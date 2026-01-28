from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import laspy

# Optional SciPy KDTree (fallback to naive)
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    KDTree = None
    HAVE_SCIPY = False

# Optional Open3D for interactive picking
try:
    import open3d as o3d  # type: ignore
    HAVE_O3D = True
except Exception:
    o3d = None
    HAVE_O3D = False


_MAIN_ID_RE_DEFAULT = r"^\d+$"


@dataclass(frozen=True)
class Control:
    ctrl_id: str
    x: float
    y: float
    z: float
    kommentar: str


def _sniff_delimiter(path: Path) -> str:
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "\t" in first:
        return "\t"
    if ";" in first:
        return ";"
    return ","


def _read_controls_filtered(
    controls_csv: Path,
    delimiter: Optional[str],
    id_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    comment_col: str,
    comment_value: str,
    main_id_regex: str,
) -> List[Control]:
    delim = delimiter if delimiter is not None else _sniff_delimiter(controls_csv)
    main_re = re.compile(main_id_regex)

    out: List[Control] = []
    with open(controls_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        fields = r.fieldnames or []
        needed = [id_col, x_col, y_col, z_col, comment_col]
        missing = [c for c in needed if c not in fields]
        if missing:
            raise ValueError(f"Missing columns in controls file: {missing}. Found: {fields}")

        for row in r:
            kommentar = (row.get(comment_col) or "").strip()
            if comment_value not in kommentar:
                continue

            cid = (row.get(id_col) or "").strip()
            if not main_re.match(cid):
                continue

            try:
                x = float(str(row[x_col]).replace(",", "."))
                y = float(str(row[y_col]).replace(",", "."))
                z = float(str(row[z_col]).replace(",", "."))
            except Exception:
                continue

            out.append(Control(ctrl_id=cid, x=x, y=y, z=z, kommentar=kommentar))

    return out


def load_pointcloud_as_arrays(laz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    las = laspy.read(laz_path)
    X = np.asarray(las.x, dtype=np.float64)
    Y = np.asarray(las.y, dtype=np.float64)
    Z = np.asarray(las.z, dtype=np.float64)

    I = None
    try:
        if "intensity" in las.point_format.dimension_names:
            I = np.asarray(las.intensity, dtype=np.float64)
    except Exception:
        I = None
    return X, Y, Z, I


# -------------------------
# KDTree (same spirit as original)
# -------------------------
def build_kdtree_xy(X: np.ndarray, Y: np.ndarray):
    xy = np.column_stack([X, Y])
    if HAVE_SCIPY:
        return KDTree(xy), xy

    class Naive:
        def __init__(self, xy_):
            self.xy = xy_

        def query_ball_point(self, q, r):
            d = np.hypot(self.xy[:, 0] - q[0], self.xy[:, 1] - q[1])
            return np.flatnonzero(d <= r)

        def query(self, q, k=1):
            d = np.hypot(self.xy[:, 0] - q[0], self.xy[:, 1] - q[1])
            k = min(k, d.size)
            idx = np.argpartition(d, k - 1)[:k]
            idx = idx[np.argsort(d[idx])]
            return d[idx], idx

    return Naive(xy), xy


def build_kdtree_xyz(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    xyz = np.column_stack([X, Y, Z])
    if HAVE_SCIPY:
        return KDTree(xyz), xyz

    class Naive3D:
        def __init__(self, xyz_):
            self.xyz = xyz_

        def query_ball_point(self, q, r):
            d = np.linalg.norm(self.xyz - q, axis=1)
            return np.flatnonzero(d <= r)

        def query(self, q, k=1):
            d = np.linalg.norm(self.xyz - q, axis=1)
            k = min(k, d.size)
            idx = np.argpartition(d, k - 1)[:k]
            idx = idx[np.argsort(d[idx])]
            return d[idx], idx

    return Naive3D(xyz), xyz


def select_neighbors_xy(tree, xy: np.ndarray, q: np.ndarray, k: Optional[int], radius: float):
    idxs = np.asarray(tree.query_ball_point(q, r=radius), dtype=int)
    if idxs.size == 0:
        return idxs, np.array([], dtype=float)

    d = np.hypot(xy[idxs, 0] - q[0], xy[idxs, 1] - q[1])
    order = np.argsort(d)
    sel = idxs[order][:k] if k else idxs[order]
    return sel, d[order][: (k if k else None)]


def select_neighbors_3d(tree, xyz: np.ndarray, q: np.ndarray, k: Optional[int], radius: float):
    idxs = np.asarray(tree.query_ball_point(q, r=radius), dtype=int)
    if idxs.size == 0:
        return idxs, np.array([], dtype=float)

    d = np.linalg.norm(xyz[idxs] - q, axis=1)
    order = np.argsort(d)
    sel = idxs[order][:k] if k else idxs[order]
    return sel, d[order][: (k if k else None)]


# -------------------------
# Open3D picking with cross markers (deterministic top-down)
# -------------------------
def _parse_rgb(s: str) -> Tuple[float, float, float]:
    try:
        a = [float(x.strip()) for x in s.split(",")]
        if len(a) != 3:
            return (1.0, 0.0, 0.0)
        return (float(a[0]), float(a[1]), float(a[2]))
    except Exception:
        return (1.0, 0.0, 0.0)


def _auto_intensity_window(I: np.ndarray, p_low: float = 2.0, p_high: float = 98.0, ignore_zeros: bool = True):
    I = np.asarray(I, dtype=np.float64)
    if ignore_zeros:
        I = I[I != 0]
    if I.size == 0:
        return None, None
    lo = np.percentile(I, p_low)
    hi = np.percentile(I, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    return float(lo), float(hi)


def _colors_from_intensity_or_z(Z: np.ndarray, I: Optional[np.ndarray], intensity_auto: bool, ignore_zeros: bool):
    Z = np.asarray(Z, dtype=np.float64)
    if I is not None and intensity_auto:
        imin, imax = _auto_intensity_window(I, 2.0, 98.0, ignore_zeros=ignore_zeros)
        if imin is not None and imax is not None:
            t = np.clip((I - imin) / max(imax - imin, 1e-9), 0.0, 1.0)
            return np.repeat(t[:, None], 3, axis=1)

    # fallback Z: blue-red-ish
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    t = (Z - zmin) / max(zmax - zmin, 1e-9)
    return np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)


def _make_cross_points(center: np.ndarray, size: float) -> np.ndarray:
    cx, cy, cz = center
    return np.array(
        [
            [cx - size, cy, cz],
            [cx + size, cy, cz],
            [cx, cy - size, cz],
            [cx, cy + size, cz],
            [cx, cy, cz - size],
            [cx, cy, cz + size],
        ],
        dtype=np.float64,
    )


def pick_one_point_with_cross(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    I: Optional[np.ndarray],
    ctrl_xyz: np.ndarray,
    point_size: float = 7.0,
    ctrl_color: str = "1,0,0",
    cross_size_m: float = 0.10,  # <-- 10 cm arms (±0.10 m)
    intensity_auto: bool = True,
    intensity_ignore_zeros: bool = True,
    window_title: str = "Pick point (Shift+LMB, Q)",
) -> Optional[Dict[str, float]]:
    """
    Deterministic viewer:
      - always centered to the control point (ctrl becomes origin)
      - always TOP-DOWN view initially
      - cross arms are ±cross_size_m in meters (default 0.10 m)
      - point size via Open3D RenderOption.point_size (pixels)

    Picking:
      - if user clicks control/cross point, remap to nearest real cloud point.
    """
    if not HAVE_O3D:
        print("[WARN] Open3D not available -> interactive pick disabled.")
        return None

    pts_world = np.column_stack([X, Y, Z]).astype(np.float64, copy=False)
    n_pts = int(pts_world.shape[0])
    if n_pts == 0:
        return None

    ctrl_world = ctrl_xyz.astype(np.float64, copy=False)
    if ctrl_world.shape[0] < 1:
        return None

    # Center EVERYTHING to control point => stable view & stable cross size in meters
    origin = ctrl_world[0].copy()
    pts = pts_world - origin
    ctrl = ctrl_world - origin  # first is [0,0,0]

    colors_cloud = _colors_from_intensity_or_z(
        Z, I, intensity_auto=intensity_auto, ignore_zeros=intensity_ignore_zeros
    )

    n_ctrl = int(ctrl.shape[0])

    # cross points around control(s) (expected 1 control here)
    cross_owner: List[int] = []
    cross_pts_list: List[np.ndarray] = []
    for i in range(n_ctrl):
        cps = _make_cross_points(ctrl[i], size=float(cross_size_m))
        cross_pts_list.append(cps)
        cross_owner.extend([i] * cps.shape[0])

        # --- DEBUG: ověř délku ramen (v tom samém prostoru, co jde do O3D) ---
    arm_x = float(np.linalg.norm(cps[1] - ctrl[i]))  # +X bod
    arm_y = float(np.linalg.norm(cps[3] - ctrl[i]))  # +Y bod
    arm_z = float(np.linalg.norm(cps[5] - ctrl[i]))  # +Z bod
    print(f"[DEBUG] cross_size_m param = {cross_size_m}")
    print(f"[DEBUG] arm lengths (x,y,z) = {arm_x:.6f}, {arm_y:.6f}, {arm_z:.6f}")
    
    cross_pts = np.vstack(cross_pts_list) if cross_pts_list else np.empty((0, 3), dtype=np.float64)
    n_cross = int(cross_pts.shape[0])

    # combined geometry
    all_xyz = np.vstack([pts, ctrl, cross_pts])

    ctrl_rgb = np.asarray(_parse_rgb(ctrl_color), dtype=np.float64)
    cross_rgb = np.array([1.0, 1.0, 0.0], dtype=np.float64)

    all_colors = np.vstack(
        [
            colors_cloud.astype(np.float64),
            np.repeat(ctrl_rgb[None, :], n_ctrl, axis=0),
            np.repeat(cross_rgb[None, :], n_cross, axis=0),
        ]
    )

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_xyz))
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_title, width=1280, height=800)
    vis.add_geometry(pcd)

    # Open3D docs: RenderOption.point_size in pixels
    try:
        ro = vis.get_render_option()
        ro.point_size = float(point_size)
        ro.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    except Exception:
        pass

    # Deterministic camera: top-down, looking at origin
    try:
        vc = vis.get_view_control()
        vc.set_lookat([0.0, 0.0, 0.0])
        vc.set_front([0.0, 0.0, -1.0])  # looking down -Z
        vc.set_up([0.0, 1.0, 0.0])      # Y up on screen
        vc.set_zoom(0.70)               # fixed zoom for clip radius ~20 m

        vis.poll_events()
        vis.update_renderer()
    except Exception:
        pass

    print(f"[PICK] cloud={n_pts}  ctrl={n_ctrl}  cross_pts={n_cross}  (Shift+LMB, Q)")
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    if not picked:
        return None

    idx = int(picked[0])

    # If clicked beyond real points -> remap to nearest real cloud point (WORLD coords)
    if idx >= n_pts:
        if idx < n_pts + n_ctrl:
            c_world = ctrl_world[idx - n_pts]
        else:
            which = idx - (n_pts + n_ctrl)
            owner = cross_owner[which]
            c_world = ctrl_world[owner]

        if HAVE_SCIPY:
            tree = KDTree(pts_world)
            _, nn = tree.query(c_world, k=1)
            idx = int(nn)
        else:
            d = np.linalg.norm(pts_world - c_world, axis=1)
            idx = int(np.argmin(d))
        print(f"[PICK] clicked ctrl/cross -> nearest cloud point #{idx}")

    return {"idx": idx, "x": float(X[idx]), "y": float(Y[idx]), "z": float(Z[idx])}


# -------------------------
# Statistics
# -------------------------
def _skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s <= 0:
        return 0.0
    m3 = float(np.mean((x - m) ** 3))
    return m3 / (s ** 3)


def _summary_cm_from_errors(err_m: np.ndarray, use_abs_for_percentile: bool = True) -> Dict[str, float]:
    e_cm = np.asarray(err_m, dtype=np.float64) * 100.0
    e_cm = e_cm[np.isfinite(e_cm)]
    if e_cm.size == 0:
        return {
            "Anzahl der Punkte": 0,
            "EMSE (cm)": float("nan"),
            "Std abw (cm)": float("nan"),
            "Durchschnitt (cm)": float("nan"),
            "Median (cm)": float("nan"),
            "Schiefe": float("nan"),
            "Min (cm)": float("nan"),
            "Max (cm)": float("nan"),
            "95%CI (cm)": float("nan"),
            "95. Perzentil (cm)": float("nan"),
        }

    n = int(e_cm.size)
    mean = float(np.mean(e_cm))
    std = float(np.std(e_cm, ddof=1)) if n > 1 else 0.0
    rmse = float(np.sqrt(np.mean(e_cm ** 2)))
    median = float(np.median(e_cm))
    sk = _skewness(e_cm)
    mn = float(np.min(e_cm))
    mx = float(np.max(e_cm))
    ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    p95 = float(np.percentile(np.abs(e_cm) if use_abs_for_percentile else e_cm, 95.0))

    return {
        "Anzahl der Punkte": n,
        "EMSE (cm)": rmse,
        "Std abw (cm)": std,
        "Durchschnitt (cm)": mean,
        "Median (cm)": median,
        "Schiefe": sk,
        "Min (cm)": mn,
        "Max (cm)": mx,
        "95%CI (cm)": ci95,
        "95. Perzentil (cm)": p95,
    }


# -------------------------
# Main runner
# -------------------------
def run_accuracy(
    cloud_dir: Path,
    controls_csv: Path,
    out_dir: Path,
    delimiter: Optional[str] = None,
    id_col: str = "Kontrollpunkt_ID",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    comment_col: str = "Kommentar",
    comment_value: str = "Kontrolle",
    main_id_regex: str = _MAIN_ID_RE_DEFAULT,
    k: int = 6,
    radius: float = 0.4,
    use_3d_radius: bool = True,
    min_points: int = 3,
    z_stat: str = "mean",  # mean|median
    interactive: bool = False,
    point_size: float = 7.0,
    ctrl_color: str = "1,0,0",
    cross_size_m: float = 0.10,  # <-- default 10 cm
    intensity_auto: bool = True,
    intensity_ignore_zeros: bool = True,
    sep: str = ";",
) -> None:
    """
    Expects clips in cloud_dir named "<Kontrollpunkt_ID>.laz" (or .LAZ).
    Writes:
      - height_check.csv
      - xy_check.csv
      - height_summary.csv
      - xy_summary.csv
      - excluded_points.csv

    NOTE:
      - If interactive=True and no pick is selected, the point is excluded from BOTH Z and XY.
      - XY summary is computed on dxy = sqrt(dx^2 + dy^2) (magnitudes).
    """
    cloud_dir = Path(cloud_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = _read_controls_filtered(
        controls_csv=Path(controls_csv),
        delimiter=delimiter,
        id_col=id_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        comment_col=comment_col,
        comment_value=comment_value,
        main_id_regex=main_id_regex,
    )
    if not controls:
        raise RuntimeError("No controls selected (check Kommentar filter / main ID regex / delimiter).")

    height_rows: List[Dict[str, object]] = []
    xy_rows: List[Dict[str, object]] = []
    excluded: List[Dict[str, object]] = []

    dz_list: List[float] = []
    dxy_list: List[float] = []

    for c in controls:
        # find clip
        p1 = (cloud_dir / f"{c.ctrl_id}.laz")
        p2 = (cloud_dir / f"{c.ctrl_id}.LAZ")
        laz_path = p1 if p1.exists() else p2

        if not laz_path.exists():
            excluded.append({"Kontrollpunkt_ID": c.ctrl_id, "Reason": "clip not found", "X": c.x, "Y": c.y, "Z": c.z})
            continue

        # load clip
        try:
            X, Y, Z, I = load_pointcloud_as_arrays(laz_path)
        except Exception as e:
            excluded.append({"Kontrollpunkt_ID": c.ctrl_id, "Reason": f"failed to read clip: {str(e)[:200]}", "X": c.x, "Y": c.y, "Z": c.z})
            continue

        if X.size == 0:
            excluded.append({"Kontrollpunkt_ID": c.ctrl_id, "Reason": "empty clip", "X": c.x, "Y": c.y, "Z": c.z})
            continue

        # height neighbors
        if use_3d_radius:
            tree, xyz = build_kdtree_xyz(X, Y, Z)
            q = np.array([c.x, c.y, c.z], dtype=np.float64)
            idxs, _d = select_neighbors_3d(tree, xyz, q, k=(k if k > 0 else None), radius=float(radius))
        else:
            tree, xy = build_kdtree_xy(X, Y)
            q = np.array([c.x, c.y], dtype=np.float64)
            idxs, _d = select_neighbors_xy(tree, xy, q, k=(k if k > 0 else None), radius=float(radius))

        if idxs.size < int(min_points):
            excluded.append({"Kontrollpunkt_ID": c.ctrl_id, "Reason": f"too few neighbors (n={int(idxs.size)})", "X": c.x, "Y": c.y, "Z": c.z})
            continue

        z_neighbors = Z[idxs]
        lidar_z = float(np.median(z_neighbors)) if z_stat.lower().strip() == "median" else float(np.mean(z_neighbors))
        lidar_x = float(np.mean(X[idxs]))
        lidar_y = float(np.mean(Y[idxs]))
        dz = lidar_z - c.z

        pick = None
        if interactive:
            ctrl_xyz = np.array([[c.x, c.y, c.z]], dtype=np.float64)
            pick = pick_one_point_with_cross(
                X=X,
                Y=Y,
                Z=Z,
                I=I,
                ctrl_xyz=ctrl_xyz,
                point_size=point_size,
                ctrl_color=ctrl_color,
                cross_size_m=cross_size_m,
                intensity_auto=intensity_auto,
                intensity_ignore_zeros=intensity_ignore_zeros,
                window_title=f"Pick XY point for Kontrollpunkt {c.ctrl_id} (Shift+LMB, Q)",
            )
            if pick is None:
                excluded.append({"Kontrollpunkt_ID": c.ctrl_id, "Reason": "no pick selected (excluded from Z & XY)", "X": c.x, "Y": c.y, "Z": c.z})
                continue

        # store height row (accepted)
        height_rows.append(
            {
                "Kontrollpunkt_ID": c.ctrl_id,
                "Kontrollpunkt_Z": c.z,
                "LiDAR_X": lidar_x,
                "LiDAR_Y": lidar_y,
                "LiDAR_Z": lidar_z,
                "Differenz Z": dz,
            }
        )
        dz_list.append(dz)

        # store XY if interactive (accepted)
        if interactive and pick is not None:
            dx = float(pick["x"] - c.x)
            dy = float(pick["y"] - c.y)
            dxy = float(math.hypot(dx, dy))

            xy_rows.append(
                {
                    "Kontrollpunkt_ID": c.ctrl_id,
                    "Kontrollpunkt_X": c.x,
                    "Kontrollpunkt_Y": c.y,
                    "LiDAR_X": float(pick["x"]),
                    "LiDAR_Y": float(pick["y"]),
                    "Differenz X": dx,
                    "Differenz Y": dy,
                }
            )
            dxy_list.append(dxy)

    # write outputs
    height_csv = out_dir / "height_check.csv"
    xy_csv = out_dir / "xy_check.csv"
    excluded_csv = out_dir / "excluded_points.csv"
    height_sum_csv = out_dir / "height_summary.csv"
    xy_sum_csv = out_dir / "xy_summary.csv"

    def _write_rows(path: Path, rows: List[Dict[str, object]], headers: List[str]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=sep)
            w.writerow(headers)
            for r in rows:
                w.writerow([r.get(h, "") for h in headers])

    _write_rows(
        height_csv,
        height_rows,
        ["Kontrollpunkt_ID", "Kontrollpunkt_Z", "LiDAR_X", "LiDAR_Y", "LiDAR_Z", "Differenz Z"],
    )

    if interactive:
        _write_rows(
            xy_csv,
            xy_rows,
            ["Kontrollpunkt_ID", "Kontrollpunkt_X", "Kontrollpunkt_Y", "LiDAR_X", "LiDAR_Y", "Differenz X", "Differenz Y"],
        )
    else:
        _write_rows(
            xy_csv,
            [],
            ["Kontrollpunkt_ID", "Kontrollpunkt_X", "Kontrollpunkt_Y", "LiDAR_X", "LiDAR_Y", "Differenz X", "Differenz Y"],
        )

    _write_rows(excluded_csv, excluded, ["Kontrollpunkt_ID", "Reason", "X", "Y", "Z"])

    # summaries
    z_summary = _summary_cm_from_errors(np.array(dz_list, dtype=np.float64), use_abs_for_percentile=True)
    xy_summary = _summary_cm_from_errors(np.array(dxy_list, dtype=np.float64), use_abs_for_percentile=True)

    z_summary_named = dict(z_summary)
    z_summary_named["EMSEz (cm)"] = z_summary_named.pop("EMSE (cm)")
    xy_summary_named = dict(xy_summary)
    xy_summary_named["EMSExy (cm)"] = xy_summary_named.pop("EMSE (cm)")

    def _write_summary(path: Path, summ: Dict[str, float]):
        if "EMSEz (cm)" in summ:
            order = [
                "Anzahl der Punkte",
                "EMSEz (cm)",
                "Std abw (cm)",
                "Durchschnitt (cm)",
                "Median (cm)",
                "Schiefe",
                "Min (cm)",
                "Max (cm)",
                "95%CI (cm)",
                "95. Perzentil (cm)",
            ]
        else:
            order = [
                "Anzahl der Punkte",
                "EMSExy (cm)",
                "Std abw (cm)",
                "Durchschnitt (cm)",
                "Median (cm)",
                "Schiefe",
                "Min (cm)",
                "Max (cm)",
                "95%CI (cm)",
                "95. Perzentil (cm)",
            ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=sep)
            w.writerow(order)
            w.writerow(
                [
                    f"{summ.get(k, float('nan')):.3f}"
                    if isinstance(summ.get(k, None), (float, int)) and k != "Anzahl der Punkte"
                    else summ.get(k, "")
                    for k in order
                ]
            )

    _write_summary(height_sum_csv, z_summary_named)
    _write_summary(xy_sum_csv, xy_summary_named)

    print(f"[accuracy] DONE -> {out_dir}")
    print(f"[accuracy] height: {height_csv}")
    print(f"[accuracy] xy:     {xy_csv}")
    print(f"[accuracy] excl:   {excluded_csv}")
    print(f"[accuracy] sumZ:   {height_sum_csv}")
    print(f"[accuracy] sumXY:  {xy_sum_csv}")
