from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import geopandas as gpd
import laspy


@dataclass(frozen=True)
class TileInput:
    tile_id: str
    area_m2: float
    laz_path: Path


@dataclass(frozen=True)
class TileResult:
    tile_id: str
    n_points: Optional[int]
    z_min: Optional[float]
    z_max: Optional[float]
    z_mean: Optional[float]
    density: Optional[float]
    comment: str


def load_tiles_from_shp(shp_path: Path, tile_id_field: str) -> List[Tuple[str, float]]:
    """
    Returns list of (tile_id, area_m2).
    NOTE: area is in CRS units -> for correct m², SHP CRS must be in meters (e.g., UTM).
    """
    gdf = gpd.read_file(shp_path)
    if tile_id_field not in gdf.columns:
        raise ValueError(
            f"Tile ID field '{tile_id_field}' not found in SHP. "
            f"Available fields: {list(gdf.columns)}"
        )

    areas = gdf.geometry.area.to_numpy()
    ids = gdf[tile_id_field].astype(str).to_numpy()

    out: List[Tuple[str, float]] = []
    for tile_id, area in zip(ids, areas):
        try:
            area_m2 = float(area)
        except Exception:
            area_m2 = float("nan")
        out.append((str(tile_id), area_m2))
    return out


def resolve_laz_path(laz_dir: Path, pattern: str, tile_id: str) -> Path:
    """
    Resolve LAZ path from pattern.
    Supports {Kachel_ID} and {tile_id}.
    Tries common extension variants if not found.
    """
    name = pattern.format(Kachel_ID=tile_id, tile_id=tile_id)
    p = (laz_dir / name)

    if p.exists():
        return p.resolve()

    # Try .laz / .LAZ variants
    if p.suffix == "":
        alt1 = p.with_suffix(".laz")
        if alt1.exists():
            return alt1.resolve()
        alt2 = p.with_suffix(".LAZ")
        if alt2.exists():
            return alt2.resolve()

    if p.suffix.lower() == ".laz":
        alt = p.with_suffix(".LAZ")
        if alt.exists():
            return alt.resolve()

    if p.suffix == ".LAZ":
        alt = p.with_suffix(".laz")
        if alt.exists():
            return alt.resolve()

    return p.resolve()


def header_stats_from_laz(laz_path: Path) -> Tuple[Optional[int], Optional[float], Optional[float], str]:
    """
    Fast stats from header:
      - point_count
      - zmin, zmax (header mins/maxs)
    """
    try:
        with laspy.open(laz_path) as reader:
            hdr = reader.header
            n = int(hdr.point_count)
            z_min = float(hdr.mins[2])
            z_max = float(hdr.maxs[2])
            return n, z_min, z_max, ""
    except Exception as e:
        return None, None, None, f"header read error: {str(e)[:200]}"


def mean_z_from_laz(
    laz_path: Path,
    mode: str = "sample",          # "none" | "sample" | "full"
    sample_target: int = 200_000,  # max samples for sample mode
    chunk_size: int = 2_000_000,
) -> Tuple[Optional[float], str]:
    """
    Compute mean Z in meters using binary reading (fast).

    Z_m = Z_int * z_scale + z_offset
    mean(Z_m) = z_scale * mean(Z_int) + z_offset

    mode:
      - none:   skip mean
      - sample: take approx sample_target points (every stride-th), stop early
      - full:   scan all points (slow)
    """
    mode = mode.lower().strip()
    if mode == "none":
        return None, ""

    try:
        with laspy.open(laz_path) as reader:
            hdr = reader.header
            n = int(hdr.point_count)
            if n <= 0:
                return None, "No points"

            z_scale = float(hdr.z_scale)
            z_offset = float(hdr.z_offset)

            if mode == "sample":
                if sample_target <= 0:
                    return None, "Invalid sample_target"
                stride = max(1, n // sample_target)
            elif mode == "full":
                stride = 1
            else:
                return None, f"Unknown mean mode: {mode}"

            total_z_int = np.int64(0)
            used = 0
            global_i = 0

            for points in reader.chunk_iterator(chunk_size):
                Z = points.Z
                m = Z.size
                if m == 0:
                    continue

                # keep stride consistent across whole file
                start = (-global_i) % stride
                sel = Z[start::stride]

                total_z_int += np.int64(sel.sum(dtype=np.int64))
                used += int(sel.size)
                global_i += m

                if mode == "sample" and used >= sample_target:
                    break

            if used == 0:
                return None, "No Z samples selected"

            mean_z = z_scale * (float(total_z_int) / float(used)) + z_offset
            return mean_z, ""
#konment
    except Exception as e:
        return None, f"mean error: {str(e)[:200]}"


def _process_one_tile(args: Tuple[TileInput, str, int, int]) -> TileResult:
    """
    Worker for multiprocessing (must be top-level on Windows).
    """
    t, mean_mode, sample_target, chunk_size = args

    if not t.laz_path.exists():
        return TileResult(t.tile_id, None, None, None, None, None, f"LAZ not found: {t.laz_path.name}")

    n_points, z_min, z_max, c1 = header_stats_from_laz(t.laz_path)
    if n_points is None:
        return TileResult(t.tile_id, None, None, None, None, None, c1)

    z_mean, c2 = mean_z_from_laz(
        t.laz_path,
        mode=mean_mode,
        sample_target=sample_target,
        chunk_size=chunk_size,
    )

    density = None
    comments = []
    if c1:
        comments.append(c1)
    if c2:
        comments.append(c2)

    if t.area_m2 and math.isfinite(t.area_m2) and t.area_m2 > 0:
        density = float(n_points) / float(t.area_m2)
        if density < 40.0:
            comments.append("Check")
    else:
        comment_parts.append("Invalid tile area")

    return TileResult(
        tile_id=t.tile_id,
        n_points=n_points,
        z_min=z_min,
        z_max=z_max,
        z_mean=z_mean,
        density=density,
        comment=" | ".join(comments),
    )


def run_tile_report(
    shp_tiles: Path,
    tile_id_field: str,
    laz_dir: Path,
    laz_pattern: str,
    out_csv: Path,
    workers: int = 8,
    mean_mode: str = "sample",      # none | sample | full
    sample_target: int = 200_000,
    chunk_size: int = 2_000_000,
    out_missing_csv: Optional[Path] = None,
    progress_every: int = 200,
) -> None:
    """
    STEP 1: Per-tile report to CSV.

    Output columns:
      Kachel-ID, Gesamtpunktzahl, Z min (m), Z max (m), Z mean, Punktdichte, Kommentar

    - n_points, zmin, zmax from header (fast)
    - zmean via laspy (sample by default) (fast-ish)
    """
    shp_tiles = Path(shp_tiles)
    laz_dir = Path(laz_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_missing_csv is not None:
        out_missing_csv = Path(out_missing_csv)
        out_missing_csv.parent.mkdir(parents=True, exist_ok=True)

    tiles = load_tiles_from_shp(shp_tiles, tile_id_field)
    inputs: List[TileInput] = []
    for tile_id, area_m2 in tiles:
        laz_path = resolve_laz_path(laz_dir, laz_pattern, tile_id)
        inputs.append(TileInput(tile_id=tile_id, area_m2=area_m2, laz_path=laz_path))

    tasks = [(t, mean_mode, sample_target, chunk_size) for t in inputs]

    results: List[TileResult] = []
    total = len(tasks)
    done = 0

    # Windows friendly
    mp.freeze_support()

    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_process_one_tile, task) for task in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if progress_every > 0 and (done % progress_every == 0 or done == total):
                print(f"[tile-report] processed {done}/{total}")

    results.sort(key=lambda r: r.tile_id)

    # Main report
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Kachel-ID", "Gesamtpunktzahl", "Z min (m)", "Z max (m)", "Z mean", "Punktdichte", "Kommentar"])
        for r in results:
            w.writerow([
                r.tile_id,
                "" if r.n_points is None else r.n_points,
                "" if r.z_min is None else f"{r.z_min:.3f}",
                "" if r.z_max is None else f"{r.z_max:.3f}",
                "" if r.z_mean is None else f"{r.z_mean:.3f}",
                "" if r.density is None else f"{r.density:.1f}",
                r.comment,
            ])

    # Missing/problem list
    if out_missing_csv is not None:
        with open(out_missing_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Kachel-ID", "Kommentar"])
            for r in results:
                is_missing = (r.comment or "").lower().startswith("laz not found")
                is_failed = (r.n_points is None)
                if is_missing or is_failed:
                    w.writerow([r.tile_id, r.comment])

    print(f"[tile-report] DONE. Main: {out_csv}")
    if out_missing_csv is not None:
        print(f"[tile-report] Missing/problem: {out_missing_csv}")
