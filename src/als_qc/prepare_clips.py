from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import geopandas as gpd
from shapely.geometry import Point

from als_qc.lastools import LastoolsRunner
from als_qc.qc_tiles import resolve_laz_path


_MAIN_ID_RE = re.compile(r"^\d+$", re.ASCII)


@dataclass(frozen=True)
class ControlPoint:
    ctrl_id: str
    x: float
    y: float
    z: Optional[float]
    kommentar: str


def _sniff_delimiter(path: Path) -> str:
    # your file is tab-separated, so default to \t if present in header
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if "\t" in first:
        return "\t"
    if ";" in first:
        return ";"
    return ","


def _read_controls(
    controls_csv: Path,
    id_col: str,
    x_col: str,
    y_col: str,
    z_col: Optional[str],
    kommentar_col: str,
    kommentar_value: str,
    delimiter: Optional[str],
) -> List[ControlPoint]:
    delim = delimiter if delimiter is not None else _sniff_delimiter(controls_csv)

    pts: List[ControlPoint] = []
    with open(controls_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        missing = [c for c in [id_col, x_col, y_col, kommentar_col] if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in controls file: {missing}. Found: {r.fieldnames}")

        for row in r:
            kommentar = (row.get(kommentar_col) or "").strip()
            if kommentar_value not in kommentar:
                continue

            ctrl_id = (row.get(id_col) or "").strip()
            if not _MAIN_ID_RE.match(ctrl_id):
                continue  # only main ids

            x = float(row[x_col])
            y = float(row[y_col])
            z = None
            if z_col:
                z_raw = row.get(z_col)
                if z_raw not in (None, ""):
                    z = float(z_raw)

            pts.append(ControlPoint(ctrl_id=ctrl_id, x=x, y=y, z=z, kommentar=kommentar))

    return pts


def prepare_control_clips(
    shp_tiles: Path,
    tile_id_field: str,
    laz_dir: Path,
    laz_pattern: str,
    controls_csv: Path,
    out_dir: Path,
    lastools_bin: Path,
    clip_radius_m: float = 20.0,
    id_col: str = "Kontrollpunkt_ID",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: Optional[str] = "Z",
    kommentar_col: str = "Kommentar",
    kommentar_value: str = "Kontrolle",
    delimiter: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    1) Load SHP tiles
    2) Load controls (only Kommentar contains `kommentar_value` AND main ids ^\\d+$)
    3) Spatial join -> assign tile id
    4) Clip LAZ per control point using LAStools las2las -keep_circle X Y R
    5) Write:
       - controls_main_kontrolle.csv (with tile + input laz + output laz)
       - clip_errors.csv (problems)
    """
    shp_tiles = Path(shp_tiles)
    laz_dir = Path(laz_dir)
    controls_csv = Path(controls_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = LastoolsRunner(Path(lastools_bin))

    controls = _read_controls(
        controls_csv=controls_csv,
        id_col=id_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        kommentar_col=kommentar_col,
        kommentar_value=kommentar_value,
        delimiter=delimiter,
    )

    if not controls:
        raise RuntimeError("No control points selected (check kommentar_value / id format / delimiter).")

    tiles_gdf = gpd.read_file(shp_tiles)
    if tile_id_field not in tiles_gdf.columns:
        raise ValueError(f"tile_id_field '{tile_id_field}' not found in SHP. Fields: {list(tiles_gdf.columns)}")

    # Create points geodataframe
    pts_gdf = gpd.GeoDataFrame(
        {
            "ctrl_id": [c.ctrl_id for c in controls],
            "X": [c.x for c in controls],
            "Y": [c.y for c in controls],
            "Z": [c.z for c in controls],
            "Kommentar": [c.kommentar for c in controls],
        },
        geometry=[Point(c.x, c.y) for c in controls],
        crs=tiles_gdf.crs,  # assume same CRS as tiles
    )

    # spatial join (point in polygon)
    joined = gpd.sjoin(pts_gdf, tiles_gdf[[tile_id_field, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={tile_id_field: "Kachel_ID"})

    map_csv = out_dir / "controls_main_kontrolle.csv"
    err_csv = out_dir / "clip_errors.csv"

    with open(map_csv, "w", newline="", encoding="utf-8") as fm, open(err_csv, "w", newline="", encoding="utf-8") as fe:
        wm = csv.writer(fm)
        we = csv.writer(fe)

        wm.writerow(["Kontrollpunkt_ID", "X", "Y", "Z", "Kachel_ID", "Input_LAZ", "Output_LAZ"])
        we.writerow(["Kontrollpunkt_ID", "X", "Y", "Reason"])

        for _, row in joined.iterrows():
            ctrl_id = str(row["ctrl_id"])
            x = float(row["X"])
            y = float(row["Y"])
            kachel = row.get("Kachel_ID")

            if kachel is None or (isinstance(kachel, float) and str(kachel) == "nan"):
                we.writerow([ctrl_id, x, y, "Point not inside any tile polygon"])
                continue

            kachel_id = str(kachel)
            input_laz = resolve_laz_path(Path(laz_dir), laz_pattern, kachel_id)
            if not input_laz.exists():
                we.writerow([ctrl_id, x, y, f"Input LAZ not found: {input_laz.name}"])
                continue

            out_laz = (out_dir / f"{ctrl_id}.laz").resolve()
            if out_laz.exists() and not overwrite:
                wm.writerow([ctrl_id, x, y, row.get("Z"), kachel_id, str(input_laz), str(out_laz)])
                continue

            exe = str(runner.exe("las2las64.exe"))
            cmd = [
                exe,
                "-i", str(input_laz),
                "-keep_circle", f"{x}", f"{y}", f"{clip_radius_m}",
                "-olaz",
                "-o", str(out_laz),
                "-quiet",
            ]
            res = runner.run(cmd)

            if res.returncode != 0:
                we.writerow([ctrl_id, x, y, f"las2las failed: {res.stderr[:200]}"])
                continue

            wm.writerow([ctrl_id, x, y, row.get("Z"), kachel_id, str(input_laz), str(out_laz)])

    print(f"[prepare-clips] DONE. Clips in: {out_dir}")
    print(f"[prepare-clips] Mapping: {map_csv}")
    print(f"[prepare-clips] Errors: {err_csv}")
