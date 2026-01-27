from __future__ import annotations
from pathlib import Path
import argparse

from als_qc.qc_tiles import run_tile_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="als-qc", description="ALS QC utilities (Step 1: tile-report)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tile-report", help="Create per-tile statistics CSV (count, Zmin/Zmax/Zmean sample, density)")
    t.add_argument("--shp-tiles", required=True, type=Path)
    t.add_argument("--tile-id-field", default="NAME", help="Field name in SHP with Kachel-ID (default: NAME)")
    t.add_argument("--laz-dir", required=True, type=Path)
    t.add_argument("--laz-pattern", default="{Kachel_ID}.laz",
                   help='e.g. "{Kachel_ID}.laz" or "20240816_DE_{Kachel_ID}.laz" or "{Kachel_ID}"')

    t.add_argument("--out-csv", required=True, type=Path)
    t.add_argument("--out-missing-csv", type=Path, default=None, help="Optional CSV with missing/problem tiles")

    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--mean-mode", choices=["none", "sample", "full"], default="sample")
    t.add_argument("--sample-target", type=int, default=200_000)
    t.add_argument("--chunk-size", type=int, default=2_000_000)
    t.add_argument("--progress-every", type=int, default=200)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "tile-report":
        run_tile_report(
            shp_tiles=args.shp_tiles,
            tile_id_field=args.tile_id_field,
            laz_dir=args.laz_dir,
            laz_pattern=args.laz_pattern,
            out_csv=args.out_csv,
            out_missing_csv=args.out_missing_csv,   # <-- TEĎ už se správně předává
            workers=args.workers,
            mean_mode=args.mean_mode,
            sample_target=args.sample_target,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
        )
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
