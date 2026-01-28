from __future__ import annotations

from pathlib import Path
import argparse

from als_qc.prepare_clips import prepare_control_clips
from als_qc.qc_tiles import run_tile_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="als-qc", description="ALS QC utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # Step 1: tile-report
    # -------------------------
    t = sub.add_parser(
        "tile-report",
        help="Create per-tile statistics CSV (count, Zmin/Zmax/Zmean sample, density)"
    )
    t.add_argument("--shp-tiles", required=True, type=Path)
    t.add_argument("--tile-id-field", default="NAME", help="Field name in SHP with Kachel-ID (default: NAME)")
    t.add_argument("--laz-dir", required=True, type=Path)
    t.add_argument(
        "--laz-pattern",
        default="{Kachel_ID}.laz",
        help='e.g. "{Kachel_ID}.laz" or "1km_{Kachel_ID}.laz" or "{Kachel_ID}"'
    )

    t.add_argument("--out-csv", required=True, type=Path)
    t.add_argument("--out-missing-csv", type=Path, default=None, help="Optional CSV with missing/problem tiles")

    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--mean-mode", choices=["none", "sample", "full"], default="sample")
    t.add_argument("--sample-target", type=int, default=200_000)
    t.add_argument("--chunk-size", type=int, default=2_000_000)
    t.add_argument("--progress-every", type=int, default=200)

    # -------------------------
    # Step 2: prepare-clips
    # -------------------------
    c = sub.add_parser(
        "prepare-clips",
        help="Clip LAZ tiles around Kontrolle main control points and save per-point clips"
    )
    c.add_argument("--shp-tiles", required=True, type=Path)
    c.add_argument("--tile-id-field", default="NAME", help="Field name in SHP with Kachel-ID (default: NAME)")
    c.add_argument("--laz-dir", required=True, type=Path)
    c.add_argument(
        "--laz-pattern",
        default="{Kachel_ID}.laz",
        help='Same pattern as in step 1, e.g. "1km_{Kachel_ID}.laz"'
    )

    c.add_argument("--controls-csv", required=True, type=Path)
    c.add_argument("--out-dir", required=True, type=Path)
    c.add_argument("--lastools-bin", required=True, type=Path)
    c.add_argument("--clip-radius", type=float, default=20.0, help="Clip radius in meters (default: 20)")

    # columns in controls file
    c.add_argument("--id-col", default="Kontrollpunkt_ID")
    c.add_argument("--x-col", default="X")
    c.add_argument("--y-col", default="Y")
    c.add_argument("--z-col", default="Z")
    c.add_argument("--comment-col", default="Kommentar")
    c.add_argument("--comment-value", default="Kontrolle", help="Substring to filter Kommentar (default: Kontrolle)")

    # delimiter: allow passing "\t" etc.
    c.add_argument("--delimiter", default=None, help=r'Optional delimiter, e.g. "\t" or ";" or ","')
    c.add_argument("--overwrite", action="store_true", help="Overwrite existing clips")

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
            out_missing_csv=args.out_missing_csv,
            workers=args.workers,
            mean_mode=args.mean_mode,
            sample_target=args.sample_target,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
        )
        return

    if args.cmd == "prepare-clips":
        delimiter = None
        if args.delimiter:
            # allows passing "\t" in PowerShell
            delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")

        prepare_control_clips(
            shp_tiles=args.shp_tiles,
            tile_id_field=args.tile_id_field,
            laz_dir=args.laz_dir,
            laz_pattern=args.laz_pattern,
            controls_csv=args.controls_csv,
            out_dir=args.out_dir,
            lastools_bin=args.lastools_bin,
            clip_radius_m=args.clip_radius,
            id_col=args.id_col,
            x_col=args.x_col,
            y_col=args.y_col,
            z_col=args.z_col,
            kommentar_col=args.comment_col,
            kommentar_value=args.comment_value,
            delimiter=delimiter,
            overwrite=args.overwrite,
        )
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
