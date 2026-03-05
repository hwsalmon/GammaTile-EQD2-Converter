"""
main.py — GammaTile EQD2 Converter — Application Entry Point

Usage (CLI / development):
    python main.py                          # launch with file-picker dialog (Phase 3)
    python main.py --input "Test Data/"     # load a specific DICOM directory
    python main.py --input "Test Data/" --trep 1.5 --ab 2.0  # with physics params

Phases:
    1  io_manager.py   — DICOM-RT ingestion      ✅
    2  physics_engine.py — EQD2 engine           ✅
    3  viewer.py        — Orthoviewer UI          🔲 (TODO)
    4  exporter.py      — DICOM-RT export         🔲 (TODO)
"""

import argparse
import logging
import sys
from pathlib import Path

from io_manager import IOManager, DICOMIngestionError
from physics_engine import PhysicsEngine, RadiobiologyParameters


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GammaTile EQD2 Converter — Cs-131 physical dose → EQD2"
    )
    parser.add_argument(
        "--input", "-i",
        metavar="DIR",
        help="Directory containing CT, RTSTRUCT, and RTDOSE DICOM files.",
    )
    parser.add_argument(
        "--trep",
        type=float,
        default=1.5,
        metavar="HOURS",
        help="Sublethal damage repair half-time in hours (default: 1.5).",
    )
    parser.add_argument(
        "--ab",
        type=float,
        default=2.0,
        metavar="GY",
        help="Tissue α/β ratio in Gy (default: 2.0).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def run_headless(input_dir: str, t_rep_hours: float, alpha_beta: float) -> None:
    """Load data, run EQD2 conversion, and print a summary (no GUI)."""
    logger = logging.getLogger(__name__)

    # --- Phase 1: Load ---
    logger.info("=== Phase 1: DICOM-RT Ingestion ===")
    manager = IOManager(input_dir)
    dataset = manager.load()

    # --- Phase 2: Convert ---
    logger.info("=== Phase 2: EQD2 Conversion ===")
    params = RadiobiologyParameters(t_rep_hours=t_rep_hours, alpha_beta=alpha_beta)
    engine = PhysicsEngine(params)

    logger.info("Engine configuration:")
    for k, v in engine.summary().items():
        logger.info("  %-25s %s", k, v)

    result = engine.compute_eqd2_volume(dataset.dose_array_gy)

    logger.info("EQD2 Result:")
    logger.info("  Max physical dose : %.4f Gy", dataset.dose_array_gy.max())
    logger.info("  Max BED           : %.4f Gy", result.bed_gy.max())
    logger.info("  Max EQD2          : %.4f Gy", result.max_eqd2_gy)
    logger.info("  Mean EQD2 (volume): %.4f Gy", result.mean_eqd2_gy)

    logger.info("Structure EQD2 summary:")
    for name, mask in dataset.structure_masks.items():
        if mask.any():
            eqd2_in_roi = result.eqd2_gy[mask]
            logger.info(
                "  %-30s mean=%.3f Gy  max=%.3f Gy  D98=%.3f Gy",
                name,
                eqd2_in_roi.mean(),
                eqd2_in_roi.max(),
                float(sorted(eqd2_in_roi)[int(len(eqd2_in_roi) * 0.02)]),
            )

    logger.info("=== Phase 3 (viewer) and Phase 4 (export) — Not yet implemented ===")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if "--headless" in sys.argv:
        if not args.input:
            logger.error("--headless requires --input <dir>")
            sys.exit(1)
        try:
            run_headless(args.input, args.trep, args.ab)
        except DICOMIngestionError as exc:
            logger.error("DICOM ingestion failed: %s", exc)
            sys.exit(1)
    else:
        # Launch PySide6 GUI
        from viewer import main as launch_viewer
        launch_viewer()


if __name__ == "__main__":
    main()
