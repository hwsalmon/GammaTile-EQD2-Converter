"""
io_manager.py — Phase 1: DICOM-RT Data Ingestion

Handles loading, validation, and extraction of a DICOM-RT dataset consisting of:
  - CT Image Series
  - RT Structure Set (RS / RTSTRUCT)
  - RT Dose (RD / RTDOSE)

All three modalities must share the same FrameOfReferenceUID.  Structure contours
are rasterized onto the RT Dose voxel grid (not the CT grid) because all downstream
EQD2 calculations operate on dose voxels.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from rt_utils import RTStructBuilder

logger = logging.getLogger(__name__)

# Structures to load — all others are silently skipped
STRUCTURES_OF_INTEREST: frozenset[str] = frozenset({
    "CTV", "PTV", "Brain", "Brainstem", "OpticChiasm", "OpticNrv_L", "OpticNrv_R",
})


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CoordinateSystem:
    """Spatial coordinate system extracted from the RT Dose file.

    All coordinates are in millimetres, in the DICOM Patient coordinate system
    (LPS: Left, Posterior, Superior positive directions).
    """

    image_position: np.ndarray   # [x, y, z] of voxel (0, 0, 0) in mm
    pixel_spacing: np.ndarray    # [row_spacing, col_spacing] in mm
    grid_frame_offsets: np.ndarray  # z-offsets of each dose plane relative to image_position[2]
    n_rows: int
    n_cols: int
    n_slices: int

    @property
    def slice_positions_z(self) -> np.ndarray:
        """Absolute z-position of each dose slice in mm (Patient CS)."""
        return self.image_position[2] + self.grid_frame_offsets

    @property
    def voxel_volume_cc(self) -> float:
        """Voxel volume in cm³ (assumes uniform z-spacing)."""
        z_spacing_mm = abs(float(self.grid_frame_offsets[1] - self.grid_frame_offsets[0])) \
            if self.n_slices > 1 else 0.0
        return (self.pixel_spacing[0] * self.pixel_spacing[1] * z_spacing_mm) / 1000.0


@dataclass
class DICOMDataset:
    """Container for the full loaded DICOM-RT dataset."""

    ct_series: list[Dataset] = field(default_factory=list)
    rt_struct: Optional[Dataset] = None
    rt_dose: Optional[Dataset] = None
    coordinate_system: Optional[CoordinateSystem] = None
    dose_array_gy: Optional[np.ndarray] = None          # shape: (n_slices, n_rows, n_cols)
    structure_masks: dict[str, np.ndarray] = field(default_factory=dict)
    frame_of_reference_uid: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return (
            self.dose_array_gy is not None
            and self.coordinate_system is not None
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DICOMIngestionError(Exception):
    """Raised when DICOM loading or validation fails."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IOManager:
    """Loads and validates a DICOM-RT dataset from a directory on disk.

    Usage::

        manager = IOManager("/path/to/dicom/export")
        dataset = manager.load()
        # dataset.dose_array_gy  → physical dose in Gy, shape (n_slices, n_rows, n_cols)
        # dataset.structure_masks → dict of {name: bool ndarray}

    Args:
        input_dir: Path to a directory containing CT, RTSTRUCT, and RTDOSE files.
            Files may be in subdirectories; the loader searches recursively.
    """

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise DICOMIngestionError(
                f"Input path is not a directory: {self.input_dir}"
            )
        self.dataset = DICOMDataset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> DICOMDataset:
        """Load the full DICOM-RT dataset.

        Steps:
            1. Discover all DICOM files in ``input_dir``.
            2. Classify files by Modality (CT / RTSTRUCT / RTDOSE).
            3. Validate shared FrameOfReferenceUID.
            4. Extract dose pixel array (scaled to Gy).
            5. Extract coordinate system from RD header.
            6. Rasterize structure contours onto the dose voxel grid.

        Returns:
            Populated :class:`DICOMDataset`.

        Raises:
            DICOMIngestionError: On missing modalities or UID mismatch.
        """
        logger.info("Loading DICOM-RT dataset from: %s", self.input_dir)

        dcm_files = self._discover_dicom_files()
        self._classify_files(dcm_files)
        self._validate_frame_of_reference()
        self._extract_dose_array()
        self._extract_coordinate_system()
        self._extract_structure_masks()

        logger.info(
            "Load complete — CT slices: %d | Structures: %s | "
            "Dose shape: %s | Max dose: %.3f Gy",
            len(self.dataset.ct_series),
            list(self.dataset.structure_masks.keys()),
            self.dataset.dose_array_gy.shape,
            self.dataset.dose_array_gy.max(),
        )
        return self.dataset

    # ------------------------------------------------------------------
    # Step 1 — File discovery
    # ------------------------------------------------------------------

    def _discover_dicom_files(self) -> list[Path]:
        """Recursively find DICOM files under ``input_dir``.

        Looks for ``*.dcm`` first; falls back to extension-less files if none
        are found (some TPS exports omit the ``.dcm`` suffix).

        Raises:
            DICOMIngestionError: If no files are found at all.
        """
        dcm_files = list(self.input_dir.rglob("*.dcm"))

        if not dcm_files:
            # Broad fallback: any file that is not a known non-DICOM format
            non_dicom_suffixes = {
                ".txt", ".csv", ".json", ".xml", ".md", ".py", ".png", ".jpg"
            }
            dcm_files = [
                p for p in self.input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() not in non_dicom_suffixes
            ]

        if not dcm_files:
            raise DICOMIngestionError(
                f"No DICOM files found under {self.input_dir}. "
                "Ensure the directory contains CT, RTSTRUCT, and RTDOSE files."
            )

        logger.info("Discovered %d candidate DICOM file(s).", len(dcm_files))
        return dcm_files

    # ------------------------------------------------------------------
    # Step 2 — Classification
    # ------------------------------------------------------------------

    def _classify_files(self, dcm_files: list[Path]) -> None:
        """Sort files into CT, RTSTRUCT, and RTDOSE buckets by Modality tag.

        Non-readable or unrecognised files are silently skipped with a debug
        log entry.  If multiple RS or RD files are found the last one wins and
        a warning is emitted.

        Raises:
            DICOMIngestionError: If any required modality is absent.
        """
        ct_series: list[Dataset] = []
        rt_struct: Optional[Dataset] = None
        rt_dose: Optional[Dataset] = None

        for path in dcm_files:
            try:
                ds = pydicom.dcmread(str(path), force=True)
                ds.filename = str(path)          # Attach path for later use by rt_utils
                modality = getattr(ds, "Modality", "").upper().strip()

                if modality == "CT":
                    ct_series.append(ds)

                elif modality == "RTSTRUCT":
                    if rt_struct is not None:
                        logger.warning(
                            "Multiple RTSTRUCT files found — using: %s", path.name
                        )
                    rt_struct = ds

                elif modality == "RTDOSE":
                    if rt_dose is not None:
                        logger.warning(
                            "Multiple RTDOSE files found — using: %s", path.name
                        )
                    rt_dose = ds

                else:
                    logger.debug("Skipping Modality='%s': %s", modality, path.name)

            except Exception as exc:
                logger.debug("Could not read '%s' as DICOM: %s", path.name, exc)

        # Guard: all three modalities required
        missing = []
        if not ct_series:
            missing.append("CT image series")
        if rt_struct is None:
            missing.append("RTSTRUCT (RS file)")
        if rt_dose is None:
            missing.append("RTDOSE (RD file)")
        if missing:
            raise DICOMIngestionError(
                "Required DICOM modalities not found: " + ", ".join(missing)
            )

        # Sort CT slices by z-position (ascending; superior slices last for LPS)
        ct_series.sort(key=lambda d: float(d.ImagePositionPatient[2]))

        self.dataset.ct_series = ct_series
        self.dataset.rt_struct = rt_struct
        self.dataset.rt_dose = rt_dose

        logger.info(
            "Classified: %d CT slices | 1 RTSTRUCT | 1 RTDOSE", len(ct_series)
        )

    # ------------------------------------------------------------------
    # Step 3 — FrameOfReferenceUID validation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_frame_of_reference_uid(ds: Dataset) -> str:
        """Extract FrameOfReferenceUID from any DICOM modality.

        For CT and RTDOSE the UID is a top-level attribute (0020,0052).
        For RTSTRUCT it is nested inside ReferencedFrameOfReferenceSequence
        (the top-level tag is not required by the standard for RT Structure Sets).

        Raises:
            DICOMIngestionError: If the UID cannot be located.
        """
        # Top-level (CT, RTDOSE, and some RTSTRUCT)
        uid = getattr(ds, "FrameOfReferenceUID", None)
        if uid:
            return str(uid)

        # RTSTRUCT: UID lives in ReferencedFrameOfReferenceSequence
        refs = getattr(ds, "ReferencedFrameOfReferenceSequence", None)
        if refs and len(refs) > 0:
            uid = getattr(refs[0], "FrameOfReferenceUID", None)
            if uid:
                return str(uid)

        raise DICOMIngestionError(
            f"Cannot determine FrameOfReferenceUID for "
            f"{getattr(ds, 'Modality', 'unknown')} dataset "
            f"(SOPInstanceUID={getattr(ds, 'SOPInstanceUID', 'unknown')})"
        )

    def _validate_frame_of_reference(self) -> None:
        """Assert that CT, RS, and RD share an identical FrameOfReferenceUID.

        A mismatch indicates that the files belong to different studies or
        were exported incorrectly, and the spatial relationship between dose
        and structures cannot be guaranteed.

        Raises:
            DICOMIngestionError: On any UID mismatch.
        """
        ref_uid = self._get_frame_of_reference_uid(self.dataset.ct_series[0])

        # Spot-check CT (first + last slice; full check would be slow for 500+ slices)
        for i in [0, len(self.dataset.ct_series) - 1]:
            uid = self._get_frame_of_reference_uid(self.dataset.ct_series[i])
            if uid != ref_uid:
                raise DICOMIngestionError(
                    f"CT slice {i} has FrameOfReferenceUID={uid!r} "
                    f"(expected {ref_uid!r})"
                )

        rs_uid = self._get_frame_of_reference_uid(self.dataset.rt_struct)
        if rs_uid != ref_uid:
            raise DICOMIngestionError(
                f"RTSTRUCT FrameOfReferenceUID={rs_uid!r} "
                f"does not match CT={ref_uid!r}"
            )

        rd_uid = self._get_frame_of_reference_uid(self.dataset.rt_dose)
        if rd_uid != ref_uid:
            raise DICOMIngestionError(
                f"RTDOSE FrameOfReferenceUID={rd_uid!r} "
                f"does not match CT={ref_uid!r}"
            )

        self.dataset.frame_of_reference_uid = ref_uid
        logger.info("FrameOfReferenceUID validated: %s", ref_uid)

    # ------------------------------------------------------------------
    # Step 4 — Dose array extraction
    # ------------------------------------------------------------------

    def _extract_dose_array(self) -> None:
        """Extract and scale the RTDOSE pixel data to physical dose in Gy.

        DICOM RT Dose stores dose as unsigned integer pixel values.  The true
        dose is obtained by multiplying by the DoseGridScaling tag (Gy per
        pixel value).

        Tag:  DoseGridScaling (3004, 000E)

        Raises:
            DICOMIngestionError: If pixel data or scaling tag is absent.
        """
        rd = self.dataset.rt_dose

        if not hasattr(rd, "PixelData"):
            raise DICOMIngestionError(
                "RTDOSE file contains no PixelData — file may be incomplete."
            )
        if not hasattr(rd, "DoseGridScaling"):
            raise DICOMIngestionError(
                "RTDOSE file is missing the DoseGridScaling (3004,000E) tag."
            )

        scaling = float(rd.DoseGridScaling)          # Gy / integer_value
        pixel_array = rd.pixel_array                 # uint32, shape: (n_slices, n_rows, n_cols)

        self.dataset.dose_array_gy = pixel_array.astype(np.float64) * scaling

        logger.info(
            "Dose array: shape=%s | max=%.4f Gy | scaling=%.6e Gy/value",
            self.dataset.dose_array_gy.shape,
            self.dataset.dose_array_gy.max(),
            scaling,
        )

    # ------------------------------------------------------------------
    # Step 5 — Coordinate system
    # ------------------------------------------------------------------

    def _extract_coordinate_system(self) -> None:
        """Build a :class:`CoordinateSystem` from the RTDOSE header.

        Relevant DICOM tags:
            - ImagePositionPatient  (0020,0032): [x, y, z] of voxel (0,0,0) in mm
            - PixelSpacing          (0028,0030): [row_spacing, col_spacing] in mm
            - GridFrameOffsetVector (3004,000C): z-offset of each dose plane (mm)
        """
        rd = self.dataset.rt_dose

        image_position = np.array(rd.ImagePositionPatient, dtype=np.float64)
        pixel_spacing = np.array(rd.PixelSpacing, dtype=np.float64)
        grid_frame_offsets = np.array(rd.GridFrameOffsetVector, dtype=np.float64)

        n_slices, n_rows, n_cols = self.dataset.dose_array_gy.shape

        self.dataset.coordinate_system = CoordinateSystem(
            image_position=image_position,
            pixel_spacing=pixel_spacing,
            grid_frame_offsets=grid_frame_offsets,
            n_rows=n_rows,
            n_cols=n_cols,
            n_slices=n_slices,
        )

        logger.info(
            "Coordinate system: origin=[%.1f, %.1f, %.1f] mm | "
            "pixel_spacing=[%.2f, %.2f] mm | z_range=[%.1f, %.1f] mm",
            *image_position,
            *pixel_spacing,
            grid_frame_offsets[0],
            grid_frame_offsets[-1],
        )

    # ------------------------------------------------------------------
    # Step 6 — Structure mask extraction
    # ------------------------------------------------------------------

    def _extract_structure_masks(self) -> None:
        """Rasterize RT Structure contours onto the dose voxel grid.

        Only structures listed in STRUCTURES_OF_INTEREST are loaded.
        Contours are rasterized onto the CT grid by rt_utils, then
        resampled to the dose grid using patient-coordinate mapping
        (nearest-neighbour) for spatial accuracy.
        """
        rs_path = self.dataset.rt_struct.filename
        ct_dir = str(Path(self.dataset.ct_series[0].filename).parent)

        try:
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=ct_dir,
                rt_struct_path=rs_path,
            )
        except Exception as exc:
            raise DICOMIngestionError(
                f"rt_utils failed to load the RT Structure Set: {exc}"
            ) from exc

        all_roi_names = rtstruct.get_roi_names()
        # Exact-name filter — avoids loading 39 structures when only 7 are needed
        roi_names = [n for n in all_roi_names if n in STRUCTURES_OF_INTEREST]
        logger.info(
            "Found %d ROI(s); loading %d of interest: %s",
            len(all_roi_names), len(roi_names), roi_names,
        )

        coord = self.dataset.coordinate_system
        ct_series = self.dataset.ct_series

        # CT coordinate info for accurate spatial resampling
        ct_z = np.array([float(ds.ImagePositionPatient[2]) for ds in ct_series])
        first_ct = ct_series[0]
        ct_origin_x = float(first_ct.ImagePositionPatient[0])
        ct_origin_y = float(first_ct.ImagePositionPatient[1])
        ct_ps_row = float(first_ct.PixelSpacing[0])   # mm per row (y-direction)
        ct_ps_col = float(first_ct.PixelSpacing[1])   # mm per col (x-direction)

        masks: dict[str, np.ndarray] = {}

        for name in roi_names:
            try:
                # rt_utils returns (n_rows, n_cols, n_slices) — transpose to (n_slices, n_rows, n_cols)
                ct_mask = np.transpose(rtstruct.get_roi_mask_by_name(name), (2, 0, 1))
                dose_mask = self._resample_mask_to_dose_grid(
                    ct_mask, coord, ct_z,
                    ct_origin_x, ct_origin_y, ct_ps_row, ct_ps_col,
                )
                masks[name] = dose_mask
                logger.info(
                    "  %-20s %7d dose voxels inside contour",
                    f"'{name}':", int(dose_mask.sum()),
                )
            except Exception as exc:
                logger.warning("Could not extract mask for '%s': %s", name, exc)

        self.dataset.structure_masks = masks

    @staticmethod
    def _resample_mask_to_dose_grid(
        ct_mask: np.ndarray,
        coord: CoordinateSystem,
        ct_z: np.ndarray,
        ct_origin_x: float,
        ct_origin_y: float,
        ct_ps_row: float,
        ct_ps_col: float,
    ) -> np.ndarray:
        """Resample a CT-grid boolean mask to the dose grid using patient coordinates.

        For each dose voxel (k, r, c) the corresponding CT voxel is found by
        mapping patient-space coordinates through both grids' ImagePositionPatient
        and PixelSpacing values.  Nearest-neighbour sampling preserves sharp
        binary mask boundaries without introducing fractional values.

        Args:
            ct_mask:      Boolean mask on the CT voxel grid, shape (n_ct_z, n_ct_r, n_ct_c).
            coord:        Dose coordinate system (target grid).
            ct_z:         Sorted z-positions [mm] of each CT slice (len = n_ct_z).
            ct_origin_x:  ImagePositionPatient[0] of first CT slice [mm].
            ct_origin_y:  ImagePositionPatient[1] of first CT slice [mm].
            ct_ps_row:    CT PixelSpacing[0] — mm per row step (y-direction).
            ct_ps_col:    CT PixelSpacing[1] — mm per col step (x-direction).

        Returns:
            Boolean mask on the dose grid, shape (n_slices, n_rows, n_cols).
        """
        n_ct_z, n_ct_row, n_ct_col = ct_mask.shape
        dose_z = coord.slice_positions_z

        # Precompute dose→CT column and row index maps (same for every z-slice)
        dc = np.arange(coord.n_cols)
        dr = np.arange(coord.n_rows)
        x_dose = coord.image_position[0] + dc * coord.pixel_spacing[1]   # x at each dose col
        y_dose = coord.image_position[1] + dr * coord.pixel_spacing[0]   # y at each dose row

        ct_col_idx = np.clip(
            np.round((x_dose - ct_origin_x) / ct_ps_col).astype(int), 0, n_ct_col - 1
        )
        ct_row_idx = np.clip(
            np.round((y_dose - ct_origin_y) / ct_ps_row).astype(int), 0, n_ct_row - 1
        )

        result = np.zeros((coord.n_slices, coord.n_rows, coord.n_cols), dtype=bool)
        for k, z_d in enumerate(dose_z):
            ct_z_idx = int(np.argmin(np.abs(ct_z - z_d)))
            # np.ix_ creates a 2-D index grid: result[k,r,c] = ct_mask[ct_z_idx, row[r], col[c]]
            result[k] = ct_mask[ct_z_idx][np.ix_(ct_row_idx, ct_col_idx)]

        return result
