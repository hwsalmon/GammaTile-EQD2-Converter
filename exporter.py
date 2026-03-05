"""
exporter.py — Phase 4: DICOM-RT Dose Export

Exports the EQD2-converted dose volume as a new RTDOSE DICOM file derived
from the original RTDOSE header.  The pixel data, DoseGridScaling, and
identification tags are updated; all spatial metadata (ImagePositionPatient,
PixelSpacing, GridFrameOffsetVector, FrameOfReferenceUID, etc.) is preserved
so the exported file registers correctly in any treatment planning system.

Usage::

    from exporter import export_eqd2_rtdose
    export_eqd2_rtdose(template_ds, eqd2_array, "EQD2_dose.dcm", metadata)
"""

import copy
import datetime
import logging
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid

logger = logging.getLogger(__name__)


def export_eqd2_rtdose(
    template_ds: pydicom.Dataset,
    eqd2_array: np.ndarray,
    output_path: str | Path,
    metadata: dict,
) -> Path:
    """Write an EQD2 dose volume as a DICOM RT Dose file.

    Args:
        template_ds:  Original RTDOSE Dataset used as spatial/header template.
        eqd2_array:   EQD2 dose volume in Gy, shape (n_slices, n_rows, n_cols).
        output_path:  Destination .dcm path.
        metadata:     Dict with keys:
                        t_rep_hours  — repair half-time used [h]
                        G_factor     — Lea-Catcheside G factor
                        structure_ab — {name: alpha_beta} dict

    Returns:
        Path to the written file.

    Raises:
        ValueError: If eqd2_array shape doesn't match template pixel data shape.
    """
    output_path = Path(output_path)

    # Deep-copy to avoid mutating the live dataset
    ds = copy.deepcopy(template_ds)

    # Verify shape consistency
    expected_shape = (
        int(getattr(ds, "NumberOfFrames", eqd2_array.shape[0])),
        int(ds.Rows),
        int(ds.Columns),
    )
    if eqd2_array.shape != expected_shape:
        raise ValueError(
            f"eqd2_array shape {eqd2_array.shape} does not match "
            f"template RTDOSE shape {expected_shape}"
        )

    # ------------------------------------------------------------------ Identity
    ds.SOPInstanceUID        = generate_uid()
    ds.InstanceCreationDate  = datetime.date.today().strftime("%Y%m%d")
    ds.InstanceCreationTime  = datetime.datetime.now().strftime("%H%M%S.%f")[:14]
    ds.SeriesInstanceUID     = generate_uid()
    ds.SeriesNumber          = str(int(getattr(ds, "SeriesNumber", 1)) + 100)
    ds.SeriesDescription     = "Cs-131 GammaTile EQD2"

    # ------------------------------------------------------------------ Dose metadata
    t_rep   = metadata.get("t_rep_hours", 1.5)
    G       = metadata.get("G_factor", 0.0)
    ab_info = metadata.get("structure_ab", {})
    ab_str  = ", ".join(f"{n}:{v:.1f}" for n, v in ab_info.items())

    ds.DoseComment = (
        f"EQD2 | Cs-131 GammaTile | T_rep={t_rep:.2f}h | G={G:.6f} | "
        f"Per-structure α/β [{ab_str}]"
    )
    ds.DoseType          = "EFFECTIVE"   # biologically effective dose
    ds.DoseSummationType = "PLAN"

    # ------------------------------------------------------------------ Pixel data
    # Scale EQD2 to uint32 with ~0.1 mGy precision
    max_dose = float(np.max(eqd2_array))
    if max_dose > 0.0:
        # Choose scaling so max maps near 2^32-1 for maximum precision
        scaling = max_dose / (2**32 - 1)
    else:
        scaling = 1.0

    pixel_array = np.round(
        np.clip(eqd2_array, 0.0, None) / scaling
    ).astype(np.uint32)

    ds.DoseGridScaling  = float(scaling)
    ds.BitsAllocated    = 32
    ds.BitsStored       = 32
    ds.HighBit          = 31
    ds.PixelRepresentation = 0
    ds.NumberOfFrames   = eqd2_array.shape[0]
    ds.PixelData        = pixel_array.tobytes()

    # Ensure explicit VR little-endian transfer syntax
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID    = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.is_implicit_VR  = False
    ds.is_little_endian = True

    pydicom.dcmwrite(str(output_path), ds, write_like_original=False)

    logger.info(
        "EQD2 RTDOSE exported → %s | max=%.3f Gy | scaling=%.4e Gy/px | shape=%s",
        output_path, max_dose, scaling, eqd2_array.shape,
    )
    return output_path
