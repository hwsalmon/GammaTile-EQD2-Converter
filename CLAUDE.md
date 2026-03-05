# GammaTile EQD2 Converter — Project Reference

## Overview
A Python desktop application for medical physicists that transforms Cs-131 GammaTile
physical dose distributions (from a TPS DICOM export) into Equivalent Dose in 2 Gy
fractions (EQD2) maps, preserving full DICOM-RT structural integrity for re-import into
a Treatment Planning System.

**Role:** Senior Medical Physics Software Engineer
**Stack:** Python 3.11+, pydicom, rt-utils, numpy, scipy, PySide6, matplotlib

---

## Project Structure

```
GammaTile EQD2 Converter/
├── CLAUDE.md               ← This file
├── requirements.txt
├── main.py                 ← Application entry point
├── io_manager.py           ← Phase 1: DICOM-RT ingestion
├── physics_engine.py       ← Phase 2: Radiobiological engine
├── viewer.py               ← Phase 3: Orthoviewer UI (TODO)
├── exporter.py             ← Phase 4: DICOM-RT export (TODO)
└── Test Data/              ← Local test DICOM dataset
```

---

## Development Phases & Status

| Phase | Module             | Description                          | Status      |
|-------|--------------------|--------------------------------------|-------------|
| 1     | `io_manager.py`    | DICOM-RT load, validate, mask extract | ✅ Complete |
| 2     | `physics_engine.py`| LQ/Lea-Catcheside EQD2 engine        | ✅ Complete |
| 3     | `viewer.py`        | Orthoviewer with dose/structure overlay | 🔲 TODO   |
| 4     | `exporter.py`      | DICOM-RT Dose export                 | 🔲 TODO    |

---

## Physics Reference

### Isotope: Cs-131
- **Half-life:** T½ = 9.7 days (fixed constant — do not allow user override)
- **Decay constant:** λ = ln(2) / (T½ × 24 h) [h⁻¹]

### LQ Model for Permanent Implants (Dale 1985)

**Parameters (user-adjustable):**
- `T_rep` — sublethal damage repair half-time [h], default 1.5 h
- `α/β` — tissue-specific ratio [Gy], default 2.0 Gy (late-responding / prostate)

**Derived constants:**
```
λ = ln(2) / (9.7 × 24)   [h⁻¹]   source decay constant
μ = ln(2) / T_rep         [h⁻¹]   repair rate constant
G = λ / (λ + μ)                    Lea-Catcheside factor (0 < G < 1)
```

**Transformation (voxel-wise):**
```
BED   = D × [1 + G × D / (α/β)]
EQD2  = BED / (1 + 2 / (α/β))
```

Where `D` is the physical dose at each voxel in Gy.

### Key References
- Dale RG (1985). The application of the linear-quadratic dose-effect equation to fractionated
  and protracted radiotherapy. Br J Radiol 58:515–528.
- Brenner DJ, Hall EJ (1991). Conditions for the equivalence of continuous to pulsed low dose
  rate brachytherapy. Int J Radiat Oncol Biol Phys 20:181–190.

---

## DICOM-RT Data Model

### Required Input Files (all must share FrameOfReferenceUID)
| Modality   | DICOM Tag  | Description                    |
|------------|------------|--------------------------------|
| CT         | (0008,0060)| Image series for anatomical ref|
| RTSTRUCT   | (0008,0060)| Structure contours (RS file)   |
| RTDOSE     | (0008,0060)| Physical dose grid (RD file)   |

### Key DICOM Tags
| Tag        | Keyword                    | Used In          |
|------------|----------------------------|------------------|
| (0020,0052)| FrameOfReferenceUID        | Validation       |
| (3004,000C)| GridFrameOffsetVector      | Z-axis mapping   |
| (0020,0032)| ImagePositionPatient       | Dose grid origin |
| (0028,0030)| PixelSpacing               | XY resolution    |
| (3004,000E)| DoseGridScaling            | Gy/pixel_value   |

### Export Requirements (Phase 4)
- Clone original RD header entirely
- Update: `SOPInstanceUID` (new UID), `PixelData` (EQD2 scaled to uint32)
- Retain: `ReferencedRTPlanSequence`, `ReferencedStructureSetSequence`
- Set: `DoseUnits = 'GY'`, `DoseType = 'PHYSICAL'`
- Optionally: `DoseComment = 'EQD2 conversion: Cs-131 GammaTile, LQ model'`

---

## Key Design Decisions

1. **Dose grid as reference frame:** Structure masks are resampled to the RTDOSE voxel
   grid (not CT grid) because EQD2 math operates on dose voxels.

2. **Nearest-neighbour resampling for masks:** Binary structure masks use order=0
   (nearest-neighbour) zoom to preserve sharp contour boundaries.

3. **Float64 for dose arithmetic:** All physics calculations use float64 to prevent
   accumulated rounding error in the LQ multiplication.

4. **G-factor is dataset-independent:** G depends only on λ and μ (source + tissue
   properties), not on dose. It is computed once per parameter set and broadcast
   across all voxels.

5. **rt-utils for contour rasterization:** Avoids re-implementing polygon-to-voxel
   conversion; handles DICOM coordinate transforms internally.

---

## Dependencies

```
pydicom>=2.4.0          # DICOM file I/O
rt-utils>=1.2.7         # RT Structure contour rasterization
numpy>=1.26.0           # Array math
scipy>=1.13.0           # ndimage.zoom for mask resampling
PySide6>=6.7.0          # Qt6 GUI (Phase 3)
matplotlib>=3.9.0       # Fallback/supplementary plotting
```

---

## Development Notes

- Test data lives in `Test Data/` directory.
- Phase 3 viewer must support real-time parameter updates: changing T_rep or α/β
  triggers re-computation on the active slice without requiring a full volume reload.
- Phase 4 exporter must pass DICOM validation (e.g., dciodvfy or Orthanc import).
