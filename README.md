# GammaTile EQD2 Converter

An interactive DICOM-RT viewer and radiobiological dose converter for **Cs-131 GammaTile** permanent brain brachytherapy implants.  Converts physical dose distributions to **EQD2** (2 Gy equivalent dose) using the Linear-Quadratic model with the Lea-Catcheside G-factor correction for low dose-rate permanent implants.

---

> **⚠️ FOR QUALIFIED USE ONLY**
>
> This software is intended exclusively for use by qualified medical physicists, radiation oncologists, and other licensed healthcare professionals with training in radiobiology and treatment planning.  It is a **research and evaluation tool** — outputs must be independently verified before any clinical use.
>
> **USE AT YOUR OWN RISK.** The authors provide no warranty of any kind, express or implied. This tool is not FDA-cleared or CE-marked and is not approved for clinical decision-making.

---

## Radiobiological Method

### Linear-Quadratic (LQ) Model

The biological effect of a radiation dose is modelled by the Linear-Quadratic framework:

$$\text{BED} = D \left(1 + \frac{G \cdot D}{\alpha/\beta}\right)$$

where:
- **D** — total physical dose (Gy)
- **α/β** — tissue-specific radiosensitivity ratio (Gy)
- **G** — Lea-Catcheside dose-protraction factor (dimensionless)

### Lea-Catcheside G-Factor for Permanent Implants

For a **permanent implant** with exponential dose-rate decay, the G-factor accounts for sublethal damage repair occurring *during* dose delivery:

$$G = \frac{\lambda}{\lambda + \mu}$$

where:
- **λ** — radioactive decay constant = ln(2) / T½ ; for Cs-131, T½ = 9.7 days
- **μ** — sublethal damage repair rate = ln(2) / T_rep ; default T_rep = 1.5 h

As λ → 0 (very long half-life) G → 0, meaning repair is complete between fractions.  As μ → 0 (no repair) G → 1, recovering the acute BED formula.  For Cs-131 (short half-life relative to tissue repair), G ≈ 0.003–0.006, substantially reducing the biological effect compared to an acute dose.

### EQD2 Conversion

The EQD2 (biologically equivalent dose in 2 Gy fractions) is derived from the BED:

$$\text{EQD2} = \frac{\text{BED}}{1 + \frac{2}{\alpha/\beta}} = \frac{D\left(1 + G \cdot D / (\alpha/\beta)\right)}{1 + 2/(\alpha/\beta)}$$

### Per-Structure α/β Assignment

EQD2 is computed **per voxel** using structure-specific α/β values with the following priority hierarchy (highest wins):

| Structure | α/β (Gy) | Rationale |
|-----------|-----------|-----------|
| CTV | — | Surgical cavity — no tissue, EQD2 = 0 |
| PTV | 10.0 | Tumour / high-proliferating tissue |
| Brainstem | 3.0 | Late-responding CNS tissue |
| OpticChiasm | 3.0 | Late-responding CNS tissue |
| OpticNrv_L / R | 3.0 | Late-responding CNS tissue |
| Brain | 3.0 | Late-responding CNS tissue |
| Uncontoured | 3.0 | Conservative default |

Sub-structures that lie within Brain (e.g. Brainstem, Optic structures) take precedence over the Brain α/β in overlapping voxels.  All α/β values are user-adjustable in the GUI.

---

## Features

- **DICOM-RT ingestion** — reads CT series, RTSTRUCT, and RTDOSE from a directory
- **Coordinate-based structure resampling** — maps structure contours from CT grid to dose grid using patient-space coordinates (no interpolation artefacts)
- **Interactive orthoviewer** — linked Axial (large, left), Coronal and Sagittal (stacked, right) panels with dose overlay (jet colourmap) and structure contours
- **Per-structure α/β controls** — adjust each structure's radiosensitivity; EQD2 volume recomputes instantly
- **Structure snap** — click a structure name to centre all three views on its geometric centroid
- **Cursor readout** — hover over any dose panel to read physical dose, EQD2, structure, and α/β at that voxel
- **DICOM RT Dose export** — saves the EQD2 volume as a new RTDOSE file, preserving all spatial metadata from the original plan

---

## Installation

### Prerequisites

- Python ≥ 3.11
- Dependencies: `pydicom`, `rt-utils`, `numpy`, `scipy`, `PySide6`, `matplotlib`

### Install from source

```bash
git clone https://github.com/hwsalmon/GammaTile-EQD2-Converter.git
cd GammaTile-EQD2-Converter
pip install -e .
```

The `gammatile-eqd2` command will be available on your PATH.

### Launch

```bash
gammatile-eqd2
```

Or directly:

```bash
python viewer.py
```

On Linux, a `.desktop` entry can be installed for the application menu — see `launch.sh` for the LD_LIBRARY_PATH wrapper needed on some systems.

---

## Usage

1. **File → Open Directory** (Ctrl+O) — select a folder containing CT, RTSTRUCT, and RTDOSE DICOM files
2. Structures load automatically; adjust **α/β** values per structure in the Parameters panel
3. Change **T_rep** (sublethal repair half-time) and click **Apply** to recompute G and EQD2
4. Toggle **EQD2 / Physical** display mode; adjust dose window and CT window level
5. Click a **structure name** to snap all views to its centroid
6. Hover over any panel to read dose/EQD2/structure values in the Cursor Readout box
7. **File → Export EQD2 RTDOSE** (Ctrl+E) or click the **Export** button to save the EQD2 volume as a DICOM RT Dose file

---

## File Structure

| File | Description |
|------|-------------|
| `viewer.py` | PySide6 GUI — orthoviewer, parameter controls, structure panel |
| `io_manager.py` | DICOM-RT ingestion, structure rasterisation, coordinate mapping |
| `physics_engine.py` | LQ model, G-factor computation, EQD2 conversion |
| `exporter.py` | DICOM RT Dose export |
| `main.py` | CLI entry point |
| `launch.sh` | Shell launcher (sets LD_LIBRARY_PATH for PySide6 on Linux) |

---

## References

- Dale R.G. (1985). *The application of the linear-quadratic dose-effect equation to fractionated and protracted radiotherapy.* Br J Radiol. 58:515–528.
- Lea & Catcheside (1942). *The mechanism of the induction by radiation of chromosome aberrations in Tradescantia.* J Genet. 44:216–245.
- Astrahan M.A. (2008). *Some implications of linear-quadratic-linear radiation dose-response with regard to hypofractionation.* Med Phys. 35(9):4161–4172.
