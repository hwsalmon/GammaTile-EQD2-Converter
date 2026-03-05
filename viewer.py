"""
viewer.py — Phase 3: Interactive Orthoviewer

PySide6 application with three linked orthogonal slice panels (Axial / Coronal /
Sagittal), a dose heatmap overlay, structure contour outlines, and live
radiobiological parameter controls.

Launch::
    python viewer.py
    python viewer.py --input "Test Data/"
"""

import sys
import logging
import threading
import queue as _queue_module
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates, binary_erosion

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox,
    QScrollArea, QDockWidget, QFileDialog, QProgressBar, QSplitter,
    QComboBox, QMessageBox, QSizePolicy, QFrame,
)
from PySide6.QtGui import QAction, QColor

from io_manager import IOManager, DICOMDataset, DICOMIngestionError
from physics_engine import PhysicsEngine, RadiobiologyParameters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-structure α/β defaults  (None = cavity, excluded from EQD2)
# ---------------------------------------------------------------------------

_STRUCTURE_DEFAULT_AB: dict[str, Optional[float]] = {
    "CTV":        None,    # Cavity — no tissue, excluded from EQD2
    "PTV":        10.0,
    "Brain":       3.0,
    "Brainstem":   3.0,
    "OpticChiasm": 3.0,
    "OpticNrv_L":  3.0,
    "OpticNrv_R":  3.0,
}

# ---------------------------------------------------------------------------
# Color assignments
# ---------------------------------------------------------------------------

_NAMED_COLORS: dict[str, str] = {
    "ctv":            "#FF8C00",
    "ptv":            "#FF0000",
    "brainstem":      "#9400D3",   # must come BEFORE "brain"
    "brain":          "#4169E1",
    "spinalcord":     "#FFFF00",
    "cord":           "#FFFF00",
    "eye":            "#00CED1",
    "opticnrv":       "#7CFC00",
    "opticchiasm":    "#FF00FF",
    "parotid":        "#87CEEB",
    "cochlea":        "#FFB6C1",
    "lens":           "#98FB98",
    "pituitary":      "#FFA07A",
    "ln_":            "#DDA0DD",
    "musc":           "#CD853F",
    "glnd":           "#F0E68C",
    "brachial":       "#E9967A",
}

_PALETTE = [
    "#00FFFF", "#FF69B4", "#32CD32", "#FF6347", "#00CED1",
    "#DDA0DD", "#F0E68C", "#CD853F", "#00FA9A", "#FF4500",
    "#1E90FF", "#FF1493", "#ADFF2F", "#FFA500", "#7B68EE",
    "#48D1CC", "#DA70D6", "#90EE90", "#FF7F50", "#40E0D0",
]


def assign_structure_colors(names: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    palette_idx = 0
    for name in names:
        key = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        matched = False
        for fragment, color in _NAMED_COLORS.items():
            if fragment.replace("_", "") in key:
                colors[name] = color
                matched = True
                break
        if not matched:
            colors[name] = _PALETTE[palette_idx % len(_PALETTE)]
            palette_idx += 1
    return colors


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


class LoadWorker:
    """Load DICOM dataset in a Python daemon thread.

    Results are communicated via a Queue polled by a QTimer in the main thread.
    Using Python threading (not QThread) avoids conflicts between Qt's thread
    lifecycle management and native threads created internally by cv2/OpenCV
    (used by rt_utils for contour rasterization).

    Queue messages: ('progress', str) | ('finished', DICOMDataset) | ('error', str)
    """

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self._q: _queue_module.Queue = _queue_module.Queue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dicom-load"
        )

    def start(self) -> None:
        self._thread.start()

    def poll(self):
        """Non-blocking: return next queued message or None."""
        try:
            return self._q.get_nowait()
        except _queue_module.Empty:
            return None

    def _run(self) -> None:
        try:
            self._q.put(("progress", "Reading DICOM files…"))
            manager = IOManager(self.input_dir)
            dataset = manager.load()
            self._q.put(("finished", dataset))
        except Exception as exc:
            self._q.put(("error", str(exc)))


class CTBuildWorker:
    """Resample CT volume to dose grid in a Python daemon thread.

    Queue messages: ('progress', (k, total)) | ('finished', np.ndarray)
    """

    def __init__(self, dataset: DICOMDataset):
        self.dataset = dataset
        self._q: _queue_module.Queue = _queue_module.Queue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ct-build"
        )

    def start(self) -> None:
        self._thread.start()

    def poll(self):
        try:
            return self._q.get_nowait()
        except _queue_module.Empty:
            return None

    def _run(self) -> None:
        coord = self.dataset.coordinate_system
        ct_series = self.dataset.ct_series

        ct_z = np.array([float(ds.ImagePositionPatient[2]) for ds in ct_series])
        dose_z = coord.slice_positions_z

        first = ct_series[0]
        ct_ps_row = float(first.PixelSpacing[0])
        ct_ps_col = float(first.PixelSpacing[1])
        ct_origin_x = float(first.ImagePositionPatient[0])
        ct_origin_y = float(first.ImagePositionPatient[1])

        dose_ps_row = float(coord.pixel_spacing[0])
        dose_ps_col = float(coord.pixel_spacing[1])
        dose_origin_x = float(coord.image_position[0])
        dose_origin_y = float(coord.image_position[1])

        dc = np.arange(coord.n_cols)
        dr = np.arange(coord.n_rows)
        dose_x = dose_origin_x + dc * dose_ps_col
        dose_y = dose_origin_y + dr * dose_ps_row

        ct_col_coords = (dose_x - ct_origin_x) / ct_ps_col
        ct_row_coords = (dose_y - ct_origin_y) / ct_ps_row

        RR, CC = np.meshgrid(ct_row_coords, ct_col_coords, indexing="ij")

        volume = np.zeros((coord.n_slices, coord.n_rows, coord.n_cols), dtype=np.float32)
        HU_AIR = -1000.0
        MAX_Z_GAP = 5.0

        for k, z in enumerate(dose_z):
            if k % 20 == 0:
                self._q.put(("progress", (k, coord.n_slices)))
            nearest_idx = int(np.argmin(np.abs(ct_z - z)))
            if abs(ct_z[nearest_idx] - z) > MAX_Z_GAP:
                volume[k] = HU_AIR
                continue
            ct_ds = ct_series[nearest_idx]
            pix = ct_ds.pixel_array.astype(np.float32)
            slope     = float(getattr(ct_ds, "RescaleSlope", 1))
            intercept = float(getattr(ct_ds, "RescaleIntercept", -1024))
            hu = pix * slope + intercept
            volume[k] = map_coordinates(
                hu, [RR.ravel(), CC.ravel()],
                order=1, mode="constant", cval=HU_AIR,
            ).reshape(coord.n_rows, coord.n_cols)

        self._q.put(("finished", volume))


# ---------------------------------------------------------------------------
# Display state
# ---------------------------------------------------------------------------


class ViewerState:
    """Mutable display settings owned by MainWindow and shared by all panels."""

    def __init__(self, n_axial: int, n_coronal: int, n_sagittal: int):
        self.axial_idx    = n_axial    // 2
        self.coronal_idx  = n_coronal  // 2
        self.sagittal_idx = n_sagittal // 2

        self.dose_opacity  = 0.55
        self.dose_min_gy   = 0.5
        self.dose_max_gy   = 60.0

        self.show_dose       = True
        self.show_structures = True
        self.display_mode    = "eqd2"   # "eqd2" | "physical"

        self.ct_wc = 40    # window centre (HU) — brain window default
        self.ct_ww = 400   # window width  (HU)

        self.structure_visible: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Computed data container
# ---------------------------------------------------------------------------


class ViewerData:
    """Holds pre-computed display arrays derived from the loaded dataset."""

    # Structure priority order for EQD2 α/β assignment (lower index = lower priority)
    _PRIORITY = ["Brain", "Brainstem", "OpticChiasm", "OpticNrv_L", "OpticNrv_R"]

    def __init__(self, dataset: DICOMDataset, engine: PhysicsEngine):
        self.dataset       = dataset
        self.engine        = engine
        self.dose_physical = dataset.dose_array_gy                          # (Z, R, C)
        self.ct_volume: Optional[np.ndarray] = None                         # set by worker
        self.structure_colors = assign_structure_colors(
            list(dataset.structure_masks.keys())
        )
        # Per-structure α/β — populated from panel defaults; updated by UI
        self.structure_ab: dict[str, float] = {
            k: v for k, v in _STRUCTURE_DEFAULT_AB.items() if v is not None
        }
        self.default_ab: float = engine.params.alpha_beta
        self.dose_eqd2 = self._compute_weighted_eqd2()

    # ------------------------------------------------------------------
    # EQD2 computation
    # ------------------------------------------------------------------

    def _compute_weighted_eqd2(self) -> np.ndarray:
        """Per-voxel EQD2 using structure-priority α/β assignment.

        Priority (highest wins):
          CTV  → 0 Gy (cavity, no tissue)
          PTV  → PTV α/β (default 10)
          Other critical structures → their α/β (default 3)
          Unassigned → self.default_ab
        """
        G    = self.engine.G_factor
        dose = self.dose_physical

        def _eqd2(d: np.ndarray, ab: float) -> np.ndarray:
            return d * (1.0 + G * d / ab) / (1.0 + 2.0 / ab)

        # Start: default α/β for every voxel
        eqd2 = _eqd2(dose, self.default_ab)

        # Apply critical structures (low priority — PTV will overwrite)
        for name in self._PRIORITY:
            mask = self.dataset.structure_masks.get(name)
            ab   = self.structure_ab.get(name)
            if mask is None or ab is None:
                continue
            d = dose[mask]
            eqd2[mask] = _eqd2(d, ab)

        # PTV overwrites all critical structures
        ptv  = self.dataset.structure_masks.get("PTV")
        ptv_ab = self.structure_ab.get("PTV")
        if ptv is not None and ptv_ab is not None:
            d = dose[ptv]
            eqd2[ptv] = _eqd2(d, ptv_ab)

        # CTV = cavity → zero
        ctv = self.dataset.structure_masks.get("CTV")
        if ctv is not None:
            eqd2[ctv] = 0.0

        return eqd2

    def update_eqd2(self, t_rep_hours: float, alpha_beta: float) -> None:
        """Update engine (T_rep → new G) and rebuild weighted EQD2."""
        self.engine.update_parameters(t_rep_hours=t_rep_hours, alpha_beta=alpha_beta)
        self.default_ab = alpha_beta
        self.dose_eqd2  = self._compute_weighted_eqd2()

    def update_structure_ab(self, structure_ab: dict[str, Optional[float]]) -> None:
        """Update per-structure α/β values and rebuild weighted EQD2."""
        self.structure_ab = {k: v for k, v in structure_ab.items() if v is not None}
        self.dose_eqd2    = self._compute_weighted_eqd2()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def get_display_dose(self, state: ViewerState) -> np.ndarray:
        return self.dose_eqd2 if state.display_mode == "eqd2" else self.dose_physical

    # ------------------------------------------------------------------
    # Per-structure statistics
    # ------------------------------------------------------------------

    def compute_structure_stats(
        self, structure_ab: dict[str, Optional[float]]
    ) -> dict[str, dict]:
        """Compute per-structure EQD2 Dmean / Dmax.

        Returns: {name: {"dmean": float|None, "dmax": float|None}}
        """
        G = self.engine.G_factor
        stats: dict[str, dict] = {}
        for name, mask in self.dataset.structure_masks.items():
            ab = structure_ab.get(name)
            if ab is None:
                stats[name] = {"dmean": None, "dmax": None}
                continue
            dose_in = self.dose_physical[mask]
            if dose_in.size == 0:
                stats[name] = {"dmean": 0.0, "dmax": 0.0}
                continue
            eqd2 = dose_in * (1.0 + G * dose_in / ab) / (1.0 + 2.0 / ab)
            stats[name] = {"dmean": float(np.mean(eqd2)), "dmax": float(np.max(eqd2))}
        return stats

    # ------------------------------------------------------------------
    # Cursor voxel info
    # ------------------------------------------------------------------

    def get_voxel_info(self, z: int, r: int, c: int) -> dict:
        """Return physical dose, EQD2, structure name, and α/β at voxel (z,r,c)."""
        dose_val = float(self.dose_physical[z, r, c])
        eqd2_val = float(self.dose_eqd2[z, r, c])

        # Determine which structure "owns" this voxel (priority order)
        struct_name = "—"
        ab: Optional[float] = self.default_ab

        ctv = self.dataset.structure_masks.get("CTV")
        if ctv is not None and ctv[z, r, c]:
            struct_name = "CTV (Cavity)"
            ab          = None
        else:
            ptv = self.dataset.structure_masks.get("PTV")
            if ptv is not None and ptv[z, r, c]:
                struct_name = "PTV"
                ab          = self.structure_ab.get("PTV")
            else:
                # Iterate reversed so sub-structures (Brainstem, Optic) trump Brain
                for name in reversed(self._PRIORITY):
                    m = self.dataset.structure_masks.get(name)
                    if m is not None and m[z, r, c]:
                        struct_name = name
                        ab          = self.structure_ab.get(name)
                        break

        return {
            "dose_physical": dose_val,
            "eqd2":          eqd2_val,
            "structure":     struct_name,
            "alpha_beta":    ab,
        }


# ---------------------------------------------------------------------------
# Ortho slice canvas
# ---------------------------------------------------------------------------


class OrthoCanvas(QWidget):
    """Matplotlib figure embedded in Qt displaying one orthogonal dose plane."""

    slice_changed = Signal(str, int)   # orientation, index
    cursor_info   = Signal(str)        # status-bar text on mouse hover

    _TITLE = {"axial": "Axial", "coronal": "Coronal", "sagittal": "Sagittal"}

    def __init__(self, orientation: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        assert orientation in ("axial", "coronal", "sagittal")
        self.orientation  = orientation
        self.viewer_data: Optional[ViewerData]  = None
        self.viewer_state: Optional[ViewerState] = None

        self._im_ct      = None
        self._im_dose    = None
        self._im_structs = None  # single RGBA imshow for all structure boundaries

        self._setup_ui()

    # ------------------------------------------------------------------ UI

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.fig = Figure(figsize=(4, 4), facecolor="#111111")
        self.fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0.02)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor("#111111")
        self.ax.axis("off")
        self._title_obj = self.ax.set_title(
            self._TITLE[self.orientation], color="white", fontsize=9, pad=3
        )

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, stretch=1)

        # Slice slider row
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.lbl_slice = QLabel("—")
        self.lbl_slice.setFixedWidth(55)
        self.lbl_slice.setAlignment(Qt.AlignCenter)
        self.lbl_slice.setStyleSheet("color:#aaaaaa; font-size:9px;")
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_slice)
        layout.addLayout(row)

        self.slider.valueChanged.connect(self._on_slider)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # Debounce timer: draw structures 80 ms after the last slice change
        self._struct_timer = QTimer(self)
        self._struct_timer.setSingleShot(True)
        self._struct_timer.setInterval(80)
        self._struct_timer.timeout.connect(self._deferred_struct_draw)

    # ------------------------------------------------------------------ Public

    def setup_data(self, data: ViewerData, state: ViewerState) -> None:
        self.viewer_data  = data
        self.viewer_state = state
        n = self._n_slices()
        self.slider.setMaximum(n - 1)
        self._sync_slider()
        self._full_draw()

    def refresh_dose_only(self) -> None:
        """Redraw dose+structure layers without re-reading CT (fast path)."""
        if self.viewer_data is None:
            return
        # Update CT window level without re-rendering the whole background
        if self._im_ct is not None and self.viewer_state is not None:
            s = self.viewer_state
            self._im_ct.set_clim(s.ct_wc - s.ct_ww / 2, s.ct_wc + s.ct_ww / 2)
        self._draw_dose_layer()
        self._draw_structure_layers()
        self.canvas.draw_idle()

    def refresh_full(self) -> None:
        """Full redraw, incl. CT background."""
        if self.viewer_data is None:
            return
        self._sync_slider()
        self._full_draw()

    # ------------------------------------------------------------------ helpers

    def _n_slices(self) -> int:
        if self.viewer_data is None:
            return 1
        d = self.viewer_data.dose_physical
        return {"axial": d.shape[0], "coronal": d.shape[1], "sagittal": d.shape[2]}[
            self.orientation
        ]

    def _idx(self) -> int:
        s = self.viewer_state
        return {"axial": s.axial_idx, "coronal": s.coronal_idx, "sagittal": s.sagittal_idx}[
            self.orientation
        ]

    def _cut(self, vol: np.ndarray, idx: int) -> np.ndarray:
        """Slice `vol` along the panel axis; flip so superior→top."""
        if self.orientation == "axial":
            return vol[idx]
        elif self.orientation == "coronal":
            return vol[:, idx, :][::-1]
        else:
            return vol[:, :, idx][::-1]

    def _full_draw(self) -> None:
        self.ax.cla()
        self.ax.set_facecolor("#111111")
        self.ax.axis("off")
        idx = self._idx()
        n   = self._n_slices()
        self.ax.set_title(
            f"{self._TITLE[self.orientation]}  {idx}/{n-1}",
            color="white", fontsize=9, pad=3,
        )
        self._im_ct = self._im_dose = self._im_structs = None

        # CT background
        if self.viewer_data.ct_volume is not None:
            s = self.viewer_state
            vmin = s.ct_wc - s.ct_ww / 2
            vmax = s.ct_wc + s.ct_ww / 2
            self._im_ct = self.ax.imshow(
                self._cut(self.viewer_data.ct_volume, idx),
                cmap="gray", vmin=vmin, vmax=vmax, origin="upper", aspect="equal",
            )

        self._draw_dose_layer()
        self._draw_structure_layers()
        self.lbl_slice.setText(f"{idx} / {n-1}")
        self.canvas.draw_idle()

    def _draw_dose_layer(self) -> None:
        if self._im_dose is not None:
            try:
                self._im_dose.remove()
            except Exception:
                pass
            self._im_dose = None

        s = self.viewer_state
        if not s.show_dose or self.viewer_data is None:
            return

        idx   = self._idx()
        dose  = self._cut(self.viewer_data.get_display_dose(s), idx)
        masked = np.ma.masked_where(dose < s.dose_min_gy, dose)

        self._im_dose = self.ax.imshow(
            masked, cmap="jet", alpha=s.dose_opacity,
            vmin=s.dose_min_gy, vmax=s.dose_max_gy,
            origin="upper", aspect="equal",
        )

    def _draw_structure_layers(self) -> None:
        if self._im_structs is not None:
            try:
                self._im_structs.remove()
            except Exception:
                pass
            self._im_structs = None

        s = self.viewer_state
        if not s.show_structures or self.viewer_data is None:
            return

        idx = self._idx()
        dose_slice = self._cut(self.viewer_data.dose_physical, idx)
        h, w = dose_slice.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        has_any = False

        for name, mask in self.viewer_data.dataset.structure_masks.items():
            if not s.structure_visible.get(name, True):
                continue
            m_slice = self._cut(mask.astype(bool), idx)
            if not m_slice.any():
                continue
            # Boundary = mask XOR eroded mask (1-pixel outline)
            eroded = binary_erosion(m_slice)
            boundary = m_slice & ~eroded
            if not boundary.any():
                boundary = m_slice  # tiny structure — show filled
            color = self.viewer_data.structure_colors.get(name, "#ffffff")
            try:
                r = int(color[1:3], 16) / 255.0
                g = int(color[3:5], 16) / 255.0
                b = int(color[5:7], 16) / 255.0
            except (ValueError, IndexError):
                r = g = b = 1.0
            rgba[boundary, 0] = r
            rgba[boundary, 1] = g
            rgba[boundary, 2] = b
            rgba[boundary, 3] = 1.0
            has_any = True

        if has_any:
            self._im_structs = self.ax.imshow(
                rgba, origin="upper", aspect="equal",
                interpolation="nearest", zorder=3,
            )

    def _sync_slider(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._idx())
        self.slider.blockSignals(False)

    def _on_slider(self, value: int) -> None:
        if self.viewer_state is None:
            return
        if self.orientation == "axial":
            self.viewer_state.axial_idx = value
        elif self.orientation == "coronal":
            self.viewer_state.coronal_idx = value
        else:
            self.viewer_state.sagittal_idx = value
        # Fast redraw: CT + dose only; structures follow after debounce
        self._fast_draw()
        self._struct_timer.start()
        self.slice_changed.emit(self.orientation, value)

    def _fast_draw(self) -> None:
        """Redraw CT and dose only (no structures) — used while scrolling."""
        self.ax.cla()
        self.ax.set_facecolor("#111111")
        self.ax.axis("off")
        idx = self._idx()
        n   = self._n_slices()
        self.ax.set_title(
            f"{self._TITLE[self.orientation]}  {idx}/{n-1}",
            color="white", fontsize=9, pad=3,
        )
        self._im_ct = self._im_dose = self._im_structs = None
        if self.viewer_data is None:
            self.canvas.draw_idle()
            return
        if self.viewer_data.ct_volume is not None:
            s = self.viewer_state
            vmin = s.ct_wc - s.ct_ww / 2
            vmax = s.ct_wc + s.ct_ww / 2
            self._im_ct = self.ax.imshow(
                self._cut(self.viewer_data.ct_volume, idx),
                cmap="gray", vmin=vmin, vmax=vmax, origin="upper", aspect="equal",
            )
        self._draw_dose_layer()
        self.lbl_slice.setText(f"{idx} / {n-1}")
        self.canvas.draw_idle()

    def _deferred_struct_draw(self) -> None:
        """Draw structure boundaries after slider settles (debounced)."""
        if self.viewer_data is None:
            return
        self._draw_structure_layers()
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if self.viewer_state is None:
            return
        delta = -1 if event.button == "up" else 1
        new_val = max(0, min(self._n_slices() - 1, self._idx() + delta))
        self.slider.setValue(new_val)

    def _on_mouse_move(self, event) -> None:
        if event.inaxes != self.ax or self.viewer_data is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        d = self.viewer_data.dose_physical
        n_z, n_r, n_c = d.shape

        col_f = event.xdata
        row_f = event.ydata

        # Convert display coords → dose grid (z, r, c)
        if self.orientation == "axial":
            z = self._idx()
            r = int(round(row_f))
            c = int(round(col_f))
        elif self.orientation == "coronal":
            # _cut flips z: display_row = n_z-1-z  →  z = n_z-1-display_row
            z = n_z - 1 - int(round(row_f))
            r = self._idx()
            c = int(round(col_f))
        else:  # sagittal
            # _cut flips z; display cols = rows of original
            z = n_z - 1 - int(round(row_f))
            r = int(round(col_f))
            c = self._idx()

        # Clip to valid bounds
        z = max(0, min(n_z - 1, z))
        r = max(0, min(n_r - 1, r))
        c = max(0, min(n_c - 1, c))

        info = self.viewer_data.get_voxel_info(z, r, c)
        ab_str = f"{info['alpha_beta']:.1f} Gy" if info['alpha_beta'] is not None else "N/A"
        text = (
            f"Physical:  {info['dose_physical']:.2f} Gy\n"
            f"EQD2:      {info['eqd2']:.2f} Gy\n"
            f"Structure: {info['structure']}\n"
            f"α/β:       {ab_str}"
        )
        self.cursor_info.emit(text)


# ---------------------------------------------------------------------------
# Parameter control panel
# ---------------------------------------------------------------------------


class ParameterPanel(QWidget):
    """Left-side dock: T_rep, per-structure α/β, and display controls (unified)."""

    params_changed     = Signal(float, float)   # (t_rep_hours, 3.0 fixed default)
    display_changed    = Signal()
    ab_changed         = Signal(str, float)     # forwarded from structure section
    visibility_changed = Signal()               # forwarded from structure section
    export_requested   = Signal()               # Export EQD2 RTDOSE button clicked
    structure_clicked  = Signal(str)            # forwarded from structure section

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # --- Physics (T_rep only — α/β is per-structure below) ---
        grp_phys = QGroupBox("Radiobiology (Cs-131)")
        grp_phys.setStyleSheet("QGroupBox { font-weight: bold; }")
        fl = QVBoxLayout(grp_phys)
        fl.addWidget(QLabel("T½ = 9.7 days  (fixed)"))
        self._add_row(fl, "T<sub>rep</sub> (h):", self._make_dspin(0.1, 24.0, 1.5, 0.1), "trep")
        self.lbl_G = QLabel("G = —")
        self.lbl_G.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        fl.addWidget(self.lbl_G)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._emit_params)
        fl.addWidget(btn_apply)
        root.addWidget(grp_phys)

        # --- Structure α/β & Visibility (embedded StructurePanel) ---
        grp_struct = QGroupBox("Structures — α/β  |  Dmean / Dmax EQD2 (Gy)")
        grp_struct.setStyleSheet("QGroupBox { font-weight: bold; }")
        sl = QVBoxLayout(grp_struct)
        sl.setContentsMargins(2, 6, 2, 2)
        self._struct_section = StructurePanel()
        self._struct_section.ab_changed.connect(self.ab_changed)
        self._struct_section.visibility_changed.connect(self.visibility_changed)
        self._struct_section.structure_clicked.connect(self.structure_clicked)
        sl.addWidget(self._struct_section)
        root.addWidget(grp_struct, 1)   # stretch=1 → takes available space

        # --- Cursor readout (below structures) ---
        grp_cursor = QGroupBox("Cursor Readout")
        grp_cursor.setStyleSheet("QGroupBox { font-weight: bold; }")
        cl2 = QVBoxLayout(grp_cursor)
        cl2.setContentsMargins(6, 6, 6, 6)
        self._lbl_cursor = QLabel("Hover over a dose panel to read values")
        self._lbl_cursor.setWordWrap(True)
        self._lbl_cursor.setStyleSheet(
            "color: #dddddd; font-size: 10px; font-family: monospace; "
            "background: #1a1a1a; padding: 4px; border-radius: 3px;"
        )
        cl2.addWidget(self._lbl_cursor)
        root.addWidget(grp_cursor)

        # --- Dose display ---
        grp_disp = QGroupBox("Dose Display")
        grp_disp.setStyleSheet("QGroupBox { font-weight: bold; }")
        dl = QVBoxLayout(grp_disp)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["EQD2 (Gy)", "Physical (Gy)"])
        self.combo_mode.currentIndexChanged.connect(lambda _: self.display_changed.emit())
        dl.addWidget(self.combo_mode)
        self.chk_dose = QCheckBox("Show dose overlay")
        self.chk_dose.setChecked(True)
        self.chk_dose.stateChanged.connect(lambda _: self.display_changed.emit())
        dl.addWidget(self.chk_dose)
        self._add_row(dl, "Opacity:",  self._make_dspin(0.0, 1.0,    0.55, 0.05), "opacity")
        self._add_row(dl, "Min (Gy):", self._make_dspin(0.0, 500.0,  10.0, 1.0),  "dmin")
        self._add_row(dl, "Max (Gy):", self._make_dspin(0.1, 5000.0, 60.0, 5.0),  "dmax")
        for w in [self._spins["opacity"], self._spins["dmin"], self._spins["dmax"]]:
            w.valueChanged.connect(lambda _: self.display_changed.emit())
        root.addWidget(grp_disp)

        # --- CT window ---
        grp_ct = QGroupBox("CT Window (HU)")
        grp_ct.setStyleSheet("QGroupBox { font-weight: bold; }")
        cl = QVBoxLayout(grp_ct)
        self._add_row(cl, "Centre:", self._make_dspin(-1000, 3000, 40,  10), "ct_wc")
        self._add_row(cl, "Width:",  self._make_dspin(1,     6000, 400, 50), "ct_ww")
        presets = QHBoxLayout()
        for label, wc, ww in [("Brain", 40, 400), ("Bone", 400, 2000), ("Soft", 50, 350)]:
            btn = QPushButton(label)
            btn.setMaximumWidth(55)
            btn.clicked.connect(lambda checked, c=wc, w=ww: self._apply_ct_preset(c, w))
            presets.addWidget(btn)
        cl.addLayout(presets)
        for w in [self._spins["ct_wc"], self._spins["ct_ww"]]:
            w.valueChanged.connect(lambda _: self.display_changed.emit())
        root.addWidget(grp_ct)

        # Export button
        self._btn_export = QPushButton("Export EQD2 RTDOSE…")
        self._btn_export.setEnabled(False)
        self._btn_export.setStyleSheet(
            "QPushButton { background:#2a5a2a; color:white; font-weight:bold; padding:5px; border-radius:4px; }"
            "QPushButton:hover { background:#3a7a3a; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._btn_export.clicked.connect(self.export_requested)
        root.addWidget(self._btn_export)

    # ------------------------------------------------------------------
    # Structure section delegation
    # ------------------------------------------------------------------

    def populate_structures(self, structure_colors: dict, state: ViewerState) -> None:
        self._struct_section.populate(structure_colors, state)

    def update_structure_stats(
        self, name: str, dmean: Optional[float], dmax: Optional[float]
    ) -> None:
        self._struct_section.update_stats(name, dmean, dmax)

    def get_structure_ab(self) -> dict[str, Optional[float]]:
        return self._struct_section.get_structure_ab()

    def get_visibility(self) -> dict[str, bool]:
        return self._struct_section.get_visibility()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_dspin(self, lo: float, hi: float, val: float, step: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setSingleStep(step)
        sb.setDecimals(2)
        return sb

    def _add_row(self, layout, label: str, widget: QWidget, key: str) -> None:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(72)
        row.addWidget(lbl)
        row.addWidget(widget)
        layout.addLayout(row)
        self._spins[key] = widget

    def _emit_params(self) -> None:
        self.params_changed.emit(self._spins["trep"].value(), 3.0)

    def _apply_ct_preset(self, wc: int, ww: int) -> None:
        self._spins["ct_wc"].setValue(wc)
        self._spins["ct_ww"].setValue(ww)

    # Public read accessors
    @property
    def t_rep_hours(self) -> float:  return self._spins["trep"].value()
    @property
    def alpha_beta(self) -> float:   return 3.0   # default for unstructured tissue
    @property
    def dose_opacity(self) -> float: return self._spins["opacity"].value()
    @property
    def dose_min(self) -> float:     return self._spins["dmin"].value()
    @property
    def dose_max(self) -> float:     return self._spins["dmax"].value()
    @property
    def ct_wc(self) -> float:        return self._spins["ct_wc"].value()
    @property
    def ct_ww(self) -> float:        return self._spins["ct_ww"].value()
    @property
    def show_dose(self) -> bool:     return self.chk_dose.isChecked()
    @property
    def show_structures(self) -> bool: return True   # always on; per-structure toggles control visibility
    @property
    def display_mode(self) -> str:
        return "eqd2" if self.combo_mode.currentIndex() == 0 else "physical"

    def update_G_label(self, G: float) -> None:
        self.lbl_G.setText(f"G = {G:.6f}")

    def set_export_enabled(self, enabled: bool) -> None:
        self._btn_export.setEnabled(enabled)

    def update_cursor(self, text: str) -> None:
        self._lbl_cursor.setText(text.strip())


# ---------------------------------------------------------------------------
# Structure visibility panel
# ---------------------------------------------------------------------------


class StructurePanel(QWidget):
    """Right-side dock: per-structure visibility, α/β spinbox, and EQD2 stats."""

    visibility_changed = Signal()
    ab_changed         = Signal(str, float)   # structure name, new α/β value
    structure_clicked  = Signal(str)          # structure name label clicked → snap views

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._checkboxes:  dict[str, QCheckBox]      = {}
        self._ab_spins:    dict[str, QDoubleSpinBox] = {}
        self._stat_labels: dict[str, QLabel]         = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hdr = QLabel("Structure · α/β · Dmean / Dmax (EQD2 Gy)")
        hdr.setStyleSheet("color:#666; font-size:9px;")
        layout.addWidget(hdr)

        self._scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_widget)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(3)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._scroll_widget)
        layout.addWidget(scroll, 1)

    def populate(self, structure_colors: dict[str, str], state: ViewerState) -> None:
        # Clear previous rows
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checkboxes.clear()
        self._ab_spins.clear()
        self._stat_labels.clear()

        for name, color in structure_colors.items():
            qc = QColor(color)
            r, g, b = qc.red(), qc.green(), qc.blue()
            ab_default = _STRUCTURE_DEFAULT_AB.get(name)

            row = QFrame()
            row.setFrameShape(QFrame.StyledPanel)
            row.setStyleSheet("QFrame { border: 1px solid #333; border-radius: 3px; }")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(4, 3, 4, 3)
            rl.setSpacing(5)

            # Visibility checkbox (colored indicator)
            cb = QCheckBox()
            cb.setChecked(state.structure_visible.get(name, True))
            cb.setStyleSheet(
                f"QCheckBox::indicator:checked   {{ background-color: rgb({r},{g},{b}); "
                f"border: 2px solid rgb({r},{g},{b}); border-radius:2px; }}"
                f"QCheckBox::indicator:unchecked {{ border: 2px solid rgb({r},{g},{b}); "
                f"border-radius:2px; background:#1e1e1e; }}"
            )
            cb.stateChanged.connect(lambda v, n=name: self._on_toggle(n, bool(v)))
            rl.addWidget(cb)
            self._checkboxes[name] = cb
            state.structure_visible[name] = cb.isChecked()

            # Structure name — clickable button to snap views to centroid
            btn_name = QPushButton(name)
            btn_name.setFlat(True)
            btn_name.setCursor(Qt.PointingHandCursor)
            btn_name.setStyleSheet(
                f"QPushButton {{ color: #ffffff; font-size:11px; font-weight:bold; border:none; "
                f"text-align:left; padding:0px; }}"
                f"QPushButton:hover {{ color: rgb({r},{g},{b}); text-decoration: underline; }}"
            )
            btn_name.setMinimumWidth(80)
            btn_name.clicked.connect(lambda checked=False, n=name: self.structure_clicked.emit(n))
            rl.addWidget(btn_name, 1)

            if ab_default is None:
                # CTV — cavity, no EQD2
                lbl_cav = QLabel("Cavity")
                lbl_cav.setStyleSheet("color:#555; font-size:9px; font-style:italic; border:none;")
                rl.addWidget(lbl_cav)
            else:
                # α/β spinbox
                lbl_ab = QLabel("α/β:")
                lbl_ab.setStyleSheet("color:#888; font-size:9px; border:none;")
                rl.addWidget(lbl_ab)
                spin = QDoubleSpinBox()
                spin.setRange(0.1, 20.0)
                spin.setValue(ab_default)
                spin.setSingleStep(0.5)
                spin.setDecimals(1)
                spin.setFixedWidth(52)
                spin.valueChanged.connect(lambda v, n=name: self.ab_changed.emit(n, v))
                rl.addWidget(spin)
                self._ab_spins[name] = spin

            # Stats label (Dmean / Dmax)
            stat = QLabel("—")
            stat.setStyleSheet("color:#777; font-size:9px; border:none;")
            stat.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            stat.setMinimumWidth(80)
            rl.addWidget(stat)
            self._stat_labels[name] = stat

            self._scroll_layout.addWidget(row)

        self._scroll_layout.addStretch()

    def update_stats(self, name: str, dmean: Optional[float], dmax: Optional[float]) -> None:
        lbl = self._stat_labels.get(name)
        if lbl is None:
            return
        if dmean is None:
            lbl.setText("Cavity")
        else:
            lbl.setText(f"{dmean:.1f} / {dmax:.1f}")

    def get_structure_ab(self) -> dict[str, Optional[float]]:
        result: dict[str, Optional[float]] = {}
        for name in self._checkboxes:
            result[name] = self._ab_spins[name].value() if name in self._ab_spins else None
        return result

    def _on_toggle(self, name: str, visible: bool) -> None:
        self.visibility_changed.emit()

    def get_visibility(self) -> dict[str, bool]:
        return {name: cb.isChecked() for name, cb in self._checkboxes.items()}


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self, initial_dir: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("GammaTile EQD2 Converter")
        self.resize(1400, 900)

        self.viewer_data:  Optional[ViewerData]  = None
        self.viewer_state: Optional[ViewerState] = None
        self._load_worker:   Optional[LoadWorker]   = None
        self._ct_worker:     Optional[CTBuildWorker] = None
        self._is_loading: bool = False

        # Poll timers — drive worker progress without Qt signals from threads
        self._load_poll_timer = QTimer(self)
        self._load_poll_timer.setInterval(200)
        self._load_poll_timer.timeout.connect(self._poll_load_worker)

        self._ct_poll_timer = QTimer(self)
        self._ct_poll_timer.setInterval(200)
        self._ct_poll_timer.timeout.connect(self._poll_ct_worker)

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._apply_param_update)

        self._build_ui()
        self._apply_dark_theme()

        if initial_dir:
            QTimer.singleShot(200, lambda: self._start_load(initial_dir))

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # Menu
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        self._act_open = QAction("&Open Directory…", self)
        self._act_open.setShortcut("Ctrl+O")
        self._act_open.triggered.connect(self._on_open)
        file_menu.addAction(self._act_open)

        self._act_export = QAction("&Export EQD2 RTDOSE…", self)
        self._act_export.setShortcut("Ctrl+E")
        self._act_export.setEnabled(False)
        self._act_export.triggered.connect(self._on_export_eqd2)
        file_menu.addAction(self._act_export)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Central widget — layout:
        #   Left (2/3):  large Axial panel (full height)
        #   Right (1/3): Coronal (top) + Sagittal (bottom) stacked vertically
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)

        self.panel_axial    = OrthoCanvas("axial",    self)
        self.panel_coronal  = OrthoCanvas("coronal",  self)
        self.panel_sagittal = OrthoCanvas("sagittal", self)

        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.slice_changed.connect(self._on_slice_changed)

        # Right column: Coronal over Sagittal
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self.panel_coronal,  1)
        right_layout.addWidget(self.panel_sagittal, 1)

        # Horizontal splitter: Axial (left, large) | right column
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.addWidget(self.panel_axial)
        h_splitter.addWidget(right_widget)
        h_splitter.setStretchFactor(0, 2)   # Axial gets ~2/3
        h_splitter.setStretchFactor(1, 1)   # Right column gets ~1/3
        h_splitter.setSizes([700, 350])
        h_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(h_splitter)

        # Left dock — parameters (includes embedded structure α/β panel)
        self.param_panel = ParameterPanel()
        self.param_panel.params_changed.connect(self._on_params_changed)
        self.param_panel.display_changed.connect(self._on_display_changed)
        self.param_panel.ab_changed.connect(self._on_structure_ab_changed)
        self.param_panel.visibility_changed.connect(self._on_visibility_changed)
        self.param_panel.export_requested.connect(self._on_export_eqd2)
        self.param_panel.structure_clicked.connect(self._on_structure_snap)

        # Connect cursor signals once at startup (not per-load) to avoid duplicates
        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.cursor_info.connect(self.param_panel.update_cursor)

        left_dock = QDockWidget("Parameters", self)
        left_dock.setWidget(self.param_panel)
        left_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        left_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # Status bar
        self.status_bar = self.statusBar()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready — open a DICOM directory with File → Open Directory")

    # ------------------------------------------------------------------ Load

    def _on_open(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open DICOM Directory", str(Path.home()))
        if d:
            self._start_load(d)

    def _start_load(self, directory: str) -> None:
        # Stop any in-progress load poll (old daemon thread completes harmlessly)
        self._load_poll_timer.stop()
        self._ct_poll_timer.stop()
        self._is_loading = True

        self.status_bar.showMessage(f"Loading: {directory} — please wait…")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self._load_worker = LoadWorker(directory)
        self._load_worker.start()
        self._load_poll_timer.start()

    # ---- polling callbacks (called from main thread via QTimer) ----

    def _poll_load_worker(self) -> None:
        if self._load_worker is None:
            self._load_poll_timer.stop()
            return
        msg = self._load_worker.poll()
        if msg is None:
            return
        kind, data = msg
        if kind == "progress":
            self.status_bar.showMessage(data)
        elif kind == "finished":
            self._load_poll_timer.stop()
            self._on_dataset_loaded(data)
        elif kind == "error":
            self._load_poll_timer.stop()
            self._finish_loading(success=False)
            QMessageBox.critical(self, "Load Error", data)

    def _poll_ct_worker(self) -> None:
        if self._ct_worker is None:
            self._ct_poll_timer.stop()
            return
        while True:   # drain all queued progress messages at once
            msg = self._ct_worker.poll()
            if msg is None:
                break
            kind, data = msg
            if kind == "progress":
                k, total = data
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(k)
            elif kind == "finished":
                self._ct_poll_timer.stop()
                self.viewer_data.ct_volume = data
                self._finish_loading(success=True)
                for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
                    panel.refresh_full()
                break

    def _finish_loading(self, success: bool) -> None:
        self._is_loading = False
        self.progress_bar.setVisible(False)
        if success:
            self.status_bar.showMessage("Ready — CT background loaded")

    def _on_dataset_loaded(self, dataset: DICOMDataset) -> None:
        self.status_bar.showMessage("Building EQD2 volume…")

        params = RadiobiologyParameters(
            t_rep_hours=self.param_panel.t_rep_hours,
            alpha_beta=self.param_panel.alpha_beta,
        )
        engine = PhysicsEngine(params)

        self.viewer_data = ViewerData(dataset, engine)
        coord = dataset.coordinate_system
        self.viewer_state = ViewerState(coord.n_slices, coord.n_rows, coord.n_cols)

        for name in dataset.structure_masks:
            self.viewer_state.structure_visible[name] = True

        # Dose display max fixed at 120 Gy
        self.viewer_state.dose_max_gy = 120.0
        self.param_panel._spins["dmax"].setValue(120.0)

        # Centre all three views on the global EQD2 maximum voxel
        peak_z, peak_r, peak_c = (
            int(i) for i in np.unravel_index(
                np.argmax(self.viewer_data.dose_eqd2), self.viewer_data.dose_eqd2.shape
            )
        )
        self.viewer_state.axial_idx    = peak_z
        self.viewer_state.coronal_idx  = peak_r
        self.viewer_state.sagittal_idx = peak_c

        self.param_panel.update_G_label(engine.G_factor)
        self.param_panel.populate_structures(self.viewer_data.structure_colors, self.viewer_state)
        self._update_structure_stats()

        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.setup_data(self.viewer_data, self.viewer_state)

        self._act_export.setEnabled(True)
        self.param_panel.set_export_enabled(True)

        self.status_bar.showMessage(
            f"Loaded {len(dataset.ct_series)} CT slices | "
            f"{coord.n_slices}×{coord.n_rows}×{coord.n_cols} dose grid | "
            "Building CT background volume…"
        )
        self.progress_bar.setRange(0, coord.n_slices)
        self.progress_bar.setVisible(True)

        self._ct_worker = CTBuildWorker(dataset)
        self._ct_worker.start()
        self._ct_poll_timer.start()

    # ------------------------------------------------------------------ Updates

    def _on_slice_changed(self, orientation: str, idx: int) -> None:
        """When one panel scrolls, update crosshair-linked views (same z-slice)."""
        pass  # Future: link coronal/sagittal positions for crosshair

    def _sync_state_from_controls(self) -> None:
        if self.viewer_state is None:
            return
        s = self.viewer_state
        p = self.param_panel
        s.dose_opacity   = p.dose_opacity
        s.dose_min_gy    = p.dose_min
        s.dose_max_gy    = p.dose_max
        s.show_dose      = p.show_dose
        s.show_structures = p.show_structures
        s.display_mode   = p.display_mode
        s.ct_wc          = p.ct_wc
        s.ct_ww          = p.ct_ww
        s.structure_visible.update(self.param_panel.get_visibility())

    def _on_display_changed(self) -> None:
        """Any display property changed — redraw dose layers."""
        self._sync_state_from_controls()
        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.refresh_dose_only()

    def _on_params_changed(self, t_rep: float, ab: float) -> None:
        """T_rep or default α/β changed — update G-factor and rebuild weighted EQD2."""
        if self.viewer_data is None:
            return
        self.status_bar.showMessage("Recomputing EQD2…")
        # update_eqd2 sets T_rep (→ new G) and default_ab, then rebuilds weighted map
        self.viewer_data.update_eqd2(t_rep, ab)
        self.param_panel.update_G_label(self.viewer_data.engine.G_factor)
        self._sync_state_from_controls()
        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.refresh_dose_only()
        self._update_structure_stats()
        self.status_bar.showMessage("Ready")

    def _on_structure_ab_changed(self, name: str, ab: float) -> None:
        """A per-structure α/β spinbox changed — rebuild EQD2 and refresh."""
        if self.viewer_data is None:
            return
        self.viewer_data.update_structure_ab(self.param_panel.get_structure_ab())
        self._sync_state_from_controls()
        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.refresh_dose_only()
        self._update_structure_stats()

    def _update_structure_stats(self) -> None:
        """Recompute per-structure EQD2 Dmean/Dmax and push to panel."""
        if self.viewer_data is None:
            return
        structure_ab = self.param_panel.get_structure_ab()
        stats = self.viewer_data.compute_structure_stats(structure_ab)
        for name, s in stats.items():
            self.param_panel.update_structure_stats(name, s["dmean"], s["dmax"])

    def _on_export_eqd2(self) -> None:
        """Export the current EQD2 volume as a DICOM RT Dose file."""
        if self.viewer_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export EQD2 RTDOSE", "EQD2_dose.dcm",
            "DICOM files (*.dcm);;All files (*)"
        )
        if not path:
            return
        try:
            from exporter import export_eqd2_rtdose
            metadata = {
                "t_rep_hours": self.viewer_data.engine.params.t_rep_hours,
                "G_factor":    self.viewer_data.engine.G_factor,
                "structure_ab": self.viewer_data.structure_ab,
            }
            export_eqd2_rtdose(
                self.viewer_data.dataset.rt_dose,
                self.viewer_data.dose_eqd2,
                path,
                metadata,
            )
            self.status_bar.showMessage(f"Exported: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _apply_param_update(self) -> None:
        self._on_params_changed(
            self.param_panel.t_rep_hours,
            self.param_panel.alpha_beta,
        )

    def _on_visibility_changed(self) -> None:
        self._sync_state_from_controls()
        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.refresh_dose_only()

    def _on_structure_snap(self, name: str) -> None:
        """Snap all three views to the centroid of the named structure's mask."""
        if self.viewer_data is None or self.viewer_state is None:
            return
        mask = self.viewer_data.dataset.structure_masks.get(name)
        if mask is None or not np.any(mask):
            return
        coords = np.argwhere(mask)
        cz, cr, cc = coords.mean(axis=0)
        self.viewer_state.axial_idx    = int(round(cz))
        self.viewer_state.coronal_idx  = int(round(cr))
        self.viewer_state.sagittal_idx = int(round(cc))
        for panel in (self.panel_axial, self.panel_coronal, self.panel_sagittal):
            panel.refresh_full()

    # ------------------------------------------------------------------ Style

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget, QDockWidget, QGroupBox { background-color: #1e1e1e; color: #dcdcdc; }
            QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 8px; padding: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
            QPushButton { background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px;
                          padding: 3px 8px; color: #dcdcdc; }
            QPushButton:hover  { background-color: #505050; }
            QPushButton:pressed { background-color: #2a2a2a; }
            QDoubleSpinBox, QComboBox { background-color: #2e2e2e; color: #dcdcdc;
                                         border: 1px solid #555; border-radius: 3px; padding: 2px; }
            QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #aaa; width: 12px; height: 12px;
                                          margin: -4px 0; border-radius: 6px; }
            QCheckBox { color: #dcdcdc; }
            QScrollArea { border: none; }
            QLabel { color: #dcdcdc; }
            QMenuBar { background-color: #1e1e1e; color: #dcdcdc; }
            QMenuBar::item:selected { background-color: #3a3a3a; }
            QMenu { background-color: #2e2e2e; color: #dcdcdc; border: 1px solid #555; }
            QMenu::item:selected { background-color: #505050; }
            QStatusBar { background-color: #151515; color: #aaaaaa; font-size: 10px; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; background: #2e2e2e;
                           text-align: center; color: #dcdcdc; }
            QProgressBar::chunk { background-color: #4169E1; }
        """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="GammaTile EQD2 Viewer")
    parser.add_argument("--input", "-i", metavar="DIR", help="DICOM directory to open on launch")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("GammaTile EQD2 Converter")
    window = MainWindow(initial_dir=args.input)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
