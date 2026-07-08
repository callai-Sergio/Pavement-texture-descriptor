"""
settings_panel.py – Left dock widget with all processing & rendering controls
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QSlider, QCheckBox, QSpinBox,
    QLabel, QPushButton,
)


class SettingsPanel(QDockWidget):
    """Dockable settings panel replicating the Streamlit sidebar."""

    def __init__(self, parent=None):
        super().__init__("⚙️ Settings", parent)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.setMinimumWidth(260)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Grid ──────────────────────────────────────────────────
        grp_grid = QGroupBox("📐 Grid Spacing")
        grid_form = QFormLayout(grp_grid)

        self.dx = QDoubleSpinBox()
        self.dx.setRange(0.001, 100.0)
        self.dx.setValue(0.011)
        self.dx.setDecimals(4)
        self.dx.setSuffix(" mm")
        grid_form.addRow("dx:", self.dx)

        self.dy = QDoubleSpinBox()
        self.dy.setRange(0.001, 100.0)
        self.dy.setValue(0.011)
        self.dy.setDecimals(4)
        self.dy.setSuffix(" mm")
        grid_form.addRow("dy:", self.dy)

        self.units_xy = QComboBox()
        self.units_xy.addItems(["mm", "m", "µm"])
        grid_form.addRow("XY units:", self.units_xy)

        self.units_z = QComboBox()
        self.units_z.addItems(["mm", "µm", "m"])
        grid_form.addRow("Z units:", self.units_z)

        self.direction = QComboBox()
        self.direction.addItems(["longitudinal", "transverse"])
        grid_form.addRow("Direction:", self.direction)

        self.every_n = QSpinBox()
        self.every_n.setRange(1, 100)
        self.every_n.setValue(1)
        grid_form.addRow("Every N-th profile:", self.every_n)

        layout.addWidget(grp_grid)

        # ── Preprocessing ─────────────────────────────────────────
        grp_pre = QGroupBox("🔧 Preprocessing")
        pre_form = QFormLayout(grp_pre)

        self.plane_mode = QComboBox()
        self.plane_mode.addItems(["plane", "none", "polynomial"])
        pre_form.addRow("Plane removal:", self.plane_mode)

        self.poly_order = QSpinBox()
        self.poly_order.setRange(1, 5)
        self.poly_order.setValue(2)
        pre_form.addRow("Poly order:", self.poly_order)

        self.outlier_method = QComboBox()
        self.outlier_method.addItems(["hampel", "median", "none"])
        pre_form.addRow("Outlier filter:", self.outlier_method)

        self.outlier_window = QSpinBox()
        self.outlier_window.setRange(3, 21)
        self.outlier_window.setValue(7)
        self.outlier_window.setSingleStep(2)
        pre_form.addRow("Window size:", self.outlier_window)

        self.outlier_thresh = QDoubleSpinBox()
        self.outlier_thresh.setRange(1.0, 6.0)
        self.outlier_thresh.setValue(3.5)
        self.outlier_thresh.setSingleStep(0.5)
        pre_form.addRow("Threshold (σ):", self.outlier_thresh)

        self.interp_gaps = QCheckBox("Interpolate gaps")
        self.interp_gaps.setChecked(True)
        pre_form.addRow(self.interp_gaps)

        self.detrend = QComboBox()
        self.detrend.addItems(["none", "mean", "linear"])
        pre_form.addRow("Detrend:", self.detrend)

        layout.addWidget(grp_pre)

        # ── Rendering ─────────────────────────────────────────────
        grp_render = QGroupBox("🎨 Rendering")
        render_form = QFormLayout(grp_render)

        self.vert_exag_slider = QSlider(Qt.Orientation.Horizontal)
        self.vert_exag_slider.setRange(1, 30)  # 0.1 to 3.0
        self.vert_exag_slider.setValue(3)  # 0.3
        self.vert_exag_label = QLabel("0.3")
        self.vert_exag_slider.valueChanged.connect(
            lambda v: self.vert_exag_label.setText(f"{v / 10:.1f}"))
        render_form.addRow("Vert. exag:", self.vert_exag_slider)
        render_form.addRow("", self.vert_exag_label)

        self.robust_color = QCheckBox("Robust colour (P1–P99)")
        self.robust_color.setChecked(True)
        render_form.addRow(self.robust_color)

        layout.addWidget(grp_render)

        # ── Aggregation ───────────────────────────────────────────
        grp_agg = QGroupBox("📊 Aggregation")
        agg_form = QFormLayout(grp_agg)
        self.agg_mode = QComboBox()
        self.agg_mode.addItems(["mean", "median", "trimmed_mean"])
        agg_form.addRow("Mode:", self.agg_mode)
        layout.addWidget(grp_agg)

        # ── Re-run Button ─────────────────────────────────────────
        self.btn_rerun = QPushButton("▶ Re-run Analysis")
        self.btn_rerun.setMinimumHeight(40)
        self.btn_rerun.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #636EFA; color: white;")
        layout.addWidget(self.btn_rerun)

        layout.addStretch()
        self.setWidget(container)

    # ── Getters ────────────────────────────────────────────────────
    @property
    def vert_exag(self) -> float:
        return self.vert_exag_slider.value() / 10.0

    def get_config(self):
        """Return a PreprocessingConfig from current UI state."""
        from engine.preprocessing import PreprocessingConfig
        return PreprocessingConfig(
            plane_removal=self.plane_mode.currentText(),
            poly_order=self.poly_order.value(),
            outlier_method=self.outlier_method.currentText(),
            outlier_window=self.outlier_window.value(),
            outlier_threshold=self.outlier_thresh.value(),
            interp_missing=self.interp_gaps.isChecked(),
            detrend_mode=self.detrend.currentText(),
        )
