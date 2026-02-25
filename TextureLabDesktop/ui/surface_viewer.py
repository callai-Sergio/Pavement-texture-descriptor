"""
surface_viewer.py – Hardware-accelerated 3D surface viewer via PyQtGraph
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

import pyqtgraph.opengl as gl
import pyqtgraph as pg


class SurfaceViewer(QWidget):
    """OpenGL 3D surface viewer with vertical exaggeration and robust color."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel("Load a surface to view")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #888; font-size: 13px;")
        layout.addWidget(self.info_label)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor(pg.mkColor(30, 30, 40))
        self.gl_widget.setCameraPosition(distance=50, elevation=30, azimuth=45)
        layout.addWidget(self.gl_widget)

        self._surface_item = None
        self._grid_item = None

    def update_surface(self, z: np.ndarray, dx: float, dy: float,
                       vert_exag: float = 0.3,
                       robust_color: bool = True,
                       max_pts: int = 500):
        """Render the surface with block-average downsampling."""
        # Clear previous
        if self._surface_item is not None:
            self.gl_widget.removeItem(self._surface_item)
            self._surface_item = None
        if self._grid_item is not None:
            self.gl_widget.removeItem(self._grid_item)
            self._grid_item = None

        ny, nx = z.shape
        step_x = max(1, nx // max_pts)
        step_y = max(1, ny // max_pts)

        # Block-average downsample
        if step_x > 1 or step_y > 1:
            ny_trim = (ny // step_y) * step_y
            nx_trim = (nx // step_x) * step_x
            z_trim = z[:ny_trim, :nx_trim]
            z_ds = np.nanmean(
                z_trim.reshape(ny_trim // step_y, step_y,
                               nx_trim // step_x, step_x),
                axis=(1, 3))
        else:
            z_ds = z.copy()

        # Replace NaN with 0 for rendering
        z_ds = np.nan_to_num(z_ds, nan=0.0)

        ny_ds, nx_ds = z_ds.shape

        # Color mapping
        valid = z_ds[z_ds != 0] if np.any(z_ds != 0) else z_ds.ravel()
        if robust_color and len(valid) > 0:
            cmin = float(np.percentile(valid, 1))
            cmax = float(np.percentile(valid, 99))
        elif len(valid) > 0:
            cmin, cmax = float(valid.min()), float(valid.max())
        else:
            cmin, cmax = 0, 1

        # Normalize to 0-1 for colormap
        z_norm = np.clip((z_ds - cmin) / (cmax - cmin + 1e-15), 0, 1)

        # Viridis-like colormap (RGBA)
        colors = np.zeros((ny_ds, nx_ds, 4), dtype=np.float32)
        # Simple viridis approximation
        colors[..., 0] = np.clip(0.267 + 2.0 * (z_norm - 0.5), 0, 1) * (1 - 0.5 * z_norm)
        colors[..., 1] = np.clip(0.004 + z_norm * 0.87, 0, 0.95)
        colors[..., 2] = np.clip(0.329 + 0.7 * (1 - z_norm), 0, 1) * (0.3 + 0.7 * z_norm)
        colors[..., 3] = 1.0

        # Apply vertical exaggeration
        z_render = z_ds * vert_exag

        # Create surface item
        self._surface_item = gl.GLSurfacePlotItem(
            z=z_render,
            colors=colors,
            shader='shaded',
            glOptions='opaque',
        )

        # Scale and center
        sx = dx * step_x
        sy = dy * step_y
        self._surface_item.scale(sx, sy, 1.0)
        cx = nx_ds * sx / 2
        cy = ny_ds * sy / 2
        self._surface_item.translate(-cx, -cy, 0)

        self.gl_widget.addItem(self._surface_item)

        # Add grid for reference
        self._grid_item = gl.GLGridItem()
        self._grid_item.setSize(nx_ds * sx, ny_ds * sy, 0)
        self._grid_item.setSpacing(nx_ds * sx / 10, ny_ds * sy / 10, 0)
        self.gl_widget.addItem(self._grid_item)

        # Update camera
        cam_dist = max(nx_ds * sx, ny_ds * sy) * 1.2
        self.gl_widget.setCameraPosition(
            distance=cam_dist, elevation=30, azimuth=45)

        self.info_label.setText(
            f"Grid: {ny}×{nx} → rendered {ny_ds}×{nx_ds}  |  "
            f"Vert. exag: ×{vert_exag:.1f}  |  "
            f"Colour: {'P1-P99' if robust_color else 'full range'}")

    def clear(self):
        if self._surface_item is not None:
            self.gl_widget.removeItem(self._surface_item)
            self._surface_item = None
        if self._grid_item is not None:
            self.gl_widget.removeItem(self._grid_item)
            self._grid_item = None
        self.info_label.setText("Load a surface to view")
