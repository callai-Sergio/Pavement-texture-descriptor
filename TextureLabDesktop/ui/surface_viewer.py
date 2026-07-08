import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSplitter, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt

import pyqtgraph.opengl as gl
import pyqtgraph as pg


class SurfaceViewer(QWidget):
    """Dual 2D/3D surface viewer with axes and view toggles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # 1) Top Control Bar
        self.controls = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        
        self.btn_2d = QRadioButton("2D View")
        self.btn_3d = QRadioButton("3D View")
        self.btn_both = QRadioButton("Both")
        self.btn_both.setChecked(True)
        
        for btn in [self.btn_2d, self.btn_3d, self.btn_both]:
            self.view_group.addButton(btn)
            self.controls.addWidget(btn)
        
        self.view_group.buttonClicked.connect(self._update_visibility)
        
        self.controls.addStretch()
        self.info_label = QLabel("Load a surface to view")
        self.info_label.setStyleSheet("color: #aaa; font-style: italic;")
        self.controls.addWidget(self.info_label)
        self.layout.addLayout(self.controls)

        # 2) Splitter for 2D and 3D
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # --- 2D WIDGET ---
        self.widget_2d = pg.PlotWidget(title="2D Top View (Heightmap)")
        self.widget_2d.setLabel('bottom', "X", units='mm')
        self.widget_2d.setLabel('left', "Y", units='mm')
        self.widget_2d.setAspectLocked(True)
        self.img_item = pg.ImageItem()
        self.widget_2d.addItem(self.img_item)
        
        # Color bar for 2D
        self.colorbar = pg.ColorBarItem(values=(0, 1))
        self.colorbar.setColorMap(pg.colormap.get('viridis'))
        self.colorbar.setImageItem(self.img_item)
        
        # --- 3D WIDGET ---
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor(pg.mkColor(30, 30, 40))
        
        # Add axes to 3D
        self.axes = gl.GLAxisItem()
        self.axes.setSize(10, 10, 10)
        self.gl_view.addItem(self.axes)
        
        self.splitter.addWidget(self.widget_2d)
        self.splitter.addWidget(self.gl_view)
        self.layout.addWidget(self.splitter)

        self._surface_item = None
        self._last_data = None # (z, dx, dy, vert_exag, robust)

    def _update_visibility(self):
        if self.btn_2d.isChecked():
            self.widget_2d.show()
            self.gl_view.hide()
        elif self.btn_3d.isChecked():
            self.widget_2d.hide()
            self.gl_view.show()
        else:
            self.widget_2d.show()
            self.gl_view.show()

    def update_surface(self, z: np.ndarray, dx: float, dy: float,
                       vert_exag: float = 0.3,
                       robust_color: bool = True,
                       max_pts: int = 512):
        """Update both 2D and 3D views."""
        self._last_data = (z, dx, dy, vert_exag, robust_color)
        
        ny, nx = z.shape
        step_x = max(1, nx // max_pts)
        step_y = max(1, ny // max_pts)

        # 1) Downsample for visualization
        if step_x > 1 or step_y > 1:
            ny_trim = (ny // step_y) * step_y
            nx_trim = (nx // step_x) * step_x
            z_ds = np.nanmean(
                z[:ny_trim, :nx_trim].reshape(ny_trim // step_y, step_y,
                                              nx_trim // step_x, step_x),
                axis=(1, 3))
        else:
            z_ds = z.copy()

        # Handle NaNs for rendering
        mask = np.isnan(z_ds)
        if mask.any():
            avg = np.nanmean(z_ds) if not np.isnan(z_ds).all() else 0.0
            z_ds[mask] = avg

        # 2) Update 2D View
        # ImageItem needs [X, Y], current z_ds is [Y, X]
        # We also need to set the lookup table and levels
        if robust_color:
            levels = np.percentile(z_ds, [1, 99])
        else:
            levels = [z_ds.min(), z_ds.max()]
            
        if levels[1] <= levels[0]: levels[1] = levels[0] + 1e-9
        
        self.img_item.setImage(z_ds)
        self.img_item.setRect(0, 0, ny * dy, nx * dx)
        self.img_item.setLevels(levels)
        self.colorbar.setLevels(levels)

        if self._surface_item is not None:
            self.gl_view.removeItem(self._surface_item)
        if hasattr(self, '_grid_item') and self._grid_item:
            self.gl_view.removeItem(self._grid_item)
        
        # Color mapping (Matplotlib colormap)
        import matplotlib.pyplot as plt
        z_norm = np.clip((z_ds - levels[0]) / (levels[1] - levels[0]), 0, 1)
        colors = plt.get_cmap("viridis")(z_norm).astype(np.float32)

        sx, sy = dy * step_y, dx * step_x
        nx_ds_swap, ny_ds_swap = z_ds.shape[0], z_ds.shape[1]
        
        span_x = nx_ds_swap * sx
        span_y = ny_ds_swap * sy
        max_span = max(span_x, span_y)
        z_span = max(1e-6, float(z_ds.max() - z_ds.min()))
        
        # Scale Z proportionally to max horizontal span for a balanced 3D view
        z_render = ((z_ds - np.mean(z_ds)) / z_span) * (0.25 * max_span) * vert_exag

        self._surface_item = gl.GLSurfacePlotItem(
            z=z_render.T, colors=colors.transpose(1, 0, 2), shader='shaded', glOptions='opaque'
        )
        
        self._surface_item.scale(sx, sy, 1.0)
        nx_ds_swap, ny_ds_swap = z_ds.shape[0], z_ds.shape[1]
        self._surface_item.translate(-(nx_ds_swap * sx) / 2.0, 
                                     -(ny_ds_swap * sy) / 2.0, 0)
        self.gl_view.addItem(self._surface_item)
        
        # Add 3D grid for dimensions
        self._grid_item = gl.GLGridItem()
        self._grid_item.setSize(nx_ds_swap * sx, ny_ds_swap * sy, 0)
        self._grid_item.setSpacing(10, 10, 0) # 10mm grid
        self.gl_view.addItem(self._grid_item)
        
        # Auto-center 3D camera
        cam_dist = max(nx_ds_swap * sx, ny_ds_swap * sy) * 1.5
        self.gl_view.setCameraPosition(distance=cam_dist, elevation=35, azimuth=-45)
        
        self.info_label.setText(f"Surface: {ny*dy:.1f} × {nx*dx:.1f} mm  |  Z Range: [{z_ds.min():.2f}, {z_ds.max():.2f}]")

    def refresh(self, vert_exag: float, robust_color: bool):
        """Re-render with different visual settings using cached data."""
        if self._last_data is None:
            return
        z, dx, dy, _, _ = self._last_data
        self.update_surface(z, dx, dy, vert_exag, robust_color)

    def clear(self):
        self.img_item.clear()
        if self._surface_item is not None:
            self.gl_view.removeItem(self._surface_item)
            self._surface_item = None
        if hasattr(self, '_grid_item') and self._grid_item:
            self.gl_view.removeItem(self._grid_item)
            self._grid_item = None
        self.info_label.setText("Load a surface to view")
