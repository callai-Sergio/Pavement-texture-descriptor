"""
main_window.py – Main application window for TextureLab Desktop
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QTabWidget,
    QStatusBar, QProgressBar, QLabel, QMessageBox, QWidget,
    QVBoxLayout, QSplitter,
)
from PyQt6.QtGui import QAction, QKeySequence

from ui.settings_panel import SettingsPanel
from ui.surface_viewer import SurfaceViewer
from ui.results_panel import ResultsPanel
from ui.workers import ProcessingWorker


class MainWindow(QMainWindow):
    """TextureLab Desktop main window."""

    APP_NAME = "TextureLab Desktop"
    APP_VERSION = "2.0.0"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{self.APP_NAME} v{self.APP_VERSION}")
        self.setMinimumSize(1200, 750)
        self.resize(1400, 850)

        self._worker: Optional[ProcessingWorker] = None
        self._results: Optional[dict] = None
        self._current_files: List[str] = []

        self._build_menu()
        self._build_ui()
        self._build_statusbar()

        # Restore window geometry
        settings = QSettings("TextureLab", "Desktop")
        geo = settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    def closeEvent(self, event):
        settings = QSettings("TextureLab", "Desktop")
        settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    # ── Menu bar ──────────────────────────────────────────────────

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_act = QAction("📂 &Open File…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_file)
        file_menu.addAction(open_act)

        batch_act = QAction("📦 Open &Batch…", self)
        batch_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        batch_act.triggered.connect(self._open_batch)
        file_menu.addAction(batch_act)

        file_menu.addSeparator()

        export_csv = QAction("📄 Export CS&V…", self)
        export_csv.triggered.connect(
            lambda: self.results_panel._export_csv()
            if self._results else None)
        file_menu.addAction(export_csv)

        export_xl = QAction("📊 Export &Excel…", self)
        export_xl.triggered.connect(
            lambda: self.results_panel._export_excel()
            if self._results else None)
        file_menu.addAction(export_xl)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menubar.addMenu("&View")
        toggle_settings = QAction("⚙️ Settings Panel", self)
        toggle_settings.setCheckable(True)
        toggle_settings.setChecked(True)
        toggle_settings.triggered.connect(
            lambda checked: self.settings_panel.setVisible(checked))
        view_menu.addAction(toggle_settings)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_act = QAction("About TextureLab", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ── UI layout ─────────────────────────────────────────────────

    def _build_ui(self):
        # Settings dock (left)
        self.settings_panel = SettingsPanel(self)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_panel)

        # Central area: splitter with 3D viewer + results tabs
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # 3D surface viewer (top)
        self.surface_viewer = SurfaceViewer()
        splitter.addWidget(self.surface_viewer)

        # Results tabs (bottom)
        self.results_panel = ResultsPanel()
        splitter.addWidget(self.results_panel)

        splitter.setSizes([400, 400])
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_statusbar(self):
        self.statusBar().setStyleSheet("font-size: 12px;")
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(250)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    # ── File dialogs ──────────────────────────────────────────────

    def _open_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Surface File",
            "",
            "Surface Files (*.csv *.txt *.laz *.las);;All Files (*)")
        if paths:
            self._process_files(paths)

    def _open_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Batch Files",
            "",
            "Surface Files (*.csv *.txt *.laz *.las);;All Files (*)")
        if paths:
            self._process_files(paths)

    # ── Processing ────────────────────────────────────────────────

    def _process_files(self, paths: List[str]):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(
                self, "Busy", "Processing is already running.")
            return

        self._current_files = paths
        cfg = self.settings_panel.get_config()

        self._worker = ProcessingWorker(
            file_paths=paths,
            dx=self.settings_panel.dx.value(),
            dy=self.settings_panel.dy.value(),
            units_xy=self.settings_panel.units_xy.currentText(),
            units_z=self.settings_panel.units_z.currentText(),
            direction=self.settings_panel.direction.currentText(),
            every_n=self.settings_panel.every_n.value(),
            agg_mode=self.settings_panel.agg_mode.currentText(),
            cfg=cfg,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing…")
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, results: dict):
        self._results = results
        self.progress_bar.setVisible(False)

        n = len(results["file_names"])
        total_time = sum(results["timing"].values())
        self.status_label.setText(
            f"✅ {n} file(s) processed in {total_time:.1f}s")

        # Update 3D viewer with first surface
        if results["surfaces"]:
            grid = results["surfaces"][0]
            self.surface_viewer.update_surface(
                grid.z,
                self.settings_panel.dx.value(),
                self.settings_panel.dy.value(),
                vert_exag=self.settings_panel.vert_exag,
                robust_color=self.settings_panel.robust_color.isChecked(),
            )

        # Update results panel
        self.results_panel.set_results(results)

        # Connect file selector to update 3D view
        self.results_panel.file_selector.currentTextChanged.connect(
            self._on_surface_selected)

    def _on_surface_selected(self, fname: str):
        if not self._results or not fname:
            return
        for grid, fn in zip(
                self._results["surfaces"],
                self._results["file_names"]):
            if fn == fname:
                self.surface_viewer.update_surface(
                    grid.z,
                    self.settings_panel.dx.value(),
                    self.settings_panel.dy.value(),
                    vert_exag=self.settings_panel.vert_exag,
                    robust_color=self.settings_panel.robust_color.isChecked(),
                )
                break

    def _on_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error")
        QMessageBox.critical(self, "Processing Error", msg)

    def _show_about(self):
        QMessageBox.information(
            self, "About TextureLab Desktop",
            f"<h2>{self.APP_NAME}</h2>"
            f"<p>Version {self.APP_VERSION}</p>"
            "<p>Standalone pavement texture analysis application.</p>"
            "<p>© 2025 Sergio Callai<br>"
            "Licensed under CC BY-NC 4.0</p>"
            "<hr>"
            "<p><b>Descriptors:</b> ISO 4287, ISO 13473, ISO 25178, "
            "ISO 13565, ISO 10844</p>"
            "<p><b>Engine:</b> NumPy, SciPy, scikit-learn</p>"
            "<p><b>UI:</b> PyQt6 + PyQtGraph (OpenGL)</p>")
