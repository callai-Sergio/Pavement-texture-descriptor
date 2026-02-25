"""
results_panel.py – Tabbed results display for TextureLab Desktop

Tabs: Summary | Charts | PCA | Feature Selection | Logs
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QComboBox, QLabel, QPushButton,
    QFileDialog, QHeaderView, QSplitter, QGroupBox, QFormLayout,
    QSpinBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from engine.analytics import run_pca, prepare_feature_matrix, correlation_pruning


class ResultsPanel(QTabWidget):
    """Central tabbed widget showing analysis results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None

        # ── Summary tab ───────────────────────────────────────────
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)

        self.file_selector = QComboBox()
        self.file_selector.currentTextChanged.connect(self._on_file_changed)
        summary_layout.addWidget(self.file_selector)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(4)
        self.summary_table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Std", "Unit"])
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.summary_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table)

        # Export buttons
        btn_row = QHBoxLayout()
        self.btn_csv = QPushButton("📄 Export CSV")
        self.btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(self.btn_csv)
        self.btn_excel = QPushButton("📊 Export Excel")
        self.btn_excel.clicked.connect(self._export_excel)
        btn_row.addWidget(self.btn_excel)
        summary_layout.addLayout(btn_row)
        self.addTab(self.summary_tab, "📋 Summary")

        # ── Charts tab ────────────────────────────────────────────
        self.charts_tab = QWidget()
        charts_layout = QVBoxLayout(self.charts_tab)

        chart_controls = QHBoxLayout()
        chart_controls.addWidget(QLabel("Parameter:"))
        self.chart_param = QComboBox()
        self.chart_param.currentTextChanged.connect(self._update_chart)
        chart_controls.addWidget(self.chart_param)
        charts_layout.addLayout(chart_controls)

        self.chart_canvas = FigureCanvasQTAgg(Figure(figsize=(8, 4)))
        self.chart_canvas.figure.set_facecolor('#1e1e28')
        charts_layout.addWidget(self.chart_canvas)
        self.addTab(self.charts_tab, "📈 Charts")

        # ── PCA tab ───────────────────────────────────────────────
        self.pca_tab = QWidget()
        pca_layout = QVBoxLayout(self.pca_tab)

        pca_controls = QHBoxLayout()
        pca_controls.addWidget(QLabel("Components:"))
        self.pca_n = QSpinBox()
        self.pca_n.setRange(2, 20)
        self.pca_n.setValue(5)
        pca_controls.addWidget(self.pca_n)
        self.pca_run_btn = QPushButton("▶ Run PCA")
        self.pca_run_btn.clicked.connect(self._run_pca)
        pca_controls.addWidget(self.pca_run_btn)
        pca_controls.addStretch()
        pca_layout.addLayout(pca_controls)

        self.pca_canvas = FigureCanvasQTAgg(Figure(figsize=(10, 4)))
        self.pca_canvas.figure.set_facecolor('#1e1e28')
        pca_layout.addWidget(self.pca_canvas)
        self.addTab(self.pca_tab, "🔬 PCA")

        # ── Feature Selection tab ─────────────────────────────────
        self.feat_tab = QWidget()
        feat_layout = QVBoxLayout(self.feat_tab)
        self.feat_run_btn = QPushButton("▶ Run Correlation Pruning")
        self.feat_run_btn.clicked.connect(self._run_feature_selection)
        feat_layout.addWidget(self.feat_run_btn)
        self.feat_table = QTableWidget()
        self.feat_table.setColumnCount(2)
        self.feat_table.setHorizontalHeaderLabels(["Feature", "Status"])
        self.feat_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        feat_layout.addWidget(self.feat_table)
        self.addTab(self.feat_tab, "🎯 Features")

        # ── Logs tab ──────────────────────────────────────────────
        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout(self.logs_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 12px;")
        logs_layout.addWidget(self.log_text)
        self.addTab(self.logs_tab, "📝 Logs")

    # ── Public methods ────────────────────────────────────────────

    def set_results(self, results: dict):
        """Populate all tabs with processing results."""
        self._results = results

        # Update file selector
        self.file_selector.blockSignals(True)
        self.file_selector.clear()
        self.file_selector.addItems(results["file_names"])
        self.file_selector.blockSignals(False)

        # Populate first file
        if results["file_names"]:
            self._on_file_changed(results["file_names"][0])

        # Chart param selector
        self.chart_param.blockSignals(True)
        self.chart_param.clear()
        if results["batch_agg"]:
            keys = [k for k in results["batch_agg"][0].keys()
                    if not k.endswith("_std")]
            self.chart_param.addItems(sorted(keys))
        self.chart_param.blockSignals(False)
        if self.chart_param.count() > 0:
            self._update_chart(self.chart_param.currentText())

        # Logs
        self.log_text.clear()
        for line in results.get("logs", []):
            self.log_text.append(line)

    def _on_file_changed(self, fname: str):
        if not self._results or not fname:
            return
        agg = self._results["aggregated"].get(fname, {})
        areal = self._results["results_areal"].get(fname, {})

        # Unit mapping
        UNITS = {
            "Ra": "µm", "Rq": "µm", "Rz": "µm", "Rt": "µm",
            "Rp": "µm", "Rv": "µm", "Rsk": "–", "Rku": "–",
            "Rk": "µm", "Rpk": "µm", "Rvk": "µm",
            "Mr1": "%", "Mr2": "%", "g_factor": "–",
            "MPD": "mm", "ETD": "mm", "ENDT": "dB",
            "TextureLevel": "dB", "FractalDim": "–",
            "Sa": "µm", "Sq": "µm", "Sdr": "%",
            "Ssk": "–", "Sku": "–", "Sz": "µm", "Sv": "µm",
            "Vv": "mm³/mm²", "Vm": "mm³/mm²",
        }

        # Merge aggregated + areal
        merged = dict(agg)
        merged.update(areal)

        # Filter base keys
        base_keys = sorted(
            k for k in merged
            if not k.endswith("_std") and not k.endswith("_P10")
            and not k.endswith("_P90"))

        self.summary_table.setRowCount(len(base_keys))
        for i, key in enumerate(base_keys):
            val = merged.get(key)
            std = merged.get(f"{key}_std")
            unit = UNITS.get(key, "")

            self.summary_table.setItem(i, 0, QTableWidgetItem(key))
            self.summary_table.setItem(
                i, 1, QTableWidgetItem(self._fmt(val)))
            self.summary_table.setItem(
                i, 2, QTableWidgetItem(self._fmt(std) if std else "–"))
            self.summary_table.setItem(i, 3, QTableWidgetItem(unit))

    def _fmt(self, val) -> str:
        if val is None:
            return "–"
        av = abs(val)
        if av == 0:
            return "0"
        elif av >= 100:
            return f"{val:.1f}"
        elif av >= 1:
            return f"{val:.3f}"
        elif av >= 0.01:
            return f"{val:.4f}"
        else:
            return f"{val:.2e}"

    def _update_chart(self, param: str):
        if not self._results or not param:
            return
        fig = self.chart_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e28')
        fig.patch.set_facecolor('#1e1e28')

        fnames = self._results["file_names"]
        vals = []
        for fn in fnames:
            agg = self._results["aggregated"].get(fn, {})
            if param in agg:
                vals.append(agg[param])
            else:
                vals.append(0)

        if len(fnames) <= 5:
            ax.bar(range(len(fnames)), vals, color='#636EFA', alpha=0.8)
            ax.set_xticks(range(len(fnames)))
            ax.set_xticklabels(
                [f[:20] for f in fnames], rotation=30,
                ha='right', fontsize=8, color='white')
        else:
            ax.hist(vals, bins=min(20, len(vals)), color='#636EFA',
                    alpha=0.8, edgecolor='white', linewidth=0.5)
            ax.set_xlabel(param, color='white')
            ax.set_ylabel("Count", color='white')

        ax.set_title(param, color='white', fontsize=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        fig.tight_layout()
        self.chart_canvas.draw()

    def _run_pca(self):
        if not self._results:
            return
        # Build dataframe from all profiles
        all_rows = []
        for fn in self._results["file_names"]:
            for row in self._results["results_1d"].get(fn, []):
                r = dict(row)
                r["_file"] = fn
                all_rows.append(r)
        if not all_rows:
            return

        df = pd.DataFrame(all_rows)
        num_cols = sorted(df.select_dtypes(include="number").columns.tolist())
        n_comp = min(self.pca_n.value(), len(num_cols), len(df))

        scores, loadings, explained, feat_names = run_pca(
            df, num_cols, n_comp)

        fig = self.pca_canvas.figure
        fig.clear()
        fig.patch.set_facecolor('#1e1e28')

        # Variance plot
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor('#1e1e28')
        ax1.bar(range(1, len(explained) + 1), explained,
                color='#636EFA', alpha=0.8)
        ax1.plot(range(1, len(explained) + 1), np.cumsum(explained),
                 'o-', color='#EF553B', linewidth=2)
        ax1.set_xlabel("Component", color='white')
        ax1.set_ylabel("Variance ratio", color='white')
        ax1.set_title("Explained Variance", color='white')
        ax1.tick_params(colors='white')
        for s in ax1.spines.values():
            s.set_color('#444')

        # Biplot (PC1 vs PC2)
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor('#1e1e28')
        ax2.scatter(scores[:, 0], scores[:, 1], s=8, alpha=0.5,
                    c='#636EFA')
        # Loading arrows
        scale = np.max(np.abs(scores[:, :2])) / (
            np.max(np.abs(loadings[:, :2])) + 1e-9)
        for i, name in enumerate(feat_names):
            ax2.annotate(
                name,
                xy=(loadings[i, 0] * scale * 0.7,
                    loadings[i, 1] * scale * 0.7),
                xytext=(0, 0), textcoords='data',
                arrowprops=dict(arrowstyle='->', color='#EF553B',
                                lw=1.5),
                fontsize=7, color='#EF553B', ha='center')
        ax2.set_xlabel("PC1", color='white')
        ax2.set_ylabel("PC2", color='white')
        ax2.set_title("Biplot", color='white')
        ax2.tick_params(colors='white')
        for s in ax2.spines.values():
            s.set_color('#444')

        fig.tight_layout()
        self.pca_canvas.draw()

    def _run_feature_selection(self):
        if not self._results:
            return
        all_rows = []
        for fn in self._results["file_names"]:
            for row in self._results["results_1d"].get(fn, []):
                all_rows.append(dict(row))
        if not all_rows:
            return

        df = pd.DataFrame(all_rows)
        kept = correlation_pruning(df, threshold=0.95)
        all_cols = sorted(df.select_dtypes(include="number").columns.tolist())

        self.feat_table.setRowCount(len(all_cols))
        for i, col in enumerate(all_cols):
            self.feat_table.setItem(i, 0, QTableWidgetItem(col))
            status = "✅ Keep" if col in kept else "❌ Dropped (r > 0.95)"
            self.feat_table.setItem(i, 1, QTableWidgetItem(status))

    # ── Export ────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "texturelab_results.csv",
            "CSV Files (*.csv)")
        if path:
            rows = []
            for fn in self._results["file_names"]:
                agg = dict(self._results["aggregated"].get(fn, {}))
                agg["file"] = fn
                rows.append(agg)
            pd.DataFrame(rows).to_csv(path, index=False)

    def _export_excel(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "texturelab_results.xlsx",
            "Excel Files (*.xlsx)")
        if path:
            rows = []
            for fn in self._results["file_names"]:
                agg = dict(self._results["aggregated"].get(fn, {}))
                agg["file"] = fn
                rows.append(agg)
            pd.DataFrame(rows).to_excel(path, index=False)
