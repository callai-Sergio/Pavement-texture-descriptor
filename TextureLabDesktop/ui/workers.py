"""
workers.py – Background processing threads for TextureLab Desktop

Uses QThread so the UI never freezes during file loading or analysis.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

import numpy as np

from engine.data_io import load_surface, SurfaceGrid
from engine.preprocessing import PreprocessingConfig, preprocess_surface
from engine.descriptors import compute_all, aggregate_profiles


class ProcessingWorker(QThread):
    """Load file → preprocess → compute descriptors in background."""

    progress = pyqtSignal(int, str)       # percent, message
    finished = pyqtSignal(dict)           # full results bundle
    error = pyqtSignal(str)               # error message

    def __init__(self, file_paths: List[str],
                 dx: float, dy: float,
                 units_xy: str, units_z: str,
                 direction: str, every_n: int,
                 agg_mode: str,
                 cfg: PreprocessingConfig,
                 parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.dx = dx
        self.dy = dy
        self.units_xy = units_xy
        self.units_z = units_z
        self.direction = direction
        self.every_n = every_n
        self.agg_mode = agg_mode
        self.cfg = cfg

    def run(self):
        try:
            results = {
                "surfaces": [],
                "file_names": [],
                "profiles": {},
                "results_1d": {},
                "results_areal": {},
                "aggregated": {},
                "batch_agg": [],
                "warnings": [],
                "logs": [],
                "timing": {},
            }

            total = len(self.file_paths)
            for fi, fpath in enumerate(self.file_paths):
                fname = Path(fpath).name
                t0 = time.perf_counter()
                self.progress.emit(
                    int(100 * fi / total),
                    f"Loading {fname} ({fi+1}/{total})…")

                # Load
                grid = load_surface(fpath, self.dx, self.dy,
                                    units_xy=self.units_xy,
                                    units_z=self.units_z)
                results["logs"].append(
                    f"📄 {fname}: grid {grid.ny}×{grid.nx}")

                # Preprocess
                self.progress.emit(
                    int(100 * (fi + 0.3) / total),
                    f"Preprocessing {fname}…")

                def _prog(pct):
                    self.progress.emit(
                        int(100 * (fi + 0.3 + 0.4 * pct / 100) / total),
                        f"Filtering {fname}…")

                z_proc, profiles, warns = preprocess_surface(
                    grid.z, self.dx, self.dy, self.cfg,
                    self.direction, self.every_n,
                    progress_cb=_prog)

                grid.z = z_proc.copy()
                results["surfaces"].append(grid)
                results["file_names"].append(fname)
                results["warnings"].extend(warns)
                for w in warns:
                    results["logs"].append(f"  ⚠ {w}")

                # Compute descriptors
                self.progress.emit(
                    int(100 * (fi + 0.7) / total),
                    f"Computing descriptors for {fname}…")

                def _desc_prog(pct):
                    self.progress.emit(
                        int(100 * (fi + 0.7 + 0.3 * pct / 100) / total),
                        f"Descriptors {fname}…")

                per_profile, areal = compute_all(
                    profiles, z_proc, self.dx, self.dy,
                    progress_cb=_desc_prog)
                agg = aggregate_profiles(per_profile, self.agg_mode)

                results["profiles"][fname] = profiles
                results["results_1d"][fname] = per_profile
                results["results_areal"][fname] = areal
                results["aggregated"][fname] = agg
                results["batch_agg"].append(agg)

                elapsed = time.perf_counter() - t0
                results["timing"][fname] = elapsed
                results["logs"].append(
                    f"  ✅ {len(profiles)} profiles, "
                    f"{len(per_profile[0]) if per_profile else 0} params, "
                    f"{elapsed:.1f}s")

            self.progress.emit(100, "Done ✅")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")
