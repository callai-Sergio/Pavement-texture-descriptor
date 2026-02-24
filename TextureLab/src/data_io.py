"""
data_io.py – Data ingestion for TextureLab
Handles LAZ point-cloud and CSV grid files with chunked / out-of-core reading.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class SurfaceGrid:
    """Regular-grid height field."""
    z: np.ndarray                # 2D array [ny, nx]
    dx: float                    # grid spacing X (m)
    dy: float                    # grid spacing Y (m)
    x0: float = 0.0
    y0: float = 0.0
    units_xy: str = "mm"
    units_z: str = "mm"
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def nx(self) -> int:
        return self.z.shape[1]

    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def x(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def y(self) -> np.ndarray:
        return self.y0 + np.arange(self.ny) * self.dy


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------
def _sniff_csv_format(path: str) -> str:
    """Detect whether CSV is 'xyz' (x,y,z columns) or 'matrix' (dense grid)."""
    with open(path, "r") as f:
        first_lines = [f.readline() for _ in range(5)]
    # Count columns in first non-empty line
    for line in first_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ncols = len(line.split(","))
        if ncols == 3:
            return "xyz"
        else:
            return "matrix"
    return "matrix"


def read_csv_xyz(path: str, dx: float, dy: float,
                 chunk_size: int = 500_000,
                 units_xy: str = "mm",
                 units_z: str = "mm") -> SurfaceGrid:
    """Read x,y,z CSV and rasterize to regular grid."""
    chunks = []
    for chunk in pd.read_csv(path, header=None, names=["x", "y", "z"],
                             chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Snap to grid
    x_vals = np.sort(df["x"].unique())
    y_vals = np.sort(df["y"].unique())

    # Build 2-D array
    nx, ny = len(x_vals), len(y_vals)
    z = np.full((ny, nx), np.nan, dtype=np.float64)

    x_idx = np.searchsorted(x_vals, df["x"].values)
    y_idx = np.searchsorted(y_vals, df["y"].values)
    z[y_idx, x_idx] = df["z"].values

    real_dx = float(np.median(np.diff(x_vals))) if len(x_vals) > 1 else dx
    real_dy = float(np.median(np.diff(y_vals))) if len(y_vals) > 1 else dy

    return SurfaceGrid(
        z=z, dx=real_dx, dy=real_dy,
        x0=float(x_vals[0]), y0=float(y_vals[0]),
        units_xy=units_xy, units_z=units_z,
        source_file=path,
        metadata={"format": "csv_xyz", "nx": nx, "ny": ny},
    )


def read_csv_matrix(path: str, dx: float, dy: float,
                    units_xy: str = "mm",
                    units_z: str = "mm") -> SurfaceGrid:
    """Read dense matrix CSV (rows=y, cols=x)."""
    # Skip comment lines
    z = np.loadtxt(path, delimiter=",", comments="#")
    if z.ndim == 1:
        z = z.reshape(1, -1)
    return SurfaceGrid(
        z=z, dx=dx, dy=dy,
        units_xy=units_xy, units_z=units_z,
        source_file=path,
        metadata={"format": "csv_matrix", "nx": z.shape[1], "ny": z.shape[0]},
    )


def read_csv(path: str, dx: float, dy: float, **kw) -> SurfaceGrid:
    """Auto-detect CSV format and read."""
    fmt = _sniff_csv_format(path)
    if fmt == "xyz":
        return read_csv_xyz(path, dx, dy, **kw)
    return read_csv_matrix(path, dx, dy, **kw)


# ---------------------------------------------------------------------------
# LAZ reader
# ---------------------------------------------------------------------------
def read_laz(path: str, dx: float, dy: float,
             units_xy: str = "mm",
             units_z: str = "mm") -> SurfaceGrid:
    """Read LAZ/LAS file and rasterize to regular grid."""
    import laspy

    with laspy.open(path) as reader:
        las = reader.read()

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    x_min, y_min = x.min(), y.min()
    x -= x_min
    y -= y_min

    col = np.round(x / dx).astype(int)
    row = np.round(y / dy).astype(int)

    nx = col.max() + 1
    ny = row.max() + 1

    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    # Simple nearest-neighbour assignment (last-write-wins)
    grid[row, col] = z

    return SurfaceGrid(
        z=grid, dx=dx, dy=dy,
        x0=float(x_min), y0=float(y_min),
        units_xy=units_xy, units_z=units_z,
        source_file=path,
        metadata={"format": "laz", "nx": int(nx), "ny": int(ny),
                  "n_points": len(z)},
    )


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------
def load_surface(path: str, dx: float, dy: float, **kw) -> SurfaceGrid:
    """Load any supported file format."""
    ext = pathlib.Path(path).suffix.lower()
    if ext in (".laz", ".las"):
        return read_laz(path, dx, dy, **kw)
    elif ext == ".csv":
        return read_csv(path, dx, dy, **kw)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
