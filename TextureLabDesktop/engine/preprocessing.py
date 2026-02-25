"""
preprocessing.py – Surface pre-processing pipeline for TextureLab

Includes: plane removal, outlier filtering (mm-based Hampel),
gap interpolation, detrending, FFT band-pass filtering,
and block-average downsampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage, signal, interpolate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """All pre-processing toggles stored in one place for reproducibility."""
    # Plane/surface removal
    plane_removal: str = "none"          # none | plane | polynomial
    poly_order: int = 2

    # Outlier
    outlier_method: str = "none"         # none | hampel | median
    outlier_window: int = 7              # in samples (fallback)
    outlier_threshold: float = 3.5

    # Missing values
    interp_missing: bool = True
    max_missing_fraction: float = 0.3    # reject profile if > this

    # Per-profile detrending
    detrend_mode: str = "none"           # none | mean | linear

    # Band filtering
    bandpass: bool = False
    bandpass_low: float = 0.5            # mm wavelength low-cut
    bandpass_high: float = 50.0          # mm wavelength high-cut
    bandpass_method: str = "fft"         # fft | iir

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Plane / polynomial removal (3-D)
# ---------------------------------------------------------------------------
def remove_plane(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Subtract best-fit plane from surface."""
    ny, nx = z.shape
    xv, yv = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    mask = np.isfinite(z)
    A = np.column_stack([xv[mask], yv[mask], np.ones(mask.sum())])
    coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
    plane = coeffs[0] * xv + coeffs[1] * yv + coeffs[2]
    return z - plane


def remove_polynomial(z: np.ndarray, dx: float, dy: float,
                      order: int = 2) -> np.ndarray:
    """Subtract best-fit polynomial surface up to given order."""
    ny, nx = z.shape
    xv, yv = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    mask = np.isfinite(z)

    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((xv[mask] ** i) * (yv[mask] ** j))
    A = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)

    surface = np.zeros_like(z)
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            surface += coeffs[idx] * (xv ** i) * (yv ** j)
            idx += 1
    return z - surface


# ---------------------------------------------------------------------------
# Outlier filtering – Hampel with mm-based window
# ---------------------------------------------------------------------------
def hampel_filter_1d(profile: np.ndarray, win: int = 7,
                     threshold: float = 3.5) -> np.ndarray:
    """Hampel identifier – replace outliers with local median.

    Parameters
    ----------
    win : int
        Number of samples on each side (total 2*win+1).
    threshold : float
        Number of robust sigma for outlier classification (K factor).
    """
    n = len(profile)
    out = profile.copy()
    for i in range(win, n - win):
        window = profile[i - win:i + win + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 3:
            continue
        med = np.nanmedian(valid)
        mad = 1.4826 * np.nanmedian(np.abs(valid - med))
        if mad > 0 and np.isfinite(profile[i]):
            if np.abs(profile[i] - med) > threshold * mad:
                out[i] = med
    return out


def median_filter_1d(profile: np.ndarray, size: int = 5) -> np.ndarray:
    return ndimage.median_filter(profile, size=size).astype(profile.dtype)


def filter_outliers_profile(profile: np.ndarray, method: str,
                            window: int = 7,
                            threshold: float = 3.5) -> np.ndarray:
    if method == "hampel":
        return hampel_filter_1d(profile, window, threshold)
    elif method == "median":
        return median_filter_1d(profile, window)
    return profile


# ---------------------------------------------------------------------------
# Missing-value interpolation
# ---------------------------------------------------------------------------
def interpolate_gaps(profile: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN gaps in a 1-D profile."""
    nans = np.isnan(profile)
    if not nans.any():
        return profile
    if nans.all():
        return profile
    x = np.arange(len(profile))
    out = profile.copy()
    out[nans] = np.interp(x[nans], x[~nans], profile[~nans])
    return out


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------
def detrend_profile(profile: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Remove mean or linear trend from a single profile."""
    if mode == "mean":
        return profile - np.nanmean(profile)
    elif mode == "linear":
        x = np.arange(len(profile), dtype=np.float64)
        mask = np.isfinite(profile)
        if mask.sum() < 2:
            return profile
        coeffs = np.polyfit(x[mask], profile[mask], 1)
        return profile - np.polyval(coeffs, x)
    return profile


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------
def fft_bandpass(profile: np.ndarray, dx: float,
                 wl_low: float, wl_high: float) -> np.ndarray:
    """FFT bandpass keeping wavelengths between wl_low and wl_high (same units as dx)."""
    n = len(profile)
    freqs = np.fft.rfftfreq(n, d=dx)
    fft_vals = np.fft.rfft(profile)

    f_low = 1.0 / wl_high if wl_high > 0 else 0.0
    f_high = 1.0 / wl_low if wl_low > 0 else freqs[-1]

    mask = (freqs >= f_low) & (freqs <= f_high)
    fft_vals[~mask] = 0.0
    return np.fft.irfft(fft_vals, n=n)


def iir_bandpass(profile: np.ndarray, dx: float,
                 wl_low: float, wl_high: float,
                 order: int = 4) -> np.ndarray:
    """Butterworth IIR bandpass filter."""
    fs = 1.0 / dx
    f_low = 1.0 / wl_high
    f_high = 1.0 / wl_low
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    low = max(low, 1e-6)
    high = min(high, 0.9999)
    if low >= high:
        return profile
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, profile).astype(profile.dtype)


# ---------------------------------------------------------------------------
# Block-average downsampling
# ---------------------------------------------------------------------------
def block_average_resample(z: np.ndarray, dx_native: float, dy_native: float,
                           dx_target: float, dy_target: float,
                           min_valid_frac: float = 0.9) -> Tuple[np.ndarray, dict]:
    """Downsample a grid by block averaging (anti-alias).

    Returns the downsampled grid and a log dict.
    """
    rx = max(1, round(dx_target / dx_native))
    ry = max(1, round(dy_target / dy_native))

    ny, nx = z.shape
    ny_trim = (ny // ry) * ry
    nx_trim = (nx // rx) * rx

    z_trim = z[:ny_trim, :nx_trim]
    blocks = z_trim.reshape(ny_trim // ry, ry, nx_trim // rx, rx)

    valid_count = np.sum(np.isfinite(blocks), axis=(1, 3))
    total_count = ry * rx
    z_ds = np.nanmean(blocks, axis=(1, 3))

    # Mark blocks with too few valid points as NaN
    invalid_mask = valid_count < (min_valid_frac * total_count)
    z_ds[invalid_mask] = np.nan

    log = {
        "rx": rx, "ry": ry,
        "input_shape": z.shape,
        "output_shape": z_ds.shape,
        "invalid_blocks": int(invalid_mask.sum()),
    }
    return z_ds, log


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------
def extract_profiles(z: np.ndarray, direction: str = "longitudinal",
                     every_n: int = 1) -> list[np.ndarray]:
    """Extract 1-D profiles from the grid.

    direction: 'longitudinal' scans along rows,
               'transverse' scans along columns.
    """
    if direction == "longitudinal":
        return [z[i, :] for i in range(0, z.shape[0], every_n)]
    else:
        return [z[:, j] for j in range(0, z.shape[1], every_n)]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_surface(z: np.ndarray, dx: float, dy: float,
                       cfg: PreprocessingConfig,
                       direction: str = "longitudinal",
                       every_n: int = 1,
                       progress_cb=None) -> Tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Run the entire pre-processing pipeline.

    Returns:
        z_processed: 2-D processed grid
        profiles: list of 1-D cleaned profiles
        warnings: list of warning strings
    """
    warnings: list[str] = []
    z_out = z.copy().astype(np.float64)

    # 1) Plane / polynomial removal (on 2-D grid)
    if cfg.plane_removal == "plane":
        z_out = remove_plane(z_out, dx, dy)
    elif cfg.plane_removal == "polynomial":
        z_out = remove_polynomial(z_out, dx, dy, cfg.poly_order)

    # 2) Clean full surface grid in BOTH directions (rows then columns)
    #    Convert window from samples to mm-based half-window:
    #    w_pts = odd(round(window_mm / dx)) where window_mm ~ outlier_window * dx
    #    The outlier_window from the UI is in samples, so we convert to half-window.

    def _clean_1d(p, step_dx):
        """Apply the full per-profile cleaning chain."""
        missing_frac = np.isnan(p).sum() / len(p)
        if missing_frac > cfg.max_missing_fraction:
            return p, False  # skip this line

        if cfg.interp_missing:
            p = interpolate_gaps(p)

        if cfg.outlier_method != "none":
            # Convert window to half-window for Hampel
            half_win = cfg.outlier_window // 2
            half_win = max(1, half_win)
            p = filter_outliers_profile(
                p, cfg.outlier_method,
                half_win, cfg.outlier_threshold)

        if cfg.detrend_mode != "none":
            p = detrend_profile(p, cfg.detrend_mode)

        if cfg.bandpass:
            if cfg.bandpass_method == "fft":
                p = fft_bandpass(p, step_dx, cfg.bandpass_low, cfg.bandpass_high)
            else:
                p = iir_bandpass(p, step_dx, cfg.bandpass_low, cfg.bandpass_high)

        return p, True

    ny, nx = z_out.shape
    total = ny + nx  # both passes

    # Pass 1: clean along rows (longitudinal)
    for i in range(ny):
        p, ok = _clean_1d(z_out[i, :].copy(), dx)
        if ok:
            z_out[i, :] = p
        if progress_cb and i % max(1, ny // 20) == 0:
            progress_cb(0.5 * i / ny)

    # Pass 2: clean along columns (transverse)
    for j in range(nx):
        p, ok = _clean_1d(z_out[:, j].copy(), dy)
        if ok:
            z_out[:, j] = p
        if progress_cb and j % max(1, nx // 20) == 0:
            progress_cb(0.5 + 0.5 * j / nx)

    # 3) Extract targeted profiles
    raw_profiles = extract_profiles(z_out, direction, every_n)
    profiles: list[np.ndarray] = []

    for idx, p in enumerate(raw_profiles):
        missing_frac = np.isnan(p).sum() / len(p)
        if missing_frac > cfg.max_missing_fraction:
            warnings.append(
                f"Profile {idx}: {missing_frac:.1%} missing – rejected "
                f"(threshold {cfg.max_missing_fraction:.1%})")
            continue
        profiles.append(p)

    if not profiles:
        warnings.append("All profiles were rejected – check data quality!")

    return z_out, profiles, warnings
