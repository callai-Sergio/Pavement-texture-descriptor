# preprocessing.py
"""Preprocessing module for regular‑grid pavement laser texture data.
Implements the stages described in the user specification:
1. Plane removal / detrending
2. Gap interpolation (linear)
3. Outlier (spike) removal using a Hampel filter
4. Macro‑texture down‑sampling (block averaging)
5. ISO‑13473‑1 macro‑texture filter chain (high‑pass & low‑pass Butterworth)
6. Spectral preparation for 1/3‑octave or PSD analysis
All functions operate on NumPy arrays and return cleaned data together with a log dictionary.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _log_entry(log: Dict[str, Any], key: str, value: Any) -> None:
    """Append a value to a list entry in the log dictionary.
    Creates the list if the key does not exist.
    """
    if key not in log:
        log[key] = []
    log[key].append(value)

# ---------------------------------------------------------------------------
# Stage A – Plane removal / detrending
# ---------------------------------------------------------------------------

def fit_plane(z: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Fit a plane z = a*x + b*y + c to the valid points of *z*.

    Parameters
    ----------
    z : 2‑D array (height field)
    mask : optional boolean array of the same shape where ``True`` marks
           valid samples. If ``None`` all finite values are used.

    Returns
    -------
    z_fit : 2‑D array of the fitted plane values
    coeffs : (a, b, c) plane coefficients
    """
    ny, nx = z.shape
    y, x = np.mgrid[0:ny, 0:nx]
    if mask is None:
        mask = np.isfinite(z)
    # Stack coordinates and a constant term for least‑squares
    A = np.column_stack((x[mask].ravel(), y[mask].ravel(), np.ones(mask.sum())))
    b = z[mask].ravel()
    # Solve using least‑squares (robust regression could be added later)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, c = coeffs
    z_fit = a * x + b_coef * y + c
    return z_fit, (a, b_coef, c)


def remove_plane(z: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Subtract the fitted plane from *z* and return the leveled surface.
    Also returns a log dictionary with the plane coefficients.
    """
    log: Dict[str, Any] = {}
    z_fit, coeffs = fit_plane(z, mask)
    _log_entry(log, "plane_coefficients", coeffs)
    # Leveled surface (preserve NaNs for invalid points)
    z_leveled = np.where(np.isfinite(z), z - z_fit, np.nan)
    return z_leveled, log

# ---------------------------------------------------------------------------
# Stage B – Gap interpolation (1‑D linear)
# ---------------------------------------------------------------------------

def interpolate_gaps_1d(profile: np.ndarray, max_gap: int = 20) -> Tuple[np.ndarray, int]:
    """Linear interpolation of NaNs in a 1‑D profile.
    Gaps longer than *max_gap* samples are left as NaN.
    Returns the interpolated profile and the number of points filled.
    """
    isnan = np.isnan(profile)
    if not isnan.any():
        return profile.copy(), 0
    filled = profile.copy()
    n_filled = 0
    # Find indices of valid points
    valid_idx = np.where(~isnan)[0]
    for start, end in zip(valid_idx[:-1], valid_idx[1:]):
        gap_len = end - start - 1
        if 0 < gap_len <= max_gap:
            # Linear interpolation between the two valid ends
            start_val, end_val = profile[start], profile[end]
            interp_vals = np.linspace(start_val, end_val, gap_len + 2)[1:-1]
            filled[start + 1 : end] = interp_vals
            n_filled += gap_len
    return filled, n_filled


def interpolate_grid_gaps(z: np.ndarray, direction: str = "rows", max_gap: int = 20) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Interpolate gaps along rows or columns.
    Returns the interpolated grid and a log with statistics.
    """
    log: Dict[str, Any] = {"interpolated_points": 0, "rejected_profiles": 0}
    ny, nx = z.shape
    if direction == "rows":
        for i in range(ny):
            profile = z[i, :]
            interp_profile, filled = interpolate_gaps_1d(profile, max_gap)
            log["interpolated_points"].append(filled)
            # Determine if the profile should be rejected (>10% invalid after interpolation)
            invalid_frac = np.isnan(interp_profile).sum() / nx
            if invalid_frac > 0.10:
                log["rejected_profiles"].append(i)
                continue
            z[i, :] = interp_profile
    elif direction == "cols":
        for j in range(nx):
            profile = z[:, j]
            interp_profile, filled = interpolate_gaps_1d(profile, max_gap)
            log["interpolated_points"].append(filled)
            invalid_frac = np.isnan(interp_profile).sum() / ny
            if invalid_frac > 0.10:
                log["rejected_profiles"].append(j)
                continue
            z[:, j] = interp_profile
    else:
        raise ValueError("direction must be 'rows' or 'cols'")
    return z, log

# ---------------------------------------------------------------------------
# Stage C – Hampel outlier filter
# ---------------------------------------------------------------------------

def hampel_filter_1d(
    profile: np.ndarray,
    window_size: int = 45,
    n_sigma: float = 3.5,
    replace: str = "median",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply a Hampel filter to a 1‑D profile.

    Parameters
    ----------
    profile : 1‑D array (NaNs are ignored)
    window_size : number of samples on each side of the centre (total length = 2*window_size+1)
    n_sigma : threshold multiplier for the robust sigma estimate
    replace : "median" (replace with local median) or "interp" (linear interpolation of neighbours)
    """
    n = len(profile)
    filtered = profile.copy()
    log = {"outliers": 0, "replacements": 0, "max_deviation": 0.0}
    half_win = window_size
    for i in range(n):
        if np.isnan(profile[i]):
            continue
        start = max(i - half_win, 0)
        end = min(i + half_win + 1, n)
        window = profile[start:end]
        # Exclude NaNs from statistics
        window = window[~np.isnan(window)]
        if len(window) < 3:
            continue
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        sigma = 1.4826 * mad if mad > 0 else 0.0
        if sigma == 0:
            continue
        deviation = np.abs(profile[i] - median)
        if deviation > n_sigma * sigma:
            log["outliers"] += 1
            log["max_deviation"] = max(log["max_deviation"], deviation)
            if replace == "median":
                filtered[i] = median
            elif replace == "interp":
                # Linear interpolation between nearest valid neighbours
                left = i - 1
                while left >= 0 and np.isnan(profile[left]):
                    left -= 1
                right = i + 1
                while right < n and np.isnan(profile[right]):
                    right += 1
                if left >= 0 and right < n:
                    filtered[i] = np.interp(i, [left, right], [profile[left], profile[right]])
                else:
                    filtered[i] = median
            log["replacements"] += 1
    return filtered, log


def hampel_filter_grid(
    z: np.ndarray,
    direction: str = "rows",
    window_mm: float = 0.5,
    dx: float = 0.011,
    n_sigma: float = 3.5,
    replace: str = "median",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the Hampel filter along rows or columns of a grid.
    ``window_mm`` is converted to a sample count using the grid spacing ``dx``.
    Returns the filtered grid and a combined log.
    """
    window_samples = max(int(np.round(window_mm / dx)), 1)
    ny, nx = z.shape
    total_log = {"outliers": 0, "replacements": 0, "max_deviation": 0.0}
    if direction == "rows":
        for i in range(ny):
            filtered, log = hampel_filter_1d(z[i, :], window_samples, n_sigma, replace)
            z[i, :] = filtered
            for k in total_log:
                total_log[k] += log.get(k, 0)
    elif direction == "cols":
        for j in range(nx):
            filtered, log = hampel_filter_1d(z[:, j], window_samples, n_sigma, replace)
            z[:, j] = filtered
            for k in total_log:
                total_log[k] += log.get(k, 0)
    else:
        raise ValueError("direction must be 'rows' or 'cols'")
    return z, total_log

# ---------------------------------------------------------------------------
# Stage D – Macro‑texture down‑sampling (block averaging)
# ---------------------------------------------------------------------------

def downsample_grid(z: np.ndarray, target_dx: float = 0.55, dx: float = 0.011) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Down‑sample a regular grid by block averaging.
    Returns the macro‑grid and a log with the down‑sampling factor.
    """
    factor = int(np.round(target_dx / dx))
    if factor <= 1:
        return z.copy(), {"downsample_factor": 1, "dx_macro": dx}
    ny, nx = z.shape
    # Trim to a multiple of factor
    ny_trim = ny - (ny % factor)
    nx_trim = nx - (nx % factor)
    z_trim = z[:ny_trim, :nx_trim]
    # Reshape and average
    z_macro = z_trim.reshape(ny_trim // factor, factor, nx_trim // factor, factor).mean(axis=(1, 3))
    log = {"downsample_factor": factor, "dx_macro": dx * factor, "dy_macro": dx * factor, "original_shape": (ny, nx), "macro_shape": z_macro.shape}
    return z_macro, log

# ---------------------------------------------------------------------------
# Stage E – ISO‑13473‑1 macro‑texture filter chain (MPD)
# ---------------------------------------------------------------------------

def iso_mpd_filter(profile: np.ndarray, dx_macro: float, hp_cut_mm: float = 140.0, lp_cut_mm: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply a zero‑phase Butterworth high‑pass and low‑pass filter to a macro‑profile.
    The cutoff wavelengths are given in millimetres.
    """
    fs = 1.0 / dx_macro  # samples per mm
    # Normalised cut‑off frequencies (Nyquist = fs/2)
    hp_norm = (1.0 / hp_cut_mm) / (fs / 2.0)
    lp_norm = (1.0 / lp_cut_mm) / (fs / 2.0)
    # Design 2nd‑order Butterworth filters
    b_hp, a_hp = signal.butter(N=2, Wn=hp_norm, btype="high", analog=False)
    b_lp, a_lp = signal.butter(N=2, Wn=lp_norm, btype="low", analog=False)
    # Apply forward‑reverse filtering to avoid phase shift
    filtered = signal.filtfilt(b_hp, a_hp, profile, method="gust")
    filtered = signal.filtfilt(b_lp, a_lp, filtered, method="gust")
    log = {
        "dx_macro": dx_macro,
        "hp_cut_mm": hp_cut_mm,
        "lp_cut_mm": lp_cut_mm,
        "hp_norm": hp_norm,
        "lp_norm": lp_norm,
    }
    return filtered, log

# ---------------------------------------------------------------------------
# Stage F – Spectral preparation (Welch / FFT, 1/3‑octave)
# ---------------------------------------------------------------------------

def spectral_prep(
    profile: np.ndarray,
    dx: float,
    window: str = "hann",
    nperseg: int = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare a profile for spectral analysis.
    Returns frequency (mm⁻¹), PSD values and a log.
    """
    if nperseg is None:
        nperseg = min(256, len(profile))
    freqs, psd = signal.welch(profile, fs=1.0 / dx, window=window, nperseg=nperseg, scaling="density")
    log = {"dx": dx, "nperseg": nperseg, "window": window}
    return freqs, psd, log

# ---------------------------------------------------------------------------
# Public API – one‑stop preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_surface(
    z: np.ndarray,
    dx: float = 0.011,
    dy: float = None,
    direction: str = "rows",
    max_gap: int = 20,
    hampel_window_mm: float = 0.5,
    hampel_n_sigma: float = 3.5,
    target_dx_macro: float = 0.55,
    iso_filter: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Full preprocessing pipeline returning cleaned full‑resolution and macro‑resolution data.
    The function returns two dictionaries:
    * ``full`` – contains ``z_clean`` and a ``log`` for the full‑resolution steps.
    * ``macro`` – contains ``z_macro`` (down‑sampled), optionally ``z_iso`` (filtered), and a ``log``.
    """
    if dy is None:
        dy = dx
    log_full: Dict[str, Any] = {}
    # 1. Plane removal
    z_plane, log_plane = remove_plane(z)
    log_full.update(log_plane)
    # 2. Gap interpolation
    z_interp, log_interp = interpolate_grid_gaps(z_plane, direction=direction, max_gap=max_gap)
    log_full.update(log_interp)
    # 3. Hampel outlier removal
    z_clean, log_hampel = hampel_filter_grid(
        z_interp,
        direction=direction,
        window_mm=hampel_window_mm,
        dx=dx,
        n_sigma=hampel_n_sigma,
        replace="median",
    )
    log_full.update(log_hampel)
    # Package full‑resolution result
    full_res = {"z_clean": z_clean, "log": log_full}
    # 4. Down‑sample for macro‑texture
    z_macro, log_macro = downsample_grid(z_clean, target_dx=target_dx_macro, dx=dx)
    macro_res: Dict[str, Any] = {"z_macro": z_macro, "log": log_macro}
    # 5. Optional ISO MPD filter on macro profiles (apply per row/col as needed by the caller)
    if iso_filter:
        # Example: filter each row profile; callers can also filter columns similarly.
        iso_profiles = []
        iso_logs = []
        for row in z_macro:
            filtered, iso_log = iso_mpd_filter(row, dx_macro=log_macro["dx_macro"])
            iso_profiles.append(filtered)
            iso_logs.append(iso_log)
        macro_res["z_iso"] = np.array(iso_profiles)
        macro_res["iso_log"] = iso_logs
    return full_res, macro_res

# ---------------------------------------------------------------------------
# End of module
# ---------------------------------------------------------------------------
