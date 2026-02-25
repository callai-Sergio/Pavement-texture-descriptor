"""
descriptors.py – Lean texture parameter calculations for TextureLab Desktop

Computes selected 1-D profile and 3-D areal texture descriptors following
ISO 13473, ISO 4287, ISO 25178, ISO 13565, and ISO 10844 standards.

Lean set: Core stats, Abbott-Firestone, 3D Areal, PSD/ENDT/TextureLevel, FractalDim.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sig


# ===================================================================
# Core profile statistics (ISO 4287)
# ===================================================================

def calc_ra(profile: np.ndarray) -> float:
    z = profile - np.nanmean(profile)
    return float(np.nanmean(np.abs(z)))


def calc_rq(profile: np.ndarray) -> float:
    z = profile - np.nanmean(profile)
    return float(np.sqrt(np.nanmean(z ** 2)))


def calc_rsk(profile: np.ndarray) -> float:
    z = profile - np.nanmean(profile)
    rq = np.sqrt(np.nanmean(z ** 2))
    if rq == 0:
        return 0.0
    return float(np.nanmean(z ** 3) / rq ** 3)


def calc_rku(profile: np.ndarray) -> float:
    z = profile - np.nanmean(profile)
    rq = np.sqrt(np.nanmean(z ** 2))
    if rq == 0:
        return 0.0
    return float(np.nanmean(z ** 4) / rq ** 4)


def calc_rz_iso(profile: np.ndarray, n_segments: int = 5) -> float:
    """Rz per ISO 4287: mean peak-to-valley across n_segments."""
    z = profile[np.isfinite(profile)]
    seg_len = len(z) // n_segments
    if seg_len < 2:
        return float(np.ptp(z))
    pv = []
    for i in range(n_segments):
        seg = z[i * seg_len:(i + 1) * seg_len]
        pv.append(np.max(seg) - np.min(seg))
    return float(np.mean(pv))


def calc_core_stats_1d(profile: np.ndarray) -> dict:
    """Comprehensive profile statistics."""
    z = profile[np.isfinite(profile)]
    if len(z) == 0:
        return {}
    zm = z - np.mean(z)
    return {
        "Ra": float(np.mean(np.abs(zm))),
        "Rq": float(np.sqrt(np.mean(zm ** 2))),
        "Rsk": calc_rsk(z),
        "Rku": calc_rku(z),
        "Rz": calc_rz_iso(z),
        "Rt": float(np.ptp(z)),
        "Rp": float(np.max(zm)),
        "Rv": float(np.abs(np.min(zm))),
        "StdDev": float(np.std(z)),
        "P5": float(np.percentile(z, 5)),
        "P10": float(np.percentile(z, 10)),
        "P50": float(np.percentile(z, 50)),
        "P90": float(np.percentile(z, 90)),
        "P95": float(np.percentile(z, 95)),
        "IQR": float(np.percentile(z, 75) - np.percentile(z, 25)),
    }


# ===================================================================
# 3-D Areal statistics (ISO 25178)
# ===================================================================

def calc_sdr(z: np.ndarray, dx: float, dy: float) -> float:
    """Developed interfacial area ratio Sdr (ISO 25178)."""
    dzdx = np.diff(z, axis=1) / dx
    dzdy = np.diff(z, axis=0) / dy
    nr = min(dzdx.shape[0], dzdy.shape[0])
    nc = min(dzdx.shape[1], dzdy.shape[1])
    dzdx = dzdx[:nr, :nc]
    dzdy = dzdy[:nr, :nc]
    actual = np.nansum(np.sqrt(1 + dzdx ** 2 + dzdy ** 2)) * dx * dy
    projected = nr * nc * dx * dy
    if projected == 0:
        return 0.0
    return float((actual - projected) / projected * 100.0)


def calc_void_material_volume(z: np.ndarray,
                              mr_pct: float = 80.0) -> Tuple[float, float]:
    """Void volume (Vv) and material volume (Vm)."""
    vals = z[np.isfinite(z)]
    if len(vals) == 0:
        return 0.0, 0.0
    sorted_z = np.sort(vals)[::-1]
    n = len(sorted_z)
    cut_idx = min(int(mr_pct / 100.0 * n), n - 1)
    z_cut = sorted_z[cut_idx]
    above = vals[vals > z_cut] - z_cut
    below = z_cut - vals[vals <= z_cut]
    return float(np.sum(below) / n), float(np.sum(above) / n)


def calc_core_stats_3d(z: np.ndarray, dx: float = 1.0,
                       dy: float = 1.0) -> dict:
    """Comprehensive 3-D areal statistics."""
    vals = z[np.isfinite(z)]
    if len(vals) == 0:
        return {}
    m = np.mean(vals)
    zc = vals - m
    sq = np.sqrt(np.mean(zc ** 2))
    ssk = np.mean(zc ** 3) / sq ** 3 if sq > 0 else 0.0
    sku = np.mean(zc ** 4) / sq ** 4 if sq > 0 else 0.0
    vv, vm = calc_void_material_volume(z)
    return {
        "Sa": float(np.mean(np.abs(zc))),
        "Sq": float(sq),
        "Ssk": float(ssk),
        "Sku": float(sku),
        "Sz": float(np.ptp(vals)),
        "Sv": float(np.abs(np.min(zc))),
        "Sdr": calc_sdr(z, dx, dy),
        "Vv": vv,
        "Vm": vm,
    }


# ===================================================================
# Abbott–Firestone / ISO 13565
# ===================================================================

def material_ratio_height(profile: np.ndarray, mr_pct: float) -> float:
    z = np.sort(profile[np.isfinite(profile)])[::-1]
    idx = int(np.clip(mr_pct / 100.0 * len(z), 0, len(z) - 1))
    return float(z[idx])


def calc_rk_params(profile: np.ndarray) -> dict:
    """Rk, Rpk, Rvk, Mr1, Mr2 via the 40% kernel approach (ISO 13565-2)."""
    z = np.sort(profile[np.isfinite(profile)])[::-1]
    n = len(z)
    if n < 10:
        return {"Rk": 0.0, "Rpk": 0.0, "Rvk": 0.0, "Mr1": 0.0, "Mr2": 0.0}
    mr = np.linspace(0, 100, n)
    best_start, best_slope = 0, np.inf
    span = int(0.4 * n)
    for s in range(n - span):
        slope = abs(z[s] - z[s + span]) / 40.0
        if slope < best_slope:
            best_slope = slope
            best_start = s
    return {
        "Rk": float(z[best_start] - z[best_start + span]),
        "Rpk": float(z[0] - z[best_start]),
        "Rvk": float(z[best_start + span] - z[-1]),
        "Mr1": float(mr[best_start]),
        "Mr2": float(mr[best_start + span]),
    }


def calc_g_factor(profile: np.ndarray) -> float:
    """g-factor (ISO 10844:2021): material ratio at mid-height."""
    z = profile[np.isfinite(profile)]
    if len(z) == 0:
        return 0.0
    z_mid = (np.max(z) + np.min(z)) / 2.0
    return float(np.sum(z >= z_mid) / len(z))


# ===================================================================
# ISO 13473-1: MPD, ETD
# ===================================================================

def calc_mpd(profile: np.ndarray, segment_mm: float = 100.0,
             dx: float = 1.0) -> Tuple[float, List[float]]:
    """Mean Profile Depth per ISO 13473-1."""
    seg_pts = max(2, int(segment_mm / dx))
    n_seg = max(1, len(profile) // seg_pts)
    msds: List[float] = []
    for i in range(n_seg):
        seg = profile[i * seg_pts:(i + 1) * seg_pts]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 2:
            continue
        half = len(seg) // 2
        pk1 = np.max(seg[:half])
        pk2 = np.max(seg[half:])
        msds.append(float((pk1 + pk2) / 2.0 - np.mean(seg)))
    return (float(np.mean(msds)) if msds else 0.0), msds


def calc_etd(mpd: float) -> float:
    return 0.2 + 0.8 * mpd


# ===================================================================
# Spectral: PSD, ENDT, TextureLevel
# ===================================================================

def calc_psd_welch(profile: np.ndarray, dx: float,
                   nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, len(profile))
    freqs, psd = sig.welch(profile, fs=1.0 / dx, nperseg=nperseg,
                           detrend="linear")
    return freqs, psd


def calc_endt(freqs: np.ndarray, psd: np.ndarray) -> float:
    """ENDT (ISO 10844:2014) – spectral weighted noise estimate."""
    if len(freqs) < 2:
        return 0.0
    df = freqs[1] - freqs[0]
    weights = freqs / (freqs[-1] + 1e-15)
    return float(10.0 * np.log10(np.sum(weights * psd * df) + 1e-15))


def calc_texture_level(profile: np.ndarray, dx: float) -> float:
    """Texture level LT in dB re 1 µm."""
    freqs, psd = calc_psd_welch(profile, dx)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_rms = float(np.sqrt(np.sum(psd) * df))
    return float(20.0 * np.log10(total_rms + 1e-12))


# ===================================================================
# Fractal dimension (box-counting)
# ===================================================================

def calc_fractal_dimension(profile: np.ndarray) -> float:
    """Fractal dimension via box-counting (1-D profile)."""
    z = profile[np.isfinite(profile)]
    if len(z) < 8:
        return 1.0
    z_norm = (z - np.min(z))
    rng = np.max(z_norm)
    if rng == 0:
        return 1.0
    z_norm = z_norm / rng

    sizes, counts = [], []
    n = len(z_norm)
    for k in [2, 4, 8, 16, 32, 64]:
        if k > n // 2:
            break
        box_size = n / k
        count = 0
        for i in range(k):
            seg = z_norm[int(i * box_size):int((i + 1) * box_size)]
            if len(seg) == 0:
                continue
            count += int(np.ceil((np.max(seg) - np.min(seg)) / (1.0 / k))) + 1
        if count > 0:
            sizes.append(1.0 / k)
            counts.append(count)

    if len(sizes) < 2:
        return 1.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(-coeffs[0])


# ===================================================================
# Compute all for a single profile
# ===================================================================

def compute_profile_params(profile: np.ndarray, dx: float) -> dict:
    """Compute all selected descriptors for one profile."""
    results = {}
    results.update(calc_core_stats_1d(profile))
    results.update(calc_rk_params(profile))
    results["g_factor"] = calc_g_factor(profile)

    mpd, _ = calc_mpd(profile, segment_mm=100.0, dx=dx)
    results["MPD"] = mpd
    results["ETD"] = calc_etd(mpd)

    freqs, psd = calc_psd_welch(profile, dx)
    results["ENDT"] = calc_endt(freqs, psd)
    results["TextureLevel"] = calc_texture_level(profile, dx)
    results["FractalDim"] = calc_fractal_dimension(profile)

    return results


def compute_all(profiles: List[np.ndarray], z_grid: np.ndarray,
                dx: float, dy: float,
                progress_cb=None) -> Tuple[List[dict], dict]:
    """Compute descriptors for all profiles + areal 3-D stats."""
    per_profile: List[dict] = []
    total = len(profiles)
    for idx, p in enumerate(profiles):
        per_profile.append(compute_profile_params(p, dx))
        if progress_cb and idx % max(1, total // 20) == 0:
            progress_cb(int(100 * idx / total))

    areal = calc_core_stats_3d(z_grid, dx, dy)
    return per_profile, areal


def aggregate_profiles(per_profile: List[dict],
                       mode: str = "mean") -> dict:
    """Aggregate per-profile results into a summary row."""
    if not per_profile:
        return {}
    keys = per_profile[0].keys()
    agg: dict = {}
    for k in keys:
        vals = [p[k] for p in per_profile
                if isinstance(p.get(k), (int, float))]
        if not vals:
            continue
        arr = np.array(vals)
        if mode == "mean":
            agg[k] = float(np.mean(arr))
        elif mode == "median":
            agg[k] = float(np.median(arr))
        elif mode == "trimmed_mean":
            from scipy.stats import trim_mean
            agg[k] = float(trim_mean(arr, 0.1))
        else:
            agg[k] = float(np.mean(arr))
        agg[f"{k}_std"] = float(np.std(arr))
    return agg
