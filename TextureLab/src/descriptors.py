"""
descriptors.py – Texture parameter calculations for TextureLab

Computes 1-D profile and 3-D areal texture descriptors following
ISO 13473, ISO 4287, ISO 25178, ISO 13565, and ISO 10844 standards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sig


# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------
@dataclass
class ParamMeta:
    """Metadata for a single descriptor."""
    name: str
    symbol: str
    definition: str
    equation_latex: str
    dim: str            # "1D" / "3D" / "1D/3D"
    standard: str       # ISO reference
    unit: str
    noise: str = "–"    # relevance: Very High / High / Medium / Low / No
    friction: str = "–"
    drainage: str = "–"
    tags: List[str] = field(default_factory=list)


PARAM_REGISTRY: Dict[str, ParamMeta] = {}


def _reg(name, symbol, defn, eq, dim, std, unit,
         noise="–", friction="–", drainage="–", tags=None):
    PARAM_REGISTRY[name] = ParamMeta(
        name, symbol, defn, eq, dim, std, unit,
        noise, friction, drainage, tags or [])


# ── Core profile statistics (ISO 4287) ────────────────────────────────────
_reg("Ra", "Ra", "Arithmetic mean absolute deviation",
     r"Ra = \frac{1}{L}\int|z(x)|dx", "2D", "ISO 4287", "µm",
     "Medium", "Medium", "Low", ["friction"])
_reg("Rq", "Rq", "Root mean square height (RMS)",
     r"Rq = \sqrt{\frac{1}{L}\int z^2 dx}", "2D", "ISO 4287", "µm",
     "High", "Medium", "Low", ["friction"])
_reg("Rz", "Rz", "Mean peak-to-valley height",
     r"Rz = \frac{1}{5}\sum(Z_{pi}-Z_{vi})", "2D", "ISO 4287", "µm",
     "High", "Medium", "Low")
_reg("Rt", "Rt", "Total profile height",
     r"Rt = z_{max} - z_{min}", "2D", "ISO 4287", "µm",
     "Medium", "Medium", "Medium")
_reg("Rp", "Rp", "Maximum peak height",
     r"Rp = z_{max}", "2D", "ISO 4287", "µm",
     "Medium", "Medium", "Low")
_reg("Rv", "Rv", "Maximum valley depth",
     r"Rv = |z_{min}|", "2D", "ISO 4287", "µm",
     "Medium", "Medium", "High", ["drainage"])
_reg("Rsk", "Rsk", "Skewness (asymmetry of height distribution)",
     r"Rsk = \frac{1}{Rq^3}\frac{1}{L}\int z^3 dx", "2D", "ISO 4287", "–",
     "High", "Medium", "Medium", ["drainage"])
_reg("Rku", "Rku", "Kurtosis (peakedness)",
     r"Rku = \frac{1}{Rq^4}\frac{1}{L}\int z^4 dx", "2D", "ISO 4287", "–",
     "High", "Low", "Low")

# ── Statistical descriptors ───────────────────────────────────────────────
_reg("StdDev", "σz", "Standard deviation of heights",
     r"\sigma = \sqrt{\frac{1}{N}\sum(z_i-\bar{z})^2}", "2D/3D",
     "Statistical", "µm", "Medium", "Medium", "Low")
_reg("P5", "P5", "5th percentile height", r"P_5", "2D/3D",
     "Statistical", "µm", "Medium", "Medium", "High")
_reg("P10", "P10", "10th percentile height", r"P_{10}", "2D/3D",
     "Statistical", "µm", "Medium", "Medium", "High")
_reg("P50", "P50", "Median height", r"P_{50}", "2D/3D",
     "Statistical", "µm", "Medium", "Medium", "High")
_reg("P90", "P90", "90th percentile height", r"P_{90}", "2D/3D",
     "Statistical", "µm", "Medium", "Medium", "High")
_reg("P95", "P95", "95th percentile height", r"P_{95}", "2D/3D",
     "Statistical", "µm", "Medium", "Medium", "High")
_reg("IQR", "IQR", "Inter-quartile range",
     r"Q_{75}-Q_{25}", "2D/3D", "Statistical", "µm",
     "Medium", "Medium", "Medium")

# ── Abbott–Firestone / ISO 13565 ──────────────────────────────────────────
_reg("Rk", "Rk", "Core roughness depth (Abbott curve)",
     r"Rk", "2D", "ISO 13565", "µm", "High", "Medium", "Medium",
     ["friction"])
_reg("Rpk", "Rpk", "Reduced peak height",
     r"Rpk", "2D", "ISO 13565", "µm", "Medium", "High", "Low")
_reg("Rvk", "Rvk", "Reduced valley depth",
     r"Rvk", "2D", "ISO 13565", "µm", "Medium", "Medium", "High",
     ["drainage"])
_reg("Mr1", "Mr1", "Upper material ratio",
     r"Mr_1", "2D", "ISO 13565", "%", "Medium", "Medium", "High")
_reg("Mr2", "Mr2", "Lower material ratio",
     r"Mr_2", "2D", "ISO 13565", "%", "Medium", "Medium", "High")
_reg("Mr10", "Mr10%", "Height at 10% material ratio",
     r"z(Mr=10\%)", "2D", "ISO 13565", "µm", "Medium", "Medium", "Medium")
_reg("Mr50", "Mr50%", "Height at 50% material ratio",
     r"z(Mr=50\%)", "2D", "ISO 13565", "µm", "Medium", "Medium", "Medium")
_reg("Mr90", "Mr90%", "Height at 90% material ratio",
     r"z(Mr=90\%)", "2D", "ISO 13565", "µm", "Medium", "Medium", "Medium")
_reg("CoreDepth", "Dcore", "Core depth (Mr 10%-80%)",
     r"z(Mr_{10})-z(Mr_{80})", "2D", "ISO 13565", "µm",
     "Medium", "Medium", "Medium")

# ── ISO 13473 macrotexture ────────────────────────────────────────────────
_reg("MPD", "MPD", "Mean Profile Depth over 100 mm base length",
     r"MPD = \frac{h_1+h_2}{2} - \bar{h}", "2D", "ISO 13473-1", "mm",
     "High", "High", "Medium", ["noise", "friction"])
_reg("ETD", "ETD", "Estimated Texture Depth",
     r"ETD = 0.2 + 0.8\,MPD", "2D", "ISO 13473-1", "mm",
     "High", "High", "Medium", ["noise", "friction"])
_reg("MSD", "MSD", "Mean Segment Depth",
     r"MSD", "2D", "ISO 13473-1", "mm", "High", "High", "Medium")

# ── ISO 10844 / Spectral ──────────────────────────────────────────────────
_reg("g_factor", "g", "Material ratio at mid-height (Abbott curve)",
     r"D_{cum}(z_{mid})", "2D", "ISO 10844:2021", "–",
     "Very High", "Medium", "Medium")
_reg("ENDT", "ENDT", "Estimated noise difference from texture spectrum",
     r"\text{Spectral weighted sum}", "2D", "ISO 10844:2014", "dB",
     "High", "No", "No")
_reg("LTX", "LTX", "Log texture level per wavelength",
     r"10\log_{10}(PSD)", "2D", "ISO 10844", "dB",
     "High", "Low", "Low")
_reg("PSD_moments", "PSD_m", "Spectral energy metrics (moments of PSD)",
     r"\text{Statistical moments of PSD}", "2D", "Derived", "–",
     "High", "Low", "Low")
_reg("BandRMS", "RMS_band", "Band-limited RMS amplitude",
     r"\sqrt{\int_{f_1}^{f_2} PSD\,df}", "2D", "ISO 13473-4", "µm",
     "High", "Low", "Low", ["noise"])
_reg("TextureLevel", "LT", "Texture level (dB re 1 µm)",
     r"L_T = 20\log_{10}(RMS / 1\,\mu m)", "2D", "ISO 13473-4", "dB",
     "High", "Low", "Low")

# ── Spacing / slope / correlation (2D) ───────────────────────────────────
_reg("PeakDensity", "Pc", "Number of peaks per unit length",
     r"Count/L", "2D", "ISO 13473-1", "peaks/m",
     "High", "High", "Low")
_reg("Sm", "Sm", "Mean spacing between profile elements",
     r"\text{Average zero-crossing spacing}", "2D", "ISO 4287", "mm",
     "High", "High", "Low")
_reg("ACL", "Sal", "Autocorrelation length (decay to 0.2)",
     r"\tau : R(\tau)=0.2", "2D/3D", "ISO 25178", "mm",
     "High", "Medium", "Low")
_reg("MeanSlope", "Rdq", "Mean absolute slope",
     r"\frac{1}{L}\int|dz/dx|dx", "2D", "Derived", "–",
     "High", "High", "Low", ["friction"])
_reg("FractalDim", "D", "Fractal dimension (box-counting)",
     r"\text{Box-counting method}", "2D/3D", "Derived", "–",
     "Medium", "Medium", "Medium")

# ── 3-D Areal (ISO 25178) ────────────────────────────────────────────────
_reg("Sa", "Sa", "Arithmetic mean height (areal)",
     r"\frac{1}{A}\iint|z(x,y)|dA", "3D", "ISO 25178", "µm",
     "High", "High", "Medium")
_reg("Sq", "Sq", "RMS height (areal)",
     r"\sqrt{\frac{1}{A}\iint z^2 dA}", "3D", "ISO 25178", "µm",
     "High", "Medium", "Low")
_reg("Ssk", "Ssk", "Areal skewness",
     r"\text{3D equivalent of Rsk}", "3D", "ISO 25178", "–",
     "High", "Medium", "Medium")
_reg("Sku", "Sku", "Areal kurtosis",
     r"\text{3D equivalent of Rku}", "3D", "ISO 25178", "–",
     "High", "Low", "Low")
_reg("Sz", "Sz", "Maximum height of the surface",
     r"Sz = z_{max} - z_{min}", "3D", "ISO 25178", "µm",
     "Medium", "Medium", "Medium")
_reg("Sv", "Sv", "Maximum pit depth (3D)",
     r"Sv = |z_{min}|", "3D", "ISO 25178", "µm",
     "Medium", "Medium", "High", ["drainage"])
_reg("Sdr", "Sdr", "Developed interfacial area ratio",
     r"\text{Surface area increase \%}", "3D", "ISO 25178", "%",
     "Medium", "High", "High")
_reg("Vv", "Vv", "Void volume (from areal bearing curve)",
     r"Vv", "3D", "ISO 25178", "mm³/mm²",
     "Medium", "Medium", "Very High", ["drainage"])
_reg("Vm", "Vm", "Material volume (from areal bearing curve)",
     r"Vm", "3D", "ISO 25178", "mm³/mm²",
     "Medium", "Medium", "Very High")


# ===================================================================
# Calculation functions
# ===================================================================

# --- Core profile statistics (1D → "2D" in user convention) ----------------

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
        "Mean": float(np.mean(z)),
        "Min": float(np.min(z)),
        "Max": float(np.max(z)),
    }


# --- 3-D Areal statistics (ISO 25178) -------------------------------------

def calc_sdr(z: np.ndarray, dx: float, dy: float) -> float:
    """Developed interfacial area ratio Sdr (ISO 25178).

    Measures % increase of actual surface area vs projected area.
    """
    # Partial derivatives via finite differences
    dzdx = np.diff(z, axis=1) / dx
    dzdy = np.diff(z, axis=0) / dy
    # Trim to common shape
    nr = min(dzdx.shape[0], dzdy.shape[0])
    nc = min(dzdx.shape[1], dzdy.shape[1])
    dzdx = dzdx[:nr, :nc]
    dzdy = dzdy[:nr, :nc]
    actual = np.nansum(np.sqrt(1 + dzdx ** 2 + dzdy ** 2)) * dx * dy
    projected = nr * nc * dx * dy
    if projected == 0:
        return 0.0
    return float((actual - projected) / projected * 100.0)


def calc_void_material_volume(z: np.ndarray, mr_pct: float = 80.0) -> Tuple[float, float]:
    """Void volume (Vv) and material volume (Vm) from areal bearing curve.

    Computed at the specified material-ratio cut-off (default 80 %).
    Units: volume per unit area.
    """
    vals = z[np.isfinite(z)]
    if len(vals) == 0:
        return 0.0, 0.0
    sorted_z = np.sort(vals)[::-1]
    n = len(sorted_z)
    cut_idx = int(mr_pct / 100.0 * n)
    cut_idx = min(cut_idx, n - 1)
    z_cut = sorted_z[cut_idx]

    # Above cut = material peaks; below cut = voids
    above = vals[vals > z_cut] - z_cut
    below = z_cut - vals[vals <= z_cut]
    vm = float(np.sum(above) / n)   # material volume per unit area
    vv = float(np.sum(below) / n)   # void volume per unit area
    return vv, vm


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
    result = {
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
    return result


# --- Abbott–Firestone / ISO 13565 ------------------------------------------

def material_ratio_height(profile: np.ndarray, mr_pct: float) -> float:
    """Height at which the material ratio equals mr_pct %."""
    z = np.sort(profile[np.isfinite(profile)])[::-1]
    idx = int(np.clip(mr_pct / 100.0 * len(z), 0, len(z) - 1))
    return float(z[idx])


def calc_abbott_firestone(profile: np.ndarray) -> dict:
    return {
        "Mr10": material_ratio_height(profile, 10),
        "Mr50": material_ratio_height(profile, 50),
        "Mr90": material_ratio_height(profile, 90),
        "CoreDepth": (material_ratio_height(profile, 10)
                      - material_ratio_height(profile, 80)),
    }


def calc_rk_params(profile: np.ndarray) -> dict:
    """Rk, Rpk, Rvk, Mr1, Mr2 via the 40 % kernel approach (ISO 13565-2)."""
    z = np.sort(profile[np.isfinite(profile)])[::-1]
    n = len(z)
    if n < 10:
        return {"Rk": 0.0, "Rpk": 0.0, "Rvk": 0.0, "Mr1": 0.0, "Mr2": 0.0}

    mr = np.linspace(0, 100, n)
    best_start = 0
    best_slope = np.inf
    span = int(0.4 * n)
    for s in range(n - span):
        slope = abs(z[s] - z[s + span]) / 40.0
        if slope < best_slope:
            best_slope = slope
            best_start = s

    mr1 = mr[best_start]
    mr2 = mr[best_start + span]
    rk = float(z[best_start] - z[best_start + span])
    rpk = float(z[0] - z[best_start])
    rvk = float(z[best_start + span] - z[-1])
    return {"Rk": rk, "Rpk": rpk, "Rvk": rvk,
            "Mr1": float(mr1), "Mr2": float(mr2)}


def calc_g_factor(profile: np.ndarray) -> float:
    """g-factor (ISO 10844:2021): material ratio at mid-height."""
    z = profile[np.isfinite(profile)]
    if len(z) == 0:
        return 0.0
    z_mid = (np.max(z) + np.min(z)) / 2.0
    return float(np.sum(z >= z_mid) / len(z))


# --- ISO 13473-1: MPD, ETD, MSD -------------------------------------------

def calc_mpd(profile: np.ndarray, segment_mm: float = 100.0,
             dx: float = 1.0) -> Tuple[float, List[float]]:
    """Mean Profile Depth per ISO 13473-1."""
    seg_pts = max(2, int(segment_mm / dx))
    n_seg = len(profile) // seg_pts
    if n_seg == 0:
        n_seg = 1
        seg_pts = len(profile)

    msds: List[float] = []
    for i in range(n_seg):
        seg = profile[i * seg_pts:(i + 1) * seg_pts]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 2:
            continue
        half = len(seg) // 2
        pk1 = np.max(seg[:half])
        pk2 = np.max(seg[half:])
        mean_z = np.mean(seg)
        msd = (pk1 + pk2) / 2.0 - mean_z
        msds.append(float(msd))

    mpd = float(np.mean(msds)) if msds else 0.0
    return mpd, msds


def calc_etd(mpd: float) -> float:
    return 0.2 + 0.8 * mpd


# --- Spectral descriptors (ISO 13473-4 / ISO 10844) -----------------------

def calc_psd_welch(profile: np.ndarray, dx: float,
                   nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, len(profile))
    freqs, psd = sig.welch(profile, fs=1.0 / dx, nperseg=nperseg,
                           detrend="linear")
    return freqs, psd


def octave_band_rms(freqs: np.ndarray, psd: np.ndarray,
                    n_octave: int = 3) -> Dict[str, float]:
    """1/n-octave band aggregation."""
    if len(freqs) < 2:
        return {}
    df = freqs[1] - freqs[0]
    f_min = freqs[freqs > 0][0] if (freqs > 0).any() else df
    f_max = freqs[-1]
    ratio = 2.0 ** (1.0 / n_octave)
    bands: Dict[str, float] = {}
    fc = f_min
    while fc <= f_max:
        f_lo = fc / ratio
        f_hi = fc * ratio
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.any():
            rms = float(np.sqrt(np.sum(psd[mask]) * df))
            bands[f"{fc:.4f}"] = rms
        fc *= ratio
    return bands


def calc_psd_moments(freqs: np.ndarray, psd: np.ndarray) -> dict:
    """Statistical moments of the PSD."""
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    m0 = float(np.sum(psd) * df)
    m1 = float(np.sum(freqs * psd) * df)
    m2 = float(np.sum(freqs ** 2 * psd) * df)
    mean_freq = m1 / m0 if m0 > 0 else 0.0
    return {"PSD_m0": m0, "PSD_m1": m1, "PSD_m2": m2,
            "PSD_mean_freq": mean_freq}


def calc_endt(freqs: np.ndarray, psd: np.ndarray) -> float:
    """ENDT (ISO 10844:2014) – simplified spectral weighted estimate.

    This is a weighted sum over texture wavelength bands correlated
    with tyre/road noise generation.
    """
    if len(freqs) < 2:
        return 0.0
    df = freqs[1] - freqs[0]
    # Weight by frequency (higher frequencies → more noise)
    weights = freqs / (freqs[-1] + 1e-15)
    return float(10.0 * np.log10(np.sum(weights * psd * df) + 1e-15))


def calc_ltx(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    """Log texture level per wavelength band LTX (ISO 10844)."""
    if len(freqs) < 2:
        return {}
    result = {}
    for i, (f, p) in enumerate(zip(freqs, psd)):
        if f > 0:
            wl = 1.0 / f
            result[f"LTX_{wl:.2f}mm"] = float(10.0 * np.log10(p + 1e-15))
    return result


def calc_spectral(profile: np.ndarray, dx: float) -> dict:
    """Full spectral descriptor bundle."""
    freqs, psd = calc_psd_welch(profile, dx)
    bands = octave_band_rms(freqs, psd)

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total_rms = float(np.sqrt(np.sum(psd) * df))
    texture_level = 20.0 * np.log10(total_rms + 1e-12)

    psd_mom = calc_psd_moments(freqs, psd)
    endt = calc_endt(freqs, psd)

    return {
        "BandRMS": total_rms,
        "TextureLevel": float(texture_level),
        "ENDT": endt,
        "PSD_m0": psd_mom["PSD_m0"],
        "PSD_m1": psd_mom["PSD_m1"],
        "PSD_m2": psd_mom["PSD_m2"],
        "PSD_mean_freq": psd_mom["PSD_mean_freq"],
        "OctaveBands": bands,
    }


# --- Spacing / slope / correlation ----------------------------------------

def calc_mean_slope(profile: np.ndarray, dx: float) -> float:
    grad = np.abs(np.diff(profile)) / dx
    return float(np.nanmean(grad))


def calc_peak_density(profile: np.ndarray, dx: float) -> float:
    peaks, _ = sig.find_peaks(profile)
    length_m = len(profile) * dx / 1000.0
    if length_m == 0:
        return 0.0
    return len(peaks) / length_m


def calc_mean_spacing(profile: np.ndarray, dx: float) -> float:
    z = profile - np.nanmean(profile)
    crossings = np.where(np.diff(np.sign(z)))[0]
    if len(crossings) < 2:
        return 0.0
    spacings = np.diff(crossings) * dx
    return float(np.mean(spacings))


def calc_acl(profile: np.ndarray, dx: float,
             threshold: float = 0.2) -> float:
    """Autocorrelation length (distance to threshold decay, default 0.2)."""
    z = profile - np.nanmean(profile)
    n = len(z)
    acf = np.correlate(z, z, mode="full")[n - 1:]
    acf = acf / (acf[0] + 1e-15)
    below = np.where(acf < threshold)[0]
    if len(below) == 0:
        return float(n * dx)
    return float(below[0] * dx)


def calc_fractal_dimension(profile: np.ndarray) -> float:
    """Estimate fractal dimension via box-counting (1-D profile).

    For 2-D profiles, D typically ranges ~1.0-2.0.
    """
    z = profile[np.isfinite(profile)]
    if len(z) < 8:
        return 1.0

    # Normalize to [0, 1]
    z_norm = (z - np.min(z))
    rng = np.max(z_norm)
    if rng == 0:
        return 1.0
    z_norm = z_norm / rng

    sizes = []
    counts = []
    n = len(z_norm)

    for k in [2, 4, 8, 16, 32, 64]:
        if k > n // 2:
            break
        box_size = n / k
        count = 0
        for i in range(k):
            lo = int(i * box_size)
            hi = int((i + 1) * box_size)
            seg = z_norm[lo:hi]
            if len(seg) == 0:
                continue
            h_range = np.max(seg) - np.min(seg)
            count += int(np.ceil(h_range / (1.0 / k))) + 1
        if count > 0:
            sizes.append(1.0 / k)
            counts.append(count)

    if len(sizes) < 2:
        return 1.0

    log_s = np.log(sizes)
    log_c = np.log(counts)
    coeffs = np.polyfit(log_s, log_c, 1)
    return float(-coeffs[0])


# ===================================================================
# Aggregate all parameters for one profile
# ===================================================================

def compute_profile_params(profile: np.ndarray, dx: float) -> dict:
    """Compute all 1-D / 2-D descriptors for a single profile."""
    results = {}
    results.update(calc_core_stats_1d(profile))
    results.update(calc_abbott_firestone(profile))
    results.update(calc_rk_params(profile))
    results["g_factor"] = calc_g_factor(profile)

    mpd, msds = calc_mpd(profile, segment_mm=100.0, dx=dx)
    results["MPD"] = mpd
    results["ETD"] = calc_etd(mpd)
    results["MSD_mean"] = float(np.mean(msds)) if msds else 0.0

    spectral = calc_spectral(profile, dx)
    results["BandRMS"] = spectral["BandRMS"]
    results["TextureLevel"] = spectral["TextureLevel"]
    results["ENDT"] = spectral["ENDT"]
    results["PSD_m0"] = spectral["PSD_m0"]
    results["PSD_m1"] = spectral["PSD_m1"]
    results["PSD_m2"] = spectral["PSD_m2"]
    results["PSD_mean_freq"] = spectral["PSD_mean_freq"]

    results["MeanSlope"] = calc_mean_slope(profile, dx)
    results["PeakDensity"] = calc_peak_density(profile, dx)
    results["Sm"] = calc_mean_spacing(profile, dx)
    results["ACL"] = calc_acl(profile, dx)
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
            progress_cb(idx / total)

    areal = calc_core_stats_3d(z_grid, dx, dy)
    return per_profile, areal


# ===================================================================
# Aggregation helpers
# ===================================================================

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
        agg[f"{k}_P10"] = float(np.percentile(arr, 10))
        agg[f"{k}_P90"] = float(np.percentile(arr, 90))
    return agg
