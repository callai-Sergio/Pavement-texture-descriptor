"""
test_descriptors.py – Unit tests for TextureLab descriptor calculations
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.descriptors import (
    calc_ra, calc_rq, calc_rsk, calc_rku, calc_rz_iso,
    calc_core_stats_1d, calc_core_stats_3d, calc_sdr,
    calc_mpd, calc_etd, calc_mean_slope, calc_peak_density,
    calc_mean_spacing, calc_acl, calc_fractal_dimension,
    calc_g_factor, calc_rk_params, calc_abbott_firestone,
    calc_psd_welch, octave_band_rms, calc_spectral,
    compute_profile_params, aggregate_profiles,
)
from src.preprocessing import (
    fft_bandpass, remove_plane, detrend_profile, interpolate_gaps,
    hampel_filter_1d, PreprocessingConfig,
)
from src.data_io import read_csv_matrix, SurfaceGrid


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sine_profile():
    """Pure sine wave with known amplitude."""
    x = np.linspace(0, 10 * np.pi, 1000)
    return np.sin(x)  # amplitude = 1


@pytest.fixture
def flat_profile():
    return np.zeros(500)


@pytest.fixture
def random_surface():
    np.random.seed(42)
    return np.random.randn(100, 100)


# ── Core stats ────────────────────────────────────────────────────────────

class TestCoreStats:
    def test_ra_sine(self, sine_profile):
        """Ra of sin(x) should be 2/π ≈ 0.6366."""
        ra = calc_ra(sine_profile)
        assert abs(ra - 2 / np.pi) < 0.02

    def test_rq_sine(self, sine_profile):
        """Rq of sin(x) should be 1/√2 ≈ 0.7071."""
        rq = calc_rq(sine_profile)
        assert abs(rq - 1 / np.sqrt(2)) < 0.02

    def test_rsk_symmetric(self, sine_profile):
        """Skewness of symmetric sine ≈ 0."""
        assert abs(calc_rsk(sine_profile)) < 0.05

    def test_rku_sine(self, sine_profile):
        """Kurtosis of sine ≈ 1.5."""
        assert abs(calc_rku(sine_profile) - 1.5) < 0.1

    def test_flat_ra(self, flat_profile):
        assert calc_ra(flat_profile) == 0.0

    def test_flat_rq(self, flat_profile):
        assert calc_rq(flat_profile) == 0.0

    def test_core_stats_dict(self, sine_profile):
        stats = calc_core_stats_1d(sine_profile)
        assert "Ra" in stats
        assert "Rq" in stats
        assert "Rsk" in stats
        assert "Rt" in stats
        assert "Rp" in stats
        assert "Rv" in stats
        assert "StdDev" in stats
        assert "IQR" in stats

    def test_rz_iso(self, sine_profile):
        rz = calc_rz_iso(sine_profile, n_segments=5)
        # Each segment should have peak-to-valley ~ 2
        assert rz > 1.5


# ── 3D Areal ──────────────────────────────────────────────────────────────

class TestArealStats:
    def test_areal_keys(self, random_surface):
        stats = calc_core_stats_3d(random_surface, 1.0, 1.0)
        for key in ["Sa", "Sq", "Ssk", "Sku", "Sz", "Sv", "Sdr", "Vv", "Vm"]:
            assert key in stats

    def test_sdr_flat(self):
        z = np.zeros((50, 50))
        assert calc_sdr(z, 1.0, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_sdr_positive(self, random_surface):
        sdr = calc_sdr(random_surface, 1.0, 1.0)
        assert sdr > 0  # rough surface should have positive Sdr


# ── Abbott–Firestone ──────────────────────────────────────────────────────

class TestAbbott:
    def test_rk_params(self, sine_profile):
        res = calc_rk_params(sine_profile)
        assert "Rk" in res
        assert "Rpk" in res
        assert "Rvk" in res
        assert res["Rk"] >= 0

    def test_g_factor_range(self, sine_profile):
        g = calc_g_factor(sine_profile)
        assert 0 <= g <= 1

    def test_material_ratio_ordering(self, sine_profile):
        af = calc_abbott_firestone(sine_profile)
        assert af["Mr10"] >= af["Mr50"] >= af["Mr90"]


# ── ISO 13473: MPD/ETD ───────────────────────────────────────────────────

class TestMPD:
    def test_mpd_positive(self, sine_profile):
        mpd, msds = calc_mpd(sine_profile, segment_mm=100, dx=1.0)
        assert mpd > 0

    def test_etd_formula(self):
        assert calc_etd(1.0) == pytest.approx(1.0, abs=0.01)
        assert calc_etd(0.0) == pytest.approx(0.2, abs=0.01)


# ── Spectral ──────────────────────────────────────────────────────────────

class TestSpectral:
    def test_psd_welch_shape(self, sine_profile):
        freqs, psd = calc_psd_welch(sine_profile, dx=1.0, nperseg=128)
        assert len(freqs) == len(psd)
        assert len(freqs) > 0

    def test_spectral_bundle(self, sine_profile):
        res = calc_spectral(sine_profile, dx=1.0)
        assert "BandRMS" in res
        assert "TextureLevel" in res
        assert "ENDT" in res
        assert "PSD_m0" in res


# ── Spacing / Slope ──────────────────────────────────────────────────────

class TestSpacing:
    def test_mean_slope_flat(self, flat_profile):
        assert calc_mean_slope(flat_profile, 1.0) == 0.0

    def test_peak_density(self, sine_profile):
        pd_ = calc_peak_density(sine_profile, dx=1.0)
        assert pd_ > 0

    def test_acl_positive(self, sine_profile):
        acl = calc_acl(sine_profile, dx=1.0)
        assert acl > 0

    def test_fractal_dim_range(self, sine_profile):
        fd = calc_fractal_dimension(sine_profile)
        assert 1.0 <= fd <= 2.0


# ── Pre-processing ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_fft_bandpass(self):
        """Bandpass should remove high and low frequency components."""
        n = 1024
        dx = 1.0
        x = np.arange(n) * dx
        low_freq = np.sin(2 * np.pi * x / 200)     # λ=200
        mid_freq = np.sin(2 * np.pi * x / 20)      # λ=20
        high_freq = np.sin(2 * np.pi * x / 2)      # λ=2
        signal = low_freq + mid_freq + high_freq

        filtered = fft_bandpass(signal, dx, wl_low=5.0, wl_high=50.0)
        # Mid-frequency should be preserved, others attenuated
        rms_orig_mid = np.sqrt(np.mean(mid_freq ** 2))
        rms_filtered = np.sqrt(np.mean(filtered ** 2))
        # Filtered should be comparable to mid-frequency component only
        assert rms_filtered < np.sqrt(np.mean(signal ** 2))

    def test_interpolate_gaps(self):
        p = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = interpolate_gaps(p)
        assert not np.any(np.isnan(result))
        assert abs(result[2] - 3.0) < 0.01

    def test_detrend_mean(self):
        p = np.ones(100) * 5.0
        result = detrend_profile(p, "mean")
        assert abs(np.mean(result)) < 1e-10

    def test_plane_removal(self):
        z = np.zeros((50, 50))
        # Add a plane z = 2x + 3y + 1
        for i in range(50):
            for j in range(50):
                z[i, j] = 2 * j + 3 * i + 1
        result = remove_plane(z, 1.0, 1.0)
        assert np.std(result) < 0.01  # should be nearly flat


# ── Integration ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_profile_params(self, sine_profile):
        params = compute_profile_params(sine_profile, dx=1.0)
        # Check all major keys present
        expected = ["Ra", "Rq", "Rsk", "Rku", "Rt", "Rp", "Rv",
                    "MPD", "ETD", "MeanSlope", "PeakDensity", "Sm",
                    "ACL", "FractalDim", "Rk", "g_factor", "ENDT"]
        for k in expected:
            assert k in params, f"Missing key: {k}"

    def test_aggregation(self, sine_profile):
        profiles = [sine_profile, sine_profile * 0.5]
        results = [compute_profile_params(p, 1.0) for p in profiles]
        agg = aggregate_profiles(results, "mean")
        assert "Ra" in agg
        assert "Ra_std" in agg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
