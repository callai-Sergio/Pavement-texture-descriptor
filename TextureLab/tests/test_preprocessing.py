import pytest
import numpy as np

from TextureLab.preprocessing import (
    fit_plane,
    remove_plane,
    interpolate_gaps_1d,
    interpolate_grid_gaps,
    hampel_filter_1d,
    hampel_filter_grid,
    downsample_grid,
    iso_mpd_filter,
    spectral_prep,
    preprocess_surface,
)

def test_fit_and_remove_plane():
    # Create a 5x5 grid with a known plane
    y, x = np.mgrid[0:5, 0:5]
    z = 2.0 * x + 3.0 * y + 1.0
    
    # Add a small flat deviation to a point
    z_with_noise = z.copy()
    z_with_noise[2, 2] += 5.0
    
    z_fit, coeffs = fit_plane(z)
    np.testing.assert_allclose(coeffs, (2.0, 3.0, 1.0), atol=1e-7)
    np.testing.assert_allclose(z_fit, z, atol=1e-7)
    
    # remove_plane
    z_leveled, log = remove_plane(z)
    np.testing.assert_allclose(z_leveled, np.zeros((5, 5)), atol=1e-7)
    assert np.allclose(log["plane_coefficients"][0], (2.0, 3.0, 1.0))

def test_interpolate_gaps_1d():
    profile = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0, np.nan, np.nan, np.nan, 10.0])
    
    # Max gap 2: the gap of 3 (indices 6, 7, 8) won't be filled.
    # Gap 1 (indices 1, 2) is between 1.0 and 4.0, should be 2.0, 3.0
    # Gap 2 (index 4) is between 4.0 and 6.0, should be 5.0
    interp_prof, n_filled = interpolate_gaps_1d(profile, max_gap=2)
    
    assert n_filled == 3
    assert np.allclose(interp_prof[:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.isnan(interp_prof[6])
    assert np.isnan(interp_prof[7])
    assert np.isnan(interp_prof[8])

def test_interpolate_grid_gaps():
    z = np.array([
        [1.0, np.nan, 3.0],
        [4.0, 5.0, np.nan]
    ])
    # by row
    z_interp, log = interpolate_grid_gaps(z.copy(), direction="rows", max_gap=5)
    assert list(log["interpolated_points"]) == [1, 0]
    np.testing.assert_allclose(z_interp[0, :], [1.0, 2.0, 3.0])
    # The last element in row 1 is NaN, but it's at the end, so interpolate_gaps_1d doesn't extrapolate
    assert np.isnan(z_interp[1, 2])

def test_hampel_filter_1d():
    # 0 1 2 300 4 5  (300 is an outlier)
    profile = np.array([0.0, 1.0, 2.0, 30.0, 4.0, 5.0])
    filtered, log = hampel_filter_1d(profile, window_size=2, n_sigma=2.0, replace="median")
    assert log["outliers"] == 1
    assert log["replacements"] == 1
    # Median of [1, 2, 30, 4, 5] is 4
    # Wait, the window for index 3 is [1, 2, 30, 4, 5]. Median is 4. -> Replaced with 4.
    assert filtered[3] == 4.0

def test_downsample_grid():
    z = np.ones((4, 4))
    z[0:2, 0:2] = 2.0
    z[0:2, 2:4] = 4.0
    # Average of 2x2 blocks:
    # 2.0 4.0
    # 1.0 1.0
    z_macro, log = downsample_grid(z, target_dx=2.0, dx=1.0)
    assert log["downsample_factor"] == 2
    assert z_macro.shape == (2, 2)
    np.testing.assert_allclose(z_macro, [[2.0, 4.0], [1.0, 1.0]])

def test_spectral_prep():
    # Just check return types and shapes for a simple signal
    x = np.linspace(0, 10, 256)
    y = np.sin(2 * np.pi * 5 * x)
    freqs, psd, log = spectral_prep(y, dx=10/256, nperseg=64)
    assert len(freqs) == len(psd)
    assert len(freqs) == 33  # nperseg/2 + 1

def test_preprocess_surface():
    z = np.random.rand(20, 20)
    z[5, 5] = np.nan
    full_res, macro_res = preprocess_surface(
        z, 
        dx=0.1, 
        target_dx_macro=0.5, 
        hampel_window_mm=0.5, 
        iso_filter=False
    )
    assert "z_clean" in full_res
    assert "z_macro" in macro_res
    assert full_res["z_clean"].shape == (20, 20)
    assert macro_res["z_macro"].shape == (4, 4) # 20 / 5 = 4
