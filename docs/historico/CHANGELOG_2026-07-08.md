# TextureLab Improvements Changelog (2026-07-08)

## 🚀 Web Application (`TextureLab/app.py` & `src/`)
- **Abbott-Firestone Enhancements**:
  - Added new `Layout Mode` options (Combined, Separate Subplots, Group by Prefix).
  - Automatically display a mini parameter table (`Rk`, `Rpk`, `Rvk`, `Mr1`, `Mr2`) alongside the curve in the Separate Subplots view.
  - Added ability to average curves automatically based on file name prefix.
- **PSD (Power Spectral Density) Smoothing**:
  - Added a toggle (default enabled) to display the PSD curve as standardized 1/3 Octave Bands RMS (dB) instead of a noisy narrow-band plot.
- **Export Refinements**:
  - Removed unstable PDF export options.
  - Ensured Excel and CSV batch exports include all 3D parameters properly.
- **Calculation Bug Fix**:
  - Fixed an issue in `aggregate_profiles` (`descriptors.py`) where parameters returning `{}` on edge profiles incorrectly removed those parameters from the batch export tables. Parameter extraction now gathers all unique keys across all profiles correctly.

## 🖥️ Desktop Application (`TextureLabDesktop/main.py`)
- **PyWebView Transformation**:
  - Replaced the entire PyQt6 UI with a `pywebview` wrapper. 
  - The Desktop App now automatically spawns the Streamlit server in the background and embeds the Web UI inside a native desktop window, guaranteeing 100% visual and functional parity between the Web and Desktop versions.
