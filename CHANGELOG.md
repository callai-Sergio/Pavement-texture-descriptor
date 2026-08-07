# TextureLab Changelog

## [2.4.0] - 2026-08-07

### Added
- **Vectorized Figure Exports**: Added a ZIP export button to download all active plots (PCA, PSD, Abbott, 3D Surfaces) as high-quality `.svg` vector files using `kaleido`.
- **Raw Data Exports**: Added CSV and Excel export buttons specifically for PSD curves and PCA coordinates.
- **Two-Step Parameter Filtering**: Added a display filter ("1D Parameters", "3D Parameters", "All", "Custom") for both single file analysis and batch comparisons. The backend now calculates all parameters by default.
- **Always Export All Data**: Batch CSV/Excel exports now always contain all computed parameters, bypassing UI display filters.

### Changed
- **Realistic 3D Surfaces**: The 3D Surface Gallery now enforces a realistic 1:1:1 scale (`aspectmode='data'`). The Z-axis is explicitly labeled with physical units (mm), and the default vertical exaggeration is set to 1.0 (true scale).
- **Abbott Curve Enhancements**: Downsampled Abbott curve generation to 250 points to improve web browser rendering performance. Y-axis now explicitly states `Height (mm)`. Added visual grouping options (Combined, Separate, Group by Prefix).
- **PSD Plot Enhancements**: X-axis explicitly labeled as `Wavelength λ (mm)`. Added a Linear/Logarithmic toggle for the Y-axis. Standardized titles and grouping layouts.

## [2.3.0] - 2026-08-05

### Changed
- **High-Resolution Performance Optimization**: Added `numba` dependency to JIT-compile core mathematically intensive routines (`Sdr`, `Vv`, `Vm`). This reduces memory consumption to $O(1)$ during local gradient calculation and speeds up processing time by >100x for dense point clouds (e.g., 0.01mm resolution).
- **Abbott-Firestone Histogram Optimization**: Changed volume and material calculation logic (Vv/Vm) from $O(N \log N)$ sorting to $O(N)$ histogram distribution.


## [1.3.0] / [2.1.0 Desktop] - 2026-07-08

### Added
- **Save/Load Project Workspace**: Export and import complete analysis sessions (`.tlp` format) using `pickle` serialization. This ensures data is not lost between sessions or when navigating between tabs.
- **PDF Report Generation**: Export tabulated results to PDF using `fpdf2`. This is available in both the Web and Desktop versions.

### Removed
- **Ollama AI Assistant**: Removed `texturelab_llm` package, dependencies (`requests`), and all AI Assistant interface elements from both Web and Desktop applications to reduce bloat and improve system stability.

### Fixed
- Fixed state persistence issues in the Streamlit Web app by allowing users to save their session state directly to their local machine.
