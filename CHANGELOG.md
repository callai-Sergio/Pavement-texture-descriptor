# TextureLab Changelog

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
