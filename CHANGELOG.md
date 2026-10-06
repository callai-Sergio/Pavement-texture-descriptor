# TextureLab Changelog

## [3.2.0] - 2026-10-06

### Added
- **g-factor (shape factor)** per DIN ISO 10844:2024-11, clause 5.3.2 and Annex B: per 100 mm MPD segment (ISO 13473-1 processing without low-pass, zero mean), `g_pct` per segment in `cadeiaA_segmentos.csv`, `g_factor_media`/`g_factor_desvio`/`g_factor_n_segmentos` per file; example segment in `visualizacao.npz`. Unit tests with known answers.
- Viewer: "DIN ISO 10844" parameter group (g-factor + skewness), g-factor histogram and an Annex B Figure B.3-style plot in the Segments tab, g-factor in the metric cards and default comparisons.

- **Wavelet texture analysis** (descriptive, Daubechies db4): profile octave spectrum (`ondaletas_perfil.csv`, micro/macro shares) and 2D directional energies and anisotropy on SL5 (`ondaletas_2d.csv`). Pipeline dependency: PyWavelets.
- White paper PDFs in `docs/pdf/` and the generator `tools/md2pdf.py` (Markdown + KaTeX → headless Chrome).

### Fixed
- `.tlproj` name for output folders with a dot (`Resultados_v3.2` was packed as `Resultados_v3.tlproj`, overwriting it).

### Changed
- Version 3.2.0 in pyproject, pipeline, viewer and desktop wrapper.

### Removed
- ETD (it is only 1.1·MPD) from the pipeline output, configuration and viewer.

### Viewer (since 3.1.0)
- English/German/Portuguese with language switch; Abbott grid with Smr1/Smr2 lines and values; true-scale 3D with vertical exaggeration; pavement grouping (section + surface course), ISO tracks kept separate; PCA hidden (SHOW_PCA).

## [3.1.0] - 2026-10-06

### Added
- **Project format** (`docs/FORMATO_PROJETO.md`): the batch output folder (or a `.tlproj` zip) with `projeto.json` index, per-file `resumo.json`, CSVs, `previa.npz` and the new `visualizacao.npz` (Abbott curves, height histograms, chain-A profiles, native-resolution line, filtered SL5 preview). JSON/CSV/NPZ only, no pickle.
- **Recipe** per result (core version + configuration + LAZ identity, hashed). `--skip-done` now recomputes only files whose recipe changed.
- Pipeline options `--config` (JSON overrides, unknown keys rejected), `--zip` (pack `.tlproj`) and `--index-only` (index existing results, e.g. `Resultados_v2`, without recomputing).
- `TextureLab/components/project_reader.py`: read-only project access (folder or zip, `allow_pickle=False`).
- `pipeline/tests/test_pipeline.py`: synthetic LAZ end to end, recipe, folder/zip equivalence, legacy results, no unpickling, optional real-file regression.
- `TextureLab/requirements-viewer.txt` (viewer only, no laspy/numba).

### Changed
- **Single core**: `pipeline/texturelab_batch.py` merges the v3.0.0 calculations, the memory/CPU optimizations of the server's v2 script (chunked LAZ reading, shared strips, in-place areal filters, block statistics, one material-ratio curve per surface, one process per file) and Hurst. On a real file all 92 parameters of the v2 results match exactly.
- **`TextureLab/app.py` is now a viewer** of projects (summary, per-file surface/profiles/spectrum/Abbott/PSD/segments, group comparison, correlation and PCA). It no longer imports the v2.4.0 calculations. The desktop wrapper opens this viewer.
- Worker count from *available* RAM and file size; `streamlit>=1.45`.

### Moved
- The previous computing app is kept unchanged as `TextureLab/app_legacy.py` (v2.4.0 calculations, see `docs/DIAGNOSTICO.md`).

## [3.0.0] - 2026-10-05

### Added
- **Headless batch pipeline** (`pipeline/texturelab_batch.py`, `pipeline/requirements.txt`): single, standards-checked core for 3dT LAZ scans, parallel over files, no interface. Reference for reported values.
  - Chain A per ISO 13473-1:2019: 0.5 mm strips and resampling, Annex E spikes, Table D.2 low-pass (designed at 2.40 mm, forward-backward), slope suppression per 100 mm segment, MSD/MPD, ETD = 1.1·MPD.
  - Texture spectrum per ISO 13473-4:2024 method 1 (one-third-octave filters, re 1 µm, l ≥ 12·λmax, Annex F mirroring).
  - Areal parameters per ISO 25178-2/-3 on SF, SL5 and provisional MICRO surfaces (Gaussian ISO 16610-61): heights, Sdq, Sdr, Sk family, volumes, Sal, Str.
  - Rk/Sk family per ISO 13565-2 (least-squares line in the 40 % window, equal-area triangles).
  - Hurst exponent and fractal dimension (descriptive), micro (0.05–0.5 mm) and macro (0.5–20 mm) fitted separately.
- `docs/WHITE_PAPER_PT.md`, `docs/WHITE_PAPER_EN.md`, `docs/DIAGNOSTICO.md`.

### Changed
- About page: describes v3.0.0 and the pipeline; Acknowledgements section removed.
- Version bumped to 3.0.0 (app, desktop, pyproject, exports).

### Removed
- Tracked `__pycache__`, `*.egg-info`, `pytest_out.txt`, `test_output.txt`, `scratch_test_tk.py`, the broken self-referencing submodule entry and `tests/test_llm_utils.py` (imported a removed package). Added `.gitignore`.

### Known issues
- The Streamlit app still uses the v2.4.0 calculations (see `docs/DIAGNOSTICO.md`), including `pickle` for `.tlp` projects.

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
