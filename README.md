# Pavement Texture Evaluator

A project to evaluate and analyze pavement textures via the **TextureLab** Streamlit application.

## 🆕 v3.4 — Compute once on the server, view on the PC

There is **one** calculation core, `pipeline/texturelab_batch.py` (core **3.4.1**). It runs in batch on the
server and writes a **project** (results folder or `.tlproj` zip: JSON/CSV/NPZ, no pickle). The Streamlit app
`TextureLab/app.py` (viewer **3.4.2**) opens the project and only renders tables, profiles, spectra, Abbott
curves, PSD/Hurst, 3D surfaces and comparisons. Nothing is recalculated in the app.

What the core computes, by version:
- **3.0 / 3.1** — standards-checked core: MPD/MSD (ISO 13473-1), third-octave spectrum (ISO 13473-4), areal
  S/V parameters on SF, SL5 and MICRO (ISO 25178, ISO 16610-61), Rk family (ISO 13565-2), 1D Hurst (descriptive).
- **3.2** — g-factor per DIN ISO 10844:2024-11 Annex B; descriptive wavelet analysis (db4); ETD dropped (it is 1.1·MPD).
- **3.3** — radial 2D PSD and 2D Hurst on SF; height percentiles P50/P70/P90/P95 and P90/P70; 3D g-factor;
  Excel export of medians per surface (`tools/exportar_medianas.py`).
- **3.4** — filter verification against the standards (`tools/verificar_filtros.py`,
  [report](docs/verificacao_filtros/VERIFICACAO_FILTROS.md)); the short-lived ML5 chain was removed in 3.4.1.

Full history in [CHANGELOG.md](CHANGELOG.md).

```bash
# server: own venv + batch with CPU/RAM/disk limits (pause, resume, stop, status)
pipeline/servidor.sh instalar
pipeline/servidor.sh iniciar /path/LAZ /path/Resultados_v3.4.1
pipeline/servidor.sh status
# PC: own venv + viewer
python -m venv .venv && .venv\Scripts\activate
pip install -r TextureLab/requirements-viewer.txt
streamlit run TextureLab/app.py        # then open Resultados_v3.4.1.tlproj or the folder
```

- **View on the PC without the server:** `tools\baixar_resultados.bat` copies finished projects to
  `%USERPROFILE%\TextureLab\projetos` (local disk), `tools\abrir_app.bat` opens the app locally — see [docs/PC_LOCAL.md](docs/PC_LOCAL.md)
- Batch usage, shared-server memory lock and options: [pipeline/README.md](pipeline/README.md)
- Project format and recipe (code version + config + LAZ id): [docs/FORMATO_PROJETO.md](docs/FORMATO_PROJETO.md)
- White paper: [Português](docs/WHITE_PAPER_PT.md) · [English](docs/WHITE_PAPER_EN.md) ·
  PDF: [PT](docs/pdf/TextureLab_White_Paper_PT.pdf) · [EN](docs/pdf/TextureLab_White_Paper_EN.pdf)
  (regenerate with `python tools/md2pdf.py`)
- Filter verification (ISO 16610-61, ISO 13473-1, ISO 13473-4 / IEC 61260-1): [docs/verificacao_filtros/VERIFICACAO_FILTROS.md](docs/verificacao_filtros/VERIFICACAO_FILTROS.md)
- Excel of medians per surface: `python tools/exportar_medianas.py Resultados_v3.4.1` (needs `openpyxl`)
- Diagnostics and corrections: [docs/DIAGNOSTICO.md](docs/DIAGNOSTICO.md)

> The previous app, which computed with the v2.4.0 calculations (see the diagnostics), is kept as
> `TextureLab/app_legacy.py` for reference only. **Everything below this line describes that legacy app**
> (v2.x), not the current core and viewer: for example it still lists ETD, the Hampel filter and PCA, which
> the current core does not use (PCA is hidden in the viewer).

## 🔬 Legacy app (v2.x): features & analysis pipeline

**TextureLab v2.x (legacy)** provides a comprehensive pipeline for evaluating 3D pavement scans (LAZ/LAS/CSV/TXT) by extracting ISO-standard descriptors across multiple profiles:

1. **Preprocessing Pipeline**:
   - **Plane Removal:** Planar detrending or polynomial surface removal (enabled by default).
   - **Gap Interpolation:** Linear interpolation for invalid / NaN points.
   - **Outlier Filtering:** Hampel filter with mm-based window, applied in **both row and column** directions (two-pass cleaning). Enabled by default with K=3.5.
   - **Block-Average Downsampling:** Anti-aliased resampling to target resolution (never stride-based).
   - **Bandpass Filtering:** ISO-13473-1 compliant FFT/Butterworth filters (e.g., for MPD).
2. **Descriptor Extraction**: 
   - Core statistics: MPD, ETD, Ra, Rq, Rsk, Rku, Sa, Sdr, g-factor, Fractal Dimension.
3. **Physically Realistic 3D Rendering**:
   - **Vertical exaggeration** slider (default 0.3) – affects rendering only, not metrics.
   - **Robust colour scale** (P1–P99 percentile clamping) so outliers don't wash out the colormap.
   - Block-averaged downsampling in the renderer for anti-aliased visualization.
4. **Data Science Analytics**: 
   - PCA (**per-surface**, **per-profile**, or **per-sample** scope), with **sample selector** to pick which files to include.
   - K-Means / GMM / Ward Clustering, Regression, Isolation Forest anomaly detection, Feature Selection.
5. **Reproducible Batch Processing & Workspace**: 
   - Save and load execution "recipes" (YAML).
   - **Save / Load Project Workspace**: Export your entire analysis session as a `.tlp` file to resume later without re-processing.
   - Results: PDF Reports, CSV, Excel, JSON with preprocessing logs.
6. **File Support**:
   - CSV, TXT (tab/comma/space delimited), LAZ, and LAS formats. Automatically ignores text headers.
   - Upload limit: **3.6 GB** per file (increased websocket message limit).

## 🚀 How to share and deploy this app for others to test (Streamlit Community Cloud)

The easiest and completely **free** way to share this application publicly is using [Streamlit Community Cloud](https://share.streamlit.io). It connects directly to your GitHub repository and automatically updates whenever you push new changes.

### Step-by-Step Deployment Guide

1. **Commit and Push to GitHub**
   Ensure all your latest code (including the `TextureLab/requirements.txt` file) is pushed to your GitHub repository.

2. **Sign up for Streamlit Community Cloud**
   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in using your GitHub account.

3. **Deploy the App**
   - Click the **"New app"** button.
   - Authorize Streamlit to access your GitHub repositories if prompted.
   - Fill in the deployment form:
     - **Repository:** Select this repository (e.g., `callai-Sergio/Pavement-texture-descriptor`)
     - **Branch:** `master` (or `main`)
     - **Main file path:** Type `TextureLab/app.py`
   - Click **"Deploy!"**

4. **Wait for Build**
   Streamlit will now provision a server, install the dependencies listed in `TextureLab/requirements.txt`, and launch the app. This usually takes 1-3 minutes. 

5. **Share the Link!**
   Once deployed, you will get a public URL (e.g., `https://pavement-texture.streamlit.app`) that you can send to anyone to test the application directly in their browser without installing Python.

*(Note: The Community Cloud has a 1GB memory limit. For extremely large LAZ files > 200MB, users might experience slow downs, but it works perfectly for standard pavement scans).*


