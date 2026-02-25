# Pavement Texture Evaluator

A project to evaluate and analyze pavement textures via the **TextureLab** Streamlit application.

## 🔬 Features & Analysis Pipeline

**TextureLab v1.2.0** provides a comprehensive pipeline for evaluating 3D pavement scans (LAZ/LAS/CSV/TXT) by extracting ISO-standard descriptors across multiple profiles:

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
5. **Reproducible Batch Processing**: 
   - Save and load execution "recipes" (YAML).
   - Results: CSV, Excel, JSON with preprocessing logs.
6. **File Support**:
   - CSV, TXT (tab/comma delimited), LAZ, and LAS formats.
   - Upload limit: **3.5 GB** per file.

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
