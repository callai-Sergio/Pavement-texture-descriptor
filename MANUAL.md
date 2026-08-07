# TextureLab Operating Manual

Welcome to TextureLab, a complete software solution for 3D pavement surface analysis.

## Overview
TextureLab allows you to process 3D pavement scans (CSV, LAS, LAZ), filter outliers, remove macroscopic inclinations, and calculate standard 1D and 3D texture parameters (like MPD, Ra, Rq, Sdr, etc.). It also provides powerful batch comparison tools, allowing you to cluster surfaces with PCA, compute Power Spectral Density (PSD), and plot Abbott-Firestone bearing area curves.

## Basic Workflow

### 1. Starting an Analysis
- **Data Import**: Click "Browse files" or drag-and-drop your `.csv`, `.las`, or `.laz` files into the sidebar.
- **Physical Grid Config**: 
  - Ensure the physical grid sizes ($dx$, $dy$) are correct (usually 1.0mm or 0.1mm depending on your scanner).
  - Select the measurement unit (mm or µm).
- **Pre-processing Options**:
  - **Plane Removal**: Crucial for removing macroscopic slopes. Select "plane" or "polynomial" to flatten the pavement surface.
  - **Outlier Filtering**: Use Hampel or Median filters to eliminate sensor spikes and anomalies.
  - **Missing Values**: Toggle gap interpolation if your scanner produced missing `NaN` values.
  - **Band Filtering**: Use IIR or FFT to filter out specific wavelengths (e.g., isolating macrotexture from megatexture).
- Click **"Run Analysis"** to process the data.

### 2. Single File Review
Once processed, you can view the resulting data for the first file:
- **Surface View**: Toggle between a 2D Heatmap or a realistic 3D Surface map.
- **Display Parameters**: Use the filter toggles to view just 1D profile parameters, 3D surface parameters, or customize your selection.
- **Export**: Export the single-file parameter table to CSV, Excel, or JSON.

### 3. Batch Comparison
If you uploaded multiple files, scroll down to the **Batch Comparison** section:
- **Key Metrics Comparison**: View a consolidated table of all calculated parameters. The backend automatically calculates *everything* by default. Use the "Display Parameters" filter to clean up the screen.
- **Batch Export**: Click "Batch CSV" or "Export Excel" to download the full parameter matrix for all files. *Note: The export will always include all parameters regardless of your UI display filters.*
- **Charts**:
  - **PCA**: Automatically clusters your pavement samples based on their mathematical parameters.
  - **PSD (Power Spectral Density)**: Displays the frequency domain of your surface roughness. You can toggle logarithmic Y-axes and change the grouping layout.
  - **Abbott-Firestone**: Displays the material ratio curves (bearing area).
  - **3D Surfaces Gallery**: Compares realistic 1:1 true-scale 3D models of all your samples side-by-side.
- **Vector Figure Export (.zip)**: At the very bottom, click the "Generate Vector Figures" button to download high-quality, vectorized `.svg` images of all active charts for use in articles and presentations.

## Troubleshooting & Tips
- **Performance**: If processing many files, browser performance may slow down when plotting the interactive 3D graphs.
- **Units**: Pay attention to the X, Y, and Z units. If your 3D plot looks completely flat, you can increase the "Vertical exaggeration" slider for visual inspection, though it defaults to 1.0 (true physical scale).
- **Recipes**: You can save your pre-processing configuration using the "YAML" download button and reload it later to ensure consistent batch processing.
