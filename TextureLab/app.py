"""
TextureLab – Pavement Surface Analysis Application
Main Streamlit entry point.

Copyright (c) 2025 Sergio Callai
Licensed under CC BY-NC 4.0
"""
import os
import sys
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Add project root to path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_io import load_surface, SurfaceGrid
from src.preprocessing import PreprocessingConfig, preprocess_surface
from src.descriptors import (
    compute_all, aggregate_profiles, PARAM_REGISTRY, compute_profile_params,
    calc_psd_welch
)
from src.analytics import (
    prepare_feature_matrix, run_pca, run_kmeans, run_gmm, run_ward,
    elbow_analysis, run_regression, run_isolation_forest,
    correlation_pruning, recursive_feature_elimination,
    bootstrap_ci, save_model, load_model,
)
from components.visualizer import (
    histogram, boxplot, heatmap_2d, surface_3d, scatter_2d, scatter_3d,
    pair_plot, correlation_heatmap, pca_variance_plot, pca_biplot,
    elbow_plot, residual_plot, actual_vs_predicted, feature_importance_plot,
    texture_spectrum_plot,
)
from components.export_manager import (
    build_results_table, build_batch_table,
    export_csv, export_excel, export_json_report, export_pdf_report
)
from components.project_manager import export_project, load_project

# ===================================================================
# Constants
# ===================================================================
APP_VERSION = "2.4.0"
APP_AUTHOR = "Sergio Callai"
APP_YEAR = "2025"


# ===================================================================
# Page configuration
# ===================================================================
st.set_page_config(
    page_title="TextureLab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1rem;
    }

    /* Hero header */
    .hero-title {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #f3f4f6;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: -0.2rem;
    }
    .version-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* Metric cards – compact */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 8px;
        padding: 0.35rem 0.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetric"] label {
        color: #9ca3af !important;
        font-weight: 500;
        font-size: 0.7rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0ff !important;
        font-weight: 600;
        font-size: 1rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #667eea;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.8rem;
    }

    /* Nav buttons */
    .nav-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        cursor: pointer;
        text-align: center;
        transition: all 0.3s ease;
        width: 100%;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }

    /* DataFrames */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Info boxes */
    .stAlert { border-radius: 8px; }

    /* License footer */
    .license-footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.7rem;
        padding: 0.5rem 0;
        border-top: 1px solid rgba(102, 126, 234, 0.1);
        margin-top: 1rem;
    }
    .license-footer a { color: #667eea; }

    /* Welcome cards */
    .welcome-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .welcome-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    .welcome-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .welcome-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #e0e0ff;
    }
    .welcome-desc {
        color: #d1d5db;
        font-size: 0.9rem;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* File uploader list – show 5 items per page */
    div[data-testid="stFileUploader"] ul {
        max-height: 500px !important;
        overflow-y: auto;
    }
    div[data-testid="stFileUploader"] ul li {
        padding: 0.15rem 0 !important;
        font-size: 0.8rem;
    }

    /* Compact expander headers so long filenames don't clutter */
    div[data-testid="stExpander"] details summary {
        font-size: 0.8rem !important;
        overflow: hidden;
    }
    div[data-testid="stExpander"] details summary span {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    div[data-testid="stExpander"] details summary svg {
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# Session state defaults
# ===================================================================
def _init_state():
    defaults = {
        "surfaces": [],
        "file_names": [],
        "profiles": {},
        "results_1d": {},
        "results_areal": {},
        "aggregated": {},
        "batch_agg": [],
        "warnings": [],
        "processed": False,
        "logs": [],
        "page": "home",
        "selected_params": [],
        "uploader_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _log(msg: str):
    st.session_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _set_page(page: str):
    st.session_state["page"] = page


# ===================================================================
# Sidebar
# ===================================================================
with st.sidebar:
    st.markdown('<p class="hero-title">🔬 TextureLab</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Pavement Surface Analysis</p>',
                unsafe_allow_html=True)
    st.markdown(f'<span class="version-badge">v{APP_VERSION}</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────
    st.markdown("### 🧭 Navigation")
    if st.button("🏠  Home", use_container_width=True, key="nav_home"):
        _set_page("home")
    if st.button("📄  New Analysis", use_container_width=True, key="nav_new"):
        _set_page("new")
    if st.button("📂  Open File", use_container_width=True, key="nav_open"):
        _set_page("open")
    if st.button("📦  Batch Analysis", use_container_width=True, key="nav_batch"):
        _set_page("batch")
    if st.button("🔍  Compare Results", use_container_width=True, key="nav_compare"):
        _set_page("compare")

    st.markdown("---")
    st.markdown("### 📚 Information")
    if st.button("❓  Help", use_container_width=True, key="nav_help"):
        _set_page("help")
    if st.button("ℹ️  About", use_container_width=True, key="nav_about"):
        _set_page("about")
    if st.button("📜  License", use_container_width=True, key="nav_license"):
        _set_page("license")

    st.markdown("---")
    
    st.markdown("### 💾 Workspace")
    uploaded_project = st.file_uploader("Load Project (.tlp)", type=["tlp"], key="load_proj_uploader", label_visibility="collapsed")
    if uploaded_project is not None and st.button("Load Workspace", use_container_width=True):
        try:
            load_project(uploaded_project.getvalue(), st.session_state)
            st.success("Project loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load: {e}")

    if st.session_state.get("processed", False):
        st.markdown("### 🧹 Data Management")
        if st.button("💾 Prepare Project for Download", use_container_width=True):
            with st.spinner("Preparing workspace data for export..."):
                st.session_state["project_export_bytes"] = export_project(dict(st.session_state))
                
        if "project_export_bytes" in st.session_state:
            st.download_button(
                label="📥 Download .tlp Project",
                data=st.session_state["project_export_bytes"],
                file_name="texturelab_workspace.tlp",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        if st.button("🗑️ Clear all loaded data", use_container_width=True, type="secondary"):
            current_key = st.session_state.get("uploader_key", 0)
            st.session_state.clear()
            _init_state()
            st.session_state["uploader_key"] = current_key + 1
            _set_page("home")
            st.rerun()
            
        st.markdown("---")

    # License footer
    st.markdown(f"""
    <div class="license-footer">
        © {APP_YEAR} {APP_AUTHOR}<br>
        <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a>
        · No commercial use
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# Pre-processing settings panel (used by New / Open / Batch pages)
# ===================================================================
def render_settings_panel():
    """Render the processing settings in sidebar and return config."""
    with st.sidebar:
        st.markdown("### ⚙️ Grid Settings")
        col1, col2 = st.columns(2)
        with col1:
            dx = st.number_input("dx (mm)", value=1.0, min_value=0.001,
                                 step=0.1, format="%.3f", key="s_dx")
        with col2:
            dy = st.number_input("dy (mm)", value=1.0, min_value=0.001,
                                 step=0.1, format="%.3f", key="s_dy")

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            units_xy = st.selectbox("XY units", ["mm", "m", "µm"], key="s_uxy")
        with col_u2:
            units_z = st.selectbox("Z units", ["mm", "µm", "m"], key="s_uz")

        direction = st.selectbox("Traffic direction",
                                 ["longitudinal", "transverse"], key="s_dir")
        every_n = st.number_input("Extract every N-th profile", value=1,
                                  min_value=1, step=1, key="s_every")

        st.markdown("### 🔧 Pre-processing")
        with st.expander("Plane / surface removal"):
            plane_mode = st.selectbox("Mode", ["plane", "none", "polynomial"],
                                      key="s_plane")
            poly_order = (st.slider("Polynomial order", 1, 5, 2, key="s_poly")
                          if plane_mode == "polynomial" else 2)

        with st.expander("Outlier filtering"):
            outlier_method = st.selectbox("Method", ["hampel", "median", "none"],
                                          key="s_out")
            outlier_window = st.slider("Window size", 3, 21, 7, step=2,
                                       key="s_outw")
            outlier_thresh = st.slider("Threshold (σ)", 1.0, 6.0, 3.0, 0.5,
                                       key="s_outt")

        with st.expander("Missing values"):
            interp = st.checkbox("Interpolate gaps", value=True, key="s_interp")
            max_miss = st.slider("Max missing fraction", 0.0, 1.0, 0.3, 0.05,
                                 key="s_miss")

        with st.expander("Detrending"):
            detrend = st.selectbox("Per-profile detrend",
                                   ["none", "mean", "linear"], key="s_detrend")

        with st.expander("Band filtering"):
            do_bp = st.checkbox("Enable bandpass", value=False, key="s_bp")
            bp_low = st.number_input("Low λ cut (mm)", value=0.5, step=0.1,
                                     key="s_bpl")
            bp_high = st.number_input("High λ cut (mm)", value=50.0, step=1.0,
                                      key="s_bph")
            bp_method = st.selectbox("Filter method", ["fft", "iir"],
                                     key="s_bpm")

        st.markdown("### 📊 Aggregation Mode")
        agg_mode = st.selectbox("Aggregation Mode", ["mean", "median", "trimmed_mean"],
                                key="s_agg", help="How to aggregate multiple profiles for 2D parameters.")

        st.markdown("### 🎨 Rendering")
        vert_exag = st.slider(
            "Vertical exaggeration", 0.1, 5.0, 1.0, 0.1,
            key="s_vexag",
            help="Scale Z for visualization only. Lower = flatter (less spiky). Does NOT affect metrics.")
        robust_color = st.checkbox(
            "Robust colour scale (P1–P99)", value=True,
            key="s_robcol",
            help="Clamp colours to 1st–99th percentile so outlier pits/spikes don't wash out the colormap.")

        # Recipe save/load
        st.markdown("### 💾 Save/Load Recipe (Batch settings)")
        st.caption("Store your pre-processing and grid settings for reproducible batch analysis later.")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            recipe = {
                "dx": dx, "dy": dy, "units_xy": units_xy, "units_z": units_z,
                "direction": direction, "every_n": every_n,
                "plane_removal": plane_mode, "poly_order": poly_order,
                "outlier_method": outlier_method, "outlier_window": outlier_window,
                "outlier_threshold": outlier_thresh,
                "interp_missing": interp, "max_missing_fraction": max_miss,
                "detrend_mode": detrend,
                "bandpass": do_bp, "bandpass_low": bp_low,
                "bandpass_high": bp_high, "bandpass_method": bp_method,
                "aggregation_mode": agg_mode,
            }
            st.download_button("⬇ YAML", yaml.dump(recipe),
                               "texturelab_recipe.yaml", "text/yaml",
                               use_container_width=True)
        with col_s2:
            recipe_file = st.file_uploader("Load", type=["yaml", "yml"],
                                           label_visibility="collapsed",
                                           key="s_recipe")
            if recipe_file:
                st.info("Recipe loaded.")

    cfg = PreprocessingConfig(
        plane_removal=plane_mode, poly_order=poly_order,
        outlier_method=outlier_method, outlier_window=outlier_window,
        outlier_threshold=outlier_thresh,
        interp_missing=interp, max_missing_fraction=max_miss,
        detrend_mode=detrend,
        bandpass=do_bp, bandpass_low=bp_low,
        bandpass_high=bp_high, bandpass_method=bp_method,
    )
    return dx, dy, units_xy, units_z, direction, every_n, agg_mode, st.session_state["selected_params"], cfg, vert_exag, robust_color


# ===================================================================
# File processing function
# ===================================================================
def process_files(uploaded_files, dx, dy, units_xy, units_z,
                  direction, every_n, agg_mode, selected_params, cfg):
    """Process uploaded file(s) and store results in session state."""
    all_warnings: list = []
    batch_agg: list = []
    surfaces: list = []
    file_names: list = []

    progress = st.progress(0, text="Processing files…")

    for fi, f in enumerate(uploaded_files):
        _log(f"📄 Loading {f.name}…")
        progress.progress(fi / len(uploaded_files),
                          text=f"Processing {f.name} ({fi+1}/{len(uploaded_files)})")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix)
        tmp.write(f.read())
        tmp.close()

        try:
            grid = load_surface(tmp.name, dx, dy,
                                units_xy=units_xy, units_z=units_z)
            _log(f"  Grid size: {grid.ny}×{grid.nx}")
            
            z_proc, profiles, warns = preprocess_surface(
                grid.z, dx, dy, cfg, direction, every_n)
            
            # Store the FILTERED surface for visualization, not the raw one
            grid.z = z_proc.copy()
            surfaces.append(grid)
            file_names.append(f.name)

            all_warnings.extend(warns)
            for w in warns:
                _log(f"  ⚠ {w}")
            _log(f"  ✅ {len(profiles)} profiles extracted")

            per_profile, areal = compute_all(profiles, z_proc, dx, dy)
            agg = aggregate_profiles(per_profile, agg_mode)
            
            # Filter results by user selection
            if selected_params:
                # Need to keep std/P10/P90 derived metrics if the base metric was selected
                filtered_agg = {}
                for k in list(agg.keys()):
                    base_k = k.replace("_std", "").replace("_P10", "").replace("_P90", "")
                    if base_k in selected_params:
                        filtered_agg[k] = agg[k]
                agg = filtered_agg
                
                areal = {k: v for k, v in areal.items() if k in selected_params}
                
                # Filter per-profile data for data science/viz tabs
                filtered_profiles = []
                for res in per_profile:
                    filtered_profiles.append({k: v for k, v in res.items() if k in selected_params})
                per_profile = filtered_profiles

            st.session_state["profiles"][f.name] = profiles
            st.session_state["results_1d"][f.name] = per_profile
            st.session_state["results_areal"][f.name] = areal
            st.session_state["aggregated"][f.name] = agg
            batch_agg.append(agg)

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            _log(f"  ❌ Error: {e}")
            for line in tb_str.strip().split('\n'):
                _log(f"      {line}")
            st.error(f"Error processing {f.name}: {e}")
        finally:
            os.unlink(tmp.name)

    st.session_state["surfaces"] = surfaces
    st.session_state["file_names"] = file_names
    st.session_state["batch_agg"] = batch_agg
    st.session_state["warnings"] = all_warnings
    st.session_state["processed"] = True
    progress.progress(1.0, text="Done ✅")
    
    total = len(uploaded_files)
    success = len(file_names)
    failed = total - success
    success_rate = (success / total) * 100 if total > 0 else 0
    
    _log("🎉 Analysis complete.")
    _log(f"📊 Summary: {success} processed, {failed} failed ({success_rate:.1f}% success rate)")
    
    if failed > 0:
        st.warning(f"Processed {success}/{total} file(s) successfully ({success_rate:.1f}%). Check logs for {failed} failure(s).")
    else:
        st.success(f"Processed {success}/{total} file(s) successfully (100%)!")


# ===================================================================
# Results display functions
# ===================================================================
def render_summary():
    """Display analysis summary: compact metrics + descriptions."""

    def fmt(val, unit=""):
        if val is None:
            return "–"
        av = abs(val)
        if av == 0:
            s = "0"
        elif av >= 100:
            s = f"{val:.1f}"
        elif av >= 1:
            s = f"{val:.3f}"
        elif av >= 0.01:
            s = f"{val:.4f}"
        else:
            s = f"{val:.2e}"
        if unit and unit != "–":
            s += f" {unit}"
        return s

    # Descriptions for each key metric
    DESCR = {
        "MPD": "Mean Profile Depth (ISO 13473-1). Computed over 100 mm baseline segments: each split in two halves, peak height averaged, minus segment mean.",
        "ETD": "Estimated Texture Depth = 0.2 + 0.8·MPD. Approximates sand-patch depth.",
        "Ra": "Arithmetic mean roughness (ISO 4287). Average absolute deviation from the mean line.",
        "Rq": "Root-mean-square roughness (ISO 4287). Sensitive to peaks/valleys.",
        "Rsk": "Skewness of height distribution. Negative → valleys dominate (good drainage). Positive → peaks dominate.",
        "Rku": "Kurtosis. >3 = sharp features (spiky surface). <3 = rounded features.",
        "Rk": "Core roughness depth (ISO 13565). Height of the linear region on the Abbott-Firestone curve.",
        "Rpk": "Reduced peak height. Peaks that wear away quickly (initial contact).",
        "Rvk": "Reduced valley depth. Valley volume available for fluid retention/drainage.",
        "Sa": "Areal arithmetic mean height (ISO 25178). 3D equivalent of Ra.",
        "Sq": "Areal RMS height (ISO 25178). 3D equivalent of Rq.",
        "Sdr": "Developed interfacial area ratio (ISO 25178). % increase of true surface over projected area.",
        "g_factor": "Material ratio at mid-height (ISO 10844). Related to tyre/road contact.",
        "FractalDim": "Fractal dimension (box-counting). Characterises surface complexity; 1.0=smooth, 2.0=space-filling.",
        "MeanSlope": "Mean absolute slope Rdq. Higher → more micro-friction.",
        "PeakDensity": "Peak count per metre. Related to contact point density.",
    }

    fnames = st.session_state["file_names"]
    for fname in fnames:
        agg = st.session_state["aggregated"].get(fname, {})
        areal = st.session_state["results_areal"].get(fname, {})
        n_profiles = len(st.session_state["results_1d"].get(fname, []))

        with st.expander(f"📄 {fname}", expanded=(len(fnames) == 1)):
            st.caption(f"Aggregated from {n_profiles} profiles")

            # ── Compact metric cards ──────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MPD", fmt(agg.get("MPD"), "mm"))
            c2.metric("Ra", fmt(agg.get("Ra"), "µm"))
            c3.metric("Rq", fmt(agg.get("Rq"), "µm"))
            c4.metric("ETD", fmt(agg.get("ETD"), "mm"))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rsk", fmt(agg.get("Rsk")))
            c2.metric("Rku", fmt(agg.get("Rku")))
            c3.metric("Sa", fmt(areal.get("Sa"), "µm"))
            c4.metric("Sdr", fmt(areal.get("Sdr"), "%"))

            # ── Summary table with descriptions ───────────────────
            st.markdown("##### Detailed surface parameters")
            summary_rows = []
            param_order = [
                ("MPD", "mm", "Profile"), ("ETD", "mm", "Profile"),
                ("Ra", "µm", "Profile"), ("Rq", "µm", "Profile"),
                ("Rsk", "–", "Profile"), ("Rku", "–", "Profile"),
                ("Rk", "µm", "Profile"), ("Rpk", "µm", "Profile"),
                ("Rvk", "µm", "Profile"),
                ("g_factor", "–", "Profile"),
                ("MeanSlope", "–", "Profile"),
                ("PeakDensity", "pk/m", "Profile"),
                ("FractalDim", "–", "Profile"),
                ("Sa", "µm", "Areal"), ("Sq", "µm", "Areal"),
                ("Sdr", "%", "Areal"),
            ]
            for key, unit, scope in param_order:
                val = agg.get(key, areal.get(key, None))
                std = agg.get(f"{key}_std", None)
                desc = DESCR.get(key, "")
                summary_rows.append({
                    "Parameter": key,
                    "Value": fmt(val, unit),
                    "Std": fmt(std) if std is not None else "–",
                    "Scope": scope,
                    "Description": desc,
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
                height=min(450, 35 * 6 + 38),  # ~5 rows visible (6 = header + 5 rows)
            )

            # Surface views (2D heatmap + 3D)
            grid_obj = [s for s, fn in zip(
                st.session_state["surfaces"], fnames) if fn == fname]
            if grid_obj:
                dx_val = st.session_state.get("s_dx", 1.0)
                dy_val = st.session_state.get("s_dy", 1.0)
                _uz = st.session_state.get("s_uz", "units")
                _ve = st.session_state.get("s_vexag", 1.0)
                _rc = st.session_state.get("s_robcol", True)
                view_mode = st.radio(
                    "Surface view", ["2D heatmap", "3D surface"],
                    horizontal=True, key=f"view_{fname}")
                if view_mode == "2D heatmap":
                    st.caption("Colour = height after pre-processing. Axes in grid units.")
                    st.plotly_chart(
                        heatmap_2d(grid_obj[0].z, dx_val, dy_val,
                                   title=f"Surface – {fname}",
                                   units_z=_uz, robust_color=_rc),
                        use_container_width=True)
                else:
                    st.caption(
                        f"Interactive 3D view. Vertical exaggeration ×{_ve} (render only). "
                        "Drag to rotate, scroll to zoom.")
                    st.plotly_chart(
                        surface_3d(grid_obj[0].z, dx_val, dy_val,
                                   title=f"3D Surface – {fname}",
                                   units_z=_uz, vert_exag=_ve,
                                   robust_color=_rc),
                        use_container_width=True)

    if st.session_state["warnings"]:
        st.warning(f"{len(st.session_state['warnings'])} warning(s)")
        with st.expander("Show warnings"):
            for w in st.session_state["warnings"]:
                st.text(w)


def render_table(dx, dy, agg_mode, cfg):
    """Show full parameter table and exports."""
    fnames = st.session_state["file_names"]
    selected = st.selectbox("Select file", fnames, key="tbl_file")
    agg = st.session_state["aggregated"].get(selected, {})
    areal = st.session_state["results_areal"].get(selected, {})

    notes = f"agg={agg_mode}, plane={cfg.plane_removal}"
    tbl = build_results_table(agg, areal, notes)

    st.markdown("### Full Parameter Table")
    
    # 1D / 3D Display Filter
    param_filter = st.radio("Display Parameters", ["All", "1D Parameters", "3D Parameters", "Custom"], horizontal=True, key="tbl_filter")
    disp_tbl = tbl.copy()
    if param_filter == "1D Parameters":
        disp_tbl = disp_tbl[disp_tbl["Dim"] == "1D"]
    elif param_filter == "3D Parameters":
        disp_tbl = disp_tbl[disp_tbl["Dim"] != "1D"]
    elif param_filter == "Custom":
        sel_params = st.multiselect("Select parameters", disp_tbl["Parameter"].tolist(), default=disp_tbl["Parameter"].tolist())
        disp_tbl = disp_tbl[disp_tbl["Parameter"].isin(sel_params)]
        
    st.dataframe(disp_tbl, use_container_width=True, height=min(600, 35 * len(disp_tbl) + 38))

    st.markdown("### Export")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.download_button("⬇ CSV", export_csv(tbl),
                           "texturelab_results.csv", "text/csv")
    with col_e2:
        st.download_button("⬇ Excel", export_excel(tbl),
                           "texturelab_results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col_e3:
        json_str = export_json_report(
            agg, areal, settings=cfg.to_dict(),
            metadata={"file": selected, "dx": dx, "dy": dy},
            logs=st.session_state.get("logs", []))
        st.download_button("⬇ JSON", json_str,
                           "texturelab_report.json", "application/json")

    if len(fnames) > 1:
        st.markdown("---")
        st.markdown("### 📊 Batch Comparison")
        areal_list = [st.session_state["results_areal"].get(fn, {}) for fn in fnames]
        batch_tbl = build_batch_table(st.session_state["batch_agg"], fnames, areal_results=areal_list)
        
        # Start index at 1 instead of 0
        batch_tbl.index = np.arange(1, len(batch_tbl) + 1)
        
        # Allow user to select parameters to show
        all_cols = batch_tbl.columns.tolist()
        all_cols.remove("File")  # We always want to show File
        
        sel_cols = st.multiselect(
            "Select parameters to display", all_cols, default=all_cols,
            key="batch_cols"
        )
        
        display_tbl = batch_tbl[["File"] + sel_cols]
        
        st.dataframe(display_tbl, use_container_width=True)
        
        st.markdown("#### Export Batch Table")
        col_be1, col_be2 = st.columns(2)
        with col_be1:
            st.download_button("⬇ Batch CSV", export_csv(batch_tbl),
                               "texturelab_batch.csv", "text/csv")
        with col_be2:
            st.download_button("⬇ Batch Excel", export_excel(batch_tbl),
                               "texturelab_batch.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                               
        st.markdown("---")
        st.markdown("### 📊 By Pavement Type (Aggregated)")
        st.caption("Group files by their prefix to compare pavement types.")
        prefix_len = st.slider("Prefix length to define pavement type (e.g. A1_MALA is 7)", 2, 30, 7, key="tbl_prefix_len")
        
        batch_tbl_type = batch_tbl.copy()
        batch_tbl_type["Pavement Type"] = batch_tbl_type["File"].apply(lambda x: x[:prefix_len])
        numeric_cols = batch_tbl_type.select_dtypes(include=np.number).columns.tolist()
        if "File" in numeric_cols:
            numeric_cols.remove("File")
            
        if len(numeric_cols) > 0:
            mean_df = batch_tbl_type.groupby("Pavement Type")[numeric_cols].mean().reset_index()
            std_df = batch_tbl_type.groupby("Pavement Type")[numeric_cols].std().reset_index()
            
            st.markdown("#### Mean Values")
            st.dataframe(mean_df, use_container_width=True)
            st.markdown("#### Standard Deviation")
            st.dataframe(std_df, use_container_width=True)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.download_button("⬇ Download Mean CSV", export_csv(mean_df), "pavement_mean.csv", "text/csv")
            with col_m2:
                st.download_button("⬇ Download Mean Excel", export_excel(mean_df), "pavement_mean.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
            st.markdown("#### Texture Level Spectrum")
            # Extract OctaveBands from batch_agg
            spectrum_data = {}
            for fn, agg_data in zip(fnames, st.session_state["batch_agg"]):
                p_type = fn[:prefix_len]
                if "OctaveBands" in agg_data:
                    if p_type not in spectrum_data:
                        spectrum_data[p_type] = []
                    spectrum_data[p_type].append(agg_data["OctaveBands"])
            
            # Average the bands per pavement type
            mean_spectrum_by_type = {}
            for p_type, bands_list in spectrum_data.items():
                if not bands_list:
                    continue
                mean_bands = {}
                band_keys = set()
                for b in bands_list:
                    band_keys.update(b.keys())
                for bk in band_keys:
                    vals = [b[bk] for b in bands_list if bk in b]
                    mean_bands[bk] = np.mean(vals)
                mean_spectrum_by_type[p_type] = mean_bands
                
            if mean_spectrum_by_type:
                fig_spectrum = texture_spectrum_plot(mean_spectrum_by_type)
                st.plotly_chart(fig_spectrum, use_container_width=True)
            else:
                st.info("No spectral data available to plot Texture Level Spectrum.")


def render_visualization():
    """Interactive visualization tab."""
    fnames = st.session_state["file_names"]
    selected = st.selectbox("Select file", fnames, key="viz_file")
    per_profile = st.session_state["results_1d"].get(selected, [])

    if not per_profile:
        st.warning("No profile results available.")
        return

    df_profiles = pd.DataFrame(per_profile)
    num_cols = sorted(df_profiles.select_dtypes(include="number").columns.tolist())
    n_profiles = len(df_profiles)

    st.info(
        f"📊 **{n_profiles} profiles** extracted from this surface. "
        f"Each profile produces one set of parameters. The plots below show "
        f"the **variability across profiles within the surface** — this helps "
        f"identify heterogeneity, measurement artefacts, and spatial trends. "
        f"For a single representative value per parameter, see the **Summary** tab."
    )

    st.markdown("### Distribution of a single parameter")
    st.caption("Histogram + boxplot showing how the parameter varies across profiles.")
    param1 = st.selectbox("Parameter", num_cols, key="viz_p1")
    meta = PARAM_REGISTRY.get(param1)
    if meta:
        st.markdown(f"*{meta.definition}* ({meta.standard})")
    col_h, col_b = st.columns(2)
    with col_h:
        st.plotly_chart(histogram(df_profiles, param1), use_container_width=True)
    with col_b:
        st.plotly_chart(boxplot(df_profiles, param1), use_container_width=True)

    st.markdown("---")
    st.markdown("### Relationship between two parameters")
    st.caption("Each dot = one profile. Look for trends or clusters.")
    col_x, col_y = st.columns(2)
    with col_x:
        px_ = st.selectbox("X axis", num_cols, index=0, key="viz_px")
    with col_y:
        py_ = st.selectbox("Y axis", num_cols,
                           index=min(1, len(num_cols) - 1), key="viz_py")
    st.plotly_chart(scatter_2d(df_profiles, px_, py_), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3D scatter – three parameters")
    st.caption("Explore interactions between three descriptors simultaneously.")
    col_3a, col_3b, col_3c = st.columns(3)
    with col_3a:
        p3x = st.selectbox("X", num_cols, index=0, key="viz3x")
    with col_3b:
        p3y = st.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1), key="viz3y")
    with col_3c:
        p3z = st.selectbox("Z", num_cols, index=min(2, len(num_cols) - 1), key="viz3z")
    st.plotly_chart(scatter_3d(df_profiles, p3x, p3y, p3z), use_container_width=True)

    st.markdown("---")
    st.markdown("### Correlation heatmap")
    st.caption("Pearson correlation between selected parameters (across profiles).")
    sel_corr = st.multiselect("Select parameters", num_cols,
                              default=num_cols[:8], key="viz_corr_sel")
    if sel_corr:
        st.plotly_chart(correlation_heatmap(df_profiles, sel_corr),
                       use_container_width=True)

    st.markdown("### Pair plot")
    st.caption("Matrix of scatter plots for quick visual screening.")
    sel_pair = st.multiselect("Parameters for pair plot", num_cols,
                              default=num_cols[:4], key="viz_pair_sel")
    if sel_pair:
        st.plotly_chart(pair_plot(df_profiles, sel_pair), use_container_width=True)


def render_data_science():
    """Data Science tab with PCA, Clustering, Regression, Anomaly, Features."""
    fnames = st.session_state["file_names"]

    # ── Sample selector ──
    st.markdown("### Sample Selection")
    selected_samples = st.multiselect(
        "Samples to include", fnames, default=fnames, key="ds_samples")
    if not selected_samples:
        st.warning("Select at least one sample.")
        return

    # ── Scope ──
    scope_options = ["Per Profile (All data)"]
    if len(selected_samples) >= 2:
        scope_options = ["Per Surface (File average)", "Per Profile (All data)", "Per Sample (Single file)"]
    else:
        scope_options = ["Per Profile (All data)", "Per Sample (Single file)"]

    scope = st.radio("Analysis Basis", scope_options, horizontal=True, key="ds_scope")

    all_rows = []
    if scope == "Per Surface (File average)":
        for fn in selected_samples:
            agg = dict(st.session_state["aggregated"].get(fn, {}))
            agg["_file"] = fn
            all_rows.append(agg)
    elif scope == "Per Sample (Single file)":
        sample_choice = st.selectbox("Select sample", selected_samples, key="ds_sample_choice")
        for row in st.session_state["results_1d"].get(sample_choice, []):
            r = dict(row)
            r["_file"] = sample_choice
            all_rows.append(r)
    else:  # Per Profile (All data)
        for fn in selected_samples:
            for row in st.session_state["results_1d"].get(fn, []):
                r = dict(row)
                r["_file"] = fn
                all_rows.append(r)

    if not all_rows:
        st.warning("No data available for selected samples.")
        return

    df_all = pd.DataFrame(all_rows)
    num_cols = sorted(df_all.select_dtypes(include="number").columns.tolist())

    if len(selected_samples) < 2:
        st.info(
            "💡 **Single sample selected.** Analyses use per-profile data within this surface. "
            "Select multiple samples or use **Batch Analysis** for between-surface comparisons."
        )

    ds_tab1, ds_tab2, ds_tab3, ds_tab4, ds_tab5 = st.tabs(
        ["PCA", "Clustering", "Regression", "Anomaly Detection", "Feature Selection"])

    with ds_tab1:
        st.markdown("### Principal Component Analysis")
        pca_feats = st.multiselect("Features for PCA", num_cols,
                                   default=num_cols[:10], key="pca_feats")
        if pca_feats and len(pca_feats) >= 2:
            pca_res = run_pca(df_all, pca_feats)
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    pca_variance_plot(pca_res["explained_variance_ratio"]),
                    use_container_width=True)
            with col_b:
                st.plotly_chart(
                    pca_biplot(pca_res["scores"], pca_res["loadings"],
                               pca_res["feature_names"]),
                    use_container_width=True)
            st.markdown(f"**Cumulative variance (2 PC):** "
                        f"{sum(pca_res['explained_variance_ratio'][:2]):.1%}")

    with ds_tab2:
        st.markdown("### Clustering")
        clust_feats = st.multiselect("Features", num_cols,
                                     default=num_cols[:8], key="clust_feats")
        clust_method = st.selectbox("Algorithm",
                                    ["K-Means", "GMM", "Ward"], key="clust_algo")
        k = st.slider("Number of clusters (k)", 2, 10, 3)

        if clust_feats and len(clust_feats) >= 2 and st.button("Run clustering"):
            X_sc, _ = prepare_feature_matrix(df_all, clust_feats)
            X_np = X_sc.values
            if clust_method == "K-Means":
                res = run_kmeans(X_np, k)
            elif clust_method == "GMM":
                res = run_gmm(X_np, k)
            else:
                res = run_ward(X_np, k)
            st.metric("Silhouette score", f"{res['silhouette']:.3f}")
            elb = elbow_analysis(X_np)
            st.plotly_chart(elbow_plot(elb["k_range"], elb["inertias"],
                                      elb["silhouettes"]),
                           use_container_width=True)
            pca_r = run_pca(df_all, clust_feats, n_components=2)
            df_pca = pd.DataFrame(pca_r["scores"][:, :2], columns=["PC1", "PC2"])
            df_pca["Cluster"] = res["labels"].astype(str)
            st.plotly_chart(
                px.scatter(df_pca, x="PC1", y="PC2", color="Cluster",
                           template="plotly_dark", title="Clusters in PCA space"),
                use_container_width=True)

    with ds_tab3:
        st.markdown("### Regression")
        target_source = st.radio("Target source",
                                 ["From computed parameters", "Import external CSV"],
                                 key="reg_src")
        if target_source == "From computed parameters":
            target_col = st.selectbox("Target", num_cols, key="reg_target")
            feature_cols = st.multiselect(
                "Features", [c for c in num_cols if c != target_col],
                default=[c for c in num_cols[:8] if c != target_col], key="reg_feats")
            ext_y = None
        else:
            ext_file = st.file_uploader("Upload target CSV", type=["csv"], key="reg_ext")
            target_col = None
            feature_cols = st.multiselect("Features", num_cols,
                                          default=num_cols[:8], key="reg_feats_ext")
            ext_y = None
            if ext_file:
                ext_df = pd.read_csv(ext_file)
                target_col = st.selectbox("Target column",
                                          ext_df.columns.tolist(), key="reg_ext_col")
                ext_y = ext_df[target_col].values[:len(df_all)]

        model_name = st.selectbox("Model",
                                  ["linear", "ridge", "lasso", "random_forest"],
                                  key="reg_model")
        split_by = st.checkbox("Split CV by file (prevent leakage)", value=True)

        if st.button("Train model", key="reg_train"):
            if feature_cols and target_col:
                X_sc, _ = prepare_feature_matrix(df_all, feature_cols)
                X_np = X_sc.values
                y = (ext_y[:len(X_np)] if ext_y is not None
                     else df_all[target_col].values[:len(X_np)])
                groups = df_all["_file"].values[:len(X_np)] if split_by else None
                reg_res = run_regression(X_np, y, model_name, groups)
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("R²", f"{reg_res['r2']:.4f}")
                col_m2.metric("CV R² (mean)", f"{reg_res['cv_r2_mean']:.4f}")
                col_m3.metric("RMSE", f"{reg_res['rmse']:.4f}")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.plotly_chart(actual_vs_predicted(y, reg_res["y_pred"]),
                                   use_container_width=True)
                with col_p2:
                    st.plotly_chart(residual_plot(y, reg_res["y_pred"]),
                                   use_container_width=True)
                if reg_res["importance"] is not None:
                    st.plotly_chart(
                        feature_importance_plot(feature_cols, reg_res["importance"]),
                        use_container_width=True)

    with ds_tab4:
        st.markdown("### Anomaly Detection (Isolation Forest)")
        anom_feats = st.multiselect("Features", num_cols,
                                    default=num_cols[:8], key="anom_feats")
        contamination = st.slider("Contamination", 0.01, 0.20, 0.05, 0.01)
        if anom_feats and st.button("Detect anomalies"):
            X_sc, _ = prepare_feature_matrix(df_all, anom_feats)
            anom_res = run_isolation_forest(X_sc.values, contamination)
            n_anom = int((anom_res["labels"] == -1).sum())
            st.metric("Anomalies detected", f"{n_anom} / {len(anom_res['labels'])}")
            pca_r = run_pca(df_all, anom_feats, n_components=2)
            df_pca = pd.DataFrame(pca_r["scores"][:, :2], columns=["PC1", "PC2"])
            df_pca["Anomaly"] = np.where(anom_res["labels"] == -1, "Anomaly", "Normal")
            st.plotly_chart(
                px.scatter(df_pca, x="PC1", y="PC2", color="Anomaly",
                           color_discrete_map={"Anomaly": "#EF553B",
                                               "Normal": "#636EFA"},
                           template="plotly_dark", title="Anomalies in PCA space"),
                use_container_width=True)

    with ds_tab5:
        st.markdown("### Feature Selection")
        st.markdown("#### Correlation-based pruning")
        corr_thresh = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01)
        if st.button("Prune correlated features"):
            to_drop = correlation_pruning(df_all[num_cols], corr_thresh)
            st.write(f"**Drop ({len(to_drop)}):** {to_drop}")
            st.write(f"**Keep:** {[c for c in num_cols if c not in to_drop]}")

        st.markdown("---")
        st.markdown("#### Bootstrap Confidence Intervals")
        ci_param = st.selectbox("Parameter", num_cols, key="ci_param")
        ci_n = st.number_input("Bootstrap iterations", 500, 5000, 1000, 100)
        if st.button("Compute CI"):
            vals = df_all[ci_param].dropna().values
            mean_val, lo, hi = bootstrap_ci(vals, int(ci_n))
            st.metric(ci_param, f"{mean_val:.4f}",
                      delta=f"95% CI: [{lo:.4f}, {hi:.4f}]")


# ===================================================================
# PAGE: Home
# ===================================================================
def page_home():
    st.markdown('<p class="hero-title">🔬 TextureLab</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Advanced Pavement Surface Texture Analysis</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # Welcome cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">📄</div>
            <div class="welcome-title">New Analysis</div>
            <div class="welcome-desc">Upload a single surface file (CSV or LAZ) and run a complete texture analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_new", use_container_width=True):
            _set_page("new")
            st.rerun()

    with col2:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">📂</div>
            <div class="welcome-title">Open File</div>
            <div class="welcome-desc">Browse and open a surface file from your local file system for analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_open", use_container_width=True):
            _set_page("open")
            st.rerun()

    with col3:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">📦</div>
            <div class="welcome-title">Batch Analysis</div>
            <div class="welcome-desc">Process multiple files at once and compare results across surfaces.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_batch", use_container_width=True):
            _set_page("batch")
            st.rerun()

    with col4:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">🔍</div>
            <div class="welcome-title">Compare Results</div>
            <div class="welcome-desc">Side-by-side comparison of analysed surfaces with statistical tests.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="home_compare", use_container_width=True):
            _set_page("compare")
            st.rerun()

    st.markdown("---")

    # Quick stats
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Texture Parameters", f"{len(PARAM_REGISTRY)}")
    col_b.metric("Supported Standards", "ISO 13473 · ISO 4287 · ISO 25178 · ISO 13565 · ISO 10844")
    col_c.metric("Input Formats", "CSV · LAZ · LAS")


# ===================================================================
# PAGE: New Analysis (single file)
# ===================================================================
def page_new():
    st.markdown("## 📄 New Analysis")
    st.markdown("Upload a single surface file for comprehensive texture analysis.")

    dx, dy, units_xy, units_z, direction, every_n, agg_mode, selected_params, cfg, vert_exag, robust_color = render_settings_panel()

    uploaded = st.file_uploader(
        "Upload surface file (CSV / TXT / LAZ / LAS)",
        type=["csv", "txt", "laz", "las"],
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.get('uploader_key', 0)}_single",
        help="Upload one surface measurement file.")

    if uploaded and st.button("🚀  Run Analysis", use_container_width=True):
        process_files([uploaded], dx, dy, units_xy, units_z,
                      direction, every_n, agg_mode, selected_params, cfg)

    if st.session_state["processed"]:
        tab_s, tab_t, tab_v, tab_d, tab_l = st.tabs(
            ["📋 Summary", "📊 Table", "📈 Visualize", "🧪 Data Science", "📝 Logs"])
        with tab_s:
            render_summary()
        with tab_t:
            render_table(dx, dy, agg_mode, cfg)
        with tab_v:
            render_visualization()
        with tab_d:
            render_data_science()
        with tab_l:
            render_logs()


# ===================================================================
# PAGE: Open File (from file path)
# ===================================================================
def page_open():
    st.markdown("## 📂 Open Existing File")
    st.markdown("Enter the path to an existing surface file on your system.")

    dx, dy, units_xy, units_z, direction, every_n, agg_mode, selected_params, cfg, vert_exag, robust_color = render_settings_panel()

    file_path = st.text_input("File path",
                              placeholder=r"C:\data\surface_scan.csv",
                              help="Full path to CSV, LAZ, or LAS file")

    if file_path and st.button("🚀  Analyse", use_container_width=True):
        path = Path(file_path)
        if not path.exists():
            st.error(f"File not found: {file_path}")
        else:
            _log(f"📄 Opening {path.name}…")
            try:
                grid = load_surface(str(path), dx, dy,
                                    units_xy=units_xy, units_z=units_z)
                _log(f"  Grid size: {grid.ny}×{grid.nx}")

                z_proc, profiles, warns = preprocess_surface(
                    grid.z, dx, dy, cfg, direction, every_n)
                for w in warns:
                    _log(f"  ⚠ {w}")
                _log(f"  ✅ {len(profiles)} profiles extracted")

                per_profile, areal = compute_all(profiles, z_proc, dx, dy)
                agg = aggregate_profiles(per_profile, agg_mode)

                # Filter results by user selection
                if selected_params:
                    filtered_agg = {}
                    for k in list(agg.keys()):
                        base_k = k.replace("_std", "").replace("_P10", "").replace("_P90", "")
                        if base_k in selected_params:
                            filtered_agg[k] = agg[k]
                    agg = filtered_agg
                    
                    areal = {k: v for k, v in areal.items() if k in selected_params}
                    
                    filtered_profiles = []
                    for res in per_profile:
                        filtered_profiles.append({k: v for k, v in res.items() if k in selected_params})
                    per_profile = filtered_profiles

                fname = path.name
                st.session_state["surfaces"] = [grid]
                st.session_state["file_names"] = [fname]
                st.session_state["profiles"][fname] = profiles
                st.session_state["results_1d"][fname] = per_profile
                st.session_state["results_areal"][fname] = areal
                st.session_state["aggregated"][fname] = agg
                st.session_state["batch_agg"] = [agg]
                st.session_state["warnings"] = warns
                st.session_state["processed"] = True
                _log("🎉 Analysis complete.")
                st.success(f"Processed {fname} successfully!")
            except Exception as e:
                _log(f"  ❌ Error: {e}")
                st.error(f"Error: {e}")

    if st.session_state["processed"]:
        tab_s, tab_t, tab_v, tab_d, tab_l = st.tabs(
            ["📋 Summary", "📊 Table", "📈 Visualize", "🧪 Data Science", "📝 Logs"])
        with tab_s:
            render_summary()
        with tab_t:
            render_table(dx, dy, agg_mode, cfg)
        with tab_v:
            render_visualization()
        with tab_d:
            render_data_science()
        with tab_l:
            render_logs()


# ===================================================================
# PAGE: Batch Analysis
# ===================================================================
def page_batch():
    st.markdown("## 📦 Batch Analysis")
    st.markdown("Upload multiple surface files to process and compare in batch.")

    dx, dy, units_xy, units_z, direction, every_n, agg_mode, selected_params, cfg, vert_exag, robust_color = render_settings_panel()

    uploaded = st.file_uploader(
        "Upload surface files (CSV / TXT / LAZ / LAS)",
        type=["csv", "txt", "laz", "las"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_key', 0)}_batch",
        help="Upload 2+ files for batch comparison.")

    if uploaded and len(uploaded) >= 1 and st.button("🚀  Run Batch Analysis",
                                                      use_container_width=True):
        process_files(uploaded, dx, dy, units_xy, units_z,
                      direction, every_n, agg_mode, selected_params, cfg)

    if st.session_state["processed"]:
        tab_s, tab_t, tab_v, tab_d, tab_l = st.tabs(
            ["📋 Summary", "📊 Table", "📈 Visualize", "🧪 Data Science", "📝 Logs"])
        with tab_s:
            render_summary()
        with tab_t:
            render_table(dx, dy, agg_mode, cfg)
        with tab_v:
            render_visualization()
        with tab_d:
            render_data_science()
        with tab_l:
            render_logs()


# ===================================================================
# PAGE: Compare Results
# ===================================================================
def page_compare():
    st.markdown("## 🔍 Compare Results")
    st.markdown("Load and compare results from multiple analysis sessions.")

    if not st.session_state["processed"] or len(st.session_state["file_names"]) < 2:
        st.info("Run a **Batch Analysis** with 2+ files first, then come here to compare.")
        return

    fnames = st.session_state["file_names"]
    st.markdown(f"**Files loaded:** {len(fnames)}")

    if "chart_styles" not in st.session_state:
        st.session_state["chart_styles"] = {}
    
    colors = px.colors.qualitative.Plotly
    styles = st.session_state["chart_styles"]
    
    for i, fn in enumerate(fnames):
        if fn not in styles:
            styles[fn] = {
                "color": colors[i % len(colors)],
                "dash": "solid"
            }

    with st.expander("🎨 Chart Styling Options"):
        st.markdown("Customize the colors and line styles for each pavement scan. These will be kept across all comparative charts.")
        for fn in fnames:
            cols = st.columns([3, 1, 1])
            cols[0].write(f"**{fn}**")
            new_color = cols[1].color_picker("Color", styles[fn]["color"], key=f"color_{fn}", label_visibility="collapsed")
            new_dash = cols[2].selectbox("Line Type", ["solid", "dash", "dot", "dashdot"], 
                                         index=["solid", "dash", "dot", "dashdot"].index(styles[fn]["dash"]), 
                                         key=f"dash_{fn}", label_visibility="collapsed")
            styles[fn]["color"] = new_color
            styles[fn]["dash"] = new_dash

    # Side-by-side key metrics
    st.markdown("### Key Metrics Comparison")
    
    export_figs = {} # Store figures for vector ZIP export
    
    areal_list = [st.session_state["results_areal"].get(fn, {}) for fn in fnames]
    full_batch_tbl = build_batch_table(
        st.session_state["batch_agg"], fnames, None, areal_results=areal_list)
        
    # 1D / 3D Display Filter for Batch
    param_filter_batch = st.radio("Display Parameters (Batch)", ["All", "1D Parameters", "3D Parameters", "Custom"], horizontal=True, key="tbl_filter_batch")
    
    # We can use the PARAM_REGISTRY to figure out what is 1D vs 3D
    from src.descriptors import PARAM_REGISTRY
    all_params = [c for c in full_batch_tbl.columns if c != "File"]
    disp_cols = ["File"]
    compare_params = []
    
    if param_filter_batch == "All":
        disp_cols += all_params
        compare_params = ["MPD", "Ra", "Rq", "Rsk", "Rk", "Sdr"] # Default for charts
    elif param_filter_batch == "Custom":
        compare_params = st.multiselect(
            "Parameters to compare",
            all_params,
            default=["MPD", "Ra", "Rq", "Rsk", "Rk", "Sdr"],
            key="compare_params")
        disp_cols += compare_params
    else:
        for p in all_params:
            meta = PARAM_REGISTRY.get(p)
            dim = meta.dim if meta else "1D"
            if param_filter_batch == "1D Parameters" and dim == "1D":
                disp_cols.append(p)
                compare_params.append(p)
            elif param_filter_batch == "3D Parameters" and dim != "1D":
                disp_cols.append(p)
                compare_params.append(p)
                
    disp_batch_tbl = full_batch_tbl[disp_cols]
    st.dataframe(disp_batch_tbl, use_container_width=True)
    
    # Batch Export (Always exports ALL parameters)
    st.markdown("#### Export Batch Table (Contains All Parameters)")
    col_be1, col_be2 = st.columns(2)
    with col_be1:
        st.download_button("⬇ Batch CSV", export_csv(full_batch_tbl), "batch_results.csv", "text/csv")
    with col_be2:
        st.download_button("⬇ Export Excel", export_excel(full_batch_tbl), "compare_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="btn_xl_comp")

    # Bar chart comparison
    if compare_params:
        color_map = {fn: styles[fn]["color"] for fn in fnames}
        # Limit to 6 charts max to avoid crowding
        for param in compare_params[:6]:
            if param not in full_batch_tbl.columns: continue
            vals = full_batch_tbl[param].tolist()
            fig = px.bar(x=fnames, y=vals, title=f"{param} comparison",
                         template="plotly_dark",
                         color=fnames,
                         color_discrete_map=color_map,
                         labels={"x": "File", "y": param})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            export_figs[f"BarChart_{param}"] = fig
        if len(compare_params) > 6:
            st.info(f"Showing first 6 parameters as charts. Select 'Custom' to plot specific parameters.")

    st.markdown("---")
    
    tab_pca, tab_psd, tab_abbott, tab_3d = st.tabs([
        "🌌 PCA Comparison", "📈 Power Spectral Density (PSD)", "📉 Abbott-Firestone Curve", "🧊 3D Surfaces Gallery"
    ])
    
    with tab_pca:
        st.markdown("### Principal Component Analysis (Batch)")
        st.markdown("Automatically clusters the analysed files in 2D space based on their parameters to find similar surfaces.")
        
        areal_list = [st.session_state["results_areal"].get(fn, {}) for fn in fnames]
        batch_tbl_pca = build_batch_table(st.session_state["batch_agg"], fnames, areal_results=areal_list)
        num_cols = batch_tbl_pca.select_dtypes(include="number").columns.tolist()
        pca_feats = st.multiselect("Features for PCA", num_cols,
                                   default=[c for c in num_cols if c in 
                                            ["MPD", "Ra", "Rq", "Rsk", "Rku", "Rk", "Sa", "Sdr", "MeanSlope"]], 
                                   key="comp_pca_feats")
        
        if pca_feats and len(pca_feats) >= 2 and len(fnames) >= 2:
            try:
                pca_res = run_pca(batch_tbl_pca, pca_feats, n_components=2)
                df_pca = pd.DataFrame(pca_res["scores"][:, :2], columns=["PC1", "PC2"])
                df_pca["File"] = fnames
                
                fig = px.scatter(df_pca, x="PC1", y="PC2", color="File", text="File",
                                 title="Surfaces in PCA space",
                                 color_discrete_map=color_map,
                                 template="plotly_dark", size_max=15)
                fig.update_traces(textposition='top center', marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)
                export_figs["PCA_Comparison"] = fig
                
                st.caption(f"**Variance explained:** PC1: {pca_res['explained_variance_ratio'][0]:.1%} | PC2: {pca_res['explained_variance_ratio'][1]:.1%}")
                
                # PCA Exports
                st.markdown("#### Export PCA Results")
                col_pca1, col_pca2, col_pca3 = st.columns(3)
                with col_pca1:
                    st.download_button("⬇ PCA Data CSV", export_csv(df_pca), "pca_results.csv", "text/csv")
                with col_pca2:
                    st.download_button("⬇ PCA Data Excel", export_excel(df_pca), "pca_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col_pca3:
                    try:
                        img_bytes = fig.to_image(format="svg", engine="kaleido")
                        st.download_button("⬇ PCA Figure (.svg)", img_bytes, "pca_plot.svg", "image/svg+xml")
                    except Exception as e:
                        st.warning(f"Figure export unavailable (requires kaleido): {e}")

            except Exception as e:
                st.warning(f"Could not run PCA: {e}")
                
    with tab_psd:
        st.markdown("### PSD Comparison")
        st.markdown("Compares the wavelength distribution of the surfaces. Data is averaged over all extracted profiles per file.")
        
        dx = st.session_state.get("s_dx", 1.0)
        col1, col2 = st.columns(2)
        with col1:
            smooth_psd = st.toggle("Smooth (1/3 Octave Bands)", value=True)
        with col2:
            layout_mode_psd = st.radio("Layout Mode (PSD)", ["Combined (Single Chart)", "Separate Subplots", "Group by Prefix"], horizontal=True)
        
        prefix_len_psd = 7
        if layout_mode_psd == "Group by Prefix":
            prefix_len_psd = st.number_input("Number of characters to define group prefix (PSD)", min_value=1, max_value=50, value=7)
            
        def _get_psd_data(fn):
            profs = st.session_state["profiles"].get(fn, [])
            if not profs: return None
            psds = []
            freqs = None
            dx_val = st.session_state.get("s_dx", 1.0)
            for p in profs:
                f, psd = calc_psd_welch(p, dx_val)
                freqs = f
                psds.append(psd)
            if freqs is None or len(psds) == 0: return None
            
            mean_psd = np.mean(psds, axis=0)
            if smooth_psd:
                from src.descriptors import octave_band_rms
                bands = octave_band_rms(freqs, mean_psd, n_octave=3)
                if bands:
                    z_unit = st.session_state.get("s_uz", "mm")
                    offset = 60 if z_unit == "mm" else 0
                    wls = [1.0 / float(fc) for fc in bands.keys() if float(fc) > 0]
                    rms_db = [20 * np.log10(v + 1e-15) + offset for v in bands.values()]
                    wls, rms_db = zip(*sorted(zip(wls, rms_db)))
                    return wls, rms_db, True
            else:
                mask = freqs > 0
                wls = 1.0 / freqs[mask]
                rms_db = 10 * np.log10(mean_psd[mask] + 1e-15)
                return wls, rms_db, False
            return None

        if layout_mode_psd == "Combined (Single Chart)":
            fig = go.Figure()
            for fn in fnames:
                data = _get_psd_data(fn)
                if data:
                    wls, rms_db, is_smooth = data
                    mode = 'lines+markers' if is_smooth else 'lines'
                    shape = 'spline' if is_smooth else 'linear'
                    fig.add_trace(go.Scatter(
                        x=wls, y=rms_db, mode=mode, name=fn,
                        line=dict(color=styles[fn]["color"], dash=styles[fn]["dash"], shape=shape)
                    ))
                    
            if len(fig.data) > 0:
                h_mode = "x unified" if len(fnames) <= 5 else "closest"
                fig.update_layout(
                    template="plotly_dark",
                    xaxis=dict(title="Wavelength λ (mm)", type="log", dtick=1),
                    yaxis=dict(title="Texture Level Spectrum L_TX (dB)" if smooth_psd else "Power Spectral Density (dB)"),
                    hovermode=h_mode
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No raw profile data found in memory. If you loaded an older project file, please re-run the Batch Analysis to view PSD curves.")
                
        elif layout_mode_psd == "Separate Subplots":
            for fn in fnames:
                data = _get_psd_data(fn)
                if data:
                    wls, rms_db, is_smooth = data
                    mode = 'lines+markers' if is_smooth else 'lines'
                    shape = 'spline' if is_smooth else 'linear'
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wls, y=rms_db, mode=mode, name=fn,
                        line=dict(color=styles[fn]["color"], dash=styles[fn]["dash"], shape=shape)
                    ))
                    
                    fig.update_layout(
                        title=fn, template="plotly_dark", 
                        xaxis=dict(title="Wavelength λ (mm)", type="log", dtick=1),
                        yaxis=dict(title="Texture Level Spectrum L_TX (dB)" if smooth_psd else "Power Spectral Density (dB)"),
                        hovermode="x unified", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    export_figs[f"PSD_Separate_{fn}"] = fig
                    
        elif layout_mode_psd == "Group by Prefix":
            groups = {}
            for fn in fnames:
                pfx = fn[:prefix_len_psd]
                if pfx not in groups:
                    groups[pfx] = []
                groups[pfx].append(fn)
                
            for idx, (pfx, fn_list) in enumerate(groups.items()):
                all_rms = []
                wls_common = None
                is_smooth = False
                for fn in fn_list:
                    data = _get_psd_data(fn)
                    if data:
                        wls, rms_db, is_smooth = data
                        wls_common = wls
                        all_rms.append(rms_db)
                        
                if wls_common is not None and all_rms:
                    mean_rms = np.mean(all_rms, axis=0)
                    color = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
                    mode = 'lines+markers' if is_smooth else 'lines'
                    shape = 'spline' if is_smooth else 'linear'
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wls_common, y=mean_rms, mode=mode, name=f"{pfx}* (n={len(all_rms)})",
                        line=dict(color=color, dash="solid", width=3)
                    ))
                    
                    y_type = "log" if log_y else None
                    fig.update_layout(
                        title=f"Prefix: {pfx}* (n={len(all_rms)})", template="plotly_dark", 
                        xaxis=dict(title="Wavelength λ (mm)", type="log", dtick=1),
                        yaxis=dict(title="Mean Texture Level Spectrum L_TX (dB)" if smooth_psd else "Mean Power Spectral Density (dB)", type=y_type),
                        hovermode="x unified", height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    export_figs[f"PSD_Group_{pfx}"] = fig
                    
        # PSD Data Export
        st.markdown("#### Export PSD Data")
        psd_export_data = []
        for fn in fnames:
            d = _get_psd_data(fn)
            if d:
                wls, rms, _ = d
                for w, r in zip(wls, rms):
                    psd_export_data.append({"File": fn, "Wavelength (mm)": w, "Amplitude (dB)": r})
        if psd_export_data:
            df_psd_ex = pd.DataFrame(psd_export_data)
            col_psd1, col_psd2 = st.columns(2)
            with col_psd1:
                st.download_button("⬇ PSD Data CSV", export_csv(df_psd_ex), "psd_data.csv", "text/csv")
            with col_psd2:
                st.download_button("⬇ PSD Data Excel", export_excel(df_psd_ex), "psd_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
    with tab_abbott:
        st.markdown("### Abbott-Firestone Curve (Bearing Area)")
        st.markdown("Shows the cumulative height distribution (Material Ratio) of the surfaces.")
        
        layout_mode = st.radio("Layout Mode", ["Combined (Single Chart)", "Separate Subplots", "Group by Prefix"], horizontal=True)
        prefix_len = 7
        if layout_mode == "Group by Prefix":
            prefix_len = st.number_input("Number of characters to define group prefix", min_value=1, max_value=50, value=7)
            
        def _get_abbott_data(fn):
            grid_obj = [s for s, name in zip(st.session_state["surfaces"], fnames) if name == fn]
            if not grid_obj: return None
            z = grid_obj[0].z
            z_valid = z[np.isfinite(z)]
            if len(z_valid) == 0: return None
            z_sorted = np.sort(z_valid)[::-1]
            step = max(1, len(z_sorted) // 250)
            z_sub = z_sorted[::step]
            mr = np.linspace(0, 100, len(z_sub))
            return mr, z_sub
            
        if layout_mode == "Combined (Single Chart)":
            fig = go.Figure()
            for fn in fnames:
                data = _get_abbott_data(fn)
                if data:
                    mr, z_sub = data
                    fig.add_trace(go.Scatter(x=mr, y=z_sub, mode='lines', name=fn, line=dict(color=styles[fn]["color"], dash=styles[fn]["dash"])))
            if len(fig.data) > 0:
                h_mode = "x unified" if len(fnames) <= 5 else "closest"
                fig.update_layout(template="plotly_dark", xaxis_title="Material Ratio (%)", yaxis_title=f"Height (mm)", hovermode=h_mode)
                st.plotly_chart(fig, use_container_width=True)
                export_figs["Abbott_Combined"] = fig
            else:
                st.warning("No raw surface data found in memory. If you loaded an older project file, please re-run the Batch Analysis to view Abbott curves.")
                
        elif layout_mode == "Separate Subplots":
            from src.descriptors import calc_rk_params
            for fn in fnames:
                data = _get_abbott_data(fn)
                if data:
                    mr, z_sub = data
                    
                    # Calculate parameters specifically for this 3D curve to draw accurate lines
                    params_3d = calc_rk_params(z_sub)
                    mr1, mr2 = params_3d["Mr1"], params_3d["Mr2"]
                    rk = params_3d["Rk"]
                    rpk, rvk = params_3d["Rpk"], params_3d["Rvk"]
                    
                    z_top = z_sub[0] - rpk
                    z_bottom = z_sub[-1] + rvk
                    
                    fig = go.Figure()
                    
                    # Rk Tangent Line (Inclination)
                    m = (z_bottom - z_top) / (mr2 - mr1) if mr2 > mr1 else 0
                    z_at_0 = z_top - m * mr1
                    z_at_100 = z_bottom + m * (100 - mr2)
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 100], y=[z_at_0, z_at_100],
                        mode='lines', name='Core Inclination (Rk Tangent)',
                        line=dict(color='cyan', dash='dash', width=2)
                    ))
                    
                    # Rpk Triangle Area
                    fig.add_trace(go.Scatter(
                        x=[0, mr1, 0, 0], y=[z_top, z_top, z_sub[0], z_top],
                        mode='lines', fill='toself', fillcolor='rgba(255,0,0,0.3)',
                        line=dict(color='red', width=1), name='Rpk Area'
                    ))
                    
                    # Rvk Triangle Area
                    fig.add_trace(go.Scatter(
                        x=[100, mr2, 100, 100], y=[z_bottom, z_bottom, z_sub[-1], z_bottom],
                        mode='lines', fill='toself', fillcolor='rgba(255,0,0,0.3)',
                        line=dict(color='red', width=1), name='Rvk Area'
                    ))
                    
                    # Horizontal Lines
                    fig.add_trace(go.Scatter(
                        x=[0, mr1], y=[z_top, z_top], mode='lines', 
                        line=dict(color='gray', dash='dash', width=1), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[mr2, 100], y=[z_bottom, z_bottom], mode='lines', 
                        line=dict(color='gray', dash='dash', width=1), showlegend=False
                    ))
                    
                    # Vertical Lines
                    z_min = z_sub[-1] - abs(rvk) - 0.5 # A bit below
                    fig.add_trace(go.Scatter(
                        x=[mr1, mr1], y=[z_top, z_min], mode='lines', 
                        line=dict(color='gray', dash='dash', width=1), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[mr2, mr2], y=[z_bottom, z_min], mode='lines', 
                        line=dict(color='gray', dash='dash', width=1), showlegend=False
                    ))

                    # The actual curve
                    fig.add_trace(go.Scatter(x=mr, y=z_sub, mode='lines', name=fn, line=dict(color=styles[fn]["color"], dash=styles[fn]["dash"])))
                    
                    # Annotations
                    fig.add_annotation(x=0, y=z_top + rpk/2, text="Rpk", showarrow=False, xanchor="right", xshift=-5, font=dict(color="red"))
                    fig.add_annotation(x=0, y=z_top - rk/2, text="Rk", showarrow=False, xanchor="right", xshift=-5, font=dict(color="green"))
                    fig.add_annotation(x=100, y=z_bottom - rvk/2, text="Rvk", showarrow=False, xanchor="left", xshift=5, font=dict(color="red"))
                    fig.add_annotation(x=mr1, y=z_min, text="Mr1", showarrow=False, yanchor="top", yshift=-5, font=dict(color="green"))
                    fig.add_annotation(x=mr2, y=z_min, text="Mr2", showarrow=False, yanchor="top", yshift=-5, font=dict(color="green"))
                    
                    fig.update_layout(title=fn, template="plotly_dark", xaxis_title="Material Ratio (%)", yaxis_title="Height (mm)", hovermode="x unified", height=500)
                    
                    col_p, col_t = st.columns([3, 1])
                    with col_p:
                        st.plotly_chart(fig, use_container_width=True)
                        export_figs[f"Abbott_Separate_{fn}"] = fig
                    with col_t:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown("**Parameters (Profile Avg):**")
                        agg = st.session_state["aggregated"].get(fn, {})
                        params = {"Parameter": [], "Value": []}
                        rk_val = agg.get("Rk")
                        mr1_val = agg.get("Mr1")
                        mr2_val = agg.get("Mr2")
                        
                        for k in ["Rk", "Rpk", "Rvk", "Mr1", "Mr2"]:
                            val = agg.get(k)
                            params["Parameter"].append(k)
                            params["Value"].append(f"{val:.3f}" if isinstance(val, (int, float)) else "-")
                            
                        if rk_val is not None and mr1_val is not None and mr2_val is not None and (mr2_val - mr1_val) > 0:
                            slope = rk_val / (mr2_val - mr1_val)
                            params["Parameter"].append("Slope (Rk/ΔMr)")
                            params["Value"].append(f"{slope:.4f}")
                            params["Parameter"].append("Slope (%)")
                            params["Value"].append(f"{slope * 100:.2f}%")
                            
                        st.dataframe(pd.DataFrame(params), hide_index=True, use_container_width=True)
                        
                        st.markdown("**Parameters (Surface):**")
                        params_surf = {"Parameter": [], "Value": []}
                        sk_val = params_3d.get("Rk")
                        smr1_val = params_3d.get("Mr1")
                        smr2_val = params_3d.get("Mr2")
                        
                        for k in ["Rk", "Rpk", "Rvk", "Mr1", "Mr2"]:
                            val = params_3d.get(k)
                            params_surf["Parameter"].append(k.replace("R", "S"))
                            params_surf["Value"].append(f"{val:.3f}" if isinstance(val, (int, float)) else "-")
                            
                        if sk_val is not None and smr1_val is not None and smr2_val is not None and (smr2_val - smr1_val) > 0:
                            slope_surf = sk_val / (smr2_val - smr1_val)
                            params_surf["Parameter"].append("Slope (Sk/ΔSMr)")
                            params_surf["Value"].append(f"{slope_surf:.4f}")
                            params_surf["Parameter"].append("Slope (%)")
                            params_surf["Value"].append(f"{slope_surf * 100:.2f}%")
                            
                        st.dataframe(pd.DataFrame(params_surf), hide_index=True, use_container_width=True)
        elif layout_mode == "Group by Prefix":
            groups = {}
            for fn in fnames:
                pfx = fn[:prefix_len]
                if pfx not in groups: groups[pfx] = []
                groups[pfx].append(fn)
                
            for idx, (pfx, fn_list) in enumerate(groups.items()):
                mr_std = np.linspace(0, 100, 1000)
                z_interp = []
                group_params = {"Rk": [], "Rpk": [], "Rvk": [], "Mr1": [], "Mr2": []}
                for fn in fn_list:
                    d = _get_abbott_data(fn)
                    if d:
                        z_interp.append(np.interp(mr_std, d[0], d[1]))
                    agg = st.session_state["aggregated"].get(fn, {})
                    for k in group_params.keys():
                        v = agg.get(k)
                        if v is not None:
                            group_params[k].append(v)
                            
                if z_interp:
                    z_mean = np.mean(z_interp, axis=0)
                    mr_mean = mr_std
                    color = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scattergl(x=mr_std, y=z_mean, mode='lines', name=f"{pfx}* (n={len(z_interp)})", line=dict(color=color, dash="solid")))
                    fig.update_layout(title=f"Prefix: {pfx}* (n={len(z_interp)})", template="plotly_dark", xaxis_title="Material Ratio (%)", yaxis_title="Mean Height (mm)", hovermode="x unified", height=400)
                    
                    col_p, col_t = st.columns([3, 1])
                    with col_p:
                        st.plotly_chart(fig, use_container_width=True)
                    with col_t:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown(f"**Mean Params ({pfx}*):**")
                        params = {"Parameter": [], "Mean Value": []}
                        
                        rk_val = np.mean(group_params["Rk"]) if group_params["Rk"] else None
                        mr1_val = np.mean(group_params["Mr1"]) if group_params["Mr1"] else None
                        mr2_val = np.mean(group_params["Mr2"]) if group_params["Mr2"] else None
                        
                        for k in ["Rk", "Rpk", "Rvk", "Mr1", "Mr2"]:
                            val_list = group_params[k]
                            mean_val = np.mean(val_list) if val_list else None
                            params["Parameter"].append(k)
                            params["Mean Value"].append(f"{mean_val:.3f}" if mean_val is not None else "-")
                            
                        if rk_val is not None and mr1_val is not None and mr2_val is not None and (mr2_val - mr1_val) > 0:
                            slope = rk_val / (mr2_val - mr1_val)
                            params["Parameter"].append("Slope (Rk/ΔMr)")
                            params["Mean Value"].append(f"{slope:.4f}")
                            
                        st.dataframe(pd.DataFrame(params), hide_index=True, use_container_width=True)
                else:
                    st.warning(f"No surface data available for prefix {pfx}*.")

        st.markdown("---")
        st.markdown("#### Export Curve Data")
        mr_std = np.linspace(0, 100, 1000)
        export_dict = {"Material Ratio (%)": mr_std}
        for fn in fnames:
            data = _get_abbott_data(fn)
            if data:
                export_dict[fn] = np.interp(mr_std, data[0], data[1])
        if len(export_dict) > 1:
            df_export = pd.DataFrame(export_dict)
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.download_button("⬇ Download CSV", export_csv(df_export),
                                   "abbott_curves.csv", "text/csv")
            with col_exp2:
                st.download_button("⬇ Download Excel", export_excel(df_export),
                                   "abbott_curves.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="btn_xl_abb_c")

        st.markdown("#### Export Abbott Parameters")
        abbott_params = {"File": []}
        prof_keys = ["Rk", "Rpk", "Rvk", "Mr1", "Mr2"]
        surf_keys = ["Rk", "Rpk", "Rvk", "Mr1", "Mr2"]
        for k in prof_keys:
            abbott_params[f"Profile_{k}"] = []
        for k in surf_keys:
            abbott_params[f"Surface_{k.replace('R', 'S')}"] = []
            
        for fn in fnames:
            abbott_params["File"].append(fn)
            agg = st.session_state["aggregated"].get(fn, {})
            params_3d = st.session_state["results_areal"].get(fn, {})
            for k in prof_keys:
                abbott_params[f"Profile_{k}"].append(agg.get(k, None))
            for k in surf_keys:
                abbott_params[f"Surface_{k.replace('R', 'S')}"].append(params_3d.get(k.replace('R', 'S'), None))
                
        df_params = pd.DataFrame(abbott_params)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.download_button("⬇ Download Params CSV", export_csv(df_params), "abbott_params.csv", "text/csv", key="btn_csv_abb_p")
        with col_p2:
            st.download_button("⬇ Download Params Excel", export_excel(df_params), "abbott_params.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="btn_xl_abb_p")

    with tab_3d:
        st.markdown("### 🧊 3D Surfaces Gallery")
        st.markdown("Compare the visual topography of all loaded surfaces side-by-side.")
        
        # Determine a reasonable grid size
        n_surfaces = len(fnames)
        cols_per_row = min(2, n_surfaces) # 2 columns max for good visibility
        
        for i in range(0, n_surfaces, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < n_surfaces:
                    fn = fnames[i + j]
                    grid_obj = [s for s, name in zip(st.session_state["surfaces"], fnames) if name == fn]
                    if grid_obj:
                        dx_val = st.session_state.get("s_dx", 1.0)
                        dy_val = st.session_state.get("s_dy", 1.0)
                        
                        with cols[j]:
                            st.markdown(f"**{fn}**")
                            _ve = st.session_state.get("s_vexag", 1.0)
                            _rc = st.session_state.get("s_robcol", True)
                            _uz = st.session_state.get("s_uz", "units")
                            fig_3d = surface_3d(grid_obj[0].z, dx_val, dy_val, title="",
                                                units_z=_uz, vert_exag=_ve, robust_color=_rc)
                            fig_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                            st.plotly_chart(fig_3d, use_container_width=True)


# ===================================================================
# PAGE: Help
# ===================================================================
def page_help():
    st.markdown("## ❓ Help – TextureLab User Guide")

    st.markdown("### 📘 Basic Operating Manual")
    st.markdown("""
    **1. Starting an Analysis**
    - **Data Import**: Click "Browse files" or drag-and-drop your `.csv`, `.las`, or `.laz` files.
    - **Physical Grid Config**: Ensure the grid sizes ($dx$, $dy$) match your scanner (e.g. 1.0mm or 0.1mm) and select the right unit.
    - **Pre-processing Options**: Choose how to filter the surface. Removing the macroscopic slope ("Plane removal") is critical.
    - Click **"Run Analysis"** to process.

    **2. Single File Review**
    - View a 2D Heatmap or a realistic 3D Surface map.
    - **Display Parameters**: Use the radio buttons (1D/3D) to filter the table. The backend always calculates all parameters.
    - Export the single-file table to CSV, Excel, or JSON.

    **3. Batch Comparison**
    - **Key Metrics Comparison**: View a consolidated table of all files. The Batch CSV/Excel download buttons will *always* export every parameter.
    - **Charts**: Easily compare PCA clusters, Power Spectral Density (PSD), and Abbott-Firestone curves. 
    - **3D Surfaces Gallery**: Visually compare all surfaces in a true-scale 3D grid.
    - **Vector Export (.zip)**: Click "Generate Vector Figures" at the bottom to download publication-ready `.svg` versions of all active plots.
    """)

    st.markdown("### Supported File Formats")
    st.markdown("""
    | Format | Extension | Description |
    |--------|-----------|-------------|
    | CSV (x,y,z) | `.csv` | Three-column coordinate file |
    | CSV (matrix) | `.csv` | Dense grid (rows = y, cols = x) |
    | LAS | `.las` | ASPRS LiDAR point cloud |
    | LAZ | `.laz` | Compressed LAS (requires laszip) |
    """)

    st.markdown("### Pre-processing Options")
    st.markdown("""
    | Option | Choices | Description |
    |--------|---------|-------------|
    | Plane removal | None / Plane / Polynomial | Remove tilt or curvature from surface |
    | Outlier filtering | None / Hampel / Median | Remove spike artifacts |
    | Gap interpolation | On / Off | Fill NaN values by linear interpolation |
    | Detrending | None / Mean / Linear | Remove trend per profile |
    | Bandpass filter | FFT / IIR | Keep wavelengths in macrotexture range only |
    """)

    st.markdown("---")
    st.markdown("### 📐 Complete Parameter Reference Table")
    st.markdown("Use the checkboxes in the **Calculate** column to select which parameters should be computed and displayed in the app tables and graphs.")

    # Build reference dataframe with a Calculate checkbox column
    ref_rows = []
    current_selected = set(st.session_state.get("selected_params", []))
    
    for name, meta in PARAM_REGISTRY.items():
        ref_rows.append({
            "Calculate": name in current_selected,
            "ID": name,
            "Parameter": meta.symbol,
            "Definition": meta.definition,
            "Dim": meta.dim,
            "Standard": meta.standard,
            "Unit": meta.unit,
        })
    ref_df = pd.DataFrame(ref_rows)
    
    # We use a callback or directly update from the returned dataframe
    edited_df = st.data_editor(
        ref_df,
        column_config={
            "Calculate": st.column_config.CheckboxColumn(
                "Calculate",
                help="Select to include this parameter in analysis",
                default=True,
            ),
            "ID": None, # Hide internal ID column
        },
        disabled=["Parameter", "Definition", "Dim", "Standard", "Unit"],
        hide_index=True,
        use_container_width=True,
        height=600,
        key="param_table_editor"
    )
    
    # Update session state based on user edits
    if edited_df is not None:
        new_selected = edited_df[edited_df["Calculate"] == True]["ID"].tolist()
        if new_selected != st.session_state["selected_params"]:
            st.session_state["selected_params"] = new_selected

    st.markdown("---")
    st.markdown("### Equations (LaTeX)")
    with st.expander("Show all equations", expanded=False):
        for name, meta in PARAM_REGISTRY.items():
            st.latex(f"{meta.symbol}: \\quad {meta.equation_latex}")

    st.markdown("---")
    st.markdown("### Data Science Tab Guide")
    st.markdown("""
    - **PCA**: Select features, view explained variance and biplot
    - **Clustering**: Choose algorithm (K-Means/GMM/Ward), use elbow plot to pick k
    - **Regression**: Predict a target variable from texture parameters (supports external CSV with CPX, friction, or drainage targets)
    - **Anomaly Detection**: Flag unusual profiles using Isolation Forest
    - **Feature Selection**: Prune correlated features or use RFE; compute bootstrap confidence intervals
    """)


# ===================================================================
# PAGE: About
# ===================================================================
def page_about():
    st.markdown("## ℹ️ About TextureLab")
    st.markdown("")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown(f"""
        **TextureLab** is a comprehensive pavement surface texture analysis tool
        designed for researchers and engineers working in road surface
        characterisation, tyre/road noise, friction, and drainage assessment.

        ### Author
        **{APP_AUTHOR}**
        Scientific Officer — Pavement Surface Engineering

        ### Version
        v{APP_VERSION} — Released {APP_YEAR}

        ### Purpose
        TextureLab provides a unified environment for:
        - Ingesting 3D surface measurements from laser scanners and profilometers
        - Computing standardised texture descriptors (ISO 13473, ISO 4287, ISO 25178, ISO 13565, ISO 10844)
        - Performing multivariate statistical analysis (PCA, clustering, regression)
        - Generating reproducible analysis reports

        ### Technology
        Built with Python, Streamlit, NumPy, SciPy, Scikit-learn, and Plotly.

        ### Contact
        For questions, bug reports, or collaboration inquiries, please reach out
        to the author.
        """)

    with col_r:
        st.markdown("""
        ### Quick Facts

        | | |
        |---|---|
        | **Parameters** | 35+ |
        | **ISO Standards** | 5 |
        | **ML Algorithms** | 7 |
        | **Export Formats** | CSV, Excel, JSON |
        | **License** | CC BY-NC 4.0 |
        """)

    st.markdown("---")
    st.markdown("### Acknowledgements")
    st.markdown("""
    This tool builds on established standards and methods from the road surface
    metrology community. Special thanks to the ISO TC 43/SC 1 and ISO TC 213
    working groups for the foundational standards.
    """)


# ===================================================================
# PAGE: License
# ===================================================================
def page_license():
    st.markdown("## 📜 License")
    st.markdown("")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
                border: 1px solid rgba(102, 126, 234, 0.25);
                border-radius: 16px; padding: 2rem; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin-top: 0;">
            Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
        </h3>
        <p style="color: #d1d5db; font-size: 0.95rem;">
            Copyright © """ + APP_YEAR + " " + APP_AUTHOR + """
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### You are free to:

    - **Share** — copy and redistribute the material in any medium or format
    - **Adapt** — remix, transform, and build upon the material

    The licensor cannot revoke these freedoms as long as you follow the license terms.

    ### Under the following terms:

    - **Attribution** — You must give appropriate credit, provide a link to the
      license, and indicate if changes were made. You may do so in any reasonable
      manner, but not in any way that suggests the licensor endorses you or your use.

    - **NonCommercial** — You may **not** use the material for commercial purposes.
      This includes, but is not limited to:
      - Selling the software or derivative works
      - Using the software in commercial consulting services without permission
      - Incorporating the software into commercial products

    ### No additional restrictions

    You may not apply legal terms or technological measures that legally restrict
    others from doing anything the license permits.

    ---

    ### Full License Text

    The full license text is available at:
    [creativecommons.org/licenses/by-nc/4.0/legalcode](https://creativecommons.org/licenses/by-nc/4.0/legalcode)

    ---

    ### Third-party licenses

    TextureLab uses the following open-source libraries:

    | Library | License |
    |---------|---------|
    | Streamlit | Apache 2.0 |
    | NumPy | BSD 3-Clause |
    | SciPy | BSD 3-Clause |
    | Pandas | BSD 3-Clause |
    | Scikit-learn | BSD 3-Clause |
    | Plotly | MIT |
    | laspy | BSD 2-Clause |
    | OpenPyXL | MIT |
    | PyYAML | MIT |
    """)


# ===================================================================
# Logs page function
# ===================================================================
def render_logs():
    st.markdown("### 📝 Processing Logs")
    if st.session_state["logs"]:
        for line in st.session_state["logs"]:
            st.text(line)
    else:
        st.info("No logs yet.")


# ===================================================================
# Page Router
# ===================================================================
page = st.session_state.get("page", "home")

if page == "home":
    page_home()
elif page == "new":
    page_new()
elif page == "open":
    page_open()
elif page == "batch":
    page_batch()
elif page == "compare":
    page_compare()
elif page == "help":
    page_help()
elif page == "about":
    page_about()
elif page == "license":
    page_license()
else:
    page_home()
