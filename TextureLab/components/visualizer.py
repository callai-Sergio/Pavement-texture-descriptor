"""
visualizer.py – Interactive Plotly visualizations for TextureLab
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ===================================================================
# Single-parameter plots
# ===================================================================

def histogram(df: pd.DataFrame, col: str, nbins: int = 30,
              title: str = "") -> go.Figure:
    fig = px.histogram(df, x=col, nbins=nbins, marginal="rug",
                       title=title or f"Distribution of {col}",
                       template="plotly_dark",
                       color_discrete_sequence=["#636EFA"])
    fig.update_layout(bargap=0.05)
    return fig


def boxplot(df: pd.DataFrame, col: str,
            group_col: Optional[str] = None,
            title: str = "") -> go.Figure:
    fig = px.box(df, y=col, x=group_col,
                 title=title or f"Box plot – {col}",
                 template="plotly_dark",
                 color=group_col,
                 points="outliers")
    return fig


def heatmap_2d(z: np.ndarray, dx: float, dy: float,
               title: str = "Surface height map",
               units_z: str = "units",
               robust_color: bool = True) -> go.Figure:
    ny, nx = z.shape
    x_coords = np.arange(nx) * dx
    y_coords = np.arange(ny) * dy
    x_range = x_coords[-1] - x_coords[0] if nx > 1 else 1.0
    y_range = y_coords[-1] - y_coords[0] if ny > 1 else 1.0
    aspect = y_range / x_range if x_range > 0 else 1.0

    # Robust color scale: clamp to P1–P99
    valid = z[np.isfinite(z)]
    if robust_color and len(valid) > 0:
        cmin = float(np.percentile(valid, 1))
        cmax = float(np.percentile(valid, 99))
    elif len(valid) > 0:
        cmin, cmax = float(valid.min()), float(valid.max())
    else:
        cmin, cmax = 0, 1

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_coords,
        y=y_coords,
        colorscale="Viridis",
        zmin=cmin, zmax=cmax,
        colorbar=dict(title=f"Height ({units_z})", thickness=15),
    ))
    plot_width = 700
    plot_height = max(300, min(800, int(plot_width * aspect)))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="X", yaxis_title="Y",
        width=plot_width, height=plot_height,
    )
    return fig


def surface_3d(z: np.ndarray, dx: float, dy: float,
               title: str = "3D Surface",
               max_pts: int = 300,
               units_z: str = "units",
               vert_exag: float = 0.3,
               robust_color: bool = True) -> go.Figure:
    """Interactive 3D surface mesh (downsampled via block averaging).

    Parameters
    ----------
    vert_exag : float
        Vertical exaggeration for rendering only. Default 0.3 flattens
        spikes visually while the colorbar still shows real height values.
    robust_color : bool
        Clamp colorbar to P1-P99 percentiles so outlier spikes/pits
        don't stretch the colour map.
    """
    ny, nx = z.shape
    step_x = max(1, nx // max_pts)
    step_y = max(1, ny // max_pts)

    # Block-average downsample (anti-alias), never stride
    if step_x > 1 or step_y > 1:
        ny_trim = (ny // step_y) * step_y
        nx_trim = (nx // step_x) * step_x
        z_trim = z[:ny_trim, :nx_trim]
        z_ds = np.nanmean(
            z_trim.reshape(ny_trim // step_y, step_y,
                           nx_trim // step_x, step_x),
            axis=(1, 3))
    else:
        z_ds = z.copy()

    ny_ds, nx_ds = z_ds.shape
    x_coords = np.arange(nx_ds) * dx * step_x
    y_coords = np.arange(ny_ds) * dy * step_y

    # Robust color scale: P1-P99
    valid = z_ds[np.isfinite(z_ds)]
    if robust_color and len(valid) > 0:
        cmin = float(np.percentile(valid, 1))
        cmax = float(np.percentile(valid, 99))
    elif len(valid) > 0:
        cmin, cmax = float(valid.min()), float(valid.max())
    else:
        cmin, cmax = 0, 1

    # Vertical exaggeration for rendering only
    z_render = z_ds * vert_exag

    fig = go.Figure(data=[go.Surface(
        z=z_render, x=x_coords, y=y_coords,
        surfacecolor=z_ds,           # colour from REAL z values
        colorscale="Viridis",
        cmin=cmin, cmax=cmax,
        colorbar=dict(title=f"Height ({units_z})", thickness=12, len=0.6),
        contours=dict(z=dict(show=False)),
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2,
                      roughness=0.7, fresnel=0.1),
        lightposition=dict(x=0, y=0, z=10000),
    )])
    fig.update_layout(
        title=title, template="plotly_dark",
        width=750, height=550,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title=f"Height ({units_z}) ×{vert_exag}",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            aspectmode="manual",
            aspectratio=dict(
                x=1.0,
                y=ny_ds / nx_ds if nx_ds > 0 else 1.0,
                z=0.25,
            ),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ===================================================================
# Two-parameter plots
# ===================================================================

def scatter_2d(df: pd.DataFrame, x: str, y: str,
               color: Optional[str] = None,
               title: str = "") -> go.Figure:
    try:
        fig = px.scatter(df, x=x, y=y, color=color,
                         title=title or f"{y} vs {x}",
                         template="plotly_dark",
                         trendline="ols")
    except Exception:
        fig = px.scatter(df, x=x, y=y, color=color,
                         title=title or f"{y} vs {x}",
                         template="plotly_dark")
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    return fig


def correlation_heatmap(df: pd.DataFrame,
                        cols: Optional[List[str]] = None) -> go.Figure:
    if cols:
        df = df[cols]
    corr = df.select_dtypes(include="number").corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation heatmap", template="plotly_dark",
                      width=800, height=700)
    return fig


# ===================================================================
# Three-parameter plots
# ===================================================================

def scatter_3d(df: pd.DataFrame, x: str, y: str, z: str,
               color: Optional[str] = None) -> go.Figure:
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color,
                        title=f"{z} vs {x}, {y}",
                        template="plotly_dark")
    return fig


def pair_plot(df: pd.DataFrame,
              cols: Optional[List[str]] = None) -> go.Figure:
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()[:6]
    fig = px.scatter_matrix(df, dimensions=cols,
                            template="plotly_dark",
                            title="Pair plot")
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    return fig


# ===================================================================
# PCA visualizations
# ===================================================================

def pca_variance_plot(explained: np.ndarray) -> go.Figure:
    cumulative = np.cumsum(explained)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=list(range(1, len(explained) + 1)),
                         y=explained, name="Individual",
                         marker_color="#636EFA"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=list(range(1, len(explained) + 1)),
                             y=cumulative, name="Cumulative",
                             line=dict(color="#EF553B", width=3)),
                  secondary_y=True)
    fig.update_layout(title="PCA – Explained variance",
                      xaxis_title="Component",
                      template="plotly_dark")
    fig.update_yaxes(title_text="Individual ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative ratio", secondary_y=True)
    return fig


def pca_biplot(scores: np.ndarray, loadings: np.ndarray,
               feature_names: List[str],
               labels: Optional[np.ndarray] = None) -> go.Figure:
    fig = go.Figure()
    # Scores
    marker = dict(size=5, opacity=0.7)
    if labels is not None:
        marker["color"] = labels
        marker["colorscale"] = "Viridis"
    fig.add_trace(go.Scatter(
        x=scores[:, 0], y=scores[:, 1], mode="markers",
        marker=marker, name="Scores"))
    # Loadings as arrows
    scale = np.max(np.abs(scores[:, :2])) / np.max(np.abs(loadings[:, :2]) + 1e-9)
    for i, name in enumerate(feature_names):
        fig.add_annotation(
            ax=0, ay=0, axref="x", ayref="y",
            x=loadings[i, 0] * scale * 0.8,
            y=loadings[i, 1] * scale * 0.8,
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="#EF553B", text=name, font=dict(size=9))
    fig.update_layout(title="PCA Biplot", template="plotly_dark",
                      xaxis_title="PC1", yaxis_title="PC2")
    return fig


# ===================================================================
# Clustering visualizations
# ===================================================================

def elbow_plot(k_range: list, inertias: list,
               silhouettes: list) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=k_range, y=inertias, name="Inertia",
                             line=dict(color="#636EFA", width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=k_range, y=silhouettes, name="Silhouette",
                             line=dict(color="#00CC96", width=2)),
                  secondary_y=True)
    fig.update_layout(title="Elbow / Silhouette analysis",
                      xaxis_title="k", template="plotly_dark")
    return fig


# ===================================================================
# Regression visualizations
# ===================================================================

def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(size=5, color="#636EFA")))
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    fig.update_layout(title="Residual plot", template="plotly_dark",
                      xaxis_title="Predicted", yaxis_title="Residual")
    return fig


def actual_vs_predicted(y_true: np.ndarray,
                        y_pred: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(size=5, color="#636EFA"),
                             name="Data"))
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(dash="dash", color="white"),
                             name="1:1"))
    fig.update_layout(title="Actual vs Predicted", template="plotly_dark",
                      xaxis_title="Actual", yaxis_title="Predicted")
    return fig


def feature_importance_plot(names: List[str],
                            importance: np.ndarray) -> go.Figure:
    order = np.argsort(importance)
    fig = go.Figure(go.Bar(
        x=importance[order], y=[names[i] for i in order],
        orientation="h", marker_color="#636EFA"))
    fig.update_layout(title="Feature importance", template="plotly_dark",
                      xaxis_title="Importance", yaxis_title="Feature",
                      height=max(400, len(names) * 25))
    return fig
