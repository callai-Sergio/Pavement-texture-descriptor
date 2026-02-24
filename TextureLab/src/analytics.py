"""
analytics.py – Multivariate analytics for TextureLab

PCA, clustering (K-Means, GMM, Ward), regression (Linear, Ridge, Lasso, RF),
anomaly detection (Isolation Forest), feature selection, and bootstrap CI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import (mean_squared_error, r2_score,
                              silhouette_score, mean_absolute_error)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
import joblib


# ===================================================================
# Helpers
# ===================================================================

def prepare_feature_matrix(df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None,
                           ) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features and return (X_scaled_df, scaler)."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if df[c].dtype in ("float64", "float32", "int64")]
    X = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=feature_cols, index=X.index), scaler


# ===================================================================
# PCA
# ===================================================================

def run_pca(df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
            n_components: Optional[int] = None
            ) -> Dict:
    """Run PCA and return scores, loadings, explained variance."""
    X, scaler = prepare_feature_matrix(df, feature_cols)
    if n_components is None:
        n_components = min(X.shape)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return {
        "scores": scores,
        "loadings": pca.components_.T,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "feature_names": list(X.columns),
        "pca_obj": pca,
        "scaler": scaler,
    }


# ===================================================================
# Clustering
# ===================================================================

def run_kmeans(X: np.ndarray, k: int) -> Dict:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else 0.0
    return {"labels": labels, "inertia": km.inertia_,
            "silhouette": sil, "model": km}


def run_gmm(X: np.ndarray, k: int) -> Dict:
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else 0.0
    return {"labels": labels, "bic": gmm.bic(X),
            "silhouette": sil, "model": gmm}


def run_ward(X: np.ndarray, k: int) -> Dict:
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = ward.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else 0.0
    return {"labels": labels, "silhouette": sil, "model": ward}


def elbow_analysis(X: np.ndarray, k_max: int = 10) -> Dict:
    """Return inertia and silhouette for k = 2..k_max."""
    inertias, silhouettes = [], []
    k_range = list(range(2, min(k_max + 1, len(X))))
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return {"k_range": k_range, "inertias": inertias,
            "silhouettes": silhouettes}


# ===================================================================
# Regression
# ===================================================================

REGRESSION_MODELS = {
    "linear": LinearRegression,
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=100, random_state=42),
}


def run_regression(X: np.ndarray, y: np.ndarray,
                   model_name: str = "linear",
                   groups: Optional[np.ndarray] = None,
                   n_splits: int = 5) -> Dict:
    """Fit regression model with cross-validation."""
    if model_name == "linear":
        model = LinearRegression()
    elif model_name in REGRESSION_MODELS:
        model = REGRESSION_MODELS[model_name]()
    else:
        model = LinearRegression()

    # Cross-validation
    if groups is not None and len(np.unique(groups)) >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(model, X, y, cv=cv, groups=groups,
                                     scoring="r2")
    else:
        cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="r2")

    # Full fit
    model.fit(X, y)
    y_pred = model.predict(X)

    # Feature importance
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = None

    return {
        "model": model,
        "y_pred": y_pred,
        "r2": r2_score(y, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "importance": importance,
        "residuals": y - y_pred,
    }


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


# ===================================================================
# Anomaly Detection
# ===================================================================

def run_isolation_forest(X: np.ndarray,
                         contamination: float = 0.05) -> Dict:
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X)
    return {"labels": labels, "scores": scores, "model": iso}


# ===================================================================
# Feature Selection
# ===================================================================

def correlation_pruning(df: pd.DataFrame, threshold: float = 0.95
                        ) -> List[str]:
    """Remove highly correlated features (keep one of each pair)."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return to_drop


def recursive_feature_elimination(X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   n_features: int = 10) -> Dict:
    """RFE with RandomForest estimator."""
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rfe = RFE(rf, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    selected = [f for f, s in zip(feature_names, rfe.support_) if s]
    ranking = dict(zip(feature_names, rfe.ranking_.tolist()))
    return {"selected": selected, "ranking": ranking, "model": rfe}


# ===================================================================
# Bootstrap Confidence Intervals
# ===================================================================

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean."""
    n = len(values)
    means = np.array([
        np.mean(np.random.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, alpha * 100))
    hi = float(np.percentile(means, (1 - alpha) * 100))
    return float(np.mean(values)), lo, hi
