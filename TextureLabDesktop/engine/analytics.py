"""
analytics.py – Lean multivariate analytics for TextureLab Desktop

PCA and Feature Selection only.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


# ===================================================================
# Helpers
# ===================================================================

def prepare_feature_matrix(df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None,
                           ) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize features and return (X_scaled_df, scaler)."""
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number").columns.tolist()
    X = df[feature_cols].dropna(axis=1, how="all").fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler


# ===================================================================
# PCA
# ===================================================================

def run_pca(df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
            n_components: Optional[int] = None
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Run PCA and return scores, loadings, explained variance, feature names."""
    X_scaled, _ = prepare_feature_matrix(df, feature_cols)
    cols = X_scaled.columns.tolist()
    n = n_components or min(len(cols), X_scaled.shape[0], 10)
    pca = PCA(n_components=n)
    scores = pca.fit_transform(X_scaled.values)
    loadings = pca.components_.T
    return scores, loadings, pca.explained_variance_ratio_, cols


# ===================================================================
# Feature Selection
# ===================================================================

def correlation_pruning(df: pd.DataFrame, threshold: float = 0.95
                        ) -> List[str]:
    """Remove highly correlated features (keep one of each pair)."""
    corr = df.select_dtypes(include="number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in df.select_dtypes(include="number").columns
            if c not in to_drop]


def recursive_feature_elimination(X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   n_features: int = 10) -> Tuple[List[str], np.ndarray]:
    """RFE with RandomForest estimator."""
    n_features = min(n_features, X.shape[1])
    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected = [f for f, s in zip(feature_names, rfe.support_) if s]
    return selected, rfe.ranking_
