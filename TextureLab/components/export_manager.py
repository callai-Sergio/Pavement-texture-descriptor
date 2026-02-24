"""
export_manager.py – Export results to CSV, Excel, and JSON for TextureLab
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Absolute import via sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.descriptors import PARAM_REGISTRY


# ===================================================================
# Build display table
# ===================================================================

def build_results_table(aggregated: dict, areal: dict,
                        processing_notes: str = "") -> pd.DataFrame:
    """Build the main results DataFrame with metadata columns."""
    rows = []

    # 1-D profile aggregated parameters
    for key, val in aggregated.items():
        if key.endswith(("_std", "_P10", "_P90")):
            continue
        meta = PARAM_REGISTRY.get(key)
        row = {
            "Parameter": meta.symbol if meta else key,
            "Value": val,
            "Std": aggregated.get(f"{key}_std", None),
            "P10": aggregated.get(f"{key}_P10", None),
            "P90": aggregated.get(f"{key}_P90", None),
            "Unit": meta.unit if meta else "–",
            "Dim": meta.dim if meta else "–",
            "Standard": meta.standard if meta else "–",
            "Noise": meta.noise if meta else "–",
            "Friction": meta.friction if meta else "–",
            "Drainage": meta.drainage if meta else "–",
            "Definition": meta.definition if meta else "",
            "Notes": processing_notes,
        }
        rows.append(row)

    # 3-D areal parameters
    for key, val in areal.items():
        meta = PARAM_REGISTRY.get(key)
        row = {
            "Parameter": meta.symbol if meta else key,
            "Value": val,
            "Std": None,
            "P10": None,
            "P90": None,
            "Unit": meta.unit if meta else "–",
            "Dim": meta.dim if meta else "3D",
            "Standard": meta.standard if meta else "–",
            "Noise": meta.noise if meta else "–",
            "Friction": meta.friction if meta else "–",
            "Drainage": meta.drainage if meta else "–",
            "Definition": meta.definition if meta else "",
            "Notes": processing_notes,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_batch_table(batch_results: List[dict],
                      file_names: List[str],
                      key_params: Optional[List[str]] = None,
                      areal_results: Optional[List[dict]] = None
                      ) -> pd.DataFrame:
    """Build batch comparison table (rows = files, cols = params)."""
    
    rows = []
    for i, (fname, res) in enumerate(zip(file_names, batch_results)):
        row = {"File": fname}
        
        # Merge areal results if provided
        areal = areal_results[i] if areal_results and i < len(areal_results) else {}
        combined = {**res, **areal}
        
        if key_params is None:
            # If no specific keys requested, use all of them except the metadata ones (like _std)
            for k, v in combined.items():
                if not k.endswith(('_std', '_P10', '_P90')):
                    row[k] = v
        else:
            for k in key_params:
                row[k] = combined.get(k, None)
                
        rows.append(row)
        
    return pd.DataFrame(rows)


# ===================================================================
# Exporters
# ===================================================================

def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def export_json_report(aggregated: dict, areal: dict,
                       settings: dict, metadata: dict) -> str:
    """Full JSON report including settings for reproducibility."""
    report = {
        "texturelab_version": "1.0.0",
        "metadata": metadata,
        "processing_settings": settings,
        "profile_results_aggregated": _sanitize(aggregated),
        "areal_results": _sanitize(areal),
    }
    return json.dumps(report, indent=2, default=str)


def _sanitize(d: dict) -> dict:
    """Convert numpy types to Python natives for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _sanitize(v)
        else:
            out[k] = v
    return out
