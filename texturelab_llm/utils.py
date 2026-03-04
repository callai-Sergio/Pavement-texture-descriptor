"""
texturelab_llm.utils – Shared helper functions for prompt building.
"""
from __future__ import annotations

from typing import List, Union

import pandas as pd


def df_to_compact_text(
    df: pd.DataFrame,
    max_rows: int = 20,
    max_cols: int = 12,
) -> str:
    """Convert a DataFrame to a compact text representation suitable for an LLM prompt.

    Includes column names, a truncated head, and basic descriptive statistics.
    """
    if df.empty:
        return "(empty DataFrame)"

    # Limit columns
    cols = list(df.columns[:max_cols])
    subset = df[cols]

    parts: list[str] = []

    # Shape
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    if len(df.columns) > max_cols:
        parts.append(f"(showing first {max_cols} of {len(df.columns)} columns)")

    # Head
    head = subset.head(max_rows).to_string(index=False)
    parts.append(f"\n--- Head ({min(len(subset), max_rows)} rows) ---")
    parts.append(head)

    # Describe (numeric only)
    num = subset.select_dtypes(include="number")
    if not num.empty:
        parts.append("\n--- Descriptive Statistics ---")
        parts.append(num.describe().to_string())

    return "\n".join(parts)


def safe_truncate(text: str, max_chars: int = 8000) -> str:
    """Truncate *text* to at most *max_chars*, appending an ellipsis marker."""
    if len(text) <= max_chars:
        return text
    _MARKER = "\n\n... [truncated for context limit]"
    return text[: max_chars - len(_MARKER)] + _MARKER



def format_bullets(text_or_list: Union[str, List[str]]) -> str:
    """Normalise a string or list into a bullet-point block.

    If *text_or_list* is already a string containing newlines, each non-empty
    line becomes a bullet.  If it is a list, each element becomes a bullet.
    """
    if isinstance(text_or_list, list):
        lines = text_or_list
    else:
        lines = [l.strip() for l in str(text_or_list).splitlines() if l.strip()]

    bullets: list[str] = []
    for line in lines:
        # Strip existing bullet markers for consistency
        clean = line.lstrip("-•*– ").strip()
        if clean:
            bullets.append(f"- {clean}")

    return "\n".join(bullets) if bullets else "(none provided)"
