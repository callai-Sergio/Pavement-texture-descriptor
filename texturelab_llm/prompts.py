"""
texturelab_llm.prompts – Prompt templates and builders for LLM tasks.

All prompts include safety guardrails:
  • Never fabricate data or numbers.
  • Explicitly cite which metrics were used.
  • If information is missing, list clarifying questions instead of guessing.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import pandas as pd

from .utils import df_to_compact_text, format_bullets, safe_truncate


# ---------------------------------------------------------------------------
# System prompts (shared safety preamble)
# ---------------------------------------------------------------------------

_SAFETY_PREAMBLE = """\
You are an expert pavement engineering and surface texture assistant.

RULES YOU MUST FOLLOW:
1. NEVER invent, fabricate, or hallucinate data, numbers, or measurement results.
2. If the user has not provided sufficient data, list the specific information you need before answering.
3. When interpreting results, ALWAYS cite which specific metrics or values you are referencing.
4. Use SI units unless the user explicitly requests otherwise.
5. Write in a technical but accessible style appropriate for civil engineering professionals.
"""


# ---------------------------------------------------------------------------
# Draft generator
# ---------------------------------------------------------------------------

def build_project_draft_prompt(inputs: dict) -> Tuple[str, str]:
    """Build system + user prompts for project draft generation.

    Expected keys in *inputs*:
        title, context, objectives, methodology, partners, budget,
        constraints (dict with keys like 'pages', 'tone', 'language').
    """
    system = _SAFETY_PREAMBLE + """
You are writing a project proposal / technical report draft.
Structure your output with clear sections:
  1. Title & Executive Summary
  2. Context & Background
  3. Objectives
  4. Methodology
  5. Partnerships & Resources
  6. Budget Overview
  7. Timeline (if information provided)
  8. Expected Impact

Adapt length and tone to the constraints provided.
If any section lacks input data, write a placeholder note "[PROVIDE: ...]" instead of inventing content.
"""

    parts: list[str] = []
    parts.append(f"## Title / Topic\n{inputs.get('title', '(not provided)')}\n")
    parts.append(f"## Context / Background\n{inputs.get('context', '(not provided)')}\n")

    obj = inputs.get("objectives", "")
    parts.append(f"## Objectives\n{format_bullets(obj)}\n")

    meth = inputs.get("methodology", "")
    parts.append(f"## Methodology\n{format_bullets(meth)}\n")

    parts.append(f"## Partners\n{inputs.get('partners', '(not provided)')}\n")
    parts.append(f"## Budget\n{inputs.get('budget', '(not provided)')}\n")

    constraints = inputs.get("constraints", {})
    if isinstance(constraints, dict):
        c_lines = []
        if constraints.get("pages"):
            c_lines.append(f"- Target length: {constraints['pages']} page(s)")
        if constraints.get("tone"):
            c_lines.append(f"- Tone: {constraints['tone']}")
        if constraints.get("language"):
            c_lines.append(f"- Language: {constraints['language']}")
        if c_lines:
            parts.append("## Constraints\n" + "\n".join(c_lines) + "\n")
    elif constraints:
        parts.append(f"## Constraints\n{constraints}\n")

    prompt = (
        "Based on the information below, write a complete project draft.\n\n"
        + "\n".join(parts)
    )
    return system.strip(), safe_truncate(prompt)


# ---------------------------------------------------------------------------
# Results interpreter
# ---------------------------------------------------------------------------

def build_results_interpret_prompt(
    metrics: Union[dict, pd.DataFrame],
) -> Tuple[str, str]:
    """Build prompts for interpreting texture analysis results.

    *metrics* is either a dict of key-value results or a pandas DataFrame.
    """
    system = _SAFETY_PREAMBLE + """
You are interpreting pavement surface texture measurement results.

INSTRUCTIONS:
1. Summarise the key findings from the data provided.
2. For each metric you discuss, cite the metric name and its value explicitly.
3. Compare values against typical ranges for asphalt surfaces when relevant:
   - MPD: 0.3–2.5 mm (low < 0.5, moderate 0.5–1.5, high > 1.5)
   - Ra:  5–80 µm for typical pavements
   - Skewness: negative → valleys dominate (better drainage)
   - Kurtosis: ~3 is Gaussian; > 3 → sharp peaks/valleys
   - Sdr: 0–50 % for typical textures
4. Highlight any anomalies, unusual values, or potential quality concerns.
5. End with a short "Recommendations" section if applicable.
"""

    if isinstance(metrics, pd.DataFrame):
        data_text = df_to_compact_text(metrics)
    elif isinstance(metrics, dict):
        lines = [f"  {k}: {v}" for k, v in metrics.items()]
        data_text = "\n".join(lines)
    else:
        data_text = str(metrics)

    prompt = (
        "Interpret the following pavement texture measurement results:\n\n"
        + safe_truncate(data_text)
    )
    return system.strip(), prompt


# ---------------------------------------------------------------------------
# Chat system prompt
# ---------------------------------------------------------------------------

def build_chat_system_prompt(context: dict | None = None) -> str:
    """Build the system prompt for interactive chat.

    *context* can include keys like 'project', 'standard', 'language'.
    """
    parts = [_SAFETY_PREAMBLE.strip()]

    parts.append(
        "\nYou are an interactive assistant for the TextureLab pavement "
        "texture analysis application. Answer questions about surface "
        "texture, ISO standards (ISO 13473, ISO 25178), measurement "
        "interpretation, and pavement engineering."
    )

    if context:
        if context.get("project"):
            parts.append(f"\nCurrent project context: {context['project']}")
        if context.get("standard"):
            parts.append(f"\nTarget standard / norm: {context['standard']}")
        if context.get("language"):
            parts.append(f"\nRespond in: {context['language']}")

    return "\n".join(parts)
