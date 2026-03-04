"""
texturelab_llm.interpreter – Texture metrics interpretation via Ollama.
"""
from __future__ import annotations

from typing import Dict, Iterator, Optional, Union

import pandas as pd

from .config import LLMConfig
from .logging_setup import get_logger
from .ollama_client import generate
from .prompts import build_results_interpret_prompt
from .utils import safe_truncate

logger = get_logger()


def interpret_results(
    config: LLMConfig,
    metrics: Union[dict, pd.DataFrame],
    context: Optional[dict] = None,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    """Interpret texture analysis results using the LLM.

    Parameters
    ----------
    config : LLMConfig
        Ollama connection and model settings.
    metrics : dict | pd.DataFrame
        The measurement results to interpret (metric names → values,
        or a full DataFrame of per-profile / per-file results).
    context : dict, optional
        Additional context such as ``{'surface_type': 'SMA 11',
        'measurement': 'laser profilometry'}``.
    stream : bool
        If True return an iterator of text chunks.

    Returns
    -------
    str | Iterator[str]
    """
    system, prompt = build_results_interpret_prompt(metrics)

    # Append optional context
    if context:
        ctx_lines = [f"  {k}: {v}" for k, v in context.items()]
        extra = "\n\nAdditional context:\n" + "\n".join(ctx_lines)
        prompt += safe_truncate(extra, max_chars=1000)

    logger.info(
        "interpret_results → model=%s metrics_type=%s stream=%s",
        config.model, type(metrics).__name__, stream,
    )
    return generate(config, prompt, system=system, stream=stream)
