"""
texturelab_llm.drafting – Project draft generation via Ollama.
"""
from __future__ import annotations

from typing import Iterator, Union

from .config import LLMConfig
from .logging_setup import get_logger
from .ollama_client import generate
from .prompts import build_project_draft_prompt

logger = get_logger()


def generate_project_draft(
    config: LLMConfig,
    inputs: dict,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    """Generate a structured project draft from user-provided inputs.

    Parameters
    ----------
    config : LLMConfig
        Ollama connection and model settings.
    inputs : dict
        Dictionary with keys: title, context, objectives, methodology,
        partners, budget, constraints.
    stream : bool
        If True return an iterator of text chunks; otherwise a full string.

    Returns
    -------
    str | Iterator[str]
    """
    system, prompt = build_project_draft_prompt(inputs)
    logger.info("generate_project_draft → model=%s stream=%s", config.model, stream)
    return generate(config, prompt, system=system, stream=stream)
