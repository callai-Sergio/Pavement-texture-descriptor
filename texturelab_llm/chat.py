"""
texturelab_llm.chat – Conversational chat with rolling history via Ollama.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Union

from .config import LLMConfig
from .logging_setup import get_logger
from .ollama_client import generate
from .prompts import build_chat_system_prompt
from .utils import safe_truncate

logger = get_logger()

# Maximum number of turns to keep (user + assistant pairs)
MAX_HISTORY_MESSAGES = 20


def chat_turn(
    config: LLMConfig,
    history: List[Dict[str, str]],
    user_message: str,
    context: Optional[dict] = None,
    stream: bool = False,
) -> Union[str, Iterator[str]]:
    """Perform one chat turn and return the assistant response.

    Parameters
    ----------
    config : LLMConfig
        Ollama connection and model settings.
    history : list[dict]
        Conversation history, each item ``{'role': 'user'|'assistant', 'content': ...}``.
        This list is **not** modified; the caller should append the new
        user message and the assistant response after this call.
    user_message : str
        The latest user message.
    context : dict, optional
        Extra context keys (project, standard, language) for the system prompt.
    stream : bool
        If True return an iterator of text chunks.

    Returns
    -------
    str | Iterator[str]
    """
    system = build_chat_system_prompt(context)

    # Build the conversation prompt from history (trim to last N messages)
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    parts: list[str] = []
    for msg in trimmed:
        role = msg.get("role", "user").capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")

    prompt = safe_truncate("\n\n".join(parts))

    logger.info(
        "chat_turn → model=%s history_len=%d stream=%s",
        config.model, len(trimmed), stream,
    )
    return generate(config, prompt, system=system, stream=stream)
