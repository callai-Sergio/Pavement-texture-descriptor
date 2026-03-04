"""
texturelab_llm.ollama_client – Low-level HTTP client for the Ollama REST API.
"""
from __future__ import annotations

import json
from typing import Iterator, List, Optional

import requests

from .config import LLMConfig
from .logging_setup import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Health & model listing
# ---------------------------------------------------------------------------

def check_ollama_health(config: LLMConfig) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(
            config.api_url("/api/tags"),
            timeout=5,
        )
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_models(config: LLMConfig) -> List[str]:
    """Return a sorted list of model names available on the Ollama server.

    Returns an empty list if the server is unreachable.
    """
    try:
        r = requests.get(
            config.api_url("/api/tags"),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return sorted(models)
    except Exception as exc:
        logger.error("list_models failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(
    config: LLMConfig,
    prompt: str,
    system: Optional[str] = None,
    stream: bool = False,
) -> str | Iterator[str]:
    """Generate text from Ollama.

    Parameters
    ----------
    config : LLMConfig
        Connection and model settings.
    prompt : str
        The user prompt.
    system : str, optional
        A system prompt prepended to the conversation.
    stream : bool
        If True, return an iterator yielding incremental text chunks.
        If False, return the full response as a single string.

    Returns
    -------
    str or Iterator[str]

    Raises
    ------
    ConnectionError
        If the Ollama server is unreachable.
    RuntimeError
        If the server returns an HTTP error.
    """
    payload: dict = {
        "model": config.model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_ctx": config.num_ctx,
        },
    }
    if system:
        payload["system"] = system

    url = config.api_url("/api/generate")
    logger.info(
        "generate → model=%s stream=%s prompt_len=%d",
        config.model, stream, len(prompt),
    )

    try:
        r = requests.post(
            url,
            json=payload,
            stream=stream,
            timeout=config.timeout_seconds,
        )
        r.raise_for_status()
    except requests.ConnectionError:
        msg = (
            f"Cannot connect to Ollama at {config.ollama_base_url}. "
            "Please ensure Ollama is running (ollama serve)."
        )
        logger.error(msg)
        raise ConnectionError(msg)
    except requests.HTTPError as exc:
        # Check for model-not-found specifically
        if r.status_code == 404:
            msg = (
                f"Model '{config.model}' not found. "
                f"Run: ollama pull {config.model}"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        msg = f"Ollama HTTP error {r.status_code}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg)
    except requests.Timeout:
        msg = (
            f"Ollama request timed out after {config.timeout_seconds}s. "
            "Try a smaller prompt or increase timeout_seconds."
        )
        logger.error(msg)
        raise TimeoutError(msg)

    if stream:
        return _stream_chunks(r)
    else:
        data = r.json()
        text = data.get("response", "")
        logger.info("generate ← response_len=%d", len(text))
        return text


def _stream_chunks(response: requests.Response) -> Iterator[str]:
    """Yield incremental text from a streaming Ollama response."""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done", False):
                return
        except json.JSONDecodeError:
            continue
