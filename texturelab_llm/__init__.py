"""
texturelab_llm – Local LLM integration for TextureLab via Ollama.

Public API
----------
LLMConfig               Configuration dataclass.
check_ollama_health      Check if Ollama server is reachable.
list_models              List available models on the Ollama server.
generate_project_draft   Generate a structured project proposal draft.
interpret_results        Interpret texture measurement results.
chat_turn                One turn of an interactive chat conversation.
"""

from .config import LLMConfig
from .ollama_client import check_ollama_health, list_models
from .drafting import generate_project_draft
from .interpreter import interpret_results
from .chat import chat_turn

__all__ = [
    "LLMConfig",
    "check_ollama_health",
    "list_models",
    "generate_project_draft",
    "interpret_results",
    "chat_turn",
]
