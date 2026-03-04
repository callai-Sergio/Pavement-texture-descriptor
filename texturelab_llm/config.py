"""
texturelab_llm.config – Configuration dataclass for Ollama LLM integration.
"""
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the Ollama LLM client.

    Attributes:
        ollama_base_url: Base URL for the Ollama API server.
        model: Model name to use (must be pulled in Ollama first).
        temperature: Sampling temperature (0.0 = deterministic, higher = more creative).
        top_p: Nucleus sampling threshold.
        num_ctx: Context window size in tokens.
        timeout_seconds: HTTP request timeout.
        stream_default: Whether to stream responses by default.
    """
    ollama_base_url: str = "http://localhost:11434"
    model: str = "gemma3"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 4096
    timeout_seconds: int = 120
    stream_default: bool = False

    def api_url(self, endpoint: str) -> str:
        """Build full API URL for a given endpoint."""
        base = self.ollama_base_url.rstrip("/")
        return f"{base}{endpoint}"
