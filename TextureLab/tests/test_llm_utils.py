"""
Tests for texturelab_llm utility functions and config.
"""
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from texturelab_llm.config import LLMConfig
from texturelab_llm.utils import df_to_compact_text, safe_truncate, format_bullets


# ── LLMConfig ────────────────────────────────────────────────────────────

class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.model == "gemma3"
        assert cfg.temperature == 0.2
        assert cfg.top_p == 0.9
        assert cfg.num_ctx == 4096
        assert cfg.timeout_seconds == 120
        assert cfg.stream_default is False

    def test_custom_values(self):
        cfg = LLMConfig(model="gemma2:2b", temperature=0.8, num_ctx=2048)
        assert cfg.model == "gemma2:2b"
        assert cfg.temperature == 0.8
        assert cfg.num_ctx == 2048

    def test_api_url(self):
        cfg = LLMConfig()
        assert cfg.api_url("/api/tags") == "http://localhost:11434/api/tags"
        # Trailing slash should be handled
        cfg2 = LLMConfig(ollama_base_url="http://localhost:11434/")
        assert cfg2.api_url("/api/tags") == "http://localhost:11434/api/tags"


# ── safe_truncate ────────────────────────────────────────────────────────

class TestSafeTruncate:
    def test_short_text_unchanged(self):
        text = "Hello, world!"
        assert safe_truncate(text, 100) == text

    def test_exact_limit(self):
        text = "x" * 100
        assert safe_truncate(text, 100) == text

    def test_truncation(self):
        text = "x" * 200
        result = safe_truncate(text, 100)
        assert len(result) <= 100
        assert "[truncated" in result

    def test_default_limit(self):
        text = "a" * 10000
        result = safe_truncate(text)
        assert len(result) <= 8000


# ── format_bullets ───────────────────────────────────────────────────────

class TestFormatBullets:
    def test_list_input(self):
        result = format_bullets(["item A", "item B", "item C"])
        assert result == "- item A\n- item B\n- item C"

    def test_string_input(self):
        result = format_bullets("line 1\nline 2\nline 3")
        assert result == "- line 1\n- line 2\n- line 3"

    def test_existing_bullets_stripped(self):
        result = format_bullets(["- already bulleted", "• dot bullet", "* star bullet"])
        assert result == "- already bulleted\n- dot bullet\n- star bullet"

    def test_empty_input(self):
        assert "(none provided)" in format_bullets("")
        assert "(none provided)" in format_bullets([])

    def test_mixed_empty_lines(self):
        result = format_bullets("line 1\n\n\nline 2\n")
        assert result == "- line 1\n- line 2"


# ── df_to_compact_text ───────────────────────────────────────────────────

class TestDfToCompactText:
    def test_basic_dataframe(self):
        df = pd.DataFrame({"MPD": [1.0, 1.5, 2.0], "Ra": [10, 20, 30]})
        text = df_to_compact_text(df)
        assert "3 rows" in text
        assert "MPD" in text
        assert "Ra" in text
        assert "Descriptive Statistics" in text

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        text = df_to_compact_text(df)
        assert "empty" in text.lower()

    def test_column_limit(self):
        import numpy as np
        df = pd.DataFrame(np.random.rand(5, 20))
        text = df_to_compact_text(df, max_cols=5)
        assert "first 5 of 20" in text

    def test_row_limit(self):
        df = pd.DataFrame({"x": range(100)})
        text = df_to_compact_text(df, max_rows=10)
        assert "10 rows" in text


# ── Import smoke test ────────────────────────────────────────────────────

class TestImports:
    def test_public_api_importable(self):
        from texturelab_llm import (
            LLMConfig,
            check_ollama_health,
            list_models,
            generate_project_draft,
            interpret_results,
            chat_turn,
        )
        # All should be callable
        assert callable(check_ollama_health)
        assert callable(list_models)
        assert callable(generate_project_draft)
        assert callable(interpret_results)
        assert callable(chat_turn)
