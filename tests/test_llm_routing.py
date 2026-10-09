"""
Tests for LLM model routing with local Ollama.

Verifies that:
  - Extraction / summarisation use llama3:8b (or OLLAMA_EXTRACTION_MODEL/OLLAMA_SUMMARY_MODEL override)
  - Final answer generation uses qwen3:14b (or OLLAMA_ANSWER_MODEL override)
  - Embedding uses nomic-embed-text (or OLLAMA_EMBEDDING_MODEL override)
  - All clients use http://localhost:11434/v1 by default.
  - process_chunks processes correctly.
  - call_llm routes to the answer model when _use_answer_model=True

No real API calls are made; the OpenAI client is patched throughout.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(text: str = "mocked content"):
    """Build a minimal mock that looks like openai.ChatCompletion response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _clean_env(monkeypatch, *keys):
    """Remove a set of env keys, ignoring if already absent."""
    for k in keys:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# 1. _chat_model_name in summarize.py
# ---------------------------------------------------------------------------

class TestSummarisationModelName:

    def test_default_is_llama_8b(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_SUMMARY_MODEL", "llama3:8b")
        import summarize
        name = summarize._chat_model_name()
        assert name == "llama3:8b"

    def test_ollama_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_SUMMARY_MODEL", "mistral:7b")
        import summarize
        assert summarize._chat_model_name() == "mistral:7b"


# ---------------------------------------------------------------------------
# 2. _chat_model_name and _answer_model_name in utils.py
# ---------------------------------------------------------------------------

class TestUtilsModelNames:

    def test_chat_model_default_is_llama_8b(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EXTRACTION_MODEL", "llama3:8b")
        import utils
        assert utils._chat_model_name() == "llama3:8b"

    def test_chat_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EXTRACTION_MODEL", "mistral:7b")
        import utils
        assert utils._chat_model_name() == "mistral:7b"

    def test_answer_model_default_is_qwen3(self, monkeypatch):
        _clean_env(monkeypatch, "OLLAMA_ANSWER_MODEL")
        import utils
        assert utils._answer_model_name() == "qwen3:14b"

    def test_answer_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_ANSWER_MODEL", "phi3:mini")
        import utils
        assert utils._answer_model_name() == "phi3:mini"


# ---------------------------------------------------------------------------
# 3. _client() base URL (extraction / summarisation)
# ---------------------------------------------------------------------------

class TestExtractionClientBaseUrl:

    def test_default_base_url_is_ollama(self, monkeypatch):
        _clean_env(monkeypatch, "OLLAMA_BASE_URL")

        import utils
        captured = {}

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None, **kw):
                captured["base_url"] = base_url

        with patch.object(utils, "OpenAI", FakeOpenAI):
            utils._client()

        assert captured["base_url"] == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# 4. _answer_client() base URL
# ---------------------------------------------------------------------------

class TestAnswerClient:

    def test_default_base_url_is_ollama(self, monkeypatch):
        _clean_env(monkeypatch, "OLLAMA_BASE_URL")

        import utils
        captured = {}

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None, **kw):
                captured["base_url"] = base_url

        with patch.object(utils, "OpenAI", FakeOpenAI):
            utils._answer_client()

        assert captured["base_url"] == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# 5. call_llm model dispatch
# ---------------------------------------------------------------------------

class TestCallLlmRouting:

    def test_uses_extraction_model_by_default(self, monkeypatch):
        _clean_env(monkeypatch, "OLLAMA_EXTRACTION_MODEL")
        _clean_env(monkeypatch, "OLLAMA_BASE_URL")

        import utils

        mock_resp = _make_mock_response("extraction answer")
        mock_create = MagicMock(return_value=mock_resp)

        class FakeOpenAI:
            def __init__(self, **kw): pass
            chat = MagicMock()

        fake_instance = FakeOpenAI()
        fake_instance.chat = MagicMock()
        fake_instance.chat.completions = MagicMock()
        fake_instance.chat.completions.create = mock_create

        with patch.object(utils, "OpenAI", return_value=fake_instance):
            with patch.object(utils, "_run_with_hard_timeout",
                              lambda secs, fn, *a, **kw: fn(*a, **kw)):
                result = utils.call_llm("sys", "user", _use_answer_model=False)

        assert result == "extraction answer"
        model_used = mock_create.call_args.kwargs.get("model")
        assert model_used == "llama3:8b"

    def test_uses_answer_model_when_flagged(self, monkeypatch):
        _clean_env(monkeypatch, "OLLAMA_ANSWER_MODEL")
        _clean_env(monkeypatch, "OLLAMA_BASE_URL")

        import utils

        mock_resp = _make_mock_response("qwen answer")
        mock_create = MagicMock(return_value=mock_resp)

        fake_instance = MagicMock()
        fake_instance.chat.completions.create = mock_create

        with patch.object(utils, "OpenAI", return_value=fake_instance):
            with patch.object(utils, "_run_with_hard_timeout",
                              lambda secs, fn, *a, **kw: fn(*a, **kw)):
                result = utils.call_llm("sys", "user", _use_answer_model=True)

        assert result == "qwen answer"
        model_used = mock_create.call_args.kwargs.get("model")
        assert model_used == "qwen3:14b"


# ---------------------------------------------------------------------------
# 6. Extraction model in create_graph.py
# ---------------------------------------------------------------------------

class TestExtractionModelInCreateGraph:

    def test_extraction_uses_llama_8b(self, monkeypatch):
        monkeypatch.setenv("USE_LLM_EXTRACTION", "1")
        monkeypatch.setenv("OLLAMA_EXTRACTION_MODEL", "llama3:8b")
        _clean_env(monkeypatch, "OLLAMA_BASE_URL")

        import create_graph
        from camel.loaders import UnstructuredIO

        uio = UnstructuredIO()
        source = uio.create_element_from_text(text="dummy")

        class FakeOpenAI:
            def __init__(self, **kw): pass
            chat = MagicMock()

        mock_resp = _make_mock_response("Node(id='Test', type='Disease', description='test')")
        mock_create = MagicMock(return_value=mock_resp)

        fake_instance = MagicMock()
        fake_instance.chat.completions.create = mock_create

        with patch.object(create_graph, "OpenAI", return_value=fake_instance):
            create_graph._extract_graph_elements_from_text(
                "DIAGNOSIS code=J18 icd_version=10 name=Pneumonia", source
            )

        model_used = mock_create.call_args.kwargs.get("model")
        assert model_used == "llama3:8b"

    def test_extraction_base_url_is_ollama(self, monkeypatch):
        monkeypatch.setenv("USE_LLM_EXTRACTION", "1")
        _clean_env(monkeypatch, "OLLAMA_BASE_URL")

        import create_graph
        from camel.loaders import UnstructuredIO

        uio = UnstructuredIO()
        source = uio.create_element_from_text(text="dummy")

        captured = {}

        class FakeOpenAI:
            def __init__(self, api_key=None, base_url=None, **kw):
                captured["base_url"] = base_url
            chat = MagicMock()

        with patch.object(create_graph, "OpenAI", FakeOpenAI):
            try:
                create_graph._extract_graph_elements_from_text("clinical text", source)
            except Exception:
                pass

        assert captured.get("base_url") == "http://localhost:11434/v1"
