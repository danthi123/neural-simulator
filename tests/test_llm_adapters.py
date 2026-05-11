"""Tests for sim.llm_adapters — Phase 3.3 scaffolding.

These tests exercise the adapter interface without requiring a real
ollama server. Live integration tests come in Phase 3.3 once the user
picks an LLM.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_llm_adapters_imports():
    """sim.llm_adapters imports cleanly + exposes OllamaLLM class."""
    from sim import llm_adapters
    assert hasattr(llm_adapters, "OllamaLLM")
    assert hasattr(llm_adapters, "LlamaCppLLM")
    assert hasattr(llm_adapters, "DEFAULT_SYSTEM_PROMPT")


def test_default_system_prompt_mentions_all_tools():
    """The default system prompt covers all 5 tool schemas so the LLM
    knows what's available."""
    from sim.llm_adapters import DEFAULT_SYSTEM_PROMPT
    for tool in ("memory_store", "memory_recall", "memory_speak",
                 "memory_forget", "memory_consolidate"):
        assert tool in DEFAULT_SYSTEM_PROMPT, (
            f"system prompt missing mention of {tool}"
        )


def test_llama_cpp_adapter_stub():
    """LlamaCppLLM raises NotImplementedError (Phase 3.3 placeholder)."""
    from sim.llm_adapters import LlamaCppLLM
    with pytest.raises(NotImplementedError):
        LlamaCppLLM(model_path="fake.gguf")


def test_ollama_adapter_requires_openai_package(monkeypatch):
    """Constructor raises ImportError when openai isn't installed."""
    # Mock the import failure
    import sys
    orig_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("openai not installed (mocked)")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    # Drop cached module if present
    sys.modules.pop("openai", None)

    from sim.llm_adapters import OllamaLLM
    with pytest.raises(ImportError, match="openai"):
        OllamaLLM()


def test_ollama_adapter_handles_connection_failure():
    """If openai is installed but ollama isn't running, OllamaLLM should
    return an error LLMResponse rather than raising."""
    pytest.importorskip("openai")
    from sim.llm_adapters import OllamaLLM
    from sim.llm_memory_orchestrator import LLMResponse

    # Point at a definitely-dead address to force connection failure
    llm = OllamaLLM(base_url="http://localhost:1/v1", timeout_s=2.0)
    conv = [{"role": "user", "content": "hello"}]
    resp = llm(conv)
    assert isinstance(resp, LLMResponse)
    # Either error message OR (rarely, if a server happens to listen)
    # a real response. Errors look like "Sorry, I hit an LLM error..."
    assert resp.message  # non-empty either way
