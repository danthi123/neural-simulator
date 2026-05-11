"""Tests for sim.llm_memory_orchestrator — Phase 3.2 scaffold.

Tests the tool-use loop with MockLLM + a mock memory. No bridge or
real LLM required.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.llm_memory_orchestrator import (
    LLMMemoryOrchestrator,
    MockLLM,
    ToolCall,
    LLMResponse,
    TOOL_SCHEMAS,
)


# ──────────────────────────────────────────────────────────────────────
# Mock memory — simulates BridgeMemory interface without a bridge
# ──────────────────────────────────────────────────────────────────────


class _MockMemory:
    """Simulates BridgeMemory's store/recall/speak with in-memory dict."""

    def __init__(self):
        self.bindings = {}  # key -> value
        self.calls = {"store": 0, "recall": 0, "speak": 0}

    def store(self, key, value, **kw):
        self.calls["store"] += 1
        self.bindings[key] = value
        return {"key": key, "value": value,
                "target_action": value[0].upper(),
                "confidence": 1.5, "bound_correctly": True,
                "n_events_run": 50}

    def recall(self, key, top_k=4, **kw):
        self.calls["recall"] += 1
        # Stub: if key bound, return high-delta for that direction;
        # else return all-zero deltas.
        if key in self.bindings:
            value = self.bindings[key]
            action_for = {"north": "N", "east": "E", "south": "S",
                          "west": "W"}.get(value, "N")
            value_for = {"N": "north", "E": "east", "S": "south",
                          "W": "west"}
            results = []
            for rank, action in enumerate(["N", "E", "S", "W"], 1):
                conf = 1.0 if action == action_for else 0.1
                results.append({
                    "action": action,
                    "value": value_for[action],
                    "confidence": conf,
                    "rank": rank,
                    "raw_delta": 100 if action == action_for else 5,
                })
            # Sort by raw_delta desc
            results.sort(key=lambda r: -r["raw_delta"])
            for i, r in enumerate(results, 1):
                r["rank"] = i
            return results[:top_k]
        return [{"action": "N", "value": "north", "confidence": 0.0,
                  "rank": 1, "raw_delta": 0}]

    def speak(self, action, top_k=4, **kw):
        self.calls["speak"] += 1
        # Stub: find binding with matching value
        action_dir = {"N": "north", "E": "east", "S": "south",
                       "W": "west"}.get(action, "north")
        for key, value in self.bindings.items():
            if value == action_dir:
                return [{"word": key, "similarity": 0.85, "rank": 1}]
        return [{"word": action_dir, "similarity": 0.5, "rank": 1}]


# ──────────────────────────────────────────────────────────────────────
# Tool schema
# ──────────────────────────────────────────────────────────────────────


def test_tool_schemas_well_formed():
    """TOOL_SCHEMAS is a list of valid OpenAI-style tool definitions."""
    assert len(TOOL_SCHEMAS) == 5  # store, recall, speak, forget, consolidate
    names = {s["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "memory_store", "memory_recall", "memory_speak",
        "memory_forget", "memory_consolidate",
    }
    for s in TOOL_SCHEMAS:
        assert "description" in s
        assert "parameters" in s
        assert s["parameters"]["type"] == "object"
        assert "properties" in s["parameters"]
        assert "required" in s["parameters"]


# ──────────────────────────────────────────────────────────────────────
# MockLLM pattern recognition
# ──────────────────────────────────────────────────────────────────────


def test_mock_llm_recognizes_store_pattern():
    """'remember that my X is direction' → memory_store call."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "Remember that my favorite is north."}
    ])
    assert response.tool_calls
    tc = response.tool_calls[0]
    assert tc.name == "memory_store"
    assert "favorite" in tc.arguments["key"]
    assert tc.arguments["value"] == "north"


def test_mock_llm_recognizes_recall_pattern():
    """'what's my X' → memory_recall call."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "What's my favorite color?"}
    ])
    assert response.tool_calls
    assert response.tool_calls[0].name == "memory_recall"


def test_mock_llm_recognizes_speak_pattern():
    """'what word goes with north' → memory_speak call."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "What word goes with north?"}
    ])
    assert response.tool_calls
    assert response.tool_calls[0].name == "memory_speak"
    assert response.tool_calls[0].arguments["action"] == "N"


def test_mock_llm_fallback_message():
    """Unrecognized input gets a fallback message (no tool calls)."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "What's the weather?"}
    ])
    assert not response.tool_calls
    assert "don't understand" in response.message.lower()


def test_mock_llm_handles_synonym_direction():
    """'up' in store/speak gets mapped to 'north'."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "Remember that my favorite is up."}
    ])
    assert response.tool_calls
    assert response.tool_calls[0].arguments["value"] == "north"


def test_mock_llm_recognizes_forget_pattern():
    """'forget my X' -> memory_forget call with decay_rate=0.5."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "Forget my favorite."}
    ])
    assert response.tool_calls
    tc = response.tool_calls[0]
    assert tc.name == "memory_forget"
    assert "favorite" in tc.arguments["key"]
    assert tc.arguments["decay_rate"] == 0.5


def test_mock_llm_recognizes_full_erase_pattern():
    """'fully forget' or 'erase' -> decay_rate=0.0."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "Fully forget my favorite."}
    ])
    assert response.tool_calls
    assert response.tool_calls[0].name == "memory_forget"
    assert response.tool_calls[0].arguments["decay_rate"] == 0.0


def test_mock_llm_recognizes_consolidate_pattern():
    """'consolidate memory' / 'sleep on it' -> memory_consolidate."""
    llm = MockLLM()
    for msg in ("Please consolidate the memory.",
                  "Sleep on it.",
                  "Time to sleep on this."):
        response = llm([{"role": "user", "content": msg}])
        assert response.tool_calls, f"no tool call for: {msg!r}"
        assert response.tool_calls[0].name == "memory_consolidate"


def test_mock_llm_extracts_n_cycles_from_consolidate():
    """'sleep for 5 cycles' -> n_sleep_cycles=5."""
    llm = MockLLM()
    response = llm([
        {"role": "user", "content": "Consolidate for 5 cycles."}
    ])
    assert response.tool_calls
    assert response.tool_calls[0].arguments["n_sleep_cycles"] == 5


# ──────────────────────────────────────────────────────────────────────
# Orchestrator end-to-end
# ──────────────────────────────────────────────────────────────────────


def test_orchestrator_chat_no_tool_calls():
    """Unrecognized input returns the fallback message directly."""
    mem = _MockMemory()
    orch = LLMMemoryOrchestrator(memory=mem)
    response = orch.chat("How tall is the Eiffel Tower?")
    assert "don't understand" in response.lower()
    assert mem.calls == {"store": 0, "recall": 0, "speak": 0}


def test_orchestrator_chat_store_then_recall():
    """End-to-end: store a fact, then recall it."""
    mem = _MockMemory()
    orch = LLMMemoryOrchestrator(memory=mem)

    # Turn 1: store
    response = orch.chat("Remember that my favorite is north.")
    assert "remember" in response.lower() or "got it" in response.lower()
    assert mem.calls["store"] == 1
    # The MockLLM regex strips the leading "my" — key is just "favorite"
    assert "favorite" in mem.bindings

    # Turn 2: recall
    response = orch.chat("What's my favorite?")
    assert "north" in response.lower()
    assert mem.calls["recall"] == 1


def test_orchestrator_chat_recall_unknown_key():
    """Recall on unbound key returns a helpful message."""
    mem = _MockMemory()
    orch = LLMMemoryOrchestrator(memory=mem)
    response = orch.chat("What's my best friend's name?")
    assert "don't remember" in response.lower() or "tell me" in response.lower()
    assert mem.calls["recall"] == 1


def test_orchestrator_speak_after_store():
    """Speak after binding returns the bound word."""
    mem = _MockMemory()
    orch = LLMMemoryOrchestrator(memory=mem)
    orch.chat("Remember that my pet is east.")
    response = orch.chat("What word goes with east?")
    assert "pet" in response.lower() or "east" in response.lower()
    assert mem.calls["speak"] == 1


def test_orchestrator_conversation_persists():
    """The conversation list grows across turns."""
    mem = _MockMemory()
    orch = LLMMemoryOrchestrator(memory=mem)
    assert len(orch.conversation) == 0
    orch.chat("Remember that my favorite is north.")
    # 1 user + 1 tool + 1 assistant = 3 messages
    assert len(orch.conversation) == 3
    orch.chat("What's my favorite?")
    # Now 1+1+1 + 1+1+1 = 6 messages
    assert len(orch.conversation) == 6


def test_orchestrator_max_iterations_safety_cap():
    """If LLM keeps calling tools, orchestrator caps iterations."""
    mem = _MockMemory()

    class _AlwaysToolCallLLM:
        """Pathological LLM that never returns a final message."""
        def __call__(self, conv):
            return LLMResponse(
                tool_calls=[ToolCall(name="memory_recall",
                                       arguments={"key": "anything"})]
            )

    orch = LLMMemoryOrchestrator(
        memory=mem, llm_callable=_AlwaysToolCallLLM(),
        max_tool_iterations=3,
    )
    response = orch.chat("Hello")
    assert "max_tool_iterations" in response


def test_orchestrator_dispatch_unknown_tool_returns_error():
    """An LLM that calls an unknown tool gets an error result."""
    mem = _MockMemory()

    class _BadToolLLM:
        call_count = 0
        def __call__(self, conv):
            self.call_count += 1
            if self.call_count == 1:
                return LLMResponse(tool_calls=[ToolCall(
                    name="totally_made_up", arguments={}
                )])
            # Second call: return final message
            return LLMResponse(message="Done.")

    orch = LLMMemoryOrchestrator(memory=mem, llm_callable=_BadToolLLM())
    response = orch.chat("test")
    # The bad tool call's result should be in conversation as an error
    tool_turn = next(t for t in orch.conversation if t["role"] == "tool")
    assert "error" in tool_turn["content"]
    assert response == "Done."


def test_mock_llm_propagates_tool_error_to_user():
    """When the dispatched tool errors, MockLLM surfaces the error
    instead of a misleading 'Got it' message."""
    llm = MockLLM()
    # Simulate the conversation: user message + a tool result that
    # carries an "error" key (as bridge_memory.store would when its
    # underlying ops fail).
    conv = [
        {"role": "user", "content": "Remember that my name is north."},
        {"role": "tool", "name": "memory_store",
         "content": {"error": "vocab full"}},
    ]
    response = llm(conv)
    msg = response.message.lower()
    # Must mention the error or apologize — not say "got it"
    assert "error" in msg or "sorry" in msg
    assert "got it" not in msg
    assert "vocab full" in response.message
