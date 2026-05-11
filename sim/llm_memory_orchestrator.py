"""LLM-Memory Orchestrator — Phase 3.2 scaffold for Path 3.

Drives the tool-use loop between an LLM and a BridgeMemory. The LLM
emits either a final user-facing message OR a tool_call to invoke
memory_store / memory_recall / memory_speak. The orchestrator
dispatches the call, feeds the result back, and continues until the
LLM produces a final message.

Phase 3.2 scaffold: uses a MOCK LLM by default (no external dependency).
Real LLM integration (Phi-3-mini via ollama, Llama 3.2 via vLLM,
Qwen2.5 via llama.cpp, etc.) plugs in via the `llm_callable` argument.

Design doc: docs/plans/2026-05-11-path3-bridge-memory-api-design.md

Usage:

    from sim.bridge_memory import BridgeMemory
    from sim.llm_memory_orchestrator import (
        LLMMemoryOrchestrator, MockLLM,
    )

    mem = BridgeMemory(lineage_name="alice", mode="synonym")
    llm = MockLLM()  # or a real LLM callable
    orch = LLMMemoryOrchestrator(memory=mem, llm_callable=llm)

    response = orch.chat("Hi! I'm Alice. Remember that my favorite direction is north.")
    # → "Hi Alice! I'll remember that your favorite direction is north."

    response = orch.chat("What's my favorite direction?")
    # → "Your favorite direction is north."

The MockLLM hand-codes a few recognized patterns for the demo; a real
LLM would do this via natural language understanding.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ──────────────────────────────────────────────────────────────────────
# Tool-use schema (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────


TOOL_SCHEMAS = [
    {
        "name": "memory_store",
        "description": ("Save a key-value fact in the brain-shaped memory. "
                        "The value must be a direction word: north, east, "
                        "south, west (or one of their synonyms in the "
                        "active vocab)."),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string",
                          "description": "The cue (e.g. user's name)"},
                "value": {"type": "string",
                            "description": "The direction word to bind"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_recall",
        "description": ("Retrieve the direction associated with a key. "
                        "Returns ranked candidates with confidence scores."),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string",
                          "description": "The cue to recall"},
                "top_k": {"type": "integer", "default": 4,
                            "description": "How many candidates to return"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "memory_speak",
        "description": ("Generate a word for a given motor direction. "
                        "Useful for verifying what word is bound to a "
                        "specific direction."),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                             "enum": ["N", "E", "S", "W"],
                             "description": "Motor pool to activate"},
                "top_k": {"type": "integer", "default": 4},
            },
            "required": ["action"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────
# LLM response schema
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A tool invocation request from the LLM."""
    name: str  # "memory_store" / "memory_recall" / "memory_speak"
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """One LLM turn's output: either a final message OR a tool call.

    If tool_calls is non-empty, the orchestrator executes them all and
    feeds the results back via a follow-up llm_callable invocation.

    If tool_calls is empty AND message is non-empty, the response is
    final and gets returned to the user.
    """
    message: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# MockLLM — hand-coded patterns for the demo
# ──────────────────────────────────────────────────────────────────────


class MockLLM:
    """A deterministic mock LLM that recognizes a few patterns.

    Not intended for production use — just demonstrates the tool-use
    protocol end-to-end so the orchestrator can be tested without
    a real LLM.

    Recognized patterns:
    - "remember that <key> is <direction>" → memory_store(key, direction)
    - "my <key> is <direction>" → memory_store(key, direction)
    - "what is my <key>" / "what's my <key>" → memory_recall(key)
    - "what word goes with <direction>" → memory_speak(action_for_direction)
    - everything else → "I don't understand. Try teaching me a fact."
    """

    DIRECTION_KEYWORDS = {
        "north": "north", "south": "south", "east": "east", "west": "west",
        "up": "north", "right": "east", "down": "south", "left": "west",
        "n": "north", "s": "south", "e": "east", "w": "west",
    }

    def __init__(self):
        self.transcript: list[dict] = []

    def _find_direction(self, text: str) -> Optional[str]:
        """Find a direction word in `text`. Returns canonical form or None."""
        for word in re.findall(r"\b\w+\b", text.lower()):
            if word in self.DIRECTION_KEYWORDS:
                return self.DIRECTION_KEYWORDS[word]
        return None

    def __call__(self, conversation: list[dict]) -> LLMResponse:
        """Process the next turn.

        Args:
            conversation: list of {"role": str, "content": str | dict} dicts.
                Roles: "user", "assistant", "tool".

        Returns:
            LLMResponse with either a message OR tool_calls.
        """
        # Get the most recent user message
        last_user = ""
        for turn in reversed(conversation):
            if turn["role"] == "user":
                last_user = turn["content"].lower()
                break

        # If the previous turn was a tool result, format a final response
        last_turn = conversation[-1] if conversation else {}
        if last_turn.get("role") == "tool":
            tool_name = last_turn.get("name", "")
            result = last_turn.get("content", {})
            if tool_name == "memory_recall":
                # result is a list of {action, value, confidence, rank, raw_delta}
                if result and len(result) > 0 and result[0].get("raw_delta", 0) > 0:
                    return LLMResponse(
                        message=f"Your answer is {result[0]['value']}."
                    )
                else:
                    return LLMResponse(
                        message="I don't remember that fact yet. "
                                 "Tell me, and I'll remember next time."
                    )
            if tool_name == "memory_store":
                return LLMResponse(
                    message=f"Got it. I'll remember: "
                              f"{result.get('key', '')} = "
                              f"{result.get('value', '')}."
                )
            if tool_name == "memory_speak":
                if result and len(result) > 0:
                    return LLMResponse(
                        message=f"The word for that direction is "
                                  f"'{result[0]['word']}'."
                    )
                return LLMResponse(
                    message="No clear word found for that direction."
                )

        # Otherwise, pattern-match on the user message
        # Pattern: "remember that <key> is <direction>"
        m = re.search(
            r"(?:remember|my)\s+(?:that\s+)?(?:my\s+)?(.+?)\s+(?:is|=|:)\s+(.+)",
            last_user,
        )
        if m:
            key = m.group(1).strip()
            value_text = m.group(2).strip()
            direction = self._find_direction(value_text)
            if direction:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(name="memory_store",
                                  arguments={"key": key, "value": direction})
                    ]
                )

        # Pattern: "what (is|was) my <key>"
        m = re.search(r"what(?:'s| is| was)\s+my\s+(.+?)[\?\.]*$", last_user)
        if m:
            key = m.group(1).strip()
            return LLMResponse(
                tool_calls=[
                    ToolCall(name="memory_recall",
                              arguments={"key": key, "top_k": 4})
                ]
            )

        # Pattern: "what word goes with <direction>"
        if "what word" in last_user or "speak" in last_user:
            direction = self._find_direction(last_user)
            if direction:
                action_map = {"north": "N", "east": "E", "south": "S", "west": "W"}
                return LLMResponse(
                    tool_calls=[
                        ToolCall(name="memory_speak",
                                  arguments={"action": action_map[direction]})
                    ]
                )

        # Fallback
        return LLMResponse(
            message=("I don't understand. Try: "
                     "'Remember that my favorite direction is north', then "
                     "'What is my favorite direction?'")
        )


# ──────────────────────────────────────────────────────────────────────
# LLMMemoryOrchestrator — the tool-use loop
# ──────────────────────────────────────────────────────────────────────


class LLMMemoryOrchestrator:
    """Drives the tool-use loop between an LLM and a BridgeMemory.

    Each chat() call:
    1. Adds the user's message to the conversation
    2. Asks the LLM for the next response
    3. If the response is a tool call, dispatches it against memory
       and feeds the result back as a "tool" role turn
    4. Repeats until the LLM produces a final message
    5. Returns the final message to the caller

    Args:
        memory: a BridgeMemory instance
        llm_callable: callable that takes a conversation list and
            returns an LLMResponse. Default: MockLLM.
        max_tool_iterations: safety cap on tool-call loops per turn
            (prevents the LLM from infinite-looping on tools)
    """

    def __init__(self,
                 memory,
                 llm_callable: Optional[Callable] = None,
                 max_tool_iterations: int = 5):
        self.memory = memory
        self.llm = llm_callable or MockLLM()
        self.max_tool_iterations = int(max_tool_iterations)
        self.conversation: list[dict] = []

    def chat(self, user_input: str) -> str:
        """Process one user turn. Returns the LLM's final message."""
        self.conversation.append({"role": "user", "content": user_input})

        for iteration in range(self.max_tool_iterations + 1):
            response = self.llm(self.conversation)
            if response.tool_calls:
                # Dispatch each tool call against memory
                for tc in response.tool_calls:
                    result = self._dispatch_tool(tc)
                    self.conversation.append({
                        "role": "tool",
                        "name": tc.name,
                        "content": result,
                    })
                # Loop again so LLM can respond to the tool result
                continue
            # No tool calls = final message
            self.conversation.append({
                "role": "assistant", "content": response.message,
            })
            return response.message

        # Hit the iteration cap — return a stub message
        msg = (f"(orchestrator: reached max_tool_iterations="
                f"{self.max_tool_iterations}; aborting)")
        self.conversation.append({"role": "assistant", "content": msg})
        return msg

    def _dispatch_tool(self, tc: ToolCall) -> Any:
        """Execute a tool call against the BridgeMemory."""
        args = tc.arguments
        try:
            if tc.name == "memory_store":
                return self.memory.store(
                    key=args["key"], value=args["value"]
                )
            elif tc.name == "memory_recall":
                return self.memory.recall(
                    key=args["key"], top_k=args.get("top_k", 4)
                )
            elif tc.name == "memory_speak":
                return self.memory.speak(
                    action=args["action"], top_k=args.get("top_k", 4)
                )
            else:
                return {"error": f"unknown tool: {tc.name}"}
        except Exception as e:
            return {"error": str(e)}
