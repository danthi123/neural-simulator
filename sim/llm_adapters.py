"""LLM adapters for LLMMemoryOrchestrator (Path 3 Phase 3.3 scaffold).

Each adapter conforms to the llm_callable signature:

    def my_adapter(conversation: list[dict]) -> LLMResponse:
        ...

Adapters wrap an external LLM (ollama / vLLM / llama.cpp / OpenAI /
Anthropic / etc.) and return LLMResponse(message=..., tool_calls=[...]).
The orchestrator drives the tool-use loop independent of which adapter
is plugged in.

Shipping with one adapter today: OllamaLLM. Phase 3.3 will add others
as the user picks LLMs.

Design: docs/plans/2026-05-11-path3-phase3.3-real-llm-design.md
"""
from __future__ import annotations

import json
from typing import Optional

from sim.llm_memory_orchestrator import LLMResponse, ToolCall, TOOL_SCHEMAS


# ──────────────────────────────────────────────────────────────────────
# Ollama adapter — works with any model served by ollama (Phi-3, Llama,
# Qwen, etc.). Uses ollama's OpenAI-compatible /v1/chat/completions API
# so we can use the standard `openai` Python client.
# ──────────────────────────────────────────────────────────────────────


DEFAULT_SYSTEM_PROMPT = (
    "You are a memory assistant. You have access to five tools that "
    "interact with a biology-grounded memory subsystem:\n"
    "- memory_store(key, value): bind a fact. value must be a direction "
    "word (north, east, south, west, up, down, left, right).\n"
    "- memory_recall(key): retrieve the direction bound to a key.\n"
    "- memory_speak(action): get the word bound to a motor pool (action "
    "is N, E, S, or W).\n"
    "- memory_forget(key, decay_rate): weaken/erase a bound fact. "
    "decay_rate=0.0 fully erases; 0.5 halves; 1.0 is no-op.\n"
    "- memory_consolidate(n_sleep_cycles): run sleep-replay consolidation "
    "(only effective on hippocampus-enabled bridges).\n\n"
    "When the user mentions remembering or storing a fact, use "
    "memory_store. When they ask what they previously stored, use "
    "memory_recall. When they ask what word goes with a direction, use "
    "memory_speak. When they want to forget, use memory_forget. When "
    "they say 'sleep on it' or want consolidation, use "
    "memory_consolidate. Always respond in plain English after consulting "
    "the memory."
)


class OllamaLLM:
    """Adapter for ollama (or any OpenAI-compatible endpoint).

    Requires:
        pip install openai      # the openai client (>=1.0)
        ollama serve            # ollama daemon running
        ollama pull llama3.2:3b  # any model you want

    Args:
        base_url: ollama API endpoint. Default 'http://localhost:11434/v1'.
        model: ollama model name. Default 'llama3.2:3b'. Try also:
            'phi3:3.8b', 'qwen2.5:3b'.
        system_prompt: prefix prompt. Defaults to a memory-assistant
            persona biased toward the five tools.
        api_key: any string works for ollama (ignored). For real OpenAI,
            pass your actual key.
        timeout_s: per-call timeout. Default 30s.
        verbose: print debug info.

    Usage:
        from sim.llm_adapters import OllamaLLM
        from sim.llm_memory_orchestrator import LLMMemoryOrchestrator
        from sim.bridge_memory import BridgeMemory

        mem = BridgeMemory(lineage_name="main", mode="synonym")
        llm = OllamaLLM(model="llama3.2:3b")
        orch = LLMMemoryOrchestrator(memory=mem, llm_callable=llm)
        print(orch.chat("Remember that my favorite is north."))
    """

    def __init__(self,
                  base_url: str = "http://localhost:11434/v1",
                  model: str = "llama3.2:3b",
                  system_prompt: Optional[str] = None,
                  api_key: str = "ollama",
                  timeout_s: float = 30.0,
                  verbose: bool = False):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OllamaLLM requires the openai package: "
                "`pip install openai`"
            ) from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout_s = float(timeout_s)
        self.verbose = bool(verbose)

    def __call__(self, conversation: list[dict]) -> LLMResponse:
        """Process the next conversation turn via ollama.

        Args:
            conversation: list of {"role", "content"} dicts. Tool turns
                have "name" + "content"=result dict.

        Returns:
            LLMResponse with .message and/or .tool_calls.
        """
        # Translate the conversation into OpenAI-compatible messages.
        messages = [{"role": "system", "content": self.system_prompt}]
        last_tool_call_id = 0
        for m in conversation:
            role = m.get("role", "user")
            if role == "tool":
                # OpenAI's tool-result format uses tool_call_id linkage;
                # ollama is permissive. We flatten to a text "tool result"
                # turn that the model treats as informational.
                last_tool_call_id += 1
                tool_name = m.get("name", "")
                content = m.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, default=str)[:1000]
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{last_tool_call_id}",
                    "content": f"[{tool_name} result] {content}",
                })
            else:
                messages.append({
                    "role": role,
                    "content": m.get("content", ""),
                })

        # Wrap TOOL_SCHEMAS into OpenAI tools format.
        tools = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]

        if self.verbose:
            print(f"[OllamaLLM] sending {len(messages)} messages to "
                  f"{self.model}", flush=True)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=self.timeout_s,
            )
        except Exception as e:
            # Surface the error so MockLLM-style fallback works
            return LLMResponse(
                message=(f"Sorry, I hit an LLM error: {type(e).__name__}: "
                          f"{str(e)[:200]}")
            )

        choice = resp.choices[0].message

        # Parse tool calls if any
        tool_calls = []
        raw_tool_calls = getattr(choice, "tool_calls", None) or []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool_calls.append(ToolCall(
                name=tc.function.name,
                arguments=args,
            ))

        message = choice.content or ""
        return LLMResponse(message=message, tool_calls=tool_calls)


# ──────────────────────────────────────────────────────────────────────
# Future adapters (sketches)
# ──────────────────────────────────────────────────────────────────────


class LlamaCppLLM:
    """Adapter for llama.cpp + llama-cpp-python (CPU-friendly).

    Phase 3.3 sketch — wire when needed. Requires:
        pip install llama-cpp-python

    Loads a GGUF model directly without an external server. Faster CPU
    inference than ollama for some configurations.
    """

    def __init__(self, model_path: str, **kwargs):
        raise NotImplementedError(
            "LlamaCppLLM is a Phase 3.3 stub. Wire when the user picks "
            "llama.cpp as the serving backend. See "
            "docs/plans/2026-05-11-path3-phase3.3-real-llm-design.md"
        )

    def __call__(self, conversation: list[dict]) -> LLMResponse:
        raise NotImplementedError
