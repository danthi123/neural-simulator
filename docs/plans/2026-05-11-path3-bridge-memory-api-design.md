---
type: plan
status: live
date: 2026-05-11
---

# Path 3 design — BridgeMemory API (LLM-callable memory subsystem)

**Date:** 2026-05-11 05:25 EDT
**Status:** DESIGN — explores Path 3 from the strategic re-evaluation
(LLM + biology-inspired memory subsystem). Spells out the
`BridgeMemory` abstraction that a locally-runnable LLM would query
via tool-use.
**Strategic context:** [`docs/plans/2026-05-11-strategic-reevaluation.md`](2026-05-11-strategic-reevaluation.md)
**Prereqs:**
- Bridge Lineage Manager (✅ shipped)
- chat_repl with binding workflow (✅ shipped)
- NumPy backend (✅ shipped — Path 3 needs CPU portability)

---

## The Path 3 thesis

A locally-runnable LLM (Phi-3-mini 3.8B / Qwen2.5 0.5-1.5B / Llama 3.2 1-3B)
handles language + cognition. The biology-grounded sim becomes the
**memory subsystem** — a continually-learning knowledge graph that
the LLM queries between turns. The distinctive thing isn't the chat
substrate; it's that the memory:

- Grows continuously across sessions (lineage-backed)
- Doesn't catastrophically forget (Phase 1.4 BRANCH A retention)
- Consolidates via sleep-replay (Phase 1.3 hippo → cortex)
- Is biology-grounded (STDP, embodied-Hebbian, no transformer-style
  attention)

This is the "ChatGPT with a brain-shaped memory that learns continuously"
product framing. Differentiated against:
- **Vanilla RAG**: forgets across sessions; vector DB doesn't consolidate
- **Fine-tuning**: monolithic update; catastrophic forgetting risk
- **Long-context**: stateless; doesn't accumulate experience

## The API

```python
class BridgeMemory:
    """LLM-callable memory subsystem backed by a SimulationBridge.

    Provides a clean key-value-ish interface that the LLM invokes via
    tool-use. The underlying bridge handles embodied-Hebbian binding,
    sleep-replay consolidation, and lineage persistence.
    """

    def __init__(self,
                 lineage_name: str = "main",
                 mode: str = "synonym",  # tier1/synonym/synonym12/synonym16
                 auto_save: bool = True,
                 verbose: bool = False):
        """Open a BridgeMemory backed by the given lineage.

        Loads the lineage if it exists (skip training); else trains
        from scratch via chat_repl's bio_three_factor + lineage save.
        """
        ...

    def store(self, key: str, value: str, **metadata) -> dict:
        """Bind a key->value association in the bridge.

        Trains the bridge on `(key, value)` pairs via embodied-Hebbian
        co-firing. Returns a dict with:
        - confidence: float (0-1, post-binding evaluation)
        - bound_correctly: bool
        - n_events_run: int (training events used)
        - elapsed_seconds: float

        Args:
            key: e.g. "user's name", "favorite color", "today's date"
            value: e.g. "Alice", "blue", "2026-05-11"
            **metadata: arbitrary k/v for the lineage growth event
                (kind="memory_bind", description="...", etc.)
        """
        ...

    def recall(self, key: str, top_k: int = 5,
                 temperature: float = 0.0) -> list[dict]:
        """Retrieve associations for a key, sorted by confidence.

        Drives `key` through `language_input`, reads motor population
        activity + language_output via the `:speak`-style path.
        Returns a list of dicts:
        - value: str (decoded word/phrase)
        - confidence: float (cosine similarity 0-1)
        - rank: int (1 = top)

        Args:
            key: the cue to recall
            top_k: how many candidates to return
            temperature: 0 = strict argmax; >0 = softmax sampling
                (for natural variation in repeated queries)
        """
        ...

    def forget(self, key: str, decay_rate: float = 0.5) -> dict:
        """Best-effort unbind. Reduces synaptic weights along the
        pathways associated with `key`. Returns a dict with
        n_synapses_decayed + estimated_retention (lower = more forgotten).

        Note: real unbinding is approximate. Plasticity decay over time
        + competing bindings are the biological mechanisms.
        """
        ...

    def consolidate(self, n_sleep_cycles: int = 3) -> dict:
        """Run sleep-replay consolidation (Phase 1.3).

        Transfers recent bindings from hippocampus to cortex. Reduces
        dependence on hippocampal traces (which decay faster) and
        strengthens long-term cortical storage. Returns stats:
        - pre_silence_acc: dict (accuracy before consolidation)
        - hippo_off_acc: dict (accuracy after, with hippo silenced)
        - retention_ratio: float (hippo_off / pre_silence — higher = better)
        """
        ...

    def stats(self) -> dict:
        """Snapshot of memory state:
        - n_bindings: int (estimated; counted from growth events)
        - cumulative_training_events: int
        - vocab_size: int
        - retention_estimate: float (last consolidation result)
        - lineage_name: str
        - mode: str
        - bridge_synapses: int
        - bridge_neurons: int
        """
        ...

    def list_keys(self) -> list[str]:
        """Return all known keys (vocab) the memory can recall."""
        ...

    def save(self, growth_kind: str = "manual_save",
              description: str = "") -> Path:
        """Force save to lineage (default: auto-saved on store/forget)."""
        ...
```

## LLM tool-use protocol

The LLM is given access to these tools via OpenAI / Anthropic / local
tool-use schema:

```json
{
  "tools": [
    {
      "name": "memory_store",
      "description": "Save a key-value fact in the brain-shaped memory.",
      "input_schema": {
        "type": "object",
        "properties": {
          "key": {"type": "string"},
          "value": {"type": "string"}
        },
        "required": ["key", "value"]
      }
    },
    {
      "name": "memory_recall",
      "description": "Retrieve facts associated with a key.",
      "input_schema": {
        "type": "object",
        "properties": {
          "key": {"type": "string"},
          "top_k": {"type": "integer", "default": 5}
        },
        "required": ["key"]
      }
    },
    {
      "name": "memory_consolidate",
      "description": "Run sleep-replay consolidation (move recent learnings to long-term storage).",
      "input_schema": {"type": "object", "properties": {}}
    }
  ]
}
```

## Interaction flow example

```
User: Hi! I'm Alice. I'm an engineer at Acme Corp.
LLM:  [tool_call: memory_store(key="user_name", value="Alice")]
      [tool_call: memory_store(key="user_role", value="engineer")]
      [tool_call: memory_store(key="user_employer", value="Acme Corp")]
      Hi Alice! Welcome. I'll remember that.

User: What's my favorite color?
LLM:  [tool_call: memory_recall(key="favorite_color")]
      [empty result — no binding]
      I don't have that on file yet. What is your favorite color?

User: It's blue.
LLM:  [tool_call: memory_store(key="favorite_color", value="blue")]
      Got it, Alice — blue. I'll remember.

[Session ends. Lineage saves bridge state.]

[NEW SESSION 3 days later — lineage loads bridge from disk.]

User: Hi, do you remember me?
LLM:  [tool_call: memory_recall(key="user_name")]
      [result: [{"value": "Alice", "confidence": 0.91, "rank": 1}]]
      Yes, you're Alice — engineer at Acme Corp. Your favorite color is
      blue. Welcome back!
```

The biology-grounded memory:
1. Persisted across sessions (lineage save/load)
2. Survived the 3-day gap (no catastrophic forgetting; STDP retention)
3. Can be queried in O(ms) (one bridge `:speak`-equivalent call)
4. Doesn't bloat (synaptic plasticity, not unbounded vector DB)

## Implementation phases

### Phase 3.1 — `BridgeMemory` scaffold (this autonomous-arc-friendly)

CPU-only. No LLM yet. Just the wrapper class with stubs that:
- `store()`: call existing chat_repl `learn` flow
- `recall()`: call existing chat_repl `:speak`-style generative inference
- `forget()`: scale specific pathway weights (set_pathway_weights with 0.5×)
- `consolidate()`: hook into Phase 1.3 sleep-replay (exists in bridge)
- `stats()`: count growth events + bridge metadata

Tests with mock bridge.

### Phase 3.2 — LLM integration (next session, GPU + LLM hosting)

- Choose local LLM (Phi-3-mini 3.8B is the natural target; quantized fits
  on RTX 3090 with VRAM headroom for the bridge)
- Host via vLLM / llama.cpp / ollama
- Define the tool-use JSON schema
- Wire BridgeMemory methods to tool-call handlers
- Smoke test: 5-turn conversation that exercises store + recall + consolidate

### Phase 3.3 — Multi-session continuity test (GPU + week-long arc)

- Day 1: 20-turn conversation; bindings stored
- Day 3: 20-turn conversation; recall accuracy on Day-1 bindings measured
- Day 7: same; should retain >80% (Phase 1.4 BRANCH A guarantee)
- Day 7 + consolidation: same; should retain >80% even with hippo silenced
  (Phase 1.3 hippo-OFF guarantee)

This validates the **distinctive product claim**: "an LLM that
remembers continuously without catastrophic forgetting."

## Why Path 3 is attractive

| Dimension | Path 1 (biology scale-up) | Path 2 (hybrid) | Path 3 (LLM + bio memory) |
|-----------|---------------------------|-----------------|---------------------------|
| Capability ceiling | Unknown | Tiny-SOTA-LLM | LLM |
| Timeline | 12-24 mo | 6-12 mo | **3-6 mo** |
| Cloud cost | $10K-50K | $1K-5K | **~$0** |
| Local hardware | RTX 3090 + cloud | RTX 3090 | **RTX 3090** |
| Risk | High | Moderate | **Low** |
| Distinctive value | Biology AGI substrate | Embodied + transformer | **Brain-shaped memory** |
| Probability of working | 20-40% | 60-80% | **90%+** |
| Useful product timeline | 12+ mo | 6-9 mo | **3 mo** |

Path 3 ships the soonest with the highest probability of working. It
also preserves the biology research as a supporting track — the
distinguishing feature is the memory subsystem, not the chat substrate.

## What's already shipped that Path 3 builds on

- **Lineage system** (continuous-learning memory across sessions)
- **Phase 1.4 BRANCH A** (5/6 retention validated)
- **Phase 1.3 consolidation** (3/3 hippo-OFF retention validated)
- **chat_repl** (the substrate for `store` and `recall`)
- **chat_speak** (the substrate for generative recall via `:speak`)
- **Tier 2.1 synonym mode** (8-word vocab as a starting binding capacity)
- **Auto-growth** (bind more vocab over time as new keys arrive)
- **NumPy backend** (LLM hosting may want CPU paths)

Path 3 is largely **integration work** on existing primitives, not new
research. That's the source of its low risk.

## Open questions

1. **LLM choice**: Phi-3-mini (3.8B), Qwen2.5 (0.5-1.5B), Llama 3.2 (1-3B)?
   Trade-offs: capability vs. local-hosting ease.
2. **Tool-use protocol**: OpenAI-style JSON, Anthropic-style XML, or
   local-LLM-specific?
3. **Memory scaling**: at what point does the bridge's tier need to grow?
   (Probably: when `vocab_size > 16` for the current arch.)
4. **Multi-key bindings**: today's bridge binds 1 key → 1 value. Real
   facts often have multiple values ("Alice's role" might be "engineer" OR
   "founder" OR "manager"). Does the multi-pop motor cortex support this?
   (Probably yes via Tier 2.1 synonym-style structure.)
5. **Privacy + persistence**: where do lineages live on disk? How does
   the user clear specific facts? `forget()` is best-effort; real
   "right-to-be-forgotten" is harder.

## Provenance

- This doc: `docs/plans/2026-05-11-path3-bridge-memory-api-design.md`
- Strategic context: `docs/plans/2026-05-11-strategic-reevaluation.md`
- Supporting infrastructure: Bridge Lineage Manager, chat_repl,
  Phase 1.3 + 1.4 work, NumPy backend, all shipped this arc.

Next autonomous-arc unit: ship `sim/bridge_memory.py` scaffold (Phase
3.1) with the API stubs + tests. LLM integration (3.2) waits for
the user's go-ahead on Path 3.
