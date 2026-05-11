"""BridgeMemory — LLM-callable memory subsystem wrapping a SimulationBridge.

Phase 3.1 of the Path 3 design: scaffold + API stubs that wire to the
existing chat_repl bind/recall flow. LLM integration (Phase 3.2) comes
next session.

Design doc: docs/plans/2026-05-11-path3-bridge-memory-api-design.md

Usage (the LLM will invoke these via tool-use; humans can use them too):

    from sim.bridge_memory import BridgeMemory

    mem = BridgeMemory(lineage_name="alice", mode="synonym")

    # Bind facts
    mem.store("user_name", "alice")
    mem.store("favorite_color", "blue")

    # Query
    results = mem.recall("user_name", top_k=3)
    # [{"value": "alice", "confidence": 0.91, "rank": 1}, ...]

    # Long-term consolidation
    consolidation_stats = mem.consolidate(n_sleep_cycles=3)

    # Inspect state
    print(mem.stats())

The bridge handles biology-grounded plasticity (STDP + embodied-Hebbian)
+ persistence (Bridge Lineage Manager). The LLM treats the bridge as
a black-box key-value store with the unique property of continuous
learning + no-catastrophic-forgetting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class BridgeMemory:
    """LLM-callable memory subsystem.

    Phase 3.1 scaffold: API + stubs. The actual bind/recall paths wire
    to chat_repl helpers (chat_inference + the embodied-Hebbian learn
    flow) which are themselves backend-aware via sim.backend.

    Args:
        lineage_name: name of the BridgeLineage backing this memory.
            Default "main" — shares state with the chat REPL's default
            lineage.
        mode: chat_repl mode ("tier1" / "synonym" / "synonym12" /
            "synonym16"). Determines the underlying arch + vocab capacity.
            Default "synonym" (Tier 2.1 = 8-word vocab; capacity to grow
            with synonyms).
        bridge: optional pre-loaded SimulationBridge. If None, the
            memory loads the lineage or trains from scratch.
        auto_save: if True (default), each store() / forget() /
            consolidate() saves back to the lineage.
        verbose: passes through to underlying chat_repl helpers.

    State:
        _bridge: SimulationBridge (lazy-loaded)
        _lineage: BridgeLineage (lazy-loaded)
        _vocab_size_estimate: count of bindings made (best-effort)
    """
    lineage_name: str = "main"
    mode: str = "synonym"
    bridge: Optional[Any] = None
    auto_save: bool = True
    verbose: bool = False

    # Internal state — initialized lazily on first call
    _lineage: Any = None
    _vocab_size_estimate: int = 0
    _last_consolidation: Optional[dict] = None

    def _ensure_loaded(self) -> None:
        """Load lineage + bridge if not already loaded."""
        if self.bridge is not None and self._lineage is not None:
            return
        from sim.lineage import BridgeLineage
        if self._lineage is None:
            self._lineage = BridgeLineage(self.lineage_name)
        if self.bridge is None:
            # Lazy-load via chat_repl's helpers — works under both
            # backends and respects the lineage state.
            if self._lineage.exists():
                from research.runners.chat_repl import _load_bridge_from_checkpoint
                # default seed 42 — matches chat_repl defaults
                self.bridge = _load_bridge_from_checkpoint(
                    str(self._lineage.current_path), self.mode, 42,
                    verbose=self.verbose,
                )
            else:
                # No saved state — train a fresh bridge at this mode
                # using chat_repl's helpers. This is a slow path; only
                # happens once per lineage.
                from research.runners.chat_repl import (
                    _load_or_train_tier1, _load_or_train_synonym,
                )
                if self.mode == "tier1":
                    self.bridge = _load_or_train_tier1(
                        seed=42, n_train_events=200, verbose=self.verbose,
                    )
                elif self.mode in ("synonym", "synonym12", "synonym16"):
                    vocab_map = {"synonym": 8, "synonym12": 12,
                                  "synonym16": 16}
                    self.bridge = _load_or_train_synonym(
                        seed=42, n_train_events=400,
                        verbose=self.verbose,
                        vocab_size=vocab_map[self.mode],
                    )
                else:
                    raise ValueError(f"unknown mode: {self.mode}")
                # Save initial state
                if self.auto_save:
                    self._save_to_lineage("init", "Memory initialized")

    # ── Store ────────────────────────────────────────────────────────

    def store(self, key: str, value: str, **metadata) -> dict:
        """Bind key → value in the bridge.

        Phase 3.1 stub: uses chat_repl's learn flow (embodied-Hebbian
        co-firing). The learn flow assumes value maps to one of N/E/S/W
        actions, which limits Phase 3.1 to action-like bindings.
        Multi-vocab extension is Phase 3.2.

        Args:
            key: cue word (e.g. "user_name", "favorite_color")
            value: target word (e.g. "alice", "blue")
            **metadata: extra fields for the lineage growth event

        Returns:
            {
              "key": str,
              "value": str,
              "confidence": float,
              "bound_correctly": bool,
              "n_events_run": int,
              "elapsed_seconds": float,
              "stub_note": str (Phase 3.1 placeholder explanation)
            }
        """
        self._ensure_loaded()
        t0 = time.time()
        # Phase 3.1 scaffold: record the request without actually doing
        # embodied-Hebbian learning. Real implementation calls
        # chat_repl's _run_embodied_learn_session, which trains the
        # bridge on (key, value) co-firing.
        result = {
            "key": key,
            "value": value,
            "confidence": 0.0,  # not yet computed
            "bound_correctly": False,
            "n_events_run": 0,
            "elapsed_seconds": time.time() - t0,
            "stub_note": (
                "Phase 3.1 scaffold: bridge binding deferred to Phase "
                "3.2. The lineage growth event IS recorded so the "
                "memory's intended state survives this stub."
            ),
        }
        self._vocab_size_estimate += 1
        if self.auto_save and self._lineage is not None:
            self._save_to_lineage(
                "memory_bind",
                f"store('{key}', '{value}')",
                key=key, value=value, **metadata,
            )
        return result

    # ── Recall ──────────────────────────────────────────────────────

    def recall(self, key: str, top_k: int = 5,
                 temperature: float = 0.0) -> list[dict]:
        """Retrieve associations for key, sorted by confidence.

        Phase 3.1 stub: returns empty list. Real implementation calls
        chat_repl's chat_inference or chat_speak path with `key` as
        input, ranks the resulting motor activity / language_output
        spikes against the known vocab.

        Args:
            key: the cue
            top_k: how many candidates to return
            temperature: 0 = argmax; >0 = softmax sampling

        Returns:
            List of {"value": str, "confidence": float, "rank": int}
            sorted by descending confidence. Empty if no binding exists
            or stub stage.
        """
        self._ensure_loaded()
        # Phase 3.1 scaffold: return empty list. Real implementation
        # routes through chat_repl.chat_inference / chat_speak.
        return []

    # ── Forget ──────────────────────────────────────────────────────

    def forget(self, key: str, decay_rate: float = 0.5) -> dict:
        """Best-effort unbind. Decays weights along the pathway
        associated with `key`.

        Phase 3.1 stub: records the request as a growth event but
        does not actually decay weights. Real implementation calls
        bridge.set_pathway_weights with weights × decay_rate for the
        edges connecting `key` neurons to motor pools.

        Returns:
            {
              "key": str,
              "decay_rate": float,
              "n_synapses_decayed": int (0 in stub),
              "estimated_retention": float (1.0 in stub),
            }
        """
        self._ensure_loaded()
        result = {
            "key": key,
            "decay_rate": decay_rate,
            "n_synapses_decayed": 0,
            "estimated_retention": 1.0,
            "stub_note": (
                "Phase 3.1 scaffold: forget deferred to Phase 3.2."
            ),
        }
        if self.auto_save and self._lineage is not None:
            self._save_to_lineage(
                "memory_forget",
                f"forget('{key}', decay={decay_rate})",
                key=key, decay_rate=decay_rate,
            )
        return result

    # ── Consolidate ─────────────────────────────────────────────────

    def consolidate(self, n_sleep_cycles: int = 3) -> dict:
        """Run sleep-replay consolidation (Phase 1.3).

        Phase 3.1 stub: returns mock stats. Real implementation calls
        the bridge's consolidation pathway (see Phase 1.3 work in
        bridge.py + the consolidation_synonym_trainer runner).

        Returns:
            {
              "pre_silence_acc": dict,
              "hippo_off_acc": dict,
              "retention_ratio": float,
              "n_sleep_cycles_run": int,
            }
        """
        self._ensure_loaded()
        result = {
            "pre_silence_acc": {"overall": 0.0},
            "hippo_off_acc": {"overall": 0.0},
            "retention_ratio": 1.0,
            "n_sleep_cycles_run": 0,
            "stub_note": (
                "Phase 3.1 scaffold: consolidation deferred to Phase 3.2. "
                "The lineage growth event IS recorded."
            ),
        }
        self._last_consolidation = result
        if self.auto_save and self._lineage is not None:
            self._save_to_lineage(
                "memory_consolidate",
                f"consolidate({n_sleep_cycles} cycles)",
                n_sleep_cycles=n_sleep_cycles,
            )
        return result

    # ── Inspection ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot of memory state."""
        self._ensure_loaded()
        meta = self._lineage.read_metadata() if self._lineage else None
        n_synapses = 0
        n_neurons = 0
        if self.bridge is not None:
            n_synapses = int(getattr(
                self.bridge, "actual_total_connections_n", 0
            )) if hasattr(self.bridge, "actual_total_connections_n") else 0
            n_neurons = int(getattr(self.bridge.core_sim_config
                                    if hasattr(self.bridge, "core_sim_config")
                                    else self.bridge.core_config,
                                    "num_neurons", 0))
        return {
            "lineage_name": self.lineage_name,
            "mode": self.mode,
            "n_bindings_estimate": self._vocab_size_estimate,
            "cumulative_training_events": (
                meta.cumulative_training_events if meta else 0
            ),
            "vocab_size": len(meta.vocab) if meta else 0,
            "last_consolidation": self._last_consolidation,
            "bridge_synapses": n_synapses,
            "bridge_neurons": n_neurons,
            "stub_phase": "3.1",
        }

    def list_keys(self) -> list[str]:
        """Return all known keys (vocab) the memory can recall.

        Phase 3.1 stub: returns the static vocab for the mode. Real
        implementation would track which keys have been bound via
        memory_bind growth events.
        """
        self._ensure_loaded()
        try:
            from research.runners.chat_repl import _vocab_for_mode
            vocab, _ = _vocab_for_mode(self.mode)
            return sorted(vocab)
        except Exception:
            return []

    def save(self, growth_kind: str = "manual_save",
              description: str = "") -> Path:
        """Force save to lineage. Returns the current_path."""
        self._ensure_loaded()
        if self._lineage is None:
            raise RuntimeError("BridgeMemory has no lineage configured")
        self._save_to_lineage(growth_kind, description or "Manual save")
        return self._lineage.current_path

    # ── Internal helpers ─────────────────────────────────────────────

    def _save_to_lineage(self, kind: str, description: str,
                            **metadata) -> None:
        """Save the bridge to the lineage with a growth event."""
        if self._lineage is None or self.bridge is None:
            return
        try:
            arch = {
                "mode": self.mode,
                "n_neurons": int(getattr(
                    self.bridge.core_sim_config
                    if hasattr(self.bridge, "core_sim_config")
                    else self.bridge.core_config,
                    "num_neurons", 0)),
            }
            self._lineage.save(self.bridge, tier=self.mode, arch=arch)
            meta = self._lineage.read_metadata()
            meta.add_growth_event(kind=kind, description=description, **metadata)
            self._lineage.write_metadata(meta)
        except Exception as e:
            if self.verbose:
                print(f"[BridgeMemory] save failed (non-fatal): {e}",
                      flush=True)
