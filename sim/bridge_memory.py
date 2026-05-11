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

    def store(self, key: str, value: str, n_events: int = 50,
                **metadata) -> dict:
        """Bind key → value in the bridge.

        Phase 3.1.5: uses chat_repl's learn_word_pairing flow
        (embodied-Hebbian co-firing). Today's bridge has 4 motor pools
        (N/E/S/W) — value must be one of these directions OR a known
        synonym/word that maps to one (per the mode's vocab table).

        Phase 3.2 will extend to multi-modal bindings as the arch grows.

        Args:
            key: cue word to bind (e.g. "alice", "favorite_color")
            value: target direction word (e.g. "north", "up", "left")
                Must be in the active mode's vocab or be a primary
                N/E/S/W letter.
            n_events: training events for the co-firing session (default 50)
            **metadata: extra fields for the lineage growth event

        Returns:
            {
              "key": str,
              "value": str,
              "target_action": str (N/E/S/W),
              "confidence": float,
              "bound_correctly": bool,
              "n_events_run": int,
              "elapsed_seconds": float,
            }

        Raises:
            ValueError: if value cannot be mapped to an N/E/S/W action.
        """
        self._ensure_loaded()
        t0 = time.time()
        target_action = self._value_to_action(value)
        from research.runners.chat_repl import learn_word_pairing

        summary = learn_word_pairing(
            self.bridge,
            word=key,
            target_action=target_action,
            n_events=n_events,
            verbose=self.verbose,
        )

        # Best-effort confidence via a follow-up chat_inference
        from research.runners.chat_repl import chat_inference
        try:
            check = chat_inference(self.bridge, key)
            confidence = float(check.get("confidence_ratio", 0.0))
            bound_correctly = (check.get("predicted_action") == target_action)
        except Exception:
            confidence = 0.0
            bound_correctly = False

        result = {
            "key": key,
            "value": value,
            "target_action": target_action,
            "confidence": confidence,
            "bound_correctly": bound_correctly,
            "n_events_run": int(summary.get("n_events_run", n_events)),
            "elapsed_seconds": time.time() - t0,
        }
        self._vocab_size_estimate += 1
        if self.auto_save and self._lineage is not None:
            self._save_to_lineage(
                "memory_bind",
                f"store('{key}', '{value}') -> motor_{target_action}, "
                f"acc={confidence:.2f}",
                key=key, value=value, target_action=target_action,
                confidence=confidence, **metadata,
            )
        return result

    def _value_to_action(self, value: str) -> str:
        """Map value → motor action letter (N/E/S/W).

        Accepts:
          - Direction letters: "N", "E", "S", "W" (any case)
          - Primary direction words: "north", "east", "south", "west"
          - Synonyms in the mode's vocab table (e.g. "up", "right")
        """
        value_lower = value.strip().lower()
        # Direct letter
        if value.upper() in ("N", "E", "S", "W"):
            return value.upper()
        # Vocab lookup
        try:
            from research.runners.chat_repl import _vocab_for_mode
            _, word_to_action = _vocab_for_mode(self.mode)
            if value_lower in word_to_action:
                return word_to_action[value_lower]
        except Exception:
            pass
        raise ValueError(
            f"value '{value}' doesn't map to N/E/S/W under mode "
            f"'{self.mode}'. Today's bridge has 4 motor pools; use a "
            f"primary direction word or a synonym from the mode vocab."
        )

    # ── Recall ──────────────────────────────────────────────────────

    def recall(self, key: str, top_k: int = 5,
                 temperature: float = 0.0) -> list[dict]:
        """Retrieve associations for key, sorted by confidence.

        Phase 3.1.5: uses chat_repl.chat_inference to drive `key`
        through language_input and read motor activity. Returns the
        top-k (action, confidence) pairs ranked by spike-delta.

        For a more semantically-rich response, use a vocab-aware
        synonym mode (synonym / synonym12 / synonym16) — recall will
        return the canonical primary word for the top motor pool.

        Args:
            key: the cue
            top_k: how many candidates to return (max 4 = N/E/S/W;
                limited by the 4-motor-pool arch)
            temperature: reserved for Phase 3.2 (softmax sampling)

        Returns:
            List of {"value": str, "confidence": float, "rank": int,
                       "action": str} sorted by descending confidence.
        """
        self._ensure_loaded()
        from research.runners.chat_repl import chat_inference
        try:
            result = chat_inference(self.bridge, key)
        except Exception:
            return []

        # delta_counts is {"N": int, "E": int, "S": int, "W": int}
        # Rank by descending delta
        deltas = result.get("delta_counts", {})
        if not deltas:
            return []
        # Compute confidence-style score: positive delta / max delta
        max_delta = max(max(deltas.values()), 1)
        ranked = sorted(deltas.items(), key=lambda kv: -kv[1])

        # Action -> primary word for the response
        action_to_word = {"N": "north", "E": "east", "S": "south", "W": "west"}
        out = []
        for rank, (action, delta) in enumerate(ranked[:top_k], 1):
            out.append({
                "action": action,
                "value": action_to_word.get(action, action),
                "confidence": float(delta / max_delta) if max_delta > 0 else 0.0,
                "rank": rank,
                "raw_delta": int(delta),
            })
        return out

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
