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
        # ── Safety check: lineage/bridge arch mismatch ──
        # A common failure mode: the lineage was saved at toy-scale
        # arch (e.g. 208 neurons from bridge_memory_demo's tier1 toy)
        # but the chat_repl loader rebuilds the standard tier1 bridge
        # (~6336 neurons), leaving region_manager at the bigger layout
        # while cp_external_input_current resizes from the checkpoint.
        # The mismatch surfaces later as IndexError on store().
        # Detect early and raise a clear message.
        if self.bridge is not None:
            try:
                n_neurons = int(self.bridge.core_config.num_neurons)
                cur_array = self.bridge.cp_external_input_current
                cur_size = int(cur_array.shape[0])
                rm_indices = self.bridge.region_manager.indices("language_input")
                max_rm_idx = max(rm_indices) if rm_indices else 0
                if max_rm_idx >= cur_size or max_rm_idx >= n_neurons:
                    raise RuntimeError(
                        f"BridgeMemory: lineage '{self.lineage_name}' "
                        f"has cp_external_input_current of size {cur_size} "
                        f"(num_neurons={n_neurons}) but region_manager's "
                        f"language_input region extends to index {max_rm_idx}. "
                        f"This indicates a toy-scale lineage being loaded "
                        f"against the standard tier1/synonym architecture. "
                        f"Re-bootstrap the lineage at the intended scale "
                        f"(e.g. `python -m research.runners.chat_repl "
                        f"--mode {self.mode} --lineage {self.lineage_name} "
                        f"--from-scratch`)."
                    )
            except RuntimeError:
                raise
            except Exception:
                # Best-effort check; don't block on attribute errors
                pass

    # ── Store ────────────────────────────────────────────────────────

    # Mode-specific default training events. Production-scale synonym
    # arches (12K neurons, ~12M synapses) need more co-firing events to
    # overcome random-init bias than the toy tier1 (~208 neurons).
    # Tuned 2026-05-11 after live smoke showed 50 events insufficient
    # for synonym scale.
    DEFAULT_N_EVENTS = {
        "tier1": 50,
        "synonym": 200,
        "synonym12": 200,
        "synonym16": 200,
    }

    def store(self, key: str, value: str, n_events: int | None = None,
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
            n_events: training events for the co-firing session. If None,
                uses DEFAULT_N_EVENTS[mode] (50 for tier1, 200 for
                synonym variants — production scale needs more events).
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
        # Resolve default training events from mode if not explicit.
        if n_events is None:
            n_events = self.DEFAULT_N_EVENTS.get(self.mode, 50)
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

    def forget(self, key: str, decay_rate: float = 0.5,
                  sparsity: float = 0.1) -> dict:
        """Decay weights along synapses originating from `key`'s
        language_input neurons.

        Phase 3.2 real-ops (2026-05-11): computes the deterministic
        embedding for `key`, identifies which language_input neurons
        it activates (top `sparsity` fraction per
        text_embeddings.vocab_to_drive_pattern), and multiplies the
        weights of all outgoing synapses from those neurons by
        `decay_rate`.

        Biology: real forgetting is gradual + neuromodulator-gated
        decay along recently-active synapses. This is a coarser
        intervention — uniform multiplicative decay on edges sourced
        from the key's active neurons. Closer to extinction-style
        forgetting than passive decay.

        Args:
            key: the cue word to forget
            decay_rate: weights are multiplied by this (0.0 = full
                erase, 0.5 = halve, 1.0 = no-op). Default 0.5.
            sparsity: must match the sparsity used when binding (must
                target the same neurons). Default 0.1 — matches the
                bind/recall default.

        Returns:
            {
              "key": str,
              "decay_rate": float,
              "n_active_neurons": int,        # active in language_input
              "n_synapses_decayed": int,      # weights touched
              "mean_weight_pre": float,
              "mean_weight_post": float,
              "estimated_retention": float,   # post/pre ratio
            }
        """
        self._ensure_loaded()

        # ── Step 1: identify language_input neurons active for `key` ──
        from sim.text_embeddings import vocab_to_drive_pattern
        from sim.backend import to_host as _to_host
        import numpy as _np

        if self.bridge.region_manager is None:
            raise RuntimeError(
                "BridgeMemory.forget: bridge has no region_manager. "
                "Brain-region framework must be enabled."
            )
        try:
            lang_indices = list(
                self.bridge.region_manager.indices("language_input")
            )
        except Exception as e:
            raise RuntimeError(
                f"BridgeMemory.forget: language_input region not found: {e}"
            ) from None

        n_lang = len(lang_indices)
        drive = vocab_to_drive_pattern(
            key, n_neurons=n_lang, drive_max_pA=200.0, sparsity=sparsity,
        )
        active_local = _np.where(drive > 0)[0]
        # Map local language_input indices to global neuron indices
        active_global = _np.array(
            [lang_indices[i] for i in active_local], dtype=_np.int64,
        )
        n_active = int(len(active_global))
        if n_active == 0:
            return {
                "key": key, "decay_rate": decay_rate,
                "n_active_neurons": 0, "n_synapses_decayed": 0,
                "mean_weight_pre": 0.0, "mean_weight_post": 0.0,
                "estimated_retention": 1.0,
                "warn": "no active neurons for this key",
            }

        # ── Step 2: locate outgoing edges in CSR ─────────────────────
        # cp_connections is CSR in (pre -> post) layout. For each
        # active pre-neuron, slice rows by indptr to get its outgoing
        # edges' data indices.
        cp_conn = self.bridge.cp_connections
        indptr_host = _to_host(cp_conn.indptr)
        # Edges-by-row index arrays into cp_conn.data
        data_indices = []
        for src in active_global:
            start = int(indptr_host[src])
            end = int(indptr_host[src + 1])
            if end > start:
                data_indices.extend(range(start, end))
        if not data_indices:
            return {
                "key": key, "decay_rate": decay_rate,
                "n_active_neurons": n_active, "n_synapses_decayed": 0,
                "mean_weight_pre": 0.0, "mean_weight_post": 0.0,
                "estimated_retention": 1.0,
                "warn": "no outgoing synapses from active neurons",
            }

        # ── Step 3: decay weights ────────────────────────────────────
        # Use backend-aware indexing: cp_conn.data is a {cupy, numpy}
        # ndarray. We update in-place.
        from sim.backend import get_backend
        xp, _backend_name = get_backend()
        idx = xp.asarray(data_indices, dtype=xp.int64)
        data = cp_conn.data
        # Snapshot pre-decay weights for stats
        pre_weights_host = _to_host(data[idx])
        mean_pre = float(_np.mean(pre_weights_host)) if len(pre_weights_host) else 0.0
        # Apply decay (cast to data's dtype to preserve precision)
        data[idx] = data[idx] * xp.asarray(decay_rate, dtype=data.dtype)
        # Snapshot post-decay for confirmation
        post_weights_host = _to_host(data[idx])
        mean_post = float(_np.mean(post_weights_host)) if len(post_weights_host) else 0.0

        retention = mean_post / mean_pre if mean_pre > 0 else 1.0
        n_decayed = int(len(data_indices))

        result = {
            "key": key,
            "decay_rate": decay_rate,
            "n_active_neurons": n_active,
            "n_synapses_decayed": n_decayed,
            "mean_weight_pre": mean_pre,
            "mean_weight_post": mean_post,
            "estimated_retention": retention,
        }
        if self.auto_save and self._lineage is not None:
            self._save_to_lineage(
                "memory_forget",
                f"forget('{key}', decay={decay_rate}) -> "
                f"{n_decayed} synapses decayed ({mean_pre:.3f} -> "
                f"{mean_post:.3f}, retention={retention:.2f})",
                key=key, decay_rate=decay_rate,
                n_synapses_decayed=n_decayed,
                mean_weight_pre=mean_pre,
                mean_weight_post=mean_post,
            )
        return result

    # ── Consolidate ─────────────────────────────────────────────────

    def consolidate(self, n_sleep_cycles: int = 3,
                       n_swr_events_per_cycle: int = 200,
                       swr_drive_pA: float = 100.0) -> dict:
        """Run sleep-replay consolidation (Phase 1.3).

        Phase 3.2 real-ops (2026-05-11): drives CA3 with SWR-style
        bursts during a sleep phase, propagating patterns through
        ca3 -> ca1 -> motor / lang_output via the consolidation
        pathways. Implements Buzsaki 2015 ripple model + McClelland
        1995 CLS theory.

        Requires a hippocampus-enabled bridge (built with
        enable_hippocampus_consolidation=True). The `main` lineage as
        of 2026-05-11 is NOT hippocampus-enabled; bootstrap a
        hippo-enabled lineage via
        research.runners.consolidation_trainer first.

        Args:
            n_sleep_cycles: number of sleep cycles to run. Each cycle
                runs `n_swr_events_per_cycle` SWR bursts.
            n_swr_events_per_cycle: SWR bursts per cycle (default 200)
            swr_drive_pA: CA3 drive amplitude during bursts (default 100)

        Returns:
            {
              "n_sleep_cycles_run": int,
              "n_swr_events_run": int,
              "elapsed_seconds": float,
              "hippocampus_enabled": bool,
              "note": str (if degenerate),
            }
        """
        self._ensure_loaded()
        t0 = time.time()

        # Detect hippocampus by looking for the "ca3" region.
        has_hippo = False
        try:
            if self.bridge.region_manager is not None:
                self.bridge.region_manager.indices("ca3")
                has_hippo = True
        except Exception:
            has_hippo = False

        if not has_hippo:
            result = {
                "n_sleep_cycles_run": 0,
                "n_swr_events_run": 0,
                "elapsed_seconds": 0.0,
                "hippocampus_enabled": False,
                "note": (
                    "Bridge lacks hippocampus (no 'ca3' region). Build a "
                    "hippo-enabled bridge with "
                    "enable_hippocampus_consolidation=True via "
                    "research.runners.consolidation_trainer."
                ),
            }
        else:
            # Run real consolidation
            from research.runners.consolidation_trainer import (
                run_swr_replay_phase,
            )
            from research.runners.text_minimal_isolation import (
                set_awake_gates, set_sleep_gates,
            )
            import numpy as _np
            rng = _np.random.default_rng()
            n_events_total = 0
            try:
                set_sleep_gates(self.bridge)
                for _ in range(int(n_sleep_cycles)):
                    run_swr_replay_phase(
                        self.bridge,
                        n_swr_events=n_swr_events_per_cycle,
                        swr_drive_pA=swr_drive_pA,
                        rng=rng,
                    )
                    n_events_total += n_swr_events_per_cycle
            finally:
                set_awake_gates(self.bridge)
            result = {
                "n_sleep_cycles_run": int(n_sleep_cycles),
                "n_swr_events_run": n_events_total,
                "elapsed_seconds": time.time() - t0,
                "hippocampus_enabled": True,
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

    def speak(self, action: str, top_k: int = 4,
                temperature: float = 0.0) -> list[dict]:
        """Generative A→W recall: drive a motor pool, decode what word.

        Phase 3.1.6 (added 2026-05-11): the inverse of recall(). Where
        recall takes a key and reads the motor activity, speak takes a
        motor action and reads the language_output cortex to decode
        which word the sim "thinks" goes with that action.

        Used for:
        - Verifying bindings: store("alice", "north"); speak("N") -> "alice"
        - LLM-driven generation: "produce a word for direction N" -> "alice"
        - Conversation paraphrase: LLM can ask the memory to "speak"
          its current binding for a motor concept

        Args:
            action: motor pool to activate ("N", "E", "S", "W")
            top_k: how many candidate words to return
            temperature: 0 = strict argmax (deterministic);
                0.01-0.05 = primary dominant with synonym variation;
                0.05+ = more variety

        Returns:
            List of {"word": str, "similarity": float, "rank": int}
            sorted by descending similarity. Mirrors the output of
            chat_repl.generative_inference for compatibility with the
            existing :speak path.
        """
        self._ensure_loaded()
        from research.runners.chat_repl import generative_inference

        action = action.upper()
        if action not in ("N", "E", "S", "W"):
            raise ValueError(
                f"speak: action must be one of N/E/S/W, got '{action}'"
            )
        try:
            result = generative_inference(
                self.bridge,
                target_action=action,
                temperature=temperature,
            )
        except Exception:
            return []

        # generative_inference returns {"rankings": [...], "delta": ..., "top_k": ...}
        # We normalize to a consistent schema
        rankings = result.get("rankings", [])
        out = []
        for rank, (word, sim) in enumerate(rankings[:top_k], 1):
            out.append({
                "word": word,
                "similarity": float(sim),
                "rank": rank,
            })
        return out

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
