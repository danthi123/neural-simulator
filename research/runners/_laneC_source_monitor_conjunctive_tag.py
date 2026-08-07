"""Conjunctive source-tag at ENCODING for the source-monitor lane.

The prior lever -- thresholded heterosynaptic depression of the encoding fan-out
(``_laneC_source_monitor_hetero_encoding.py``) -- was a NO-GO for a STRUCTURAL
reason the smoke measured directly: the shared core fires in EVERY recall, so any
post-hoc synapse pruning REDISTRIBUTES the rival burden across sources rather than
removing it (2026-08-07-source-monitor-hetero-encoding-NO-GO-...). The reframe the
NO-GO earned: **separation must live in WHICH CELLS FIRE, not in which synapses
survive.** This runner is the pre-scoped fallback that does exactly that.

MECHANISM (Komorowski-Manns-Eichenbaum 2009 item-in-context conjunctive cells):
during ``experience`` (ENCODING ONLY) the physical source afferent WEAKLY modulates
the overlap/episode layer, so a source-specific SUBSET of the shared-core cells
fires preferentially for each source. Different subsets -> different source-specific
assemblies form at encoding, so the separation is in which cells fire. Operationally
the modulation is delivered as a weak additive drive (``source_tag_gain * drive_pA``)
to a fixed, source-specific random subset of the driven episode cells -- the
sensory/motor context reaching association cortex, a documented host scaffold of the
SAME class as the episode drive and the source-afferent drive (a fully-synaptic
source_afferent -> episode projection is the biologization step if this clears). One
knob ``--source-tag-gain``; ``--source-tag-gain 0`` adds nothing and is asserted
byte-identical to the symmetric-Hebbian overlap NO-GO baseline (the null control).

THE LOAD-BEARING HONESTY GUARD: the tag shapes ENCODING ONLY. RECALL STAYS
EPISODE-ONLY -- the source afferent (and the tag) are ABSENT at recall. The tag is
applied ONLY inside ``experience`` (``_drive_with_tag``); ``recall`` uses the base
``_drive`` unchanged. If any source-label information reached the recall path the
margin would be a trivial source-label read (the exact cheat v6/v9 GOs were RETRACTED
as instrument artifacts for). Anti-cheat (b) VERIFIES per arm that during recall the
source-afferent external current is 0, the source-afferent firing is 0, and the
episode drive equals ``drive_pA`` exactly (no tag boost leaked in). A non-vacuity
diagnostic (``probe_label_leak``) confirms that a tag forcibly injected AT recall
WOULD move the winner -- proving the guard is excluding a real label path, not a
no-op.

Anti-cheats reported EVERY arm (all must hold for a promising smoke):
  (a) source_tag_gain=0 reproduces the overlap NO-GO byte-identically (asserted);
  (b) recall is episode-only -- source afferent current + firing 0 at recall, and
      the episode drive is unmodulated (THE cheat guard);
  (c) commitment-distribution entropy spans all THREE sources (the tag actually
      differentiated the core into source-specific subsets, not a recency collapse);
  (d) reliability guard: all_dominant_correct stays True AND no source's own recall
      rate drops vs the gain=0 baseline;
  (e) zero-learned-weight control stays strict=False (no stepping-history artifact).
NumPy backend, deterministic.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from sim.backend import get_backend, to_host
from research.runners._laneC_source_monitor_coresidency_gate import (
    SOURCES,
    SOURCE_LEARNING_GATE,
    _dominant_source,
    _source_margin,
)
from research.runners._laneC_source_monitor_coresidency_gate_v5 import (
    SOURCE_COMPETITION_GATE,
    SourceMonitorConfigV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
    SourceMonitorCoresidencyGateV6,
)
from research.runners._laneC_source_monitor_overlap_sweep import (
    CALIBRATION_SEEDS,
    _recall_three,
    evaluate_overlap as _orig_evaluate_overlap,
    make_overlapping_episode_patterns,
)
from tools.lab import attributable_to

HELD_OUT_SEEDS = (655, 656, 657)
DEFAULT_OVERLAPS = (0.2, 0.4)
DEFAULT_TAG_FRAC = 0.5  # fraction of episode cells tagged per source (source-specific)
FLOOR = 0.15  # frozen v6 min-source-margin floor F


class ConjunctiveTagGate(SourceMonitorCoresidencyGateV6):
    """v6 gate whose ``experience`` weakly modulates the overlap layer with a
    fixed, source-specific tag (encoding-only). ``recall`` is inherited unchanged
    and stays episode-only, so the tag never reaches the recall path."""

    def __init__(self, *, seed: int, config=None, tag_frac: float = DEFAULT_TAG_FRAC):
        super().__init__(seed=seed, config=config)
        n_ep = int(self.config.n_episode)
        # Per-source tag mask over episode-local cells. Distinct RNG stream per
        # source -> the tagged subsets DIFFER across sources, so within the shared
        # core different subsets get the preferential drive (the whole point).
        self._tag_mask = {}
        for si, s in enumerate(SOURCES):
            rng = np.random.default_rng(int(self.seed) * 131 + si + 1)
            self._tag_mask[s] = rng.random(n_ep) < float(tag_frac)
        # enc_fire[e, s] = cumulative firing of episode cell e during source s's
        # encoding events (measures which cells fired preferentially for each source).
        self._enc_fire = np.zeros((n_ep, len(SOURCES)), dtype=np.float64)

    # -- Encoding-only tag drive -------------------------------------------------
    def _drive_with_tag(
        self,
        episode_global: np.ndarray,
        active_sources: Sequence[str],
        pattern_local: np.ndarray,
        tag_gain: float,
    ) -> None:
        """Base ``_drive`` PLUS a weak source-specific additive boost to the tagged
        subset of the driven episode cells. tag_gain=0 -> identical to base _drive."""

        xp, _ = get_backend()
        self.bridge.cp_external_input_current[:] = 0.0
        self.bridge.cp_external_input_current[
            xp.asarray(episode_global, dtype=xp.int64)
        ] = float(self.config.drive_pA)
        for source in active_sources:
            self.bridge.cp_external_input_current[
                xp.asarray(self._source_afferent_indices[source], dtype=xp.int64)
            ] = float(self.config.drive_pA)
        if tag_gain > 0.0:
            boost = float(tag_gain) * float(self.config.drive_pA)
            for source in active_sources:
                tagged_local = pattern_local[self._tag_mask[source][pattern_local]]
                if tagged_local.size:
                    g = self._episode_indices[tagged_local]
                    self.bridge.cp_external_input_current[
                        xp.asarray(g, dtype=xp.int64)
                    ] += boost

    def experience(
        self,
        episode_pattern: Sequence[int],
        *,
        visual_activity: bool = False,
        auditory_activity: bool = False,
        corollary_discharge: bool = False,
        learning_enabled: bool = True,
        source_afferent_lesion: bool = False,
        source_tag_gain: float = 0.0,
    ) -> dict:
        """Base ``experience`` VERBATIM except the drive is tag-aware and a
        side-effect-free per-step firing read accumulates ``_enc_fire``. With
        source_tag_gain=0 no extra current is applied and the firing read advances
        no RNG, so weights + recall are byte-identical to the base gate."""

        episode_global = self._episode_global_indices(episode_pattern)
        active_sources = self._active_sources(
            visual_activity=visual_activity,
            auditory_activity=auditory_activity,
            corollary_discharge=corollary_discharge,
        )
        pattern_local = np.asarray(episode_pattern, dtype=np.int64)
        active_idx = [SOURCES.index(s) for s in active_sources]
        before = self.weight_summary()
        self.bridge.set_plasticity_gate(
            SOURCE_LEARNING_GATE, 1.0 if learning_enabled else 0.0
        )
        self.bridge.set_transmission_gate(
            "source_afferent_transmission", 0.0 if source_afferent_lesion else 1.0
        )
        try:
            for _ in range(int(self.config.training_cycles)):
                self._drive_with_tag(
                    episode_global, active_sources, pattern_local, float(source_tag_gain)
                )
                for _ in range(int(self.config.training_steps)):
                    self.bridge._run_one_simulation_step()
                    firing = np.asarray(
                        to_host(self.bridge.cp_firing_states), dtype=np.float64
                    )
                    pre = firing[episode_global]  # per driven episode cell
                    for si in active_idx:
                        self._enc_fire[pattern_local, si] += pre
                self._rest()
        finally:
            self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
            self.bridge.set_transmission_gate("source_afferent_transmission", 1.0)
            self.bridge.cp_external_input_current[:] = 0.0
        after = self.weight_summary()
        return {
            "active_afferents": list(active_sources),
            "learning_enabled": bool(learning_enabled),
            "source_tag_gain": float(source_tag_gain),
            "weight_l1_before": float(before["l1"]),
            "weight_l1_after": float(after["l1"]),
            "weight_l1_delta": float(after["l1"] - before["l1"]),
        }

    def commitment(self, core_local: np.ndarray) -> dict:
        """Per-core-cell committed source (argmax cumulative encoding firing) + the
        commitment-distribution entropy across the three sources (anti-cheat c)."""

        core_local = np.asarray(core_local, dtype=np.int64)
        counts = {s: 0 for s in SOURCES}
        for e in core_local:
            row = self._enc_fire[e]
            if float(row.sum()) <= 0.0:
                continue
            counts[SOURCES[int(np.argmax(row))]] += 1
        total = sum(counts.values())
        entropy_norm = 0.0
        if total > 0:
            probs = [counts[s] / total for s in SOURCES if counts[s] > 0]
            ent = -sum(p * math.log(p) for p in probs)
            entropy_norm = ent / math.log(len(SOURCES))  # 1.0 == uniform over 3 sources
        return {
            "core_cells": int(core_local.size),
            "committed_counts": counts,
            "commitment_entropy_norm": float(entropy_norm),
            "n_sources_committed": int(sum(1 for s in SOURCES if counts[s] > 0)),
        }

    # -- Cheat guard (b): prove recall is episode-only ---------------------------
    def verify_recall_episode_only(self, patterns: Sequence[np.ndarray]) -> dict:
        """For each pure-source recall pattern, drive the episode pattern with the
        BASE drive (as recall does) and verify the source afferent carries zero
        current and zero firing and the episode drive is unmodulated. This is a
        direct proof that no source-label / tag path exists at recall."""

        aff_idx = np.concatenate([self._source_afferent_indices[s] for s in SOURCES])
        drive_pA = float(self.config.drive_pA)
        aff_current_max = 0.0
        aff_spikes = 0.0
        ep_drive_ok = True
        for pattern in patterns:
            self.reset_dynamical_state()
            self._settle_to_quiescence()
            episode_global = self._episode_global_indices(pattern)
            self._drive(episode_global)  # base drive: NO afferent, NO tag
            cur = np.asarray(
                to_host(self.bridge.cp_external_input_current), dtype=np.float64
            )
            aff_current_max = max(aff_current_max, float(cur[aff_idx].max()))
            ep_cur = cur[episode_global]
            # episode cells must all carry exactly drive_pA (no tag boost leaked in)
            if float(ep_cur.max()) != drive_pA or float(ep_cur.min()) != drive_pA:
                ep_drive_ok = False
            for _ in range(int(self.config.read_steps)):
                self.bridge._run_one_simulation_step()
                firing = np.asarray(
                    to_host(self.bridge.cp_firing_states), dtype=np.float64
                )
                aff_spikes += float(firing[aff_idx].sum())
            self._rest()
        self.bridge.cp_external_input_current[:] = 0.0
        return {
            "afferent_current_max_at_recall": aff_current_max,
            "afferent_spikes_at_recall": aff_spikes,
            "episode_drive_unmodulated": bool(ep_drive_ok),
            "recall_is_episode_only": bool(
                aff_current_max == 0.0 and aff_spikes == 0.0 and ep_drive_ok
            ),
        }

    def probe_label_leak(self, pattern: Sequence[int], tag_source: str, tag_gain: float) -> str:
        """NON-VACUITY DIAGNOSTIC (not a gate): drive the recall pattern but FORCE
        the tag for ``tag_source`` on. If the tag carries source-label information,
        the recalled dominant source should move toward ``tag_source``. This proves
        the episode-only guard is excluding a REAL label path, not a no-op. Never
        used by the honest measurement."""

        self.reset_dynamical_state()
        self._settle_to_quiescence()
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        episode_global = self._episode_global_indices(pattern)
        pattern_local = np.asarray(pattern, dtype=np.int64)
        source_spikes = {s: 0.0 for s in SOURCES}
        try:
            self._drive_with_tag(
                episode_global, (tag_source,), pattern_local, float(tag_gain)
            )
            # remove the afferent drive component -- keep ONLY the episode+tag drive,
            # so this isolates the tag's effect on the episode layer at read time.
            xp, _ = get_backend()
            self.bridge.cp_external_input_current[
                xp.asarray(self._source_afferent_indices[tag_source], dtype=xp.int64)
            ] = 0.0
            for _ in range(int(self.config.read_steps)):
                self.bridge._run_one_simulation_step()
                firing = np.asarray(
                    to_host(self.bridge.cp_firing_states), dtype=np.float64
                )
                for s in SOURCES:
                    source_spikes[s] += float(firing[self._source_memory_indices[s]].sum())
        finally:
            self._rest()
            self.bridge.cp_external_input_current[:] = 0.0
        return max(SOURCES, key=lambda s: source_spikes[s])


def run_arm(
    seed: int,
    overlap_fraction: float,
    source_tag_gain: float,
    tag_frac: float,
    config: SourceMonitorConfigV2 | None = None,
) -> dict:
    """One conjunctive-tag arm at one (seed, overlap, source_tag_gain)."""

    c = config or SourceMonitorConfigV2()
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()

    # -- Instrument verification: zero-learned-weight control (no experience) -----
    ctrl = ConjunctiveTagGate(seed=seed + 30000, config=c, tag_frac=tag_frac)
    ctrl_on = _recall_three(ctrl, patterns)
    ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        ctrl_off = _recall_three(ctrl, patterns)
    finally:
        ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- Real arm: learn with the conjunctive tag at ENCODING, then recall --------
    intact = ConjunctiveTagGate(seed=seed, config=c, tag_frac=tag_frac)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True, source_tag_gain=source_tag_gain)
    intact.experience(patterns[1], auditory_activity=True, source_tag_gain=source_tag_gain)
    intact.experience(patterns[2], corollary_discharge=True, source_tag_gain=source_tag_gain)
    intact.experience(
        patterns[3], visual_activity=True, auditory_activity=True,
        source_tag_gain=source_tag_gain,
    )
    learned = intact.weight_summary()
    commit = intact.commitment(core)

    on = _recall_three(intact, patterns)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        off = _recall_three(intact, patterns)
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    # -- Cheat guard (b): recall is episode-only ---------------------------------
    episode_only = intact.verify_recall_episode_only(patterns[:3])
    # non-vacuity diagnostic: a tag forced ON at recall WOULD move the winner
    leak_probe = None
    if source_tag_gain > 0.0:
        leak_probe = {
            "recall_seen_forced_heard_tag_winner": intact.probe_label_leak(
                patterns[0], "heard", source_tag_gain
            ),
            "recall_seen_forced_self_tag_winner": intact.probe_label_leak(
                patterns[0], "self_generated", source_tag_gain
            ),
        }

    margins_M = {s: _source_margin(on[s], s) for s in SOURCES}
    margins_L = {s: _source_margin(off[s], s) for s in SOURCES}
    own_rate = {s: float(on[s]["source_rates"][s]) for s in SOURCES}
    dominant_correct = {s: bool(_dominant_source(on[s]) == s) for s in SOURCES}
    weakest_strict = bool(min(margins_M.values()) > min(margins_L.values()))
    weak_src = min(SOURCES, key=lambda s: margins_L[s])
    weak_attr = attributable_to(
        f"weakest-source ({weak_src}) margin vs SOURCE_COMPETITION_GATE",
        treatment_value=margins_M[weak_src],
        control_value=margins_L[weak_src],
    )

    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "source_tag_gain": float(source_tag_gain),
        "tag_frac": float(tag_frac),
        "core_size": int(core.size),
        "weights_learned_l1": float(learned["l1"]),
        "control_zero_weight_strict": control_strict,
        "commitment": commit,
        "recall_episode_only": episode_only,
        "label_leak_probe": leak_probe,
        "margins_M": margins_M,
        "margins_L": margins_L,
        "own_rate": own_rate,
        "min_margin_M": float(min(margins_M.values())),
        "min_margin_L": float(min(margins_L.values())),
        "min_own_rate": float(min(own_rate.values())),
        "weakest_source_strictly_improved": weakest_strict,
        "weakest_source_by_L": weak_src,
        "weak_margin_attributable_to_competition": weak_attr,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": bool(all(dominant_correct.values())),
        "clears_floor": bool(min(margins_M.values()) >= FLOOR),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def evaluate(
    seed: int,
    overlap_fraction: float,
    source_tag_gain: float,
    tag_frac: float,
) -> dict:
    """Treatment arm (source_tag_gain) + the gain=0 null baseline, all anti-cheats."""

    base = run_arm(seed, overlap_fraction, 0.0, tag_frac)
    treat = run_arm(seed, overlap_fraction, source_tag_gain, tag_frac)

    # (a) source_tag_gain=0 byte-identical to the original overlap NO-GO instrument.
    orig = _orig_evaluate_overlap(int(seed), float(overlap_fraction))
    byte_identical = bool(
        base["min_margin_M"] == orig["min_margin_M"]
        and base["min_margin_L"] == orig["min_margin_L"]
        and base["weights_learned_l1"] == orig["weights_learned_l1"]
    )

    # (d) reliability guard: dominant stays correct AND no source's own recall rate drops.
    no_rate_drop = all(
        treat["own_rate"][s] >= base["own_rate"][s] - 1e-9 for s in SOURCES
    )
    reliability_ok = bool(treat["all_dominant_correct"] and no_rate_drop)

    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "source_tag_gain": float(source_tag_gain),
        "tag_frac": float(tag_frac),
        "baseline_arm": base,
        "treatment_arm": treat,
        "anti_cheats": {
            "a_gain0_byte_identical_to_nogo": byte_identical,
            "b_recall_is_episode_only": bool(
                treat["recall_episode_only"]["recall_is_episode_only"]
                and base["recall_episode_only"]["recall_is_episode_only"]
            ),
            "b_afferent_current_max_at_recall": treat["recall_episode_only"][
                "afferent_current_max_at_recall"
            ],
            "b_afferent_spikes_at_recall": treat["recall_episode_only"][
                "afferent_spikes_at_recall"
            ],
            "b_label_leak_probe_would_move_winner": treat["label_leak_probe"],
            "c_commitment_spans_three_sources": bool(
                treat["commitment"]["n_sources_committed"] == 3
            ),
            "c_commitment_entropy_norm": treat["commitment"]["commitment_entropy_norm"],
            "d_reliability_preserved": reliability_ok,
            "d_no_own_rate_drop": bool(no_rate_drop),
            "e_zero_weight_control_strict_false": bool(
                not base["control_zero_weight_strict"]
                and not treat["control_zero_weight_strict"]
            ),
        },
        "headline": {
            "min_margin_M_treatment": treat["min_margin_M"],
            "min_margin_L_treatment": treat["min_margin_L"],
            "clears_floor_0.15": treat["clears_floor"],
            "beats_lesion_arm": bool(treat["min_margin_M"] > treat["min_margin_L"]),
            "min_margin_M_baseline": base["min_margin_M"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conjunctive source-tag at encoding for the source-monitor lane "
        "(numpy, deterministic)."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--overlaps", type=float, nargs="+", default=list(DEFAULT_OVERLAPS))
    parser.add_argument("--source-tag-gain", type=float, default=0.0)
    parser.add_argument("--tag-frac", type=float, default=DEFAULT_TAG_FRAC)
    parser.add_argument(
        "--mode",
        choices=("calibration", "development", "held_out"),
        default="calibration",
        help="seed-partition label recorded in the artifact (does not gate seeds).",
    )
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=None,
                        help="explicit seed override (parent-supplied validation seeds).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = args.dev_seeds if args.dev_seeds is not None else args.seeds
    rows = []
    for overlap in args.overlaps:
        for seed in seeds:
            row = evaluate(
                int(seed), float(overlap), float(args.source_tag_gain), float(args.tag_frac)
            )
            rows.append(row)
            ac = row["anti_cheats"]
            hl = row["headline"]
            print(
                "[conjunctive-tag] "
                f"seed={row['seed']} overlap={row['overlap_fraction']:.2f} "
                f"gain={row['source_tag_gain']:.2f} tagfrac={row['tag_frac']:.2f} | "
                f"minM={hl['min_margin_M_treatment']:.4f} "
                f"minL={hl['min_margin_L_treatment']:.4f} "
                f"minM_base={hl['min_margin_M_baseline']:.4f} | "
                f"floor={hl['clears_floor_0.15']} beatsL={hl['beats_lesion_arm']} | "
                f"a_byteid={ac['a_gain0_byte_identical_to_nogo']} "
                f"b_recall_epionly={ac['b_recall_is_episode_only']}"
                f"(aff_I={ac['b_afferent_current_max_at_recall']:.1f},"
                f"aff_spk={ac['b_afferent_spikes_at_recall']:.1f}) "
                f"c_3src={ac['c_commitment_spans_three_sources']}"
                f"({ac['c_commitment_entropy_norm']:.2f}) "
                f"d_reliab={ac['d_reliability_preserved']} "
                f"e_ctrl={ac['e_zero_weight_control_strict_false']}",
                flush=True,
            )

    out = {
        "runner": "research/runners/_laneC_source_monitor_conjunctive_tag.py",
        "mode": args.mode,
        "seeds": list(seeds),
        "overlaps": list(args.overlaps),
        "source_tag_gain": float(args.source_tag_gain),
        "tag_frac": float(args.tag_frac),
        "floor": FLOOR,
        "instrument": "v6 fixed (per-recall reset_dynamical_state) + conjunctive tag at encoding",
        "mechanism": "Komorowski-Manns-Eichenbaum 2009 item-in-context conjunctive cells "
        "(encoding-only weak source-specific overlap modulation; recall episode-only)",
        "rows": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[conjunctive-tag] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
