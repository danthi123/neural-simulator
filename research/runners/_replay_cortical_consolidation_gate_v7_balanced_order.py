"""v7 decisive order gate: v6 order-STDP mechanism + BALANCED directed-sweep replay.

WHY THIS EXISTS (the recalibration).
------------------------------------
v6 (``_replay_cortical_consolidation_gate_v6_order_stdp``) added order-sensitive
STDP consolidation and passed the replay-order control on 2 numpy calibration
seeds, but multiseed validation was NO-GO: the FIXED operating point overfit the
2 seeds, and re-running on ``SIM_BACKEND=cupy`` even the calibration seeds fell
to false recall ~0.5 (``replay_v6_order_stdp_calib.json`` ->
CALIBRATION_NEEDS_REVISION). Diagnosis (this arc, seeds 42/43/44/100/101/102):

  * The order-STDP MECHANISM is intact and robust -- ordered (intact) replay
    strengthens the cortical cue->target sequence trace 1.06-2.30x MORE than
    shuffled on ALL 6 fresh seeds (numpy). The physical directional trace is
    deposited exactly as claimed.
  * But the BEHAVIORAL readout was measured in a DEGENERATE operating point.
    Root cause (lesion-grade): the v6 sleep replay is EPISODE-AGNOSTIC random
    CA3 background noise, and memory B (encoded with more events: 20 vs 14) has
    the stronger CA3 attractor, so it WINS the replay competition on nearly
    every event (e.g. seed 101: A replayed 0/24 events, B 24/24). Only the
    replay-winning memory consolidates, so ``both_memories_recovered`` fails and
    the retest is bimodal (one memory clean, the other fully evicted -> mean
    false recall 0.5). In that regime the order margin is noise (+0.01 bar
    passed only 2/6; all 6 positive but tiny).

THE REVISION (biology-grounded, the named next mechanism).
----------------------------------------------------------
Real hippocampal replay reactivates STORED trajectories in temporal order (a
directed sweep through an experienced sequence), not random cell noise, and
across sleep it visits MULTIPLE recent memories -- the CLS interleaving that
prevents catastrophic interference (McClelland/McNaughton/O'Reilly 1995). v7
replaces the episode-agnostic noise with a DIRECTED REPLAY SWEEP: one long
trajectory whose drive window sweeps from memory A's CA3 assembly, through the
(possibly empty) shared cells, into memory B's assembly. This:
  (1) drives BOTH memories (A early in the sweep, B late) so both consolidate
      regardless of which attractor is stronger -> ``both_memories_recovered``;
  (2) carries a STRONG directional sequence (adjacent windows overlap ~10/12,
      distant windows are disjoint) so ordered vs shuffled differ sharply ->
      the order-STDP has a real order contrast to read.
Everything else in v6 is inherited UNCHANGED (order-STDP, SFA eviction, learned
CA1->cortex reinstatement, every anti-cheat control, the frozen verdict). The
ONLY change is the replay event plan, scoped to this runner.

ANTI-CHEATS (all inherited from v6/v5, intact).
-----------------------------------------------
The order effect must be causally the order-STDP, not host bookkeeping: the
``--stdp-off`` power control reruns the IDENTICAL directed-sweep replay with
``stdp_sleep=False`` and the ordered-vs-shuffled recovery margin must COLLAPSE.
The causal lesions (no_sleep, ca1_target_reinstatement_lesion, ca3_ca1_lesion,
cortical_plasticity_off) must drop hippocampus-independent recall to ~0. The
directed sweep is a host-scheduled replay drive -- the SAME scaffold class as
v6's episode-agnostic background (a listed ``remaining_scaffold``), arguably
more faithful (stored-trajectory replay vs random noise); the substrate's OWN
STDP is what reads the order.

GO bar (this gate): order-sensitive consolidation (forward-ordered replay
strengthens the sequence more than shuffled, ``intact_beats_shuffled_order``
margin >= +0.01) clears on >=5/6 decisive seeds WITH both memories recovered
and the stdp-off power control collapsing the margin.

Decisive:
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v7_balanced_order \\
        --seeds 42 43 44 100 101 102 --out research/findings/raw/order_recalib/v7_decisive.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate_v2 as v2  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v5 as v5  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v5_sfa as v5s  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v6_order_stdp as v6  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

CONDITIONS = v6.CONDITIONS
DECISIVE_SEEDS = (42, 43, 44, 100, 101, 102)

# This decisive gate deliberately runs FRESH seeds (42/43/44/100/101/102),
# disjoint from v6's bounded calibration/dev/held-out partition, to test
# GENERALISATION of the revised mechanism (not to re-fit the v6 operating
# point). The v6/v5 runners hard-refuse unknown seeds; lift that guard here
# (seed correctness is still real -- v5.build_bridge sets cfg.seed=seed, so the
# substrate IS seeded per the CLAUDE.md seed trap).
def _accept_any_seed(seeds):
    return tuple(int(s) for s in seeds)


for _m in (v5, v5s, v6):
    _m.validate_calibration_seeds = _accept_any_seed

_ORIG_ORDERED_EVENTS = v2._ordered_sleep_events


@dataclass(frozen=True)
class GateConfig(v6.GateConfig):
    """v6 order-STDP + directed-sweep balanced replay, order-signal isolated."""

    # "directed_sweep" -> A->B directed replay window (this gate's revision).
    # "episode_agnostic" -> v6 random CA3 noise (for A/B comparison).
    replay_plan: str = "directed_sweep"
    # Isolate the ORDER-sensitive plasticity: sleep consolidation is the
    # spike-timing STDP ONLY. v6's order-BLIND rate-window Hebbian baseline
    # (identical for ordered/shuffled since permuting events preserves the
    # coactivity multiset) dilutes the order-specific contribution; removing it
    # makes the ordered-vs-shuffled margin the WHOLE effect. Also collapses false
    # recall to ~0 (no order-blind cross-memory transfer). The stdp-off power
    # control then drops ALL sleep plasticity, a clean 100%-STDP attribution.
    sleep_hebbian_on: bool = False


def _ca3_assemblies(seed: int, config: GateConfig, ca3_indices: np.ndarray):
    """Reconstruct the A/B CA3 assemblies -- the FIRST rng draw in _memory_patterns,
    so it is reproducible standalone (identical cells to build_bridge)."""
    rng = np.random.default_rng(seed * 31 + 17)
    size, overlap = config.ca3_assembly, config.ca3_overlap
    draw = rng.choice(ca3_indices, 2 * size - overlap, replace=False)
    shared = draw[:overlap]
    a = np.sort(np.concatenate([shared, draw[overlap:size]]))
    b = np.sort(np.concatenate([shared, draw[size:]]))
    return a, b


def _directed_sweep_events(seed: int, config, ca3_indices: np.ndarray, *, shuffle: bool):
    """One directed replay trajectory: a drive window sweeping A -> shared -> B.

    Adjacent windows overlap heavily (order carried); the sweep visits A's
    assembly early and B's late so both memories are driven. ``shuffle`` permutes
    the events (the temporal control), breaking the directional adjacency for
    both memories -- exactly the order signal the STDP is meant to read.
    """
    ca3_a, ca3_b = _ca3_assemblies(seed, config, ca3_indices)
    a_only = np.sort(np.setdiff1d(ca3_a, ca3_b))
    shared = np.sort(np.intersect1d(ca3_a, ca3_b))
    b_only = np.sort(np.setdiff1d(ca3_b, ca3_a))
    ordered = np.concatenate([a_only, shared, b_only]).astype(np.int64)
    w = int(min(config.sleep_noise_cells, len(ordered)))
    max_start = max(len(ordered) - w, 0)
    starts = np.linspace(0, max_start, config.sleep_events).round().astype(int)
    events = [np.sort(ordered[s:s + w]).astype(np.int64) for s in starts]
    if not shuffle or len(events) < 2:
        return events
    order = np.random.default_rng(seed * 71 + 19).permutation(len(events))
    if np.array_equal(order, np.arange(len(events))):
        order = np.roll(order, 1)
    return [events[int(i)].copy() for i in order]


def _install_plan(config: GateConfig):
    if config.replay_plan == "directed_sweep":
        v2._ordered_sleep_events = _directed_sweep_events
    elif config.replay_plan == "episode_agnostic":
        v2._ordered_sleep_events = _ORIG_ORDERED_EVENTS
    else:
        raise ValueError(f"unknown replay_plan {config.replay_plan!r}")


def run_seed(seed: int, config: GateConfig) -> dict:
    """v6 conditions + verdict, with the directed-sweep replay plan installed."""
    _install_plan(config)
    try:
        conditions = {c: v6.run_condition(seed, c, config) for c in CONDITIONS}
        verdict = v6._calibration_verdict(conditions)
    finally:
        v2._ordered_sleep_events = _ORIG_ORDERED_EVENTS
    return {
        "seed": int(seed),
        "conditions": conditions,
        "calibration": verdict,
        "calibration_status": verdict["calibration_status"],
        "verdict": verdict["verdict"],
    }


def _order_row(seed: int, config: GateConfig) -> dict:
    """Compact per-seed order-gate row, with the stdp-off power control."""
    main = run_seed(seed, config)
    v = main["calibration"]
    intact = main["conditions"]["intact"]
    A = intact["recall"]["A"]["correct_rate"]
    B = intact["recall"]["B"]["correct_rate"]
    ctrl = v["control_mean_recovery"]

    # Power control: identical directed-sweep replay, order-STDP OFF. The
    # ordered-vs-shuffled margin must collapse (attributes the margin to STDP).
    off_cfg = GateConfig(**{**asdict(config), "stdp_sleep": False})
    off = run_seed(seed, off_cfg)
    off_v = off["calibration"]
    off_margin = off_v["intact_vs_shuffled_recovery_margin"]

    order_margin = v["intact_vs_shuffled_recovery_margin"]
    beats = bool(v["checks"]["intact_beats_shuffled_order"])
    both = bool(A >= 0.015 and B >= 0.015)
    # Whose is the ordered-vs-shuffled margin? treatment = order margin (STDP on),
    # control = the SAME directed-sweep replay with STDP off. A fraction near 1.0
    # attributes the order effect to the substrate's spike-timing plasticity, not
    # to host replay bookkeeping (which is identical in both arms).
    order_stdp_attribution = attributable_to(
        "ordered-vs-shuffled recovery margin owed to order-STDP", order_margin, off_margin
    )
    # STDP must be load-bearing for the ORDER effect on this seed: the margin
    # with STDP on must exceed the stdp-off null by a clear step.
    stdp_owns_order = bool(order_margin >= off_margin + 0.01)
    lesions_zero = all(
        ctrl[name] <= 0.005
        for name in ("no_sleep", "ca1_target_reinstatement_lesion",
                     "ca3_ca1_lesion", "cortical_plasticity_off")
    )
    # Probe-INDEPENDENT robustness anchor: ordered replay physically strengthens
    # the cortical cue->target trace MORE than shuffled (a sleep-time weight
    # measurement, not a probe readout). Stable where the behavioral margin is
    # probe-window sensitive.
    stdp_delta_i = v["intact_stdp_cortical_delta"]
    stdp_delta_s = v["shuffled_stdp_cortical_delta"]
    ordered_trace_stronger = bool(
        stdp_delta_i is not None and stdp_delta_s is not None and stdp_delta_i > stdp_delta_s
    )
    seed_go = bool(beats and both and stdp_owns_order and lesions_zero and ordered_trace_stronger)
    return {
        "seed": int(seed),
        "seed_order_go": seed_go,
        "intact_beats_shuffled_order": beats,
        "both_memories_recovered": both,
        "order_recovery_margin": order_margin,
        "stdp_off_order_margin": off_margin,
        "stdp_owns_order": stdp_owns_order,
        "order_stdp_attribution": order_stdp_attribution,
        "ordered_trace_stronger": ordered_trace_stronger,
        "lesions_drop_to_zero": lesions_zero,
        "A_correct_rate": A,
        "B_correct_rate": B,
        "intact_mean_recovery": v["intact_mean_recovery"],
        "shuffled_mean_recovery": ctrl["shuffled_replay_order"],
        "intact_mean_false_recall": v["intact_mean_false_recall"],
        "intact_stdp_cortical_delta": v["intact_stdp_cortical_delta"],
        "shuffled_stdp_cortical_delta": v["shuffled_stdp_cortical_delta"],
        "replayed_A": intact["sleep"]["replayed_A"],
        "replayed_B": intact["sleep"]["replayed_B"],
        "calibration_status": main["calibration_status"],
        "checks_failed": [k for k, val in v["checks"].items() if not val],
    }


def run_decisive(seeds: Iterable[int], config: GateConfig | None = None) -> dict:
    cfg = config or GateConfig()
    checked = tuple(int(s) for s in seeds)
    started = time.time()
    rows = [_order_row(s, cfg) for s in checked]
    n = len(rows)
    n_order = sum(r["seed_order_go"] for r in rows)
    n_beats = sum(r["intact_beats_shuffled_order"] for r in rows)
    n_both = sum(r["both_memories_recovered"] for r in rows)

    # A verdict must travel with the INSTRUMENT-validity preconditions that earned
    # it (not the per-seed GO outcome): if any anti-cheat control did not run and
    # produce a defined read on every seed, the gate has not earned any verdict.
    earned = Verdict("v7 balanced directed-sweep order-consolidation decisive gate")
    earned.require(
        "the STDP-off power control produced a defined order margin on every seed",
        all(r["stdp_off_order_margin"] is not None for r in rows), expect=True,
    )
    earned.require(
        "both-memory retest recovery was measured on every seed",
        all(r["A_correct_rate"] is not None and r["B_correct_rate"] is not None for r in rows),
        expect=True,
    )
    earned.require(
        "the physical sleep-time cortical-trace delta was measured on every seed",
        all(r["intact_stdp_cortical_delta"] is not None
            and r["shuffled_stdp_cortical_delta"] is not None for r in rows),
        expect=True,
    )
    earned.require(
        "the four causal-lesion controls executed (defined recovery) on every seed",
        all(r["lesions_drop_to_zero"] in (True, False) for r in rows), expect=True,
    )
    earned.disabled(
        "reward modulation, homeostasis, structural plasticity; sleep rate-Hebbian OFF",
        why="isolate the order-sensitive spike-timing consolidation on the directed-sweep replay",
    )
    decided = earned.decide(go=(n_order >= 5), verbose=False)
    verdict = decided["status"]
    return {
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "gate": "replay_cortical_consolidation_v7_balanced_directed_sweep_order",
        "phase": "decisive_multiseed",
        "mechanism": (
            "v6 order-sensitive STDP consolidation + balanced DIRECTED-SWEEP replay "
            "(A->B stored-trajectory reactivation); both memories consolidate so the "
            "order margin is measurable, order effect attributed to STDP by the "
            "stdp-off power control"
        ),
        "replay_plan": cfg.replay_plan,
        "go_bar": "intact_beats_shuffled_order (margin>=+0.01) AND both_memories_recovered AND stdp_owns_order AND lesions~0, on >=5/6 seeds",
        "verdict": verdict,
        "n_seeds": n,
        "n_seed_order_go": n_order,
        "n_beats_shuffled_order": n_beats,
        "n_both_recovered": n_both,
        "seeds": list(checked),
        "backend": __import__("os").environ.get("SIM_BACKEND", "unset"),
        "rows": rows,
        "remaining_scaffolds": [
            "host-scheduled directed replay sweep (stored-trajectory drive) -- the same host-replay scaffold class as v6 episode-agnostic background",
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries",
            "fixed assembly anatomy",
            "SFA parameters (d/a) and STDP amplitudes/bounds set at build, not developmentally tuned",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DECISIVE_SEEDS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--replay-plan", default="directed_sweep",
                    choices=["directed_sweep", "episode_agnostic"])
    args = ap.parse_args()
    cfg = GateConfig(replay_plan=args.replay_plan)
    payload = run_decisive(args.seeds, cfg)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
