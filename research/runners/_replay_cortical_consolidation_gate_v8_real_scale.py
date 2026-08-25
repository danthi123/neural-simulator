"""v8: PORT the v7 balanced directed-sweep + order-STDP consolidation mechanism off the 72-neuron
TOY calibration circuit onto a REAL-SCALE CA1 substrate (board #130's next rung, 2026-08-26).

WHY THIS EXISTS.
-----------------
v7 (``_replay_cortical_consolidation_gate_v7_balanced_order.py``) is 6/6 GO on the decisive seeds
(``research/findings/2026-08-25-order-consolidation-recalib-...GO.md``): balanced directed-sweep
replay + isolated order-STDP makes ordered replay strengthen the cortical cue->target sequence trace
more than shuffled, causally attributed to the substrate's own STDP. But every version in this chain
(v1..v7) runs the SAME hand-picked TOY population: n_ca3=72, n_ca1=n_cue=n_target=48, assemblies of
16-24 cells, wired ALL-TO-ALL. That is a real ``sim.bridge.SimulationBridge`` (no host-computed
plasticity), but it is a small idealised calibration circuit, not a network at anything like a
biologically-plausible hippocampal scale.

THE PORT.
---------
Two changes, nothing else in v7 touched:

1. EVERY extensive (population / assembly) GateConfig field is multiplied by ``SCALE_FACTOR=25``,
   landing CA3 at n_ca3=1800 -- close to this project's OWN established "real" hippocampal-CA3 scale
   convention (n_ca3=2000, the production D5 episodic organ,
   ``research/runners/_episodic_dap_dialogue_memory.py``). Every INTENSIVE parameter (per-synapse
   weights, drive currents in pA, learning rates, STDP/SFA per-neuron amplitudes, event/step counts)
   is inherited UNCHANGED from v7 -- those are biophysical properties of a single neuron or synapse,
   not properties of population size, and must NOT scale with N.

2. Wiring density. v1-v7 build every projection ALL-TO-ALL (``_all_to_all``, self-edges excluded for
   recurrent populations). At 25x bigger assemblies, naive all-to-all would give each postsynaptic
   neuron 25x MORE converging synapses at the SAME per-synapse weight -- 25x more aggregate drive,
   silently retuning the whole calibrated operating point (encode/sleep/probe drive currents, Hebbian
   and STDP rates, SFA eviction strength) rather than testing the mechanism on a bigger substrate. So
   every ``_all_to_all`` call is routed through a SPARSE random bipartite projection at an in-degree
   of ``round(len(pre) / SCALE_FACTOR)``. Because every extensive population/assembly field was scaled
   by the SAME uniform factor, this recovers the EXACT toy in-degree for every one of the 9 wiring
   populations (verified by hand below), while genuinely growing the network and making its
   connectivity properly sparse -- a real step toward the "sparse, near-disjoint" CA1 code this
   project's other findings establish biologically (not "same toy, bigger label").
   Per-pathway check (pre size at SCALE_FACTOR=25 -> indegree = round(pre/25) -> toy indegree):
     ca3 recurrent            600 -> 24  (toy: 24, off-by-1 for the excluded self-edge, negligible)
     ca3 -> ca1                600 -> 24  (toy: 24)
     ca1 -> cortical_cue       400 -> 16  (toy: 16)
     ca1 -> cortical_target    400 -> 16  (toy: 16, the v5 reinstatement wire)
     cortical_cue -> target   1200 -> 48  (toy: 48 -- this one spans the FULL region, not an assembly)
     target recurrent          400 -> 16  (toy: 16, off-by-1 as above)
     target -> FS               400 -> 16  (toy: 16)
     ca1 -> FS                  400 -> 16  (toy: 16)
     FS -> target (opponent)   150 ->  6  (toy: 6)
   Every pathway lands on its exact toy in-degree.

THE v5 "REINSTATEMENT_MEMORY_SPECIFIC" PRECONDITION, RECOMPUTED FOR SPARSE WIRING.
------------------------------------------------------------------------------------
``v5.build_bridge`` asserts the CA1->cortical_target reinstatement wire is memory-specific by
comparing its edge COUNT to the dense all-to-all product per memory. That equality assumes dense
wiring and would read False on every sparse build here even though sparsification NEVER creates a
cross-memory edge (each ``_all_to_all`` call receives one memory's pre/post pair at a time). Rather
than weaken or skip that precondition, this runner recomputes the actual invariant it protects
DIRECTLY FROM THE INSTALLED SUBSTRATE (the real pre/post neuron ids of every synapse under
``INDEX_TARGET_GATE``, via ``cp_connections.tocoo()``) -- a genuine bridge-truth check, not a
trusted-by-construction claim.

Everything else -- the balanced directed-sweep replay plan, isolated order-STDP, intrinsic SFA
one-of-N eviction, learned CA1->cortex reinstatement, every anti-cheat control (stdp-off power
control, the four causal lesions, the shuffled-order temporal control) -- is v7, byte-for-byte.

GO bar (inherited from v7, unchanged): >=5/6 decisive seeds pass
``intact_beats_shuffled_order`` (margin >= +0.01) AND both memories recovered AND the stdp-off
power control collapses the margin AND lesions ~0.

Numpy decisive (this arc's own established primary-decisive backend; see v7's
``v7_decisive_numpy.json``, cupy confirmation queued the same way there):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v8_real_scale \\
        --seeds 42 43 44 100 101 102 --out research/findings/raw/order_recalib/v8_real_scale_decisive_numpy.json

Small-scale smoke (fast correctness check, NOT a scientific verdict):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v8_real_scale \\
        --smoke-scale 3 --seeds 42
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

from research.runners import _replay_cortical_consolidation_gate as v1  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v5 as v5  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v6_order_stdp as v6  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v7_balanced_order as v7  # noqa: E402

DECISIVE_SEEDS = v7.DECISIVE_SEEDS

# The uniform real-scale factor applied to EVERY extensive (population/assembly) config field, and to
# the sparsify in-degree denominator (see module docstring for the per-pathway indegree recovery).
# 25 lands CA3 at 1800, close to this project's own established "real" CA3 scale (2000, the
# production D5 episodic organ).
SCALE_FACTOR = 25


def _scaled_fields(scale: int) -> dict:
    return dict(
        n_ca3=72 * scale,
        n_ca1=48 * scale,
        n_cue=48 * scale,
        n_target=48 * scale,
        n_target_fs=12 * scale,
        ca3_assembly=24 * scale,
        ca3_overlap=0,
        ca1_assembly=16 * scale,
        cue_assembly=16 * scale,
        cue_overlap=6 * scale,
        target_assembly=16 * scale,
        sleep_noise_cells=12 * scale,
    )


@dataclass(frozen=True)
class GateConfig(v7.GateConfig):
    """v7's mechanism, UNCHANGED, at SCALE_FACTOR x the toy's population/assembly sizes. Every
    intensive field (weights, drive currents, learning rates, STDP/SFA amplitudes, event/step
    counts) is inherited from v7 verbatim -- see the module docstring for why those must not scale.
    """

    n_ca3: int = 72 * SCALE_FACTOR
    n_ca1: int = 48 * SCALE_FACTOR
    n_cue: int = 48 * SCALE_FACTOR
    n_target: int = 48 * SCALE_FACTOR
    n_target_fs: int = 12 * SCALE_FACTOR
    ca3_assembly: int = 24 * SCALE_FACTOR
    ca3_overlap: int = 0
    ca1_assembly: int = 16 * SCALE_FACTOR
    cue_assembly: int = 16 * SCALE_FACTOR
    cue_overlap: int = 6 * SCALE_FACTOR
    target_assembly: int = 16 * SCALE_FACTOR
    sleep_noise_cells: int = 12 * SCALE_FACTOR


def make_config(scale: int = SCALE_FACTOR, **overrides) -> GateConfig:
    """A GateConfig at an arbitrary scale (default SCALE_FACTOR, the "real" port). Used for cheap
    smoke-scale variants (``scale=2`` or ``3``) that share every mechanism knob and the sparsify
    machinery -- only the network size differs, so a correctness bug is caught in seconds rather
    than the ~25 minutes the full real-scale decisive gate costs (see `_order_row` docstring below).
    """
    fields = _scaled_fields(scale)
    fields.update(overrides)
    return GateConfig(**fields)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# SPARSE wiring: a random bipartite projection at a fixed in-degree per postsynaptic neuron, NOT
# dense all-to-all. Monkeypatches the ONE choke point every wiring builder in v1/v2/v5/v5s/v6 routes
# through (`v1._all_to_all`, called from `v2._population`/`v2._merge_pairs` and directly from both
# v1.build_bridge and v5.build_bridge) -- the same idiom v7 itself already uses to swap in the
# directed-sweep replay plan (`v2._ordered_sleep_events`), scoped here to one build via try/finally.
# ────────────────────────────────────────────────────────────────────────────────────────────────
_ORIG_ALL_TO_ALL = v1._all_to_all
_ORIG_V6_BUILD_BRIDGE = v6.build_bridge


def _make_sparse_all_to_all(seed: int, scale: int):
    """Deterministic given `seed` (its own RNG stream -- multiplier 211/offset 3 is unused by every
    other seed-derived draw in the v1..v7 chain, checked by grep). In-degree = round(len(pre)/scale)
    per postsynaptic neuron, which -- because every extensive population/assembly size here is
    exactly toy_value*scale -- recovers the ORIGINAL toy in-degree for every pathway (see module
    docstring). `self_edges=False` excludes the postsynaptic neuron itself from its own candidate
    pool (used for the two recurrent populations, ca3 and cortical_target)."""
    rng = np.random.default_rng(int(seed) * 211 + 3)

    def _sparse(pre, post, *, self_edges: bool = True):
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        if pre.size == 0 or post.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        indegree = max(1, int(round(pre.size / scale)))
        edge_pre_parts = []
        edge_post_parts = []
        for p in post.tolist():
            cand = pre if self_edges else pre[pre != p]
            k = min(indegree, cand.size)
            if k <= 0:
                continue
            chosen = rng.choice(cand, size=k, replace=False)
            edge_pre_parts.append(chosen)
            edge_post_parts.append(np.full(k, p, dtype=np.int64))
        if not edge_pre_parts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return (
            np.concatenate(edge_pre_parts).astype(np.int64),
            np.concatenate(edge_post_parts).astype(np.int64),
        )

    return _sparse


def _verify_reinstatement_memory_specific(bridge, handles: dict) -> bool:
    """Bridge-truth replacement for v5's dense-count-equality check (see module docstring): every
    installed synapse under INDEX_TARGET_GATE (the CA1->cortical_target reinstatement wire) must
    connect a memory's own ca1 assembly to that SAME memory's target assembly, never the other
    memory's -- read directly from the substrate's connection matrix, not trusted from the
    sparsify code's own construction."""
    from sim.backend import to_host

    idx = np.asarray(to_host(bridge._plasticity_gate_indices_gpu[v5.INDEX_TARGET_GATE]), dtype=np.int64)
    if idx.size == 0:
        return False
    coo = bridge.cp_connections.tocoo(copy=False)
    row = np.asarray(to_host(coo.row), dtype=np.int64)[idx]
    col = np.asarray(to_host(coo.col), dtype=np.int64)[idx]
    pat = handles["patterns"]
    ca1_a, ca1_b = set(pat["A"]["ca1"].tolist()), set(pat["B"]["ca1"].tolist())
    tgt_a, tgt_b = set(pat["A"]["target"].tolist()), set(pat["B"]["target"].tolist())
    for p, c in zip(row.tolist(), col.tolist()):
        if p in ca1_a:
            if c not in tgt_a:
                return False
        elif p in ca1_b:
            if c not in tgt_b:
                return False
        else:
            return False
    return True


def build_bridge(seed: int, config) -> tuple[object, dict]:
    """v6's build_bridge (v5 anatomy + STDP allocation), with sparse convergence-preserving wiring
    installed for the duration of this ONE build (restored after, so no global state leaks to any
    other module sharing this process -- several of these gate modules get imported together).

    The sparsify in-degree denominator is derived from THIS config's own ``n_ca3`` (``n_ca3/72``),
    not the module-level ``SCALE_FACTOR`` constant -- ``make_config(scale=...)`` builds smoke-scale
    configs at scales other than SCALE_FACTOR, and the in-degree recovery in the module docstring
    only holds when the sparsify denominator matches the ACTUAL size multiplier used to build this
    particular config (a smoke run at scale=2 sparsified at denominator 25 gives indegree~2 -- far
    below the tuned operating point -- and was caught exactly this way during development)."""
    effective_scale = float(config.n_ca3) / 72.0
    v1._all_to_all = _make_sparse_all_to_all(seed, effective_scale)
    try:
        bridge, handles = _ORIG_V6_BUILD_BRIDGE(seed, config)
    finally:
        v1._all_to_all = _ORIG_ALL_TO_ALL
    handles["reinstatement_memory_specific"] = _verify_reinstatement_memory_specific(bridge, handles)
    return bridge, handles


def _order_row(seed: int, config: GateConfig) -> dict:
    """v7's per-seed order-gate row (mechanism, controls, stdp-off power control -- byte-for-byte),
    with `v6.build_bridge` swapped for the sparse real-scale version for the duration of the call.
    ~25 minutes for the full SCALE_FACTOR=25 decisive gate on numpy (6 seeds); a `make_config(scale=3)`
    smoke config runs in seconds."""
    v6.build_bridge = build_bridge
    try:
        return v7._order_row(seed, config)
    finally:
        v6.build_bridge = _ORIG_V6_BUILD_BRIDGE


def run_decisive(seeds: Iterable[int], config: GateConfig | None = None) -> dict:
    cfg = config or GateConfig()
    v6.build_bridge = build_bridge
    try:
        checked = tuple(int(s) for s in seeds)
        started = time.time()
        rows = [_order_row(s, cfg) for s in checked]
    finally:
        v6.build_bridge = _ORIG_V6_BUILD_BRIDGE
    n = len(rows)
    n_order = sum(r["seed_order_go"] for r in rows)
    n_beats = sum(r["intact_beats_shuffled_order"] for r in rows)
    n_both = sum(r["both_memories_recovered"] for r in rows)

    from tools.lab import attributable_to
    from tools.verdict import Verdict

    # Aggregate (across-seed) attribution, complementing the per-seed `order_stdp_attribution` v7
    # already computes in each row: whose is the real-scale order margin -- the substrate's own
    # order-sensitive STDP (treatment: the mean intact-vs-shuffled order margin with STDP on), or
    # something running identically in both arms (control: the SAME directed-sweep replay with
    # stdp_sleep=False)? A fraction near 1.0 attributes the margin to STDP, not to host replay
    # bookkeeping -- exactly the question CLAUDE.md's gap#5 lesson says never to skip when a
    # treatment/control pair is banked side by side.
    mean_order_margin = float(np.mean([r["order_recovery_margin"] for r in rows]))
    mean_stdp_off_margin = float(np.mean([r["stdp_off_order_margin"] for r in rows]))
    aggregate_order_stdp_attribution = attributable_to(
        "real-scale-CA1 order-consolidation margin owed to order-sensitive STDP (aggregate across seeds)",
        mean_order_margin, mean_stdp_off_margin,
    )

    earned = Verdict("v8 real-scale-CA1 order-consolidation decisive gate")
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
        why="isolate the order-sensitive spike-timing consolidation on the directed-sweep replay (inherited from v7)",
    )
    decided = earned.decide(go=(n_order >= 5), verbose=False)
    verdict = decided["status"]
    return {
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "gate": "replay_cortical_consolidation_v8_real_scale_ca1_port",
        "phase": "decisive_multiseed",
        "mechanism": (
            "v7 balanced directed-sweep order-STDP consolidation (unchanged), PORTED onto a "
            f"{SCALE_FACTOR}x-scaled CA1 substrate (n_ca3={cfg.n_ca3}) with sparse convergence-"
            "preserving wiring instead of the toy's dense all-to-all"
        ),
        "scale_factor": SCALE_FACTOR,
        "toy_gate": "replay_cortical_consolidation_v7_balanced_directed_sweep_order",
        "replay_plan": cfg.replay_plan,
        "go_bar": "intact_beats_shuffled_order (margin>=+0.01) AND both_memories_recovered AND stdp_owns_order AND lesions~0, on >=5/6 seeds",
        "verdict": verdict,
        "n_seeds": n,
        "n_seed_order_go": n_order,
        "n_beats_shuffled_order": n_beats,
        "n_both_recovered": n_both,
        "mean_order_recovery_margin": mean_order_margin,
        "mean_stdp_off_order_margin": mean_stdp_off_margin,
        "aggregate_order_stdp_attribution": aggregate_order_stdp_attribution,
        "seeds": list(checked),
        "backend": __import__("os").environ.get("SIM_BACKEND", "unset"),
        "rows": rows,
        "remaining_scaffolds": [
            "host-scheduled directed replay sweep (stored-trajectory drive) -- inherited from v7",
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries",
            "fixed assembly anatomy",
            "SFA parameters (d/a) and STDP amplitudes/bounds set at build, not developmentally tuned",
            f"sparse wiring is a fixed in-degree RANDOM projection (round(pre/{SCALE_FACTOR})), not "
            "a developmentally self-organized (e.g. activity-dependent pruning) connectivity",
            "assembly:region size ratio (~duplicate of the toy's ~33-67%) is NOT the biological "
            "sparse (~1-5% active) CA1 code; a further scale-up toward true sparsity is the next rung",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DECISIVE_SEEDS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--replay-plan", default="directed_sweep",
                    choices=["directed_sweep", "episode_agnostic"])
    ap.add_argument("--smoke-scale", type=int, default=None,
                     help="Build at this (small) scale instead of SCALE_FACTOR, for a fast correctness check.")
    args = ap.parse_args()
    if args.smoke_scale is not None:
        cfg = make_config(scale=args.smoke_scale, replay_plan=args.replay_plan)
    else:
        cfg = GateConfig(replay_plan=args.replay_plan)
    payload = run_decisive(args.seeds, cfg)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
