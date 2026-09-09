"""v9: a further scale-up toward TRUE biological sparsity, decoupling the CA1/CA3/cortical REGION
size from the ASSEMBLY size (board #130's own named next rung after v8, 2026-09-08).

WHY THIS EXISTS.
-----------------
v8 (``_replay_cortical_consolidation_gate_v8_real_scale.py``) ported the v7 balanced directed-sweep
+ order-STDP consolidation mechanism onto a real-scale CA1 substrate (n_ca3=1800) with genuinely
sparse (not all-to-all) wiring -- 6/6 GO, unchanged margins. But v8's own finding named an explicit
residual it did NOT close (``2026-08-25-order-consolidation-v8-real-scale-ca1-port-6seed-GO.md``,
"What this does and does not establish", item 3): v8 scaled REGION and ASSEMBLY by the exact SAME
factor (25x), so the assembly:region ratio stayed an exact duplicate of the hand-picked toy's
~33-67% -- nowhere near the ~1-5% active sparse code this project's other CA1/hippocampal findings
establish biologically. The finding is explicit that closing this "would need a SEPARATE de-risk
since it changes the assembly:region ratio the sparsify indegree formula [v8] assumes stays fixed."
This is that de-risk.

THE PORT (one change, isolated).
---------------------------------
v8's sparsify function derives in-degree as ``round(pre.size / effective_scale)`` where
``effective_scale = config.n_ca3 / 72`` -- a SINGLE scale factor uniformly applied to both region
and assembly, which only recovers the toy's per-pathway in-degree because every population/assembly
field happens to be exactly ``toy_value * effective_scale``. Growing ONLY the region (n_ca3, n_ca1,
n_cue, n_target) while holding every assembly/overlap/FS field at v8's absolute values breaks that
assumption for the ONE pathway that wires the FULL region rather than an assembly
("cortical_association", cue-region -> target-region, v8's docstring: "spans the FULL region, not
an assembly") -- its pre-population size grows with the region, so the ratio formula would silently
inflate its in-degree instead of holding it fixed.

The fix: in-degree is a property of the POSTSYNAPTIC NEURON's dendritic convergence (how many
synapses land on one cell), not of the size of the population it is drawn from -- the SAME
"intensive, must-not-scale-with-N" reasoning v8's own docstring already applies to synaptic weights,
drive currents, and learning rates, extended here to synaptic fan-in count. So v9 replaces the
ratio-derived in-degree with v8's OWN ALREADY-VALIDATED ABSOLUTE per-pathway values, looked up by
the (unchanged) presynaptic population size rather than recomputed from a region-dependent ratio:
    600 (ca3/ca1 assembly)      -> in-degree 24   (v8: ca3 recurrent, ca3->ca1)
    400 (ca1/cue/target assembly) -> in-degree 16 (v8: ca1->cue, ca1->target, target recurrent,
                                                      target->FS, ca1->FS)
    150 (FS half-population)    -> in-degree 6    (v8: FS->target, opponent)
    n_cue (the FULL cue region) -> in-degree 48   (v8: cortical_association, FIXED regardless of
                                                      how big the region grows)
Every assembly/overlap/FS/noise field is inherited from v8 UNCHANGED (ca3_assembly=600,
ca1/cue/target_assembly=400, cue_overlap=150, sleep_noise_cells=300, n_target_fs=300) -- confirmed
by hand that v8's own per-pathway pre-population sizes are IDENTICAL between v8 and v9 for every
assembly-scoped pathway (only the region pools themselves grow), so this lookup is unambiguous and
exactly reproduces v8's own validated in-degree on every pathway except the one that is SUPPOSED to
change (cortical_association, held fixed at 48 instead of scaling with the now-bigger region).

REGION_SCALE = 2 (this rung; a further scale-up toward the ~1-5% target is banked, not attempted
here in one jump -- matching this arc's OWN precedent of incremental, verified steps v1->v8):
    n_ca3:   1800 -> 3600   (ca3_assembly/n_ca3   33.3% -> 16.7%)
    n_ca1:   1200 -> 2400   (ca1_assembly/n_ca1   33.3% -> 16.7%)
    n_cue:   1200 -> 2400   (cue_assembly/n_cue   33.3% -> 16.7%)
    n_target:1200 -> 2400   (target_assembly/n_target 33.3% -> 16.7%)
    n_target_fs: 300 -> 300 (unchanged; the FS population IS the opponent-channel pair itself, not
                              a sparse subset of a bigger inhibitory region -- v5's build_bridge
                              splits the WHOLE n_target_fs region into fs_a/fs_b, so growing it would
                              be a DIFFERENT experiment (more interneurons per opponent channel), not
                              "the same channel, sparser code")
Total network size grows from v8's 5,700 neurons to v9's 11,100 (~1.95x), NOT the naive 25x of a
uniform re-scale -- because assemblies (the load-bearing populations) stay fixed and only the idle
surrounding region grows, exactly the biological picture (a small active ensemble embedded in a
much larger quiescent population).

Everything else -- the balanced directed-sweep replay plan, isolated order-STDP, intrinsic SFA
one-of-N eviction, learned CA1->cortex reinstatement, every anti-cheat control (stdp-off power
control, the four causal lesions, the shuffled-order temporal control) -- is v8/v7, byte-for-byte.

GO bar (inherited from v7/v8, unchanged): >=5/6 decisive seeds pass ``intact_beats_shuffled_order``
(margin >= +0.01) AND both memories recovered AND the stdp-off power control collapses the margin
AND lesions ~0.

Numpy decisive (this arc's own established primary-decisive backend; cupy confirmation queued the
same way v7/v8's own decisive artifacts were):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v9_true_sparsity \\
        --seeds 42 43 44 100 101 102 --out research/findings/raw/order_recalib/v9_true_sparsity_decisive_numpy.json

Small-scale smoke (fast correctness check, NOT a scientific verdict):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v9_true_sparsity \\
        --smoke-scale 3 --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate as v1  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v6_order_stdp as v6  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v7_balanced_order as v7  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v8_real_scale as v8  # noqa: E402

DECISIVE_SEEDS = v7.DECISIVE_SEEDS

# Region-only growth factor relative to v8 (see module docstring for why assemblies do NOT scale
# with this). v8's own region was already 25x the toy; REGION_SCALE further multiplies v8's region
# sizes (not the toy's), so the toy-relative region factor here is 25*REGION_SCALE = 50.
REGION_SCALE = 2

# v8's ALREADY-VALIDATED absolute per-pathway in-degrees, keyed by the (region-growth-INVARIANT)
# presynaptic population size for every assembly/FS-scoped pathway. Verified by hand against v8's
# own docstring table; unchanged from v8 because every assembly/overlap/FS field here is identical
# to v8's. See module docstring for the full derivation.
_FIXED_INDEGREE_BY_PRESIZE = {
    600: 24,   # ca3 recurrent, ca3 -> ca1               (ca3_assembly, unchanged from v8)
    400: 16,   # ca1->cue, ca1->target(reinstatement),
               # cortical_target_recurrent, target->FS, ca1->FS  (ca1/cue/target_assembly)
    150: 6,    # FS -> target (opponent)                   (n_target_fs // 2, unchanged from v8)
}
# The ONE full-region pathway (cortical_association, cue-region -> target-region): held at v8's
# fixed absolute in-degree regardless of how big the region grows -- this is the entire point of
# this de-risk (see module docstring). Filled in once n_cue is known (module-level GateConfig below).
_CORTICAL_ASSOCIATION_INDEGREE = 48


def _scaled_region_fields(region_scale: int) -> dict:
    """v8's region fields, further multiplied by `region_scale`; every assembly/overlap/FS/noise
    field is NOT included here -- it stays at v8's absolute value (see GateConfig below)."""
    return dict(
        n_ca3=1800 * region_scale,
        n_ca1=1200 * region_scale,
        n_cue=1200 * region_scale,
        n_target=1200 * region_scale,
    )


@dataclass(frozen=True)
class GateConfig(v8.GateConfig):
    """v8's mechanism, UNCHANGED, with the REGION (n_ca3/n_ca1/n_cue/n_target) further scaled by
    REGION_SCALE while every ASSEMBLY/overlap/FS/noise field stays at v8's absolute value (see
    module docstring for why: in-degree, like a synaptic weight or a drive current, is a property of
    one postsynaptic neuron, not of the population size)."""

    n_ca3: int = 1800 * REGION_SCALE
    n_ca1: int = 1200 * REGION_SCALE
    n_cue: int = 1200 * REGION_SCALE
    n_target: int = 1200 * REGION_SCALE
    # Inherited UNCHANGED from v8 (re-stated for clarity; dataclass field order requires re-declaring
    # every field of a frozen dataclass subclass that changes any field's default):
    n_target_fs: int = 300
    ca3_assembly: int = 600
    ca3_overlap: int = 0
    ca1_assembly: int = 400
    cue_assembly: int = 400
    cue_overlap: int = 150
    target_assembly: int = 400
    sleep_noise_cells: int = 300


def make_config(region_scale: int = REGION_SCALE, **overrides) -> GateConfig:
    """A GateConfig at an arbitrary REGION scale (default REGION_SCALE). Used for cheap smoke-scale
    variants (region_scale=1, i.e. v8-identical regions) that share every mechanism knob and the
    sparsify machinery -- only the region size differs, so a correctness bug is caught in seconds."""
    fields = _scaled_region_fields(region_scale)
    fields.update(overrides)
    return GateConfig(**fields)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# SPARSE wiring: FIXED absolute in-degree per pathway (looked up by presynaptic population size),
# NOT a ratio recomputed from the region size. See module docstring for the full argument.
# ────────────────────────────────────────────────────────────────────────────────────────────────
_ORIG_ALL_TO_ALL = v1._all_to_all
_ORIG_V6_BUILD_BRIDGE = v6.build_bridge


def _make_fixed_indegree_all_to_all(seed: int, n_cue: int):
    """Deterministic given `seed` (same RNG-stream convention as v8's `_make_sparse_all_to_all`:
    multiplier 211 / offset 3, confirmed unused by any other seed-derived draw in the v1..v8 chain).
    In-degree is looked up by `pre.size`: v8's fixed absolute values for every assembly/FS-scoped
    pathway (identical presynaptic sizes to v8, since assemblies don't scale here), and the ONE
    full-region pathway (pre.size == n_cue, however big the region grows) pinned at v8's own
    absolute value instead of a region-dependent ratio. Raises (fails loud) on an unrecognised
    pre-population size rather than silently guessing an in-degree -- a correctness bug in a config
    field must not be masked by a fallback."""
    rng = np.random.default_rng(int(seed) * 211 + 3)
    table = dict(_FIXED_INDEGREE_BY_PRESIZE)
    table[int(n_cue)] = _CORTICAL_ASSOCIATION_INDEGREE

    def _sparse(pre, post, *, self_edges: bool = True):
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        if pre.size == 0 or post.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if int(pre.size) not in table:
            raise ValueError(
                f"v9 fixed-indegree sparsify: unrecognised presynaptic population size {pre.size} "
                f"(known: {sorted(table)}) -- a config field changed without updating the lookup."
            )
        indegree = max(1, int(table[int(pre.size)]))
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


def build_bridge(seed: int, config) -> tuple[object, dict]:
    """v6's build_bridge (v5 anatomy + STDP allocation), with FIXED-absolute-in-degree wiring
    installed for the duration of this ONE build (restored after, so no global state leaks to any
    other module sharing this process). Reuses v8's own reinstatement-memory-specific bridge-truth
    verification unchanged (sparse wiring never creates a cross-memory edge regardless of which
    sparsify function is installed)."""
    v1._all_to_all = _make_fixed_indegree_all_to_all(seed, config.n_cue)
    try:
        bridge, handles = _ORIG_V6_BUILD_BRIDGE(seed, config)
    finally:
        v1._all_to_all = _ORIG_ALL_TO_ALL
    handles["reinstatement_memory_specific"] = v8._verify_reinstatement_memory_specific(bridge, handles)
    return bridge, handles


def _order_row(seed: int, config: GateConfig) -> dict:
    """v7's per-seed order-gate row (mechanism, controls, stdp-off power control -- byte-for-byte),
    with `v6.build_bridge` swapped for the v9 fixed-indegree real-scale version for the duration of
    the call."""
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

    mean_order_margin = float(np.mean([r["order_recovery_margin"] for r in rows]))
    mean_stdp_off_margin = float(np.mean([r["stdp_off_order_margin"] for r in rows]))
    aggregate_order_stdp_attribution = attributable_to(
        "v9 true-sparsity order-consolidation margin owed to order-sensitive STDP (aggregate across seeds)",
        mean_order_margin, mean_stdp_off_margin,
    )

    earned = Verdict("v9 true-sparsity order-consolidation decisive gate")
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
        why="isolate the order-sensitive spike-timing consolidation on the directed-sweep replay (inherited from v7/v8)",
    )
    decided = earned.decide(go=(n_order >= 5), verbose=False)
    verdict = decided["status"]
    ca3_ratio = cfg.ca3_assembly / cfg.n_ca3
    ca1_ratio = cfg.ca1_assembly / cfg.n_ca1
    return {
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "gate": "replay_cortical_consolidation_v9_true_sparsity",
        "phase": "decisive_multiseed",
        "mechanism": (
            "v8 real-scale-CA1 order-consolidation (unchanged), with the REGION further scaled "
            f"{REGION_SCALE}x beyond v8 (n_ca3={cfg.n_ca3}) while every ASSEMBLY/FS field stays at "
            "v8's absolute size -- fixed-absolute in-degree wiring (not a region-dependent ratio)"
        ),
        "region_scale_beyond_v8": REGION_SCALE,
        "assembly_region_ratio_ca3": ca3_ratio,
        "assembly_region_ratio_ca1_cue_target": ca1_ratio,
        "v8_gate": "replay_cortical_consolidation_v8_real_scale_ca1_port",
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
            "host-scheduled directed replay sweep (stored-trajectory drive) -- inherited from v7/v8",
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries",
            "fixed assembly anatomy",
            "SFA parameters (d/a) and STDP amplitudes/bounds set at build, not developmentally tuned",
            "sparse wiring is a fixed in-degree RANDOM projection (now held at v8's absolute value "
            "independent of region size), not a developmentally self-organized connectivity",
            f"assembly:region ratio moved from v8's ~33-67% to ~{ca3_ratio*100:.1f}%/{ca1_ratio*100:.1f}% "
            "-- a real step toward the biological ~1-5% target, but NOT yet there; a further "
            "region-scale-up (same fixed-indegree method) is the next rung",
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
                     help="Build at this (small) REGION scale instead of REGION_SCALE, for a fast correctness check.")
    args = ap.parse_args()
    if args.smoke_scale is not None:
        cfg = make_config(region_scale=args.smoke_scale, replay_plan=args.replay_plan)
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
