"""v10: a region-growth-COMPENSATING in-degree law for the ONE pathway v9's own finding named as the
root cause of its NO-GO (board #130's next-named rung after v9, 2026-09-08/09).

WHY THIS EXISTS.
-----------------
v9 (``_replay_cortical_consolidation_gate_v9_true_sparsity.py``) scaled the CA1/CA3/cue/target
REGION 2x beyond v8 while holding every ASSEMBLY/FS field fixed (assembly:region ratio 33.3% ->
16.7%, a real step toward the ~1-5% biological target) -- and applied v8's own "hold in-degree
absolute, not ratio-derived" rule UNIFORMLY to every wiring pathway, including the one pathway v9's
own module docstring already flagged as different in kind: ``cortical_association``
(cue-region -> target-region, "spans the FULL region, not an assembly"). Its finding
(``2026-09-08-order-consolidation-v9-true-sparsity-scaleup-NO-GO-readout-threshold-not-mechanism.md``)
traced the resulting 0/6 NO-GO to exactly this: holding ``cortical_association``'s in-degree fixed
at v8's absolute value (48) while its presynaptic candidate pool (``n_cue``) DOUBLES dilutes the
probability that any of a target neuron's 48 converging synapses lands on one of the FEW
partial-cue cells the retrieval probe actually drives -- while the order-STDP consolidation
mechanism itself stayed fully intact on all 6 seeds (attribution 1.0, exact 0.0000 stdp-off
collapse, physical trace stronger on every seed). The finding named two independent, compatible
fixes; this is fix (1), the more biological one (a developmental connectivity LAW, not a re-tuned
threshold): "give `cortical_association` specifically a region-growth-COMPENSATING in-degree law
(in-degree x driven-fraction held constant, i.e. in-degree scaling with `n_cue`...) instead of
applying the same 'hold in-degree constant' rule ... uniformly."

THE LAW (one pathway, one new knob, everything else v9 byte-for-byte).
------------------------------------------------------------------------
``cortical_association``'s in-degree becomes a function of the CURRENT region size relative to the
region size its fixed value (48) was VALIDATED at (v8/v9's own baseline, n_cue=1200):

    indegree(n_cue) = round(48 * n_cue / 1200)

This holds the DRIVEN FRACTION -- P(a given target neuron's cortical_association synapses include
at least one of the partial-cue driven cells) -- approximately scale-invariant as the region grows,
because the expected number of driven-cue synapses per target neuron is
``indegree * (n_driven_cue / n_cue)``; substituting the law makes ``n_cue`` cancel, leaving
``48 * n_driven_cue / 1200`` -- a quantity that depends on how many cue cells the probe drives, not
on how large the surrounding (mostly idle) region has grown. This is exactly the "synaptic-scaling
/ homeostatic gain" companion process this project's wall-reframe finding
(``2026-07-31-why-we-hit-walls-the-missing-companion-process.md``) says biology runs and v1-v9 never
implemented for this ONE pathway (every OTHER, assembly-scoped pathway in this substrate correctly
keeps in-degree fixed, because ITS presynaptic pool does not grow with the region -- see v9's
docstring; only ``cortical_association``'s presynaptic pool IS the growing region itself).

At v8/v9's own baseline (``n_cue=1200``, ``region_scale=1``), the law evaluates to
``round(48*1200/1200) == 48`` -- IDENTICAL to the fixed rule. So the two in-degree laws
(``"fixed"`` vs ``"scale_invariant"``) are byte-identical at region_scale=1 by construction, and
only diverge once the region grows beyond the baseline it was calibrated at (region_scale=2:
fixed stays 48, scale_invariant becomes round(48*2400/1200)=96).

FLAG, GUARDED, ADDITIVE, DEFAULT OFF.
---------------------------------------
``GateConfig.indegree_law`` (new field): ``"fixed"`` (default -- v9's own behaviour, unchanged) or
``"scale_invariant"`` (this fix, opt-in). No v9 file is modified; this is a NEW module built the
same way v9 was built on v8 -- import the prior version, override only what changes (here: which
in-degree the ONE full-region pathway receives), inherit everything else (the balanced
directed-sweep replay plan, isolated order-STDP, intrinsic SFA one-of-N eviction, learned
CA1->cortex reinstatement, every anti-cheat control) byte-for-byte from v9/v8/v7.

GO bar: inherited from v7/v8/v9, unchanged (>=5/6 decisive seeds pass
``intact_beats_shuffled_order`` (margin >= +0.01) AND both memories recovered AND the stdp-off
power control collapses the margin AND lesions ~0).

Numpy decisive (STOCK ON THE POOL, not run in-agent -- see CLAUDE.md cost-routing):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v10_indegree_law \\
        --indegree-law scale_invariant --seeds 42 43 44 100 101 102 \\
        --out research/findings/raw/order_recalib/v10_indegree_law_decisive_numpy.json

Small-scale / single-seed smoke (fast correctness + transfer check, NOT a scientific verdict):
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v10_indegree_law \\
        --indegree-law fixed --smoke-scale 1 --seeds 42            # byte-identical-to-v8 check
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v10_indegree_law \\
        --indegree-law scale_invariant --seeds 42 43               # does the driven-fraction now transfer?
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
from research.runners import _replay_cortical_consolidation_gate_v9_true_sparsity as v9  # noqa: E402

DECISIVE_SEEDS = v7.DECISIVE_SEEDS
REGION_SCALE = v9.REGION_SCALE

# The region size cortical_association's fixed in-degree (48) was VALIDATED at -- v8's own n_cue,
# also v9's region_scale=1 baseline. The law is anchored here so "fixed" and "scale_invariant"
# coincide exactly at this scale (see module docstring).
_BASE_N_CUE = 1200
_BASE_CORTICAL_ASSOCIATION_INDEGREE = 48

INDEGREE_LAWS = ("fixed", "scale_invariant")


def _cortical_association_indegree(n_cue: int, law: str) -> int:
    """The ONE knob this module adds. `law="fixed"` reproduces v9 exactly (constant 48,
    regardless of region size -- the NO-GO'd method). `law="scale_invariant"` scales in-degree
    proportionally with n_cue so the driven-fraction probability the retrieval probe depends on
    stays constant as the region grows (see module docstring for the derivation). Both laws agree
    exactly at n_cue == _BASE_N_CUE by construction."""
    if law == "fixed":
        return _BASE_CORTICAL_ASSOCIATION_INDEGREE
    if law == "scale_invariant":
        return max(1, round(_BASE_CORTICAL_ASSOCIATION_INDEGREE * (int(n_cue) / _BASE_N_CUE)))
    raise ValueError(f"unknown indegree_law {law!r} (known: {INDEGREE_LAWS})")


@dataclass(frozen=True)
class GateConfig(v9.GateConfig):
    """v9's mechanism and region scale, UNCHANGED, plus ONE new opt-in field controlling which
    in-degree law `cortical_association` uses. Every other field is inherited from v9 (which itself
    re-states every v8 field it does not change, per dataclass frozen-subclass requirements)."""

    indegree_law: str = "fixed"


def make_config(region_scale: int = REGION_SCALE, indegree_law: str = "fixed", **overrides) -> GateConfig:
    """A GateConfig at an arbitrary REGION scale and in-degree law. `region_scale=1` (v8-identical
    regions) + `indegree_law="fixed"` reproduces v8/v9 byte-for-byte -- the cheap correctness check
    this fix must pass before its transfer claim means anything."""
    fields = v9._scaled_region_fields(region_scale)
    fields.update(overrides)
    fields.setdefault("indegree_law", indegree_law)
    return GateConfig(**fields)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# SPARSE wiring: identical to v9's fixed-absolute-in-degree sparsify for every assembly/FS-scoped
# pathway; the ONE full-region pathway (cortical_association) now looks its in-degree up via
# `_cortical_association_indegree(n_cue, law)` instead of v9's hardcoded constant.
# ────────────────────────────────────────────────────────────────────────────────────────────────
_ORIG_ALL_TO_ALL = v1._all_to_all
_ORIG_V6_BUILD_BRIDGE = v6.build_bridge


def _make_indegree_law_all_to_all(seed: int, n_cue: int, law: str):
    """Same RNG-stream convention as v8/v9's sparsify (multiplier 211 / offset 3), so `law="fixed"`
    draws from an IDENTICAL stream to v9's own sparsify and is therefore byte-identical to v9 at
    every scale, not merely at the baseline. Only the in-degree TABLE VALUE for the one full-region
    pathway (pre.size == n_cue) differs by law; the assembly/FS-scoped table entries are unchanged
    from v9's own `_FIXED_INDEGREE_BY_PRESIZE`."""
    rng = np.random.default_rng(int(seed) * 211 + 3)
    table = dict(v9._FIXED_INDEGREE_BY_PRESIZE)
    table[int(n_cue)] = _cortical_association_indegree(int(n_cue), law)

    def _sparse(pre, post, *, self_edges: bool = True):
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        if pre.size == 0 or post.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if int(pre.size) not in table:
            raise ValueError(
                f"v10 indegree-law sparsify: unrecognised presynaptic population size {pre.size} "
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
    """v6's build_bridge (v5 anatomy + STDP allocation), with the in-degree-law wiring installed for
    the duration of this ONE build (restored after, so no global state leaks to any other module
    sharing this process). `config.indegree_law` selects fixed (v9-identical) vs scale_invariant."""
    law = getattr(config, "indegree_law", "fixed")
    v1._all_to_all = _make_indegree_law_all_to_all(seed, config.n_cue, law)
    try:
        bridge, handles = _ORIG_V6_BUILD_BRIDGE(seed, config)
    finally:
        v1._all_to_all = _ORIG_ALL_TO_ALL
    handles["reinstatement_memory_specific"] = v8._verify_reinstatement_memory_specific(bridge, handles)
    return bridge, handles


_ORIG_V7_GATECONFIG = v7.GateConfig


def _order_row(seed: int, config: GateConfig) -> dict:
    """v7's per-seed order-gate row (mechanism, controls, stdp-off power control -- byte-for-byte),
    with `v6.build_bridge` swapped for this module's in-degree-law build for the duration of the call.

    Also swaps v7's own module-level `GateConfig` name for this module's subclass: v7's power-control
    line builds its stdp-off config as ``GateConfig(**{**asdict(config), "stdp_sleep": False})``,
    resolving `GateConfig` from v7's own namespace at call time. v8/v9 never hit this because they only
    override existing field DEFAULTS; v10 adds a genuinely NEW field (`indegree_law`), so
    `asdict(config)` carries a key v7's own GateConfig does not accept unless patched the same way
    `build_bridge` already is."""
    v6.build_bridge = build_bridge
    v7.GateConfig = GateConfig
    try:
        return v7._order_row(seed, config)
    finally:
        v6.build_bridge = _ORIG_V6_BUILD_BRIDGE
        v7.GateConfig = _ORIG_V7_GATECONFIG


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
        "v10 indegree-law order-consolidation margin owed to order-sensitive STDP (aggregate across seeds)",
        mean_order_margin, mean_stdp_off_margin,
    )

    earned = Verdict("v10 indegree-law order-consolidation decisive gate")
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
        why="isolate the order-sensitive spike-timing consolidation on the directed-sweep replay (inherited from v7/v8/v9)",
    )
    decided = earned.decide(go=(n_order >= 5), verbose=False)
    verdict = decided["status"]
    ca3_ratio = cfg.ca3_assembly / cfg.n_ca3
    ca1_ratio = cfg.ca1_assembly / cfg.n_ca1
    law = getattr(cfg, "indegree_law", "fixed")
    return {
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "gate": "replay_cortical_consolidation_v10_indegree_law",
        "phase": "decisive_multiseed",
        "mechanism": (
            "v9 region-scaled order-consolidation (unchanged), with `cortical_association`'s "
            f"in-degree governed by the '{law}' law "
            f"(fixed=v9-identical constant {_BASE_CORTICAL_ASSOCIATION_INDEGREE}; "
            "scale_invariant=in-degree scales with n_cue to hold the driven-cue-fraction constant)"
        ),
        "indegree_law": law,
        "cortical_association_indegree": _cortical_association_indegree(cfg.n_cue, law),
        "region_scale_beyond_v8": REGION_SCALE if cfg.n_ca3 == v9.GateConfig.n_ca3 else cfg.n_ca3 / 1800,
        "assembly_region_ratio_ca3": ca3_ratio,
        "assembly_region_ratio_ca1_cue_target": ca1_ratio,
        "v9_gate": "replay_cortical_consolidation_v9_true_sparsity",
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
            "host-scheduled directed replay sweep (stored-trajectory drive) -- inherited from v7/v8/v9",
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries",
            "fixed assembly anatomy",
            "SFA parameters (d/a) and STDP amplitudes/bounds set at build, not developmentally tuned",
            "sparse wiring is a fixed-or-law in-degree RANDOM projection, not a developmentally "
            "self-organized connectivity -- the LAW governing cortical_association's in-degree is "
            "still a host-derived formula, not something the substrate grew itself",
            f"assembly:region ratio at ~{ca3_ratio*100:.1f}%/{ca1_ratio*100:.1f}% "
            "-- a real step toward the biological ~1-5% target, but NOT yet there",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DECISIVE_SEEDS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--replay-plan", default="directed_sweep",
                    choices=["directed_sweep", "episode_agnostic"])
    ap.add_argument("--indegree-law", default="fixed", choices=list(INDEGREE_LAWS),
                     help="'fixed' (default, v9-identical) or 'scale_invariant' (this fix's opt-in law).")
    ap.add_argument("--smoke-scale", type=int, default=None,
                     help="Build at this (small) REGION scale instead of REGION_SCALE, for a fast correctness check.")
    args = ap.parse_args()
    if args.smoke_scale is not None:
        cfg = make_config(region_scale=args.smoke_scale, indegree_law=args.indegree_law,
                           replay_plan=args.replay_plan)
    else:
        cfg = GateConfig(replay_plan=args.replay_plan, indegree_law=args.indegree_law)
    payload = run_decisive(args.seeds, cfg)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
