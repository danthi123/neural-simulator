"""Consolidation probe: does replay-driven hippocampo-cortical
consolidation lift the engram tag's cortical readout off the noise
floor?

The storage-locus probe (2026-05-21) found the root cause: the
compositional engram tag is hippocampal-only by construction, and
stimulating it drives the cortical concept pools only at the noise
floor (0.001-0.004). The binding is stored but stranded. The
hypothesized missing ingredient is replay-driven consolidation
(Phase 1.3 mechanism: ca3 -> ca1 -> cortex STDP transfer during
NREM-equivalent replay).

This probe runs the project's VALIDATED consolidation mechanism on the
same engram tags and measures whether the tag's cortical readout --
both concept-pool firing AND the language-output bound-attribute
signal -- rises off the noise floor.

Measurement points (cumulative replay cycles): 0, 20, 60.
  - 0 cycles: the un-consolidated baseline (reproduces storage-locus).
  - 20 cycles: the validated default (the sixth arc's replay count).
  - 60 cycles: tests whether more consolidation continues to help.

DIAGNOSTIC -- no PASS/FAIL bar. Pre-registered routing rule (fixed):
- If consolidation lifts the tag-stimulated cortical readout off the
  noise floor AND the bound attribute becomes selectively strongest:
  consolidation is the missing ingredient; the forward path is a full
  pre-registered arc (consolidate the compositional binding, then
  read).
- If consolidation does NOT lift it through 60 cycles: replay-driven
  consolidation does not establish the tag-to-cortex pathway for
  compositional bindings; the next question is substrate-level
  (whether ca1 projects to the concept pools at all).

Reuse-by-import only: set_sleep_gates, freeze_all_gates,
run_concept_replay_phase, lang_output_pattern_during_stim,
_ranked_from_pattern, the substrate builder, _encode_facts -- all
byte-unchanged. No protected/frozen/moat module modified. No autograd.
Controller-only; single seed 42; cached 200-event unified substrate.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_recipe,
    _encode_facts,
    _all_words_word_to_idx,
    _all_pool_regions,
)
from research.runners.compose_retrieval_runner import (
    _N_WORDS_ORTHOGONAL,
    _ranked_from_pattern,
)
from research.runners.compose_concept_engram import lang_output_pattern_during_stim
from research.runners.consolidation_trainer import run_concept_replay_phase
from research.runners.text_minimal_isolation import set_sleep_gates, freeze_all_gates

SEED = 42
CACHE_DIR = "research/findings/raw/unified_per_regime/phase1"
STIM_STEPS = 100
FACTS = [("apple", "big"), ("river", "small"), ("dog", "hot"), ("cat", "cold")]
REPLAY_CHECKPOINTS = [0, 20, 60]  # cumulative replay cycles per tag

_POOL_OF = {
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}


def _pool_firing_during_tag_stim(bridge, tag_name, pools, drive_pA=1500.0,
                                   stim_steps=100):
    """Mean per-pool firing rate during engram-tag stimulation."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    pool_arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64)
                 for p in pools}
    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()
    bridge.stimulate_tag(tag_name, drive_pA=drive_pA, additive=False)
    accum = {p: 0.0 for p in pools}
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        firing = bridge.cp_firing_states
        for p in pools:
            accum[p] += float(cp.sum(firing[pool_arrs[p]].astype(cp.float32)))
    bridge.clear_tag_drive(tag_name)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    return {p: accum[p] / (stim_steps * max(1, len(pool_arrs[p])))
            for p in pools}


def _measure(bridge, tags, pools, dims):
    """For each tag: pool firing + language-output bound-attribute
    rate/rank during tag stimulation. Gates frozen so the measurement
    itself induces no plasticity."""
    freeze_all_gates(bridge)
    per = []
    for i, tag in enumerate(tags):
        noun, adj = FACTS[i]
        rates = _pool_firing_during_tag_stim(bridge, tag, pools,
                                               stim_steps=STIM_STEPS)
        adj_pools = [p for p in pools if p.startswith("adjective_pool_")]
        adj_ranked = sorted(((p, rates[p]) for p in adj_pools),
                             key=lambda kv: -kv[1])
        bound_adj_pool = _POOL_OF[adj]
        adj_rank = [p for p, _ in adj_ranked].index(bound_adj_pool) + 1
        top_pool, top_rate = max(rates.items(), key=lambda kv: kv[1])

        lo_pat, n_lo = lang_output_pattern_during_stim(
            bridge, tag, drive_pA=1500.0, stim_steps=STIM_STEPS)
        lo_ranked = _ranked_from_pattern(lo_pat, n_lo, dims, exclude=noun)
        lo_rate = {w: float(r) for (w, r, _t) in lo_ranked}
        lo_sorted = sorted(lo_rate.items(), key=lambda kv: -kv[1])
        lo_top = lo_sorted[0][0] if lo_sorted else None
        lo_adj_rank = ([w for w, _ in lo_sorted].index(adj) + 1
                       if adj in lo_rate else -1)
        per.append({
            "tag": tag, "noun": noun, "adj": adj,
            "pool_top": top_pool, "pool_top_rate": top_rate,
            "bound_adj_pool_rate": rates[bound_adj_pool],
            "bound_adj_pool_rank_among_adj": adj_rank,
            "langout_top": lo_top,
            "langout_bound_adj_rate": lo_rate.get(adj, 0.0),
            "langout_bound_adj_rank": lo_adj_rank,
        })
    return per


def main():
    print("=== Consolidation probe ===")
    print(f"seed={SEED}; cache={CACHE_DIR}; replay checkpoints={REPLAY_CHECKPOINTS}")

    bridge = _build_bridge_with_phase1_recipe(SEED, tiny_synth=False)
    cache_path = _phase1_cache_path(CACHE_DIR, SEED)
    print(f"Loading {cache_path}")
    bridge.load_checkpoint(str(cache_path))

    recipe_dims = _phase1_recipe(False)
    all_words, _ = _all_words_word_to_idx()
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05, "dt_ms": 0.5,
        "n_words_for_orthogonal": max(_N_WORDS_ORTHOGONAL, len(all_words)),
    }
    tags = _encode_facts(bridge, FACTS, dims, encoding_steps=200)
    print(f"Encoded tags: {tags}")
    pools = _all_pool_regions(enable_adjective=True)

    rng = np.random.default_rng(SEED)
    snapshots = []
    cycles_done = 0
    for target in REPLAY_CHECKPOINTS:
        delta = target - cycles_done
        if delta > 0:
            set_sleep_gates(bridge)
            stats = run_concept_replay_phase(
                bridge, tags, n_replays_per_tag=delta, rng=rng)
            cycles_done = target
            print(f"  ran {delta} replay cycles/tag "
                  f"(cumulative {cycles_done}); n_replays={stats['n_replays']}")
        per = _measure(bridge, tags, pools, dims)
        snapshots.append({"cumulative_replay_cycles": cycles_done,
                           "per_binding": per})
        # summary line
        mean_pool = np.mean([b["bound_adj_pool_rate"] for b in per])
        n_adj_top = sum(b["bound_adj_pool_rank_among_adj"] == 1 for b in per)
        n_lo_top = sum(b["langout_top"] == b["adj"] for b in per)
        mean_lo = np.mean([b["langout_bound_adj_rate"] for b in per])
        print(f"  [{cycles_done:>2} cycles] bound-adj pool rate mean="
              f"{mean_pool:.4f} top-among-adj={n_adj_top}/4 | "
              f"langout bound-adj rate mean={mean_lo:.2f} "
              f"langout-top-correct={n_lo_top}/4")

    # Trajectory verdict
    print(f"\n=== CONSOLIDATION TRAJECTORY (seed {SEED}) ===")
    base = snapshots[0]
    last = snapshots[-1]
    base_pool = np.mean([b["bound_adj_pool_rate"] for b in base["per_binding"]])
    last_pool = np.mean([b["bound_adj_pool_rate"] for b in last["per_binding"]])
    base_lo = np.mean([b["langout_bound_adj_rate"] for b in base["per_binding"]])
    last_lo = np.mean([b["langout_bound_adj_rate"] for b in last["per_binding"]])
    last_adj_top = sum(b["bound_adj_pool_rank_among_adj"] == 1
                       for b in last["per_binding"])
    last_lo_top = sum(b["langout_top"] == b["adj"]
                      for b in last["per_binding"])
    print(f"  bound-adj pool rate: {base_pool:.4f} (0 cyc) -> "
          f"{last_pool:.4f} ({last['cumulative_replay_cycles']} cyc)")
    print(f"  langout bound-adj rate: {base_lo:.2f} -> {last_lo:.2f}")

    # Noise floor reference: direct-binding pool rates run 0.2-0.8.
    NOISE_FLOOR = 0.02
    lifted = (last_pool > NOISE_FLOOR) or (last_lo > 2.0 * base_lo + 1.0)
    selective = (last_adj_top > 2) or (last_lo_top > 2)
    if lifted and selective:
        verdict = "CONSOLIDATION_HELPS"
        print("  --> Consolidation lifts the tag's cortical readout AND the "
              "bound attribute becomes selectively strongest. Consolidation "
              "is the missing ingredient -> full pre-registered arc.")
    elif lifted:
        verdict = "CONSOLIDATION_LIFTS_NOT_SELECTIVE"
        print("  --> Consolidation lifts the readout off the noise floor but "
              "the bound attribute is NOT selectively strongest. Partial; "
              "the selectivity question is the next probe.")
    else:
        verdict = "CONSOLIDATION_DOES_NOT_LIFT"
        print("  --> Consolidation does NOT lift the tag-to-cortex readout "
              "through 60 cycles. Replay-driven consolidation does not "
              "establish the tag-to-cortex pathway for compositional "
              "bindings -> substrate-level question (does ca1 project to "
              "the concept pools at all).")

    out = {
        "seed": SEED, "cache_dir": CACHE_DIR, "facts": FACTS, "tags": tags,
        "replay_checkpoints": REPLAY_CHECKPOINTS, "stim_steps": STIM_STEPS,
        "snapshots": snapshots, "verdict": verdict,
        "base_pool_rate": base_pool, "last_pool_rate": last_pool,
        "base_langout_rate": base_lo, "last_langout_rate": last_lo,
    }
    out_path = "research/findings/raw/consolidation_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
