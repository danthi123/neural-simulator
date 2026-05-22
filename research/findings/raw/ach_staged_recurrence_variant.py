"""ACh-staged recurrent excitation: consolidation variant.

Design: docs/plans/2026-05-22-acetylcholine-staged-recurrence-consolidation-variant-design.md

The ca1-variant arc found the missing ca1->concept-pool wire is
necessary but not sufficient: the concept pools' deliberately-weak
internal dynamics cannot ignite into a consolidated attractor.
Hasselmo's SPEAR biology dissolves the tension: recurrent excitation
is suppressed during encoding (stable multi-pattern training) and
RELEASED during consolidation (attractor formation).

This variant implements that staging. It reuses the ca1-variant
substrate AND its Phase-1 checkpoint (the recurrence is installed
AFTER loading, so Phase-1 stability is preserved by construction --
no retrain). After load, before consolidation, it INSTALLS recurrent
excitatory connectivity into each concept pool via the documented
set_pathway_weights(add_missing=True) post-build install API -- the
"low-ACh release of recurrent excitation".

Time-boxed: one run, no iteration. Pre-registered decision rule with
anti-cheat (selectivity must EMERGE from consolidation; permuted-tag
control; RUNAWAY is a distinct honest sub-case).

Reuse-by-import: build_variant_bridge (the ca1-variant substrate),
_encode_facts, run_concept_replay_phase, set_sleep_gates,
freeze_all_gates -- all byte-unchanged. No protected/frozen/moat
module modified. No autograd. Controller-only; single seed 42.
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

# Reuse the ca1-variant substrate builder byte-unchanged.
from importlib import util as _iu
_v_path = os.path.join(_HERE, "ca1_concept_pool_variant.py")
_spec = _iu.spec_from_file_location("_ca1v", _v_path)
_ca1v = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_ca1v)
build_variant_bridge = _ca1v.build_variant_bridge

from research.runners.unified_per_regime_monitor_runner import (
    _phase1_recipe, _encode_facts, _all_words_word_to_idx, _all_pool_regions,
)
from research.runners.compose_retrieval_runner import _N_WORDS_ORTHOGONAL
from research.runners.consolidation_trainer import run_concept_replay_phase
from research.runners.text_minimal_isolation import set_sleep_gates, freeze_all_gates
import research.runners.concept_pool_demo as cpd

SEED = 42
VARIANT_CACHE = "research/findings/raw/unified_per_regime/phase1_ca1variant/seed42.simstate.h5"
STIM_STEPS = 100
FACTS = [("apple", "big"), ("river", "small"), ("dog", "hot"), ("cat", "cold")]
REPLAY_CHECKPOINTS = [0, 20, 60]
# Staged recurrence: mirror the validated attractor-capable motor-pool
# canon recurrence (density 0.10, weight 2.0).
RECUR_DENSITY = 0.10
RECUR_WEIGHT = 2.0
RUNAWAY_RATE = 0.40  # pre-registered: mean-rate-across-all-pools above this = runaway

_POOL_OF = {
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}


def install_staged_recurrence(bridge, concept_pools, density, weight, rng):
    """Install recurrent excitatory connectivity into each concept pool
    -- the 'low-ACh release of recurrent excitation'. Excitatory
    neurons only (Dale's law: exc pre -> excitatory synapse). One
    set_pathway_weights(add_missing=True) call for all pools."""
    rm = bridge.region_manager
    all_pre, all_post = [], []
    for pool in concept_pools:
        all_idx = list(rm.indices(pool))
        inh = set(rm.inhibitory_indices(pool))
        exc = np.array([i for i in all_idx if i not in inh], dtype=np.int64)
        nE = len(exc)
        if nE < 2:
            continue
        mask = rng.random((nE, nE)) < density
        np.fill_diagonal(mask, False)
        pre_loc, post_loc = np.where(mask)
        all_pre.append(exc[pre_loc])
        all_post.append(exc[post_loc])
    pre = np.concatenate(all_pre)
    post = np.concatenate(all_post)
    w = np.full(pre.shape, float(weight), dtype=np.float32)
    n = bridge.set_pathway_weights(
        "staged_recurrence", pre, post, w, add_missing=True)
    return n, int(pre.size)


def pool_firing_during_tag_stim(bridge, tag, pools, drive_pA=1500.0,
                                  stim_steps=100):
    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager
    arrs = {p: cp.asarray(list(rm.indices(p)), dtype=cp.int64) for p in pools}
    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()
    for _ in range(30):
        bridge._run_one_simulation_step()
    bridge.stimulate_tag(tag, drive_pA=drive_pA, additive=False)
    accum = {p: 0.0 for p in pools}
    for _ in range(stim_steps):
        bridge._run_one_simulation_step()
        firing = bridge.cp_firing_states
        for p in pools:
            accum[p] += float(cp.sum(firing[arrs[p]].astype(cp.float32)))
    bridge.clear_tag_drive(tag)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
    return {p: accum[p] / (stim_steps * max(1, len(arrs[p]))) for p in pools}


def measure(bridge, tags, pools):
    """Per-tag: bound-adjective-pool selectivity among adjective pools;
    permuted-tag control; mean rate across all pools (runaway check)."""
    freeze_all_gates(bridge)
    adj_pools = [p for p in pools if p.startswith("adjective_pool_")]
    groundable, all_rates = [], []
    for i, tag in enumerate(tags):
        noun, adj = FACTS[i]
        rates = pool_firing_during_tag_stim(bridge, tag, pools, stim_steps=STIM_STEPS)
        all_rates.extend(rates.values())
        adj_ranked = sorted(((p, rates[p]) for p in adj_pools), key=lambda kv: -kv[1])
        bound = _POOL_OF[adj]
        rank = [p for p, _ in adj_ranked].index(bound) + 1
        groundable.append({
            "tag": tag, "noun": noun, "adj": adj,
            "bound_adj_pool_rate": rates[bound],
            "bound_adj_rank_among_adj": rank,
            "is_selective": rank == 1,
        })
    permuted = []
    for i, tag in enumerate(tags):
        j = (i + 1) % len(tags)
        rates = pool_firing_during_tag_stim(bridge, tags[j], pools, stim_steps=STIM_STEPS)
        adj_ranked = sorted(((p, rates[p]) for p in adj_pools), key=lambda kv: -kv[1])
        permuted.append({
            "cue_index": i, "stimulated_tag": tags[j],
            "stimulated_adj": FACTS[j][1], "top_adj_pool": adj_ranked[0][0],
            "control_ok": adj_ranked[0][0] == _POOL_OF[FACTS[j][1]],
        })
    return {
        "groundable": groundable, "permuted_control": permuted,
        "n_selective": sum(g["is_selective"] for g in groundable),
        "n_control_ok": sum(p["control_ok"] for p in permuted),
        "mean_bound_adj_rate": float(np.mean([g["bound_adj_pool_rate"] for g in groundable])),
        "mean_all_pool_rate": float(np.mean(all_rates)),
    }


def main():
    print("=== ACh-staged recurrent excitation consolidation variant ===")
    print(f"seed={SEED}; recurrence density={RECUR_DENSITY} weight={RECUR_WEIGHT}")

    bridge = build_variant_bridge(SEED)
    print(f"Loading ca1-variant Phase-1 checkpoint {VARIANT_CACHE}")
    bridge.load_checkpoint(VARIANT_CACHE)

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
    concept_pools = (
        ["noun_pool_%s" % n for n in cpd.NOUN_NAMES]
        + ["verb_pool_%s" % v for v in cpd.VERB_NAMES]
        + ["adjective_pool_%s" % a for a in cpd.ADJECTIVE_NAMES]
    )

    rng = np.random.default_rng(SEED)

    # Measurement 1: pre-recurrence baseline (reproduces ca1-variant).
    pre_recur = measure(bridge, tags, pools)
    print(f"  [pre-recurrence] bound-adj rate mean={pre_recur['mean_bound_adj_rate']:.4f} "
          f"| selective={pre_recur['n_selective']}/4 | control={pre_recur['n_control_ok']}/4")

    # Install the staged recurrence (the low-ACh release).
    n_edges, n_attempted = install_staged_recurrence(
        bridge, concept_pools, RECUR_DENSITY, RECUR_WEIGHT, rng)
    print(f"  installed staged recurrence: {n_attempted} recurrent exc edges "
          f"across {len(concept_pools)} concept pools")

    # Measurement 2: post-install, pre-consolidation (anti-cheat baseline).
    post_install = measure(bridge, tags, pools)
    print(f"  [post-install pre-consol] bound-adj rate mean="
          f"{post_install['mean_bound_adj_rate']:.4f} | selective="
          f"{post_install['n_selective']}/4 | control={post_install['n_control_ok']}/4 "
          f"| mean-all-pool-rate={post_install['mean_all_pool_rate']:.4f}")

    # Consolidation: replay with the new recurrence released.
    snapshots = [{"phase": "pre_recurrence", "cumulative_replay_cycles": -1,
                  **{k: pre_recur[k] for k in
                     ("n_selective", "n_control_ok", "mean_bound_adj_rate",
                      "mean_all_pool_rate")}},
                 {"phase": "post_install_pre_consol", "cumulative_replay_cycles": 0,
                  **{k: post_install[k] for k in
                     ("n_selective", "n_control_ok", "mean_bound_adj_rate",
                      "mean_all_pool_rate")}}]
    cycles = 0
    for target in REPLAY_CHECKPOINTS:
        if target == 0:
            continue
        delta = target - cycles
        set_sleep_gates(bridge)
        bridge.set_plasticity_gate("ca1_to_concept_pool", 1.0)
        stats = run_concept_replay_phase(bridge, tags, n_replays_per_tag=delta, rng=rng)
        cycles = target
        snap = measure(bridge, tags, pools)
        snapshots.append({"phase": "post_consol", "cumulative_replay_cycles": cycles,
                          **{k: snap[k] for k in
                             ("n_selective", "n_control_ok", "mean_bound_adj_rate",
                              "mean_all_pool_rate")},
                          "groundable": snap["groundable"],
                          "permuted_control": snap["permuted_control"]})
        print(f"  [{cycles:>2} replay cyc] bound-adj rate mean={snap['mean_bound_adj_rate']:.4f} "
              f"| selective={snap['n_selective']}/4 | control={snap['n_control_ok']}/4 "
              f"| mean-all-pool-rate={snap['mean_all_pool_rate']:.4f}")

    last = snapshots[-1]
    print(f"\n=== VERDICT (seed {SEED}) ===")
    print(f"  post-install pre-consolidation selective: {post_install['n_selective']}/4")
    print(f"  post-consolidation selective:             {last['n_selective']}/4")
    print(f"  post-consolidation permuted-control:      {last['n_control_ok']}/4")
    print(f"  post-consolidation mean-all-pool-rate:    {last['mean_all_pool_rate']:.4f}")

    if last["mean_all_pool_rate"] > RUNAWAY_RATE:
        verdict = "RUNAWAY"
        print(f"  --> RUNAWAY: installed recurrence drove the pools to "
              f"saturation (mean-all-pool-rate {last['mean_all_pool_rate']:.3f} > "
              f"{RUNAWAY_RATE}). Released recurrence is unstable without matched "
              f"inhibition -> routes to ISN matched-inhibition tuning.")
    elif post_install["n_selective"] >= 3:
        verdict = "VOID"
        print("  --> VOID: selectivity present PRE-consolidation (the recurrence "
              "prior, not consolidation, did the work). Cannot attribute to "
              "consolidation.")
    elif last["n_selective"] >= 3 and last["n_control_ok"] >= 3:
        verdict = "PASS"
        print("  --> PASS: selectivity EMERGED from consolidation AND the "
              "permuted-tag control holds. Staged recurrence enables "
              "compositional consolidation. Motivates a full multi-seed arc.")
    else:
        verdict = "NEGATIVE"
        print("  --> NEGATIVE: even with staged recurrence released for "
              "consolidation, replay does not establish selective compositional "
              "retrieval. Converges with the SPEAR negative; the next major "
              "direction is phase-coded vector-symbolic composition.")

    out = {
        "seed": SEED, "facts": FACTS, "tags": tags,
        "recur_density": RECUR_DENSITY, "recur_weight": RECUR_WEIGHT,
        "replay_checkpoints": REPLAY_CHECKPOINTS, "runaway_rate": RUNAWAY_RATE,
        "n_recurrent_edges_installed": n_attempted,
        "snapshots": snapshots, "verdict": verdict,
    }
    out_path = "research/findings/raw/ach_staged_recurrence_variant.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
