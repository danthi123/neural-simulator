"""Experimental substrate variant: ca1 -> concept-pool consolidation pathway.

Design: docs/plans/2026-05-21-ca1-to-concept-pool-consolidation-pathway-variant-design.md

The consolidation probe's terminal finding: the unified substrate has
ca1 -> motor + ca1 -> language_output consolidation pathways but no
ca1 -> concept-pool pathway, so compositional bindings cannot be
consolidated from the hippocampal engram into the cortical concept
pools. This variant ADDS that pathway -- by appending RegionPathway
objects to the list build_biological_brain_regions returns, WITHOUT
modifying the builder -- and tests whether compositional consolidation
then works.

Pre-registered decision rule (anti-cheat baked in): a weight-2.0
pathway lifts tag-stim pool firing off the noise floor on its own;
that proves nothing. The real test is SELECTIVITY EMERGING FROM
CONSOLIDATION -- the bound adjective's pool must become selectively
strongest AFTER replay (diffuse pre, selective post), and the
permuted-tag control must hold. If selectivity is present
pre-consolidation, the result is VOID.

Reuse-by-import: build_biological_brain_regions, _encode_facts,
run_concept_replay_phase, set_sleep_gates, freeze_all_gates,
apply_concept_topographic_bias, train_word_to_pool, measure_pool_firing
-- all byte-unchanged. No protected/frozen/moat module modified. No
autograd. Controller-only; single seed 42.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _phase1_recipe,
    _phase1_train_kwargs,
    _encode_facts,
    _all_words_word_to_idx,
    _all_pool_regions,
    _direct_pool_target,
    _N_WORDS_ORTHOGONAL,
)
from research.runners.consolidation_trainer import run_concept_replay_phase
from research.runners.text_minimal_isolation import set_sleep_gates, freeze_all_gates

SEED = 42
VARIANT_CACHE = "research/findings/raw/unified_per_regime/phase1_ca1variant/seed42.simstate.h5"
STIM_STEPS = 100
FACTS = [("apple", "big"), ("river", "small"), ("dog", "hot"), ("cat", "cold")]
REPLAY_CHECKPOINTS = [0, 20, 60]
NOISE_FLOOR = 0.02  # pre-registered; direct-binding pool rates run 0.2-0.8

_POOL_OF = {
    "apple": "noun_pool_APPLE", "river": "noun_pool_RIVER",
    "dog": "noun_pool_DOG", "cat": "noun_pool_CAT",
    "big": "adjective_pool_BIG", "small": "adjective_pool_SMALL",
    "hot": "adjective_pool_HOT", "cold": "adjective_pool_COLD",
}


def build_variant_bridge(seed):
    """Build the unified substrate AUGMENTED with ca1 -> concept-pool
    consolidation pathways. build_biological_brain_regions is called
    byte-unchanged; the new pathways are APPENDED to the list it
    returns before the bridge is constructed."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import RegionPathway
    from research.runners.text_minimal_isolation import build_biological_brain_regions

    dims = _phase1_recipe(False)
    n_lang_input = int(dims["n_lang_input"])
    n_per_pool = int(dims["n_per_pool"])
    n_fs_per_pool = int(dims["n_fs_per_pool"])
    n_dlpfc_verb = int(dims["n_dlpfc_verb"])

    # Identical kwargs to _build_bridge_with_phase1_recipe.
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=0.10,
        motor_exc_weight_mean=2.0,
        motor_inh_weight_mean=4.0,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=cpd.NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=cpd.VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=True,
        adjective_pool_names=cpd.ADJECTIVE_NAMES,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8,
        enable_hippocampus_consolidation=True,
        enable_dlpfc_verb=True,
        n_dlpfc_verb=n_dlpfc_verb,
        dlpfc_verb_internal_density=0.15,
    )

    # --- THE VARIANT CHANGE: append ca1 -> concept-pool pathways ---
    # Mirrors the existing ca1 -> motor pathway exactly (density 0.20,
    # weight 2.0, jitter 0.3, plastic, gated). One per noun / verb /
    # adjective pool. The builder is NOT modified -- we augment its
    # returned pathway list.
    pathways = list(pathways)
    concept_pools = (
        ["noun_pool_%s" % n for n in cpd.NOUN_NAMES]
        + ["verb_pool_%s" % v for v in cpd.VERB_NAMES]
        + ["adjective_pool_%s" % a for a in cpd.ADJECTIVE_NAMES]
    )
    n_added = 0
    for pool in concept_pools:
        pathways.append(RegionPathway(
            from_region="ca1", to_region=pool,
            density=0.20, weight_mean=2.0, weight_jitter=0.3,
            plastic=True, plasticity_gate="ca1_to_concept_pool",
        ))
        n_added += 1
    print(f"  variant: appended {n_added} ca1 -> concept-pool pathways")

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def train_phase1_variant(bridge, seed):
    """Phase-1 training on the variant -- the v14/v16 200-event recipe
    (apply_concept_topographic_bias + interleaved train_word_to_pool),
    identical to the proven longer_phase1_diagnostic logic."""
    tk = _phase1_train_kwargs(False)
    all_words_ordered = (
        list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB)
        + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(all_words_ordered)}
    n_words_total = len(all_words_ordered)

    cpd.apply_concept_topographic_bias(
        bridge, n_lang_input=int(tk["n_lang_input"]),
        topographic_factor=float(tk["topographic_factor"]),
        off_target_factor=float(tk["off_target_factor"]),
        sparsity=float(tk["sparsity"]), orthogonal_codes=bool(tk["orthogonal_codes"]),
        n_words_for_orthogonal=int(n_words_total), word_to_idx=word_to_idx,
        skip_motor=False, verbose=False)

    targets = []
    for w, a in cpd.DIRECTION_VOCAB.items():
        targets.append((w, "motor_%s" % a))
    for w, nm in cpd.NOUN_VOCAB.items():
        targets.append((w, "noun_pool_%s" % nm))
    for w, nm in cpd.VERB_VOCAB.items():
        targets.append((w, "verb_pool_%s" % nm))
    for w, nm in cpd.ADJECTIVE_VOCAB.items():
        targets.append((w, "adjective_pool_%s" % nm))

    n_events = int(tk["n_train_events"])
    rng = np.random.default_rng(int(seed))
    buf = [(w, t) for (w, t) in targets for _ in range(n_events)]
    rng.shuffle(buf)
    print(f"  Phase-1: {len(buf)} events ({len(targets)} words x {n_events})")
    t0 = time.time()
    for i, (w, t) in enumerate(buf):
        cpd.train_word_to_pool(
            bridge, w, t, n_events=1, reset_steps=50,
            n_lang_input=int(tk["n_lang_input"]),
            n_lang_output=int(tk["n_lang_input"]),
            sparsity=float(tk["sparsity"]),
            orthogonal_codes=bool(tk["orthogonal_codes"]),
            n_words_for_orthogonal=int(n_words_total),
            word_to_idx=word_to_idx, verbose=False)
        if (i + 1) % 800 == 0:
            el = (time.time() - t0) / 60.0
            print(f"    {i+1}/{len(buf)} ({el:.1f} min)")
    print(f"  Phase-1 done; {(time.time()-t0)/60.0:.1f} min")


def direct_binding_sanity(bridge):
    """16-word direct binding on the variant -- sanity check that the
    added pathway did not break Phase-1 (base 200ev seed 42 ~ 68.8%)."""
    dims = _phase1_recipe(False)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words = max(_N_WORDS_ORTHOGONAL, len(all_words))
    pools = _all_pool_regions(enable_adjective=True)
    n_ok = 0
    for w in all_words:
        try:
            target = _direct_pool_target(w)
        except KeyError:
            continue
        per = cpd.measure_pool_firing(
            bridge, w, pools, stim_steps=100, reset_steps=50, drive_pA=200.0,
            sparsity=0.05, n_lang_input=int(dims["n_lang_input"]),
            orthogonal_codes=True, n_words_for_orthogonal=int(n_words),
            word_to_idx=word_to_idx)
        if max(per.items(), key=lambda kv: kv[1])[0] == target:
            n_ok += 1
    return n_ok, len(all_words)


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
    """For each tag: which adjective pool is strongest on tag stim,
    and the rank of the bound adjective among adjective pools.
    Permuted control: stim tag_{i+1}, expect FACTS[i+1] adjective."""
    freeze_all_gates(bridge)
    adj_pools = [p for p in pools if p.startswith("adjective_pool_")]
    groundable = []
    for i, tag in enumerate(tags):
        noun, adj = FACTS[i]
        rates = pool_firing_during_tag_stim(bridge, tag, pools, stim_steps=STIM_STEPS)
        adj_ranked = sorted(((p, rates[p]) for p in adj_pools), key=lambda kv: -kv[1])
        bound = _POOL_OF[adj]
        rank = [p for p, _ in adj_ranked].index(bound) + 1
        groundable.append({
            "tag": tag, "noun": noun, "adj": adj,
            "bound_adj_pool_rate": rates[bound],
            "bound_adj_rank_among_adj": rank,
            "top_adj_pool": adj_ranked[0][0], "top_adj_rate": adj_ranked[0][1],
            "is_selective": rank == 1,
        })
    # permuted control
    permuted = []
    for i, tag in enumerate(tags):
        j = (i + 1) % len(tags)
        perm_tag = tags[j]
        perm_adj = FACTS[j][1]
        rates = pool_firing_during_tag_stim(bridge, perm_tag, pools, stim_steps=STIM_STEPS)
        adj_ranked = sorted(((p, rates[p]) for p in adj_pools), key=lambda kv: -kv[1])
        ok = (adj_ranked[0][0] == _POOL_OF[perm_adj])
        permuted.append({
            "cue_index": i, "stimulated_tag": perm_tag,
            "stimulated_adj": perm_adj, "top_adj_pool": adj_ranked[0][0],
            "control_ok": ok,
        })
    return groundable, permuted


def main():
    print("=== ca1 -> concept-pool consolidation pathway variant ===")
    print(f"seed={SEED}; facts={FACTS}; replay checkpoints={REPLAY_CHECKPOINTS}")

    bridge = build_variant_bridge(SEED)

    cache_path = Path(VARIANT_CACHE)
    if cache_path.exists():
        print(f"Loading variant Phase-1 cache {cache_path}")
        bridge.load_checkpoint(str(cache_path))
    else:
        print("Training Phase-1 on the variant...")
        train_phase1_variant(bridge, SEED)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bridge.save_checkpoint(str(cache_path))
        print(f"Saved variant Phase-1 cache to {cache_path}")

    # Sanity: direct binding should be near the base 200ev seed-42 ~68.8%.
    db_ok, db_n = direct_binding_sanity(bridge)
    print(f"Direct-binding sanity: {db_ok}/{db_n} = {100.0*db_ok/db_n:.1f}% "
          f"(base 200ev seed 42 ~ 68.8%)")

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
    cycles = 0
    for target in REPLAY_CHECKPOINTS:
        delta = target - cycles
        if delta > 0:
            set_sleep_gates(bridge)
            # open the NEW consolidation gate (set_sleep_gates does not
            # know about it -- runner-side, no protected modification).
            bridge.set_plasticity_gate("ca1_to_concept_pool", 1.0)
            stats = run_concept_replay_phase(
                bridge, tags, n_replays_per_tag=delta, rng=rng)
            cycles = target
            print(f"  ran {delta} replay cycles/tag (cumulative {cycles}); "
                  f"n_replays={stats['n_replays']}")
        groundable, permuted = measure(bridge, tags, pools)
        n_sel = sum(g["is_selective"] for g in groundable)
        n_lifted = sum(g["bound_adj_pool_rate"] > NOISE_FLOOR for g in groundable)
        n_ctrl = sum(p["control_ok"] for p in permuted)
        mean_rate = float(np.mean([g["bound_adj_pool_rate"] for g in groundable]))
        snapshots.append({
            "cumulative_replay_cycles": cycles, "groundable": groundable,
            "permuted_control": permuted, "n_selective": n_sel,
            "n_lifted": n_lifted, "n_control_ok": n_ctrl,
            "mean_bound_adj_rate": mean_rate,
        })
        print(f"  [{cycles:>2} cyc] bound-adj pool rate mean={mean_rate:.4f} | "
              f"lifted(>{NOISE_FLOOR})={n_lifted}/4 | selective={n_sel}/4 | "
              f"permuted-control={n_ctrl}/4")

    base = snapshots[0]
    last = snapshots[-1]
    # Anti-cheat: selectivity must EMERGE from consolidation.
    pre_selective = base["n_selective"]
    post_selective = last["n_selective"]
    post_control = last["n_control_ok"]
    post_lifted = last["n_lifted"]

    print(f"\n=== VARIANT VERDICT (seed {SEED}) ===")
    print(f"  pre-consolidation selective:  {pre_selective}/4")
    print(f"  post-consolidation selective: {post_selective}/4")
    print(f"  post-consolidation permuted-control: {post_control}/4")
    print(f"  post-consolidation lifted off noise floor: {post_lifted}/4")

    if pre_selective >= 3:
        verdict = "VOID"
        print("  --> VOID: selectivity present PRE-consolidation -- the "
              "weight-2.0 prior, not consolidation, is doing the work. "
              "Cannot attribute to consolidation.")
    elif post_selective >= 3 and post_control >= 3 and post_lifted >= 3:
        verdict = "PASS"
        print("  --> PASS: selectivity EMERGED from consolidation AND the "
              "permuted-tag control holds. The missing ca1 -> concept-pool "
              "pathway is the fix. Motivates a full multi-seed arc.")
    else:
        verdict = "NEGATIVE"
        print("  --> NEGATIVE: even with the ca1 -> concept-pool pathway, "
              "replay-driven consolidation does not establish selective "
              "compositional retrieval. Routes to the consolidation "
              "learning rule.")

    out = {
        "seed": SEED, "facts": FACTS, "tags": tags,
        "replay_checkpoints": REPLAY_CHECKPOINTS,
        "direct_binding_sanity": {"n_ok": db_ok, "n_total": db_n},
        "snapshots": snapshots, "verdict": verdict,
        "pre_selective": pre_selective, "post_selective": post_selective,
        "post_control_ok": post_control, "post_lifted": post_lifted,
        "noise_floor": NOISE_FLOOR,
    }
    out_path = "research/findings/raw/ca1_concept_pool_variant.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
