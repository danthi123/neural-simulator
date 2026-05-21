"""V14-only substrate cross-substrate generalization test (Direction K).

The autonomous arc's substrate characterization (4 regimes; per-word
attractor sensitivity; oscillatory silent-interval dynamics) was done
on the UNIFIED substrate (v14/v16 concept pools + hippocampus +
dlpfc). Direction K tests whether those findings are SUBSTRATE-SPECIFIC
to the unified architecture or SUBSTRATE-GENERAL.

Cheap-first single-seed probe: train v14-only substrate at 800ev seed
42 (the saturated training budget), run 16-word direct binding
diagnostic. Compare to unified 800ev seed 42 (15/16 = 93.8%).

Decision rule (pre-registered):
- If v14-only seed 42 800ev direct binding >= unified 93.8%: v14-only
  has stronger direct binding (consistent with v14 documented baseline
  88.75% multi-seed; the hippocampus + dlpfc additions modestly
  DEGRADE direct binding which is fully recovered with 800ev).
- If v14-only seed 42 800ev direct binding < unified 93.8%: unified
  substrate's hippocampus + dlpfc additions IMPROVE direct binding
  over v14-only at extended training; unexpected.
- If v14-only matches unified: the substrate findings are substrate-
  general; the hippocampus + dlpfc don't significantly affect direct
  binding capability at saturation.

Reuse: mirror longer_phase1_diagnostic.py's train_longer_phase1
helper byte-for-byte; only the substrate builder call changes
(enable_hippocampus_consolidation=False, enable_dlpfc_verb=False).
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
    _phase1_cache_path,
    _phase1_recipe,
    _phase1_train_kwargs,
    _freeze_phase1_gates,
    _all_words_word_to_idx,
    _all_pool_regions,
    _direct_pool_target,
    _N_WORDS_ORTHOGONAL,
)


def _build_v14_only_bridge(seed: int, tiny_synth: bool):
    """Build v14-only substrate (concept pools only; NO hippocampus;
    NO dlpfc). Mirrors _build_bridge_with_phase1_recipe except for the
    enable_hippocampus_consolidation + enable_dlpfc_verb flags."""
    from sim.config import (
        CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    recipe = _phase1_recipe(tiny_synth)
    n_lang_input = int(recipe["n_lang_input"])
    n_per_pool = int(recipe["n_per_pool"])
    n_fs_per_pool = int(recipe["n_fs_per_pool"])

    concept_internal_density = 0.05
    concept_exc_weight = 0.3
    concept_inh_weight = 0.8
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
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
        # V14-ONLY: disable hippocampus + dlpfc
        enable_hippocampus_consolidation=False,
        enable_dlpfc_verb=False,
    )

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
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def train_v14_only_phase1(seed: int, n_train_events: int, cache_dir: str):
    cache_path = _phase1_cache_path(cache_dir, seed)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Cache already exists at {cache_path}; skipping training.")
        return cache_path

    print(f"=== V14-only Phase-1 training at {n_train_events} events/word ===")
    print(f"Seed: {seed}; cache_dir: {cache_dir}")

    train_kwargs = _phase1_train_kwargs(False)
    bridge = _build_v14_only_bridge(int(seed), False)

    all_words_ordered = (
        list(cpd.DIRECTION_VOCAB)
        + list(cpd.NOUN_VOCAB)
        + list(cpd.VERB_VOCAB)
        + list(cpd.ADJECTIVE_VOCAB)
    )
    word_to_idx = {w: i for i, w in enumerate(all_words_ordered)}
    n_words_total = len(all_words_ordered)

    cpd.apply_concept_topographic_bias(
        bridge,
        n_lang_input=int(train_kwargs["n_lang_input"]),
        topographic_factor=float(train_kwargs["topographic_factor"]),
        off_target_factor=float(train_kwargs["off_target_factor"]),
        sparsity=float(train_kwargs["sparsity"]),
        orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
        n_words_for_orthogonal=int(n_words_total),
        word_to_idx=word_to_idx,
        skip_motor=False,
        verbose=False,
    )

    all_targets = []
    for word, action in cpd.DIRECTION_VOCAB.items():
        all_targets.append((word, f"motor_{action}"))
    for word, name in cpd.NOUN_VOCAB.items():
        all_targets.append((word, f"noun_pool_{name}"))
    for word, name in cpd.VERB_VOCAB.items():
        all_targets.append((word, f"verb_pool_{name}"))
    for word, name in cpd.ADJECTIVE_VOCAB.items():
        all_targets.append((word, f"adjective_pool_{name}"))

    print(f"Vocab: {len(all_targets)} (word, pool) targets")
    print(f"Total events: {len(all_targets)} x {n_train_events} = "
          f"{len(all_targets) * n_train_events}")

    rng = np.random.default_rng(int(seed))
    buffer = []
    for word, target in all_targets:
        for _ in range(n_train_events):
            buffer.append((word, target))
    rng.shuffle(buffer)

    print(f"Training buffer: {len(buffer)} events; starting train loop...")
    t_start = time.time()
    last_print = t_start
    for i, (word, target) in enumerate(buffer):
        cpd.train_word_to_pool(
            bridge, word, target, n_events=1, reset_steps=50,
            n_lang_input=int(train_kwargs["n_lang_input"]),
            n_lang_output=int(train_kwargs["n_lang_input"]),
            sparsity=float(train_kwargs["sparsity"]),
            orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
            n_words_for_orthogonal=int(n_words_total),
            word_to_idx=word_to_idx, verbose=False,
        )
        now = time.time()
        if now - last_print > 60:
            elapsed = now - t_start
            rate = (i + 1) / elapsed
            eta = (len(buffer) - i - 1) / rate / 60.0
            print(f"  step {i+1}/{len(buffer)} "
                  f"({100.0*(i+1)/len(buffer):.1f}%); elapsed {elapsed/60.0:.1f}min; "
                  f"ETA {eta:.1f}min")
            last_print = now

    elapsed_total = time.time() - t_start
    print(f"Training complete; {elapsed_total/60.0:.1f}min wall-clock")
    print(f"Saving checkpoint to {cache_path}...")
    bridge.save_checkpoint(str(cache_path))
    print(f"Saved.")
    return cache_path


def test_one_checkpoint_v14(seed, cache_dir, label):
    """Test direct binding on v14-only substrate."""
    print(f"\n=== {label} ===")
    bridge = _build_v14_only_bridge(seed=seed, tiny_synth=False)
    cache_path = _phase1_cache_path(cache_dir, seed)
    print(f"Loading {cache_path}")
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(False)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    all_pools = _all_pool_regions(enable_adjective=True)

    print(f"Querying {len(all_words)} trained words...")
    n_correct = 0
    per_word = []
    for word in all_words:
        try:
            target_pool = _direct_pool_target(word)
        except KeyError:
            continue
        per_pool = cpd.measure_pool_firing(
            bridge, word, all_pools,
            stim_steps=100, reset_steps=50, drive_pA=200.0, sparsity=0.05,
            n_lang_input=int(recipe_dims["n_lang_input"]),
            orthogonal_codes=True,
            n_words_for_orthogonal=int(n_words_for_orthogonal),
            word_to_idx=word_to_idx,
        )
        top_pool = max(per_pool.items(), key=lambda x: x[1])[0]
        correct = (top_pool == target_pool)
        if correct: n_correct += 1
        per_word.append({
            "word": word, "target_pool": target_pool, "top_pool": top_pool,
            "top_rate": float(per_pool[top_pool]),
            "target_rate": float(per_pool[target_pool]),
            "correct": correct,
        })
        marker = "OK " if correct else "XX "
        print(f"  {marker} {word:>8} -> target {target_pool:>22}; "
              f"top={top_pool:>22} rate={per_pool[top_pool]:.3f} "
              f"(target_rate={per_pool[target_pool]:.3f})")

    accuracy = n_correct / len(all_words)
    print(f"\n  {label}: {n_correct}/{len(all_words)} = {100.0*accuracy:.1f}% direct binding accuracy")
    return {
        "label": label, "cache_dir": cache_dir,
        "n_correct": n_correct, "n_total": len(all_words),
        "accuracy": accuracy, "per_word": per_word,
    }


def main():
    SEED = 42
    EVENTS = 800
    CACHE_DIR = "research/findings/raw/v14_only_per_regime/phase1_800ev"

    cache = train_v14_only_phase1(SEED, EVENTS, CACHE_DIR)
    result = test_one_checkpoint_v14(SEED, CACHE_DIR, f"v14-only 800ev seed {SEED}")

    bar = 0.80
    print(f"\n=== DIRECTION K RESULT (v14-only 800ev seed {SEED}) ===")
    print(f"  n_correct/n_total: {result['n_correct']}/{result['n_total']}")
    print(f"  accuracy: {100.0*result['accuracy']:.1f}%")
    print(f"  bar 0.80: {'PASS' if result['accuracy'] >= bar else 'FAIL'}")
    print(f"  vs unified 800ev seed 42 (15/16=93.8%): {'>=' if result['accuracy'] >= 15/16 else '<'}")

    out = "research/findings/raw/v14_only_phase1_diagnostic_seed42.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "events_per_word": EVENTS, "cache_dir": CACHE_DIR,
            "result": result,
            "unified_800ev_seed42_n_correct": 15,
            "unified_800ev_seed42_n_total": 16,
            "unified_800ev_seed42_accuracy": 15/16,
            "comparison_to_unified": (
                "matches" if result["accuracy"] == 15/16
                else ("v14_only_higher" if result["accuracy"] > 15/16 else "v14_only_lower")
            ),
        }, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
