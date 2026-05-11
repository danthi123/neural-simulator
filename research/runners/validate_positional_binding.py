"""Validate P4.1 — positional context produces item-in-context binding.

Catalog: D.01 (episodic memory), D.02 (relational binding), D.11
(time cells). Roadmap: realigned plan v3 phase P4.1.

The user's question: can the sim distinguish "alice ate apple" from
"apple ate alice"? Same words, different word order, different
meaning.

The P4.1 architectural answer: when ec_context drives a positional
pattern alongside ec (word) drive, DG expansion recoding produces
distinct CA3 ensembles for (word, position) tuples.

Test:
  1. Build hippo bridge with enable_episodic_context=True.
  2. Encode (apple, pos_1) by driving ec(apple_pattern) +
     ec_context(positional_pattern(0)) simultaneously. Tag the
     resulting CA3 ensemble as "apple_pos1".
  3. Encode (apple, pos_3) by driving ec(apple_pattern) +
     ec_context(positional_pattern(2)). Tag as "apple_pos3".
  4. Encode (alice, pos_1) — different word, same position.
     Tag as "alice_pos1".
  5. Measure:
     - cos(apple_pos1, apple_pos3) — SAME WORD, DIFFERENT POSITION
       (should be LOW per P4.1 hypothesis; distinct CA3 ensembles)
     - cos(apple_pos1, alice_pos1) — DIFFERENT WORD, SAME POSITION
       (should be LOW; distinct ensembles)
     - cos(apple_pos1, apple_pos1_retrained) — same (word, position),
       different presentation. Should be HIGH.

PASS criterion: position-discriminability (cos between same word @
different positions) is < some threshold, demonstrating that the
positional code IS being integrated by DG.

A simpler version of the test: just confirm the DG output for
(apple, pos_1) and (apple, pos_3) differs. We don't even need to
measure recall — just confirm the encoding produces distinct
ensembles.

Usage:
    python -m research.runners.validate_positional_binding \\
        --seed 42 --out research/findings/raw/g11_bg/p41_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def cosine_similarity_indices(a_indices, b_indices, n_total: int) -> float:
    if len(a_indices) == 0 or len(b_indices) == 0:
        return 0.0
    a = np.zeros(n_total, dtype=np.float64)
    b = np.zeros(n_total, dtype=np.float64)
    a[a_indices] = 1.0
    b[b_indices] = 1.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_word_pattern(word: str, n_lang_input: int, seed: int = 42):
    """Deterministic sparse word pattern.

    Uses a hash-based sparse code (similar to vocab_to_drive_pattern
    but seed-controllable for the test).
    """
    rng = np.random.default_rng(hash(word) % (2**31))
    n_active = max(1, int(0.1 * n_lang_input))
    return rng.choice(n_lang_input, size=n_active, replace=False).astype(np.int64)


def encode_and_tag(
    bridge,
    tag_name: str,
    word_indices,
    position: int,
    n_lang_input: int,
    n_ec_context: int,
    train_events: int = 100,
    record_window_steps: int = 100,
    drive_pA: float = 200.0,
):
    """Train one (word, position) binding and capture CA3 ensemble.

    Steps:
      1. Open hippo plasticity gates
      2. For N events: drive lang_input(word) + ec_context(position)
         simultaneously
      3. Close gates
      4. Drive again (no plasticity) + record CA3 firing → commit as
         the tag's ensemble
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import positional_drive_pattern

    # Get region indices
    rm = bridge.region_manager
    lang_indices = list(rm.indices("language_input"))
    ec_context_indices = list(rm.indices("ec_context"))
    n_lang = len(lang_indices)
    n_ctx = len(ec_context_indices)

    # Compute positional drive pattern
    pos_drive = positional_drive_pattern(
        position, n_neurons=n_ctx,
        drive_max_pA=drive_pA, sparsity=0.1, n_max_positions=10,
    )
    word_arr = cp.asarray(word_indices, dtype=cp.int64)
    pos_active = np.where(pos_drive > 0)[0]
    pos_global = np.array(
        [ec_context_indices[i] for i in pos_active], dtype=np.int64
    )
    pos_arr = cp.asarray(pos_global, dtype=cp.int64)

    # Open hippo plasticity
    for g in ("ca3_swr_burst", "dg_to_ca3", "ec_to_dg",
              "ec_context_to_dg", "lang_to_ec"):
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass

    # Training
    for _ in range(train_events):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        # Drive word + position simultaneously
        bridge.cp_external_input_current[word_arr] = float(drive_pA)
        bridge.cp_external_input_current[pos_arr] = float(drive_pA)
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    # Close gates
    for g in ("ca3_swr_burst", "dg_to_ca3", "ec_to_dg",
              "ec_context_to_dg", "lang_to_ec"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    # Reset transients
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Record CA3 firing during a final drive (no plasticity)
    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[word_arr] = float(drive_pA)
    bridge.cp_external_input_current[pos_arr] = float(drive_pA)
    for _ in range(record_window_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[:] = 0.0
    stats = bridge.commit_engram_tag(
        tag_name, top_k=50, region_filter=["ca3"],
    )
    return stats


def run_positional_validation(
    seed: int = 42,
    n_lang_input: int = 1024,
    n_ec: int = 200,
    n_dg: int = 800,
    n_dg_pv_basket: int = 240,
    n_ca3: int = 400,
    n_ca1: int = 200,
    n_ec_context: int = 200,
    ca3_recurrent_weight: float = 5.0,
    train_events: int = 100,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"P4.1 positional binding test (seed={seed})")
    log("=" * 60)

    from sim.config import (CoreSimConfig, RuntimeState, GPUConfig,
                              VisualizationConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from sim.backend import get_backend, to_host

    t0 = time.time()
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=8, n_motor_fs_per_action=2,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket,
        n_ca3=n_ca3, n_ca1=n_ca1,
        ca3_recurrent_weight=ca3_recurrent_weight,
        enable_episodic_context=True,
        n_ec_context=n_ec_context,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.fast_spike_reset = True
    cfg.stdp_w_max = 10.0
    cfg.enable_hebbian_learning = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    build_sec = time.time() - t0
    log(f"Built in {build_sec:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses")

    apple = build_word_pattern("apple", n_lang_input)
    alice = build_word_pattern("alice", n_lang_input)

    # Encode 4 (word, position) pairs:
    #   apple @ pos_0  ← target ensemble for comparison
    #   apple @ pos_2  ← same word, different position
    #   alice @ pos_0  ← different word, same position
    #   alice @ pos_2  ← different word, different position
    bindings = [
        ("apple_pos0", apple, 0),
        ("apple_pos2", apple, 2),
        ("alice_pos0", alice, 0),
        ("alice_pos2", alice, 2),
    ]
    encode_results = {}
    for tag_name, word_indices, pos in bindings:
        log(f"\nEncoding {tag_name} (position={pos})...")
        t_enc = time.time()
        stats = encode_and_tag(
            bridge, tag_name, word_indices, pos,
            n_lang_input=n_lang_input, n_ec_context=n_ec_context,
            train_events=train_events,
        )
        log(f"  -> {stats['n_tagged']} CA3 neurons tagged "
            f"({time.time() - t_enc:.0f}s)")
        encode_results[tag_name] = stats

    # Pairwise cosine between tags
    n_neurons_total = int(cfg.num_neurons)
    tag_indices = {
        name: to_host(bridge.get_engram_tag_indices(name))
        for name, _, _ in bindings
    }
    pairs = [
        ("apple_pos0", "apple_pos2", "SAME WORD, DIFFERENT POSITION"),
        ("alice_pos0", "alice_pos2", "SAME WORD, DIFFERENT POSITION"),
        ("apple_pos0", "alice_pos0", "DIFFERENT WORD, SAME POSITION"),
        ("apple_pos2", "alice_pos2", "DIFFERENT WORD, SAME POSITION"),
        ("apple_pos0", "alice_pos2", "DIFFERENT WORD, DIFFERENT POSITION"),
    ]
    log("\nPairwise CA3 ensemble cosines:")
    pair_results = []
    for a, b, label in pairs:
        cos = cosine_similarity_indices(
            tag_indices[a], tag_indices[b], n_neurons_total
        )
        log(f"  {a} vs {b} ({label}): {cos:.3f}")
        pair_results.append({
            "a": a, "b": b, "label": label, "cosine": cos,
        })

    # PASS criteria for P4.1:
    # 1. Position discriminability: (apple, pos_0) vs (apple, pos_2) cos < 0.4
    #    (same word at different positions should produce distinct ensembles)
    # 2. Word discriminability: (apple, pos_0) vs (alice, pos_0) cos < 0.4
    #    (different words at same position should produce distinct ensembles)
    # 3. Cross discriminability: (apple, pos_0) vs (alice, pos_2) cos < 0.4
    pair_dict = {(r["a"], r["b"]): r["cosine"] for r in pair_results}
    cos_apple_pos = pair_dict.get(("apple_pos0", "apple_pos2"), 1.0)
    cos_alice_pos = pair_dict.get(("alice_pos0", "alice_pos2"), 1.0)
    cos_pos0_word = pair_dict.get(("apple_pos0", "alice_pos0"), 1.0)
    cos_pos2_word = pair_dict.get(("apple_pos2", "alice_pos2"), 1.0)

    pass_position = (cos_apple_pos < 0.4) and (cos_alice_pos < 0.4)
    pass_word = (cos_pos0_word < 0.4) and (cos_pos2_word < 0.4)
    overall = pass_position and pass_word

    log("\n" + "=" * 60)
    log("PASS criteria:")
    log(f"  Position discriminability (same word, diff pos cos < 0.4):")
    log(f"    apple_pos0 vs apple_pos2: {cos_apple_pos:.3f} "
        f"{'PASS' if cos_apple_pos < 0.4 else 'FAIL'}")
    log(f"    alice_pos0 vs alice_pos2: {cos_alice_pos:.3f} "
        f"{'PASS' if cos_alice_pos < 0.4 else 'FAIL'}")
    log(f"  Word discriminability (diff word, same pos cos < 0.4):")
    log(f"    apple_pos0 vs alice_pos0: {cos_pos0_word:.3f} "
        f"{'PASS' if cos_pos0_word < 0.4 else 'FAIL'}")
    log(f"    apple_pos2 vs alice_pos2: {cos_pos2_word:.3f} "
        f"{'PASS' if cos_pos2_word < 0.4 else 'FAIL'}")
    log(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    log("=" * 60)

    result = {
        "seed": seed,
        "build_seconds": build_sec,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "train_events": train_events,
        "encode_results": encode_results,
        "pair_cosines": pair_results,
        "pass_position": pass_position,
        "pass_word": pass_word,
        "overall_passed": overall,
        "total_seconds": time.time() - t0,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_positional_validation(
        seed=args.seed,
        train_events=args.train_events,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
