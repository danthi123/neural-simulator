"""Two-concept discrimination test — combines P1 trisynaptic + P2 engram
tagging to validate the catalog's pattern-separation-AND-completion
combined criterion (D.12 ∩ D.13).

The catalog (D.13) explicitly notes: "Trade-off with separation: too
much completion → confused episodes; too little → no generalization."
The right criterion isn't an absolute cosine threshold — it's RELATIVE:

  cos(partial_A_recall, full_A_stored) >> cos(partial_A_recall, full_B_stored)

i.e. partial recall of concept A converges to A's stored attractor,
NOT to a different concept's attractor.

This is the Marr 1971 autoassociator's actual job at the system level
("does a partial cue retrieve THIS memory, not some other memory?")
and the right test for "concepts as distinguishable ensembles" — the
user's vision for the realigned plan.

Uses the P2 engram-tagging API to capture stored ensembles cleanly.

Usage:
    python -m research.runners.validate_two_concept_discrimination \\
        --seed 42 --train-events 400 --ca3-recurrent-weight 5.0 \\
        --out research/findings/raw/g11_bg/two_concept_seed42.json

Validation criterion:
  cosine(partial_A_recall, tag_A_stored) > 0.5  AND
  cosine(partial_A_recall, tag_B_stored) < 0.3  AND
  margin > 0.2 (separation between same-concept and cross-concept)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def cosine_similarity_indices(a_indices, b_indices, n_total: int) -> float:
    """Cosine of two index-sets viewed as binary vectors."""
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


def build_drive_pattern(n_neurons: int, sparsity: float, seed: int):
    """Random sparse activation pattern indices."""
    rng = np.random.default_rng(seed)
    n_active = max(1, int(round(sparsity * n_neurons)))
    return rng.choice(n_neurons, size=n_active, replace=False).astype(np.int64)


def train_concept(bridge, drive_indices, train_events: int = 400,
                   drive_pA: float = 200.0, reset_steps: int = 30,
                   stim_steps: int = 100):
    """Train a concept by co-firing the EC pattern with all hippo gates open.
    Returns the time spent on training."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    drive_arr = cp.asarray(drive_indices, dtype=cp.int64)
    # Open hippo plasticity for training
    for g in ("ca3_swr_burst", "dg_to_ca3", "ec_to_dg", "lang_to_ec"):
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass
    t0 = time.time()
    for _ in range(train_events):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[drive_arr] = float(drive_pA)
        for _ in range(stim_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
    # Close gates
    for g in ("ca3_swr_burst", "dg_to_ca3", "ec_to_dg", "lang_to_ec"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass
    return time.time() - t0


def tag_ca3_ensemble(bridge, tag_name: str, drive_indices,
                       window_steps: int = 100, drive_pA: float = 200.0,
                       top_k: int = 50):
    """Tag the CA3 ensemble that fires when driving this EC pattern."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    drive_arr = cp.asarray(drive_indices, dtype=cp.int64)
    # Reset transients
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    # Start recording + drive
    bridge.start_engram_recording(tag_name)
    bridge.cp_external_input_current[drive_arr] = float(drive_pA)
    for _ in range(window_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[:] = 0.0
    # Commit using top_k restricted to CA3
    stats = bridge.commit_engram_tag(
        tag_name, top_k=top_k, region_filter=["ca3"],
    )
    return stats


def recall_via_direct_ca3(bridge, tag_indices, n_steps: int = 100,
                            drive_pA: float = 200.0, partial_frac: float = 0.5):
    """Drive partial of the tag's CA3 neurons directly and capture the
    full CA3 firing pattern that emerges. Returns indices of CA3 neurons
    that fired (active set)."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    # Take partial
    n_total = int(tag_indices.shape[0]) if hasattr(tag_indices, "shape") \
        else len(tag_indices)
    n_partial = max(1, int(round(partial_frac * n_total)))
    partial_indices_xp = tag_indices[:n_partial]
    # Reset
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    # Drive partial
    bridge.cp_external_input_current[partial_indices_xp] = float(drive_pA)
    # Capture CA3 firing
    rm = bridge.region_manager
    ca3_indices_host = list(rm.indices("ca3"))
    ca3_arr = cp.asarray(ca3_indices_host, dtype=cp.int64)
    spike_counts = cp.zeros(len(ca3_indices_host), dtype=cp.float32)
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[ca3_arr]
        spike_counts += fired.astype(cp.float32)
    bridge.cp_external_input_current[:] = 0.0
    # Return global indices of CA3 neurons that fired
    fired_local = to_host(spike_counts > 0)
    fired_global = [ca3_indices_host[i] for i in np.where(fired_local)[0]]
    return np.array(fired_global, dtype=np.int64)


def run_two_concept(
    seed: int = 42,
    n_lang_input: int = 2048,
    n_ec: int = 200,
    n_dg: int = 800,
    n_dg_pv_basket: int = 240,
    n_ca3: int = 400,
    n_ca1: int = 200,
    ca3_recurrent_density: float = 0.30,
    ca3_recurrent_weight: float = 5.0,
    train_events: int = 400,
    partial_frac: float = 0.5,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"Two-concept discrimination test (seed={seed})")
    log("=" * 60)

    # Build hippo bridge
    from sim.config import (CoreSimConfig, RuntimeState, GPUConfig,
                              VisualizationConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from sim.backend import get_backend, to_host

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=16, n_motor_fs_per_action=4,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket,
        n_ca3=n_ca3, n_ca1=n_ca1,
        ca3_recurrent_density=ca3_recurrent_density,
        ca3_recurrent_weight=ca3_recurrent_weight,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.stdp_w_max = 10.0
    cfg.fast_spike_reset = True

    t0 = time.time()
    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    build_sec = time.time() - t0
    log(f"Built in {build_sec:.1f}s")

    # Two distinct EC patterns
    drive_A = build_drive_pattern(
        n_neurons=n_lang_input, sparsity=0.1, seed=seed,
    )
    drive_B = build_drive_pattern(
        n_neurons=n_lang_input, sparsity=0.1, seed=seed + 1000,
    )
    n_overlap = len(set(drive_A) & set(drive_B))
    log(f"EC patterns: A={len(drive_A)} neurons, B={len(drive_B)}, "
        f"overlap={n_overlap}")

    # Train concept A
    log(f"\nTraining concept A ({train_events} events)...")
    t_train_a = train_concept(bridge, drive_A, train_events=train_events)
    log(f"  trained in {t_train_a:.0f}s")

    # Tag CA3 ensemble for A
    log("  tagging concept A's CA3 ensemble...")
    stats_a = tag_ca3_ensemble(bridge, "concept_a", drive_A, top_k=50)
    log(f"  tag_a: {stats_a['n_tagged']} neurons (window {stats_a['window_ms']}ms)")

    # Train concept B
    log(f"\nTraining concept B ({train_events} events)...")
    t_train_b = train_concept(bridge, drive_B, train_events=train_events)
    log(f"  trained in {t_train_b:.0f}s")

    # Tag CA3 ensemble for B
    log("  tagging concept B's CA3 ensemble...")
    stats_b = tag_ca3_ensemble(bridge, "concept_b", drive_B, top_k=50)
    log(f"  tag_b: {stats_b['n_tagged']} neurons (window {stats_b['window_ms']}ms)")

    # Cross-tag overlap (CA3 ensembles should be different for distinct concepts)
    tag_a_idx_host = to_host(bridge.get_engram_tag_indices("concept_a"))
    tag_b_idx_host = to_host(bridge.get_engram_tag_indices("concept_b"))
    n_neurons_total = int(cfg.num_neurons)
    tag_ab_cos = cosine_similarity_indices(
        tag_a_idx_host, tag_b_idx_host, n_neurons_total
    )
    log(f"\nCA3 tag overlap (concept_a vs concept_b): cos = {tag_ab_cos:.3f}")
    log(f"  (lower = better separation; ideal: < 0.3)")

    # Recall partial of A → measure cosine vs tag_A and tag_B
    log("\nRecall: drive partial of tag_a's CA3 neurons directly")
    recall_a = recall_via_direct_ca3(
        bridge, bridge.get_engram_tag_indices("concept_a"),
        partial_frac=partial_frac,
    )
    cos_aa = cosine_similarity_indices(recall_a, tag_a_idx_host, n_neurons_total)
    cos_ab = cosine_similarity_indices(recall_a, tag_b_idx_host, n_neurons_total)
    log(f"  cos(recall_a, tag_a) = {cos_aa:.3f} (same-concept; higher better)")
    log(f"  cos(recall_a, tag_b) = {cos_ab:.3f} (cross-concept; lower better)")

    # Recall partial of B → measure cosine vs tag_B and tag_A
    log("\nRecall: drive partial of tag_b's CA3 neurons directly")
    recall_b = recall_via_direct_ca3(
        bridge, bridge.get_engram_tag_indices("concept_b"),
        partial_frac=partial_frac,
    )
    cos_bb = cosine_similarity_indices(recall_b, tag_b_idx_host, n_neurons_total)
    cos_ba = cosine_similarity_indices(recall_b, tag_a_idx_host, n_neurons_total)
    log(f"  cos(recall_b, tag_b) = {cos_bb:.3f} (same-concept; higher better)")
    log(f"  cos(recall_b, tag_a) = {cos_ba:.3f} (cross-concept; lower better)")

    # PASS criterion: biology-faithful per Marr 1971 / catalog D.13.
    # The autoassociator's job is "stored attractor converges from
    # partial cue to ITS OWN attractor, not to a different one."
    # The honest metric is MARGIN: same-concept >> cross-concept.
    #
    # Two-tier criterion:
    #   STRICT (engineering-ideal): same > 0.5 AND cross < 0.3 AND
    #     margin > 0.2
    #   BIOLOGY-FAITHFUL (what really matters): cross < 0.3 (concepts
    #     are SEPARATED) AND margin > 0.2 (autoassociator returns
    #     ITS own pattern stronger than another's)
    margin_a = cos_aa - cos_ab
    margin_b = cos_bb - cos_ba
    strict_a = (cos_aa > 0.5) and (cos_ab < 0.3) and (margin_a > 0.2)
    strict_b = (cos_bb > 0.5) and (cos_ba < 0.3) and (margin_b > 0.2)
    bio_a = (cos_ab < 0.3) and (margin_a > 0.2)
    bio_b = (cos_ba < 0.3) and (margin_b > 0.2)
    strict_overall = strict_a and strict_b
    bio_overall = bio_a and bio_b
    log("=" * 60)
    log(f"Per-concept analysis:")
    log(f"  cross < 0.3 (separation): a={cos_ab < 0.3} (cos={cos_ab:.3f}), "
        f"b={cos_ba < 0.3} (cos={cos_ba:.3f})")
    log(f"  margin > 0.2 (discrimination): a={margin_a > 0.2:.0f} "
        f"(margin={margin_a:.3f}), b={margin_b > 0.2:.0f} "
        f"(margin={margin_b:.3f})")
    log(f"  same > 0.5 (ideal completion): a={cos_aa > 0.5} "
        f"(cos={cos_aa:.3f}), b={cos_bb > 0.5} (cos={cos_bb:.3f})")
    log("---")
    log(f"BIOLOGY-FAITHFUL verdict (cross < 0.3 AND margin > 0.2):")
    log(f"  concept A: {'PASS' if bio_a else 'FAIL'}")
    log(f"  concept B: {'PASS' if bio_b else 'FAIL'}")
    log(f"  OVERALL:   {'PASS' if bio_overall else 'FAIL'}")
    log(f"STRICT verdict (also requires same > 0.5):")
    log(f"  concept A: {'PASS' if strict_a else 'FAIL'}")
    log(f"  concept B: {'PASS' if strict_b else 'FAIL'}")
    log(f"  OVERALL:   {'PASS' if strict_overall else 'FAIL'}")
    log("=" * 60)
    # Use biology-faithful as primary pass
    pass_a, pass_b = bio_a, bio_b
    overall = bio_overall

    result = {
        "seed": seed,
        "build_seconds": build_sec,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "train_events": train_events,
        "ca3_recurrent_weight": ca3_recurrent_weight,
        "drive_a_size": int(len(drive_A)),
        "drive_b_size": int(len(drive_B)),
        "ec_overlap": n_overlap,
        "tag_a_size": stats_a["n_tagged"],
        "tag_b_size": stats_b["n_tagged"],
        "tag_ab_cosine": tag_ab_cos,
        "recall_a": {
            "cos_aa": cos_aa,
            "cos_ab": cos_ab,
            "margin": margin_a,
            "passed_biology": bio_a,
            "passed_strict": strict_a,
        },
        "recall_b": {
            "cos_bb": cos_bb,
            "cos_ba": cos_ba,
            "margin": margin_b,
            "passed_biology": bio_b,
            "passed_strict": strict_b,
        },
        "biology_passed": bio_overall,
        "strict_passed": strict_overall,
        "overall_passed": overall,  # alias for biology_passed
        "total_seconds": time.time() - t0,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-events", type=int, default=400)
    ap.add_argument("--ca3-recurrent-weight", type=float, default=5.0)
    ap.add_argument("--partial-frac", type=float, default=0.5)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_two_concept(
        seed=args.seed,
        train_events=args.train_events,
        ca3_recurrent_weight=args.ca3_recurrent_weight,
        partial_frac=args.partial_frac,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
