"""Lane C gate for learned episodic source support on the spiking substrate.

The learner receives a complete proposition and an external-experience event
during observation. Retrieval receives only a live proposition. The result is
source support from learned synaptic weights, without an expected-answer table.

Run one seed per worker for parallel validation::

    SIM_BACKEND=numpy python -m research.runners._laneC_plastic_source_memory_derisk \
      --seeds 42 --json research/findings/raw/lanes/metacog/plastic_source_s42.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners.plastic_source_memory import PlasticSourceConfig, PlasticSourceMemory
from tools.lab import attributable_to


def _facts(n_facts: int):
    return [(f"agent_{i}", f"action_{i}", f"patient_{i}") for i in range(int(n_facts))]


def _measure(memory: PlasticSourceMemory, rows):
    out = []
    for kind, cue, candidate in rows:
        rec = memory.support(kind=kind, cue=cue, candidate=candidate)
        out.append({
            "kind": kind,
            "cue": list(cue),
            "candidate": candidate,
            "support": rec["support"],
            "source_consistent": rec["source_consistent"],
            "bank_support": rec["bank_support"],
        })
    return out


def _rate(rows):
    return float(np.mean([bool(r["source_consistent"]) for r in rows])) if rows else 0.0


def _mean_support(rows):
    return float(np.mean([float(r["support"]) for r in rows])) if rows else 0.0


def evaluate_seed(seed: int, config: PlasticSourceConfig, n_facts: int, n_controls: int):
    t0 = time.time()
    facts = _facts(n_facts)
    controls = facts[: min(int(n_controls), len(facts))]
    memory = PlasticSourceMemory(seed=seed, config=config)

    initial_weights = memory.weight_summary()
    for agent, action, patient in controls:
        memory.observe(
            kind="what_does",
            cue=(agent, action),
            candidate=patient,
            learning_enabled=False,
        )
    frozen_weights = memory.weight_summary()
    no_learning_rows = _measure(
        memory,
        [("what_does", (agent, action), patient) for agent, action, patient in controls],
    )

    for agent, action, patient in facts:
        memory.observe(kind="what_does", cue=(agent, action), candidate=patient)
    learned_weights = memory.weight_summary()

    seen_rows = _measure(
        memory,
        [("what_does", (agent, action), patient) for agent, action, patient in facts],
    )
    wrong_rows = _measure(
        memory,
        [
            ("what_does", (agent, action), facts[(i + 1) % len(facts)][2])
            for i, (agent, action, _patient) in enumerate(facts)
        ],
    )
    unknown_rows = _measure(
        memory,
        [
            ("what_does", (f"unknown_agent_{i}", f"unknown_action_{i}"), patient)
            for i, (_agent, _action, patient) in enumerate(controls)
        ],
    )
    lesion_rows = []
    for agent, action, patient in controls:
        rec = memory.support(
            kind="what_does",
            cue=(agent, action),
            candidate=patient,
            lesion=True,
        )
        lesion_rows.append({
            "cue": [agent, action],
            "candidate": patient,
            "support": rec["support"],
            "source_consistent": rec["source_consistent"],
        })
    post_retrieval_weights = memory.weight_summary()
    source_path_attribution = attributable_to(
        "plastic proposition-to-source pathway",
        _mean_support(seen_rows),
        _mean_support(lesion_rows),
        warn_below=0.90,
    )

    permuted = PlasticSourceMemory(seed=seed + 500003, config=config)
    for i, (agent, action, _patient) in enumerate(controls):
        permuted.observe(
            kind="what_does",
            cue=(agent, action),
            candidate=controls[(i + 1) % len(controls)][2],
        )
    permuted_taught_rows = _measure(
        permuted,
        [
            ("what_does", (agent, action), controls[(i + 1) % len(controls)][2])
            for i, (agent, action, _patient) in enumerate(controls)
        ],
    )
    permuted_original_rows = _measure(
        permuted,
        [("what_does", (agent, action), patient) for agent, action, patient in controls],
    )

    seen_min = min(float(r["support"]) for r in seen_rows)
    wrong_max = max(float(r["support"]) for r in wrong_rows)
    components = {
        "weights_start_zero": bool(initial_weights["l1"] == 0.0),
        "learning_gate_off_keeps_weights_zero": bool(frozen_weights["l1"] == 0.0),
        "no_learning_accept_rate_zero": bool(_rate(no_learning_rows) == 0.0),
        "learning_grows_synaptic_weights": bool(learned_weights["l1"] > 0.0),
        "retrieval_keeps_learned_weights_frozen": bool(
            post_retrieval_weights["l1"] == learned_weights["l1"]
        ),
        "seen_accept_rate_at_least_0_95": bool(_rate(seen_rows) >= 0.95),
        "wrong_answer_false_accept_rate_zero": bool(_rate(wrong_rows) == 0.0),
        "unknown_proposition_false_accept_rate_zero": bool(_rate(unknown_rows) == 0.0),
        "seen_wrong_worst_case_margin_positive": bool(seen_min > wrong_max),
        "source_path_lesion_accept_rate_zero": bool(_rate(lesion_rows) == 0.0),
        "permuted_teaching_is_followed": bool(_rate(permuted_taught_rows) >= 0.90),
        "permuted_original_is_rejected": bool(_rate(permuted_original_rows) == 0.0),
    }
    return {
        "seed": int(seed),
        "verdict": "GO" if all(components.values()) else "NEGATIVE",
        "components": components,
        "metrics": {
            "seen_accept_rate": _rate(seen_rows),
            "wrong_answer_false_accept_rate": _rate(wrong_rows),
            "unknown_false_accept_rate": _rate(unknown_rows),
            "lesion_accept_rate": _rate(lesion_rows),
            "permuted_taught_accept_rate": _rate(permuted_taught_rows),
            "permuted_original_accept_rate": _rate(permuted_original_rows),
            "seen_support_min": seen_min,
            "seen_support_mean": _mean_support(seen_rows),
            "wrong_support_max": wrong_max,
            "wrong_support_mean": _mean_support(wrong_rows),
            "no_learning_support_mean": _mean_support(no_learning_rows),
            "lesion_support_mean": _mean_support(lesion_rows),
            "permuted_taught_support_mean": _mean_support(permuted_taught_rows),
            "permuted_original_support_mean": _mean_support(permuted_original_rows),
            "worst_case_seen_wrong_margin": float(seen_min - wrong_max),
            "retrieval_weight_l1_delta": float(
                post_retrieval_weights["l1"] - learned_weights["l1"]
            ),
            "source_path_attribution": source_path_attribution,
        },
        "weights": {
            "initial": initial_weights,
            "learning_disabled": frozen_weights,
            "learned": learned_weights,
        },
        "samples": {
            "seen": seen_rows[:3],
            "wrong": wrong_rows[:3],
            "unknown": unknown_rows[:3],
            "lesion": lesion_rows[:3],
            "permuted_taught": permuted_taught_rows[:3],
            "permuted_original": permuted_original_rows[:3],
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Gate learned episodic source support on the spiking substrate.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-facts", type=int, default=48)
    ap.add_argument("--n-controls", type=int, default=12)
    ap.add_argument("--proposition-neurons-per-bank", type=int, default=16384)
    ap.add_argument("--support-threshold", type=float, default=0.34)
    ap.add_argument(
        "--json",
        default="research/findings/raw/lanes/metacog/laneC_plastic_source_memory_6seed.json",
    )
    args = ap.parse_args()

    config = replace(
        PlasticSourceConfig(),
        proposition_neurons_per_bank=int(args.proposition_neurons_per_bank),
        support_threshold=float(args.support_threshold),
    )
    print(
        "[plastic-source] learned proposition -> source support | "
        f"backend={os.environ.get('SIM_BACKEND')} seeds={args.seeds} "
        f"facts={args.n_facts} config={asdict(config)}",
        flush=True,
    )
    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        row = evaluate_seed(seed, config, args.n_facts, args.n_controls)
        per_seed.append(row)
        print(
            f"[plastic-source] seed={seed} verdict={row['verdict']} "
            f"seen={row['metrics']['seen_accept_rate']:.3f} "
            f"wrong_FA={row['metrics']['wrong_answer_false_accept_rate']:.3f} "
            f"margin={row['metrics']['worst_case_seen_wrong_margin']:+.3f}",
            flush=True,
        )

    verdict = "GO" if all(row["verdict"] == "GO" for row in per_seed) else "NEGATIVE"
    out = {
        "runner": "research/runners/_laneC_plastic_source_memory_derisk.py",
        "faculty": "Lane C episodic source monitoring for honesty",
        "mechanism": "gated symmetric Hebbian proposition-to-external-source association on a spiking bridge",
        "backend": os.environ.get("SIM_BACKEND", "(unset)"),
        "seeds": list(args.seeds),
        "n_facts": int(args.n_facts),
        "n_controls": int(args.n_controls),
        "config": asdict(config),
        "verdict": verdict,
        "per_seed": per_seed,
        "preconditions": [
            {
                "name": "all_required_controls_recorded",
                "ok": all(
                    all(
                        key in row["components"]
                        for key in (
                            "learning_gate_off_keeps_weights_zero",
                            "retrieval_keeps_learned_weights_frozen",
                            "source_path_lesion_accept_rate_zero",
                            "permuted_teaching_is_followed",
                        )
                    )
                    for row in per_seed
                ),
            },
            {
                "name": "verdict_derived_from_seed_components",
                "ok": verdict == (
                    "GO" if all(all(row["components"].values()) for row in per_seed) else "NEGATIVE"
                ),
            },
        ],
        "honest_scope": (
            "The proposition-to-source association is learned in zero-initialized spiking-bridge synapses and "
            "retrieval receives no expected answer or stored proposition table. Proposition hashing, the explicit "
            "external-source teaching event, and the dedicated bridge remain scaffolds. This gate does not claim "
            "co-residency with the full brain or a learned ACC/aPFC confidence circuit."
        ),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[plastic-source] === VERDICT: {verdict} === wrote {out_path}", flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
