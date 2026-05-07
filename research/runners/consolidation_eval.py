"""Phase 1.3 -- Hippocampus-OFF and sleep-recovery evaluation.

Two test modes per design:

1. evaluate_with_hippo_off(): drive hippocampus regions (ec, dg,
   ca3, ca1) with strong negative current (silence them), then
   measure W->A on cortex alone. Pass if accuracy stays at >=
   50% of pre-silence baseline -- proves cortex consolidated
   the patterns rather than relying on hippocampus.

2. evaluate_sleep_recovery(): standard Phase 1.4 + sleep cycle.
   Train Phase A primaries -> sleep cycle -> train Phase B
   synonyms -> sleep cycle -> measure primary retention. Pass
   if retention >= 80%. Tests whether consolidation prevents
   catastrophic forgetting more effectively than awake-only.

Both reuse evaluate_word_to_action's reset/stim/readout machinery.

Status: SKELETON. Untested on GPU. Pending GPU validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


HIPPO_REGIONS = ["ec", "dg", "dg_pv_basket", "ca3", "ca1"]


def evaluate_with_hippo_off(
    bridge,
    n_trials_per_word: int = 25,
    silence_current_pA: float = -200.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Silence hippocampus regions and run W->A test on cortex alone.

    Implementation: set strong negative external_input_current on
    ec/dg/ca3/ca1 indices to push them below firing threshold for
    the duration of the eval. Then call evaluate_word_to_action.

    Pass criterion (per design): accuracy >= 50% of pre-silence
    baseline. We treat this as 'consolidation proof' -- if cortex
    can produce W->A without hippocampus, it consolidated the
    patterns from sleep replay rather than relying on hippo route.

    Args:
        bridge: Bridge with hippocampus_consolidation enabled.
        n_trials_per_word: passed to evaluate_word_to_action.
        silence_current_pA: external current to apply to hippo
            regions during eval. -200 pA reliably silences cortex
            pyramidals.
        verbose: passed through.

    Returns dict with accuracy + confusion + pass.
    """
    import cupy as cp
    from research.runners.text_eval import evaluate_word_to_action

    rm = bridge.region_manager
    # Gather all hippo neuron indices
    hippo_indices = []
    for region_name in HIPPO_REGIONS:
        try:
            idx = rm.indices(region_name)
            if idx is not None:
                hippo_indices.extend(list(idx))
        except Exception:
            pass

    if not hippo_indices:
        # No hippocampus regions -- this eval doesn't apply
        return {
            "name": "hippo_off",
            "skipped": True,
            "reason": "no hippocampus regions in bridge",
        }

    hippo_arr = cp.asarray(hippo_indices, dtype=cp.int64)

    # Apply silencing: monkey-patch _run_one_simulation_step to
    # inject silencing current on every step. This is more robust
    # than a one-time current set because the standard eval loop
    # zeros external_input_current between trials.
    original_step = bridge._run_one_simulation_step

    def silenced_step():
        # Force hippo current to silence value before each step
        bridge.cp_external_input_current[hippo_arr] = float(
            silence_current_pA
        )
        return original_step()

    bridge._run_one_simulation_step = silenced_step
    try:
        wa = evaluate_word_to_action(
            bridge, n_trials_per_word=n_trials_per_word,
            stim_steps_per_trial=100, n_reset_steps=50,
            token_sparsity=0.1, verbose=verbose,
        )
    finally:
        # Restore
        bridge._run_one_simulation_step = original_step
        # Clear silencing current from any state
        bridge.cp_external_input_current[hippo_arr] = 0.0

    return {
        "name": "hippo_off",
        "accuracy": wa["accuracy"],
        "confusion_matrix": wa.get("confusion_matrix", {}),
        "n_hippo_neurons_silenced": len(hippo_indices),
        # Pass criterion deferred to caller (needs pre-silence baseline)
    }


def evaluate_consolidation_proof(
    bridge,
    n_trials_per_word: int = 25,
    silence_current_pA: float = -200.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Combined eval: pre-silence baseline + hippo-off + ratio.

    Returns:
        {
            "name": "consolidation_proof",
            "pre_silence_acc": float,
            "hippo_off_acc": float,
            "ratio": hippo_off / pre_silence,
            "pass": ratio >= 0.5,
        }
    """
    from research.runners.text_eval import evaluate_word_to_action

    if verbose:
        print("\n  [CONSOLIDATION] Pre-silence baseline", flush=True)
    wa_pre = evaluate_word_to_action(
        bridge, n_trials_per_word=n_trials_per_word,
        stim_steps_per_trial=100, n_reset_steps=50,
        token_sparsity=0.1, verbose=verbose,
    )
    pre_acc = wa_pre["accuracy"]
    if verbose:
        print(f"    Pre-silence W->A: {pre_acc:.1%}", flush=True)

    if verbose:
        print("\n  [CONSOLIDATION] Hippo-OFF test", flush=True)
    res = evaluate_with_hippo_off(
        bridge, n_trials_per_word=n_trials_per_word,
        silence_current_pA=silence_current_pA, verbose=verbose,
    )
    if res.get("skipped"):
        return {
            "name": "consolidation_proof",
            "skipped": True,
            "pre_silence_acc": pre_acc,
            "reason": res["reason"],
        }
    hippo_off_acc = res["accuracy"]
    ratio = (hippo_off_acc / pre_acc) if pre_acc > 0 else 0.0
    if verbose:
        print(f"    Hippo-OFF W->A: {hippo_off_acc:.1%}", flush=True)
        print(f"    Ratio (hippo-off / pre): {ratio:.0%}", flush=True)

    return {
        "name": "consolidation_proof",
        "pre_silence_acc": pre_acc,
        "hippo_off_acc": hippo_off_acc,
        "ratio": ratio,
        "pass": ratio >= 0.5,
        "pre_confusion": wa_pre.get("confusion_matrix", {}),
        "hippo_off_confusion": res.get("confusion_matrix", {}),
    }


def main():
    """Standalone consolidation eval: requires loading a checkpoint."""
    raise NotImplementedError(
        "Standalone CLI requires checkpoint loader for the 10+1 "
        "region consolidation architecture. Recommend calling "
        "evaluate_consolidation_proof() directly from "
        "consolidation_trainer.py's run() after training."
    )


if __name__ == "__main__":
    main()
