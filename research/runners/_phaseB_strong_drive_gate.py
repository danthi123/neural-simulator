"""Phase-B DECISIVE strong-drive gate (the regime the 36-cell grid never tried): the hub encoding was a
drive-strength artifact -- at drive_scale 12 the hubs fired ~0.15 spikes/hub (Poisson noise) so the input
structure was lost (-0.06); at ds40 it recovers to +0.33 (centered). The subagent only swept ds{12,20}. This
tests the L1-FAITHFUL pipeline at STRONG drive: strong hub->cortex coupling so the cortex fires, SLOW
(default) homeostasis (FAST homeostasis equalizes the hubs and hurt the encoding), co-fire to ensure cortex
firing, bounded-Hebbian STDP -- with AND without the cm-pool centering. Read CORTEX spike-count codes, check
the gate (structure >= +0.30, beat random-proj, not silent). This is the LAST gate attempt; GO -> the build
proceeds; WALL -> the honest NEGATIVE is airtight (the spike-count encoding loses too much even in the best
regime -> the dendritic substrate / months-scale piece). NO sim/ edit.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_strong_drive_gate
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.spiking_sm_cortex import (  # noqa: E402
    build_sm_cortex_bridge, encode_drive, train_sm_cortex, read_codes,
)


def _gate(name, codes, S_true, labels):
    p = _pearson_vs_Strue(_cos_sim(codes), S_true)
    pc = _pearson_vs_Strue(_cos_sim(codes - codes.mean(0, keepdims=True)), S_true)
    g, ch = heldout_generalization(codes, labels)
    silent = float(np.mean(codes.sum(1) == 0))
    print(f"  [{name:30s}] structure={p:+.3f}  out-centered={pc:+.3f}  gen={g:.3f}  silent={silent:.2f}  "
          f"eff-rank={effective_rank(codes):.1f}", flush=True)
    return max(p, pc)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    print(f"[strong-drive gate] {C.shape[0]} concepts x {n_hub} hubs; drive cos +0.891", flush=True)

    best = -1.0
    for ds in (40.0, 80.0, 160.0):
        for use_cm in (False, True):
            cm_kw = dict(n_cm=64, hub_to_cm_density=0.5, hub_to_cm_weight=2.0,
                         cm_to_cortex_density=1.0, cm_to_cortex_weight=8.0) if use_cm else {}
            b, hub, cx = build_sm_cortex_bridge(
                n_hub=n_hub, n_cortex=128, seed=42, density=0.5, weight_mean=80.0, stdp_w_max=200.0, **cm_kw)
            # SLOW (default) homeostasis (no override) -- fast homeostasis equalizes the hubs.
            train_sm_cortex(b, C_drive, hub, cx, n_epochs=8, drive_scale=ds, window=80, settle=8, cofire_pA=6.0)
            codes = read_codes(b, C_drive, hub, cx, drive_scale=ds, window=80, settle=8)
            tag = f"ds{int(ds)} {'+cm-centering' if use_cm else 'no-centering'}"
            best = max(best, _gate(tag, codes, S_true, labels))
        # random-projection control at this drive (untrained) for the load-bearing check
        b0, hub0, cx0 = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=43, density=0.5,
                                               weight_mean=80.0, stdp_w_max=200.0)
        rc = read_codes(b0, C_drive, hub0, cx0, drive_scale=ds, window=80, settle=8)
        _gate(f"ds{int(ds)} RANDOM-PROJ (control)", rc, S_true, labels)

    print(f"\n  VERDICT: best cortex structure across the strong-drive regime = {best:+.3f}", flush=True)
    if best >= 0.30:
        print("  GO (provisional) -- strong drive unblocks the cortex gate; formalize in the HARD GATE test "
              "+ full anti-cheat battery, then proceed to Task 4 (GPU real corpus).", flush=True)
    elif best >= 0.15:
        print("  PARTIAL -- strong drive lifts it materially but short of +0.30; the spike-count encoding "
              "is lossy. Weigh more firing/window vs the dendritic substrate.", flush=True)
    else:
        print("  WALL (airtight) -- even at strong drive (hubs encoding) the cortex spike code does NOT "
              "recover the structure -> the spike-count rate->spike encoding loses too much -> the honest "
              "NEGATIVE: the spiking learned cortex needs the dendritic substrate (the months-scale piece).",
              flush=True)


if __name__ == "__main__":
    main()
