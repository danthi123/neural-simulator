"""Phase-B GO-confirmation: the dense-firing readout cracked the spike-count boundary (+0.42 untrained). Now
the RIGOROUS gate in the dense regime -- the LEARNED cortex + the full anti-cheat battery -- to confirm GO
WITHOUT a 3rd premature positive (tonight: WALL -> BOUNDARY -> crack; be disciplined). The untrained random
projection ALREADY reaches +0.42 (the uncentered category-magnitude structure survives a dense readout), so
the load-bearing question is whether the LEARNED cortex (STDP) GENERALIZES to held-out concepts where the
random projection does not (the L1 distinction: random gen ~chance, learned gen 0.875). Controls: random-proj
baseline (same dense regime), permuted-similarity (~0), held-out generalization (learned > random).

GO ⇒ the spiking learned cortex works (Task-3 HARD GATE PASSES in the dense regime); proceed to Task 4 (GPU
real corpus). PARTIAL ⇒ the structure transmits but the LEARNING isn't load-bearing (a fixed dense random
projection suffices, not a "learned cortex") -- honest, still useful (the dense readout is the fix), reframe.
NO sim/ edit. Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_dense_gate
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


def _stat(name, codes, S_true, labels):
    p = _pearson_vs_Strue(_cos_sim(codes), S_true)
    g, ch = heldout_generalization(codes, labels)
    sil = float(np.mean(codes.sum(1) == 0))
    print(f"  [{name:22s}] Pearson={p:+.3f}  gen={g:.3f} (chance {ch:.3f})  silent={sil:.2f}  "
          f"eff-rank={effective_rank(codes):.1f}", flush=True)
    return p, g, ch


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    # the DENSE regime that cracked it: strong coupling + long window + homeostasis off.
    bp = dict(n_hub=n_hub, n_cortex=128, density=0.5, weight_mean=400.0, stdp_w_max=2000.0,
              enable_homeostasis=False)
    rp = dict(drive_scale=40.0, window=1000, settle=8)
    print(f"[dense gate] {C.shape[0]}c x {n_hub}h; DENSE regime (wm400, homeo off, ds40, win1000)", flush=True)

    # LEARNED cortex
    bL, hub, cx = build_sm_cortex_bridge(seed=42, **bp)
    hub = np.asarray(hub); cx = np.asarray(cx)
    train_sm_cortex(bL, C_drive, hub, cx, n_epochs=8, drive_scale=40.0, window=200, settle=8)
    learned = read_codes(bL, C_drive, hub, cx, **rp)
    lp, lg, ch = _stat("LEARNED", learned, S_true, labels)

    # UNTRAINED random-projection control (same dense regime, different seed wiring)
    bR, hubR, cxR = build_sm_cortex_bridge(seed=43, **bp)
    rand = read_codes(bR, C_drive, np.asarray(hubR), np.asarray(cxR), **rp)
    rp_p, rg, _ = _stat("RANDOM-PROJ", rand, S_true, labels)

    # permuted-similarity anti-cheat on the learned codes
    rng = np.random.RandomState(42 * 2718281 + 1)
    perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = _pearson_vs_Strue(_cos_sim(learned), S_perm)
    print(f"  [anti-cheat] learned permuted-similarity Pearson={perm_p:+.3f} (~0)", flush=True)

    structure_ok = lp >= 0.30
    permuted_ok = abs(perm_p) <= 0.15
    generalizes = lg > ch + 0.10
    learning_load_bearing = (lp >= rp_p + 0.10) or (lg >= rg + 0.15)  # higher structure OR better held-out gen
    print(f"\n  gates: structure(≥0.30)={structure_ok}  permuted_collapses={permuted_ok}  "
          f"generalizes={generalizes}  learning_load_bearing={learning_load_bearing}", flush=True)
    print(f"  (learned {lp:+.3f}/gen {lg:.3f}  vs  random {rp_p:+.3f}/gen {rg:.3f})", flush=True)
    if structure_ok and permuted_ok and generalizes and learning_load_bearing:
        print("\n  GO -- the LEARNED spiking cortex recovers the structure in the dense regime, generalizes, "
              "beats the random projection, permuted-clean. The Phase-B spike-readout boundary is CRACKED by "
              "dense firing; the build proceeds (formalize the HARD GATE + Task 4 GPU real corpus).", flush=True)
    elif structure_ok and permuted_ok:
        print(f"\n  PARTIAL/HONEST -- the structure transmits densely (learned {lp:+.3f}, permuted clean) but "
              f"the LEARNING is not clearly load-bearing over the dense random projection ({rp_p:+.3f}). The "
              f"FIX (dense readout) is real; whether STDP adds 'learned cortex' value needs the real corpus / "
              f"more training. Honest reframe, not a 3rd premature positive.", flush=True)
    else:
        print(f"\n  NOT CLEARED -- structure {lp:+.3f}, permuted {perm_p:+.3f}; the dense crack does not survive "
              f"the full battery. Re-assess.", flush=True)


if __name__ == "__main__":
    main()
