"""Phase-B C1b DE-RISK (host-side instrument, NO sim/ edit): does PER-POSTSYNAPTIC-NEURON centering of the
cortex drive recover the category structure that a single (rank-1) inhibitory pool could not?

Context: Task-3 attempt 2 (centering via a feedforward cm pool + synaptic scaling) returned a WALL — the
cortex g_e (analog drive) cosine is −0.063 ≈ the spike-code −0.074 (the common mode survives into the analog
drive; the spiking threshold is NOT the destroyer). The subagent localized WHY: L1's load-bearing op is
`X − X.mean(0)` (a PER-INPUT-DIMENSION subtraction), but a single inhibitory pool can only deliver rank-1
(scalar/uniform) inhibition, while the common mode's contribution to the cortex is PER-NEURON-VARYING
(random hub→cortex connectivity). So the framework cannot express it; C1b would be a small guarded sim/ edit
(per-postsynaptic-neuron subtractive drive centering = the direct spiking `x − col_mean`, brain-plausible as
intrinsic subtractive adaptation / slow feedback inhibition).

BEFORE any sim/ edit, this HOST-SIDE PROBE tests whether the OP actually works on the bridge's analog drive:
read the cortex g_e codes, subtract EACH cortex neuron's mean drive across concepts (the per-neuron centering
the sim/ edit would do neurally), and check whether the cosine jumps from ~0 to positive. The decisive arm is
the UNTRAINED (random-W) g_e centered = the bridge analogue of L1's `random_proj_centered` (+0.169): if
centering recovers the random projection's structure on the bridge, the per-neuron centering OP is sound and
a sim/ edit is warranted; if even that stays ~0, the per-neuron-centering hypothesis is wrong and the wall is
deeper. (Host centering here is a TEST INSTRUMENT, not the deliverable — a host X.mean subtraction would be a
cheat; if it works, C1b implements it NEURALLY.)

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_c1b_derisk_perneuron_centering
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
    build_sm_cortex_bridge, encode_drive, train_sm_cortex, _set_hub_drive, _step_with_time,
)
from sim.backend import to_host  # noqa: E402


def read_ge_codes(bridge, C_drive, hub_idx, cortex_idx, *, drive_scale, window, settle, cofire_pA=0.0):
    """Read per-concept cortex g_e (excitatory conductance = the analog hub-driven drive) over the window,
    plasticity frozen. Mirrors read_codes but accumulates cp_conductance_g_e instead of spike counts."""
    cortex_idx = np.asarray(cortex_idx)
    Nc = int(np.asarray(C_drive).shape[0])
    codes = np.zeros((Nc, cortex_idx.size), dtype=np.float64)
    gate_names = list(getattr(bridge, "_plasticity_gate_values", {}).keys()) or ["hub_to_cortex"]
    for g in gate_names:
        bridge.set_plasticity_gate(g, 0.0)
    try:
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale,
                           cortex_idx=cortex_idx, cofire_pA=cofire_pA)
            acc = np.zeros(cortex_idx.size, dtype=np.float64)
            for t in range(int(settle) + int(window)):
                _step_with_time(bridge)
                if t >= settle:
                    acc += np.asarray(to_host(bridge.cp_conductance_g_e))[cortex_idx].astype(np.float64)
            codes[i] = acc / max(1, window)
            bridge.cp_external_input_current[:] = 0.0
    finally:
        for g in gate_names:
            bridge.set_plasticity_gate(g, 1.0)
    return codes


def _score(name, codes, S_true, labels):
    p_unc = _pearson_vs_Strue(_cos_sim(codes), S_true)
    cen = codes - codes.mean(0, keepdims=True)          # PER-NEURON centering (the C1b op)
    p_cen = _pearson_vs_Strue(_cos_sim(cen), S_true)
    g_unc, ch = heldout_generalization(codes, labels)
    g_cen, _ = heldout_generalization(cen, labels)
    silent = float(np.mean(codes.sum(1) == 0))
    print(f"  [{name:26s}] uncentered={p_unc:+.3f} (gen {g_unc:.3f})  PER-NEURON-CENTERED={p_cen:+.3f} "
          f"(gen {g_cen:.3f})  silent={silent:.2f}  eff-rank={effective_rank(codes):.1f}", flush=True)
    return p_unc, p_cen


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    # synthetic 64-concept, strong common mode (host PPMI+SVD ceiling ~0.96)
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    rate_cos = _pearson_vs_Strue(_cos_sim(C_drive), S_true)
    rate_cos_cen = _pearson_vs_Strue(_cos_sim(C_drive - C_drive.mean(0, keepdims=True)), S_true)
    print(f"[C1b per-neuron-centering de-risk] {C.shape[0]} concepts x {n_hub} hubs", flush=True)
    print(f"  reference: rate log-input cos uncentered={rate_cos:+.3f}  CENTERED={rate_cos_cen:+.3f} "
          f"(the L1 op on the input)", flush=True)
    print("  --- bridge cortex g_e codes: does PER-NEURON centering recover the structure? ---", flush=True)

    bp = dict(n_hub=n_hub, n_cortex=128, seed=42, density=0.5, weight_mean=80.0,
              homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03, stdp_w_max=200.0)
    rp = dict(drive_scale=12.0, window=40, settle=8)

    # (A0) THE HUB spike-rate codes themselves -- does the INPUT spiking layer preserve the structure (so a
    #      per-HUB-input centering, i.e. a subtractive per-presynaptic-source gain, would work), or does the
    #      input spiking nonlinearity already destroy it (the deeper wall)? Read hub spike counts over a window.
    b, hub, cx = build_sm_cortex_bridge(**bp)
    hub = np.asarray(hub)
    Nc = C_drive.shape[0]
    hub_codes = np.zeros((Nc, hub.size), dtype=np.float64)
    for g in (list(getattr(b, "_plasticity_gate_values", {}).keys()) or ["hub_to_cortex"]):
        b.set_plasticity_gate(g, 0.0)
    for i in range(Nc):
        _set_hub_drive(b, hub, C_drive[i], rp["drive_scale"])
        acc = np.zeros(hub.size)
        for t in range(rp["settle"] + rp["window"]):
            _step_with_time(b)
            if t >= rp["settle"]:
                acc += np.asarray(to_host(b.cp_firing_states))[hub].astype(np.float64)
        hub_codes[i] = acc
        b.cp_external_input_current[:] = 0.0
    _score("HUB spike-rate codes", hub_codes, S_true, labels)

    # (A) UNTRAINED random-W g_e -- the bridge analogue of L1 random_proj_centered (+0.169). DECISIVE arm.
    ge0 = read_ge_codes(b, C_drive, hub, cx, cofire_pA=0.0, **rp)
    a_unc, a_cen = _score("UNTRAINED random-W g_e", ge0, S_true, labels)

    # (B) TRAINED g_e -- the STDP learned on the UNCENTERED drive, so post-hoc centering may not fully fix it
    #     (the W may already encode the common mode); informative either way.
    b2, hub2, cx2 = build_sm_cortex_bridge(**bp)
    train_sm_cortex(b2, C_drive, hub2, cx2, n_epochs=8, drive_scale=12.0, window=40, settle=8, cofire_pA=4.0)
    ge1 = read_ge_codes(b2, C_drive, hub2, cx2, cofire_pA=0.0, **rp)
    b_unc, b_cen = _score("TRAINED g_e", ge1, S_true, labels)

    print("\n  VERDICT:", flush=True)
    if a_cen >= 0.15 and a_cen >= a_unc + 0.10:
        print(f"  C1b VALIDATED -- per-neuron centering recovers the structure on the bridge g_e "
              f"(untrained {a_unc:+.3f}->{a_cen:+.3f}). The OP is sound; a guarded sim/ edit (per-postsynaptic-"
              f"neuron subtractive drive centering = intrinsic adaptation) is warranted. Training on the "
              f"centered drive should lift it toward the L1 learned ceiling. Trained-post-hoc {b_unc:+.3f}->"
              f"{b_cen:+.3f} (post-hoc centering of a W learned on UNcentered drive -- the sim/ edit centers "
              f"DURING training, which this under-states).", flush=True)
    elif a_cen >= a_unc + 0.10:
        print(f"  PARTIAL -- per-neuron centering helps ({a_unc:+.3f}->{a_cen:+.3f}) but stays below +0.15; "
              f"the OP is directionally right but the bridge g_e loses too much. Weigh C1b vs the deeper wall.",
              flush=True)
    else:
        print(f"  WALL CONFIRMED DEEPER -- per-neuron centering does NOT recover the bridge g_e structure "
              f"(untrained {a_unc:+.3f}->{a_cen:+.3f}); the per-neuron-centering hypothesis is insufficient. "
              f"The rate->spike wall is below the centering op -> the honest NEGATIVE (the spiking learned "
              f"cortex needs the deeper dendritic substrate / the months-scale piece).", flush=True)


if __name__ == "__main__":
    main()
