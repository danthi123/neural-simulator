"""De-risk (B) regression test: a SUBSTRATE-HELD bound fact (its (ON,OFF) vector imprinted in connection WEIGHTS
of a small dedicated population, retrieved in SPIKES) unbinds at NUMPY PARITY.

Pins the (B) crux GATE validated in `research/findings/raw/_b_substrate_weight_store_probe.py` (finding
`2026-06-05-B-substrate-store-fidelity-GO.md`): the Crawford-Gingerich-Eliasmith-style per-fact weight-store --
a trigger population whose OUTPUT weights onto two D-neuron readout banks ARE the bound vector -- reconstructs the
bound vector from SPIKES well enough (recon cosine ~0.97) that unbinding the substrate-retrieved B' recovers the
SAME fillers as unbinding the numpy-held B, across all roles. The cleanup is held CONSTANT (the deterministic
numpy argmax oracle) so the STORE is what's tested.

These build a real SimulationBridge and run spiking bind/retrieve, so they are heavier than a pure-numpy unit
test; they run on the available backend (GPU when present). On the numpy backend the composer's spiking bind
produces a DEGENERATE (all-zero) bound vector at this operating point, so the test skips when the bound vector is
not graded -- the de-risk is a GPU result (per the project's GPU-for-real-runs mandate), and a degenerate bound
vector would make 'parity' a meaningless tautology."""
import numpy as np
import pytest

from research.runners.core_sim_composition import CoreSimComposer

# small-but-non-degenerate config (validated equal to the production proj_dim=800 result on GPU):
PROJ_DIM = 400
N_TRIG = 40
N_PER = 2
W_GAIN = 250.0
TRIG_DRIVE = 600.0
RUN_STEPS = 250
ROLES = ("agent", "action", "patient")


def _setup(seed=42):
    """Build a composer + import the probe's store/retrieve, skipping gracefully if the concept cache is absent
    or (numpy backend) the spiking bind is degenerate."""
    try:
        comp = CoreSimComposer(seed=seed, proj_dim=PROJ_DIM)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present (run activity_level_integration to build it)")
    from research.findings.raw._b_substrate_weight_store_probe import (
        build_store_bridge, retrieve_bound, _cos)
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    # guard: the composer's spiking bind must produce a GRADED bound vector for the parity test to be meaningful
    probe = comp.bind_fact({"agent": usable[0], "action": usable[1], "patient": usable[2]})
    if float(probe[0].max()) <= 0.0 and float(probe[1].max()) <= 0.0:
        pytest.skip("spiking bind degenerate on this backend (all-zero bound vector) -- de-risk is a GPU result")
    return comp, usable, build_store_bridge, retrieve_bound, _cos


def test_substrate_weight_store_unbind_parity():
    """Store each fact's bound vector in the substrate weight-store, retrieve it in spikes, and confirm every
    role's substrate-store unbind+cleanup == the numpy-store unbind+cleanup (the cleanup held constant)."""
    seed = 42
    comp, usable, build_store_bridge, retrieve_bound, _cos = _setup(seed)
    rng = np.random.default_rng(seed)
    n_total = 0; n_match = 0; cosines = []
    for _ in range(3):
        a, ac, p = rng.choice(usable, size=3, replace=False)
        fact = {"agent": str(a), "action": str(ac), "patient": str(p)}
        B = comp.bind_fact(fact)                                   # numpy-held bound vector
        bon, boff = B
        bridge, idx, D = build_store_bridge(seed, B, N_TRIG, N_PER, W_GAIN)
        bon_p, boff_p = retrieve_bound(bridge, idx, D, N_PER, TRIG_DRIVE, RUN_STEPS)   # spiking read
        Bp = (bon_p, boff_p)
        cosines.append(_cos(np.concatenate([bon_p, boff_p]), np.concatenate([bon, boff])))
        for role in ROLES:
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_sub, e_off_sub = comp._unbind_onoff(Bp, role)
            filler_sub = comp._cleanup(e_on_sub - e_off_sub, comp.words)
            n_total += 1; n_match += int(filler_sub == filler_np)
    # GATE: substrate-store recall == numpy-store recall (parity) across all roles
    assert n_match == n_total, f"substrate-store unbind parity {n_match}/{n_total} (recon cos {np.mean(cosines):.3f})"
    # the reconstruction is a faithful (scaled) spiking copy, not perfect (f-I nonlinearity) -- ~0.97
    assert np.mean(cosines) > 0.90, f"reconstruction cosine {np.mean(cosines):.3f} too low"


def test_substrate_store_read_is_from_spikes_not_numpy():
    """Smell-test: the read is GENUINELY from spikes. Zeroing the trigger drive silences the readout banks ->
    the reconstructed bound vector collapses to the OU-noise floor (no numpy passthrough copies B into B')."""
    seed = 42
    comp, usable, build_store_bridge, retrieve_bound, _cos = _setup(seed)
    B = comp.bind_fact({"agent": usable[0], "action": usable[1], "patient": usable[2]})
    bridge, idx, D = build_store_bridge(seed, B, N_TRIG, N_PER, W_GAIN)
    on_driven, _ = retrieve_bound(bridge, idx, D, N_PER, TRIG_DRIVE, RUN_STEPS)   # trigger ON
    on_silent, _ = retrieve_bound(bridge, idx, D, N_PER, 0.0, RUN_STEPS)          # trigger OFF
    # with the trigger silent the readout is just OU noise -> the reconstruction collapses (>=10x lower)
    assert on_silent.sum() < 0.10 * on_driven.sum(), (
        f"reconstruction did not collapse without trigger spikes (driven={on_driven.sum():.3f}, "
        f"silent={on_silent.sum():.3f}) -- read may be a numpy passthrough")
    assert on_driven.sum() > 1.0, "trigger-driven readout did not fire (operating point off)"
