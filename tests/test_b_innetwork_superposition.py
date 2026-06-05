"""De-risk (B) regression test: the bound-fact SUPERPOSITION + ON/OFF OPPONENCY done IN-NETWORK (spiking) --
the last two numpy ops in `CoreSimComposer.bind_fact`'s compute path. Pins the de-risk OUTCOME, which is
NEGATIVE (`research/findings/2026-06-05-B-innetwork-superposition-NEGATIVE.md`).

The de-risk built a SHARED ACCUMULATOR on the bridge: the per-role coincidence banks drive acc_on <- A+B,
acc_off <- C+D (this SUMS the superposition across roles in spikes -- genuine, pinned by the smell-test below),
with acc_on -| acc_off mutual lateral inhibition (inhibitory-trait routing) for the `onoff(bon-boff)` opponency.
The GATE was: in-network unbind == numpy unbind (parity ~1.000). It FAILED: the SIGNED difference `bon-boff` that
the unbind consumes is destroyed (signed cos ~0.41), because the conductance-based shunting opponency cannot
perform the precise small-signal subtraction of the two strongly-correlated channels (`cos(o,f)~0.89`). Even
PERFECT numpy opponency applied to the in-network superposition only recovers ~0.64 -- so the superposition read
(cos-0.97 per channel) differenced into a cos-0.41 signed vector is the root blocker, not just the spiking
opponency. The documented next idea is a gated NEF integrator (a recurrent linear integrator representing the
signed value, so the subtraction happens before the lossy f-I read).

These tests PIN that documented boundary: (1) the accumulator genuinely SUMS across roles in spikes (the
superposition piece works), and (2) the in-network unbind does NOT reach parity at this tuned-bank operating
point (the opponency/subtraction-fidelity wall). If a future mechanism (NEF integrator) LIFTS parity to
~1.000, test (2) will fail loudly -- the correct signal to update this de-risk to GO and flip the assertion.

These build a real SimulationBridge and run spiking bind/accumulate, so they are heavier than a pure-numpy unit
test; they run on the available backend (GPU when present). On the numpy backend the composer's spiking bind
produces a DEGENERATE (all-zero) bound vector at this operating point, so the tests skip when the bound vector is
not graded -- the de-risk is a GPU result (per the project's GPU-for-real-runs mandate)."""
import numpy as np
import pytest

from research.runners.core_sim_composition import CoreSimComposer

# small-but-non-degenerate config (matches the production proj_dim=800 NEGATIVE on GPU):
PROJ_DIM = 400
ROLES = ("agent", "action", "patient")


def _setup(seed=42):
    """Build a composer + the bind+accumulator bridge, skipping gracefully if the concept cache is absent or
    (numpy backend) the spiking bind is degenerate."""
    try:
        comp = CoreSimComposer(seed=seed, proj_dim=PROJ_DIM)
    except FileNotFoundError:
        pytest.skip("denoise64 concept-code cache not present (run activity_level_integration to build it)")
    from research.findings.raw._b_innetwork_superposition_probe import (
        build_bind_accumulator_bridge, bind_fact_in_network, ACC_OP, _cos)
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    probe = comp.bind_fact({"agent": usable[0], "action": usable[1], "patient": usable[2]})
    if float(probe[0].max()) <= 0.0 and float(probe[1].max()) <= 0.0:
        pytest.skip("spiking bind degenerate on this backend (all-zero bound vector) -- de-risk is a GPU result")
    op = dict(ACC_OP)
    bridge, idx = build_bind_accumulator_bridge(seed, comp.D, op)
    return comp, usable, bridge, idx, op, bind_fact_in_network, _cos


def test_innetwork_superposition_unbind_does_not_reach_parity():
    """Pins the NEGATIVE de-risk OUTCOME: the in-network (spiking) superposition+opponency does NOT unbind at
    numpy parity at this tuned-bank operating point. The mechanism is genuinely ENGAGED (the in-network bound
    vector is non-degenerate and the unbind recovers SOME fillers), but the strict parity GATE fails because the
    signed `bon-boff` difference is destroyed by the shunting opponency. A future NEF-integrator fix that lifts
    parity to ~1.000 will fail this test -- the signal to flip the de-risk to GO."""
    seed = 42
    comp, usable, bridge, idx, op, bind_fact_in_network, _cos = _setup(seed)
    rng = np.random.default_rng(seed)
    n_total = 0; n_match = 0; cosines = []
    bound_nonzero = False
    for _ in range(3):
        a, ac, p = rng.choice(usable, size=3, replace=False)
        fact = {"agent": str(a), "action": str(ac), "patient": str(p)}
        B = comp.bind_fact(fact)                                # numpy superposition/opponency
        bon, boff = B
        Bp = bind_fact_in_network(bridge, idx, comp, fact, op)  # IN-NETWORK superposition/opponency
        bon_p, boff_p = Bp
        if float(bon_p.max()) > 0.0 or float(boff_p.max()) > 0.0:
            bound_nonzero = True
        cosines.append(_cos(np.concatenate([bon_p, boff_p]), np.concatenate([bon, boff])))
        for role in ROLES:
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_in, e_off_in = comp._unbind_onoff(Bp, role)
            filler_in = comp._cleanup(e_on_in - e_off_in, comp.words)
            n_total += 1; n_match += int(filler_in == filler_np)
    # the mechanism is genuinely engaged: the in-network bound vector is non-degenerate (spiking, not all-zero)
    assert bound_nonzero, "in-network bound vector degenerate (all-zero) -- accumulator not firing"
    # the de-risk OUTCOME (NEGATIVE): the strict parity GATE is NOT met -- some roles disagree with numpy.
    # If a future mechanism reaches parity (n_match == n_total) this assertion fires -> flip the de-risk to GO.
    assert n_match < n_total, (
        f"in-network superposition reached numpy PARITY {n_match}/{n_total} (recon cos {np.mean(cosines):.3f}) "
        f"-- the de-risk NEGATIVE no longer holds; update 2026-06-05-B-innetwork-superposition-NEGATIVE.md to GO")


def test_innetwork_accumulator_sums_in_spikes_not_passthrough():
    """Smell-test: the accumulator genuinely SUMS in spikes. A 2-role fact's accumulated acc read should be
    close to the SUM of the two single-role acc reads (superposition), and substantially larger than either
    single role's read alone -- i.e. neither role dominates and it is not a numpy passthrough of one role."""
    seed = 42
    comp, usable, bridge, idx, op, bind_fact_in_network, _cos = _setup(seed)
    a, ac, p = usable[0], usable[1], usable[2]

    # single-role accumulations (each reset + one role)
    f_agent = {"agent": a}
    (s_on, s_off), pr_a = bind_fact_in_network(bridge, idx, comp, f_agent, op, return_per_role=True)
    s_agent = s_on.sum() + s_off.sum()
    f_action = {"action": ac}
    (t_on, t_off), pr_b = bind_fact_in_network(bridge, idx, comp, f_action, op, return_per_role=True)
    s_action = t_on.sum() + t_off.sum()

    # the 2-role fact: accumulator should integrate BOTH role windows
    f_two = {"agent": a, "action": ac}
    (two_on, two_off), per_role = bind_fact_in_network(bridge, idx, comp, f_two, op, return_per_role=True)
    s_two = two_on.sum() + two_off.sum()

    # both single roles must actually fire the accumulator (operating point on)
    assert s_agent > 1.0 and s_action > 1.0, (
        f"single-role accumulator did not fire (agent={s_agent:.3f}, action={s_action:.3f})")
    # the 2-role accumulation is larger than EITHER single role (genuine superposition, not one dominating)
    assert s_two > 1.10 * max(s_agent, s_action), (
        f"2-role accumulation {s_two:.3f} not larger than max single role {max(s_agent, s_action):.3f} "
        f"-- accumulator is not summing across roles")
    # and it tracks the SUM of the two (within a tolerance band; spiking dynamics are not perfectly additive)
    expected = s_agent + s_action
    assert 0.5 * expected <= s_two <= 1.6 * expected, (
        f"2-role accumulation {s_two:.3f} not ~ sum of single roles {expected:.3f} "
        f"(agent={s_agent:.3f}, action={s_action:.3f}) -- superposition is off")
