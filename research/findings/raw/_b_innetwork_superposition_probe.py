"""(de-risk B, the LAST numpy in the composer's compute path) Can the bound-fact SUPERPOSITION
(`bon += o; boff += f` summed across roles) + ON/OFF OPPONENCY (`onoff(bon - boff)`) be done IN-NETWORK
(spiking) so the resulting bound vector unbinds at NUMPY PARITY?

Context: (A) cleared the READOUT shortcut (numpy argmax cleanup -> spiking NEF cleanup,
`2026-06-05-composer-cleanup-NEF-GO.md`); (B) the STORE shortcut (numpy `kb` list -> Crawford-style spiking
weight-store, `2026-06-05-B-substrate-store-fidelity-GO.md`). Those two GOs left exactly TWO linear numpy ops
in `CoreSimComposer.bind_fact` (disclosed by the audit as "linear inter-phase ops"):
  for each role:  o, f = self._op(role, fon, foff)   # SPIKING coincidence bind (A/B/C/D banks)
                  bon += o; boff += f                 # <-- numpy SUPERPOSITION (rate-sum across roles)
  return onoff(bon - boff)                            # <-- numpy OPPONENCY (ON/OFF rectified difference)
The host read of `(o, f)` per role is the numpy boundary. To remove it the per-role binds' coincidence banks
must drive a SHARED ACCUMULATOR on the bridge that SUMS across roles, with ON/OFF lateral inhibition for the
opponency.

THE MECHANISM BUILT HERE (NO sim/ edits; the BIND machinery REUSED BY IMPORT from `build_bind_bridge`):
  ONE standalone SimulationBridge of 10*D Izhikevich neurons:
    - the 8*D-neuron coincidence circuit (role_ON/OFF + fill_ON/OFF -> A/B/C/D), wired by REUSING
      `core_sim_composition.build_bind_bridge(shared_bridge=...)` (no re-implementation of the +-1 Hadamard);
    - PLUS two accumulator regions acc_on[D] @ [8D,9D), acc_off[D] @ [9D,10D).
  Accumulator wiring (fixed identity weights, w_acc):
    A[k] -> acc_on[k],  B[k] -> acc_on[k]   (acc_on accumulates A+B = the ON pattern)
    C[k] -> acc_off[k], D[k] -> acc_off[k]  (acc_off accumulates C+D = the OFF pattern)
  Opponency (lateral inhibition): acc_on[k] -| acc_off[k] and acc_off[k] -| acc_on[k] (mutual shunt). acc_on
  and acc_off carry an INHIBITORY trait so their outgoing synapses route through g_i (the routing keys on the
  PRESYNAPTIC inhibitory trait, per the A divnorm finding `_spiking_cleanup_divnorm_probe.py` / bridge.py
  5046-5070 -- NOT on the conn_type string). Each ON/OFF pair settles to the rectified opponent difference
  (the `onoff(bon-boff)` analogue computed in spikes).

  bind_fact_in_network(fact): reset the bridge + accumulator ONCE; for EACH role drive role/fill currents +
  run the bind window WITHOUT resetting the accumulator -> the coincidence banks fire -> drive the accumulator
  -> it ACCUMULATES the superposition across roles; after all roles read (acc_on, acc_off) rates -> (bon',boff').

SATURATION HANDLING: 2-4 superposed role(x)filler patterns can saturate the accumulator Izhikevich f-I. The
accumulator drive gain `w_acc` (banks->acc weight), the readout window, and the opponency strength `w_opp` are
tunable and swept so the accumulated sum stays in the responsive (sub-saturation) band and the relative
magnitudes are preserved (a tuned bank, per the research note that a gated NEF integrator is the heavier
alternative if a tuned bank does not suffice).

THE TEST (the de-risk GATE):
  1. Build a CoreSimComposer (proj_dim 800; the harder/noisier regime); store a few SVO + one-attribute facts ->
     for each, the NUMPY bound vector B = comp.bind_fact(fact).
  2. For each fact: compute the IN-NETWORK bound vector B' = bind_fact_in_network(fact) (this probe); then for
     each role compare comp._unbind_onoff(B', role) -> cleanup vs comp._unbind_onoff(B, role) -> cleanup. The
     CLEANUP is HELD CONSTANT (the deterministic numpy argmax oracle) to isolate the SUPERPOSITION. Recovery =
     fraction of roles whose in-network-built unbind returns the SAME filler as the numpy-built unbind.
  3. Multi-seed (42/43/44). Also report the cosine (bon',boff').(bon,boff). GATE: in-network == numpy at parity
     (~1.000). Smell-test: the accumulator genuinely SUMS in spikes (a 2-role fact's acc ~ sum of the two
     single-role binds, not one dominating); it is not a numpy passthrough.

  python -u -m research.findings.raw._b_innetwork_superposition_probe --out research/findings/raw/_b_innetwork_superposition.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.core_sim_composition import (
    CoreSimComposer, build_bind_bridge, onoff, _scale_to_current,
    ROLE_DRIVE, FILL_DRIVE, RESET_STEPS, DEFAULT_RUN_STEPS, DEFAULT_BIAS)
from research.runners.unified_brain_bridge import merge_population_into_shared_bridge

ROLES = ("agent", "action", "patient")
INH_TRAIT = 1

# in-network superposition+opponency accumulator operating point (the tuned-bank choice). The accumulator
# Izhikevich f-I saturates if driven too hard; w_acc / run_steps / w_opp keep the accumulated sum in the
# responsive band and realize the rectified opponent difference in spikes. The best tuned-bank point found by
# the sweep (mutual ON/OFF lateral inhibition); see the NEGATIVE finding 2026-06-05-B-innetwork-superposition-
# NEGATIVE.md for why this does NOT reach numpy parity (the signed bon-boff difference is destroyed by the
# spiking opponency — a small-signal-in-correlated-channels problem).
ACC_OP = dict(w_acc=500.0, w_opp=200.0, einh=-80.0, run_steps=DEFAULT_RUN_STEPS, ou_std=20.0)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_bind_accumulator_bridge(seed, D, op):
    """Build the standalone 10*D-neuron bind+accumulator bridge.

    REUSE: the 8*D coincidence circuit is wired by `core_sim_composition.build_bind_bridge(shared_bridge=...)`
    (the +-1 Hadamard is NOT re-implemented). This probe only ADDS the accumulator regions + their wiring:
      acc_on[k] @ 8D+k, acc_off[k] @ 9D+k;
      A[k],B[k] -> acc_on[k] (w_acc);  C[k],D[k] -> acc_off[k] (w_acc);
      acc_on[k] -| acc_off[k], acc_off[k] -| acc_on[k] (w_opp, mutual shunt).
    acc_on/acc_off carry the inhibitory trait so their outgoing synapses route through g_i (the opponency is a
    true conductance shunt, per the A divnorm finding). Returns (bridge, idx) with idx including acc_on/acc_off.
    """
    N = 10 * D
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 2
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = float(op["ou_std"])
    cfg.enable_inhibitory_neurons = True
    cfg.inhibitory_trait_indices = [INH_TRAIT]
    cfg.syn_reversal_potential_i = float(op["einh"])

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    xp, _ = get_backend()

    acc_on = np.arange(8 * D, 9 * D)
    acc_off = np.arange(9 * D, 10 * D)
    # mark accumulator banks INHIBITORY BEFORE the first step (mask is cached on first use)
    tr = bridge.cp_traits
    tr[:] = 0
    tr[xp.asarray(acc_on, dtype=tr.dtype)] = INH_TRAIT
    tr[xp.asarray(acc_off, dtype=tr.dtype)] = INH_TRAIT
    bridge.cp_traits = tr
    bridge._cached_inhibitory_mask = None

    # REUSE the bind wiring (build_bind_bridge accumulates the "bind" population onto this shared bridge and
    # returns the offset bank index arrays). On a Hebbian-OFF bridge the gate-zero is a harmless no-op.
    bridge, idx = build_bind_bridge(seed, D, shared_bridge=bridge, index_offset=0)

    # ADD the accumulator wiring: banks -> acc (identity, w_acc) + opponency (mutual lateral inhibition, w_opp).
    A = np.arange(4 * D, 5 * D); B = np.arange(5 * D, 6 * D)
    C = np.arange(6 * D, 7 * D); Dd = np.arange(7 * D, 8 * D)
    w_acc = float(op["w_acc"]); w_opp = float(op["w_opp"])
    pre_e, post_e, w_e = [], [], []
    for k in range(D):
        for src in (A, B):
            pre_e.append(int(src[k])); post_e.append(int(acc_on[k])); w_e.append(w_acc)
        for src in (C, Dd):
            pre_e.append(int(src[k])); post_e.append(int(acc_off[k])); w_e.append(w_acc)
    pre_i, post_i, w_i = [], [], []
    for k in range(D):
        pre_i.append(int(acc_on[k])); post_i.append(int(acc_off[k])); w_i.append(w_opp)
        pre_i.append(int(acc_off[k])); post_i.append(int(acc_on[k])); w_i.append(w_opp)
    plan = {
        "acc_in": {"pre_indices": pre_e, "post_indices": post_e,
                   "initial_weights": np.array(w_e, dtype=np.float32), "plastic": False,
                   "conn_type": "E_TO_E", "count": len(pre_e)},
        "acc_opp": {"pre_indices": pre_i, "post_indices": post_i,
                    "initial_weights": np.array(w_i, dtype=np.float32), "plastic": False,
                    "conn_type": "I_TO_E", "count": len(pre_i)},
    }
    merge_population_into_shared_bridge(bridge, plan)

    idx = dict(idx)
    idx["acc_on"] = xp.asarray(acc_on, dtype=xp.int64)
    idx["acc_off"] = xp.asarray(acc_off, dtype=xp.int64)
    return bridge, idx


def _drive_one_role(bridge, idx, role_vec, fill_on_cur, fill_off_cur, D, run_steps, coinc_bias,
                    accumulate_acc=True):
    """Drive ONE role's bind window WITHOUT resetting the accumulator. The coincidence banks fire -> the
    accumulator integrates (acc_on gets A+B, acc_off gets C+D); acc_on -| acc_off realizes the opponency.
    Returns the per-step-summed (acc_on, acc_off) firing for THIS window (used by the spike-sum smell-test);
    the persistent accumulation across roles lives in the bridge's conductance/voltage state."""
    xp, _ = get_backend()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[idx["role_on"]] = xp.asarray((role_vec > 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["role_off"]] = xp.asarray((role_vec < 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = coinc_bias
    bridge.cp_external_input_current[:] = cur
    acc_on = xp.zeros(D, dtype=xp.float64)
    acc_off = xp.zeros(D, dtype=xp.float64)
    for _ in range(run_steps):
        bridge._run_one_simulation_step()
        if accumulate_acc:
            acc_on += bridge.cp_firing_states[idx["acc_on"]].astype(xp.float64)
            acc_off += bridge.cp_firing_states[idx["acc_off"]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc_on) / run_steps, to_host(acc_off) / run_steps


def bind_fact_in_network(bridge, idx, comp, fact, op, return_per_role=False):
    """Compute the IN-NETWORK bound vector for `fact`: reset the bridge + accumulator ONCE; for EACH present
    role drive role/fill currents + run the bind window WITHOUT resetting the accumulator -> the accumulator
    SUMS the superposition across roles in spikes; the acc_on -| acc_off opponency settles to the rectified
    difference. Read (acc_on, acc_off) firing rates accumulated across ALL role windows -> (bon', boff').

    The accumulator firing is summed across every role window (the persistent conductance/voltage state ALSO
    carries the sum, but reading the across-window firing sum is the spike read of the superposition). Returns
    (bon', boff'); with return_per_role also the per-role single-window acc reads (for the spike-sum smell-test).
    """
    xp, _ = get_backend()
    # reset ONCE (drain conductance + settle to rest with zero drive)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    D = comp.D
    run_steps = int(op["run_steps"]); coinc_bias = comp.coinc_bias
    total_on = np.zeros(D); total_off = np.zeros(D)
    per_role = {}
    for role in comp.ROLES:
        if role not in fact:
            continue
        c_on, c_off = onoff(comp._filler_signed(fact[role]))
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        a_on, a_off = _drive_one_role(bridge, idx, comp.roles[role], fon, foff, D, run_steps, coinc_bias)
        total_on += a_on; total_off += a_off
        per_role[role] = (a_on, a_off)
    bridge.cp_external_input_current[:] = 0.0
    if return_per_role:
        return (total_on, total_off), per_role
    return (total_on, total_off)


def numpy_raw_superposition(comp, fact):
    """The numpy PRE-opponency superposition: bon = sum_role rates(A+B), boff = sum_role rates(C+D), via the SAME
    spiking coincidence bind the composer uses, but WITHOUT the final `onoff(bon-boff)`. Used as the diagnostic
    reference for the in-network accumulator's superposition fidelity + the signed-difference fidelity (the
    quantity the unbind actually consumes)."""
    D = comp.D
    bon = np.zeros(D); boff = np.zeros(D)
    for role in comp.ROLES:
        if role not in fact:
            continue
        c_on, c_off = onoff(comp._filler_signed(fact[role]))
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        o, f = comp._op(comp.roles[role], fon, foff)
        bon += o; boff += f
    return bon, boff


def eval_seed(seed, proj_dim, n_flat, n_attr, op):
    """Build a composer + the bind+accumulator bridge; store n_flat SVO + n_attr one-attribute facts; for each
    fact compare the IN-NETWORK bound vector's unbind vs the NUMPY bound vector's unbind across all roles. The
    cleanup is the SAME (deterministic numpy argmax oracle) for both arms -- the SUPERPOSITION is what's tested.

    Also captures the two load-bearing DIAGNOSTICS for the finding:
      - `signed_cos`: cos(acc_on - acc_off, bon - boff) -- fidelity of the SIGNED difference the unbind consumes;
      - `upper_bound_recovery`: parity when NUMPY opponency `onoff(acc_on - acc_off)` is applied to the in-network
        superposition (isolates whether the SUPERPOSITION read or the spiking OPPONENCY is the blocker)."""
    comp = CoreSimComposer(seed=seed, proj_dim=proj_dim)
    bridge, idx = build_bind_accumulator_bridge(seed, comp.D, op)
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    rng = np.random.default_rng(seed)

    def pick(k):
        return [str(x) for x in rng.choice(usable, size=k, replace=False)]

    facts = []
    for _ in range(n_flat):
        a, ac, p = pick(3)
        facts.append(({"agent": a, "action": ac, "patient": p}, ROLES))
    for _ in range(n_attr):
        a, ac, adj, noun = pick(4)
        # one-attribute fact via comp.store's tuple form -> roles agent/action/patient/attribute
        f = {"agent": a, "action": ac, "patient": noun, "attribute": adj}
        facts.append((f, ("agent", "action", "patient", "attribute")))

    per_fact = []
    n_total = 0; n_match = 0; cos_list = []
    signed_cos_list = []
    n_ub_total = 0; n_ub_match = 0   # upper bound: in-network superposition + NUMPY opponency
    for fact, roles in facts:
        B = comp.bind_fact(fact)               # numpy superposition/opponency bound vector
        bon, boff = B
        Bp = bind_fact_in_network(bridge, idx, comp, fact, op)  # IN-NETWORK bound vector (spiking opponency)
        bon_p, boff_p = Bp
        cos = _cos(np.concatenate([bon_p, boff_p]), np.concatenate([bon, boff]))
        cos_list.append(cos)
        # diagnostics: signed-difference fidelity + upper-bound (numpy opponency on the in-network superposition)
        raw_bon, raw_boff = numpy_raw_superposition(comp, fact)
        signed_cos_list.append(_cos(bon_p - boff_p, raw_bon - raw_boff))
        Bp_ub = onoff(bon_p - boff_p)          # apply NUMPY opponency to the in-network superposition
        for role in roles:
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_ub, e_off_ub = comp._unbind_onoff(Bp_ub, role)
            filler_ub = comp._cleanup(e_on_ub - e_off_ub, comp.words)
            n_ub_total += 1; n_ub_match += int(filler_ub == filler_np)
        role_rec = {}
        for role in roles:
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_in, e_off_in = comp._unbind_onoff(Bp, role)
            filler_in = comp._cleanup(e_on_in - e_off_in, comp.words)
            match = (filler_in == filler_np)
            role_rec[role] = {"numpy": filler_np, "in_network": filler_in, "match": bool(match),
                              "truth": fact[role]}
            n_total += 1; n_match += int(match)
        per_fact.append({"fact": {k: (v if isinstance(v, str) else str(v)) for k, v in fact.items()},
                         "recon_cos": round(cos, 4), "roles": role_rec})

    recovery = n_match / max(n_total, 1)
    return {"recovery": recovery, "mean_recon_cos": float(np.mean(cos_list)),
            "min_recon_cos": float(np.min(cos_list)), "n_facts": len(facts),
            "n_roles": n_total, "per_fact": per_fact,
            "mean_signed_cos": float(np.mean(signed_cos_list)),
            "upper_bound_recovery": n_ub_match / max(n_ub_total, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=3)
    ap.add_argument("--n-attr", type=int, default=1)
    ap.add_argument("--w-acc", type=float, default=ACC_OP["w_acc"])
    ap.add_argument("--w-opp", type=float, default=ACC_OP["w_opp"])
    ap.add_argument("--einh", type=float, default=ACC_OP["einh"])
    ap.add_argument("--run-steps", type=int, default=ACC_OP["run_steps"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    op = dict(w_acc=args.w_acc, w_opp=args.w_opp, einh=args.einh, run_steps=args.run_steps,
              ou_std=ACC_OP["ou_std"])

    per_seed = {}
    for s in args.seeds:
        r = eval_seed(s, args.proj_dim, args.n_flat, args.n_attr, op)
        per_seed[s] = r
        print(f"[B-innet] seed {s}: recovery={r['recovery']:.3f} ({r['n_roles']} roles) "
              f"mean_recon_cos={r['mean_recon_cos']:.4f} signed_cos={r['mean_signed_cos']:.4f} "
              f"upper_bound(numpy-opponency)={r['upper_bound_recovery']:.3f}", flush=True)

    recoveries = {s: per_seed[s]["recovery"] for s in args.seeds}
    min_rec = min(recoveries.values()); mean_rec = sum(recoveries.values()) / len(recoveries)
    mean_cos = float(np.mean([per_seed[s]["mean_recon_cos"] for s in args.seeds]))
    mean_signed = float(np.mean([per_seed[s]["mean_signed_cos"] for s in args.seeds]))
    mean_ub = float(np.mean([per_seed[s]["upper_bound_recovery"] for s in args.seeds]))
    verdict = "GO" if min_rec >= 0.999 else "NEGATIVE"
    print(f"\n[B-innet ROBUST] min_recovery={min_rec:.3f} mean_recovery={mean_rec:.3f} mean_recon_cos={mean_cos:.4f}"
          f" mean_signed_cos={mean_signed:.4f} mean_upper_bound={mean_ub:.3f}")
    print(f"[VERDICT] in-network superposition+opponency unbind == numpy unbind (parity) -> {verdict} "
          f"(GATE: per-seed recovery == 1.000)")
    if args.out:
        json.dump({"op": op, "per_seed": per_seed, "recoveries": recoveries, "min_recovery": min_rec,
                   "mean_recovery": mean_rec, "mean_recon_cos": mean_cos, "mean_signed_cos": mean_signed,
                   "mean_upper_bound_recovery": mean_ub, "verdict": verdict},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
