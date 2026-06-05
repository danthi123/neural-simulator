"""(de-risk B, the crux GATE) Can a SUBSTRATE-HELD bound fact -- its (ON,OFF) vector imprinted in connection
WEIGHTS of a small dedicated population, retrieved in SPIKES -- be unbound at NUMPY PARITY?

This GATES the whole (B) build (substrate-held memory). (A) cleared the READOUT shortcut (the numpy argmax
cleanup -> a spiking NEF cleanup, `2026-06-05-composer-cleanup-NEF-GO.md`). (B) is the deeper MEMORY shortcut:
the composer's bound fact is a numpy (ON,OFF) vector held in a Python list (`CoreSimComposer.kb`); the memory is
NOT in the substrate. The (B) research VERDICT (`2026-06-05-substrate-held-memory-literature-synthesis.md`):
do NOT use an engram-tag for the GRADED bound vector (engrams BINARIZE); the recommended store is a
Crawford-Gingerich-Eliasmith-style per-fact associative memory -- the fact's bound vector imprinted in STATIC
connection weights of a small dedicated population, retrieved by firing it (validated to 117,659 facts at D=512).

THE MECHANISM (Crawford-style weight-store, simplest faithful version; NO sim/ edits; reuse-by-import):
  On a SEPARATE small SimulationBridge of Izhikevich neurons:
    - a per-fact "memory" TRIGGER population of `n_trig` neurons;
    - two D-neuron readout banks: readout_ON[0,D), readout_OFF[D,2D);
    - the fact's bound vector lives in the OUTPUT weights: every trigger neuron i projects
        trigger_i -> readout_ON[k]  with weight  bon[k] * w_gain     (the ON channel of B)
        trigger_i -> readout_OFF[k] with weight  boff[k] * w_gain    (the OFF channel of B)
      so the bound (ON,OFF) vector is HELD IN THE WEIGHTS, not in numpy.
  RETRIEVE (a genuine spiking read, NOT a numpy passthrough):
    - drive the trigger population at a constant current -> the trigger neurons fire steadily -> each readout
      neuron fires at a rate ~ f(sum_i w_i) = f(n_trig * bon[k] * w_gain), the f-I nonlinearity of the readout
      Izhikevich neuron;
    - accumulate readout firing over the readout window -> (bon', boff') = the reconstructed bound vector,
      read out of SPIKES (the trigger drives nothing in numpy; the bound values are recovered only because the
      readout banks fire at rates set by the imprinted synaptic weights).

THE TEST (the de-risk):
  1. Build a CoreSimComposer (proj_dim 800; the harder/noisier regime), store a few SVO facts -> each fact's
     numpy bound vector B=(bon,boff) via comp.bind_fact(fact).
  2. For each fact: imprint B into the substrate weight-store; RETRIEVE B'=(bon',boff') in spikes; then for each
     role in {agent, action, patient} compare comp._unbind_onoff(B', role) -> cleanup vs
     comp._unbind_onoff(B, role) -> cleanup. The CLEANUP is HELD CONSTANT (the validated numpy argmax oracle,
     OR --spiking-cleanup), because the POINT is the STORE. Recovery = fraction of roles whose substrate-store
     -retrieved unbind returns the SAME filler as the numpy-store unbind.
  3. Multi-seed (42/43/44). GATE: substrate-store recall == numpy-store recall at parity (~1.000 per the
     literature's high-fidelity regime). Report per-seed recovery + the reconstruction cosine (bon',boff').(bon,boff).

  python -u -m research.findings.raw._b_substrate_weight_store_probe --out research/findings/raw/_b_substrate_store.json

Smell-test for a real GO: the read is from SPIKES (trigger drives only the readout banks via synapses; no numpy
shortcut copies B into B'); multi-seed; the cleanup is held constant so the STORE is what's tested.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host
from research.runners.core_sim_composition import CoreSimComposer

RESET_STEPS = 20
ROLES = ("agent", "action", "patient")

# weight-store operating point (the Crawford-style per-fact memory). n_trig trigger neurons summing onto each
# readout neuron + a per-dimension readout averaging pool (n_per readout neurons per bound dimension) smooth the
# f-I noise (representational error ~ 1/sqrt(N), Singh-Eliasmith). w_gain * trig_drive set so the readout f-I is
# in its graded (sub-saturation) band -> the reconstructed rates track the imprinted weights.
STORE_OP = dict(n_trig=40, n_per=4, w_gain=250.0, trig_drive=600.0, run_steps=300)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_store_bridge(seed, bound_onoff, n_trig, n_per, w_gain, ou_std=20.0):
    """Imprint ONE fact's bound (ON,OFF) vector into a per-fact substrate weight-store and return the bridge +
    index. Layout: trigger[0,n_trig) + readout_ON[base, base+D*n_per) + readout_OFF[..., +D*n_per).
      The bound vector is HELD IN THE WEIGHTS: trigger_i -> readout_ON[k*n_per+j] weight = bon[k]*w_gain;
      trigger_i -> readout_OFF[k*n_per+j] weight = boff[k]*w_gain. n_per readout neurons share dimension k's
      weight (population-average the f-I noise on the read). FIXED (no plasticity)."""
    bon, boff = bound_onoff
    D = bon.shape[0]
    base_on = n_trig
    base_off = base_on + D * n_per
    N = base_off + D * n_per
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = float(ou_std)

    trig = np.arange(0, n_trig)
    ron = np.arange(base_on, base_on + D * n_per)        # readout_ON[k*n_per + j]
    roff = np.arange(base_off, base_off + D * n_per)     # readout_OFF[k*n_per + j]
    pre, post, w = [], [], []
    for t in trig:
        for k in range(D):
            won = float(bon[k] * w_gain); woff = float(boff[k] * w_gain)
            for j in range(n_per):
                if won != 0.0:
                    pre.append(int(t)); post.append(int(ron[k * n_per + j])); w.append(won)
                if woff != 0.0:
                    pre.append(int(t)); post.append(int(roff[k * n_per + j])); w.append(woff)
    plan = {"store": {"pre_indices": pre, "post_indices": post,
                      "initial_weights": np.array(w, dtype=np.float32), "plastic": False,
                      "conn_type": "E_TO_E", "count": len(pre)}}
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.inject_explicit_wiring(plan)
    xp, _ = get_backend()
    idx = {"trig": xp.asarray(trig, dtype=xp.int64),
           "ron": xp.asarray(ron, dtype=xp.int64), "roff": xp.asarray(roff, dtype=xp.int64)}
    return bridge, idx, D


def retrieve_bound(bridge, idx, D, n_per, trig_drive, run_steps):
    """RETRIEVE the bound vector in SPIKES: drive the trigger population at a constant current -> the readout
    banks fire at rates set by the imprinted weights -> (bon', boff') = per-dimension averaged readout firing
    rate. A genuine spiking read: the trigger neurons drive ONLY the readout banks (via the fixed store
    synapses); the bound values appear only because those synaptic weights ARE the bound vector."""
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[idx["trig"]] = float(trig_drive)
    bridge.cp_external_input_current[:] = cur
    acc_on = xp.zeros(D * n_per, dtype=xp.float64)
    acc_off = xp.zeros(D * n_per, dtype=xp.float64)
    for _ in range(run_steps):
        bridge._run_one_simulation_step()
        acc_on += bridge.cp_firing_states[idx["ron"]].astype(xp.float64)
        acc_off += bridge.cp_firing_states[idx["roff"]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    # per-dimension average over the n_per readout neurons (smooth the f-I noise)
    bon_p = (to_host(acc_on) / run_steps).reshape(D, n_per).mean(axis=1)
    boff_p = (to_host(acc_off) / run_steps).reshape(D, n_per).mean(axis=1)
    return bon_p, boff_p


def eval_seed(seed, proj_dim, n_facts, op, spiking_cleanup):
    """Build a composer, store n_facts SVO facts, and for each fact compare substrate-store unbind vs numpy-store
    unbind across all 3 roles. The cleanup is the SAME for both arms (numpy argmax oracle, or the spiking NEF
    cleanup when spiking_cleanup) -- the STORE is what's tested."""
    comp = CoreSimComposer(seed=seed, proj_dim=proj_dim, enable_spiking_cleanup=spiking_cleanup)
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    rng = np.random.default_rng(seed)
    facts = []
    for _ in range(n_facts):
        a, ac, p = rng.choice(usable, size=3, replace=False)
        facts.append({"agent": str(a), "action": str(ac), "patient": str(p)})

    per_fact = []
    n_roles_total = 0; n_match = 0
    cos_list = []
    for fact in facts:
        B = comp.bind_fact(fact)                                   # numpy bound vector (the held value)
        bon, boff = B
        # SUBSTRATE STORE: imprint B in weights, RETRIEVE B' in spikes
        bridge, idx, D = build_store_bridge(seed, B, op["n_trig"], op["n_per"], op["w_gain"])
        bon_p, boff_p = retrieve_bound(bridge, idx, D, op["n_per"], op["trig_drive"], op["run_steps"])
        Bp = (bon_p, boff_p)
        cos = _cos(np.concatenate([bon_p, boff_p]), np.concatenate([bon, boff]))
        cos_list.append(cos)
        # per-role: substrate-store unbind+cleanup  vs  numpy-store unbind+cleanup (cleanup held constant)
        role_rec = {}
        for role in ROLES:
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_sub, e_off_sub = comp._unbind_onoff(Bp, role)
            filler_sub = comp._cleanup(e_on_sub - e_off_sub, comp.words)
            match = (filler_sub == filler_np)
            role_rec[role] = {"numpy": filler_np, "substrate": filler_sub, "match": bool(match),
                              "truth": fact[role]}
            n_roles_total += 1; n_match += int(match)
        per_fact.append({"fact": fact, "recon_cos": round(cos, 4), "roles": role_rec})

    recovery = n_match / max(n_roles_total, 1)
    return {"recovery": recovery, "mean_recon_cos": float(np.mean(cos_list)),
            "min_recon_cos": float(np.min(cos_list)), "n_facts": len(facts),
            "n_roles": n_roles_total, "per_fact": per_fact}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-facts", type=int, default=4)
    ap.add_argument("--n-trig", type=int, default=STORE_OP["n_trig"])
    ap.add_argument("--n-per", type=int, default=STORE_OP["n_per"])
    ap.add_argument("--w-gain", type=float, default=STORE_OP["w_gain"])
    ap.add_argument("--trig-drive", type=float, default=STORE_OP["trig_drive"])
    ap.add_argument("--run-steps", type=int, default=STORE_OP["run_steps"])
    ap.add_argument("--spiking-cleanup", action="store_true",
                    help="hold the SPIKING NEF cleanup constant instead of the numpy argmax oracle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    op = dict(n_trig=args.n_trig, n_per=args.n_per, w_gain=args.w_gain,
              trig_drive=args.trig_drive, run_steps=args.run_steps)

    per_seed = {}
    for s in args.seeds:
        r = eval_seed(s, args.proj_dim, args.n_facts, op, args.spiking_cleanup)
        per_seed[s] = r
        print(f"[B-store] seed {s}: recovery={r['recovery']:.3f} ({sum(rr['roles'][role]['match'] for rr in r['per_fact'] for role in ROLES)}"
              f"/{r['n_roles']}) mean_recon_cos={r['mean_recon_cos']:.4f} min_recon_cos={r['min_recon_cos']:.4f}",
              flush=True)

    recoveries = {s: per_seed[s]["recovery"] for s in args.seeds}
    min_rec = min(recoveries.values()); mean_rec = sum(recoveries.values()) / len(recoveries)
    mean_cos = float(np.mean([per_seed[s]["mean_recon_cos"] for s in args.seeds]))
    verdict = "GO" if min_rec >= 0.999 else "NEGATIVE"
    print(f"\n[B-store ROBUST] min_recovery={min_rec:.3f} mean_recovery={mean_rec:.3f} mean_recon_cos={mean_cos:.4f}")
    print(f"[VERDICT] substrate weight-store unbind == numpy-store unbind (parity) -> {verdict} "
          f"(GATE: per-seed recovery == 1.000)")
    if args.out:
        json.dump({"op": op, "per_seed": per_seed, "recoveries": recoveries, "min_recovery": min_rec,
                   "mean_recovery": mean_rec, "mean_recon_cos": mean_cos, "verdict": verdict},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
