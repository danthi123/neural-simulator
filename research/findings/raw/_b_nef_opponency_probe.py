"""(de-risk B, piece (B) final) Can the composer's ON/OFF OPPONENCY -- the SIGNED difference `s = bon - boff`
the unbind consumes -- be represented ROBUSTLY in SPIKES via an NEF random-projection signed-value population,
so the bound vector unbinds at NUMPY PARITY?

WHY this, after the simple-accumulator NEGATIVE (`2026-06-05-B-innetwork-superposition-NEGATIVE.md`):
  The simple in-network accumulator READS each channel faithfully (acc_on~bon, acc_off~boff at cos 0.97), but the
  unbind consumes the SIGNED DIFFERENCE `bon - boff`, which is SMALL relative to a large COMMON MODE
  (cos(bon,boff) ~ 0.94, ||bon-boff||/||bon|| ~ 0.33). A 3% per-channel spiking read-error swamps the small true
  difference -> signed cos collapses to 0.41. PLUS conductance shunting `g_i*(E_inh-V)` is DIVISIVE, not a clean
  linear subtraction. Even PERFECT numpy opponency on the in-network superposition recovered only 0.64.

THE MECHANISM (NEF signed-value representation; the two insights that target the diagnosis):
  Represent `s = bon - boff` NOT via per-component subtraction (noise-amplifying) but via a population of M
  neurons each computing a DOT PRODUCT `e_i . s = e_i . bon - e_i . boff` (a sum over D components -> per-component
  read-noise AVERAGES out; the common mode CANCELS in the difference). Decode `s' = sum_i a_i(s) d_i` with offline
  least-squares decoders d_i (NEF principle, Eliasmith-Anderson 2003; decode error ~ 1/sqrt(M)).

  1. RANDOM-PROJECTION (NEF) ENCODING of s, in SPIKES on the core Izhikevich bridge. The signed dot product
     `e_i . bon - e_i . boff` is realized by SIGN-routed conductance:
       - bon/boff are turned into spiking SOURCE banks (D neurons each, plus an INHIBITORY-trait copy of each).
       - source bank `k` drives encoder `i` with weight `w_nef * gain_i * |e_i[k]|`, routed by SIGN:
           e_i[k] > 0:  src_bon[k]  --(g_e)-->  enc[i]   (e_i . bon, positive part)   [excitatory]
                        src_boff_i[k]--(g_i)-->  enc[i]   (-e_i . boff, e_i[k]>0 -> -) [inhibitory copy]
           e_i[k] < 0:  src_bon_i[k]--(g_i)-->  enc[i]   (e_i . bon, negative part)    [inhibitory copy]
                        src_boff[k] --(g_e)-->  enc[i]   (-e_i . boff, e_i[k]<0 -> +)  [excitatory]
       so the NET drive into enc[i] ~ gain_i * (e_i . bon - e_i . boff) + bias_i (the NEF intercept, a constant
       bias current). The SUBTRACTION happens in the represented dot product BEFORE the lossy f-I read, and the
       per-component noise averages over the D-term sum.
  2. LINEAR-REGIME SUBTRACTION. The NEF subtraction is NEGATIVE connection WEIGHTS = (here) inhibitory conductance
     operated in the SMALL-inhibition LINEAR band (g_i small, V near rest, E_inh far -> g_i*(E_inh-V) ~ const*g_i,
     ~linear in the boff/bon rate), NOT the bridge's divisive shunting at large g_i. The tuning is verified by the
     `--check-linearity` arm: the represented `e_i . s` must track the true `e_i . (bon - boff)` (encoder-rate vs
     ideal-projection cos > 0.95) across a sweep of test vectors.
  Then `onoff(s')` -> the bound vector (the rectify is cheap + local; done in numpy here, it is a trivial per-dim
  max with no common-mode issue).

DECODER PRECOMPUTE (the NEF caveat: weights precomputed offline, injected as fixed pathways):
  encoders e_i ~ random Gaussian (unit), gains/biases ~ random (fixed by seed); decoders d_i by LEAST SQUARES on
  the encoder population's spiking tuning A (M x Ntrain) over a SAMPLE of signed vectors s (real composer binds +
  jitter), solving min ||D A - S||^2 + reg. d_i are then FIXED (a numpy linear readout of the encoder firing --
  the standard NEF decode; the ENCODING is the spiking computation that removes the common mode).

THE TEST (the de-risk GATE):
  1. CoreSimComposer (proj_dim 800); store a few SVO + one-attr facts -> numpy bound vector B=(bon,boff) =
     comp.bind_fact(fact), AND the underlying raw bon,boff BEFORE opponency (replicate the bind loop).
  2. NEF signed read: feed bon,boff -> s' (the spiking-represented bon-boff) -> B' = onoff(s'). Compare
     comp._unbind_onoff(B', role) -> cleanup (numpy argmax oracle, held constant) vs comp._unbind_onoff(B, role)
     for each role. Recovery = fraction of roles matching numpy. Multi-seed 42/43/44.
  3. Report recovery + the signed cosine s'.(bon-boff) (the simple accumulator got 0.41 -- beat it toward 1.0) +
     the M-sweep (decode error ~ 1/sqrt(M)).
  GATE: recovery == 1.000 multi-seed. Smell-test: the NEF read is genuinely spiking (M-dependence; zeroing the
  source input collapses s'); the signed cosine genuinely improves over 0.41 with M.

  python -u -m research.findings.raw._b_nef_opponency_probe --out research/findings/raw/_b_nef_opponency.json

NO sim/ edits; the bind machinery + composer are REUSED BY IMPORT.
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
    CoreSimComposer, onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS)

ROLES = ("agent", "action", "patient")
INH_TRAIT = 1

# NEF signed-value operating point. The encoder f-I must stay in its GRADED band (not saturated) so the population
# code is informative; src_drive sets the source-bank rates, w_nef the encoder drive gain, einh keeps the
# inhibitory (negative-encoder) routing in the LINEAR small-inhibition regime (g_i*(E_inh-V) ~ const*g_i).
NEF_OP = dict(M=8000, src_drive=500.0, w_nef=40.0, einh=-80.0, gain_lo=1.0, gain_hi=3.0, bias_lo=20.0, bias_hi=80.0,
              read_steps=120, settle_frac=0.35, ou_std=20.0, n_train=80, reg=0.02, train_jitter=0.15)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def numpy_raw_superposition(comp, fact):
    """The numpy PRE-opponency superposition bon = sum_role rates(A+B), boff = sum_role rates(C+D), via the SAME
    spiking coincidence bind the composer uses but WITHOUT the final `onoff(bon-boff)`. This is the (bon,boff) the
    NEF signed read consumes; `s = bon - boff` is what the unbind ultimately needs."""
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


def build_nef_opponency_bridge(seed, D, M, op):
    """Build the standalone NEF signed-value bridge (4*D source neurons + M encoder neurons).

    Layout:  src_bon[0,D)  src_bon_i[D,2D)  src_boff[2D,3D)  src_boff_i[3D,4D)  enc[4D, 4D+M).
      src_bon / src_boff are EXCITATORY (route to g_e); src_bon_i / src_boff_i are INHIBITORY-trait copies (route
      to g_i). All four are driven by the SAME external current (~ bon[k] / boff[k]) so the excitatory and
      inhibitory copies fire at matched rates -> the sign-routed encoder weights realize `e_i . bon - e_i . boff`.

    Encoder weights (NEF random projection, sign-routed conductance):
      e_i[k] > 0:  src_bon[k] -> enc[i] (E, w = w_nef*gain_i* e_i[k]); src_boff_i[k] -> enc[i] (I, same magnitude)
      e_i[k] < 0:  src_bon_i[k] -> enc[i] (I, w = w_nef*gain_i*|e_i[k]|); src_boff[k] -> enc[i] (E, same magnitude)
    Plus a per-encoder constant bias current `bias_i` (the NEF intercept; injected at read time on cp_external).

    Returns (bridge, idx, E, gain, bias) where E[M,D] are the encoders, gain[M], bias[M] are the NEF gains/biases.
    """
    xp, _ = get_backend()
    N = 4 * D + M
    src_bon = np.arange(0, D); src_bon_i = np.arange(D, 2 * D)
    src_boff = np.arange(2 * D, 3 * D); src_boff_i = np.arange(3 * D, 4 * D)
    enc = np.arange(4 * D, 4 * D + M)

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

    # mark the inhibitory-trait source copies (so their OUTGOING synapses route through g_i) BEFORE first step.
    tr = bridge.cp_traits
    tr[:] = 0
    tr[xp.asarray(src_bon_i, dtype=tr.dtype)] = INH_TRAIT
    tr[xp.asarray(src_boff_i, dtype=tr.dtype)] = INH_TRAIT
    bridge.cp_traits = tr
    bridge._cached_inhibitory_mask = None

    # NEF encoders / gains / biases (FIXED by seed -- precomputed offline, injected as fixed pathways).
    rng = np.random.default_rng(seed * 100003 + 7)
    E = rng.standard_normal((M, D)); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    gain = rng.uniform(op["gain_lo"], op["gain_hi"], M)
    bias = rng.uniform(op["bias_lo"], op["bias_hi"], M)

    # build the sign-routed encoder wiring. Excitatory plan (E_TO_E) and inhibitory plan (I_TO_E) separately.
    w_nef = float(op["w_nef"])
    pre_e, post_e, w_e = [], [], []
    pre_i, post_i, w_i = [], [], []
    Ew = (E * (w_nef * gain[:, None]))    # M x D scaled weights
    for i in range(M):
        row = Ew[i]
        pos = np.where(E[i] > 0)[0]; neg = np.where(E[i] < 0)[0]
        ci = int(enc[i])
        for k in pos:
            # e_i[k] > 0: +bon (excitatory), -boff (inhibitory copy)
            pre_e.append(int(src_bon[k])); post_e.append(ci); w_e.append(float(row[k]))
            pre_i.append(int(src_boff_i[k])); post_i.append(ci); w_i.append(float(row[k]))
        for k in neg:
            # e_i[k] < 0: +bon via inhibitory copy (magnitude), +boff excitatory (magnitude)
            mag = float(-row[k])
            pre_i.append(int(src_bon_i[k])); post_i.append(ci); w_i.append(mag)
            pre_e.append(int(src_boff[k])); post_e.append(ci); w_e.append(mag)

    plan = {
        "nef_e": {"pre_indices": pre_e, "post_indices": post_e,
                  "initial_weights": np.array(w_e, dtype=np.float32), "plastic": False,
                  "conn_type": "E_TO_E", "count": len(pre_e)},
        "nef_i": {"pre_indices": pre_i, "post_indices": post_i,
                  "initial_weights": np.array(w_i, dtype=np.float32), "plastic": False,
                  "conn_type": "I_TO_E", "count": len(pre_i)},
    }
    bridge.inject_explicit_wiring(plan)

    idx = {"src_bon": xp.asarray(src_bon, dtype=xp.int64),
           "src_bon_i": xp.asarray(src_bon_i, dtype=xp.int64),
           "src_boff": xp.asarray(src_boff, dtype=xp.int64),
           "src_boff_i": xp.asarray(src_boff_i, dtype=xp.int64),
           "enc": xp.asarray(enc, dtype=xp.int64), "bias": xp.asarray(bias, dtype=xp.float64)}
    return bridge, idx, E, gain, bias


def encode_spiking(bridge, idx, bon, boff, M, op, zero_input=False):
    """Drive the 4 source banks with currents ~ bon[k]/boff[k] (matched excitatory + inhibitory copies), add the
    per-encoder NEF bias current, run the read window, return the encoder population firing-rate vector a(s) in M.

    `zero_input=True`: drive the source banks with ZERO current (the bias still applied) -> the smell-test that the
    encoder rates collapse without the bon/boff drive (the read is genuinely a spiking response to the input)."""
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    if not zero_input:
        # source-bank drive: scale bon/boff by a common factor so the source rates encode the magnitudes. Use a
        # SHARED scale across bon and boff (NOT per-channel) so the common mode is preserved into the encoder
        # (the NEF dot product is what removes it -- not the source normalization).
        sd = float(op["src_drive"])
        m = max(float(bon.max()), float(boff.max()), 1e-9)
        bon_cur = (bon / m * sd).astype(np.float32)
        boff_cur = (boff / m * sd).astype(np.float32)
        cur[idx["src_bon"]] = xp.asarray(bon_cur)
        cur[idx["src_bon_i"]] = xp.asarray(bon_cur)
        cur[idx["src_boff"]] = xp.asarray(boff_cur)
        cur[idx["src_boff_i"]] = xp.asarray(boff_cur)
    # per-encoder constant NEF bias current (the intercept) is ALWAYS applied (even on the zero-input smell-test).
    cur[idx["enc"]] = xp.asarray(np.asarray(to_host(idx["bias"]), dtype=np.float32))
    bridge.cp_external_input_current[:] = cur
    acc = xp.zeros(M, dtype=xp.float64)
    read_steps = int(op["read_steps"]); settle = int(read_steps * op["settle_frac"])
    for t in range(read_steps):
        bridge._run_one_simulation_step()
        if t >= settle:
            acc += bridge.cp_firing_states[idx["enc"]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    return to_host(acc) / max(read_steps - settle, 1)


def gen_signed_sample(comp, rng, n, jitter):
    """A training sample of signed s = bon - boff vectors, from REAL composer binds (the actual distribution) +
    small jitter (so the NEF decoder generalizes around the operating manifold). Returns list of (bon, boff, s)."""
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    out = []
    for _ in range(n):
        a, ac, p = rng.choice(usable, size=3, replace=False)
        bon, boff = numpy_raw_superposition(comp, {"agent": str(a), "action": str(ac), "patient": str(p)})
        if jitter > 0:
            bon = np.maximum(bon + jitter * bon.std() * rng.standard_normal(comp.D), 0.0)
            boff = np.maximum(boff + jitter * boff.std() * rng.standard_normal(comp.D), 0.0)
        out.append((bon, boff, bon - boff))
    return out


def fit_decoders(bridge, idx, comp, M, op, seed):
    """Precompute the NEF least-squares decoders OFFLINE: run a sample of signed vectors through the SPIKING
    encoder -> rate matrix A (M x Ntrain); solve D = argmin ||D A - S||^2 + reg*I (S = D x Ntrain targets). The
    decoders are then FIXED (a numpy linear readout of the encoder firing). Returns Ddec (D x M)."""
    rng = np.random.default_rng(seed * 7 + 11)
    sample = gen_signed_sample(comp, rng, int(op["n_train"]), float(op["train_jitter"]))
    A = np.zeros((M, len(sample))); S = np.zeros((comp.D, len(sample)))
    for j, (bon, boff, s) in enumerate(sample):
        A[:, j] = encode_spiking(bridge, idx, bon, boff, M, op)
        S[:, j] = s
    reg = float(op["reg"]) * (A.shape[1])
    G = A @ A.T + reg * np.eye(M)
    Ddec = np.linalg.solve(G, A @ S.T).T          # D x M
    return Ddec


def nef_signed_read(bridge, idx, Ddec, bon, boff, M, op):
    """The NEF signed read: encode (bon,boff) in spikes -> a(s); decode s' = Ddec @ a(s)."""
    a = encode_spiking(bridge, idx, bon, boff, M, op)
    return Ddec @ a


def eval_seed(seed, proj_dim, n_flat, n_attr, op, check_linearity=False):
    """Build a composer + the NEF opponency bridge; fit decoders offline; for each stored fact compare the
    NEF-built bound vector's unbind vs the NUMPY bound vector's unbind across all roles. Cleanup held constant
    (numpy argmax oracle) -- the OPPONENCY (the signed read) is what is tested. Also captures the signed cosine
    s'.(bon-boff) (the load-bearing diagnostic) and (optionally) the encoder linearity check."""
    comp = CoreSimComposer(seed=seed, proj_dim=proj_dim)
    M = int(op["M"])
    bridge, idx, E, gain, bias = build_nef_opponency_bridge(seed, comp.D, M, op)
    Ddec = fit_decoders(bridge, idx, comp, M, op, seed)

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
        facts.append(({"agent": a, "action": ac, "patient": noun, "attribute": adj},
                      ("agent", "action", "patient", "attribute")))

    # smell-test: zeroing the source input collapses the encoder rates (the read is genuinely spiking)
    a_live = encode_spiking(bridge, idx, *numpy_raw_superposition(comp, facts[0][0])[:2], M, op)
    a_zero = encode_spiking(bridge, idx, np.zeros(comp.D), np.zeros(comp.D), M, op, zero_input=True)
    smell_collapse = float(a_zero.mean()) / (float(a_live.mean()) + 1e-12)

    lin_cos = None
    if check_linearity:
        # the encoder should represent `e_i . s` ~ linearly: a sample's decoded s' must track the IDEAL projection
        # decode (D @ (E @ s) ...). Simpler faithful check: across a held-out sample, the spiking-read s' tracks
        # the TRUE s (cos), and tracks the IDEAL (rate-model) NEF s' computed from the SAME E/gain/bias. Report the
        # spiking-vs-true cos as the linearity proxy (the represented signed value tracks the true signed value).
        rng2 = np.random.default_rng(seed * 13 + 5)
        sample = gen_signed_sample(comp, rng2, 12, 0.0)
        lc = []
        for bon, boff, s in sample:
            s_hat = nef_signed_read(bridge, idx, Ddec, bon, boff, M, op)
            lc.append(_cos(s_hat, s))
        lin_cos = float(np.mean(lc))

    per_fact = []
    n_total = 0; n_match = 0; signed_cos_list = []; recon_cos_list = []
    for fact, roles in facts:
        B = comp.bind_fact(fact)                       # numpy superposition/opponency bound vector
        bon, boff = numpy_raw_superposition(comp, fact)  # raw pre-opponency (s = bon - boff)
        s_hat = nef_signed_read(bridge, idx, Ddec, bon, boff, M, op)  # spiking NEF signed read
        signed_cos_list.append(_cos(s_hat, bon - boff))
        Bp = onoff(s_hat)                              # the bound vector from the NEF signed read
        recon_cos_list.append(_cos(np.concatenate(Bp), np.concatenate(B)))
        role_rec = {}
        for role in roles:
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_in, e_off_in = comp._unbind_onoff(Bp, role)
            filler_in = comp._cleanup(e_on_in - e_off_in, comp.words)
            match = (filler_in == filler_np)
            role_rec[role] = {"numpy": filler_np, "nef": filler_in, "match": bool(match), "truth": fact[role]}
            n_total += 1; n_match += int(match)
        per_fact.append({"fact": {k: (v if isinstance(v, str) else str(v)) for k, v in fact.items()},
                         "signed_cos": round(signed_cos_list[-1], 4), "recon_cos": round(recon_cos_list[-1], 4),
                         "roles": role_rec})

    return {"recovery": n_match / max(n_total, 1), "n_roles": n_total, "n_facts": len(facts),
            "mean_signed_cos": float(np.mean(signed_cos_list)), "mean_recon_cos": float(np.mean(recon_cos_list)),
            "smell_zero_collapse": round(smell_collapse, 4), "lin_cos": lin_cos, "per_fact": per_fact}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=3)
    ap.add_argument("--n-attr", type=int, default=1)
    ap.add_argument("--M", type=int, default=NEF_OP["M"])
    ap.add_argument("--m-sweep", type=int, nargs="*", default=None,
                    help="sweep M (signed cosine vs M, the key curve) on seed 42 only")
    ap.add_argument("--src-drive", type=float, default=NEF_OP["src_drive"])
    ap.add_argument("--w-nef", type=float, default=NEF_OP["w_nef"])
    ap.add_argument("--einh", type=float, default=NEF_OP["einh"])
    ap.add_argument("--read-steps", type=int, default=NEF_OP["read_steps"])
    ap.add_argument("--check-linearity", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base_op = dict(NEF_OP)
    base_op.update(M=args.M, src_drive=args.src_drive, w_nef=args.w_nef, einh=args.einh, read_steps=args.read_steps)

    m_sweep_res = None
    if args.m_sweep:
        m_sweep_res = {}
        for M in args.m_sweep:
            op = dict(base_op); op["M"] = M
            r = eval_seed(42, args.proj_dim, args.n_flat, args.n_attr, op, check_linearity=False)
            m_sweep_res[M] = {"recovery": r["recovery"], "mean_signed_cos": r["mean_signed_cos"],
                              "mean_recon_cos": r["mean_recon_cos"]}
            print(f"[B-NEF M-sweep] M={M}: recovery={r['recovery']:.3f} signed_cos={r['mean_signed_cos']:.4f} "
                  f"recon_cos={r['mean_recon_cos']:.4f}", flush=True)

    per_seed = {}
    for s in args.seeds:
        r = eval_seed(s, args.proj_dim, args.n_flat, args.n_attr, base_op, check_linearity=args.check_linearity)
        per_seed[s] = r
        print(f"[B-NEF] seed {s}: recovery={r['recovery']:.3f} ({r['n_roles']} roles) "
              f"signed_cos={r['mean_signed_cos']:.4f} recon_cos={r['mean_recon_cos']:.4f} "
              f"smell_zero_collapse={r['smell_zero_collapse']:.3f}"
              + (f" lin_cos={r['lin_cos']:.4f}" if r['lin_cos'] is not None else ""), flush=True)

    recoveries = {s: per_seed[s]["recovery"] for s in args.seeds}
    min_rec = min(recoveries.values()); mean_rec = sum(recoveries.values()) / len(recoveries)
    mean_signed = float(np.mean([per_seed[s]["mean_signed_cos"] for s in args.seeds]))
    mean_recon = float(np.mean([per_seed[s]["mean_recon_cos"] for s in args.seeds]))
    verdict = "GO" if min_rec >= 0.999 else "NEGATIVE"
    print(f"\n[B-NEF ROBUST] min_recovery={min_rec:.3f} mean_recovery={mean_rec:.3f} "
          f"mean_signed_cos={mean_signed:.4f} mean_recon_cos={mean_recon:.4f}")
    print(f"[VERDICT] NEF signed-value opponency unbind == numpy unbind (parity) -> {verdict} "
          f"(GATE: per-seed recovery == 1.000; cf. simple-accumulator signed cos 0.41)")
    if args.out:
        json.dump({"op": base_op, "per_seed": per_seed, "recoveries": recoveries, "min_recovery": min_rec,
                   "mean_recovery": mean_rec, "mean_signed_cos": mean_signed, "mean_recon_cos": mean_recon,
                   "m_sweep": m_sweep_res, "verdict": verdict}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
