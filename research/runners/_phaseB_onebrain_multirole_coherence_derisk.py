"""ROADMAP PHASE 2 (the real "one brain"), STEP 3c -- PHASE-COHERENCE capacity of the persistent bridge as a fact
gains ROLES. Steps 1-3b validated a 2-role fact (agent, action) chained register->register on ONE persistent bridge
(bind -> bundle -> unbind -> cleanup, ~5 ops, 1.000). The explicitly-named TOP RISK is "phase coherence as the chain
lengthens". A realistic conversational fact has MORE roles -- "the big warm dog goes" = agent(dog) + action(go) +
attribute(big) + attribute2(warm). Each extra role adds one bound vector to the bundled composite, so every unbind
sees the other R-1 binds as superposition CROSSTALK (the Fourier-Holographic-Reduced-Representation noise floor rises
~ sqrt(R-1)/sqrt(D)). This de-risk sweeps R = 2, 3, 4 and measures whether the FULL on-bridge chain (bind all R ->
bundle all R -> unbind all R in parallel into separate registers -> on-bridge cleanup) still recovers EVERY role.

Everything runs register->register on ONE persistent bridge, NO host round-trip: the R fillers are kicked, the binds
settle, the bundle settles the composite C, then all R unbind synapses fire in parallel into R separate Q registers,
and R separate concept-score blocks read their Q via conj(codebook) -- the answer is the argmax of each block's
membrane (re), exactly the step-3a/3b on-bridge cleanup, just R-wide.

GATE (3 seeds x 2 D, per R): on-bridge recovers all R roles == ground truth AND == the host composer pipeline
(`_encode` -> `_unbind_phases` -> `_cleanup`, the validated oracle). on-bridge < host at some R = the SUBSTRATE
phase-coherence cost (a reportable finding: that R needs a phase-latch on C before the production build scales to it).
Anti-cheat: a RANDOM-codebook cleanup (no real code matches Q) must collapse each block to chance. Reuse-by-import
(RFPhasorComposer + _build_rf_bridge); ADDITIVE runner, NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_multirole_coherence_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import to_host  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

# A richer vocab so a 4-role fact draws 4 DISTINCT fillers (noun, verb, adjective, adjective2).
AGENTS = ["dog", "cat", "bird", "river", "apple"]
ACTIONS = ["go", "come", "look", "stop", "swim"]
ADJ1 = ["big", "small", "red", "blue", "fast"]
ADJ2 = ["warm", "cold", "soft", "hard", "loud"]
VOCAB = AGENTS + ACTIONS + ADJ1 + ADJ2
ROLE_ORDER = ["agent", "action", "attribute", "attribute2"]
FILLERS_BY_ROLE = {"agent": AGENTS, "action": ACTIONS, "attribute": ADJ1, "attribute2": ADJ2}


def _store_and_query_onbridge(b, comp, fact, period, random_codebook=None):
    """Store an R-role fact on ONE persistent bridge and read every role's on-bridge cleanup answer.

    fact: dict role->filler_word, R = len(fact). Registers (each D wide):
      filler_r = r              (r in [0,R))     -- kicked with the filler phasor
      bound_r  = R + r                            -- bind output
      C        = 2R              (the composite)  -- bundle output (stored fact)
      Q_r      = 2R + 1 + r                        -- per-role unbind output
    concept-score neurons: R separate V-blocks at base = (3R+1)*D, role r -> [base+r*V : base+(r+1)*V].
    `random_codebook` (anti-cheat): if given (a list of R*V random phasor arrays), the cleanup reads via those
    instead of the true conj-codes -> no real match -> chance. Returns list[str] of length R (the argmax word per role).
    """
    D = comp.D
    V = len(VOCAB)
    roles = list(fact.keys())
    R = len(roles)
    role_ph = {r: comp._to_phasor(comp.roles[r]) for r in roles}
    fill_ph = {r: comp._to_phasor(comp.concepts[fact[r]]) for r in roles}

    binds = []
    for ri, r in enumerate(roles):
        binds += [((R + ri) * D + k, ri * D + k, complex(role_ph[r][k])) for k in range(D)]
    C = 2 * R
    bundle = []
    for ri in range(R):
        bundle += [(C * D + k, (R + ri) * D + k, 1.0) for k in range(D)]
    queries = []
    for ri, r in enumerate(roles):
        qreg = 2 * R + 1 + ri
        queries += [(qreg * D + k, C * D + k, complex(np.conj(role_ph[r][k]))) for k in range(D)]
    base = (3 * R + 1) * D
    cleanup = []
    for ri in range(R):
        qreg = 2 * R + 1 + ri
        for j in range(V):
            cc = (random_codebook[ri * V + j] if random_codebook is not None
                  else np.conj(comp._to_phasor(comp.concepts[VOCAB[j]])))
            cleanup += [(base + ri * V + j, qreg * D + d, complex(cc[d])) for d in range(D)]

    n_total = base + R * V
    kick = np.zeros(n_total, dtype=np.complex128)
    for ri, r in enumerate(roles):
        kick[ri * D:(ri + 1) * D] = fill_ph[r]
    # reset the bridge's RF register state before this fact (residual phasors would leak into the bundle on a
    # persistent bridge; in the production composer the STORED facts live in synapse weights, not this register state).
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    b.rf_set_complex_weights(binds)            # window 1: all R binds settle
    b.rf_kick(kick, period=period, lam=0.0)
    b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(bundle)           # window 2: bundle settles the composite C
    b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(queries)          # window 3: all R unbinds fire in parallel -> separate Q registers
    b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(cleanup)          # window 4: R concept-score blocks read their Q (parallel, no drift)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    answers = []
    for ri in range(R):
        scores = np.maximum(mem[base + ri * V:base + (ri + 1) * V], 0.0)
        answers.append(VOCAB[int(np.argmax(scores))])
    return answers


def _make_facts():
    """5 facts; each is a full 4-role tuple (we slice to the first R roles per sweep point)."""
    facts = []
    for i in range(5):
        facts.append({r: FILLERS_BY_ROLE[r][i] for r in ROLE_ORDER})
    return facts


def run_seed(seed, D, R):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    roles = ROLE_ORDER[:R]
    n_total = (3 * R + 1) * D + R * len(VOCAB)
    b = _build_rf_bridge(n_total, seed)
    facts = _make_facts()
    rng = np.random.default_rng(seed + 7919)
    rand_cb = [np.conj(np.exp(2j * np.pi * rng.uniform(0.0, 1.0, D))) for _ in range(R * len(VOCAB))]

    ob_truth = 0      # on-bridge answer == ground-truth filler
    ob_host = 0       # on-bridge answer == host composer oracle
    host_truth = 0    # host oracle == ground-truth filler (FHRR-capacity baseline)
    rand_truth = 0    # random-codebook anti-cheat (want chance ~ 1/V)
    n_role = 0
    for fact in facts:
        sub = {r: fact[r] for r in roles}
        ob = _store_and_query_onbridge(b, comp, sub, comp.period)
        rb = _store_and_query_onbridge(b, comp, sub, comp.period, random_codebook=rand_cb)
        # host oracle: encode the full R-role fact, unbind+cleanup each role
        comp_phases = comp._encode(sub)
        for ri, r in enumerate(roles):
            host_ans = comp._cleanup(comp._unbind_phases(comp_phases, r), VOCAB)
            truth = sub[r]
            ob_truth += int(ob[ri] == truth)
            ob_host += int(ob[ri] == host_ans)
            host_truth += int(host_ans == truth)
            rand_truth += int(rb[ri] == truth)
            n_role += 1
    row = {"seed": seed, "D": D, "R": R,
           "ob_truth": ob_truth / n_role, "ob_host": ob_host / n_role,
           "host_truth": host_truth / n_role, "rand_truth": rand_truth / n_role}
    print(f"  [seed {seed} D={D} R={R}] on-bridge=={row['ob_truth']:.2f} truth | =={row['ob_host']:.2f} host | "
          f"host=={row['host_truth']:.2f} truth | rand_cb {row['rand_truth']:.2f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--roles", type=str, default="2,3,4")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_multirole_coherence.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    role_counts = [int(r) for r in args.roles.split(",")]
    t0 = time.time()
    print("[one-brain multi-role coherence de-risk] does the on-bridge chain recover EVERY role as a fact gains "
          "roles (2->3->4)? phase coherence under bundle crosstalk on ONE persistent bridge.\n", flush=True)
    rows = [run_seed(s, D, R) for R in role_counts for s in seeds for D in dims]

    # per-R summary
    per_R = {}
    for R in role_counts:
        rs = [r for r in rows if r["R"] == R]
        per_R[R] = {k: float(np.mean([r[k] for r in rs])) for k in ("ob_truth", "ob_host", "host_truth", "rand_truth")}
    print(f"\n{'='*108}", flush=True)
    for R in role_counts:
        p = per_R[R]
        print(f"  R={R}: on-bridge {p['ob_truth']:.3f} truth | {p['ob_host']:.3f} host-parity | host {p['host_truth']:.3f}"
              f" truth | rand_cb {p['rand_truth']:.3f} (chance {1.0/len(VOCAB):.3f})", flush=True)
    # GATE: on-bridge tracks the host oracle at every swept R (substrate faithful), the anti-cheat is at chance, and
    # the highest R is still usefully recovered. The headline is the max R at which on-bridge==host AND on-bridge truth
    # stays >= 0.95 (the conversational-complexity reach before a phase-latch is needed).
    chance = 1.0 / len(VOCAB)
    parity_ok = all(per_R[R]["ob_host"] >= 0.95 for R in role_counts)
    anticheat_ok = all(per_R[R]["rand_truth"] <= 3 * chance for R in role_counts)
    reach = [R for R in role_counts if per_R[R]["ob_truth"] >= 0.95]
    max_reach = max(reach) if reach else 0
    go = parity_ok and anticheat_ok and (max_reach >= max(role_counts))
    if go:
        print(f"\n  GO: on-bridge tracks the host oracle at every R (substrate-faithful, parity >= 0.95) and recovers "
              f"all roles through R={max(role_counts)} -- a persistent bridge holds a MULTI-ATTRIBUTE fact and answers "
              f"every role with no host round-trip; the named phase-coherence risk does NOT bite to R={max(role_counts)}"
              f" at these D. Anti-cheat at chance.", flush=True)
    elif parity_ok and anticheat_ok and max_reach >= 2:
        print(f"\n  BOUNDARY: on-bridge stays faithful to the host (parity >= 0.95) but ground-truth recovery falls "
              f"below 0.95 past R={max_reach} -- that is the FHRR capacity limit (host degrades too), NOT a substrate "
              f"fault; facts beyond R={max_reach} need larger D or split binds. Anti-cheat at chance. Reportable.", flush=True)
    else:
        worst = min(per_R[R]["ob_host"] for R in role_counts)
        print(f"\n  SUBSTRATE-COST: on-bridge diverges from the host oracle (min parity {worst:.3f}) -- the bundle "
              f"crosstalk degrades the on-bridge phase code faster than the host; the composite register needs a "
              f"phase-latch before the production composer holds multi-attribute facts. Reportable finding.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)
    out = {"verdict": "GO" if go else ("BOUNDARY" if (parity_ok and anticheat_ok) else "SUBSTRATE-COST"),
           "seeds": seeds, "dims": dims, "roles": role_counts, "per_R": per_R, "max_reach": max_reach,
           "vocab_size": len(VOCAB), "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
