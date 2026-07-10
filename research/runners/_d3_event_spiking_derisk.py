"""D3 EVENT COMPOSITION — the SPIKING port: the running FACTORED (agent, patient) MEANING re-discretized ON SPIKES.
The rate de-risks proved the discrete-attractor maintains a running two-slot event (per-step GO 0.993; weak-supervisable
GO 0.996). THIS moves the re-discretization onto the project's OWN spiking substrate — the master-directive "fully
spiking on one brain" requirement — for the factored event: each step's transition produces TWO K-way score vectors
(agent, patient); each drives its OWN K-pool Izhikevich attractor bridge with a shared FS lateral-inhibition pool (the
CA3/NEF clean one-of-K winner); the two spiking winners = the next (a, p); iterate. So the running who-did-what-to-whom
MEANING is maintained as TWO co-evolving spiking attractors, composing the relational role-shift to held-out-DEEPER depth.

RUNG SCOPE (mirrors the group spiking port `_d3_spiking_attractor_derisk`): the TRANSITION (delta) is the rate-learned
`factored_event_rnn` weights; only the RE-DISCRETIZATION is on-spikes (two FS-WTA slots). Anti-cheats: (a) the per-slot
spiking-WTA winner == the host-argmax winner (the two WTAs are faithful); (b) held-out-DEEPER spiking EVENT-track (both
slots) >> chance (1/K^2) == the rate result; (c) the running state is load-bearing (inherited: the rate recurrence-lesion
collapses). Reuse-by-import (`factored_event_rnn` weights + `build_fswta_score_bridge`/`fswta_drive`); numpy backend
(small bridge); NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, factored_event_rnn
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive


def spiking_event_rollout(task, W, split, sb_a, sb_p, K, settle=25, n_eval=60, seed=42):
    """Autoregressive rollout with TWO on-bridge FS-WTA re-discretizations (agent slot + patient slot). Each step: the
    rate transition -> (a_scores, p_scores) -> drive the two attractor bridges -> the two spiking winners = next (a,p)."""
    emb, Wr, Wi = W["emb"], W["Wr"], W["Wi"]; Wa, ba, Wp, bp = W["Wa"], W["ba"], W["Wp"], W["bp"]
    ident = task["ident"]
    X, Ya, Yp, L, SEQ, STA, STP = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(L), min(n_eval, len(L)), replace=False)
    ok_evt = ok_a = ok_p = 0; agree_a = agree_p = 0; steps = 0
    for n in idx:
        a = p = ident
        for t in range(int(L[n])):
            h = np.tanh(np.concatenate([emb[a], emb[p]]) @ Wr.T + X[n, t] @ Wi.T)
            a_scores = h @ Wa.T + ba; p_scores = h @ Wp.T + bp     # the two K-way transition scores
            _, acc_a = fswta_drive(sb_a, K, a_scores, settle=settle)   # agent slot re-discretized ON SPIKES
            _, acc_p = fswta_drive(sb_p, K, p_scores, settle=settle)   # patient slot re-discretized ON SPIKES
            na_spk = int(np.argmax(acc_a)) if acc_a.max() > 0 else ident
            np_spk = int(np.argmax(acc_p)) if acc_p.max() > 0 else ident
            agree_a += int(na_spk == int(np.argmax(a_scores))); agree_p += int(np_spk == int(np.argmax(p_scores))); steps += 1
            a, p = na_spk, np_spk                                  # ROLL OUT on the SPIKING winners
        ta = int(STA[n, int(L[n]) - 1]); tp = int(STP[n, int(L[n]) - 1])
        ok_evt += int(a == ta and p == tp); ok_a += int(a == ta); ok_p += int(p == tp)
    m = len(idx)
    return {"evt": ok_evt / m, "agent": ok_a / m, "patient": ok_p / m,
            "agree_a": agree_a / max(steps, 1), "agree_p": agree_p / max(steps, 1)}


def run_seed(seed, K, n_hid, epochs, settle, fs_inh):
    task = make_event_task(seed, K=K, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    fac = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)
    W = fac["weights"]
    sb_a = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)       # agent-slot attractor bridge
    sb_p = build_fswta_score_bridge(seed=seed + 7, K=K, fs_to_exc=fs_inh)   # patient-slot attractor bridge (distinct init)
    spk = spiking_event_rollout(task, W, "test_deeper", sb_a, sb_p, K, settle=settle, seed=seed)
    return {"seed": seed, "K": K, "rate_event_deeper": round(fac["event_deeper"], 3),
            "SPK_event_deeper": round(spk["evt"], 3), "SPK_agent": round(spk["agent"], 3), "SPK_patient": round(spk["patient"], 3),
            "agree_agent": round(spk["agree_a"], 3), "agree_patient": round(spk["agree_p"], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--fs-inh", type=float, default=9.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT SPIKING] K={a.K} | the running FACTORED (agent, patient) MEANING re-discretized ON SPIKES (two FS-WTA Izhikevich attractor slots)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.settle, a.fs_inh); rows.append(r)
        print(f"  [seed {s}] rate event DEEPER={r['rate_event_deeper']} || SPIKING event DEEPER={r['SPK_event_deeper']} "
              f"(a={r['SPK_agent']} p={r['SPK_patient']}) || per-slot host-agree a={r['agree_agent']} p={r['agree_patient']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        spk, aa, ap_ = _m("SPK_event_deeper"), _m("agree_agent"), _m("agree_patient")
        chance = 1.0 / (a.K * a.K)
        go = (spk > 0.85) and (aa > 0.95) and (ap_ > 0.95)
        print(f"\n  AGGREGATE (K={a.K}, event chance {chance:.3f}): SPIKING event DEEPER={spk:.3f} | per-slot host-agree agent={aa:.3f} patient={ap_:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the running FACTORED (agent, patient) MEANING is re-discretized ON SPIKES (two co-evolving FS-WTA Izhikevich attractor slots) and composes the relational role-shift to held-out-DEEPER depth (spiking event-track '+format(spk,'.2f')+', both per-slot WTAs faithful == host argmax a='+format(aa,'.2f')+'/p='+format(ap_,'.2f')+') -> the anti-RAG running who-did-what-to-whom MEANING runs on the project spiking substrate = the simulated recurrent sequence/language cortex maintaining a composed event; next: wrap the composer fixed per-slot bind + learn the transition on-substrate' if go else 'the spiking two-slot event did not hold cleanly (tune fs-inh/settle; read the per-slot host-agree)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
