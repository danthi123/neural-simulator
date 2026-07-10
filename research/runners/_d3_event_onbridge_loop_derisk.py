"""D3 EVENT COMPOSITION — the fully-spiking ONE-LOOP: the WHOLE two-slot event step in ONE spiking loop (the LIF
transition-forward + the FS-WTA re-discretization, both slots, feeding back). The event arc validated the two halves
SEPARATELY: the transition LEARNED through a spiking LIF hidden (rung 6) and the re-discretization ON SPIKES (the FS-WTA
port, rung 3). THIS composes them into ONE loop for the FACTORED event — each rollout step:
    (i)   the LEARNED spiking LIF transition-forward -> two K-way score vectors (agent, patient)
    (ii)  each score vector drives its OWN K-pool Izhikevich FS-WTA attractor bridge -> the spiking winner
    (iii) the two spiking winners = the next (a, p), FED BACK as the next state
So the running who-did-what-to-whom MEANING is maintained by a single spiking loop whose transition is LEARNED-on-spikes
and whose re-discretization is on-spikes — the master-directive "fully spiking, one loop" for the event. Transition
trained SHALLOW (len 1/2/3); the loop rolled out held-out DEEP (len 6/7/8), a genuinely-deep task (AGENT-COREF).

ANTI-CHEATS: (a) the full-spiking-loop deeper-track >> the LAST-2-OBJECTS shallow reader (genuinely deep) AND >> chance;
(b) per-slot spiking-winner == host-argmax (the FS-WTAs are faithful); (c) multi-seed dev+blind. Reuse-by-import
(`train_event_spiking_weak` weights + `build_fswta_score_bridge`/`fswta_drive` + `lif_rate`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_onbridge_loop_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, last2_objects_floor
from research.runners._d3_event_spiking_learning_derisk import train_event_spiking_weak
from research.runners._d3_spiking_weak_learning_derisk import lif_rate
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive


def spiking_loop_eval(task, W, split, sb_a, sb_p, K, settle=25, n_eval=60, seed=42):
    """Autoregressive rollout: the LEARNED spiking LIF transition -> scores -> two FS-WTA re-discretizations -> next
    (a,p) fed back. Both halves on spikes, in one loop."""
    emb, W1 = W["emb"], W["W1"]; Wa, ba, Wp, bp = W["Wa"], W["ba"], W["Wp"], W["bp"]; T = W["T"]
    ident = task["ident"]
    X, Ya, Yp, L, SEQ, STA, STP = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(L), min(n_eval, len(L)), replace=False)
    ok_evt = 0; agree_a = agree_p = 0; steps = 0
    for n in idx:
        a = p = ident
        for t in range(int(L[n])):
            feat = np.concatenate([emb[a], emb[p], X[n, t]])[None, :]        # [1, n_in]
            rate, _ = lif_rate(feat, W1, T)                                   # (i) LEARNED spiking LIF transition-forward
            a_scores = (rate @ Wa.T + ba)[0]; p_scores = (rate @ Wp.T + bp)[0]
            _, acc_a = fswta_drive(sb_a, K, a_scores, settle=settle)          # (ii) FS-WTA re-discretize agent slot
            _, acc_p = fswta_drive(sb_p, K, p_scores, settle=settle)          #      FS-WTA re-discretize patient slot
            na = int(np.argmax(acc_a)) if acc_a.max() > 0 else ident
            npp = int(np.argmax(acc_p)) if acc_p.max() > 0 else ident
            agree_a += int(na == int(np.argmax(a_scores))); agree_p += int(npp == int(np.argmax(p_scores))); steps += 1
            a, p = na, npp                                                    # (iii) spiking winners fed back
        ok_evt += int(a == int(STA[n, int(L[n]) - 1]) and p == int(STP[n, int(L[n]) - 1]))
    m = len(idx)
    return {"evt": ok_evt / m, "agree_a": agree_a / max(steps, 1), "agree_p": agree_p / max(steps, 1)}


def run_seed(seed, K, n_hid, T, epochs, settle):
    task = make_event_task(seed, K=K, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    W = train_event_spiking_weak(task, seed=seed, n_hid=n_hid, T=T, epochs=epochs)["weights"]   # LEARNED spiking transition
    sb_a = build_fswta_score_bridge(seed=seed, K=K); sb_p = build_fswta_score_bridge(seed=seed + 7, K=K)
    r = spiking_loop_eval(task, W, "test_deeper", sb_a, sb_p, K, settle=settle, seed=seed)
    return {"seed": seed, "K": K, "LOOP_event_deeper": round(r["evt"], 3),
            "agree_agent": round(r["agree_a"], 3), "agree_patient": round(r["agree_p"], 3),
            "last2_objects_floor": round(last2_objects_floor(task), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT ONE-LOOP] K={a.K} | the WHOLE two-slot event step in ONE spiking loop (LEARNED LIF transition-forward + FS-WTA re-discretization, both slots, feeding back)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.T, a.epochs, a.settle); rows.append(r)
        print(f"  [seed {s}] full-spiking-LOOP event DEEPER={r['LOOP_event_deeper']} || per-slot host-agree a={r['agree_agent']} p={r['agree_patient']} || LAST-2-OBJ(shallow)={r['last2_objects_floor']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ev, aa, ap_, l2 = _m("LOOP_event_deeper"), _m("agree_agent"), _m("agree_patient"), _m("last2_objects_floor")
        go = (ev > 0.7) and (ev - l2 > 0.2) and (aa > 0.9) and (ap_ > 0.9)
        print(f"\n  AGGREGATE (K={a.K}): full-spiking-LOOP event DEEPER={ev:.3f} | per-slot host-agree agent={aa:.3f} patient={ap_:.3f} | LAST-2-OBJ(shallow)={l2:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the WHOLE two-slot event step runs in ONE spiking loop -- the LEARNED-on-spikes LIF transition-forward + the FS-WTA re-discretization, both slots, feeding back -- length-generalizing to the genuinely-DEEP task (LOOP event-track '+format(ev,'.2f')+' >> the LAST-2-OBJECTS shallow reader '+format(l2,'.2f')+', per-slot WTAs faithful == host argmax a='+format(aa,'.2f')+'/p='+format(ap_,'.2f')+') -> the running who-did-what-to-whom MEANING is maintained by a single spiking loop whose transition is LEARNED-on-spikes and whose re-discretization is on-spikes = the simulated recurrent sequence/language cortex step for a composed EVENT, fully realized on spikes end-to-end' if go else 'the event one-loop did not hold cleanly (tune epochs/T/fs-settle; read the per-slot host-agree)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
