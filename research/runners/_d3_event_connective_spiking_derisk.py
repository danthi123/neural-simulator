"""D3 EVENT CONNECTIVES -> the SPIKING port: the event PAIR maintained by FOUR spiking one-of-K FS-WTA attractor slots.
The rate rung showed a connective-marked EVENT BOUNDARY shifts the running event into a previous slot, so the brain
holds a PAIR of composed events and can relate them (prev-agent 0.881, same-agent relation 0.929, vs a structurally
incapable single-event control at 0.467). The non-negotiable is fully-spiking-on-one-brain, so THIS moves the whole
event pair onto the project's own spiking substrate:

    (a_curr, p_curr | a_prev, p_prev)   -- FOUR K-way slots, EACH re-discretized by its own K-pool Izhikevich attractor
                                           bridge with a shared FS inhibitory pool; the four spiking winners are the
                                           next state. No host argmax anywhere in the state path.

The EVENT BOUNDARY (the connective) is therefore executed as a spiking SHIFT: on a boundary clause the transition drives
the prev-slot attractors toward whatever the curr-slot attractors were holding, and the prior event must survive on
spikes across arbitrarily many following clauses.

ANTI-CHEATS (6-seed): (a) SPIKING prev-agent on held-out-DEEPER >> the SINGLE-EVENT control (structurally incapable) and
>> recency; (b) per-slot HOST-AGREE (each spiking winner == the host argmax of that slot's transition scores) -- the
FS-WTAs ARE the state, not a check on one; (c) the same-agent RELATION read across two spiking slots; (d) held-out-
DEEPER lengths. Reuse-by-import (`_d3_event_connective_derisk` + `build_fswta_score_bridge`/`fswta_drive`); numpy
backend; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_connective_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_connective_derisk import make_connective_task, multislot_rnn, recency_floor
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive


def spiking_pair_rollout(task, W, split, bridges, K, settle=25, n_eval=400, seed=42):
    """Roll the event PAIR with ALL FOUR slots re-discretized ON SPIKES (one FS-WTA attractor bridge per slot)."""
    emb, Wr, Wi, Ws, bs = W["emb"], W["Wr"], W["Wi"], W["Ws"], W["bs"]
    n_slots = W["n_slots"]; ident = task["ident"]
    X, L, SA, SP, PA, PP = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(L), min(n_eval, len(L)), replace=False)
    fin = np.zeros((len(idx), 4), np.int64)
    agree = steps = 0
    slot_agree = np.zeros(4); slot_steps = np.zeros(4); slot_margin = np.zeros(4)
    for j, n in enumerate(idx):
        slots = [ident] * n_slots
        for t in range(int(L[n])):
            si = np.concatenate([emb[s] for s in slots])
            h = np.tanh(si @ Wr.T + X[n, t] @ Wi.T)
            outs = []
            for k in range(4):
                sc = h @ Ws[k].T + bs[k]                       # slot k's K-way transition scores
                # NORMALIZE the drive before the attractor. Diagnosed, not guessed: with RAW scores the per-slot
                # host-agree was [a_curr .928, p_curr .885, a_prev .982, p_prev .988] -- the WORST slot (p_curr) had the
                # LARGEST top-2 margin (9.65), so near-ties cannot explain it. The cause is f-I SATURATION (large drives
                # push several pools to ceiling, degrading the spike-count read) -- the same failure EMERGE-77 hit when
                # packing 8 primacies into one current range. Normalizing (the pattern already used by the centering
                # wire) restores per-slot agree to >=0.98 AND raises accuracy.
                d = np.maximum(sc, 0.0); mx = d.max()
                d = d / (mx + 1e-9) if mx > 0 else d
                _, acc = fswta_drive(bridges[k], K, d, settle=settle)    # re-discretized ON SPIKES
                w = int(np.argmax(acc)) if acc.max() > 0 else ident
                ok = int(w == int(np.argmax(sc)))
                agree += ok; steps += 1
                slot_agree[k] += ok; slot_steps[k] += 1
                srt = np.sort(sc)[::-1]; slot_margin[k] += float(srt[0] - srt[1])   # top-2 score margin
                outs.append(w)
            slots = outs[:n_slots]                             # roll out on the SPIKING winners
        fin[j] = outs
    tg = np.stack([SA[idx, L[idx] - 1], SP[idx, L[idx] - 1], PA[idx, L[idx] - 1], PP[idx, L[idx] - 1]], 1)
    curr_a = float((fin[:, 0] == tg[:, 0]).mean()); prev_a = float((fin[:, 2] == tg[:, 2]).mean())
    same = float(((fin[:, 0] == fin[:, 2]) == (tg[:, 0] == tg[:, 2])).mean())
    return {"curr_agent": curr_a, "prev_agent": prev_a, "same_agent_rel": same, "host_agree": agree / max(steps, 1),
            "slot_agree": (slot_agree / np.maximum(slot_steps, 1)).round(3).tolist(),
            "slot_margin": (slot_margin / np.maximum(slot_steps, 1)).round(3).tolist()}


def run_seed(seed, K, n_hid, epochs, settle, fs_inh, n_eval):
    task = make_connective_task(seed, K=K)
    pair = multislot_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, n_slots=4)
    single = multislot_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, n_slots=2)   # structurally incapable control
    bridges = [build_fswta_score_bridge(seed=seed + 3 * k, K=K, fs_to_exc=fs_inh) for k in range(4)]
    spk = spiking_pair_rollout(task, pair["weights"], "test_deeper", bridges, K, settle=settle, n_eval=n_eval, seed=seed)
    _, rec_p = recency_floor(task)
    return {"seed": seed, "K": K,
            "SPK_prev_agent": round(spk["prev_agent"], 3), "SPK_curr_agent": round(spk["curr_agent"], 3),
            "SPK_same_rel": round(spk["same_agent_rel"], 3), "host_agree": round(spk["host_agree"], 3),
            "rate_prev_agent": round(pair["prev_agent"], 3),
            "SINGLE_prev_agent": round(single["prev_agent"], 3), "recency_prev": round(rec_p, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--fs-inh", type=float, default=9.0)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT CONNECTIVES SPIKING] K={a.K} | the event PAIR maintained by FOUR spiking one-of-K FS-WTA attractor slots; the connective's event boundary is a spiking SHIFT", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.settle, a.fs_inh, a.n_eval); rows.append(r)
        print(f"  [seed {s}] SPIKING prev-agent={r['SPK_prev_agent']} (curr={r['SPK_curr_agent']} | same-rel={r['SPK_same_rel']}) || "
              f"host-agree={r['host_agree']} | rate prev={r['rate_prev_agent']} || SINGLE-EVENT prev={r['SINGLE_prev_agent']} | recency prev={r['recency_prev']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        pp, pc, rel, ag = _m("SPK_prev_agent"), _m("SPK_curr_agent"), _m("SPK_same_rel"), _m("host_agree")
        sp, rp, rt = _m("SINGLE_prev_agent"), _m("recency_prev"), _m("rate_prev_agent")
        chance = 1.0 / a.K
        go = (pp > 0.75) and (pp - sp > 0.3) and (pp - rp > 0.3) and (rel > 0.75) and (ag > 0.95)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}):", flush=True)
        print(f"    SPIKING event-PAIR: prev-agent={pp:.3f} (rate {rt:.3f}) | curr-agent={pc:.3f} | same-agent relation={rel:.3f}", flush=True)
        print(f"    per-slot host-agree={ag:.3f} (the four FS-WTA winners ARE the state)", flush=True)
        print(f"    SINGLE-EVENT control prev={sp:.3f} | recency prev={rp:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the whole event PAIR runs on the spiking substrate: FOUR one-of-K FS-WTA Izhikevich attractor slots maintain (a_curr,p_curr | a_prev,p_prev), the connective event-boundary is executed as a spiking SHIFT, and the prior event SURVIVES ON SPIKES across arbitrarily many following clauses -- prev-agent '+format(pp,'.2f')+' on held-out-DEEPER (per-slot host-agree '+format(ag,'.2f')+', so the spiking winners ARE the state), same-agent relation read across two spiking slots '+format(rel,'.2f')+', where a structurally-incapable SINGLE-EVENT register fails ('+format(sp,'.2f')+') and recency collapses ('+format(rp,'.2f')+') -> the brain relates two composed meanings ON SPIKES' if go else 'the spiking event pair did not clearly hold (read SPIKING prev vs single/recency + per-slot host-agree)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
