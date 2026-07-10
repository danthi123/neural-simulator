"""D3 EVENT -> the SELF-SUPERVISED delta, EXECUTED ON SPIKES: a transition learned with NO state label, whose running
agent slot is re-discretized by the project's own FS-WTA Izhikevich attractor. This closes the last two directives at
once for the event composition:

    EMERGENT   -- delta is learned from an agent-emission cross-entropy ALONE (no (agent,patient) label anywhere;
                  `_d3_event_selfsup_derisk`, 6-seed GO, adversarially verified by 3 skeptics), AND
    SPIKING    -- the running slot is maintained by a spiking one-of-K attractor (FS lateral inhibition), not a host
                  softmax/argmax.

So the running who-did-what-to-whom MEANING both EMERGES from prediction and RUNS on the spiking substrate.

THE ROLLOUT: at each clause the self-supervised transition produces a K-way score vector for the agent slot; that vector
drives a K-pool Izhikevich attractor bridge with a shared FS inhibitory pool; the spiking winner IS the next slot state
(fed back as a clean one-hot). No host argmax anywhere in the state path.

ANTI-CHEATS (6-seed): (a) the SPIKING probe on held-out-DEEPER >> chance, and on the coref-DEEP subset (>=3 trailing
corefs) it holds while a FAIR echo-state reservoir sits at chance; (b) per-step HOST-AGREE (the spiking winner == the
host argmax of the transition scores) -- the FS-WTA is faithful, i.e. it IS the state, not a check on one; (c) the
PROMOTE-bound subset (~51% of finals) where the honest label-free `last-named-subject` floor structurally fails;
(d) an EMISSION-SEVERED model (agent->emission link cut) collapses through the same spiking rollout -- so the spikes are
executing a LEARNED delta, not a generic attractor. Reuse-by-import (`_d3_event_selfsup_derisk` + `build_fswta_score_bridge`/
`fswta_drive`); numpy backend (small bridge); NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_selfsup_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_selfsup_derisk import (
    make_selfsup_event_task, train_selfsup, fair_reservoir, last_named_subject_floor,
    final_depth_and_promote, linear_probe, COREF, INTRO, PROMOTE)
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive


def soft_states_onehot(roll, task, split):
    """The (cheap) host rollout, hardened to one-hot -- used ONLY to fit the probe's slot->agent permutation on TRAIN."""
    fsa, y = roll(split)
    oh = np.zeros_like(fsa); oh[np.arange(len(fsa)), fsa.argmax(1)] = 1.0
    return oh, y


def spiking_selfsup_rollout(task, W, split, sb, K, settle=25, n_eval=600, seed=42):
    """Roll the SELF-SUPERVISED delta with the agent slot re-discretized ON SPIKES (FS-WTA). Returns the final one-hot
    slot states, the true agents, the item indices, and the per-step host-agreement."""
    emb, Wr, Wi, Wa, ba = W["emb"], W["Wr"], W["Wi"], W["Wa"], W["ba"]
    ident = task["ident"]
    S, SI, O, E, OP, Ls, T = task[split]
    rng = np.random.RandomState(seed + 1)
    idx = rng.choice(len(Ls), min(n_eval, len(Ls)), replace=False)
    states = np.zeros((len(idx), K), np.float32); ys = np.zeros(len(idx), np.int64)
    agree = steps = 0
    for j, n in enumerate(idx):
        a = ident; p = ident
        for t in range(int(Ls[n])):
            st_in = np.concatenate([emb[a], emb[p]])
            h = np.tanh(st_in @ Wr.T + S[n, t] @ Wi.T)
            la = h @ Wa.T + ba                                  # the self-supervised transition's K-way agent scores
            _, acc = fswta_drive(sb, K, la, settle=settle)      # re-discretized ON SPIKES (one-of-K attractor)
            a_spk = int(np.argmax(acc)) if acc.max() > 0 else ident
            agree += int(a_spk == int(np.argmax(la))); steps += 1
            a = a_spk                                           # ROLL OUT on the SPIKING winner
            p = int(O[n, t])                                    # the patient is the OBSERVED object
        states[j, a] = 1.0; ys[j] = int(T[n, int(Ls[n]) - 1])
    return states, ys, idx, agree / max(steps, 1)


def run_seed(seed, K, n_hid, epochs, theta_peak, settle, fs_inh, n_eval):
    task = make_selfsup_event_task(seed, K=K, theta_peak=theta_peak)
    depth, promo = final_depth_and_promote(task, "test_deeper")

    roll = train_selfsup(task, seed=seed, n_hid=n_hid, epochs=epochs)          # SELF-SUPERVISED (no state label)
    roll_sev = train_selfsup(task, seed=seed, n_hid=n_hid, epochs=epochs, random_emit=True)   # emission-severed control

    trX, trY = soft_states_onehot(roll, task, "train")                          # fit the probe's slot->agent permutation
    sb = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)
    teX, teY, idx, agree = spiking_selfsup_rollout(task, roll.W, "test_deeper", sb, K, settle=settle, n_eval=n_eval, seed=seed)
    pred = linear_probe(trX, trY, teX, teY, K)

    sb2 = build_fswta_score_bridge(seed=seed + 5, K=K, fs_to_exc=fs_inh)
    trXs, trYs = soft_states_onehot(roll_sev, task, "train")
    teXs, teYs, idxs, _ = spiking_selfsup_rollout(task, roll_sev.W, "test_deeper", sb2, K, settle=settle, n_eval=n_eval, seed=seed)
    pred_sev = linear_probe(trXs, trYs, teXs, teYs, K)

    p_res, y_res = fair_reservoir(task, seed=seed)                             # the FAIR architecture floor (host)
    p_lns, _ = last_named_subject_floor(task, "test_deeper")                    # the honest label-free floor (host)

    d = depth[idx] >= 3; pr = promo[idx]

    def acc(p, yy, mask=None):
        m = np.ones(len(yy), bool) if mask is None else mask
        return round(float((p[m] == yy[m]).mean()), 3) if m.sum() else float("nan")

    dres = depth >= 3
    return {"seed": seed, "K": K, "n_eval": int(len(idx)),
            "SPK_selfsup": acc(pred, teY), "SPK_selfsup_deep": acc(pred, teY, d), "SPK_selfsup_promote": acc(pred, teY, pr),
            "SPK_emission_severed": acc(pred_sev, teYs),
            "host_agree": round(agree, 3),
            "fair_reservoir_deep": acc(p_res, y_res, dres), "last_named_subject_promote": acc(p_lns, y_res, promo),
            "frac_deep": round(float(d.mean()), 3), "frac_promote": round(float(pr.mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--theta-peak", type=float, default=3.0)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--fs-inh", type=float, default=9.0)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT SELF-SUP SPIKING] K={a.K} | a delta learned with NO state label, its running agent slot re-discretized ON SPIKES (FS-WTA one-of-K attractor)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.theta_peak, a.settle, a.fs_inh, a.n_eval); rows.append(r)
        print(f"  [seed {s}] SPIKING self-sup={r['SPK_selfsup']} (deep>=3: {r['SPK_selfsup_deep']} | promote: {r['SPK_selfsup_promote']}) || "
              f"host-agree={r['host_agree']} | emission-severed(spiking)={r['SPK_emission_severed']} || "
              f"fair-reservoir deep={r['fair_reservoir_deep']} | last-named-subj promote={r['last_named_subject_promote']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        ss, ssd, ssp = _m("SPK_selfsup"), _m("SPK_selfsup_deep"), _m("SPK_selfsup_promote")
        sev, ag = _m("SPK_emission_severed"), _m("host_agree")
        resd, lnsp = _m("fair_reservoir_deep"), _m("last_named_subject_promote")
        chance = 1.0 / a.K
        go = (ss > 0.75) and (ssd > 0.65) and (ssd - resd > 0.3) and (ssp - lnsp > 0.3) and (ss - sev > 0.3) and (ag > 0.95)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}):", flush=True)
        print(f"    SPIKING self-sup={ss:.3f}  (coref-DEEP>=3: {ssd:.3f} | promote-bound: {ssp:.3f})", flush=True)
        print(f"    per-step host-agree={ag:.3f} (the FS-WTA winner IS the state, not a check on a host argmax)", flush=True)
        print(f"    emission-severed through the SAME spiking rollout={sev:.3f} (the spikes execute a LEARNED delta)", flush=True)
        print(f"    floors: fair reservoir (deep)={resd:.3f} | last-named-subject (promote)={lnsp:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the event transition delta -- LEARNED with NO (agent,patient) state label, from agent-emission prediction alone -- is EXECUTED ON SPIKES: its running agent slot is re-discretized by a spiking one-of-K FS-WTA Izhikevich attractor (per-step host-agree '+format(ag,'.2f')+', so the spiking winner IS the state), decoding the running agent at '+format(ss,'.2f')+' on held-out-DEEPER and '+format(ssd,'.2f')+' on the coref-DEEP subset where a FAIR echo-state reservoir sits at '+format(resd,'.2f')+'; the promote-bound edge holds ('+format(ssp,'.2f')+' vs the honest label-free floor '+format(lnsp,'.2f')+'); an EMISSION-SEVERED model collapses through the SAME spiking rollout ('+format(sev,'.2f')+') so the spikes execute a LEARNED delta, not a generic attractor -> the running who-did-what-to-whom MEANING both EMERGES from prediction and RUNS on the spiking substrate' if go else 'the spiking self-supervised delta did not clearly hold (read SPIKING vs the deep/promote contrasts + host-agree)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
