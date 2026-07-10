"""D3 EVENT QA -> the FULLY-SPIKING read-out: the RANK-3 reasoning/QA over the composed running event, end-to-end ON THE
SPIKING SUBSTRATE. The rate QA (`_d3_event_qa_derisk`) is 6-seed GO (deep-agent QA ~0.982 vs recency ~0.375); the
non-negotiable is fully-spiking-on-one-brain, so THIS moves BOTH QA stages onto the project's spiking FS-WTA substrate:

  STAGE 1 (resolve the referent, SPIKING): the running FACTORED event is re-discretized by two co-evolving FS-WTA
          Izhikevich attractor slots (the `_d3_event_spiking_derisk` port) -> the resolved coref-DEEP agent, on spikes.
  STAGE 2 (key the fact store, SPIKING): the resolved entity's code drives a THIRD FS-WTA property bridge through a
          learned associative map (emb[entity] -> property) -> the answer property spikes out. This is the composer's
          associative-recall role realized as a spiking read-out (NEF/engram-style), so the fact store is keyed ON SPIKES.

=> "what does HE eat?" is answered by SPIKES end-to-end: spiking-composed running agent -> spiking associative recall of
its stored property. The anti-RAG payoff (situation model x fact store) on the substrate, no host argmax in the QA path.

ANTI-CHEATS (6-seed): (a) SPIKING deep-agent QA >> chance (1/K) == the rate QA; (b) recency (spiking-composed but resolve
to last-mentioned) FAILS; (c) the two read-out FS-WTAs are FAITHFUL (spiking winner == host argmax) for the resolve AND
the property stages; (d) property-store keyed on spikes (the property FS-WTA winner == prop[resolved]). Reuse-by-import
(`factored_event_rnn` + `build_fswta_score_bridge`/`fswta_drive` + `build_fact_store`/`recency_resolved` from the rate QA);
numpy backend (small bridges); NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_qa_spiking_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, factored_event_rnn
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive
from research.runners._d3_event_qa_derisk import build_fact_store, recency_resolved


def train_property_readout(emb, prop, K, seed, epochs=400, lr=0.2):
    """Learned associative map emb[entity] -> property label (the fact store as a spiking-read-out weight). Softmax
    regression over the K distinct entity embeddings (trivially separable); returns Wprop[K_prop, n_hid], bprop."""
    rng = np.random.RandomState(seed + 31); n_hid = emb.shape[1]
    Wp = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(K, np.float32)
    eye = np.eye(K, dtype=np.float32)
    for _ in range(epochs):
        z = emb @ Wp.T + bp; z -= z.max(1, keepdims=True); e = np.exp(z); sm = e / e.sum(1, keepdims=True)
        d = (sm - eye[prop]) / K; Wp -= lr * (d.T @ emb); bp -= lr * d.sum(0)
    return Wp, bp


def spiking_qa_rollout(task, W, Wprop, bprop, split, sb_a, sb_p, sb_prop, K, prop, settle=25, n_eval=120, seed=42):
    """Compose the event ON SPIKES (two FS-WTA slots) -> the resolved agent; then key the fact store ON SPIKES (a third
    FS-WTA property bridge driven by the learned associative map) -> the answer property. Returns spiking deep-agent QA +
    per-stage host-agreement + a spiking-recency comparison."""
    emb, Wr, Wi = W["emb"], W["Wr"], W["Wi"]; Wa, ba, Wp, bp = W["Wa"], W["ba"], W["Wp"], W["bp"]
    ident = task["ident"]; X, Ya, Yp, L, SEQ, STA, STP = task[split]
    rng = np.random.RandomState(seed + 1); idx = rng.choice(len(L), min(n_eval, len(L)), replace=False)
    ra, rp = recency_resolved(task, split, K)
    ok_qa = ok_rec = agree_resolve = agree_prop = steps = qsteps = 0
    for n in idx:
        a = p = ident
        for t in range(int(L[n])):
            h = np.tanh(np.concatenate([emb[a], emb[p]]) @ Wr.T + X[n, t] @ Wi.T)
            a_sc = h @ Wa.T + ba; p_sc = h @ Wp.T + bp
            _, acc_a = fswta_drive(sb_a, K, a_sc, settle=settle); _, acc_p = fswta_drive(sb_p, K, p_sc, settle=settle)
            na = int(np.argmax(acc_a)) if acc_a.max() > 0 else ident
            npp = int(np.argmax(acc_p)) if acc_p.max() > 0 else ident
            agree_resolve += int(na == int(np.argmax(a_sc))); steps += 1
            a, p = na, npp
        # STAGE 2: key the fact store ON SPIKES for the resolved (deep) AGENT
        pr_sc = emb[a] @ Wprop.T + bprop                              # the learned associative-map property scores
        _, acc_pr = fswta_drive(sb_prop, K, pr_sc, settle=settle)     # property FS-WTA (spiking associative recall)
        ans = int(np.argmax(acc_pr)) if acc_pr.max() > 0 else ident
        agree_prop += int(ans == int(np.argmax(pr_sc))); qsteps += 1
        ta = int(STA[n, int(L[n]) - 1])                              # the TRUE deep agent
        ok_qa += int(ans == int(prop[ta]))                          # spiking QA correct?
        ok_rec += int(int(prop[ra[n]]) == int(prop[ta]))            # recency floor (resolve->last-mentioned)
    m = len(idx)
    return {"SPK_qa": ok_qa / m, "recency_qa": ok_rec / m,
            "agree_resolve": agree_resolve / max(steps, 1), "agree_prop": agree_prop / max(qsteps, 1)}


def run_seed(seed, K, n_hid, epochs, settle, fs_inh):
    task = make_event_task(seed, K=K, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    W = factored_event_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs, joint=False)["weights"]
    prop = build_fact_store(seed, K)
    Wprop, bprop = train_property_readout(W["emb"], prop, K, seed)
    sb_a = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)
    sb_p = build_fswta_score_bridge(seed=seed + 7, K=K, fs_to_exc=fs_inh)
    sb_prop = build_fswta_score_bridge(seed=seed + 13, K=K, fs_to_exc=fs_inh)   # the property (fact-store) FS-WTA
    r = spiking_qa_rollout(task, W, Wprop, bprop, "test_deeper", sb_a, sb_p, sb_prop, K, prop, settle=settle, seed=seed)
    return {"seed": seed, "K": K, "SPK_deep_agent_QA": round(r["SPK_qa"], 3), "recency_QA": round(r["recency_qa"], 3),
            "agree_resolve": round(r["agree_resolve"], 3), "agree_property": round(r["agree_prop"], 3)}


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
    print(f"[D3 EVENT QA SPIKING] K={a.K} | QA over the composed running event, BOTH stages ON SPIKES: spiking-composed agent -> spiking associative recall of its stored property", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, a.settle, a.fs_inh); rows.append(r)
        print(f"  [seed {s}] SPIKING deep-agent QA={r['SPK_deep_agent_QA']} vs recency={r['recency_QA']} || host-agree resolve={r['agree_resolve']} property={r['agree_property']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        spk, rec, ares, aprop = _m("SPK_deep_agent_QA"), _m("recency_QA"), _m("agree_resolve"), _m("agree_property")
        chance = 1.0 / a.K
        go = (spk > 0.75) and (spk - rec > 0.3) and (ares > 0.9) and (aprop > 0.9)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}): SPIKING deep-agent QA={spk:.3f} | recency={rec:.3f} || host-agree resolve={ares:.3f} property={aprop:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the RANK-3 QA over the composed running event runs END-TO-END ON SPIKES ('+format(spk,'.2f')+'): STAGE-1 the running agent re-discretized on the FS-WTA attractor (resolve host-agree '+format(ares,'.2f')+'), STAGE-2 the fact store keyed by a spiking associative-recall FS-WTA (property host-agree '+format(aprop,'.2f')+'); a recency resolver FAILS ('+format(rec,'.2f')+') -> the anti-RAG situation-model x fact-store QA answers a question on the project spiking substrate, no host argmax in the QA path = the simulated cortex reasoning over a running meaning; next: multi-turn/connectives, wire into the live agent QA' if go else 'the spiking QA did not cleanly hold (tune fs-inh/settle; read the per-stage host-agree)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
