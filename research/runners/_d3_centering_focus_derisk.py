"""D3 -> the AGENT's actual discourse focus (Centering over SVO): the production wire-in's foundation. The anaphora
integration (`_d3_anaphora_integration_derisk`) drove the resolution with D3's focus tracked on the POSSESSION delta;
the deployed `MultiTurnAgent` hears SVO facts and resolves pronouns by the host `content_bias_target` (feature
compatibility). To CLOSE that host shortcut with D3's brain-based composed focus, D3 must track the agent's actual
discourse center. THIS models it: Centering Theory's backward-looking center Cb over SVO utterances --
    delta(Cb, (subj=s, obj=o)) = Cb   if Cb in {s, o}   (the center CONTINUES -- it is realized in this utterance)
                                 s     otherwise          (SHIFT -- the center moves to the new subject)
a state-dependent single-K-way focus update (Grosz-Joshi-Weinstein 1995: Cb = the highest-ranked realized center;
subject-preferred). The pronoun binds to Cb. D3 tracks it; the RECENCY baseline (bind to the most-recently-mentioned
entity = the last object) FAILS when the center has continued while a new object was mentioned (Cb != most-recent).
DISTINCT delta from the possession task -> a second genuinely-linguistic iterative-compose focus rule.

ANTI-CHEATS: (a) D3 held-out-DEEPER Cb-track >> chance; (b) the RECENCY (last-object) baseline fails at chance on
focus-continued discourses; (c) order matters (non-commutative); (d) held-out-DEEPER (longer discourses); multi-seed.
Reuse-by-import (`discrete_attractor_rnn` [D3 tracker]); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_centering_focus_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import discrete_attractor_rnn


def make_centering_task(seed, K=6, n_pool=64, noise=0.6, train_lens=(1, 2, 3), test_lens=(6, 7, 8),
                        n_per_len=2500, p_continue=0.6):
    """SVO discourse; state = the Centering backward-looking center Cb (0..K-1; ident=0). Emits the make_group_task
    dict (X[N,Lmax,n_pool], y, L, SEQ=s*K+o, STATE=Cb)."""
    rng = np.random.RandomState(seed)
    half = n_pool // 2; code_k = max(3, half // 4)
    base = -np.ones((K, half), dtype=np.float32)                  # entity codes (subj half + obj half)
    for e in range(K):
        base[e, rng.choice(half, code_k, replace=False)] = 1.0
    color = rng.randint(0, 2, size=K); ident = 0
    Lmax = max(tuple(train_lens) + tuple(test_lens))

    def gen(lens, n_each):
        X, Y, L, SEQ, STATE = [], [], [], [], []
        for L_ in lens:
            for _ in range(n_each):
                cb = ident
                codes = np.zeros((Lmax, n_pool), dtype=np.float32)
                cb_seq = np.full(Lmax, -1, dtype=np.int64); so_seq = np.full(Lmax, -1, dtype=np.int64)
                for t in range(L_):
                    force_shift_last = (t == L_ - 1) and (L_ >= 2)   # FORCE the last utterance to CONTINUE cb as SUBJECT
                    #                                                  + a NEW object -> recency(=new object) != cb (the
                    #                                                  center), so "last-mentioned" fails while cb persists.
                    if force_shift_last:
                        s = cb; o = int(rng.choice([e for e in range(K) if e != cb]))
                    elif rng.rand() < p_continue:                   # CONTINUE: mention cb (as subj or obj)
                        if rng.rand() < 0.5:
                            s = cb; o = int(rng.randint(0, K))
                        else:
                            s = int(rng.randint(0, K)); o = cb
                    else:                                           # SHIFT: cb not realized -> new subject
                        others = [e for e in range(K) if e != cb]
                        s = int(rng.choice(others)); o = int(rng.choice([e for e in range(K) if e != cb]))
                    cb = cb if cb in (s, o) else s                  # the Centering delta
                    cb_seq[t] = cb; so_seq[t] = s * K + o
                    c = np.concatenate([base[s], base[o]]).copy()
                    flip = rng.rand(n_pool) < (noise * 0.15); c[flip] = -c[flip]
                    codes[t] = c
                X.append(codes); Y.append(int(color[cb])); L.append(L_); SEQ.append(so_seq); STATE.append(cb_seq)
        return (np.asarray(X, np.float32), np.asarray(Y, np.int64), np.asarray(L, np.int64),
                np.asarray(SEQ, np.int64), np.asarray(STATE, np.int64))

    return {"train": gen(train_lens, n_per_len), "test_same": gen(train_lens, max(400, n_per_len // 4)),
            "test_deeper": gen(test_lens, max(400, n_per_len // 4)),
            "K": K, "ident": ident, "n_pool": n_pool, "color": color}


def recency_floor(task):
    """RECENCY baseline: bind the pronoun to the most-recently-mentioned entity (the last utterance's OBJECT). K-way vs
    the true Cb (chance 1/K)."""
    K = task["K"]; Xe, ye, Le, SEQe, Se = task["test_deeper"]
    ok = tot = 0
    for n in range(len(Le)):
        L = int(Le[n])
        if L < 2:
            continue
        tot += 1
        _, o_last = divmod(int(SEQe[n][L - 1]), K)
        ok += int(o_last == int(Se[n, L - 1]))
    return ok / max(tot, 1)


def run_seed(seed, K=6, n_hid=192, epochs=60):
    task = make_centering_task(seed, K=K, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    da = discrete_attractor_rnn(task, seed=seed, epochs=epochs, n_hid=n_hid, temperature=0.7)
    rec = recency_floor(task)
    return {"seed": seed, "K": K, "D3_Cb_track": round(da["state_deeper"], 3), "step_delta": round(da["step_transition_acc"], 3),
            "recency_floor": round(rec, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 CENTERING focus] K={a.K} | the discrete-attractor tracks the discourse CENTER (Cb) over SVO utterances (Centering Theory), vs the recency baseline", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, K=a.K, n_hid=a.n_hid, epochs=a.epochs)
        rows.append(r)
        print(f"  [seed {s}] D3 Cb-track DEEPER={r['D3_Cb_track']} (step-delta={r['step_delta']}) | RECENCY(last-object) floor={r['recency_floor']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        tr, rec = _m("D3_Cb_track"), _m("recency_floor")
        chance = 1.0 / a.K
        go = (tr > 0.75) and (rec < 0.4)
        print(f"\n  AGGREGATE (K={a.K}, chance {chance:.3f}): D3 Cb-track DEEPER={tr:.3f} | RECENCY(last-object) floor={rec:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the discrete-attractor tracks the discourse CENTER (Cb, Centering Theory) over SVO utterances to held-out-DEEPER lengths ('+format(tr,'.2f')+') where the RECENCY (last-object) baseline FAILS at chance ('+format(rec,'.2f')+') -> D3 models the AGENT s actual discourse focus (Centering over SVO) = the foundation for the production MultiTurnAgent wire-in (D3 s composed Cb replaces the host content_bias_target)' if go else 'the Centering Cb tracking did not clearly GO (read D3 vs recency; tune epochs/n_hid/p_continue)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
