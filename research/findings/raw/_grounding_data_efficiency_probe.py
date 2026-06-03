"""Cheap-first probe: does GROUNDED (structured/shared sensory features) input make word-learning DATA-EFFICIENT
vs ORTHOGONAL tokenizer symbols? (Owner input-side-fidelity insight, 2026-06-02.)

The sim's language input is GIVEN as orthogonal tokenizer codes (vocab_to_drive_pattern / one-hot): distinct
words -> distinct/orthogonal codes with NO grounding and NO shared structure. So "red apple" and "red ball"
have INDEPENDENT codes -> the feature "red" cannot transfer -> the net must learn each combination from
co-occurrence statistics (data-hungry, the regime we hit). Real sensory grounding gives words SHARED structure
(red things share a redness feature), which should enable few-shot generalization.

Controlled test (the ONLY variable is the input representation; same simple learner for both):
  concepts = (color, object) pairs. GROUNDED code = concat(color_feature, object_feature) -- structured,
  SHARES color_feature across same-color concepts. ORTHOGONAL code = an independent random code per pair (the
  tokenizer regime) -- no sharing. Task: classify the COLOR from the code. Train on K (color,object) pairs;
  test on HELD-OUT pairs whose color AND object were each seen (in OTHER pairs) but THIS combination was not.
  A grounded learner reads the shared color_feature -> generalizes to novel combinations from few pairs; an
  orthogonal learner must memorize each pair -> needs ~all of them.

Decisive read (pre-registered): grounded held-out accuracy reaches >= 0.9 at FAR fewer train pairs than
orthogonal -> grounding is data-efficient; quantify the ratio. Same logistic-regression learner for both
conditions (isolates the INPUT representation, not the learning rule). Stdlib + numpy only; no protected import.

  python -m research.findings.raw._grounding_data_efficiency_probe
"""
from __future__ import annotations
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def train_logreg(X, Y, n_cls, epochs=300, lr=0.2, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((X.shape[1], n_cls)) * 0.01
    for _ in range(epochs):
        p = softmax(X @ W)
        g = X.T @ (p - np.eye(n_cls)[Y]) / len(Y) + l2 * W
        W -= lr * g
    return W


def acc(W, X, Y):
    return float((softmax(X @ W).argmax(1) == Y).mean())


def run(seed, n_colors=6, n_objects=8, Dc=48, Do=48, train_pairs_list=(4, 8, 12, 18, 24, 32, 40)):
    rng = np.random.default_rng(seed)
    color_feat = rng.standard_normal((n_colors, Dc))                      # shared grounded sub-features
    obj_feat = rng.standard_normal((n_objects, Do))
    pairs = [(c, o) for c in range(n_colors) for o in range(n_objects)]
    rng.shuffle(pairs)
    # held-out = last 25% of pairs whose color AND object also appear in the train portion
    n_ho = max(n_colors, len(pairs) // 4)
    ho_pairs = pairs[-n_ho:]; pool = pairs[:-n_ho]
    D = Dc + Do
    # grounded code = concat(color_feat, obj_feat); orthogonal code = independent random per pair
    orth = {p: rng.standard_normal(D) for p in pairs}

    def code(p, grounded):
        c, o = p
        return np.concatenate([color_feat[c], obj_feat[o]]) if grounded else orth[p]

    Xho_g = np.array([code(p, True) for p in ho_pairs]); Xho_o = np.array([code(p, False) for p in ho_pairs])
    Yho = np.array([c for c, o in ho_pairs])
    out = {"grounded": [], "orthogonal": []}
    for K in train_pairs_list:
        # ensure every color appears in the train subset (else its held-out is unlearnable for either)
        tr = pool[:K]
        seen_colors = {c for c, o in tr}
        for c in range(n_colors):
            if c not in seen_colors:
                cand = next((q for q in pool[K:] if q[0] == c), None)
                if cand:
                    tr = tr + [cand]
        for grounded in (True, False):
            Xtr = np.array([code(p, grounded) for p in tr]); Ytr = np.array([c for c, o in tr])
            W = train_logreg(Xtr, Ytr, n_colors, seed=seed)
            Xho = Xho_g if grounded else Xho_o
            out["grounded" if grounded else "orthogonal"].append((len(tr), acc(W, Xho, Yho)))
    return out


def main():
    print("=== grounding data-efficiency: held-out color accuracy vs # train pairs (grounded vs orthogonal) ===",
          flush=True)
    seeds = [42, 43, 44]
    agg = {"grounded": {}, "orthogonal": {}}
    for s in seeds:
        r = run(s)
        for cond in agg:
            for k, a in r[cond]:
                agg[cond].setdefault(k, []).append(a)
    ks = sorted(agg["grounded"])
    print("  #train_pairs | grounded held-out acc | orthogonal held-out acc", flush=True)
    g_thr = o_thr = None
    for k in ks:
        gm = float(np.mean(agg["grounded"][k])); om = float(np.mean(agg["orthogonal"][k]))
        print(f"     {k:>3}        |        {gm:.3f}        |        {om:.3f}", flush=True)
        if g_thr is None and gm >= 0.9:
            g_thr = k
        if o_thr is None and om >= 0.9:
            o_thr = k
    print(f"\n  train pairs to reach >=0.90 held-out: grounded={g_thr}  orthogonal={o_thr}", flush=True)
    if g_thr and (o_thr is None or g_thr < o_thr):
        ratio = (o_thr / g_thr) if o_thr else float("inf")
        print(f"VERDICT: RESOLVES -- GROUNDED (shared sensory structure) generalizes from "
              f"{g_thr} pairs; orthogonal tokens need {o_thr if o_thr else '>'+str(ks[-1])} "
              f"({'%.1fx' % ratio if o_thr else '>'+ '%.1fx'%(ks[-1]/g_thr)} more). Grounding makes word-learning "
              f"DATA-EFFICIENT via feature-sharing/transfer -- the tokenizer's orthogonal symbols force the "
              f"data-hungry regime. Supports raw-transduction + learned grounding as a missing data-efficient "
              f"structure -> design the faithful build (text-as-pixels through the visual pathway).", flush=True)
    else:
        print("VERDICT: grounded did not clearly beat orthogonal -- re-examine the structure/task.", flush=True)


if __name__ == "__main__":
    main()
