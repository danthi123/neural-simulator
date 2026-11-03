# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md (B4 selection benchmark)
"""Which episode does the hippocampus select for a question, and is the told fact then recalled? By key variant.

No sleep of the cortex; hippocampus ON. Per seed, 24 of the 48 facts (facts_bank.py) are stored, 12 are never told
(their questions + NEVER_TOLD are the "should not fire" cues), 12 are the teacher-key training pool (never stored,
never tested). Cues for each stored fact: same-words question (orig), light rewording (para_a), heavy rewording
(para_b). Read-outs per variant:
  sel_<cue>     top-1 selection accuracy among the 24 stored episodes (chance 1/24)
  auroc         top score of told cues vs untold cues (can the familiarity bar separate them; 0.5 = none)
  rec_<cue>     recall: the selected episode's delta reinstated at the hippocampus gain, then the cortex answers the
                question (greedy); hit if the answer word is in the reply. Oracle = correct episode, none = no memory.
Learned keys (brain/hippocampus_keys.py LearnedKey on pool_all), trained per seed OFFLINE by this script (not by
Brain.sleep) on cached read-outs. NOTE: this probe recomputes similarities from Gram matrices and never runs
KeyedHippocampus.recall; the real-path probe is selection_realpath.py.
  learned_selfq    pairs = (stored episode, a question the cortex itself generated about it while replaying it in this
                   script), no teacher text in training (the generation prompt is a declared scaffold)
  learned_shuf     same pairs with questions shuffled across episodes (control: must not beat pool_all)
  learned_teacher  pairs = teacher-written questions/rewordings of the 12 training-pool facts (disjoint facts; scaffold)

  python -m brain.experiments.selection_probe --model <pth> --seeds 42 43 44 100 101 102 --out <json>
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.facts_bank import FACTS48, NEVER_TOLD
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import LearnedKey, read_activity
from brain.plastic_cortex import PlasticCortex

KINDS = ("S", "att", "ffn", "last", "pool")
CUES = ("orig", "para_a", "para_b")
SKETCH = 256  # random-projection dims per layer for the matrix-memory sketch (learnedS_* arms)
# Scaffold (SCAFFOLDS.md): while replaying an episode the cortex predicts the friend's next turn, opened with "What" (a
# question onset): its own guess at a question about the episode. (A direct "quiz me" request gave refusals at 0.1B;
# 8-fact prompt trial 11-03.)
SELFQ_PROMPT = "User: {s}\n\nAssistant: Okay, I'll remember that.\n\nUser: What"


def selfq_text(o):
    """Re-attach the "What" onset (speak() strips the continuation's leading space)."""
    o = o.split("\n")[0].rstrip()
    return "What" + ("" if o.startswith(("'", "?", ",")) else " ") + o


def stmt(s):
    return f"User: {s}\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-store", type=int, default=24)
    ap.add_argument("--n-selfq", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--cache", required=True, help="directory for the per-model activity and answer caches")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-recall", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    os.makedirs(a.cache, exist_ok=True)
    tag = os.path.basename(a.model).split("-")[2]
    c = Cortex(model_path=a.model)
    pc = PlasticCortex(c, rank=1, layers=[])
    pc.enabled = False
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    L = c.n_layers

    # ---- the brain's own questions about each episode (replay + generation), cached per model ----
    selfq_path = os.path.join(a.cache, f"selfq_{tag}.json")
    if os.path.exists(selfq_path):
        selfq = json.load(open(selfq_path))
    else:
        selfq = {}
        for s, *_ in FACTS48:
            lg, st = c.read(BIRTH + SELFQ_PROMPT.format(s=s), None)
            outs = [c.speak(lg, st, max_tokens=24, temperature=0)[0]]
            for k in range(a.n_selfq - 1):
                torch.manual_seed(k)
                outs.append(c.speak(lg, st, max_tokens=24, temperature=1.0, top_p=0.7)[0])
            selfq[s] = [selfq_text(o) for o in outs]
            print("selfq", s, "->", selfq[s], flush=True)
        json.dump(selfq, open(selfq_path, "w"), indent=1)

    # ---- every text the probe needs, read once: per-layer Gram matrices + pooled read vectors ----
    texts = []
    for s, q, pa, pb, _ in FACTS48:
        texts += [stmt(s), stmt(q), stmt(pa), stmt(pb)] + [stmt(x) for x in selfq[s]]
    texts += [stmt(x) for x in NEVER_TOLD]
    texts = list(dict.fromkeys(texts))
    idx = {t: i for i, t in enumerate(texts)}
    act_path = os.path.join(a.cache, f"gram_{tag}.pt")
    if os.path.exists(act_path):
        blob = torch.load(act_path)
    else:
        n = len(texts)
        gram = {k: torch.zeros(L, n, n) for k in KINDS}
        vecs = {k: [] for k in KINDS}
        Sdisk = None
        for j, t in enumerate(texts):
            act = read_activity(pc, base, t)
            for k in ("att", "ffn", "last", "pool"):
                vecs[k].append(torch.stack(act[k]))
            if Sdisk is None:
                d = act["S"][0].numel()
                Sdisk = np.memmap(os.path.join(a.cache, f"S_{tag}.bin"), dtype=np.float16, mode="w+", shape=(n, L, d))
            Sdisk[j] = torch.stack(act["S"]).half().numpy()
            if j % 25 == 0:
                print(f"read {j}/{n} ({time.time() - t0:.0f}s)", flush=True)
        for k in ("att", "ffn", "last", "pool"):
            V = torch.stack(vecs[k])  # n, L, d
            for l in range(L):
                gram[k][l] = V[:, l] @ V[:, l].T
        for l in range(L):
            V = torch.from_numpy(np.asarray(Sdisk[:, l], dtype=np.float32))
            gram["S"][l] = V @ V.T
        blob = {"texts": texts, "gram": gram, "pool": torch.stack(vecs["pool"])}
        torch.save(blob, act_path)
        del Sdisk
        os.remove(os.path.join(a.cache, f"S_{tag}.bin"))
    assert blob["texts"] == texts
    gram, pool = blob["gram"], blob["pool"]
    pool_flat = LearnedKey.flat([pool[:, l] for l in range(L)])  # n, L*C
    # A fixed random projection (sketch) of every layer's normalised matrix-memory delta, SKETCH dims per layer (one
    # random Gaussian matrix shared by all layers, seed 0): keeps S_all's cosines approximately, at a size a learned
    # key can sit on. Biology: a fixed random fan-in, as cortex-to-hippocampus input is often modelled.
    sk_path = os.path.join(a.cache, f"sketch_{tag}.pt")
    if os.path.exists(sk_path):
        sk = torch.load(sk_path)
    else:
        R = None
        rows = []
        for j, t in enumerate(texts):
            Ss = read_activity(pc, base, t)["S"]
            if R is None:
                R = torch.randn(Ss[0].numel(), SKETCH, generator=torch.Generator().manual_seed(0)) / SKETCH ** 0.5
            rows.append(torch.stack([torch.nn.functional.normalize(s, dim=0) @ R for s in Ss]))
            if j % 50 == 0:
                print(f"sketch {j}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        sk = {"texts": texts, "sketch": torch.stack(rows)}
        torch.save(sk, sk_path)
        del R
    assert sk["texts"] == texts
    feats = {"pool": pool_flat, "Ssk": LearnedKey.flat([sk["sketch"][:, l] for l in range(L)])}

    def sims_fixed(variant, stored_rows, query_rows):
        """Centred cosine (mean of stored keys removed) from Gram matrices, exactly as KeyedHippocampus computes it: one
        layer = the raw read-out (S0 = the default Hippocampus key); several layers = each layer's vector normalised,
        concatenated, then centred."""
        kind = next(k for k in KINDS if variant.startswith(k))
        rest = variant[len(kind):]
        layers = range(L) if rest == "_all" else [L // 2] if rest == "_mid" else [int(rest)]
        if len(layers) == 1:
            G = gram[kind][layers[0]]
        else:
            G = 0.0
            for l in layers:
                d = gram[kind][l].diagonal().clamp_min(1e-12).sqrt()
                G = G + gram[kind][l] / d[:, None] / d[None]
        si, qi = torch.tensor(stored_rows), torch.tensor(query_rows)
        Gss, Gqs, Gqq = G[si][:, si], G[qi][:, si], G[qi, qi]
        mm = Gss.mean()
        qmu, smu = Gqs.mean(1, keepdim=True), Gss.mean(1)  # q.mu, k.mu
        num = Gqs - qmu - smu[None] + mm
        qn = (Gqq - 2 * qmu[:, 0] + mm).clamp_min(1e-12).sqrt()
        kn = (Gss.diagonal() - 2 * smu + mm).clamp_min(1e-12).sqrt()
        return num / qn[:, None] / kn[None]

    def sims_learned(lk, X, stored_rows, query_rows):
        with torch.no_grad():
            K = lk(X[stored_rows])
            Q = lk(X[query_rows])
        mu = K.mean(0, keepdim=True)
        return torch.nn.functional.normalize(Q - mu, dim=1) @ torch.nn.functional.normalize(K - mu, dim=1).T

    # ---- recall: answer to a question with one episode reinstated (cache across seeds) ----
    ans_path = os.path.join(a.cache, f"answers_{tag}.json")
    answers = json.load(open(ans_path)) if os.path.exists(ans_path) else {}
    deltas = {}

    def answer(q, s):
        k = f"{q}||{s or ''}"
        if k not in answers:
            st0 = base
            if s is not None:
                if s not in deltas:
                    _, st = c.read(stmt(s), base)
                    deltas[s] = c.matrix_delta(st, base)
                st0 = c.add_delta(base, deltas[s], Hippocampus.GAIN)
            lg, st = c.read(f"User: {q}\n\nAssistant:", st0)
            answers[k] = c.speak(lg, st, max_tokens=a.max_tokens, temperature=0)[0]
        return answers[k]

    fixed = [f"{k}{l}" for k in KINDS for l in range(L)] + [f"{k}_all" for k in KINDS] + [f"{k}_mid" for k in KINDS]
    recall_variants = ["S0", "S_mid", "S_all", "att_all", "ffn_all", "last_all", "pool_mid", "pool_all",
                       "learned_selfq", "learned_shuf", "learned_teacher", "Ssk_all", "learnedS_selfq", "learnedS_shuf",
                       "learnedS_teacher"]
    result = {"selfq_prompt": SELFQ_PROMPT, "model": a.model, "n_layers": L, "n_store": a.n_store, "seeds": {}, "selfq": selfq}
    for seed in a.seeds:
        rng = random.Random(seed)
        order = list(range(48))
        rng.shuffle(order)
        stored, untold, trainpool = order[:a.n_store], order[a.n_store:a.n_store + 12], order[a.n_store + 12:a.n_store + 24]
        srows = [idx[stmt(FACTS48[i][0])] for i in stored]
        cue_rows = {cue: [idx[stmt(FACTS48[i][1 + j])] for i in stored] for j, cue in enumerate(CUES)}
        untold_rows = [idx[stmt(FACTS48[i][1])] for i in untold] + [idx[stmt(x)] for x in NEVER_TOLD]
        # learned keys
        lks = {}
        sq_rows = [[idx[stmt(x)] for x in selfq[FACTS48[i][0]]] for i in stored]
        perm = list(range(len(stored)))
        while any(p == j for j, p in enumerate(perm)):
            rng.shuffle(perm)
        tr_rows = [idx[stmt(FACTS48[i][0])] for i in trainpool]
        tr_pos = [[idx[stmt(x)] for x in FACTS48[i][1:4]] for i in trainpool]
        arms = []
        for fname, prefix in (("pool", "learned"), ("Ssk", "learnedS")):
            X = feats[fname]
            arms += [(f"{prefix}_selfq", X, X[srows], sq_rows), (f"{prefix}_shuf", X, X[srows], [sq_rows[p] for p in perm]),
                     (f"{prefix}_teacher", X, X[tr_rows], tr_pos)]
        for name, X, anc, pos_lists in arms:
            lk = LearnedKey(X.shape[1], seed=seed)
            g = random.Random(seed + 1)
            # one positive per anchor per step, drawn from its cue list (replay varies)
            draws = [X[[g.choice(pl) for pl in pos_lists]] for _ in range(300)]
            torch.manual_seed(seed)
            lk.mu.data.copy_(anc.mean(0))
            opt = torch.optim.Adam([lk.A, lk.B], lr=3e-3, weight_decay=1e-3)
            tgt = torch.arange(anc.shape[0])
            for step in range(300):
                ka = torch.nn.functional.normalize(lk(anc), dim=1)
                kp = torch.nn.functional.normalize(lk(draws[step]), dim=1)
                lg = ka @ kp.T / 0.1
                loss = 0.5 * (torch.nn.functional.cross_entropy(lg, tgt) + torch.nn.functional.cross_entropy(lg.T, tgt))
                opt.zero_grad()
                loss.backward()
                opt.step()
            lks[name] = (lk, X)
        lks["Ssk_all"] = (None, feats["Ssk"])  # the fixed sketch itself (sanity: close to S_all)
        res = {"stored": stored, "untold": untold, "trainpool": trainpool, "variants": {}}
        for v in fixed + list(lks):
            if v == "Ssk_all":
                f = lambda qr: sims_learned(lambda x: x, feats["Ssk"], srows, qr)  # noqa: E731
            elif v in lks:
                f = lambda qr: sims_learned(lks[v][0], lks[v][1], srows, qr)  # noqa: E731
            else:
                f = lambda qr: sims_fixed(v, srows, qr)  # noqa: E731
            r = {}
            told_top = []
            for cue in CUES:
                S = f(cue_rows[cue])
                pred = S.argmax(1)
                r[f"sel_{cue}"] = float((pred == torch.arange(len(stored))).float().mean())
                r[f"pred_{cue}"] = [stored[int(p)] for p in pred]
                told_top += S.max(1).values.tolist()
            un_top = f(untold_rows).max(1).values.tolist()
            r["auroc"] = sum((t > u) + 0.5 * (t == u) for t in told_top for u in un_top) / (len(told_top) * len(un_top))
            res["variants"][v] = r
        if not a.no_recall:
            for v in recall_variants + ["oracle", "none"]:
                r = res["variants"].setdefault(v, {})
                for j, cue in enumerate(CUES):
                    hits = []
                    for n_i, i in enumerate(stored):
                        q, keys = FACTS48[i][1 + j], FACTS48[i][4]
                        if v == "oracle":
                            s = FACTS48[i][0]
                        elif v == "none":
                            s = None
                        else:
                            s = FACTS48[res["variants"][v][f"pred_{cue}"][n_i]][0]
                        hits.append(any(k in answer(q, s).lower() for k in keys))
                    r[f"rec_{cue}"] = sum(hits) / len(hits)
                json.dump(answers, open(ans_path, "w"))
                print(seed, v, {k: round(x, 3) for k, x in r.items() if k.startswith(("sel_", "rec_", "auroc"))}, flush=True)
        result["seeds"][str(seed)] = res
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    result["elapsed_s"] = time.time() - t0
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
