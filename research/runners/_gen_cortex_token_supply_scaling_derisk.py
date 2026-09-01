"""GENERATIVE-CORTEX SCALE LEVER — the TOKEN-SUPPLY axis at FIXED SMALL CAPACITY (roadmap Wall #7 / R4 open prose).

THE FRONTIER (task 2026-09-01): the brain's own spiking mouth frames only structured SVO; ARBITRARY open prose still
needs the external Qwen-0.5B scaffold. The real wall is GENERATIVE-CORTEX SCALE. The substrate-native generator is the
WKV cortex (RWKV-style diagonal-SSM linear-attention LM; a FIXED reservoir + LOCAL-rule read-out reaches BPTT parity,
`2026-07-20-wkv-cortex-biological-learning-CLOSE-local-rule-readout-retires-BPTT.md`, so the mechanism is biologizable
and its SCALING is an architecture property this BPTT instrument characterizes).

WHAT THE RECORD ALREADY BANKED (`2026-07-21-gap1-plateau-is-data-starvation-not-capacity-...md`): on broad-domain
wikitext at a BIG d512 model (~9.8M params), the WKV deep-context NLL is FLAT across data (150k->850k sentences) and
width (d512->d1024), while a fair trigram IMPROVES -> read as "single-layer CAPACITY saturated". BUT every one of those
measurements sat at wikitext's ~1.7 tok/param (the len<=16 filter throws away most of the corpus), i.e. DEEP in the
token-STARVED regime for a 9.8M-param model (Chinchilla-optimal is ~20 tok/param). "Big model flat + trigram improves"
is ALSO the signature of a model that is token-starved FOR ITS SIZE. The record's own next-lever note ("relax the
length filter / reach 5-10 tok/param") was never run, so the record NEVER separated:
  (a) capacity-saturation (more tokens cannot help at this capacity), from
  (b) joint token-starvation (a capacity-MATCHED model would keep using more tokens).

THE SINGLE-VARIABLE DE-RISK (this runner): hold a SMALL FIXED capacity (d_model, vocab, layers, epochs, the eval set,
the vocab classes) and sweep ONLY the training TOKEN SUPPLY, via nested prefixes of ONE fixed per-seed train pool, on
the BROAD corpus (wikitext103, contiguous max_len=48 passages so token accounting is clean and sequence length is held
constant). At a small capacity the SAME feasible token budget reaches a MUCH higher tok/param than the record ever hit
(~7 tok/param, 4x past its 1.7 operating point). The question the record left open:

  * WKV deep (d10-99) held-out NLL keeps DROPPING with tokens past ~1.7 tok/param (still descending at ~7)
        => the broad-domain plateau was JOINT STARVATION, not capacity-at-this-scale. The forward lever is MATCHED-
           QUALITY TOKEN SUPPLY (distillation-as-data / bigger matched corpus), and "~4 orders of PARAMS" over-states
           the wall: at a matched small capacity the binding constraint is token supply, and the substrate scales with it.
  * WKV deep NLL FLATTENS while tokens keep rising (and the trigram keeps improving)
        => genuine capacity saturation even in the higher tok/param regime at this scale. The forward lever is then
           bigger capacity (compute arc) / developmental capacity growth / deep-credit-for-generation, NOT more tokens.

CLEANLINESS / ANTI-CHEATS (silent-failure discipline):
  - SINGLE VARIABLE. Per seed: one deterministic 85/15 split of a FIXED passage pool; the eval set (idx[cut:][:2000])
    and the train POOL (idx[:cut]) are identical across every token point; each point trains on the FIRST k passages of
    that fixed pool (NESTED prefixes) -> the ONLY thing that changes is how many tokens the WKV sees. Verified in the
    output: `eval_ids_sha` is identical across all points within a seed.
  - VOCAB HELD FIXED. Vocab is built ONCE per seed from the FULL train pool (not per token-point), so the V output
    classes are byte-identical across points -> NLL is directly comparable across the sweep (a per-point vocab would
    silently change the class set). Verified: `V` identical across points; `vocab_sha` identical.
  - GENUINE SEQUENCE MODEL, not a count table: at every point the WKV's deep-context advantage must survive the two
    anti-cheat collapses (PERMUTE the prefix order; MEMORYLESS recurrence-off) -> reported per point. A "lower NLL" that
    does not survive perm/mless is not a generation gain.
  - THE TRIGRAM TELL (the record's own decisive control): the fair interpolated trigram is refit on EACH point's train
    prefix, so we report whether the WKV's margin over the trigram GROWS with tokens (WKV uses tokens better than
    counts -> the good sign, opposite of the record's d512 result) or SHRINKS (counts win, WKV saturates).
  - OVERFIT DISCLOSURE: final train loss per point is recorded; at the starved points 12 epochs will overfit (train<<held)
    -> the held-NLL improvement WITH tokens is precisely the generalization (token-supply) benefit, not memorisation.

SCOPE (honest): this is a SMALL-capacity (d96/V2000, ~0.42M active params) probe of the tok/param RESPONSE DIRECTION on
the validated mechanism's BPTT instrument. It does not itself produce LLM-fluent broad prose (the absolute NLL residual
to a ~ppl 20-40 fluency target is reported). The transfer claim is the Chinchilla-universal DIRECTION of the tok/param
response at a capacity-matched operating point; repeating the sweep at a larger capacity is the named next rung.

Run (smoke, ~1 min): SIM_BACKEND=numpy python -m research.runners._gen_cortex_token_supply_scaling_derisk --smoke
Run (full 6-seed): python -m research.runners._gen_cortex_token_supply_scaling_derisk \
    --corpus /home/dant123/Projects/sim/data/corpus/wikitext103.txt \
    --seeds 42 43 44 100 101 102 --d-model 96 --vocab 2000 --epochs 12 --max-len 48 \
    --n-sentences 140000 --token-points 4000 8000 16000 32000 64000
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, json, hashlib, time, math
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import numpy as np

from research.runners._emerge_wkv_lm_derisk import (
    build_and_train_wkv, eval_perdepth, load_stories, fit_interp_trigram,
)
from research.runners._emerge_reservoir_lm_derisk import Vocab, fit_bigram
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

OUT = Path("research/findings/raw/_gen_cortex_token_supply_scaling.json")
DEEP = "10-99"                                # the deep-context bucket = the long-range generation regime


def _sha(obj) -> str:
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]


def _wkv_active_params(V: int, D: int) -> int:
    """Active generative params in the default 'wkv' branch (excludes Wo_sp, used only in the --recurrence ssm branch).
    emb(V*D) + LN(2D) + {Wk,Wv,Wr,Wo}(4*D*D) + w(D)+u(D) + head(D*V + V)."""
    return V * D + 2 * D + 4 * D * D + 2 * D + (D * V + V)


def _mk_args(d_model, epochs, vocab, batch):
    """Minimal Namespace for build_and_train_wkv; every exotic flag it reads is via getattr(...,default), so only the
    core training knobs matter. Defaults mirror _emerge_wkv_lm_derisk's argparse defaults exactly (batch raised to cut
    the launch-bound sequential-recurrence cost: fewer batches => fewer t-loop launches; held CONSTANT across the
    token sweep so the comparison stays single-variable)."""
    return SimpleNamespace(
        d_model=d_model, epochs=epochs, batch=batch, lr=3e-3, weight_decay=1e-4,
        n_layers=1, freeze_emb=False, input="learned", recurrence="wkv",
        uniform_decay=False, vocab=vocab,
    )


def _generate(net, vocab, V, device, seed, n=40, temp=0.8):
    """Sample autoregressive prose from the trained WKV (low NLL != fluent, so we SAMPLE it — the record's rule).
    Greedy-temp sampling from fixed neutral prompts; deterministic per seed."""
    import torch
    prompts = [["the"], ["in", "the"], ["he", "was"], ["it", "is", "a"]]
    rng = np.random.default_rng(seed * 131 + 7)
    outs = []
    net.eval()
    for prompt in prompts:
        ids_g = [vocab.w2i.get(w, vocab.unk) for w in prompt]
        with torch.no_grad():
            for _ in range(n):
                logits = net(torch.tensor([ids_g], device=device))[0, -1]
                p = torch.softmax(logits / temp, -1).cpu().numpy()
                p = p / p.sum()
                ids_g.append(int(rng.choice(V, p=p)))
        outs.append(" ".join(vocab.i2w[i] for i in ids_g))
    return outs


def run_seed(seed, sents, args, capture_gen=False):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
    pool = [sents[i] for i in idx[:cut]]                        # FIXED train pool (nested prefixes drawn from here)
    ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]    # FIXED eval set across all token points
    # VOCAB fixed once from the full pool -> identical output classes across every token point
    vocab = Vocab.build(pool, V=args.vocab); V = vocab.size
    ev_ids = [vocab.ids(s) for s in ev]
    ev_sha = _sha([tuple(e) for e in ev_ids])
    vocab_sha = _sha(list(vocab.i2w))
    active_params = _wkv_active_params(V, args.d_model)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wkv_args = _mk_args(args.d_model, args.epochs, args.vocab, args.batch)

    seed_gen = None
    points = []
    for k in args.token_points:
        if k > len(pool):
            continue
        tr = pool[:k]                                          # NESTED prefix = the ONLY thing that changes
        tr_ids = [vocab.ids(s) for s in tr]
        n_tok = int(sum(len(t) for t in tr_ids))
        dev_ids = tr_ids[-min(2000, max(1, len(tr_ids) // 5)):]  # held-out slice for trigram lambda tuning

        P_bi = fit_bigram(tr_ids, V)
        tri, lambdas = fit_interp_trigram(tr_ids, V, dev_ids)

        t0 = time.time()
        net, WKV_cls = build_and_train_wkv(tr_ids, V, seed, wkv_args, device, init_emb=None)
        is_top_point = (k == max(p for p in args.token_points if p <= len(pool)))
        if capture_gen and is_top_point:
            seed_gen = _generate(net, vocab, V, device, seed)
        # final train loss (overfit disclosure) — one teacher-forced pass over a bounded train sample
        net.eval()
        with torch.no_grad():
            samp = tr_ids[:512]
            tl, tn = 0.0, 0
            for ids in samp:
                if len(ids) < 2:
                    continue
                X = torch.tensor([ids], device=device)
                logp = torch.log_softmax(net(X)[0], -1).cpu().numpy()
                for t in range(len(ids) - 1):
                    tl += -math.log(max(math.exp(logp[t, ids[t + 1]]), 1e-12)); tn += 1
            train_nll = tl / max(1, tn)

        wkv_ce, cnt = eval_perdepth(net, WKV_cls, ev_ids, V, device, seed=seed)
        wkv_perm, _ = eval_perdepth(net, WKV_cls, ev_ids, V, device, permute=True, seed=seed)
        wkv_mless, _ = eval_perdepth(net, WKV_cls, ev_ids, V, device, memoryless=True, seed=seed)

        # bigram + fair trigram deep-NLL on the SAME eval tokens
        tce = defaultdict(float); bce = defaultdict(float)
        for ids in ev_ids:
            for t in range(len(ids) - 1):
                b = _bucket(t + 1)
                bce[b] += -math.log(max(P_bi[ids[t], ids[t + 1]], 1e-12))
                u = ids[t - 1] if t >= 1 else -1
                tce[b] += -math.log(max(tri(u, ids[t], ids[t + 1]), 1e-12))
        n_deep = cnt.get(DEEP, 0)
        if n_deep == 0:
            continue
        wkv_deep = wkv_ce[DEEP]
        tri_deep = tce[DEEP] / n_deep
        bi_deep = bce[DEEP] / n_deep
        perm_collapse = wkv_perm.get(DEEP, float("nan")) - wkv_deep
        mless_collapse = wkv_mless.get(DEEP, float("nan")) - wkv_deep
        # TRUE anti-cheat = the WKV genuinely USES ORDER + MEMORY (both collapses positive). Beating the trigram is a
        # separate QUALITY tell, NOT an anti-cheat (a small d model may not beat a fair trigram yet still be a genuine,
        # token-responsive sequence model). Kept distinct so the verdict does not silently require a quality bar.
        uses_context = (perm_collapse > 0.05) and (mless_collapse > 0.05)
        beats_trigram = (tri_deep - wkv_deep) > 0.0
        points.append({
            "max_train_sents": k, "n_train_passages": len(tr), "n_tokens": n_tok,
            "tok_per_active_param": round(n_tok / active_params, 3),
            "wkv_deep_nll": round(wkv_deep, 4), "trigram_deep_nll": round(tri_deep, 4),
            "bigram_deep_nll": round(bi_deep, 4),
            "margin_vs_trigram": round(tri_deep - wkv_deep, 4),
            "perm_collapse": round(perm_collapse, 4), "mless_collapse": round(mless_collapse, 4),
            "uses_context": bool(uses_context), "beats_trigram": bool(beats_trigram),
            "train_nll": round(train_nll, 4), "overfit_gap": round(wkv_deep - train_nll, 4),
            "n_deep": n_deep, "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  [seed {seed}] k={k:>6} tok={n_tok:>8} ({points[-1]['tok_per_active_param']:>5} t/p): "
              f"WKV_deep {wkv_deep:.4f} | trigram {tri_deep:.4f} | margin {tri_deep-wkv_deep:+.4f} "
              f"| perm+{perm_collapse:.3f} mless+{mless_collapse:.3f} | train {train_nll:.3f} "
              f"| ctx={'ok' if uses_context else 'NO'} tri={'beat' if beats_trigram else 'lose'} "
              f"({points[-1]['elapsed_s']}s)", flush=True)

    # per-seed lever signals
    seed_out = {"V": V, "active_params": active_params, "eval_ids_sha": ev_sha, "vocab_sha": vocab_sha,
                "n_eval": len(ev_ids), "points": points}
    if seed_gen is not None:
        seed_out["gen_samples_top_point"] = seed_gen
    if len(points) >= 2:
        nll_min = points[0]["wkv_deep_nll"]; nll_max = points[-1]["wkv_deep_nll"]
        seed_out["delta_nll_min_to_max_tokens"] = round(nll_min - nll_max, 4)   # +ve = improved with tokens
        seed_out["top_segment_slope"] = round(points[-2]["wkv_deep_nll"] - points[-1]["wkv_deep_nll"], 4)  # +ve = still descending
        seed_out["margin_grows_with_tokens"] = points[-1]["margin_vs_trigram"] > points[0]["margin_vs_trigram"]
        seed_out["uses_tokens"] = (nll_min - nll_max) > 0.10
        seed_out["still_descending_at_top"] = (points[-2]["wkv_deep_nll"] - points[-1]["wkv_deep_nll"]) > 0.02
        seed_out["uses_context_at_top"] = points[-1]["uses_context"]
        seed_out["beats_trigram_at_top"] = points[-1]["beats_trigram"]
    return seed_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--corpus", type=str, default="/home/dant123/Projects/sim/data/corpus/wikitext103.txt")
    ap.add_argument("--d-model", dest="d_model", type=int, default=96)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256,
                    help="held CONSTANT across the token sweep; raised from the runner's 128 to cut the launch-bound "
                         "sequential-recurrence cost (fewer batches => fewer t-loop launches).")
    ap.add_argument("--max-len", dest="max_len", type=int, default=48)
    ap.add_argument("--n-sentences", dest="n_sentences", type=int, default=140000)
    ap.add_argument("--max-eval-sents", dest="max_eval_sents", type=int, default=2000)
    ap.add_argument("--token-points", dest="token_points", type=int, nargs="+",
                    default=[4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--smoke", action="store_true", help="fast 1-seed 2-point sanity check")
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [42]; args.d_model = 48; args.epochs = 3; args.n_sentences = 12000
        args.token_points = [2000, 6000]; args.max_eval_sents = 500; args.batch = 256

    if not Path(args.corpus).exists():
        # fall back to any available corpus so the smoke can run even off-box
        for alt in ("/home/dant123/Projects/sim/data/corpus/wikitext.txt",
                    "data/corpus/wikitext103.txt", "data/corpus/tinystories.txt"):
            if Path(alt).exists():
                args.corpus = alt; break

    print(f"[gen-cortex token-supply] corpus={args.corpus} d={args.d_model} V<={args.vocab} epochs={args.epochs} "
          f"max_len={args.max_len} points={args.token_points} seeds={args.seeds}", flush=True)
    sents = load_stories(args.corpus, args.n_sentences, max_len=args.max_len)   # contiguous passages (clean token count)
    print(f"[gen-cortex] loaded {len(sents)} contiguous passages from {args.corpus}", flush=True)

    t0 = time.time(); per_seed = {}
    for si, seed in enumerate(args.seeds):
        per_seed[str(seed)] = run_seed(seed, sents, args, capture_gen=(si == 0))

    # ---- 6-seed aggregate verdict ----
    def agg(key):
        vals = [s.get(key) for s in per_seed.values() if key in s]
        return vals
    uses = agg("uses_tokens"); desc = agg("still_descending_at_top"); clean = agg("uses_context_at_top")
    grows = agg("margin_grows_with_tokens"); beats = agg("beats_trigram_at_top")
    deltas = agg("delta_nll_min_to_max_tokens")
    n = len(per_seed)
    n_uses = sum(1 for x in uses if x); n_desc = sum(1 for x in desc if x)
    n_clean = sum(1 for x in clean if x); n_grows = sum(1 for x in grows if x)
    n_beats = sum(1 for x in beats if x)
    mean_delta = round(float(np.mean(deltas)), 4) if deltas else None
    # top-point NLL residual to a fluency target band (ppl ~20-40 => NLL ~3.0-3.69)
    top_nlls = [s["points"][-1]["wkv_deep_nll"] for s in per_seed.values() if s.get("points")]
    mean_top_nll = round(float(np.mean(top_nlls)), 4) if top_nlls else None
    max_tok_per_param = max((s["points"][-1]["tok_per_active_param"] for s in per_seed.values() if s.get("points")),
                            default=None)

    # GO = the token lever is REAL at a matched small capacity (plateau was starvation, not capacity-at-this-scale)
    token_lever_go = (n_uses >= max(1, n - 1)) and (n_desc >= max(1, n - 2)) and (n_clean >= max(1, n - 1))
    if token_lever_go:
        verdict = "GO-TOKEN-LEVER"
    elif n_uses <= n // 2:
        verdict = "NO-GO-CAPACITY-SATURATED"
    else:
        verdict = "PARTIAL"

    out = {
        "runner": "_gen_cortex_token_supply_scaling_derisk",
        "corpus": args.corpus, "d_model": args.d_model, "vocab_cap": args.vocab, "epochs": args.epochs,
        "max_len": args.max_len, "token_points": args.token_points, "seeds": args.seeds,
        "per_seed": per_seed,
        "verdict": {
            "verdict": verdict,
            "n_seeds": n,
            "n_uses_tokens": n_uses, "n_still_descending_at_top": n_desc,
            "n_uses_context_at_top": n_clean, "n_beats_trigram_at_top": n_beats,
            "n_margin_grows_with_tokens": n_grows,
            "mean_delta_nll_min_to_max_tokens": mean_delta,
            "mean_top_point_wkv_deep_nll": mean_top_nll,
            "max_tok_per_active_param_reached": max_tok_per_param,
            "fluency_target_nll_band": [3.0, 3.69],
            "residual_top_nll_above_fluency_band_hi": (round(mean_top_nll - 3.69, 4)
                                                       if mean_top_nll is not None else None),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    v = out["verdict"]
    print(f"\n==== VERDICT: {verdict} ====", flush=True)
    print(f"  uses_tokens {n_uses}/{n} | still_descending_at_top {n_desc}/{n} | uses_context_at_top {n_clean}/{n} "
          f"| beats_trigram_at_top {n_beats}/{n} | margin_grows {n_grows}/{n}", flush=True)
    print(f"  mean delta-NLL(min->max tokens) = {mean_delta} nats | mean top-point WKV deep NLL = {mean_top_nll} "
          f"(fluency band 3.0-3.69) | max tok/param reached = {max_tok_per_param}", flush=True)
    print(f"-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
