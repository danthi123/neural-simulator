"""Generator-F pre-registered MULTI-SEED capability gate. Decides
whether a self-contained small Transformer LM (sim.tiny_transformer
.TinyGPT, trained on the authorized public corpus) clears the SAME
unmodified HARDENED gate_core that 9 spiking/order-blind/statistical
mechanisms failed. Anti-cheat: the SAME HARDENED gate_core (FIXED bars
0.20/1.5/0.5/0.20 + abs-competence floor; >=3 seeds; NEVER tuned
here); load-bearing control = BPE-invariant WORD-SHUFFLE of the train
split (identical tokenizer/token distribution, ONLY sequence order
destroyed). A small Transformer on a bounded corpus CAN memorize ->
the hardened verbatim-copy + generalization bars are load-bearing.
HONEST CEILING: small-Transformer TinyStories-class coherent
SIMPLE-STORY generation, explicitly NOT an LLM (never spun). Kill-safe
resume. Honest propagation is the CONTROLLER's post-run job; main()
ONLY computes+prints+writes JSON (exit 0 == verdict computed; exit 2
== not runnable). ASCII only."""
from __future__ import annotations
import argparse


def _word_shuffle(text, rng):
    """BPE-invariant faithful control: same words (-> same word
    frequencies -> identical BPE merges -> identical tokenizer/token
    distribution), sequence order destroyed."""
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def _heldout_nll(model, tok, text, block_size, device,
                 max_positions):
    """Teacher-forced per-window mean next-token CE over held-out ids.
    SAME logits semantics as training. Returns list[float]."""
    import torch
    import torch.nn.functional as F
    ids = tok.encode(text)
    V = tok.vocab_size
    n = len(ids)
    out = []
    if n < block_size + 2:
        return out
    n_windows = (n - 1) // block_size
    step_w = max(1, n_windows // max(1, int(max_positions)))
    model.eval()
    with torch.no_grad():
        wi = 0
        for w in range(0, n_windows, step_w):
            s = w * block_size
            x = torch.tensor(ids[s:s + block_size],
                             dtype=torch.long,
                             device=device)[None]
            y = torch.tensor(ids[s + 1:s + 1 + block_size],
                             dtype=torch.long,
                             device=device)[None]
            lg = model(x)
            ce = F.cross_entropy(lg.reshape(-1, V),
                                 y.reshape(-1))
            out.append(float(ce))
            wi += 1
            if wi >= int(max_positions):
                break
    return out


def _generate(model, tok, prompt_ids, n_tokens, block_size,
              device, seed):
    """Autoregressive multinomial sampling (temperature 1.0),
    seeded/reproducible. Returns ONLY the generated id list."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    seq = list(prompt_ids) if prompt_ids else [0]
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(n_tokens)):
            ctx = seq[-block_size:]
            x = torch.tensor(ctx, dtype=torch.long,
                             device=device)[None]
            lg = model(x)[0, -1].float().cpu()
            p = torch.softmax(lg, dim=-1)
            nxt = int(torch.multinomial(p, 1, generator=g).item())
            seq.append(nxt)
            out.append(nxt)
    return out


def main():
    import json
    import time
    from pathlib import Path
    import numpy as np

    from research.runners.corpus_fetch import (
        fetch_corpus, split_corpus)
    from research.runners.tiny_transformer_train import (
        train_tiny_gpt)
    from research.runners.subword_lm_gate_core import (
        perplexity, distinct_ngram_ratio, verbatim_copy_fraction,
        gs_verdict, gs_aggregate_multiseed,
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--corpus", type=str, default="tinystories")
    ap.add_argument("--max-corpus-mb", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=200)
    ap.add_argument("--eval-positions", type=int, default=2000)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_f_gate.json")
    ap.add_argument("--ckpt", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_f_gate.ckpt")
    a = ap.parse_args()

    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]

    print("=" * 64, flush=True)
    print("GENERATOR-F PRE-REGISTERED MULTI-SEED CAPABILITY GATE",
          flush=True)
    print("(self-contained small Transformer LM on the authorized "
          "public corpus;", flush=True)
    print(" SAME HARDENED gate_core 0.20/1.5/0.5/0.20 + "
          "abs-competence floor NEVER tuned here;", flush=True)
    print(" load-bearing control = BPE-invariant word-shuffle; >= 3 "
          "seeds;", flush=True)
    print(" HONEST CEILING: small-Transformer TinyStories-class, "
          "explicitly NOT an LLM)", flush=True)
    print("=" * 64, flush=True)

    if len(seeds) < 3:
        print("[NOT RUNNABLE] %d seed(s); >= 3 MANDATORY (single-seed "
              "is NOT a pass -- gate_core enforces, this is the early "
              "exit)." % len(seeds), flush=True)
        return 2

    resume_path = str(a.ckpt) + ".resume.json"
    completed = {}
    if Path(resume_path).exists():
        try:
            completed = {int(k): v for k, v in json.loads(
                Path(resume_path).read_text("utf-8")).get(
                "completed", {}).items()}
            if completed:
                print("[RESUME] %d seed(s) done: %s (skipped)"
                      % (len(completed), sorted(completed)),
                      flush=True)
        except (ValueError, OSError):
            completed = {}

    cinfo = fetch_corpus(name=a.corpus,
                         max_bytes=int(a.max_corpus_mb) * 1_000_000)
    train_text, heldout_text = split_corpus(cinfo["text"],
                                            heldout_frac=0.1)
    print("[corpus] used=%s degraded=%s n_chars=%d train=%d "
          "heldout=%d"
          % (cinfo["corpus_used"], cinfo["degraded"],
             cinfo["n_chars"], len(train_text), len(heldout_text)),
          flush=True)

    cdir = Path(a.ckpt).parent
    cdir.mkdir(parents=True, exist_ok=True)
    tr_file = str(cdir / "gen_f_train.txt")
    Path(tr_file).write_text(train_text, encoding="utf-8")

    per_seed_verdicts = []
    per_seed_records = []
    vocab_size_actual = None
    t0 = time.time()

    def _flush_resume(comp):
        tmp = resume_path + ".tmp"
        Path(tmp).write_text(json.dumps(
            {"completed": {str(k): v for k, v in comp.items()},
             "seeds": seeds}), encoding="utf-8")
        import os
        os.replace(tmp, resume_path)

    for seed in seeds:
        if seed in completed:
            v = completed[seed]
            per_seed_verdicts.append(v)
            per_seed_records.append({"seed": seed, "resumed": True,
                                     "verdict": v})
            print("[SEED %d] RESUMED (verdict reused)" % seed,
                  flush=True)
            continue

        print("\n" + "-" * 64 + "\n[SEED %d]" % seed + "\n"
              + "-" * 64, flush=True)
        rr = train_tiny_gpt(
            seed=seed, corpus_path=tr_file,
            vocab_size=a.vocab_size, d_model=a.d_model,
            n_layer=a.n_layer, n_head=a.n_head,
            block_size=a.block_size, steps=a.steps,
            batch_size=a.batch_size,
            ckpt_path="%s.s%d.real" % (a.ckpt, seed),
            bpe_path="%s.s%d.real.bpe.json" % (a.ckpt, seed),
            device=a.device, verbose=False)
        ctl_file = str(cdir / ("gen_f_ctl_s%d.txt" % seed))
        Path(ctl_file).write_text(
            _word_shuffle(train_text,
                          np.random.default_rng(seed * 911 + 1)),
            encoding="utf-8")
        cr = train_tiny_gpt(
            seed=seed, corpus_path=ctl_file,
            vocab_size=a.vocab_size, d_model=a.d_model,
            n_layer=a.n_layer, n_head=a.n_head,
            block_size=a.block_size, steps=a.steps,
            batch_size=a.batch_size,
            ckpt_path="%s.s%d.ctl" % (a.ckpt, seed),
            bpe_path="%s.s%d.ctl.bpe.json" % (a.ckpt, seed),
            device=a.device, verbose=False)

        rtok, rmodel = rr["_tok"], rr["_model"]
        ctok, cmodel = cr["_tok"], cr["_model"]
        dev = rr["device"]
        V = rtok.vocab_size
        vocab_size_actual = V
        ho_ppl = perplexity(_heldout_nll(
            rmodel, rtok, heldout_text, a.block_size, dev,
            a.eval_positions))
        ctl_ppl = perplexity(_heldout_nll(
            cmodel, ctok, heldout_text, a.block_size, dev,
            a.eval_positions))
        tr_ppl = perplexity(_heldout_nll(
            rmodel, rtok, train_text[:len(heldout_text)],
            a.block_size, dev, a.eval_positions))

        prompt_ids = rtok.encode(
            " ".join(heldout_text.split()[:8]))
        gen_ids = _generate(rmodel, rtok, prompt_ids,
                            a.gen_tokens, a.block_size, dev,
                            seed * 13 + 5)
        tr_ids = rtok.encode(train_text)
        distinct = distinct_ngram_ratio(gen_ids, n=3)
        copy_frac = verbatim_copy_fraction(gen_ids, tr_ids, n=8)

        v = gs_verdict(heldout_ppl=ho_ppl, shuffled_ppl=ctl_ppl,
                       train_ppl=tr_ppl, distinct=distinct,
                       copy_frac=copy_frac,
                       has_shuffled_control=True,
                       uniform_ppl=V)
        v["seed"] = seed
        per_seed_verdicts.append(v)
        per_seed_records.append({
            "seed": seed, "resumed": False,
            "heldout_ppl": ho_ppl, "shuffled_ctl_ppl": ctl_ppl,
            "train_ppl": tr_ppl, "uniform_ppl": V,
            "distinct_trigram": distinct,
            "verbatim_copy_frac": copy_frac,
            "real_final_loss": rr.get("final_loss"),
            "ctl_final_loss": cr.get("final_loss"),
            "gen_sample": rtok.decode(gen_ids)[:300],
            "verdict": v})
        completed[seed] = v
        _flush_resume(completed)
        print("[SEED %d] ho_ppl=%.3f ctl_ppl=%.3f tr_ppl=%.3f "
              "uni=%d distinct=%.3f copy=%.3f -> %s"
              % (seed, ho_ppl, ctl_ppl, tr_ppl, V, distinct,
                 copy_frac, v["GATE"]), flush=True)

    agg = gs_aggregate_multiseed(per_seed_verdicts)
    result = {
        "task": "Generator-F pre-registered MULTI-SEED capability "
                "gate",
        "mechanism": ("self-contained small Transformer LM "
                      "(TinyGPT) trained on the authorized public "
                      "corpus; honest ceiling: small-Transformer "
                      "TinyStories-class, NOT an LLM"),
        "corpus_used": cinfo["corpus_used"],
        "corpus_degraded": cinfo["degraded"],
        "seeds": seeds, "n_seeds": len(seeds),
        # config.vocab_size = ACTUAL realized tokenizer vocab (BPE prepends <UNK> -> requested+1); uniform_ppl MUST stay this real vocab for the HARDENED competence floor; vocab_size_requested = the CLI request. (Generator-E-precedented reconciliation; the grounding pin asserts uniform_ppl == config.vocab_size.)
        "config": {"vocab_size": (vocab_size_actual
                                  if vocab_size_actual is not None
                                  else a.vocab_size),
                   "vocab_size_requested": a.vocab_size,
                   "d_model": a.d_model, "n_layer": a.n_layer,
                   "n_head": a.n_head, "block_size": a.block_size,
                   "steps": a.steps, "batch_size": a.batch_size,
                   "gen_tokens": a.gen_tokens,
                   "eval_positions": a.eval_positions},
        "anti_cheat": {
            "hardened_gate_core": "0.20/1.5/0.5/0.20 + "
                                   "abs-competence floor; >=3 "
                                   "seeds; NEVER tuned; uniform_ppl "
                                   "passed",
            "load_bearing_control": "BPE-invariant word-shuffle of "
                                     "the train split",
            "memorization_caught_by": "hardened verbatim-copy + "
                                       "generalization + word-"
                                       "shuffle bars + mandatory "
                                       "smell-test",
            "honest_ceiling": "small-Transformer TinyStories-class "
                              "coherent SIMPLE-STORY, NOT an LLM; "
                              "never spun",
            "honest_propagation": "CONTROLLER's post-run job; "
                                   "runner only computes+writes "
                                   "JSON"},
        "per_seed": per_seed_records,
        "aggregate_verdict": agg,
        "GATE": agg["GATE"],
        "OVERALL": "PASS" if agg["GATE"] == "PASS" else "FAIL",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8")

    print("\n" + "=" * 64, flush=True)
    print("GENERATOR-F GATE VERDICT", flush=True)
    print("=" * 64, flush=True)
    for r in per_seed_records:
        vv = r["verdict"]
        print("  seed %s: %s (ho_ppl=%s ctl_ppl=%s tr_ppl=%s "
              "uni=%s distinct=%s copy=%s)"
              % (r["seed"], vv["GATE"],
                 r.get("heldout_ppl"), r.get("shuffled_ctl_ppl"),
                 r.get("train_ppl"), r.get("uniform_ppl"),
                 r.get("distinct_trigram"),
                 r.get("verbatim_copy_frac")), flush=True)
    print("  AGGREGATE: %s (n_seeds=%d n_pass=%d; >=3 mandatory; "
          "HARDENED bars untouched)"
          % (agg["GATE"], agg["n_seeds"], agg["n_pass"]),
          flush=True)
    if agg["GATE"] != "PASS":
        print("  NOTE: a maxed FAIL is an HONEST finding -> "
              "propagate (terminal decision-relevant); do NOT "
              "config-crank.", flush=True)
    else:
        print("  NOTE: a PASS is reported STRICTLY at the honest "
              "ceiling (small-Transformer TinyStories-class, NOT "
              "an LLM); controller smell-tests "
              "regurgitation+coherence before propagating.",
              flush=True)
    print("  -> %s" % a.out, flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
