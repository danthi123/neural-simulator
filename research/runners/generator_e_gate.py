"""Generator-E pre-registered MULTI-SEED capability gate. Decides
whether a SELF-CONTAINED n-gram generative LM clears the SAME
unmodified HARDENED gate_core that 9 neural attempts failed. The model
is sim.ngram_teacher.NgramTeacher (reused UNMODIFIED -- it IS the
generative model now). Anti-cheat: the SAME HARDENED gate_core
(FIXED bars 0.20/1.5/0.5/0.20 + absolute-competence floor; >=3 seeds;
NEVER tuned here); the load-bearing control is a WORD-SHUFFLE of the
train split (BPE-invariant -> identical tokenizer/token distribution,
ONLY sequence order destroyed -> the control n-gram has no trigram
structure); an n-gram's chief cheat is REGURGITATION, caught by the
hardened verbatim-copy bar. Honest ceiling (NOT spun): n-gram-class
LOCAL coherence only, explicitly NOT an LLM. Kill-safe resume. Honest
propagation is the CONTROLLER's post-run job; main() ONLY
computes+prints+writes JSON (exit 0 == verdict computed; exit 2 ==
not runnable). NO GPU. ASCII only."""
from __future__ import annotations
import argparse


def _word_shuffle(text, rng):
    """BPE-invariant faithful control: same words (-> same word
    frequencies -> identical BPE merges -> identical tokenizer/token
    distribution), sequence order destroyed."""
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def main():
    import json
    import time
    from pathlib import Path
    import numpy as np

    from research.runners.corpus_fetch import fetch_corpus, split_corpus
    from sim.bpe_tokenizer import BPETokenizer
    from sim.ngram_teacher import NgramTeacher
    from sim.ngram_ppl import ngram_heldout_nll
    from sim.ngram_generate import ngram_generate
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
    ap.add_argument("--gen-tokens", type=int, default=200)
    ap.add_argument("--eval-positions", type=int, default=4000)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_e_gate.json")
    ap.add_argument("--ckpt", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_e_gate.ckpt")
    a = ap.parse_args()

    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]

    print("=" * 64, flush=True)
    print("GENERATOR-E PRE-REGISTERED MULTI-SEED CAPABILITY GATE",
          flush=True)
    print("(SELF-CONTAINED n-gram generative LM through the SAME "
          "HARDENED gate_core;", flush=True)
    print(" FIXED bars 0.20/1.5/0.5/0.20 + abs-competence floor NEVER "
          "tuned here;", flush=True)
    print(" load-bearing control = BPE-invariant word-shuffle; n-gram "
          "cheat = REGURGITATION (verbatim-copy bar); >= 3 seeds;",
          flush=True)
    print(" HONEST CEILING: n-gram-class LOCAL coherence, NOT an LLM)",
          flush=True)
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
                      % (len(completed), sorted(completed)), flush=True)
        except (ValueError, OSError):
            completed = {}

    cinfo = fetch_corpus(name=a.corpus,
                         max_bytes=int(a.max_corpus_mb) * 1_000_000)
    train_text, heldout_text = split_corpus(cinfo["text"],
                                            heldout_frac=0.1)
    print("[corpus] used=%s degraded=%s n_chars=%d train=%d heldout=%d"
          % (cinfo["corpus_used"], cinfo["degraded"],
             cinfo["n_chars"], len(train_text), len(heldout_text)),
          flush=True)

    Path(a.ckpt).parent.mkdir(parents=True, exist_ok=True)

    per_seed_verdicts = []
    per_seed_records = []
    # The BPE tokenizer is deterministic on the (seed-invariant) train
    # split, so the ACTUAL achieved vocab size is identical every seed.
    # We record THIS (not the raw --vocab-size request, which the
    # tokenizer rounds via its <UNK>/merge accounting) so config is
    # internally consistent with uniform_ppl (== the vocab the model
    # actually predicts over). Deviation from the reference's
    # config={"vocab_size": a.vocab_size}: the grounding pin requires
    # per_seed.uniform_ppl == config.vocab_size, and uniform_ppl MUST
    # stay V (HARDENED gate competence floor is over the real vocab);
    # reporting the achieved vocab is the honest reconciliation.
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

        print("\n" + "-" * 64 + "\n[SEED %d]" % seed + "\n" + "-" * 64,
              flush=True)
        # BPE on the train split (deterministic; n-gram counts are
        # deterministic too -- the seed varies the GENERATION sampling
        # rng AND the word-shuffle rng, so multi-seed tests a real
        # generative result holds across sampling seeds).
        tok = BPETokenizer()
        tok.train(train_text, vocab_size=a.vocab_size)
        V = tok.vocab_size
        vocab_size_actual = V
        tr_ids = tok.encode(train_text)
        ho_ids = tok.encode(heldout_text)

        real = NgramTeacher()
        real.train(tr_ids, vocab_size=V)
        # WORD-SHUFFLED control (BPE-invariant; identical vocab/token
        # distribution; trigram structure destroyed).
        ctl_text = _word_shuffle(train_text,
                                 np.random.default_rng(seed * 911 + 1))
        ctl_ids = tok.encode(ctl_text)
        ctl = NgramTeacher()
        ctl.train(ctl_ids, vocab_size=V)

        ho_ppl = perplexity(ngram_heldout_nll(real, ho_ids))
        ctl_ppl = perplexity(ngram_heldout_nll(ctl, ho_ids))
        tr_ppl = perplexity(
            ngram_heldout_nll(real, tr_ids[:len(ho_ids)]))

        prompt_ids = tok.encode(
            " ".join(heldout_text.split()[:8]))
        gen_ids = ngram_generate(
            real, prompt_ids, a.gen_tokens,
            np.random.default_rng(seed * 13 + 5), 1.0)
        distinct = distinct_ngram_ratio(gen_ids, n=3)
        copy_frac = verbatim_copy_fraction(gen_ids, tr_ids, n=8)

        # HARDENED gate_core: MUST pass uniform_ppl (fail-closed
        # without it). The gate judges THIS n-gram generator.
        v = gs_verdict(heldout_ppl=ho_ppl, shuffled_ppl=ctl_ppl,
                       train_ppl=tr_ppl, distinct=distinct,
                       copy_frac=copy_frac, has_shuffled_control=True,
                       uniform_ppl=V)
        v["seed"] = seed
        per_seed_verdicts.append(v)
        per_seed_records.append({
            "seed": seed, "resumed": False,
            "heldout_ppl": ho_ppl, "shuffled_ctl_ppl": ctl_ppl,
            "train_ppl": tr_ppl, "uniform_ppl": V,
            "distinct_trigram": distinct,
            "verbatim_copy_frac": copy_frac,
            "gen_sample": tok.decode(gen_ids)[:240], "verdict": v})
        completed[seed] = v
        _flush_resume(completed)
        print("[SEED %d] ho_ppl=%.3f ctl_ppl=%.3f tr_ppl=%.3f "
              "uni=%d distinct=%.3f copy=%.3f -> %s"
              % (seed, ho_ppl, ctl_ppl, tr_ppl, V, distinct,
                 copy_frac, v["GATE"]), flush=True)

    agg = gs_aggregate_multiseed(per_seed_verdicts)
    result = {
        "task": "Generator-E pre-registered MULTI-SEED capability gate",
        "mechanism": ("self-contained n-gram generative LM "
                      "(NgramTeacher) through the SAME HARDENED "
                      "gate_core; honest ceiling: n-gram-class LOCAL "
                      "coherence, NOT an LLM"),
        "corpus_used": cinfo["corpus_used"],
        "corpus_degraded": cinfo["degraded"],
        "seeds": seeds, "n_seeds": len(seeds),
        "config": {"vocab_size": (vocab_size_actual
                                  if vocab_size_actual is not None
                                  else a.vocab_size),
                   "vocab_size_requested": a.vocab_size,
                   "gen_tokens": a.gen_tokens,
                   "eval_positions": a.eval_positions},
        "anti_cheat": {
            "hardened_gate_core": "0.20/1.5/0.5/0.20 + abs-competence "
                                   "floor; >=3 seeds; NEVER tuned; "
                                   "uniform_ppl passed",
            "load_bearing_control": "BPE-invariant word-shuffle of the "
                                     "train split; control n-gram has "
                                     "no trigram structure",
            "ngram_cheat_is_regurgitation": "the hardened verbatim-"
                                             "copy<=0.20 bar is the "
                                             "load-bearing adjudicator",
            "honest_ceiling": "n-gram-class LOCAL coherence ONLY, "
                               "explicitly NOT an LLM; never spun",
            "honest_propagation": "CONTROLLER's post-run job; runner "
                                   "only computes+writes JSON"},
        "per_seed": per_seed_records,
        "aggregate_verdict": agg,
        "GATE": agg["GATE"],
        "OVERALL": "PASS" if agg["GATE"] == "PASS" else "FAIL",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=str),
                           encoding="utf-8")

    print("\n" + "=" * 64, flush=True)
    print("GENERATOR-E GATE VERDICT", flush=True)
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
          % (agg["GATE"], agg["n_seeds"], agg["n_pass"]), flush=True)
    if agg["GATE"] != "PASS":
        print("  NOTE: a maxed FAIL is an HONEST finding -> propagate "
              "(terminal decision-relevant) ; do NOT config-crank.",
              flush=True)
    else:
        print("  NOTE: a PASS is reported STRICTLY at the honest "
              "ceiling (n-gram-class LOCAL coherence, NOT an LLM); "
              "controller smell-tests regurgitation before "
              "propagating.", flush=True)
    print("  -> %s" % a.out, flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
