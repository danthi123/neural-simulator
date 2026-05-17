"""Generator-D pre-registered MULTI-SEED capability gate. Decides
whether a subword spiking LM trained by KNOWLEDGE DISTILLATION (soft
cross-entropy against a competent trigram teacher) on a real public
corpus generates coherent held-out text where Generator-S (hard
one-hot target) failed. Anti-cheat: the SAME HARDENED gate_core
(FIXED bars 0.20/1.5/0.5/0.20 + absolute-competence floor; >=3 seeds;
NEVER tuned here) judges the STUDENT (teacher discarded post-training);
the load-bearing control is a WORD-SHUFFLE of the train split
(BPE-invariant -> identical tokenizer/token distribution, ONLY
sequence order destroyed; its teacher is the trigram on that shuffled
text -> no sequential structure). Kill-safe resume. Honest propagation
is the CONTROLLER's post-run job; main() ONLY computes+prints+writes
JSON (exit 0 == verdict computed; exit 2 == not runnable). ASCII
only."""
from __future__ import annotations
import argparse


def _word_shuffle(text, rng):
    """BPE-invariant faithful control: same words (-> same word
    frequencies -> identical BPE merges -> identical tokenizer/token
    distribution), sequence order destroyed."""
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def _heldout_nll(layers, tok, text, T, xp, max_positions):
    """Teacher-forced per-token NLL over held-out ids (STUDENT). REUSES
    the validated forward_unroll_xp + cross_entropy_loss_np (DRY); SAME
    logits semantics as training. Returns list[float] nll."""
    import numpy as np
    from sim.bptt_snn_gpu import forward_unroll_xp
    from sim.bptt_snn import cross_entropy_loss_np
    ids = list(tok.encode(text))
    V = tok.vocab_size
    nll = []
    n = len(ids)
    if n < 2:
        return nll
    step = max(1, (n - 1) // max(1, int(max_positions)))
    for pos in range(T, n, step):
        ctx = ids[pos - T:pos]
        oh = np.zeros((T, 1, V), dtype=np.float32)
        for t, tid in enumerate(ctx):
            if 0 <= tid < V:
                oh[t, 0, tid] = 1.0
        x = xp.asarray(oh) if xp.__name__ == "cupy" else oh
        st = forward_unroll_xp(x, layers, xp=xp)
        lg = st["spikes"][-1].sum(axis=0)
        lg = lg.get() if hasattr(lg, "get") else lg
        nll.append(cross_entropy_loss_np(lg[0:1], int(ids[pos])))
    return nll


def _teacher_heldout_ppl(teacher, tok, text, max_positions):
    """TRANSPARENCY ONLY (NOT a gate input): the trigram teacher's own
    held-out perplexity, mirroring the grounded probe."""
    import math
    ids = list(tok.encode(text))
    n = len(ids)
    if n < 3:
        return None
    step = max(1, (n - 2) // max(1, int(max_positions)))
    nll = []
    for i in range(2, n, step):
        p = float(teacher.soft_dist((ids[i - 2], ids[i - 1]))[ids[i]])
        nll.append(-math.log(max(p, 1e-12)))
    if not nll:
        return None
    return math.exp(sum(nll) / len(nll))


def main():
    import json
    import time
    from pathlib import Path
    import numpy as np

    from research.runners.corpus_fetch import fetch_corpus, split_corpus
    from research.runners.distill_subword_lm_train import (
        train_distill_subword_lm)
    from research.runners.subword_lm_generate import generate
    from research.runners.subword_lm_gate_core import (
        perplexity, distinct_ngram_ratio, verbatim_copy_fraction,
        gs_verdict, gs_aggregate_multiseed,
    )
    from sim.bptt_snn_gpu import _get_backend

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--corpus", type=str, default="tinystories")
    ap.add_argument("--max-corpus-mb", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--hidden-layers", type=str, default="256,256")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-train-samples", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--gen-tokens", type=int, default=80)
    ap.add_argument("--eval-positions", type=int, default=400)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_d_gate.json")
    ap.add_argument("--ckpt", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_d_gate.ckpt")
    a = ap.parse_args()

    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    hidden = [int(x) for x in a.hidden_layers.split(",") if x.strip()]

    print("=" * 64, flush=True)
    print("GENERATOR-D PRE-REGISTERED MULTI-SEED CAPABILITY GATE",
          flush=True)
    print("(subword spiking LM via KNOWLEDGE DISTILLATION on a real "
          "public corpus;", flush=True)
    print(" SAME HARDENED gate_core (0.20/1.5/0.5/0.20 + abs-competence "
          "floor) NEVER tuned here;", flush=True)
    print(" load-bearing control = BPE-invariant word-shuffle; >= 3 "
          "seeds)", flush=True)
    print("=" * 64, flush=True)

    if len(seeds) < 3:
        print("[NOT RUNNABLE] %d seed(s); >= 3 MANDATORY (single-seed "
              "is NOT a pass -- gate_core enforces, this is the early "
              "exit)." % len(seeds), flush=True)
        return 2

    xp, is_gpu = _get_backend(prefer_gpu=True)
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

    cdir = Path(a.ckpt).parent
    cdir.mkdir(parents=True, exist_ok=True)
    tr_file = str(cdir / "gen_d_train.txt")
    Path(tr_file).write_text(train_text, encoding="utf-8")

    per_seed_verdicts = []
    per_seed_records = []
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
        # Real model on the train split (DISTILLED).
        rr = train_distill_subword_lm(
            seed=seed, corpus_path=tr_file, vocab_size=a.vocab_size,
            hidden_layers=hidden, T=a.T, epochs=a.epochs,
            batch_size=a.batch_size, lr=a.lr,
            n_train_samples=a.n_train_samples,
            ckpt_path="%s.s%d.real.npz" % (a.ckpt, seed),
            bpe_path="%s.s%d.real.bpe.json" % (a.ckpt, seed),
            backend="auto", verbose=False)
        # WORD-SHUFFLED control (BPE-invariant; only order destroyed;
        # its teacher is the trigram on the shuffled text -> no
        # sequential structure). Identical model/config, DISTILLED.
        ctl_file = str(cdir / ("gen_d_ctl_s%d.txt" % seed))
        Path(ctl_file).write_text(
            _word_shuffle(train_text, np.random.default_rng(
                seed * 911 + 1)), encoding="utf-8")
        cr = train_distill_subword_lm(
            seed=seed, corpus_path=ctl_file, vocab_size=a.vocab_size,
            hidden_layers=hidden, T=a.T, epochs=a.epochs,
            batch_size=a.batch_size, lr=a.lr,
            n_train_samples=a.n_train_samples,
            ckpt_path="%s.s%d.ctl.npz" % (a.ckpt, seed),
            bpe_path="%s.s%d.ctl.bpe.json" % (a.ckpt, seed),
            backend="auto", verbose=False)

        rtok, rlay = rr["_tok"], rr["_layers"]
        ctok, clay = cr["_tok"], cr["_layers"]
        ho_nll = _heldout_nll(rlay, rtok, heldout_text, a.T, xp,
                              a.eval_positions)
        ctl_nll = _heldout_nll(clay, ctok, heldout_text, a.T, xp,
                               a.eval_positions)
        tr_nll = _heldout_nll(rlay, rtok, train_text[:len(heldout_text)],
                              a.T, xp, a.eval_positions)
        ho_ppl = perplexity(ho_nll)
        ctl_ppl = perplexity(ctl_nll)
        tr_ppl = perplexity(tr_nll)
        teach_ppl = _teacher_heldout_ppl(
            rr["_teacher"], rtok, heldout_text, a.eval_positions)

        grng = np.random.default_rng(seed * 13 + 5)
        prompt = " ".join(heldout_text.split()[:8])
        gen_ids, gen_txt = generate(rlay, rtok, prompt,
                                    a.gen_tokens, a.T, xp=xp,
                                    rng=grng, temperature=1.0)
        train_ids = list(rtok.encode(train_text))
        distinct = distinct_ngram_ratio(gen_ids, n=3)
        copy_frac = verbatim_copy_fraction(gen_ids, train_ids, n=8)

        # HARDENED gate_core: MUST pass uniform_ppl (fail-closed
        # without it). The gate judges the STUDENT; teacher_ppl is
        # transparency only, NOT an input.
        v = gs_verdict(heldout_ppl=ho_ppl, shuffled_ppl=ctl_ppl,
                       train_ppl=tr_ppl, distinct=distinct,
                       copy_frac=copy_frac, has_shuffled_control=True,
                       uniform_ppl=rtok.vocab_size)
        v["seed"] = seed
        per_seed_verdicts.append(v)
        per_seed_records.append({
            "seed": seed, "resumed": False,
            "heldout_ppl": ho_ppl, "shuffled_ctl_ppl": ctl_ppl,
            "train_ppl": tr_ppl, "uniform_ppl": rtok.vocab_size,
            "teacher_heldout_ppl": teach_ppl,
            "distinct_trigram": distinct,
            "verbatim_copy_frac": copy_frac,
            "real_final_loss": rr.get("final_loss"),
            "ctl_final_loss": cr.get("final_loss"),
            "gen_sample": gen_txt[:240], "verdict": v})
        completed[seed] = v
        _flush_resume(completed)
        print("[SEED %d] ho_ppl=%.3f ctl_ppl=%.3f tr_ppl=%.3f "
              "teacher_ppl=%s distinct=%.3f copy=%.3f -> %s"
              % (seed, ho_ppl, ctl_ppl, tr_ppl, str(teach_ppl),
                 distinct, copy_frac, v["GATE"]), flush=True)

    agg = gs_aggregate_multiseed(per_seed_verdicts)
    result = {
        "task": "Generator-D pre-registered MULTI-SEED capability gate",
        "mechanism": ("subword spiking LM via knowledge distillation "
                      "(soft-xent vs competent trigram teacher), real "
                      "public corpus, self-contained at runtime "
                      "(teacher discarded post-training)"),
        "corpus_used": cinfo["corpus_used"],
        "corpus_degraded": cinfo["degraded"],
        "seeds": seeds, "n_seeds": len(seeds),
        "config": {"vocab_size": a.vocab_size, "hidden_layers": hidden,
                   "T": a.T, "epochs": a.epochs,
                   "batch_size": a.batch_size,
                   "n_train_samples": a.n_train_samples, "lr": a.lr},
        "anti_cheat": {
            "hardened_gate_core": "0.20/1.5/0.5/0.20 + abs-competence "
                                   "floor; >=3 seeds; NEVER tuned in "
                                   "the runner; uniform_ppl passed",
            "load_bearing_control": "BPE-invariant word-shuffle of the "
                                     "train split (identical tokenizer/"
                                     "token distribution; only sequence "
                                     "order destroyed; its teacher has "
                                     "no sequential structure)",
            "teacher_is_transparency_only": "teacher_heldout_ppl "
                                             "recorded for transparency; "
                                             "the gate judges the "
                                             "STUDENT; teacher discarded "
                                             "post-training",
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
    print("GENERATOR-D GATE VERDICT", flush=True)
    print("=" * 64, flush=True)
    for r in per_seed_records:
        vv = r["verdict"]
        print("  seed %s: %s (ho_ppl=%s ctl_ppl=%s tr_ppl=%s "
              "teacher_ppl=%s distinct=%s copy=%s)"
              % (r["seed"], vv["GATE"],
                 r.get("heldout_ppl"), r.get("shuffled_ctl_ppl"),
                 r.get("train_ppl"), r.get("teacher_heldout_ppl"),
                 r.get("distinct_trigram"),
                 r.get("verbatim_copy_frac")), flush=True)
    print("  AGGREGATE: %s (n_seeds=%d n_pass=%d; >=3 mandatory; "
          "HARDENED bars untouched)"
          % (agg["GATE"], agg["n_seeds"], agg["n_pass"]), flush=True)
    if agg["GATE"] != "PASS":
        print("  NOTE: a maxed FAIL is an HONEST finding -> propagate "
              "+ proceed to pre-staged Generator-E; do NOT "
              "config-crank.", flush=True)
    print("  -> %s" % a.out, flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
