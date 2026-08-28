"""De-risk: does a CPU-cheap DECODE-TIME repetition guard suppress the WKV mouth's repetition/looping
residual, through the PRODUCTION `webapp.wkv_mouth_generator.generate()` entry point? (2026-08-28)

CONTEXT -- the diagnosis this runner tests, already done, not re-derived here
(`research/findings/2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md`, the A/B GO's own SS5
"honest residuals, named as next levers"). That finding read all 8 of the e-prop LOCALLY-LEARNED read-out
head's generated continuations verbatim (not just automated distinct-2/max-repeat-run scores) and found a
repeated word or short phrase in 5/8 (2 severe) -- e.g. index 1's `"...and again ever again and again ever
again and again ever again and again again and again i was safe again and again..."`, and index 6's `"...or
again or again or again or again or again or again"`, a strict period-2 alternation (`or`, `again`, `or`,
`again`, ...) the existing `max_repeat_run` metric (longest run of IMMEDIATELY CONSECUTIVE identical tokens)
is BLIND to (index 6 scored `max_repeat_run=1` despite the visible loop, because no two `again`s are
adjacent -- `or` always sits between them). That same finding also showed self-NLL can be ADVERSARIALLY
FOOLED by this exact failure mode: index 6's self-NLL was *better* than native's on the identical prompt
(1.291 vs 1.504 nats) *despite* the visible loop -- a model can be highly self-confident of its OWN
repetition, so self-NLL under-penalizes this class of artifact and must never be the PRIMARY gate here.

WHY the generator had zero repetition guard to begin with (read by inspecting
`webapp/wkv_mouth_generator.py::_free_gen` before this rung, lines ~286-304 pre-change): the driving loop
samples the next word via the genuine few-spike Izhikevich spiking population-coded winner-take-all read
(`FewSpikeWordRead.read`, GO-verified in `research.runners._wkv_fewspike_read_derisk`) over the top-`topk`
candidates by raw logit -- but `gen` (the token history) was never fed back into the logits before that
top-k cut, so nothing in the decode path discourages the smoothly-evolving WKV recurrent state from locking
onto a small set of high-row-norm tokens ("high", "again", ...) once it starts favoring them.

WHAT THIS RUNNER BUILDS AND MEASURES (the fix + the diagnostic in one pass):
  1. A CYCLE-AWARE repetition metric (`cycle_metrics`, below) that EXTENDS the existing period-1
     `max_repeat_run` with (a) the max repeat COUNT of any 2-gram or 3-gram within a sliding window of the
     last ~20 tokens, and (b) a period-2 alternation run-length (`token[i] == token[i-2]`) that catches
     exactly the "or again or again..." shape the period-1 metric missed. GATES on repeated-{2,3}-gram-count
     >= 4 OR period-2 run >= 4 -- self-NLL is used ONLY as a SECONDARY non-degradation guard (learned mean
     must not exceed the native head's own self-NLL mean, 1.469 nats
     (`2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md` SS4), plus ~0.15 nats slack), NEVER
     the primary gate -- exactly because that finding showed self-NLL can read BETTER on a looping sample.
  2. A decode-time repetition guard threaded into the PRODUCTION `webapp.wkv_mouth_generator.generate()` /
     `_free_gen()` (`repetition_penalty`, `no_repeat_ngram_size`, both new kwargs, DEFAULT 1.0/0 = an EXACT
     no-op -- see that module's own `_apply_repetition_controls` docstring). Applied to the FULL-vocab
     logits immediately after the unk mask and BEFORE the top-k cut (so a banned/penalized token cannot
     re-enter the candidate set the spiking reader samples over) -- `reader.read(p)` itself, the genuine
     spiking-population read, is never touched. This is legitimately HOST territory per `docs/`'s
     brain-based-only boundary: a decode control is the same category as the pre-existing `topk`/`gen_temp`
     knobs, not a cognitive computation the brain should be doing.
  3. THIS runner: loads the persisted e-prop LOCALLY-LEARNED head (`BRAIN_WKV_MOUTH_LEARNED_HEAD`, pointed
     at `research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz` -- the SAME
     artifact, SAME seed=102 checkpoint pairing, the A/B GO used; see that finding SS1 for why seed=102 is
     the only seed this specific npz legitimately represents), regenerates each of N prompts through the
     SAME production `generate()` call TWICE -- once with `repetition_penalty=1.0, no_repeat_ngram_size=0`
     (baseline, expected to REPRODUCE the looping) and once with the caller's penalty/n-gram knobs (expected
     to SUPPRESS it) -- and scores both arms with `cycle_metrics`.

THE DIAGNOSTIC VERDICT this runner exists to answer, per prompt from the caller (not assumed):
  `decode_suppressible = True`  if the baseline arm reproduces >=1 gated loop AND the penalty arm has ZERO
                                  gated loops -- the repetition is a DECODE-TIME artifact, a cheap band-aid
                                  suffices, and the objective (retraining / a decorrelation-read primitive)
                                  stays an optional future lever, not a mandatory one.
  `decode_suppressible = False` if the penalty arm STILL gates a loop on at least one sample where the
                                  baseline also looped -- the degeneracy is at least partly BAKED INTO the
                                  learned head's own logit landscape (`W_hat`) and a decode-time guard alone
                                  cannot fully close it; the objective fix (SS5's other two candidate levers)
                                  becomes the load-bearing next step, not merely a nice-to-have.
  `decode_suppressible = None/"undefined_no_baseline_loop"` if the baseline arm reproduces ZERO gated loops
                                  on this specific prompt/seed set (a real possibility -- the A/B GO's own
                                  5/8 rate is not guaranteed to replicate on a different sample; see
                                  `tools.lab.undefined_if_empty` for why this is reported as UNDEFINED, not
                                  fabricated as a negative result).

NOT re-derived here (already GO'd / measured elsewhere, cited not repeated): the few-spike Izhikevich read's
own anti-cheats (`_wkv_fewspike_read_derisk`'s own GO); the learned head's coherence-vs-chance and
lever/fail-safe/RNG soundness (`_wkv_learned_vs_native_head_ab.py`'s own GO, whose `_self_nll` /
`_ngram_distinct` / `_max_repeat_run` helpers and 8-prompt set this runner REUSES verbatim, not re-derives).

CPU/numpy only (~512 neurons per read, read_window=40). Detached-run friendly: prints progress per
prompt/arm so a `nohup ... &` caller can tail it; each generation is seconds on the persisted head.

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_repetition_penalty_derisk \\
        --head research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz \\
        --seed 102 --repetition-penalty 1.3 --no-repeat-ngram 3 --gen-seeds 0,1,2,3,4,5,6,7 \\
        --out research/findings/raw/_wkv_rep_penalty_derisk.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools.verdict import Verdict  # noqa: E402
from tools.lab import lever, void_if, undefined_if_empty  # noqa: E402
# REUSED verbatim (not re-derived): the A/B GO's own teacher-forced self-NLL replay + coherence n-gram
# helpers, and its 8 in-vocab TinyStories-domain prompts (each independently verified via
# `webapp.wkv_mouth_generator.in_vocab_scope` by that runner).
from research.runners._wkv_learned_vs_native_head_ab import (  # noqa: E402
    _self_nll, _ngram_distinct, _max_repeat_run, PROMPTS,
)

DEFAULT_HEAD_NPZ = _REPO / "research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz"
DEFAULT_OUT = _REPO / "research/findings/raw/_wkv_rep_penalty_derisk.json"
DEFAULT_SEED = 102          # the ONLY seed the default npz's e-prop-learned head actually represents (A/B GO SS1)
MAX_NEW_TOKENS = 50         # matches the A/B GO's own generation length, for a like-for-like replication
READ_WINDOW = 40
POP = 8
TOPK = 64
GEN_TEMP = 0.8

# ── the cycle-aware gate thresholds (this rung's own contribution) ─────────────────────────────────────────────────
CYCLE_WINDOW = 20           # sliding window: only the last ~15-20 generated tokens are scored for cycling
FAIL_NGRAM_COUNT = 4        # gate: a single 2-gram or 3-gram repeated >=4x within the window
FAIL_PERIOD2_RUN = 4        # gate: an A-B-A-B... alternation (token[i]==token[i-2]) of length >=4 within the window
NATIVE_SELF_NLL_MEAN = 1.469    # cited, not re-derived: the A/B GO's own native-head self-NLL mean (SS4 table)
SELF_NLL_SLACK = 0.15           # secondary non-degradation guard's allowed margin above that native mean


def cycle_metrics(words: list[str], window: int = CYCLE_WINDOW) -> dict:
    """Cycle-aware repetition metric over the LAST `window` tokens of a generated continuation.

    EXTENDS the period-1 `max_repeat_run` (longest run of IMMEDIATELY CONSECUTIVE identical tokens -- blind
    to an A-B-A-B cycle, per the module docstring's index-6 example) with:
      (a) the max repeat COUNT of any single 2-gram or 3-gram within the window (a `Counter.most_common`
          top count -- a loop that cycles through >2 distinct tokens, e.g. "and again ever again and again
          ever again", shows up here even though no single word repeats consecutively);
      (b) `period2_max_run`: the longest run of consecutive positions i where `words[i] == words[i-2]` --
          catches the "or again or again or again" shape directly (each `words[i]` equals the word two
          positions back, even though the immediate neighbor differs every time).

    GATE (`gate_fail`): True if `max(2gram_count, 3gram_count) >= FAIL_NGRAM_COUNT` OR
    `period2_max_run >= FAIL_PERIOD2_RUN`. This is the PRIMARY pass/fail criterion this runner uses --
    self-NLL is a separate, secondary, non-degradation-only guard (see module docstring for why: self-NLL
    can read BETTER on a looping sample, so it must never gate primarily).

    `unresolved_at_cutoff`: True if the cycle detected is STILL ACTIVE at the very last token of the window
    (as opposed to a cycle that occurred earlier in the window and then broke on its own) -- a diagnostic
    field, not part of the gate: tells us whether letting `max_new_tokens` run longer would likely extend
    the SAME loop, vs one that had already self-terminated before generation was cut off.
    """
    win = words[-window:] if len(words) > window else list(words)
    n = len(win)

    def _max_ngram(k):
        if n < 2 * k:
            return 0, None
        grams = [tuple(win[i:i + k]) for i in range(n - k + 1)]
        counts = Counter(grams)
        top_gram, top_count = counts.most_common(1)[0]
        return top_count, top_gram

    g2_count, g2_top = _max_ngram(2)
    g3_count, g3_top = _max_ngram(3)

    p2_best = p2_cur = 0
    for i in range(2, n):
        if win[i] == win[i - 2]:
            p2_cur += 1
            p2_best = max(p2_best, p2_cur)
        else:
            p2_cur = 0

    # is the period-2 alternation STILL RUNNING at the very last window token?
    p2_active_at_end = 0
    if n >= 3:
        i = n - 1
        while i >= 2 and win[i] == win[i - 2]:
            p2_active_at_end += 1
            i -= 1

    # does the LAST n-gram in the window match an EARLIER occurrence (n-gram loop still active at cutoff)?
    ngram_unresolved = False
    for k, count in ((2, g2_count), (3, g3_count)):
        if n >= k and count >= 2:
            last_gram = tuple(win[-k:])
            earlier = [tuple(win[i:i + k]) for i in range(n - k)]
            if last_gram in earlier:
                ngram_unresolved = True

    unresolved_at_cutoff = bool(ngram_unresolved or p2_active_at_end >= 2)
    gate_fail = bool(max(g2_count, g3_count) >= FAIL_NGRAM_COUNT or p2_best >= FAIL_PERIOD2_RUN)

    return {
        "window_tokens": n,
        "max_2gram_repeat_count": g2_count, "max_2gram": list(g2_top) if g2_top else None,
        "max_3gram_repeat_count": g3_count, "max_3gram": list(g3_top) if g3_top else None,
        "period2_max_run": p2_best, "period2_active_at_end": p2_active_at_end,
        "unresolved_at_cutoff": unresolved_at_cutoff,
        "gate_fail": gate_fail,
    }


def _selftest_cycle_metrics():
    """Cheap, deterministic self-check run at import/argparse time -- the two exact shapes named in the
    diagnosis must gate FAIL, and a clean non-repeating continuation must gate PASS. Not a formal test
    suite (none exists for this module yet); a lightweight guard that the metric can actually fail."""
    period2 = "or again or again or again or again or again or again".split()
    m = cycle_metrics(period2)
    assert m["gate_fail"] is True and m["period2_max_run"] >= FAIL_PERIOD2_RUN, \
        "period-2 A-B-A-B loop must gate FAIL (was the #1 case max_repeat_run missed): %r" % m
    assert _max_repeat_run(period2) < FAIL_PERIOD2_RUN, \
        "sanity: the period-1 metric must NOT catch this case (else it wouldn't be a new failure mode): %r" \
        % _max_repeat_run(period2)

    # a period-3 cycle ("the cat sat" x4) -- distinct from the period-2 case above, and needs >=4
    # repeats of the 3-gram to clear FAIL_NGRAM_COUNT, exercising the (a) branch of the gate on its own.
    ngram_loop = "the cat sat the cat sat the cat sat the cat sat".split()
    m2 = cycle_metrics(ngram_loop)
    assert m2["gate_fail"] is True and m2["max_3gram_repeat_count"] >= FAIL_NGRAM_COUNT, \
        "repeated 3-gram loop must gate FAIL via the n-gram branch: %r" % m2
    assert m2["period2_max_run"] < FAIL_PERIOD2_RUN, \
        "sanity: a period-3 cycle should not ALSO trip the period-2 branch: %r" % m2

    clean = ("once upon a time there was a little boy named tim who had a dog and they went "
              "to the park together and played all day long").split()
    m3 = cycle_metrics(clean)
    assert m3["gate_fail"] is False, "a clean non-repeating continuation must gate PASS: %r" % m3


def _run_arm(W, prompt: str, seed: int, rp: float, nrn: int) -> dict:
    ro, _vocab, word_to_id = W._get_readout(seed)
    status = W.learned_head_status(seed)
    text, secs = W.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS, topk=TOPK,
                             read_window=READ_WINDOW, pop=POP, gen_temp=GEN_TEMP,
                             repetition_penalty=rp, no_repeat_ngram_size=nrn)
    cont = text[len(prompt):].strip() if text.startswith(prompt) else text
    cont_words = cont.split()
    self_nll, n_scored = _self_nll(ro, word_to_id, text)
    cm = cycle_metrics(cont_words)
    return {
        "prompt": prompt, "repetition_penalty": rp, "no_repeat_ngram_size": nrn,
        "text": text, "continuation": cont, "gen_seconds": secs,
        "learned_head_status": status,
        "self_nll": self_nll, "n_words_scored": n_scored, "n_continuation_words": len(cont_words),
        "distinct_1": _ngram_distinct(cont_words, 1), "distinct_2": _ngram_distinct(cont_words, 2),
        "distinct_3": _ngram_distinct(cont_words, 3), "max_repeat_run": _max_repeat_run(cont_words),
        "cycle_metrics": cm,
    }


def main():
    _selftest_cycle_metrics()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", type=str, default=str(DEFAULT_HEAD_NPZ))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--gen-seeds", type=str, default="0,1,2,3,4,5,6,7",
                     help="comma-separated indices into the SAME 8-prompt PROMPTS list "
                          "`_wkv_learned_vs_native_head_ab.py` uses -- see module docstring: `generate()`'s "
                          "own `seed` kwarg conflates checkpoint-selection + head-path + RNG-timeline, so "
                          "diversity across generations comes from (a) distinct prompts and (b) that "
                          "seed's private RNG timeline continuing to advance across successive calls "
                          "(the SAME mechanism the A/B GO already used), not from varying `--seed` itself.")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    head_path = Path(args.head)
    gen_idx = [int(x) for x in args.gen_seeds.split(",") if x.strip() != ""]
    void_if(any(i < 0 or i >= len(PROMPTS) for i in gen_idx),
            f"--gen-seeds index out of range for the {len(PROMPTS)}-prompt PROMPTS list: {gen_idx}")
    prompts = [PROMPTS[i] for i in gen_idx]

    os.environ["BRAIN_WKV_MOUTH_LEARNED_HEAD"] = "1"
    os.environ["BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH"] = str(head_path)   # read at IMPORT time -> set FIRST
    from webapp import wkv_mouth_generator as W  # noqa: E402  (deliberately late; see line above)

    npz = np.load(head_path, allow_pickle=True)
    npz_meta = {k: (npz[k].item() if npz[k].shape == () else npz[k].shape) for k in npz.files}
    print(f"[reppen] learned-head artifact: {head_path.name}  meta={npz_meta}")
    void_if(int(npz_meta.get("seed", -1)) != args.seed,
            f"npz seed={npz_meta.get('seed')} != --seed={args.seed} -- checkpoint/head mismatch")

    host_rng_before = np.random.get_state()[1].copy()

    baseline_runs, treated_runs = [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        r_base = _run_arm(W, p, args.seed, 1.0, 0)                                    # penalty OFF (no-op)
        r_treat = _run_arm(W, p, args.seed, args.repetition_penalty, args.no_repeat_ngram)  # penalty ON
        baseline_runs.append(r_base)
        treated_runs.append(r_treat)
        print(f"[reppen] {i+1}/{len(prompts)}  gen_seed_idx={gen_idx[i]}  "
              f"baseline_gate_fail={r_base['cycle_metrics']['gate_fail']}  "
              f"treated_gate_fail={r_treat['cycle_metrics']['gate_fail']}  "
              f"baseline_nll={r_base['self_nll']}  treated_nll={r_treat['self_nll']}  "
              f"elapsed={time.time()-t0:.1f}s")

    host_rng_after = np.random.get_state()[1].copy()
    rng_untouched = bool((host_rng_before == host_rng_after).all())

    all_applied = all((r["learned_head_status"] or {}).get("applied") is True
                       for r in baseline_runs + treated_runs)
    lever("repetition_penalty knob", 1.0, args.repetition_penalty)
    lever("no_repeat_ngram_size knob", 0, args.no_repeat_ngram)

    n_baseline_loop = sum(1 for r in baseline_runs if r["cycle_metrics"]["gate_fail"])
    n_treated_loop = sum(1 for r in treated_runs if r["cycle_metrics"]["gate_fail"])
    n_treated_unresolved = sum(1 for r in treated_runs if r["cycle_metrics"]["unresolved_at_cutoff"])

    print(f"[reppen] baseline loops (gate_fail): {n_baseline_loop}/{len(prompts)}   "
          f"treated loops (gate_fail): {n_treated_loop}/{len(prompts)}")

    decode_suppressible = undefined_if_empty(
        "decode_suppressible (baseline loops to suppress)", n_baseline_loop,
        score=(n_treated_loop == 0), total=len(prompts))
    # undefined_if_empty prints/returns None when n_baseline_loop==0; otherwise it returns the boolean score
    # (True iff the penalty arm has ZERO gated loops among the prompts that gated a loop in baseline).

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    baseline_nll_mean = _mean([r["self_nll"] for r in baseline_runs])
    treated_nll_mean = _mean([r["self_nll"] for r in treated_runs])
    self_nll_ceiling = NATIVE_SELF_NLL_MEAN + SELF_NLL_SLACK

    art = {
        "probe": "wkv_mouth_repetition_penalty_derisk", "backend": "numpy", "seed": args.seed,
        "head_path": str(head_path.relative_to(_REPO)) if head_path.is_relative_to(_REPO) else str(head_path),
        "npz_meta": npz_meta, "gen_seed_indices": gen_idx, "n_prompts": len(prompts),
        "max_new_tokens": MAX_NEW_TOKENS, "read_window": READ_WINDOW,
        "repetition_penalty": args.repetition_penalty, "no_repeat_ngram_size": args.no_repeat_ngram,
        "cycle_window": CYCLE_WINDOW, "fail_ngram_count": FAIL_NGRAM_COUNT,
        "fail_period2_run": FAIL_PERIOD2_RUN,
        "native_self_nll_mean_cited": NATIVE_SELF_NLL_MEAN, "self_nll_slack": SELF_NLL_SLACK,
        "self_nll_ceiling": self_nll_ceiling,
        "baseline_runs": baseline_runs, "treated_runs": treated_runs,
        "baseline_self_nll_mean": baseline_nll_mean, "treated_self_nll_mean": treated_nll_mean,
        "n_baseline_loop": n_baseline_loop, "n_treated_loop": n_treated_loop,
        "n_treated_unresolved_at_cutoff": n_treated_unresolved,
        "decode_suppressible": decode_suppressible,
        "self_nll_secondary_guard_ok": (treated_nll_mean is not None and treated_nll_mean <= self_nll_ceiling),
        "all_learned_applied": all_applied, "rng_untouched_across_run": rng_untouched,
        "elapsed_s": round(time.time() - t0, 1),
    }

    # The self-NLL secondary guard is REPORTED (art["self_nll_secondary_guard_ok"] above, and printed below)
    # but deliberately NOT registered as a `Verdict.require` -- per this runner's own module docstring and
    # the task this rung was scoped from, self-NLL is "ONLY a secondary non-degradation guard... NEVER the
    # primary gate" (the A/B GO's own SS3 showed self-NLL can read BETTER on a looping sample, so making it
    # a hard blocking precondition here would let the exact instrument failure that finding surfaced veto a
    # result the PRIMARY cycle-aware gate has already earned). The `aggregate GO` this runner's own task
    # scope defines is "all penalty-ON samples pass the [primary, cycle-aware] gate" -- registered below.
    v = Verdict("decode-time repetition_penalty/no_repeat_ngram_size guard suppresses the learned-head "
                "repetition/looping residual named by the A/B GO's SS5, through the production entry point")
    v.require("(lever) the learned head loader reports applied=True on every call (no silent fallback)",
              all_applied, expect=True)
    v.require("(RNG) host process-global numpy RNG state is byte-identical before/after the whole run",
              rng_untouched, expect=True)
    v.require("(primary gate) every penalty-ON sample passes the cycle-aware repetition gate",
              n_treated_loop, expect=lambda x: x == 0)

    go = bool(all_applied and rng_untouched and n_treated_loop == 0)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["GO"] = go
    art["preconditions"] = decided.get("preconditions", [])   # gates/verdict_preconditions reads this TOP-LEVEL
    if not art["self_nll_secondary_guard_ok"]:
        print(f"  ⚠️  SECONDARY GUARD (non-blocking, reported not gated): treated self-NLL mean "
              f"{treated_nll_mean} exceeds the ceiling {self_nll_ceiling} "
              f"(native {NATIVE_SELF_NLL_MEAN} + {SELF_NLL_SLACK} slack) -- disclosed, does not change GO.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art, indent=1))
    print(json.dumps({
        "n_baseline_loop": n_baseline_loop, "n_treated_loop": n_treated_loop,
        "decode_suppressible": decode_suppressible,
        "baseline_self_nll_mean": baseline_nll_mean, "treated_self_nll_mean": treated_nll_mean,
        "self_nll_ceiling": self_nll_ceiling, "all_learned_applied": all_applied,
        "rng_untouched_across_run": rng_untouched, "GO": go,
    }, indent=1))
    print(f"wrote {out_path} -> {decided['status']}   decode_suppressible={decode_suppressible}")
    return decided["status"]


if __name__ == "__main__":
    main()
