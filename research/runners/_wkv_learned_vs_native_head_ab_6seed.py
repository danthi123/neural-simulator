"""6-SEED A/B: e-prop LOCALLY-LEARNED WKV-mouth read-out head vs the checkpoint's NATIVE (host-trained/copied)
head, through the PRODUCTION `webapp.wkv_mouth_generator.generate()` entry point, at the DEFAULT (un-overridden)
learned-head path template. (2026-09-01, mouth crutch-burndown rung-1, part 2+3 of 3.)

WHY A SEPARATE RUNNER FROM `_wkv_learned_vs_native_head_ab.py` (not a `--seeds` flag added to it). That runner is
deliberately PINNED to `SEED=102` and to the single literal artifact `wkv_eprop_learned_head_6seed.npz` (which,
despite its name, holds only seed=102's head -- a real templating bug in the run that produced it, documented in
that runner's own docstring and in `2026-08-28-wkv-learned-vs-native-head-AB-worth-keeping-opt-in.md` SS1). That
bug was independently fixed on the PERSIST side (`eprop_learn_persist_6seed.json`, 2026-08-28: `--save-w-hat`
re-run with proper `{seed}` templating, producing six genuinely distinct per-seed files,
`wkv_eprop_learned_head_0p94_s{42,43,44,100,101,102}.npz`, `sub_recov_ratio` mean 0.9273 min 0.8906 -- verified
here by SHA1 that the six files are NOT byte-identical, i.e. not a repeat of the same overwrite bug) AND on the
LOAD side (`aa7c3a23c`, `webapp/wkv_mouth_generator.py:76-81`: `_LEARNED_HEAD_PATH_TEMPLATE` default now points
at `_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s{seed}.npz` with `{seed}` templating, replacing a
prior default that pointed at a nonexistent `_wkv_eprop_learned_head_seed{seed}.npz`). THIS runner is the first
one to actually EXERCISE that fixed default end-to-end across all six seeds it names -- it deliberately does
**not** set `BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH`, so `_learned_head_path(seed)` must resolve the correct,
seed-distinct file from the module's own default alone. A regression in that default (the exact class of bug
this rung exists to close) shows up here as `applied=False` / `reason=file_missing` on every learned-arm call,
which the Verdict below treats as an automatic NO-GO, not a silently-degraded run.

METHOD, per (seed, prompt) pair -- 6 seeds x 8 in-vocab TinyStories-domain prompts (same 8 prompts as the
single-seed A/B, imported from it verbatim, not re-typed) x 2 arms (native, learned) = 96 production
`generate()` calls:
  (1) self-NLL (nats) of the generated continuation under the ACTIVE head's own teacher-forced next-word
      distribution, vs chance `log(1000)=6.9078` nats -- reusing `_self_nll` from the single-seed A/B verbatim
      (same convention: gate on the PREVIOUS token to predict the NEXT).
  (2) coherence beyond self-NLL: distinct-1/2/3 n-gram ratios + longest consecutive-repeated-word run, reusing
      `_ngram_distinct` / `_max_repeat_run` verbatim.
  (3) LEVER sanity PER SEED: `ro.head_w` must actually differ between native and learned arms (else the
      "learned" arm silently ran on native -- catches both a broken flag AND a broken path); the loader's own
      provenance dict must read `applied=True` (never a silent fail-safe fallback) on every learned-arm call,
      for EVERY one of the 6 seeds -- this is the regression pin for the rung-1 path bug.
  (4) RNG discipline: host process-global numpy RNG state byte-identical before/after the WHOLE 6-seed run.

HONEST FRAMING (unchanged from the single-seed A/B, restated so this finding is not read as a different claim):
the learned head recovers ~93% mean / ~89% worst-case of the native head's own recovery of the checkpoint's
target (`sub_recov_ratio`), so it is EXPECTED to generate somewhat worse than native on average, not better. A
GO on this runner's Verdict means "the fixed default path genuinely loads a distinct, per-seed-correct learned
head everywhere, and that head generates coherently (decisively separated from chance) on every seed" -- it is
NOT a "learned beats native" claim, and per this repo's own convention (a multi-seed A/B is still a SINGLE
lever/config point) does not by itself license a `BRAIN_WKV_MOUTH_LEARNED_HEAD` default-on flip.

CPU/numpy only, no GPU needed (6x the single-seed A/B's ~1.5s workload). Detached-run friendly: prints progress
per (seed, prompt).

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_learned_vs_native_head_ab_6seed
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools.verdict import Verdict  # noqa: E402
from tools.lab import lever, void_if  # noqa: E402
# Reuse VERBATIM -- same prompts, same scoring convention as the single-seed A/B (do not re-derive/drift).
from research.runners._wkv_learned_vs_native_head_ab import (  # noqa: E402
    PROMPTS, _self_nll, _ngram_distinct, _max_repeat_run, _sha1,
)

SEEDS = [42, 43, 44, 100, 101, 102]           # the 6-seed non-negotiable (CLAUDE.md)
MAX_NEW_TOKENS = 50
READ_WINDOW = 40
POP = 8
TOPK = 64
GEN_TEMP = 0.8

OUT = _REPO / "research" / "findings" / "raw" / "_wkv_learned_vs_native_head_ab_6seed.json"


def _run_arm(W, prompt: str, seed: int, learned: bool) -> dict:
    os.environ["BRAIN_WKV_MOUTH_LEARNED_HEAD"] = "1" if learned else "0"
    ro, vocab, word_to_id = W._get_readout(seed)
    head_hash = _sha1(ro.head_w)
    status = W.learned_head_status(seed) if learned else None
    text, secs = W.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS, topk=TOPK,
                             read_window=READ_WINDOW, pop=POP, gen_temp=GEN_TEMP)
    cont = text[len(prompt):].strip() if text.startswith(prompt) else text
    cont_words = cont.split()
    self_nll, n_scored = _self_nll(ro, word_to_id, text)
    return {
        "prompt": prompt, "seed": seed, "learned": learned, "text": text, "continuation": cont,
        "gen_seconds": secs, "head_hash": head_hash, "learned_head_status": status,
        "learned_head_path": (status or {}).get("path"),
        "self_nll": self_nll, "n_words_scored": n_scored, "n_continuation_words": len(cont_words),
        "distinct_1": _ngram_distinct(cont_words, 1), "distinct_2": _ngram_distinct(cont_words, 2),
        "distinct_3": _ngram_distinct(cont_words, 3), "max_repeat_run": _max_repeat_run(cont_words),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def main():
    # Deliberately NOT setting BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH -- this run exercises the module's own DEFAULT
    # template (the exact thing rung-1 fixed) end to end, for every one of the 6 seeds.
    os.environ.pop("BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH", None)
    from webapp import wkv_mouth_generator as W  # noqa: E402  (deliberately late: import-time-fixed constant)

    # Anti-cheat: the 6 per-seed npz files must NOT be byte-identical (a repeat of the old overwrite bug would
    # make every "seed-distinct" head secretly the same file).
    npz_hashes = {}
    for seed in SEEDS:
        p = Path(W._learned_head_path(seed))
        void_if(not p.exists(), f"seed={seed}: default learned-head path does not exist: {p} "
                                 f"-- THIS IS THE RUNG-1 REGRESSION THIS RUNNER EXISTS TO CATCH")
        npz_hashes[seed] = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    n_distinct = len(set(npz_hashes.values()))
    print(f"[ab6] per-seed learned-head npz SHA1s: {npz_hashes}  (distinct={n_distinct}/{len(SEEDS)})")

    host_rng_before = np.random.get_state()[1].copy()

    per_seed = {}
    t0 = time.time()
    for si, seed in enumerate(SEEDS):
        native_runs, learned_runs = [], []
        for i, p in enumerate(PROMPTS):
            r_native = _run_arm(W, p, seed, learned=False)
            r_learned = _run_arm(W, p, seed, learned=True)
            native_runs.append(r_native)
            learned_runs.append(r_learned)
        applied_flags = [(r["learned_head_status"] or {}).get("applied") for r in learned_runs]
        reasons = [(r["learned_head_status"] or {}).get("reason") for r in learned_runs]
        native_hashes = {r["head_hash"] for r in native_runs}
        learned_hashes = {r["head_hash"] for r in learned_runs}
        heads_differ = (native_hashes != learned_hashes and len(native_hashes) == 1 and len(learned_hashes) == 1)
        lever(f"seed={seed} head_w hash native vs learned", next(iter(native_hashes)), next(iter(learned_hashes)))
        native_nll_mean = _mean([r["self_nll"] for r in native_runs])
        learned_nll_mean = _mean([r["self_nll"] for r in learned_runs])
        native_maxrun_mean = _mean([r["max_repeat_run"] for r in native_runs])
        learned_maxrun_mean = _mean([r["max_repeat_run"] for r in learned_runs])
        n_learned_wins = sum(
            1 for rn, rl in zip(native_runs, learned_runs)
            if rn["self_nll"] is not None and rl["self_nll"] is not None and rl["self_nll"] < rn["self_nll"]
        )
        per_seed[seed] = {
            "seed": seed, "npz_sha1": npz_hashes[seed], "heads_differ": heads_differ,
            "all_learned_applied": all(f is True for f in applied_flags),
            "learned_reasons": reasons,
            "native_self_nll_mean": native_nll_mean, "learned_self_nll_mean": learned_nll_mean,
            "native_max_repeat_run_mean": native_maxrun_mean, "learned_max_repeat_run_mean": learned_maxrun_mean,
            "n_prompts": len(PROMPTS), "n_learned_wins_of_8": n_learned_wins,
            "native_runs": native_runs, "learned_runs": learned_runs,
        }
        print(f"[ab6] seed={seed} ({si+1}/{len(SEEDS)})  native_nll={native_nll_mean:.3f}  "
              f"learned_nll={learned_nll_mean:.3f}  applied={all(f is True for f in applied_flags)}  "
              f"heads_differ={heads_differ}  wins={n_learned_wins}/8  elapsed={time.time()-t0:.1f}s")

    host_rng_after = np.random.get_state()[1].copy()
    rng_untouched = bool((host_rng_before == host_rng_after).all())

    all_heads_differ = all(v["heads_differ"] for v in per_seed.values())
    all_applied = all(v["all_learned_applied"] for v in per_seed.values())
    chance_nll = float(np.log(1000))
    native_means = [v["native_self_nll_mean"] for v in per_seed.values()]
    learned_means = [v["learned_self_nll_mean"] for v in per_seed.values()]
    native_6seed_mean = _mean(native_means)
    learned_6seed_mean = _mean(learned_means)
    learned_6seed_worst = max(learned_means) if all(m is not None for m in learned_means) else None
    total_wins = sum(v["n_learned_wins_of_8"] for v in per_seed.values())

    art = {
        "probe": "wkv_learned_vs_native_head_ab_6seed", "backend": "numpy", "seeds": SEEDS,
        "n_prompts_per_seed": len(PROMPTS), "max_new_tokens": MAX_NEW_TOKENS, "read_window": READ_WINDOW,
        "per_seed": per_seed, "npz_sha1_by_seed": npz_hashes, "n_distinct_npz": n_distinct,
        "chance_nll": chance_nll,
        "native_self_nll_6seed_mean": native_6seed_mean, "learned_self_nll_6seed_mean": learned_6seed_mean,
        "learned_self_nll_6seed_worst": learned_6seed_worst,
        "all_heads_differ_6of6": all_heads_differ, "all_learned_applied_6of6": all_applied,
        "rng_untouched_across_run": rng_untouched,
        "total_learned_wins_of_48": total_wins,
        "elapsed_s": round(time.time() - t0, 1),
    }

    v = Verdict("the e-prop LOCALLY-LEARNED WKV-mouth read-out head, loaded via the FIXED default per-seed path "
                "template, generates coherently through the production entry point on EVERY one of 6 seeds "
                "(rung-1 path-fix regression pin + quality A/B, not a default-on claim)")
    v.require("(anti-cheat) the 6 per-seed npz artifacts are NOT byte-identical (not a repeat of the prior "
              "single-seed overwrite bug)", n_distinct == len(SEEDS), expect=True)
    v.require("(lever, 6/6) native and learned head_w actually differ on every seed",
              all_heads_differ, expect=True)
    v.require("(fail-safe, 6/6) the learned head loader reports applied=True on every learned-arm call, every "
              "seed -- THE RUNG-1 REGRESSION PIN (a wrong default path shows up here as applied=False)",
              all_applied, expect=True)
    v.require("(RNG) host process-global numpy RNG state is byte-identical before/after the whole 6-seed run",
              rng_untouched, expect=True)
    if learned_6seed_worst is not None:
        v.control("learned-head self-NLL vs chance, WORST seed (not just mean)",
                  treatment=chance_nll - learned_6seed_worst, control=0.0, min_separation=2.0,
                  note=f"worst-seed learned self_nll={learned_6seed_worst:.3f} nats vs chance={chance_nll:.3f}")
    else:
        v.require("learned-head self-NLL was measurable on every seed", False, expect=True)

    go = (n_distinct == len(SEEDS) and all_heads_differ and all_applied and rng_untouched
          and learned_6seed_worst is not None and (chance_nll - learned_6seed_worst) > 2.0)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["GO"] = bool(go)
    art["preconditions"] = decided.get("preconditions", [])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=1))
    print(json.dumps({
        "native_self_nll_6seed_mean": native_6seed_mean, "learned_self_nll_6seed_mean": learned_6seed_mean,
        "learned_self_nll_6seed_worst": learned_6seed_worst, "chance_nll": chance_nll,
        "all_heads_differ_6of6": all_heads_differ, "all_learned_applied_6of6": all_applied,
        "rng_untouched_across_run": rng_untouched, "total_learned_wins_of_48": total_wins, "GO": go,
    }, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
