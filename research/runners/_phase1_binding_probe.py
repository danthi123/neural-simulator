"""Phase-1 word→pool BINDING probe — the reusable instrument for the 2026-07-25 binding investigation.

WHY THIS EXISTS: the investigation into why word→pool binding fails burned several hours on measurements that could
not have detected the effect they were asked about — three separate FLOORED comparisons (a 1-word/200-event weight
delta, a 200-event old-vs-new A/B, and a saturating learning rate at the other extreme). Each produced a confident,
wrong conclusion that had to be retracted. This probe exists so that never happens silently again:

  * it reports a GRADED margin alongside the binary 16-way accuracy (a continuous score discriminates far below the
    training budget at which a 16-way argmax starts to win, so it can serve as a cheap bisect instrument);
  * it prints the RAW magnitudes (target rate, best non-target rate, and how many words produced ANY pool activity),
    so a floored / saturated run is visible immediately rather than being read as a null;
  * it emits an explicit FLOOR WARNING when the measurement cannot discriminate, so its own output cannot be
    mistaken for evidence.

It builds via the REFERENCE harness's own builder (`unified_per_regime_monitor_runner._build_bridge_with_phase1_recipe`)
so results are comparable to the recorded 87.5%-at-800ev figure, and it runs the same `train_word_to_pool` protocol.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._phase1_binding_probe --events 800
  # against an older checkout (bisect):
  SIM_BACKEND=cupy PYTHONPATH=/path/to/worktree python -m research.runners._phase1_binding_probe --events 800
"""
import os, sys, json, time, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
import numpy as np
import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe, _phase1_recipe, _phase1_train_kwargs,
    _all_words_word_to_idx, _N_WORDS_ORTHOGONAL)


def _targets():
    t = []
    for w, a in cpd.DIRECTION_VOCAB.items():  t.append((w, "motor_%s" % a))
    for w, n in cpd.NOUN_VOCAB.items():       t.append((w, "noun_pool_%s" % n))
    for w, n in cpd.VERB_VOCAB.items():       t.append((w, "verb_pool_%s" % n))
    for w, n in cpd.ADJECTIVE_VOCAB.items():  t.append((w, "adjective_pool_%s" % n))
    return t


def _pools():
    return (["motor_%s" % a for a in ("N", "E", "S", "W")]
            + ["noun_pool_%s" % n for n in cpd.NOUN_NAMES]
            + ["verb_pool_%s" % v for v in cpd.VERB_NAMES]
            + ["adjective_pool_%s" % a for a in cpd.ADJECTIVE_NAMES])


def run(seed=42, events=800, n_words=None, verbose=True):
    b0 = _build_bridge_with_phase1_recipe(int(seed), False)
    b = b0[0] if isinstance(b0, (tuple, list)) else b0
    dims = _phase1_recipe(False); tk = _phase1_train_kwargs(False)
    allw, w2i = _all_words_word_to_idx(); nw = max(_N_WORDS_ORTHOGONAL, len(allw))
    targets = _targets()
    if n_words:                       # reduced word set = cheaper bisect step
        targets = targets[:int(n_words)]
    tset = {w for w, _ in targets}
    rng = np.random.default_rng(int(seed))
    buf = [(w, t) for (w, t) in targets for _ in range(int(events))]
    rng.shuffle(buf)
    t0 = time.time()
    for i, (w, t) in enumerate(buf):
        cpd.train_word_to_pool(b, w, t, n_events=1, reset_steps=50,
                               n_lang_input=int(tk["n_lang_input"]), n_lang_output=int(tk["n_lang_input"]),
                               sparsity=float(tk["sparsity"]), orthogonal_codes=True,
                               n_words_for_orthogonal=int(nw), word_to_idx=w2i, verbose=False)
        if verbose and (i + 1) % 2000 == 0:
            print(f"    train {i+1}/{len(buf)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    pools = _pools()
    tgt_of = dict(targets)
    ok, margins, tgt_rates, best_other, n_any = 0, [], [], [], 0
    for w, t in targets:
        per = cpd.measure_pool_firing(b, w, pools, stim_steps=100, reset_steps=50, drive_pA=200.0,
                                      sparsity=0.05, n_lang_input=int(dims["n_lang_input"]),
                                      orthogonal_codes=True, n_words_for_orthogonal=int(nw), word_to_idx=w2i)
        tr = float(per.get(t, 0.0))
        others = [v for k, v in per.items() if k != t]
        bo = float(max(others)) if others else 0.0
        ok += (max(per.items(), key=lambda kv: kv[1])[0] == t)
        margins.append(tr - bo); tgt_rates.append(tr); best_other.append(bo)
        n_any += (max(per.values()) > 0)

    n = len(targets)
    res = dict(seed=int(seed), events=int(events), n_words=n,
               accuracy=round(ok / n, 4), n_correct=int(ok),
               mean_margin=round(float(np.mean(margins)), 5),
               mean_target_rate=round(float(np.mean(tgt_rates)), 5),
               mean_best_other=round(float(np.mean(best_other)), 5),
               words_with_any_activity=int(n_any), minutes=round((time.time() - t0) / 60, 1))
    # FLOOR / CEILING self-check — this probe must never be read as evidence when it cannot discriminate
    floored = res["mean_target_rate"] < 1e-4 and res["mean_best_other"] < 1e-4
    dead = res["words_with_any_activity"] < max(2, n // 4)
    res["can_discriminate"] = bool(not floored and not dead)
    res["floor_warning"] = ("ALL pool rates ~0 — measurement is FLOORED, its result is NOT evidence"
                            if floored else
                            ("most words produce NO pool activity — near-floored, treat with suspicion"
                             if dead else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--events", type=int, default=800, help="events per word (recorded 87.5%% used 800; 200 is FLOORED)")
    ap.add_argument("--n-words", type=int, default=None, help="reduced word set for a cheaper bisect step")
    ap.add_argument("--out", default="research/findings/raw/phase1_binding")
    args = ap.parse_args()
    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)
    r = run(args.seed, args.events, args.n_words)
    Path(f"{args.out}/binding_e{args.events}_w{r['n_words']}_seed{args.seed}.json").write_text(json.dumps(r, indent=2))
    print(f"[seed {args.seed}] events/word={r['events']} words={r['n_words']} ({r['minutes']} min)")
    print(f"  ACCURACY (16-way argmax): {r['n_correct']}/{r['n_words']} = {100*r['accuracy']:.1f}%   [record: 87.5% @800ev]")
    print(f"  GRADED MARGIN (target - best other): {r['mean_margin']:+.5f}   <- continuous; discriminates below argmax")
    print(f"  raw: mean target rate={r['mean_target_rate']}  mean best-other={r['mean_best_other']}  "
          f"words with ANY activity={r['words_with_any_activity']}/{r['n_words']}")
    if r["floor_warning"]:
        print(f"  ⚠️  {r['floor_warning']}")
    print(f"  CAN DISCRIMINATE: {r['can_discriminate']}"
          + ("" if r["can_discriminate"] else "  <- do NOT use this run as evidence for or against anything"))
    print("PHASE1-BINDING-PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
