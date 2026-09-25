"""Seed-7 DEV CHECK of the frame-junction referent lexicon (pre-registration
research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md, AMENDMENT 1). Seed 7 is not an
evaluation seed; nothing here carries evaluation weight, and no six-seed run or default flip happens here.

Steps (one process; numpy backend):
  D0 DEFAULT OFF, asserted in data: `BRAIN_LEARNED_REFERENT_JUNCTION` unset -> the parse instrument at seed 7 must
     reproduce the pinned pre-change hashes (diag_frame_s7_gt3.json) exactly.
  D1 JUNCTION LEXICON at seed 7 (flag set): build + train, curriculum accuracy without the teacher (report only).
  D2 AND SMOKE at the frozen constants on 256 sampled junctions of the trained circuit; and the `coincidence`
     lesion must make a lone afferent fire its junction (OR).
  D3 PARSE (the G2/G3 instrument): intact and `coincidence` arms over the 112 battery turns.
  D4 ROUTE (the G1 instrument's logic): `_d6_learned_referent_env_flag_derisk.run_seed(7, ...)` with the route's
     lexicon singleton pinned to THIS seed-7 junction lexicon. The production singleton trains at seed 42, so the
     harness patches `get_lexicon` (declared in the pre-registration; dev only) instead of touching seed 42.

  bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_junction_dev --corpus /path/to/tinystories.txt \
      --out research/findings/raw/_lexicon_closed_class/dev_s7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEV_SEED = 7
PINNED_OFF = {"parse_sha256": "81737f9706d815e56244e7e1886aa622617fe72cc22a5cc780626b67a2bc0d29",
              "decisions_sha256": "94a29a45b55112beab7383cb7ef6b1749593a2e3b2ee602aa5d1d7d89dd1871d"}


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=str)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--out", default="research/findings/raw/_lexicon_closed_class/dev_s7")
    ap.add_argument("--skip-route", action="store_true")
    a = ap.parse_args()
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(_REPO, a.out)
    from research.runners import lexicon_spiking_frame_category as L
    from research.runners import _lexicon_closed_class_parse_diag as P
    summary = {"seed": DEV_SEED, "backend": os.environ.get("SIM_BACKEND"), "corpus_path": corpus}
    summary["corpus_sha256"], summary["corpus_bytes"] = P._sha256(corpus)
    t0 = time.time()

    # D0 default OFF
    os.environ.pop("BRAIN_LEARNED_REFERENT_JUNCTION", None)
    off = P.run(DEV_SEED, corpus)
    _dump(os.path.join(out_dir, "off_frame_s7.json"), off)
    summary["D0_default_off"] = {
        "parse_sha256": off["parse_sha256"], "decisions_sha256": off["decisions_sha256"], "variant": off["variant"],
        "identical_to_pinned": bool(off["parse_sha256"] == PINNED_OFF["parse_sha256"]
                                    and off["decisions_sha256"] == PINNED_OFF["decisions_sha256"]),
        "junction_module_imported": "research.runners.lexicon_frame_junction" in sys.modules}
    print("D0", summary["D0_default_off"], flush=True)
    from research.runners import lexicon_frame_junction as J
    summary["constants"] = {"W_J": J.W_J, "I_TONIC_J": J.I_TONIC_J, "T_ON_J": J.T_ON_J,
                            "DRIVE_MATCH_S": J.DRIVE_MATCH_S, "W_INIT_J": J.W_INIT_J, "ETA_J": J.ETA_J,
                            "OJA_BETA_J": J.OJA_BETA_J, "OR_LESION_FACTOR": J.OR_LESION_FACTOR}

    # D1 + D3: junction lexicon, parse arms (P.run builds and trains through get_lexicon)
    os.environ["BRAIN_LEARNED_REFERENT_JUNCTION"] = "1"
    t1 = time.time()
    jr = P.run(DEV_SEED, corpus, lesions=(None, "coincidence"))
    _dump(os.path.join(out_dir, "junction_s7.json"), jr)
    lex = L._LEXICON
    assert lex is not None and getattr(lex, "variant", "") == "junction", "the junction lexicon was not built"
    ai, al = jr["arms"]["intact"], jr["arms"]["coincidence"]
    summary["D3_parse"] = {
        "build_train_s": jr["build_train_s"], "elapsed_s": round(time.time() - t1, 1), "peak_rss_mb": jr["peak_rss_mb"],
        "intact": {k: ai[k] for k in ("n_changed", "n_mismatch", "mismatch_labels", "offending_words",
                                      "unknown_admits", "unexplained_drops", "new_gt_nouns_recovered",
                                      "tom_fb_on", "tom_fb_anne_kept", "admitted_by_decision")},
        "coincidence": {k: al[k] for k in ("n_changed", "n_mismatch", "mismatch_labels", "offending_words",
                                           "unknown_admits", "new_gt_nouns_recovered", "admitted_by_decision")},
        "g3_lever_moved": bool(al["n_mismatch"] > ai["n_mismatch"])}
    # ATTRIBUTION. (a) the coincidence lesion is the G3 lever: did removing the AND move the mismatch count?
    # (b) how much of the single-offset lexicon's mismatch count (D0, the v2 arm at this seed) does the junction
    #     variant remove? treatment = v2 mismatches, control = junction-intact mismatches.
    from tools.lab import attributable_to, lever
    summary["D3_parse"]["g3_lever_recorded"] = bool(lever(
        "coincidence lesion -> battery parse mismatches (seed 7)", ai["n_mismatch"], al["n_mismatch"],
        required=False))
    summary["D3_parse"]["fraction_of_v2_mismatches_removed"] = attributable_to(
        "v2 single-offset vs junction-intact battery parse mismatches (seed 7)", off["arms"]["intact"]["n_mismatch"],
        ai["n_mismatch"])
    print("D3", json.dumps(summary["D3_parse"], default=str), flush=True)

    # D1 curriculum accuracy without the teacher (report only)
    lex.set_lesion(None)
    words, labels = L.seed_curriculum(lex.env)
    dec = [lex.decide(w)[0][0] for w in words]
    correct = sum(1 for d, lab in zip(dec, labels) if d is not None and d == (lab > 0))
    summary["D1_curriculum"] = {"n": len(words), "correct": int(correct),
                                "abstain": int(sum(d is None for d in dec)),
                                "wrong": [w for w, d, lab in zip(words, dec, labels) if d is not None and d != (lab > 0)]}
    print("D1", summary["D1_curriculum"], flush=True)

    # D2 AND smoke on the trained circuit + the OR lesion
    sm = J.and_smoke(lex, n_sample=256, seed=1)
    lex.set_lesion("coincidence")
    rng = np.random.default_rng(2)
    pairs = [(int(x), int(y)) for x, y in zip(rng.integers(0, lex.C, 64), rng.integers(0, lex.C, 64))]
    lone, _ = lex.junction_response(pairs, "left")
    lex.set_lesion(None)
    summary["D2_and_smoke"] = {**sm, "coincidence_lesion_lone_left_fired": int((lone > 0).sum()),
                               "coincidence_lesion_n": len(pairs)}
    print("D2", summary["D2_and_smoke"], flush=True)

    # D4 route logic with the seed-7 junction lexicon pinned as the singleton
    if not a.skip_route:
        from research.runners import _d6_learned_referent_env_flag_derisk as R
        from research.runners._lexicon_learned_referent_derisk import FIXTURE
        trained = lex
        orig_get = L.get_lexicon

        def _pinned(*_a, **_k):
            L._LEXICON = trained
            return trained
        L.get_lexicon = _pinned
        L._DEFAULT_CORPUS = corpus          # the worktree has no data/ copy; record the file the lexicon read
        try:
            route = R.run_seed(DEV_SEED, corpus, json.load(open(FIXTURE))["pos"])
        finally:
            L.get_lexicon = orig_get
        _dump(os.path.join(out_dir, "route_s7.json"), route)
        summary["D4_route"] = {k: route.get(k) for k in ("r1_pass", "r1_input_order", "r2_pass",
                                                        "r3_recovered_both_rate", "r3_pass",
                                                        "r4_lesion_recovered_both_rate", "r4_lever_moved", "r4_pass",
                                                        "hand_baseline_recovered_both_rate", "elapsed_s")}
        print("D4", summary["D4_route"], flush=True)

    os.environ.pop("BRAIN_LEARNED_REFERENT_JUNCTION", None)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    try:
        import resource
        summary["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001
        summary["peak_rss_mb"] = None
    _dump(os.path.join(out_dir, "dev_s7_summary.json"), summary)


if __name__ == "__main__":
    main()
