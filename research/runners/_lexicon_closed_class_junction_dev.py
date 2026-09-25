"""DEV CHECK of the frame-junction referent lexicon (pre-registration
research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md, AMENDMENTS 1 + 2). Runs at a
GIVEN dev seed (7 by default; AMENDMENT 2 also runs this at 42, the PRODUCTION seed, per review issue #5 -- seed 7
alone never exercised what actually deploys). Neither seed is an evaluation seed; nothing here carries evaluation
weight, and no six-seed run or default flip happens here.

Steps (one process; numpy backend):
  D0 DEFAULT OFF, asserted in data: `BRAIN_LEARNED_REFERENT_JUNCTION` unset -> the parse instrument at this seed
     must reproduce the pinned pre-change hashes (`PINNED_OFF`, seed 7 only -- pinned before any mechanism build;
     seed 42 has no such pin, so D0 there only asserts variant=="frame" and junction_module_imported==False).
  D1 JUNCTION LEXICON at this seed (flag set): build + train, curriculum accuracy without the teacher (report only).
  D2 AND SMOKE at the frozen constants: `and_smoke` (256 sampled junctions) AND `and_population` (AMENDMENT 2,
     ALL 10,000 junctions -- the 256-sample missed the 'day'-column violation entirely) on the trained circuit;
     the `coincidence` lesion must make a lone afferent fire its junction (OR).
  D3 PARSE (the G2/G3 instrument): intact, `coincidence` and `coincidence_matched` (AMENDMENT 2 drive-matched OR
     control) arms over the 112 battery turns, token-level ground truth, per-word margins, silent-NON reporting.
  D4 ROUTE (the G1 instrument's logic): `_d6_learned_referent_env_flag_derisk.run_seed(seed, ...)` with the route's
     lexicon singleton pinned to THIS seed's junction lexicon. The production singleton trains at seed 42, so the
     harness patches `get_lexicon` (declared in the pre-registration; dev only) when seed != 42.

  bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_junction_dev --seed 7 --corpus /path/to/tinystories.txt \
      --out research/findings/raw/_lexicon_closed_class/dev_s7_amendment2
  bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_junction_dev --seed 42 --corpus /path/to/tinystories.txt \
      --out research/findings/raw/_lexicon_closed_class/dev_s42_amendment2

AMENDMENT 3 (`--elemental`): the junction lexicon WITH the elemental partial-match edge
(BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL=1 set in-process after D0). D3's arms become intact + the two G3' lesions
(`elemental`, `conjunctive`); the 2x `coincidence` arms are not run for this variant. D2 adds: under `conjunctive`
no junction fires to its pair. The summary reports G2, G3'a, G3'b (with the drive condition), G4 and G1 R1-R4.
  bash tools/mem_ok.sh 4 && bash tools/memcap.sh 4 -- env SIM_BACKEND=numpy python -u -m \
      research.runners._lexicon_closed_class_junction_dev --elemental --seed 7 --corpus /path/to/tinystories.txt \
      --out research/findings/raw/_lexicon_closed_class/dev_s7_amendment3
Without --elemental the script runs exactly AMENDMENT 2's dev check.

NOTE ON --out: use a NEW directory per amendment, never the prior round's `dev_s<seed>/` -- round 1's
`dev_s7/{off_frame_s7,junction_s7,route_s7,dev_s7_result}.json` are cited by
research/findings/2026-09-24-lexicon-closed-class-frame-junction-dev-s7-not-ready.md's own frontmatter; reusing
that directory overwrites the artifacts a committed finding points to (caught the hard way: AMENDMENT 2's first
run did exactly this before it was reverted).
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

PINNED_OFF = {7: {"parse_sha256": "81737f9706d815e56244e7e1886aa622617fe72cc22a5cc780626b67a2bc0d29",
                  "decisions_sha256": "94a29a45b55112beab7383cb7ef6b1749593a2e3b2ee602aa5d1d7d89dd1871d"}}


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=str)
    # `--out` names a DIRECTORY, which the provenance door cannot sidecar on its own (it stamps FILES named by an
    # output flag); declare each written file so it gets a `.prov.json` at exit (AMENDMENT 3 dev check, found when
    # its artifacts came back unstamped; they were backfilled by _lexicon_closed_class_a3_backfill_prov.py).
    try:
        from research.runners import declare_output
        declare_output(path)
    except Exception:  # noqa: BLE001 -- provenance must never be why a run dies
        pass
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", default=os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    ap.add_argument("--out", default="research/findings/raw/_lexicon_closed_class/dev_s7")
    ap.add_argument("--skip-route", action="store_true")
    ap.add_argument("--elemental", action="store_true",
                    help="AMENDMENT 3: the junction lexicon with the elemental partial-match edge; G3' arms")
    a = ap.parse_args()
    ELEM_ENV = "BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL"
    seed = a.seed
    corpus = a.corpus if os.path.isabs(a.corpus) else os.path.join(_REPO, a.corpus)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(_REPO, a.out)
    from research.runners import lexicon_spiking_frame_category as L
    from research.runners import _lexicon_closed_class_parse_diag as P
    summary = {"seed": seed, "backend": os.environ.get("SIM_BACKEND"), "corpus_path": corpus,
              "git_sha": P._git_sha()}
    summary["corpus_sha256"], summary["corpus_bytes"] = P._sha256(corpus)
    t0 = time.time()

    # D0 default OFF. IMPORTANT: `lexicon_frame_junction` must NOT be imported (by this script or anything it calls)
    # before this check runs, or "junction_module_imported" trivially reads True from OUR OWN later import instead
    # of from get_lexicon()'s own variant choice (caught the hard way: importing it above for the constants block
    # made D0 assert-fail on every seed the first time this ran).
    os.environ.pop("BRAIN_LEARNED_REFERENT_JUNCTION", None)
    os.environ.pop(ELEM_ENV, None)
    off = P.run(seed, corpus)
    _dump(os.path.join(out_dir, f"off_frame_s{seed}.json"), off)
    pinned = PINNED_OFF.get(seed)
    summary["D0_default_off"] = {
        "parse_sha256": off["parse_sha256"], "decisions_sha256": off["decisions_sha256"], "variant": off["variant"],
        "junction_module_imported": "research.runners.lexicon_frame_junction" in sys.modules}
    if pinned is not None:
        summary["D0_default_off"]["identical_to_pinned"] = bool(
            off["parse_sha256"] == pinned["parse_sha256"] and off["decisions_sha256"] == pinned["decisions_sha256"])
    else:
        summary["D0_default_off"]["identical_to_pinned"] = None
        summary["D0_default_off"]["note"] = f"no pinned pre-change hash exists for seed {seed}; asserting " \
            "variant=='frame' and the junction module stayed unimported is the only default-off check available here"
    assert off["variant"] == "frame" and not summary["D0_default_off"]["junction_module_imported"], \
        "default-off must be the v2 lexicon with the junction module unimported"
    print("D0", summary["D0_default_off"], flush=True)
    from research.runners import lexicon_frame_junction as J   # AFTER the D0 check (see the comment above)
    summary["constants"] = {"W_J": J.W_J, "I_TONIC_J": J.I_TONIC_J, "T_ON_J": J.T_ON_J,
                            "DRIVE_MATCH_S": J.DRIVE_MATCH_S, "W_INIT_J": J.W_INIT_J, "ETA_J": J.ETA_J,
                            "OJA_BETA_J": J.OJA_BETA_J, "OR_LESION_FACTOR": J.OR_LESION_FACTOR,
                            "OR_MATCH_FACTOR": J.OR_MATCH_FACTOR, "STP_ENABLED": J.STP_ENABLED}
    if a.elemental:
        summary["constants"].update({"ELEMENTAL": True, "W_INIT_E": J.W_INIT_E, "ETA_E": J.ETA_E,
                                     "OJA_BETA_E": J.OJA_BETA_E, "ELEMENTAL_JITTER_SEED": J.ELEMENTAL_JITTER_SEED})

    # D1 + D3: junction lexicon, parse arms (P.run builds and trains through get_lexicon)
    os.environ["BRAIN_LEARNED_REFERENT_JUNCTION"] = "1"
    if a.elemental:
        os.environ[ELEM_ENV] = "1"
    _keys = ("n_changed", "n_mismatch", "mismatch_labels", "offending_words", "unknown_admits",
             "new_gt_nouns_recovered", "admitted_by_decision", "admitted_margins",
             "n_non_heard", "silent_non_words", "silent_non_fraction")
    t1 = time.time()
    if a.elemental:
        jr = P.run(seed, corpus, lesions=(None, "elemental", "conjunctive"))
        _dump(os.path.join(out_dir, f"junction_elemental_s{seed}.json"), jr)
        lex = L._LEXICON
        assert lex is not None and getattr(lex, "variant", "") == "junction_elemental", \
            "the junction_elemental lexicon was not built"
        ai = jr["arms"]["intact"]
        _dkeys = _keys + ("mean_afferent_drive", "n_drive_words")
        g3p = P.g3prime_seed(jr["arms"])
        summary["D3_parse"] = {
            "build_train_s": jr["build_train_s"], "elapsed_s": round(time.time() - t1, 1),
            "peak_rss_mb": jr["peak_rss_mb"],
            "intact": {k: ai[k] for k in _dkeys + ("unexplained_drops", "tom_fb_on", "tom_fb_anne_kept")},
            "elemental": {k: jr["arms"]["elemental"][k] for k in _dkeys},
            "conjunctive": {k: jr["arms"]["conjunctive"][k] for k in _dkeys},
            "G2_pass": bool(ai["n_mismatch"] == 0),
            "G3prime": g3p,
            "G4_pass": bool(ai["silent_non_fraction"] is not None
                            and ai["silent_non_fraction"] <= P.G4_SILENT_NON_MAX),
            "v2_silent_non_fraction": off["arms"]["intact"]["silent_non_fraction"],
            "v2_silent_non_words": off["arms"]["intact"]["silent_non_words"],
            "v2_n_mismatch": off["arms"]["intact"]["n_mismatch"],
            "v2_offending_words": off["arms"]["intact"]["offending_words"]}
        from tools.lab import attributable_to
        summary["D3_parse"]["fraction_of_v2_mismatches_removed"] = attributable_to(
            f"v2 single-offset vs junction_elemental-intact battery parse mismatches (seed {seed})",
            off["arms"]["intact"]["n_mismatch"], ai["n_mismatch"])
        print("D3", json.dumps({k: v for k, v in summary["D3_parse"].items()
                                if k not in ("intact", "elemental", "conjunctive")}, default=str), flush=True)
    else:
        jr = P.run(seed, corpus, lesions=(None, "coincidence", "coincidence_matched"))
        _dump(os.path.join(out_dir, f"junction_s{seed}.json"), jr)
        lex = L._LEXICON
        assert lex is not None and getattr(lex, "variant", "") == "junction", "the junction lexicon was not built"
        _amendment2_d3(summary, jr, off, seed, t1, _keys)

    # a partial summary now (the parse arms are the expensive part), overwritten by the final one below
    _dump(os.path.join(out_dir, f"dev_s{seed}_result.json"), {**summary, "partial": True})
    _d1_d2_d4(summary, lex, J, L, a, seed, corpus, out_dir)
    os.environ.pop("BRAIN_LEARNED_REFERENT_JUNCTION", None)
    os.environ.pop(ELEM_ENV, None)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    try:
        import resource
        summary["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001
        summary["peak_rss_mb"] = None
    # NAMED "_result", not "_summary": `.gitignore` excludes `*_summary.json` repo-wide, which silently dropped
    # this exact file from version control in round 1 (worked around there by a manual rename before commit,
    # replicated here in the filename itself so a future run does not need the same manual step).
    _dump(os.path.join(out_dir, f"dev_s{seed}_result.json"), summary)


def _amendment2_d3(summary, jr, off, seed, t1, _keys):
    """AMENDMENT 2's D3 summary (unchanged; the round-2 arms)."""
    ai, al, am = jr["arms"]["intact"], jr["arms"]["coincidence"], jr["arms"]["coincidence_matched"]
    summary["D3_parse"] = {
        "build_train_s": jr["build_train_s"], "elapsed_s": round(time.time() - t1, 1), "peak_rss_mb": jr["peak_rss_mb"],
        "intact": {k: ai[k] for k in _keys + ("unexplained_drops", "tom_fb_on", "tom_fb_anne_kept")},
        "coincidence": {k: al[k] for k in _keys},
        "coincidence_matched": {k: am[k] for k in _keys},
        "g3_lever_moved": bool(al["n_mismatch"] > ai["n_mismatch"]),
        "g3_matched_lever_moved": bool(am["n_mismatch"] > ai["n_mismatch"])}
    # ATTRIBUTION. (a) the coincidence lesion is the G3 lever: did removing the AND move the mismatch count?
    # (b) how much of the single-offset lexicon's mismatch count (D0, the v2 arm at this seed) does the junction
    #     variant remove? treatment = v2 mismatches, control = junction-intact mismatches.
    from tools.lab import attributable_to, lever
    summary["D3_parse"]["g3_lever_recorded"] = bool(lever(
        f"coincidence lesion -> battery parse mismatches (seed {seed})", ai["n_mismatch"], al["n_mismatch"],
        required=False))
    summary["D3_parse"]["g3_matched_lever_recorded"] = bool(lever(
        f"coincidence_matched (drive-matched OR) lesion -> battery parse mismatches (seed {seed})",
        ai["n_mismatch"], am["n_mismatch"], required=False))
    summary["D3_parse"]["fraction_of_v2_mismatches_removed"] = attributable_to(
        f"v2 single-offset vs junction-intact battery parse mismatches (seed {seed})",
        off["arms"]["intact"]["n_mismatch"], ai["n_mismatch"])
    # AMENDMENT 2 G2-2: silent-NON comparison against v2 AT THE SAME SEED (never an absolute/cross-seed bar).
    summary["D3_parse"]["v2_silent_non_fraction"] = off["arms"]["intact"]["silent_non_fraction"]
    summary["D3_parse"]["v2_silent_non_words"] = off["arms"]["intact"]["silent_non_words"]
    print("D3", json.dumps({k: v for k, v in summary["D3_parse"].items()
                            if k not in ("intact", "coincidence", "coincidence_matched")}, default=str), flush=True)


def _d1_d2_d4(summary, lex, J, L, a, seed, corpus, out_dir):
    """D1 (curriculum, report only), D2 (the AND on the trained circuit + its lesions) and D4 (the route runner's
    R1-R4 logic), shared by both variants; AMENDMENT 3 adds, for the elemental variant, the `conjunctive` lesion's
    integrity check (no junction fires to its pair)."""
    # D1 curriculum accuracy without the teacher (report only)
    lex.set_lesion(None)
    words, labels = L.seed_curriculum(lex.env)
    dec = [lex.decide(w)[0][0] for w in words]
    correct = sum(1 for d, lab in zip(dec, labels) if d is not None and d == (lab > 0))
    summary["D1_curriculum"] = {"n": len(words), "correct": int(correct),
                                "abstain": int(sum(d is None for d in dec)),
                                "wrong": [w for w, d, lab in zip(words, dec, labels) if d is not None and d != (lab > 0)]}
    print("D1", summary["D1_curriculum"], flush=True)

    # D2 AND smoke (sample) + AND POPULATION (AMENDMENT 2: every junction) on the trained circuit + the OR lesion
    sm = J.and_smoke(lex, n_sample=256, seed=1)
    pop = J.and_population(lex)
    lex.set_lesion("coincidence")
    rng = np.random.default_rng(2)
    pairs = [(int(x), int(y)) for x, y in zip(rng.integers(0, lex.C, 64), rng.integers(0, lex.C, 64))]
    lone, _ = lex.junction_response(pairs, "left")
    lex.set_lesion(None)
    summary["D2_and_smoke"] = {**sm, "coincidence_lesion_lone_left_fired": int((lone > 0).sum()),
                               "coincidence_lesion_n": len(pairs)}
    summary["D2_and_population"] = {k: v for k, v in pop.items()
                                    if k not in ("lone_left_full_rows", "lone_right_full_cols")}
    _dump(os.path.join(out_dir, f"and_population_trained_s{seed}.json"), pop)
    print("D2", summary["D2_and_smoke"], flush=True)
    print("D2_population", summary["D2_and_population"], flush=True)
    if getattr(lex, "elemental", False):
        # AMENDMENT 3: the `conjunctive` lesion must leave EVERY junction silent to its own pair (the AND removed).
        lex.set_lesion("conjunctive")
        popc = J.and_population(lex)
        lex.set_lesion(None)
        summary["D2_conjunctive_lesion"] = {
            "pair_silent": popc["pair_silent"], "n_junctions": popc["n_junctions"],
            "lone_left_fired": popc["lone_left_fired"], "lone_right_fired": popc["lone_right_fired"],
            "holds": bool(popc["pair_silent"] == popc["n_junctions"] and popc["lone_left_fired"] == 0
                          and popc["lone_right_fired"] == 0)}
        print("D2_conjunctive_lesion", summary["D2_conjunctive_lesion"], flush=True)

    # D4 route logic with this seed's junction lexicon pinned as the singleton (seed 42 IS the production seed, so
    # no patch is needed there -- get_lexicon()'s own default already trains at 42).
    if not a.skip_route:
        from research.runners import _d6_learned_referent_env_flag_derisk as R
        from research.runners._lexicon_learned_referent_derisk import FIXTURE
        trained = lex
        orig_get = L.get_lexicon
        if seed != 42:
            def _pinned(*_a, **_k):
                L._LEXICON = trained
                return trained
            L.get_lexicon = _pinned
        L._DEFAULT_CORPUS = corpus          # the worktree has no data/ copy; record the file the lexicon read
        try:
            route = R.run_seed(seed, corpus, json.load(open(FIXTURE))["pos"])
        finally:
            L.get_lexicon = orig_get
        _dump(os.path.join(out_dir, f"route_s{seed}.json"), route)
        summary["D4_route"] = {k: route.get(k) for k in ("r1_pass", "r1_input_order", "r2_pass",
                                                        "r3_recovered_both_rate", "r3_pass",
                                                        "r4_lesion_recovered_both_rate", "r4_lever_moved", "r4_pass",
                                                        "hand_baseline_recovered_both_rate", "elapsed_s")}
        # the route's own lexicon variant (seed 42 rebuilds it through the real get_lexicon; others reuse `lex`)
        summary["D4_route"]["route_lexicon_variant"] = getattr(L._LEXICON, "variant", None)
        print("D4", summary["D4_route"], flush=True)


if __name__ == "__main__":
    main()
