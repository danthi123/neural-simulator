"""CAPSTONE+CONSOLE — the BRAIN-LOAD SPEEDUP (steps (2)+(1)): a LOADED console brain LOADS its trained state in
low-seconds instead of RE-TRAINING the parser / RE-RESONATING every fact (the owner's per-op-latency priority).

Per the scoping (`research/findings/2026-06-24-brain-load-speedup-scoping.md` + `_scoping_brain_load_speedup.json`),
the first-load cost is TWO recomputation loops, both pure recomputation of state that could be persisted:

  (2) LAZY PARSER — the ~75K-step Hebbian PARSER training (BridgeParser + AttributedBridgeParser, run in the agent
      `__init__`). A LOADED brain restores its facts via `composer.store()` directly (bypassing the parser), so a
      pure Q&A session NEVER uses the parser -> DEFER its construction+training to the FIRST runtime teach. Removes
      the entire FIXED ~75K-step cost (dominates the tiny-demo's ~112 s).

  (1) PERSIST KB COMPOSITES — the per-fact RF resonate re-store (~832 steps/fact, ~43K for a 52-fact brain) whose
      output is just the `[D]` numpy composite already cached in `composer.kb`. Persist it in the developed-brain
      bundle (`kb_composites.npz`) -> on load set `composer.kb` directly, skipping the resonate. The composite IS
      the deterministic resonate output -> byte-identical recall. Removes the PER-FACT cost (dominates a
      knowledge-rich brain).

This runner VALIDATES the LOGIC on a SMALL brain (NOT a full self-knowledge brain-load — that is the CONTROLLER's
GPU measurement). NumPy/CPU, FOREGROUND-only, reuse-by-import, NO `sim/` edit. It checks:

  (2) defer-train: a LOADED brain constructs the parser DEFERRED (no `_train` ran — `_parser_trained_count == 0`);
      a Q&A query still ANSWERS (from the loaded facts) + the MOAT ABSTAINS on untaught; teaching a NEW fact still
      works (the deferred parser builds lazily on the first teach + the fact recalls).
  (1) kb round-trip: the persisted composites round-trip (save -> load -> `composer.kb` composite == the original
      array BIT-EXACT, recall byte-identical, moat 0-FA).
  ANTI-CHEAT: the loaded composite == a FRESH `store()` re-resonate BIT-EXACT (the persisted composite IS the same
      array); recall is identical to the re-resonate path; the moat is 0-FA; the EAGER (defer-OFF) standalone path
      is byte-unchanged (the eager agent trains its parser, count == 2).

Run:
    SIM_BACKEND=numpy python -u -m research.runners._capstone_console_brain_load_speedup \
        --out research/findings/raw/_capstone_console_brain_load_speedup.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.rf_phasor_composer import Clause  # noqa: E402
from research.runners.developed_brain_io import (  # noqa: E402
    save_developed_brain, load_developed_brain, _read_manifest, _load_kb_composites,
    extract_kb_composites,
)

# a SMALL brain — varied fact shapes: plain SVO, polarity tag, attributed entity, embedded clause.
_VOCAB = ["dog", "cat", "go", "chase", "look", "eat", "north", "south", "fish", "big", "apple", "river", "bird"]
_FACTS = [
    ("store", ("dog", "go", "north"), {"polarity": "AFFIRM"}),
    ("store", ("cat", "eat", "fish"), {"polarity": "NEGATE"}),
    ("store", ("dog", "chase", ("big", "apple")), {}),                 # attributed entity
    ("store", ("dog", "look", Clause("cat", "go", "south")), {}),      # embedded-clause patient
]
# Q&A probes (taught) + a MOAT probe (untaught) — answers must be identical loaded vs fresh.
_QA = [("dog", "go"), ("cat", "eat"), ("dog", "chase"), ("dog", "look")]
_MOAT = ("bird", "jump")        # never taught -> must abstain (None)


def _build_small_brain(seed, *, defer_parser=False):
    a = BrainConversationalAgent(seed=seed, concepts={w: None for w in _VOCAB},
                                 composer_kind="rf", enable_neural_render=False,
                                 defer_parser=defer_parser)
    for _kind, args, kw in _FACTS:
        a.composer.store(*args, **kw)
    return a


def _qa_snapshot(agent):
    """The recall answers (what_does for each probe + the moat) — the byte-identity oracle for recall."""
    inner = getattr(agent, "agent", agent)
    out = {f"what_does({a},{v})": inner.what_does(a, v) for (a, v) in _QA}
    out[f"moat_what_does({_MOAT[0]},{_MOAT[1]})"] = inner.what_does(*_MOAT)
    # a yes/no probe (exercises the polarity tag round-trip)
    out["yes_no(dog,go,north)"] = inner.is_it_true("dog", "go", "north")
    out["yes_no(cat,eat,fish)"] = inner.is_it_true("cat", "eat", "fish")     # NEGATE -> 'no'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_capstone_console_brain_load_speedup.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    print("=" * 110, flush=True)
    print("[BRAIN-LOAD SPEEDUP — (2) lazy parser + (1) persist kb composites; LOGIC validation on a SMALL brain]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  seed={a.seed}  facts={len(_FACTS)}", flush=True)
    print("=" * 110 + "\n", flush=True)

    res = {
        "probe": "capstone_console_brain_load_speedup__lazy_parser_plus_persist_kb_composites",
        "backend": os.environ.get("SIM_BACKEND"),
        "seed": a.seed,
        "scope": "LOGIC/CPU validation on a SMALL brain (NOT a full self-knowledge brain-load; the controller "
                 "measures the full GPU load-time separately).",
        "sim_edit": "NONE (reuse-by-import: brain_conversational_agent + multi_turn_agent + developed_brain_io only).",
    }

    # ======================================================================================================
    # BASELINE (the EAGER reference) — build a brain, teach facts, save the bundle.
    # ======================================================================================================
    t0 = time.time()
    eager = _build_small_brain(a.seed, defer_parser=False)
    res["eager_build_seconds"] = round(time.time() - t0, 3)
    res["eager_parser_trained_count"] = int(eager._parser_trained_count)   # expect 2 (BridgeParser + AttributedBridgeParser)
    eager_qa = _qa_snapshot(eager)
    print(f"[baseline] eager build {res['eager_build_seconds']}s -> parser_trained_count="
          f"{res['eager_parser_trained_count']} (expect 2: BridgeParser + AttributedBridgeParser)", flush=True)

    bundle = tempfile.mkdtemp(prefix="blspeed_")
    try:
        manifest = save_developed_brain(eager, bundle, seed=a.seed, D=128, composer_kind="rf")
        res["bundle_files"] = sorted(os.listdir(bundle))
        res["manifest_n_kb_composites"] = manifest.get("n_kb_composites")
        res["kb_composites_npz_written"] = bool((Path(bundle) / "kb_composites.npz").exists())
        print(f"[baseline] saved bundle: files={res['bundle_files']}  n_kb_composites="
              f"{res['manifest_n_kb_composites']}", flush=True)

        # the persisted composites (option 1) — index -> [D]
        saved_composites = _load_kb_composites(bundle)
        in_mem_composites = extract_kb_composites(eager)
        res["n_composites_persisted"] = len(saved_composites)
        res["n_facts"] = len(eager.composer.kb)

        # ==================================================================================================
        # (1) KB-COMPOSITE ROUND-TRIP — load (no resonate) + BIT-EXACT vs in-memory + vs a FRESH re-resonate.
        # ==================================================================================================
        t0 = time.time()
        loaded, _m = load_developed_brain(bundle, use_multiturn=False, enable_neural_render=False)
        res["loaded_build_seconds"] = round(time.time() - t0, 3)
        li = getattr(loaded, "agent", loaded)

        # (1a) saved composite == in-memory composite, BIT-EXACT (the npz round-trips the array verbatim)
        saved_bit_exact = all(
            np.array_equal(np.asarray(in_mem_composites[k]), np.asarray(saved_composites.get(int(k))))
            for k in in_mem_composites)
        # (1b) loaded composer.kb composite == the in-memory composite, BIT-EXACT (the direct-set, no resonate)
        loaded_eq_inmem = (len(li.composer.kb) == len(eager.composer.kb)) and all(
            np.array_equal(np.asarray(eager.composer.kb[i][1]), np.asarray(li.composer.kb[i][1]))
            for i in range(len(eager.composer.kb)))
        # (1c) ANTI-CHEAT: loaded composite == a FRESH store() RE-RESONATE, BIT-EXACT (the persisted composite IS the
        #      deterministic resonate output -> not a re-derive, the same array)
        fresh = _build_small_brain(a.seed, defer_parser=False)
        loaded_eq_fresh = all(
            np.array_equal(np.asarray(fresh.composer.kb[i][1]), np.asarray(li.composer.kb[i][1]))
            for i in range(len(li.composer.kb)))
        # (1d) the fact dicts the direct-set rebuilt == store()'s fact dicts (the _store_fact_dict_from_operand mirror)
        dicts_match = all(eager.composer.kb[i][0] == li.composer.kb[i][0] for i in range(len(eager.composer.kb)))

        res["option1_persist_kb_composites"] = {
            "saved_composite_bit_exact_vs_inmemory": bool(saved_bit_exact),
            "loaded_kb_composite_bit_exact_vs_inmemory": bool(loaded_eq_inmem),
            "loaded_composite_bit_exact_vs_fresh_reresonate": bool(loaded_eq_fresh),  # the anti-cheat
            "fact_dicts_identical_to_store": bool(dicts_match),
        }
        print(f"[option 1] kb composites: saved_bit_exact={saved_bit_exact} loaded==inmem={loaded_eq_inmem} "
              f"loaded==fresh_reresonate={loaded_eq_fresh} (ANTI-CHEAT) dicts_match={dicts_match}", flush=True)

        # ==================================================================================================
        # (2) DEFER-TRAIN — the loaded brain's parser is DEFERRED (no _train ran); Q&A answers; teach still works.
        # ==================================================================================================
        loaded_trained_count = int(li._parser_trained_count)   # expect 0 BEFORE any teach
        loaded_parser_is_none = li.parser is None
        # Q&A on the loaded brain (no teach) — answers must be identical to the eager reference
        loaded_qa = _qa_snapshot(loaded)
        qa_identical = (loaded_qa == eager_qa)
        # the parser STILL untrained after pure Q&A (proves a read session never pays the ~75K-step training)
        trained_count_after_qa = int(li._parser_trained_count)
        # the MOAT: untaught cue abstains
        moat_abstains = (loaded_qa[f"moat_what_does({_MOAT[0]},{_MOAT[1]})"] is None)

        print(f"[option 2] loaded: parser_is_None={loaded_parser_is_none} trained_count={loaded_trained_count} "
              f"(expect 0); Q&A identical to eager={qa_identical}; trained_count after Q&A="
              f"{trained_count_after_qa} (still 0); moat_abstains={moat_abstains}", flush=True)

        # teach a NEW fact on the loaded (deferred) brain -> the lazy parser builds + the fact recalls
        t0 = time.time()
        taught_roles = li.hear("dog eat fish", polarity="AFFIRM")
        teach_seconds = round(time.time() - t0, 3)
        trained_count_after_teach = int(li._parser_trained_count)   # expect 1 (lazy BridgeParser build on first teach)
        new_fact_recalls = (li.what_does("dog", "eat") == "fish")
        old_fact_intact = (li.what_does("dog", "go") == "north")
        # the new-fact parse must equal a FRESH eager parser's parse (deferred build == eager build, same seed)
        eager_parse = fresh._ensure_parser().parse(["dog", "eat", "fish"], "active")
        parse_equiv = (taught_roles == eager_parse)

        res["option2_lazy_parser"] = {
            "loaded_parser_is_none_before_teach": bool(loaded_parser_is_none),
            "loaded_parser_trained_count_on_load": loaded_trained_count,                 # 0
            "no_train_on_load": (loaded_trained_count == 0),
            "qa_answers_identical_to_eager": bool(qa_identical),
            "parser_still_untrained_after_pure_qa": (trained_count_after_qa == 0),
            "moat_abstains_on_untaught": bool(moat_abstains),
            "teach_new_fact_seconds": teach_seconds,
            "parser_trained_count_after_first_teach": trained_count_after_teach,         # 1
            "lazy_parser_built_on_first_teach": (trained_count_after_teach == 1),
            "new_fact_recalls": bool(new_fact_recalls),
            "old_fact_intact_after_teach": bool(old_fact_intact),
            "deferred_parse_equals_eager_parse": bool(parse_equiv),
        }
        print(f"[option 2] teach 'dog eat fish' ({teach_seconds}s) -> trained_count={trained_count_after_teach} "
              f"(expect 1); new_fact_recalls={new_fact_recalls}; old_fact_intact={old_fact_intact}; "
              f"deferred_parse==eager_parse={parse_equiv}", flush=True)

        # ==================================================================================================
        # REGRESSION — the EAGER standalone path is byte-unchanged (defer OFF default: parser trained eagerly).
        # ==================================================================================================
        eager_default = _build_small_brain(a.seed, defer_parser=False)
        eager_default_trains = (int(eager_default._parser_trained_count) == 2)
        eager_qa_unchanged = (_qa_snapshot(eager_default) == eager_qa)
        res["regression_eager_path_unchanged"] = {
            "eager_default_trains_two_parsers": bool(eager_default_trains),   # BridgeParser + AttributedBridgeParser
            "eager_default_qa_identical": bool(eager_qa_unchanged),
        }
        print(f"[regression] eager default trains 2 parsers={eager_default_trains}; "
              f"eager Q&A unchanged={eager_qa_unchanged}", flush=True)

        # ==================================================================================================
        # the CONSOLE default path = MultiTurnAgent (use_multiturn=True) — exercise the same load path the
        # console uses, to confirm defer+composites flow through the wrapper.
        # ==================================================================================================
        t0 = time.time()
        mt_loaded, _m = load_developed_brain(bundle, use_multiturn=True, enable_neural_render=False)
        mt_seconds = round(time.time() - t0, 3)
        mti = getattr(mt_loaded, "agent", mt_loaded)
        mt_is_multiturn = hasattr(mt_loaded, "held_referent")
        mt_trained = int(mti._parser_trained_count)
        mt_qa_ok = (mti.what_does("dog", "go") == "north" and mti.what_does("cat", "eat") == "fish"
                    and mti.what_does(*_MOAT) is None)
        res["console_multiturn_load"] = {
            "is_multiturn": bool(mt_is_multiturn),
            "load_seconds": mt_seconds,
            "inner_parser_trained_count": mt_trained,            # 0
            "no_train_on_load": (mt_trained == 0),
            "qa_and_moat_ok": bool(mt_qa_ok),
        }
        print(f"[console] MultiTurnAgent load ({mt_seconds}s): is_multiturn={mt_is_multiturn} "
              f"inner_trained_count={mt_trained} (expect 0) qa_and_moat_ok={mt_qa_ok}", flush=True)

    finally:
        import shutil
        shutil.rmtree(bundle, ignore_errors=True)

    # ======================================================================================================
    # VERDICT
    # ======================================================================================================
    o1 = res["option1_persist_kb_composites"]
    o2 = res["option2_lazy_parser"]
    reg = res["regression_eager_path_unchanged"]
    con = res["console_multiturn_load"]

    option1_ok = all([o1["saved_composite_bit_exact_vs_inmemory"], o1["loaded_kb_composite_bit_exact_vs_inmemory"],
                      o1["loaded_composite_bit_exact_vs_fresh_reresonate"], o1["fact_dicts_identical_to_store"],
                      res["n_composites_persisted"] == res["n_facts"]])
    option2_ok = all([o2["no_train_on_load"], o2["qa_answers_identical_to_eager"],
                      o2["parser_still_untrained_after_pure_qa"], o2["moat_abstains_on_untaught"],
                      o2["lazy_parser_built_on_first_teach"], o2["new_fact_recalls"],
                      o2["old_fact_intact_after_teach"], o2["deferred_parse_equals_eager_parse"]])
    regression_ok = all([reg["eager_default_trains_two_parsers"], reg["eager_default_qa_identical"]])
    console_ok = all([con["is_multiturn"], con["no_train_on_load"], con["qa_and_moat_ok"]])
    moat_clean = bool(o2["moat_abstains_on_untaught"] and con["qa_and_moat_ok"])

    go = bool(option1_ok and option2_ok and regression_ok and console_ok and moat_clean)
    res["option1_ok"] = option1_ok
    res["option2_ok"] = option2_ok
    res["regression_ok"] = regression_ok
    res["console_ok"] = console_ok
    res["moat_clean"] = moat_clean
    res["GO"] = go
    res["verdict"] = (
        "GO — the brain-LOAD speedup LOGIC is validated on a small brain. (2) LAZY PARSER: a LOADED brain "
        "constructs the comprehension parser DEFERRED (parser_trained_count==0 on load, still 0 after pure Q&A — no "
        "~75K-step training paid), Q&A answers IDENTICALLY to the eager reference, the moat ABSTAINS on an untaught "
        "cue, and teaching a NEW fact builds the parser lazily (count->1) + recalls correctly, with the deferred "
        "parse BIT-EQUAL to a fresh eager parse. (1) PERSIST KB COMPOSITES: the bundle's kb_composites.npz round-trips "
        "BIT-EXACT (loaded == in-memory == a FRESH re-resonate — the persisted composite IS the deterministic "
        "resonate output, NOT a re-derive), recall is byte-identical, the fact dicts match store() exactly, and the "
        "load skips the per-fact resonate. The EAGER standalone path is byte-unchanged (trains 2 parsers, Q&A "
        "identical) and the CONSOLE MultiTurnAgent load path flows defer+composites through (no train on load, "
        "Q&A+moat OK). NO sim/ edit; moat 0-FA throughout."
        if go else
        f"HONEST/SNAG — option1_ok={option1_ok} option2_ok={option2_ok} regression_ok={regression_ok} "
        f"console_ok={console_ok} moat_clean={moat_clean}. See the per-section detail for the localize."
    )

    res["anti_cheats"] = [
        "(1) the loaded composer.kb composite == a FRESH store() re-resonate BIT-EXACT (np.array_equal) — the "
        "persisted composite IS the same deterministic resonate array, NOT a re-derive; recall is byte-identical.",
        "(1) the rebuilt fact dicts == store()'s fact dicts (attribute split + clause + polarity), so the direct-set "
        "matches store() in structure, not just the composite array.",
        "(2) parser_trained_count == 0 on a Q&A-only load (and still 0 after pure Q&A) proves the deferred parser "
        "never paid its ~75K-step training; the first TEACH builds it lazily (count->1) and the deferred parse is "
        "BIT-EQUAL to a fresh eager parse, so a teaching session is identical to a never-deferred one.",
        "the no-confab MOAT is 0-FA on the loaded brain (an untaught cue abstains -> None) — only the saved facts "
        "are in kb.",
        "REGRESSION: defer-OFF (the standalone build default) trains 2 parsers eagerly + Q&A is byte-identical, so "
        "the standalone path is unchanged.",
    ]
    res["expected_load_time_speedup"] = {
        "removed_loops": [
            "(2) the ~75K-step Hebbian parser training (BridgeParser ~25K + AttributedBridgeParser ~50K), "
            "fact-INDEPENDENT -> the dominant term of the tiny-demo's ~112 s first load; now 0 for a Q&A session.",
            "(1) the ~832-step-per-fact RF resonate re-store (~43K steps for a 52-fact self-knowledge brain) -> now "
            "a direct kb-composite set from kb_composites.npz; ~0 per fact.",
        ],
        "expected": "a developed-brain BUNDLE that previously re-trained the parser (~75K steps) + re-resonated each "
                    "fact (~832/fact) now LOADS in bundle-deserialize time (low seconds): both dominant loops are "
                    "removed for a Q&A session. The residual is the bundle npz/json read + (for use_multiturn) the "
                    "SpikingLoopContextBuffer WM build — both sub-second to a few seconds.",
        "note": "the SELF-KNOWLEDGE built-in (`_load_self_knowledge`) re-teaches its curriculum via hear() (parser + "
                "resonate) rather than load_developed_brain; to get the same speedup for it, save it as a bundle "
                "(developed_brain_io.save_developed_brain) and load via the bundle path — the generic path this build "
                "speeds up. The controller measures the bundle load-time before vs after.",
    }

    out_path = a.out if os.path.isabs(a.out) else os.path.join(_REPO, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  VERDICT: {res['verdict']}", flush=True)
    print(f"  [saved] {os.path.relpath(out_path, _REPO)}\n{'=' * 110}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
