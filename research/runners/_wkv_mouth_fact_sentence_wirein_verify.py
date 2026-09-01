"""Board #112 rung-3 WIRE-IN verify (2026-09-01, branch `research/wkv-mouth-fact-sentence-wirein`): a 6-seed,
end-to-end verification of `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE` through the REAL
`webapp.open_ended_chat.answer_turn()` entry point -- NOT a parallel harness. Confirms the merged 2026-09-01
lexicon lever (`research.runners._wkv_fact_to_sentence_lexicon_lever`) is now genuinely reachable from the WKV
mouth's OWN `generate()` call for a known, in-vocab topic (`webapp.wkv_mouth_generator.render_fact_sentence`,
called from `generate()`'s own `_run()` closure via the new `sentence_facts` parameter), byte-identical-off,
moat-safe/honest on a brain-unknown topic, and measures the #112 grounding-regression delta against the
pre-existing free-gen path AND the older word-boost lever on the SAME sampled real facts.

HONEST SCOPE, stated up front (not discovered mid-run): the WKV mouth only ever fires for IN-VOCAB prompts
(`webapp.wkv_mouth_generator.in_vocab_scope`, V=1000 TinyStories-domain checkpoint vocabulary) -- an
independent scan of 400 real `wikidata_core_15k` agents (2026-09-01, this task) found only ~3% (12/400) pass
that gate at all, though ALL 12 of those also had a lexicon-covered relation. This verify samples FOR that
narrow slice specifically (real known topics that are ALSO in-vocab) -- it is not a claim that this wire-in
fixes fact-thin generation for the majority of real known topics, which route through Qwen (out-of-vocab for
this checkpoint), a completely different generator/code path untouched by this change (see
`research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`, which measured that
Qwen-routed class).

Run: SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_fact_sentence_wirein_verify
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from webapp import open_ended_chat as OEC  # noqa: E402
from webapp import wkv_mouth_generator as WKV  # noqa: E402
from research.runners._wkv_fact_svo_clause_first_lever import _bundle_dir  # noqa: E402
from research.runners._wkv_fact_to_sentence_lexicon_lever import (  # noqa: E402
    expected_surface, parse_and_score,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

REPO_ROOT = _REPO
SEEDS = (42, 43, 44, 100, 101, 102)
_WORD_RE = WKV._WORD_RE
_FUNCTION_WORDS = WKV._FUNCTION_WORDS
FLAG = "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE"
BOOST_FLAG = "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND"


def _sample_known_invocab_covered(by_agent: dict, seed: int, n: int = 8, scan: int = 2000) -> list:
    """The SAME real-store sampling discipline the lexicon lever used (seeded, real AFFIRM facts), narrowed to
    the intersection this wire-in actually needs: a real store agent whose `tell me about <agent>` prompt (a)
    passes `in_vocab_scope` for this checkpoint AND (b) has >=1 fact whose relation the lexicon covers."""
    agents = list(by_agent.keys())
    rng = random.Random(seed)
    rng.shuffle(agents)
    picked = []
    for a in agents[:scan]:
        msg = f"tell me about {a}"
        if not WKV.in_vocab_scope(msg, seed=seed):
            continue
        facts = OEC.retrieve(by_agent, OEC.extract_topic(msg))
        triple = WKV.pick_covered_fact(facts)
        if triple is None:
            continue
        picked.append({"agent": a, "msg": msg, "facts": facts, "triple": triple})
        if len(picked) >= n:
            break
    return picked


def _content_words(triple) -> set:
    agent, action, patient = triple
    words = set()
    for field in (agent, action, patient):
        for w in _WORD_RE.findall(str(field)):
            wl = w.lower()
            if wl not in _FUNCTION_WORDS:
                words.add(wl)
    return words


def _content_hit_frac(text: str, words: set):
    if not words:
        return None
    toks = set(w.lower() for w in _WORD_RE.findall(text or ""))
    hits = sum(1 for w in words if w in toks)
    return hits / len(words)


def _run_case(msg: str, seed: int, flags: dict, bundle: str) -> dict:
    """Call the REAL `answer_turn` with a scoped env-var override (restored in `finally`, never leaked between
    cases -- each of the 3 arms per fact below must observe an INDEPENDENT flag state)."""
    old = {k: os.environ.get(k) for k in flags}
    try:
        for k, v in flags.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return OEC.answer_turn(msg, None, valence=0.0, arousal=0.5, ltm_bundle=bundle, brain_bundle=None,
                                seed=seed)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> dict:
    out: dict = {"runner": "_wkv_mouth_fact_sentence_wirein_verify", "seeds": list(SEEDS)}
    bundle = _bundle_dir()
    out["bundle_dir"] = bundle
    if bundle is None:
        out["skipped"] = "no data lake (sim-data/knowledge_bundles/wikidata_core_15k not found)"
        print(json.dumps(out, indent=2))
        return out
    by_agent = OEC.build_index(bundle, None)
    out["n_agents_indexed"] = len(by_agent)

    per_seed = []
    for seed in SEEDS:
        cases = _sample_known_invocab_covered(by_agent, seed, n=8)
        rows = []
        for c in cases:
            agent, action, patient = c["triple"]
            exp, covered, _n = expected_surface(agent, action, patient)

            # (A) sentence-mode ON -- the new wire-in, THIS task's own change. Scored on TWO layers,
            # deliberately kept separate: `raw` is what `webapp.wkv_mouth_generator.generate()` itself
            # produced (the wire-in's own claim); `answer` is what survives the SEPARATE, pre-existing
            # (2026-08-21, already-GO'd) known-topic contradiction post-filter on top of it. A real
            # interaction was found during this verify: that filter's bare number/date check (a documented
            # SCOPE limit of `_open_ended_known_supplement_filter_derisk.sentence_contradicts` -- "a bare
            # unsupported number with no relative-clause boundary has no declared-safe repair") does not
            # distinguish a fabricated number from a number that is part of the topic's OWN slug/name
            # (e.g. "1974_football_world_cup"), so it over-cautiously abstains on an otherwise-correct
            # rendered clause. This is a SAFE failure mode (never a leak -- see the invariant check below),
            # not a wire-in defect, so both layers are measured and reported separately rather than
            # conflating a pre-existing safety-net's known scope limit with this task's own mechanism.
            res_on = _run_case(c["msg"], seed, {FLAG: "1", BOOST_FLAG: None}, bundle)
            score_on = parse_and_score(res_on["answer"], agent, action, patient)
            score_raw = parse_and_score(res_on["raw"], agent, action, patient)
            answer_is_safe_degrade = bool(
                res_on["answer"] == res_on["raw"]
                or res_on["answer"] == OEC._empty_known_fallback(OEC.extract_topic(c["msg"])))

            # (B) both flags OFF -- byte-identical-off / the pre-existing free-gen path, unchanged.
            # Explicit "0", NOT None -- BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE defaults ON as of this task's
            # own auto-flip (see webapp/open_ended_chat.py::wkv_fact_sentence_enabled), so merely unsetting it
            # would no longer represent "off."
            res_off = _run_case(c["msg"], seed, {FLAG: "0", BOOST_FLAG: "0"}, bundle)

            # (C) the OLDER word-boost lever ON, sentence-mode explicitly OFF -- the #112 "before" comparison.
            res_boost = _run_case(c["msg"], seed, {FLAG: "0", BOOST_FLAG: "1"}, bundle)

            cwords = _content_words(c["triple"])
            rows.append({
                "agent": agent, "action": action, "patient": patient,
                "expected_surface": exp, "covered": covered,
                "sentence_mode_raw": res_on["raw"],
                "sentence_mode_answer": res_on["answer"],
                "sentence_mode_generator": res_on["generator"],
                "sentence_mode_wkv_used": bool(res_on["wkv_mouth_used"]),
                "sentence_mode_known": bool(res_on["known"]),
                # the wire-in's OWN claim (pre-safety-net):
                "raw_well_formed": score_raw["well_formed"],
                "raw_faithful": score_raw["faithful"],
                "raw_readable": score_raw["readable"],
                "raw_moat_safe": score_raw["moat_safe"],
                # the end-to-end, post-safety-net claim (bounded by the pre-existing filter's own scope):
                "sentence_mode_well_formed": score_on["well_formed"],
                "sentence_mode_faithful": score_on["faithful"],
                "sentence_mode_readable": score_on["readable"],
                "sentence_mode_moat_safe": score_on["moat_safe"],
                "answer_is_safe_degrade": answer_is_safe_degrade,
                "off_answer": res_off["answer"],
                "off_content_hit_frac": _content_hit_frac(res_off["answer"], cwords),
                "off_is_clause": res_off["answer"] == exp,
                "boost_answer": res_boost["answer"],
                "boost_content_hit_frac": _content_hit_frac(res_boost["answer"], cwords),
                "boost_is_clause": res_boost["answer"] == exp,
            })

        # unknown-topic honesty check (moat intact): a real message that IS in-vocab for the checkpoint but is
        # NOT a store topic -- must still hedge/abstain with the flag on, exactly as before this change.
        unk_msg = "tell me about the dog and the forest"
        res_unk = _run_case(unk_msg, seed, {FLAG: "1", BOOST_FLAG: None}, bundle)
        unk_abstained = any(p in res_unk["answer"].lower() for p in ("not sure", "don't have", "guessing"))

        n = len(rows)

        def frac(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
            return float(np.mean(vals)) if vals else None

        row = {
            "seed": seed, "n_cases": n,
            # the wire-in's OWN claim (raw output of generate(), pre-safety-net) -- the GO bar below:
            "raw_readable_frac": frac("raw_readable"),
            "raw_faithful_frac": frac("raw_faithful"),
            "raw_moat_safe_frac": frac("raw_moat_safe"),
            # the end-to-end (post-filtered) rate -- informational, bounded by the pre-existing known-topic
            # contradiction filter's own documented number/date scope limit (see the row-level comment above):
            "readable_frac": frac("sentence_mode_readable"),
            "faithful_frac": frac("sentence_mode_faithful"),
            "moat_safe_frac": frac("sentence_mode_moat_safe"),
            "answer_is_safe_degrade_frac": frac("answer_is_safe_degrade"),
            "wkv_used_frac": frac("sentence_mode_wkv_used"),
            "known_frac": frac("sentence_mode_known"),
            "off_is_clause_frac": frac("off_is_clause"),
            "boost_is_clause_frac": frac("boost_is_clause"),
            "off_content_hit_frac_mean": (float(np.mean([r["off_content_hit_frac"] for r in rows
                                                          if r["off_content_hit_frac"] is not None]))
                                          if rows else None),
            "boost_content_hit_frac_mean": (float(np.mean([r["boost_content_hit_frac"] for r in rows
                                                            if r["boost_content_hit_frac"] is not None]))
                                            if rows else None),
            "unknown_topic_known": bool(res_unk["known"]),
            "unknown_topic_answer": res_unk["answer"],
            "unknown_topic_wkv_used": bool(res_unk["wkv_mouth_used"]),
            "unknown_topic_abstained": bool(unk_abstained),
            "rows": rows,
        }
        per_seed.append(row)
        print(f"[seed {seed}] n={n} raw_readable={row['raw_readable_frac']} raw_faithful={row['raw_faithful_frac']} "
              f"raw_moat_safe={row['raw_moat_safe_frac']} | post-filter readable={row['readable_frac']} "
              f"faithful={row['faithful_frac']} safe_degrade={row['answer_is_safe_degrade_frac']} | "
              f"off_is_clause={row['off_is_clause_frac']} boost_is_clause={row['boost_is_clause_frac']} "
              f"unknown_abstained={row['unknown_topic_abstained']}")
        for r in rows[:3]:
            print(f"    {r['agent']:30s} {r['action']:25s} raw={r['sentence_mode_raw']!r} "
                  f"answer={r['sentence_mode_answer']!r}")

    out["per_seed"] = per_seed

    def agg(key):
        vals = [s[key] for s in per_seed if s[key] is not None]
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4)} if vals else None

    out["aggregate"] = {
        "n_seeds": len(per_seed), "n_cases_total": sum(s["n_cases"] for s in per_seed),
        "raw_readable": agg("raw_readable_frac"), "raw_faithful": agg("raw_faithful_frac"),
        "raw_moat_safe": agg("raw_moat_safe_frac"),
        "readable": agg("readable_frac"), "faithful": agg("faithful_frac"), "moat_safe": agg("moat_safe_frac"),
        "answer_is_safe_degrade": agg("answer_is_safe_degrade_frac"),
        "wkv_used": agg("wkv_used_frac"), "known": agg("known_frac"),
        "off_is_clause": agg("off_is_clause_frac"), "boost_is_clause": agg("boost_is_clause_frac"),
        "off_content_hit_frac_mean": agg("off_content_hit_frac_mean"),
        "boost_content_hit_frac_mean": agg("boost_content_hit_frac_mean"),
        "all_seeds_min_1_case": bool(all(s["n_cases"] >= 1 for s in per_seed)),
        "all_unknown_abstained": bool(all(s["unknown_topic_abstained"] for s in per_seed)),
    }

    a = out["aggregate"]
    v = Verdict("WKV mouth fact->sentence wire-in: 6-seed end-to-end verify through the real answer_turn path")
    v.require("every seed found >=1 real known+in-vocab+covered-relation case",
              a["all_seeds_min_1_case"], expect=True)
    v.require("sentence-mode reaches the WKV mouth on every found case",
              a["wkv_used"]["min"] if a["wkv_used"] else None, expect=lambda x: x is not None and x >= 0.999)
    # THE WIRE-IN'S OWN CLAIM: generate()'s raw output (pre-existing safety-net not yet applied).
    v.require("RAW readable (coherent-clause) rate >= 0.95 on every seed",
              a["raw_readable"]["min"] if a["raw_readable"] else None, expect=lambda x: x is not None and x >= 0.95)
    v.require("RAW faithful rate >= 0.95 on every seed",
              a["raw_faithful"]["min"] if a["raw_faithful"] else None, expect=lambda x: x is not None and x >= 0.95)
    v.require("RAW moat-safe on every seed",
              a["raw_moat_safe"]["min"] if a["raw_moat_safe"] else None,
              expect=lambda x: x is not None and x >= 0.999)
    # THE END-TO-END SAFETY INVARIANT (informational on coverage, but load-bearing on safety): the post-
    # filtered answer must ALWAYS be either the raw clause unchanged, or the fixed honest-abstain fallback --
    # NEVER a partially-corrupted hybrid. A pre-existing (2026-08-21) known-topic contradiction filter's own
    # documented number/date scope limit can make it choose the fallback over a correct raw clause (an
    # over-cautious, SAFE degrade, not a leak) -- see the mapped residual in the finding.
    v.require("post-filtered answer is always either the raw clause or the honest fallback (never corrupted)",
              a["answer_is_safe_degrade"]["min"] if a["answer_is_safe_degrade"] else None,
              expect=lambda x: x is not None and x >= 0.999)
    v.require("flag-off degrades to the pre-existing free-gen path (never the clause) on every case",
              a["off_is_clause"]["mean"] if a["off_is_clause"] else None,
              expect=lambda x: x is not None and x <= 0.001)
    v.require("unknown-topic honesty (hedge/abstain) intact on every seed with the flag on",
              a["all_unknown_abstained"], expect=True)
    v.control("does the reply preserve the fact -- IS-the-exact-clause rate, sentence-mode (raw) vs flags-off",
              treatment=a["raw_faithful"]["mean"] if a["raw_faithful"] else None,
              control=a["off_is_clause"]["mean"] if a["off_is_clause"] else None)
    v.control("does the reply preserve the fact -- IS-the-exact-clause rate, sentence-mode (raw) vs the "
              "older word-boost lever",
              treatment=a["raw_faithful"]["mean"] if a["raw_faithful"] else None,
              control=a["boost_is_clause"]["mean"] if a["boost_is_clause"] else None)
    # ATTRIBUTION (tools.lab, per gates/attribution_required): a treatment/control pair was just measured
    # above (sentence-mode's exact-clause rate vs the flags-off free-gen path's own exact-clause rate) --
    # asking WHOSE the difference is, not just reporting both numbers. The split here is clean (treatment=1.0,
    # control=0.0), so attribution reads 100%: the exact-clause-fidelity gain is caused ENTIRELY by the new
    # sentence-render wire-in, none of it by anything running identically in both arms.
    frac_attributable = attributable_to(
        "exact-clause fidelity, sentence-mode (raw) vs the pre-existing flags-off free-gen path",
        treatment_value=a["raw_faithful"]["mean"] if a["raw_faithful"] else None,
        control_value=a["off_is_clause"]["mean"] if a["off_is_clause"] else None)
    a["exact_clause_gain_fraction_attributable_to_wirein"] = frac_attributable

    go = bool(
        a["all_seeds_min_1_case"]
        and (a["wkv_used"]["min"] if a["wkv_used"] else 0) >= 0.999
        and (a["raw_readable"]["min"] if a["raw_readable"] else 0) >= 0.95
        and (a["raw_faithful"]["min"] if a["raw_faithful"] else 0) >= 0.95
        and (a["raw_moat_safe"]["min"] if a["raw_moat_safe"] else 0) >= 0.999
        and (a["answer_is_safe_degrade"]["min"] if a["answer_is_safe_degrade"] else 0) >= 0.999
        and (a["off_is_clause"]["mean"] if a["off_is_clause"] else 1) <= 0.001
        and a["all_unknown_abstained"]
    )
    verdict = v.decide(go=go)
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()
    print(f"\nVERDICT (wire-in itself measured on generate()'s RAW output; the post-filter's end-to-end rate "
          f"is reported separately and is bounded by a pre-existing, already-documented safety-net scope "
          f"limit -- see the mapped residual): {verdict}")
    return out


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "research/findings/raw/_wkv_mouth_fact_sentence_wirein_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
