"""Board #112 rung 3: from a structurally well-formed but INCOHERENT SVO clause (rung 2's
`_wkv_fact_svo_clause_first_lever.py` first lever, `research/findings/2026-09-01-wkv-fact-to-svo-clause-first-
lever-investigation.md`) to a clause that reads as a genuine FACTUAL SENTENCE.

THE TWO RESIDUALS rung 2 named precisely, closed here:
  1. the "verb" was a naive relation-label morphology guess (`emerge_v3(action)`): `"follows"` -> `"followses"`
     (double-inflected), `"country_of_citizenship"` -> raw (untouched). Closed here by `RELATION_LEXICON`, a
     curated relation -> English-predicate map covering ALL 34 distinct relation types in the real shipped
     `wikidata_core_15k` store (verified by `_check_lexicon_coverage` below against the live facts.json), e.g.
     `employer` -> "works for", `place_of_birth` -> "was born in", `country_of_citizenship` -> "is a citizen of".
  2. entities rendered as raw underscored slugs (`asimov_isaac`, `u_s_of_a`) with NO NP structure. Closed here by
     `slug_to_np`, a small closed-class casing rule (underscore-split, Title-Case, a short list of connective
     words kept lowercase) -- e.g. `asimov_isaac` -> "Asimov Isaac", `united_kingom` -> "United Kingom" (the
     store's own truncated spelling preserved verbatim, exactly the honest-data-artifact precedent rung 2's own
     finding already named for this same slug).

BOTH drive the SAME already-6-seed-GO `SpikingClauseProducer` (`_spiking_fluent_surface_derisk.py`, EMERGE-
59/60/61) COMPLETELY UNMODIFIED -- this file only changes WHAT CONTENT is handed to `emit()` (the predicate
string, the NP strings, and which of two fixed determiner templates is used), never the producer's own spiking
order mechanism. Content is the brain's own recalled fact (agent/action/patient, unchanged from rung 2's
sampling); the lexicon + NP-casing are host articulation scaffolding of the SAME sanctioned Broca-like class
already in this codebase (`_FUNCTION_WORDS`, `_LEADIN_WORDS`, `_IRREGULAR_3SG`, the RA `VERBS` table) -- a small,
closed, hand-curated lookup table, not a learned/generative mechanism.

THE HONEST MAP OF WHAT REMAINS (mapped, not built here -- rung 2's own residual #3, still open):
  3. this is STILL a PARALLEL renderer, exactly like rung 2 -- it does not read or write
     `webapp/wkv_mouth_generator.py::generate()`'s own recurrent hidden state and is not wired into
     `answer_turn`. SS5 below states precisely what wiring it in would require.

For any relation type the lexicon does NOT cover (none exist in the live store today -- `RELATION_LEXICON`
covers all 34 -- but the mechanism must still degrade honestly for an unseen relation), the FALLBACK path is
BYTE-IDENTICAL to rung 2's own mechanism: naive `emerge_v3` morphology, raw underscored slugs, the fixed
`PLAIN_TRANSITIVE` template. `_check_fallback_byte_identical_to_rung2` asserts this directly against rung 2's
own `_render_facts` function, imported unchanged.

Run: SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_fact_to_sentence_lexicon_lever
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners._spiking_fluent_surface_derisk import (  # noqa: E402
    SpikingClauseProducer, DET, SUBJ, VERB, OBJ,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3  # noqa: E402
from research.runners._wkv_fact_svo_clause_first_lever import (  # noqa: E402
    _bundle_dir, _sample_facts, _render_facts as _rung2_render_facts, PLAIN_TRANSITIVE,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44, 100, 101, 102)


# =====================================================================================================================
# RESIDUAL 1: relation -> English predicate. Curated closed-class lookup (the sanctioned host-scaffold class).
# Value = (predicate_text, obj_article) where obj_article in {"the", "a_an", ""}:
#   "the"  -> the object NP takes the definite article ("works for the University Of Boston")
#   "a_an" -> the object NP takes the indefinite article, chosen by the object NP's own first letter
#             ("is a Human Specie" / "is an Association Football Club")
#   ""     -> no article (the predicate phrase already supplies the right shape: "is a type of the Y" reads
#             worse than "is a type of Y")
# Covers all 34 distinct relation types measured live in wikidata_core_15k/facts.json (verified below).
# =====================================================================================================================
RELATION_LEXICON: dict[str, tuple[str, str]] = {
    "country": ("is located in", "the"),
    "instance_of": ("is", "a_an"),
    "sport": ("is associated with the sport of", "the"),
    "located_in_time_zone": ("is located in the time zone of", "the"),
    "located_in_the_administrative_territoria": ("is located in", "the"),
    "subclass_of": ("is a type of", ""),
    "headquarters_location": ("is headquartered in", "the"),
    "shares_border_with": ("shares a border with", "the"),
    "language_of_work_or_name": ("is in the language of", "the"),
    "member_of": ("is a member of", "the"),
    "part_of": ("is part of", "the"),
    "occupation": ("works as", "a_an"),
    "taxon_rank": ("has the taxonomic rank of", ""),
    "country_of_citizenship": ("is a citizen of", "the"),
    "follows": ("follows", "the"),
    "followed_by": ("is followed by", "the"),
    "genre": ("is in the genre of", ""),
    "contains_administrative_territorial_enti": ("contains", "the"),
    "languages_spoken_written_or_signed": ("speaks", "the"),
    "country_of_origin": ("originates from", "the"),
    "award_received": ("received", "the"),
    "participant_of": ("participated in", "the"),
    "given_name": ("has the given name", ""),
    "place_of_birth": ("was born in", "the"),
    "place_of_death": ("died in", "the"),
    "educated_at": ("was educated at", "the"),
    "record_label": ("is signed to", "the"),
    "work_location": ("works in", "the"),
    "original_language_of_film_or_tv_show": ("was originally produced in", "the"),
    "member_of_political_party": ("is a member of the political party", ""),
    "position_held": ("holds the position of", ""),
    "parent_taxon": ("belongs to the taxon", "the"),
    "employer": ("works for", "the"),
    "family_name": ("has the family name", ""),
}


def _check_lexicon_coverage() -> dict:
    """Coverage against the REAL live store -- not asserted from memory. Fails loudly if a relation type exists
    in facts.json that the lexicon does not know about (the honest-map obligation is mechanical, not recalled)."""
    bundle = _bundle_dir()
    if bundle is None:
        return {"checked": False, "reason": "no data lake"}
    raw = json.loads((Path(bundle) / "facts.json").read_text(encoding="utf-8"))
    live_relations = sorted({r["fact"]["action"] for r in raw
                             if r.get("fact", {}).get("polarity", "AFFIRM") == "AFFIRM"})
    covered = [r for r in live_relations if r in RELATION_LEXICON]
    uncovered = [r for r in live_relations if r not in RELATION_LEXICON]
    return {"checked": True, "n_live_relations": len(live_relations), "n_covered": len(covered),
            "n_uncovered": len(uncovered), "uncovered": uncovered,
            "coverage_frac": round(len(covered) / len(live_relations), 4) if live_relations else None}


# =====================================================================================================================
# RESIDUAL 2: underscored slug -> natural(ish) NP. A small closed-class casing rule -- NOT a name-order inference
# (the store's own slug word order is preserved verbatim; see the honest limit in SS4 of the finding).
# =====================================================================================================================
_NP_LOWER_CONNECTORS = {"of", "the", "and", "in", "on", "for", "de", "la", "van", "der", "von", "al"}


def slug_to_np(slug: str) -> str:
    words = [w for w in slug.split("_") if w]
    # A slug whose OWN first word is "the" (e.g. "the_republic_of_turkey", a real store entry -- the formal
    # name's determiner got baked into the slug) is a DIFFERENT case from a mid-slug "the" like
    # "bounce_around_the_ground": every caller that places this NP into a clause (`_dctx_and_slots`/
    # `_slots_for`) already prepends its OWN fixed DET("the") immediately before the SUBJ/OBJ slot, so keeping
    # the slug's leading "the" doubles the determiner -- "the The Republic of Turkey" (one-brain Stage-2
    # Touchpoint-A build-ahead smoke finding, 2026-09-04, research/coordination/build_ahead_ready.md). A
    # mid-slug "the" is unaffected (i > 0 below already lowercases it as an ordinary connector). Drop the
    # leading one here rather than merely lowering it -- lowering alone would still leave "the the ..." (the
    # SAME word twice); a bare one-word slug of "the" (degenerate; no real store entry) is left untouched so
    # this can never reduce a one-word slug to nothing.
    if len(words) > 1 and words[0].lower() == "the":
        words = words[1:]
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _NP_LOWER_CONNECTORS:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out) if out else slug


def _article_for(np_text: str) -> str:
    first = np_text.split()[0] if np_text.split() else ""
    return "an" if first[:1].lower() in "aeiou" else "a"


# =====================================================================================================================
# TEMPLATE + REALIZATION: build the (still-fixed, still-typed) DET/SUBJ/VERB/[DET]/OBJ slot list per relation, and
# realize it through `SpikingClauseProducer.emit()` UNMODIFIED. Two distinct slot-list LENGTHS occur (5 for "the"/
# "a_an", 4 for ""); a producer is taught once per length actually used (an untaught cross-length reuse would
# silently blend two primacy gradients -- avoided by construction, not by accident).
# =====================================================================================================================
def _slots_for(article: str, obj_art_word: str | None = None):
    if article == "the":
        return [(DET, "the"), (SUBJ, None), (VERB, None), (DET, "the"), (OBJ, None)]
    if article == "":
        return [(DET, "the"), (SUBJ, None), (VERB, None), (OBJ, None)]
    if article == "a_an":
        return [(DET, "the"), (SUBJ, None), (VERB, None), (DET, obj_art_word), (OBJ, None)]
    raise ValueError(article)


def _dctx_and_slots(agent: str, action: str, patient: str):
    """Return (slots, dctx, covered). `covered=False` -> the BYTE-IDENTICAL rung-2 fallback shape."""
    entry = RELATION_LEXICON.get(action)
    if entry is None:
        return PLAIN_TRANSITIVE, {"subject": agent, "verb_3sg": emerge_v3(action), "object": patient}, False
    predicate, article = entry
    subj_np, obj_np = slug_to_np(agent), slug_to_np(patient)
    art_word = _article_for(obj_np) if article == "a_an" else None
    slots = _slots_for(article, art_word)
    dctx = {"subject": subj_np, "verb_3sg": predicate, "object": obj_np}
    return slots, dctx, True


def expected_surface(agent: str, action: str, patient: str) -> tuple[str, bool, int]:
    """Independent ground-truth reconstruction -- built directly from the fact + the SAME lexicon/NP-casing rule
    used to drive rendering (ground truth reused as data, exactly how rung 2's own parser reused `emerge_v3` as
    the ground truth for verb agreement), NEVER by asking the producer what it intended. Returns
    (expected_surface_in_correct_order, covered, expected_token_count)."""
    entry = RELATION_LEXICON.get(action)
    if entry is None:
        toks = ["the", agent, emerge_v3(action), "the", patient]
        return " ".join(toks), False, len(toks)
    predicate, article = entry
    subj_np, obj_np = slug_to_np(agent), slug_to_np(patient)
    if article == "the":
        toks = ["the"] + subj_np.split() + predicate.split() + ["the"] + obj_np.split()
    elif article == "":
        toks = ["the"] + subj_np.split() + predicate.split() + obj_np.split()
    else:
        toks = ["the"] + subj_np.split() + predicate.split() + [_article_for(obj_np)] + obj_np.split()
    return " ".join(toks), True, len(toks)


# =====================================================================================================================
# THE INDEPENDENT PARSER -- does NOT trust the producer. `faithful` = exact match to the independently-
# reconstructed correct-order surface (this is STRONGER than rung 2's own faithful check, and doubles as the
# order-causation check: under the permuted-teaching control the emitted slot ORDER differs, so the exact match
# fails, exactly mirroring rung 2's own permuted-control-collapses design). `well_formed` = a SEPARATE, weaker,
# purely-positional structural check (the predicate phrase must appear at the position it would occupy if the
# canonical DET-SUBJ-VERB-[DET]-OBJ order held) -- independent of whether the exact NP content also matches.
# `readable` = covered AND faithful AND no leftover underscore character (the composite "coherent clause" bar).
# =====================================================================================================================
def parse_and_score(surface: str, agent: str, action: str, patient: str) -> dict:
    exp, covered, exp_len = expected_surface(agent, action, patient)
    toks = surface.split()
    if not covered:
        well_formed = (len(toks) == 5 and toks[0] == "the" and toks[3] == "the"
                       and all(t for t in toks))
    else:
        entry = RELATION_LEXICON[action]
        predicate, article = entry
        subj_np = slug_to_np(agent)
        pred_toks = predicate.split()
        i0 = 1 + len(subj_np.split())
        pred_slice = toks[i0:i0 + len(pred_toks)] if len(toks) >= i0 + len(pred_toks) else []
        well_formed = (len(toks) == exp_len and toks and toks[0] == "the"
                       and pred_slice == pred_toks and all(t for t in toks))
    faithful = (surface == exp)
    readable = bool(covered and faithful and ("_" not in surface))
    # moat sanity: every token in the surface must come from the fact's own content + fixed closed-class words
    # (subject NP, predicate, object NP, determiners) -- no fabricated token could appear here by construction,
    # checked directly rather than merely asserted.
    allowed = set((slug_to_np(agent) + " " + (RELATION_LEXICON.get(action, (emerge_v3(action), ""))[0])
                  + " " + slug_to_np(patient)).lower().split()) | {"the", "a", "an", agent, patient,
                                                                    emerge_v3(action)}
    # `.rstrip(".!?")` on each token: `webapp.wkv_mouth_generator.render_fact_sentence` (2026-09-04 Touchpoint-A
    # Stage-2 prose fix) now appends a sentence-final "." directly onto the last word (proper typography -- no
    # preceding space), so a genuinely moat-safe surface's LAST token can legitimately be e.g. "boston." rather
    # than "boston". Stripping a trailing terminator before the membership check recognizes these as the SAME
    # word; it cannot let a fabricated token through (a token failing membership before the strip still fails
    # after it unless the trim exposes an allowed word, which only ever happens for this exact sentence-final-
    # punctuation case) and is a no-op for every pre-existing caller whose tokens never carried one.
    moat_safe = all(t.lower().rstrip(".!?") in allowed for t in toks)
    return {"well_formed": bool(well_formed), "faithful": bool(faithful), "readable": readable,
            "covered": covered, "moat_safe": bool(moat_safe), "expected": exp}


# =====================================================================================================================
# RENDER (main + permuted-teaching control), grouped by the template LENGTH each fact actually needs.
# =====================================================================================================================
def _render_facts(seed: int, facts: list, permute_order: bool = False):
    by_len: dict[int, SpikingClauseProducer] = {}
    results = []
    any_spiked = False
    for f in facts:
        agent, action, patient = f["agent"], f["action"], f["patient"]
        slots, dctx, covered = _dctx_and_slots(agent, action, patient)
        n = len(slots)
        prod = by_len.get(n)
        if prod is None:
            prod = SpikingClauseProducer(seed, permute_order=permute_order)
            prod.learn(n)
            by_len[n] = prod
        words = prod.emit(slots, dctx)
        surface = " ".join(words)
        pr = parse_and_score(surface, agent, action, patient)
        any_spiked = any_spiked or prod.spiked
        results.append({"agent": agent, "action": action, "patient": patient, "surface": surface, **pr})
    return results, any_spiked


def _check_fallback_byte_identical_to_rung2(seed: int) -> dict:
    """The 'byte-identical-off' analog for a runner with no boolean flag (this remains a parallel renderer, not
    wired into any default-on path): for an UNCOVERED relation, the new mechanism's output must be BYTE-
    IDENTICAL to rung 2's own `_render_facts`, called unmodified. Proves the new content-realization layer adds
    nothing and changes nothing when it does not apply."""
    fake_fact = {"agent": "zzz_synthetic_agent", "action": "zzz_totally_uncovered_relation_xyz",
                 "patient": "zzz_synthetic_patient", "polarity": "AFFIRM"}
    assert fake_fact["action"] not in RELATION_LEXICON
    new_results, _ = _render_facts(seed, [fake_fact])
    rung2_results, _ = _rung2_render_facts(seed, [fake_fact])
    new_surface = new_results[0]["surface"]
    rung2_surface = rung2_results[0]["surface"]
    return {"new": new_surface, "rung2": rung2_surface, "byte_identical": bool(new_surface == rung2_surface)}


def main() -> dict:
    out: dict = {"runner": "_wkv_fact_to_sentence_lexicon_lever", "seeds": list(SEEDS)}
    bundle = _bundle_dir()
    out["bundle_dir"] = bundle
    if bundle is None:
        out["skipped"] = "no data lake (sim-data/knowledge_bundles/wikidata_core_15k not found)"
        print(json.dumps(out, indent=2))
        return out

    coverage = _check_lexicon_coverage()
    out["lexicon_coverage_vs_live_store"] = coverage
    print(f"Lexicon coverage vs live store: {coverage}")

    per_seed = []
    for seed in SEEDS:
        facts = _sample_facts(seed, n=8)
        main_results, spiked = _render_facts(seed, facts, permute_order=False)
        perm_results, _ = _render_facts(seed, facts, permute_order=True)
        fallback_check = _check_fallback_byte_identical_to_rung2(seed)

        def frac(rs, k):
            return float(np.mean([r[k] for r in rs])) if rs else None

        row = {
            "seed": seed, "n": len(facts), "spiked": bool(spiked),
            "well_formed_frac": frac(main_results, "well_formed"),
            "faithful_frac": frac(main_results, "faithful"),
            "readable_frac": frac(main_results, "readable"),
            "moat_safe_frac": frac(main_results, "moat_safe"),
            "covered_frac": frac(main_results, "covered"),
            "permuted_control_well_formed_frac": frac(perm_results, "well_formed"),
            "permuted_control_faithful_frac": frac(perm_results, "faithful"),
            "fallback_byte_identical_to_rung2": fallback_check["byte_identical"],
            "examples": main_results,
        }
        per_seed.append(row)
        print(f"[seed {seed}] n={len(facts)} covered={row['covered_frac']} well_formed={row['well_formed_frac']} "
              f"faithful={row['faithful_frac']} readable={row['readable_frac']} moat_safe={row['moat_safe_frac']} "
              f"(permuted control: well_formed={row['permuted_control_well_formed_frac']} "
              f"faithful={row['permuted_control_faithful_frac']}) spiked={spiked} "
              f"fallback_byte_identical={fallback_check['byte_identical']}")
        for r in main_results[:4]:
            print(f"    {r['agent']:30s} {r['action']:45s} -> {r['surface']!r}  "
                  f"readable={r['readable']} faithful={r['faithful']}")

    def agg(key):
        vals = [s[key] for s in per_seed if s[key] is not None]
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4)} if vals else None

    out["per_seed"] = per_seed
    out["aggregate"] = {
        "n_seeds": len(per_seed),
        "well_formed": agg("well_formed_frac"),
        "faithful": agg("faithful_frac"),
        "readable": agg("readable_frac"),
        "moat_safe": agg("moat_safe_frac"),
        "covered": agg("covered_frac"),
        "permuted_control_well_formed": agg("permuted_control_well_formed_frac"),
        "permuted_control_faithful": agg("permuted_control_faithful_frac"),
        "all_seeds_spiked": bool(all(s["spiked"] for s in per_seed)),
        "all_seeds_fallback_byte_identical": bool(all(s["fallback_byte_identical_to_rung2"] for s in per_seed)),
    }

    a = out["aggregate"]
    v = Verdict("WKV fact-to-sentence lexicon lever: coherent-clause rate, all 6 seeds")
    v.require("every seed's bridge genuinely spiked", a["all_seeds_spiked"], expect=True)
    v.require("lexicon covers every relation type live in the real store",
              coverage.get("coverage_frac"), expect=lambda x: x is not None and x >= 1.0)
    v.require("readable (coherent-clause) rate >= 0.95 on every seed",
              a["readable"]["min"] if a["readable"] else None, expect=lambda x: x is not None and x >= 0.95)
    v.require("faithful rate >= 0.95 on every seed",
              a["faithful"]["min"] if a["faithful"] else None, expect=lambda x: x is not None and x >= 0.95)
    v.require("moat-safe (no fabricated tokens) on every seed, every fact",
              a["moat_safe"]["min"] if a["moat_safe"] else None, expect=lambda x: x is not None and x >= 0.999)
    v.require("fallback path byte-identical to rung 2 on every seed",
              a["all_seeds_fallback_byte_identical"], expect=True)
    v.control("faithful rate, correct-taught order vs PERMUTED-teaching control",
              treatment=a["faithful"]["mean"] if a["faithful"] else None,
              control=a["permuted_control_faithful"]["mean"] if a["permuted_control_faithful"] else None)
    frac_attributable = attributable_to(
        "faithful clause order, correct-taught vs permuted-teaching",
        treatment_value=a["faithful"]["mean"] if a["faithful"] else None,
        control_value=a["permuted_control_faithful"]["mean"] if a["permuted_control_faithful"] else None)
    a["order_effect_fraction_attributable_to_manipulation"] = frac_attributable
    v.require("order effect is genuinely attributable to the manipulation (not a control-shared artifact)",
              frac_attributable, expect=lambda x: x is not None and x >= 0.95)

    go = bool(
        a["all_seeds_spiked"]
        and (coverage.get("coverage_frac") or 0) >= 1.0
        and (a["readable"]["min"] if a["readable"] else 0) >= 0.95
        and (a["faithful"]["min"] if a["faithful"] else 0) >= 0.95
        and (a["moat_safe"]["min"] if a["moat_safe"] else 0) >= 0.999
        and a["all_seeds_fallback_byte_identical"]
        and (frac_attributable or 0) >= 0.95
    )
    verdict = v.decide(go=go)
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()
    print(f"\nVERDICT (lexicon-driven coherent-clause lever -- STILL a PARALLEL renderer, not wired into "
          f"wkv_mouth_generator.generate(); see the finding's SS5): {verdict}")
    return out


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "research/findings/raw/_wkv_fact_to_sentence_lexicon_lever.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
