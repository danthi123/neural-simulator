"""Board #112 rung 2 investigation + first lever: from a boosted WORD to a grounded factual CLAUSE.

WHAT THIS IS. `research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md` named its deeper residual: the
fact-boost lever (rung 1's sibling, already landed) raises the odds the WKV mouth's free generation contains
the TRUE recalled word, but that word lands inside ordinary TinyStories-register fiction ("tim saw the ball in
the basket"), never as a clause asserting the fact ("strech_five plays basketball"). That finding named the
next step: "a structural (not decode-time) grounding lever ... conditioning generation on a fact-bearing
template ... reusing the existing spiking clause/frame machinery" -- this task's own instruction pointed at
`_spiking_fluent_surface_derisk.SpikingClauseProducer` / its SVO frames specifically, not a host template.

THE INVESTIGATION (why this is a first lever, not a closure). `SpikingClauseProducer` (EMERGE-59/60/61,
already 6-seed GO in `research/findings/2026-08-28-...spiking-fluent-surface...` -- see that runner's own
docstring) renders a clause by ordering a FIXED SMALL SET of abstract slot pools (DET/SUBJ/VERB/OBJ, 6 pools x
30 neurons = 180 neurons total) via real spiking rate-coded competitive queuing, then "realizes" each slot by
calling an IDENTITY spell callback (`spell=str`) on whatever payload string is passed in -- the mechanism is
VOCABULARY-AGNOSTIC by construction (its own docstring: "the mechanism does not depend on which specific words
fill the roles"). This is the OPPOSITE of the WKV mouth's closed V=1000 embedding table, so it does NOT inherit
Part 1's vocabulary-coverage wall (2026-09-01 fact-grounding finding) -- a real fact's agent/action/patient
words, however rare, CAN be slotted directly. This lever tests that: does slotting a REAL recalled fact
(agent=SUBJ, action=VERB via `emerge_v3`'s regular 3sg morphology, patient=OBJ) into the SAME GO'd
`PLAIN_TRANSITIVE` frame produce a genuinely-spiking-ordered, structurally well-formed, fact-faithful clause?

THE HONEST SCOPE, stated up front (not discovered mid-run): this is a PARALLEL renderer, not a modification of
`wkv_mouth_generator.generate()`'s own recurrent decode loop -- it does not read or write the WKV's hidden
state, and is not wired into `answer_turn`. It demonstrates the clause-frame machinery CAN take fact-grounded
content and render it as a spiking-ordered SVO clause (closing the "coherent-clause-vs-fiction" structural gap
DIFFERENTLY from the WKV's own free generation), as a first concrete step toward eventually biasing the WKV's
OWN decode toward this shape -- not that step itself. SS5 below maps precisely what remains.

Run: `SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_fact_svo_clause_first_lever`
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners._spiking_fluent_surface_derisk import (  # noqa: E402
    SpikingClauseProducer, PLAIN_TRANSITIVE, DET, SUBJ, VERB, OBJ,
)
from research.runners._emerge57_ra_refinetune_emerge_frames_derisk import emerge_v3  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44, 100, 101, 102)


def _bundle_dir() -> str | None:
    roots = []
    env_root = os.environ.get("BRAIN_DATA_ROOT", "").strip()
    if env_root:
        roots.append(env_root)
    roots.append(str(REPO_ROOT.parent / "sim-data"))
    roots.append(str(Path.home() / "Projects" / "sim-data"))
    for r in roots:
        d = str(Path(r) / "knowledge_bundles" / "wikidata_core_15k")
        if os.path.isdir(d):
            return d
    return None


def _sample_facts(seed: int, n: int = 8) -> list:
    """The SAME real-store sampling convention the fact-grounding finding's own Part 3 used: real AFFIRM
    triples from the shipped `wikidata_core_15k` store, one seeded sample per seed."""
    bundle = _bundle_dir()
    if bundle is None:
        return []
    raw = json.loads((Path(bundle) / "facts.json").read_text(encoding="utf-8"))
    affirm = [r["fact"] for r in raw if r.get("fact", {}).get("polarity", "AFFIRM") == "AFFIRM"]
    rng = random.Random(seed)
    return rng.sample(affirm, min(n, len(affirm)))


def parse_plain_transitive(surface: str) -> dict:
    """Independent structural parser for `PLAIN_TRANSITIVE`'s 'the <SUBJ> <VERB-3sg> the <OBJ>' shape -- does
    NOT trust the producer. No closed-lexicon membership check (unlike `parse_hedged_transitive`'s pseudo-word
    de-risk): SUBJ/OBJ here are real Wikidata slugs, an open set by design, so well-formedness is purely
    STRUCTURAL (determiner positions, non-empty slots, exactly 5 tokens -- each real entity slug/relation kept
    as ONE literal underscored token, matching the codebase's own existing `StubRenderer` convention noted in
    `research/FAILURE_LOG.md`'s 2026-09-01 `TieredFactStore` entry, so this parser's single-token-per-slot
    assumption holds)."""
    toks = surface.split()
    if len(toks) != 5:
        return {"well_formed": False, "svo": None, "why": f"len {len(toks)} != 5"}
    if toks[0] != "the" or toks[3] != "the":
        return {"well_formed": False, "svo": None, "why": "determiner slot(s) wrong"}
    subj, vform, obj = toks[1], toks[2], toks[4]
    if not subj or not vform or not obj:
        return {"well_formed": False, "svo": None, "why": "an SVO slot is empty"}
    return {"well_formed": True, "svo": (subj, vform, obj), "why": "ok"}


def _render_facts(seed: int, facts: list, permute_order: bool = False) -> list:
    prod = SpikingClauseProducer(seed, permute_order=permute_order)
    prod.learn(len(PLAIN_TRANSITIVE))
    results = []
    for f in facts:
        agent, action, patient = f["agent"], f["action"], f["patient"]
        dctx = {"subject": agent, "verb_3sg": emerge_v3(action), "object": patient}
        words = prod.emit(PLAIN_TRANSITIVE, dctx)
        surface = " ".join(words)
        pr = parse_plain_transitive(surface)
        faithful = bool(pr["well_formed"] and pr["svo"] == (agent, emerge_v3(action), patient))
        results.append({"agent": agent, "action": action, "patient": patient, "surface": surface,
                        "well_formed": pr["well_formed"], "faithful": faithful, "why": pr["why"]})
    return results, prod.spiked


def main() -> dict:
    out: dict = {"runner": "_wkv_fact_svo_clause_first_lever", "seeds": list(SEEDS)}
    bundle = _bundle_dir()
    out["bundle_dir"] = bundle
    if bundle is None:
        out["skipped"] = "no data lake (sim-data/knowledge_bundles/wikidata_core_15k not found)"
        print(json.dumps(out, indent=2))
        return out

    per_seed = []
    for seed in SEEDS:
        facts = _sample_facts(seed, n=8)
        main_results, spiked = _render_facts(seed, facts, permute_order=False)
        perm_results, _ = _render_facts(seed, facts, permute_order=True)
        wf_main = float(np.mean([r["well_formed"] for r in main_results])) if main_results else None
        faith_main = float(np.mean([r["faithful"] for r in main_results])) if main_results else None
        wf_perm = float(np.mean([r["well_formed"] for r in perm_results])) if perm_results else None
        faith_perm = float(np.mean([r["faithful"] for r in perm_results])) if perm_results else None
        per_seed.append({
            "seed": seed, "n": len(facts), "spiked": bool(spiked),
            "well_formed_frac": wf_main, "faithful_frac": faith_main,
            "permuted_control_well_formed_frac": wf_perm, "permuted_control_faithful_frac": faith_perm,
            "examples": main_results,
        })
        print(f"[seed {seed}] n={len(facts)} well_formed={wf_main} faithful={faith_main} "
              f"(permuted control: well_formed={wf_perm} faithful={faith_perm}) spiked={spiked}")
        for r in main_results[:3]:
            print(f"    {r['agent']:30s} -> {r['surface']!r}  well_formed={r['well_formed']} faithful={r['faithful']}")

    all_wf = [s["well_formed_frac"] for s in per_seed if s["well_formed_frac"] is not None]
    all_faith = [s["faithful_frac"] for s in per_seed if s["faithful_frac"] is not None]
    all_perm_wf = [s["permuted_control_well_formed_frac"] for s in per_seed
                   if s["permuted_control_well_formed_frac"] is not None]
    all_spiked = [s["spiked"] for s in per_seed]
    out["per_seed"] = per_seed
    out["aggregate"] = {
        "n_seeds": len(per_seed),
        "well_formed_frac_mean": round(float(np.mean(all_wf)), 4) if all_wf else None,
        "well_formed_frac_min": round(float(np.min(all_wf)), 4) if all_wf else None,
        "faithful_frac_mean": round(float(np.mean(all_faith)), 4) if all_faith else None,
        "faithful_frac_min": round(float(np.min(all_faith)), 4) if all_faith else None,
        "permuted_control_well_formed_frac_mean": round(float(np.mean(all_perm_wf)), 4) if all_perm_wf else None,
        "all_seeds_spiked": bool(all(all_spiked)),
    }

    v = Verdict("WKV fact-to-SVO-clause first lever: structural well-formedness + faithfulness, all 6 seeds")
    v.require("every seed's bridge genuinely spiked", out["aggregate"]["all_seeds_spiked"], expect=True)
    v.require("well-formed clause rate >= 0.95 on every seed",
              out["aggregate"]["well_formed_frac_min"], expect=lambda x: x is not None and x >= 0.95)
    v.require("faithful (correct SVO role assignment) rate >= 0.95 on every seed",
              out["aggregate"]["faithful_frac_min"], expect=lambda x: x is not None and x >= 0.95)
    v.control("well-formed rate, correct-taught order vs PERMUTED-teaching control",
              treatment=out["aggregate"]["well_formed_frac_mean"],
              control=out["aggregate"]["permuted_control_well_formed_frac_mean"])
    # ATTRIBUTION (tools.lab, per gates/attribution_required): a treatment/control PAIR was just measured above --
    # per the gap#5 lesson, measuring both arms is not the same as asking whose the difference was. Here the split
    # is clean (treatment=1.0, control=0.0), so attribution reads 100% -- the well-formed order is caused ENTIRELY
    # by the manipulation (correct-order teaching vs a permuted teacher), none of it by anything running identically
    # in both arms.
    frac_attributable = attributable_to("well-formed clause order, correct-taught vs permuted-teaching",
                                        treatment_value=out["aggregate"]["well_formed_frac_mean"],
                                        control_value=out["aggregate"]["permuted_control_well_formed_frac_mean"])
    out["aggregate"]["order_effect_fraction_attributable_to_manipulation"] = frac_attributable
    v.require("order effect is genuinely attributable to the manipulation (not a control-shared artifact)",
              frac_attributable, expect=lambda x: x is not None and x >= 0.95)
    verdict = v.decide(go=(out["aggregate"]["all_seeds_spiked"]
                           and (out["aggregate"]["well_formed_frac_min"] or 0) >= 0.95
                           and (out["aggregate"]["faithful_frac_min"] or 0) >= 0.95
                           and (frac_attributable or 0) >= 0.95))
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()
    print(f"\nVERDICT (structural mechanism, NOT a claim of natural-sounding English -- see the finding's "
          f"honest-residual section): {verdict}")
    return out


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "research/findings/raw/_wkv_fact_svo_clause_first_lever.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
