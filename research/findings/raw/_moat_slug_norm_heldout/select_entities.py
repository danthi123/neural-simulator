"""HELD-OUT ENTITY SELECTION for the BRAIN_MOAT_SLUG_NORM validation -- the rule, written BEFORE any held-out sentence was
generated or looked at (the commit that adds this file precedes the generation run).

WHY HELD OUT. The memo's article/relation-name guard ("X is a country in Y") was written after seeing its one leak on the
memo corpus, so that corpus (4 gathered sets: angora_turkey x2, ukraine, l_quipe_de_france) cannot validate it.

RULE (fixed here, applied by this script, not tuned afterwards)
  store  : the production wikidata_100k bundle facts.json (the store the memo's turns ran over).
  pool   : every entity that is the AGENT of >= 4 AFFIRM facts in that file.
  order  : lexicographic by slug (Python str order).
  pick   : every N-th entity of the ordered pool starting at index 0, N = len(pool) // 28   (=> ~28 candidates).
  exclude: any pick whose slug occurs (as agent or patient) in a fact of the memo's four gathered sets; excluded picks are
           dropped, NOT replaced.
  drive  : one fresh-session RICH turn per pick, "What is <slug>?" (the memo's x3/x4 form), BRAIN_CHAT_SEED=1, production
           renderer = off-bridge Qwen-0.5B on CPU, flag BRAIN_MOAT_SLUG_NORM UNSET while generating so every fact gets both
           its render_svo and render_svo_regen call recorded (mouth_trace).
  keep   : every turn whose mouth_trace has >= 1 call is part of the corpus; turns with none are reported as
           "no facts gathered" and are not replaced either.
No outcome of any verifier is consulted by this rule.
"""
import json, sys

FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
# slugs of the memo's four gathered sets (moat-memo/corpus.py output, 2026-10-18)
MEMO_SLUGS = {
    "angora_turkey", "located_in_time_zone", "kaliningrad_time", "instance_of", "city_work", "country",
    "the_republic_of_turkey", "twinned_administrative_body", "washington_dc", "described_by_source", "catholic_encyclopedia",
    "ukraine", "west_africa_summer_time", "shares_border_with", "pole_land", "brockhaus_and_efron_encyclopaedia",
    "contains_administrative_territorial_enti", "m_kyiv",
    "l_quipe_de_france", "sport", "association_football_club", "armenian_soviet_encyclopedia", "hayastani_hanrapetut_yun",
    "franzoesische_republik", "participant_of", "world_championship_football_2010",
}
ENTITY_SLUGS = MEMO_SLUGS - {"located_in_time_zone", "instance_of", "country", "twinned_administrative_body",
                              "described_by_source", "shares_border_with", "contains_administrative_territorial_enti",
                              "sport", "participant_of"}   # relation names are not entities


def select():
    import collections
    cnt = collections.Counter()
    for r in json.load(open(FACTS)):
        f = r["fact"]
        if f.get("polarity") == "AFFIRM":
            cnt[f["agent"]] += 1
    pool = sorted(e for e, c in cnt.items() if c >= 4)
    n = len(pool) // 28
    picks = pool[0::n]
    kept = [e for e in picks if e not in ENTITY_SLUGS]
    return {"pool_size": len(pool), "N": n, "picks": picks, "excluded_memo_overlap": [e for e in picks if e in ENTITY_SLUGS],
            "entities": kept, "facts_per_entity": {e: cnt[e] for e in kept}}


if __name__ == "__main__":
    out = select()
    out["backend"] = "numpy"
    out["script"] = "research/findings/raw/_moat_slug_norm_heldout/select_entities.py"
    out["argv"] = sys.argv
    out["_doc"] = "output of select_entities.py (rule in its docstring); no verifier was run before this file was written"
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(out["pool_size"], out["N"], len(out["picks"]), len(out["entities"]), out["excluded_memo_overlap"])
