"""S3 ENTITY SELECTION for the BRAIN_MOAT_SLUG_NORM one-fact-routing stratum -- the rule, written BEFORE any S3 sentence was
generated or looked at (the commit that adds this file + s3_entities.json precedes the generation run).

WHY. With BRAIN_MOAT_SLUG_NORM on, RichAnswerComposer._verify_rendered sends a ONE-fact turn through the claim verifier
instead of the single-triple ChatBrain._verify. Stratum S1 (select_entities.py) kept only agents with >= 4 AFFIRM facts and so
excluded 65.6 % of the store's agents; no real Qwen sentence for a small entity existed, so that routing had no real-data record.

RULE (fixed here, applied by this script, not tuned afterwards)
  store    : the production wikidata_100k bundle facts.json (the store the turns run over), AFFIRM facts only.
  count(e) : number of AFFIRM facts whose agent is e.
  frame A  ("natural"): every entity with 1 <= count(e) <= 3 (15,301 entities).  Ordered lexicographically by slug.
             N_A = len(frame) // 30 ; picks = frame[N_A//2 :: N_A]   (every N_A-th, starting at offset N_A//2)  => ~30 picks.
  frame B  ("leaf")   : entities with count(e) == 1 whose single patient is never the agent of any fact AND which never occurs
             as the patient of any fact (so the rich gather has nothing to chain to or elaborate from; the turn is expected to
             gather exactly one fact).  Ordered lexicographically.  N_B = len(frame) // 30 ; picks = frame[N_B//2 :: N_B]
             => ~30 picks.
  exclude  : any pick that is a slug of the memo's four gathered sets or an S1 entity (entities.json) is dropped, NOT replaced;
             a frame-B pick already picked in frame A is dropped from B (kept in A), NOT replaced.
  drive    : one fresh-session RICH turn per pick, "What is <slug>?" (the S1 form), BRAIN_CHAT_SEED=1, the production renderer =
             off-bridge Qwen-0.5B on the GPU (fp16, `qwen-gpu`), flag BRAIN_MOAT_SLUG_NORM UNSET while generating so every
             fact gets both its render_svo and render_svo_regen call recorded (mouth_trace).  Turn labels s3a01.. / s3b01..
             (frame order).  One GPU-queue job; ~12 turns per process (a 29-turn process was OOM-killed under a 12 GB cap).
  member   : the S3 stratum = every generated turn whose RichAnswerComposer.render_paragraph gathered set has 1..3 facts
             ("gathered" = the facts handed to render_paragraph, as in S1).  The ONE-FACT ROUTING subset = turns whose gathered
             set has exactly 1 fact.  Turns with >= 4 gathered facts are reported separately ("overflow", measured, not in S3);
             turns with no facts are reported as "no facts gathered".  Nothing is replaced or re-drawn after the fact.
  labels   : every unique (turn, svo, surface) sentence is hand-labelled F/B/U/N under the S1 rubric (make_labels.py) BEFORE any
             verifier verdict on S3 exists; the labels file is committed before the measurement.
No outcome of any verifier is consulted by this rule.
"""
import collections, json, sys

import research.runners  # noqa: F401  (the provenance door: stamps the output artifact with run id, host, backend)

FACTS = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
S1_ENTITIES = "research/findings/raw/_moat_slug_norm_heldout/entities.json"
MEMO = {  # slugs of the memo's four gathered sets (same list as select_entities.py, entities + relation names)
    "angora_turkey", "located_in_time_zone", "kaliningrad_time", "instance_of", "city_work", "country",
    "the_republic_of_turkey", "twinned_administrative_body", "washington_dc", "described_by_source", "catholic_encyclopedia",
    "ukraine", "west_africa_summer_time", "shares_border_with", "pole_land", "brockhaus_and_efron_encyclopaedia",
    "contains_administrative_territorial_enti", "m_kyiv",
    "l_quipe_de_france", "sport", "association_football_club", "armenian_soviet_encyclopedia", "hayastani_hanrapetut_yun",
    "franzoesische_republik", "participant_of", "world_championship_football_2010",
}
K = 30


def select():
    cnt = collections.Counter(); patients = collections.defaultdict(list); is_patient = set()
    for r in json.load(open(FACTS)):
        f = r["fact"]
        if f.get("polarity") != "AFFIRM":
            continue
        cnt[f["agent"]] += 1
        patients[f["agent"]].append(f["patient"]); is_patient.add(f["patient"])
    s1 = set(json.load(open(S1_ENTITIES))["entities"])
    excl = MEMO | s1
    frame_a = sorted(e for e, c in cnt.items() if 1 <= c <= 3)
    n_a = len(frame_a) // K
    picks_a = frame_a[n_a // 2::n_a]
    frame_b = sorted(e for e, c in cnt.items() if c == 1 and e not in is_patient and patients[e][0] not in cnt)
    n_b = len(frame_b) // K
    picks_b = frame_b[n_b // 2::n_b]
    ent_a = [e for e in picks_a if e not in excl]
    ent_b = [e for e in picks_b if e not in excl and e not in set(ent_a)]
    turns = [["s3a%02d" % i, e, "A"] for i, e in enumerate(ent_a, 1)] + [["s3b%02d" % i, e, "B"] for i, e in enumerate(ent_b, 1)]
    return {"frame_a_size": len(frame_a), "N_A": n_a, "picks_a": picks_a, "excluded_a": [e for e in picks_a if e in excl],
            "frame_b_size": len(frame_b), "N_B": n_b, "picks_b": picks_b,
            "excluded_b": [e for e in picks_b if e in excl or e in set(ent_a)],
            "turns": turns, "facts_per_entity": {e: cnt[e] for _l, e, _f in turns},
            "patients": {e: patients[e] for _l, e, _f in turns}}


if __name__ == "__main__":
    out = select()
    out["backend"] = "numpy"
    out["device"] = "cpu"
    out["script"] = "research/findings/raw/_moat_slug_norm_heldout/select_s3.py"
    out["argv"] = sys.argv
    out["_doc"] = "output of select_s3.py (rule in its docstring); no verifier was run and no S3 sentence existed when this was written"
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print("frame A", out["frame_a_size"], "N", out["N_A"], "picks", len(out["picks_a"]), "excluded", out["excluded_a"])
    print("frame B", out["frame_b_size"], "N", out["N_B"], "picks", len(out["picks_b"]), "excluded", out["excluded_b"])
    print("turns", len(out["turns"]))
