import os, sys, json
os.environ["SIM_BACKEND"] = "cupy"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
os.environ["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed42.npz"
os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"
sys.path.insert(0, ".")
from webapp import wkv_mouth_generator as WKV

prompt = "Tell me about kanton genf."
facts_A = [('kanton_genf', 'located_in_time_zone', 'rome_time'), ('kanton_genf', 'country', 'conf_d_ration_suisse'), ('kanton_genf', 'shares_border_with', 'ain_d_partement')]
facts_B = [('history_of_rochester_minnesota', 'country', 'u_s_of_a'), ('history_of_rochester_minnesota', 'instance_of', 'city_work'), ('history_of_rochester_minnesota', 'located_in_time_zone', 'central_daylight_time')]

def gen(facts, boost=6.0):
    text, secs = WKV.generate(prompt, seed=42, max_new_tokens=50, repetition_penalty=1.3,
                               no_repeat_ngram_size=3, facts=facts, fact_boost=boost)
    return text

out_none = gen(None)
out_A = gen(facts_A)
out_B = gen(facts_B)
out_A_noboost = gen(facts_A, boost=0.0)

print("FACTS=None:      ", out_none)
print("FACTS=A (genf):  ", out_A)
print("FACTS=B (roch):  ", out_B)
print("FACTS=A boost=0: ", out_A_noboost)
print()
print("None == A_noboost (lesion should match None):", out_none == out_A_noboost)
print("A == B (should differ if grounded):", out_A == out_B)
print("A == None (boost should differ from no-facts):", out_A == out_none)

out = {
    "prompt": prompt, "seed": 42, "fact_boost_default": 6.0,
    "facts_A": facts_A, "facts_B": facts_B,
    "out_none": out_none, "out_A": out_A, "out_B": out_B, "out_A_noboost": out_A_noboost,
    "lesion_boost0_matches_none": out_none == out_A_noboost,
    "A_differs_from_B": out_A != out_B,
    "A_differs_from_none": out_A != out_none,
    "note": "A/B outputs are non-linguistic repetitive BPE-fragment garbage at boost=6.0 (the caller's own "
            "default, used unmodified by webapp/open_ended_chat.py's WKV-mouth branch) -- see the finding doc "
            "for why: fact_grounding_ids matches mostly 1-3-letter slug fragments against the BPE vocabulary, "
            "not real content words, and the resulting boost overwhelms the decode.",
}
with open("research/findings/raw/_linattn_live_verify_phase3_fact_boost.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote research/findings/raw/_linattn_live_verify_phase3_fact_boost.json")
