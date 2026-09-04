import os, sys, json
os.environ["SIM_BACKEND"] = "cupy"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
os.environ["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed42.npz"
os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
sys.path.insert(0, ".")
from webapp import wkv_mouth_generator as WKV
from webapp import open_ended_chat as OE

from research.runners._wkv_fact_to_sentence_lexicon_lever import RELATION_LEXICON
print("lexicon relations:", sorted(RELATION_LEXICON.keys())[:10], "... total", len(RELATION_LEXICON))

facts_path = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k/facts.json")
data = json.loads(open(facts_path, encoding="utf-8").read())
covered_examples = []
for rec in data:
    f = rec.get("fact", rec) if isinstance(rec, dict) else None
    if not isinstance(f, dict) or f.get("polarity", "AFFIRM") != "AFFIRM":
        continue
    a, v, p = f.get("agent"), f.get("action"), f.get("patient")
    if v in RELATION_LEXICON:
        covered_examples.append((a, v, p))
    if len(covered_examples) >= 5:
        break
print("covered examples:", covered_examples)

for ex in covered_examples[:2]:
    sent = WKV.render_fact_sentence([ex], seed=42)
    print("render_fact_sentence:", ex, "->", sent)

text_with_sf, _ = WKV.generate("Tell me about it.", seed=42, sentence_facts=[covered_examples[0]])
text_without_sf, _ = WKV.generate("Tell me about it.", seed=42, sentence_facts=None)
print("WITH sentence_facts:", text_with_sf)
print("WITHOUT sentence_facts (free-gen):", text_without_sf)
print("differ:", text_with_sf != text_without_sf)

ltm_dir = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")
os.environ["BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK"] = "0"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE"] = "0"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND"] = "0"
r_lo = OE.answer_turn("Tell me about kanton genf", None, -0.9, 0.1, ltm_bundle=ltm_dir, brain_bundle=None, seed=42)
r_hi = OE.answer_turn("Tell me about kanton genf", None, 0.9, 0.9, ltm_bundle=ltm_dir, brain_bundle=None, seed=42)
print("valence=-0.9 generator:", r_lo["generator"], "raw:", r_lo["raw"])
print("valence=+0.9 generator:", r_hi["generator"], "raw:", r_hi["raw"])
print("raw identical across valence:", r_lo["raw"] == r_hi["raw"])
print("state differs (valence field itself):", r_lo["state"], "vs", r_hi["state"])

out = {
    "lexicon_relation_count": len(RELATION_LEXICON),
    "covered_examples": covered_examples,
    "sentence_facts_render": {str(ex): WKV.render_fact_sentence([ex], seed=42) for ex in covered_examples[:2]},
    "sentence_facts_vs_freegen": {"with_sf": text_with_sf, "without_sf": text_without_sf, "differ": text_with_sf != text_without_sf},
    "valence_isolation": {
        "lo_valence": -0.9, "hi_valence": 0.9,
        "raw_lo": r_lo["raw"], "raw_hi": r_hi["raw"],
        "raw_identical_across_valence": r_lo["raw"] == r_hi["raw"],
        "generator_lo": r_lo["generator"], "generator_hi": r_hi["generator"],
        "state_lo": r_lo["state"], "state_hi": r_hi["state"],
    },
}
with open("research/findings/raw/_linattn_live_verify_phase4_5.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote research/findings/raw/_linattn_live_verify_phase4_5.json")
