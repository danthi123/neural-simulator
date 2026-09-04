import os, sys, json, time
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ["BRAIN_OPEN_ENDED"] = "1"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
os.environ["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed42.npz"
os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
os.environ["BRAIN_OPEN_ENDED_NP_ENTAILMENT"] = "0"
os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "0"
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"
sys.path.insert(0, ".")

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

import webapp.server as S

RENDERER = "stub"
SESSION = "linattn_live_verify"

def chat(msg, reset, extra_env=None):
    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
    resp = S.brain_chat(S.BrainChatRequest(session=SESSION, message=msg, brain="tiny-demo",
                                            reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))

rows = {}

log("=== turn 1: build session (default flags -- fact_sentence/fact_clause ON) known topic (underscored slug -- ")
log("    extract_topic/retrieve is an exact-lowercase-match on the store's own agent key, no space normalization) ===")
d1 = chat("Tell me about frank_lincoln_wright", reset=True)
oe1 = d1.get("open_ended") or {}
log("generator:", oe1.get("generator"), "known:", oe1.get("known"), "affect:", d1.get("affect"))
rows["known_default_flags"] = {"topic": "frank_lincoln_wright", "oe": oe1, "affect": d1.get("affect")}

log("=== turn 1b: a DIFFERENT known topic, default flags -- do two different real entities get two different ===")
log("     correct spiking_clause renders (the real production grounding path)? ===")
d1b = chat("Tell me about harold_clayton_lloyd", reset=False)
oe1b = d1b.get("open_ended") or {}
log("generator:", oe1b.get("generator"), "known:", oe1b.get("known"), "answer:", d1b.get("answer"))
rows["known_default_flags_topic2"] = {"topic": "harold_clayton_lloyd", "oe": oe1b, "answer": d1b.get("answer")}

log("=== turn 2: same known topic, fact routing forced OFF (force raw linattn free-gen) ===")
d2 = chat("Tell me about frank_lincoln_wright", reset=False,
          extra_env={"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"})
oe2 = d2.get("open_ended") or {}
log("generator:", oe2.get("generator"), "raw:", oe2.get("raw"))
rows["known_fact_routing_off"] = {"topic": "frank_lincoln_wright", "oe": oe2, "affect": d2.get("affect")}

log("=== turn 2b: a DIFFERENT known topic, fact routing OFF + fact_ground boost ON -- does the raw free-gen ===")
log("     actually change content between two different real entities' facts (the decode-boost grounding lever)? ===")
d2b = chat("Tell me about harold_clayton_lloyd", reset=False,
           extra_env={"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0",
                      "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND": "1"})
oe2b = d2b.get("open_ended") or {}
log("generator:", oe2b.get("generator"), "raw:", oe2b.get("raw"))
rows["known_fact_routing_off_boost_on_topic2"] = {"topic": "harold_clayton_lloyd", "oe": oe2b}
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND"] = "0"

log("=== turn 3: unknown topic (moat: brain-unknown, should abstain / hedge) ===")
d3 = chat("Tell me about zorplaxian quibberflax", reset=False,
          extra_env={"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "1", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "1"})
oe3 = d3.get("open_ended") or {}
log("generator:", oe3.get("generator"), "known:", oe3.get("known"), "answer:", d3.get("answer"))
rows["unknown_topic"] = {"topic": "zorplaxian quibberflax", "oe": oe3, "answer": d3.get("answer"), "abstained": d3.get("abstained")}

log("=== turn 4: dangerous topic (Qwen-known / brain-unknown famous entity) ===")
d4 = chat("Tell me about Albert Einstein", reset=False)
oe4 = d4.get("open_ended") or {}
log("generator:", oe4.get("generator"), "known:", oe4.get("known"), "answer:", d4.get("answer"))
rows["dangerous_topic"] = {"topic": "Albert Einstein", "oe": oe4, "answer": d4.get("answer"), "abstained": d4.get("abstained")}

log("=== turn 5+6: affect-vary via different messages, same forced-free-gen known topic ===")
d5 = chat("I am so thrilled and happy right now! Tell me about frank lincoln wright", reset=False,
          extra_env={"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"})
d6 = chat("I feel awful and miserable today. Tell me about frank lincoln wright", reset=False)
oe5, oe6 = d5.get("open_ended") or {}, d6.get("open_ended") or {}
log("affect5:", d5.get("affect"), "affect6:", d6.get("affect"))
log("topic5:", oe5.get("topic"), "topic6:", oe6.get("topic"))
rows["affect_vary_natural"] = {
    "msg5": "I am so thrilled and happy right now! Tell me about frank lincoln wright",
    "msg6": "I feel awful and miserable today. Tell me about frank lincoln wright",
    "affect5": d5.get("affect"), "affect6": d6.get("affect"),
    "oe5": oe5, "oe6": oe6,
}

log("=== turn 7+8: BRAIN_AFFECT_LESION off vs on, SAME message ===")
d7 = chat("Tell me about frank lincoln wright", reset=False, extra_env={"BRAIN_AFFECT_LESION": "0"})
d8 = chat("Tell me about frank lincoln wright", reset=False, extra_env={"BRAIN_AFFECT_LESION": "1"})
oe7, oe8 = d7.get("open_ended") or {}, d8.get("open_ended") or {}
log("affect7 (lesion=0):", d7.get("affect"))
log("affect8 (lesion=1):", d8.get("affect"))
log("raw7 == raw8:", oe7.get("raw") == oe8.get("raw"))
rows["affect_lesion"] = {
    "affect_lesion0": d7.get("affect"), "affect_lesion1": d8.get("affect"),
    "raw_lesion0": oe7.get("raw"), "raw_lesion1": oe8.get("raw"),
    "raw_identical": oe7.get("raw") == oe8.get("raw"),
}
os.environ["BRAIN_AFFECT_LESION"] = "0"

out_path = "research/findings/raw/_linattn_live_verify_phase7_pipeline.json"
with open(out_path, "w") as fh:
    json.dump({"runner": "linattn_live_verify_phase7 (ad hoc, hand-authored)", "seed": 42,
               "backend": os.environ.get("SIM_BACKEND"), "rows": rows,
               "wall_seconds": round(time.time() - T0, 1)}, fh, indent=1)
log("wrote", out_path)
