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
from research.runners._open_ended_state_driven_generation_derisk import (
    uncertainty_signaled, _UNKNOWN_ENTITIES, _QWEN_KNOWN_STORE_UNKNOWN,
)

RENDERER = "stub"
SESSION = "linattn_moat_soak"

def chat(msg, reset, extra_env=None):
    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
    resp = S.brain_chat(S.BrainChatRequest(session=SESSION, message=msg, brain="tiny-demo",
                                            reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))

unknown_topics = _UNKNOWN_ENTITIES[:5]
dangerous_topics = _QWEN_KNOWN_STORE_UNKNOWN[:5]
known_topics = ["frank_lincoln_wright", "harold_clayton_lloyd", "atlantic_jazz"]

rows = {"unknown": [], "dangerous": [], "known_default": [], "known_stress_fact_routing_off": []}
first = True
for topic in unknown_topics:
    d = chat(f"Tell me about {topic}", reset=first)
    first = False
    oe = d.get("open_ended") or {}
    raw, final = oe.get("raw") or "", oe.get("filtered") or d.get("answer") or ""
    row = {"topic": topic, "generator": oe.get("generator"), "known": oe.get("known"),
           "raw": raw, "filtered": final,
           "fab_raw": bool(not uncertainty_signaled(raw)), "fab_filtered": bool(not uncertainty_signaled(final)),
           "abstained": d.get("abstained")}
    rows["unknown"].append(row)
    log("UNKNOWN", topic, "fab_raw:", row["fab_raw"], "fab_filtered:", row["fab_filtered"])

for topic in dangerous_topics:
    d = chat(f"Tell me about {topic}", reset=False)
    oe = d.get("open_ended") or {}
    raw, final = oe.get("raw") or "", oe.get("filtered") or d.get("answer") or ""
    row = {"topic": topic, "generator": oe.get("generator"), "known": oe.get("known"),
           "raw": raw, "filtered": final,
           "fab_raw": bool(not uncertainty_signaled(raw)), "fab_filtered": bool(not uncertainty_signaled(final)),
           "abstained": d.get("abstained")}
    rows["dangerous"].append(row)
    log("DANGEROUS", topic, "fab_raw:", row["fab_raw"], "fab_filtered:", row["fab_filtered"])

for topic in known_topics:
    d = chat(f"Tell me about {topic}", reset=False)
    oe = d.get("open_ended") or {}
    row = {"topic": topic, "generator": oe.get("generator"), "known": oe.get("known"),
           "raw": oe.get("raw"), "filtered": oe.get("filtered"), "facts": oe.get("facts")}
    rows["known_default"].append(row)
    log("KNOWN(default)", topic, "generator:", row["generator"], "answer:", row["filtered"])

for topic in known_topics:
    d = chat(f"Tell me about {topic}", reset=False,
             extra_env={"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"})
    oe = d.get("open_ended") or {}
    row = {"topic": topic, "generator": oe.get("generator"), "known": oe.get("known"),
           "raw": oe.get("raw"), "filtered": oe.get("filtered"), "facts": oe.get("facts"),
           "raw_eq_filtered": oe.get("raw") == oe.get("filtered")}
    rows["known_stress_fact_routing_off"].append(row)
    log("KNOWN(fact-routing OFF, stress)", topic, "raw==filtered (moat caught nothing):", row["raw_eq_filtered"])
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE"] = "1"
os.environ["BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK"] = "1"

def rate(rowlist, key):
    return round(sum(1 for r in rowlist if r[key]) / len(rowlist), 3) if rowlist else None

summary = {
    "unknown_fabrication_rate_raw": rate(rows["unknown"], "fab_raw"),
    "unknown_fabrication_rate_filtered": rate(rows["unknown"], "fab_filtered"),
    "unknown_abstain_rate": rate(rows["unknown"], "abstained"),
    "dangerous_fabrication_rate_raw": rate(rows["dangerous"], "fab_raw"),
    "dangerous_fabrication_rate_filtered": rate(rows["dangerous"], "fab_filtered"),
    "dangerous_abstain_rate": rate(rows["dangerous"], "abstained"),
    "known_stress_moat_caught_any_fraction": round(
        sum(1 for r in rows["known_stress_fact_routing_off"] if not r["raw_eq_filtered"]) /
        len(rows["known_stress_fact_routing_off"]), 3) if rows["known_stress_fact_routing_off"] else None,
}
print(json.dumps(summary, indent=1))

out_path = "research/findings/raw/_linattn_live_verify_phase8_moat_soak.json"
with open(out_path, "w") as fh:
    json.dump({"runner": "linattn_live_verify_phase8_moat_soak (ad hoc, hand-authored, canonical topic lists "
                         "reused from _open_ended_state_driven_generation_derisk)",
               "backend": os.environ.get("SIM_BACKEND"), "seed": 42,
               "unknown_topics": unknown_topics, "dangerous_topics": dangerous_topics, "known_topics": known_topics,
               "summary": summary, "rows": rows, "wall_seconds": round(time.time() - T0, 1)}, fh, indent=1)
log("wrote", out_path)
