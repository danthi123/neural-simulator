"""Phase 3 -- REAL live-pipeline `BRAIN_AFFECT_LESION` on/off test, through `webapp.server.brain_chat`
in-process (the exact function `/api/brain-chat` dispatches to), real onebrain composer, real spiking affect
organ, SHIPPED-default recurrence (`BRAIN_WKV_MOUTH_RECURRENCE` unset -> 'ssm'). Mirrors turns 5-8 of the
ORIGINAL 2026-09-03 linattn live-verification's `phase7_live_pipeline_fluent_grounded_lesion.py` methodology
EXACTLY (same `chat()` helper shape, same env-override pattern, same `BRAIN_AFFECT_LESION` toggle) -- with ONE
deliberate addition: fact-routing (`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE` / `BRAIN_OPEN_ENDED_FACT_CLAUSE_
FALLBACK`) is forced OFF on the topic-query turns, so the reply is genuinely written by `_free_gen` (the path
this fix touches) rather than the fact-clause template (which is INTENTIONALLY affect-neutral by design and
would show `raw_identical: true` regardless of this fix -- testing it would silently test the wrong code path).

Sequence: (1) a sentiment-laden priming message (establishes a real appraised mood via `_update_session_mood`'s
cross-turn EMA, exactly the mechanism `webapp/server.py`'s live Gate-B block already uses); (2) the SAME
known-topic query asked twice in the SAME session, `BRAIN_AFFECT_LESION=0` then `=1`, fact-routing forced off
both times, so `oe.raw` is genuinely the mouth's own free generation under the SAME session mood, differing
only in whether the organ's neural differential (hence the mapped valence) is lesioned.
"""
import json
import os
import sys
import time

sys.path.insert(0, ".")
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ["BRAIN_OPEN_ENDED"] = "1"
os.environ["BRAIN_OPEN_ENDED_WKV_MOUTH"] = "1"
os.environ["BRAIN_WKV_MOUTH_SCOPE"] = "broad"      # test-only override -- see module docstring
os.environ["BRAIN_OPEN_ENDED_NP_ENTAILMENT"] = "0"
os.environ["BRAIN_OPEN_ENDED_GEN_TIME_HONESTY"] = "0"
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"
# BRAIN_WKV_MOUTH_RECURRENCE left UNSET -> the shipped default ('ssm') -- this is the production-flip-relevant arm.

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


import webapp.server as S  # noqa: E402  (deliberately late: env-fixed constants read at import)

RENDERER = "stub"
SESSION = "affect_wkv_mouth_verify_ssm"


def chat(msg, reset, extra_env=None):
    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
    resp = S.brain_chat(S.BrainChatRequest(session=SESSION, message=msg, brain="tiny-demo",
                                            reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))


rows = {}
TOPIC_MSG = "Tell me about frank_lincoln_wright"
FORCE_FREEGEN = {"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"}

log("=== turn 0: build session (sentiment-laden priming message, establishes a real appraised mood) ===")
d0 = chat("I am absolutely thrilled and overjoyed today, everything is wonderful!", reset=True,
          extra_env=FORCE_FREEGEN)
log("affect0:", d0.get("affect"))
rows["priming_turn"] = {"msg": "I am absolutely thrilled and overjoyed today, everything is wonderful!",
                         "affect": d0.get("affect")}

log("=== turn 1: known topic, fact-routing OFF (forces genuine free-gen), BRAIN_AFFECT_LESION=0 ===")
d1 = chat(TOPIC_MSG, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "0"})
oe1 = d1.get("open_ended") or {}
log("generator:", oe1.get("generator"), "wkv_used:", oe1.get("wkv_mouth_used"), "known:", oe1.get("known"))
log("affect1 (lesion=0):", d1.get("affect"))
log("raw1:", oe1.get("raw"))

log("=== turn 2: SAME topic + SAME fact-routing-off config, BRAIN_AFFECT_LESION=1 ===")
d2 = chat(TOPIC_MSG, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "1"})
oe2 = d2.get("open_ended") or {}
log("generator:", oe2.get("generator"), "wkv_used:", oe2.get("wkv_mouth_used"), "known:", oe2.get("known"))
log("affect2 (lesion=1):", d2.get("affect"))
log("raw2:", oe2.get("raw"))

log("=== turn 3: lesion back OFF, SAME topic -- a second lesion=0 reading (repeatability sanity) ===")
d3 = chat(TOPIC_MSG, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "0"})
oe3 = d3.get("open_ended") or {}
log("affect3 (lesion=0 again):", d3.get("affect"))
log("raw3:", oe3.get("raw"))

os.environ["BRAIN_AFFECT_LESION"] = "0"

raw_identical_1_2 = oe1.get("raw") == oe2.get("raw")
raw_identical_1_3 = oe1.get("raw") == oe3.get("raw")
log("raw1 == raw2 (lesion0 vs lesion1):", raw_identical_1_2)
log("raw1 == raw3 (lesion0 vs lesion0 repeat):", raw_identical_1_3)

rows["lesion_test"] = {
    "topic_msg": TOPIC_MSG,
    "generator": {"lesion0": oe1.get("generator"), "lesion1": oe2.get("generator"), "lesion0_repeat": oe3.get("generator")},
    "wkv_used": {"lesion0": oe1.get("wkv_mouth_used"), "lesion1": oe2.get("wkv_mouth_used"), "lesion0_repeat": oe3.get("wkv_mouth_used")},
    "affect": {"lesion0": d1.get("affect"), "lesion1": d2.get("affect"), "lesion0_repeat": d3.get("affect")},
    "raw": {"lesion0": oe1.get("raw"), "lesion1": oe2.get("raw"), "lesion0_repeat": oe3.get("raw")},
    "raw_identical_lesion0_vs_lesion1": raw_identical_1_2,
    "raw_identical_lesion0_vs_lesion0_repeat": raw_identical_1_3,
}

out_path = "research/findings/raw/_affect_wkv_mouth_verify_phase3_live_pipeline_lesion_ssm.json"
with open(out_path, "w") as fh:
    json.dump({"runner": "affect_wkv_mouth_verify_phase3 (ad hoc, hand-authored)", "seed": 42,
               "backend": os.environ.get("SIM_BACKEND"), "recurrence": "ssm (shipped default, unset)",
               "rows": rows, "wall_seconds": round(time.time() - T0, 1)}, fh, indent=1)
log("wrote", out_path)
