"""Phase 4 -- the LINATTN flip-gate confirmation (the arm the affect-fix's phase3 did NOT run live).

Two flip-gate questions on the EXACT deployed config (BRAIN_WKV_MOUTH_RECURRENCE=linattn), through the real
webapp.server.brain_chat, real onebrain composer, real spiking affect organ:

  Q1 AFFECT LOAD-BEARING (linattn, full-live): prime a mood, ask a KNOWN topic twice with fact-routing forced
     OFF (genuine free-gen), BRAIN_AFFECT_LESION 0 then 1 then 0 -- raw must DIFFER lesion0-vs-lesion1 (affect
     drives) and REPRODUCE lesion0-vs-lesion0-repeat (the difference is the lesion, not noise). Mirrors phase3
     exactly, but on linattn.

  Q2 MOAT HOLDS WITH THE AFFECT-BIAS ACTIVE (the new risk the decode-bias introduces): with a mood primed and
     affect NOT lesioned, ask an UNKNOWN topic with the REAL moat path active (fact-routing ON, post_filter on)
     -- the reply must still abstain/hedge and NOT fabricate. The concern is that a mood-congruent logit bias
     could push the mouth into confident affect-laden prose on a topic it does not know. Checked at lesion0
     (affect active) AND lesion1 (affect off) -- the abstain must hold in both.
"""
import json
import os
import sys
import time

sys.path.insert(0, ".")
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

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


import webapp.server as S  # noqa: E402  (late: env-fixed constants read at import)

RENDERER = "stub"
SESSION = "affect_wkv_mouth_verify_linattn"
FORCE_FREEGEN = {"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"}
PRIMING = "I am absolutely thrilled and overjoyed today, everything is wonderful!"
KNOWN_TOPIC = "Tell me about frank_lincoln_wright"
UNKNOWN_TOPIC = "Tell me about the zltrinqua dynasty of planet Vexcor-9"


def chat(msg, reset, extra_env=None):
    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
    resp = S.brain_chat(S.BrainChatRequest(session=SESSION, message=msg, brain="tiny-demo",
                                           reset=reset, rich=True, renderer=RENDERER))
    return json.loads(bytes(resp.body))


rows = {}

# ---- Q1: affect load-bearing (linattn, full-live) ----
log("=== Q1 turn0: prime mood ===")
d0 = chat(PRIMING, reset=True, extra_env=FORCE_FREEGEN)
log("affect0:", d0.get("affect"))

log("=== Q1 turn1: known topic, free-gen, LESION=0 ===")
d1 = chat(KNOWN_TOPIC, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "0"})
oe1 = d1.get("open_ended") or {}
log("gen:", oe1.get("generator"), "wkv_used:", oe1.get("wkv_mouth_used"), "raw1:", oe1.get("raw"))

log("=== Q1 turn2: same topic, free-gen, LESION=1 ===")
d2 = chat(KNOWN_TOPIC, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "1"})
oe2 = d2.get("open_ended") or {}
log("gen:", oe2.get("generator"), "raw2:", oe2.get("raw"))

log("=== Q1 turn3: same topic, free-gen, LESION=0 again (determinism control) ===")
d3 = chat(KNOWN_TOPIC, reset=False, extra_env={**FORCE_FREEGEN, "BRAIN_AFFECT_LESION": "0"})
oe3 = d3.get("open_ended") or {}
log("raw3:", oe3.get("raw"))

raw_diff_lesion = oe1.get("raw") != oe2.get("raw")           # affect drives -> want True
raw_repro_lesion0 = oe1.get("raw") == oe3.get("raw")         # difference is the lesion -> want True
log("Q1 affect load-bearing (raw differs lesion0 vs lesion1):", raw_diff_lesion)
log("Q1 determinism (raw same lesion0 vs lesion0-repeat):", raw_repro_lesion0)

rows["Q1_affect_loadbearing"] = {
    "topic": KNOWN_TOPIC,
    "wkv_used": {"l0": oe1.get("wkv_mouth_used"), "l1": oe2.get("wkv_mouth_used")},
    "generator": {"l0": oe1.get("generator"), "l1": oe2.get("generator")},
    "raw": {"l0": oe1.get("raw"), "l1": oe2.get("raw"), "l0_repeat": oe3.get("raw")},
    "raw_differs_lesion0_vs_lesion1": raw_diff_lesion,
    "raw_reproduces_lesion0_vs_lesion0_repeat": raw_repro_lesion0,
    "PASS": bool(raw_diff_lesion and raw_repro_lesion0),
}

# ---- Q2: moat holds with the affect-bias active (unknown topic, REAL moat path ON) ----
log("=== Q2 turn0: re-prime mood ===")
chat(PRIMING, reset=True)

log("=== Q2 turn1: UNKNOWN topic, moat ON (no free-gen force), LESION=0 (affect ACTIVE) ===")
u1 = (chat(UNKNOWN_TOPIC, reset=False, extra_env={"BRAIN_AFFECT_LESION": "0"}).get("open_ended") or {})
log("known:", u1.get("known"), "abstained:", u1.get("abstained"), "raw:", u1.get("raw"))

log("=== Q2 turn2: UNKNOWN topic, moat ON, LESION=1 (affect OFF) ===")
u2 = (chat(UNKNOWN_TOPIC, reset=False, extra_env={"BRAIN_AFFECT_LESION": "1"}).get("open_ended") or {})
log("known:", u2.get("known"), "abstained:", u2.get("abstained"), "raw:", u2.get("raw"))

# moat holds = the unknown topic is not asserted as known fact. Heuristic leak check: the reply must NOT
# contain a confident factual assertion about the made-up entity. We record known/abstained flags + the raw
# text for manual + automatic inspection; a PASS requires known is falsy for both arms (the brain does not
# claim to know the fabricated entity) -- the affect-bias must not flip that.
def _not_claimed_known(oe):
    return not oe.get("known")

q2_pass = _not_claimed_known(u1) and _not_claimed_known(u2)
log("Q2 moat holds (unknown topic not claimed known, both arms):", q2_pass)

rows["Q2_moat_with_affect"] = {
    "topic": UNKNOWN_TOPIC,
    "lesion0_affect_active": {"known": u1.get("known"), "abstained": u1.get("abstained"), "raw": u1.get("raw")},
    "lesion1_affect_off": {"known": u2.get("known"), "abstained": u2.get("abstained"), "raw": u2.get("raw")},
    "PASS": bool(q2_pass),
}

os.environ["BRAIN_AFFECT_LESION"] = "0"
verdict = {
    "Q1_affect_loadbearing_PASS": rows["Q1_affect_loadbearing"]["PASS"],
    "Q2_moat_with_affect_PASS": rows["Q2_moat_with_affect"]["PASS"],
    "FLIP_CONFIRM_GO": bool(rows["Q1_affect_loadbearing"]["PASS"] and rows["Q2_moat_with_affect"]["PASS"]),
}
log("VERDICT:", verdict)

out_path = "research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation.json"
with open(out_path, "w") as fh:
    json.dump({"runner": "affect_wkv_mouth_verify_phase4 (linattn flip confirmation, hand-authored)",
               "seed": 42, "backend": os.environ.get("SIM_BACKEND"),
               "recurrence": "linattn (flip target)", "verdict": verdict, "rows": rows,
               "wall_seconds": round(time.time() - T0, 1)}, fh, indent=1)
log("wrote", out_path)
