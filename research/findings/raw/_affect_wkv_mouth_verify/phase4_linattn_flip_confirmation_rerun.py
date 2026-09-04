"""Phase 4 RERUN (2026-09-04) -- the SAME two flip-gate questions as `phase4_linattn_flip_confirmation.py`
(byte-identical scenario: same priming message, same known/unknown topics, same env config), re-run against the
sharpness-aware / saturating / habituating `_apply_affect_bias` coupling fix in `webapp/wkv_mouth_generator.py`
(margin-to-top1 concentrated assist + recent-affect-word habituation, `affect_boost` default raised 5.0->10.0 --
see that function's own mechanism comment for the full diagnosis + design). Writes a SEPARATE `..._AFTER.json`
so the original FAIL verdict (`..._BEFORE.json`, preserved verbatim alongside this file) stays on record.

Adds ONE thing phase4 did not check: a repeated LESION=0 raw-text FLUENCY spot-check (not just byte-diff/
determinism) -- salad-collapse is a qualitative failure mode a byte-diff test cannot see (byte-different is
necessary but not sufficient; "byte-different because it degenerated into repeated affect words" would still
pass phase4's own Q1 while failing the task's fluency requirement), so this script also reports the raw text
for manual/automated fluency inspection and a same-token-repetition-fraction heuristic.

Forced to CPU (CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy set by the caller) -- the GPU was busy with an
unrelated scale probe + the mouth's torch OOMs against it when sharing the card (see the task brief this rung
worked from); this ENTIRE mechanism is checkpoint-numpy + Izhikevich-CPU/NumPy-backend, so CPU-forcing changes
nothing about which code path is exercised, only which device runs it.
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ["BRAIN_OPEN_ENDED"] = "1"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
os.environ["BRAIN_WKV_MOUTH_CKPT"] = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
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
SESSION = "affect_wkv_mouth_verify_linattn_rerun"
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


def salad_frac(text):
    toks = (text or "").split()
    if not toks:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)


rows = {}

# ---- Q1: affect load-bearing (linattn, full-live) ----
log("=== Q1 turn0: prime mood ===")
d0 = chat(PRIMING, reset=True, extra_env=FORCE_FREEGEN)
log("affect0:", d0.get("affect"))
rows["priming_turn"] = {"msg": PRIMING, "affect": d0.get("affect")}   # persisted (was log-only before this rung)

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
sfrac_l0 = salad_frac(oe1.get("raw"))
log("Q1 affect load-bearing (raw differs lesion0 vs lesion1):", raw_diff_lesion)
log("Q1 determinism (raw same lesion0 vs lesion0-repeat):", raw_repro_lesion0)
log("Q1 lesion0 salad_frac (fluency heuristic):", sfrac_l0)

rows["Q1_affect_loadbearing"] = {
    "topic": KNOWN_TOPIC,
    "wkv_used": {"l0": oe1.get("wkv_mouth_used"), "l1": oe2.get("wkv_mouth_used")},
    "generator": {"l0": oe1.get("generator"), "l1": oe2.get("generator")},
    "raw": {"l0": oe1.get("raw"), "l1": oe2.get("raw"), "l0_repeat": oe3.get("raw")},
    "raw_differs_lesion0_vs_lesion1": raw_diff_lesion,
    "raw_reproduces_lesion0_vs_lesion0_repeat": raw_repro_lesion0,
    "lesion0_salad_frac": sfrac_l0,
    "lesion0_fluent_not_salad": bool(sfrac_l0 < 0.3),   # heuristic threshold -- see the finding for manual read
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
    "Q1_lesion0_fluent_not_salad": rows["Q1_affect_loadbearing"]["lesion0_fluent_not_salad"],
    "Q2_moat_with_affect_PASS": rows["Q2_moat_with_affect"]["PASS"],
    "FLIP_CONFIRM_GO": bool(rows["Q1_affect_loadbearing"]["PASS"]
                             and rows["Q1_affect_loadbearing"]["lesion0_fluent_not_salad"]
                             and rows["Q2_moat_with_affect"]["PASS"]),
}
log("VERDICT:", verdict)

out_path = "research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_AFTER.json"
with open(out_path, "w") as fh:
    json.dump({"runner": "affect_wkv_mouth_verify_phase4_rerun (linattn flip confirmation AFTER the "
                         "sharpness-aware/saturating/habituating affect-bias fix, hand-authored)",
               "seed": 42, "backend": os.environ.get("SIM_BACKEND"),
               "recurrence": "linattn (flip target)", "verdict": verdict, "rows": rows,
               "wall_seconds": round(time.time() - T0, 1)}, fh, indent=1)
log("wrote", out_path)
