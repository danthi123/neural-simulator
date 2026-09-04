"""Phase 6 -- CLEAN affect-load-bearing isolation for the linattn flip gate.

The original phase4 ran turn1(lesion0)/turn2(lesion1)/turn3(lesion0) in ONE session, so the mood EMA
(_update_session_mood) AND the affect-fix's habituation state evolve ACROSS turns -- which makes lesion0 vs
lesion0-repeat legitimately DIFFER (session-dynamics, not noise) and CONFOUNDS the lesion0-vs-lesion1 attribution.

This test isolates the lesion: each arm uses a FRESH session (unique id + reset=True on the prime), so the mood
+ habituation state at the topic turn are IDENTICAL across arms. Then:
  - affect load-bearing: raw(fresh, lesion0) != raw(fresh, lesion1)   [only the lesion differs]
  - determinism:         raw(fresh, lesion0) == raw(fresh, lesion0-repeat)   [fresh session -> reproducible]
Merged-main config (linattn flip target), CPU-forced, real webapp.server.brain_chat.
"""
import json, os, sys, time
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
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
import webapp.server as S  # noqa: E402
FORCE_FREEGEN = {"BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0", "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0"}
PRIMING = "I am absolutely thrilled and overjoyed today, everything is wonderful!"
TOPIC = "Tell me about frank_lincoln_wright"

def one_arm(session, lesion):
    """Fresh session: prime the mood, then ask the topic under the given lesion. Return the topic reply raw."""
    for k, v in FORCE_FREEGEN.items():
        os.environ[k] = v
    os.environ["BRAIN_AFFECT_LESION"] = lesion
    S.brain_chat(S.BrainChatRequest(session=session, message=PRIMING, brain="tiny-demo", reset=True, rich=True, renderer="stub"))
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=TOPIC, brain="tiny-demo", reset=False, rich=True, renderer="stub"))
    oe = (json.loads(bytes(resp.body)).get("open_ended") or {})
    return oe.get("raw"), oe.get("generator"), oe.get("wkv_mouth_used")

log("=== arm A: fresh session, lesion=0 (affect ACTIVE) ===")
rawA, genA, usedA = one_arm("phase6_l0_A", "0"); log("genA:", genA, "usedA:", usedA, "rawA:", repr(rawA)[:180])
log("=== arm B: fresh session, lesion=1 (affect OFF) ===")
rawB, genB, usedB = one_arm("phase6_l1_B", "1"); log("genB:", genB, "rawB:", repr(rawB)[:180])
log("=== arm C: fresh session, lesion=0 again (determinism, fresh) ===")
rawC, genC, usedC = one_arm("phase6_l0_C", "0"); log("rawC:", repr(rawC)[:180])

os.environ["BRAIN_AFFECT_LESION"] = "0"
affect_loadbearing = (rawA != rawB)          # only the lesion differs across fresh sessions
determinism = (rawA == rawC)                 # fresh session -> reproducible
verdict = {
    "affect_loadbearing_rawA_ne_rawB": affect_loadbearing,
    "determinism_rawA_eq_rawC": determinism,
    "wkv_mouth_used": {"A": usedA, "B": usedB, "C": usedC},
    "CLEAN_FLIP_GO": bool(affect_loadbearing and determinism),
}
log("VERDICT:", verdict)
out = "research/findings/raw/_affect_wkv_mouth_verify_phase6_clean_isolation.json"
with open(out, "w") as fh:
    json.dump({"runner": "phase6_clean_isolation (fresh-session-per-arm)", "backend": os.environ.get("SIM_BACKEND"),
               "recurrence": "linattn", "verdict": verdict,
               "raw": {"A_l0": rawA, "B_l1": rawB, "C_l0rep": rawC}, "wall_seconds": round(time.time()-T0, 1)}, fh, indent=1)
log("wrote", out)
