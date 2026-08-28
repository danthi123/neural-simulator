"""Timing probe: how long does ONE real /api/brain-chat rich turn cost on numpy, with LTM attach disabled
(BRAIN_LTM_SHIP_DEFAULT=off -- honest for this specific retest since the tested facts/mechanism are buffer-tier
only, see the main script's docstring), so the main verification script can be sized correctly instead of
guessing a timeout blind again."""
import os, time
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE",
           "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE",
           "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM",
           "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES",
           "BRAIN_DA_DRIVES", "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM",
           "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN", "BRAIN_BG_SELECT", "BRAIN_SILENT_WM",
           "BRAIN_SPIKING_MOUTH_RECALL", "BRAIN_CONFIDENCE_FORTHCOMING"):
    os.environ[_k] = "0"

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

log("importing webapp.server ...")
import webapp.server as S
log("import done")

log("turn 1 (fresh session, first-ever build) starting ...")
resp = S.brain_chat(S.BrainChatRequest(session="timing_probe_1", message="what does the brain use",
                                       brain="tiny-demo", reset=True, rich=True))
log(f"turn 1 done: {resp.body[:200]}")

log("turn 2 (SAME process, DIFFERENT fresh session) starting ...")
resp2 = S.brain_chat(S.BrainChatRequest(session="timing_probe_2", message="what does the dog chase",
                                        brain="tiny-demo", reset=True, rich=True))
log(f"turn 2 done: {resp2.body[:200]}")

log("turn 3 (SAME process, THIRD fresh session) starting ...")
resp3 = S.brain_chat(S.BrainChatRequest(session="timing_probe_3", message="what does the brain use",
                                        brain="tiny-demo", reset=True, rich=True))
log(f"turn 3 done: {resp3.body[:200]}")
log("ALL DONE")
