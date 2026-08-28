"""Find a synaptic-noise sigma that, through the FULL real /api/brain-chat rich-answer-composer + VERIFY
pipeline (not the raw composer.query_patient probe the recalib arc used), still ANSWERS the same fact (no
misrecall, no abstain) but reads a mean_role_confidence clearly below ROLE_CONF_LO=0.30. sigma=2.2 (matching
the raw-composer sweep's "clearly degraded" pick) turned out to abstain OUTRIGHT through this stricter,
VERIFY-gated pipeline -- this sweep finds a sigma where the rich path still answers.

Tries DESCENDING sigma (most-degraded first) so the search favors maximal genuine separation from the
confident band while still producing an answer; stops at the first success. Each trial is a FRESH session
(avoids discourse-thread "already said" contamination across trials).
"""
import os, json, time
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
           "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
os.environ.pop("BRAIN_METACOG", None)
os.environ.pop("BRAIN_METACOG_LESION", None)
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", None)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import numpy as np
import webapp.server as S
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import mean_role_confidence, ROLE_CONF_LO

Q = "what does the brain use"
RENDERER = "stub"


def _prebuild(session):
    ck = (session, "tiny-demo", RENDERER)
    chat, source = S._build_chat_brain("tiny-demo", RENDERER)
    S._BRAIN_CHATS[ck] = chat
    return chat


def _composer_of(session):
    ck = (session, "tiny-demo", RENDERER)
    chat = S._BRAIN_CHATS.get(ck)
    return chat, getattr(getattr(chat, "inner", None), "composer", None)


results = []
for sigma in (1.8, 1.3, 0.9):
    session = f"sigmasweep_{str(sigma).replace('.', 'p')}"
    _prebuild(session)
    _, comp = _composer_of(session)
    base = list(comp.store_conns)
    comp.store_conns = _noise(base, sigma, np.random.default_rng(9000 + int(sigma * 10)))
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=Q, brain="tiny-demo",
                                           reset=False, rich=True, renderer=RENDERER))
    d = json.loads(bytes(resp.body))
    mrc = mean_role_confidence(d.get("activity"))
    same_fact = d.get("recalled_svo") == ["brain", "use", "spikes"]
    log(f"sigma={sigma}: abstained={d.get('abstained')} answer={d.get('answer')!r} recalled_svo={d.get('recalled_svo')} "
        f"mrc={mrc} metacog_confident={(d.get('metacog') or {}).get('confident')}")
    results.append({"sigma": sigma, "abstained": d.get("abstained"), "answer": d.get("answer"),
                     "recalled_svo": d.get("recalled_svo"), "mrc": mrc, "same_fact": same_fact,
                     "metacog": d.get("metacog"), "below_lo": bool(mrc is not None and mrc < ROLE_CONF_LO)})
    if (not d.get("abstained")) and same_fact and mrc is not None and mrc < ROLE_CONF_LO:
        log(f"FOUND: sigma={sigma} answers + same fact + mrc={mrc} < LO={ROLE_CONF_LO} -- stopping sweep")
        break

with open("research/findings/raw/_confidence_forthcoming_retest/_sigma_sweep.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
log("wrote sweep results")
