"""BOUNDED confidence-read discrimination test (controller-owned, replacing the killed agent's runaway sweep).
Question: does mean_role_confidence DROP below the metacog HIGH band (ROLE_CONF_HI) as the decode is degraded,
while the turn still answers? If yes -> the read discriminates given ambiguity; the tiny-demo saturation is purely
its unambiguous content. HARD-CAPPED: 5 sigmas, 1 seed each. No sweep, no custom-LTM (avoids the mrc=null bug)."""
import os, sys, json, time
REPO = "/home/dant123/Projects/sim"
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC",
           "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP",
           "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY",
           "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES",
           "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM", "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN",
           "BRAIN_BG_SELECT", "BRAIN_SILENT_WM", "BRAIN_SPIKING_MOUTH_RECALL"):
    os.environ[_k] = "0"
os.environ.pop("BRAIN_METACOG", None)  # metacog default-ON
import numpy as np
import webapp.server as S
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import mean_role_confidence, ROLE_CONF_LO, ROLE_CONF_HI
print(f"ROLE_CONF_LO={ROLE_CONF_LO} ROLE_CONF_HI={ROLE_CONF_HI}", flush=True)
chat, source = S._build_chat_brain("tiny-demo", "stub")
comp = getattr(getattr(chat, "inner", None), "composer", None)
assert comp is not None, "no composer"
base_conns = list(comp.buffer.store_conns)
Q = "what does the brain use"
_sid = [0]
def ask(noised=None):
    comp.buffer.store_conns = noised if noised is not None else list(base_conns)
    _sid[0] += 1
    ck = (f"cd{_sid[0]:03d}", "tiny-demo", "stub")
    S._BRAIN_CHATS[ck] = chat
    try:
        r = S.brain_chat(S.BrainChatRequest(session=f"cd{_sid[0]:03d}", message=Q, brain="tiny-demo",
                                            reset=False, rich=True, renderer="stub"))
        return json.loads(bytes(r.body))
    finally:
        comp.buffer.store_conns = list(base_conns)
d0 = ask(None)
mrc0 = mean_role_confidence(d0.get("activity"))
print(f"CLEAN  mrc={mrc0} n_sentences={d0.get('n_sentences')} abstained={d0.get('abstained')} "
      f"answer={(d0.get('answer') or '')[:60]!r}", flush=True)
rows = [("clean", mrc0, d0.get("n_sentences"), d0.get("abstained"))]
for sigma in [0.3, 0.6, 0.9, 1.2, 1.5]:
    dn = ask(_noise(base_conns, sigma, np.random.default_rng(42)))
    mrc = mean_role_confidence(dn.get("activity"))
    print(f"sigma={sigma:4} mrc={mrc} n_sentences={dn.get('n_sentences')} abstained={dn.get('abstained')} "
          f"answer={(dn.get('answer') or '')[:50]!r}", flush=True)
    rows.append((f"sigma{sigma}", mrc, dn.get("n_sentences"), dn.get("abstained")))
# verdict: is there a turn with mrc BELOW ROLE_CONF_HI that STILL answered (not abstain, mrc not None)?
low_answering = [r for r in rows if r[1] is not None and r[1] < ROLE_CONF_HI and not r[3]]
print("\n=== VERDICT ===")
print(f"clean mrc={mrc0} (HIGH band={ROLE_CONF_HI})")
print(f"low-but-answering turns found: {len(low_answering)} -> {low_answering}")
print("DISCRIMINATES:" , bool(low_answering) and mrc0 is not None and mrc0 >= ROLE_CONF_HI)
