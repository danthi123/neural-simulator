"""PREP RUNG 2 verification: the DA-encoding substrate-homeostasis CONSOLIDATION TRIGGER wired into the idle tick.

`webapp/continuous_engine.consolidate_substrate_homeostasis(cache_key, chat)` runs the Turrigiano synaptic-scaling pass
(`da_encoding_drives_chat.apply_substrate_homeostasis` -> `OneBrainComposer.apply_homeostatic_scaling`) on the idle
tick, alongside the D5 learn-through-use pass. This runner proves the between-turn cadence is load-bearing AND safe
against compounding:
  (A) FIRES on store-growth: after teaching DA-gated facts, the pass runs -> a REAL change to the composer's store
      synapses (store_conns mean |w| moves), returning a record with n_engrams == the stored count.
  (B) NO-OPS with no new writes: a second call with the store unchanged returns None (the new-writes-since-last-pass
      trigger prevents re-scaling, which would compound strong engrams toward unit and erase the DA-salience order).
  (C) RE-FIRES after a new fact is taught (the store grew again).
  (D) LESION disarms it (BRAIN_DA_ENCODING_LESION=1 -> None); (D2) =0 disarms it (byte-identical off).

Run (numpy-CPU, foreground, ~a minute): SIM_BACKEND=numpy python -u -m research.runners._da_encoding_homeo_trigger
"""
from __future__ import annotations
import json, logging, os, sys, types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger().setLevel(logging.ERROR)
import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from webapp import continuous_engine as CE

FACTS = [("dog", "eat", "grass"), ("cat", "see", "fish"), ("bird", "chase", "worm"), ("wolf", "hunt", "deer")]
GAINS = [1.0, 2.4, 0.6, 1.8]   # a spread DA-salience distribution (some strong, some weak) for the scaler to regulate
VOCAB = sorted({t for f in FACTS for t in f})
CK = ("homeo_test", "tiny-demo", "stub")


def _mean_mag(comp):
    ws = [abs(w) for (_p, _q, w) in comp.store_conns]
    return float(np.mean(ws)) if ws else 0.0


def _fake_chat(comp):
    return types.SimpleNamespace(inner=types.SimpleNamespace(composer=comp))


def _build():
    holder = {"g": 1.0}
    c = OneBrainComposer(seed=42, D=48, vocab=VOCAB, k_max=16, enable_batched=False,
                         enable_rf_cudagraph=False, enable_csr_cache=False, enable_spiking_cleanup=False,
                         encoding_gain_fn=lambda: holder["g"], homeostatic_scaling=True)
    return c, holder


def main():
    for k in ("BRAIN_DA_ENCODING", "BRAIN_DA_ENCODING_SUBSTRATE", "BRAIN_DA_ENCODING_LESION"):
        os.environ.pop(k, None)                                   # faculty + substrate homeostat DEFAULT-ON, no lesion
    CE._LAST_HOMEO_KB.pop(CK, None)

    comp, holder = _build()
    for (a, act, p), g in zip(FACTS[:3], GAINS[:3]):
        holder["g"] = float(g); comp.store(a, act, p)
    chat = _fake_chat(comp)

    mag0 = _mean_mag(comp)
    recA = CE.consolidate_substrate_homeostasis(CK, chat)          # (A) FIRES
    magA = _mean_mag(comp)
    pA = bool(recA is not None and recA.get("n_engrams") == 3 and abs(magA - mag0) > 1e-9)

    recB = CE.consolidate_substrate_homeostasis(CK, chat)          # (B) NO-OP (no growth)
    magB = _mean_mag(comp)
    pB = bool(recB is None and abs(magB - magA) < 1e-12)

    holder["g"] = float(GAINS[3]); comp.store(*FACTS[3])           # teach a 4th fact -> store grew
    recC = CE.consolidate_substrate_homeostasis(CK, chat)         # (C) RE-FIRES
    pC = bool(recC is not None and recC.get("n_engrams") == 4)

    os.environ["BRAIN_DA_ENCODING_LESION"] = "1"                   # (D) LESION disarms
    comp2, holder2 = _build()
    for (a, act, p), g in zip(FACTS[:2], GAINS[:2]):
        holder2["g"] = float(g); comp2.store(a, act, p)
    CE._LAST_HOMEO_KB.pop(("les_test", "tiny-demo", "stub"), None)
    recL = CE.consolidate_substrate_homeostasis(("les_test", "tiny-demo", "stub"), _fake_chat(comp2))
    pL = bool(recL is None)
    os.environ.pop("BRAIN_DA_ENCODING_LESION", None)

    os.environ["BRAIN_DA_ENCODING"] = "0"                         # (D2) =0 disarms
    comp3, holder3 = _build()
    for (a, act, p), g in zip(FACTS[:2], GAINS[:2]):
        holder3["g"] = float(g); comp3.store(a, act, p)
    CE._LAST_HOMEO_KB.pop(("off_test", "tiny-demo", "stub"), None)
    recOff = CE.consolidate_substrate_homeostasis(("off_test", "tiny-demo", "stub"), _fake_chat(comp3))
    pOff = bool(recOff is None)
    os.environ.pop("BRAIN_DA_ENCODING", None)

    go = bool(pA and pB and pC and pL and pOff)
    # EARN the verdict — the preconditions travel with the result (tools.verdict.Verdict; verdict-preconditions gate).
    from tools.verdict import Verdict
    v = Verdict("da-encoding substrate-homeostasis consolidation trigger (idle-tick cadence)")
    v.require("(A) FIRES on store-growth: a real store-synapse change, n_engrams==3", pA, expect=True,
              note=f"mean |w| {mag0} -> {magA}")
    v.require("(B) NO-OP with no new writes (compounding guard)", pB, expect=True)
    v.require("(C) RE-FIRES after a new fact is taught: n_engrams==4", pC, expect=True)
    v.require("(D) LESION (BRAIN_DA_ENCODING_LESION=1) disarms the pass", pL, expect=True)
    v.require("(D2) =0 (BRAIN_DA_ENCODING=0) disarms the pass (byte-identical off)", pOff, expect=True)
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])
    out = {
        "runner": "research/runners/_da_encoding_homeo_trigger.py",
        "A_fires_on_growth": {"rec": recA, "mag_before": mag0, "mag_after": magA, "PASS": pA},
        "B_noop_no_growth": {"rec": recB, "PASS": pB},
        "C_refires_after_new_fact": {"rec": recC, "PASS": pC},
        "D_lesion_disarms": {"rec": recL, "PASS": pL},
        "D2_off_disarms": {"rec": recOff, "PASS": pOff},
        "GO": go,
        "status": decided["status"],
        "preconditions": decided["preconditions"],
    }
    op = os.path.join(_REPO, "research", "findings", "raw", "_da_encoding_wired", "homeo_trigger.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=2, default=str)
    bar = "=" * 90
    print("\n" + bar)
    print("  DA-ENCODING SUBSTRATE-HOMEOSTASIS CONSOLIDATION TRIGGER (PREP RUNG 2)")
    print(bar)
    print(f"  (A) FIRES on growth:        |w| {mag0:.5f} -> {magA:.5f}  n_engrams={recA and recA.get('n_engrams')}  -> {pA}")
    print(f"  (B) NO-OP no new writes:    rec={recB}  -> {pB}")
    print(f"  (C) RE-FIRES after teach:   n_engrams={recC and recC.get('n_engrams')}  -> {pC}")
    print(f"  (D) LESION disarms:         rec={recL}  -> {pL}")
    print(f"  (D2) =0 disarms:            rec={recOff}  -> {pOff}")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'}")
    print(f"  [saved] {op}\n" + bar)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
