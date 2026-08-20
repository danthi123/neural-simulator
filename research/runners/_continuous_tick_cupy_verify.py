"""Verify the continuous-state engine's idle TICK runs end-to-end on the CUPY (production) substrate.

The tick (webapp/continuous_engine.tick_session) does two spiking reads that, before the cupy sim-step fix
(commit 34bbdfd7), crashed on cupy via the scipy-hybrid matmul:
  (a) affect_organ.read_differential(...)   -> the spiking affect-ladder read
  (b) selfinit_organ.speak(...)             -> the self-initiation CA3 wander (FULL 4000-step op point on cupy)
This replicates exactly what the server's background loop calls, on cupy, and asserts both reads succeed.

Run (from repo root):
  SIM_BACKEND=cupy BRAIN_CONTINUOUS=1 BRAIN_SELF_INITIATE_STORE=1 \
    .venv/bin/python -m research.runners._continuous_tick_cupy_verify
Writes: research/findings/raw/2026-08-20-continuous-tick-cupy-verify.json ; exit 0 iff VERDICT==GO.
"""
import os, sys, time, json

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("BRAIN_CONTINUOUS", "1")
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")  # force the heavy CA3 wander even off the cupy auto-gate

OUT = os.path.join("research", "findings", "raw", "2026-08-20-continuous-tick-cupy-verify.json")


def main() -> int:
    out = {"backend_requested": os.environ["SIM_BACKEND"],
           "produced_by": "research/runners/_continuous_tick_cupy_verify.py"}

    from sim.backend import get_backend
    xp, name = get_backend()
    out["backend_actual"] = name
    assert name == "cupy", f"NOT ON CUPY (got {name}) -- verification would be meaningless"

    from webapp import continuous_engine as CE
    from research.runners.affect_production_organ import get_organ as get_affect
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan

    t0 = time.time(); affect = get_affect(seed=42); out["affect_build_s"] = round(time.time() - t0, 2)
    t0 = time.time(); selfinit = SelfInitiationOrgan(seed=42); out["selfinit_build_s"] = round(time.time() - t0, 2)

    cache_key = "verify-cupy-session"
    session_mood = {cache_key: {"valence": 0.6, "arousal": 0.4}}
    CE.mark_request(cache_key)

    t0 = time.time()
    rec = CE.tick_session(cache_key, session_mood, affect, now=time.time(), selfinit_organ=selfinit)
    out["tick_s"] = round(time.time() - t0, 2)

    out["tick_returned"] = rec is not None
    if rec is not None:
        out["valence_relaxed_from_0.6_to"] = round(rec["valence"], 4)
        out["arousal_relaxed_from_0.4_to"] = round(rec["arousal"], 4)
        out["affect_differential"] = rec.get("differential")
        out["wandered_concept"] = rec.get("wandered")
        out["note"] = rec.get("note")

    ok_relax = rec is not None and abs(rec["valence"] - 0.51) < 1e-6 and abs(rec["arousal"] - 0.34) < 1e-6
    ok_affect = rec is not None and isinstance(rec.get("differential"), (int, float))
    ok_wander = rec is not None and bool(rec.get("wandered"))
    out["preconditions"] = {
        "on_cupy": {"ok": name == "cupy"},
        "affect_spiking_read_returned_number": {"ok": bool(ok_affect)},
        "selfinit_ca3_wander_ran_on_cupy": {"ok": bool(ok_wander)},
        "mood_relaxed": {"ok": bool(ok_relax)},
    }
    out["VERDICT"] = "GO" if (ok_relax and ok_affect and ok_wander) else "UNDEFINED"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return 0 if out["VERDICT"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
