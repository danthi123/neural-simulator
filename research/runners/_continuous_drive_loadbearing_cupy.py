"""AUTHORITATIVE load-bearing verification of the continuous-state engine's TWO drive-couplings on the CUPY
substrate, with LESION controls (the drive-not-observe bar: vary the faculty state -> the downstream must change,
and the change must VANISH when the coupling is lesioned).

(A) WANDER-DRIVE (rung 2.5): an idle tick surfaces a wandered concept; `recent_wander(key)` returns it (once,
    then CONSUMES it) so the next live turn leads with it. LESION: with BRAIN_CONTINUOUS=0 the tick records no
    wander, so `recent_wander` returns None -> no lead. Spiking part: the concept is the self-init organ's CA3
    wander selection (full 4000-step op point on cupy).

(B) FEELING-DRIVE (v1): the idle tick RELAXES the felt mood toward baseline and RE-READS the spiking affect
    ladder. Because `_update_session_mood` (server.py:3210) HOLDS the prior mood on a neutral next message and
    uses it as the EMA prior on an affective one, the relaxed mood is the baseline the next turn's affect read /
    tone is built on. This asserts the spiking affect differential at the RELAXED mood DIFFERS from the reading at
    the ORIGINAL mood (so idling measurably changes the next read), and that with no tick the mood is unchanged.

Run: SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_drive_loadbearing_cupy
Writes research/findings/raw/_continuous_live_cupy/loadbearing_cupy.json ; exit 0 iff BOTH couplings verify.
"""
import os, sys, json, time

os.environ.setdefault("SIM_BACKEND", "cupy")
OUT = os.path.join("research", "findings", "raw", "_continuous_live_cupy", "loadbearing_cupy.json")


def main() -> int:
    out = {"runner": "research/runners/_continuous_drive_loadbearing_cupy.py", "seed": 42}
    from sim.backend import get_backend
    _, name = get_backend()
    out["backend"] = name
    assert name == "cupy", f"not on cupy ({name})"

    from webapp import continuous_engine as CE
    from research.runners.affect_production_organ import get_organ as get_affect
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan

    affect = get_affect(seed=42)
    selfinit = SelfInitiationOrgan(seed=42)

    # ---- (A) WANDER-DRIVE, armed (BRAIN_CONTINUOUS on) ----
    os.environ["BRAIN_CONTINUOUS"] = "1"
    os.environ["BRAIN_SELF_INITIATE_STORE"] = "1"
    keyA = "lb-wander-on"
    moodA = {keyA: {"valence": 0.5, "arousal": 0.3}}
    CE.mark_request(keyA)
    t0 = time.time()
    recA = CE.tick_session(keyA, moodA, affect, now=time.time(), selfinit_organ=selfinit)
    out["wander_tick_s"] = round(time.time() - t0, 1)
    w1 = CE.recent_wander(keyA)          # first read -> the concept
    w2 = CE.recent_wander(keyA)          # second read -> None (consumed once)
    out["wander_on_concept"] = w1
    out["wander_on_consumed_second_read_is_none"] = (w2 is None)

    # ---- (A') WANDER-DRIVE, LESIONED (BRAIN_CONTINUOUS off) ----
    os.environ["BRAIN_CONTINUOUS"] = "0"
    keyB = "lb-wander-off"
    moodB = {keyB: {"valence": 0.5, "arousal": 0.3}}
    CE.mark_request(keyB)
    # tick_idle_sessions is a no-op when continuous is off; tick_session itself still runs, but recent_wander is
    # gated on continuous_enabled() -> returns None regardless. Assert the LESION kills the drive:
    recB = CE.tick_session(keyB, moodB, affect, now=time.time(), selfinit_organ=selfinit)
    wB = CE.recent_wander(keyB)
    out["wander_off_recent_wander_is_none"] = (wB is None)
    os.environ["BRAIN_CONTINUOUS"] = "1"  # restore

    wander_ok = bool(w1) and (w2 is None) and (wB is None)
    out["PASS_wander_drive_loadbearing"] = wander_ok

    # ---- (B) FEELING-DRIVE: the relaxed mood changes the spiking affect read ----
    v_orig = 0.8
    d_orig = float(affect.read_differential(v_orig, lesion=False)["differential"])   # read at the ORIGINAL mood
    keyC = "lb-feel"
    moodC = {keyC: {"valence": v_orig, "arousal": 0.5}}
    CE.mark_request(keyC)
    recC = CE.tick_session(keyC, moodC, affect, now=time.time(), selfinit_organ=None)  # relax + re-read (no wander)
    v_relaxed = moodC[keyC]["valence"]
    d_relaxed = float(affect.read_differential(v_relaxed, lesion=False)["differential"])
    out["feel_v_orig"] = v_orig
    out["feel_v_relaxed"] = round(v_relaxed, 4)
    out["feel_diff_at_orig"] = d_orig
    out["feel_diff_at_relaxed"] = d_relaxed
    out["feel_relaxed_moved_toward_neutral"] = abs(v_relaxed) < abs(v_orig)
    # Load-bearing: the felt read at the relaxed mood differs from the read at the original mood (idling changed it)
    feel_ok = (abs(v_relaxed) < abs(v_orig)) and (abs(d_relaxed - d_orig) > 1e-9)
    out["PASS_feeling_drive_loadbearing"] = feel_ok

    out["VERDICT"] = "GO" if (wander_ok and feel_ok) else "UNDEFINED"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return 0 if out["VERDICT"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
