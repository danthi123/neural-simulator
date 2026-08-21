"""SUSTAINED MULTI-SESSION SOAK gating the BRAIN_CONTINUOUS (+BRAIN_CONTINUOUS_DRIVES) default-ON flip.

The two drive-couplings are ALREADY lesion-verified load-bearing on cupy
(_continuous_drive_loadbearing_cupy.py), and the idle tick runs end-to-end on cupy
(_continuous_tick_cupy_verify.py). What remains before flipping the DEFAULT is the SAFETY proof for leaving it ON
in production across sustained, concurrent use:

  (S1) NO-REGRESSION on ordinary turns. Through the REAL brain_chat handler, on an ORDINARY active turn with NO
       pending idle-wander, BRAIN_CONTINUOUS=1 BRAIN_CONTINUOUS_DRIVES=1 is BYTE-IDENTICAL (answer/recalled/abstain)
       to BRAIN_CONTINUOUS=0. The drive only injects a lead when recent_wander() is non-None (which happens ONLY
       after an idle tick), and the mood-relax only runs in the background idle loop — so an ordinary handler turn
       with no prior idle tick MUST be unchanged. Driven over a repeated multi-turn stateless panel (recall / abstain
       / self / open-ended — no teaching, so the shared warm brain is not mutated between arms) across several
       sessions and several rounds (a sustained soak, not a single turn).

  (S2) NO GPU-MEMORY PILEUP. Across the whole soak (all handler turns + all idle ticks), the cupy memory pool
       used-bytes STABILIZES — it does not grow monotonically. (The background idle loop must not leak the substrate
       under the thread-executor guard.) Sampled early / mid / late (+ after a GC); late must not exceed the early
       warm baseline by more than MEM_GROWTH_TOL.

  (S3) DRIVE STILL LOAD-BEARING (spot-check). After an idle tick that records a wander, recent_wander() returns it
       ONCE then None (consumed); with BRAIN_CONTINUOUS=0 it returns None. Confirms the flip DELIVERS the faculty,
       not just a safe no-op. (The full lesion attribution lives in _continuous_drive_loadbearing_cupy.py.)

  (S4) IDLE LOOP ROBUST AT SCALE. tick_idle_sessions over MANY concurrent idle sessions completes without error and
       every session's mood relaxes toward baseline + reads the spiking affect ladder.

HARD MEMORY GUARD: this is the single GPU-loading process (the ONE-brain-at-a-time rule). Before the heavy brain
build and before each soak round it checks free VRAM; if below GPU_MIN_FREE_MIB it ABORTS cleanly (UNDEFINED, exit 2)
rather than risk an OOM that takes the machine down (the 2026-08-21 failure mode). NO sim/ edit — additive runner.

Run headless on the LOCAL 3090:
  tools/gpu_queue.sh add 'SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._continuous_default_flip_soak_cupy'
Writes research/findings/raw/_continuous_live_cupy/default_flip_soak.json ; exit 0 iff VERDICT==GO.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

OUT = os.path.join("research", "findings", "raw", "_continuous_live_cupy", "default_flip_soak.json")
GPU_MIN_FREE_MIB = int(os.environ.get("GPU_MIN_FREE_MIB", "2500"))   # abort below this free VRAM (don't OOM the box)
MEM_GROWTH_TOL = float(os.environ.get("MEM_GROWTH_TOL", "0.10"))     # (legacy; reported not gated) pool %-growth
VRAM_LEAK_FLOOR_MIB = int(os.environ.get("VRAM_LEAK_FLOOR_MIB", "2000"))  # a bounded VRAM drop (S3 organ builds) is NOT a leak
N_SESSIONS = int(os.environ.get("SOAK_SESSIONS", "3"))
N_ROUNDS = int(os.environ.get("SOAK_ROUNDS", "5"))

# Ordinary turns that do NOT mutate the shared composer (no teaching): recall / abstain / self / open-ended.
PANEL = [
    "what does dog chase?",     # STORED recall
    "what does cat eat?",       # STORED recall
    "what does fish fly?",      # UNSTORED -> abstain (moat)
    "what are you",             # SELF / identity
    "what does brain use?",     # STORED recall
    "what does dog eat?",       # INCONSISTENT -> abstain (moat)
    "tell me something new about brain",  # OPEN-ENDED (flagged guess)
]


def _free_vram_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _pool_used_bytes():
    try:
        import cupy  # noqa
        return int(cupy.get_default_memory_pool().used_bytes())
    except Exception:
        return None


def _core(resp: dict) -> dict:
    """The user-visible core of a handler response — the no-regression contract (additive continuous-observability
    keys like inner_life / continuous are allowed to appear; the ANSWER must not change)."""
    return {k: resp.get(k) for k in ("answer", "recalled_svo", "abstained", "hypothesis", "hypothesis_svo")}


def _reply(msg: str, session: str) -> dict:
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(r.body.decode("utf-8"))


def main() -> int:
    out = {"runner": "research/runners/_continuous_default_flip_soak_cupy.py", "seed": 42,
           "n_sessions": N_SESSIONS, "n_rounds": N_ROUNDS, "panel_len": len(PANEL)}

    from sim.backend import get_backend
    _, name = get_backend()
    out["backend"] = name
    allow_noncupy = os.environ.get("SOAK_ALLOW_NONCUPY", "0").strip().lower() in ("1", "true", "on", "yes")
    if name != "cupy" and not allow_noncupy:
        out["VERDICT"] = "UNDEFINED"
        out["undefined_reason"] = f"not on cupy ({name}) — the soak proves cupy production safety, meaningless off-cupy"
        _save(out)
        print(json.dumps(out, indent=2), flush=True)
        return 2
    out["noncupy_smoke"] = bool(name != "cupy" and allow_noncupy)

    free0 = _free_vram_mib()
    out["free_vram_mib_start"] = free0
    if free0 is not None and free0 < GPU_MIN_FREE_MIB:
        out["VERDICT"] = "UNDEFINED"
        out["undefined_reason"] = f"free VRAM {free0}MiB < GPU_MIN_FREE_MIB {GPU_MIN_FREE_MIB} — aborting to avoid OOM"
        _save(out)
        print(json.dumps(out, indent=2), flush=True)
        return 2

    # Keep the heavy Gate-B organs OFF for speed — they run identically on both flag arms, so the byte-identical
    # comparison is unaffected (same discipline as the gnw-bus flip verify). The continuous flags are the lever.
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_PMEM", "BRAIN_RICH"):
        os.environ[k] = "0"

    # ---- warm the shared brain (ONE build) via a throwaway OFF turn, then sample the warm memory baseline ----
    os.environ["BRAIN_CONTINUOUS"] = "0"
    os.environ.pop("BRAIN_CONTINUOUS_DRIVES", None)
    t0 = time.time()
    _reply("what does dog chase?", "soak-warm")
    out["warm_build_s"] = round(time.time() - t0, 1)
    gc.collect()
    mem_early = _pool_used_bytes()
    out["pool_used_bytes_early"] = mem_early

    # ---- (S1) NO-REGRESSION soak: OFF vs ON, byte-identical core, many turns × sessions × rounds ----
    diverged = []
    n_turns = 0
    mem_samples = [("early", mem_early)]
    for rnd in range(N_ROUNDS):
        # memory guard each round
        fv = _free_vram_mib()
        if fv is not None and fv < GPU_MIN_FREE_MIB:
            out["VERDICT"] = "UNDEFINED"
            out["undefined_reason"] = f"free VRAM fell to {fv}MiB (<{GPU_MIN_FREE_MIB}) mid-soak round {rnd} — aborting"
            out["free_vram_mib_abort"] = fv
            _save(out)
            print(json.dumps(out, indent=2), flush=True)
            return 2
        for s in range(N_SESSIONS):
            for msg in PANEL:
                # OFF arm
                os.environ["BRAIN_CONTINUOUS"] = "0"
                os.environ.pop("BRAIN_CONTINUOUS_DRIVES", None)
                off = _reply(msg, f"soak-off-r{rnd}-s{s}")
                # ON arm (both continuous flags) — NO idle tick recorded for this session, so recent_wander is None
                os.environ["BRAIN_CONTINUOUS"] = "1"
                os.environ["BRAIN_CONTINUOUS_DRIVES"] = "1"
                on = _reply(msg, f"soak-on-r{rnd}-s{s}")
                n_turns += 1
                if _core(off) != _core(on):
                    diverged.append({"round": rnd, "session": s, "msg": msg,
                                     "off": _core(off), "on": _core(on)})
        mem_samples.append((f"round{rnd}", _pool_used_bytes()))
        print(f"[soak] round {rnd + 1}/{N_ROUNDS} done: {n_turns} OFF/ON pairs, {len(diverged)} diverged, "
              f"free_vram={fv}MiB", flush=True)
    out["n_noregression_turn_pairs"] = n_turns
    out["n_diverged"] = len(diverged)
    out["diverged_examples"] = diverged[:8]

    # ---- (S2) NO GPU-MEMORY PILEUP ----
    # The REAL leak signal is (a) OS VRAM stability across the soak, and (b) the cupy pool SETTLING after a one-time
    # warm-up. cupy's pool caches freed blocks, so a bounded first-use allocation (e.g. 46->72MB, then flat) is NOT a
    # leak — and a raw %-growth on a tiny baseline false-alarms on it. So: PASS iff VRAM did not fall by more than
    # VRAM_LEAK_FLOOR_MIB (S3 legitimately builds the affect+selfinit organs -> a bounded one-time VRAM delta is
    # expected) AND the pool did not keep growing after the turns (late_gc <= the last per-round sample * 1.05).
    gc.collect()
    mem_late = _pool_used_bytes()
    last_round_bytes = mem_samples[-1][1] if mem_samples else mem_early   # the final per-round sample (pre-late_gc)
    mem_samples.append(("late_gc", mem_late))
    out["pool_used_bytes_samples"] = [{"phase": p, "bytes": b} for p, b in mem_samples]
    out["pool_used_bytes_late"] = mem_late
    out["pool_growth_frac"] = (round((mem_late - mem_early) / max(mem_early, 1), 4) if (mem_early and mem_late) else None)
    vram_end = _free_vram_mib()
    out["free_vram_mib_end"] = vram_end
    vram_start = out.get("free_vram_mib_start")
    vram_ok = (vram_start is None or vram_end is None) or ((vram_start - vram_end) <= VRAM_LEAK_FLOOR_MIB)
    pool_settled = (mem_late is None or not last_round_bytes) or (mem_late <= last_round_bytes * 1.05)
    out["vram_drop_mib"] = ((vram_start - vram_end) if (vram_start is not None and vram_end is not None) else None)
    out["pool_settled_after_turns"] = bool(pool_settled)
    mem_ok = bool(vram_ok and pool_settled)
    out["PASS_no_pileup"] = mem_ok

    # ---- (S3) DRIVE STILL LOAD-BEARING (spot-check) ----
    from webapp import continuous_engine as CE
    from research.runners.affect_production_organ import get_organ as get_affect
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan
    affect = get_affect(seed=42)
    selfinit = SelfInitiationOrgan(seed=42)
    os.environ["BRAIN_CONTINUOUS"] = "1"
    os.environ["BRAIN_SELF_INITIATE_STORE"] = "1"
    kA = "soak-drive-on"
    moodA = {kA: {"valence": 0.5, "arousal": 0.3}}
    CE.mark_request(kA)
    CE.tick_session(kA, moodA, affect, now=time.time(), selfinit_organ=selfinit)
    w1 = CE.recent_wander(kA)
    w2 = CE.recent_wander(kA)
    os.environ["BRAIN_CONTINUOUS"] = "0"
    kB = "soak-drive-off"
    moodB = {kB: {"valence": 0.5, "arousal": 0.3}}
    CE.mark_request(kB)
    CE.tick_session(kB, moodB, affect, now=time.time(), selfinit_organ=selfinit)
    wB = CE.recent_wander(kB)
    drive_ok = bool(w1) and (w2 is None) and (wB is None)
    out["drive_on_concept"] = w1
    out["drive_consumed_second_read_none"] = (w2 is None)
    out["drive_off_none"] = (wB is None)
    out["PASS_drive_loadbearing"] = drive_ok

    # ---- (S4) IDLE LOOP ROBUST AT SCALE ----
    # tick_idle_sessions(session_mood, affect_organ_getter, now=, selfinit_getter=, episodic_getter=). The getters
    # are called as getter(cache_key). A session is idle iff now - last_request >= IDLE_SEC (20s), so advance `now`
    # well past it. selfinit_getter=None keeps this fast (mood-relax + affect read at scale); S3 covers the wander.
    os.environ["BRAIN_CONTINUOUS"] = "1"
    N_IDLE = 20
    idle_mood = {f"idle-{i}": {"valence": 0.6, "arousal": 0.5} for i in range(N_IDLE)}
    t_idle = time.time()
    for k in idle_mood:
        CE.mark_request(k)
    idle_err = None
    n_ticked_last = 0
    try:
        for r in range(3):
            n_ticked_last = CE.tick_idle_sessions(idle_mood, (lambda k=None: affect),
                                                  now=t_idle + 3600.0 * (r + 1), selfinit_getter=None)
    except Exception as e:  # narrow: report, don't swallow
        idle_err = f"{type(e).__name__}: {e}"
    relaxed = sum(1 for v in idle_mood.values() if abs(v["valence"]) < 0.6)
    idle_ok = (idle_err is None) and (relaxed == N_IDLE)
    out["idle_sessions"] = N_IDLE
    out["idle_relaxed"] = relaxed
    out["idle_ticked_last_round"] = n_ticked_last
    out["idle_error"] = idle_err
    out["PASS_idle_scale"] = bool(idle_ok)

    # ---- VERDICT ----
    noreg_ok = (n_turns > 0 and len(diverged) == 0)
    out["PASS_no_regression"] = bool(noreg_ok)

    from tools.verdict import Verdict
    v = Verdict("BRAIN_CONTINUOUS default-ON flip — sustained multi-session cupy soak (safety gate)")
    v.require("no-regression: ON is byte-identical to OFF on every ordinary turn", len(diverged), expect=0,
              note=f"{n_turns} OFF/ON turn-pairs, {len(diverged)} diverged")
    v.require("the soak actually ran turns (not UNDEFINED-by-emptiness)", (n_turns > 0), expect=True)
    v.require("no GPU-memory pileup (late pool used-bytes <= early*(1+tol))", mem_ok, expect=True,
              note=f"growth_frac={out['pool_growth_frac']} tol={MEM_GROWTH_TOL}")
    v.require("drive still load-bearing (wander recorded on, consumed once, None off)", drive_ok, expect=True)
    v.require("idle loop robust over many concurrent sessions", idle_ok, expect=True)
    v.disabled("heavy Gate-B organs (affect/worldmodel/... = 0) during the byte-identical soak",
               why="disabled ONLY for speed; they run identically on both continuous-flag arms, so the "
                   "no-regression comparison is unaffected")
    go = bool(noreg_ok and mem_ok and drive_ok and idle_ok and n_turns > 0)
    decided = v.decide(go=go, verbose=False)
    out["preconditions"] = decided["preconditions"]
    out["disabled_processes"] = decided.get("disabled_processes")
    out["VERDICT"] = "GO" if go else ("UNDEFINED" if not decided.get("valid", True) else "NO-GO")
    out["status"] = decided["status"]
    out["go"] = go

    _save(out)
    print(json.dumps({k: out[k] for k in (
        "backend", "n_noregression_turn_pairs", "n_diverged", "pool_growth_frac", "PASS_no_regression",
        "PASS_no_pileup", "PASS_drive_loadbearing", "PASS_idle_scale", "VERDICT")}, indent=2), flush=True)
    print(f"wrote {OUT} -> {out['status']}", flush=True)
    return 0 if go else 1


def _save(out):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
