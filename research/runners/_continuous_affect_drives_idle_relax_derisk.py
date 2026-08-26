"""ANTI-HOLLOW verification of #91 (2026-08-26): the idle-tick "felt mood keeps evolving" relaxation extended to
the FLAGSHIP, default-ON, most user-visible affect->tone coupling (board #84, `webapp.affect_drives_chat`).

THE GAP THIS CLOSES. `continuous_engine.tick_session`'s headline mechanism only relaxed+re-read the LEGACY Gate-B
mood (`_SESSION_MOOD` + `_get_affect_organ().read_differential`). Board #84's OWN persistent EMA
(`AffectDrivesWorkspace.ema_valence/ema_arousal`) was written ONLY inside a live `observe_turn` and NEVER touched
by an idle tick -- so telling the brain something emotionally charged, waiting idle, then sending a neutral
follow-up produced the IDENTICAL #84 lead marker as zero idle time. `webapp.affect_drives_chat.relax_idle` +
`webapp.continuous_engine._affect_relax_drive_enabled`/the new idle-tick call close that gap.

THIS RUNNER goes through the REAL `/api/brain-chat` FastAPI endpoint (an in-process Starlette TestClient calling
the actual `webapp.server.app` -- the identical route function real HTTP would invoke, not a hand-rolled
reimplementation), with the STUB renderer forced BEFORE `webapp.server` is even imported
(`BRAIN_CHAT_RENDERER=stub`) so the startup Qwen-warm never fires and no request ever selects the qwen mouth --
per the anti-wedge note (the Qwen corpus + developed bridges live only in the primary checkout; a warm attempt in
an isolated worktree hangs). The heavy idle self-initiation WANDER is independently neutralized
(`BRAIN_WANDER_BUDGET=0`) so neither our own manual tick calls NOR the server's live 20s background tick loop can
ever trigger the ~55s CA3 wander during this run.

COST CONTROL (a real production `/api/brain-chat` first turn builds the full one-brain composer + GNW bus +
several co-resident organs -- several minutes on this worktree's GPU). To keep this de-risk tractable, ONE session
pays that build cost ONCE; the 4 conditions below then run on the SAME warm chat, resetting ONLY the two pieces of
per-condition state that must start neutral each time: (a) `chat._affect_drives_workspace` (the #84 EMA under
test -- deleting it makes the next `observe_turn` build a FRESH workspace, i.e. exactly the "new session" reset
this coupling cares about, at zero extra composer-build cost) and (b) the continuous-engine per-session dicts via
`forget_session` + `_SESSION_MOOD.pop`. This does not touch the coupling under test: `relax_idle` reads/writes
ONLY the (freshly-reset) workspace, never the composer.

DESIGN (no real sleep(), matching the sibling `_continuous_drive_loadbearing_cupy.py` convention): after the
induction turn, idle time is simulated by calling `continuous_engine.tick_idle_sessions` directly N times with an
explicit, advancing `now` (its documented parameter for exactly this purpose) rather than waiting on the real
20s-cadence background loop -- this is calling the SAME function production's background loop calls, on the SAME
global session-state dicts the live server request path just wrote to, so it genuinely exercises the tick's
production code path, just without a real wall-clock wait.

FOUR CONDITIONS (2x2: idle-vs-immediate x armed-vs-lesioned), run in sequence on the ONE warm session:
  I_on   -- induce, THEN send the neutral follow-up IMMEDIATELY (coupling ARMED, but no idle time passed)
            -> the #84 level should be the un-decayed (persisted) level -- observe()'s existing hold-prior logic.
  A_on   -- induce, THEN idle-tick N times (coupling ARMED)      -> the #84 level should DECAY toward neutral.
  I_off  -- induce, THEN immediate follow-up, coupling LESIONED -> the un-decayed baseline for the lesioned arm.
  A_off  -- induce, THEN idle-tick N times, coupling LESIONED (BRAIN_CONTINUOUS_AFFECT_RELAX=0)
            -> the #84 level should be IDENTICAL to I_off (no decay -- the LOAD-BEARING vanish).

GO bar: level(I_on) > level(A_on)  [idling with the coupling armed measurably decays the felt lead]
    AND level(I_off) == level(A_off)  [the SAME idle gap with the coupling lesioned changes NOTHING -- vanish]
    AND level(I_on) == level(I_off)  [the lesion flag itself does not touch the immediate/no-idle read]
    AND recalled_svo/abstained/verified IDENTICAL across all 4 conditions for the same neutral follow-up message
        [content is untouched; only the affect surface changes -- the no-regression floor]

Run: SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_affect_drives_idle_relax_derisk
Writes research/findings/raw/_continuous_affect_relax/idle_relax_derisk.json ; exit 0 iff GO.
"""
from __future__ import annotations

import os
import sys
import json
import time

# ── MUST be set before `webapp.server` is imported: forces the out-of-box renderer to 'stub' regardless of
# GPU/backend, so the startup Qwen-warm background thread never fires and no /api/brain-chat call can select the
# qwen mouth (the anti-wedge: the Qwen corpus + developed bridges live only in the primary checkout). ──────────────
os.environ["BRAIN_CHAT_RENDERER"] = "stub"
os.environ.setdefault("SIM_BACKEND", "cupy")
# Neutralize the heavy idle self-initiation wander (~55s CA3 run) so neither our manual tick calls nor the live
# server's own 20s-cadence background loop can ever trigger it while this session is idle.
os.environ["BRAIN_WANDER_BUDGET"] = "0"
# Keep the run minimal/fast + deterministic: arm only what this de-risk tests.
os.environ["BRAIN_CONTINUOUS"] = "1"       # master continuous-engine switch (tick_idle_sessions no-ops without it)
os.environ["BRAIN_AFFECT_DRIVES"] = "1"    # board #84 itself must be on for there to be a lead to test
os.environ.setdefault("BRAIN_D5_CONSOLIDATE", "0")   # unrelated coupling; keep this run's inner-life clean
os.environ.setdefault("BRAIN_DA_ENCODING", "0")      # unrelated coupling; keep this run's inner-life clean
os.environ.setdefault("BRAIN_CONTINUOUS_IDEATE", "0")  # unrelated; the wander budget=0 already blocks it anyway

OUT = os.path.join("research", "findings", "raw", "_continuous_affect_relax", "idle_relax_derisk.json")

INDUCE_MSG = "I am so wonderfully thrilled and joyful right now -- this is genuinely the best news I've heard all year!"
NEUTRAL_MSG = "Okay, noted."
N_IDLE_TICKS = 14   # 0.85**14 ~= 0.10 of the induced EMA -- enough to cross a Koulakov LEVEL boundary (not just an
                    # emphasis change within the same level), so the anti-hollow bar is a legible discrete-level drop
SESSION = "lb91-shared"
BRAIN, RENDERER = "tiny-demo", "stub"


def _turn(client, message: str) -> dict:
    r = client.post("/api/brain-chat", json={
        "session": SESSION, "message": message, "brain": BRAIN, "renderer": RENDERER, "rich": False,
    })
    r.raise_for_status()
    return r.json()


def _affect(resp: dict) -> dict:
    ad = resp.get("affect_drives")
    return ad if isinstance(ad, dict) else {}


def _reset_condition_state(CE, cache_key) -> None:
    """Reset ONLY the per-condition state (never the expensive warm ChatBrain/composer): drop the #84 workspace
    (so the next observe_turn starts from the neutral h=0.5/a=0 set-point, exactly like a fresh session) and the
    continuous-engine per-session bookkeeping. `chat` itself (and its composer/GNW-bus/organs) stays warm."""
    from webapp.server import _BRAIN_CHATS, _SESSION_MOOD
    chat = _BRAIN_CHATS.get(cache_key)
    if chat is not None and hasattr(chat, "_affect_drives_workspace"):
        delattr(chat, "_affect_drives_workspace")
    CE.forget_session(cache_key)
    _SESSION_MOOD.pop(cache_key, None)


def _run_condition(client, CE, cache_key, *, idle: bool, lesion: bool) -> dict:
    os.environ["BRAIN_CONTINUOUS_AFFECT_RELAX"] = "0" if lesion else "1"
    _reset_condition_state(CE, cache_key)

    r1 = _turn(client, INDUCE_MSG)
    a1 = _affect(r1)

    if idle:
        # simulate N idle ticks with an ADVANCING explicit `now` (documented tick_idle_sessions parameter) --
        # calling the identical production function the live server's background loop calls, on the SAME global
        # session-state dicts the request path above just wrote to. selfinit_getter=None / episodic_getter=None
        # deliberately skip the wander/D5 side-channels (already independently neutralized/disabled above) so
        # this isolates the ONE coupling under test.
        from webapp.server import _SESSION_MOOD, _get_affect_organ, _get_chat_existing
        now = time.time()
        for _ in range(N_IDLE_TICKS):
            now += CE.IDLE_SEC + 1.0
            CE._LAST_REQUEST[cache_key] = now - CE.IDLE_SEC - 1.0   # force "idle" for this tick
            CE.tick_idle_sessions(_SESSION_MOOD, _get_affect_organ, now=now,
                                   selfinit_getter=None, episodic_getter=None,
                                   chat_getter=_get_chat_existing)

    r2 = _turn(client, NEUTRAL_MSG)
    a2 = _affect(r2)
    return {
        "idle": idle, "lesion": lesion,
        "induce_level": a1.get("level"), "induce_lead": a1.get("lead"), "induce_mood": a1.get("mood"),
        "followup_level": a2.get("level"), "followup_lead": a2.get("lead"), "followup_mood": a2.get("mood"),
        "followup_abstained": r2.get("abstained"), "followup_recalled_svo": r2.get("recalled_svo"),
        "followup_verified": r2.get("verified"),
    }


def main() -> int:
    from sim.backend import get_backend
    _, backend_name = get_backend()

    from webapp.server import app
    from webapp import continuous_engine as CE
    from starlette.testclient import TestClient

    out = {
        "runner": "research/runners/_continuous_affect_drives_idle_relax_derisk.py",
        "run_id": os.environ.get("SIM_RUN_ID", "unset"),
        "backend": backend_name,
        "device": "cuda:0" if backend_name == "cupy" else "cpu",
        "seed": 42,
        "n_idle_ticks": N_IDLE_TICKS,
        "idle_sec": CE.IDLE_SEC,
        "relax": CE.RELAX,
        "induce_msg": INDUCE_MSG,
        "neutral_msg": NEUTRAL_MSG,
        "session_reuse": "ONE warm ChatBrain reused across all 4 conditions; only the #84 workspace + "
                          "continuous-engine per-session dicts are reset between conditions (see _reset_condition_state)",
    }

    cache_key = (SESSION, BRAIN, RENDERER)
    t0 = time.time()
    with TestClient(app) as client:
        # pay the one-time full chat-brain build (composer + GNW bus + co-resident organs) on a throwaway turn,
        # so it is not counted inside any one condition's timing and every condition starts from the SAME warm chat.
        _turn(client, "hello")
        t_build = time.time()
        out["chat_build_s"] = round(t_build - t0, 2)

        cond = {}
        cond["I_on"] = _run_condition(client, CE, cache_key, idle=False, lesion=False)
        cond["A_on"] = _run_condition(client, CE, cache_key, idle=True, lesion=False)
        cond["I_off"] = _run_condition(client, CE, cache_key, idle=False, lesion=True)
        cond["A_off"] = _run_condition(client, CE, cache_key, idle=True, lesion=True)
    out["wall_s"] = round(time.time() - t0, 2)
    out["conditions"] = cond

    lvl = {k: cond[k]["followup_level"] for k in cond}
    out["levels"] = lvl

    # ---- GO CHECKS ----
    induced_ok = all((cond[k]["induce_level"] or 0) > 0 for k in cond)   # the induction actually moved the mood
    idle_decays_when_armed = (lvl["I_on"] is not None and lvl["A_on"] is not None
                               and lvl["I_on"] > lvl["A_on"])
    vanishes_under_lesion = (lvl["I_off"] == lvl["A_off"])
    lesion_flag_itself_inert_on_immediate = (lvl["I_on"] == lvl["I_off"])

    # content fields (abstained / recalled_svo / verified) must be identical across all 4 for the SAME follow-up msg
    keys = ["followup_abstained", "followup_recalled_svo", "followup_verified"]
    content_vals = {k: [cond[c][k] for c in cond] for k in keys}
    content_identical = all(len(set(json.dumps(v, sort_keys=True) for v in vals)) == 1
                             for vals in content_vals.values())

    out["PASS_induced_nonzero"] = induced_ok
    out["PASS_idle_decays_when_armed"] = idle_decays_when_armed
    out["PASS_vanishes_under_lesion"] = vanishes_under_lesion
    out["PASS_lesion_flag_inert_on_immediate_read"] = lesion_flag_itself_inert_on_immediate
    out["PASS_content_fields_identical"] = content_identical

    verdict_go = (induced_ok and idle_decays_when_armed and vanishes_under_lesion
                  and lesion_flag_itself_inert_on_immediate and content_identical)
    out["VERDICT"] = "GO" if verdict_go else "UNDEFINED"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str), flush=True)
    return 0 if verdict_go else 1


if __name__ == "__main__":
    sys.exit(main())
