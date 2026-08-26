"""ANTI-HOLLOW verification of #92 (2026-08-26): the idle-tick "keeps evolving between turns" relaxation extended
to the flagship, default-ON, most user-visible ENGAGEMENT->forthcomingness coupling (board #76/#79,
`webapp.da_mode_drives_chat`) -- a SECOND, independent axis alongside #91 (energy/engagement, not valence/warmth).

THE GAP THIS CLOSES. `continuous_engine.tick_session`'s "the brain keeps feeling between turns" mechanism relaxes
the legacy Gate-B mood, and (board #91) board #84's own affect EMA -- but `DaModeDrivesWorkspace.ema_engagement`
(the state behind the DA-mode engagement SUFFIX -- " -- worth going further here.", "there's plenty more to dig
into here!") was written ONLY inside a live `observe_turn`, and a content-free turn merely HOLDS it; an idle
session with NO turns at all never touches it either. So telling the brain something highly engaging, waiting
idle, then sending a neutral follow-up produced the IDENTICAL DA-mode engagement suffix as zero idle time.
`webapp.da_mode_drives_chat.relax_idle` + `webapp.continuous_engine._da_relax_drive_enabled`/the new idle-tick
call close that gap, mirroring #91's own closure of the analogous gap on #84.

THIS RUNNER goes through the REAL `/api/brain-chat` FastAPI endpoint (an in-process Starlette TestClient calling
the actual `webapp.server.app` -- the identical route function real HTTP would invoke), with the STUB renderer
forced BEFORE `webapp.server` is even imported (`BRAIN_CHAT_RENDERER=stub`) so the startup Qwen-warm never fires
and no request ever selects the qwen mouth -- per the anti-wedge note (the Qwen corpus + developed bridges live
only in the primary checkout; a warm attempt in an isolated worktree hangs). The heavy idle self-initiation WANDER
is independently neutralized (`BRAIN_WANDER_BUDGET=0`) so neither our own manual tick calls NOR the server's live
20s background tick loop can ever trigger the ~55s CA3 wander during this run.

COST CONTROL (mirrors `_continuous_affect_drives_idle_relax_derisk.py`): a real production `/api/brain-chat` first
turn builds the full one-brain composer + GNW bus + several co-resident organs -- several minutes on this
worktree's GPU. ONE session pays that build cost ONCE; the 4 conditions below then run on the SAME warm chat,
resetting ONLY the per-condition state that must start neutral each time: (a) `chat._da_drives_workspace` (the
DA-mode engagement EMA under test -- deleting it makes the next `observe_turn` build a FRESH workspace, i.e.
exactly the "new session" reset this coupling cares about, at zero extra composer-build cost) and (b) the
continuous-engine per-session dicts via `forget_session`. This does not touch the coupling under test: `relax_idle`
reads/writes ONLY the (freshly-reset) workspace, never the composer.

DESIGN (no real sleep(), matching the #91 convention): after the induction turn, idle time is simulated by calling
`continuous_engine.tick_idle_sessions` directly N times with an explicit, advancing `now` (its documented parameter
for exactly this purpose) rather than waiting on the real 20s-cadence background loop.

INDUCTION (different mechanics than #91's `BRAIN_AFFECT_DRIVES_INDUCE`): the DA-mode `afferent_override` affordance
(`BRAIN_DA_DRIVES_INDUCE`) does NOT persist into `ema_engagement` (it drives that one read's afferent directly,
bypassing the EMA fold -- see `DaModeDrivesWorkspace.observe`), so it would leave nothing for `relax_idle` to decay.
Induction here instead uses a genuinely rich/novel MESSAGE so the real `engagement_of()` fold raises the persistent
EMA exactly as production does -- the same mechanism a real engaged user turn would exercise.

FOUR CONDITIONS (2x2: idle-vs-immediate x armed-vs-lesioned), run in sequence on the ONE warm session:
  I_on   -- induce, THEN send the neutral follow-up IMMEDIATELY (coupling ARMED, but no idle time passed)
            -> the DA-mode should be the un-decayed (persisted) mode -- observe()'s existing hold-prior logic.
  A_on   -- induce, THEN idle-tick N times (coupling ARMED)      -> the DA-mode should DECAY toward rest.
  I_off  -- induce, THEN immediate follow-up, coupling LESIONED (BRAIN_CONTINUOUS_DA_RELAX=0) -> the un-decayed
            baseline for the lesioned arm.
  A_off  -- induce, THEN idle-tick N times, coupling LESIONED
            -> the DA-mode should be IDENTICAL to I_off (no decay -- the LOAD-BEARING vanish).

GO bar: mode_rank(I_on) > mode_rank(A_on)  [idling with the coupling armed measurably decays the engagement suffix]
    AND mode(I_off) == mode(A_off)  [the SAME idle gap with the coupling lesioned changes NOTHING -- vanish]
    AND mode(I_on) == mode(I_off)  [the lesion flag itself does not touch the immediate/no-idle read]
    AND recalled_svo/abstained/verified IDENTICAL across all 4 conditions for the same neutral follow-up message
        [content is untouched; only the DA-mode surface changes -- the no-regression floor]

Run: SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_da_drives_idle_relax_derisk
Writes research/findings/raw/_continuous_da_relax/da_idle_relax_derisk.json ; exit 0 iff GO.
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
os.environ["BRAIN_DA_DRIVES"] = "1"        # the DA-mode coupling itself must be on for there to be a suffix to test
os.environ.setdefault("BRAIN_AFFECT_DRIVES", "0")            # unrelated coupling; keep this run's surface clean
os.environ.setdefault("BRAIN_CONTINUOUS_AFFECT_RELAX", "0")  # unrelated (#91); isolate the coupling under test
os.environ.setdefault("BRAIN_D5_CONSOLIDATE", "0")            # unrelated coupling; keep this run's inner-life clean
os.environ.setdefault("BRAIN_DA_ENCODING", "0")               # unrelated coupling; keep this run's inner-life clean
os.environ.setdefault("BRAIN_CONTINUOUS_IDEATE", "0")         # unrelated; the wander budget=0 already blocks it anyway

OUT = os.path.join("research", "findings", "raw", "_continuous_da_relax", "da_idle_relax_derisk.json")

# A rich, highly-novel first turn: >=8 distinct non-stopword content tokens all unseen this session -> richness=1.0,
# novelty=1.0 -> turn_engagement=1.0 -> ema_engagement = (1-_EMA_DECAY)*1.0 = 0.6 (the single-turn ceiling) ->
# afferent ~840pA -> DA-mode FOCUS (a non-empty engagement suffix).
INDUCE_MSG = ("Wow, the quantum crystalline nebula fascinates ancient volcanic mountains beneath glimmering "
              "coral reefs today")
# NOT "Okay, noted." (the #91 sibling's neutral message) -- unlike #84's VALENCE appraisal (a fixed sentiment
# lexicon; "okay"/"noted" score zero hits -> the HOLD branch), the DA-mode ENGAGEMENT read is driven by
# novelty+richness of ANY non-stopword content token >=3 letters -- "okay"/"noted" are BOTH novel content words, so
# that message would itself re-engage ema_engagement on the follow-up turn (verified: _content_tokens("Okay,
# noted.") == ['okay','noted']), masking the very idle-decay this de-risk measures. "Ok." has ZERO qualifying
# tokens ("ok" is 2 letters, below _MIN_CONTENT_LEN=3) -> _content_tokens("Ok.") == [] -> the HOLD branch fires,
# exactly like the induce message's the true engagement-neutral probe this coupling's contract requires.
NEUTRAL_MSG = "Ok."
# 0.85**10 ~= 0.197 of the induced EMA (0.6 -> ~0.118 -> afferent ~165pA -> DA ~0.24, comfortably below the REST
# threshold 0.40); crosses the FOCUS->REST bin boundary with margin while costing less real-background-thread
# interference than 14 ticks (each manual tick this E2E harness runs also risks a competing tick from the server's
# own live 20s-cadence background loop, since BRAIN_CONTINUOUS=1 is required for either to fire).
N_IDLE_TICKS = 10
SESSION = "lb92-shared"
BRAIN, RENDERER = "tiny-demo", "stub"

_MODE_RANK = {"rest": 0, "neutral": 1, "focus": 2, "arousal": 3}


def _turn(client, message: str) -> dict:
    r = client.post("/api/brain-chat", json={
        "session": SESSION, "message": message, "brain": BRAIN, "renderer": RENDERER, "rich": False,
    })
    r.raise_for_status()
    return r.json()


def _da(resp: dict) -> dict:
    dd = resp.get("da_drives")
    return dd if isinstance(dd, dict) else {}


def _reset_condition_state(CE, cache_key) -> None:
    """Reset ONLY the per-condition state (never the expensive warm ChatBrain/composer): drop the DA-mode
    workspace (so the next observe_turn starts from ema_engagement=0.0, exactly like a fresh session) and the
    continuous-engine per-session bookkeeping. `chat` itself (and its composer/GNW-bus/organs) stays warm."""
    from webapp.server import _BRAIN_CHATS, _SESSION_MOOD
    chat = _BRAIN_CHATS.get(cache_key)
    if chat is not None and hasattr(chat, "_da_drives_workspace"):
        delattr(chat, "_da_drives_workspace")
    CE.forget_session(cache_key)
    _SESSION_MOOD.pop(cache_key, None)


def _run_condition(client, CE, cache_key, *, idle: bool, lesion: bool) -> dict:
    os.environ["BRAIN_CONTINUOUS_DA_RELAX"] = "0" if lesion else "1"
    _reset_condition_state(CE, cache_key)

    r1 = _turn(client, INDUCE_MSG)
    d1 = _da(r1)

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
    d2 = _da(r2)
    return {
        "idle": idle, "lesion": lesion,
        "induce_mode": d1.get("mode"), "induce_lead": d1.get("lead"), "induce_da_level": d1.get("da_level"),
        "induce_ema_engagement": d1.get("ema_engagement"),
        "followup_mode": d2.get("mode"), "followup_lead": d2.get("lead"), "followup_da_level": d2.get("da_level"),
        "followup_ema_engagement": d2.get("ema_engagement"),
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
        "runner": "research/runners/_continuous_da_drives_idle_relax_derisk.py",
        "run_id": os.environ.get("SIM_RUN_ID", "unset"),
        "backend": backend_name,
        "device": "cuda:0" if backend_name == "cupy" else "cpu",
        "seed": 42,
        "n_idle_ticks": N_IDLE_TICKS,
        "idle_sec": CE.IDLE_SEC,
        "relax": CE.RELAX,
        "induce_msg": INDUCE_MSG,
        "neutral_msg": NEUTRAL_MSG,
        "session_reuse": "ONE warm ChatBrain reused across all 4 conditions; only the DA-mode workspace + "
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
        for _name, _idle, _lesion in (("I_on", False, False), ("A_on", True, False),
                                       ("I_off", False, True), ("A_off", True, True)):
            _tc0 = time.time()
            cond[_name] = _run_condition(client, CE, cache_key, idle=_idle, lesion=_lesion)
            print("[timing] condition %s took %.1fs" % (_name, time.time() - _tc0), flush=True)
    out["wall_s"] = round(time.time() - t0, 2)
    out["conditions"] = cond

    modes = {k: cond[k]["followup_mode"] for k in cond}
    out["modes"] = modes
    ranks = {k: _MODE_RANK.get(modes[k], -1) for k in cond}
    out["mode_ranks"] = ranks

    # ---- GO CHECKS ----
    induced_ok = all((cond[k]["induce_lead"] or "") != "" for k in cond)   # the induction actually engaged the mode
    idle_decays_when_armed = (ranks["I_on"] >= 0 and ranks["A_on"] >= 0 and ranks["I_on"] > ranks["A_on"])
    vanishes_under_lesion = (modes["I_off"] == modes["A_off"])
    lesion_flag_itself_inert_on_immediate = (modes["I_on"] == modes["I_off"])

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
