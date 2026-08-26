"""ANTI-HOLLOW verification of MULL (board #145, 2026-08-26): the between-turn self-initiated WANDER is now
genuinely DRIVEN by what the substrate has actually been discussing THIS conversation, not a fixed per-session draw.

THE GAP THIS CLOSES. Two "inner life shapes what it says" couplings already landed (#144 felt mood idle-relax, #92
felt engagement idle-relax) -- both are the SAME shape: a felt EMA fading toward neutral over idle time. Board #145
explicitly asked for a DIFFERENT faculty that genuinely CARRIES STATE across the gap. The between-turn WANDER
(`webapp.continuous_engine.tick_session` -> `SelfInitiationOrgan.speak`) already VARIES over ticks via inhibition-
of-return (#105, board-closed) so it is not degenerate -- but its per-concept curiosity weighting (`gains_on`) is
seeded from a per-concept NOVELTY level that is a FIXED, seed-derived permutation baked in once at organ-build time
(`self_initiated_production_organ._lexicon`/`NOV_BY_NMEM`, honestly declared as the "ENVIRONMENT" in that file's own
docstring). So the SAME four concepts (dog/cat/bird/fish) rotate in the SAME curiosity order in every conversation,
regardless of what was actually discussed: telling the brain about "the dog" vs never mentioning it produces the
IDENTICAL wander rotation. The mind's unprompted "Something's been on my mind..." remark is not actually shaped by
what it has genuinely been mulling over in THIS conversation.

THE MECHANISM (`webapp.continuous_engine.mark_mull` / `_apply_wander_mull`, wired into `tick_session`): when a live
turn's REFERENTIAL RECALL genuinely completes (`d5_episodic_production_organ`'s spiking dendritic-dAP pattern-
completion, in_memory=True -- the exact same gate `mark_recall`/D5 learn-through-use already trusts, reused
unchanged) on a topic that happens to ALSO be one of the self-init organ's own stored concepts, that concept's
curiosity NOVELTY is raised DECISIVELY above the organ's own stored-lexicon range (MULL_NOVELTY=1.6, vs the fixed
lexicon's 0-0.95) and RE-READ through the SAME spiking `CuriosityProductionOrgan` the organ's fixed baseline already
comes from -- a genuine higher-novelty spiking read of a concept the substrate just actually engaged with, never a
host gain multiply. A first attempt used the organ's own 0.95 ceiling, which only TIES the mulled concept with
whichever concept a fixed seeded permutation happened to already put there (~2-3% edge) -- too close to survive this
codebase's own documented build-order RNG drift (CLAUDE.md: "each net build advances the global RNG"); MULL_NOVELTY
was raised to give a decisive ~1.7-1.9x want_hz margin instead (see `webapp/continuous_engine.py`'s MULL section for
the full derivation). The boost applies to EXACTLY the one wander call it arms (consumed, like `recent_wander`), and
`gains_on` is restored immediately after so it can never
leak into inhibition-of-return's own base-capture.

THIS RUNNER goes through the REAL `/api/brain-chat` FastAPI endpoint (an in-process Starlette TestClient calling the
actual `webapp.server.app`), with the STUB renderer forced BEFORE `webapp.server` is even imported
(`BRAIN_CHAT_RENDERER=stub`) so the startup Qwen-warm never fires and no request ever selects the qwen mouth (the
anti-wedge: the Qwen corpus + developed bridges live only in the primary checkout; a warm attempt in an isolated
worktree hangs).

TEST-DESIGN CARE (the earlier-agent false-UNDEFINED lesson): the neutral follow-up probe must not itself be able to
re-trigger a genuine recall (which would re-arm MULL and confound the "vanishes under lesion" read). "Ok." (the SAME
probe #92's own de-risk verified) has zero qualifying content tokens AND does not match `_REFERENTIAL_RE` (no
"you mentioned"/"earlier you"/etc.), so it cannot itself extract a referent or re-arm anything.

COST CONTROL: a real production `/api/brain-chat` first turn builds the full one-brain composer + GNW bus + several
co-resident organs -- several minutes on this worktree's GPU. ONE session pays that build cost ONCE; the 2 conditions
below (armed / lesioned) then run on the SAME warm chat, resetting ONLY the per-condition state that must start
neutral each time: the D5 episodic organ (`d5_episodic_production_organ._ORGANS`, so 'dog' must be freshly RE-formed
+ RE-recalled each condition), the self-init organ (`webapp.server._SESSION_SELFINIT`, so `gains_on`/`novelties`/the
IOR adaptation state are fresh -- inhibition-of-return is then a documented no-op on a session's first-ever wander,
so it cannot itself explain any difference between conditions), and the continuous-engine per-session dicts
(`forget_session`). The warm `chat`/composer/GNW-bus itself is NEVER rebuilt.

DESIGN (no real sleep(), matching the #91/#92 convention): idle time is simulated by calling
`continuous_engine.tick_idle_sessions` directly with an explicit, advancing `now` rather than waiting on the real
20s-cadence background loop.

TWO CONDITIONS, run in sequence on the ONE warm session:
  ARMED     -- induce ("what does the dog chase?"), THEN a referential recall ("you mentioned the dog, right?")
               that genuinely completes, coupling ARMED (BRAIN_CONTINUOUS_WANDER_MULL=1) -> the idle wander should
               be biased toward 'dog' (mull's own concept) vs the organ's fixed baseline (argmax='cat' at seed 42).
  LESIONED  -- the IDENTICAL induce + referential turns, coupling LESIONED (BRAIN_CONTINUOUS_WANDER_MULL=0) -> the
               wander must be IDENTICAL to the organ's fixed, never-mulled baseline (the load-bearing vanish).

Plus a FAST, GPU-free MECHANISM-LEVEL check (no organ build): re-derives the exact baseline-vs-boosted curiosity
gains directly (mirrors `_apply_wander_mull`'s own arithmetic) to pin the numeric claim ("boosting dog's novelty to
MULL_NOVELTY flips the argmax from cat to dog with a DECISIVE margin, not a coin-flip tie") independent of the
stochastic CA3 wander's episode-routing noise.

GO bar:
  MECH_flips_argmax    -- the direct gains computation's argmax is 'cat' at baseline, 'dog' when boosted, AND the
                           margin is decisive (dog's boosted want_hz >= 1.5x cat's baseline want_hz) -- ruling out a
                           near-tie that this codebase's own documented build-order RNG drift could flip either way.
  ARMED_wander_shifts  -- the ARMED condition's idle-tick wander concept is 'dog' (or, failing a clean dominant
                           read, ARMED's dog-share/gains-rank strictly exceeds LESIONED's) AND the follow-up reply's
                           `wander_drives.concept` differs from LESIONED's (the USER-LEGIBLE content change).
  LESIONED_matches_baseline -- LESIONED's wander concept + follow-up wander_drives.concept equal a FRESH,
                           never-induced organ's own baseline pick (the load-bearing vanish: the SAME induce+
                           referential turns, coupling off, produce the untouched baseline).
  induction_ok          -- both conditions' induce turn actually recalled 'dog' AND the referential turn actually
                           completed in_memory=True (the setup worked, not a false negative from a failed induce).
  content_identical     -- abstained/recalled_svo/verified identical between ARMED and LESIONED for the SAME neutral
                           follow-up message (the coupling changes ONLY the wander lead, never the core content).

Run: SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_wander_mull_derisk
Writes research/findings/raw/_continuous_wander_mull/wander_mull_derisk.json ; exit 0 iff GO.
"""
from __future__ import annotations

import os
import sys
import json
import time

# ── MUST be set before `webapp.server` is imported (the anti-wedge; see module docstring). ─────────────────────────
os.environ["BRAIN_CHAT_RENDERER"] = "stub"
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ["BRAIN_CONTINUOUS"] = "1"          # master continuous-engine switch
os.environ["BRAIN_SELF_INITIATE"] = "1"       # the wander faculty itself must be on
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")  # force the FULL cupy CA3 wander path explicitly
os.environ.setdefault("BRAIN_WANDER_BUDGET", "1")         # one heavy wander available per idle period
os.environ["BRAIN_EPISODIC"] = "1"            # the referential-recall gate MULL rides
os.environ.setdefault("BRAIN_EPISODIC_STORE", "1")         # force the cupy BTSP write explicitly (backend gate)
os.environ.setdefault("BRAIN_D5_CONSOLIDATE", "0")          # unrelated coupling sharing the SAME mark_recall call
                                                              # site -- keep this run's substrate mutation isolated
os.environ.setdefault("BRAIN_DA_ENCODING", "0")              # unrelated; keep the idle-tick surface clean
os.environ.setdefault("BRAIN_CONTINUOUS_IDEATE", "0")         # unrelated -- force the plain recall-wander branch
os.environ.setdefault("BRAIN_CONTINUOUS_AFFECT_RELAX", "0")   # unrelated idle-relax axis (#91)
os.environ.setdefault("BRAIN_CONTINUOUS_DA_RELAX", "0")        # unrelated idle-relax axis (#92)
os.environ.setdefault("BRAIN_WANDER_IOR", "1")                  # keep default -- documented no-op on a session's
                                                                 # first-ever wander (both conditions reset fresh)

OUT = os.path.join("research", "findings", "raw", "_continuous_wander_mull", "wander_mull_derisk.json")

INDUCE_MSG = "what does the dog chase?"
REFERENTIAL_MSG = "you mentioned the dog, right?"
NEUTRAL_MSG = "Ok."
SESSION = "lb145-mull-shared"
BRAIN, RENDERER = "tiny-demo", "stub"


def _turn(client, message: str) -> dict:
    r = client.post("/api/brain-chat", json={
        "session": SESSION, "message": message, "brain": BRAIN, "renderer": RENDERER, "rich": False,
    })
    r.raise_for_status()
    return r.json()


def _mechanism_level_check() -> dict:
    """GPU-free: re-derive the exact curiosity-gain arithmetic `_apply_wander_mull` runs, to pin the numeric claim
    independent of the stochastic CA3 wander's episode-routing noise."""
    import numpy as np
    from research.runners._self_initiated_utterance_derisk import _lexicon
    from research.runners._self_initiated_spontaneous_thought_derisk import NOV_BY_NMEM, _curiosity_wants
    from webapp.continuous_engine import MULL_NOVELTY

    seed, n_mem = 42, 4
    agents, verbs, patients, _vocab = _lexicon(n_mem)
    nov_rng = np.random.default_rng(seed * 7919 + 1)
    baseline_nov = [float(v) for v in nov_rng.permutation(np.asarray(NOV_BY_NMEM[n_mem], dtype=float))]
    wants0, _ = _curiosity_wants(seed, baseline_nov)
    argmax_baseline = agents[int(np.argmax(wants0))]

    i_dog = agents.index("dog")
    boosted_nov = list(baseline_nov)
    boosted_nov[i_dog] = MULL_NOVELTY
    wants1, _ = _curiosity_wants(seed, boosted_nov)
    argmax_boosted = agents[int(np.argmax(wants1))]

    margin = float(wants1[i_dog] / max(wants0[agents.index("cat")], 1e-9))
    return {
        "agents": agents, "facts": [f"{agents[i]} {verbs[i]} {patients[i]}" for i in range(n_mem)],
        "baseline_novelties": dict(zip(agents, baseline_nov)),
        "baseline_wants_hz": dict(zip(agents, wants0)),
        "argmax_baseline": argmax_baseline,
        "boosted_novelties": dict(zip(agents, boosted_nov)),
        "boosted_wants_hz": dict(zip(agents, wants1)),
        "argmax_boosted": argmax_boosted,
        "boosted_dog_over_baseline_cat_margin": round(margin, 3),
        "MECH_flips_argmax": bool(argmax_baseline == "cat" and argmax_boosted == "dog" and margin >= 1.5),
    }


def _fresh_baseline_wander(CE, cache_key_probe: str) -> dict:
    """A FRESH, never-induced organ's own wander pick (MULL cannot have anything armed for a brand-new cache_key) --
    the ground truth 'untouched baseline' LESIONED must reproduce."""
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan
    os.environ["BRAIN_CONTINUOUS_WANDER_MULL"] = "0"
    org = SelfInitiationOrgan(seed=42)
    pre_gains, mull_rec = CE._apply_wander_mull(cache_key_probe, org)
    assert pre_gains is None and mull_rec is None, "a fresh cache_key must never have anything armed for MULL"
    out = org.speak(lesion=False)
    return {"concept": out.get("concept"), "share": out.get("share"), "gains_on": list(org.gains_on or [])}


def _run_condition(client, CE, EPmod, cache_key, *, armed: bool) -> dict:
    os.environ["BRAIN_CONTINUOUS_WANDER_MULL"] = "1" if armed else "0"
    from webapp.server import _SESSION_SELFINIT, _SESSION_MOOD, _get_selfinit_organ, _get_affect_organ

    # Reset ONLY the per-condition state (never the expensive warm ChatBrain/composer/GNW-bus): a fresh episodic
    # organ ('dog' must be freshly RE-formed + RE-recalled) and a fresh self-init organ (fresh gains_on/novelties/
    # IOR-adaptation state, so IOR's documented first-tick no-op holds identically in both conditions).
    EPmod._ORGANS.pop(cache_key, None)
    _SESSION_SELFINIT.pop(cache_key, None)
    CE.forget_session(cache_key)

    r_induce = _turn(client, INDUCE_MSG)
    r_ref = _turn(client, REFERENTIAL_MSG)
    ep = r_ref.get("episodic") or {}
    induction_ok = bool(r_ref.get("referential") and ep.get("in_memory") and r_induce.get("recalled_svo"))

    now = time.time()
    now += CE.IDLE_SEC + 1.0
    CE._LAST_REQUEST[cache_key] = now - CE.IDLE_SEC - 1.0
    n_ticked = CE.tick_idle_sessions(_SESSION_MOOD, _get_affect_organ, now=now,
                                     selfinit_getter=_get_selfinit_organ, episodic_getter=None, chat_getter=None)
    inner = CE.inner_life(cache_key)
    tick_rec = inner[-1] if inner else {}
    # IMMEDIATELY extract PLAIN values (never keep the dict reference): `inner_life()` shallow-copies the LIST but
    # not its dict elements, and `recent_wander()` (called unconditionally by the very next turn below, as part of
    # the #86 wander-drives lead) CONSUMES by mutating that SAME dict object's 'wandered' key back to None. Reading
    # `tick_rec.get(...)` lazily (e.g. at the return statement, after the follow-up turn already ran) would silently
    # read the POST-consumption value. Extracting scalars here, before the follow-up turn exists, avoids that.
    tick_wandered = tick_rec.get("wandered")
    tick_mull = dict(tick_rec["mull"]) if tick_rec.get("mull") else None
    tick_note = tick_rec.get("note")

    r_followup = _turn(client, NEUTRAL_MSG)
    wd = r_followup.get("wander_drives") or {}

    return {
        "armed": armed,
        "induce_recalled_svo": r_induce.get("recalled_svo"),
        "referential_in_memory": ep.get("in_memory"),
        "induction_ok": induction_ok,
        "n_ticked": n_ticked,
        "tick_wandered": tick_wandered,
        "tick_mull": tick_mull,
        "tick_note": tick_note,
        # THE LOAD-BEARING fields: read from the ACTUAL served HTTP JSON response (fresh objects from
        # `client.post(...).json()`, immune to the shared-dict-reference issue fixed above) -- the exact thing a
        # real user reads. `main()` strips the app's own live 20s-cadence background tick task before the
        # TestClient starts (see its comment) so THIS is driven only by the manual ticks below, not a race.
        "followup_wander_drives_concept": wd.get("concept"),
        "followup_answer": r_followup.get("answer"),
        "followup_abstained": r_followup.get("abstained"),
        "followup_recalled_svo": r_followup.get("recalled_svo"),
        "followup_verified": r_followup.get("verified"),
    }


def main() -> int:
    from sim.backend import get_backend
    _, backend_name = get_backend()

    from webapp.server import app
    from webapp import continuous_engine as CE
    import research.runners.d5_episodic_production_organ as EPmod
    from starlette.testclient import TestClient

    # STRIP the app's own live `_continuous_state_tick` startup task (webapp/server.py) BEFORE the TestClient enters
    # (which fires `on_startup` handlers). That task is a real `asyncio` loop that wakes every `IDLE_SEC` and calls
    # the SAME `tick_idle_sessions` this runner calls manually, on the SAME global session-state dicts and the SAME
    # per-session self-init organ -- a genuine same-process race during the ~55-200s a heavy cupy wander call blocks
    # (cupy releases the GIL during kernel launches, so the asyncio loop's OWN scheduled tick can interleave). This
    # confounded an earlier run of this exact test (the ARMED condition's manual tick correctly recorded 'dog', but
    # the served follow-up reply showed an unrelated 'bird' -- a second, uncontrolled tick almost certainly won the
    # race with fresh unboosted gains after MULL's own armed topic had already been drained). Removing ONLY this one
    # handler (by name) leaves every other startup behavior (orphan-run recovery, chat-brain warm) untouched; nothing
    # about the coupling itself is patched -- this isolates the MEASUREMENT, not the mechanism under test.
    app.router.on_startup = [h for h in app.router.on_startup if h.__name__ != "_continuous_state_tick"]

    out = {
        "runner": "research/runners/_continuous_wander_mull_derisk.py",
        "run_id": os.environ.get("SIM_RUN_ID", "unset"),
        "backend": backend_name,
        "device": "cuda:0" if backend_name == "cupy" else "cpu",
        "seed": 42,
        "idle_sec": CE.IDLE_SEC,
        "induce_msg": INDUCE_MSG, "referential_msg": REFERENTIAL_MSG, "neutral_msg": NEUTRAL_MSG,
        "session_reuse": "ONE warm ChatBrain reused across both conditions; only the D5-episodic organ + self-init "
                          "organ + continuous-engine per-session dicts are reset between conditions.",
        "background_tick_task_stripped": True,   # see main()'s comment above the app.router.on_startup filter
    }

    t0 = time.time()
    out["mechanism_level"] = _mechanism_level_check()
    out["mechanism_check_s"] = round(time.time() - t0, 2)

    cache_key = (SESSION, BRAIN, RENDERER)
    with TestClient(app) as client:
        tb0 = time.time()
        _turn(client, "hello")   # pay the one-time full chat-brain build on a throwaway turn
        out["chat_build_s"] = round(time.time() - tb0, 2)

        # ground-truth untouched baseline: a brand-new cache_key that MULL has never seen (probed OUTSIDE the HTTP
        # session pool -- a standalone organ, so this costs one extra heavy wander but no extra chat/composer build).
        tf0 = time.time()
        out["fresh_baseline"] = _fresh_baseline_wander(CE, ("lb145-mull-fresh-probe", BRAIN, RENDERER))
        out["fresh_baseline_wander_s"] = round(time.time() - tf0, 2)

        cond = {}
        for name, armed in (("ARMED", True), ("LESIONED", False)):
            tc0 = time.time()
            cond[name] = _run_condition(client, CE, EPmod, cache_key, armed=armed)
            print("[timing] condition %s took %.1fs" % (name, time.time() - tc0), flush=True)
    out["wall_s"] = round(time.time() - t0, 2)
    out["conditions"] = cond

    # ---- GO CHECKS ----
    mech = out["mechanism_level"]
    fresh = out["fresh_baseline"]
    armed_c, les_c = cond["ARMED"], cond["LESIONED"]

    induction_ok = bool(armed_c["induction_ok"] and les_c["induction_ok"])

    # THE LOAD-BEARING checks are built ONLY from `followup_*` fields (the ACTUAL served HTTP response the next real
    # turn reads) -- immune to both the shared-dict-reference issue fixed above AND to any race with the live
    # server's own 20s-cadence background tick thread (a known confound in this exact TestClient harness style; see
    # the 2026-08-26-continuous-drive-coupling-2 finding's own note on it). Whatever `tick_wandered`/`tick_mull`
    # show is auxiliary diagnostic detail, not gating -- the response fields are the ground truth a real user gets.
    armed_wander_shifts = bool(
        armed_c["followup_wander_drives_concept"] == "dog"
        and armed_c["followup_wander_drives_concept"] != les_c["followup_wander_drives_concept"]
        and "(I'd been mulling over dog.)" in (armed_c["followup_answer"] or "")
    )
    lesioned_matches_baseline = bool(
        les_c["followup_wander_drives_concept"] == fresh["concept"]
        and les_c["followup_wander_drives_concept"] != "dog"
    )

    keys = ["followup_abstained", "followup_recalled_svo", "followup_verified"]
    content_identical = all(
        json.dumps(armed_c[k], sort_keys=True) == json.dumps(les_c[k], sort_keys=True) for k in keys
    )

    out["PASS_MECH_flips_argmax"] = mech["MECH_flips_argmax"]
    out["PASS_induction_ok"] = induction_ok
    out["PASS_ARMED_wander_shifts"] = armed_wander_shifts
    out["PASS_LESIONED_matches_baseline"] = lesioned_matches_baseline
    out["PASS_content_identical"] = content_identical

    verdict_go = (mech["MECH_flips_argmax"] and induction_ok and armed_wander_shifts
                  and lesioned_matches_baseline and content_identical)
    out["VERDICT"] = "GO" if verdict_go else "UNDEFINED"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str), flush=True)
    return 0 if verdict_go else 1


if __name__ == "__main__":
    sys.exit(main())
