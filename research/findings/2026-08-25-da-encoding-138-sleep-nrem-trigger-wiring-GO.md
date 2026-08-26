---
type: finding
status: contributing
date: 2026-08-25
mechanism: da-gated-encoding
lane: integration
integration_faculty: da-gated-encoding
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
instrument: research/runners/_da_encoding_138_sleep_trigger_derisk.py -- per seed, 6 scenarios through the REAL
  production call path (`continuous_engine.tick_idle_sessions` -> `consolidate_substrate_homeostasis` ->
  `da_encoding_drives_chat.apply_substrate_homeostasis` -> `OneBrainComposer.apply_homeostatic_scaling`, never a
  re-implementation): OFF/light-idle fires, ON/light-idle no-ops, ON/sleep-depth fires with a real store-synapse
  rescale, the sleep-triggered scale vector is byte-equal to the idle-tick-triggered one (mechanism identity), a
  re-tick of an already-consolidated sleeping session no-ops (compounding guard), and LESION still no-ops under the
  sleep trigger (delegation, not a bypass).
runner: research/runners/_da_encoding_138_sleep_trigger_derisk.py
external: NO-EXTERNAL-NEEDED -- this is a control-flow (WHEN the pass fires) question about code already in this
  repo; the underlying Turrigiano synaptic-scaling MATH has its own separate cupy 6-seed GO
  (2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP.md) and is not re-derived here.
artifacts:
  - research/findings/raw/_da_encoding_138_sleep_trigger/numpy_6seed.json
  - research/findings/raw/_da_encoding_wired/homeo_trigger.json (pre-existing idle-tick regression instrument,
    re-run unmodified as a no-regression check on the OFF/default path)
---
# GO (6-seed): the Turrigiano consolidation pass fires on a genuine sleep/NREM-depth event, additive/default-off, and cannot double-apply with the idle tick

Board #138. `webapp/continuous_engine.consolidate_substrate_homeostasis` runs the on-substrate Turrigiano synaptic-
scaling consolidation pass and is already production-default (2026-08-25 flip GO). Until this landing it fired on
EVERY idle tick (>= `IDLE_SEC` ~20s) whenever the store grew -- a documented stopgap, since Turrigiano scaling is
canonically a slow OFFLINE/SLEEP process, not a between-utterance one. This rung adds a second trigger, gated behind
`BRAIN_DA_ENCODING_SLEEP_TRIGGER` (unset/0 = default, byte-identical to HEAD).

## What changed (`webapp/continuous_engine.py`)

* `SLEEP_IDLE_SEC = 300.0` -- a session idle this long (minutes) is genuine sleep-depth, distinct from the light
  between-turn `IDLE_SEC` pause (~20s) that only lets the mood relax.
* `substrate_homeostasis_sleep_trigger_enabled()` -- the new flag getter.
* `_is_sleep_tick(cache_key, now)` -- True iff this session's recorded idle time (`_LAST_REQUEST`) is
  `>= SLEEP_IDLE_SEC`. A session with no recorded last-request is conservatively never "asleep" (no manufactured
  sleep state).
* `consolidate_substrate_homeostasis` gained an additive `trigger: str = "idle_tick"` parameter, stamped into the
  returned inner-life record (`"idle_tick"` | `"nrem_sleep"`) so the caller (and a test) can see which path fired.
  The default preserves every existing 2-positional-argument call site (`_da_encoding_homeo_trigger.py`,
  `_da_encoding_flip_verify.py`) unchanged.
* The `tick_idle_sessions` call site is now an **if/else on the same call**, not an addition beside it:
  `substrate_homeostasis_sleep_trigger_enabled()` FALSE -> the original unconditional call runs exactly as before
  (byte-identical); TRUE -> ONLY `_is_sleep_tick(...)`-gated calls reach the pass, tagged `"nrem_sleep"`. The
  two triggers are therefore **structurally mutually exclusive within one activation** -- there is no code path
  where both fire for the same tick. The pass's own `_LAST_HOMEO_KB` new-writes-since-last-pass guard (unchanged,
  shared across callers) is the second, independent line of defense, e.g. across an OFF->ON flag flip mid-session.

## The 6-seed verdict

Per seed (42/43/44/100/101/102), on a fresh lean `OneBrainComposer` per scenario (the same 12-word vocab / 9-fact /
Latin-square DA battery `_da_encoding_leansoak.py` validated -- reused, not re-derived), through the real production
call path:

| Check | Result (6/6 seeds) |
|---|---|
| OFF + light idle: the pass FIRES, tagged `idle_tick` (byte-identical to pre-#138 HEAD) | GO |
| ON + light idle (< `SLEEP_IDLE_SEC`): the pass does **not** fire | GO |
| ON + sleep-depth idle (>= `SLEEP_IDLE_SEC`): the pass FIRES, tagged `nrem_sleep`, weights change | GO |
| MECHANISM IDENTITY: sleep-triggered scale vector byte-equal to idle-tick-triggered (same seed/facts/gains) | GO |
| COMPOUNDING GUARD: re-ticking an already-consolidated sleeping session is a no-op, weights unchanged | GO |
| LESION DELEGATION: `BRAIN_DA_ENCODING_LESION=1` still no-ops under the sleep trigger | GO |

15 checks/seed x 6 seeds, all GO, `research/findings/raw/_da_encoding_138_sleep_trigger/numpy_6seed.json`.

**The anti-cheat that matters most here is MECHANISM IDENTITY.** #138 could have silently introduced a second,
untested computation path under the new flag. It does not: `scales_off == scales_on_sleep` to float equality on
every seed -- the sleep trigger calls the exact same `apply_substrate_homeostasis` -> `apply_homeostatic_scaling`
already carrying the 2026-08-25 6-seed cupy GO. This landing changes **when** the pass fires, never **what** it
computes.

**Store-synapse rescale is a real weight change, not a metadata event**: the `reaches` checks compare a weight
fingerprint (sum of `|w|` over every stored synapse) before/after, and it moves on both the OFF and the ON-sleep
arm (e.g. seed 42: 764.16 -> 763.44) -- the values differ per-engram (documented direction: weak/low-DA engrams
scale up toward the set-point, strong/high-DA engrams partially scale down, tonic engrams stay ~1.0), consistent
with the already-validated Turrigiano rule.

**No-regression on the pre-existing OFF/default path**: the pre-existing, independently-authored regression
instrument `research/runners/_da_encoding_homeo_trigger.py` (FIRES / NO-OP / RE-FIRES / LESION / `=0` disarms) was
re-run unmodified against this change and still reports GO -- the default (flag unset) idle-tick behaviour this
runner exercises is untouched.

## Scoping note: numpy 6-seed is the decisive check here; a cupy production-backend confirmation is queued

The claim under test is **control flow** -- a Python `if`/`else` in `continuous_engine.py` with zero cupy/numpy
array operations of its own. It cannot differ by backend by construction, and the Turrigiano scaling MATH it calls
into already carries its own separate cupy 6-seed GO. The numpy-backend 6-seed run above (`SIM_BACKEND=numpy`,
~1 min wall-clock, no GPU) is therefore treated as decisive for the trigger-wiring question this rung asks.

A `SIM_BACKEND=cupy` run of the identical runner (same `--out` flag, a sibling `cupy_6seed.json` file not yet
produced as of this writing) was additionally queued via `tools/gpu_queue.sh add` for production-backend parity
due-diligence, not required to decide this question, non-blocking: the GPU queue held 10-12 jobs ahead of it at
landing time (the 2026-08-26 four-day autonomous harvest), so it was not waited on. **Residual, not deferred**:
if that queued run reports anything other than 6/6 GO when it lands, it supersedes this finding's scoping call and
the mechanism-identity claim above must be re-examined.

## What this is NOT

Per `docs/TERMS.md`: this is **wired (default-off)**, not on-by-default and not integrated/closed -- the new
trigger only reaches production traffic when an owner sets `BRAIN_DA_ENCODING_SLEEP_TRIGGER=1`; the default
config an owner gets today is unchanged (the pre-existing every-idle-tick trigger stays the production path). No
claim of "consolidation" beyond the code condition already earned by the underlying 2026-08-25 flip finding (this
rung changes its trigger, not its replay/reactivation status).

## Next rung

Owner product decision, not a wall: whether/when to flip `BRAIN_DA_ENCODING_SLEEP_TRIGGER` default-on (retargeting
production from the light-idle stopgap to genuine sleep-depth), and whether `SLEEP_IDLE_SEC=300` is the right
depth threshold for real conversational sessions (a host-timed scaffold, like `IDLE_SEC` itself -- open to
recalibration against real session-idle distributions, not a biological measurement). A more faithful next
mechanism (tracked separately, #107/#64): key the trigger off an actual on-substrate SWR-replay/NREM event
(the Ecker AdEx CA3 forward-replay GO, 2026-08-20) rather than a host wall-clock threshold -- this rung's
`SLEEP_IDLE_SEC` gate is explicitly a step toward that, not a substitute for it.
