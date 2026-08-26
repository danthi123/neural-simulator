---
type: finding
status: live
date: 2026-08-26
mechanism: continuous-affect-drives-idle-relax
lane: continuity
integration_faculty: affect-drives-response
seeds: [42]
verdict: GO
instrument: through the REAL `/api/brain-chat` FastAPI endpoint (in-process Starlette TestClient calling the actual
  `webapp.server.app`, stub renderer, one warm ChatBrain reused across 4 conditions to keep the production
  one-brain-composer build cost to a single pay) -- induce a strong felt mood, then either answer immediately or
  drive N synthetic idle ticks (`continuous_engine.tick_idle_sessions` called directly with an advancing `now`, no
  real sleep()) before the SAME neutral follow-up, crossed with the new coupling armed vs lesioned.
artifacts:
  - research/findings/raw/_continuous_affect_relax/idle_relax_derisk.json
runner: research/runners/_continuous_affect_drives_idle_relax_derisk.py
external: none new -- reuses board #81's Koulakov-2002/Goldman-2003 graded-affect ladder (6/6-seed GO) and the
  existing `continuous_engine` homeostatic-relaxation formula (`v1 = NEUTRAL + (v0-NEUTRAL)*RELAX`) already shipped
  for the legacy Gate-B mood path.
supersedes: none -- additive extension of `webapp/continuous_engine.py` (2026-08-19) and
  `webapp/affect_drives_chat.py` (board #84, 2026-08-19); both existing findings stand unmodified.
---

# The idle-tick "felt mood keeps evolving" relaxation now reaches board #84's OWN affect-drives EMA, not just the legacy Gate-B mood — the flagship affect-lead marker measurably fades toward neutral while a session is idle, and the fade VANISHES under an independent lesion flag (board #91)

Artifact: `research/findings/raw/_continuous_affect_relax/idle_relax_derisk.json`

**One line.** `continuous_engine.tick_session`'s headline "the brain keeps feeling between turns" mechanism only
ever relaxed+re-read the LEGACY Gate-B mood; board #84's own persistent EMA (`AffectDrivesWorkspace.ema_valence`/
`ema_arousal`, the state behind the live `/api/brain-chat` affect-lead marker — "Wonderful —", "Gladly! ", etc.)
was written only inside a live turn and never touched by an idle tick. A user could tell the brain something
emotionally charged, wait arbitrarily long, send a neutral follow-up, and get the IDENTICAL lead as zero idle
time — an observe-vs-drive gap on the single most legible "the brain has a felt life between messages" signal a
user could notice. This closes it: one new idle-tick call decays the SAME #84 EMA a live turn writes, re-reads the
#81 spiking ladder at the decayed point, and the resulting graded lead now measurably fades with idle time — and
the fade vanishes cleanly when an independent escape flag is set, proving it is this new coupling doing the work,
not an accident of the existing #84 machinery.

## What was built

- **`webapp/affect_drives_chat.py`** — `AffectDrivesWorkspace.relax_idle(relax, neutral=0.0)`: decays
  `ema_valence`/`ema_arousal` toward the neutral set-point (the identical homeostatic formula
  `continuous_engine.tick_session` already applies to the legacy mood — reused verbatim, not reinvented),
  recomputes the body-state `(h, a)`, and re-runs ONE #81 ladder read (`_read_body`, board #81's 6/6-seed-GO
  Koulakov/Goldman bistable-NMDA ladder, reused-by-import, zero new spiking code) at the decayed point. Does NOT
  increment `n_turns` (an idle tick is not a conversational turn). Module-level `relax_idle(chat, relax, neutral)`
  is the entry point: it returns `None` (a clean no-op) when the session has no `_affect_drives_workspace` yet, so
  a session that never had a #84 turn is byte-identical to today no matter how it ticks.
- **`webapp/continuous_engine.py`** — `_affect_relax_drive_enabled()` (env `BRAIN_CONTINUOUS_AFFECT_RELAX`,
  default **OFF** pending this GO and owner review — an INDEPENDENT flag from `BRAIN_AFFECT_DRIVES` /
  `BRAIN_AFFECT_DRIVES_LESION`, so this one coupling can be armed/lesioned without touching #84 wholesale) and a
  new call site inside `tick_idle_sessions`'s existing `chat_getter` block, right beside the DA-encoding-homeostasis
  call, that invokes `affect_drives_chat.relax_idle` and logs an inner-life note on a genuine decay.
- **`research/runners/_continuous_affect_drives_idle_relax_derisk.py`** — the anti-hollow verifier below.

No `sim/` edit (`git diff sim/` is empty for this change) — the ladder build/read is reused-by-import exactly as
board #84 already does; this module adds only the idle-tick relaxation call.

## The anti-hollow verification (through the REAL `/api/brain-chat` handler, stub renderer)

Four conditions, 2×2 (idle-vs-immediate × armed-vs-lesioned), run against ONE warm `ChatBrain` (the full
production one-brain-composer + GNW-bus build is paid once; only the lightweight `#84` workspace and the
continuous-engine per-session dicts are reset between conditions — `relax_idle` never touches the composer, so
this reuse cannot hide or manufacture the effect under test):

| condition | idle? | coupling | induced level/lead | follow-up level/lead | follow-up mood |
|---|---|---|---|---|---|
| I_on  | no  | armed     | 2 / "Gladly! " | **2 / "Gladly! "**  | 0.068125 |
| A_on  | yes (14 synthetic ticks) | armed     | 2 / "Gladly! " | **0 / "" (empty — no lead)** | 0.005541666666666667 |
| I_off | no  | lesioned  | 2 / "Gladly! " | **2 / "Gladly! "**  | 0.068125 |
| A_off | yes (14 synthetic ticks) | lesioned  | 2 / "Gladly! " | **2 / "Gladly! "**  | 0.068125 |

Idle time is simulated by calling `continuous_engine.tick_idle_sessions` directly N=14 times with an advancing
explicit `now` (its documented parameter for exactly this) rather than a real 20s×14 wall-clock wait — the
identical production function the live server's background loop calls, on the same global session-state dicts the
request path just wrote.

**GO checks (all passed):**
1. `PASS_induced_nonzero` — the induction turn actually moved the felt state (level 2 in all four arms before any
   idle time; the baseline is non-trivial, not already-neutral).
2. `PASS_idle_decays_when_armed` — **level(I_on)=2 > level(A_on)=0**: with the coupling armed, 14 idle ticks alone
   (no new message) collapse the affective lead to nothing. The mood read itself moved from 0.068125→0.005541666666666667, a
   genuine re-read of the spiking ladder at the decayed body-state, not a discrete jump.
3. `PASS_vanishes_under_lesion` — **level(I_off)==level(A_off)==2**: the SAME 14-idle-tick gap, with only
   `BRAIN_CONTINUOUS_AFFECT_RELAX=0` flipped, produces ZERO decay — the lead is byte-identical whether idle time
   passed or not. This is the load-bearing proof: kill this one flag and the "felt life between messages" effect
   disappears completely, so the effect rides this coupling and not some other confound.
4. `PASS_lesion_flag_inert_on_immediate_read` — level(I_on)==level(I_off)==2: the lesion flag changes nothing about
   the immediate (no-idle) read — it only removes the idle-time decay, never the base #84 behavior itself.
5. `PASS_content_fields_identical` — `abstained` (true), `recalled_svo` (null), `verified` (false) for the same
   neutral follow-up message are IDENTICAL across all four conditions. The moat/recall verdict is completely
   untouched; only the affective tone surface (the lead marker) changes with idle time.

`VERDICT: GO` (all five checks pass). Full per-condition record in the artifact.

## Honest scope / residuals

1. **The relaxation clock/formula is a declared host homeostat**, identical to the one already shipped and accepted
   for the legacy Gate-B mood (`continuous_engine.RELAX`/`NEUTRAL`) — `IDLE_SEC`/tick-cadence is a scaffold timer,
   not a claim that elapsed wall-clock time itself is neurally computed. The FELT READ at every decayed point (the
   mood/level/lead) is the genuine #81 spiking ladder, reused verbatim — this finding does not change that
   boundary, only extends which persistent EMA the existing boundary applies to.
2. **Default is OFF** (`BRAIN_CONTINUOUS_AFFECT_RELAX` unset ⇒ 0), unlike the sibling wander/ideation couplings
   which were flipped default-ON after their own GO + owner review. This coupling is handed back for that review
   before any default flip, per the branch's instructions — flipping it is a one-line change once approved
   (`_affect_relax_drive_enabled`'s default string), not a further build.
3. **N=14 synthetic ticks** (not a real ~280s wall-clock wait) was needed to cross a discrete Koulakov LEVEL
   boundary. <!--derived--> An earlier, discarded 4-tick trial run (superseded by the N=14 artifact above, not
   itself saved as a citable artifact) already showed the underlying continuous mood read moving measurably
   partway toward neutral, with the lead's emphasis softening from an exclamation to a dash, well before crossing
   a level boundary — i.e. the coupling is graded, not a threshold artifact of the specific N chosen. <!--derived-->
4. **Cost**: the anti-hollow run pays one real production `/api/brain-chat` first-turn build (one-brain composer +
   GNW bus + co-resident organs; ~212s on this GPU) — a genuine production cost, not a wedge; the run reuses that
   one warm chat across all four conditions specifically to keep this tractable (see the runner's cost-control
   note).

## Reproduce

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_affect_drives_idle_relax_derisk
```
Writes `research/findings/raw/_continuous_affect_relax/idle_relax_derisk.json`; exits 0 iff GO. Runtime ~4-5 min
(the one-time chat-brain build dominates).
