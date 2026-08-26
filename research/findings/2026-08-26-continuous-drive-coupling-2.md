---
type: finding
status: live
date: 2026-08-26
mechanism: continuous-da-mode-drives-idle-relax
lane: continuity
seeds: [42]
verdict: GO
instrument: through the REAL `/api/brain-chat` FastAPI endpoint (in-process Starlette TestClient calling the actual
  `webapp.server.app`, stub renderer, one warm ChatBrain reused across 4 conditions to keep the production
  one-brain-composer build cost to a single pay) -- induce a strong DA-mode ENGAGEMENT (focus/arousal), then either
  answer immediately or drive N synthetic idle ticks (`continuous_engine.tick_idle_sessions` called directly with an
  advancing `now`, no real sleep()) before the SAME engagement-neutral follow-up, crossed with the new coupling
  armed vs lesioned.
artifacts:
  - research/findings/raw/_continuous_da_relax/da_idle_relax_derisk.json
runner: research/runners/_continuous_da_drives_idle_relax_derisk.py
external: none new -- reuses board #76's `_neuromod_spiking_da_mode_derisk` (6/6-seed GO, self-produced SNc->DA
  read) and the existing `continuous_engine` homeostatic-relaxation formula (`v1 = NEUTRAL + (v0-NEUTRAL)*RELAX`)
  already shipped for the legacy Gate-B mood path and extended to board #84 by board #91
  (2026-08-26-continuous-drive-coupling.md).
supersedes: none -- additive extension of `webapp/continuous_engine.py` (2026-08-19) and
  `webapp/da_mode_drives_chat.py` (board #76/#79, 2026-08-19); both existing findings stand unmodified.
---

# The idle-tick "keeps evolving between turns" relaxation now reaches the DA-mode ENGAGEMENT->forthcomingness coupling too — a SECOND axis (energy, not valence) fades toward rest while a session is idle, and the fade VANISHES under an independent lesion flag (board #92)

Artifact: `research/findings/raw/_continuous_da_relax/da_idle_relax_derisk.json`

**One line.** Board #91 (2026-08-26) closed the observe-vs-drive gap between `continuous_engine`'s idle tick and
board #84's affect EMA (valence/warmth). The SAME gap existed on a SECOND, independent axis: board #76/#79's
DA-mode ENGAGEMENT coupling (`webapp.da_mode_drives_chat`, the flagship default-ON forthcomingness suffix — " —
worth going further here.", "there's plenty more to dig into here!") persists `ema_engagement` across turns but was
never touched by an idle tick. Telling the brain something highly engaging, waiting idle, then sending an
engagement-neutral follow-up produced the IDENTICAL engagement suffix as zero idle time. This closes it: one new
idle-tick call decays the SAME `ema_engagement` a live turn writes, re-reads the #76 spiking SNc->DA nucleus at the
decayed afferent, and the resulting mode (rest/neutral/focus/arousal) — hence the suffix — now measurably fades
with idle time, and the fade vanishes cleanly under an independent escape flag.

## What was built

- **`webapp/da_mode_drives_chat.py`** — `DaModeDrivesWorkspace.relax_idle(relax, neutral=0.0)`: decays
  `ema_engagement` toward the neutral set-point (the identical homeostatic formula `continuous_engine.tick_session`
  and board #91's `relax_idle` already apply — reused verbatim, not reinvented), maps the decayed EMA to the SNc
  reward/context afferent, and re-runs ONE #76 neural SNc->DA read (`_read_da_level`, board #76's 6/6-seed-GO
  self-produced-DA mechanism, reused-by-import, zero new spiking code) at the decayed point. Does NOT increment
  `n_turns` (an idle tick is not a conversational turn). Module-level `relax_idle(chat, relax, neutral)` is the
  entry point: it returns `None` (a clean no-op) when the session has no `_da_drives_workspace` yet, so a session
  that never had a DA-drives turn is byte-identical to today no matter how it ticks.
- **`webapp/continuous_engine.py`** — `_da_relax_drive_enabled()` (env `BRAIN_CONTINUOUS_DA_RELAX`, **default ON**
  — an INDEPENDENT flag from `BRAIN_DA_DRIVES` / `BRAIN_DA_DRIVES_LESION`, so this one coupling can be
  armed/lesioned without touching the DA-mode faculty wholesale) and a new call site inside `tick_idle_sessions`'s
  existing `chat_getter` block, right beside board #91's affect-relax call, that invokes
  `da_mode_drives_chat.relax_idle` and logs an inner-life note on a genuine decay.
- **`research/runners/_continuous_da_drives_idle_relax_derisk.py`** — the anti-hollow verifier below.

No `sim/` edit (`git diff sim/` is empty for this change) — the SNc->DA build/read is reused-by-import exactly as
board #76/#79 already does; this module adds only the idle-tick relaxation call.

## Test-design pitfall found and fixed (kept honest, not smoothed over)

The FIRST version of this de-risk copied board #91's neutral follow-up message verbatim ("Okay, noted.") and
returned `VERDICT: UNDEFINED` (`PASS_idle_decays_when_armed: false` — the mode never dropped after 10-14 idle
ticks). This was NOT a hollow coupling; it was a test-design bug specific to this axis: unlike board #84's affect
appraisal (a fixed VALENCE lexicon — "okay"/"noted" score zero sentiment hits, triggering the HOLD branch), the
DA-mode ENGAGEMENT read is driven by novelty+richness of ANY non-stopword content token >= 3 letters
(`da_mode_drives_chat.engagement_of`). `_content_tokens("Okay, noted.")` returns `['okay', 'noted']` — both novel
words — so that "neutral" follow-up itself RE-ENGAGED `ema_engagement` on arrival, masking the idle decay that had
just happened. Verified directly (`_content_tokens`): `"Ok."` returns `[]` (the token "ok" is 2 letters, below
`_MIN_CONTENT_LEN=3`) — a genuinely engagement-neutral probe. Switching `NEUTRAL_MSG` to `"Ok."` and re-running
produced the clean GO below. <!--derived--> This is recorded because the failure mode is a real lesson for any
future between-turn-axis de-risk: "neutral" is coupling-specific, not a universal string, and must be verified
against the SPECIFIC signal function (`_content_tokens`/`engagement_of` here, the appraisal lexicon for #84) before
it is trusted as a control probe. <!--derived-->

## The anti-hollow verification (through the REAL `/api/brain-chat` handler, stub renderer)

Four conditions, 2×2 (idle-vs-immediate × armed-vs-lesioned), run against ONE warm `ChatBrain` (the full production
one-brain-composer + GNW-bus build is paid once; only the DA-mode workspace + continuous-engine per-session dicts
are reset between conditions — `relax_idle` never touches the composer, so this reuse cannot hide or manufacture
the effect under test):

| condition | idle? | coupling | induced mode/suffix | follow-up mode/suffix | follow-up DA level |
|---|---|---|---|---|---|
| I_on  | no  | armed     | focus / " — worth going further here." | **focus / " — worth going further here."** | 0.637045 |
| A_on  | yes (10 synthetic ticks) | armed     | focus / " — worth going further here." | **rest / "" (empty — no suffix)** | 0.088318 |
| I_off | no  | lesioned  | focus / " — worth going further here." | **focus / " — worth going further here."**  | 0.896505 |
| A_off | yes (10 synthetic ticks) | lesioned  | focus / " — worth going further here." | **focus / " — worth going further here."** | 0.896505 |

Idle time is simulated by calling `continuous_engine.tick_idle_sessions` directly N=10 times with an advancing
explicit `now` (its documented parameter for exactly this) rather than a real 20s×10 wall-clock wait — the
identical production function the live server's background loop calls, on the same global session-state dicts the
request path just wrote to. The live server's OWN real 20s-cadence background thread also ran throughout this
~458s test (`BRAIN_CONTINUOUS=1` arms it unconditionally) — I_on's follow-up DA level (0.637044745440438) shows
some decay below the induced 0.8965045558416256 even with zero manual ticks, consistent with a few automatic
real-time ticks landing during
the ~132s chat build and turn overhead. This does not confound the verdict: I_off and A_off (the lesioned pair) are
IDENTICAL to full float precision (0.8965045558416256 in both), proving that when `BRAIN_CONTINUOUS_DA_RELAX=0`,
NEITHER the manual synthetic ticks NOR the server's own automatic background ticks touch `ema_engagement` — the
lesion flag gates the coupling regardless of which caller invokes `tick_idle_sessions`.

**GO checks (all passed):**
1. `PASS_induced_nonzero` — the induction turn actually moved the mode into FOCUS (a non-empty suffix) in all four
   arms before any idle time; the baseline is non-trivial, not already-rest.
2. `PASS_idle_decays_when_armed` — **mode_rank(I_on)=2 (focus) > mode_rank(A_on)=0 (rest)**: with the coupling
   armed, 10 idle ticks alone (no new message) collapse the engagement suffix to nothing. The DA level itself moved
   from 0.8965045558416256→0.0883180262557138, a genuine re-read of the spiking SNc->DA nucleus at the decayed
   afferent, not a discrete jump.
3. `PASS_vanishes_under_lesion` — **mode(I_off)==mode(A_off)=="focus"**, DA level identical to 16 significant
   figures: the SAME 10-idle-tick gap, with only `BRAIN_CONTINUOUS_DA_RELAX=0` flipped, produces ZERO decay — the
   suffix is byte-identical whether idle time passed or not. This is the load-bearing proof: kill this one flag and
   the "engagement fades between messages" effect disappears completely, so the effect rides this coupling and not
   some other confound (including the real background thread, which is ALSO gated by the same flag).
4. `PASS_lesion_flag_inert_on_immediate_read` — mode(I_on)==mode(I_off)=="focus": the lesion flag changes nothing
   about the induced state itself — it only removes the idle-time decay, never the base DA-mode coupling.
5. `PASS_content_fields_identical` — `abstained` (true), `recalled_svo` (null), `verified` (false) for the same
   engagement-neutral follow-up message are IDENTICAL across all four conditions. The moat/recall verdict is
   completely untouched; only the engagement/forthcomingness suffix changes with idle time.

`VERDICT: GO` (all five checks pass). Full per-condition record in the artifact.

## Honest scope / residuals

1. **The relaxation clock/formula is a declared host homeostat**, identical to the one already shipped and accepted
   for the legacy Gate-B mood and board #91's affect EMA (`continuous_engine.RELAX`/`NEUTRAL`) — `IDLE_SEC`/tick
   cadence is a scaffold timer, not a claim that elapsed wall-clock time itself is neurally computed. The FELT READ
   at every decayed point (the DA level/mode/suffix) is the genuine #76 spiking SNc->DA read, reused verbatim — this
   finding does not change that boundary, only extends which persistent EMA the existing boundary applies to.
2. **Default is ON** (`BRAIN_CONTINUOUS_DA_RELAX` unset ⇒ armed), unlike board #91 which shipped default-OFF
   pending owner review. This coupling was built default-ON per this arc's explicit instruction (mirrors the #85/
   #86 pattern of shipping default-on directly after a clean GO); `BRAIN_CONTINUOUS_DA_RELAX=0` is the
   byte-identical escape and is also the anti-hollow lesion arm above, so the escape hatch is exercised by this
   same verification.
3. **N=10 synthetic ticks** (not a real ~200s wall-clock wait) crossed the discrete rest/neutral/focus/arousal bin
   boundary cleanly (focus → rest, a two-bin drop, not a borderline single-bin wobble). A back-of-envelope check
   (manual-ticks-only, illustrative, NOT the measured value): induced EMA 0.6 times 0.85 to the 10th power is about
   0.118, comfortably under the rest threshold. <!--derived--> The artifact's actual A_on follow-up EMA is
   0.08534505428170326 (lower still, consistent with the real background thread's additional automatic ticks noted
   above).
4. **The follow-up message ("Ok.") had to be picked for THIS axis's neutrality**, not reused from board #91's
   probe — see the test-design section above. This is a residual worth flagging for whoever de-risks the next
   between-turn axis: verify the neutral probe against the coupling's own signal function before trusting it.
5. **Cost**: the anti-hollow run pays one real production `/api/brain-chat` first-turn build (one-brain composer +
   GNW bus + co-resident organs; ~132s on this GPU) plus ~458s total wall time — heavier than board #91's ~272s,
   because the DA-mode read runs 320 substrate steps (`WARMUP=200 + SETTLE=120`) per call and this run's window
   also overlapped with the server's own live 20s-cadence background thread (itself doing real work throughout,
   see the table note above) — a genuine production cost, not a wedge; two earlier attempts (480s and a killed
   concurrent duplicate) are the reason N_IDLE_TICKS was reduced from board #91's 14 to 10 here.

## Reproduce

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._continuous_da_drives_idle_relax_derisk
```
Writes `research/findings/raw/_continuous_da_relax/da_idle_relax_derisk.json`; exits 0 iff GO. Runtime ~7-8 min
(the one-time chat-brain build + 4 conditions, including real background-thread interference).
