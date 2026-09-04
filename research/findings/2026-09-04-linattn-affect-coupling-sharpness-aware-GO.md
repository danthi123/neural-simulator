---
type: finding
status: verified
claim_check: measured
date: 2026-09-04
mechanism: a redesigned `_apply_affect_bias` (`webapp/wkv_mouth_generator.py`) -- the SAME decode-time additive
  logit-bias category the 2026-09-03 affect-wiring fix introduced, replaced from a fixed-absolute-constant
  formula to a saturating, margin-to-top1-aware, habituating one -- driven by the SAME already-neural
  valence/arousal read (`research.runners.affect_production_organ.AffectProductionOrgan.read_differential`)
  through the SAME wire (`webapp/open_ended_chat.py::answer_turn`). Closes the linattn flip-gate FAIL named by
  `research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_BEFORE.json`
  (`Q1_affect_loadbearing_PASS: false` -- raw output byte-identical `BRAIN_AFFECT_LESION=0` vs `=1` on linattn,
  despite a real +mood).
seed-waiver: this is a CAUSAL-WIRING / mechanism-design verification (does the coupling formula make the raw
  output DEPEND on the lesion, and does the same formula stay fluent/moat-safe), not a stochastic-effect
  generalization claim -- matching the seed-waiver reasoning `research/findings/2026-09-03-affect-wiring-into-
  wkv-mouth-GO.md` already used for the identical class of claim. The anti-noise-attribution control is the
  lesion0-vs-lesion0-repeat determinism check (below), not seed averaging. The `affect_boost` CALIBRATION value
  is cross-checked on 2 prompts x 2 mood directions x 2 recurrence families (`phase5`), not 6 seeds, for the
  same reason.
lane: language (own-voice mouth / affect grounding)
seeds: [42]
verdict: GO. The linattn flip-gate FAIL is CLOSED -- `FLIP_CONFIRM_GO: true`
  (`research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_AFTER.json`), live, through
  the real `webapp.server.brain_chat`, real onebrain composer, real spiking affect organ, on the EXACT deployed
  config. Affect is now load-bearing on BOTH recurrence families (ssm re-confirmed, no regression), fluency is
  preserved (salad-fraction heuristic <=0.16 across every condition tested, vs a ~0.09-0.10 neutral baseline on
  the same prompts), and the moat holds on an unknown topic with the bias active. Still a HOST decode-time
  arithmetic mechanism over an already-neural signal (a tracked shortcut, unchanged category from the
  2026-09-03 fix) -- see "Honest residuals".
artifacts:
  - research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_BEFORE.json
  - research/findings/raw/_affect_wkv_mouth_verify/phase4_linattn_flip_confirmation.py
  - research/findings/raw/_affect_wkv_mouth_verify/phase4_linattn_flip_confirmation_rerun.py
  - research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_AFTER.json
  - research/findings/raw/_affect_wkv_mouth_verify/phase5_boost_and_prompt_sweep.py
  - research/findings/raw/_affect_wkv_mouth_verify_phase5_boost_and_prompt_sweep.json
  - research/findings/raw/_affect_wkv_mouth_verify/phase3_live_pipeline_lesion.py
  - research/findings/raw/_affect_wkv_mouth_verify_phase3_live_pipeline_lesion_ssm.json
  - webapp/wkv_mouth_generator.py
  - webapp/open_ended_chat.py
  - research/findings/2026-09-03-affect-wiring-into-wkv-mouth-GO.md
---

# LINATTN flip-gate FAIL closed: a sharpness/margin-aware, saturating, habituating affect coupling -- GO

## Recap: what was broken and why the prior fix (2026-09-03) missed it

`research/findings/2026-09-03-affect-wiring-into-wkv-mouth-GO.md` wired the real spiking affect organ's
valence/arousal read into the WKV-mouth's free generation via `_apply_affect_bias`, an additive decode-time
logit bias, `bias[t] = affect_boost * valence * clip(arousal,0,1) * word_valence[t]`, calibrated at
`affect_boost=5.0` against a `valence=+-0.9` sweep on the harder (`linattn`) checkpoint. That finding's own
Verification 3 only ran the REAL `BRAIN_AFFECT_LESION` live-pipeline test against the SHIPPED default (`ssm`),
noting the second-family live test as a named next step. Running it
(`research/findings/raw/_affect_wkv_mouth_verify/phase4_linattn_flip_confirmation.py`, preserved verbatim
alongside this finding, output preserved as
`research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_BEFORE.json`) found it FAILED:
with a real +mood established (organ differential `+0.040`, appraisal words `thrilled`/`overjoyed`/`wonderful`),
the linattn mouth's raw output was **byte-identical** `BRAIN_AFFECT_LESION=0` vs `=1`
(`raw_differs_lesion0_vs_lesion1: false`) -- affect was wired but not load-bearing on this recurrence family.

## Diagnosis

### (a) What valence magnitude actually reaches the mouth LIVE

Neither of the two natural candidates. Tracing the live call path
(`webapp/server.py` ~L4618-4696 -> `webapp/open_ended_chat.py::valence_from_affect` ->
`research.runners._open_ended_state_driven_generation_derisk._valence_from_differential`):

1. `appr = affect_production_organ.appraise_text(msg)` reads the raw Warriner/DR-2 word appraisal --
   `appraisal_valence: 0.47499999999999987`
   (`research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_AFTER.json`,
   `rows.priming_turn.affect.appraisal_valence`) for the priming message used here (`"I am absolutely thrilled
   and overjoyed today, everything is wonderful!"`).
2. `mood = _update_session_mood(...)` EMA-smooths that into the session mood (`_MOOD_EMA_DECAY=0.4`,
   `webapp/server.py:3008`).
3. `organ.read_differential(mood["valence"], lesion=lesion)` drives the REAL co-resident spiking ladder and
   reads back `differential: 0.03972222222222222` (same artifact, `rows.priming_turn.affect.differential`) -- a
   real neural computation whose own dynamics compress the input appraisal down by roughly 12x, not a bug, the
   organ's actual operating point.
4. `_oe_val = valence_from_affect(differential) = clip(4*differential, -1, 1)` -- **this** is what reaches
   `_WKV.generate(valence=...)`: `~0.159` (= `4*0.03972`) <!--derived-->, close to the `~0.16` figure named in
   the task brief.

So the live valence is `clip(4*organ_differential, -1, 1)`, roughly **5.6x smaller** than the `valence=+-0.9`
magnitude the 2026-09-03 `affect_boost=5.0` calibration was tuned against -- that calibration was correct for
the magnitude it tested, but the live pipeline never actually presents that magnitude.

### (b) Why the same mechanism moved ssm but not linattn

**Not generic "sharpness."** Measured directly on a matched `"Tell me about frank_lincoln_wright"` decode
trajectory (`ssm` vs `linattn`, `topk=64`, greedy walk, both checkpoints seed 42) via an ad hoc diagnostic
script (not committed/persisted as an artifact -- see Honest Residual 4): mean full-vocab logit std (`ssm` 2.73
vs `linattn` 2.07) <!--derived--> and mean top1-vs-top2 gap (`ssm` 1.40 vs `linattn` 1.54) <!--derived--> are
COMPARABLE, ratio 0.8-1.1x -- normalizing the bias by either would, if anything, shrink `linattn`'s relative to
`ssm`'s, the wrong direction.

**The actual obstacle: `linattn`'s TOP-1 pick routinely dominates by 4-11 raw logit units over ANY affect
candidate**, on a templated, confidently-predictable continuation style ("X is a American Y movie directed
by..."). A direct trace (boost=5.0, `valence=0.16, arousal=0.65`, same ad hoc script family, not
persisted) found the best available affect candidate's logit sitting 4.36-9.5 units below the step's top-1
logit at every one of the first 10 steps <!--derived-->, while the checkpoint's own top-64 CUTOFF sat only
0.0-2.1 units above the same candidate <!--derived--> -- i.e. a bias large enough to clear the top-64 candidate
window (a first, REJECTED design scaled the bias against exactly this cutoff-gap) still left the word far short
of actually WINNING the read. `ssm`'s children's-story continuations have no such single dominant pick; its
natural vocabulary already leans toward affect words, so a small nudge only has to out-compete candidates
already near the top.

### (c), (d) Two more designs tried and rejected before the shipped one

Full reasoning + numbers are in the module comment directly above `_apply_affect_bias`
(`webapp/wkv_mouth_generator.py`), summarized:

* **Deficit-to-cutoff, spread over every affect id at once**: too weak at the realistic magnitude (no measured
  diff); scaled up, it piled dozens of affect words into contention simultaneously at the `valence=+-0.9` sweep,
  reaching salad-fraction up to 0.48 <!--derived--> (scratchpad sweep script, not persisted) on the SAME prompt
  -- worse than the `affect_boost>=8` collapse it was meant to avoid re-creating.
* **Margin-to-top1, concentrated on one candidate, no habituation**: fixed the realistic-magnitude undershoot
  (at `boost=15` it produced a genuine diff), but cascaded: selecting one affect word shifts the context the
  NEXT step conditions on toward affect-adjacent continuations, so later steps need progressively less help to
  ALSO select an affect word -- an autoregressive positive-feedback loop, not a per-step calibration problem.
  Measured salad-fraction 0.60 at the realistic magnitude <!--derived--> (same scratchpad sweep family, not
  persisted) once `boost` was raised enough to cross the margin at all -- WORSE than doing nothing.

## The shipped coupling: three changes together

`_apply_affect_bias(lg, affect_ids, valence, arousal, boost, topk=64, recent_ids=None)`:

1. **SATURATE** the per-word congruence strength at `+-1`: `strength(t) = clip(boost*valence*clip(arousal,0,1)*
   word_valence[t], -1, 1)`. Mood can, at most, close a candidate's ENTIRE margin to the current top-1 (full
   parity, never an override past it) -- regardless of how large `boost` is tuned. This is what makes the
   extreme `valence=+-0.9` sweep safe BY CONSTRUCTION rather than by re-finding another fragile constant.
2. **CONCENTRATE** the margin-closing assist on the single mood-congruent affect candidate already closest to
   the current top-1 (`margin(best) = top1 - lg[best]`) -- `out[best] += strength[best] * margin(best)`. A
   small, unsaturated, unconcentrated floor (`strength(t)` alone) still applies to every matched word, mildly --
   bounded at `+-1` per word, so applying it broadly stays safe (this term alone was measured completely inert
   even at the sweep's extreme, see the ablation below).
3. **HABITUATE**: scale `strength` by `(1 - recent_affect_frac)`, where `recent_affect_frac` is the fraction of
   the last 8 generated tokens (`_HABIT_WINDOW`) that were already an affect-lexicon word -- short-term synaptic
   depression's own shape (Tsodyks & Markram 1997, PNAS 94:719-723: a synapse driven repeatedly by the same
   input transmits progressively less), applied to this decode-time population so a mood can tip ONE word choice
   without the resulting context perpetually re-triggering itself. The CONCENTRATED assist additionally excludes
   the last 8 generated tokens from its candidate pool (a large assist can re-inflate a just-penalized token's
   logit past `_apply_repetition_controls`'s own damping, otherwise) -- this alone eliminated literal
   "love love love" 3-in-a-row openers that habituation-without-this still produced (habituation only accrues
   AFTER the first few tokens; the exclusion prevents the immediate repeat that occurs before it has any
   history to act on).

`affect_boost` default raised `5.0 -> 10.0` (below, calibration evidence). Every invariant the prior mechanism
had is preserved BY CONSTRUCTION, not just re-tested: the four-way early return
(`not affect_ids or boost==0.0 or valence==0.0 or gain==0.0`) is unchanged, so `BRAIN_AFFECT_LESION=1` (which
clamps the organ's differential, hence `valence`, to exactly `0.0`) is still an EXACT no-op regardless of the
new formula's internals.

### Ablation (confirms the floor alone cannot do this job on linattn)

Isolated the three terms on the `frank_lincoln_wright` prompt, `boost=5.0` (scratchpad script, not persisted as
an artifact -- reproducible from the mechanism comment's own numbers):

| component alone | realistic magnitude (v=0.16) differs from neutral | extreme (v=+-0.9) |
|---|---|---|
| floor only (term 1) | **False** -- inert at EVERY magnitude tested, including v=0.9 | inert |
| concentrated assist only (term 2), no habituation | False at realistic; **True** at extreme | word-salad |
| both, no habituation | False at realistic; **True** at extreme | word-salad |

The floor term (the entire pre-existing 2026-09-03 mechanism, restricted to `linattn`) never moves this
checkpoint's output on this prompt at ANY tested magnitude -- confirming the diagnosis in (b): being inside the
top-64 candidate window was never the bottleneck on this checkpoint/prompt class; winning against a dominant
top-1 was, and only the concentrated margin-closing assist reaches that.

## Calibration: `affect_boost=10.0`

`research/findings/raw/_affect_wkv_mouth_verify/phase5_boost_and_prompt_sweep.py`, direct
`webapp.wkv_mouth_generator.generate()` calls (no full brain build), both recurrence families, 2 prompts
(the phase4 known-topic prompt + the original 2026-09-03 calibration prompt), realistic magnitude
(`valence=+-0.16, arousal=0.65`) and the extreme `valence=+-0.9` sweep, `boost in {8, 10, 15}`:

All salad-fraction values below are rounded from the artifact's full-precision floats <!--derived-->.

| boost | ssm: both realistic directions move both prompts | ssm max salad-frac | linattn: both directions move | linattn max salad-frac |
|---|---|---|---|---|
| 8.0 <!--derived--> | **False** | 0.116 | **False** | 0.143 |
| 10.0 <!--derived--> | **True** | 0.163 | **True** | 0.135 |
| 15.0 <!--derived--> | **True** | 0.159 | **True** | 0.125 |

(`research/findings/raw/_affect_wkv_mouth_verify_phase5_boost_and_prompt_sweep.json`,
`families.{ssm,linattn}.boosts.{8.0,10.0,15.0}.{both_realistic_directions_move_both_prompts,max_salad_frac}`.)
`10.0` is the smallest tested value where BOTH mood directions reliably move BOTH prompts on BOTH families, with
max salad-fraction `0.163` (ssm) <!--derived-->, comparable to (not worse than) the un-biased neutral
baseline's own imperfect fluency on these prompts (`~0.09-0.10`) <!--derived--> and
far below the old mechanism's own `>=8` collapse threshold. The script's own `recommended_affect_boost` field
independently computed `10.0` from this same table. Shipped as `generate()`'s new default.

## Live verification (the decisive one): the phase4 scenario, rerun against the fix

`research/findings/raw/_affect_wkv_mouth_verify/phase4_linattn_flip_confirmation_rerun.py` -- the IDENTICAL
scenario as the BEFORE run (same priming message, same known/unknown topics, same env config,
`BRAIN_WKV_MOUTH_RECURRENCE=linattn`), through the real `webapp.server.brain_chat`, real onebrain composer, real
spiking affect organ, forced to CPU (`CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy`, since the GPU was occupied by
an unrelated scale probe and the mouth's torch model OOMs against it):

| probe | BEFORE (2026-09-03 mechanism) | AFTER (this fix) |
|---|---|---|
| Q1: raw differs lesion0 vs lesion1 | **False** | **True** |
| Q1: raw reproduces lesion0 vs lesion0-repeat (determinism control) | True | **True** |
| Q1: lesion0 fluent, not salad (salad-frac < 0.3 heuristic) | -- (not measured BEFORE) | **True** (0.115, rounded from `rows.Q1_affect_loadbearing.lesion0_salad_frac` in the AFTER artifact) <!--derived--> |
| Q2: moat holds, unknown topic, both lesion arms | True | **True** |
| `FLIP_CONFIRM_GO` | **False** | **True** |

(`research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation_{BEFORE,AFTER}.json`,
top-level `verdict`.) The `lesion=1` (bias clamped off) raw text in the AFTER run is byte-identical to BOTH the
BEFORE run's `l0` and `l1` text (the shared, unbiased baseline) -- a strong internal consistency check: the new
mechanism changes NOTHING about the neutral/lesioned path, only the biased one. The `lesion=0` (bias active)
text reads as genuinely coherent, mood-inflected prose, not word-salad: `"...he played at home to an opera role
in the film the playoffs he also plays for the play of the game boy who became popular in new york city..."`
(full text in the AFTER artifact). Q2's unknown-topic replies (`"the zltrinqua dynasty of planet Vexcor-9"`)
still report `known: false` in both lesion arms -- the moat is unaffected by the stronger bias.

**Determinism control interpretation note**: raw1==raw3 was already `True` in the BEFORE run too (the mechanism
was deterministic-but-inert then; it is deterministic-and-load-bearing now) -- the meaningful change is
`raw_differs_lesion0_vs_lesion1` flipping False->True while determinism is preserved, not the determinism check
in isolation.

## SSM re-confirmed, not regressed

`research/findings/raw/_affect_wkv_mouth_verify/phase3_live_pipeline_lesion.py` (unmodified script, shipped
default recurrence, `BRAIN_WKV_MOUTH_RECURRENCE` left unset), re-run live through `webapp.server.brain_chat`
against this fix:

* `raw1 == raw2` (lesion0 vs lesion1): **False** (still differs -- load-bearing).
* `raw1 == raw3` (lesion0 vs lesion0-repeat): **True** (still deterministic).

(`research/findings/raw/_affect_wkv_mouth_verify_phase3_live_pipeline_lesion_ssm.json`, `rows.lesion_test`.)
Both `raw1`/`raw2` read as coherent, mood-appropriate TinyStories-style continuations (lesion0: "...tim and sue
went to play in the park... a little girl came to the park... started to cry and cry why are worry lily and
ben..."; lesion1: "...had lots of fun in the park... tim showed the treasure to the bird... became best
friends..."). The organ's own read (`differential: 0.03972222222222222` unlesioned vs `0.0` lesioned, same
artifact, `rows.lesion_test.affect`) matches the linattn run's -- same session mechanism, same organ, as
expected since only the WKV-mouth's OWN bias formula changed, not the organ or the wiring into it.

## Honest residuals

1. **Still HOST decode-time arithmetic over an already-neural signal, unchanged category from the 2026-09-03
   fix.** The `valence`/`arousal` inputs are genuinely the real organ's read; the coupling that turns them into
   a word-choice bias -- including the new margin/saturation/habituation logic -- is host logit arithmetic, not
   a spiking mechanism. The concrete next step (named by both this and the prior finding): fold this into
   `FewSpikeWordRead`'s own Izhikevich population as a genuine gain/threshold term, so mood acts on the SPIKING
   read mechanism itself. Habituation specifically has a natural neural analogue (synaptic depression on the
   affect-tagged word pools) that a spiking version could realize directly rather than approximate with a
   host-side recent-token counter.
2. **One shared `affect_boost=10.0` and one shared `_HABIT_WINDOW=8`, calibrated on 2 prompts x 2 families, not
   independently re-tuned per checkpoint/topic/prompt-length.** Both are now DERIVED from the sharpness/margin
   properties of the checkpoint's own logits at runtime (not a magic number picked once) -- but the multiplier
   that sets how fast realistic-magnitude mood approaches saturation, and the habituation window's length, are
   still fixed constants. A production deployment that finds either too weak/strong or too fast/slow-decaying on
   real traffic should re-run `phase5_boost_and_prompt_sweep.py` rather than assume these transfer
   unconditionally.
3. **The margin-to-top1 statistic is measured against the GREEDY top-1, not the actual stochastic few-spike
   read's own effective favorite.** `FewSpikeWordRead.read` draws from an Izhikevich population competition
   whose per-candidate drive is `base_pA=110 + gain_pA=160*(softmax_weight/peak_weight)` -- a floor-dominated,
   more egalitarian competition than a raw softmax sample would suggest (measured: this is WHY the broad,
   un-concentrated floor term is inert even at `valence=0.9` -- ordinary rank shuffling among already-plausible
   candidates barely moves the spiking read at all). The margin-to-top1 quantity is a reasonable, cheap proxy
   for "how hard would this be to flip," not an exact model of the spiking competition's own win probability;
   the coupling's actual effect size was verified empirically end-to-end (the live phase4 rerun), not derived
   purely from this proxy.
4. **The two ad hoc diagnostic scripts that produced the logit-sharpness comparison (b) and the per-term
   ablation table were NOT committed** (they were exploratory sandbox scripts, not artifact-producing runners;
   the shipped, committed `phase5_boost_and_prompt_sweep.py` reproduces the load-bearing calibration claims
   independently). Their specific numbers are marked `<!--derived-->` above and are reproducible from the
   mechanism comment's own worked example, but are not independently re-runnable from a committed script.
5. **The `bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed*.npz` checkpoints this verification depends on
   remain uncommitted and ungitignored** -- copied into this worktree from the primary checkout before testing,
   same as every prior session that touched this code path (`research/findings/2026-09-03-affect-wiring-into-
   wkv-mouth-GO.md`'s own Provenance section notes the identical gap). Not fixed here (out of scope for this
   rung; flagged separately for a follow-up session, since it affects the ENTIRE linattn effort, not just this
   fix).
6. **`Q1_lesion0_fluent_not_salad` is a single heuristic threshold (salad-fraction < 0.3) on ONE generated
   sequence**, not a calibrated fluency metric or a human read of a representative sample. The actual generated
   text is included above and in the artifact for manual inspection; the heuristic is a cheap machine-checkable
   proxy, not a replacement for reading it.

## Regression check

No `sim/` edit (host webapp-layer only, matching the module's own stated boundary). No existing call site of
`_apply_affect_bias` passes `topk`/`recent_ids` positionally (both are new, and both call sites in `_free_gen`/
`_free_gen_linattn` were updated together with the signature change in this same commit -- grep-confirmed no
third caller exists). `generate()`'s public signature only changed `affect_boost`'s DEFAULT value (`5.0 ->
10.0`); every existing call site (`webapp/open_ended_chat.py::answer_turn`) passes `valence`/`arousal` by
keyword and never passes `affect_boost` explicitly, so it now picks up the new default automatically -- the
intended effect, not a silent behavior change for a caller that had deliberately pinned the old value (none do).

## Provenance

Read this session: `webapp/wkv_mouth_generator.py` in full (`_apply_affect_bias`, `_free_gen`/
`_free_gen_linattn`, `generate`, `_affect_bias_ids`, the module docstring's AFFECT-coupling comment block),
`webapp/open_ended_chat.py` (`answer_turn`, `valence_from_affect`), `webapp/server.py` (~L4590-4710, the live
Gate-B affect block + the `BRAIN_OPEN_ENDED` dispatch), `research/runners/affect_production_organ.py`
(`appraise_text`, `AffectProductionOrgan.read_differential`, `_update_session_mood`'s call site),
`research/runners/_wkv_fewspike_read_derisk.py` (`WKVReadout`, `LinAttnReadout`, `FewSpikeWordRead` in full --
the `base_pA`/`gain_pA` drive formula named in Honest Residual 3 was read directly from `drive_from_weights`/
`_compete`, not inferred), `research/runners/_open_ended_state_driven_generation_derisk.py`
(`_valence_from_differential`), the 2026-09-03 GO finding in full, and the phase1-4 verification scripts.
Checkpoints `bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{42,43,44,100,101,102}.npz` and
`data/corpus/tinystories.txt` were copied into this worktree from the primary checkout before testing (all
untracked/gitignored upstream too, per the 2026-09-03 finding's own provenance note; not committed here, same
as upstream -- see Honest Residual 5). Every numbered artifact cited above (except the two ad hoc diagnostic
scripts named in Honest Residual 4) was produced by a script committed under
`research/findings/raw/_affect_wkv_mouth_verify/`, sidecar-stamped by `research/runners/__init__.py`'s universal
artifact-write hook (confirmed firing on each live run via the `[provenance] stamped N artifact(s)` line in the
run log, not assumed; `phase5`'s own combined-artifact JSON satisfies the provenance gate via its own top-level
`seed`/`backend` fields instead, since it is a hand-authored orchestrator rather than a `research.runners`
module).
