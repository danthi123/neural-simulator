---
type: finding
status: undefined
claim_check: measured
date: 2026-09-23
lane: A · Affect — D5 "feel" over the OPEN reply (roadmap §8)
seeds: [42, 43, 44, 100, 101, 102]
mechanism: webapp/affect_conditioned_mouth.py (BRAIN_OPEN_ENDED_AFFECT_CONDITIONED=prompt|resid, default-OFF) conditions
  the Qwen articulation mouth's GENERATION on the spiking affect organ's held valence. 'prompt' = graded,
  dead-zone-free MOOD line in the existing build_prompt channel; 'resid' = c*K*u added to Qwen's layer-12 residual
  stream (contrastive activation addition), u = the mouth's own WARRINER-contrast axis.
artifacts:
  - research/findings/raw/_affect_conditioned_mouth/amend1_prompt/affect_conditioned_mouth_prompt_verdict.json
  - research/findings/raw/_affect_conditioned_mouth/amend1_resid/affect_conditioned_mouth_resid_verdict.json
---

# Affect-conditioned Qwen mouth — why the brain's valence never reached the mouth, and the staged 6-seed de-risk (2026-09-23)

Status: **UNDEFINED on both modes** (2026-09-23, AMENDMENT 2 — scored). The diagnosis below is measured on all 6
seeds. The 6-seed AMENDMENT-1 tone runs (both `prompt` and `resid`, 42 arms each) completed and were scored with
the literal preregistered command. **Neither mode reaches a GO or a NO-GO: an instrument precondition fails on
both, which by this gate's own design means UNDEFINED, not a negative.** See "AMENDMENT 2: the scored result"
below. **AMENDMENT 1 (fix round, 2026-09-23) changed the instrument before any tone result was read; see the
amendment log at the end.**

## What was measured: the live-valence magnitude problem has two parts

<!--derived from research/findings/raw/_affect_conditioned_mouth/magnitude_audit.json + research/findings/raw/_affect_conditioned_mouth/calibration/calibration_fold.json -->
**1. The mouth's own conditioning channel had a dead zone that swallowed every live read.** The production
`_mood_phrase` renders "even and steady" for |valence| < 0.25. The live valence is `clip(4 * differential)`.
It read +0.141..+0.162 after positive priming and -0.180..-0.073 after negative priming. So on **12/12** pos/neg  <!--derived-->
arms of the committed NO-GO data, `_mood_phrase` would render "even and steady". **Correction (fix round):** this
is a COUNTERFACTUAL computed from the WKV-mouth NO-GO valences. Qwen was NOT the mouth in those runs, so it does not
explain why those NO-GOs failed. It shows that at those live valences the Qwen MOOD channel would have been
dead-zoned: the brain's affect would reach the Qwen prompt only as a two-decimal number.

<!--derived from research/findings/raw/_affect_conditioned_mouth/calibration/calibration_fold.json (per-seed files organ_calib_s<seed>.json alongside) -->
**2. The organ's full scale is symmetric, but its NEGATIVE side is weak at the production operating point.**
(Correction, fix round: the build said "the organ is not weak"; its own data contradict that on 4/6 seeds, below.)
On the positive side it runs at about half its range after one priming message. The affect organ is
690 neurons. It was swept over appraisal -1..+1 on each seed (`--calibrate-organ`, numpy). Results:
- Full scale is symmetric. The held differential at |appraisal| = 1 is +0.075..+0.086 and -0.076..-0.093.  <!--derived-->
  VALENCE_FS = 4 x the mean full-scale differential = 0.309..0.341 across seeds.
- The ladder has a threshold dead zone. At |appraisal| <= 0.25 no rung ignites and the differential is 0.0.
- One strongly affective message moves the session-mood EMA to appraisal ±0.475. That is where the live read
  sits: 0.449..0.525 of full scale for positive priming, 0.237..0.528 for negative priming.
- The negative-side weakness is a MID-RANGE effect, not a ceiling. At appraisal -0.475, 4/6 seeds  <!--derived-->
  (43, 100, 101, 102) hold only -0.018..-0.024. That is below the organ's own neutral tolerance of 0.03.  <!--derived-->
  At -0.75 and -1.0 the same seeds read -0.058..-0.082.  <!--derived-->

The calibration reproduces the production read on the positive side: the sweep point at +0.475 matches the NO-GO
arms' priming differential to 4 decimals for all 6 seeds. On the negative side it does NOT match exactly, and the
cause is not read noise (the organ read is deterministic: ctrl_neg == neg exactly in the NO-GO arms). The
production negative priming appraises to **-0.4725**, while the sweep samples -0.475.  <!--derived from research/findings/raw/_affect_conditioned_mouth/magnitude_audit.json -->

So the earlier description "~0.16 clipped" was wrong: nothing was clipped, and on the positive side 0.16 is about
half the organ's range. What would hold affect back on a Qwen mouth is the production MOOD dead zone, plus a
single-message EMA that holds the organ in its mid-range, plus (negative side, 4/6 seeds) an organ read below its
own neutral tolerance.

**Terminology correction (fix round).** The build called `c = valence / VALENCE_FS` "readout gain normalization
(divisive normalization, Carandini & Heeger 2012), the companion process the NO-GOs lacked". That overclaims. It is
ONE STATIC, PER-SEED CALIBRATED GAIN CONSTANT, equivalent to raising the steering gain to K_eff = K / VALENCE_FS
(about 12 on raw valence). Divisive normalization divides by pooled activity AS IT CHANGES; this divides by a
number measured once. It is itself the "constant substituted for a process" anti-pattern. The real companion
process, a gain-adapting reader population normalized by the organ's live pooled activity, is NOT built.

## The method (built, default-OFF)

The conditioning signal is `c = clip(valence / VALENCE_FS, -1, 1)`. `valence` is exactly what brain_chat
already hands the mouth, the organ's held differential. VALENCE_FS is that seed's measured full-scale value.
One constant is used for both signs, so the organ's mid-range sign asymmetry is kept, not normalized away. Live
c is about +0.45..+0.53 (pos) and -0.24..-0.53 (neg).

- **prompt**: the MOOD line is graded monotonically in |c| (faintly / slightly / moderately / clearly /
  intensely) and has no dead zone. c == 0 gives the production neutral wording. No descriptor word is in the
  scoring lexicon ('upbeat', a scored word, became 'buoyant'), so the ruler cannot score an echo of the
  conditioning text.
- **resid** (inference-time contrastive activation addition, CAA. Correction, fix round: this is NOT the
  Affect-LM / VAD-conditioning method, which conditions the model at TRAINING time. CAA shares only the locus, the
  generation representation rather than the decode): the prompt stays exactly as production builds it. During
  generation a forward hook adds `c*K*u` to
  the output of decoder layer 12. u is the difference of mean layer-12 activations of Qwen on 6 positive vs 6
  negative sentences built only from WARRINER words; the selftest enforces zero overlap with the scoring
  lexicon. K = 4.0 is fixed. When c == 0 no hook is registered.

Brain-based boundary: the conditioning SIGNAL comes from the spiking organ, whose input is the host appraisal
lexicon (`appraise_text`, an upstream named shortcut). The division by VALENCE_FS is host code, a **named
shortcut** on the scaffold boundary. The Qwen mouth, its prompt and its internal axis u are the
owner-ratified articulation scaffold (2026-09-19), not brain computation.

Off path, checked in data: `tests/test_affect_conditioned_mouth.py` uses a fake generator. With the flag unset,
answer_turn hands the generator exactly build_prompt's (system, user) and never imports the module.

## PREREGISTERED GO gate for the staged runs (written before any conditioned-mouth tone result existed)

This is the NO-GOs' gate verbatim, via `score_and_gate(mouth="qwen")`:
(1) DIRECTIONAL. On each seed pos_gap > +delta AND neg_gap < -delta, on 6/6 seeds, or 5/6 with the 6th null.
delta is the pooled std of the lesion arm's per-prompt tone, with the independent 356-word lexicon.
(2) ATTRIBUTION. |ctrl_pos - ctrl_neg| < delta on every seed. The shuffled-valence control injects ±0.16
random-sign valence.
(3) Content identity holds and the moat holds.
(4) Fluency: salad <= 0.16.
(5) lesion == lesion_rep, byte-identical.

Plus instrument preconditions. Any one unmet makes the verdict UNDEFINED, not a negative:
(L) the conditioning lever moved: c != 0 on pos/neg/ctrl rows, the hook fired in resid mode, and c == 0 on lesion
rows;
(O) each arm's priming differential equals the committed NO-GO arm's value, so all three verdicts see the same
brain state.

AMENDMENT 1 adds (see the log): (L) also requires ONE steering-axis hash across every conditioned row of every
seed; (3a-reply) content identity over the GENERATED reply; (R) the 6 seeds replicate the mouth; (5b) pos ==
pos_rep byte-identical. The base (3a) compares pre-generation retrieval and cannot fail for this method, so here it
is an integrity smoke, kept only for comparability.

**Claim scope (AMENDMENT 1).** A GO would show only that CAA or graded-prompt conditioning of the Qwen mouth,
signed and scaled by the spiking organ's held differential, shifts the open reply's tone past the lesion null on
the independent ruler. CAA is known to shift tone, so that part is expected. The gate does NOT separate the organ's
contribution from the host appraisal that drives it, or from the steering itself: the lesion removes organ output
and steering together, and (2) tests only priming-text leakage. Measuring the organ's own contribution needs an
extra arm, organ lesioned with a host constant c = sign(appraisal) x const. That arm is named here as the next
method and is not built in this round.

Literal scoring command:
`.venv/bin/python -m research.runners._lbf_affect_conditioned_mouth_derisk --score-only --mode <prompt|resid>`

## Staged runs

<!--derived from research/findings/raw/_affect_conditioned_mouth/smoke/smoke_s42_pos_resid.json -->
Smoke (seed 42, pos arm, resid, 2 prompts, CPU Qwen float32): the whole path runs end to end. Qwen wrote both
replies (generator=qwen). The hook registered and fired (c = 0.448, 96 and 67 hook calls). Replies were fluent
(salad 0.073 / 0.086). Footprint: maxrss 9.5 GB, about 52 s per CPU generation. That is too slow for the pool.
The pool also cannot host this run: its venv has no torch/transformers, it has no Qwen weights or
data/corpus, and provisioning excludes research/findings/raw/, which holds the calibration and the NO-GO
reference arms that (O) needs. So the run goes to the local box via gpu_queue, with Qwen on CUDA.

One gpu_queue line per (mode, seed). Qwen runs on CUDA; the brain runs numpy, the same organ numerics as the
NO-GOs. Arms run one after another inside each line, so only one brain is live at a time. Results go to
`research/findings/raw/_affect_conditioned_mouth/amend1_{prompt,resid}/arm_s<seed>_<arm>.json`, and the verdict to
`affect_conditioned_mouth_<mode>_verdict.json` in the same directory. The controller now defaults XDG_RUNTIME_DIR
and refuses to run without the memory cap. **Both queue lines completed 2026-09-23** (all 6 seeds x 7 arms, both
modes); see "AMENDMENT 2" below for the scored result.

## AMENDMENT 2: the scored result (2026-09-23) — UNDEFINED on both modes

Both AMENDMENT-1 runs are complete: 6 seeds x 7 arms (pos, neg, lesion, lesion_rep, ctrl_pos, ctrl_neg, pos_rep) x
2 modes = 84 arm files, all present. Scored with the literal preregistered command,
`.venv/bin/python -m research.runners._lbf_affect_conditioned_mouth_derisk --score-only --mode <prompt|resid>`,
on a clean checkout of `origin/main` (this doc's own runner, unmodified). `--selftest` passes first, including
every planted-failure check (the gate can fail in its failing direction). Verdict artifacts:
`research/findings/raw/_affect_conditioned_mouth/amend1_{prompt,resid}/affect_conditioned_mouth_{prompt,resid}_verdict.json`.

**Both modes: UNDEFINED, not a negative.** The preregistered instrument precondition **(3a-reply)** — content
identity over the GENERATED known reply, fact-word recall vs. the lesion reply >= 0.75 on every conditioned
arm/seed — is UNMET on both modes. Prompt mode: 0/6 seeds pass; 3/6 seeds (44, 101, 102) are unmeasurable because
the lesion reply carries fewer than the 2 fact words the check needs, and on the 3 measurable seeds recall is 0.0 on
9 of 15 conditioned pairs. Resid mode: 3/6 seeds (42, 44, 102) are unmeasurable the same way, and on the 3 measurable
seeds recall is 0.0 on 14 of 15 conditioned pairs. This
precondition exists exactly to prevent a tone-shift reading from being reported when the conditioned generation
has stopped reproducing the same underlying facts as its lesion baseline — unmet here means the verdict is
UNDEFINED by the gate's own design (`tools/gates/verdict_preconditions`), not that the method failed.

**resid mode also fails (4) fluency:** `max_salad_frac = 1.0` (worst: seed 43, `ctrl_neg`, "What do you think
about the future" — a fully degenerate/repetitive reply), against the <=0.16 precondition. This is a second,
independent reason resid mode cannot be read as a tone result: at K=4.0 on layer 12, the residual-stream steering
sometimes collapses generation into salad rather than shifting its tone.

**What the raw (non-gating) numbers say, honestly, since UNDEFINED still permits describing what was measured:**
- **prompt mode** does not read as a tone effect even setting the instrument failure aside: the preregistered
  directional gate is NO-GO (0/6 seeds pass the raw-std band on both signs; on the SEM-secondary the positive side
  is 3/6 correct-sign, and the NEGATIVE side is 1/6 correct and 4/6 WRONG sign — negative conditioning pushed the
  tone the opposite way on 4 of 6 seeds), and **only 14.8% of the treatment-vs-control tone gap is attributable to the
  real organ signal** — 85.2% of it is also present in the shuffled-valence control, i.e. it is not specifically
  the brain's mood driving it.
- **resid mode** reads as a much larger, more consistently-signed effect (4/6 seeds pos-correct, 2/6 neg-correct
  on the raw-std gate; 6/6 both signs on the non-gating SEM-secondary; 94.8% of the tone gap is attributable to
  the real signal, not the control) — but this is the same run whose fluency precondition failed, and the
  incoherent output sits mostly in the CONTROL arms: of the 18 resid tone rows above salad 0.16, 13 are
  ctrl_pos/ctrl_neg, 5 are neg and 0 are pos. Salad lowering the control gap would inflate the 94.8% attribution,
  so the apparent effect and the fluency collapse are confounded and cannot be disentangled from these arms.

**Provenance caveat.** The copied arm artifacts' `.prov.json` sidecars record `git_sha: 008cd8893` (merged into
`origin/main` as the second parent of `5e9a7955b`) with `git_dirty: true` in the source worktree at run time —
this is explained by provenance bookkeeping: the source worktree is still at 008cd8893 and its only tracked diff
is research/findings/raw/_provenance/runs.jsonl (plus untracked raw outputs). An edit-then-revert during the runs
cannot be excluded from git alone. The (O) organ-read-equality precondition (every arm's priming differential matches
the committed NO-GO runs' value to 1e-12) PASSED on every seed/arm in both modes, which is strong indirect
evidence the brain-side path was unchanged; it does not by itself rule out an uncommitted change to the
mouth-side code. Flagged here rather than silently assumed clean.

**Next-method note (not built this round):** the (3a-reply) precondition caught real content divergence, not a
crude check being too strict. Example: prompt mode s43, known prompt 'Tell me about frank_lincoln_wright' — the
lesion arm describes Frank Lincoln Wright, while both pos AND ctrl_pos describe Franklin D. Roosevelt. Any
conditioning, the shuffled-valence control included, changes the sampled decode path and therefore WHAT is said
about the topic; a richer content check would likely fail the same way. So the method (conditioning the mouth's
generation on affect) changes content along with tone. The next method must shape tone while holding the reply's
content fixed (for example, choosing among content-equivalent candidate replies by the brain's affect read, or
conditioning only the style channel), and resid mode additionally needs a fluency-constrained K. THE LAW applies:
this is a verdict on the generation-conditioning method, not on affect shaping the reply.

## Honest residuals
- **(AMENDMENT 2) The tone result exists and is UNDEFINED on both modes**, not a negative and not a positive: an
  instrument precondition failed on both. This doc makes no claim that affect is load-bearing over the open
  reply — the data do not support a claim in either direction yet.
- The normalization is host arithmetic, a named shortcut. The brain-based version would be a gain-adapting
  reader population.
- (Superseded by AMENDMENT 1.) Before the amendment, Qwen decoded with the server's fixed seed 42 in every arm,
  so the 6 seeds replicated only the organ. The decode seed is now the brain seed, and (R) checks the replication.
- The salad<=0.16 precondition was designed for WKV salad. First-person Qwen prose repeats "I", so short
  replies could trip it. If it does, the verdict is UNDEFINED and the instrument, not the method, needs a fix.
- The organ's mid-range V- weakness (4/6 seeds below tolerance at -0.475) may still cap neg_gap. If so, the  <!--derived-->
  surpass is on the organ/appraisal side (the session-mood EMA protocol or the V- rung-2 ignition), not the
  mouth.

Honesty boundary: this is a functional read-out only. Reply tone tracking the organ's valence is a coupling.
Nothing here claims felt experience.

## AMENDMENT LOG

**AMENDMENT 1: 2026-09-23, about 12:10 EDT, fix round after the adversarial review.** When I wrote it I had seen
NO conditioned-mouth tone result. Pre-amendment resid arms existed for seeds 42 and 43 (all 6 arms) and 44 (4
arms) in the build worktree. I read only their instrument fields: CUDA path, maxrss, generator, c, hook_calls and
the known-row fact list. I read no tone score, no reply text and no verdict. Those arms are SUPERSEDED: the job was
stopped, the 9 queued lines were removed, and the arms are never scored, because the instrument changed under them.
The changes:
1. The decode seed is now the brain seed. Before, the server passed no seed, so every arm used 42. New
   precondition (R): 6 distinct lesion realizations, with decode_seed == seed.
2. New (3a-reply): fact-word recall of the generated known reply vs the lesion reply is >= 0.75 for pos, neg,
   ctrl_pos, ctrl_neg and pos_rep on every seed. Fewer than 2 lesion fact words counts as unmet. There is only one
   known prompt per seed, so this check is thin; it can still fail.
3. `affect_axis` now runs under a fixed noise seed, and the SPK.gen state is restored afterwards. The axis hash is
   traced, and (L) requires one hash everywhere. There is a new pos_rep arm and a new precondition (5b).
4. Memory cap: the controller defaults XDG_RUNTIME_DIR and refuses to run uncapped.
5. Output goes to new `amend1_<mode>` directories. The calibration fold is renamed to `calibration_fold.json`,
   because `*_summary.json` is gitignored and the cited file had never been committed.
6. The claim scope and terminology are corrected as described above. The base gate and all 9 of its
   preconditions are unchanged, for comparability.

**AMENDMENT 2: 2026-09-23, scoring pass (worktree `research/d5-affect-conditioned-verdict`).** Both AMENDMENT-1
runs (prompt, resid; 6 seeds x 7 arms each) had completed in a separate build worktree. This pass copied the arm
JSON + `.prov.json` sidecars into a clean `origin/main` checkout (no runner code changed) and ran the literal
preregistered scoring command for both modes. Result: **UNDEFINED on both modes** — see "AMENDMENT 2: the scored
result" above for the full breakdown (failed preconditions, the raw non-gating numbers, and the provenance
caveat on the source worktree's `git_dirty: true` at run time). No new runs were staged by this pass: the 6-seed,
both-mode coverage the preregistration calls for is complete; what is missing is not a run but a better
instrument (thin content-identity check; resid mode's untuned steering magnitude).
