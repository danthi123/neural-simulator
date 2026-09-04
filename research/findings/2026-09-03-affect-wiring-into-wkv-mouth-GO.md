---
type: finding
status: verified
claim_check: measured
date: 2026-09-03
mechanism: a mood-congruent, decode-time additive logit bias inside `webapp/wkv_mouth_generator.py`'s
  `_free_gen`/`_free_gen_linattn` free-generation loops (`_apply_affect_bias`), driven by the REAL spiking affect
  organ's live valence/arousal read (`research.runners.affect_production_organ.AffectProductionOrgan.
  read_differential`), wired through `webapp/open_ended_chat.py::answer_turn`'s pre-existing (but, before this
  fix, unused-by-the-WKV-branch) `valence`/`arousal` parameters. Closes the affect-hollow gap named by
  `research/findings/2026-09-03-linattn-mouth-live-brain-grounded-honest-verification-PARTIAL-affect-gap.md` (ii-c).
seed-waiver: this is a CAUSAL-WIRING verification (does toggling BRAIN_AFFECT_LESION / a valence argument change
  a deterministic function's output), not a stochastic-effect generalization claim -- the project's 6-seed
  policy targets ruling out a favorable-seed artifact in a LEARNED/statistical effect, which does not apply the
  same way to "is this code path connected." The analogous anti-noise-attribution control used here is the
  lesion0-vs-lesion0-repeat determinism check (Verification 3), not seed averaging.
lane: language (own-voice mouth / affect grounding)
seeds: [42]
verdict: GO on the primary claim -- affect is now load-bearing on the WKV mouth's free generation, for BOTH
  recurrence families (`ssm` shipped-default and `linattn`), measured three ways (an isolated direct-`generate()`
  vary/kill-switch sweep, an isolated `answer_turn`-level valence sweep, and the REAL `BRAIN_AFFECT_LESION`
  live-pipeline test through `webapp.server.brain_chat` with a genuine onebrain composer + spiking affect organ).
  PARTIAL on completeness -- this is a HOST decode-time logit-bias mechanism (a tracked shortcut, not a neural/
  spiking coupling), one shared boost constant was empirically calibrated rather than independently tuned per
  checkpoint, and arousal-alone sensitivity + prose-level legibility of the mood shift are measurably weaker on
  the `linattn`/BPE checkpoint than on `ssm`/TinyStories at that constant. See "Honest residuals".
artifacts:
  - research/findings/2026-09-03-linattn-mouth-live-brain-grounded-honest-verification-PARTIAL-affect-gap.md
  - webapp/wkv_mouth_generator.py
  - webapp/open_ended_chat.py
  - research/runners/affect_production_organ.py
  - research/findings/raw/_affect_wkv_mouth_verify/phase1_direct_generate_vary_lesion.py
  - research/findings/raw/_affect_wkv_mouth_verify_phase1_direct_ssm.json
  - research/findings/raw/_affect_wkv_mouth_verify_phase1_direct_linattn.json
  - research/findings/raw/_affect_wkv_mouth_verify/phase2_answer_turn_valence_isolation.py
  - research/findings/raw/_affect_wkv_mouth_verify_phase2_answer_turn_ssm.json
  - research/findings/raw/_affect_wkv_mouth_verify_phase2_answer_turn_linattn.json
  - research/findings/raw/_affect_wkv_mouth_verify/phase3_live_pipeline_lesion.py
  - research/findings/raw/_affect_wkv_mouth_verify_phase3_live_pipeline_lesion_ssm.json
  - tests/test_wkv_invocab_scope_leadin_fix.py
  - tests/test_wkv_mouth_learned_head_path.py
  - tests/test_wkv_mouth_bpe_decode_wiring.py
  - tests/test_linattn_readout_parity.py
---

# Wiring the real affect organ into the WKV/SSM own-voice mouth's free generation -- GO (primary), PARTIAL (completeness)

## Diagnosis: NEVER-IMPLEMENTED, not a dropped wire

The cited verification (`research/findings/2026-09-03-linattn-mouth-live-brain-grounded-honest-verification-
PARTIAL-affect-gap.md`, property ii-c) measured, via two independent probes (an isolated valence sweep and a
live `BRAIN_AFFECT_LESION` pipeline test), that valence/arousal has ZERO effect on the WKV-mouth family's output.
Reading the source before touching anything confirms WHY, precisely:

* `webapp/open_ended_chat.py::answer_turn` (pre-fix, line ~566) already builds `state = StateContext(...,
  valence=float(valence), arousal=float(arousal), ...)` and `system, user = build_prompt(state)` -- these DO
  reach the Qwen path (`gen.generate(system, user, ...)`) and the gen-time-veto path
  (`generate_with_generation_time_veto(gen, chat, topic, seed, system, user, ...)`), both of which consume the
  affect-conditioned prompt text.
* The WKV-mouth branch (pre-fix, line ~597) called `_WKV.generate(msg, seed=seed, max_new_tokens=max_new_tokens,
  repetition_penalty=1.3, no_repeat_ngram_size=3, facts=ground_facts, sentence_facts=sentence_facts)` --
  `valence`/`arousal` are simply never passed as arguments.
* `webapp/wkv_mouth_generator.py::generate()` (pre-fix) had NO `valence`/`arousal` parameter at all, and neither
  did its two driving loops, `_free_gen`/`_free_gen_linattn`.

This is **NEVER-IMPLEMENTED**, not INTENDED-BUT-BROKEN: there was no dropped wire to reconnect, because the
parameter did not exist on the callee. The design doc's own Sec 1 table claim ("affect ... still conditions the
prompt/state") was true for the Qwen/gen-time-veto paths and false for the WKV-mouth path -- exactly what the
cited finding's "Corrections to the design doc" section already states. The exact site of the gap is the WKV-
mouth branch call in `answer_turn` (now `webapp/open_ended_chat.py` lines ~597-611) plus the two driving loops
in `webapp/wkv_mouth_generator.py` (`_free_gen`, `_free_gen_linattn`).

## The coupling mechanism

**Brain-based input, host-arithmetic coupling (a tracked shortcut).** The `valence`/`arousal` floats
`answer_turn` receives are themselves genuinely neural: they originate from
`research.runners.affect_production_organ.AffectProductionOrgan.read_differential`, a real co-resident spiking
ladder read (`rate(aff_pos_readout) - rate(aff_neg_readout)` through the `affect_out` transmission gate), mapped
through `_valence_from_differential` (`webapp/server.py`'s live Gate-B block, unchanged by this fix). What was
missing was purely the WIRING from that already-neural signal to this one generator; the fix does not touch how
the signal itself is produced.

The mechanism added (`webapp/wkv_mouth_generator.py`, new `_affect_bias_ids`/`_apply_affect_bias`/
`wkv_mouth_affect_enabled`, ~100 new lines placed immediately before `_free_gen`):

1. **A per-checkpoint affect lexicon** (`_affect_bias_ids(seed)`, cached): every word in the checkpoint's OWN
   vocabulary that is a strongly affect-bearing word in `research.runners.affect_production_organ`'s EXISTING
   salience-gated lexicon (the Warriner-norm gate + the DR-2 learned per-word value when
   `affect_production_organ.dr2_enabled()`) gets a signed valence in `[-1,1]`. This REUSES the exact appraisal
   artifact the shipped Gate-B affect coupling already uses for the strict/rich path -- not a fresh host
   sentiment formula invented for this module.
2. **An additive decode-time logit bias** (`_apply_affect_bias`), applied in the SAME decode-control category as
   the pre-existing `_apply_fact_boost`/`_apply_repetition_controls` (full-vocab logits, before the top-k cut;
   the genuine few-spike Izhikevich soft-WTA `reader.read(p)` still makes the actual selection):
   `bias[t] = affect_boost * valence * clip(arousal, 0, 1) * word_valence[t]` -- positive (favored) when the
   turn's mood and the word's own valence AGREE in sign (mood-congruent production, the direction Bower 1981's
   mood-congruent recall/production effect describes), with arousal acting as a pure GAIN term (amplifies an
   existing directional signal; supplies no direction of its own -- the same shape as LC-noradrenergic
   arousal-dependent gain modulation, Aston-Jones & Cohen 2005).
3. **An exact no-op at `valence == 0.0`** -- the parameter's own default, AND exactly what
   `AffectProductionOrgan.read_differential(..., lesion=True)` clamps the organ's differential (hence the mapped
   valence) to. This is what makes `BRAIN_AFFECT_LESION=1` a genuine lesion of this specific coupling, not merely
   a dampener.
4. **A BPE word-boundary normalization**, discovered empirically before it could become a silent gap: the
   `linattn` checkpoint's BPE vocabulary spells a whole word that survived merging intact as `"happy</w>"`,
   `"angry</w>"`, `"good</w>"`, `"bad</w>"`, `"love</w>"` (confirmed directly against the shipped checkpoint --
   see "What I checked before trusting the fix" below), not `"happy"`/`"angry"` -- the exact BPE-vs-word-level
   mismatch class `fact_grounding_ids`'s own docstring already names for the fact-boost lever. A naive lookup
   would have silently matched zero of them. `_affect_bias_ids` strips a trailing `"</w>"` before matching
   against the (whole-word) lexicon; a no-op for the word-level `ssm` checkpoint, so one lookup path serves both
   recurrence families.
5. **An independent kill switch**, `BRAIN_WKV_MOUTH_AFFECT` (default-ON -- `wkv_mouth_affect_enabled()`), for
   rollback/diagnosis, mirroring the sibling organ's own `affect_enabled()` convention.

**The wire-in itself** (`webapp/open_ended_chat.py`, the WKV-mouth branch): `_WKV.generate(...)` now additionally
passes `valence=float(valence), arousal=float(arousal)` -- the SAME `valence`/`arousal` `answer_turn` already
had. Since `generate()`'s new defaults are `0.0`/`0.0` (an exact no-op), this is purely additive.

**Scope boundary, deliberate:** `render_fact_sentence`'s fact-clause path (the `sentence_facts` short-circuit) is
NOT touched. A known-topic reply answered by that closed-class template stays tone-neutral by construction,
matching Gate-B's own stated honesty floor ("affect ... NEVER enters the certainty band"). Only `_free_gen`/
`_free_gen_linattn`'s free generation is affect-sensitive.

## What I checked before trusting the fix (the calibration that almost made this a silent no-op)

Before running any end-to-end test, I measured `_affect_bias_ids(42)` against BOTH checkpoints directly:

* `ssm` (word-level, V=1000): **63** affect-tagged vocabulary ids (46 positive, 17 negative) --
  `research/findings/raw/_affect_wkv_mouth_verify_phase1_direct_ssm.json` (`affect_ids_count`, `affect_ids_pos`,
  `affect_ids_neg`).
* `linattn` (BPE, V=8001): **57** affect-tagged ids (34 positive, 23 negative) -- same fields in
  `research/findings/raw/_affect_wkv_mouth_verify_phase1_direct_linattn.json`, only reached AFTER adding the
  `</w>`-stripping normalization in item 4 above (a first pass, unshipped, found literally `"happy</w>"`,
  `"angry</w>"`, `"good</w>"`, `"bad</w>"`, `"love</w>"` present verbatim in the checkpoint's own vocabulary,
  confirming the mismatch empirically rather than assuming it away).

I then swept `affect_boost` on the harder (`linattn`) checkpoint with a deliberately topic-empty adversarial
prompt (`"Tell me about it."`, seed 42, the production repetition guard applied
`repetition_penalty=1.3, no_repeat_ngram_size=3`, matching what `answer_turn` always passes):

* `affect_boost=4.0`: `valence=-0.9` and `valence=+0.9` produced **byte-identical** text (the affect-tagged
  tokens sit ranked ~700-5000 of 8001 for this prompt's natural continuation -- far outside the `topk=64` cut;
  a boost of 4 logits is not enough to lift them in).
* `affect_boost=8.0` and above: a real, dramatic, mood-congruent split appeared (negative valence:
  `"...pain pain pain painpain pain dead pain pain kill pain pain..."`; positive valence:
  `"...love love love you love love good love love..."`) -- but degenerated into repetitive word-salad, the
  same SHAPE of failure (not the same severity) as the pre-existing `fact_boost=6.0` NO-GO already named for
  this checkpoint family.
* `affect_boost=5.0` (the value shipped): the largest value tested that stayed genuinely coherent while still
  differing -- e.g. `valence=-0.9` -> `"...he served as president of the senate and lost his seat at the"`
  vs `valence=+0.9` -> `"...he served as president of the senate and became deputy leader of the"` on the SAME
  adversarial prompt. Shipped as `generate()`'s new `affect_boost` default; documented in its docstring with
  the exact sweep values above so a future re-tune has the evidence, not just a number.

This calibration is itself evidence against a hollow fix: had I shipped the naive lookup (item 4) or the
too-small default (4.0), the wiring would have been technically present but empirically INERT on the `linattn`
checkpoint for a large class of realistic (topic-poor) prompts -- a second, quieter way to fail the same
anti-hollow bar the original finding measured.

## Verification 1 -- isolated direct `generate()` vary + kill-switch, both recurrences

`research/findings/raw/_affect_wkv_mouth_verify/phase1_direct_generate_vary_lesion.py`, prompt `"Tell me about
it."`, seed 42, `affect_boost=5.0` (the shipped default), production repetition guard applied.

| recurrence | affect ids | valence -0.9 vs +0.9 differ | neutral vs +0.9 differ | arousal 0.05 vs 0.95 (@v=+0.9) differ | kill switch @ +0.9 matches neutral |
|---|---|---|---|---|---|
| `ssm` (default) | 63 <!--derived--> | **True** | **True** | **True** | **True** |
| `linattn` | 57 <!--derived--> | **True** | False | False | **True** |

(`research/findings/raw/_affect_wkv_mouth_verify_phase1_direct_ssm.json` /
`..._phase1_direct_linattn.json`, fields `lo_vs_hi_differ`/`neutral_vs_hi_differ`/`lowaro_vs_hiaro_differ`/
`killswitch_matches_neutral`.) The `ssm` checkpoint separates cleanly on every axis, with legible mood-congruent
text (neutral: `"...it was very happy and thanked the little girl... not scared of the cake..."`; valence=-0.9:
`"...it was very sad but he could not find his friends..."`; valence=+0.9: `"...said thank you... happy thank
his mom and dad came home... glad they could play together..."`). `linattn` passes the PRIMARY vary test
(lo-vs-hi valence, the one that matters for the `BRAIN_AFFECT_LESION` mechanism, since lesion only ever zeroes
valence -- see below) but not the secondary neutral-vs-hi and arousal-alone probes on this specific adversarial,
topic-empty prompt at this boost -- an honest, reported asymmetry, not hidden.

The kill switch (`BRAIN_WKV_MOUTH_AFFECT=0`) reproduces the neutral (`valence=0.0`) text byte-for-byte on BOTH
checkpoints even when called at `valence=0.9` -- confirming it is a genuine escape hatch, not cosmetic.

## Verification 2 -- isolated `answer_turn`-level valence sweep, both recurrences

`research/findings/raw/_affect_wkv_mouth_verify/phase2_answer_turn_valence_isolation.py` -- mirrors the ORIGINAL
finding's own `phase4_5` `valence_isolation` probe exactly (same call shape: `OE.answer_turn(msg, None, valence,
arousal, ltm_bundle=None, brain_bundle=None, seed=42)`), fact-routing flags forced OFF so the reply is genuinely
`_free_gen`/`_free_gen_linattn`'s own output (not the affect-neutral-by-design fact-clause path), message
`"Tell me about kanton genf"` (an unknown topic with no bundle loaded, forcing free generation deterministically).

| recurrence | `raw` identical, valence -0.9 vs +0.9 | `raw` identical, neutral vs +0.9 |
|---|---|---|
| `ssm` | **False** | **False** |
| `linattn` | **False** | **False** |

(`research/findings/raw/_affect_wkv_mouth_verify_phase2_answer_turn_ssm.json` /
`..._phase2_answer_turn_linattn.json`, fields `raw_identical_lo_hi`/`raw_identical_neu_hi`; the ORIGINAL finding
measured `raw_identical_across_valence: true` for this exact probe shape pre-fix.) `ssm`'s text is clearly
legible mood-congruent narrative ("...sad and angry at each other... started to cry and cry" vs "...happy and
proud of him... laughed and played together... laugh and laugh and have fun together"); `linattn`'s text differs
genuinely (confirmed by exact string inequality) but drifts into a repeated "play"/"playoffs"/"playhouse" theme
under positive valence rather than an obviously legible warm tone -- real, causally-connected, but qualitatively
weaker than `ssm`'s separation, consistent with Verification 1's finding.

## Verification 3 -- the REAL `BRAIN_AFFECT_LESION` live-pipeline test (the decisive one)

`research/findings/raw/_affect_wkv_mouth_verify/phase3_live_pipeline_lesion.py`, through the ACTUAL
`webapp.server.brain_chat` in-process handler (the function `/api/brain-chat` dispatches to), a real `onebrain`
composer, the real spiking affect organ, `SIM_BACKEND=cupy` on the free RTX 3090 (GPU queue confirmed idle
before starting: `bash tools/gpu_queue.sh status` reported `state: running`, `current: (idle)`, `queued: 0`,
`VRAM free: 21115MiB`), **`BRAIN_WKV_MOUTH_RECURRENCE` left UNSET -- the SHIPPED PRODUCTION DEFAULT (`ssm`)**,
one session, `reset=True` only on the first turn (peak RSS during the run stayed under 1.6GB, well inside the
4GB budget: `ps -o rss=` sampled `1567364` KB mid-run).

Sequence: (1) a sentiment-laden priming message ("I am absolutely thrilled and overjoyed today, everything is
wonderful!") establishes a real appraised mood via `_update_session_mood`'s cross-turn EMA -- the SAME mechanism
`webapp/server.py`'s live Gate-B block already uses in production; (2) the SAME known-topic query ("Tell me
about frank_lincoln_wright") asked three times in the SAME session with fact-routing forced OFF (so the reply is
genuinely `_free_gen`'s own output, not the fact-clause template): `BRAIN_AFFECT_LESION=0`, then `=1`, then `=0`
again.

**The lesion mechanism itself is confirmed to hold at the moment of measurement** (`docs/TERMS.md`'s own bar for
the word "lesion"): under `BRAIN_AFFECT_LESION=1`, the organ's own differential reads exactly `0.0`
(`valence_sign: "0"`, `tone_level: 0`, `pos_rate: 0.0`), vs `differential: 0.03805555555555556` (`valence_sign:
"+"`, `tone_level: 2`) unlesioned on the identical session/message -- both from
`research/findings/raw/_affect_wkv_mouth_verify_phase3_live_pipeline_lesion_ssm.json`, `rows.lesion_test.affect`.

**The result:**

| probe | value |
|---|---|
| `raw` identical, lesion=0 vs lesion=1 | **False** |
| `raw` identical, lesion=0 vs lesion=0 (repeat, determinism control) | **True** |

(same artifact, `rows.lesion_test.raw_identical_lesion0_vs_lesion1` / `raw_identical_lesion0_vs_lesion0_repeat`.)
The repeatability control is the load-bearing negative control: with the SAME (unlesioned) affect state, two
separate calls in the same session reproduce the identical `raw` text byte-for-byte, so the difference against
the lesioned arm cannot be attributed to some other source of nondeterminism (sampling drift, session-state
churn) -- it is specifically attributable to the lesion.

Text (both real, both through the shipped default `ssm` recurrence, same session, same seed): lesion=0
(unlesioned, real organ differential `0.03805555555555556`, mapped valence ~0.15 <!--derived--> (`min(1.0,
differential*4.0)`, the pre-existing `_valence_from_differential` formula unchanged by this fix) -- small,
because the REAL organ signal at this operating point is small, unlike the synthetic +-0.9 sweeps above) ends
`"...his friends were very excited
to play here and have fun with the new ball but it was too hard and he was very sad"`; lesion=1 (valence forced
to exactly 0.0) diverges partway through the SAME sentence and ends instead `"...his friends were very fast and
the puzzle at the park tim saw a big red and shiny on the ground and the bird became"`. This is a **small but
genuine, causally-attributable divergence at the REAL operating point** -- not the dramatic swing the synthetic
`+-0.9` isolated sweeps produced (those exist to prove the wire is CONNECTED; this run proves it is connected
**at the magnitude the real organ actually produces**, which is the number that matters for a production flip).

Per the owner's own anti-hollow bar ("vary changes it, lesion vanishes it"): vary (Verifications 1-2, and the
priming-vs-neutral-mood contrast baked into this very run) changes the output; lesion removes exactly that
change, confirmed against a determinism control. **The primary claim is GO.**

## Does this fix BOTH recurrence families?

**Yes, from one shared mechanism**, confirmed directly: `_apply_affect_bias` is called from both `_free_gen` and
`_free_gen_linattn` (the only difference between the two loops is the state object each `Readout` class uses),
and Verifications 1-2 above were run against BOTH `ssm` and `linattn` checkpoints. The SHIPPED production default
is `ssm` (`BRAIN_WKV_MOUTH_RECURRENCE` unset), which is also the ONLY recurrence exercised by Verification 3 (the
real live-pipeline test) -- running that same live-pipeline lesion test a second time against `linattn` would
cost another ~184s GPU-backed brain-build (`wall_seconds: 184.4` this run) for a check whose PLUMBING is already
covered by Verifications 1-2's direct tests on that checkpoint; given Verification 1 already found `linattn`
measurably weaker on secondary axes (neutral-vs-hi, arousal-alone) at the shared boost, I judged the marginal
evidence from a second full live-pipeline build not to change the verdict, and did not run it -- named here as
the concrete next check if `linattn` becomes the shipped default.

## Honest residuals

1. **Host decode-time arithmetic, not a spiking/neural coupling.** The `valence`/`arousal` INPUTS are genuinely
   neural (the organ's own differential read); the MECHANISM that turns them into a word-choice bias is host
   logit arithmetic, the same category the module already claims for `_apply_fact_boost`. The concrete next step
   (named in `_apply_affect_bias`'s own module comment): fold valence/arousal into `FewSpikeWordRead`'s own
   Izhikevich population as a genuine gain/threshold term (e.g. modulating the soft-WTA population's excitability
   directly), so arousal-as-gain and valence-as-direction act on the SPIKING read mechanism itself rather than
   the logits feeding it.
2. **One shared `affect_boost=5.0`, empirically calibrated on the harder checkpoint, not independently re-tuned
   per checkpoint/topic.** Documented with the exact sweep evidence in the code (`generate()`'s own docstring)
   so a future re-tune starts from measurements, not a guess. A production deployment that finds this too weak
   or too strong on real traffic should re-run the sweep in
   `research/findings/raw/_affect_wkv_mouth_verify/phase1_direct_generate_vary_lesion.py` rather than assume this
   value transfers unconditionally.
3. **`linattn` is measurably weaker than `ssm`** on two of four Verification-1 sub-probes (neutral-vs-hi,
   arousal-alone) at the shared boost, and its Verification-2 mood-congruent shift is less legible (a repeated
   theme-word drift rather than clearly warm/cold prose). The PRIMARY valence-driven vary test (the one
   `BRAIN_AFFECT_LESION` actually exercises, since lesion only ever zeroes valence, never arousal) passes on both
   checkpoints. If `linattn` becomes the shipped recurrence, re-running Verification 3 against it (see above) and
   possibly a `linattn`-specific `affect_boost` are the named next steps.
4. **Arousal is not itself organ-lesionable.** `BRAIN_AFFECT_LESION` clamps the organ's neural differential
   (hence `valence`) to exactly `0.0`; the `arousal` float `answer_turn` receives comes from
   `appraise_text`'s host Warriner/DR-2 word-level appraisal of the message text, a SEPARATE, pre-existing signal
   this fix does not add or change. `_apply_affect_bias`'s design (gain-only role for arousal, direction-setting
   role for valence) makes this harmless for the lesion test specifically (arousal without a valence sign has no
   direction to amplify, so `valence=0.0` still forces an exact no-op regardless of arousal) -- but it means
   "arousal is load-bearing" is a claim about the HOST appraisal signal's reach into generation, not about a
   second neurally-lesionable channel. Not a regression (arousal was never neurally sourced before this fix
   either), but worth stating precisely rather than implying a second organ-lesion test would be meaningful here.
5. **The fact-clause/`sentence_facts` path remains intentionally affect-neutral** -- a scope boundary, not a gap:
   a factual clause's tone should not vary with mood, matching Gate-B's own stated honesty floor.

## Regression check

`tests/test_wkv_invocab_scope_leadin_fix.py`, `tests/test_wkv_mouth_learned_head_path.py`,
`tests/test_wkv_mouth_bpe_decode_wiring.py`, `tests/test_linattn_readout_parity.py` -- **81 passed** on
`SIM_BACKEND=numpy` (no signature of these tests calling the two modified files' new parameters positionally;
all real call sites across the repo use keyword arguments, confirmed by grep before editing, so the new
trailing keyword-only parameters cannot have broken an existing positional call). Two pre-existing, unrelated
failures were found in `tests/test_open_ended_generation_fluent.py`
(`test_render_hypothesis_fluent_flagged_guess_stub`, `test_render_hypothesis_template_fallback_without_mouth`,
both about `research/runners/brain_chat_tui.py::HypothesisSVO`/`render_hypothesis_verified`, a module this fix
never touches) -- confirmed pre-existing by reverting this change (`git stash`) and re-running: identical
failures reproduce on the unmodified checkout. Not caused by, and not fixed by, this change; not investigated
further here (out of scope).

## Provenance

Read this session: `webapp/open_ended_chat.py` (full `answer_turn`, the module docstring's bullet list),
`webapp/wkv_mouth_generator.py` (`generate`, `_free_gen`/`_free_gen_linattn`, `_apply_fact_boost`,
`in_vocab_scope`, the module docstring), `research/runners/affect_production_organ.py` (`appraise_text`,
`AffectProductionOrgan.read_differential`, `affect_enabled`/`affect_lesioned`/`dr2_enabled`,
`MoodConditionedRenderer` -- the sibling Qwen-path affect coupling, itself host-prompt-injection-based, used
as the precedent for "host decode-territory is an established category in this module"), `webapp/server.py`
(the live Gate-B affect block ~L4597-4657, the `BRAIN_OPEN_ENDED` dispatch ~L4683-4711), the cited linattn
live-verification finding in full. Checkpoints `bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{42,43}.npz`
and the Qwen priming corpus `data/corpus/tinystories.txt` were copied into this worktree from the primary
checkout before testing (both untracked/gitignored upstream too, per the cited finding's own provenance note;
not committed here, same as upstream). Every numbered artifact above was produced by a script committed under
`research/findings/raw/_affect_wkv_mouth_verify/`, sidecar-stamped by `research/runners/__init__.py`'s universal
artifact-write hook (confirmed firing on each run via the `[provenance] stamped N artifact(s)` line, not
assumed).
