---
type: preregistration
status: preregistered
date: 2026-09-24
lane: A · Affect — D5 content-preserving tone
seeds: [42, 43, 44, 100, 101, 102]
mechanism: webapp/affect_tone_selection.py (BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT=1, default-OFF). The Qwen mouth drafts
  the reply ONCE from the production prompt with an affect-free MOOD line, rewrites that one draft in 4 fixed tones,
  a host content lock drops any rewrite that loses or adds a number, proper name or retrieved-fact word, the spiking
  affect organ evaluates every admissible candidate, and the candidate whose organ-evoked valence is closest to the
  organ's held valence is released (host comparator + argmin, a declared shortcut).
runner: research/runners/_lbf_affect_tone_selection_derisk.py
verdict: PREREGISTERED; 6-seed run NOT STAGED. The seed-7 design probe (AMENDMENT 1/2) predicts a neg-direction
  failure for every proposal-generator variant (coverage <= 0.133 < 0.5). Blockers are the rewrite generator, the
  180-word appraisal vocabulary and a lock hole; no evaluation seed has run.
---

# Affect selects the reply's tone among content-locked rewrites of one draft (D5), preregistration

## Why a new method

D5 conditioned the Qwen mouth's generation on the spiking affect organ's valence, in two ways: a graded MOOD line
(prompt mode) and a residual-stream steering vector (resid mode). The 6-seed amend1 run scored **UNDEFINED in both
modes**. The unmet precondition in both was (3a-reply), content identity over the generated known reply.
Conditioning changed WHAT the reply said, not only how it said it. On seed 42 the lesion reply was about "Frank
Lincoln Wright". The pos, neg and ctrl_pos replies were all about Franklin D. Roosevelt. The control also moved,
so the cause is not the valence signal. The greedy spiking-Qwen decode follows a different path under any
perturbation of its prompt or residual stream. Resid mode also broke fluency (max salad 1.0). Prompt-mode
negative conditioning pushed tone the wrong way on 4/6 seeds (SEM band, secondary). The earlier decode-point
methods (the additive word nudge; the neural coupling) are separate NO-GOs. (Source: the D5 finding on main plus
the amend1 verdict JSONs in the D5 build worktree.)

<!--derived-->
THE LAW applies: those are verdicts on a METHOD. The companion process that D5 did not have is a separation of the
two acts. D5 formed content and tone in ONE generation, so moving tone moved content. In speech production the two
come apart. The speaker plans the message, and the comprehension system monitors the planned utterance before it is
released (Levelt 1983's perceptual loop; Hartsuiker & Kolk 2001, doi:10.1006/cogp.2000.0744). That monitor is
sensitive to emotional valence: halting was sensitive to valence (Slevc & Ferreira 2006,
doi:10.1016/j.jml.2005.11.002). The ML analogue that holds content while changing style is overgenerate-and-rerank
(Prompt-and-Rerank, Suzgun, Melas-Kyriazi & Jurafsky, EMNLP 2022, https://arxiv.org/abs/2205.11503).

## The method

1. DRAFT. The mouth writes the reply once. It uses the production prompt with the MOOD line replaced by the fixed
   line "MOOD: you feel even and steady.", so the draft cannot depend on the affect state.
2. REWRITES. The mouth rewrites that one draft in 4 fixed tones: very warm, slightly warmer, slightly subdued, very
   sad. Each rewrite uses a fixed decode seed (brain seed + 7919·(k+1)). No descriptor or instruction word is in the
   tone-scoring lexicon (selftest).
3. CONTENT LOCK (host, declared). A rewrite is admissible only if it keeps every number, every non-sentence-initial
   capitalised name and every retrieved-fact object word the draft carries. It must also add no number or name the
   draft lacks. The draft is always candidate 0.
4. EVALUATION (brain). Each admissible candidate goes through `appraise_text`, the host appraisal lexicon that also
   feeds the held mood. The result then goes through the SAME spiking affect organ:
   `read_differential(..., lesion=False)`. The read resets its bridge, so evaluating a candidate does not change the
   held mood.
5. SELECTION (host comparator + argmin, declared). The released candidate is the one with the smallest
   |v_candidate - v_held|, both in the organ's own units (valence = clip(4 x differential)). Ties go to the lowest
   index. No calibration constant is used. The organ's dead zone and sign asymmetry act on the held mood and on the
   candidate reads alike.

Specific-edge lesion: BRAIN_AFFECT_LESION=1 zeroes the HELD mood, which is the edge into the selector. The organ's
evaluation of candidates stays intact. The lesion arm therefore releases the most neutral-reading candidate through
the same route.

Brain-based boundary. The brain part is the held mood and each candidate's evoked differential, both from the
spiking organ. The host part is the appraisal lexicon, the content lock, the comparator + argmin and the fixed style
list. The Qwen draft and rewrites are the owner-ratified articulation scaffold. **A GO is not credited to a spiking
selection.** The named next rung is to run the selection as the basal-ganglia race
(`bg_action_selection_production_organ.py` extended to N channels).

## The gate, and how each part can fail

Arms per seed (7, each a fresh process): pos, neg, lesion, lesion_rep, ctrl_pos, ctrl_neg, pos_rep. Prompts: the
base runner's 10 tone prompts, plus 5 known topics. The known topics are D5's frank_lincoln_wright plus
wolfgang_amadeus_mozart, enrico_fermi, john_adams and john_von_neumann. All are in the shipped wikidata_100k LTM.
The extra topics are there because with 1 topic, (3a-reply) was unmeasurable on 3/6 D5 seeds.

GO requires all four gate components AND every instrument precondition.

<!--derived from research/findings/raw/_affect_conditioned_mouth/calibration/calibration_fold.json (d_neg_0475 of the weak V- seeds) -->

| component | FAILS if (a realistic outcome) |
|---|---|
| (1) DIRECTIONAL (base gate verbatim): per seed pos_gap > +delta and neg_gap < -delta, 6/6 or 5/6 with the 6th null; delta = pooled std of the lesion arm's per-prompt tone, on the independent 356-word lexicon | the rewrites do not span tone on the independent lexicon; or the organ reads most candidates as 0 (ladder dead zone at \|appraisal\| <= 0.25), so selection rarely leaves the draft; or the weak V- seeds (43/100/101/102 hold only -0.018..-0.024 at the priming appraisal) release a candidate no more negative than the lesion's <!--derived--> |
| (P) seed-level exact sign-flip null over the 6 per-seed D_s = tone_pos - tone_neg: mean > 0 and p <= 0.05 (64 assignments) | the direction is inconsistent across seeds (one sizeable wrong-signed seed is enough) |
| (3a-reply) D5's check verbatim: fact-word recall of every conditioned arm's known replies vs the lesion's >= 0.75 on every seed | the draft differs between arms, so the lock cannot hold content between arms (it only holds content between a draft and its own rewrites) |
| (C2) general content-word recall vs the lesion replies (words of length >= 4 that are not stop words, not WARRINER, not the tone lexicon), pooled over all 15 prompts, >= 0.60 per conditioned arm per seed | the rewrites keep the names/numbers/fact words (which the lock enforces) but paraphrase the rest of the content away. The lock does not check these words, so this gate is not satisfied by construction |

Instrument preconditions (any unmet -> UNDEFINED, never a negative):
- Base: all 36 base arms present; lexicon disjoint from WARRINER; generator == qwen on tone rows; moat; fluency
  (salad <= 0.16 on every row); lesion == lesion_rep byte-identical; (2) attribution |ctrl_pos - ctrl_neg| < delta
  on every seed. (Fluency FAILS if a rewrite degenerates into repetition. Attribution FAILS if the priming text leaks
  into the draft or rewrites independently of the held valence.)
- (3a) base retrieval identity. This is an INTEGRITY SMOKE here, because it is fixed before generation.
- pos_rep present; (5b) pos == pos_rep byte-identical.
- (R) 6 distinct lesion realizations, with decode_seed == brain seed.
- (L) held valence != 0 on every pos/neg/ctrl row and == 0 on every lesion row; every row ran the selection with
  at least 1 evaluated candidate. Fails if the flag did not reach answer_turn or the lesion did not zero the held mood.
- (O) the priming differential equals the committed NO-GO arm's value. Same brain state as the NO-GOs and D5.
- (C1) INTEGRITY: the draft sha is identical across all 7 arms, for every seed and prompt. Fails if any non-affect
  session state that the priming touches leaks into the draft prompt.
- (P-ctrl) the control's seed-level directional gap must NOT reach sign-flip significance. The shuffled-valence
  control must not show the effect.
- Measurability: (3a-reply) needs >= 2 lesion fact words per seed; (C2) needs >= 10 lesion content words per
  seed.

Honest limits of the design, written down before any data:
- In the ctrl arms, ctrl_pos and ctrl_neg get the SAME random-sign ±0.16 sequence. If the drafts are identical (C1),
  their replies are identical and (2) passes. So (2) and (P-ctrl) test only priming leakage (as in D5). Whether the
  selector follows its target at all shows in (1) and in the released-style diagnostics.
- (3a-reply) can fail only through draft divergence. The lock enforces fact words between a draft and its own
  rewrites. (C2) is the content gate that is not guaranteed by construction.
- Delta comes from the lesion arm. Here the lesion releases the most neutral-reading candidate, not the raw draft,
  which may narrow delta relative to D5. The definition is the base gate's, unchanged.

Diagnostics reported, not gated: released-style counts per arm; lock pass rate per style; mean evoked valence per
style and the fraction that is nonzero; held differential per row.

Literal scoring command (after all 42 arms exist):
`.venv/bin/python -m research.runners._lbf_affect_tone_selection_derisk --score-only`

## Staging

- A smoke on seed **7** (NOT an evaluation seed): arms pos/neg/lesion, 2 tone + 2 known prompts, Qwen on CUDA via
  gpu_queue. It checks the instrument only (the path runs end to end, lock pass rate, organ reads nonzero, footprint,
  time). It carries no weight in the gate. If it shows the design predicts failure (e.g. the lock rejects every
  rewrite, or every organ read is 0), the design is fixed and amended here with a log of what was seen, before the
  6-seed run is staged.
- 6-seed run: one gpu_queue line per seed, arms sequential in the line, each arm under `tools/memcap.sh 12` (D5 arms
  peaked at 8.8 GB). Output goes to `research/findings/raw/_affect_tone_selection/run1/`.

Honesty boundary: functional read-out only. The released reply's tone tracking the organ's valence is a
coupling. Nothing here claims felt experience.

## AMENDMENT LOG

**AMENDMENT 1: 2026-09-24, about 00:45 EDT, after the seed-7 smoke and before any evaluation-seed run.**

What I had seen when I wrote this. The seed-7 smoke (NOT an evaluation seed) ran 1 of its 3 arms: `neg`, with 2
tone prompts and 2 known prompts. I read everything in that arm: the candidate texts, the lock details, the
appraisals and organ valences, the released styles, and its tone scores. `pos` crashed: the worktree had no `data/`
symlink, so `data/corpus/tinystories.txt` was missing. That is fixed by linking the shared `data/` as the D5
worktree does. `lesion` was still running. No evaluation seed has run.

<!--derived-->
What the smoke showed (rounded from `research/findings/raw/_affect_tone_selection/smoke_s7/arm_s7_neg.json`). The prediction below says the preregistered generator (v0) would fail the neg direction by
construction:
- The "slightly more subdued and wistful" and "very sad, somber and melancholy" rewrites carried no strongly
  negative word that the appraisal reads. Both appraised 0.0 on both tone prompts ("The sea is a vast and intricate
  system..."). The affect-free drafts appraised POSITIVE (+0.60, +0.69), so the most neutral candidate was a
  "subdued" rewrite at 0.0. The `neg` arm (held valence -0.157) released that same candidate. The lesion arm (held
  valence 0) must release it too, so neg_gap = 0 on such prompts.
- The lock rejected most rewrites of the known replies. They dropped dates or names, and one invented "Aged 103 ...
  April 14, 2023". This is the lock working, but it leaves few candidates.
- Some admissible rewrites were cut at 160 tokens before a name the draft carried ("Earth").

The change. The proposal generator becomes a choice among 4 variants, all in `webapp/affect_tone_selection.py`
`VARIANTS`:
- v0: the preregistered generator.
- v1: descriptors that name the feeling plainly ("joyful, delighted and happy" / "warm and pleased" / "sad and
  unhappy" / "deeply sad, hurt and miserable"), an instruction to let the feeling show in word choice, and 220
  tokens.
- v2: v1 plus a two-line happy/sad demonstration on an unrelated fact.
- v3: v1 plus a FIXED-sign residual steer, D5's CAA axis at c = ±0.5 / ±0.25 per style, applied during the rewrite
  only.

In every variant the proposals depend only on the draft, the fixed variant and the decode seed, never on the
organ's state. No descriptor, instruction or demonstration word is in the tone lexicon (selftest).

The choice rule, fixed before the probe runs. `--restyle-probe` runs on seed 7 only: Qwen, no brain, all 15
prompts, the affect-free draft and each variant's 4 rewrites. Among variants whose admissible candidates all have
salad <= 0.16, pick the highest COVERAGE. Coverage = min(fraction of prompts with an admissible candidate whose
appraisal is <= -0.30, fraction with one >= +0.30), where 0.30 is just past the ladder's 0.25 dead zone. Ties go
to the lower index. The rule uses only the organ's INPUT (the appraisal), the lock and fluency. It never uses the
independent tone lexicon, which the probe does not compute. **If the best coverage is < 0.5, no variant is chosen:
the design is predicted to fail, and the 6-seed run is NOT staged.** The chosen variant becomes `ACTIVE_VARIANT`,
and the seed-7 smoke is re-run on it before the 6-seed run is staged.

Unchanged: every gate and precondition above, the thresholds, the arms, the prompts and the seeds.

**AMENDMENT 2: 2026-09-24, about 01:15 EDT. The probe result: no variant is chosen, so the 6-seed run is NOT
staged.** No evaluation seed has run.

<!--derived-->
The probe summary, from `research/findings/raw/_affect_tone_selection/amend1_probe/restyle_probe_s7.json`, seed 7,
15 prompts:

| variant | prompts with an admissible candidate appraised <= -0.30 | ... >= +0.30 | coverage | lock pass | max salad (admissible) |
|---|---|---|---|---|---|
| v0 | 0.0 | 0.667 | 0.0 | 0.633 | 0.167 |
| v1 | 0.0 | 0.667 | 0.0 | 0.55 | 0.222 |
| v2 | 0.067 | 0.4 | 0.067 | 0.683 | 0.217 |
| v3 | 0.133 | 0.667 | 0.133 | 0.567 | 0.447 |

<!--derived-->
No variant is fluent: all four have max admissible salad above 0.16. The best coverage is 0.133, below the 0.5
floor, so `choose_variant` returns NONE. Under the preregistered rule the design predicts a neg-direction
failure, and the 6-seed run is not staged.

<!--derived-->
Why, from a post-hoc diagnosis (`research/findings/raw/_affect_tone_selection/amend1_probe/probe_diagnosis_s7.json`,
seed 7 only, written after the choice). There are three separate blockers:
1. **The proposal generator seldom writes a negative rewrite.** Of 30 "sad"-style rewrites per variant, the
   independent tone lexicon reads 3 / 9 / 7 / 10 as negative (v0 / v1 / v2 / v3). Qwen-0.5B often answers a
   request for a sad rewrite with a neutral paraphrase, a refusal ("I'm sorry, but I can't do that"), or
   unrelated text. v3's fixed CAA steer moves more rewrites negative but breaks fluency.
2. **The brain's appraisal cannot perceive most negative words.** The appraisal's salience gate is the 180-word
   WARRINER set. "sad", "pain", "grief", "hurt" and "despair" read strongly negative. But "sadness", "saddened",
   "unhappy", "sorrow", "melancholy", "loss", "loneliness", "decay" and "alone" read 0.0: they are not in the set.
   Even rewrites that the ruler reads as negative mostly appraise above -0.30 (1 / 0 / 1 / 6 of 30). So the organ
   cannot tell a negative candidate from a neutral one. The held mood has the same problem one step earlier: it
   is read through the same 180-word gate.
3. **The content lock has a hole for drafts with no names or numbers.** Unknown-topic drafts carry nothing for
   the lock to check. 3 or 4 admissible candidates per variant are refusals, which (C2) would catch only at the
   gate. The lock needs a general content-word recall term.

What held (seed 7, 2 arms; an integrity smoke, not evidence): the affect-free draft was byte-identical between
the `neg` and `lesion` arms on all 4 prompts. Separating draft from tone removes the content drift that made D5
UNDEFINED, by construction.

THE LAW: this is a verdict on the method's current COMPONENTS, not on the capability. The next rungs, in order:
- (a) The brain's affect PERCEPTION vocabulary. This is the missing companion process: the mouth speaks
  open-vocabulary text, but the appraisal hears 180 words. A learned salience gate over the DR-2 map, or
  morphological generalisation (sadness -> sad, loneliness -> lonely, un- negation), would widen what the organ
  perceives in both the held mood and each candidate. It is an appraisal-lane build with its own neutral-fact
  invariant (see `affect_production_organ.appraise_text`'s note on why the learned gate was not adopted).
- (b) A proposal generator that writes negative rewrites fluently, and a lock that rejects refusals (content-word
  recall against the draft).

The 6-seed run is staged only after (a) and (b) pass this same probe rule on seed 7.

**AMENDMENT 3: 2026-09-24, about 01:20 EDT, branch `research/affect-learned-vocabulary`. The lock gains a content-word
recall term (rung (b), lock half).** No evaluation seed has run.

What I had seen. Only the seed-7 probe artifact named in AMENDMENT 2. I recomputed, from its stored texts, the share
of each draft's content words that each rewrite keeps. Content words are the (C2) class: length >= 4, not a stop
word, not WARRINER, not the independent tone lexicon. Every refusal or unrelated rewrite in that artifact keeps
less than 0.10 of them.

The change, in `webapp/affect_tone_selection.content_lock`. A rewrite is also inadmissible when it keeps less than
`LOCK_RECALL_MIN = 0.25` of the draft's content words. The term applies only when the draft has at least
`LOCK_RECALL_MIN_WORDS = 4` content words; otherwise the detail records `content_recall: None`. 0.25 is well below
the (C2) gate's 0.60, so (C2) stays a separate gate that can fail. Pinned by
`tests/test_affect_tone_selection.py::test_lock_content_recall_rejects_refusal_on_a_draft_without_names_or_numbers`.

Unchanged: every gate, threshold, arm, prompt and seed above, and the rule that the 6-seed run is staged only after
rungs (a) and (b) pass the seed-7 probe rule. The proposal-generator half of rung (b) is not addressed here. Rung
(a) is the learned affect vocabulary, pre-registered separately in
`research/findings/2026-09-24-affect-learned-vocabulary-PREREG.md`.
