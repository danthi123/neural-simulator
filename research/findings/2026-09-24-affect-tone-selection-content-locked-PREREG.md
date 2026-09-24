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
verdict: PREREGISTERED. No run of this runner exists yet (not even the smoke). A separate finding reads the 6-seed
  result against the gate fixed here.
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
(none yet)
