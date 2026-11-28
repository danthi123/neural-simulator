# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md

# Local rule v3q prereg (2026-11-28, REGISTERED): v3's rule with question-form targets (candidate 3)

STATUS: REGISTERED (2026-11-28, pre-data; nothing has run). Engine specs
`tools/engine/specs/local_rule_v3q_{01b,04b}.yaml` are `status: REGISTERED`. The file keeps its `-DRAFT` name because the
specs and the harness cite this path. Launch precondition (B2, not yet met): `brain/experiments/local_rule_v3q_life_cues.json`
must be committed with `brain/` clean, because its sha256 is pinned in the harness (`LIFE_CUES_SHA256`); unshipped, every
job fails `load_life_cues` as ERROR-SMOKE.
- **2026-11-28 registration review (pre-data), fixes applied.** B1: `report()` now gives paraphrase /12 per arm (mean,
  kept vs excluded facts, arm minus no_sleep) and a `paraphrase_transfer` block (v3q minus no_sleep on all facts, on kept
  facts, on eval-cued facts and on the other kept facts), tested; the engine digest summary now prints the eval-cue share,
  the "eval-cue carried" caveat when set (N6), and paraphrase v3q minus no_sleep. Prereg text: sham_ctx as the exposure
  control (B1), STOP-G0Q expected at 0.1B and 0.4B launched directly in parallel (N3), cue-build cost revised (N4), NLMS
  dilution by post-key rows (N5), cross-node cue-set divergence as a known loud failure (N1), the floor risk of the graded
  read-out (N2). Not changed: N1's frozen cue file, N2's second graded read-out, N5's answer-span restriction (later
  variants). Tests: 158 passed (v3q + v3). Milestone: ARCHITECTURE section 4, 2026-12-31, local sleep rule. The research note
moves this to a 12-12 decision gate. This is candidate 3 of `2026-11-28-local-rule-research-round.md` section 2.3. It is
the cheapest of the three and runs in parallel with candidate 1. It does not re-open v3's NEGATIVE-DEV.

## Record brief
- **Knew.** v3 fits its own layerwise copy-mode targets: error -70%. Recall is 0-2/12 at 0.1B and 0.4B (NEGATIVE-DEV).
  The research note's claim: v3's target was the whole copy-mode activity at every position and every layer,
  unweighted. The part that matters for the output can sit in the 30% left unfitted.
- **v3's named follow-up.** Make the target the cortex answering a question with the telling in context, and take it
  only at the answer positions. G0 already showed the teacher right 12/12 with the telling in context.
- **Caution, M3 (11-03).** Under gradient descent, retrieval practice lost to restudy: -0.5/12 at 0.1B and -2.67/12 at
  0.4B. A question-form target is retrieval practice. Nothing predicts it beats the telling form under a local rule.
- **Prerequisite (research note section 4, "Replay cue form").** Questions are not stored. The hippocampus cannot
  generate a cue from part of an episode. Host-templated cues are forbidden. This prereg uses only two cue sources, both
  from the brain or its life, and counts the facts that have neither.

## Method (module `brain/experiments/local_rule_v3q.py`; imports `local_rule_v3` and `local_rule_v2`, edits neither)
The rule is v3's, unchanged: `v3_update`, normalised LMS, the KP copies, the clip, B decay, the keep step and its
weight, the RNG draw pattern (one draw for the fact, one for its cue, one keep text per step), and 240 steps. Only the
clamped phase and the target rows change.
1. **Clamped phase (teacher).** Adapters off. The cortex reads the fact's telling (the stored episode text,
   `sleep_probe._told`). Then it reads the cue in the chat frame `User: <cue>\n\nAssistant:`, then its OWN in-context
   answer: greedy, adapters off, telling in context, up to 16 tokens, ending at the turn break. The teacher records, per
   block, the residual and the channel-mix hidden at the rows that predict the answer tokens. Those rows run from the
   last prompt position to the second-to-last answer token.
2. **Free phase (student).** Adapters on. The cortex reads the same cue + answer ids with no telling in context.
3. **Update.** `v3_update` on those rows only. The channel mix works position by position and token-shift is already in
   its input, so slicing the rows is exactly the masked rule (tested). The keep step is v3's.

## Cues (the brain's own; no host-written question)
- **Life cues.** The questions the user actually asked about these facts in the three KR DEV lives (s319, s320, s321;
  `lived`, trunk line, kinds sameday and remention). `cues-from-life` extracts them once into
  `brain/experiments/local_rule_v3q_life_cues.json`. Each source life's sha256 is recorded in that file, and the file's
  own sha256 is pinned in the harness (`LIFE_CUES_SHA256`). Lives of split test or test2 are refused (tested).
- **Result of the extraction.** 3 of 12 facts have a life question:
  - pixel: "What is my cat called?" and "What name did I give my cat?";
  - marguerite: "What is my sister's name?";
  - volvo: "Which car do I own?".
  The other 9 facts are never asked about in a dev life.
- **Self cues.** The cortex itself, adapters off, reads the telling and the next turn marker `User:` and writes the next
  user turn. It does this greedy once, then 7 seeded samples (temperature 1, CUE_SEED 0, fixed across seeds so all six
  seeds train on one cue set). `User:` is the chat format's own role marker, which every telling already carries. It is
  not a question template. The host never supplies question words.
- **Kept only if.** The cue is a one-line question (ends with "?"). It does not name the answer key. The teacher with
  the telling in context answers it with the key. At most 3 cues per fact, life cues first.
- **Exclusion.** A fact with no kept cue is EXCLUDED, with its reason recorded. It is still scored in recall /12, the
  same 12 as v3, so the bar is unchanged. The counts are reported: kept facts, kept cues by source, excluded facts and
  why, rejected candidates and why.
- **Gate G0Q** (free, before any night). At least 6 of 12 facts have a kept cue. The bar needs no_sleep + 2 facts that
  can be trained. A failing G0Q writes the g0q unit and exits non-zero before any night, both in the smoke (ERROR-SMOKE)
  and on resume (tested). The pick returns STOP-G0Q (engine HOLD-DEV).
- **Disclosed overlap.** Two life cues are word for word the eval questions: pixel "What is my cat called?" and
  marguerite "What is my sister's name?". The user did ask them, so they stay. Each unit flags `cue_is_eval`. The report
  gives the share of v3q's recall gain carried by eval-cued facts (`eval_cue_gain_share`). If that share is above 0.5,
  the verdict is reported with the caveat "eval-cue carried" (report only: it does not change the word; the digest
  summary prints the share and the caveat). Paraphrase recall is reported beside it as the transfer check
  (`report.paraphrase_transfer`: v3q minus no_sleep on all 12, on kept facts, on eval-cued facts and on the other kept
  facts). A gain on the eval-cued facts that does not show under paraphrase is read as training on the test prompt.
- **Exposure is controlled on the student side.** `sham_ctx` reads the same cue + answer ids as v3q, without the fact's
  telling; clause (b) requires v3q to beat it on at least 5 of 6 seeds. Seeing the eval question during the night is
  therefore not enough to pass.

## Arms (per model, per seed; 42, 43, 44, 100, 101, 102)
1. `v3q`: the rule at (lr*, keep*), targets at every block, teacher = own telling.
2. `gradient`: `brain.sleep.sleep` on the 12 tellings (Adam 5e-4, KL weight 1). This is v3's reference, unchanged.
3. `no_sleep`: adapters at zero.
4. `sham_ctx`: the teacher reads ANOTHER kept fact's telling before the same cue + answer ids, by a seeded derangement
   over the kept facts. It controls for cue and answer exposure without the knowledge.
5. `no_target` (lesion): no telling, so the target is the pretrained cortex's own activity. It must stay exactly inert.
6. `top_only` (depth lesion): the last block only.
7. `grad_distill` (report only): Adam 5e-4 on KL(teacher || student) at the answer rows, plus the keep KL. It separates
   "bad target" from "bad rule".
8. `gradient_q` (report only): `brain.sleep` on the cue + answer exchanges, each fact weighted 1. This is retrieval
   practice under gradient on these cues: the M3 contrast against `gradient` (restudy).
Models: 0.1B `rwkv7-g1d-0.1b-20260129-ctx8192.pth`, then 0.4B `rwkv7-g1d-0.4b-20260210-ctx8192.pth`.

## Dev (seed 7, per model; v3's grid and pick)
- lr in {0.03, 0.1, 0.3} x keep in {0.3, 1, 3}: 9 `v3q` nights, plus `ref` (G0Q, no_sleep, gradient). That is 10
  engine jobs.
- The pick is `local_rule_v3.dev_pick` with G0 replaced by G0Q, so a failed G0Q gives STOP-G0Q. Everything else is v3's:
  - recall >= max(0.5 x gradient, no_sleep + 2);
  - generic KL and dev-half held-out KL each <= 1.25 x gradient's, and <= 1;
  - engaged: norm > 0 and err_end < err_start;
  - max recall wins; ties go to lower held-out KL, lower generic KL, lower lr, lower keep.
- If no cell qualifies: NEGATIVE-DEV, no six seeds.
- The pick refuses unless the dir holds exactly the G0Q unit, the 2 refs and the 9 cells, all from one model, commit,
  prompts list and cue set (cue_sha256).
- The firewall is v3's: the VERDICT half of the held-out prompts is never computed on seed 7 (tested).

## Measures (hippocampus off; one atomic json unit per (seed, arm); resume by skip)
- **Battery.** `sleep_probe.evaluate`: recall /12, false recall /4, generic KL, paraphrase /12 (reported per arm, kept vs
  excluded, minus no_sleep). The held-out KL is v3's
  (pinned prompts, dev and verdict halves). Adapter and factor norms.
- **Graded read-out (research note section 4).**
  - Per fact, at the eval question's frame `User: <q>\n\nAssistant:`, the log-probability of ` <key as told>` (summed
    over its tokens, and its first token alone), and the rank of its first token.
  - Per arm: the mean log-probability, the median rank, and the number of facts ranked in the top 10.
  - The report gives arm minus no_sleep on all 12 facts and on the kept facts. Recall is split the same way: kept
    facts and excluded facts.
  - Known limit (N2): it scores the bare key at the first answer position, while the cortex's natural answer is a
    sentence ("Your cat is called Pixel") and recall counts the key anywhere. It may sit at the floor and not track
    recall; a flat graded read-out is not evidence by itself. A teacher-forced read-out inside the in-context answer is a
    later addition, not in this registration.
- **Engagement record** (every arm):
  - local arms: teacher kind, `target_form: question`, layers on, units and facts, mean target rows, mean context
    tokens, err_start/err_end (a fixed-order, forward-only probe at the answer rows over every unit, at the night's
    start and end), clip fractions, head/tail;
  - gradient, grad_distill, gradient_q: loss_start/loss_end (fixed-order probes);
  - no_sleep: steps 0.
- **Provenance.** Host, host_env, commit, device, precision, model id, prompts sha, cue_sha256, life-cues sha, lr,
  keep, grad_lr, last layer, elapsed, cue-build seconds. Every night unit stores the cue summary. The g0q unit stores the
  full cue set (ids, rows, answers, rejected).
- **Guard.** POOL_NODE=1 or BRAIN_ALLOW_W3090=1.

## Registered rule (planned; `verdict()` in the module, tested)
- **What it is.** `local_rule_v3.verdict` on the seven v3 arms with v3 := v3q. Its clauses (a)-(e), its engagement
  gates and its order are unchanged, and `verdict` raises unless every one of the 8 arms holds exactly the six seeds.
- **Before that.** A missing or failed G0Q record gives UNDEFINED.
- **Words.** GO-V3 is renamed GO-V3Q. GO-SHALLOW, NEGATIVE and UNDEFINED keep their names.
- **Core NEGATIVE reading, from grad_distill on the same question-form target.**
  - It passes (a) and (c): rule-limit.
  - It fails: target-limit. Neither target form stores at this scale, consistent with M3.
  - It is not engaged: undetermined.
- **Report only (never changes the word):** the graded read-out, recall and paraphrase on kept vs excluded facts, the
  paraphrase transfer, the eval-cue share,
  the M3 contrast (gradient_q minus gradient), and report-arm engagement.
- Tests: each clause failing alone gives the exact word; each engagement gate alone gives UNDEFINED; any non-finite
  gated value gives UNDEFINED; report arms never change the word.
- **Two scales** (`combine`): GO-V3Q, GO-V3Q-PARTIAL, GO-SHALLOW, NEGATIVE, or UNDEFINED (any UNDEFINED, STOP, HOLD or
  ERROR).

## Expected outcome (written before data)
- **Most likely: NEGATIVE-DEV or NEGATIVE at both scales**, for three reasons:
  1. **M3.** Retrieval practice lost to restudy under gradient descent at both sizes (-0.5/12 at 0.1B, -2.67/12 at
     0.4B). A layerwise local fit of a retrieval-form target has no reason to do better than gradient on it.
     `gradient_q` measures that directly here. If gradient_q < gradient again, a v3q negative is the expected one.
  2. **The layerwise fit.** It may again leave the output-relevant direction unfitted (the research note's claim, which
     T0 tests).
  3. **Few facts to train.** Only 3 facts have a life cue. Self cues from the 0.1B cortex may be rare or not answerable,
     so **STOP-G0Q at 0.1B is the expected outcome**: `sleep_probe.py` documents the 0.1B in-context ceiling at 6/12,
     and G0Q needs at least 6 facts whose cue the teacher answers correctly in context. It would be a verdict on cue
     availability (the missing prerequisite), not on the rule. Because the 0.1B ladder stops on ERROR-SMOKE and
     HOLD-DEV, **`local_rule_v3q_04b` is launched directly, in parallel with 0.1B**, not chained.
- **What would surprise us.** The graded read-out moving on the kept facts (v3q log-probability above no_sleep by a
  clear margin) while recall stays under the bar. That would mean the direction is right and the size is too small,
  and it feeds candidate 2 (a key-value write at the answer rows).

## Cost (estimate, not measured)
- v3's units plus one arm, plus the cue build once per process: up to 12 x (8 generations of 24 tokens + up to 6
  answers of 16 tokens), forward-only: roughly 5k sequential decode steps per job.
- Cue build guessed at about 1-3 min per job at 0.1B and 5-9 min at 0.4B on CPU (revised up at review, N4); it repeats
  in every job. The smoke measures it.
- Memory caps are v3's (5 and 10 GB), provisional until the smoke's peak sets them; cached targets are only the answer
  rows.
- **Known risk (N1): cue-set divergence across nodes.** The cue set is generated per job on CPU; different CPUs can flip
  near-ties in greedy or sampled generation. `pick` and `score` refuse mixed `cue_sha256`, so the failure is loud, but
  it surfaces only at pick or score. If the smoke or dev shows it, the self-cue set is frozen into a pinned committed
  file (as the life cues are) by an amendment before six seeds.
- **Step size (N5).** `nlms` divides by the sum of `||a||^2` over all kept answer rows (up to 16), and rows after the
  key have near-trivial error, so the informative rows get a smaller effective step. The lr grid (to 0.3) and the clip
  partly compensate; restricting rows to the span up to the key is a later variant, not this one.

## Scaffolds and host code (declared)
- The life-cue extraction is host code over world data: the user's own words, from the life transcript.
- The chat frame and the `User:` turn marker are the interface format.
- The greedy and seeded generation is the cortex's own read-out.
- The "kept only if" filter is a host check standing in for the hippocampus's cue validation.
- Nothing here writes question words.

## Not done here
- Partial-cue self-generation by the hippocampus (the prerequisite, not built).
- Life cues for the other 9 facts (would need new lives that ask about them).
- 1.5B.
