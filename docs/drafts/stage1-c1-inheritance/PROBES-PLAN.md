# C1 generalization probes: do the flagged one-hop guesses behave like learning? (dev seeds; bars before data)

Written 2026-10-13 by Claude, BEFORE any probe run (the bars in section 3 are copied verbatim into
`research/runners/c1_generalization_probes.py` and fixed in the same commit). Owed by RULING c1-moat-ruled-1013
(`research/coordination/LOCAL_LLM_RUNBOOK.md`, Claude, owner-delegated): KEEP the brain's flagged one-hop resemblance
guesses (the bat that eats the bug guesses that it swims in the lake, because the fish also eat the bug), but show they
behave like learning, not templating, with two dev-seed probes before the 11-01 flips. If either fails, file NEEDS.

## 1. What separates learning from templating

A template applies a fixed rule ("shares a trait with C, so has C's property") at one strength, whatever else the
system has been told. A learner (a) lets SPECIFIC evidence win over the inference (the 07-08 cancellation finding:
"birds move, but the penguin walks"; `research/findings/2026-07-08-cancellation-member-exception-overrides-inheritance-real-corpus-GO.md`)
and (b) holds the inference with a strength that grows with the evidence for it. The organ has no priority rule
between a told frame and an inherited one: both are frame slots competing on spike counts under the asked verb, so
(a) and (b) are properties of what the synapses learned.

## 2. Design (one fresh process and one fresh ChatBrain per (seed, probe); production defaults)

`BRAIN_CATEGORY_INHERIT` and `BRAIN_CATEGORY_INHERIT_HOP` are UNSET (the flipped defaults apply) and
`BRAIN_CATEGORY_INHERIT_LESION` is removed from the env; each shard records the effective values
(`stage1_exit_gate.effective_c1_flags`). Turns go through the real handler (`webapp.server.brain_chat`, tiny-demo,
stub renderer, the Stage-1 `BASE_ENV`).

**OVERRIDE session.** The `cat_bridge` teach turns of `c1_flip_dev_check` (the C1 birds and fish, `the bat eats the
bug`, `the bat flies in the sky`, `the canary flies in the sky`); `what does the bat swim` (the inherited guess,
lake: 8/8 dev seeds in the flip dev rerun at 2c2d55952); `the bat swims in the pond` (the specific told fact: frame
`swim pond`, admitted under the same verb as `swim lake`, so the two compete in the organ's own frame space);
`what does the bat swim` again; then three reported collateral reads (`what does the bat eat`, `what does the canary
eat`, `what does the canary swim`).

**GRADED session.** A fish category: trout and salmon each `eat bug`, `bite hook`, `hide weed`, `swim lake`. Three
targets get THREE told facts each, of which k are fish traits and the rest are frames no one else carries, so the
number of facts is constant and only the number of shared traits varies: eel k=3 (eat bug, bite hook, hide weed),
toad k=2 (eat bug, bite hook, find stone), newt k=1 (eat bug, see moon, like rain). Targets are taught in DESCENDING
k and each target's shared traits come first: the k=1 target is encoded last, against the strongest lateral
structure, and its unshared facts get extra integrative-encoding passes, so the order favors the low-k targets
(against the hypothesis). Probes: `what does the eel swim`, `... toad ...`, `... newt ...`. Then the override against
the strongest guess: `the eel swims in the river`, `what does the eel swim`.

**Direct reads.** After the named probe turns the worker calls `infer(agent, verb)` on the session's own organ
(`chat._category_inherit_organ`; read-only: it restores the dynamical state and the weights). This is the organ's
decision in its own frame space even when the strict route answers the reply (it answers a told fact first: host
routing precedence). After a probe the organ already answered, the direct read must reproduce that turn's trace.

**Vocabulary pre-screen (host functions only, no brain, 2026-10-13).** The comprehension monitor drops a teach
sentence whose two nouns it judges ambiguous (why `the bird eats the worm` never stored in `cat_told`). Every new
teach sentence was screened with `ComprehensionProductionOrgan.competent` and the B3 extractor + `lemma_verb`: all
new agents (eel, toad, newt) are outside the animacy lexicon, so the monitor passes the sentences through (as for
trout, bat, canary), and every sentence extracts to the intended (agent, lemma, patient). A CPU test pins the
extraction.

## 3. The bars (fixed before any run)

**OVERRIDE, per dev seed.**
- (O1) organ: after `the bat swims in the pond`, the direct organ read of (bat, swim) DECIDES pond (reason
  `reached`: pond clears the floor and beats lake by the organ's own 1.25x margin).
- (O2) reply: the reply to the second `what does the bat swim` names pond (recalled or flagged) and names lake
  nowhere (not in either SVO field, not as a word of the answer).
- (O3, O4) the same for the eel and river (graded session), when the eel's pre-override guess reached lake. When it
  did not, the eel case is VACUOUS: reported, it decides nothing.
- Seed precondition: the first `what does the bat swim` was an organ read that reached lake (a guess existed to
  override; the 07-08 cancellation probe selected its exception member the same way). Without it the seed is
  UNDEFINED.

**GRADED, per dev seed.** The lake rate recorded in the organ trace of `what does the <target> swim` rises STRICTLY
with the shared traits: newt (1) < toad (2) < eel (3). A tie fails. The rate is spikes per neuron per step over the
read window, recorded whether or not it clears the floor (the trace's `admitted` list).

**Decision, each probe.** PASS iff all 8 dev seeds (1-8) are DEFINED and pass; UNDEFINED if any seed is undefined
(re-run that shard once); otherwise FAIL naming each seed and clause. Deterministic per seed (no OU noise in the
organ), so one failing seed is a defect, not noise.

**Bar-sanity (n 1000 each, both OK, run before any probe data):**
`bar_sanity_record_probes_override.json` (perfect 1.000 and perfect-with-a-vacuous-eel 1.000; template, one-seed
template / tie / reply-names-lake / eel-template and random all 0.000) and `bar_sanity_record_probes_graded.json`
(perfect 1.000; flat, count-only, inverted, one-seed top tie and random all 0.000). Power note, recorded before
data: the reported `info_noisy_grading` brain (a real grading of 0.01 per shared trait with per-target slot noise of
sd 0.004) passes the every-seed strict bar only 0.419 of the time. So a GRADED FAIL on one or two seeds says the
per-trait step is not large against slot heterogeneity; the RESULT must report the per-seed steps, and the bar is not
loosened after the data.

**DEFINED (tools.verdict preconditions; data quality only, the bar outcome is never a precondition).** Both shards of
the seed valid (no errored turn, source `tiny-demo +LTM`, PYTHONHASHSEED=0, corpus present); the effective
`BRAIN_CATEGORY_INHERIT` and `BRAIN_CATEGORY_INHERIT_HOP` both true; the organ ENGAGED (every teach turn observed by
the organ with its expected SVO, and every probe the bar reads before an override carries an organ query read); every
organ read (trace and direct) records `hop: true` and no lesion; no swallowed organ error; every direct read present
and, where the turn was an organ read, reproducing its trace. The graded probe needs only its own shard.

**Reported, never barred:** lake / told-object rates before and after each override and their ratio; which route
answered the post-override reply (recalled vs flagged); the collateral reads (does the canary, which shares `fly
sky` with the bat, now guess `pond` on swim: the same one-hop inference, kept by the ruling); the graded margins and
replies; the session-end weight audit; and an ORGAN-FEED diagnostic (`--organ-feed`: the same facts fed straight
into a standalone organ, no brain; it should equal the chat organ's reads exactly, which the dev-7 smoke checks).

## 4. Lanes

16 pool jobs (dev seeds 1-8 x 2 probes), numpy, `mem_gb=9`, `max_age_h=72`, ~25 min each; Hermes queues them
(`--emit-pool-jobs <SHA>`; out dir `research/findings/raw/_stage1_c1_generalization/`), then scores with `--score`.
Gate seeds are refused by the module. One dev-7 smoke (both probes) runs first, by Claude, under the brain lock.

## 5. Host-decision debt (for the host-share instrument)

What decides the answer that is not neurons, on this route:
1. **Verb-string admission.** Only frames whose verb STRING equals the asked verb are candidates (`infer`: `fr[0] ==
   v`). The organ never decides which relation is being asked; a spiking relation cue would replace it.
2. **The floor and margin compare.** `FLOOR_RATE` and `TIE_MARGIN` are a host comparison of spike counts (a k-select).
   The spiking commit stage (`docs/drafts/spiking-commit/PLAN.md`) is the named replacement (C1 PLAN section 5.2).
3. Also host on this route: slot recruitment (new word -> next slot), the SVO extraction + `lemma_verb`, the 30-step
   read window, and the strict-route-first precedence. Because of that precedence the reply-level clauses (O2, O4)
   mostly measure host routing; the organ clauses (O1, O3) and the graded rates are the evidence.
4. Probe-only (not production): the direct `infer` calls are instrument reads, outside the chat's routing.

## 6. If a probe fails

File `NEEDS [c1-generalization] RESULT` with the per-seed clause table (ruling). No retune of the organ against
these dev seeds without a new plan; a fix is judged on fresh dev seeds 9-16 first.

## 7. Reads after the bars were committed (2026-10-13; the bars above are unchanged)

- **Organ feed, dev 1-8 (diagnostic, `organ_feed/organ_feed_dev1-8.json`, commit 1b4dfffd4).** Bat O1 8/8; eel O3 7/8
  (seed 5 ties, river/lake 1.24 < 1.25); graded strictly rising on 2/8 (seeds 5, 8): lake rates flat at about
  0.075-0.093 across k. In the graded session the fish frames and even other targets' unshared frames read near the
  told frames (newt reaches `find stone`, the toad's frame), so the frame layer associates nearly everything.
- **Dev-7 smoke through the handler (`smoke_dev7/`).** Both shards DEFINED; every direct read equals the organ feed
  exactly. OVERRIDE seed 7 PASS: bat lake 0.0467 before; after the told pond, pond 0.105 vs lake 0.0517 (reached),
  reply recalled pond; eel river 0.1067 vs lake 0.0833 (reached, ratio 1.28), reply recalled river; canary swim still
  abstains. GRADED seed 7 FAIL: newt 0.0817 = toad 0.0817 < eel 0.0933 (tie). Peak RSS 6.67 / 6.81 GB, wall 19 / 21
  min. The dev 1-8 pool run decides; the organ feed predicts OVERRIDE FAIL on seed 5 (eel) and GRADED FAIL 6/8.
