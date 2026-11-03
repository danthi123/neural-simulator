# 2026-11-03: the live brain says it does not know (abstention inside `Brain`), 0.1B CPU, six seeds

## Pre-registration (written and committed before any evaluation question was asked)

**Question.** After one sleep the live brain answers never-told questions with learned names ("Your horse is called
Pixel"). With a feeling-of-knowing read-out inside `Brain.hear` (switch `Brain(abstain=True)`, default OFF), does
false recall on FRESH never-told questions drop to at most a quarter of the no-abstention rate while it keeps at
least 75% of its correct answers to fresh rewordings of told facts, on every seed?

**Record.** `2026-11-03-false-recall-feeling-of-knowing/README.md` (answer confidence fails, AUC 0.47; question
familiarity works: hippocampal AUC 0.88, cortical cue response AUC 0.79); old track: abstention is a separate
familiarity signal (`research/findings/2026-05-22-...abstention-is-a-separate-familiarity-signal...`,
`2026-06-11-familiarity-gate-v320-GO`). PREREQUISITES "Metacognition": PARTIAL; informative = abstains on
never-told facts. Biology: familiarity (perirhinal) is judged on the cue and is separate from recollection;
confabulation is a monitoring failure.

**Mechanism (brain path, `brain/brain.py`, `brain/hippocampus.py`).** For a request turn (`not _is_telling`), when
`abstain` is on: hippocampus on -> `Hippocampus.familiarity` (best recall score over all stored episodes, no side
effects); hippocampus off -> `LiveCortex.cue_response` (largest per-token log-prob gain of the question's tokens,
consolidated synapses on minus off, from the birth base). At or above the bar: unchanged path. Below it: no pattern
completion, no reinstatement, and the cortex reads the read-out as one fixed interoceptive line (`NOT_KNOWN`, declared
scaffold) before `Assistant:`; the reply is the cortex's own continuation, sampled exactly as in chat. No reply
string exists in the host. The wording of `NOT_KNOWN` may be chosen on seed 42's FITTING never-told questions only,
before any evaluation run, and is then frozen.

**Bars (fitting set only).** Fitting set = the false-recall branch's 36 questions (12 told, 12 paraphrases, 12
never-told "my X"), measured in each seed's own slept brain. Rule: answer iff signal >= t, t = the value maximising
(kept told+para) - (kept never-told) among those keeping >= 85% of told+para. Hippocampus: one bar, pooled over the
six seeds (the layer-0 signal does not depend on sleep). Cortex: leave-one-seed-out (seed s uses the bar fitted on the
other five seeds' fitting questions).

**Setting.** 0.1B, live `Brain` (sleep 240 steps, lr 5e-4, KL 1), day 1 tells the 12 facts of `facts.py`, restart =
one night (15 episodes incl. 3 identity). Each evaluation question is asked of the same day-2 brain (state and
episode strengths reset per question), chat sampling seeded by (seed, question index), identical across arms.
Arms: hippocampus on/off x abstention off/on. Method 2 (declared scaffold, confidence_probe's 8 "I don't know"
pairs replayed at strength 1.0 in the same night, read-out off): hippocampus on/off.

**Fresh sets** (`brain/experiments/abstain_in_brain.py`, none in the fitting set; no topic shared with the fitting
never-told questions or the method-2 pairs):
- `fresh_never` 24: 16 untold topics in varied forms ("What did I say about my rabbit?", "Do you remember my
  husband's name?", "Tell me about my garden.", "Which band do I like?", ...) + 8 told topics with untold attributes
  ("What colour is my cat?", "What is my brother's name?", ...).
- `fresh_told` 12: rewordings of each told fact in the same varied forms ("Do you remember my cat's name?", "Tell me
  what car I drive.", "What did I say the bakery I work at is called?", ...).
- `general` 8: general-knowledge/chat requests (reported only: collateral abstention).

**Scoring (strict, automatic).** Correct = the fact's name in a sentence of the reply that is not a first-person echo
(a sentence with I/my/me and no you/your gets no credit). False recall = any learned name (word-start match) other
than the question's own fact name (for told-topic/untold-attribute items, the topic's own name is allowed).
Abstained = no false recall, not correct, and a non-answer marker ("don't know", "haven't told", "can't", ...).

**Pass criteria, per hippocampus arm (on, off), every one of the six seeds 42/43/44/100/101/102:**
- C1: false recall on `fresh_never` with abstention ON <= 0.25 x with abstention OFF (0 <= 0 passes, flagged
  uninformative);
- C2: correct on `fresh_told` with abstention ON >= 0.75 x with abstention OFF (a seed with 0 correct OFF fails C2:
  nothing shown kept).
An arm is a PASS only if C1 and C2 hold on 6/6 seeds. Method 2 is scored with the same C1/C2 against the same seed's
abstention-off arm (comparison only). Even a PASS is a lead, not a GO (one sleep, 0.1B, 12 facts, keyword scorer).

**Order of work.** The sleep+fitting phase started a few minutes before this file was committed (it computes only
fitting-set signals; no evaluation question is asked in it).

**Amendment before any evaluation run (mechanism frozen on seed 42's FITTING questions, as allowed above).**
Prototype (`brain/experiments/abstain_wording_proto.py`, logs `dev_wording_proto_*.log`; the ungated log came from the same script before the gate existed, variants off/v1/v2/v3; read-out forced to "not known" on the 12 fitting never-told + 3 told + 2 general
questions, seed 42 slept brain): with the consolidated synapses still on, the `NOT_KNOWN` line made it WORSE (12/12
learned-name confabulations, e.g. "Nova: My horse is called Pixel."; without the gate 8/12): the confabulation lives
in the consolidated synapses, and a line of text does not override them. With retrieval gated (no reinstatement
AND the reply spoken with the consolidated synapses off) learned-name confabulation was 0/12 with no line, 0/12 with
either line; the line added speaker confusion ("Nova: What is your fish called?") and no more abstentions (4-5/12
non-answers in all three). Frozen mechanism: below the bar, no reinstatement and the reply is spoken without the
consolidated synapses; NO interoceptive line, so no English text of any kind is added by the host. What the brain
then says is the pretrained cortex's own reply: sometimes "I'm not sure of the name of your horse", sometimes a
refusal, sometimes an invented non-learned answer ("The name of my parrot is Amelia"). The strict scorer counts only
LEARNED names as false recall, so invented non-learned answers are reported separately (`abstained` vs other) and
quoted, not hidden. Scaffold declared in `brain/SCAFFOLDS.md` (all-or-none retrieval gate). Criteria unchanged.

## Results

(to be filled from the result files)
