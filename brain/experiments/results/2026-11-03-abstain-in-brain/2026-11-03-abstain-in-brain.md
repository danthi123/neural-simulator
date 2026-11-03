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

Files: `fit_s<seed>.json` (fitting signals), `bars.json`, `eval_main_s<seed>.json` (every reply), `report_main.json`
(C1/C2), `eval_dk_s<seed>.json` + `report_all.json` (method 2). Bars: hippocampus 0.230 (one bar); cortex
1.08 / 1.02 / 0.73 / 0.73 / 0.73 / 0.73 (s42/43/44/100/101/102, leave-one-seed-out). Run: workstation CPU,
`python -m brain.experiments.abstain_in_brain {sleep,fit,eval,report}`.

Per seed (fresh_never false recall out of 24, OFF -> ON; fresh_told strict correct out of 12, OFF -> ON):

| seed | hippo ON: false recall | hippo ON: correct | hippo OFF: false recall | hippo OFF: correct |
|---|---|---|---|---|
| 42 | 12 -> 7 | 3 -> 3 | 8 -> 5 | 1 -> 1 |
| 43 | 11 -> 4 | 5 -> 5 | 9 -> 7 | 3 -> 3 |
| 44 | 7 -> 4 | 4 -> 4 | 3 -> 3 | 2 -> 2 |
| 100 | 12 -> 6 | 3 -> 3 | 7 -> 4 | 2 -> 2 |
| 101 | 10 -> 6 | 4 -> 4 | 10 -> 10 | 5 -> 5 |
| 102 | 11 -> 7 | 6 -> 6 | 9 -> 9 | 8 -> 8 |
| **sum** | **63 -> 34 (-46%)** | **25 -> 25** | **46 -> 38 (-17%)** | **21 -> 21** |
| C1 (<= 1/4) / C2 (>= 75%) | **0/6** / 6/6 | | **0/6** / 6/6 | |

**Pre-registered verdict: FAIL on both arms** (C1 0/6 seeds; needs 6/6). C2 holds trivially: almost no told
question fell below the bar (hippocampus 6/72 gated, cortex 0/72).

Diagnostics (reported, not criteria):
- **The gate works; the signal does not.** Among the answers the read-out gated, learned-name false recall is
  0 (hippocampus 0/72 gated, cortex 0/19 gated); the same questions without the read-out had 29 and 8. All residual
  false recall comes from never-told questions the signal judged familiar: hippocampal familiarity passes 12/24 fresh
  never-told questions on every seed (it is lexical and seed-independent: "Do you remember my husband's name?",
  "What colour is my cat?", "What is my brother's name?" share words with told facts), and the cortical cue response
  passes 18-23/24 (it reacts to the "my X" / "Do you remember" frame that sleep trained, not to the content).
  The fitted bars do not transfer from the 36 "my X" questions to fresh forms.
- **Gated is not "I don't know".** Some gated replies are non-answers ("I'm not sure of the name of your horse",
  "I'm sorry, but I can't assist with that"): never-told non-answers went 0 -> 14 of 144 (hippocampus arm) and
  6 -> 9 of 144 (cortex arm). Most gated replies are the pretrained cortex inventing a non-learned answer ("My
  grandmother's name is Joanna.", "My stepfather is a software engineer."). The strict scorer does not count those as
  false recall; they are still confabulation, of a different kind (no intrusion of lived facts).
- **General questions improve, none lost to abstention.** Learned-name intrusions into general-knowledge replies
  fell 38 -> 17 (hippocampus) and 16 -> 11 (cortex); gated general questions are answered by the pretrained cortex
  ("Answer: Paris" instead of "Lyon is called Lyon").
- Strict told recall is low at baseline (25/72, 21/72): many answers are first-person echoes ("My grandmother lives in
  Oslo.") or the identity loop ("Of course, you're Dan, my friend."), which get no credit.
**Method 2 (declared scaffold: 8 "I don't know. You haven't told me that." pairs replayed in the night at strength 1;
read-out off), same seeds, against the main brain's read-out-off arm (`report_all.json`):**

| | false recall fresh_never (OFF -> M2) | strict correct fresh_told | never-told non-answers | general non-answers | C1 / C2 seeds |
|---|---|---|---|---|---|
| hippocampus on | 63 -> 38 | 25 -> 4 | 62/144 | 16/48 | 1/6 / 0/6 |
| hippocampus off | 46 -> 7 | 21 -> 3 | 103/144 | 38/48 | 4/6 / 1/6 |

Method 2 FAILS too, the other way round: the replayed string becomes the cortex's default reply to "my X"-shaped
questions and to general ones. With the hippocampus off it cuts false recall by 85% but answers told facts with "I
don't know. You haven't told me that." (correct 21 -> 3) and refuses 38/48 general questions. With the hippocampus
on, reinstated episodes still pull learned names in (38). It is a memorised string, not a judgement.

## Verdict and next step

**FAIL (pre-registered), both methods.** No method meets "false recall at most a quarter, told answers kept >= 75%"
on 6/6 seeds. What was learned:
1. The brain-side mechanism is sound: when the read-out says "not known", gating retrieval (no reinstatement,
   consolidated synapses off for that reply) removes learned-name intrusion completely (0 of 91 gated answers) and
   costs no general knowledge. The default-OFF switch is byte-equal to the old brain (tests pass).
2. The failure is the signal, not the gate: both question-familiarity signals, fitted on one question template, do
   not transfer to new forms (hippocampal: lexical word overlap; cortical: responds to the trained "my X" frame).
   This is the record's own warning ("fitted on 36 questions in one template") confirmed.
3. Without the consolidated content, the 0.1B pretrained cortex rarely says it does not know; mostly it invents a
   plausible non-learned answer. Saying "I don't know" in its own words needs a cortex that has that behaviour
   for personal questions (scale: re-check at 0.4B/1.5B) or a learned link from the signal to that behaviour.
Per PREREQUISITES "Metacognition": PARTIAL; a familiarity signal that generalises over question forms (a semantic,
not lexical, key) is missing: an expected failure for the signal, not a verdict on abstention.

**Next step:** swap the familiarity key for the meaning-based key (research/meaning-selection-1103, S_all) and
measure transfer on these same fresh sets (bars refit on the fitting set only); separately, check at 0.4B whether
the gated cortex says "I don't know" more often. Keep `abstain` OFF by default.

Note: above-bar replies in the ON arms were reused from the OFF arm (identical path, `tests/test_brain_abstain.py`).
