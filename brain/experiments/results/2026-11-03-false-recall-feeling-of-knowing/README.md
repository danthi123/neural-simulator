# False recall after sleep: is there a feeling-of-knowing signal? (2026-11-03, 0.1B, six seeds)

**Question.** After sleep the cortex answers never-told questions ("What is my horse called?") with learned names.
Before training a fix (method 2), does a no-training signal tell real recall from confabulation?

**Record used.** The old track found abstention is a separate familiarity signal, not a property of recall
(`research/findings/2026-05-22-...abstention-is-a-separate-familiarity-signal...`, `2026-06-11-familiarity-gate-v320-GO`).
Biology agrees: familiarity (perirhinal) is separate from recollection (hippocampal), and confabulation is a failure of
monitoring, not of memory. PREREQUISITES "Metacognition": PARTIAL, informative = abstaining on never-told facts.

**What ran.** `brain/experiments/confidence_probe.py`, the sleep_true arm (same seed and replay as sleep_probe, so
the answers match), hippocampus off for answering. For each of 36 questions (12 told, 12 paraphrase, 4 NEVER_TOLD,
8 more never-told) it logs:
- the answer's own confidence: mean, min and first-token log-prob, the name token's log-prob, entropy, and the
  adapter on-minus-off shift;
- the cortex's cue familiarity before answering: how much the sleep-changed synapses raise the log-prob of the
  question's own tokens (`cue_gain`, `cue_gain_max`) or the KL (`cue_kl`);
- the hippocampus's familiarity score for the question (`Hippocampus.recall`, the told sentences stored).

`abstain_analysis.py` gives each signal's AUC and an abstention read-out: answer only if the signal is at or above t.
t is set by Youden's index on correct vs confab. **Strict** = the held-out item's seed AND question are both unseen
when t is fitted.

    python -m brain.experiments.confidence_probe --model rwkv7-g1d-0.1b-20260129-ctx8192.pth --steps {240,480} --seed S --threads 3 --out sS_st{steps}.json
    python -m brain.experiments.abstain_analysis <dir> --out summary.json

The runs used pool40/41/42 on CPU, seeds 42/43/44/100/101/102. The 12 runs were done twice: the second batch added
the cue read-outs, and its answers matched the first batch on every item.

**Results** (six seeds pooled; correct = told + paraphrase answered with the right name; confab = never-told
question answered with a learned name, out of 12 per seed; fr4 = the original 4 NEVER_TOLD).

| read-out (strict) | 240 steps: correct / confab / fr4 | 480 steps: correct / confab / fr4 | AUC correct vs confab (240 / 480) |
|---|---|---|---|
| none | 36 / 42 / 13 | 44 / 54 / 18 | - |
| answer mean log-prob | 24 / 25 / 9 | 22 / 21 / 5 | 0.63 / 0.65 |
| answer min log-prob | 15 / 15 / 6 | 29 / 22 / 10 | 0.55 / 0.54 |
| name-token log-prob | 32 / 30 / 10 | 33 / 37 / 11 | 0.47 / 0.47 |
| answer entropy | 32 / 31 / 10 | 19 / 14 / 5 | 0.58 / 0.62 |
| cortical cue familiarity (`cue_gain_max`, hippocampus OFF) | 20 / 2 / 1 | 23 / 8 / 4 | 0.79 / 0.78 |
| hippocampal familiarity (learned bar ~0.22) | 29 / 12 / 2 | 37 / 15 / 3 | 0.88 / 0.88 |
| hippocampal familiarity at the current bar 0.12 | 35 / 39 / 10 | 43 / 49 / 13 | - |

The per-seed mean AUC agrees with the pooled AUC: cue_gain_max 0.80 / 0.79, hippocampus 0.89 / 0.89, answer
log-prob 0.72 / 0.65.

**What it shows.**
1. **Method 1 as specified fails.** The answer's own confidence is not a usable feeling-of-knowing signal. A
   confabulated name is said as fluently as a recalled one (name-token AUC 0.47). Thresholds keep about as many
   confabulations as correct answers.
2. **Two familiarity signals on the question do separate the classes. Both are leads, not GOs.**
   - The hippocampal score with a learned bar (about 0.22 instead of 0.12) keeps 81-84% of correct answers and removes
     71-72% of confabulations at both step counts.
   - With the hippocampus off, the cortex's own cue familiarity (how strongly the sleep-changed synapses respond to
     the question) removes 85-95% of confabulations but drops about half the correct answers.
   - This fits the biology: "do I know this" is judged on the cue (familiarity), not on how fluent the answer is.

**What it does not show.**
- Neither signal is wired into `Brain`.
- The thresholds are fitted on 36 questions in one template ("my X"). They need a fresh question set.
- The hippocampal score is lexical and depends only on the question, so it is identical across seeds. It flags
  "never told" mainly by topic word overlap.
- Neither signal fixes intrusions: told questions answered with the wrong learned name (wrong_named 66-69) stay.
- Untested at 0.4B and 1.5B.
- The method-2 scaffold arm (`--dont-know`, declared "I don't know" replay pairs) is built but not run.
