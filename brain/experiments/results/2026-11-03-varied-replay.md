# Varied replay at 0.1B: six seeds (2026-11-03)

## Pre-registered analysis (committed before the six-seed aggregate was computed)

The runs were launched overnight from `research/varied-replay-1103` @ b8fd34840 (no criteria were committed then),
so this pre-registers the ANALYSIS, not the run. Seed 42's JSON was opened once before this file was written;
no other seed was looked at and no aggregate was computed. Code on all three nodes matches the branch
(md5 of `sleep_probe.py`, `facts_rewordings.py`, `sleep.py` checked on pool40/41/42).

Run: `python -m brain.experiments.sleep_probe --steps 240 --seed <s>` (0.1B, lr 5e-4, kl_weight 1.0), arms
no_sleep / sleep_true / sleep_varied / sleep_shuffled, six seeds 42/43/44/100/101/102 on pool40/41/42 (CPU).
All read-outs with the hippocampus OFF.

Skeptic's demand: the rewordings of items 5 (dog: "My dog goes by Biscuit.") and 12 (son: "My son was born in
March.") reuse the wording of their PARA question ("What's my dog called?" / "What month was my son born?"), so
paraphrase on those two items can be a surface match, not a reworded memory. Primary read-out is therefore
paraphrase on the CLEAN 10 items (items 1-4, 6-11, 1-indexed).

Criteria (varied replay "helps reworded recall" only if ALL hold):
1. Clean paraphrase (/10): sleep_varied > sleep_true on >= 5 of 6 seeds, and mean gain >= +1.0.
2. Clean paraphrase: sleep_varied > sleep_shuffled on 6 of 6 seeds (shuffled trains the same name words).
3. Recall on the original questions (/12) not hurt: varied mean >= sleep_true mean - 0.5.
4. False recall (/4, never-told questions) not worse: varied mean <= sleep_true mean + 0.5.
Also reported: the all-12 paraphrase, a per-item paraphrase table (seeds passing, per arm), drift (KL).
Outcome labels: all four hold = LEAD (a lead is not a GO; land-result skeptic + lesion still needed);
1 or 2 fails = NOT SHOWN; 3 or 4 fails with 1-2 holding = TRADE (reported as such).
