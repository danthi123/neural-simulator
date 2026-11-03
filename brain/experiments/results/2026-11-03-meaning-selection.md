# 2026-11-03 Meaning-based selection: which key picks the right episode for a reworded question

Milestone 2026-11-30 "recall from reworded questions (meaning-based selection)". Hippocampus ON, no sleep of the
cortex. Record: `brain/research/2026-11-02-record-brief-consolidation-and-selection.md` part B (learned keys beat fixed
keys; >= 40 facts; confusables; shuffled-key control). `tools/before_you_build.sh` refused (stale corpus index on W);
the record brief above is the record mining for this capability.

## What ran

`brain/experiments/selection_probe.py` (variants in `brain/hippocampus_keys.py`, behind `KeyedHippocampus(key=...)`;
Hippocampus defaults untouched; `key="S0"` reproduces the default selection, `tests/test_hippocampus_keys.py`).
48 teacher-written facts (`facts_bank.py`, 24+ confusable pairs). Per seed: 24 stored, 12 untold (their questions + 12
never-told questions are the "should not fire" cues), 12 held out for the teacher-key arm. Cues per stored fact:
same-words (orig), light rewording (para_a), heavy rewording (para_b). Selection = top-1 centred cosine, forced (no
familiarity bar). Recall = selected episode's delta reinstated at gain 4, greedy reply, answer word present. Seeds
42/43/44/100/101/102 vary the fact split and learned-key init.

    python -m brain.experiments.selection_probe --model /home/dant123/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
        --threads 4 --cache _scratch/brainv2/meaning-selection-1103/cache --out sel_0.1b_6seeds.json   # and the 0.4b .pth
    python -m brain.experiments.selection_summary brain/experiments/results/2026-11-03-meaning-selection/*.json

Keys: `S0` default (layer-0 matrix-memory delta); `S_mid`; `S_all` (every layer's matrix-memory delta, each layer
normalised, concatenated); `att_all`/`ffn_all` (token-shift states); `last_all` (last-token residual); `pool_mid`/
`pool_all` (residual averaged over the turn); `Ssk_all` (fixed random 256-d projection per layer of S_all); learned
low-rank correction (A=0 at start) on `pool_all` (`learned_*`) or `Ssk_all` (`learnedS_*`), trained by InfoNCE on pairs:
`_selfq` = stored episode vs a question the cortex itself generated while replaying it (prompt scaffold, no teacher
text); `_shuf` = same with questions shuffled across episodes (control); `_teacher` = teacher questions of 12 disjoint facts.

## Results (mean over 6 seeds, min-max; chance 0.04)

| model | key | sel orig | sel para_a | sel para_b | recall orig | recall para_a | recall para_b | told-vs-untold AUROC |
|---|---|---|---|---|---|---|---|---|
| 0.1B | S0 (default) | 0.92 | 0.57 | 0.33 (0.25-0.38) | 0.67 | 0.51 | 0.22 | 0.62 |
| 0.1B | **S_all** | 0.99 | 0.79 | 0.69 (0.58-0.83) | 0.72 | 0.70 | 0.46 (0.38-0.62) | 0.65 |
| 0.1B | pool_all | 0.88 | 0.60 | 0.38 | 0.64 | 0.51 | 0.28 | 0.58 |
| 0.1B | learnedS_selfq | 0.94 | 0.76 | 0.58 | 0.69 | 0.67 | 0.41 | 0.67 |
| 0.1B | learnedS_shuf | 0.52 | 0.27 | 0.19 | 0.34 | 0.25 | 0.15 | 0.58 |
| 0.1B | learnedS_teacher | 1.00 | 0.88 | 0.76 | 0.74 | 0.78 | 0.52 | 0.68 |
| 0.1B | oracle / none | | | | 0.74 / 0 | 0.88 / 0 | 0.72 / 0 | |
| 0.4B | S0 (default) | 0.45 | 0.10 | 0.10 (0.04-0.17) | 0.42 | 0.10 | 0.05 (0.00-0.08) | 0.54 |
| 0.4B | **S_all** | 0.99 | 0.82 | 0.76 (0.67-0.88) | 0.86 | 0.70 | 0.66 (0.54-0.79) | 0.62 |
| 0.4B | pool_all | 0.95 | 0.69 | 0.52 | 0.82 | 0.57 | 0.40 | 0.57 |
| 0.4B | learnedS_selfq | 0.85 | 0.67 | 0.57 | 0.73 | 0.58 | 0.49 | 0.59 |
| 0.4B | learnedS_shuf | 0.40 | 0.24 | 0.19 | 0.35 | 0.22 | 0.14 | 0.53 |
| 0.4B | learnedS_teacher | 0.99 | 0.87 | 0.81 | 0.87 | 0.75 | 0.71 | 0.63 |
| 0.4B | oracle / none | | | | 0.87 / 0 | 0.88 / 0 | 0.88 / 0 | |

Token-shift and last-token keys are near chance on both sizes (every turn ends on the same "\n\n" token). Per seed:
S_all beats S0 on selection and recall for both rewordings on 6/6 seeds at both sizes. Self-question learned key vs
its own fixed base: 0.1B learnedS_selfq - Ssk_all -0.06 (better 2/12 seed x cue); 0.4B -0.15 (0/12); vs the shuffled
control +0.44 / +0.40 (12/12). Full tables, per-layer sweeps and per-seed deltas: `selection_summary` on the JSONs.

Reply lesion (0.4B, seed 42, heavy rewording, S0 vs S_all selection, same question): "What do I call my prickly plant?"
S0 reinstates the Bristol episode -> "Bristol?"; S_all -> "Spike, Spike."; no memory -> "That's a great question!...".
"In which city did my dad's life begin?" S0 -> "Ingrid, Ingrid." (piano teacher episode); S_all -> "Porto."

## What it shows

- The default key is the bottleneck for reworded questions, and it breaks with scale (0.4B: 10% selection on
  rewordings, 45% even on same words). Using the whole matrix-memory delta (all layers, each normalised) restores it:
  0.4B reworded recall 0.05-0.10 -> 0.66-0.70, about 75-80% of the oracle ceiling (0.88). No training, no host text;
  the key is the delta the hippocampus already stores, so no extra memory.
- Wrong selection is a confabulation source: S0's misses reinstate another episode and the cortex speaks its name.
- The learned key works as a mechanism (12/12 over the shuffled control: the replay pairs carry the signal) but the
  brain's own questions as pairs, 24 episodes, 300 steps, do NOT beat the fixed S_all key at either size; teacher pairs
  on disjoint facts add +0.05-0.07 selection on heavy rewordings over the fixed key (a scaffold, small).

## What it does not show

- Selection is forced top-1. Told-vs-untold AUROC is 0.62-0.65 for S_all: a familiarity bar on this score will NOT stop
  untold questions recalling something (false recall stays open; FAMILIARITY/MARGIN were fit for S0 and must be refit).
- 24 stored episodes, one-sentence tellings, teacher-written rewordings; no 1.5B run; no conversation context in the cue.
- Per-layer sweep (S layer 8 alone 0.78 at 0.1B, layer 7 0.81 at 0.4B on para_b) was read on test data: exploratory only.
- Not run through the live Brain/weekly transcript.

## Recommendation

Adopt `S_all` as the hippocampus key (KeyedHippocampus(key="S_all"), or `Cortex.key_of` over all layers), refit
FAMILIARITY/MARGIN per model on dev facts, then wire it into `Brain` and check the weekly transcript. Park the learned
key: revisit only with many more episodes or pairs from questions the brain actually answered correctly.
