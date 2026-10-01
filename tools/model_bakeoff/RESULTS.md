# Model bake-off results (2026-09-30 to 10-01) and the Hermes model set it chose

**Decision (owner 2026-10-01): cloud main.** Hermes drives on MiMo v2.6 Flash; the workstation's local Qwen3.8-27B is
the automatic fallback; MiMo v2.6 Pro serves delegated judgment work; DeepSeek V4.1 Flash serves auxiliary calls
(compression, titles, web extraction, session search), never design. Every OpenRouter request carries
`provider.data_collection: deny` (owner: no provider that may train on or store our data).

**Method.** Real project tasks replayed from their base commits (`suite.json`): implement (hidden tests), verdict
(command-graded), review and design (rubric with a reference answer from what the project actually did). Each run in
its own container with the leak audit (`run.py`). Rubric answers graded blind by two Claude graders (Opus and
Sonnet), the mean taken; a hard-zero disagreement is adjudicated against the rubric text. Per-run data:
`results/runs_2026-10-01.csv` (screen 1's per-run data was not kept; its summary is below).

**Spend:** $8.87 of OpenRouter in total (owner budget $15).

## Screen 1 (6 models x 6 tasks, before the per-run write-root fix)

| Model | Real $/run | Review | Design | Verdicts | Implement |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 0.02 | .94 | .81 | 2/2 | 1/1 |
| MiMo v2.6 Flash | 0.04 | .97 | .83 | 2/2 | – |
| MiMo v2.6 Pro | 0.08 | .97 | .81 | 2/2 | – |
| GPT-6 Luna | 0.07 | .86 | .65 | 1/2 | 1/1 |
| DeepSeek V4 Pro | 0.16 | .92 | .78 | 2/2 | – |
| GLM-5.3 | 0.37 | .96 | .73 | 2/2 | – |

## Finalists, all on the fixed harness (screen1b + stage2 + impl2) and the local model (local1)

| Model | Implement | Verdicts | Reviews (mean, n) | Designs (mean, n) | Rubric hard zeros | Min/run |
|---|---|---|---|---|---|---|
| MiMo v2.6 Flash | 6/6 | 4/5 | 0.86 (3) | 0.71 (5) | 0 | 16-33 |
| DeepSeek V4.1 Flash | 6/6 | 4/5 | 0.85 (3) | 0.46 (5) | 2 | 1-6 |
| MiMo v2.6 Pro | not run | 3/4 | 0.91 (3) | 0.72 (5) | 0 | 12-35 |
| Local Qwen3.8-27B (IQ4_NL, MTP) | 2/2 | 2/2 | 0.85 (1) | 0.63 (1) | 0 | ~21 |

Like for like on the six screen tasks: implement and verdicts tie (2/2, 2/2); review-gate local 0.85 vs 0.94-0.99
cloud; design-d5 local 0.63 vs 0.80-0.91 cloud. The local model's rubric n is small; the gap points the same way as
public benchmarks.

DeepSeek V4.1 Flash's two hard zeros (adjudicated with the Sonnet grader; Opus had scored 51 and 45, both below the
item-1 bar anyway): re-enabling homeostasis changes read-time conditions (curiosity-d6wm), and a quench recruited by a
host-timed drive co-timed with the challenger (gnw-eviction) is host-triggered eviction. It crossed the brain-based-only
line in 2 of 5 designs, so it is kept off design work.

## Why cloud main (serving research, 2026-10-01)

Qwen3.8-27B is the best model that fits one 3090 (no in-VRAM, MoE-offload or engine alternative beats it here). As
the driver it holds the GPU (research GPU jobs and the driver exclude each other), serves one request at a time,
runs the card at the sustained load under which it fell off the bus on 09-26, and draws an estimated $1-2.5/day of
power; a cloud main costs an estimated $1-3/day (both estimates, not measured). A cloud main keeps Hermes working
while the owner games and during GPU windows. No second local model is added: nothing fits beside the 27B, and a
second llama-swap model id unloads the first.

## Harness lessons

- `--max-spend` reads the OpenRouter KEY-WIDE usage counter: run paid batches one at a time, or a concurrent batch's
  spend counts against this one (stage2 stopped at its $4 cap while screen1b ran beside it).
- Audit hits on a path that does not exist (a mistyped own run dir, `cd /bk/runners 2>/dev/null`) leak nothing; they
  are cleared by hand with a note (`audit_cleared` in the CSV).
