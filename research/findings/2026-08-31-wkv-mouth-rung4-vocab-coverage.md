---
status: measurement-complete
lane: gap#80
type: finding
date: 2026-08-31
seed-waiver: measurement-only (coverage profile; GO/NO-GO verdicts require 6 seeds)
instrument: direct corpus intersection — count unique words in corpus that are in each checkpoint's vocab; gate-level pass rate computed by applying in_vocab_scope to all 337294 prompts
---

# Mouth rung-4: wider-vocab checkpoint in-vocab coverage (board #80)

**STATUS: GO (measurement complete)** — the V=4000 checkpoint (`bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz`) covers significantly more of the TinyStories corpus than the V=1000 checkpoint.

## Results (corpus: 10450 unique words, 10360 content words)

| Metric | V=1000 (998 words) | V=4000 (3999 words) | Delta |
|--------|---------------------|----------------------|-------|
| All unique words covered | 998/10450 (9.55%) | 3981/10450 (38.10%) | +2983 (28.55%) |
| Content words covered | 917/10360 (8.85%) | 3894/10360 (37.59%) | +2977 (28.74%) |

## Gate-level coverage (in_vocab_scope, 5–200 word prompts, 337294 prompts)

| V=1000 | V=4000 | Delta |
|--------|--------|-------|
| 321812/337294 (95.41% pass) | 334056/337294 (99.04% pass) | +12244 (+3.63%) |

## Interpretation

- The V=4000 checkpoint covers ~4.2× more unique words and ~4.2× more content words than V=1000.
- The gate pass rate improves from 95.41% to 99.04%, meaning ~4% more prompts will be accepted by the WKV mouth rather than falling back to the Qwen path.
- This is a measurement, not a GO/NO-GO on the rung-4 capability itself — the capability (the checkpoint) already exists; this documents its coverage profile.
- The residual ~1% of prompts that fail the gate at V=4000 are out-of-vocab or heavily function-word-dominated.

## Artifact

- Runner: `research/runners/_measure_wkv_ckpt_coverage.py` (stdlib Python + numpy, corpus: `data/corps/tinystories.txt`)
- Raw output: this file

## Next

- The coverage measurement is complete. The next rung in the mouth crutch-burndown is rung-3 (flip `BRAIN_OPEN_ENDED_WKV_MOUTH` default-ON, conditional on `BRAIN_OPEN_ENDED`), or the broader-wiring work as directed by the owner.