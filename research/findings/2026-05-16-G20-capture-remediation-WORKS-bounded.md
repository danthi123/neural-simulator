# Capture-quality remediation WORKS for under-recall — real, artifact-safe, but bounded (honest)

## TL;DR

The mechanism-grounded fix hypothesis is **confirmed**: boosted
re-capture of an under-recalling per-concept engram tag fixes its
self-recall, on an existing bridge, with NO retrain and NO
`generate_sparse_patterns` change (artifact-safe). **Bounded
honestly:** n=1 under-recaller; this fixes *self-recall*, NOT (yet)
the 86.7% *cross-bridge* number — and most cross-bridge misses are
on other bridges/indices, so under-recall is a real but PARTIAL
contributor, not the whole story.

## Result (bridgeA_nouns_sparse64, seed 42)

Baseline probe of all 64 indices: **only 1/64 under-recall**
(self_rank>1) — idx-12 `ball` (self_cum 161, self_rank 20). The
other 63 self-win. After boosted re-capture (teacher 400 vs
training's 100; 250 steps vs 100):

| idx | word | self_cum | self_rank |
|---|---|---|---|
| 12 | ball | 161 → **1361** (+1200) | 20 → **1 (FIXED)** |

1361 exceeds the all-index median (654) and rivals the robust
control `apple` (1157). The fix is **mechanism-consistent** (the
original tag under-captured the pattern; boosted capture spans
enough of it to reignite) — not a measurement artifact.

## Honest bounds (what this does NOT show)

1. **n = 1.** Exactly one under-recaller existed on this bridge.
   "1/1 fixed" is true but not a generalizable rate. Other bridges'
   weak indices (idx-42 `touch`/bridgeB, idx-10 `every`/bridgeE from
   the cross-benchmark analysis) were not probed here.
2. **Self-recall ≠ cross-bridge retrieval.** The probe measures
   stim-own-tag → own-pattern-fires. The 86.7% benchmark measures
   encode `A is B` cross-bridge → query A → B fires. Fixing
   per-concept self-recall is necessary-plausible but **not shown**
   to lift the cross-bridge number. That requires remediating +
   saving all 5 bridges and re-running the cross-bridge benchmark —
   not done here.
3. **Under-recall is a PARTIAL contributor.** Only 1/64 on bridgeA
   under-recalls, yet cross-bridge miss rate is ~13%. So most
   cross-bridge failures are NOT bridgeA self-under-recall; they
   involve other bridges/indices and plausibly a partly different
   sub-mechanism. Under-recall remediation addresses one real piece,
   not the entire 13%.

## What is genuinely established

- The dynamical under-recall mechanism is **confirmed** (probe) AND
  **remediable** (this experiment) via an **artifact-safe, post-hoc,
  per-index** capture-quality gate — exactly the lever the
  under-recall finding predicted, and the one that sidesteps the
  reproducibility-invariant concern that kept recovery flagged.
- This converts the flagged recovery from "open dynamical question"
  to "validated remediation primitive (n=1) + a clear honest next
  measurement": apply the gate to all 5 bridges' under-recallers,
  save, re-run the cross-bridge + sentence benchmarks, measure the
  actual end-to-end lift (if any). That is the correctly-scoped,
  honestly-bounded next step for the dedicated session.

## Recommendation (not overclaimed)

A post-hoc `capture-quality gate` (after training: probe each
per-concept tag's self-recall; re-capture any with self_rank>1 at
boosted drive) is a **safe, validated-at-n=1 production add-on**.
Its end-to-end conversational impact (cross-bridge / sentence
benchmarks) is **unmeasured** and should be quantified before it is
claimed to improve the 86.7%/80% headline numbers. Do not state it
"fixes" the cap until that measurement exists.

## Files

- `research/runners/g20_capture_remediation.py`
- `research/findings/raw/g11_bg/g20_capture_remediation.json`
- Chain: …→ dynamical under-recall [identified] → **remediation
  WORKS, bounded [here]** → (next: end-to-end cross-bridge impact,
  flagged-task-scoped).
