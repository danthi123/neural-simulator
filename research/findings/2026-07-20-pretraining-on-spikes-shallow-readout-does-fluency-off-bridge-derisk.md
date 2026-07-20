# Pretraining-on-spikes — cheap-first de-risk: a SHALLOW exact-delta read-out does FLUENCY (ppl ~40) over a fixed reservoir

**Date:** 2026-07-20 · **Status:** off-bridge cheap-first de-risk GO — the shallow exact-delta read-out that closed
the grounded task on-substrate ALSO learns TinyStories FLUENCY (ppl ~40, close to the multi-layer ~35), so the WKV
cortex's PRETRAINING is learnable on-substrate by the same pure local rule. Off-bridge, NO `sim/` edit. First step of
the owner-steered "biologize the WKV's FULL learning on the one shared spiking substrate" frontier.

## The frontier (owner steer 2026-07-20)

"Fully closing all gaps INHERENTLY means fully-spiking, one brain, single shared substrate as the end goal." The
grounded-render TASK learning is now on-substrate by a pure exact delta rule (~0.94, `2026-07-20-onbridge-ssm-readout-
learning-...`); the cortex's PRETRAINING (its TinyStories fluency) is still off-bridge BPTT — the deeper spiking-purity
item. This de-risk asks the read-out-expressiveness question CHEAPLY (off-bridge) BEFORE the slow on-bridge build:
does a SHALLOW exact-delta read-out learn full-vocab FLUENCY over a FIXED reservoir, or does fluency need the
multi-layer?

## Result — a shallow exact-delta read-out does fluency (`_gap_pretraining_shallow_fluency_derisk.py`)

The WKV cortex (emb/Wv/decay) is the FIXED reservoir (detached → no BPTT-through-time). A SINGLE-linear read-out
`logits = state @ Wsl^T + h @ Wh^T` (state = the leaky reservoir state, h = the current token) trained on TinyStories
next-token by the EXACT DELTA rule (the softmax gradient is LOCAL + exact for a single output layer — no FA, no weight
transport, no BPTT, plain SGD). Held-out TinyStories ppl:

| config | ppl | vs |
|---|---|---|
| shallow exact-delta (pretrained reservoir) | **~40** (12000 steps; 47 @ 6000, still dropping) | BPTT ceiling ~29.5; multi-layer FA/KP ~35 |
| — **NO current-token** (state-only) | 62.95 | ⇒ the current-token term is LOAD-BEARING (40.6 vs 63) |
| — **RANDOM reservoir** (Rung B for fluency) | 58.06 | working fluency over a random reservoir; the pretrained features help (40.6 vs 58) |

## Read-out

- **⇒ the WKV cortex's PRETRAINING (fluency) is shallow-learnable on-substrate by the SAME pure exact delta rule that
  closed the grounded task** — ppl ~40, CLOSE to the multi-layer BPTT ceiling (~29.5-35), far below chance (11598). No
  BPTT, no weight transport, no FA, no adaptive optimizer. The current-token term is load-bearing; a random reservoir
  still works (ppl ~58, the pretrained input features add ~18 ppl).
- **Honest:** the shallow ppl ~40 is a bit above the multi-layer ~35 (and the BPTT ~29.5) — a modest read-out-
  expressiveness gap (unlike the grounded copy, where the shallow read-out reached the ceiling). The multi-layer gated
  read-out (FA/KP) is the top-up if the last ~10 ppl matters.
- **Next (the end-goal deliverable):** the ON-BRIDGE realization — run this fluency read-out learning ON the spiking
  substrate (the committed `cp_ssm_readout_w` graded read-out forward + the exact-delta update reading `cp_ssm_state`,
  now on the full-vocab fluency task) — so the cortex's pretraining, not just the grounded-task adaptation, is learned
  on the one fully-spiking shared substrate. Then the single-shared-substrate consolidation (everything on ONE bridge).

Runner: `_gap_pretraining_shallow_fluency_derisk.py` (`--no-token`/`--random-input`).
