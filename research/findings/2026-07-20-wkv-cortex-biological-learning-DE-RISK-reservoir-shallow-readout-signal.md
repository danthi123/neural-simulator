# WKV cortex biological-learning — cheap-first DE-RISK: the grounded task adaptation is SHALLOW-READOUT-learnable over a FIXED cortex

**Date:** 2026-07-20 · **Status:** cheap-first de-risk GO (a signal for the reservoir/emergence path) — the grounded
render's TASK adaptation does NOT need deep BPTT of the recurrence; the input→state map + recurrence can be FROZEN.
Read-only research gate (agent) running in parallel to scope the deeper question. NO `sim/` edit.

## The frontier

The north-star grounded-fluent-conversation is achieved on spikes (De-risk 0-5), but the WKV language cortex is
trained by **BPTT** — the mission's last tracked shortcut (end state = learned by a biological LOCAL rule, no
weight transport, no backprop-through-time). The project's own R3 reframe: "reservoir / shallow-readout is the
emergence path." This de-risk tests one concrete piece: **is the grounded-render TASK adaptation (the format
fine-tune) a SHALLOW-READOUT skill over a FIXED cortex, or does it need deep BPTT of the recurrence?**

## Result — the grounded copy is shallow-readout-learnable over a fixed cortex

`_gap_grounded_wkv_finetune.py --freeze-input`: freeze `Wv` (the input→state map) + `decay` (the recurrence dynamics)
at the pretrained values; train ONLY the read-out (`Wr`/`Wo_sp`/`head`) + `emb`. Eval on all 22 held-out curriculum
facts:
- **focused-grounded 0.86 (19/22)**, **RA-faithful 1.00 (44/44, 0 bias)** — **identical to the full fine-tune**
  (0.83-0.85 / 1.00). anti-forget stable (TinyStories ppl 28.12 → 28.20). Grounded loss → 0.015.

⇒ freezing the recurrence's input map + decay costs NOTHING for the grounded task — the adaptation lives entirely in
the read-out (+ token emb). The grounded-render capability is a **shallow-readout adaptation over a fixed cortex**,
NOT a deep-credit problem. This is exactly the reservoir/shallow-readout emergence-path shape (the R3 reframe): the
recurrence is a fixed reservoir; the task is learned by the read-out.

## Honest scope — what this does NOT yet show

- The frozen `Wv` was itself **pretrained by BPTT** on TinyStories — so this shows the TASK adaptation is
  shallow-readout-learnable over the fixed pretrained cortex, NOT that the PRETRAINING (the TinyStories fluency) is
  biologizable. `emb` was also still trained (the token encoding).
- The deeper open question — **can the WKV cortex's PRETRAINING be learned by a biological local rule (e-prop /
  feedback-alignment / burstprop, no BPTT-through-time) and still reach fluency?** — is what the parallel read-only
  research gate is scoping (ranked cheap-first de-risks: e.g. random-`Wv` true-reservoir + shallow read-out; a local
  read-out rule replacing BPTT; whether an existing on-bridge local rule [BDSP/e-prop] applies to a diagonal-leaky
  language cortex). Build the ranked ladder when the gate returns (trust-but-verify its cited claims).

Runner: `_gap_grounded_wkv_finetune.py --freeze-input`. Out: `research/findings/raw/_gap_grounded_wkv_reservoir.json`.
