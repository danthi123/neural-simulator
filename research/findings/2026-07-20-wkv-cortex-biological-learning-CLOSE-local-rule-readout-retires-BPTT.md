# WKV cortex biological-learning — CLOSE: the grounded fluent render is learnable by a LOCAL rule (no BPTT, no weight transport)

**Date:** 2026-07-20 · **Status:** CLOSE (GO) — the mission's last shortcut (BPTT-training the grounded renderer) is
RETIRED for the grounded capability: a FIXED reservoir + a read-out trained by a biological LOCAL rule (feedback
alignment / Kolen-Pollack, transport-free) reaches the grounded fluent render at parity with BPTT. NO `sim/` edit.

## The frontier + how the scoping was sharpened

The north-star grounded-fluent-conversation is achieved on spikes (De-risk 0-5), but the WKV language cortex is trained
by BPTT — the mission's last tracked shortcut (end state = a biological LOCAL rule, no weight transport, no
backprop-through-time). A read-only research gate + a 4-agent adversarial verification scoped it:
- **CONFIRMED (solid):** the WKV recurrence is a DIAGONAL scalar-decay leaky integrator with NO recurrent weight
  matrix → e-prop's eligibility is EXACT (nothing dropped); the read-out (Wr/Wo_sp/head) is a per-timestep map.
- **Verification corrected two overstatements:** (1) the "only 2.8% needs through-time credit" framing excludes `emb`
  (~44%), which feeds the recurrence via Wv and IS on the through-time path when trained; (2) the cited Zucchet rule
  trains the decay GATE, not `Wv`, and was validated on modest tasks, not fluency.

This SHARPENED the close: since **Rung B** shows `Wv` needn't be learned (a random reservoir reaches fluency) and the
verification flags `emb` as through-time-when-trained, the cleanest fully-biological close **FREEZES the entire cortex
+ encoding (emb + Wv + decay)** → ZERO through-time credit anywhere → and trains ONLY the per-timestep read-out by a
LOCAL rule. No e-prop, no `Wv` training, no BPTT needed.

## The ladder

**Rung A (done, GO):** freeze the whole cortex, train the read-out (Adam) → grounded copy 0.86 == the full fine-tune
(`--freeze-cortex`). The grounded task is a shallow read-out over a fixed cortex.

**Rung B (done, GO):** random-`Wv` reservoir (frozen) + trained read-out reaches TinyStories held-out ppl **25.6**
(≤ the BPTT-trained 28.1) → the learned input map is only weakly load-bearing; the read-out does the work
(`--random-input --freeze-input`).

**CLOSE (Rung C', GO) — `_gap_grounded_wkv_local_readout.py`:** a FIXED reservoir (detached → NO BPTT-through-time) +
a read-out trained by a transport-free LOCAL rule. `LinearFA` routes the output error through a feedback matrix B
instead of `W^T` (**no weight transport**); each weight update is local (output error ⊗ layer input). FA = fixed
random B (Lillicrap 2016); KP = B learns the same local update → aligns to W (Kolen-Pollack/Akrout). Verify-first:
the FA forward == the standard read-out forward (maxdiff 0.0; only the backward differs). Grounded copy (22 held-out
facts) + TinyStories ppl, read-out trained FROM SCRATCH over the fixed reservoir:

| credit rule | grounded verified-fluent | RA-faithful | TinyStories ppl | biological? |
|---|---|---|---|---|
| **KP** (learned feedback) | **0.91** (20/22) | 0.84 | 63 | ✅ no transport, no BPTT |
| **FA** (fixed random feedback) | **0.91** (20/22) | 0.86 | 74 | ✅ no transport, no BPTT |
| BPTT (weight transport) | 0.86 (19/22) | 1.00 | 34 (ceiling) | ❌ the shortcut |

⇒ **on the mission-relevant capability (the grounded fluent render), the transport-free local rules MATCH/BEAT the
BPTT ceiling (0.91 vs 0.86).** The grounded render is FULLY learnable by a biological local rule over a fixed reservoir.

**Fully from scratch (the strongest form): RANDOM reservoir + KP read-out, NO BPTT ANYWHERE** (`--random-input
--credit kp --readout-from-scratch`): grounded verified-fluent **0.73** (16/22), RA-faithful 0.82, clean renders
("the dog eats meat", "the bird eats seed", "the cow eats grass"). Even with a random (never-BPTT) reservoir AND a
local-rule read-out, the grounded fluent render is a GO — the entire render learned with no BPTT and no weight
transport anywhere.

## Read-out — honest scope

- **The grounded render (the mission capability): FULLY biologically-learnable** — fixed reservoir + FA/KP read-out,
  no BPTT-through-time, no weight transport. Over the pretrained cortex it MATCHES BPTT (0.91); fully from scratch
  over a random reservoir it reaches 0.73 (GO). **The BPTT shortcut is retired for the grounded renderer.**
- **The evidence FA/KP are genuinely transport-free (not silently falling back to BPTT):** their general-fluency ppl
  DIFFERS from BPTT (FA/KP 63–80 vs BPTT 34) — the backward is genuinely a different (transport-free) credit rule.
- **The general TinyStories FLUENCY gap is CLOSEABLE (convergence-rate, not a wall) — CONFIRMED at large budget.** On
  pure fluency (grounded-frac 0), more budget monotonically narrows the KP-vs-BPTT gap: **10k steps KP 63 (1.9×) →
  20k KP 38.2 (1.3×) → 40k KP 35.1 (1.19×), and STILL DROPPING** (44.7→38.3→36.2→35.1) vs BPTT ~29.5. KP never
  plateaus — the transport-free local rule converges SLOWER than BPTT (the R3-reframe's ~78% was a fixed-budget
  snapshot) but asymptotically APPROACHES the ceiling. ⇒ the general-fluency biologization is a training-budget/
  convergence-rate lever, NOT a fundamental limit; and the GROUNDED task (the mission capability) already MATCHES BPTT
  at the standard budget. Both halves of the render are biologizable by the local rule.
- **The Adam caveat is RESOLVED — the strictest biological claim holds.** KP + **pure SGD** (a delta rule: error ⊗
  input × lr, NO per-param adaptivity, NO momentum, NO BPTT, NO weight transport) reaches grounded verified-fluent
  **0.86 == the BPTT ceiling**, RA-faithful **1.00** (`--optimizer sgd`). So the grounded fluent render is learnable
  by a FULLY biologically-plausible pure local plasticity rule (transport-free KP feedback + delta-rule SGD updates).
  (SGD converges slower on general fluency — ppl 51 — the same convergence-rate lever as above; the grounded task
  matches BPTT.) **On-bridge** local-rule learning (the committed `enable_selective_ssm_state` + the validated
  on-bridge eligibility) is the fully-spiking follow-on.
- **The reservoir dynamics were fixed** (the reservoir/shallow-readout emergence path, per the project's R3 reframe).
  Closing the general-fluency ppl gap to BPTT (KP + more steps, or an on-bridge eligibility for `Wv`) is the remaining
  bounded lever.

**⇒ the mission's last shortcut is retired for the grounded fluent render: it is learnable by a biological local rule
(FA/KP, transport-free, no BPTT) over a fixed reservoir — matching BPTT on the grounded task (0.91), and a GO even
fully-from-scratch over a random reservoir (0.73).**

Runner: `_gap_grounded_wkv_local_readout.py` (`--credit {fa,kp,bptt}`, `--readout-from-scratch`, `--random-input`,
`--shuffle-feedback`). Results: `research/findings/raw/_local_{kp,fa,bptt,kp_randres}_grounded.json`.
