# On-bridge fully-spiking read-out LEARNING (rung ii) — MECHANISM GO (delta-rule on the substrate), single-layer bigram-limited

**Date:** 2026-07-20 · **Status:** rung (ii) GO for the MECHANISM (the on-bridge graded read-out learns by a pure
local plasticity rule, on the substrate) — with an honest limit (a single LINEAR read-out is bigram-bound; the
memory-dependent grounded content needs the multi-layer read-out = rung iii). First on-bridge realization of the
biological-learning close. Additive default-off `sim/` mechanism (the forward), verified byte-identical when off.

## What was built

**The `sim/` mechanism (committed, additive, default-off byte-identical):** an on-bridge GRADED read-out over the SSM
state — `cp_ssm_readout_out = cp_ssm_readout_w @ cp_ssm_state`, computed IN the bridge step loop (the read-out value
carried through the synapse weights, graded transmission = the OUTPUT analogue of M2's synaptic INPUT decode). Guarded
by `cp_ssm_readout_w is None` → skipped → byte-identical (15 SSM+determinism tests pass); ON-path verified byte-exact
(`max|out − W@state| = 0`).

**The rung-(ii) de-risk (`_gap_onbridge_ssm_readout_learn_derisk.py`):** the WKV cortex (emb/Wv/decay) is the FIXED
reservoir; per token it charges the on-bridge graded state; a SINGLE-layer read-out (`logits = W @ state`) is trained
ON the substrate by the pure **DELTA RULE** `dw = −eta·error[post]·state[pre]` — for a single output layer the error
is LOCAL (no feedback pathway, no weight transport), and `cp_ssm_state` is the presynaptic eligibility (no BPTT — the
state is the fixed reservoir's own dynamics). Reduced grounded vocab (112 words). Metric: grounded next-token accuracy.

## Result — the MECHANISM learns (GO), the single linear layer is bigram-limited (honest)

- **MAIN: grounded next-token acc 0.42 (46× chance 1/112)** — the on-bridge graded read-out LEARNS the grounded map
  by the pure local rule, on the substrate (verify-first 0→36/20-frames).
- **Anti-cheats:** FROZEN (no update) → **0.004 = chance** (the learning is load-bearing); SHUFFLE-ELIG (shuffle the
  state→readout association) → **0.105** (4× collapse — the state association matters); **MEMORYLESS (k_leak=1) →
  0.420 == MAIN (did NOT collapse).**
- **The honest read of MEMORYLESS-not-collapsing:** the single LINEAR read-out reaches 0.42 by learning the LOCAL /
  bigram structure (the function words `the`, the `<EOS>` marker — locally predictable from the current token), NOT
  the memory-dependent CONTENT copy (recalling the subject 5-6 tokens back needs the graded memory). A single linear
  layer over the state cannot express the memory-dependent copy — the same reason the off-bridge single-layer under-
  performs the multi-layer gated read-out (Wr·Wo_sp·head), which reached **0.998** on the reduced vocab.

## Read-out

- **⇒ the on-bridge fully-spiking read-out LEARNING mechanism is GO:** a graded read-out over the on-bridge SSM state
  learns by a pure local plasticity rule (delta rule — no BPTT, no weight transport, no adaptive optimizer), on the
  substrate, load-bearing (frozen→chance, shuffle→4× collapse). This is the first on-bridge realization of the
  biological-learning close's local-rule read-out learning.
- **Honest limit + the next rung:** a single LINEAR read-out is bigram-bound (memory not load-bearing at 0.42). The
  memory-dependent grounded CONTENT (the copy) needs the MULTI-LAYER read-out — **rung (iii): the gated read-out
  `head @ (sigmoid(Wr@h) * (Wo_sp@state))` on-bridge, with the FA feedback pathway for the hidden layers** (the D3
  clean-error channel / `enable_bdsp_graded_credit`). The off-bridge multi-layer FA reached 0.998 reduced, so the
  on-bridge multi-layer read-out is de-risked at the rule level; realizing the gated forward on-bridge (rung iv) +
  the FA hidden-layer credit is the remaining build.
- NO `sim/` edit beyond the additive default-off `cp_ssm_readout_w` forward. ⚠ `cfg.seed` set; anti-cheats named.

Runners: `_gap_onbridge_ssm_readout_learn_derisk.py` (`--frozen`/`--shuffle-elig`/`--memoryless`). `sim/bridge.py` (the
additive `cp_ssm_readout_w`/`cp_ssm_readout_out` forward).
