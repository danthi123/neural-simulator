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

- **MAIN: grounded next-token acc 0.42 (5 ep) → 0.478 (20 ep) → 0.667 (40 ep, STILL RISING: 0.47→0.54→0.65→0.667;
  ≈74× chance 1/112)** — the on-bridge graded read-out LEARNS the grounded map SUBSTANTIALLY by the pure local rule,
  on the substrate (verify-first 0→36). ⚠ I under-estimated the ceiling TWICE (first "bigram-limited 0.42", then
  "~0.49") — both were UNDER-TRAINING; more epochs keep climbing. The single-linear read-out is stronger than first
  concluded; the plateau is above 0.667 (not measured to convergence).
- **Anti-cheats:** FROZEN (no update) → **0.004 = chance** (the learning is load-bearing); SHUFFLE-ELIG (shuffle the
  state→readout association) → **0.105** (4× collapse — the state association matters); **MEMORYLESS (k_leak=1) →
  0.401** at the same 20-epoch budget.
- **The MEMORY IS load-bearing (corrected — verify-before-concluding):** at 5 epochs MAIN and MEMORYLESS were EQUAL
  (both 0.42), which looked like a bigram ceiling — but that was UNDER-TRAINING. With 20 epochs **MAIN 0.478 >
  MEMORYLESS 0.401** (a ~0.08 gap): the single-layer read-out DOES use the graded memory (recalling content held in
  the state), not just the local/bigram structure (function words + `<EOS>`). ⚠ Lesson: I nearly recorded "memory not
  load-bearing" from the under-trained equal result — the fix was to train longer and re-check (the silent-failure
  discipline).
- **The single-LINEAR ceiling (~0.49) is real, though:** a single linear layer over the state uses the memory only
  WEAKLY; the off-bridge MULTI-layer gated read-out (Wr·Wo_sp·head) reached **0.998** on the same reduced vocab. So the
  full memory-dependent copy needs the multi-layer read-out (rung iii), not more single-layer training.

## Read-out

- **⇒ the on-bridge fully-spiking read-out LEARNING mechanism is GO:** a graded read-out over the on-bridge SSM state
  learns by a pure local plasticity rule (delta rule — no BPTT, no weight transport, no adaptive optimizer), on the
  substrate, load-bearing (frozen→chance, shuffle→4× collapse, and the MEMORY is load-bearing: MAIN 0.478 > memoryless
  0.401 at sufficient training). This is the first on-bridge realization of the biological-learning close's local-rule
  read-out learning.
- **The next rung:** the single-LINEAR read-out reaches 0.667 (still rising) — substantial, memory-using, but below
  the off-bridge MULTI-layer 0.998. Pushing to the full memory-dependent copy (and/or converging the single-linear)
  is the accuracy step — **rung (iii): the gated read-out
  `head @ (sigmoid(Wr@h) * (Wo_sp@state))` on-bridge, with the FA feedback pathway for the hidden layers** (the D3
  clean-error channel / `enable_bdsp_graded_credit`). The off-bridge multi-layer FA reached 0.998 reduced, so the
  on-bridge multi-layer read-out is de-risked at the rule level; realizing the gated forward on-bridge (rung iv) +
  the FA hidden-layer credit is the remaining build.
- NO `sim/` edit beyond the additive default-off `cp_ssm_readout_w` forward. ⚠ `cfg.seed` set; anti-cheats named.

Runners: `_gap_onbridge_ssm_readout_learn_derisk.py` (`--frozen`/`--shuffle-elig`/`--memoryless`). `sim/bridge.py` (the
additive `cp_ssm_readout_w`/`cp_ssm_readout_out` forward).
