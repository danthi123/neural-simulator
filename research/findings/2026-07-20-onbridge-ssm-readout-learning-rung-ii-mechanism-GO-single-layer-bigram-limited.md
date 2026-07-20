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

- **MAIN: grounded next-token acc climbs to ~0.8 with training — 0.42 (5 ep) → 0.478 (20 ep) → 0.667 (40 ep) →
  0.55/0.67/0.847/0.767 (25/50/75/100 ep), peak ~0.85 (≈90× chance 1/112; the 0.767 final is 60-frame eval noise).**
  The on-bridge graded read-out LEARNS the grounded map to ~0.8 by the pure local rule, on the substrate. ⚠ I
  under-estimated the ceiling THREE times (bigram-0.42 → 0.49 → 0.667), each an UNDER-TRAINING artifact — the single
  LINEAR read-out over the on-bridge state is genuinely STRONG (~0.8), not bigram-bound. The repeated lesson:
  verify-to-convergence before stamping a ceiling.
- **MULTI-SEED (dev) FIRMED:** at 70 epochs the grounded accuracy is **seed 42 ~0.80 / 43 0.786 / 44 0.797** — tight,
  consistent, ~87× chance. The on-bridge local-rule read-out learning is dev-multi-seed GO (blind seeds 100/101/102 =
  a follow-on).
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

- **⇒ the on-bridge fully-spiking read-out LEARNING is GO AND STRONG:** a graded read-out over the on-bridge SSM state
  learns the grounded map to **~0.8** by a pure local plasticity rule (delta rule — no BPTT, no weight transport, no
  adaptive optimizer), on the substrate, load-bearing (frozen→chance, shuffle→4× collapse, memory load-bearing >
  memoryless 0.401). This is the first on-bridge realization of the biological-learning close's local-rule read-out
  learning — and the single-LINEAR read-out already reaches ~0.8 (not just "mechanism works").
- **Rung (iii) — a GENERIC 2-layer FA read-out UNDER-performs (honest negative):** a hidden relu layer over the state
  with fixed-random FA credit (`--n-hidden 256`, W1 by FA + W2 by delta) reaches only **~0.47** (70 epochs, plateaued)
  — WORSE than the single-linear ~0.8. The FA-routed hidden credit is coarser than the EXACT single-linear delta
  (the output layer's gradient is local + exact), so the approximation cost exceeds the capacity benefit. ⇒ for THIS
  on-bridge read-out the exact single-linear delta is the better rule; adding a generic FA hidden layer HURTS.
- **The next rung (accuracy top-up):** the single-linear ~0.8 is below the off-bridge MULTI-layer 0.998 — closing the
  last ~0.2 needs the SPECIFIC off-bridge recipe (the current-token GATE `sigmoid(Wr@h)` + the graded clean-error /
  KP, which reached 0.998), NOT a generic 2-layer — **rung (iv): the gated read-out
  `head @ (sigmoid(Wr@h) * (Wo_sp@state))` on-bridge, with the FA feedback pathway for the hidden layers** (the D3
  clean-error channel / `enable_bdsp_graded_credit`). The off-bridge multi-layer FA reached 0.998 reduced, so the
  on-bridge multi-layer read-out is de-risked at the rule level; realizing the gated forward on-bridge (rung iv) +
  the FA hidden-layer credit is the remaining build.
- NO `sim/` edit beyond the additive default-off `cp_ssm_readout_w` forward. ⚠ `cfg.seed` set; anti-cheats named.

Runners: `_gap_onbridge_ssm_readout_learn_derisk.py` (`--frozen`/`--shuffle-elig`/`--memoryless`). `sim/bridge.py` (the
additive `cp_ssm_readout_w`/`cp_ssm_readout_out` forward).
