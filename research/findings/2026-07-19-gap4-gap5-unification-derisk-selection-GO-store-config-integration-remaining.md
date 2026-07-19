# gap#4↔#5 unification de-risk — the DG-selection is GO (this session), but one-shot DEFAULT-Hebbian store is too weak to form a self-sustaining attractor; the fix = integrate the ALREADY-VALIDATED completion-arc store config onto the emergent-DG bridge (recombination, not a new capability).

**2026-07-19.** Per the owner's "close the gaps, your call," pursued the gap#4↔#5 unification (one-shot BTSP-store the
DG-selected assembly → self-sustain + complete) as the path that advances gap#5 (ii) AND demonstrates biological
local-credit learning (one-shot plateau-gated plasticity — the gap#4 keystone mechanism) while SIDESTEPPING the
supervised-deep-credit wall (which the research gate showed is deeply blocked).

## Result (3-seed, `unify_btsp_attractor.py`, on the emergent-DG amplify bridge)
- **PHASE 1 — mossy SELECTION works** (this session's 6-seed-GO result reproduced): the DG volley selects a sparse CA3
  assembly (4/20/13 cells, seeds 42/43/44), plasticity OFF.
- **PHASE 2 — one-shot DEFAULT-Hebbian store is TOO WEAK:** driving the selected cells directly for 60 steps with
  `enable_hebbian_learning=True` does NOT form a self-sustaining attractor — self-sustain 0.00, completion 0.00, SAME as
  the no-store control (which correctly collapses). A de-risk-implementation subtlety was caught + fixed first: turning on
  the GLOBAL Hebbian flag DURING the DG-volley drive corrupted the mossy selection itself (selected → 0); separating the
  SELECT phase (plasticity off) from the STORE phase (drive selected cells directly, plasticity on) fixed that.

## Why + the fix (a recombination of ALREADY-VALIDATED pieces, not a new capability)
The self-sustaining completable attractor is ALREADY validated on this substrate — the **emergent CA3 completion
(`2026-07-09-riii-emergent-ca3-completion-kopsick-formation`, 6-seed GO, within/cross 12.6×)** and THIS session's **SWR
readout closing** both store + complete network-selected assemblies. But both use the FULLER store config: RATE-Hebbian
(`hebb_rate`/`hebb_sym`) + `hebb_max≈120` + MANY drive events (train_events≈100) + the k_thresh specificity knob — NOT a
brief 60-step default-Hebbian drive. My emergent-DG `_build_bridge` sets `enable_hebbian_learning=False` + a plastic
recurrent but does NOT set the completion-arc Hebbian params. ⇒ **the unification = mossy-SELECTION (GO, this session) +
the VALIDATED store+complete config (2026-07-09 + this session) — an INTEGRATION, not a new mechanism.** The remaining
step: thread the completion-arc store config (rate-Hebbian, hebb_max, multi-event drive of the SELECTED cells) onto the
emergent-DG bridge, then re-test self-sustain + complete. NO `sim/` edit (all reuse-by-import / config).

## Status
- **DE-RISKED:** the DG-volley SELECTION of a sparse assembly (6-seed GO) + the honest identification that a brief default
  store is insufficient. **REMAINING (recombination):** wire the validated completion-arc store config onto the selected
  assembly. The pieces are individually GO; the integration is the next concrete step.
- Context: this session CLOSED gap#5 (i) SWR readout (6/6 GO + anti-cheat) and de-risked gap#5 (ii) selection (6-seed GO);
  the supervised gap#4 BDSP-to-accuracy is deeply walled (research gate — the apical-decoupled bug + forward collapse), so
  this unification (one-shot local BTSP forming an emergent attractor) is the local-credit-keystone path that sidesteps it.
