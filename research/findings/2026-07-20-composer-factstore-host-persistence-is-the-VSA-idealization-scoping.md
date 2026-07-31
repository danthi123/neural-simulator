---
type: finding
status: contributing
date: 2026-07-20
mechanism: fact-store
---

# The composer fact-store's host-persistence is the VSA idealization, not a spiking-computation shortcut (scoping)

**Date:** 2026-07-20 · **Status:** SCOPING (verified read of our own code) — the last "not on the shared substrate"
item after the single-substrate capstone is the composer's FACT-STORE. Reading `one_brain_composer.py` +
`rf_phasor_composer.py` shows the fact DATA persists in a HOST list (`store_conns` / the numpy-kb) in BOTH composers;
the bridge synapses are installed transiently per read. So "everything on the substrate" is COMPLETE for all SPIKING
COMPUTATION; the host residual is the store's DATA persistence, which is the composer's documented VSA "principled
idealization", not a spiking-computation shortcut. Closing it truly-in-synapses is a scoped `sim/`-level change.

## What the code actually does (verified, not assumed)

- **`OneBrainComposer`**: `self.store_conns = []` (a HOST list, `:345`). `_write_block` APPENDS a fact's `(1+D)` block
  to `store_conns` (or rewrites in place for reconsolidation, `:586-589`) + sets `_store_dirty`. A query (`_scan`)
  calls `b.rf_set_complex_weights(self.store_conns)` (`:718`) — REBUILDS the store CSR from the host list + installs it
  onto the bridge's `cp_rf_w_*` — then kicks the trigger + reads the read-out. `_store_csr_cached` (`:779`) rebuilds
  the CSR from `store_conns` only when dirty.
- **`RFPhasorComposer`**: the numpy-kb holds the composites (phase arrays, or per-fact `(1+D)` substrate bridges with
  `enable_substrate_store`); `_store_substrate` builds a SEPARATE `(1+D)` bridge per fact.
- **The single-array constraint**: `rf_set_complex_weights` REBUILDS `cp_rf_w_*` from a conn list (REPLACE, not
  append), and the bridge has ONE `cp_rf_w_*`. So the per-op bind and the store cannot BOTH persist in `cp_rf_w_*`
  simultaneously — the store DATA must live host-side (`store_conns`) and be re-installed for each read.

## The honest reframe

- **The SPIKING COMPUTATION is fully on the substrate:** bind / unbind / bundle / cleanup (the composer's resonate
  ops), the WKV read-out forward, the WKV RF spike-encoder, AND the render-LEARNING (delta rule over `cp_ssm_state`) —
  all on ONE `SimulationBridge` (the capstone + learning-coresident findings, 6-seed). What the composer computes on
  spikes is consolidated.
- **The host residual is the fact-store DATA persistence** (`store_conns` / numpy-kb), which is the composer's
  **documented "principled idealization"** (the exact-inverse VSA algebra + its store; `CLAUDE.md` "COMPOSER IS A
  PRINCIPLED IDEALIZATION"). It is a memory-representation choice, NOT a spiking-computation the brain is shortcutting
  — the store's READ is already a spiking scan; only the cross-op DATA persistence is a host list.
- **To make it truly-persistent-in-synapses (close the idealization):** a `sim/`-level change — a PERSISTENT store
  weight tensor DISJOINT from the per-op `cp_rf_w_*` (so the store survives an `rf_set_complex_weights` op), OR a
  biological memory-in-weights consolidation (Crawford-Eliasmith / Hebbian LTP into a dedicated store synapse array).
  This is a scoped mechanism, additive/guarded, not a runner-level consolidation — the runner-level attempt would fight
  the single-`cp_rf_w_*` replace-on-set design.

## Read-out

- ⇒ the single-shared-substrate consolidation is COMPLETE for every SPIKING computation in the grounded conversational
  turn; the fact-store host-list is the composer's documented VSA idealization (a memory-representation, not a
  shortcut), closable by a scoped additive `sim/` store-synapse tensor when prioritized.
- This is the honest characterization of the last item (verified by reading our own substrate, per the standing lesson
  "read your own substrate before theorizing"), NOT a quick runner-level win — it needs the `sim/` store-tensor
  mechanism or the acknowledged idealization.
