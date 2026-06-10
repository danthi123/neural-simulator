# STEP 2b — RF composer co-resident on the one bridge (implementation plan)

> **Prerequisite:** STEP 2a complete (the 6-seed navigation gate (a) GREEN). The protected `sim/bridge.py`
> `rf_kick(..., neuron_mask=)` / `_rf_advance_one` masked-write edit is **owner-approved** (committed
> default-off byte-identical; review doc `2026-06-10-unification-sliced-RF-ops-edit-byte-review.md`).
> **Scope:** make the resonate-and-fire (RF) FHRR composer run on a SLICE of the merged navigation+conversational
> bridge — the strict single-instance unification — instead of on its own per-op bridges. Replacing the
> composer's exact-inverse VSA algebra with a learned spiking cortex (step 3) stays out of scope.

## Why this is small (the de-risks already did the hard part)

5b proved the minimal mechanism: the composer is stateless across operations (re-kicks each op) and stores
its memory in COMPLEX synapses (`cp_rf_w_re`/`cp_rf_w_im`, array-disjoint from `v`/`u` AND `cp_connections`),
so the RF slice need NOT survive a navigation step. The masked RF ops (already committed, owner-approved) let
the composer drive only the RF slice's `v`/`u`; navigation's Izhikevich step harmlessly clobbers the idle RF
slice between ops (re-kicked each op). `derisk_unification_5b_edited.py` already proved an RF op on a masked
slice of a larger Izhikevich bridge equals a standalone RF bridge EXACTLY while the Izhikevich slice stays
byte-identical.

## The build

### Task 1 — add the `rf` region to the merged bridge
- In `conv_extra_regions_pathways` (or a dedicated builder), append an `rf` `BrainRegion` of `2*D` neurons
  (`D` = the composer's projection dim; e.g. 128). `internal_density=0.0`, NO outgoing `RegionPathway` into
  navigation — the region exists only to reserve a contiguous neuron slice. `enable_nmda=False`.
- **Constraint (anti-cheat):** the `rf` region must have NO `cp_connections` out-edges into navigation, so its
  incidental Izhikevich firing between composer ops injects nothing into the navigation cascade. (Verify: no
  `RegionPathway(from_region="rf", ...)` to any nav region; the framework's internal-density is 0.)
- Resolve `rf_base = region_manager.indices("rf")[0]`; `rf_mask` = a bool array True on the rf slice.

### Task 2 — route the composer op onto the RF slice (runner-side; no further `sim/` edit)
- The composer's per-op `_resonate(n, conns, kick)` (`rf_phasor_composer.py:101-111`) addresses indices
  `0..2D-1`. Adapt it for the merged bridge: shift the index arithmetic by `rf_base`, build a FULL-`n` complex
  kick (zeros off the rf slice), set the complex weights via `rf_set_complex_weights` at the merged bridge's
  full `(n,n)` shape (the diagonal bind/unbind synapses are O(D) sparse, so size is fine), and call
  `rf_kick(full_kick, neuron_mask=rf_mask)` + `rf_resonate_steps`. Read phases from the rf slice.
- This is a thin adapter (a `MergedRFComposer` or a `composer` mode in `MergedNavConvAgent`) — reuse-by-import
  of `RFPhasorComposer`'s op bodies, only the index base + the mask differ.

### Task 3 — wire it into the agent
- `MergedNavConvAgent` (STEP 2a) currently delegates `store`/`query_*` to a separate-bridge `RFPhasorComposer`.
  STEP 2b swaps that for the co-resident adapter (Task 2) when an `rf` region is present. `parse` (parser
  slices) and `elaborate` (dlPFC slices) are unchanged from STEP 2a.

## Acceptance gates (re-run on the STEP-2b bridge)

1. **5b edited-version re-validation at merged-bridge `n`:** re-point `derisk_unification_5b_edited.py` (or a
   merged variant) so the RF slice is `[rf_base : rf_base+2D]` of the full merged `n` — assert the RF op on
   the masked slice == a standalone RF bridge EXACTLY, and a co-resident driven Izhikevich slice's `v`/`u` are
   byte-identical across the RF op. (`tests/test_rf_neuron_mask_coexistence.py` is the unit form.)
2. **Conversational gate (b):** `tests/test_nav_conv_merged_agent.py` passes VERBATIM with the co-resident RF
   composer (incl. the three `is None` no-confab assertions) — the composer's fact memory / QA / abstention
   now computed on the merged bridge's RF slice.
3. **Navigation gate (a):** the 6-seed merged navigation score stays within noise of standalone (the `rf`
   region is silent + disjoint during navigation; same conv-inertness test as STEP 2a).
4. **Anti-cheat:** assert the composer op ran on the RF SLICE of the merged bridge — `merged_bridge.cp_rf_w_re`
   is non-None and the RF read came from `merged_bridge`, NOT a throwaway RF bridge.

## Honest could-be-NEGATIVE

- **The full-`n` complex matvec is too slow / too large at the merged `n`.** Mitigation: the complex CSR is
  O(nnz) sparse (diagonal bind/unbind = O(D)); confirm the per-op wall-clock is comparable to the standalone
  per-op bridge. If a mostly-empty `(n,n)` complex matvec is a real cost, keep the composer on its own bridge
  (the STEP-2a looser bar is the shipped deliverable) and document why true co-residence is not warranted.
- **The masked RF op perturbs the navigation Izhikevich path despite the edit.** The 5b guard (the navigation
  slice `v`/`u` byte-identical across an RF op) is the control; if it fails on the full merged bridge, revert
  to the STEP-2a separate-bridge composer.

## Sequence
1. Re-point the 5b edited-version de-risk at the merged `n` (cheapest first). 2. Tasks 1-3. 3. Acceptance
gates 1-4. Each step independently valuable; a failure at any gate is an honest finding (the measured cost of
true RF co-residence), not hidden.
