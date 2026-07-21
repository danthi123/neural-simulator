# gap#5 SWR specificity stack — driver built; diagnosis confirms the emergent-DG completion is required UPSTREAM

**2026-07-21.** The gap-close research gate's Rank-1 unifies gap#5 (CA3 completion / SWR readout) with #2/#3 via the
competitive read. The gate's ranked de-risk (a-1 `2026-07-19-gap5-SWR-readout-specificity-research-gate`): the SWR
replay-readout near-tie is a dense-random-Schaffer artifact (Valero 2017), fixed by a STACK — pattern-separated CA3
completion (emergent-DG, 6-seed GO) UPSTREAM → structured+potentiated sparse Schaffer → E%-max/FFI → brief read.

## Built + first attempt (the diagnosis)

`_gap5_swr_specificity_stack_derisk.py` drives the SWR runner's `run()` with the readout stack ON
(`swr_learn_schaffer=True`, `read_ca1=True`, optional `swr_ca1_topk` E%-max) vs the dense-random baseline, reading
`ca1_match`/`ca1_cross`. First attempt (n_ca3=1000, n_mem=3, DEFAULT pre-assigned assemblies): **all-zeros — and
crucially `h_comp=0` (the CA3 COMPLETION itself fails), so `ca1_match=0` downstream.**

## Read-out — the diagnosis confirms the gate's shared-unlock

- **⇒ the SWR specificity stack cannot even start on the runner's DEFAULT completion** (random-disjoint pre-assigned
  assemblies, `h_comp=0` at n_mem=3) — exactly the gate's conclusion: **6-seed-robust CA1 specificity REQUIRES the
  pattern-separated CA3 completion UPSTREAM** (no CA1 readout can manufacture a distinction absent from CA3). The
  runner's default completion is the near-tie source; the emergent-DG mossy-selected assemblies (already 6-seed GO,
  `_gap5_emergent_dg_selection_derisk`) are the required input.
- **The precise next-step (staged):** source the emergent-DG completed assemblies → feed as `assemblies_ext` to the SWR
  `run()` → THEN the readout stack (`swr_learn_schaffer` + `swr_ca1_topk` E%-max + `swr_ca1_ff_inhib`) → read
  `ca1_match`/`cross`, 6-seed. GO bar match≥0.6 / cross≤0.3 / 3× / dense-random-collapses. This is the intricate
  multi-step build (the emergent-DG assemblies wiring + the readout params) — a research-frontier de-risk deserving
  careful, verify-first execution.
- **Completion probe (n_mem=2, encode∈{700,1400} × train∈{120,300}, n_ca3=1000): h_comp=0.000 ALL 4 — but w_ratio
  88-212 (within-assembly weights strongly POTENTIATED).** ⇒ CONFIRMS the record's characterized boundary EXACTLY: the
  within-assembly recurrent weights ARE learned (potentiated), but the completion is FUNCTIONALLY SILENT (the absolute
  recurrent drive is ~1000× too weak to fire the held members from a partial cue — the 2026-07-08/07-17 diagnosis). And
  **simple encode/train tuning does NOT fix it** (a new data point) — the pinned fix is the MULTI-KNOB Kopsick-Ascoli
  2024 recipe (strong synchronous encoding + `mossy_density`/`dg_ffi_weight`/`sync`/`ca3_fb_inhib` joint tuning at
  n_ca3~1000-2000, the record's pinned-but-only-run-at-150 attempt), an intricate multi-param arc.
- **Honest:** this cycle BUILT the SWR-stack driver + CONFIRMED the completion boundary (functionally-silent despite
  potentiated weights; encode/train alone don't fix it) — a genuine characterization, not a GO. The pinned NEXT
  MECHANISM (per the mission's boundary=undiscovered-mechanism law) is the multi-knob Kopsick recipe at scale → then
  the readout stack; an intricate multi-param hippocampal-completion de-risk deserving careful, verify-first execution.

Runner: `_gap5_swr_specificity_stack_derisk.py`; the SWR `run()` in `_riii_ca3_synchronous_assembly_derisk.py`
(`assemblies_ext`, `swr_learn_schaffer`, `swr_ca1_topk`, `read_ca1`).
