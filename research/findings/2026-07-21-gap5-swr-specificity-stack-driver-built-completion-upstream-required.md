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

---

## ⛔ CORRECTION (2026-07-21, same day — the a-1 RAG check I skipped) — the "completion is an open boundary / needs the multi-knob Kopsick recipe at scale" read above is WRONG

The `h_comp=0` probe above ran `run()` at the runner's **DEFAULT** completion params (no `bistable`, no dendritic
bistability) — so it re-derived "silent" by NOT using the closing config. **The CA3 completion is CLOSED** (2026-07-18,
`2026-07-18-gap5-CA3-completion-CLOSED-intrinsic-dendritic-bistability-resolves-the-trilemma.md`): **5/6 GO + 6/6
mechanism** (perfect specificity + bistability every seed; cue 0.18-0.33, nocue/perm 0.000, no-encode collapses) via
**intrinsic dendritic bistability** — the exact config `_gap4_btsp_completion_unification_6seed.GO_CFG`
(n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000, recall_k_thresh=110, bistable=True,
structural_sep=1, selective_inhib=True, plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=1.0, apical_gc_read=5.0).
The trilemma (magnitude vs specificity vs bistability) is resolved; the retracted self-sustaining/plasticity-noise
artifacts were caught + superseded.

## ⛔ CORRECTION 2 (2026-07-21, same day — the SWR readout is ALSO already CLOSED)
Correction-1 (above) said "the real next step is the SWR readout on the completing GO_CFG." That too is superseded:
**the SWR readout SPECIFICITY is already CLOSED — 6/6 GO + anti-cheat clean (2026-07-19)** (see
`2026-07-18-gap5-SWR-replay-readout-BLOCKED-...` → its "🎯🎯 CLOSED" block: `k30_hm150_d1200` + E%-max
`swr_ca1_topk=0.1`, match 0.700 / cross 0.065 / **10.79×** / 6 seeds; no-learn dense-random collapses to 1.02× =
load-bearing). And my own attempt this cycle CONFIRMED the trap: my `_gap5_swr_specificity_stack_derisk.py` embedded a
WRONG "GO_CFG" (`assembly_frac=0.12` async, NO `hebb_lr`/`coact_thresh`) → the completion was DEAD (held_cue=0.000) →
the `ca1_match 0.966` it printed was a CUE-DRIVEN ARTIFACT (the readout's PHASE-1 drives the cue cells directly), which
the `held_cue=0` verify-first guard caught. The ACTUAL closed config is sparse `assembly_frac=0.03` + synchronous
(`no_sync=False`) + `recall_k_thresh=30` + `hebb_max=150` + `recall_drive=1200` + `hebb_lr=4` + `swr_disjoint` +
learned Schaffer + E%-max.

⇒ **BOTH gap#5 sub-pieces (completion + SWR readout) are CLOSED.** The one genuine open piece is the **EMERGENT-DG
assemblies** (`_gap5_emergent_dg_selection_derisk`, 6-seed GO) wired as `assemblies_ext` into the CLOSED stack — the
emergence-bar (selected-from-experience) version of the sparse+sync completion, which the SWR arc repeatedly names as
"the shared unlock." This whole cycle is the drift-#12 lesson twice over: I trusted my own stale 2026-07-21 summary
over the authoritative 2026-07-18/19 CLOSED findings; the a-1 RAG check (reading the actual findings) is what corrected
it. The `_gap5_completion_kopsick_sweep.py` + the GO_CFG path in `_gap5_swr_specificity_stack_derisk.py` are moot.
