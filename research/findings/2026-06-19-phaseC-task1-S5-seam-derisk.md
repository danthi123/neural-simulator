# Phase C — Task 1 (the novel seam S5): result→sequencer coupling on-substrate — HONEST NEGATIVE (option b) (2026-06-19)

**Verdict: the on-substrate result→sequencer coupling (S5, option a — a fixed `cp_connections` projection) WALLS.
The honest-negative form holds: the result→sequencer DATA hand-off stays a host read (option b), while the
CONTROL (the match comparison + answer/abstain decision) is fully on-substrate (Phase B GO).** The no-confab moat
was NEVER weakened — a fixed projection that washes the match out either abstains or breaks the moat, and the
runner reports that boundary rather than relaxing the moat to manufacture a pass.

This is the one genuinely-new piece Phase C asked of the substrate (Phase B's sequencer already ran the control
flow in spikes, but read the cleanup *result* to host via `block_cleanup_scores → scores_to_drive`). Task 1 asked:
can a fixed on-bridge projection carry that graded result without the host read? Answer: no — and the reason is
structural, not a tuning miss.

## Result (3 seeds, D=64, real `OneBrainComposer`; runner `_phaseC_task1_S5_seam_derisk.py`)

```
SUMMARY: optB(control) GO 3/3   optA1 GO 0/3 (moat 0/3)   optA2 GO 0/3 (moat 0/3)
```

- **OPT-B (Phase B host coupling, the CONTROL): GO 3/3** — ==host True, moat True. The control flow (match →
  answer/abstain) is on-substrate; only the score→drive number is a host read. (Confirms the harness is sound: the
  failure below is specific to option a, not the test.)
- **OPT-A1 (raw cleanup membrane → fixed projection): 0/3, moat 0/3.** The decoded word-lines all fire (`lit` =
  the whole row, e.g. (12,12)), both match pools m0 AND m1 cross threshold, so the cascade cannot discriminate
  which block matched → it answers (e.g. "north") even for absent-agent / absent-action / cross cues that MUST
  abstain → moat breach. Reported as the boundary, not accepted.
- **OPT-A2 (rescaled-current + WTA projection): 0/3, moat 0/3.** Same wall (a per-row WTA on the projection does
  not recover it; seed 44 shows partial collapse but still moat False).
- **lesion-fails-safe = True (3/3):** severing the projection → decoded lines silent → abstain.

## Why option a walls (structural — confirmed by the cleanup probe, not just argued)

The cleanup result is graded RF state `Re(c)` on `cp_membrane_potential_v`: winner ≈ peak (~1.3×10⁶), runner-up ≈
0.4·peak — the discrimination is **relative**. The cleanup probe (`_phaseC_task1_cleanup_probe.json`) measured
scores 1.3–3.3×10⁵ with `both_suprathreshold_vs_izh_30mV: true`: the winner AND the runner-up (AND every off-target
with positive `Re`) are all far above the Izhikevich firing threshold (~+30 mV). To drive an Izhikevich decoded
line through `cp_connections` (the only host-read-free route) the cleanup neuron must FIRE — but a binary spike is
identical for winner and runner-up, so it DESTROYS the relative magnitude the selection needs. The
peak-normalization that `scores_to_drive`/`_spiking_cleanup` perform is a host op; a fixed `cp_connections`
projection cannot express it (the normalizer is per-query — the peak varies query to query).

This is the point-neuron graded-magnitude limit in another guise (cf. the conversational whitening wall and the
rate-code wall, Mikulasch-Priesemann): a fixed synaptic projection cannot carry a graded, per-query-normalized
score through a spike without an analog/dendritic or a normalizing-circuit stage.

## What it means for Phase C

The integrated loop (Task 2) runs with the **control on-substrate** (the on-substrate sequencer decides which fact
to scan and whether to answer or abstain, host `_scan` gone) and **one residual host DATA read at S5** (the cleanup
score → the decoded-line drive). That is the honest scope of "the brain runs the whole turn as one loop" at the
cheap-first option: all the computation and all the control are spikes; one number is read to host between the
cleanup and the sequencer.

## Follow-on to close the last host read (the lever, not on the cheap-first path)

A **NEF-style thresholded cleanup** (Stewart-Tang-Eliasmith — the same mechanism already validated as the composer
cleanup, `NEF_CLEANUP_OP`) places per-concept firing thresholds so off-target emits ZERO spikes, discretizing the
graded score to argmax WITHOUT firing the whole row. Its prerequisite is **on-bridge input normalization** (so the
per-concept thresholds are robust to the per-query peak). If that normalization can be a point-neuron divisive
pool, S5 becomes fully on-substrate and the loop has zero host round-trips; if it needs analog/dendritic
normalization, that maps the boundary to the deferred dendritic substrate. Either way it is a separable next step.

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseC_task1_cleanup_probe --seed 42 --dim 64
SIM_BACKEND=cupy python -u -m research.runners._phaseC_task1_S5_seam_derisk --seeds 42,43,44 --dim 64
```

Design: `2026-06-19-tier2-phaseC-integrated-loop-design.md` §2.3, §4.3, §6. Phase B (control flow GO):
`2026-06-19-onebrain-sequencer-derisk.md`. Task 0 (op-level spiking GREEN): commit `a49cd9a6`.
