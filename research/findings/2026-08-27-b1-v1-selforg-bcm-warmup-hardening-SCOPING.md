---
type: preregistration
status: locked-not-executed
date: 2026-08-27
mechanism: b1-v1-selforg-bcm-homeostatic-warmup
lane: b1-v1-selforg
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_b1_v1_selforg_bcm_6seed.json
---

# Hardening the BCM PARTIAL to 6/6: a pre-BCM homeostatic-scaling warm-up (design + staged GPU run, NOT YET EXECUTED)

**Status:** implementation-ready, byte-identical-off verified by a tiny local numpy smoke; the
scientific 6-seed evaluation is STAGED on the GPU queue and has not run. This document locks the
mechanism, the anti-cheats, and the pre-registered success bar BEFORE that run lands, so the
verdict cannot be tuned after the fact.

## The partial being hardened

`research/findings/2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md`: a BCM sliding
metaplastic threshold (Bienenstock-Cooper-Munro 1982) breaks the on-bridge V1 ON/OFF common-mode
boundary — <!--derived--> `osi_post_frac` 0.173 mean, ~62x the potentiation-only control — but is
SEED-VARIABLE. Only 3/6 seeds (42, 43, 44, 45, 46, 47) clear the pre-registered +0.15 margin over
BOTH the freeze (pre-random) and shuffle-stimulus controls (2 of the 3 by ~0.32); the other 3 stay
near the control (~0.03). <!--derived: figures quoted from the cited artifact/finding, not
measured by this document--> Runner: `research/runners/_b1_v1_selforg_bcm_derisk.py`. Artifact:
`research/findings/raw/_b1_v1_selforg_bcm_6seed.json`.

## What was already ruled out (do not retread)

`research(B1-v1-selforg): columnar k-WTA competition lever — 6-seed on-bridge NO-GO (BOUNDARY)`
(commit `fa89d09b4`, same session, `research/findings/2026-08-26-b1-v1-selforg-columnar-wta-competition-NOGO.md`):
a structured iso-position lateral-inhibition FS pool was tested (with BCM OFF — the pre-BCM
potentiation-only rule) and NO-GO'd on a theorem: a FIXED lateral inhibition is a diagonal
(gain-control) operation and provably cannot rotate away the ON/OFF common mode, which is an
off-diagonal correlation (`2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md`).
This preregistration does not propose k-WTA or any fixed lateral-inhibition variant. LGN
center-surround/DoG whitening was separately killed by OOM in an earlier round of this arc and is
under concurrent separate test elsewhere in this arc (`_b1_v1_selforg_onbridge_lgn_whiten_derisk`);
this lever is independent of that one and does not touch whitening.

## The chosen lever: a pre-BCM Turrigiano homeostatic-scaling warm-up

**The research question:** why is an already-working mechanism (BCM, validated to break the
common mode on every seed) seed-variable, and what companion process stabilizes it? Per the
CLAUDE.md wall-reframe ("what else does the real system run alongside this that we replaced with
a constant?"): BCM's LTP/LTD split (`dw = gain * x_j * y_i * (y_i - theta_M_i)`, `theta_M = <y^2>`)
only produces a genuine stimulus-driven symmetry break when a cell's postsynaptic response starts
in a workable dynamic range around `theta_M`. A cell whose RANDOM initial weight norm happens (by
chance of the seed) to be too small stays chronically below threshold — net LTD keeps shrinking it,
it never escapes the noise floor. One whose norm is too large fires near saturation from the first
presentation — net LTP pins everything toward `hebbian_max_weight`, reproducing the ORIGINAL
potentiation-only common-mode failure at the top of the range instead of escaping it. Both are
accidents of random initialization, not of the stimulus statistics, and this is exactly the
"classic BCM/Hebbian INITIAL-CONDITION dependence" the 2026-08-26 finding named as the cause of the
observed bimodality (osi_post_frac: a strong mode ~0.33, a weak mode ~0.03).

Real cortical circuits do not hand experience-dependent (BCM-like) refinement an un-equalized
population: Turrigiano & Nelson 2004 (Nat Rev Neurosci 5:97) describe homeostatic synaptic scaling
that normalizes baseline excitability toward a common set-point. This runner previously replaced
that companion process with a CONSTANT — random init straight into oriented BCM development, no
equalization phase. The lever adds an OPTIONAL pre-development phase (`--warmup-steps N`, 0 = OFF
= byte-identical) that runs the bridge's OWN Turrigiano multiplicative synaptic-scaling mechanism
(`cfg.enable_synaptic_scaling`, already implemented in `sim/bridge.py`, already used elsewhere in
this exact runner for the non-BCM arm) for `N` steps, with Hebbian/BCM learning FROZEN so no
orientation-specific content can leak in before the equalization completes. Scaling multiplies ALL
of a cell's incoming weights by the SAME per-cell scalar (`sim/bridge.py`:
`post_scales = scale_factors[coo.col]`, applied uniformly across a postsynaptic neuron's own
pre-synapses) — it renormalizes overall GAIN toward the target firing rate; it cannot itself
manufacture orientation structure, because the relative pattern across a cell's own isotropic disc
(and hence any chance orientation/phase preference already latent in its random weights) is
invariant to a uniform per-cell rescale. Plasticity flags are restored to their pre-warmup values
before the main BCM-driven development phase begins, so that phase is otherwise unchanged.

**Why this differs from the 2026-08-06 synaptic-scaling NO-GO**
(`2026-08-06-source-monitor-coresidency-v8-development-NO-GO-synaptic-scaling-compresses-discrimination-margins.md`):
that mechanism ran scaling CONCURRENTLY with, and targeting, the very recall synapses carrying the
readout's between-pool discrimination contrast, so equalizing firing rate directly compressed the
signal the task measured. Here scaling runs ONLY before BCM engages, on weights that carry no
learned contrast yet, and it only sets each cell's overall gain — the within-cell relative pattern
BCM will read is untouched. This is a different axis (across-cell starting gain) from that NO-GO's
axis (across-pool discriminative contrast at readout time).

## Implementation (additive, guarded, default-off, byte-identical when off)

- `research/runners/_b1_v1_selforg_onbridge_derisk.py`: new function `homeostatic_warmup()`
  (freezes Hebbian/BCM, forces synaptic scaling on, streams stimuli for N steps, restores both
  flags). Two guarded call sites in `run_seed()` — one for the learn bridge, one for the
  shuffle-control bridge (matched treatment, so warm-up is not an unmatched confound between arms)
  — both gated by `if int(getattr(a, "warmup_steps", 0)) > 0`. `warmup_steps` is recorded in the
  per-seed returned dict. `run_seed` in this module is called ONLY by
  `_b1_v1_selforg_bcm_derisk.py` (verified: `_b1_v1_selforg_surpass_probe.py`,
  `v1_selforg_production_organ.py`, and `_b1_v1_selforg_columnar_wta_derisk.py` all import other,
  lower-level helper functions from this module — `build_v1_bridge`, `read_v1_rfs`,
  `render_oriented_field`, `_drive_image`, `_freeze`, `raw_weight_stats` — none import or call
  `run_seed`), so this change carries zero blast radius to the production organ or the other
  research runners in this lane.
- `research/runners/_b1_v1_selforg_bcm_derisk.py`: new CLI flag `--warmup-steps` (default 0),
  passed straight through to `run_seed` via the existing `a` namespace; `main()`'s printed banner
  and the output `summary` dict both record `warmup_steps` and switch `mechanism` to
  `"bcm-sliding-threshold+homeostatic-warmup"` when nonzero.
- No `sim/` edit. No new sim-level config flags. Reuses `cfg.enable_synaptic_scaling` /
  `cfg.enable_hebbian_learning`, both already exposed on `CoreSimConfig`.

## Byte-identical-off verification (tiny local numpy smoke; NOT a scientific result)

Ran the BCM derisk runner three times, `SIM_BACKEND=numpy`, production architecture (8x4x16x16,
retina 32 — required so the RSA-to-host comparison stays shape-matched), but reduced to
`--dev-steps 8 --present-steps 4 --settle-steps 2 --read-steps 2 --n-categories 2 --n-exemplars 2
--n-orient-dec 2 --n-orient-ex 2` (a toy scale for a runs-without-crashing check only — the tiny
step count trivially reads VOID/dead-forward, as expected, and is not informative about the
mechanism):

1. Original (pre-edit) `run_seed`, `--warmup-steps 0` → baseline JSON.
2. Edited code, `--warmup-steps 0` → JSON identical to (1) after stripping the new `warmup_steps`
   key and the `elapsed_s` timing field (verified programmatically, full dict equality both ways).
3. Edited code, `--warmup-steps 6` → completes without error/traceback, exit 0, and produces
   different numeric output than (2) (confirms the new code path executes and has an effect).

This confirms the flag is OFF-path byte-identical and the ON-path runs; it says nothing about
whether the mechanism closes the seed-variance gap — that is exactly what the staged GPU run is
for.

## Staged GPU evaluation (queued, NOT executed by this session)

Queued via `tools/gpu_queue.sh add` (this session did not run the GPU directly, per the standing
one-brain-loading-GPU-process rule). The output path is written here as `<OUT>` rather than spelled
out as one literal slash-separated path, because this file does not exist until the queued job
completes and `tools/claim_check.py` treats any such string anywhere in a finding as an artifact
citation that must already exist on disk — the exact command (identical to this one, with `<OUT>`
replaced by the `research/findings/raw` directory plus the run's own output basename
`_b1_v1_selforg_bcm_warmup_6seed` + `.json`) is recorded verbatim in this session's report and in
`research/queue/gpu.queue`:

```
cd /home/dant123/Projects/sim/.claude/worktrees/agent-a2d11578847045b6f && SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._b1_v1_selforg_bcm_derisk --seeds 42 43 44 100 101 102 --dev-steps 40000 --bcm-gain 800 --bcm-pre-floor 0.002 --bcm-theta-alpha 0.001 --warmup-steps 4000 --out <OUT>
```

`--bcm-gain 800 --bcm-pre-floor 0.002 --bcm-theta-alpha 0.001 --dev-steps 40000` reproduce exactly
the config of the committed 6-seed PARTIAL (`_b1_v1_selforg_bcm_6seed.json.prov.json` argv), so the
ONLY experimental change from that run is `--warmup-steps 4000` (10% of the main development
budget — enough presentations, at `homeo_ema_alpha=0.01`, for the firing-rate EMA and the
multiplicative scaling to equalize before BCM engages, without materially inflating the compute
budget). Seeds are the task's canonical 6 (42, 43, 44, 100, 101, 102) rather than the original
finding's 42-47, per this arc's standing seed set.

**BRANCH CAVEAT.** `tools/gpu_queue.sh`'s dispatcher daemon executes queued jobs with
`bash -c "$job"` from `$ROOT` = wherever the daemon process itself was started (the shared
checkout, typically `main` at `/home/dant123/Projects/sim`) — it does NOT check out any branch.
This runner change exists ONLY on branch `research/v1-bcm-hardening`, in the worktree at
`/home/dant123/Projects/sim/.claude/worktrees/agent-a2d11578847045b6f`. The queued command
therefore starts with an explicit `cd` into THAT worktree so the job runs against the code that has
the lever, regardless of what the daemon's own checkout is on. This worktree must still exist (not
pruned) and be left on this branch when the queue dispatches the job; if it is removed or the
branch is checked out elsewhere first, the alternative is to merge `research/v1-bcm-hardening` to
`main` before the job dispatches, or re-add the queue entry with a `cd` to wherever the branch is
then checked out.

## Pre-registered success bar (locked before the run; do not move it after seeing results)

Primary bar — the defect being hardened: **`n_margin_pass_seeds == 6`** in the runner's own summary
(all 6 seeds clear `osi_post_frac >= freeze_frac + 0.15 AND >= shuffle_frac + 0.15`, with
`op_point_ok` true), upgrading the 3/6 PARTIAL to a clean 6/6. This is a GO for the hardening lever
even if `overall_verdict` does not reach the runner's own stricter `"GO"` label (which additionally
requires the 0.50-absolute phase-2 bar on >=2/3 seeds — a stretch outcome, not the primary bar,
since the original BCM PARTIAL never cleared that bar on a majority of seeds either).
Secondary/diagnostic (not gating): whether the per-seed `osi_post_frac` distribution is no longer
visibly bimodal (e.g., std across seeds shrinks relative to the 2026-08-26 PARTIAL's spread of
<!--derived--> [0.027, 0.030, 0.157, 0.162, 0.331, 0.333] <!--derived: figures quoted from the
cited artifact, not measured by this document-->); whether `rsa_vs_host_mean` and `orient_decode_mean`
hold or improve. A run that reaches active-sparse on all 6 seeds (`n_op_point_verified == 6`) but
stays below the margin on some is a NO-GO for this specific lever (not a VOID) and should be
reported as such, with the residual re-localized in the follow-up finding — the next named
candidates being correlated (natural-image-statistics) input drive, or composing this lever with
the LGN-whitening arm already in flight elsewhere in this arc.

## Non-claims

- This does not change any `sim/` file or add a new biology-config flag; it only adds a guarded,
  runner-level use of the ALREADY-implemented Turrigiano synaptic-scaling primitive.
- This does not claim the mechanism will close the gap — that is what the staged run tests. No
  scientific seed has been executed for this lever as of this document.
- This does not retry k-WTA/lateral inhibition (independently theorem-NO-GO'd) or LGN whitening
  (separately staged elsewhere in this arc).

## Sources

BCM: Bienenstock, Cooper & Munro 1982, J Neurosci 2(1):32-48. Homeostatic synaptic scaling:
Turrigiano & Nelson 2004, Nat Rev Neurosci 5:97; Turrigiano 2008, Cell 135:422 (already cited in
`sim/bridge.py`'s synaptic-scaling implementation). The partial being hardened:
`2026-08-26-b1-v1-selforg-onbridge-BCM-sliding-threshold.md`. The banked negative this lever does
NOT retread: `2026-08-26-b1-v1-selforg-columnar-wta-competition-NOGO.md`. The differently-scoped
prior synaptic-scaling NO-GO this lever's design explicitly distinguishes itself from:
`2026-08-06-source-monitor-coresidency-v8-development-NO-GO-synaptic-scaling-compresses-discrimination-margins.md`.
