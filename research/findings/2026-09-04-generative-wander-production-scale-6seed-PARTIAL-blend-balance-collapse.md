---
type: finding
status: partial
lane: continuous-life/generation
date: 2026-09-04
mechanism: on-substrate CA3 generative attractor-wander (dendritic dAP bistable-latch blend-completion), production
  scale (n_ca3=2000, EMERGENT DG-selected assemblies, BTSP-formed membership) -- board #104 rung 2
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_generative_attractor_wander_onsubstrate_derisk.py (unmodified; `--emergent` path,
  `main()`'s own coded GO gate)
runner: research/runners/_generative_attractor_wander_onsubstrate_derisk.py
artifacts:
  - research/findings/raw/_generative_attractor_wander_onsubstrate/production_n_ca3_2000_6seed.json
  - research/findings/raw/_generative_attractor_wander_onsubstrate/production_n_ca3_2000_6seed.json.prov.json
---

# The production-scale (n_ca3=2000, emergent) generative-wander 6-seed verify RAN on 2026-08-28 and was never written up: 1/6 seeds clear the runner's own strict per-seed bar; the dominant failure is a BLEND-BALANCE collapse (4/6), with a genuine open question about its cause

Artifact: `research/findings/raw/_generative_attractor_wander_onsubstrate/production_n_ca3_2000_6seed.json` (the
decisive 6-seed run this finding reports on).

## Why this finding exists now, a week after the run

[[2026-08-28-generative-wander-production-verify-restaged-after-queue-loss]] root-caused why the 2026-08-27 staged
verify never ran (a queue append lost against a non-primary checkout) and correctly re-staged the identical
command on the PRIMARY `gpu.queue` against `main`. `research/queue/gpu_queue.log` (line 519214 onward) shows that
re-staged job actually START and DONE(rc=0) at 2026-08-28 11:22-11:30, printing the runner's own verdict line
verbatim: `VERDICT: PARTIAL/NEGATIVE (not yet) -- on-substrate port of the generative attractor-wander mechanism
(does NOT yet clear the on-substrate bar -- see the failing metric(s) above)`. The output file
(`production_n_ca3_2000_6seed.json`) exists on disk with an `Aug 28 11:30` mtime. No finding, board update, or
Vikunja note ever recorded this decisive result -- board #104's own text still reads (as of this session)
"production-scale 6-seed verify STAGED ... verdict pending", which is now nine days stale. This finding closes
that specific gap: it changes no code and runs nothing new, it only reports what the already-completed run
already measured.

## This does NOT retract the earlier #104 GO

[[2026-08-27-generative-attractor-wander-onsubstrate-GO]] (6/6 GO, the mechanism de-risk) is a real, separate,
smaller-scope result: non-emergent, hand-assigned assembly membership, `n_ca3` at the de-risk scale. This finding
is about the SAME mechanism class ported to PRODUCTION scale (n_ca3=2000) with EMERGENT DG-selected, BTSP-formed
assemblies -- a harder, more faithful test the smaller GO was never asked to clear (the exact "small easy version
passes, harder graded version doesn't" pattern this project has hit before on the learn-through-use arc). Both
results stand; they answer different questions.

## Result -- the runner's OWN coded gate: 1/6 seeds individually clear every criterion

<!--derived--> The runner's `main()` computes `go = all(genuine) and all(n<0.85 for novelty) and all(b>0.35 for
balance) and all(b-o>0.10 for balance,others) and all(g<0.20 for persist) and all(s>0.50 for single_rec) and
all(s<0.20 for single_oth) and all(ub<0.20 for untrained_best)` -- an ALL-6-SEEDS-AND-ALL-CRITERIA bar, read
directly from the runner's source, not paraphrased. Applying it per seed to this artifact's rows:

| seed | novelty | balance_min | blend_overlap_others | persistence_gap | single_recovered | single_overlap_others | untrained_best | per-seed PASS |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.000 | 0.000 | 0.000 | 0.188 | 0.818 | 0.826 | 0.000 | no |
| 43 | 0.552 | 0.500 | 0.118 | 0.077 | 0.966 | 0.154 | 0.000 | **YES** |
| 44 | 0.143 | 0.000 | 0.000 | 0.250 | 0.893 | 0.077 | 0.000 | no |
| 100 | 0.458 | 0.444 | 0.143 | 0.111 | 0.778 | 0.333 | 0.000 | no |
| 101 | 0.360 | 0.040 | 0.048 | 0.000 | 0.880 | 0.048 | 0.000 | no |
| 102 | 0.160 | 0.000 | 0.000 | 0.120 | 0.833 | 0.100 | 0.000 | no |

Every column above is quoted directly from the cited artifact (`blend_overlap_others` replaces the earlier
`balance-other` column so the table needs no derived arithmetic; the GO gate's own `balance - other > 0.10` test
is stated in prose above, and its per-row pass/fail is in the "per-seed PASS" column). <!--derived--> **1/6 seeds
(43) individually clear every criterion** -- consistent with, and now quantifying precisely, the runner's own
aggregate verdict `PARTIAL/NEGATIVE` (the ALL-6-seeds `go` boolean is False because even one failing seed fails
the whole gate).

<!--derived--> Failure tally across the 5 failing seeds: **`balance_min<=0.35` fails on 4/6** (42, 44, 101, 102) --
the dominant failure mode -- **`balance-other<=0.10` fails on the SAME 4/6** (mechanically downstream of the same
low balance_min), **`single_overlap_others>=0.20` fails on 2/6** (42 at 0.826, 100 at 0.333) -- the recovered
single-cue memory bleeds into a different stored memory's representation -- and **`persistence_gap>=0.20` fails on
1/6** (44, at 0.250). `genuine_formation=True` and `untrained_best_overlap=0.000` hold on **6/6** -- the load-bearing
BTSP write and the untrained-network anti-cheat are both clean; the residual is specific to the blend/completion
read, not to formation or to a leaky positive control.

## What "blend-balance collapse" looks like, and an honest open question about its cause

<!--derived--> `blend_balance_min` is the weaker of the two cue-driven assemblies' overlap with the released
(post-cue) settled state; a value near 0 means the settle did not preserve a meaningful trace of BOTH driven
memories. The per-assembly `blend_overlaps_released` rows show this is not one uniform pattern: seed 101's
released overlaps are `[0.360, 0.040, 0.048]` -- the state collapses almost entirely onto ONE of the two driven
assemblies (memory 0); seed 44's are `[0.393, 0.154, 0.062]`, a milder version of the same lean; seed 102's are
`[0.0, 0.28, 0.0]` -- collapses onto the OTHER driven assembly (memory 1) instead, so "which memory wins" is not
consistently the first- or second-driven one; seed 42's are `[0.0, 0.188, 0.0]` -- the settled state barely
overlaps ANY of the three stored assemblies substantially, even though that same seed's single-cue positive
control still recovers cleanly (`single_recovered=0.818`), so the blend/settle dynamics specifically are implicated,
not a general encoding failure for that seed.

<!--derived--> The most readily-available covariate is emergent assembly size (`assembly_sizes` per seed, e.g. seed
42 `[33, 16, 23]`, a ~2x spread). That story does NOT hold up against this artifact's own seed 101, whose sizes are
`[25, 25, 21]` -- the MOST EQUAL of all six seeds -- yet whose balance is among the worst (`0.040`). **This finding
does not name a mechanism** for the blend-balance collapse; naming one without a covariate that actually explains
seed 101 would be exactly the premature-verdict pattern this project's own workflow flags ("the comfortable verdict
is the START of the research, never the end"). The honest next step is a DIAGNOSTIC, not a fix: instrument the
blend-settle window directly per seed (per-driven-assembly firing-rate trajectory across `drive_steps`/`reset_steps`
/`hold_steps`, not just the endpoint overlap) against a wider covariate set (assembly size, `w_within`, `cross_dw`
at encode time, and cue drive ORDER/timing) before proposing a write-side or gain-side correction.

## Scaffolds / residuals

<!--derived--> Unchanged from [[2026-08-27-generative-wander-production-scale-PARTIAL]]: the emergent DG-selection
+ BTSP-formed membership and the dendritic dAP bistable latch are the substrate's own mechanism (no `sim/` edit,
no host-computed blend); the blend/drive/read protocol (`blend_cells_each`, `drive_pA`, `hold_steps`, etc.) is a
host-timed experimental protocol, the same scaffold class the whole `_gap5_`/`_generative_attractor_wander_*` lane
already accepts. `BRAIN_CONTINUOUS_IDEATE_SPIKING` (the default-OFF live idle-tick wire-in) is unaffected by this
finding -- no code changed, no flag moved; this finding only reports the previously-unrecorded verdict of the
verify that flag's own production-default decision was explicitly conditioned on ([[2026-08-27-generative-wander-production-scale-PARTIAL]]:
"any production-default flip [is conditioned] on a staged 6-seed --emergent cupy GPU verify"). That verify has now
run and reads PARTIAL/NEGATIVE (1/6) -- so the flag should stay default-OFF pending the diagnostic above, not be
flipped on the strength of the smaller-scope GO alone.

## Next (no-defer -- a wall defers a METHOD, not the capability)

Named, not built (out of scope for this pass -- see the companion prep doc
[[2026-09-04-swr-ltu-forward-band-homeostasis-and-ca3-wander-blend-prep]] for why): (1) the per-seed blend-settle
instrumentation above, cheap (reuses the existing `--emergent` build path, adds only a read of the existing firing
trace during the settle window that the runner already advances); (2) once a covariate is found, the natural
mechanism class is a homeostatic/competitive balancing process during the BLEND drive itself (analogous in FORM to
the forward-band homeostatic scaling this session's companion runner applies to the CA3 replay band, and to the
Turrigiano-class scaling already validated at `webapp/da_encoding_drives_chat.py::apply_substrate_homeostasis`) --
but proposing that mechanism now, before the diagnostic identifies what it would correct, is precisely the
overclaim this finding avoids.
