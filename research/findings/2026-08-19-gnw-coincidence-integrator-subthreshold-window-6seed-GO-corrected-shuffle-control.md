---
type: finding
status: de-risk-GO-window-6seed
date: 2026-08-19
mechanism: gnw-workspace
artifacts:
  - research/findings/raw/_gnw_coincidence_integrator/dsub_robust_1300_6seed.json
  - research/findings/raw/_gnw_coincidence_integrator/dsub_robust_1500_6seed.json
  - research/findings/raw/_gnw_coincidence_integrator/dsub_robust_1700_6seed.json
  - research/findings/raw/_gnw_coincidence_integrator/calibration_seed42.json
---

# GNW coincidence-integrator — the subthreshold-drive coincidence is a WIDE WINDOW, not a knife-edge tuned at 1400: 6/6 GO at d_sub = 1300, 1500, 1700 pA (18 seed-runs), all controls clean

## What this adds over the 2026-08-12 6/6-GO finding
The 2026-08-12 finding
([`2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md`](2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md),
`status: de-risk-GO-6of6`) established the SUBSTRATE combining two subthreshold organ reads via coincidence-ignition +
2-hop re-entry, 6/6 GO at a SINGLE drive point (d_sub=1400 pA) with the corrected shuffle control. Its anti-cheat #2
— "D_SUB sits cleanly in the coincidence window, not a threshold-tuning artifact" — was supported by the single-seed
(seed-42) solo-drive calibration curve plus that single 1400 operating point. This de-risk widens that to a
DRIVE-MAGNITUDE SWEEP: three independent 6-seed runs at d_sub = 1300, 1500, 1700 pA. Every one is a full 6/6 GO with
all anti-cheat controls collapsed, so the mechanism holds across a ~400 pA band on every seed — a genuine window, not
a value tuned to 1400. No `sim/` edit; reuse-by-import of the P1.2 spiking workspace; the corrected shuffle control
(route organ C's off-target vote to an EMPTY slot, established 2026-08-12) is the one used throughout.

## Result — the window sweep (numpy, runner SHA f3d15f7, seeds 42/43/44/100/101/102)
Each drive point is an independent 6-seed run. All three: `all_go=true`, `n_go=6/6`, `all_moat_ok=true`; per-seed
`seed_go=true` on all six (18/18 GO).

<!--derived-->

| d_sub (pA) | n_go | coincidence_2hop_acc | query_chain (host parity) | r_only | c_only | disagree | shuffle | onecycle | lesion | single_hop_reflex | mutual_exclusion |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1300 | 6/6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1500 | 6/6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1700 | 6/6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

At every point `mean_coincidence_2hop_acc` equals `mean_query_chain_2hop_acc` (1.000): the synaptic coincidence path
reaches the same 2-hop conclusion as the host one-shot baseline (parity). Every ablation of the synaptic mechanism
collapses to 0.000 on all six seeds — R-only, C-only (single organ subthreshold, the anti-host-if-else), disagree
(consensus-veto), shuffle (spatial-slot congruence, corrected control), onecycle (no re-entry), lesion (no ignition)
— while the single-hop recall reflex survives at 1.000 (the dissociation) and single-content access holds
(`mutual_exclusion_frac=1.000`). The relation-blind spreading-activation baseline `mean_spreading_floor` averages
0.083 (per-seed 0.000 or 0.125), well under the gate's `coincidence >= spreading_floor + 0.5` bar and above the
per-concept guess level of 0.025 (1/40 concepts).

## Why this is a WIDE window, not a knife-edge (the hardening of anti-cheat #2)
The committed seed-42 calibration (`calibration_seed42.json`, `knee_pA=2400`, `THR=0.167`) shows a single organ's
solo drive stays subthreshold across the whole span (solo-rate 0.037/0.027/0.024/0.030 at 1200/1400/1500/1700 pA, all
< 0.05) while the ignition knee is 2400 pA — so `2*d_sub` in [2600, 3400] is suprathreshold at every window point and
a lone `d_sub` is not. This sweep promotes that seed-42 curve to a 6-seed statement at three drive magnitudes:
`r_only_acc = c_only_acc = 0.000` on all 6 seeds at all three d_sub is "one read alone never ignites, on every seed,
across the band"; `coincidence_2hop_acc = 1.000` is "the coincidence of two does". Because both the LOW edge (1300)
and the HIGH edge (1700) — roughly ±300 pA around the 1400 midpoint main already de-risked — are equally clean 6/6,
the coincidence bifurcation is a property of the ignition dynamics over a wide subthreshold band, not an artifact of
tuning the drive to a single value. The corrected shuffle control (empty-slot reroute → a single subthreshold vote)
collapses to `shuffle_acc=0.000` at each of the three points too, confirming the 2026-08-12 fix generalises off the
1400 midpoint.

## Honest scope (this HARDENS a de-risk; it is not a closure)
1. **Not "closed" (docs/TERMS.md).** A default-off de-risk runner, not the shipped production path. Closure = wiring
   the ignition bus into `webapp/server.py`'s host organ-orchestration; the 2026-08-12 finding lays out that path
   (its (a)–(c) as the write+integrate+re-enter primitive; (d)+(e) and a genuinely-distinct second organ remain).
2. **Both organ reads still come from the composer** (recall under two relations), as in the 2026-08-12 de-risk. This
   sweep hardens the robustness of the SUBSTRATE-COMBINATION claim across drive magnitude; it does not add a second
   independent organ.
3. **Drive points, not a continuous sweep.** Three points (1300/1500/1700) plus the 1400 midpoint already on main make
   four clean 6-seed points spanning ~1300–1700 pA; the claim is scoped to those measured points plus the calibration
   argument, not a continuously-verified interval.
4. **Backend.** The window sweep is numpy (the project's CPU-cheap-first path); the seed-42 calibration/smoke and the
   1400 6-seed on main were cupy. Same runner; the mechanism is backend-agnostic.

## Files
- Window sweep (this finding): `research/findings/raw/_gnw_coincidence_integrator/dsub_robust_{1300,1500,1700}_6seed.json`
  (+ `.prov.json` sidecars; argv records `--d-sub {1300,1500,1700} --D 256 --backend numpy --seeds 42 43 44 100 101 102`).
- Ignition-knee calibration (seed 42): `research/findings/raw/_gnw_coincidence_integrator/calibration_seed42.json`.
- Runner (committed on main, corrected shuffle control, reuse-by-import of P1.2; NO `sim/` edit):
  `research/runners/_gnw_coincidence_integrator_derisk.py`.
