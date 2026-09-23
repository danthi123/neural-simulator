---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
---

# All-fixes adequate-probe battery, 6 seeds, ONE consistent run: robust core 24 of 26 exercised; mean load-bearing fraction 0.949 ± 0.018 (2026-09-23) <!--derived-->

The first single, consistent 6-seed measurement of the #1 metric (lesion-verified LOAD-BEARING FRACTION) on current main
with the adequate drive probes AND every merged fix flag on. Until now the robust-core count (20 → 22 → 23) was assembled
from separate per-faculty verifications; this run measures all faculties together under one configuration.

## Configuration
- Runner `research/runners/load_bearing_fraction.py`, sharded one faculty × one seed per job by `tools/lb_shard.py`
  (tag `allfixes2`, 31 measurable faculties × 6 seeds = 186 shards), `--repeats 2`, `SIM_BACKEND=numpy` (deterministic).
- Flags (all default-OFF on main, all with code on main): fixes `BRAIN_EPISODIC_STORE_VERIFY`, `BRAIN_PMEM_FACILITATION`,
  `BRAIN_PMEM_OP_STABILIZER`, `BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE`; adequate probes `LB_EPISODIC_DRIVE_PROBE`,
  `LB_SURPRISE_CONFIRM_PROBE`, `LB_DISCOURSE_REGISTER_DRIVE_PROBE`, `LB_CG_DRIVE_PROBE`, `LB_NONCONTRADICTION_DRIVE_PROBE`,
  `LB_AFFECT_DRIVE_PROBE`, `LB_BG_SELECT_DRIVE_PROBE`, `LB_PMEM_DRIVE_PROBE`, `LB_OPEN_ENDED_DISTRIB_PROBE`.
  `BRAIN_AFFECT_MARKER_SETTLE` was NOT on (it was not yet merged when the battery started).
- Compute: two AWS r7i.4xlarge instances, 15 shards in parallel each (s100-s102 on one, s42-s44 on the other). Each got
  a code snapshot of main's working tree (≈09:45 and ≈11:40 local) plus `data/corpus/`; both snapshots contain all four
  fixes. The instances were synced without `.git`, so every shard's provenance sidecar records `git_sha: unknown` —
  the revision is asserted by the deploy procedure, not by provenance. Backend `numpy` is recorded on every shard.
- An earlier tag (`allfixes`) is INVALID and excluded: it ran before `data/corpus/` was shipped, and corpus-learned
  comprehension faculties read a false NOT-load-bearing (`research/findings/raw/_load_bearing/_shards/allfixes/INVALID.txt`).

## Result
<!--derived from research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json -->
Artifact: `research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json` (186 per-shard `lb.json` under
`research/findings/raw/_load_bearing/_shards/allfixes2/s<seed>/<faculty>/`).

- **Exercised: 26 faculties on every seed.** **Robust core (load-bearing on all 6 seeds): 24 of 26.**
- Per-seed load-bearing fraction: s42 0.9231, s43 0.9231, s44 0.9615, s100 0.9615, s101 0.9615, s102 0.9615 —
  **mean 0.949, SD 0.018** (the 2026-09-21 adequate battery, before the four fixes: 0.85 ± 0.03, robust core 20). <!--derived-->
- No exercised faculty has a dirty null control or an UNRELIABLE flag on any seed.
- Outside the robust core, among exercised faculties:
  - `affect-marker-spiking-wta` 4/6 (not load-bearing on s42, s43). The separate SETTLE contrast
    (`2026-09-23-affect-marker-settle-fullbrain-contrast-PARTIAL-6seed.md`) measured it 6/6 WITH `BRAIN_AFFECT_MARKER_SETTLE`
    — that has not been measured inside a combined battery, so it is not counted here.
  - `da-gated-encoding` 0/6 (`pass` on every seed). Today's merged v3 mechanism (DA-gated synaptic tag/late-LTP under a
    natural drive, `2026-09-23` DA findings) was not enabled in this battery.
- Not exercised by the battery's probes (thin probe, reported separately, not in the denominator):
  `swap-drives-response`, `wm-binding-advanced`.

## Option-C report (the owner-ratified pairing)
**Robust core 24/26 exercised under the adequate probes with every merged default-OFF fix flag on**, paired with the
shipped default: the thin-probe production battery with all flags OFF read **~0.59** (robust core 14) on 2026-09-20
(`2026-09-20-load-bearing-fraction-6seed-0.59-robust-core-14.md`) and was not re-measured in this run. The gap is probe
coverage plus default-off fixes, not a new brain capability, and no production default changed.

## Honest residuals
- Code revision not recorded in provenance (see Configuration); both snapshots postdate the four fix merges.
- 24 is a count under one configuration and one probe set; `affect-marker` and `da-gated-encoding` have mechanisms merged
  today that this battery did not switch on — a combined battery with those flags is the next measurement.
- `swap-drives-response` and `wm-binding-advanced` remain unmeasured (thin probes); the language lane's learned referent
  detector is the named route to exercising `wm-binding-advanced`.

## Honesty
Functional read-outs only: "load-bearing" means lesioning the faculty's spiking contribution provably changes the reply
with a clean null control. No felt or phenomenal claim.
