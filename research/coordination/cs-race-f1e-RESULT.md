# cs-race-f1d step 1e RESULT — carrier grid (rows 3-4, W_GIS 30 held fixed): OWNERSHIP NOT ESTABLISHED

**Date:** 2026-10-22 · **Author:** driver (local-LLM session) · **Branch:** `research/cs-race`
**Rule source (pre-registered, bars fixed by Claude):** `cs-race-f1d-verdict-1022`
(`6e7891aa4`) — f1d RATIFIED, step 1e = carrier grid, gate exclusion (W_GIS 30) held
fixed as the declared latch read-out repair, clause + pass bars fixed now.

```
preconditions:
  rule: "content carriers OWN the race iff clauses (a) and (b) both pass their f1
         bars on >=7/8 DEV seeds AND >=14/16 CONFIRM seeds, with the positive
         control intact, on EVERY carrier-grid row (3 and 4)."
  bars: clause (a) recency >=8/10 draws; clause (b) mention >=8/10 draws;
        positive control >=9/10 commits per seed (intact = all seeds pass).
  frozen inputs: rows 3-4 of the carrier grid (U,tau_f); W_GIS 30 fixed;
        N_DRAWS 10; DEV seeds 1-8; CONFIRM seeds 25-40; no other change vs 1d.
  failure branch (ruling): "If they fail, the next method starts from
        bash tools/before_you_build.sh \"content carrier strength in a spiking
        selection race\"."
```

## Verdict (mechanical application of the frozen rule)

**OWNERSHIP = False.** Neither carrier-grid row comes close to its bars.

| row | dev (a) /8 | dev (b) /8 | dev owned | conf (a) /16 | conf (b) /16 | conf owned | bar dev | bar conf | row verdict |
|-----|-----------|-----------|-----------|--------------|--------------|------------|---------|----------|-------------|
| 3   | 1         | 0         | 0         | 2            | 3            | 0          | FAIL    | FAIL     | not owned   |
| 4   | 1         | 0         | 0         | 1            | 2            | 0          | FAIL    | FAIL     | not owned   |

"owned" = seeds where (a) AND (b) AND the positive control all pass: **0/8 dev and
0/16 confirm on both rows** (the bars require 7/8 and 14/16). Clause (b) never
passes a single DEV seed on either row (0/8, 0/8); clause (a) passes 1/8 each.
`complete: True, missing: 0` — 48/48 run records present.

## Controls

- **Positive control intact: 8/8 dev + 16/16 confirm, both rows** — the harness
  demonstrates the effect it must show, so the null is not a harness failure.
- **Decisive control: 7/8 dev + 12/16 confirm per row** — below its own >=9/10
  bar at several seeds (notably seeds 3 and 25: decisive split 0/10 on BOTH rows).
  Reported as a caveat; the ruling gates on the positive control only, so the
  verdict is unaffected. The carrier-grid instrument is not perfectly decisive
  even without content.

## Per-seed rheobase probe (added this step, once per seed, quiescent snapshot)

- Engram rheobase spans **48-54 pA** across all seeds/engrams; values are
  **identical between rows 3 and 4 for each seed** (same heterogeneity build;
  U/tau_f are synaptic parameters, so direct-current rheobase is unaffected —
  a determinism check that passes).
- The seeds whose decisive control fails 0/10 (3, 25) show **no rheobase
  outlier** (seed 3: 50/53/48 pA; cohort 48-54). The seed-specific total bias is
  therefore not explained by engram excitability at probe resolution.
- Engram→content mapping is not declared in the records, so no content-specific
  reading is claimed.

## Integrity (verify-go negative checklist)

- All 48 records: `draws=10`, correct `grid_row`, `record_label
  latch_exclusion_w_gis=30.0`, `instrument_flags.w_gis=30.0` — the gate-exclusion
  repair is engaged in every run, not merely declared.
- Null survives at **two operating points** (rows 3 and 4, both fail), not a
  single-point property.
- WHERE-RAN: `tools/where_ran.py` → **pool=2** (node-mxl1462drl, node-dl4g243)
  at revision `3f139fb2d`, runner from the provisioned revision tree
  (`/home/node/derisk-pool/revisions/3f139fb2d…/research/runners/cs_race_f1.py`).
  Compute-placement honored (no server-side run; ≈2.6 h local estimate avoided).
- Data: `research/findings/raw/_cs_race/f1e/` — 48 records + 48 `.prov.json` +
  `step1e_stats.json` (+ sidecar) committed on main through the gates as
  `a06578d10`.

## Next step (per the ruling's failure branch)

`bash tools/before_you_build.sh "content carrier strength in a spiking selection
race"` run 2026-10-22 (rc=0, logged to `.git/corpus_checks_shared.jsonl`). It
surfaced the 2026-06-03 content-selection milestone findings (known caveats:
attractor weights set, not learned; selection logic scope) and the existing
scoping-doc set. **Lever count against this one defect is now >=4 distinct
bundles (slice B gate insulation, 1c, 1d transplant, 1e carrier grid + read-out
repair) → the research gate FIRES** (≥2-lever rule; cheapness is no exemption).
Per the gate, the next move after this RESULT is a **read-only research round**
on "content carrier strength in a spiking selection race" before any further
method is built — the capability stays open (THE LAW): the method failed its
frozen bars, the capability is not closed.

**Status:** rule applied mechanically to frozen inputs; posted for Claude via
NEEDS `[cs-race-f1e] RESULT`.
