---
type: result
status: live
date: 2026-11-02
mechanism: swap-drives-response (GNW thought-swap drive), adequate battery probe LB_SWAP_DRIVE_PROBE
prereg: research/findings/2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md
seeds: [42, 43, 44, 100, 101, 102]
verdict: LOAD-BEARING 6/6 under the adequate probe (decision credited to the spiking circuit; reply lead is a host template; probe brain, CPU)
runner: research/runners/load_bearing_fraction.py
---
# Swap-drives adequate-probe gate: LOAD-BEARING 6/6 (with caveats)

Date: 2026-11-02 · prereg: `research/findings/2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md` · pin
`b453350e5` · scorer `research/runners/load_bearing_fraction.py` (`_swap_drive_score`, `LB_SWAP_DRIVE_PROBE=1`) ·
data (dev seed s7 alongside):

- `research/findings/raw/_load_bearing/swap_drive_probe_b453350/s42/lb.json`
- `research/findings/raw/_load_bearing/swap_drive_probe_b453350/s43/lb.json`
- `research/findings/raw/_load_bearing/swap_drive_probe_b453350/s44/lb.json`
- `research/findings/raw/_load_bearing/swap_drive_probe_b453350/s100/lb.json`
- `research/findings/raw/_load_bearing/swap_drive_probe_b453350/s101/lb.json`
- `research/findings/raw/_load_bearing/swap_drive_probe_b453350/s102/lb.json`

## Result (the pre-registered headline rule: all six seeds LOAD-BEARING)

**6/6 LOAD-BEARING.** Every gate seed reads verdict `regressed`, `attributable_fraction` 1.0, and passes G1-G6:
exercised; contrast defined (intact swap trace on `sw_open` and `sw_hold`); null clean (intact_a == intact_b on all
three turns); contrast clean (`contrast_null_diffs` 0 on the no-swap-due turns); the reply changes on `sw_switch`;
the lesion rebuild reproduces. Dev seed s7 agrees.

- Intact `sw_switch`: `swapped=True`, `topic_change_swap`, `mm_peak` 0.24-0.33 <!--derived-->, answer "Setting the held thread
  aside — On cat, then — the cat eats the fish".
- Lesion (`BRAIN_SWAP_DRIVES_LESION=1`, mismatch/salience proposal drive zeroed): `swapped=False`,
  `mismatch_held_no_swap`, `mm_peak` 0.0-0.017 <!--derived-->, answer "the cat eats the fish". No arm abstained on any turn.

## Verification (verify-go, three independent refuters, none refuted)

- **Gate-cheat:** every bar is computed from recorded data; UNDEFINED conditions are checked before any positive;
  `LB_SWAP_DRIVE_PROBE=1` is in all 35 provenance sidecars.
- **Control integrity:** the arms differ only by the lesion env var; the lesion zeroes one drive (`mm_drive`).
- **Reproducibility / hosts:** all 35 sidecars carry `b453350e5`, `git_dirty` false, one `source_manifest_sha256`.
  `where_ran`: pool ran s42/s101 (+s7), workstation ran s43/s44/s100/s102; validity does not depend on host (the
  inputs are the in-code tiny-demo KB, numpy backend everywhere). The s42 rebuild matches the earlier `7f90034df`
  rebuild field for field.

## Caveats that travel with this result

1. **What changes is a templated prefix.** The reply difference is the host-articulation lead. Per the prereg only
   the swap-vs-hold DECISION is credited to the spiking circuit; topic extraction and the transition string are
   host templates. The circuit gates a template, it does not change factual content.
2. **Probe brain, opt-in flag.** Measured on the tiny-demo brain under `LB_SWAP_DRIVE_PROBE`, not the integrated
   production brain. Per the prereg, a pass grants no default change.
3. **Binary effect, identical text across seeds.** `boost_max` sits near 0.16 <!--derived--> on every intact switch; the seeds
   vary in margins, not in behaviour, so they are six clean replications of one binary outcome.
4. **CPU only.** Nothing here ran on the GPU backend.
