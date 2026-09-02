---
type: finding
status: live
date: 2026-09-02
mechanism: memory-separator-continuous-ring-parameter-screen
board: 91 (memory-separator: fix the READ, continuous-ring bump-attractor successor to sticky-slot NO-GO)
artifact: research/findings/raw/_consol_ring_screen/SCREEN_SUMMARY.json
---

# #91 memory-separator: a 216-config dendritic continuous-ring screen (first-pass, pool) — ring_spacing=2 is the promising region; a with-control re-run of the top configs is the next step

**2026-09-02.** The board-#91 successor to the sticky-slot storage NO-GO is a continuous-ring bump-attractor
(lay each memory as a stable bump on a ring). This session cluster-screened its parameter space on the mini-PC
pool (`_consol_dendritic_lineattractor_derisk --config-index 0..215 --seed 42 --dendritic-only`, 24-way parallel
after the pool concurrency fix). Artifact: `research/findings/raw/_consol_ring_screen/SCREEN_SUMMARY.json`.

## Honest scope: a parameter-space MAP, not per-config GO/NO-GO
`--dendritic-only` skipped the LINEAR control arm (to halve the screen's runtime), so each config reports the
dendritic arm's learning signal (`dw`, the CA1-slot weight change) but NOT the dendritic-vs-linear separability
verdict — every config's `VERDICT` field is `None` by construction. So this maps WHERE the ring learns strongly,
which is the input to picking configs for a full (with-control) separability run — it does not itself declare a
memory-separator GO.

## Result (116 of 216 configs harvested so far via pool_sync)
<!--derived-->
- Dendritic learning signal `dw` spans **0.0002 → 0.0489**.
- **`ring_spacing=2` is the clear promising region:** all top-8 configs by `dw` (~0.0486–0.0489) use spacing=2
  (vs spacing=1). k_thresh=2.0 and self_regen=0.1 dominate the top; lateral_exc / surround_inhib / slot_drive vary
  across the leaders (0.5–3.0 / 3.0–6.0 / 700–1400), so those are second-order within the spacing=2 basin. <!--derived-->
- Top configs: cfg 77 (spacing2, self_regen0.1, surr_inhib6, slot700), 95, 91, 87, 73 (see SCREEN_SUMMARY.json `top8_by_dw`).

## Next step (NO-DEFER, names its own next lever)
Re-run the top ~8 spacing=2 configs WITH the linear control arm (drop `--dendritic-only`) at 6 seeds → the
dendritic-vs-linear separability verdict (`both_win`) that the board-#91 read-side fix actually needs. A high `dw`
is necessary-not-sufficient (learning strongly ≠ keeping the two memories orthogonal); the with-control run is
what distinguishes them. This screen narrows a 216-config search to a ~8-config confirmatory run.
