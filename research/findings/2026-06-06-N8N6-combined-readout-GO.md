# N8+N6 combined: genuine GPi→thal disinhibition + thalamic-source readout REMOVES the N8 cheat and BEATS the baseline — GO multi-seed (grid-8); N6 signal-source biologized + a documented host-argmax residual — 2026-06-06

**Status:** GO (N8 removed), gated on the cheat-5 multi-goal navigation score (sum of per-phase final-quarter mean
distance; LOWER is better), with the original tonic+motor-argmax cheat as the baseline, **multi-seed (42/43/44) at
grid-8**. Grid-32 production-scale confirmation IN FLIGHT. The opt-in flags are shipped; NO `sim/` edits. This is the
combined resolution of the N8 boundary (`2026-06-06-N8-thalamic-disinhibition-BOUNDARY.md`) — N8 was coupled to N6.

## The one-line result

The N8 boundary (genuine GPi→thal disinhibition navigates 3.4–4.4× worse than the tonic cheat) was ENTIRELY the
weak-signal readout. Fix: read action selection from the **cleanly-selective THALAMUS** (`--readout-source thal`,
the argmax over the four thal pool rates) instead of the **weak motor spike-counts** (the original cheat N6). With
that, genuine disinhibition navigates **as well as the tonic drive, and both BEAT the original cheat baseline.**

## The decisive table (grid-8, cheat-5 multi-goal sum-finalQ, LOWER is better)

| condition | seed 42 | seed 43 | seed 44 | reading |
|---|---|---|---|---|
| tonic + motor-argmax (original cheat baseline) | 5.03 | — | — | the "before" |
| genuine + motor-argmax (the N8-alone boundary) | 22.14 | — | — | weak motor readout fails |
| **genuine disinhibition + thal-readout** | **2.34** | **2.76** | **2.18** | **GO — seed-robust, beats baseline** |
| tonic + thal-readout (control) | 2.00 | 2.00 | 2.00 | the matched control |

Genuine disinhibition + thal-readout holds at 2.2–2.8 across all 3 seeds — within noise of the tonic control (2.0),
and far below the original cheat (5.0). **N8 is removable, multi-seed, at no performance cost (an improvement).**

## Why — N8 was coupled to N6

The tonic thalamic drive wasn't only a cheat; it compensated for a fragile readout. Genuine disinhibition produces a
*clean but weak* signal (the selected thal/motor released, the others at exactly 0.000). The original readout — a host
argmax over WEAK motor spike-counts — can't track a weak winner over a multi-goal run (→ 22). Reading instead from the
**thalamus** (the BG output relay, cleanly selective under disinhibition) gives the argmax a strong, unambiguous signal
(→ 2.3). The thalamic-source readout *also* helps the tonic baseline (5.0 → 2.0), confirming the readout was the
bottleneck. So the two cheats had to be converted together, as one BG output stage.

## Honest scope — N8 resolved, N6 PARTIAL

- **N8 (thalamic tonic drive → genuine disinhibition): RESOLVED.** The BG output gate now genuinely runs (GPi clamps
  the thalamus; the selected action's D1 releases it), and it navigates ≈ the tonic cheat (better than the original
  motor-argmax baseline), multi-seed. The opt-in `--genuine-thal-disinhibition` (gpi_tonic=1300, thal_tonic=750) is
  the new biologically-correct default candidate.
- **N6 (host argmax over motor counts): PARTIAL.** `--readout-source thal` biologizes the SIGNAL SOURCE — selection is
  read from the thalamus (the BG output relay; downstream motor areas do read thalamic output, so this is
  biologically more meaningful than reading weak motor counts). BUT it is STILL a host-side argmax, not a spiking
  decision. The biological-ideal spiking winner-take-all readouts were tested and were WORSE: motor-pool WTA 14.7,
  thalamic-reticular WTA 20.0 (vs thal-argmax 2.3). So the fully-spiking selection readout did NOT pan out; the
  host-argmax MECHANISM remains a documented residual (a deeper future target, analogous to the analog-whitening
  residual on the conversational side).
- **Operating point is sensitive** (gpi1300/thal750 works; gpi2200/thal600 = 22.5) — multi-seed confirms gpi1300/thal750
  is seed-robust, but the conversion ships with that specific operating point, not a wide basin.
- **Scale:** grid-8 multi-seed (the smoke scale). Grid-32 production-scale confirmation IN FLIGHT (genuine+thalread vs
  the original tonic+motor baseline) — appended below when it lands.

## Artifacts
- `research/runners/g11_bg_runner.py` — `--genuine-thal-disinhibition` (+ `--genuine-gpi-tonic-pa`,
  `--genuine-thal-tonic-pa`) and `--readout-source {motor|thal}` (default `motor` = original preserved). Additive,
  default-preserving, NO `sim/` edit.
- `research/findings/raw/_n8n6_*thalread*.json` (grid-8 multi-seed), `_n8n6_g32_*` (grid-32, pending).
- Prior: `2026-06-06-N8-thalamic-disinhibition-BOUNDARY.md` (the boundary this resolves),
  `2026-06-06-N8-step1-verification.md`.

## Net

The first navigational cheat is converted: the BG output stage (N8 thalamic gating) now runs as genuine GPi→thal
disinhibition, reading selection from the thalamus — and it BEATS the cheats it replaced (2.3 vs 5.0), multi-seed at
grid-8. N6's signal source is biologized; its host-argmax mechanism is a documented residual (the spiking-WTA ideal
was worse). Next: grid-32 production confirmation, then cheat N5 (Manhattan-distance reward → sensed beacon gradient).
