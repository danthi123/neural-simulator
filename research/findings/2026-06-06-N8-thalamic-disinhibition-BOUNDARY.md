# N8 (thalamic-drive cheat) via genuine GPi→thal disinhibition = BOUNDARY when removed ALONE — the disinhibition is mechanistically PERFECT (selectivity) but the nav score collapses 3.4–4.4× because the released rates are too WEAK for the host-argmax readout (cheat N6); N8 is COUPLED to N6 — 2026-06-06

**Status:** BOUNDARY (N8 removed alone), gated on the cheat-5 multi-goal navigation score (sum of per-phase
final-quarter mean distance; LOWER is better), with the tonic-drive cheat as the matched CONTROL, seed 42 robust
across 5 configs (a gpi/thal sweep + cluster-A ablation). First conversion attempt of the navigational
cheat-removal arc. The opt-in `--genuine-thal-disinhibition` flag is shipped (default OFF = the cheat preserved);
NO `sim/` edits.

## The one-line result

Replacing the thalamic tonic drive (N8: 300 pA externally pacing the relay) with genuine GPi→thal disinhibition
(GPi pacemaker → the selected action's D1 silences its GPi → thalamus released) is **mechanistically PERFECT** — at
the probe level, driving the selected action silences its GPi, releases its thalamus, and leaves all non-selected
motor pools at **exactly 0.000 firing** (strictly better selectivity than the tonic cheat, where all four thalamic
pools fire roughly equally). **But the navigation SCORE collapses 3.4–4.4×, and no (gpi, thal) tuning or cluster
ablation closes the gap.** Diagnosis: the released motor rates are CLEAN but WEAK (motor_selected ~0.016
spikes/neuron/step), and the production action readout — cheat **N6**, a host-side argmax over motor SPIKE COUNTS —
cannot reliably read weak rates over a full multi-goal run. **The tonic drive was COMPENSATING for the readout's
fragility. N8 is COUPLED to N6.**

## The decisive table (seed 42, cheat-5 multi-goal sum-finalQ, LOWER is better)

| config | genuine disinhibition (N8 fixed) | tonic cheat (control) | ratio |
|---|---|---|---|
| base (A+E flagship) | 22.34 | 5.03 | 4.4× worse |
| thal_tonic = 900 | 19.42 | — | ~3.9× |
| gpi = 1500, thal = 1200 | 20.78 | — | ~4.1× |
| no-cluster-A | 16.99 | 4.99 | 3.4× worse |

The best genuine config (no-cluster-A, 16.99) is still 3.4× worse than its tonic control (4.99). The gap is robust
to every tuning tried — NOT a tuning artifact.

## Why — the mechanism (N8 is coupled to N6)

- **Tonic cheat (N8 on):** all four thalamic pools are externally driven to fire strongly; the heuristic perturbs
  one channel; the host-argmax readout reads a strong differential → reliable selection (score ~5.0).
- **Genuine disinhibition (N8 fixed):** only the selected thalamus is released, cleanly (others exactly 0.000), but
  the released rate is WEAK (motor ~0.016). The host-argmax readout, over a noisy multi-goal run, can't track the
  weak winner reliably → degraded selection (score 17–22).
- So the thalamic tonic drive was not merely a cheat — it was **compensating for the weak-readout problem (N6, the
  host-side argmax).** Genuine disinhibition exposes that the readout is too fragile to read clean-but-weak release
  signals.

## What this means for the conversion plan

**N8 cannot be removed in isolation; it must be converted TOGETHER with N6.** The principled biological fix is the
BG output stage as a whole: GPi→thal disinhibition releases the selected action, AND an on-substrate spiking
competition (motor-pool winner-take-all / mutual inhibition) robustly amplifies the released action into a clean
selection — replacing the host-side argmax (N6). This removes TWO cheats at once and is the biologically-correct
selection circuit (downstream spiking competition, not a host computer counting spikes). **Next step: the combined
N8+N6 de-risk — does a spiking WTA readout on the genuine-disinhibition signal close the 3.4× gap?**

## Honest scope

- Seed 42, but robust across 5 configs (gpi/thal sweep + cluster-A ablation). A 3.4–4.4× gap is far outside seed
  noise (~±0.5–1), so the BOUNDARY is decisive without multi-seed; multi-seed would only confirm a settled gap.
- The disinhibition MECHANISM is validated (perfect selectivity) — this is NOT a failure of the disinhibition; it is
  the coupling to the downstream readout (N6).
- `--genuine-thal-disinhibition` is shipped (default OFF; chosen operating point gpi_tonic = 1000, thal_tonic = 600)
  — preserved for the combined N8+N6 work.

## Artifacts
- `research/runners/g11_bg_runner.py` — `--genuine-thal-disinhibition` opt-in flag (default OFF, additive).
- `research/runners/_n8_thal_disinhibition_probe.py` — the STEP 1 mechanism probe (per-action GPi/D1/thal/motor
  firing, tonic vs genuine).
- `research/findings/2026-06-06-N8-step1-verification.md` — the mechanism verification + weight-scale de-risk.
- `research/findings/raw/_n8_smoke_*.json` — the nav-score runs (genuine vs tonic, 5 configs).
