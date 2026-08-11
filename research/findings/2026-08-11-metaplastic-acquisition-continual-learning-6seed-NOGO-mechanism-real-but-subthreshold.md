---
type: finding
status: contributing
date: 2026-08-11
mechanism: metaplastic e-prop acquisition (Fusi/Benna-Fusi per-synapse consolidation state gating lr_eff = lr/(1+g*c)) against the continual acquisition-at-scale forgetting
lane: H-memory / continual-learning
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO (1/6 strict) — the mechanism is REAL, load-bearing, and attributable, but the mean improvement is just below the strict +0.15 margin bar on 5/6 seeds
runner: research/runners/_teacher_loop_metaplastic_acquisition_derisk.py
artifacts:
  - research/findings/raw/metaplastic_acq_s42.json
  - research/findings/raw/metaplastic_acq_s43.json
  - research/findings/raw/metaplastic_acq_s44.json
  - research/findings/raw/metaplastic_acq_s100.json
  - research/findings/raw/metaplastic_acq_s101.json
  - research/findings/raw/metaplastic_acq_s102.json
instrument: N-sweep {16,32,50,100} facts acquired by e-prop weight change; `frac_recalled` over the acquired set. Four arms: vanilla (metaplasticity OFF, the acquisition-at-scale forgetting baseline), metaplastic (per-synapse consolidation state `c` raising effective threshold on already-consolidated synapses), meta_lesion (state frozen at 0 — the load-bearing control), meta_permute (state applied to the WRONG synapses — the attribution control). de-clamp bdsp_wmax held constant across arms so it is not the lever. SIM_BACKEND=numpy.
---

# Metaplastic e-prop against continual acquisition-at-scale forgetting — the mechanism is REAL (load-bearing + attributable) but SUB-THRESHOLD at 6 seeds (1/6 strict GO): a NO-GO that names the next mechanism

The interleaved-generative-replay NEGATIVE (`2026-08-11-continual-retention-interleaved-generative-replay-N50fact-NEGATIVE.md`)
isolated the real continual-learning bottleneck: facts acquired by the e-prop WEIGHT CHANGE show `frac_recalled ~ 1/N`
as N grows — later facts overwrite earlier ones in the shared weights. It named METAPLASTICITY (a per-synapse
consolidation state that protects already-acquired facts) as the next mechanism. This de-risk builds + 6-seed-tests it.

## Result — 6/6 seeds, N∈{16,32,50,100} (`research/findings/raw/metaplastic_acq_s*.json`)

<!--derived-->
Cross-seed means (derived over the 6 per-seed artifacts): **metaplastic `frac_recalled` 0.257 vs vanilla 0.120**
(a +0.137 mean improvement — the acquisition-at-scale forgetting IS moved the right way), with the two anti-cheat
controls collapsing back to vanilla: **meta_lesion 0.120** (freezing the consolidation state at 0 lands exactly on
vanilla — the STATE, not the code path, does the work) and **meta_permute 0.125** (applying the state to the wrong
synapses also collapses — the SPECIFIC per-synapse targeting is the drive). Immediate-acquisition ~0.97 in both arms
(no cost to learning new facts). So the mechanism is genuine, load-bearing, and correctly attributed.

**But the strict GO gate is NOT met: 1/6 seeds.** The gate requires metaplastic > vanilla+0.15 AND > lesion+0.15 AND
> permute+0.15 on EVERY seed; the mean improvement (+0.137) sits just BELOW the +0.15 margin, so 5/6 seeds miss the
bar. **VERDICT: NO-GO (strict), mechanism-real-but-sub-threshold.** The 1-seed smoke (+0.19–0.25) was the optimistic
tail of the seed distribution.

## Scope / honesty + the named next mechanism (per THE LAW — the capability stays OPEN)

<!--derived-->
NO-EXTERNAL-NEEDED: this is a quantitative sub-threshold verdict on a specific single-timescale implementation, not a
fundamental-limit claim — the biology (Fusi/Drew/Abbott 2005 cascade; Benna & Fusi 2016 multi-timescale) is cited and
the named surpass is a DEEPER version of the SAME mechanism, so no new external read is required to bank this negative.

- **What holds:** metaplasticity moves acquisition-at-scale forgetting in the right direction with a load-bearing,
  correctly-targeted per-synapse state and no cost to new-fact acquisition. That is a real positive signal on the crux.
- **Why it is sub-threshold:** the single hidden consolidation variable `c` lifts the MIDDLE of the retention curve
  but does not fully protect the very-OLDEST facts (fact 0 is still overwritten at N=100 in both arms). The +0.137
  mean is the average of "middle protected, oldest lost."
- **Named next mechanism:** a TRUE multi-timescale **Benna-Fusi consolidation CHAIN** (a cascade of variables at
  increasing timescales) rather than the single hidden variable here — the slow variables protect the oldest facts the
  single variable cannot reach. Secondary lever: a `meta_gain` sweep (the current g=8 may be sub-optimal). No `sim/`
  edit was needed (the per-synapse state lives in the runner subclass `MetaplasticEpropNet`).
- Runner-side, reuse-by-import of `OnBridgeEpropNet` + the teacher-loop machinery. de-clamp held constant across arms.
