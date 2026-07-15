# The vectorized selective-SSM scale trajectory's apparent "fluency crossover" (sel beats bigram as V grows) is the BIGRAM-STARVATION CONFOUND — against a TUNED add-k bigram the selective SSM LOSES at every V and the gap GROWS with V; the deep-tail mechanism (vs the memoryless bag) is genuine. Second catch of this confound this turn.

**Date:** 2026-07-15 · **Status:** HONEST NEGATIVE on the "fluency crossover" (the deliverable); the deep-tail mechanism sub-result stands. numpy CPU, gaming-compatible; NO `sim/` edit. The tuned-bigram control was run BEFORE any GO — the discipline caught it.

## What was tempting

The vectorized selective-SSM scale trainer (`_reslm_scale_trained_selssm_vectorized_derisk.py`, the enabler for the fluency scale run) was validated at the smoke (sel_lift +0.731 reproduces the slow runner ~+0.62). Extending it up the vocabulary axis on TinyStories (np=200, n_train=3000) produced a trajectory that LOOKS like the fluency crossover:

| V | sel_ce | add-1 bi_ce | sel_lift (vs bag) | sel_over_**add-1**-bigram |
|---|---|---|---|---|
| 120 (smoke) | 2.996 | 2.963 | +0.731 | −0.033 |
| 200 | 3.264 | 3.214 | +0.977 | −0.050 |
| 400 | 3.996 | 4.025 | +1.069 | **+0.029** |
| 600 | 4.355 | 4.518 | +1.170 | **+0.163** |

`sel_over_bigram` goes from negative to **positive** as V grows — the "selective SSM overtakes the bigram as the language gets richer" story the fluency trajectory wants.

## The tuned-bigram control REFUTES it (the same P2 discipline as the reservoir-LM finding today)

The `bi_ce` above is the **add-1** bigram, which starves as the V×V table sparsifies (V=600 → 360K cells, ~30–60K train pairs). Replicating the runner's exact data/split (`default_rng(42).permutation`, `ev`=last 400, `tr`=first 3000) and fitting a **tuned add-k** bigram (k swept):

| V | sel_ce | add-1 bi | **TUNED bi (k≈0.03)** | sel − tuned |
|---|---|---|---|---|
| 200 | 3.264 | 3.214 | **2.862** | **+0.402 (sel LOSES)** |
| 400 | 3.996 | 4.025 | **3.380** | **+0.616 (sel LOSES)** |
| 600 | 4.355 | 4.518 | **3.726** | **+0.629 (sel LOSES)** |

**Against a properly-tuned bigram the selective SSM loses at EVERY V, and the gap GROWS with V** (+0.402 → +0.629) — the OPPOSITE of a crossover. The apparent overtaking was purely the add-1 bigram inflating (2.96 → 4.52) faster than the model as the table starved. No fluency crossover.

## What is genuine (survives)

`sel_lift` is measured vs the **memoryless bag** (a fair baseline, not a starving bigram) and grows monotonically (+0.731 → +1.170) — the **deep-tail selective mechanism holds and strengthens with V**, consistent with the validated Rung-3/4 result. The per-depth deep margins (sel < bigram at d≥6/10) are also vs the add-1 bigram, so their magnitude is starvation-inflated, but the mechanism (a learned input-dependent gate holds distal context the reservoir/bag can't) is real and independently established.

## The mission implication (and the real lever)

- **Overall fluency is tuned-bigram-bound at tractable scale** — re-confirmed for the selective-SSM generator, exactly as the earlier honest scoping (ROADMAP §12) states: the overall model is bigram-level until real-conversation scale.
- **Scaling VOCAB at FIXED data DIVERGES from the tuned bigram** (the gap grows) — so vocab is the wrong lever; the mission-central lever is **DATA** (more tokens at fixed V). Whether more data closes the gap to the tuned bigram is the genuinely-open question (probe running: `raw/_reslm_data_lever_probe.log`, V=200, n_train 3000→6000→10000, sel vs tuned bigram). That data-scale run is **core-parallelism-bound** (gaming-incompatible at full parallelism).

## Method note — the confound caught TWICE this turn

This is the SECOND time this turn the bigram-starvation confound produced a tempting-but-false "beats the bigram / crossover" (first: the emergent-input-representation gate, `2026-07-15-emergent-input-representation-...`; now: the selective-SSM scale trajectory). Both were caught by running the **tuned add-k bigram control BEFORE the GO**, never the add-1. Standing rule reinforced (matches ROADMAP §12's own warning): on any "beats the bigram" language result, the baseline MUST be a tuned/interpolated n-gram on the same split — the add-1 bigram is a starved strawman at any non-trivial V.

## Artifacts
- Trajectory: `raw/_reslm_vec_scale_trajectory.log` + `raw/_reslm_vec_scale_V{200,400,600}.json`; smoke anchor `raw/_reslm_scale_trained_selssm_vec_smoke.json`.
- Tuned-bigram control: inline (replicated the runner's split); data-lever probe: `raw/_reslm_data_lever_probe.log`.
