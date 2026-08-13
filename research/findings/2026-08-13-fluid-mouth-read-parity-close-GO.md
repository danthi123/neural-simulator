---
type: finding
status: qualified
claim_check: synthesis
verdict: >
  GO (6-seed, 6/6). CLOSING the fluid-mouth GRADED-conductance read's last ~8% (recov_argmax_mass 0.921 -> 0.9775 mean /
  0.9643 min, argmax_agree 0.711 -> 0.85) with TWO measured, brain-based, additive mechanisms on top of the parent
  graded read — NOT the finding's first-named
  rung. A READ-WINDOW sweep (150/450/900 steps, FLAT at recov ~0.88) proves the residual is NOT variance-limited, so a
  facilitating LIP ramp-to-threshold accumulator (which re-integrates the SAME margin, argmax-preserving) CANNOT move
  it; a LINEAR learned read fails too (per-position corr(margin, head_w@h) ~ 0, least-squares [g_e,g_i]->logit
  degenerate). The residual DECOMPOSES into two SYSTEMATIC biases, each closed by a substrate mechanism: (1)
  FEATURE-CODE FIDELITY — the hidden feature is rate-coded by ONE neuron per dim; a DENSER population code (hid_pop
  1->4; Zohary-Shadlen-Newsome pooling) lifts reconstruction argmax 0.82->0.875 and recov 0.879->0.942 (plateaus by
  hid_pop=8); (2) the BASE-RATE PRIOR — the true logit is head_w@h + HEAD_B and the read OMITS head_b (ceiling 0.856
  argmax); injected as a per-pool TONIC BASELINE CONDUCTANCE (prior-as-starting-point, Mulder 2012), scaled s =
  0.5*std_over_pools(margin) (calibrated ONCE on seed 42, a WIDE 0.35-0.5 plateau; consistent across seeds), lifts recov
  0.942 -> ~0.98. Both are argmax-CHANGING and substrate-native (a population size; a fixed per-pool conductance), 0
  host categorical draws on the read path. Anti-cheats 6/6: scramble->chance, zero-feature-input->chance, provenance 0
  host draws; the inhibitory shadow (NEGATIVE weights) stays LOAD-BEARING 6/6 (signed argmax-agree > positive-only). NOT
  "fully spiking" / NOT wired into production / NOT "the mouth works": the hidden h is still a host residual, the WKV
  store BPTT-trained, the read-out weights host-designed; runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1
date: 2026-08-13
mechanism: >
  Graded-conductance signed read (parent) + (1) a DENSER hidden population rate code (hid_pop 1->4: N neurons per
  feature-dim instead of 1 -> lower rate-code variance -> faithful rendering of head_w@h as the net-current margin) +
  (2) a per-pool BASE-RATE tonic baseline conductance encoding head_b (the LM's per-word base-rate/frequency prior),
  added to the net-current margin at s = hb_k * std_over_pools(margin), hb_k=0.5 calibrated once on seed 42. The winner
  is argmax over the substrate net signed synaptic-current margin (df_e*g_e + df_i*g_i + s*head_b off
  cp_conductance_g_e/g_i), 0 host categorical draws. Runner-only, default-off, NO sim/ edit.
artifacts:
  - research/runners/_wkv_mouth_read_parity_close_derisk.py
  - research/findings/raw/_wkv_read_parity_close_6seed.json
  - research/findings/raw/_probe_read_parity_bias_structure.py
---

# gap#1 / A1 — closing the fluid-mouth graded read's last ~8% (recov 0.921 -> 0.978): it was NOT the accumulator (GO, 6/6)

## The residual, and what it actually was

The graded-conductance read (`2026-08-13-fluid-mouth-graded-conductance-read-GO`) recovers 0.921 of the perfect-argmax
mass by reading the winner word-pool from the CONTINUOUS net signed synaptic-current margin. Its named residual #1
proposed two next rungs to close the last ~8%: (a) a facilitating LIP ramp-to-threshold accumulator (Mongillo-Wang
short-term facilitation, an intrinsic ramp that "sharpens near-ties over the read window"), or (b) a learned end-to-end
sign-preserving read. **We measured which residual dominates BEFORE building, and neither named rung is the lever.**

**Rung (a) presumes the misses are VARIANCE-limited** (finite-window near-tie noise a longer ramp averages out). A
READ-WINDOW SWEEP at seed 42 (150 / 450 / 900 steps) is **FLAT**: recov_argmax 0.879 / 0.864 / 0.883, argmax_agree
~0.62 unchanged. Integrating the conductance longer does nothing -> the residual is **NOT variance-limited**. A
facilitating accumulator re-integrates the SAME margin and is argmax-PRESERVING on it, so it structurally cannot move a
SYSTEMATIC miss. **Rung (b) as a LINEAR read fails too**: per-position corr(margin, head_w@h) ~ 0 (the margin is a
high-variance renderer whose ARGMAX is informative but whose vector is not linearly aligned) and a least-squares
[g_e, g_i] -> logit re-fit is numerically degenerate (no generalisable low-dim linear correction; the per-pool absolute
level is load-bearing — z-scoring it destroys the read, argmax 0.82 -> 0.12). (`_probe_read_parity_bias_structure.py`.)

## The residual decomposes into two SYSTEMATIC biases — each closed by a substrate mechanism

**(1) FEATURE-CODE FIDELITY.** The hidden feature `h = r_h*(Wo_sp@state)` is rate-coded by the `hid`/`hidinh`
populations at **hid_pop = 1 neuron PER feature-dim** — a minimal population that renders `head_w@h` with a ~18% argmax
loss (argmax(margin) vs argmax(head_w@h) = 0.82). This is the "what did we replace with a constant?" answer: the hidden
population was under-provisioned to a single unit per dimension. Raising **hid_pop 1 -> 4** (a DENSER population rate
code) lifts reconstruction 0.82 -> 0.875 and recov_argmax 0.879 -> 0.942, and PLATEAUS by hid_pop=8 (0.945) — the
canonical population-coding law (more pooled neurons -> lower rate-code variance).

**(2) THE BASE-RATE PRIOR.** The true logit is `head_w@h + HEAD_B`; the graded read reconstructs `head_w@h` and
**OMITS head_b** (the parent's `head_b_gain=0`). Omitting head_b **caps** argmax-agreement at 0.856 (measured ceiling:
argmax(head_w@h) vs the true argmax). `head_b` is the per-word base-rate/frequency prior. We inject it as a **per-pool
TONIC BASELINE CONDUCTANCE** (an intrinsic pool excitability proportional to head_b — frequent words rest more
excitable), scaled to the pool net-current operating point `s = hb_k * std_over_pools(margin)`, `hb_k=0.5` **calibrated
ONCE on seed 42** (a WIDE plateau: recov 0.985-0.989 across hb_k 0.35-0.5, falling by 0.75 — not a knife-edge). This
lifts recov_argmax 0.942 -> ~0.98. (A first attempt scaled head_b by the wrong reference and made it dominate — recov
collapsed to 0.62, argmax(head_b)=always-the-frequent-word — a documented near-miss the calibration fixes.)

Neither component is a host softmax/argmax refinement: hid_pop is a substrate population SIZE; head_b is a fixed
per-pool baseline CONDUCTANCE the pools carry. The winner stays argmax over the substrate net-current margin, 0 host
categorical draws.

## RESULT — 6-seed A/B (42/43/44/100/101/102; V=1000; P=4; n_eval=200; GPU; 813s)

<!--derived: research/findings/raw/_wkv_read_parity_close_6seed.json summary (mean over 6 seeds)-->

| arm | hid_pop | head_b | recov_argmax (mean / min) | argmax_agree | silent | signed LB | GO |
|---|---|---|---|---|---|---|---|
| baseline (parent)     | 1 | off | 0.9210 / 0.9069 | 0.711 | 0.0 | 6/6 | 0/6 |
| +code (feature-code)  | 4 | off | 0.9424 / 0.9330 | 0.757 | 0.0 | 6/6 | 2/6 |
| +baserate (head_b)    | 1 | on  | 0.9611 / 0.9532 | 0.808 | 0.0 | 6/6 | 6/6 |
| **parity_close (both)** | **4** | **on** | **0.9775 / 0.9643** | **0.850** | **0.0** | **6/6** | **6/6** |

Per-seed parity_close recov_argmax: 42=0.989, 43=0.968, 44=0.982, 100=0.979, 101=0.964, 102=0.983. The baseline arm
REPRODUCES the parent graded read exactly (0.921 mean; per-seed 0.9069/0.9172/0.9157/0.9125 match the parent's table),
confirming the A/B is clean. The two components are roughly ADDITIVE and INDEPENDENT: feature-code alone lifts +0.021,
base-rate alone +0.040, and TOGETHER they reach recov_argmax **0.9775 (min 0.9643)** — materially above 0.921 (it
closes ~72% of the 0.079 gap to a perfect argmax), 6/6 GO. The inhibitory shadow (NEGATIVE weights) stays LOAD-BEARING
6/6 on every arm (signed argmax-agree > positive-only; the base-rate term is added to BOTH the signed and the
positive-only margin, so this comparison still isolates the SIGN).

## Anti-cheats (all 6 seeds, parity_close arm)

- **Scramble -> chance:** the post-hoc pool->word relabel collapses argmax-agreement to 0 on every seed.
- **Zero-feature collapse (cache-immune):** silencing the signed-projection INPUT (zero feature) drops argmax-agreement
  to 0 — the feature drives the read; the base-rate term alone does not carry it.
- **Provenance:** winner from cp_conductance_g_e/g_i, host_rng_draws_on_read_path = 0 on every seed.
- **Signed load-bearing 6/6:** the inhibitory shadow picks the right word MORE often than the excitatory drive alone on
  identical conductances.

## External grounding

According to PubMed: **Zohary, Shadlen & Newsome (1994)**, Nature 370(6485):140-3
([DOI](https://doi.org/10.1038/370140a0)) — "Pooling responses across neurons should average out noise in the activity
of single cells, leading to substantially improved psychophysical performance." This grounds mechanism (1): a denser
hidden population per feature-dim reduces rate-code variance and closes the reconstruction gap. **Mulder, Wagenmakers,
Ratcliff, Boekel & Forstmann (2012)**, J Neurosci 32(7):2335-43
([DOI](https://doi.org/10.1523/JNEUROSCI.4156-11.2012)) — prior probability biases perceptual choice "primarily due to a
change in the starting point of the accumulation process," a common frontoparietal substrate. This grounds mechanism
(2): the base-rate prior (head_b) implemented as a per-pool baseline / starting-point offset, not a change of the
evidence weights.

## Honest residuals (why this is a read-fidelity GO, not "the mouth works" / not "closed")

1. **recov_argmax ~0.98, not exactly 1.0.** The head_b-omission ceiling on argmax-AGREEMENT is 0.856, and the
   reconstruction of head_w@h caps at ~0.875; the parity_close arm exceeds both on the MASS metric (the residual misses
   are near-ties carrying little mass) but the exact argmax is not fully recovered. The genuine next rung for the last
   ~2% is a higher-fidelity feature code still (hid_pop>4 with a larger spike budget) or a learned NONLINEAR read that
   inverts the substrate's feature->conductance transfer (the linear one is degenerate here).
2. **The base-rate term is added in HOST ARITHMETIC, not yet charged through a spiking synapse.** The feature-code
   lift (hid_pop 1->4) is FULLY on-substrate (just a denser population; the `code` arm reaches 0.9424 with only the
   parent's on-substrate conductance margin). The head_b term, by contrast, is added as a per-pool CONSTANT to the
   conductance-derived margin in the runner (`margin + s*head_b`) — equivalent to a per-pool tonic baseline
   CONDUCTANCE, but NOT driven by an actual bias-input population firing onto the pools. Wiring head_b as a tonic
   bias-input population (a small pool firing at a head_b-proportional rate onto each word-pool's `g_e`/`g_i`, so the
   base-rate prior is a real synaptic current) is the named next step to make the head_b half fully on-substrate.
3. **NOT "fully spiking" / NOT wired into production / NOT "retires the mouth".** The hidden `h = r_h*(Wo_sp@state)` is a
   host residual; the WKV store is BPTT-trained; the read-out weights are host-designed (labelled-line); head_b is read
   from the trained checkpoint. This rung moves the near-tie resolution onto two substrate mechanisms AT THE RUNNER
   LEVEL (default-off); it does not retire the upstream host state and is not integrated into the production endpoint.
4. **hid_pop is partly a hyperparameter.** Part of the lift is that the parent under-provisioned the hidden population
   to 1 neuron/dim; recognising this (per the 2026-08-11 "gap4 wall was a hyperparameter" lesson) is itself a finding.
   The base-rate mechanism is the genuinely new one; hid_pop is the honest "the code was too sparse" correction.

## Files
- Runner: `research/runners/_wkv_mouth_read_parity_close_derisk.py`
- Raw: `research/findings/raw/_wkv_read_parity_close_6seed.json`
- Diagnostic probe (window sweep / bias structure / calibration): `research/findings/raw/_probe_read_parity_bias_structure.py`
- Builds on: `2026-08-13-fluid-mouth-graded-conductance-read-GO.md` (the 0.921 read this closes),
  `2026-08-13-fluid-mouth-signed-read-parity-BOUNDARY.md` (the sparse-count wall the graded read broke).
