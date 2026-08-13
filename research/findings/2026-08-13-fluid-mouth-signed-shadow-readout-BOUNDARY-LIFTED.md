---
type: finding
status: qualified
verdict: BOUNDARY-LIFTED (6-seed, NOT a GO). The TRUE SIGNED read-out (negative head_w on an INHIBITORY SHADOW of the hidden layer, no Dale-shift) ROBUSTLY removes the rung-2 signed-projection common-mode wall — read_fidelity 0.44-0.69 (mean 0.55, 6/6 seeds) vs the rung-2 Dale-shift's 0.035, a ~16x lift — and the substrate generates SEMI-COHERENT open prose on the read path (free-gen self-NLL 3.8-6.8 on the coherent prompts vs ~10 gibberish; e.g. "tom and his dog were very happy to meet their diamond but he didn't know what to do"). The projection RANKS by logit (net-vs-logit rate corr 0.987, exact-argmax match 0.81; spiking per-pool input-current vs logit corr 0.89-0.92, the true argmax at/near the top pool rank 0-1); scramble->chance and 0-host-draw provenance hold on all 6 seeds. NOT a parity GO: projection_recovery vs the perfect-current ORACLE is 0.43 (oracle 1.30), ~19% of positions are SILENT at the FIXED operating point, and whether the NEGATIVE weights are specifically load-bearing (signed > positive-only) is SEED-FRAGILE (3/6) and confounded by the operating-point silence — echoing the 2026-07-04 conductance-signed seed fragility. NOT "fully spiking" / NOT "retires the mouth": the hidden h=r_h*(Wo_sp@state) is a host residual + the WKV store is BPTT-trained. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — retiring the state->logits matmul on the read path)
date: 2026-08-13
mechanism: TRUE SIGNED synaptic read-out (Dale's principle) — Wp=max(Wfull,0) on EXCITATORY hid synapses + Wn=max(-Wfull,0) on an INHIBITORY SHADOW hidinh (a rate-matched copy of hid, same feature drive, uniformized thresholds), Wfull=concat(head_w,-head_w); net pool drive Wp@rate(hid) - ratio*Wn@rate(hidinh) ~ head_w@h, NO global Dale-shift -> NO common mode. A read floor JUST BELOW rheobase puts pools at threshold so the winner (the max-synaptic-current pool) tips over while negative-current losers stay silent; the production FS-WTA resolves the winner. ratio compensates the conductance driving-force asymmetry (low floor keeps g_i SUBTRACTIVE — the 2026-07-04 lesson)
artifacts:
  - research/runners/_wkv_signed_shadow_read_derisk.py
  - research/findings/raw/_wkv_signed_shadow_6seed.json
  - research/findings/raw/_wkv_signed_shadow_smoke.json
  - research/findings/raw/_wkv_signed_shadow_f84.json
  - research/findings/raw/_wkv_signed_shadow_projection_probe.json
---

# gap#1 / A1 — the TRUE SIGNED read-out (inhibitory shadow) LIFTS the Dale common-mode wall that made the state->logits synaptic read gibberish (rung-2 read_fid 0.035 -> 0.55, 6-seed), and the substrate speaks semi-coherent prose on the read path

## The lever, and why it is the single most tractable next rung (RAG-grounded)

`before_you_build` + the A1 mouth corpus (`rag_search "open-ended generation spiking mouth Broca ... matmul retirement"`)
locate the boundary exactly. The mouth arc had reached: the fluent open-prose WKV/SSM generation (`2026-07-20`
RF-PHASE, 6-seed GO) reads its next word by a HOST argmax over a graded conductance state;
`2026-08-13-gap1-A1-fewspike-...` put that read onto the PRODUCTION few-spike Izhikevich POPULATION read (6/6 GO,
P>=8); `2026-08-13-gap1-A1-fswta-...` added the FS-WTA (rung 1, 6/6 GO) BUT routing the `head_w @ h` logit projection
through read-out NEURONS (rung 2) was a **0/6 BOUNDARY**: realising `head_w` as Dale-shifted EXCITATORY synapses
(`head_w - gmin >= 0`) injects a COMMON MODE `gmin * sum(hidden spikes)` orders of magnitude larger than the ~3.5%
top1-top2 logit margin (measured here); a scalar canceller cannot subtract it -> read_fidelity **0.035**, gibberish.
That finding's ORACLE (a perfect host-logit current through the SAME FS-WTA) reached read_fid 0.57 (P=1) .. 0.93
(P=16), proving the FS-WTA RESOLUTION is not the wall — the **SIGNED SYNAPTIC PROJECTION fidelity** is. Its
explicitly-named next lever (item 1) is a TRUE signed read-out via an inhibitory shadow (Dale's principle). This is the
highest tractability x leverage rung: it attacks the mapped primary wall directly, the oracle already bounds the
reachable target, and the rung-2 scaffold is reusable. It also directly re-opens the 2026-07-04 conductance-signed
cautionary record (there the signed machinery was DECORATIVE + overfit at G^2=25 / 18 slots) — this finding tests
whether, over V=1000 with a ~3.5% margin, the sign is load-bearing (answer: partially, and seed-fragile — see below).

## The mechanism (Dale's principle — the biology the Dale-shift replaced)

A signed weight is carried by TWO populations, one excitatory, one inhibitory (Dale). Split
`Wfull = concat(head_w, -head_w)` [V, 2D] (so `Wfull @ feat = head_w @ h`, feat = [h+, h-] the dual-nonneg hidden
state) into `Wp = max(Wfull, 0)` and `Wn = max(-Wfull, 0)` (both >= 0). Rate-code feat by TWO matched populations
driven by the SAME feature current: an EXCITATORY `hid` and an INHIBITORY SHADOW `hidinh`. Wire `Wp` as EXCITATORY
synapses `hid -> pools` (g_e) and `Wn` as INHIBITORY synapses `hidinh -> pools` (g_i). Net pool drive
`Wp@rate(hid) - ratio*Wn@rate(hidinh) ~ (Wp - Wn)@feat = head_w @ h` — NO Dale-shift, NO common mode. The firing
thresholds are uniformized so hid and hidinh rate-MATCH (removing the per-neuron heterogeneity that would BIAS feat_i
vs feat_e — measured shadow rate-match corr 0.95-0.97 across seeds); independent OU keeps the winner stochastic. The
winner emerges from the production FS-WTA over all V pools. Reuse-by-import of the rung-1/2 scaffold + WKVReadout;
cfg.seed-controlled substrate; NO `sim/` edit; runner-only, default-off.

**The operating point is load-bearing (measured, not assumed).** The signed projection ranks by logit, but the
winner's synaptic current sits only ~+4 pA (the logit distribution is centred negative — most words are unlikely). A
LOW read floor JUST BELOW rheobase (~78-80 pA) puts every pool near threshold so the winner's few-pA excess tips it
over while negative-current losers stay silent — and critically, v stays near REST, where the ratio-compensated
inhibitory shadow is SUBTRACTIVE. Pushing the floor UP to erase silence (floor 84, ratio-matched 3.5) lifts read_fid
to 0.71 with 0 silence BUT the inhibition turns DIVISIVE/shunting and the negative weights become DECORATIVE
(positive-only 0.78 > signed 0.71 at that point) — reproducing the 2026-07-04 outcome. So the regime in which the
SIGNED mechanism is genuinely load-bearing is the low-floor subtractive one; that is the operating point reported here.

## RESULT — 6-seed (42/43/44/100/101/102; V=1000; P=8; 120 held-out positions; GPU; 874s)

<!--derived-->

| seed | read_fidelity | positive-only | signed>1.1x pos? | argmax_agree (chance 1e-3) | proj_recovery (vs oracle) | silent |
|---|---|---|---|---|---|---|
| 42  | 0.603 | 0.384 | YES (1.57x) | 0.317 | 0.489 | — |
| 43  | 0.688 | 0.641 | no (1.07x)  | 0.417 | 0.549 | — |
| 44  | 0.540 | 0.332 | YES (1.63x) | 0.267 | 0.386 | — |
| 100 | 0.516 | 0.593 | REVERSED    | 0.250 | 0.426 | — |
| 101 | 0.439 | 0.454 | REVERSED    | 0.267 | 0.346 | — |
| 102 | 0.523 | 0.430 | YES (1.22x) | 0.217 | 0.367 | — |
| **mean** | **0.5515** (min 0.439) | 0.472 | **3/6** | **0.289** (~29x chance) | **0.427** (oracle 1.30) | **0.187** |

**The robust headline: the signed shadow REMOVES the common-mode wall.** read_fidelity is 0.44-0.69 on every seed
(mean 0.55) versus the rung-2 Dale-shift's 0.035 — a ~16x lift — and the substrate GENERATES semi-coherent open prose
on the read path (each next word read from the signed synaptic current + FS-WTA, NO host logit matmul, NO top-K on the
read path). Free-generation (seed 42): *"tom and his dog were very happy to meet their diamond but he didn't know what
to do"* (self-NLL 3.75), *"once upon a time there was a little girl named lucy wanted to ..."* (6.82) — coherent
clauses that then degrade into `<unk>` loops (14.07 on the third prompt). This is far above the P=1/Dale-shift
gibberish (self-NLL ~10-12) but below the parent's clean fluent host-argmax read.

**The projection ranks faithfully (probe artifact, seed 42, `_wkv_signed_shadow_projection_probe.json`):** the
rate-limit reconstruction `Wp@rate(hid) - Wn@rate(hidinh)` correlates with the true logit at corr 0.987 and recovers
the exact argmax 0.81 of the time; the actual SPIKING per-pool input current correlates 0.89-0.92 with the logit and
places the true argmax at the top pool (rank 0-1); the exc/inh shadow rate-matches at corr 0.96. So the signed
projection RANKS by logit — the Dale common-mode wall is genuinely removed; the residual is purely the read regime.

**Anti-cheats (all 6 seeds):** scramble -> chance (post-hoc pool->word relabel collapses argmax agreement to ~0 on
every seed — the labelled-line readout carries the discrimination); provenance -> 0 host categorical draws on the read
path (winner from `cp_firing_states`); shadow rate-match corr 0.95-0.97. The intact argmax_agree (0.22-0.42, ~22-42x
the 1/V chance) together with the scramble collapse establishes the read is genuinely SIGNED-PROJECTION-driven.

## Honest residuals (the reasons this is a LIFT, not a GO — TERMS.md)

1. **projection_recovery 0.43 (< the oracle ceiling 1.30).** The oracle drives pools with a CONTRASTIVE softmax-
   normalized (exponential) current; the signed synaptic read delivers a LINEAR logit current whose ~3.5% margin the
   FS-WTA resolves only partially at a fixed budget. The exponential sharpening is a missing companion process.
2. **~19% of positions are SILENT** (winner current below rheobase at the FIXED floor). The winning pool's current is
   always the maximum, but its ABSOLUTE value drifts per position; a fixed floor cannot sit at rheobase for all.
3. **The NEGATIVE weights are LOAD-BEARING only seed-fragilely (3/6; mean signed 0.55 vs positive-only 0.47).** On
   seeds 100/101 the positive-only read (Wn lesioned) equals or beats signed — because at the near-rheobase point the
   inhibition also suppresses firing, so the silence-vs-ranking trade-off CONFOUNDS the signed-vs-positive comparison.
   This is the same seed fragility the 2026-07-04 conductance-signed arc mapped; disentangling it needs residual (2)
   fixed first. The readout-lesion-collapse control likewise fails on 3/6 seeds — it is contaminated by vocab-
   frequency ordering (with the readout off, floor+OU-driven pool firing lands on frequent low-index words); the
   scramble + provenance controls are the load-bearing ones and pass 6/6.
4. **NOT "fully spiking" / NOT "retires the mouth".** The hidden `h = r_h*(Wo_sp@state)` is a host residual (the
   Wo_sp projection + the multiplicative r_h gate); the WKV store is BPTT-trained. This rung retires the DOMINANT
   `head_w @ h` matmul + the top-K argpartition onto signed read-out neurons; the upstream state is still host/Qwen-
   scaffolded. The labelled-line feature drive + read-out weights are host-DESIGNED (not self-organized). Default-off.

## The named next rung (NOT deferred — the companion process the fixed floor replaced with a constant)

A NEURAL HOMEOSTATIC operating point (the parent's rung-2 item 2): a divisive-normalisation inhibitory pool that
tracks total word-pool drive and adapts the set-point per position (the companion process the fixed floor replaced
with a constant), so every position's winner sits at rheobase (erasing residual 2), PLUS stronger recurrent WTA
amplification to supply the exponential sharpening the linear read lacks (closing residual 1 toward the oracle 0.93).
With the operating point fixed, the signed-vs-positive-only comparison (residual 3) becomes clean. That rung would
convert this LIFT into the parity GO on the fully-synaptic mouth read.

## Files
- Runner: `research/runners/_wkv_signed_shadow_read_derisk.py`
- Raw: `research/findings/raw/_wkv_signed_shadow_6seed.json` (+ `_wkv_signed_shadow_smoke.json`,
  `_wkv_signed_shadow_f84.json` — the high-floor decorative-regime control)
- Builds on: `2026-08-13-gap1-A1-fswta-lowers-spike-budget-3x-synaptic-logit-readout-boundary.md` (the rung-2
  signed-projection BOUNDARY this lifts), `2026-08-13-gap1-A1-fewspike-izhikevich-read-...` (the population few-spike
  read), `2026-07-04-conductance-domain-signed-readout-SURPASS.md` (the conductance-signed seed-fragility record),
  `2026-07-20-gap1-RF-PHASE-ENCODE-...` (the fluent WKV generation).
