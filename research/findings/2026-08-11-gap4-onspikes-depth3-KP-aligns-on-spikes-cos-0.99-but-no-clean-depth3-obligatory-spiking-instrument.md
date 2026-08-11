---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes — transport-free KP learned feedback at genuine DEPTH-3 on the LIF SNN, + the depth-3 FIT-instrument limit on the spiking substrate
lane: gap#4 / deep-credit
verdict: transport-free KP-learned feedback ALIGNS at genuine depth-3 on the ONE spiking substrate (cos(Y_deep, W^T) ~0 -> ~0.99 on 6/6 seeds) and beats the frozen-feedback (freeze-Y) lever where the target is depth-separating; BUT the STRICT depth-3-obligatory spiking FIT instrument is UNDEFINED (not GO/NO-GO) — depth-separation is seed-fragile (3/6) and DIRECT-DFA does NOT fail at depth-3 on the shallow narrow LIF net, so the "KP beats fixed-DFA where DFA fails" premise does not reproduce. An honest instrument-limit map, NOT a fabricated negative.
artifacts:
  - research/findings/raw/_gap4_onspikes_depth3_credit_fidelity.json
instrument: research/runners/_gap4_onspikes_depth3_credit_fidelity_derisk.py — a tent^k regression FIT on the BPTT-viable LIF SNN (sim/bptt_snn_gpu) with a rate-normalized linear population read-out, four matched-budget credit arms (surrogate-BPTT oracle / direct fixed-DFA / chained fixed feedback = freeze-Y / chained KP-learned feedback), per-seed ceiling-gating + N-testable, and the transport-free / freeze-Y / permuted / determinism anti-cheats. SIM_BACKEND=numpy, device=cpu.
---

# gap#4 deep-credit ON SPIKES at genuine DEPTH-3 — transport-free KP feedback ALIGNS on the LIF substrate (cos ~0 -> ~0.99, 6/6 seeds), but a CLEAN depth-3-obligatory spiking FIT instrument is not constructible at toy scale (UNDEFINED)

This ports the RATE-side depth-3 FIT instrument (`2026-08-11-gap4-layer3-credit-fidelity...` + `_gap4_learned_feedback_derisk.py`) onto the ONE spiking substrate, to answer the last gap#4 residual: the transport-free KP surpass is 6-seed GO at DEPTH-2 on spikes (`2026-08-11-gap4-onspikes-KP-learned-feedback-ALIGNS...`), but genuine DEPTH-3 credit on spikes was never tested because no depth-3-obligatory *spiking* instrument existed. The question: on a depth-3-engaging tent^3 FIT target run on the LIF SNN, does transport-free KP-learned feedback reach genuine depth-3 credit (approaching the surrogate-BPTT depth-3 oracle, beating depth-2 and fixed-DFA) where fixed-DFA fails — the depth-3 analog of the depth-2 on-spikes GO?

## The decisive POSITIVE — KP transport-free feedback ALIGNS at genuine depth-3 on spikes (6/6 seeds)

<!--derived-->
On the depth-3 LIF net (3 hidden LIF layers + a linear population read-out of the top layer's mean firing rate), the transport-free chained KP-learned feedback's alignment signature cos(Y_deep, W^T) — the deepest-from-output feedback matrix vs its matched forward weight-transpose, the credit that must traverse 3 hops — RISES from ~0 at init to ~0.99 at convergence on EVERY seed (from `research/findings/raw/_gap4_onspikes_depth3_credit_fidelity.json`):

| seed | 42 | 43 | 44 | 100 | 101 | 102 |
|---|---|---|---|---|---|---|
| cos(Y_deep, W^T) init | 0.07 | 0.13 | 0.00 | -0.17 | 0.10 | 0.27 |
| cos(Y_deep, W^T) final | 0.992 | 0.996 | 0.997 | 0.995 | 0.995 | 0.995 |

This is the depth-3 extension of the depth-2 on-spikes alignment GO: KP co-adapts a separate random feedback stream toward W^T by the matched local delta (Kolen-Pollack), transport-free, and the alignment reaches the 3rd hidden layer on the spiking substrate. The mechanism is NOT the wall.

## The FIT reach — partial + seed-variable; the freeze-Y lever is POSITIVE (learning the feedback helps)

<!--derived-->
On the 3 depth-separating seeds (see below), aggregate FIT quality (best-over-training MSE / target variance; gap-close = fraction of the BP2->BP3 loss gap closed, 1.0 = reaches the depth-3 oracle):

| arm | loss/var | gap-close |
|---|---|---|
| BPTT depth-3 (oracle) | 0.081 | 100% |
| BPTT depth-2 (control) | 0.187 | 0% |
| direct fixed-DFA | 0.083 | 100% |
| chained KP-learned | 0.178 | 7% |
| chained frozen (freeze-Y lever) | 0.283 | -96% |

The chained KP credit (0.178) clearly beats the FROZEN chained feedback (freeze-Y, 0.283) — the freeze-Y lever is positive, so LEARNING the feedback helps the chained credit (per-seed: 42 KP 76% vs freeze-Y -166%; 43 KP 16% vs -45%; 102 KP -72% vs -76%). But KP's aggregate gap-close is only ~7% (dragged by seed 102), i.e. the aligned chained credit reaches roughly the depth-2 floor, not the depth-3 oracle — the perfect ALIGNMENT does not translate into a competitive FIT on this shallow narrow LIF net, because the chained sequential surrogate credit attenuates through the LIF membrane surrogate at each hop.

## Why the STRICT instrument is UNDEFINED (not GO/NO-GO) — two independent, mapped obstructions

<!--derived-->
**(1) Depth-separation is SEED-FRAGILE and the ceiling is marginal.** Only 3/6 seeds have depth-2 UNDERFITTING tent^3 (bp2 clearly > bp3); on 3/6 (seeds 44, 100, 101) depth-2 fits tent^3 as well as depth-3 (sep ~0), so width substitutes for depth exactly as the rate finding's Telgarsky obstruction predicts — and the separable/fittable-window overlap is EVEN TIGHTER on the coarse spiking substrate. Even the depth-3 BPTT ceiling only reaches ~0.05-0.11*var at the narrow separating width (not ~0), so the ceiling is a genuine >=88%-variance-reduction fit but NOT loss~0. Per-seed depth-separation (bp2-bp3)/var: 42 +0.096, 43 +0.130, 44 -0.001, 100 -0.001, 101 -0.002, 102 +0.093.

<!--derived-->
**(2) Direct fixed-DFA does NOT fail at depth-3 on the shallow narrow LIF net.** On every separating seed, direct DFA closes 82-134% of the BP2->BP3 gap (0.083 loss/var, MATCHING the surrogate-BPTT oracle at 0.081), so the "fixed-DFA fails to reach the deep layers" premise from the RATE finding does NOT reproduce on spikes for a SCALAR-output target.
The fixed-feedback deep-layer limit (Nokland) is a MULTI-output / deep-Jacobian phenomenon, and a 1-D-output 3-hidden-layer LIF net evades it (the direct random projection of a scalar error is an adequate credit signal here). The runner's `fixed_dfa_baseline_fails_to_reach_layer3` precondition therefore FAILS -> the verdict is UNDEFINED (per the verdict-preconditions discipline: a failed precondition is UNDEFINED, never a fabricated negative), NOT a NO-GO.

<!--derived-->
**The ceiling value is immaterial to the verdict.** The runner was aggregated at ceil_frac 0.05 / 0.10 / 0.15 / 0.20: N_testable = 0 / 2 / 3 / 3 respectively, and the status is UNDEFINED at ALL of them — at 0.05 because no seed's marginal ceiling holds, and at 0.10-0.20 because direct-DFA-does-not-fail. The default ceil_frac=0.12 is LIF-substrate-calibrated (a >=88%-variance-reduction fit) and reports 3/6 testable so the real blocker (DFA-does-not-fail) is the surfaced reason rather than "no testable seeds".

## What a VALID spiking depth-3 instrument would need (naming the path, not deferring)

<!--derived-->
- **A regime where the separable window (depth-2 underfits) and the strong-ceiling window (depth-3 fits to ~0) overlap robustly across seeds.** The rate finding proved (Telgarsky 2016) this overlap is exponentially thin at a depth-gap of 1; it is thinner still on the coarse LIF substrate. This needs LARGER scale — a richer input encoding than a single rate-coded scalar (more input dimensions / temporal structure), and more neurons — so a depth-3 LIF net fits tent^3 to ~0 while a same-width depth-2 net still cannot.
- **A MULTI-DIMENSIONAL output and/or a DEEPER net (>=4 hidden layers)** so the direct-DFA deep-Jacobian failure actually manifests — the "fixed-DFA fails at depth" phenomenon (Nokland) is a multi-output / deep-Jacobian effect; a scalar-output shallow net lets direct DFA succeed, which is why the depth-3 KP-vs-DFA question is not posable here.

## Anti-cheats (all EXECUTED, not asserted)

<!--derived-->
transport-free: max |cos(Y_fb, W^T)| at init < 0.8 every seed (separate random stream, not a W^T copy); no Y_fb byte-equal any forward W or its transpose; the credit path computes e @ Y_fb, never a forward W^T; KP updates Y_fb by the matched Adam-step transpose (activity-derived), and cos(Y_fb, W^T) RISES through training (co-adapted, not copied). freeze-Y lever: chained KP moved Y_fb every seed; chained fixed left Y_fb frozen. permuted-target KP: no fit (perm loss ~0.5*var = the mean-predictor floor, every seed). determinism: two fresh builds at one seed give byte-identical forward weights on all seeds (the substrate RNG is the runner's default_rng(seed); this runner does not touch the bridge, so there is no cfg.seed to mis-set). backend=numpy device=cpu emitted.

## Bottom line

The transport-free KP deep-credit MECHANISM reaches genuine depth-3 on the ONE spiking substrate — cos(Y_deep, W^T) -> ~0.99 on 6/6 seeds, and learning the feedback beats freezing it. The blocker to a decisive depth-3 GO/NO-GO is the INSTRUMENT, not the mechanism: at toy LIF scale a clean depth-3-obligatory spiking FIT target is not constructible (seed-fragile depth-separation, a marginal ceiling, and a direct-DFA baseline that does not fail for a scalar-output shallow net).
This corroborates the rate-side finding that the depth-3 measurement is fundamentally hard at toy scale and names the two things a valid spiking depth-3 instrument needs (scale for the separable/fittable overlap; multi-output / deeper net for the DFA-fails premise). No sim/ edit — the LIF forward / BPTT / atan surrogate are reuse-by-import; the tent^k data, the linear population read-out, and the four credit rules are runner-side.
