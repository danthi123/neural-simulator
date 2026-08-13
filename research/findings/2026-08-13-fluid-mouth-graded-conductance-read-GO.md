---
type: finding
status: qualified
claim_check: synthesis
verdict: GO (6-seed, 6/6). Reading the mouth's next-word from the GRADED / CONDUCTANCE domain — the CONTINUOUS net signed synaptic-current margin the pools already compute (off cp_conductance_g_e/g_i), NOT a sparse 1-2-SPIKE COUNT — BREAKS the signed-read parity wall. read_fidelity 1.1832 mean / 1.1552 min (6/6 ABOVE the ideal sampler) vs the SAME-seed spike-count signed read's 0.5515 (a 2.15x lift); it recovers 0.921 mean / 0.907 min of the PERFECT-ARGMAX mass (parity in the graded-native sense, >=0.85 on all 6) and 0.953 mean of the SPIKING contrastive-oracle ceiling (vs the parent's 0.427); ~19% SILENCE is CLOSED (silent 0.0 on all 6); and the inhibitory-shadow (NEGATIVE weights) are LOAD-BEARING 6/6 on the sensitive identical-conductance argmax instrument (signed argmax_agree 0.711 vs positive-only 0.567) — generalising to the 5 UNSEEN seeds (calibrated once on 42), NOT the 2026-07-04 decorative/overfit trap. Anti-cheats 6/6: scramble->chance, zero-feature-input->chance (cache-immune), 0 host draws. Free-gen coherent (self-NLL 0.86-1.24 vs the parent's 3.8-6.8). NOT "fully spiking" / NOT wired: the hidden h=r_h*(Wo_sp@state) is a host residual, the WKV store is BPTT-trained, the read-out weights are host-designed. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — the state->logits read-out)
date: 2026-08-13
mechanism: GRADED / CONDUCTANCE-DOMAIN signed read-out. The parent's signed wiring is reused verbatim (Wp EXCITATORY hid->pools accumulating cp_conductance_g_e; Wn INHIBITORY hidinh->pools accumulating cp_conductance_g_i). The winner word-pool is read NOT from a sparse 1-2-SPIKE COUNT but from the CONTINUOUS net signed synaptic-current margin the substrate itself computes: margin_k = (E_e - v_ref)*g_e[pool_k] + (E_i - v_ref)*g_i[pool_k] (v_ref=rest, off the public cp_conductance_g_e/g_i arrays — the same combination fused_conductance_decay_and_current does at bridge.py:8375), integrated over the read window. Pools kept SUBTHRESHOLD (floor 0) so g_i reflects only the inhibitory shadow and v stays near rest where the shadow is SUBTRACTIVE (Holt & Koch 1997). The inhibitory:excitatory SYNAPTIC strength ratio is calibrated ONCE (seed 42; a WIDE plateau ratio 0.15-0.7, not a knife-edge) to balance the two current terms so margin ~ (Wp-Wn)@feat = head_w @ h; NOT the parent's spike-regime ratio=6.5 which DOUBLE-compensates the driving force here and makes g_i over-dominant. Winner = argmax over the graded substrate margin (0 host draws). Runner-only, default-off, NO sim/ edit.
artifacts:
  - research/runners/_wkv_graded_conductance_read_derisk.py
  - research/findings/raw/_wkv_graded_conductance_6seed.json
  - research/findings/raw/_wkv_graded_calib_probe.py
---

# gap#1 / A1 — reading the mouth's next-word in the GRADED CONDUCTANCE domain BREAKS the sparse-spike-count parity wall (GO, 6/6)

## The wall, and the named lever

The signed-read parity BOUNDARY (`2026-08-13-fluid-mouth-signed-read-parity-BOUNDARY`) mapped the root cause exactly.
The TRUE SIGNED read-out lifted read_fidelity 0.035 -> 0.55, but PARITY stayed blocked at projection_recovery 0.43, and
its named companion processes (a neural divisive-norm pool + recurrent-WTA sharpening) moved NOTHING (0.4265 vs 0.4269)
because the sign is load-bearing ONLY in a SPARSE ~1.5-SPIKE near-threshold regime — too sparse for a feedback pool to
sense or an attractor to ignite, too noisy for rank-order. Its verdict: "the wall is the sparse spike-COUNT margin
itself." Its #1 named next lever (cost order): read in the GRADED / conductance domain — the continuous g_e/g_i the
pools already compute — so the winner-vs-loser margin is CONTINUOUS and the sparse-count noise floor no longer dominates.
This runner is that lever, and it clears the wall.

## The mechanism — read the substrate's OWN net signed synaptic current, not a spike count

The parent already accumulates the signed projection as conductances on the substrate: Wp on EXCITATORY hid->pool
synapses charges `cp_conductance_g_e`, Wn on INHIBITORY hidinh->pool synapses charges `cp_conductance_g_i`. Every step
the bridge fuses them into the neuron's own signed synaptic current `I_syn = g_e*(E_e-v) + g_i*(E_i-v)`
(`fused_conductance_decay_and_current`, bridge.py:8375) — a CONTINUOUS, graded quantity. This runner keeps the pools
SUBTHRESHOLD (floor 0) and reads the winner from the net signed current DRIVE at rest, integrated over the window:
`margin_k = (E_e - v_ref)*g_e[pool_k] + (E_i - v_ref)*g_i[pool_k]` (v_ref = rest ~ -50 mV; df_e=+50, df_i=-25). The
~5-10 ms conductance taus average out the OU noise a 1.5-spike count cannot; every pool has a defined continuous margin
(so there is no near-rheobase SILENCE). The winner is argmax over that graded substrate margin — the graded analog
read-out a distributed code affords (Mikulasch-Priesemann; the 2026-06-20 graded-dendritic-plateau GO is the template),
a genuine conductance read (NOT a host softmax, NOT a host argmax over host logits): 0 host categorical draws on the read
path. Reuse-by-import of the parent SignedShadowLogitRead (wiring/oracle/hidden-feature); cfg.seed-controlled substrate;
NO `sim/` edit; runner-only, default-off. The ONLY changes vs the parent: a SUBTHRESHOLD floor, a conductance reset
between reads (the parent's _reset left g_e/g_i carrying over), and the calibrated inhibitory:excitatory ratio.

## RESULT — 6-seed (42/43/44/100/101/102; V=1000; P=4; GPU; 290s)

<!--derived-->

| seed | read_fidelity | recov_argmax_mass | proj_recov (vs oracle) | argmax_agree | agree (pos-only) | signed>pos | silent | zerofeat_agree | GO |
|---|---|---|---|---|---|---|---|---|---|
| 42  | 1.1552 | 0.9069 | 0.9524 | 0.680 | 0.565 | YES | 0.0 | 0.0 | ✓ |
| 43  | 1.1715 | 0.9400 | 0.9731 | 0.745 | 0.575 | YES | 0.0 | 0.0 | ✓ |
| 44  | 1.1842 | 0.9335 | 0.9827 | 0.730 | 0.605 | YES | 0.0 | 0.0 | ✓ |
| 100 | 1.1698 | 0.9172 | 0.9328 | 0.710 | 0.540 | YES | 0.0 | 0.0 | ✓ |
| 101 | 1.1928 | 0.9157 | 1.0202 | 0.710 | 0.585 | YES | 0.0 | 0.0 | ✓ |
| 102 | 1.2259 | 0.9125 | 0.8582 | 0.690 | 0.535 | YES | 0.0 | 0.0 | ✓ |
| **mean** | **1.1832** (min 1.1552) | **0.921** (min 0.907) | **0.953** | **0.711** | 0.567 | **6/6** | **0.0** | **0.0** | **6/6** |

The SAME spike-count signed read on the SAME seeds (parent, GPU): read_fidelity 0.5515, projection_recovery 0.427,
silent 0.187, signed-load-bearing 3/6. This runner: read_fidelity 1.1832 (2.15x, and ABOVE the ideal sampler on all 6),
recovers 0.921 of the perfect-argmax mass (parity in the graded-native sense) / 0.953 of the spiking oracle ceiling,
silent 0.0 across all 6, signed load-bearing 6/6. Free generation stays coherent (self-NLL 0.86-1.24 on the coherent
prompts, e.g. seed 102 *"once upon a time there was a little boy named tim found a big box in the yard outside ..."*).

## Why this is NOT the 2026-07-04 conductance-signed trap (retracted twice)

That arc failed on two counts: (a) the SIGNED machinery was DECORATIVE (the positive Wp rows carried the read), and
(b) it OVERFIT to 3 tuned seeds (0-6/18 on the unseen 100/101/102). Both guarded here.
- **Single fixed operating point on ALL 6 seeds.** The inhibitory:excitatory ratio (0.3) is calibrated ONCE on seed 42
  (`_wkv_graded_calib_probe.py`) over a WIDE plateau (read_fid flat across ratio 0.15-0.7 — not a knife-edge; margin<->
  logit corr peaks 0.985), then FIXED; seeds 43/44/100/101/102 are the UNSEEN generalisation test and pass 6/6.
- **The negative weights are LOAD-BEARING on the sensitive instrument.** Because the graded read removes the
  near-rheobase silence confound, the signed-vs-positive comparison is clean and re-sim-free: from IDENTICAL per-position
  conductances, the signed margin (df_e*g_e + df_i*g_i) picks the RIGHT word MORE often than the excitatory drive alone
  (df_e*g_e) — argmax_agree 0.711 vs 0.567, 6/6. (The MASS metric SATURATES near the argmax ceiling and cannot see this:
  read_fid ~= positive-only in mass; argmax-agreement on identical conductances is the discriminating read.)

## External grounding

According to PubMed, Holt & Koch (1997), Neural Computation 9(5):1001-13
([DOI](https://doi.org/10.1162/neco.1997.9.5.1001)): shunting inhibition is DIVISIVE on subthreshold EPSPs but
SUBTRACTIVE on firing rate, because the spiking mechanism clamps the somatic membrane above rest so the shunt current
becomes rate-independent. This is the mechanistic reason the graded SUBTHRESHOLD read at rest recovers the sign as a
clean continuous margin where the near-rheobase spike-COUNT read cannot: at rest |E_i - v| is set and the net current is
a faithful LINEAR signed combination; once the pool spikes, v is clamped above rest and the sign turns noisy on a
1.5-spike count. It also grounds the low-floor operating point (the 2026-07-04 high-floor -> divisive/shunting lesson,
now with the mechanism named).

## Anti-cheats (all 6 seeds)

- **Zero-feature collapse (cache-immune):** silencing the signed-projection INPUT (drive hid/hidinh with a zero feature)
  drops argmax-agreement to 0.0 (vs the intact 0.711) — the feature drives the read; it is not a floor/frequency
  artifact. (This replaces the weight-lesion control, which is unreliable in this wiring: zeroing the read-out weights
  does not fully remove the conductance the pools already hold — the same class of contamination the PARENT documented
  on its own readout-lesion, 3/6. Verified by a direct probe: after the weight-lesion the pool g_e/g_i persist at full
  magnitude even though the CSR weights read 0.)
- **Scramble -> chance:** the post-hoc pool->word relabel collapses argmax-agreement to 0.0 on every seed — the
  labelled-line pool->word map carries the discrimination.
- **Provenance:** winner from cp_conductance_g_e/g_i, host_rng_draws_on_read_path = 0 on every seed.
- **argmax_agree 0.711 (~711x the 1/V chance)** together with the scramble + zero-feature collapse establishes the read
  is genuinely signed-projection driven.

## Honest residuals (why this is a read-fidelity GO, not "the mouth works" / not "fully spiking")

1. **The read is LINEAR** (no exponential/contrastive sharpening), so recov_argmax 0.921, not 1.0: on ~8% of positions
   the graded margin's argmax differs from the host argmax (rate-limited reconstruction; the calib probe reads
   margin<->logit corr 0.985). Named next rung: a facilitating LIP-style ramp accumulator, or a LEARNED read-out /
   sign-preserving normalisation end-to-end, to close the last few %.
2. **NOT "fully spiking" / NOT "retires the mouth" / NOT wired.** The hidden `h = r_h*(Wo_sp@state)` is a host residual;
   the WKV store is BPTT-trained; the read-out weights are host-DESIGNED (labelled-line). This rung moves the DOMINANT
   `head_w @ h` matmul + top-K argpartition onto a graded synaptic conductance read AT THE RUNNER LEVEL (default-off);
   it does not retire the upstream host state, and it is not integrated into the production endpoint.
3. **The winner is selected in the graded (subthreshold) domain**, read at a fixed v_ref=rest (the "input current at
   rest") off the substrate conductances — a genuine graded analog read, but the winning pool is not driven to a
   suprathreshold spike here; converting the graded winner to an emitted spike (a threshold/plateau or feeding the
   margin into the FS-WTA) is a small downstream step, not done in this rung.

## Files
- Runner: `research/runners/_wkv_graded_conductance_read_derisk.py`
- Raw: `research/findings/raw/_wkv_graded_conductance_6seed.json`; the one-time seed-42 ratio calibration
  `research/findings/raw/_wkv_graded_calib_probe.py`
- Builds on: `2026-08-13-fluid-mouth-signed-read-parity-BOUNDARY.md` (the wall this breaks),
  `2026-08-13-fluid-mouth-signed-shadow-readout-BOUNDARY-LIFTED.md` (the signed wiring reused),
  `2026-06-20-dendrite-derisk-A-graded-plateau-readout.md` (the graded analog read template),
  `2026-07-04-conductance-domain-signed-readout-SURPASS.md` (the conductance-signed seed-fragility record, avoided here).
