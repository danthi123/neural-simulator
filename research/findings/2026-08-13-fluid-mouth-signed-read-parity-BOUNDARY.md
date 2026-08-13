---
type: finding
status: qualified
claim_check: synthesis
verdict: BOUNDARY (6-seed, NOT a parity GO). The signed read-out's named next rung — a NEURAL divisive-normalisation homeostatic set-point pool + recurrent-WTA exponential sharpening — does NOT reach parity, because BOTH mechanisms are INERT at the sign-preserving operating point. With the divisive-norm pool (dn) + recurrent within-pool excitation wired and active (6 seeds, floor=78), read_fidelity 0.543 / projection_recovery 0.4265 / silent 0.193 / signed-load-bearing 4/6 are STATISTICALLY IDENTICAL to the parent's fixed-floor baseline (0.5515 / 0.4269 / 0.187 / 3/6) — proj_recovery 0.4265 vs 0.4269. Controlled same-position diagnostics give the mechanistic why. (1) The dn pool NEVER fires from the sparse pool drive (dn_fire=0 across floor 78-90, dn_exc 0.6-8); given a tonic floor so it CAN fire, it fires constantly (~60 spk) and blanket-suppresses the read (winner spikes 4.6->0.2, host-argmax agreement 0.30->0.00). (2) Recurrent within-pool excitation leaves the winner margin unchanged (1.4->1.5 spikes across rec_gain 0->20) — the winner fires only ~4-8 spikes/150 steps, too few to bootstrap an attractor. (3) Rank-order / first-to-fire (the research-gate RANK-1 alternative) is WORSE than integrate-rate (mass 0.072 vs 0.120, agree 0.125 vs 0.200) — OU noise dominates the first-crossing at the ~1.5-spike margin. ROOT CAUSE: the signed read is load-bearing ONLY in a sparse near-threshold SUBTRACTIVE regime (winner ~4-8 spikes, margin ~1.5, ~15-20 active pools) that is TOO SPARSE for a feedback-inhibition pool to sense proportionally and for recurrent attractors to ignite, and TOO NOISY for rank-order — while the denser regime the companion processes need is exactly the high-floor regime the parent showed turns the sign DECORATIVE. The sign and the companion processes are in TENSION in this point-neuron / 150-step substrate. The silence residual IS a genuine operating-point problem (an idealised host operating-point search — the CEILING — recovers 14/15 silent positions with correct-rate winners, agree 0.333), but the NEURAL set-point that would do it is blocked by the same sparse-sensing limit. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose mouth — the state->logits read-out)
date: 2026-08-13
mechanism: (1) divisive-normalisation homeostatic pool `dn` (Louie-Glimcher two-stage `R_i <- V_i/(B+Σ_j R_j)`, Carandini-Heeger): all word-pools -> dn (excitatory sum), dn -> all pools (inhibitory, divisive). (2) recurrent within-pool EXCITATION pool_k[i]->pool_k[j] (Rutishauser soft/hard-WTA gain; Wong-Wang). Both wired onto the parent's TRUE SIGNED read-out (Wp exc hid, Wn inh shadow hidinh). Both INERT at the sparse operating point.
artifacts:
  - research/runners/_wkv_signed_read_parity_derisk.py
  - research/findings/raw/_wkv_signed_read_parity_6seed.json
  - research/findings/raw/_wkv_signed_read_parity_diag1.txt
  - research/findings/raw/_wkv_signed_read_parity_diag2.txt
  - research/findings/raw/_wkv_signed_read_parity_diag3.txt
  - research/findings/raw/_wkv_signed_read_parity_sweep.txt
  - research/findings/raw/_wkv_signed_read_parity_diag2.py
  - research/findings/raw/_wkv_signed_read_parity_diag3.py
---

# gap#1 / A1 — the signed read-out's named parity rung (neural homeostatic set-point + recurrent-WTA sharpening) is a BOUNDARY: BOTH companion processes are INERT at the sign-preserving sparse operating point (6-seed proj_recovery 0.4265 == parent 0.4269)

## The rung, and what it was meant to close

The parent (`2026-08-13-fluid-mouth-signed-shadow-readout-BOUNDARY-LIFTED`) put the state->logits `head_w @ h`
matmul + top-K onto spiking read-out neurons via a TRUE SIGNED read-out (Dale: `Wp` on an excitatory `hid`, `Wn` on
an inhibitory shadow `hidinh`, no Dale-shift, no common mode), lifting read_fidelity 0.035 -> 0.55 (~16x, 6-seed). It
was a LIFT not parity, with three residuals: projection_recovery 0.43 vs the perfect-current ORACLE 1.30 (the LINEAR
read lacks exponential sharpening), ~19% of positions SILENT at the fixed read floor, and the negative weights
load-bearing only seed-fragilely (3/6). Its explicitly-named next rung (attacked here): a NEURAL divisive-normalisation
homeostatic pool that adapts the read set-point per position (to erase silence + disentangle the sign confound) PLUS
recurrent-WTA amplification (to supply the exponential sharpening). RAG-grounded on the 2026-07-05 objrel research
gate's RANK-2 (Louie-Glimcher two-stage normalise-then-WTA, `R_i <- V_i/(B+Σ_j R_j)`; Carandini-Heeger 2012
divisive-normalisation; Rutishauser-Douglas-Slotine 2011 soft/hard-WTA recurrent gain).

## What was built (additive, reuse-by-import, NO sim/ edit)

`_wkv_signed_read_parity_derisk.py` subclasses the parent `SignedShadowLogitRead` and adds two fully-synaptic
companion processes: (1) a `dn` inhibitory region — all word-pools -> dn (excitatory sum, so dn tracks total pool
drive), dn -> all pools (inhibitory, divisive at the operating point); (2) recurrent within-pool excitation
`pool_k[i] -> pool_k[j]` (i != j). `dn_inh=0 & rec_gain=0` reproduces the parent. Both are lesionable. The parent's
FS-WTA, signed wiring, oracle, hidden feature, metrics + anti-cheats are reused verbatim. cfg.seed-controlled
substrate; default-off.

## RESULT 1 — the 6-seed A/B: the companion processes move NOTHING (`_wkv_signed_read_parity_6seed.json`)

6 seeds (42/43/44/100/101/102; V=1000; P=8; floor=78 — matched to the parent; dn_inh=2.0; rec_gain=0.9; GPU; 541s):

| metric | PARITY (dn + recurrent active) | PARENT (fixed floor, no companion) |
|---|---|---|
| read_fidelity (mean / min) | 0.5432 / 0.4245 | 0.5515 / 0.4388 |
| projection_recovery (vs oracle) | **0.4265** | **0.4269** |
| oracle_read_fidelity | 1.2813 | 1.2992 |
| silent_frac | 0.1933 | 0.187 |
| positive_only_fidelity | 0.4559 | 0.4724 |
| signed-load-bearing count | 4/6 | 3/6 |
| GO | 0/6 | (lift, not GO) |

projection_recovery 0.4265 vs 0.4269 is byte-level identical; read_fid, silence and the sign-fragility are all within
seed noise of the parent. The wired-and-active companion processes are inert. (The `dn_lesion`/`recurrent_lesion` mass
columns in the JSON are measured on a DIFFERENT position subset than `mass_synaptic` — a measurement confound — so they
do NOT establish load-bearingness; the controlled same-position diagnostics below are the reliable evidence.)

## RESULT 2 — the controlled diagnostics: WHY both mechanisms are inert (`_wkv_signed_read_parity_diag{1,2,3}.txt`)

**The dn pool cannot be driven proportionally by the sparse pool activity.** Same-position probe (diag1/diag2): with
no tonic drive, `dn_fire = 0.0` at every setting (floor 78-90, dn_exc 0.6-8, dn_inh 2-30) — the sparse pool spikes
never charge dn to threshold, so dn_inh has NO effect. Give dn a tonic floor so it CAN fire and it fires ~60 spikes
CONSTANTLY (its own floor, not the pools, drives it) and blanket-suppresses the read: winner spikes 4.6 -> 0.2, active
pools 13 -> 1.5, host-argmax agreement 0.30 -> 0.00. There is no middle setting: the pool activity (winner ~4-6 spikes
over 150 steps) is too sparse to drive a feedback sensor proportionally (the `R_i <- V_i/(B+Σ R_j)` denominator can't
be computed by a pool that either never fires or fires tonically).

**Recurrent within-pool excitation cannot ignite an attractor.** The winner margin is unchanged: 1.4 (rec 0) -> 1.5
(rec 8) -> 1.5 (rec 20) spikes at floor 78; 1.4 -> 1.5 at floor 86-92. The winner fires only ~4-8 spikes / 150 steps
across its 8 pool neurons — ~1 spike/neuron, far too sparse for within-pool recurrence to bootstrap the Wong-Wang
positive-feedback runaway before the window ends and lateral inhibition + reset quench it.

**Rank-order / first-to-fire (research-gate RANK-1) is WORSE.** Reading the winner as the earliest-first-spike pool
instead of the most-firing pool DROPS fidelity (mass 0.072 vs 0.120, agree 0.125 vs 0.200): the OU noise (std 40 pA)
dominates the first threshold-crossing at the ~4 pA / ~1.5-spike signal margin.

**The silence IS a real operating-point problem — the CEILING confirms it.** An idealised per-position adaptive floor
(a HOST operating-point search: ramp the floor until a pool fires, then integrate) recovers 14/15 silent positions
with a correct-winner rate (agree 0.333) equal to the non-silent positions. So the ~19% silence is genuinely fixable
by an adaptive set-point — but that host ramp is a SHORTCUT, and the NEURAL set-point that would do it synaptically is
blocked by the exact sparse-sensing limit that kills dn.

## The root cause (one cause under all three residuals)

The signed read-out is load-bearing ONLY in a SPARSE, near-threshold, SUBTRACTIVE regime (winner ~4-8 spikes, margin
~1.5 spikes, ~15-20 of 1000 pools active), because only near rest is the inhibitory shadow `Wn` subtractive rather
than divisive/shunting (the 2026-07-04 conductance-signed lesson; the parent's floor-84 control showed the high-floor
dense regime makes the sign DECORATIVE, positive-only 0.78 > signed 0.71). But that same sparse regime is:
- too SPARSE for a feedback-inhibition pool to sense the total drive proportionally (divisive normalisation dies), and
- too SPARSE for recurrent excitation to ignite an attractor (exponential sharpening dies), and
- too NOISY for rank-order to read a ~1.5-spike margin (intensity-invariant read dies).

So the sign and its two named companion processes are in genuine TENSION in this point-neuron / 150-step-window
substrate: the operating point that keeps the sign load-bearing is precisely the one that starves the mechanisms meant
to sharpen it and stabilise it. This is why the three parent residuals did not separate — they share ONE root: a fixed
sparse operating point that no per-draw, spike-driven companion process can adapt.

## Anti-cheats / provenance (hold, 6/6)

Scramble -> chance on every seed (argmax_agree_scramble ~0, post-hoc pool->word relabel collapses the read).
Provenance: 0 host categorical draws on the read path (winner from `cp_firing_states`). host_rng_draws_on_read_path=0.
argmax_agree 0.21-0.34 (~210-340x the 1/V chance) with the scramble collapse establishes the read is genuinely
signed-projection-driven — as in the parent; the companion processes changed none of it.

## The named next rung (NOT deferred)

The wall is the SPARSE 1.5-spike margin itself, not the missing sharpener. Setting the operating point and sharpening
the decision both require signal the spike-count read does not carry. Three candidate surpasses, in cost order:
1. **Read in the GRADED / conductance domain, not a 1-2-spike count.** The 2026-07-04 conductance-domain signed
   read-out is the template: an analog per-pool drive has a continuous margin an inhibitory-shadow subtraction and a
   divisive gain-pool can both act on, without needing spikes to be dense. Move the signed read off the integrate-
   and-fire spike-count and onto the graded g_e/g_i the pools already compute.
2. **A facilitating LIP-style ramp-to-threshold accumulator (Mongillo-Wang short-term facilitation).** A synaptic
   facilitating drive supplies the per-position adaptive lift INTRINSICALLY (no sensor of sparse activity), giving the
   winner time to separate before it crosses — then test whether the sign survives the ramp.
3. **LEARN the read-out + a sign-preserving normalisation end-to-end** (not transplant host `head_w`), so the trained
   margin is large enough to read robustly and the normalisation is fit to the code, not imposed by a feedback pool.

## Files
- Runner: `research/runners/_wkv_signed_read_parity_derisk.py`
- Raw: `research/findings/raw/_wkv_signed_read_parity_6seed.json` (+ the diag1/2/3 + sweep txt + diag2/3 scripts)
- Builds on: `2026-08-13-fluid-mouth-signed-shadow-readout-BOUNDARY-LIFTED.md` (the rung this attacks),
  `2026-07-05-objrel-spiking-wta-read-research-gate.md` (the divisive-norm / rank-order grounding),
  `2026-07-04-conductance-domain-signed-readout-SURPASS.md` (the conductance-signed template + the sign-decorative
  high-floor lesson).
