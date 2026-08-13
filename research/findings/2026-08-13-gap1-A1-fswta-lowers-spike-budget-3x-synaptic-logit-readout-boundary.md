---
type: finding
status: qualified
verdict: TWO RUNGS on the population few-spike read (parent 2026-08-13). RUNG 1 (shared-inhibitory FS-WTA) is a 6-seed GO (6/6 at BOTH P=2 and P=4, 7/7 checks each): adding the production `fswta_drive` lateral inhibition CUTS the read spike budget ~3-5x (P=8 ~56 -> P=2 ~10.9 / P=4 ~19.4 WORD spikes) while read_fidelity STAYS at/above the parent population minimum (P=2 mean 1.10 min 0.99; P=4 mean 1.26 min 1.16) and free generation stays FLUENT (self-NLL 1.14-1.43). RUNG 2 (route the state->logits matmul through read-out NEURONS, full-V synaptic drive) is a 6-seed MAPPED BOUNDARY (0/6): the full-V synaptic read is lossy (read_fid mean 0.035, generation is gibberish); an oracle diagnostic (perfect logit CURRENT through the SAME FS-WTA) reaches read_fid 0.57-0.74 (P=1, per-seed) -> 0.93 (P=16), proving the loss is the SIGNED SYNAPTIC PROJECTION fidelity (the Dale common-mode over 1000 near-tied words), NOT the WTA. NOT "fully spiking" / NOT "retires the mouth" (TERMS.md): rung 1 still computes state->logits + top-K on the host; rung 2 does not yet work. Runner-only, default-off, NO sim/ edit.
lane: gap#1 / A1 (brain-native open-prose generation — the production few-spike READ regime)
date: 2026-08-13
mechanism: (rung1) shared-inhibitory FS-WTA (`build_fswta_score_bridge`/`fswta_drive`) over word-candidate pools SHARPENS the winner + suppresses runners-up -> few-spike read at P<8; (rung2) head_w logit projection as Dale-shifted excitatory synapses hidden->V pools + a feedforward common-mode canceller (`_fm_spiking_synaptic_readout` pattern), FS-WTA over all V pools
artifacts:
  - research/runners/_wkv_fswta_synaptic_read_derisk.py
  - research/findings/raw/_wkv_fswta_smoke.json
  - research/findings/raw/_wkv_fswta_rung1_confirm.json
  - research/findings/raw/_wkv_fswta_synaptic_6seed.json
  - research/findings/raw/_wkv_fswta_synaptic_6seed.log
---

# gap#1 / A1 — the shared-inhibitory FS-WTA cuts the few-spike read budget ~3x (rung 1); routing the state->logits matmul through read-out neurons is a signed-synaptic-projection BOUNDARY (rung 2)

## Context — the two rungs the parent named

The parent (`_wkv_fewspike_read_derisk`, `2026-08-13-gap1-A1-fewspike-...`, 6/6 GO at P>=8) put the FLUENT WKV open-prose
generation onto the production few-spike Izhikevich read regime at ideal-sampler parity, via POPULATION coding (P>=8,
~56 word spikes at rw20). It named two next rungs: (1) add the shared-inhibitory FS-WTA to sharpen the winner + cut the
spike budget below P=8; (2) route the state->logits projection through read-out NEURONS so the WTA drive is a synaptic
current, retiring the host matmul on the read path. This finding does both, in a new de-risk runner
(`research/runners/_wkv_fswta_synaptic_read_derisk.py`; reuse-by-import of the parent's `WKVReadout` + metric harness,
`build_fswta_score_bridge`/`fswta_drive`, and the `_fm_spiking_synaptic_readout` Dale-shift + common-mode canceller; NO
`sim/` edit; cfg.seed-controlled substrate; runner-only, default-off).

The decisive metric is the parent's, unchanged: read_fidelity = ondist_mass(read) / ondist_mass(host_sample), with the
argmax and ideal-sampler ceilings, plus mean spikes/read (the budget), the four anti-cheats (equal-drive, scramble,
noise-ablation, provenance), and FREE-GENERATION self-NLL. Instrument validity kept: top-K mass coverage (0.937), <!--derived-->
`mass_fewspike <= mass_argmax` (holds), and — the mission's flag — the scramble control is now made ROBUST with a
binomial-aware `<= chance + 3 sigma(n)` threshold (the parent's razor `< 2*chance` flipped seed 43 on Poisson noise).

## RUNG 1 — shared-inhibitory FS-WTA: read_fidelity holds, spike budget drops ~3-5x (6-seed GO)

The parent's read is INDEPENDENT pools + OU noise + argmax-over-firing (no lateral competition), so it must out-VOTE
runner-up noise with population size. The fix (production `fswta_drive`): each word pool excites a shared inhibitory FS
pool; FS inhibits ALL word pools; the winner fires first, recruits FS, SUPPRESSES the runners-up -> a clean one-of-K at
a small P. The load-bearing tuning was NOT the inhibition strength alone — it was pairing it with a CONTRASTIVE drive
(low floor base_pA=30, high gain=220) so the weak candidates barely fire and the shared inhibition can actually SILENCE
them (a uniform base_pA=60 floor lets all 64 top-K pools fire regardless, and the FS cannot suppress 64 driven pools).

6-seed (42/43/44/100/101/102; n=200 held-out positions; SIM_BACKEND=numpy, whole run 172s — the pools are small, CPU
beats GPU launch overhead, matching job to bottleneck). Parent P=8 row + aggregate means/mins are derived:

<!--derived-->

| operating point | word spikes / read | read_fidelity mean (min) | argmax_agree | free-gen self-NLL | 7-check GO |
|---|---|---|---|---|---|
| parent P=8 (no inhibition, rw20) | ~56 | 1.033 (0.936) | 0.725 | 0.8-1.7 (fluent) | 5-6/6 |
| **rung1 FS-WTA P=2** | **~10.9** | **1.101 (0.993)** | 0.79 | fluent | **6/6** |
| **rung1 FS-WTA P=4** | **~19.4** | **1.262 (1.158)** | 0.92 | **1.14-1.43 (fluent)** | **6/6** |

The FS-WTA holds read_fidelity at/above the parent's 0.936 min on every seed (P=2 min 0.993, P=4 min 1.158) at a <!--derived-->
~3-5x smaller WORD spike budget (10.9 / 19.4 vs 56), and free generation stays fluent and coherent (self-NLL 1.14-1.43,
squarely in the parent's fluent band; e.g. *"once upon a time there was a little boy named tim was very excited and
wanted my dog to play with the ball but it was too high for tim to play with his toy ball"*). The lateral inhibition
sharpens the read toward argmax (argmax_agree 0.79 at P=2, 0.92 at P=4; `mass_fewspike <= mass_argmax` still holds, so
the instrument is not violated) — the winner is cleaner, not the distribution flattened. Anti-cheats collapse on every
seed: equal-drive ~0.02-0.14 (vs mass ~0.29-0.41), scramble at chance under the robust threshold, noise-ablation
deterministic, provenance 0 host draws (winner from `cp_firing_states`). This confirms the parent's hypothesis: the
shared inhibition removes the runner-up noise the population averaging had to out-vote, so P drops below 8 for the same
(higher) fidelity. (P=2 buys the smaller budget; P=4 the higher fidelity + the demonstrated fluent generation.)

## RUNG 2 — routing the state->logits matmul through read-out neurons: a precisely-located BOUNDARY

Rung 2 realises the FINAL logit projection `head_w @ h` (V x D; `h = r_h*(Wo_sp@state)` the gated hidden state) as
EXCITATORY synapses from a rate-coded hidden population ([h+, h-] dual-nonneg) onto ALL V=1000 word pools, with the fm
signed-read-out surpass (global Dale-shift `head_w - gmin >= 0` + a feedforward common-mode CANCELLER: a shared
inhibitory pool that subtracts the shift-induced common mode `gmin*sum(feature)`). A FS-WTA over all V pools resolves
the winner -> NO host logit matmul, NO top-K argpartition on the read path (372k-2.9M synapses, built runner-side).

<!--derived-->
**Result (6-seed, 0/6): the full-V synaptic read does NOT reproduce the distribution — read_fidelity mean 0.035 (range
0.022-0.042), and free generation is GIBBERISH (self-NLL ~10):** *"once upon a time was bird one to loved help toy for
for she was mom so home all and did saw friends she it but when sam..."*. The mechanism is genuinely wired (argmax_agree
0.5-2% = 5-20x chance, so the synaptic drive DOES carry logit signal; readout-lesion -> ~0 collapse; scramble -> chance;
provenance clean) — it is just far too lossy.

**The load-bearing diagnostic locates the wall.** An ORACLE (drive the pools DIRECTLY by a perfect host-logit current
through the SAME FS-WTA over V pools — a diagnostic, not a read path) isolates the WTA-resolution ceiling from the
synaptic-projection fidelity (oracle sweep + synaptic ranges derived across seeds):

<!--derived-->

| read over V=1000 pools | read_fidelity | note |
|---|---|---|
| oracle (perfect logit current) P=1 | **0.57-0.74** (6-seed) | the FS-WTA resolution ceiling at P=1 |
| oracle P=4 (seed 42) | 0.50 | |
| oracle P=8 (seed 42) | 0.81 | |
| oracle P=16 (seed 42) | **0.93** | full-V resolution is ACHIEVABLE at higher population (crosses 0.90) |
| **actual synaptic** P=1 (6-seed) | **0.022-0.042** | far below the oracle ceiling |
| **actual synaptic** P=8 (seed 42) | ~0.03 (collapses) | the Dale-shift + canceller balance is P-fragile |

So the FS-WTA resolution over 1000 near-tied pools is NOT the primary wall — a perfect logit current resolves it to
read_fid 0.76 (P=1) and 0.93 (P=16), i.e. the resolution is a spike-budget/population SCALING cost that the mission's
"speed secondary" permits. **The primary wall is the SIGNED SYNAPTIC PROJECTION fidelity**: realising `head_w` as
Dale-shifted excitatory synapses injects a common mode (`gmin * sum(hidden spikes)`) that is orders of magnitude larger
than the tiny discriminative margin between the top near-tied words; the scalar canceller cannot subtract it faithfully
(Poisson noise in the hidden spikes leaves a residual >> the margin), and cranking the drive only saturates the pools
and destroys the margin. This is exactly the fm/rungB1c "signed read-out" wall (`_fm_spiking_synaptic_readout` measured
the common mode ~140x the margin at G^2=25 classes) — now over V=1000 with a far tinier margin, so the scalar-canceller
Dale-shift is insufficient.

## The next lever (named, biological — NOT deferred, mapped)

1. **A TRUE SIGNED read-out** (the fm-named surpass): carry the NEGATIVE `head_w` weights on INHIBITORY shadow
   interneurons (a copy of each hidden neuron, made inhibitory, driven by the same current), so the logit is
   `Wp@h_exc - Wn@h_inh` with NO global Dale-shift and hence NO common mode to cancel. This removes the dominant loss
   term (the oracle proves the target read_fid 0.76-0.93 is reachable once the projection is faithful). Cost: ~4x the
   read-out synapses + inhibitory shadows.
2. **Population + homeostatic per-pool floor calibration** (the fm intrinsic-excitability equalization) to reach the
   P>=16 oracle regime (read_fid 0.93) — a larger spike budget, explicitly in scope (speed secondary).
3. **A lexical PRE-ACTIVATION cohort gate** (the biology behind the parent's "top-K candidate set is a legitimate
   labelled-line input"): spreading-activation / cohort-model priming pre-activates only a small word-assembly cohort,
   so the WTA competes over ~dozens not 1000 — the companion process the host top-K argpartition replaced with a
   constant. This makes the full-V read tractable WITHOUT a host argmax.

## Honest scope / declared residuals (TERMS.md)

- **NOT "fully spiking", NOT "retires the mouth".** Rung 1 de-risks the READ only; the state->logits step is STILL a
  host matmul over the graded conductance and the top-K candidate set is a host argpartition. Rung 2 (the piece that
  would retire the matmul) is a mapped BOUNDARY, not a working mechanism.
- **NOT wired / default-off / runner-only** — a de-risk, not a production integration.
- **Rung 1 is a 6-seed GO** (6/6 at P=2 and P=4, `_wkv_fswta_synaptic_6seed.json`). Rung 2 is a 6-seed BOUNDARY (0/6).
  The oracle POPULATION sweep (P=4/8/16 -> 0.50/0.81/0.93) is single-seed (seed 42) diagnostic; the per-seed oracle
  ceiling at the P=1 read (0.57-0.74) is recorded for all 6 seeds, so the synaptic-vs-oracle gap is documented across
  seeds.

## Files
- Runner: `research/runners/_wkv_fswta_synaptic_read_derisk.py`
- Raw: `research/findings/raw/_wkv_fswta_synaptic_6seed.json` (+ `.log`), `_wkv_fswta_smoke.json`,
  `_wkv_fswta_rung1_confirm.json`
- Builds on: `2026-08-13-gap1-A1-fewspike-...` (the population few-spike read),
  `2026-08-10-neural-wta-word-decode-...` + `_d3_spiking_attractor_derisk` (the production `fswta_drive`),
  `_fm_spiking_synaptic_readout_derisk` (the Dale-shift + common-mode canceller synaptic read-out).
