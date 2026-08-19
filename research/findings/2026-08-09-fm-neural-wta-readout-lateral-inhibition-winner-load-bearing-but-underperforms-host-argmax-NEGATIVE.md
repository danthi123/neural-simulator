---
type: finding
status: contributing
date: 2026-08-09
mechanism: forward-model-neural-wta-readout-lateral-inhibition-winner
lane: world-model
---
# The world-model read-out WINNER as a genuine load-bearing NEURAL WTA (lateral inhibition) — HONEST NEGATIVE: it UNDERPERFORMS the host argmax (6-seed 0.33 vs 0.45) and does not approach the ceiling — the residual is the spike-count evidence, not the winner op

All 6-seed means below are derived; per-seed source `research/findings/raw/_fm_neural_wta_readout_s*.json`, aggregate `research/findings/raw/_fm_neural_wta_readout_6seed_agg.json`.

<!--derived-->
**One line.** The held two-pathway spiking read-out (`2026-08-08-fm-learned-twopath-spiking-readout...`, held off-main
branch `worktree-wf_122d7930-bc4-2`) reached held-out ~0.447 with a delta-trained TWO-PATHWAY (W+ excitatory / W-
feedforward-inhibitory) Dale realization, but the WINNER was a host `np.argmax(x_spk)`. We replaced that argmax with a
genuine NEURAL WTA — the ensemble's own recurrent LATERAL-INHIBITION biased competition (Wang 2002; Carandini-Heeger
divisive normalization), read from `cp_firing_states`. **The neural winner is genuinely LOAD-BEARING** (remove the
inhibition and it collapses to chance; 6-seed fallback 0.087, gap +0.247) and matches the accepted project neural-WTA
pattern (`2026-07-04-rungB1b-neural-role-wta-GO`, `2026-07-13-PAST-RESERVOIR-ONBRIDGE...-6seed-GO`). **But it is a
NEGATIVE (GO 0/6):** at 6 seeds the neural-WTA held-out is **0.333** (8x chance, 45% of the ceiling) — BELOW the
baseline host-argmax **0.447** and far below the two-pathway rate ceiling **0.747** — and it is highly seed-variable
(0.04-0.60: 3/6 seeds land 0.52-0.60, 3/6 collapse to 0.04-0.16 when the swept inhibition over-suppresses). **The
winner op was never the bottleneck — the spike-count EVIDENCE is; re-competing it neurally, if anything, adds noise.**

**NO-EXTERNAL-NEEDED:** this is a METHOD-level negative (a neural WTA winner underperforms the host argmax on THIS
substrate), NOT a fundamental/capability limit — it explicitly names the next biological mechanism to close the residual
(current-subtractive, not conductance/shunting, inhibition at the read-out membrane so relative firing tracks the rate
logits). The winner mechanism itself is grounded in cited biology (Wang 2002 biased competition; Carandini & Heeger
2012 divisive normalization) and two in-repo GO precedents (rungB1b, PAST-RESERVOIR); the local corpus was checked via
`tools/before_you_build.sh` before building. No fundamental-ceiling verdict is banked.

## What was actually done (the burn-down of the host-argmax residual)
The baseline's declared residual was "the winner is a host `np.argmax` over spike-counts". We reframed it (CLAUDE.md:
"what runs alongside this constant?"): the baseline had proxied the ensemble's biased-competition GAIN with a weak
constant (`WTA_W_IE=10`), so the ensemble counts were near-tied and the host argmax carried the resolving. Restoring the
companion process — a genuine **lateral-inhibition WTA** (each output ensemble -> a shared inhibitory pool -> all
ensembles, sweeping the inhibition strength on TRAIN) — makes the SPIKING competition resolve the winner: the correct
ensemble suppresses its rivals, and the winner is read from `cp_firing_states` (which competed ensemble fires most, a
motor-style read). **The RESOLVING is neural** — proven load-bearing by the I->E lesion below — **not the argmax.**

## The winner is NEURAL and LOAD-BEARING (the task's exact teeth — this part holds)
<!--derived-->
Zeroing the lateral-inhibition I->E synapses (`x_wta`->`x_ens`, `y_wta`->`y_ens`) removes the biased competition; the
argmax over the un-competed ensemble counts (the host-argmax fallback) collapses to ~chance (6-seed mean **0.087** vs
neural-WTA **0.333**, gap **+0.247**). WITH the competition the read-out resolves above chance; WITHOUT it, it does not
— the winner is the neural WTA, not the argmax. The selected inhibition strength is an inverted-U (ie=0 -> chance,
ie~40-70 optimum, ie>=110 over-suppresses the ensembles into a noisy sparse winner); the TRAIN sweep selects it per
seed (mean `wta_ie_selected` ~72).

## The honest ceiling residual (NO-GO on the strict bar — mapped precisely)
The neural-WTA held-out does NOT approach the two-pathway rate ceiling. Three measurements locate the residual in the
spike-count EVIDENCE, not the winner:
1. **More integration does not help.** Held-out is FLAT (~0.44-0.56) from 2 to 16 replays; the top1-top2 ensemble
   spike-count margin is a STRUCTURAL ~8%, not Poisson/sampling-limited. So the gap is not sampling noise.
2. **A downstream stochastic decision WTA DEGRADES the winner** (measured 0.12-0.31 < host argmax) — its own firing
   variability exceeds the 8% margin. Re-competing the evidence cannot recover information the sparse counts lost.
3. **Stronger inhibition over-suppresses** (ie>=110 drops held-out and silences the ensembles) — it cannot lift the
   margin either.
The gap is therefore the **rate->spike-count representational loss** at the ensemble membrane (the W+/W- current sum
plus the Izhikevich f-I nonlinearity distort the linear rate logits; the "conductance vs current-subtractive
inhibition" residual the baseline named). The fully argmax-FREE one-hot spiking latch (a sole-surviving ensemble) does
NOT resolve at the ~0.01 mean firing rate (dominant-rate ~0 even at strong inhibition), so the final index-read remains
a motor-style argmax over the competed population — declared, and consistent with the two GO precedents above.

## Results (6 seeds: 42 43 44 100 101 102)
<!--derived-->

Per-seed order 42/43/44/100/101/102.

| metric | per-seed | 6-seed mean | note |
|---|---|---|---|
| host ridge / two-pathway RATE held-out (ceiling) | 0.84/0.76/0.72/0.76/0.76/0.64 | 0.747 | == ridge to 1e-6 (decomposition exact) |
| **neural-WTA held-out (deliverable)** | 0.60/0.56/0.04/0.16/0.52/0.12 | **0.333** | lateral-inhibition winner; BELOW baseline, seed-variable |
| lateral-inhibition LESION held-out (argmax fallback) | 0.16/0.00/0.00/0.04/0.16/0.16 | 0.087 | remove competition -> ~chance (winner load-bearing) |
| prior baseline host-argmax (weak WTA) | — | 0.447 | the host op the neural WTA underperforms |
| selected lateral-inhibition strength (wta_ie) | 40/70/70/70/70/110 | 71.7 | swept on TRAIN (inverted-U optimum) |
| W+ read-out lesion held-out | 0.00/0.00/0.04/0.04/0.08/0.08 | 0.040 | collapses (teeth) |
| reservoir-silence lesion held-out | 0.00/0.00/0.04/0.04/0.08/0.16 | 0.053 | collapses (teeth) |
| matched-sham (decoy lesion) held-out | 0.60/0.48/0.04/0.16/0.48/0.12 | 0.313 | UNCHANGED vs deliverable (teeth; s43 borderline |Δ|=0.08) |
| untrained-control held-out | 0.00/0.04/0.04/0.04/0.00/0.04 | 0.027 | chance — the MAP carries it (teeth) |
| ensemble mean firing (read) | 0.010/0.010/0.005/0.008/0.009/0.006 | 0.008 | sparse; high ie over-suppresses (s44/s102) |
| chance = 1/(G·G), G=5 | — | 0.040 | |
| verdict (strict approaches-ceiling bar) | NG/UND/NG/NG/NG/NG | **GO 0/6** | honest NEGATIVE (s43 UNDEFINED: sham |Δ|=0.08 boundary) |

All seeds: `seeded=True` (byte-identical substrate, `cfg.seed`), `content_path_clean=True` (winner from
`cp_firing_states`, no host map-matmul / logit-argmax), `twopath_rate == ridge` to 1e-6. **The neural-WTA (0.333) is
below the baseline host-argmax (0.447): re-competing the sparse spike-count evidence neurally does not help and, on the
3 over-suppressed seeds, hurts. The winner is load-bearing (chance without it) but it is not a better winner than the
argmax — the bottleneck is the evidence.**

## Anti-cheats (teeth), per seed
(i) NEURAL winner off `cp_firing_states` (competed ensemble firing), reservoir + ensembles active; (ii) content path
grep-clean of the map matmul / logit-argmax; (iii) **LOAD-BEARING WINNER** — zeroing the lateral-inhibition I->E
synapses collapses held-out to ~chance (the neural competition, not the argmax, resolves); (iv) REAL LESION — zeroing
the W+ read-out synapses OR silencing the reservoir collapses held-out to ~chance; (v) MATCHED SHAM — count-matched
lesion of an OFF-DECODE decoy read-out leaves held-out UNCHANGED (|Δ|≤0.08); (vi) UNTRAINED control — random
non-negative weights of matched magnitude → chance (the MAP carries it, not the wiring); (vii) seeded byte-identical
substrate (`cfg.seed`).

## Corrects the baseline finding's residual attribution
The baseline named its residual as "spike-count noise at ~0.02 mean rate" and proposed integration. This arc MEASURES
that integration does NOT close it (flat 2->16 replays; margin structural ~8%). The residual is the rate->spike-count
representation, not sampling noise — and the WINNER op is not the bottleneck at all (a genuine neural WTA reaches the
same ~0.45 the host argmax did). The next mechanism is an explicitly current-subtractive (not conductance/shunting)
inhibitory read so the ensemble membrane computes `exc - inh` linearly and its relative firing tracks the rate logits —
the same "companion process still proxied" the baseline flagged, now localized to the read-out MEMBRANE, not the winner.

## Repro
- SMOKE (single seed, numpy): `SIM_BACKEND=numpy python -u -m research.runners._fm_neural_wta_readout_derisk --seeds 42 --smoke`
- 6-SEED (per-seed parallel then aggregate):
  `for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -u -m research.runners._fm_neural_wta_readout_derisk --seeds $s --out research/findings/raw/_fm_neural_wta_readout_s$s.json & done; wait`
  then `SIM_BACKEND=numpy python -u -m research.runners._aggregate_fm_neural_wta_seeds`
- Aggregate (with argv/git-sha/inputs provenance): `research/findings/raw/_fm_neural_wta_readout_6seed_agg.json`.
- Per-seed artifacts: `research/findings/raw/_fm_neural_wta_readout_s{42,43,44,100,101,102}.json` (+ `.prov.json`).
- Runners: `research/runners/_fm_neural_wta_readout_derisk.py`, `research/runners/_aggregate_fm_neural_wta_seeds.py`.
- NO `sim/` edit (all wiring runner-side via `inject_explicit_wiring`/`set_pathway_weights`); reuse-by-import.
