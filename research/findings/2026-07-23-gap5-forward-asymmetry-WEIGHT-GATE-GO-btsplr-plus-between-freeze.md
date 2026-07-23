# gap#5 forward-asymmetry: WEIGHT-GATE GO (seed 42) — btsp_lr strength + between-refresh freeze (2026-07-23)

## The arc: STDP refuted -> research gate -> the two-lever fix
After R0 proved the store is near-symmetric (the numpy replay GO rode start=A) and hetero-dep + the within_refresh/
chain_fwd knobs were NEGATIVE, the causal-STDP prescription was **refuted on the substrate**: a pure-cranked-LTP test
(`a_plus=0.5, a_minus=0`) left the ca3->ca3 recurrent FLAT (5.0/5.0/5.0) — **STDP structurally does not write this
recurrent** (the theta reset `_silence_soma_apical` empties STDP's Dt window; the recurrent is written by the
substrate's Hebbian + BTSP rules). The corrected research gate (`2026-07-23-gap5-forward-asymmetry-mechanism-research-
gate.md`) reframed it: BTSP IS a Blum-Abbott temporally-asymmetric RATE rule (`dw = eta*Etilde_pre*IS_post*(w_max-w)`),
its 1000ms eligibility survives the theta reset, and it ALREADY produces a real forward bias — the residual is a
**SEPARATION problem**: `W = W_within(~174, needed) + W_chain_fwd(~6, the real signal) + W_refresh_between(~137, a
SYMMETRIC byproduct)`; the within-refresh's bistable plateau spreads to the neighbor assembly and writes symmetric
between-links that swamp the ~6 forward bias.

## The two levers (both confirmed, seed 42, encode-only, NO sim/ edit)
**Lever 1 - btsp_lr strength** (the forward mechanism scales; reverse pinned at init, wr=0):
```
btsp_lr 0.02 -> ratio 1.27x (adj_fwd 6.3,  adj_rev 5.0)      [default]
btsp_lr 0.1  -> ratio 2.33x (adj_fwd 11.7, adj_rev 5.0)
btsp_lr 0.5  -> ratio 7.65x (adj_fwd 38.3, adj_rev 5.0)
btsp_lr 2.0  -> ratio 27.5x (adj_fwd 137.7,adj_rev 5.0)      <- adj_rev NEVER written (pure forward)
btsp_lr 0.5, wr=4 -> ratio 0.99x (adj_fwd 31.6 ~ adj_rev 31.9)  <- the within-refresh RE-SYMMETRIZES it
```
**Lever 2 - freeze_between_refresh** (isolate the pure forward + preserve the within): freeze the between-assembly
synapses during the within-refresh (`cp_plasticity_rate_gain[between_flat]=0`; between_flat computed from
`asm_of_local`) so the refresh rebuilds ONLY the within-attractor. At default weak btsp_lr, wr=8:
```
freeze OFF (control): within 173.7  adj_fwd 143.3 adj_rev 142.0  ratio 1.01x  [== baseline, byte-identical default]
freeze ON:            within 190.7  adj_fwd 6.3   adj_rev 5.0    ratio 1.27x  [pure chain-only forward + within PRESERVED]
```
The freeze removes the ~137 symmetric contaminant (between drops to the pure 6.3/5.0) and PRESERVES/improves the
within-attractor (190.7).

## WEIGHT-GATE GO (the combination: btsp_lr up + freeze + full wr=8)
```
btsp_lr 0.5 + freeze + wr=8:  within 40.0  adj_fwd 38.3  adj_rev 5.0  ratio 7.65x  => GO
btsp_lr 2.0 + freeze + wr=8:  within 39.0  adj_fwd 137.7 adj_rev 5.0  ratio 27.5x  => GO
```
Both clear the weight gate (ratio >= 2-3x AND within >= 27 AND adj_fwd > adj_rev). Forward asymmetry ACHIEVED with
the within-attractor preserved above the reactivation floor. NOTE: at lr=0.5 within(40) ~ adj_fwd(38) is a BALANCED
hold/push; at lr=2.0 adj_fwd(137) >> within(39) risks the forward PUSH overwhelming the HOLD in the spiking readout
-> lr=0.5 is the likely operating point.

## Biology (decisive, Ecker 2022 eLife 71850 — our substrate class)
Asymmetric rule -> FORWARD-ONLY replay ("absence of backward replay"); symmetric rule -> bidirectional; Tsodyks-Romani
2015 symmetric+STD -> RANDOM direction (confirms the banked STD-negative). A separable co-requirement: cellular
ADAPTATION makes the bump TRAVEL (else stationary), distinct from the weights' DIRECTION (the intrinsic-fatigue lever,
already calibrated).

## Next (the REAL gate): the SPIKING READOUT de-risk
The weight asymmetry is NECESSARY but the load-bearing proof is on SPIKES: a noise-ignited replay on these asymmetric
weights must travel FORWARD with **START-INVARIANCE** (holds regardless of ignition point, unlike the numpy start=0
artifact) AND **ASYM-LESION** (symmetrize -> forward collapses to chance), plus SCRAMBLE/NO-ENCODE, 6-seed. Ecker's
travel/adaptation lever is the separable piece if the ratio-GO doesn't yield functional forward replay.

## Edits (all default-OFF, byte-identical when off; NO sim/ edit)
`_gap5_sequence_replay_derisk.py`: `chain_rule="stdp"` (research tool, refuted), `freeze_between_refresh` (+ between_flat),
`chain_rev` passthrough. `_gap5_R1_hetero_encode_sweep.py`: `--btsp-lr --freeze-between-refresh --chain-rev --chain-rule`.
Byte-identity verified (freeze_off == baseline asym+1.26; btsp control == baseline).
