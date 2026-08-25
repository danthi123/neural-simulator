---
type: finding
status: qualified
lane: gap#5
date: 2026-08-25
---
# gap#5 RUNG-A/B/C on the reproducible substrate: the a1-EXCLUSION was the BDSP CLAMP (not BTSP skip-links); the fix makes a1 fire and the store completes a FORWARD-ORDERED a0->a1->a2 sweep in all 6 seeds — store-driven, but the shipped detector MASKED it and the readout is not discretely gated => NO-GO 0/6 on the strict bar, residual = readout discreteness

Artifacts (n_ca3=2000, cupy, byte-reproducible substrate): `research/findings/raw/gap5_r4/rungABC_6seed_chainwiden_tau10.json`
(6-seed GO + all controls + reverse-detonation) and per-seed rest-firing dumps
`research/findings/raw/gap5_r4/rungA_traces/seed{42,43,44,100,101,102}_traces.npz` (GO / no-detonator / reverse F,
packbits). Config: `--chain-elig-tau 10 --chain-bdsp-widen --chain-btsp-lr 0.08 --fb-drive 10`. Runners: additive
default-off knobs on `_gap5_sequence_replay_derisk._prepare_sequence` (`chain_bdsp_widen`, `chain_elig_tau_ms`) +
`_gap5_dg_detonator_ignition_derisk` (`--chain-elig-tau/--chain-bdsp-widen`, reverse-detonation control, `--dump-traces`,
RUNG-B detector flags) + `_gap5_rungB_offline_calib.py` (re-scores the dumps offline). NO `sim/` edit.

## RUNG-A — the a1-exclusion root cause is the BDSP CLAMP, not the BTSP skip-link the prior finding hypothesized
The prior finding (2026-08-25 pv-basket) blamed `asm_peaks=[a0,0,a2]` on BTSP tau-1000ms eligibility writing a0->a2
SKIP-links. Direct per-edge measurement (numpy, seed 42) DISPROVES that. The 3x3 assembly weight matrix is
a0->a1 = **5.0** (exactly `bdsp_w_max`, unwritten), a1->a2 = 71, a0->a2 = 70, all reverse = 5.0. The FIRST forward link
a0->a1 is pinned at the clamp floor while links INTO a2 escape. The BTSP write gate `elig[pre] x (v_apical[post]-v_hold)`
IS satisfied for a0->a1 (instrumented: when a1 is driven it reaches a full apical plateau `vap=22.9, plat=1.00` and a0
stays eligible `elig=0.008`) — but `fused_bdsp_update` HARD-CLAMPS every BDSP-active synapse to [-5,5] EVERY step and
the cold-start first link's BTSP write cannot outrun the clamp (the stronger later links do). Shortening the eligibility
tau alone does NOT fix it (a0->a1 stays 5.0 at tau=25) — the clamp is the cause, tau is not.

FIX (two additive, default-off knobs): `chain_bdsp_widen` widens the BDSP clip for the forward btsp chain (as the
stdp/hebb_sym paths already did) so a0->a1 forms (5.0 -> 155); `chain_elig_tau_ms` shortens the CHAIN eligibility to a
gamma-cycle window (theta-gamma sequence compression, Skaggs-McNaughton precession) so ADJACENT dominates the skip.
Validated at n_ca3=400 (numpy, sweep-count-independent at chain_fwd=24): tau=10 lr=0.08 -> a0->a1=155 a1->a2=158
skip=38 (ADJ/SKIP 4.05x) reverse ~9. At n_ca3=2000 (6 seeds, the artifact): adj_fwd 50.2-68.9, adj_rev 5.6-5.8
(~9-12x forward-asymmetric, up from the shipped 7.65x). <!--derived from rungABC_6seed_chainwiden_tau10.json encode_decoupled-->

## The a1-exclusion is ELIMINATED and the readout is a FORWARD-ORDERED sweep — measured directly, all 6 seeds
Detector-INDEPENDENT raw per-assembly peak (smoothed frac, whole GO readout) shows a1 NOW FIRES strongly and comparably
to a0: raw_peak(a0,a1,a2) per seed = 100:[2.34,2.54,1.02] 101:[2.65,2.31,0.99] 102:[2.61,2.22,1.09] 42:[2.23,1.67,0.95]
43:[2.18,1.99,1.26] 44:[2.28,2.46,0.94]. The shipped store had a1 dead (`per_asm=[1,0,1]`, cross=0). The ONSET times
(first step each assembly crosses 0.5) are STRICTLY FORWARD in every seed after the step-50 detonation: onset(a0,a1,a2)
= 100:[52,62,70] 101:[52,58,62] 102:[52,58,62] 42:[52,60,89] 43:[52,61,70] 44:[52,60,66]. So a0 fires first, then a1
~6-10 steps later, then a2 — a genuine temporal a0->a1->a2 sweep, forward-dominant (corrected detector FWD>REV every
seed), and store-driven: the no-detonator control is SILENT (n_ev=0, all 6) and the REVERSE-DETONATION control (detonate
the terminal assembly) produces NO backward chain (n_multi=0, all 6) — the order is learned, not a protocol artifact.

## RUNG-B — the shipped event detector MASKED the sweep; the root fix is a baseline threshold, not mean-smoothing
The in-run verdict is NO-GO 0/6 with `per_asm=[0,0,0]`, `forward_frac=0` on EVERY seed — an INSTRUMENT failure, not an
absence. `_event_windows`/`_detect_events` set `thr = max(med + ev_k*mad, ev_floor*asize)` with med/mad over the WHOLE
1500-step trace; the sustained/repeated readout (10 detonation cycles + latching) inflates `med+4*mad` to ~1268 while
the floor is 120, so only 1-2 step spikes at the peak cross (shorter than `min_ev_len=4` => discarded) and the discrete
sweep at pop~150-285 is never seen. The `ev_mean_smooth` (transient mean-rate) knob does NOT fix this (identical result
across ev_floor). The ROOT fix is `ev_baseline_q`: take med/mad from ONLY the quietest fraction of S (the quiet
inter-sweep steps), which de-inflates the threshold to ~ the floor. Offline re-score of the dumps with it: no-detonator
stays SILENT (n_ev=0 all 6), reverse-detonation stays clean (n_multi=0 all 6), and the GO readout resolves forward
events (FWD>REV every seed; seed 102 fwd_frac 0.38 >2x chance; seed 101 3/3 forward at q=0.15). All three RUNG-B knobs
default to the shipped behaviour (verified byte-identical: `default == ev_baseline_q=None`).

## Verdict — NO-GO 0/6 on the strict bar; the residual is READOUT DISCRETENESS, not the encode
GO bar = discrete single-assembly ignition + clean forward A->B->C transition, >=5/6. With the corrected detector
(baseline q=0.15) only 1/6 (seed 102) clears the combined `forward_transition (fwd_frac>=2x chance, >reverse, n_multi>=2)
AND duty<=0.40` bar: the readout DUTY is high (0.34-0.77) — sustained multi-cycle activity, not clean isolated sweeps —
so forward_frac is diluted (0.11-0.75) even though FWD>REV in every seed. Anti-cheats all clean: no-detonator silent,
shuffled-detonator/no-encode/shuffled-within collapse, symmetric positive control ignites 5/6, plasticity frozen
byte-verified, reverse-detonation produces no backward chain. So the encode is FIXED (a1 fires, forward-asymmetric
9-12x) and the sequence is REAL + store-driven; what remains is that the READOUT is not discretely gated.

NEXT MECHANISM (no-defer, downstream of the now-fixed encode): make each detonation produce ONE clean transient sweep
instead of sustained latched activity — the biology is the theta-gamma READ clock + adaptation-timed hand-off. Levers to
de-risk next: a single (non-periodic) detonation with a longer settle; stronger/slower spike-frequency adaptation on the
just-fired assembly so a0 silences as a1 peaks (the Ecker intrinsic-fatigue transition, currently out-run by the strong
a0->a1); a slower NMDA-mediated a0->a1 transmission so a1 lags rather than co-rises; and ship the `ev_baseline_q`
detector as the scoring default. This is a READOUT-DYNAMICS rung; the encode/store rung (a1-exclusion) is resolved
(a1 fires, forward-ordered, measured 6/6) — the wall moves downstream to the read, it is not deferred.
