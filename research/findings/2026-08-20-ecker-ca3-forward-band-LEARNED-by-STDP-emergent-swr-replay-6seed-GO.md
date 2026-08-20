---
type: finding
status: live
date: 2026-08-20
mechanism: swr-sequence-replay
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_ecker_adex_ca3_stdp_band_derisk.py — the Ecker AdEx CA3 forward-asymmetric band GROWN by spiking STDP from an A->B->C encoding sweep (not installed), then frozen + replayed, with the full anti-cheat battery incl. NO-ENCODE
runner: research/runners/_gap5_ecker_adex_ca3_stdp_band_derisk.py
external: NO-EXTERNAL-NEEDED — the emergence (STDP-learned) version of the already-banked Ecker AdEx CA3 replay GO ([[2026-08-20-ecker-adex-ca3-forward-replay-6seed-GO-closes-the-swr-wall]]); STDP LTP/LTD is standard, the question is whether it GROWS a load-bearing SWR band, answered here.
artifacts:
  - research/findings/raw/gap5_ecker_adex/ecker_adex_ca3_stdp_band_6seed.json
---
# GO (6-seed 5/6): the Ecker CA3 forward-replay band is now LEARNED by STDP, not hand-wired — the SWR replay is EMERGENT

Artifact: research/findings/raw/gap5_ecker_adex/ecker_adex_ca3_stdp_band_6seed.json

**One line.** The Ecker AdEx CA3 forward-replay GO
([[2026-08-20-ecker-adex-ca3-forward-replay-6seed-GO-closes-the-swr-wall]]) used a HAND-WIRED forward band (w_fwd=800 /
w_rev=15 installed) — its #1 residual per the emergence bar. This grows that band by spiking STDP from experience
instead, and the discrete forward replay holds: **6-seed GO 5/6, with NO-ENCODE = 0.000 on ALL six seeds** — the order
came from LEARNING, not from any residual hand-wiring.

## The band EMERGED (measured, not assigned)
<!--derived-->
The between-assembly links START symmetric + weak (forward AND reverse both injected at `between_init=15`, marked
`plastic=True`); within-assembly recurrence (w=60) is injected `plastic=False` so the engine's `cp_synapse_plastic_mask`
lets ONLY the between-edges learn. An ENCODING phase sweeps a moving external cue A->B->C->...->F (onset lag 8 ms, lap
gap > the 5·tau STDP window so cross-lap pairings are skipped); the engine's fused STDP kernel then potentiates forward
(pre-before-post, Δt>0 -> LTP) and depresses reverse (post-before-pre, Δt<0 -> LTD). **Result (6-seed mean): adj_fwd
15.0 -> 330.3 (~22×), adj_rev 15.0 -> 11.5 (DEPRESSED)** — the intended LTP/LTD signature. `band_before` is asserted
exactly 15/15 at build (ratio 1.0), so the ~29× asymmetry is 100% STDP-produced, never host-assigned. (Engine gotcha
handled: `_run_one_simulation_step` does not advance `current_time_ms`, so without advancing it every step every spike
shares one timestamp, Δt≡0, and STDP is silently inert — the first smoke reproduced that exactly, then it was fixed;
`cp_last_spike_time` also only allocates when STDP is enabled at init, so build enables STDP then flips it off until encode.)

## The 6-seed verdict (GO 5/6) — the KEY control is NO-ENCODE
<!--derived-->
Frozen the learned band, then replayed (SWR envelope + non-specific prefix seed): forward-from-seed **0.935** vs reverse
**0.000**, well above chance; per-seed forward [0.957, 0.95, 1.0, 0.938, 0.765, 1.0]; discrete (rests silent);
weights frozen at replay (byte-hash identical all arms). Anti-cheats:
- **NO-ENCODE (the emergence control): 0.000 on ALL 6 seeds.** Skip the STDP encoding phase -> the band stays
  symmetric-weak -> NO forward replay. The ONLY difference from the GO arm is the learning, so the forward order is 100%
  attributable to LEARNING — there is no residual hand-wiring to fall back on.
- **REVERSE-ASYM-LESION** collapses forward to 0.313 (fwd≈rev) on 5/6 — the order rides the LEARNED asymmetry.
- SHUFFLED-STORE 0.346 (collapses), PERMUTED-ASSEMBLY 0.000, NO-SEED silent, ADAPT-LESION 0.74–0.95 (refractoriness also
  self-terminates — honest), FROZEN-during-replay byte-identical, NUMPY-REFERENCE guard by construction.

Seed 100 is the sole NO — it fails ONLY `reverse_lesion_collapses` (its symmetrized-band forward landed 0.571 > thr).
Every other gate, including `no_encode_collapses`, passes on all 6 seeds. The runner's top-level `tools.verdict.Verdict`
resolves GO with all 7 preconditions ok + an `attributable_to` (forward vs no-encode).

## Honest scope + next
The forward-replay band is now EMERGENT (STDP-grown), removing the flagged hand-wired-band residual. Declared scope,
not hidden: (1) within-assembly recurrence is FIXED — the assemblies are pre-formed cell groups; only the inter-assembly
SEQUENCE is learned (the correct scope for THIS residual). (2) The A->B->C encoding cue is a host-provided
teaching/experience signal (the WORLD driving learning), analogous to the replay prefix; the brain learns the band from
it via its OWN STDP. (3) The reverse-asym control is edge-seed-sensitive (assembly-0 seeds can only cascade forward,
inflating the symmetrized-band forward — present in the hand-wired GO too, seed 100 marginal); a scorer refinement
(exclude edge-seed events), not a substrate issue. NEXT: wire this STDP-learned discrete-forward-replay reactivation into
the D5 episodic organ (learn-through-use) + the spiking-CA3 sleep-replay store (brain-pure consolidation) — now with a
LEARNED, not scaffolded, replay. Not wired live. (Agent-built; parent verified the 5/6 GO + band_before=15/15 +
NO-ENCODE=0.000 from the artifact.)
