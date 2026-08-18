---
type: finding
status: go
date: 2026-08-17
mechanism: dmn-per-basin-encode-equalization-post-encode-consolidation-settle
lane: F-self-initiation-DMN
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_dmn_per_basin_encode_equalization_derisk.py
artifacts:
  - research/findings/raw/_dmn_per_basin_encode_equalization_derisk.json
builds_on:
  - research/findings/2026-08-14-self-initiated-all-basins-ignite-PARTIAL.md
  - research/findings/2026-08-13-self-initiation-multibasin-GO.md
  - research/findings/2026-08-13-self-initiated-utterance-GO.md
---

# Per-basin encode equalization CLOSES the all-basins-ignite boundary: the last-encoded basin failed because its one-shot BTSP eligibility trace never converted to a weight (no subsequent encode followed it) -- a POST-ENCODE CONSOLIDATION settle converts it like every other basin, and all N=4 disjoint CA3 basins then ignite SOLO on 6/6 seeds, the noise-driven wander visits all 4, and the closed loop speaks about all 4

<!--derived-->

**One-line verdict: GO 6/6.** The 2026-08-14 boundary (`self-initiated-all-basins-ignite-PARTIAL`) showed the LAST-encoded
disjoint basin fails to ignite EVEN SOLO on 6/6 seeds -- an absolute-threshold weakness that STP, intrinsic-excitability
and a 3x within-recurrent boost could not move. This rung IDENTIFIES the cause and CLOSES it. The cause is not the cells,
not weight magnitude, not competition, and not accumulated dendritic state: it is that the one-shot BTSP write is not
instantaneous -- the plateau sets a slow eligibility trace that converts to a synaptic weight over SUBSEQUENT steps, and
every basin's trace is converted by the NEXT basin's drive EXCEPT the last, which has no subsequent encode. A POST-ENCODE
CONSOLIDATION settle (BTSP still active, zero input) converts the final basin's trace like all the others. Result on 6/6
seeds: all N=4 disjoint basins ignite SOLO (up from 3/4; mean solo member 0.44-0.46 vs a ~0.03 random floor), the balanced
uniform-gain wander visits all 4 (`n_visited_coherent==4`), and the curiosity-on closed loop speaks about all 4 concepts
(about-selected 1.00 vs scramble 0.00; novel-share 0.92 on vs 0.09 reversed, 91% attributable to the curiosity gain).
FUNCTIONAL CORRELATE only, no phenomenal claim.

## The diagnosis (GPU-measured, seed 42) -- five levers falsified, then the cause isolated

The boundary named two candidate levers (interleaved encode; homeostatic ignitability). Both were FALSIFIED here, and
three more with them -- which is what let the true cause surface:

- **Interleaved encode (round-robin the BTSP passes)** -- BANKED NEGATIVE. It DESTROYS the consecutive-drive compounding
  a strong write needs: every basin's write collapses to w~25 (vs sequential's strong 150-210). Reordering is not the fix.
- **Isolated encode (hold co-basins at weak init during each write)** -- BANKED NEGATIVE. Removing the ambient recurrence
  removes the network amplification that builds a strong write; all writes collapse to ~25.
- **Homeostatic recall-time gain to a single w_within setpoint** -- BANKED NEGATIVE. Scaling the tail's within-recurrent
  gain by 5.6x to MATCH the strong basins' mean weight (w 28 -> ~157) STILL does not ignite it (member 0.15, dwell 0).
  The failure is NOT weight magnitude -- confirming the boundary's own claim, now with the mechanism attached.
- **Clamp every competitor silent during each write (-4000 pA)** -- BANKED NEGATIVE. Co-basin activity is not the cause:
  the tail still writes w~23 with all competitors held silent. (It did make the early basins stronger, 233/259/281.)
- **Full `_hard_silence` dynamic-state reset between basins** -- BANKED NEGATIVE. Resetting membrane / recovery /
  conductance / apical-latch state before each basin leaves the tail at w~25. Not accumulated dendritic/adaptation state.

**The cause is POSITIONAL, and definitively so.** Re-ordering the encode moves the collapse to WHICHEVER basin lands
LAST: order [0,1,2,3] -> basin 3 collapses (w 33); [3,2,1,0] -> basin 0 collapses (w 28), basin 3 now first is strong
(156); [1,3,0,2] -> basin 2 collapses (w 28). The weak basin tracks the last POSITION, never the cells -- which rules out
the "connectivity-poor 240-cell subset" reading of the boundary. **The mechanism:** the BTSP write is plateau-gated onto a
slow eligibility trace (`btsp_elig_tau_ms=1000`) that converts to a synaptic weight over ONGOING post-plateau steps. In
the sequential encode each basin's eligibility is converted by the drive of the NEXT basin; the LAST basin has no
subsequent encode, so its eligibility never fully converts -> a sparse (connectivity-poor) write that no recall-time gain
can rescue. This is a PROTOCOL gap -- the missing companion process the GO substrate's encode replaced with "stop after
the last basin". The settle sweep proves it directly (seed 42, per-basin w_within): plain [145, 154, 198, 29.5];
settle-150 [155, 184, 221, 98]; settle-600 [176, 204, 240, 281]; a throwaway dummy pass after the real basins
[120, 154, 159, 188]. More post-encode consolidation -> the tail's trace converts -> a dense write.

## The surpass that works: post-encode CONSOLIDATION settle ("consolidated" encode)

The sequential BTSP encode is followed by `settle_steps` (=600) of simulation with BTSP still active and ZERO input, so
the final basin's eligibility converts to a weight like every other basin's. It is ONE global parameter, NOT a per-basin
thumb -- the last basin benefits most because it had the least subsequent activity, so the equalization EMERGES. Biology:
synaptic / behavioural-timescale consolidation AFTER the plateau -- the eligibility-trace-to-weight conversion IS the
consolidation, and it needs offline time after encoding (the reason systems consolidation is not instantaneous).

## The decisive result: per-basin SOLO ignition, 6 seeds (consolidated vs the sequential GO baseline)

Each basin is run SOLO (every other basin's within-recurrence zeroed) on a fresh build, so it completes UNCONTESTED on its
own encoded weights -- the boundary's own diagnostic. `member` is the assembly-active fraction during detected events
(ignition floor 0.30, and > 2x the random non-member floor ~0.04); `w` is the mean within-assembly recurrent weight. All
per-seed numbers below are read from the committed artifact `research/findings/raw/_dmn_per_basin_encode_equalization_derisk.json`
(runner `research/runners/_dmn_per_basin_encode_equalization_derisk.py --encode-mode consolidated --settle-steps 600`).

| seed | consolidated per-basin member | consolidated w_within | sequential per-basin member (tail last) | seq w tail |
|------|-------------------------------|-----------------------|-----------------------------------------|-----------|
| 42   | 0.45 0.45 0.45 0.44 (4/4)     | 168 203 244 283       | 0.34 0.39 0.37 **0.10** (3/4)           | 31 |
| 43   | 0.45 0.46 0.45 0.43 (4/4)     | 185 207 244 276       | 0.39 0.39 0.38 **0.12** (3/4)           | 30 |
| 44   | 0.45 0.46 0.45 0.43 (4/4)     | 172 202 243 283       | 0.38 0.37 0.40 **0.13** (3/4)           | 31 |
| 100  | 0.46 0.46 0.45 0.44 (4/4)     | 167 200 257 274       | 0.39 0.40 0.33 **0.13** (3/4)           | 25 |
| 101  | 0.45 0.45 0.44 0.43 (4/4)     | 179 212 246 276       | 0.41 0.40 0.39 **0.14** (3/4)           | 35 |
| 102  | 0.46 0.45 0.46 0.44 (4/4)     | 196 207 253 283       | 0.35 0.37 0.39 **0.12** (3/4)           | 31 |

Every seed: consolidated ignites all 4 basins SOLO (member 0.43-0.46, well above the 0.30 floor and the ~0.04 random
floor); the sequential GO baseline ignites exactly 3/4, the tail (last-encoded) collapsing to member 0.10-0.14 at w 25-35.
The equalization is load-bearing (4/4 vs 3/4) every seed.

## The wander and the closed loop cover the WHOLE store

On the consolidated store, with a byte-frozen conn.data during the wander (`array_equal` before/after = True every run):
the balanced UNIFORM-gain noise-driven wander visits all 4 disjoint basins (`n_visited_coherent == 4`, overlap 0) on every
seed -- the tail wins fewer races (lower dwell) but does ignite, so competition no longer locks it out once it can
complete. The curiosity-on closed loop then speaks about all 4 concepts every seed (`n_concepts_spoken == 4`, about-rate
1.00, coherence member ~0.41-0.45 vs random ~0.03-0.04).

## Anti-cheats (each verified in the artifact, per seed)

- **DISJOINT** -- max pairwise assembly overlap == 0 every seed (genuinely pattern-separated basins).
- **Byte-FROZEN recall** -- the equalization is entirely at ENCODE (the settle); conn.data is `array_equal` before/after
  every measured wander and every solo probe. No plasticity during the measurement.
- **Equalization LOAD-BEARING** -- the sequential GO baseline regresses to 3/4 solo every seed; consolidated is 4/4.
- **INTERNALLY-triggered** -- NO-NOISE (gains on, noise off) -> 0 utterances every seed; apical stays at/below the
  bistable hold voltage. The ignition is genuinely noise-seeded, not a self-igniting runaway.
- **SUBSTRATE-attributable** -- STORE-LESION (same noise/gains, encode skipped) collapses the utterance stream every seed.
- **NOT a host thumb** -- the consolidation is a SINGLE global `settle_steps`, run as the substrate's OWN BTSP with zero
  input; per-basin equalization emerges because the last basin had the least subsequent conversion time. No per-basin set.
- **Determinism** -- substrate seeded via `cfg.seed` (the build is byte-deterministic; the GPU encode is per-synapse
  non-deterministic, so all comparisons are FUNCTIONAL solo-ignition across seeds, not per-synapse byte-identity).
- **NO STP** -- short-term depression is banked from the boundary; not used here.

## Honesty boundary + host/spiking split

FUNCTIONAL CORRELATE only, no phenomenal claim. The reactivation, the solo completion, the noise-seeded wander and the
between-event silence are the substrate's own spiking dynamics. The consolidation settle is the substrate's OWN BTSP
eligibility-to-weight conversion running with zero external input -- the only host element is the DECISION to run those
steps (a consolidation protocol, analogous to giving the system offline time after encoding). The per-concept novelty
levels and the projection of the spiking curiosity want onto the engram as a recurrent gain remain the declared host
boundary carried from the multibasin/utterance GO findings (named next rung there: release the modulator onto CA3).

## What is banked, what is next

- **BANKED NEGATIVE (this arc):** interleaved encode; isolated (init-ambient) encode; clamped-competitor encode;
  full `_hard_silence` inter-basin reset; homeostatic recall-time gain scaling to a common w_within setpoint (5.6x does
  not ignite the tail -- magnitude is not the cause).
- **CLOSED:** all N=4 disjoint basins ignite SOLO (the boundary's absolute-threshold residual) via the post-encode
  consolidation settle; the wander and the closed loop cover the whole store.
- **NEXT RUNG:** scale N (n_mem 5/6/8) -- the settle should generalise (it is a global consolidation, not tuned to N=4),
  but the shared-basket headroom at larger N is untested; and fold this consolidated multi-basin store into the
  integrated production loop (the standing integration spine).
