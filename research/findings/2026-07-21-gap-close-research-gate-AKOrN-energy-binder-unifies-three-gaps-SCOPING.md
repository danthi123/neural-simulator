# Gap-close research gate — the AKOrN/Kuramoto energy binder on the RF phasor substrate unifies gaps #2/#3/#5 (SCOPING)

**2026-07-21.** Owner directive: "gap-close workflows, no deferrals, then LLM-like conversation." A deep-research-gate
Workflow (a-1 RAG our record + a0 read substrate + external lit + ranked cheap-first de-risks) on the two biggest open
gaps (open generation; a learned binder over structured codes). The a-1 discipline REFRAMED gap#2 precisely (drift #12
avoided): the naive "learn the bind from scratch" is a CLOSED boundary, but the actual open path is narrow and buildable.

## The precise gap-map (from the gate's a-1 RAG of our own record)

- **gap#1 (open generation)** — NOT mechanism-bound at scale (this session's ceiling: the WKV cortex beats a fair
  bigram 3.35× / a fair trigram +0.811 nats at depth on unseen TinyStories, generates coherent prose). The residual is
  DATA/SCALE (a lever to measure) + broader corpus. The gate's Rank-3 = the ANN generation ceiling + bigram at matched
  token budget — which the running gap#1 scale sweep (`_emerge_wkv_lm_derisk` d256/d512 × data) already covers.
- **gap#2 (learned binder over correlated codes)** — the naive from-scratch coincidence-product bind is a CLOSED
  STRUCTURAL boundary (shallow, deep-eprop, deep-dendritic, AND the BPTT/true-gradient ORACLE all fail held-out; only a
  FIXED self-inverse structure generalizes — `2026-07-14-deep-eprop-CONFIRMED-BOUNDARY`, `2026-06-24-burndown-3B`). BUT
  a DIFFERENT unsupervised structure — **key-addressed fast-weight** — is GO: delta-rule fast-weight bind recalls 1.000
  on correlated fillers, K-sweep 3→12 all 1.000 (`2026-07-17-keystone-2-deltarule-fastweight-bind-RATE-GO`, rate rung);
  SINGLE-bind on-bridge SPIKING store GO 6-seed (`2026-07-15-edge5-rung2`). **The precise residual = the fully-spiking
  MULTI-bind (P≥3) SHARED-store realization** — the additive-STP shared store collapses P1 0.92 → P2 0.11, and the
  on-bridge delta-WRITE error-correction is REFUTED (caps ~2 binds, `2026-07-15-edge5-rung3`). The record says the fix
  is a self-organizing COMPETITIVE SLOT / energy-attractor binder, NOT a store-side write rule.

## The decision (Rank-1 top pick) — the AKOrN/Kuramoto energy binder on the EXISTING RF phasor substrate

External lit (AKOrN, Miyato-Lowe-Geiger ICLR 2025 arXiv:2410.13821 + Resonator Networks, Frady-Sommer 2020) and our
record CONVERGE on the same primitive: **the RF resonate-and-fire neurons ARE unit-magnitude Kuramoto oscillators,
`cp_rf_w_re/im` IS the coupling matrix J, and `rf_resonate` IS the energy fixed-point iteration.** AKOrN's energy
`E = -½ xᵀJx − cᵀx` is Hopfield-form, so **J is set by a LOCAL Hebbian outer-product** of role⊗filler phasors (our
keystone-2 delta/BTSP write) — no backprop, no transport, no supervised loss → binding-by-phase-synchrony as attractor
completion. It **sidesteps BOTH refuted walls by construction**: separate phase-attractor basins have NO summed
superposition to invert (kills the bundling systematicity wall), and completion RIDES on code overlap so it WANTS
correlated codes (kills the self-defeating decorrelation demand). **ONE harness advances THREE of the five gaps:** #2
(the learned J write), #3 multi-referent disambiguation (the phase-cluster biased-competition read), #5 CA3 completion
(the energy descent). **Emergence bar: PASSES** — J self-organizes by a local rule from the unsupervised stream-cortex
codes; the fixed FHRR is retained ONLY as the numeric resonator ceiling to beat (the OFF arm).

## The plan (cheap-first, ceiling-first, ONE variable)

1. **CEILING FIRST (this cycle, numpy):** fixed-FHRR + iterative resonator cleanup, multi-bind retrieve@P=1..6 over the
   788 correlated stream-cortex codes (`bridges/developed/scale787/day_*/grounded_codes.npz`, phasor). Sets the
   best-achievable target (≥0.80 at P≥3). `_gap2_binder_resonator_ceiling.py`.
2. **MECHANISM (next):** ONE VARIABLE — J-write = LOCAL-Hebbian-outer-product ON vs OFF (OFF = fixed FHRR baseline),
   swept P=1..6, read by phase-cluster; GATE = reach the ceiling ≥0.80 at P≥3, 6-seed. Anti-cheats: permuted-role → 0;
   decorrelated-code must NOT help (proves it wants correlated codes); no-write → chance.
3. **Then #3 (multi-referent)** = the phase-cluster WTA read on the SAME attractor (biased-competition, Desimone-Duncan)
   vs the two prior NEGATIVEs (recency, salience) — essentially free once Rank-1 lands.
4. Rank-2 (on-bridge competitive-slot BTSP binder) is GATED behind Rank-1 (may be unnecessary if the RF energy binder
   hits the ceiling with a local J). Rank-5 (developmental self-organization of the conjugate-symmetry wiring) is a
   parallel emergence-bar polish on the fixed binder.

Full ranked de-risks in the workflow output; runners: `_gap2_binder_resonator_ceiling.py` (ceiling), reuse
`rf_phasor_composer.py` + `_phaseB_deltarule_bind_bundled_derisk.py` (the local write) for the mechanism.
