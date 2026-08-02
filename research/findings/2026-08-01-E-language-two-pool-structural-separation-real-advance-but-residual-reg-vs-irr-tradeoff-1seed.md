---
type: finding
status: contributing
date: 2026-08-01
artifacts:
  - research/findings/raw/lanes/morph/two_pool_oppoint_map_s42.json
---

# E·Language dual-route morphology — TWO-POOL structural separation is a REAL advance (each route reaches its gate separately) but a residual reg-vs-irr TRADEOFF remains: no single op-point clears both gates; next = the faithful di-synaptic inhibition (1-seed op-point map)

<!--derived-->
**One-line verdict.** This session's two prior morphology negatives both failed on CO-LOCATION (single shared
pool caps regular-rule generalization at reg_acc ~0.25 because rule and store compete in one WTA, and single
pool + whole-form→affix inhibition still caps at ~0.25 because a novel stem spuriously retrieves whole-forms
through the shared recurrent). The record named "genuinely separate pools" as the untested lever; this builds it
(`_productive_morphology_two_pool_derisk.py`: two structurally-isolated excitatory pools, lex + proc, each with
its own FS-WTA, no lex↔proc associative recurrent, blocking via a retrieval-gated lex→proc whole-form→affix
inhibition). Result (1-seed op-point MAP, 12 op-points): **structural separation is a real advance** — regular
generalization reaches **reg_acc 1.0** (vs the single-pool 0.25 cap) and irregular blocking reaches **irr_acc
0.857**, EACH at its own op-point. **But no single op-point clears BOTH gates** (reg≥0.90 AND irr≥0.85):
0 of 12. A residual reg-vs-irr tradeoff, cleanly located. No `sim/` edit (additive runner).

## The op-point map — 1 seed (42)

Artifact: `research/findings/raw/lanes/morph/two_pool_oppoint_map_s42.json` (12 op-points, backend numpy/CPU).

<!--derived-->
| regime | reg_acc | irr_acc | reads |
|---|---|---|---|
| single-pool baseline (both prior negatives) | ~0.25 (cap) | ~0.857 | routes COMPETE in one WTA |
| two-pool, low rule (cyc 40), inhib≥3 | 0.750 | 0.857 | irr clears, reg below gate |
| two-pool, high rule (cyc 60–120), inhib 3 | 1.000 | 0.571–0.714 | **reg clears**, irr below gate |
| two-pool, high rule (cyc 100), high inhib 5–8 | 0.875–1.000 | 0.571–0.714 | raising inhib does NOT restore blocking |

The two levers DECOUPLE the way separation predicts (reg is set by proc rule strength; irr by the blocking
inhibition) — a genuine structural improvement over the single pool, where raising rule strength collapsed
blocking. Regular generalization jumps from the single-pool 0.25 cap to 1.0. BUT the blocking inhibition cannot
be scaled up to counter the strong affix drive that full reg needs: at cyc=100, raising inhib 3→5→8 leaves irr
at 0.571–0.714 (and starts to hurt reg). So the routes are separated, but the BLOCKING does not scale.

## Root cause + the next mechanism (named, not deferred)

<!--derived-->
The blocking is the flagged **Dale-law shortcut**: a sign-inverted EXCITATORY synapse (negative g_e from
excitatory lex neurons onto the affix), not a Dale-compliant GABAergic cell. A single sign-flipped weight cannot
be scaled to overpower a strong affix drive without collateral damage (it also starts suppressing regulars, and
raising it hurts reg). **Next mechanism:** the faithful **di-synaptic feedforward inhibition** — whole-form →
dedicated inhibitory interneuron → affix — which scales the suppression with the retrieved whole-form strength
(a strong irregular whole-form drives strong inhibition; a novel regular, with no stored whole-form, drives
none). That is the honest burn-down of the shortcut AND the mechanism most likely to break the tradeoff. Also
still flagged: the hand-wired routing (which pool each item projects to is host-assigned, S2) → a developmental
self-organized split is the deeper burn-down. This is a 1-seed op-point map (a characterized partial advance,
not a GO); the di-synaptic version + a 6-seed confirmation is the next step.

## Update (2026-08-02) — the di-synaptic burn-down 6-seed = NEGATIVE (the 1-seed GO was a fluke; reg generalization is seed-fragile)

<!--derived-->
Built the faithful di-synaptic inhibition (whole-form(exc) → dedicated GABAergic interneuron → affix; Dale-
compliant, the minus sign in the cell; additive/default-off/byte-identical-when-off; corpus-confirmed untested).
Artifact: `research/findings/raw/lanes/morph/two_pool_disynaptic_6seed.json`. At the seed-42 op-point it cleared
BOTH gates (reg 1.0 + irr 0.857) where the sign-inverted version was 0/12 — but that did NOT hold at 6 seeds:
both-gates **1/6** (only seed 42), because **reg_acc is seed-fragile** (per-seed 1.0 / 0.375 / 0.625 / 0.750 /
0.875 / 0.250). So the di-synaptic inhibition does the right thing to the BLOCKING (irr clears at several seeds,
and reg is untouched by inhibition strength as designed) but the deeper residual is that the **procedural rule
route's generalization itself is unstable across seeds** — the tradeoff-break is real per-op-point but the whole
two-pool is not a robust GO. NEXT: the reg-route seed-fragility (a more robust PAST→AFFIX generalization —
stronger/wider proc pool, or more training) is the real residual, not the blocking. The structural-separation
advance (reg 0.25→up-to-1.0) stands; a robust 6-seed both-gates GO does not.
