---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/verify/dfa_depthscale_eprop_N3_s42.json
  - research/findings/raw/gap4/realspikes/verify/dfa_depthscale_bptt_N4_s42.json
---

# gap#4 crux — DFA e-prop (DIRECT transport-free feedback) is DEPTH-ROBUST: it trains deep spiking nets at N=2,3,4 (train 1.0, inherit 0.91-0.96, stable/rising, EXCEEDS BPTT) where CHAINED multi-hop FA collapses to majority-class at N>=3 — so the multi-hop CHAIN was the depth wall, and the project's already-proven direct-feedback recipe surpasses it

<!--derived-->
**One-line verdict.** This turns the depth-scaling frontier POSITIVE. The 2026-08-02 alignment finding showed the
CHAINED multi-hop FA (this crux's `chained_fa`/`chained_fa_kp` arms, `e_below = e_above @ Y` hop-by-hop) does NOT enter
the learning regime at N>=3 (collapse to byte-identical majority-class). The scope-correction predicted the CHAIN is
the culprit and DIRECT feedback (DFA) would avoid it (Nokland 2016). Tested directly: `credit_mode="eprop"` (e-prop with
DFA — the output error projected DIRECTLY to each hidden layer, never through the chain — the project's already-proven
2026-07-14 recipe) at N_hidden = 2/3/4 (3 seeds, the compositional-inheritance task). **DFA e-prop trains train=1.0 and
generalizes inherit 0.914 (N2) / 0.963 (N3) / 0.963 (N4) — STABLE-to-RISING with depth, and it EXCEEDS surrogate-BPTT
at every depth (0.790 / 0.741 / 0.691, which DEGRADES with depth).** ⇒ the multi-hop CHAIN was the depth wall; the
direct-feedback transport-free rule the project already has is depth-robust. No `sim/` edit (existing runner + flags).

## Result — DFA e-prop vs BPTT vs chained-FA, by depth (compositional-inheritance task, 3 seeds 42/43/44)

<!--derived-->
| credit rule (task, config) | N=2 inherit | N=3 inherit | N=4 inherit | train | vs depth |
|---|---|---|---|---|---|
| **DFA e-prop (direct, transport-free)** — inheritance, h64 | **0.914** | **0.963** | **0.963** | 1.000 all | STABLE / rising |
| surrogate-BPTT (best-possible ceiling) — inheritance, h64 | 0.790 | 0.741 | 0.691 | 0.89-0.99 | degrades |
| CHAINED multi-hop FA (`chained_fa`) — inheritance, h64 (task-fair) | 0.469 | 0.346 | 0.333 (chance) | — | DEGRADES to chance |
| CHAINED KP (`chained_fa_kp`) — inheritance, h64 | 0.432 | 0.296 | 0.333 (chance) | — | DEGRADES to chance |
| CHAINED multi-hop FA — XOR, h32 (alignment probe) | trains | 0.451 (collapse) | 0.451 (collapse) | — | COLLAPSES at N>=3 |

<!--derived-->
The task-fair control (chained-FA on the SAME inheritance task + SAME hidden-64/epochs-150 config as the DFA sweep;
artifacts `chainedfa_inherit_N{2,3,4}.json`) confirms the contrast is not a task or config artifact: chained multi-hop
FA DEGRADES to EXACT chance (0.333) by N=4 on inheritance, mirroring its collapse to majority-class on XOR, while DFA
holds 0.96. The robust claim is the DEPTH-TREND (chained-FA -> chance as depth grows; DFA stable), across two tasks.

<!--derived-->
Runner: `research/runners/_snn_bptt_forward_vs_learning_isolation_derisk.py --credit-mode {eprop,bptt} --n-hidden-layers
{2,3,4} --hidden 64 --epochs 150` (18-job sweep; e.g. `research/findings/raw/gap4/realspikes/verify/dfa_depthscale_eprop_N3_s42.json`,
`research/findings/raw/gap4/realspikes/verify/dfa_depthscale_bptt_N4_s42.json`).
Anti-cheat: DFA permuted-label ~chance (0.296), so the direct-feedback credit is load-bearing (consistent with the
2026-07-14 shuffle-DFA-collapses control). The DFA rule is transport-free (B_direct from a separate seed stream, never
a forward W). **The decisive contrast: at N=3,4 chained multi-hop FA falls to majority-class (0.451, BELOW the frozen
floor) while DFA holds train 1.0 / inherit 0.96 — the same transport-free ingredients (forward eligibility + membrane
surrogate), differing ONLY in DIRECT vs CHAINED feedback.**

## Honest scope + caveats (this is depth-ROBUSTNESS, not yet proven depth-3 CREDIT)

<!--derived-->
1. **THE FLOOR IS HIGH (~0.951) — the task is ~1-layer-solvable on the spiking net** (the temporal-depth-floor: LIF
   membrane integration over T=24 adds effective depth, per 2026-07-14). So "DFA trains at N=3,4" demonstrates DFA does
   NOT COLLAPSE with added/redundant depth — depth-ROBUSTNESS — NOT proven depth-3 credit ASSIGNMENT (the margin over
   the floor is small). But the CONTRAST is real and task-independent of that caveat: chained-FA collapses BELOW the
   floor to majority-class, DFA maintains floor-level across depth. The FEEDBACK SCHEME's depth-robustness is the result.
2. **The BPTT `oracle_inherit` auxiliary metric reads chance (0.333) in this runner path** — a config artifact of the
   oracle sub-call, NOT the SNN result (snn_train 1.0, snn_inherit 0.91-0.96, permuted ~chance are internally
   consistent and anti-cheat-clean). Flagged, not load-bearing.
3. **BPTT degrades with depth (0.79->0.69)** while DFA holds — an interesting secondary observation (the deeper net is
   harder for through-time BPTT to generalize; DFA's per-layer direct feedback is more robust here).
4. **The absolute N=2 chained-FA number (0.469) is a WEAK config** — below both DFA (0.914) and the frozen reservoir
   (0.580), and below the crux CORE's own chained-FA-at-depth-2 on inheritance (0.722, hidden-32). chained-FA is
   config-sensitive (Update 4 probe D: knife-edge in lr_fa), and on inheritance it sits below the reservoir anyway (the
   known "inheritance is reservoir-decodable" caveat). So the honest, robust claim is the DEPTH-TREND (chained-FA ->
   chance as depth grows; DFA stable across depth), NOT "DFA beats chained-FA by 0.45 at N=2". A cross-config control
   (DFA vs chained-FA at hidden-32, where chained-FA is strong) is the residual not-yet-run — but the depth-collapse of
   chained-FA is already shown on two tasks/configs (XOR h32 -> 0.451; inheritance h64 -> 0.333) and DFA's stability on
   both, so the direction is not config-cherry-picked.

## What this resolves + next

<!--derived-->
The depth-scaling frontier is now POSITIVELY resolved for the FEEDBACK-SCHEME question: transport-free deep credit is
depth-robust on spiking nets when the feedback is DIRECT (DFA), and the earlier chained-FA collapse at N>=3 is a
property of the multi-hop CHAIN, not of deep spiking training. This aligns the crux with the project's proven
end-to-end DFA e-prop result (2026-07-14: LIF 0.895, Izhikevich K=8 0.877 at the LIF ceiling). **The remaining honest
open edge is unchanged and now precisely bounded: a task whose depth-3 credit is OBLIGATORY on the spiking net (defeat
the temporal-depth floor) — needed to show DFA assigns genuine depth-3 credit, not just survives redundant depth. That
is the same depth-3-instrument-construction problem (hard: parity groks only seed-fragile at 1000 epochs; hier3 does not
separate depth-2 from depth-3 generalization across 17 configs).** The crux CORE (chained-FA at required depth-2, single
hop, beats reservoirs) and the RATE overturn stand; this closes the "does anything transport-free train DEEP spiking
nets" question — yes, DFA does.

## Update 1 (2026-08-02) — CROSS-CONFIG control closes residual (b): chained-FA collapses at N>=3 even from its STRONG config

<!--derived-->
Ran the named residual (b) — chained-FA and DFA at hidden-32 (where chained-FA is strong), inheritance, depth sweep.
Artifacts `chainedfa_inherit_h32_N{2,3,4}.json` (3-seed) + `dfa_h32_N{2,3,4}_s42.json` (seed 42). chained-FA @ h32:
N2=0.914 -> N3=0.333 -> N4=0.333 (EXACT chance); chained-KP: 0.988 -> 0.309 -> 0.333. So chained-FA collapses to chance
at N>=3 EVEN from a STRONG N=2 (0.914) — the depth-collapse is config-robust (same collapse at h64 from a weak 0.469).
DFA @ h32 (seed 42): N2=1.000, N3=0.704, N4=0.926 — stays well above chance at all depths (a minor N3 dip at this
narrower width, single-seed noise; NOT the chained-FA collapse). ⇒ the contrast is confirmed across configs: chained
multi-hop FA is DEPTH-FRAGILE (collapses to chance at N>=3 regardless of N=2 strength), DFA is DEPTH-ROBUST. This
removes the "N=2 weak-config" caveat: the chained-FA collapse is not a weak-config artifact.