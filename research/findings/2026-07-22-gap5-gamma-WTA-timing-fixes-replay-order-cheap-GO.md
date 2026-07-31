---
type: finding
status: live
date: 2026-07-22
mechanism: gamma-wta-replay
---

# gap#5 TIMING — cheapest-first isolation GO: a gamma-WTA + post-fire silence turns RANK 2's marginal weight-only replay order into a RELIABLE forward order, on the REAL learned weights

**2026-07-22, CPU/numpy, coexisting with the fluency training.** The cheapest-first de-risk named in
`2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS...` (NO `sim/` edit): feed RANK 2's EXISTING learned
forward-chain weights through a numpy gamma-WTA + post-fire silence and test whether phase-timing fixes the replay order,
before committing any bridge build. `research/runners/_gap5_gamma_wta_replay_derisk.py`: run RANK 2's proven encode
(n_mem=3, `--rank1-encode` + within-refresh), extract the between-assembly transition matrix W, then replay as a
gamma-cycle sequence of winners from assembly 0 (drive[j] = W[current][j] + noise; winner fires; the current is reset out
of the slot).

## Result (seed 42, valid controls) — GO
| ARM_A (weight-only, no self-avoid) | ARM_B (gamma-WTA + post-fire silence) | SCRAMBLE (per-trial) | NO-ENCODE |
|---|---|---|---|
| **0.500** (chance) | **1.000 forward (full 1.000)** | **0.505** | **0.492** |

- The gamma-WTA + post-fire silence turns the marginal weight-only order (0.500 = chance, the analogue of RANK 2's 4/6)
  into a **reliable 1.000 forward replay**.
- **Both proper controls collapse to chance:** a PER-TRIAL scramble of the off-diagonal W (0.505) and a NO-ENCODE bridge
  (0.492) ⇒ the forward order comes from the LEARNED chain, NOT from self-avoidance imposing an arbitrary permutation.
  (The earlier single-shuffle scramble was INVALID — one lucky shuffle preserved the order; caught + fixed before claiming.)

## Mechanism (from the extracted W) — why it works, and why RANK 2's weight-only path did not
W structure: within~173.7, **adjacent-forward 143.3, adjacent-reverse 142.0 (asym only +1.26), skip-forward 22.0.** The
chain forms STRONG adjacent forward links (143) and WEAK skips (22); the forward-vs-reverse asymmetry (+1.26) is tiny and
swamped by noise (=8) — which is exactly why RANK 2's weight-only replay was only 4/6 (the order rode a marginal
asymmetry). Gamma-WTA + self-avoidance does NOT use that marginal asymmetry: from A the strongest link is A->B (143 >>
skip 22) so it fires B; the gamma reset then SILENCES A (self-avoidance), so from B the strongest REMAINING link is B->C
-> reliable A->B->C. The gamma reset DECOUPLES "hold this memory" from "push to the next" and forbids backward, turning a
marginal graded difference into a deterministic order.

## Implication
Validates the deep-research's promotion of phase-organized timing to a first-class item, cheaply and on the REAL learned
weights: the timing mechanism is the single fix for BOTH open ordered-replay threads (RANK 2 uniform order + RANK 3
recombination), and ~90% of the spiking parts already exist. NEXT (RUNG 2, a thin additive default-off `sim/` driver):
literal temporal theta/gamma over the CA3 slice — a gamma FS pool for the E%-max WTA + a post-fire reset that silences
the just-fired assembly, over RANK 2's existing BTSP chain + RANK 1 within-attractors. Honest scope: this is a RATE-level
isolation on the real learned W (grounds the mechanism); the on-spikes RUNG-2 realization is the build it justifies.
Multi-seed (43/44) confirming. `research/findings/raw/gap5_r4/gamma_wta_seed42_v2.log`.

## Multi-seed confirm (42/43/44) — 3/3 GO, and the DECISIVE evidence: it works even when the raw asymmetry is REVERSE
Seeds 43/44: ARM_B(gamma-WTA+silence)=**1.000** forward, SCRAMBLE(per-trial)=0.505/0.500, NO-ENCODE=0.568/0.535
(chance). **Crucially, seeds 43 and 44 have a NEGATIVE forward/reverse asymmetry** (asym −1.01, −1.56 — the weights
lean REVERSE), yet the gamma-WTA STILL produces perfect forward order. This is the strongest evidence for the mechanism:
gamma-WTA + self-avoidance does NOT use the marginal fwd/rev asymmetry (which was RANK 2's fragile 4/6 crutch, and is
here even reverse-signed) — it rides the robust ADJACENT-vs-SKIP structure (adj ~130-143 >> skip ~22-23) and forbids
backward via self-avoidance. ⇒ phase-timing fixes the replay order EVEN WHERE the learned weight asymmetry points the
wrong way. 3/3 GO. `research/findings/raw/gap5_r4/gamma_wta_{seed42_v2,2seed}.log`.
