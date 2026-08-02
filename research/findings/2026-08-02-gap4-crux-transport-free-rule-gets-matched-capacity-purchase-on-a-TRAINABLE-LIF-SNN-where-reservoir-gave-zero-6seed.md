---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/bptt_snn_chained_fa_6seed.json
---

# gap#4 crux — the transport-free rule gets GENUINE (matched-capacity) directed-credit purchase on a TRAINABLE LIF SNN where the SAME rule got ZERO on the movable-plateau reservoir: the substrate was the wall — but the task is reservoir-decodable given width, so not yet a categorical unlock (6-seed, adversarially verified)

<!--derived-->
**One-line verdict.** This bridges the session's two results. The RATE overturn showed the transport-free rule
(chained fixed-random feedback-alignment + σ′ + graded credit) trains deep credit at rate; the SPIKING terminus
showed it has ZERO purchase on the movable-plateau RESERVOIR substrate (5 controls). This test applies the SAME
rule on a TRAINABLE LIF SNN (`sim/bptt_snn_gpu`, the forward that reached 0.82 via surrogate-BPTT in 2026-07-14):
it gets **genuine directed-credit purchase, 6/6** — chained_fa 0.722 (KP 0.870) vs frozen-reservoir 0.451 vs
permuted 0.333 (chance), directed-over-permuted +0.389. **⇒ the reservoir SUBSTRATE was the wall, not the
transport-free rule.** Adversarially verified (4 skeptics): transport-free confirmed (runtime probe: perturbing
W_in leaves the hidden credit bit-identical, scrambling Y changes it), held-out genuinely disjoint, permuted =
exactly chance on all 6 seeds, forward bit-identical to the 0.82 source. **BUT (the verification's catch, load-
bearing): the clean "unlock" is QUALIFIED — the task is linearly reservoir-decodable given width.** No `sim/`
edit (reuse-by-import of the LIF SNN + BPTT + task; the chained-FA credit is runner-side).

## Result — 6 seeds (42/43/44/100/101/102), the trainable LIF SNN (2 hidden × 32, depth-2 inheritance)

Artifact: `research/findings/raw/gap4/realspikes/bptt_snn_chained_fa_6seed.json` (numpy/CPU).

<!--derived-->
| arm (same LIF SNN forward + task; only the CREDIT differs) | 6-seed inherit held-out |
|---|---|
| **chained_fa (transport-free, fixed-random Y)** | **0.722** (GO 6/6) |
| chained_fa_kp (KP-learned, transport-approximating) | 0.870 (GO 6/6) |
| frozen_reservoir (hidden frozen; output-learning IDENTICAL to chained_fa's) | 0.451 |
| permuted (shuffled labels) | 0.333 = chance (exactly, all 6 seeds) |
| directed-over-permuted / purchase-over-frozen | +0.389 / +0.272 |

The frozen arm's output-update is byte-identical to chained_fa's, so `chained − frozen` isolates EXACTLY the
hidden-layer directed credit. On the movable-plateau RESERVOIR, the SAME rule gave directed ≈ 0 (5 controls,
`2026-08-02-gap4-crux-wall-LOCATED-...`); here it gives +0.389. That contrast is the finding: a TRAINABLE
(plastic-hidden) spiking substrate lets the transport-free rule assign directed credit; the fixed movable-plateau
reservoir did not.

## Adversarial verification — 4 skeptics, and the FOUR load-bearing caveats they earned

<!--derived-->
**Confirmed:** (1) transport-free — the hidden descent reads only fixed-random Y, never W_in (static audit +
runtime probe); (2) task not leaking — held-out disjoint (0 duplicate rows; held members carry unseen random
codes; only the abstract super→property rule is shared = systematic generalization), permuted collapses to
exactly chance; (3) purchase survives a FAIR readout at MATCHED WIDTH — an optimal ridge/logistic read on the
frozen-32 hidden reaches 0.568 (5-fold-CV λ), still below chained_fa 0.778 (+0.21); reading BOTH hidden reps
optimally gives chained 1.000 vs frozen 0.568 (+0.43).
**The FOUR caveats (the finding must carry them or it overclaims):**
1. **THE BIG ONE — the task is reservoir-decodable GIVEN WIDTH.** A 256-wide fixed-random frozen reservoir,
   optimally read, reaches **0.840 — EXCEEDING chained_fa 0.778** (wins 2/3 seeds). So directed credit's
   advantage over the NARROW frozen is partly *denied width*; on this linearly-decodable task it does NOT
   demonstrate something a (wider) reservoir categorically cannot do. This is a matched-capacity advance, NOT a
   categorical unlock.
2. **Magnitude inflated:** ~40–60% of the headline +0.358 purchase-over-frozen is a weak-local-delta-readout
   artifact (frozen local-delta 0.420 → optimal 0.568); the genuine matched-width purchase is ~+0.14 to +0.21.
3. **The BPTT ceiling (0.457) is INVALID** — the BPTT arm was under-tuned (train 0.665; the record's proper
   config gives ~0.82). `bptt_fraction_captured` is meaningless (denominator broken); dropped.
4. **KP is transport-APPROXIMATING, not transport-avoiding** (Akrout/KP: Y tracks Wᵀ in direction). The primary
   headline is the FIXED arm (0.722), which is unambiguously transport-free.

## Honest scope + next

<!--derived-->
A small depth-2 semantic-inheritance task (2 hidden × 32, T=24, numpy/CPU), matched-width. The verified positive:
the transport-free rule assigns GENUINE matched-capacity directed credit on a TRAINABLE spiking substrate where
it gave ZERO on the reservoir — so the exhaustively-characterized spiking wall was the reservoir SUBSTRATE, and a
trainable LIF SNN is the surpass direction (grounded in the field's e-prop/DECOLLE trainable substrates). **NEXT
(to earn a categorical unlock, not just matched-capacity):** (a) a task where inheritance is NOT linearly
reservoir-decodable (so a wide reservoir CANNOT solve it and directed credit is provably load-bearing), and (b) a
width-matched-capacity control (does a reservoir at chained_fa's effective capacity match it?), plus (c) more
hidden layers / real depth (where KP's rescue should widen). The rate overturn + this matched-capacity spiking
purchase together move gap#4 from "transport-free deep credit on spikes is a wall" to "it works on a trainable
substrate at matched capacity; the categorical demonstration needs a non-reservoir-decodable task."
