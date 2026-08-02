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

## Update (2026-08-02) — the harder XOR task RESOLVES caveat #1: chained_fa BEATS a wide reservoir (+0.150, 6/6) AND EXCEEDS surrogate-BPTT — a stronger positive, though the ELM reservoir still partly decodes XOR so not the STRICT categorical claim

<!--derived-->
Ran the named next test — the SAME LIF SNN bridge on the depth-2 XOR→threshold task (the rate-overturn task,
byte-identical `make_task`; XOR is NOT linearly separable, and the BPTT arm is now properly tuned — hidden 128,
200 epochs — so its ceiling is VALID: it solves the task, held-out 0.782, train 0.99). Artifact:
`bptt_snn_chained_fa_XOR_6seed.json`. 6-seed (chance ~0.524):

| arm (XOR-threshold on the LIF SNN) | 6-seed held-out |
|---|---|
| **chained_fa (transport-free)** | **0.839** (GO 6/6) |
| chained_fa_kp | 0.867 (GO 6/6) |
| **bptt (properly tuned, VALID ceiling)** | 0.782 |
| **frozen-reservoir OPTIMAL WIDE-256** | 0.689 |
| frozen-reservoir OPTIMAL matched-32 | 0.609 |
| permuted | 0.502 (chance) |

**This RESOLVES caveat #1.** On the inheritance task the wide-256 reservoir (0.840) BEAT chained_fa (0.778); on
the non-linearly-decodable XOR task chained_fa (0.839) BEATS the wide-256 optimally-read reservoir (0.689) by
**+0.150, 6/6** — so on a task a reservoir cannot fully solve, directed credit provides a real, robust advantage
a reservoir does not. And **chained_fa (0.839) EXCEEDS the valid surrogate-BPTT ceiling (0.782), 6/6**
(bptt_fraction 1.26) — the transport-free LOCAL rule ≥ the non-local best-possible on this spiking task (chained-
FA's per-step σ′ eligibility is evidently a more robust optimizer here than BPTT's exact-but-harder through-time
credit). directed-over-permuted +0.337, 6/6. **Honest scope — still NOT the STRICT categorical unlock:** the
wide-256 reservoir is a NONLINEAR (ELM-like) random feature map, so it partly decodes XOR (0.689, +0.165 over
chance, `wide_optimal_at_chance` 0/6) rather than sitting at chance — chained_fa clearly beats it (+0.150) but
does not drive it to chance. So the claim is "directed credit robustly beats reservoir computing AND matches/
exceeds BPTT on a non-linearly-decodable spiking task", not "a reservoir is provably at chance". **The gap#4
crux headline is now strong: transport-free deep credit works on a TRAINABLE spiking substrate — beating wide
reservoirs on hard tasks and matching/exceeding surrogate-BPTT (6/6) — the reservoir substrate was the wall.**
Next for the strict-categorical form: a task no NONLINEAR reservoir can decode (deeper composition), + real depth
where KP's rescue widens.

## Update 2 (2026-08-02) — DEPTH scope: the strong result holds at the REQUIRED depth (2); adding REDUNDANT depth (3/4) COLLAPSES both FA and KP below frozen — KP's depth-rescue is UNTESTED on spikes (the XOR sweep is confounded by redundant depth)

<!--derived-->
Ran the depth sweep (N = 2/3/4 hidden layers, XOR task, seed 42) to test whether KP's rate-result depth-rescue
(fixed-FA collapses at depth-4, KP recovers) holds on the trainable LIF SNN. Artifacts:
`bptt_snn_chained_fa_XOR_depth{3,4}_seed42.json`. Result: at N=2 the strong result holds (FA 0.839, KP 0.867);
at **N=3 and N=4 BOTH FA and KP COLLAPSE to 0.451 (identical), BELOW the frozen reservoir (0.546/0.515) —
purchase-over-frozen goes NEGATIVE**, and KP-over-fixed-FA = 0.0 (no rescue). **But this does NOT cleanly test
KP's obligatory-depth rescue** — the confound the build agent flagged and this confirms: the XOR→threshold task
only REQUIRES depth-2 (level-1 pair-XOR → level-2 threshold), so hidden layers 3-4 are REDUNDANT capacity, not
obligatory credit hops. The transport-free credit degrades through the redundant spiking hops (both arms fall
below frozen), and KP doesn't rescue redundant depth. **The clean test of KP's depth-rescue on spikes needs a
task whose REQUIRED depth is 3-4** (the spiking analog of the rate MNIST-depth-4 result), where the deeper layers
are obligatory — untested. **Honest scope correction to the headline:** transport-free deep credit works on a
trainable spiking substrate AT THE REQUIRED DEPTH (depth-2, robust 6/6, beats reservoirs on XOR, matches/exceeds
BPTT); it does NOT yet scale to deeper spiking nets (redundant depth collapses it), and the KP-depth-rescue that
carried the rate result is UNTESTED on spikes. NEXT: a required-depth-3-4 spiking task (obligatory hops) — the
clean KP-depth-rescue test.

## Update 3 (2026-08-02) — the required-depth-3 test via a BOOLEAN-obligatory task is ILL-POSED: stacked parity forces obligatory depth but ALSO defeats the backprop ceiling (the oracle cannot fit deep parity either) — the surpass is a SMOOTH depth-3 task

<!--derived-->
Built the named required-depth-3 test as `--task-nestedxor`: pair-XOR (L1) -> group-MAJORITY (L2) -> top-XOR (L3),
with the MAJORITY inserted between the two XOR levels specifically to break the parity fold (a pure nested-XOR
tree collapses to a wide parity = depth-1; MAJ != parity, so the three ops cannot be folded). The construction is
correct — label perfectly balanced (chance 0.504), a linear read of the raw bits sits at chance (0.490), the
three ops genuinely stack. **But the stage0 depth-genuineness gate FAILS, and the failure is diagnostic, not a
tuning miss — 6/6 seeds (l3_train 0.502-0.512, all at chance; depth3_requiring 0/6).** Artifacts:
`nestedxor_stage0_confirm_seed42.json`, `nestedxor_stage0_confirm_6seed.json` (both under
`research/findings/raw/gap4/realspikes/`).

<!--derived-->
| stage0 oracle (DendriticMLP, hidden 96, 250 epochs, lr 0.3) | held-out | TRAIN |
|---|---|---|
| linear (0 hidden) | 0.490 | 0.506 |
| 1 hidden | 0.496 | 0.502 |
| 2 hidden | 0.466 | 0.514 |
| **3 hidden (the depth regime that SHOULD clear it)** | **0.496 (chance)** | **0.502 (chance)** |

<!--derived-->
**The load-bearing read is the TRAIN column: the depth-3 oracle cannot even FIT the training set (l3_train =
0.502, chance).** This is NOT a generalization failure (which would show high train, chance held-out) — it is an
OPTIMIZATION wall: a 96-wide 3-hidden-layer backprop net trained 250 epochs cannot memorize the nested-XOR
function. Root cause: stacked XOR/parity is the canonical hard case for gradient descent — parity gradients are
near-zero and carry no learning signal (Shalev-Shwartz, Shamir & Shammah 2017, *Failures of Gradient-Based Deep
Learning*, arxiv:1703.07950; the classic parity-hardness result). The nested-XOR function is *representable* at
depth-3 but not *optimizable* by backprop at any depth 0-3.

<!--derived-->
**THE METHODOLOGICAL DELIVERABLE (why this matters beyond one bad task):** a BOOLEAN-obligatory-depth task cannot
test "does KP rescue obligatory depth-3 on spikes," because the very construct that forces obligatory depth
(stacked parity) simultaneously destroys the BACKPROP CEILING the spiking rule must be measured against. With no
depth-3 ceiling (backprop itself at chance), a null result on the spiking chained-FA/KP arms would be
uninterpretable — indistinguishable between "KP fails to rescue" and "the task is unfittable by anyone." The rate
MNIST-depth-4 rescue that carried the rate result worked precisely because natural-image depth is SMOOTH and
optimizable, NOT parity-obligatory. So "obligatory boolean depth" was the wrong operationalization of the depth
test.

<!--derived-->
**THE SURPASS (no-defer — the negative launches the next method):** a SMOOTH depth-3-REQUIRING task where the
depth requirement comes from COMPOSITIONAL BINDING, not parity — a 3-level taxonomy (member -> mid-category ->
super-category -> property, the proven depth-2 semantic-inheritance genre extended one level), which a depth-3
backprop oracle CAN fit+clear (l3_train high, l2 underfits held-out) so the ceiling exists. Building now
(`--task-hier3`). **Scope guard — this does NOT touch the crux's core result:** transport-free deep credit still
works on a trainable spiking substrate at the required depth-2 (6/6, beats reservoirs, matches/exceeds BPTT); this
refines the OPEN edge (does KP's depth-rescue hold on spikes) by correcting a test design, and the correction is
itself a finding about how the depth test must be posed.

## Update 4 (2026-08-02) — ADVERSARIAL-VERIFICATION CORRECTIONS (a 9-agent verify workflow refuted/re-scoped two committed sub-claims; the CORE beats-reservoir leg STRENGTHENED)

<!--derived-->
A 9-agent adversarial-verification workflow (4 probes + skeptics + synthesis; artifact
`research/findings/raw/gap4/realspikes/verify/crux_verification_workflow.json`) re-tested this finding's load-bearing
claims. Two require correction; the core leg strengthened.

<!--derived-->
**⛔ CORRECTION 1 (Update 3's parity claim is REFUTED as stated).** Update 3 (and the sibling finding
`2026-08-02-gap4-crux-transport-free-rule-...-ILL-POSED` framing) declared the nested-XOR depth-3 task "UNDEFINED /
unfittable — a parity OPTIMIZATION wall, no depth-3 ceiling exists" from a 250-epoch stage0. That was UNDER-TRAINING,
not a wall: at the SAME width (96), lr (0.3), batch (128), seed (42), extending 250->1000 epochs takes the depth-3
oracle from chance (0.502) to a PERFECT fit AND perfect held-out generalization (train 1.0, held-out 1.0 on 1434 unseen
disjoint patterns) via an abrupt grokking-style transition (250=0.502, 500=0.520, 750=0.528, 1000=1.000). So the
nested-XOR IS a valid depth-3 ceiling GIVEN a 1000-epoch budget. HONEST residual scope: the grok is SEED-FRAGILE — 1/2
seeds (seed 43 H96/1000ep = 0.519, did not grok) and 0/? at H=256/512 — so it is a marginal, not robust, instrument.
The `nestedxor_stage0_confirm_*.json` self-declared `below_chance=true / UNDEFINED_task_unfittable` verdict is correct
FOR 250 EPOCHS but its word "unfittable" is void — read it as "not-fitted-at-250-epochs (groks by 1000)". The parity
GRADIENTS-vanish citation (Shalev-Shwartz 2017) is a real phenomenon but does NOT bound THIS task at THIS scale.

<!--derived-->
**⛔ CORRECTION 2 (the "exceeds surrogate-BPTT" margin is INFLATED; drop the 6/6 per-seed headline).** Update 1's XOR
result claimed chained_fa (0.839) EXCEEDS the valid BPTT ceiling (0.782) by +0.057, 6/6 (bptt_fraction 1.26). A
BPTT-tuning sweep (probe C, bptt_train=1.000 in every config, so NOT under-tuned) puts the best-tuned BPTT at 0.808
(best single seed 0.822): the like-for-like margin is only **+0.011** (3-seed chained_fa 0.819 vs best-tuned BPTT
0.808), i.e. ~80% of the committed lead was BPTT under-tuning. The DIRECTION survives (chained_fa >= best-tuned BPTT)
but the MAGNITUDE and the 6/6 per-seed dominance do NOT — at the best-tuned BPTT, FA>BPTT is only 2/3 (seed 42: BPTT
0.802 > FA 0.749). ⇒ re-scope the claim to "chained_fa MATCHES / marginally exceeds a well-tuned BPTT (+0.011
like-for-like)"; the 6/6 per-seed headline is WITHDRAWN pending a full 6-seed re-run vs the tuned BPTT (bh256/be400).

<!--derived-->
**CAVEAT 3 (knob sensitivity, probe D).** chained_fa's XOR result is robust DOWNWARD in the FA learning rate (lr_fa=0.02
-> 0.881, even better) and to removing sigma-norm (-> 0.764, still > reservoir 0.689, GO 6/6), but KNIFE-EDGE on the
high side (lr_fa=0.10 -> 0.589 collapse). lr_fa is a load-bearing knob; the committed 0.839 sits on the safe side.

<!--derived-->
**✓ STRENGTHENED (probe A, the CORE leg).** The beats-reservoir claim is the most robust: the optimally-read frozen
reservoir SATURATES in a 0.65-0.73 band across W256->W2048 (0.689/0.655/0.714/0.726), per-doubling gain decelerating to
+0.012, and NO single reservoir seed at any tested width (max 0.760) reaches chained_fa 0.839 — directed transport-free
credit beats an optimally-read nonlinear reservoir of every tested width on XOR. Caveat preserved: the reservoir is a
partial ELM (~0.65-0.73 >> chance 0.524), so this is "width-saturates-below", near-categorical, not "reservoir provably
at chance". (extreme-width CLOSED: W4096=0.757, W8192=0.773 — the optimally-read reservoir saturates and NEVER crosses chained_fa 0.839 up to 32x width.) **Net: the crux CORE direction stands and
its beats-reservoir leg strengthened; the exceeds-BPTT magnitude and the parity-wall sub-claim are corrected here.**
