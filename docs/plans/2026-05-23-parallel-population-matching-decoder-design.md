# Parallel-population-matching order-bearing decoder for biologized spiking mode-unification: design

## Status

Design, pre-registered. The biology-grounded alternative to the FHRR-
biologization arc's TPAM attractor for order-bearing identification
in the mode-unification arc. Justified by the biologized-spiking
mode-unification NEGATIVE_ORDER_INVARIANT_ONLY result (TPAM crosses
its V=8-V=20 capacity window at the 32-concept tier) and the
built-in diagnostic showing argmax-of-phase-similarities to substrate-
grounded vocab symbols recovers per-slot items at multi-seed 1.000
on the same data.

## Background

Mode-unification's biologized spiking implementation tested both
readouts on the same encoded code C. Order-invariant (marginal-sum
of per-slot phase-similarities) PASSed at multi-seed 1.000. Order-
bearing via the FHRR-biologization arc's TPAM attractor MISSed
(multi-seed 0.53/0.34/0.09 at loads {2,3,5}). The diagnostic
localised the failure: the per-slot unbinds are clean (top-1 by
phase-similarity is always the correct item); the TPAM attractor
converges to a spurious mass-attractor on 32 grounded symbols
(V=8-V=20 capacity window mapped subsequently). A simpler order-
bearing decoder — per-slot argmax over `phase_similarity(unbind_k,
grounded[w])` for w in the substrate's grounded vocabulary —
gives 1.000 across all tested loads/seeds on the same data.

The owner authorised this build as the focused (b) follow-up,
explicitly preceding the generative-replay (c) build since (c)
depends on order-bearing mode-unification for the PFC compositional
frame.

## The honest oracle-adjacency framing (front and centre)

The FHRR-biologization arc's shortcut 3 critique was that the
ORIGINAL composition layer used "argmax over an explicitly stored
vocabulary table" — an engineered hand-supplied table without
substrate grounding. The TPAM attractor was the biology-faithful
replacement: a recurrent Hopfield-class network whose fixed points
are the stored vocabulary patterns. The TPAM IS biologically
grounded (cortical attractor networks; Amit & Treves 1989).

Parallel-population-matching is a DIFFERENT biology-grounded
mechanism — feedforward similarity comparison across a population
of neurons each tuned to one stored concept, followed by lateral-
inhibition winner-take-all. The output is "the concept whose tuned
neuron fires hardest." This is what real cortex implements when
many concept-specific neurons compete (mountains of literature on
cortical lateral inhibition / Mexican-hat WTA / population vector
decoding).

The oracle-adjacency concern: parallel matching IS structurally
close to "argmax over a stored vocabulary." The KEY distinction
that keeps it biology-grounded rather than engineered: the
"vocabulary" in this implementation is the substrate's own derived
grounded symbols — mean-centred consolidated activity → fixed-seed
deriver → spike-phase representation. These symbols are derived
from the substrate's per-concept activity (validated by the FHRR-
biologization arc); they are NOT a hand-supplied engineered table.
The "argmax" is the parallel-population WTA biology naturally
implements. Comparing the unbind output to each substrate-derived
concept symbol via phase-similarity is a biological operation
(dendritic integration); picking the maximum is a biological
operation (lateral inhibition WTA).

So parallel matching is biology-grounded in a different way than
TPAM:
- TPAM: recurrent attractor settling; biology-faithful via Hopfield-
  class dynamics; has a per-vocab-size capacity ceiling.
- Parallel matching: feedforward similarity + WTA; biology-faithful
  via lateral inhibition; scales with vocab without the attractor
  ceiling.

Both are honest biologizations with different scaling properties.
This runner tests the second alternative head-to-head against the
pre-registered TPAM result.

## The build, structurally

What is reused, byte-unchanged:
- All substrate machinery (`train_substrate`,
  `capture_concept_activity`, `_save_cache`, `_load_cache`).
- All FHRR-biologization arc's primitives (`ResonateFireFHRR.encode`/
  `query`, `phases_to_spikes`, `phase_similarity`, `make_deriver`).
- The 160-ensemble bridgeA_nouns trained-substrate cache.
- Task 1 helper `gamma_slot_positions`.
- The frozen 0.80 compositional bar; the K=16 PASS recipe.

What changes from the pre-registered mode-unification runner:
- The ORDER-BEARING decoder is now per-slot
  `recovered_k = argmax(phase_similarity(unbinds[k], grounded[w])
                         for w in vocab)`.
- The ORDER-INVARIANT decoder is unchanged (marginal-sum of
  per-slot phase-similarities, top-K).
- Both readouts STILL share the same encoded C and same per-slot
  unbinds.

No TPAM. No attractor. No new substrate code.

## Pre-registered test (fixed; never tuned)

Same substrate (bridgeA_nouns, 32 concepts, K=16 PASS recipe). Same
multi-seed {42, 43, 44}. Same loads {2, 3, 5}. Same 200 trials per
load per seed. Same gamma-slot positions (deterministic per seed).
Same encoding (sum_k bind(grounded[item_k], position_k) via
ResonateFireFHRR.encode). Same frozen 0.80 bar.

ONLY the order-bearing decoder differs. The order-invariant decoder
is identical to the pre-registered mode-unification runner's (which
PASSed at 1.000).

PRE-REGISTERED reading:
- **MODE_UNIFICATION_BIOLOGIZED_PASS_VIA_PARALLEL_MATCHING**: BOTH
  readouts multi-seed-mean >= 0.80 at every load. The biologized
  mode-unification's both-readouts capability is realised on the
  project's substrate via parallel-population-matching identification
  (a biology-grounded alternative to the TPAM attractor with
  different scaling properties). Subject to a fresh dedicated
  adversarial review before any capability claim.
- **NEGATIVE_PARALLEL_MATCHING_INSUFFICIENT**: either readout
  misses the bar. Surprising given the diagnostic; would require
  investigation of why pre-registered run diverges from diagnostic
  prediction.

## Soundness considerations

A PASS at this runner must survive these adversarial checks before
being claimed:

1. **No oracle leak.** The "vocabulary" the decoder argmaxes over
   is the substrate's own grounded symbols (`grounded[w] for w in
   words`), where `words` is the bridge's full 32-concept vocab
   from `g20_vocab_spec` and `grounded` is derived from mean-centred
   consolidated activity via the fixed-seed deriver. NOT a hand-
   supplied engineered table.

2. **The true items NEVER index the decoder.** The decoder uses
   `phase_similarity(unbinds[k], grounded[w]) for w in words` —
   the comparison set is the FULL vocabulary, never restricted to
   the true items.

3. **The decoder is the ONLY change.** Encoding, unbinding, gamma-
   slot positions, grounded symbol derivation, substrate, training
   — all reused byte-unchanged.

4. **Both readouts on the SAME C.** Same encoding, same per-slot
   unbinds shared by both decoders. The mode-unification claim's
   "one code" is preserved.

5. **The frozen 0.80 bar is unchanged.**

6. **No automatic differentiation.**

7. **Smell-test on PASS HARDER than FAIL.** Could degenerate runs
   produce 1.000? The diagnostic at full scale already shows simple-
   OB = 1.000 with this exact decoder on the SAME cache. A
   degenerate runner that bypassed the unbind would not produce
   1.000 — it would produce chance. The 1.000 reflects the
   substrate-derived grounded symbols' separability under phase-
   similarity comparison.

8. **The TPAM-comparison framing is preserved.** The mode-
   unification BOUNDARY pillar (n=92) from the pre-registered TPAM
   runner stands; this is an ALTERNATIVE biologization, not a
   replacement of the TPAM. The capability_status documentation
   should record BOTH biologizations and the comparison (TPAM has
   V=8-V=20 capacity window; parallel matching scales past V=32
   with the trade-off of oracle-adjacency).

## Implementation outline (TDD plan to follow separately)

Task 0: grounding pin (same constants, frozen bar, K=16 recipe; new
runner module exists; runner re-uses bridgeA_nouns substrate cache;
red until Task 2).

Task 1: no new helper needed — `gamma_slot_positions` already exists
from the mode-unification arc; the decoder is a one-line argmax that
can be inlined.

Task 2: the runner
`research/findings/raw/biologized_spiking_mode_unification_parallel_matching_runner.py`.
Mirrors the structure of the pre-registered mode-unification runner;
ONLY the order-bearing decoder changes (per-slot argmax of
`phase_similarity(unbinds[k], grounded[w])` over the full vocab).
Both readouts share the same encoded C + same per-slot unbinds.

Task 3: soundness tests at the structural level. Pin: same decoder
recipe constants (K=16, M_OBS=16, gamma slots 7, vocab 32); both
readouts share the SAME C structurally (verify by code inspection
via Task 4 reviewer); the decoder uses `grounded[w] for w in words`
(full vocab, not restricted to true items).

Task 4: dedicated adversarial reviewer. Re-runs the 11 exploit-class
checks from the pre-registered mode-unification arc plus a new
check: confirm the parallel-matching decoder is biology-grounded
(substrate-derived symbols; full-vocab argmax; no engineered table)
and the oracle-adjacency caveat is recorded in the runner header.

Task 5: controller-only decisive run. CPU-only (substrate cache
exists from the mode-unification arc; pipeline is pure CPU).
Estimated ~minutes total. Mandatory smell-test recompute. On PASS:
fresh dedicated adversarial review before the capability_status
pillar claim.

## Honest scope

A focused biology-grounded alternative to the TPAM identification
mechanism, with the same disciplined arc the FHRR-biologization +
mode-unification arcs used. Whatever the verdict, it's a clean
biology-translatable comparison of two biologized identification
mechanisms (recurrent attractor vs feedforward parallel matching),
each with different scaling properties and biological trade-offs.

A PASS here completes the biologization of mode-unification's both-
readouts capability on the project's substrate — the prerequisite
for the (c) generative replay arc that would build the
conversational loop on top. The honest oracle-adjacency caveat is
recorded up front; the capability claim is "biologized mode-
unification with parallel-population-matching identification,"
explicitly distinct from the TPAM-based identification (which has
its own validated scope at V=8-V=20 for fact composition).

Frozen 0.80 bar never tuned. Reuse-by-import only. No protected,
frozen, or moat module modified. No automatic differentiation. The
no-confab moat must stay 7/7 green throughout.
