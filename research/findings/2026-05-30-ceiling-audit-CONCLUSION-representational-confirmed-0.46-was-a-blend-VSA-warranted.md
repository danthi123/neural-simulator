# Conversational-ceiling audit, CONCLUSION (Phase 2 decisive): the 6th-arc compositional ceiling is REPRESENTATIONAL, not a readout limit. No held-out decoder (cosine / linear / NN, on the lang_output OR the pool-firing state) recovers the episode-specific binding above chance on held-out episodes. AND the headline "0.46" is a DIRECT+COMPOSITIONAL BLEND -- compositional-only is ~chance. The premise behind phase-coded vector-symbolic composition is CONFIRMED (with an honest framing correction); the big VSA arc is warranted.

**Date:** 2026-05-30
**Status:** CONCLUSION of the owner-chosen "audit the ceiling" direction. Decisive, multi-seed, pre-registered-verdict, leakage-guarded. The phase-coded VSA arc is now justified by a verified (not assumed) premise. Honest correction of my own Phase-1 optimism included.

## The decisive test + result

Question: is the 6th-arc compositional ceiling REPRESENTATIONAL (the composed neural state genuinely does not carry the episode-specific noun->adj binding) or a READOUT LIMIT (a richer held-out decoder recovers the binding the spiking cosine readout misses)?

Probe (`research/findings/raw/_ceiling_audit_phase2_decode.py`, throwaway analysis script): reuse the EXACT 6th-arc FULL-arm machinery (generative_replay_pfc_frame_runner + unified_per_regime_monitor + run_concept_replay_phase + the validated full-scale Phase-1 cache `unified_per_regime/phase1/seed{42,43,44}.simstate.h5`); per compositional query capture the composed lang_output state (primary) + the pool-firing state (secondary) + the cosine readout decision + the true answer; train HELD-OUT linear (sklearn LogisticRegression) + nearest-neighbour decoders with EPISODE-LEVEL GroupKFold (train/test never share an episode -> the decoder must generalize, not memorize; different (noun,adj) pair-sets per episode); compare decoder held-out accuracy (B) vs the cosine readout (A) on identical held-out sets. Decoder is offline analysis (sklearn/numpy), NOT a learning rule in the sim, NOT autograd.

Decisive run: 3 seeds (42/43/44) x 8 episodes = 24 episodes, 120 compositional query instances, 4 answer classes, chance 0.25.

| quantity | value |
|---|---|
| compositional cosine readout (answer-subspace, best-case fair) A | **0.24** (~chance) |
| compositional cosine readout (full-vocab) | 0.04 |
| held-out LINEAR decoder B_lin | 0.21 |
| held-out NN decoder B_nn | 0.204 |
| **B_best** | **0.21** (~chance) |
| secondary pool-firing-state decoder (lin / nn) | 0.218 / 0.208 (~chance) |
| chance | 0.25 |
| **pre-registered verdict** | **REPRESENTATIONAL-CEILING-CONFIRMED** (B_best 0.21 < A 0.24 + 0.10) |

## Scrutiny (why this is not an artifact)

- NOT leakage: episode-level GroupKFold; decoders at chance. Leakage would push the decoder ABOVE chance (memorizing episode-specific patterns); it is at/below chance.
- NOT an underpowered decoder: the SECONDARY pool-firing state is only 16-dim and WELL-sampled (96 train samples >> 16 features), so it is not the 2048-dim lang_output being underdetermined -- yet it ALSO decodes at chance. If composition were present in either state (linearly or via NN), at least one decoder would beat chance. None does.
- Real regime: STEP B confirmed the probe reproduces the genuine 6th-arc substrate (runner-blend full_acc 0.4545, in the 0.35-0.55 band); the real generative_replay_pfc_frame_runner reproduces (full_acc 0.40 N2, 0.4583 N3) -- NOT a regression.
- The result is the CONSERVATIVE one (it confirms the prior premise), so it does not carry the false-PASS risk a surprising READOUT-LIMIT would; the scrutiny above is the artifact check, and it passes.

## Honest framing correction (my own Phase-1 optimism)

Phase 1 (earlier today) reported "composition is decodably represented at ~0.46" and called the "not representable" framing OVERSTATED. STEP B + this decisive run CORRECT that: the 6th-arc `full_acc 0.458` is a BLEND of easy DIRECT-retrieval queries (high accuracy) + hard COMPOSITIONAL queries; the COMPOSITIONAL-only readout is ~CHANCE (0.04 full-vocab, 0.24 answer-subspace), not 0.46. So composition in the current representation is essentially at chance -- the original "representation is the bottleneck" framing was RIGHT for composition specifically; my Phase-1 reframe over-read the blended number. The Phase-1 finding's pipeline-conflation observation still stands (SPEAR firing-rate@650 vs 6th-arc cosine@0.1977 ARE different pipelines), but its optimistic "composition decodable at 0.46" claim is hereby corrected to "the 0.46 is a direct+compositional blend; compositional-only is ~chance and -- per this decisive probe -- not decodable above chance by any held-out decoder."

## Conclusion + what it means for the next arc

The conversational-composition ceiling is REPRESENTATIONAL: across 8+ dynamics-gating arcs the composed neural state never carried the binding in a decodable form, and a trained held-out decoder confirms the binding is not recoverable from the state (not cosine, not linear, not NN, not from lang_output, not from pool firing). Dynamics-gating cannot fix this because the representation, not the readout, is the bottleneck. This CONFIRMS (now by a verified premise, not an assumed one) that the prescribed phase-coded vector-symbolic-architecture arc (Orchard spiking-phasor FHRR: the shared theta-gamma rhythm CARRIES compositional content as spike phase, so the composed state becomes a structured DECODABLE object) is the warranted next big build.

The audit did its job: it (1) corrected a headline-number framing conflation (Phase 1), (2) discovered the "0.46" is a direct+compositional blend with compositional-only at chance (STEP B), and (3) decisively confirmed the ceiling is representational not readout-limited (Phase 2). The big arc is now justified by evidence, which is exactly why the owner chose to audit before building.

## Discipline

No protected/frozen/moat/sim/runner/compose module modified (only the throwaway probe script; controller verified git status). No bars moved. Pre-registered verdict honored. Decoder is offline analysis, no autograd in the sim. Honest self-correction of Phase-1 optimism recorded. Phase-coded VSA arc NOT started in this audit -- the audit's job was to gate it, and it now passes the gate.
