# Dedicated adversarial review of the integrated compositional capability = CLEAR; the multi-seed PASS is genuine; the integrated compositional retrieval capability is a validated milestone and is recorded in capability_status.json

## Status

The integrated compositional capability (multi-seed PASS, findings
`2026-05-22-INTEGRATED-compositional-capability-multi-seed-PASS-...`,
commit `aee3707`) is a load-bearing positive result. Per the project's
standing discipline -- scrutinise a nominal PASS HARDER than a FAIL --
a dedicated, independent adversarial reviewer was tasked to run the
exploit-class probes before any capability claim. This records the
review outcome.

## The review

An independent reviewer (fresh agent, full tool access, no
controller context) read the load-bearing files
(`spiking_phasor_fhrr.py`, `spiking_phasor_integration.py`, the three
cheap-first probes, the result JSON, the findings doc) and RAN the
exploit probes rather than reading only.

## Probes and findings

1. **Integrator-neuron genuineness (the prime suspect).** The
   `phase_sum_neuron` claims to be a genuine time-stepped p/q
   integrator (Orchard Algorithm 1) and has a fallback
   `out = where(fired, out, first+second)`. The concern: if the
   integrator never fires, the fallback (relabeled arithmetic) does
   all the work and the "neuron" is inert. The reviewer RAN it: the
   integrator branch fires for 512/512 dimensions on typical inputs;
   the fallback is never reached; integrator output matches the true
   phase sum within the one-step threshold-crossing delay. Genuine
   time-stepped integrator, not relabeled arithmetic.

2. **Symbol / answer leakage.** `pool_symbol` is drawn from `rng(seed)`
   BEFORE any fact is sampled; facts use a separate `qrng(seed+1)`.
   Symbols cannot encode the facts. The `hit` scoring uses the
   ground-truth `target_pool` only as the scoring oracle -- it does
   not feed the pipeline. No leak.

3. **Recognition load-bearing.** `word_symbol` is keyed on the
   substrate's RECOGNISED pool (`top_pool`), so a misrecognised word
   genuinely gets the wrong symbol. Verified against the JSON: seed
   42 (12/12 task words recognised) -> integrated 1.000; seeds 43/44
   (fewer recognised) -> integrated drops exactly where misrecognition
   lands. Recognition is genuinely in the loop. Live recognition
   reproduced (seed 42, 15/16, real 8440-neuron GPU bridge).

4. **Artifact / cherry-pick.** The clean-up is a genuine 4-way choice
   over the 4 filler words (chance 0.25; result 0.96-0.99).
   `composition-only` correctly restricts to facts whose words were
   all correctly recognised; `integrated <= composition-only` holds
   in all 9 cells. Not degenerate, not cherry-picked.

5. **Abstention moat.** `cleanup()` returns abstain (-1) below
   threshold; groundable vs ungroundable separate cleanly.

6. **Protected-set + autograd.** `git diff e8a99a2..HEAD` over `sim/`
   and `abstention_gate.py` is EMPTY; no `torch`/autograd in any
   shipped file; the subsystem self-test re-runs PASS.

## Verdict: CLEAR

The integration PASS is genuine. The reviewer found no defect; the
findings doc's caveats (two-system architecture, identity-level
interface, recognition-bounded, biology-inspired-engineering phasor
neurons) are stated honestly.

## Propagation

Per the discipline -- a load-bearing PASS, multi-seed, adversarially
reviewed CLEAR -- the integrated compositional retrieval capability is
recorded as a validated pillar in `webapp/capability_status.json`
(as_of bumped to 2026-05-22) with the honest two-system framing in
the metric text. Discipline checks at propagation: capability_status
schema 6/6 green; no-confabulation moat 7/7 green; protected set
byte-empty diff vs `e8a99a2`.

## Honest standing

This is a genuine, validated, adversarially-reviewed milestone: the
project's first working compositional capability. It is NOT a unified
biology-grounded substrate and NOT fluent language -- those framings
are held honestly. The next arcs (recorded in AUTONOMOUS_STATE):
activity-level integration (the substrate's neural activity feeding
the phasor layer directly, rather than an identity-level lookup),
and scaling beyond the small-load task.

## Files / evidence

- Reviewed: `research/runners/spiking_phasor_fhrr.py`,
  `research/findings/raw/spiking_phasor_integration.py`, the three
  cheap-first probes, `spiking_phasor_integration.json`.
- Capability pillar: `webapp/capability_status.json`.
