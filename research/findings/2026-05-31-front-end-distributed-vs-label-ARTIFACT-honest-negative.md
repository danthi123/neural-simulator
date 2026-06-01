# Front-end "distributed-vs-label" was an ARTIFACT (honest negative, scrutiny caught it) — 2026-05-31

## Claim tested
Insight #5 angle: at 28 words where the concept-pool LABEL recognition is ~50% (v17 wall, motor pools
dominate the winner-take-all readout), does the spiking bind/QA on the DISTRIBUTED concept-pool activity
EXCEED the pool-label? If yes, the front-end limit would be partly a READOUT artifact and the effective
conversational vocabulary larger than the label suggests.

## Result + the decisive control
Trained 28-word bridge (concept_pool_demo_v2, 50 events/word; Phase-1 pool-label 13/28 = 46%):
- POOL-LABEL recognition 0.571 (16/28)
- DISTRIBUTED-code bind/QA 1.000

This looked like a breakthrough. It is NOT. The mandatory "scrutinize a PASS harder than a FAIL" control
-- an UNTRAINED bridge (random weights, checkpoint NOT loaded) -- gives:
- POOL-LABEL recognition 0.036 (1/28 = chance, as expected for random weights)
- DISTRIBUTED-code bind/QA **1.000** (SAME as trained!)

## Verdict: ARTIFACT (drive-echo), not learned separability
The distributed-code bind/QA is 1.000 even UNTRAINED, where the learned routing is at chance. So the
metric does NOT measure learned concept separability -- it measures the ORTHOGONAL-DRIVE ECHO: each word
is driven by a distinct orthogonal lang_input pattern, so the captured pool activity is distinct per word
regardless of training (distinct inputs -> distinct outputs even with random weights). The bind/cleanup
trivially separates orthogonally-distinct codes. So "distributed >> label" is an artifact of the input
encoding, NOT a path past the v17 wall.

The HONEST finding: the 28-word recognition limit is REAL. The learned word->concept-pool routing genuinely
fails at 28 words (pool-label 57%, the documented structural-imbalance wall); the high distributed-bind-QA
is not evidence against that -- it is a flawed measurement.

## Broader honest implication (must be stated)
Captured concept codes (denoise64 16-word AND the 28-word codes) are pool activity in response to ORTHOGONAL
lang_input drives, so their separability includes a large drive-echo component, not purely learned semantics.
This does NOT undermine the COMPOSITION result -- the bind/unbind genuinely forms and retrieves structured
representations and GENERALIZES to novel role-filler combinations (8/8 random nonsense sentences; 60/60 in
compose_vsa_demo; multi-seed; adversarial CLEAR). The bind composes whatever separable fillers it is given.
But it refines the honest scope: the system's concepts are distinguished substantially by their orthogonal
INPUT encoding (the project's standard v14 orthogonal_drive_pattern), not (only) by deep learned semantic
representations. The scaling demos at V=64/160/320 used independent random sparse codes (gen_sparse), so
those show the composition handles many distinct codes -- a separate, valid result -- but are not evidence
about learned concept formation either.

## Discipline note
This is exactly the bug-discovery-first / scrutinize-the-PASS / never-overclaim discipline working: a
"too good" 1.000 was NOT reported as a breakthrough; the drive-echo artifact was anticipated, the untrained
control run, and the result is an honest NEGATIVE. The probe's auto-verdict ("real path past the wall") was
WRONG without the control and is corrected here. The real path to larger real-word vocabulary remains the
documented hard front-end problem (learned word->concept routing), unchanged by this test.
