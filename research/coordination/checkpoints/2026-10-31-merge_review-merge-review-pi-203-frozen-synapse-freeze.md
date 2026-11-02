# Checkpoint: merge-review-pi-203-frozen-synapse-freeze

kind: merge_review
branch: research/pi-203-frozen-synapse-freeze
sha: c81cbf3b090cafeb024c5697134479977bd86e49

## Question

Should branch research/pi-203-frozen-synapse-freeze (3 commits off main 710883be5, HEAD c81cbf3b090cafeb024c5697134479977bd86e49) merge? It adds cfg.enforce_plastic_mask_beyond_hebbian (DEFAULT OFF) and gates six cp_connections.data write sites (synaptic scaling, homeostatic clip, reward clip, reward UPDATE, structural elimination, mask remap/compaction/formation labelling) behind it. Board #203's bar: "frozen weights unchanged over 30 reads, and byte-identical behaviour when the flag is off". Evidence: (a) 72-run battery, frozen delta 0.0 ON vs drift OFF on 6 seeds, positive control population moves in every ON run; (b) 36/36 SHA-identity between the flag-OFF arm and the genuine pre-fix tree (git archive 710883be5), plus determinism repeats; (c) pytest tests/test_plastic_mask_beyond_hebbian.py 15 passed/6 skipped (the 6 skips are pre-existing cupy-required tests in other files); (d) the reply-level arm is a NEGATIVE: on the production comprehension organ the flag changes the reply ZERO times (12 runs, all digests identical) — I report this as a negative, not a comfort, and named the follow-up (which other write paths reach that organ's read). (e) Two MiMo second-opinion passes on the diff: three length hazards found and fixed (compaction mask truncation, remap short-array clamp, fill=True un-freezing existing synapses); the final pass found no merge-blocking defects. Three commits were blocked by gates and are recorded in research/FAILURE_LOG.md. Please rule: merge as-is, or name the specific defect/missing evidence that must land first.

## Bar

Board #203 (research/coordination/PI_LANE.md line 2): "Enforce the freeze in the runtime path that still moves frozen weights, behind a default-OFF flag, with tests: frozen weights unchanged over 30 reads, and byte-identical behaviour when the flag is off. Merge via a checkpoint merge_review."

## Evidence

- /home/dant123/Projects/neural-simulator-pi/research/findings/raw/_pi203_frozen_synapse_residual/branch_vs_main.diff
- /home/dant123/Projects/neural-simulator-pi/research/findings/2026-10-31-plastic-mask-residual-the-six-write-sites-the-hebbian-fix-never-reached.md
- /home/dant123/Projects/neural-simulator-pi/research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json
- /home/dant123/Projects/neural-simulator-pi/research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json
- /home/dant123/Projects/neural-simulator-pi/research/findings/raw/_pi203_frozen_synapse_residual/comprehension_no_regression.json
- /home/dant123/Projects/neural-simulator-pi/tests/test_plastic_mask_beyond_hebbian.py
- /home/dant123/Projects/neural-simulator-pi/tests/_plastic_mask_beyond_hebbian_scenario.py
- /home/dant123/Projects/neural-simulator-pi/research/coordination/PI_LANE.md
- /home/dant123/Projects/neural-simulator-pi/research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md
- /home/dant123/Projects/neural-simulator-pi/research/FAILURE_LOG.md
