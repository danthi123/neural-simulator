# Pi lane (pilot, from 2026-10-31)
- [x] Board #203: a synapse marked non-plastic (plastic=False / plastic_internal=False) still drifts from use. Research the record first (the finding 2026-09-02-read-isolation-audit-29runner-followup-map-plus-a-bigger-adjacent-bug.md and the board text; an earlier fix may cover part of it). Enforce the freeze in the runtime path that still moves frozen weights, behind a default-OFF flag, with tests: frozen weights unchanged over 30 reads, and byte-identical behaviour when the flag is off. Merge via a checkpoint merge_review.
  - DONE 2026-10-31, branch `research/pi-203-frozen-synapse-freeze`. Six further write sites + the dead
    `cp_plastic_mask` remap + deferred compaction + formation labelling, all behind
    `cfg.enforce_plastic_mask_beyond_hebbian` (DEFAULT OFF). Evidence:
    `research/findings/2026-10-31-plastic-mask-residual-the-six-write-sites-the-hebbian-fix-never-reached.md`,
    `research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json` (72 runs; frozen delta 0.0 ON,
    drift OFF; OFF SHA-identical to pre-fix `710883be5` 36/36) and
    `research/findings/raw/_pi203_frozen_synapse_residual/comprehension_no_regression.json` (12 runs; the flag
    changes the comprehension reply ZERO times — reported as a negative). Next: the owner flip decision per
    `research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md`, and pinning WHICH write sites reach the
    comprehension organ's read.
