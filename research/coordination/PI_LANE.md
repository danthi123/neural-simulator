# Pi lane (pilot, from 2026-10-31)
- [x] Board #203: a synapse marked non-plastic (plastic=False / plastic_internal=False) still drifts from use. Research the record first (the finding 2026-09-02-read-isolation-audit-29runner-followup-map-plus-a-bigger-adjacent-bug.md and the board text; an earlier fix may cover part of it). Enforce the freeze in the runtime path that still moves frozen weights, behind a default-OFF flag, with tests: frozen weights unchanged over 30 reads, and byte-identical behaviour when the flag is off. Merge via a checkpoint merge_review.
  - DONE 2026-10-31, branch `research/pi-203-frozen-synapse-freeze` (HEAD `fd42e0120`, 4 commits off main
    `710883be5`). merge_review APPROVED by Claude
    (`research/coordination/checkpoints/2026-10-31-merge_review-merge-review-pi-203-frozen-synapse-freeze.VERDICT.md`):
    merge as-is, flag stays DEFAULT OFF, no capability/reply-level claim, cupy confirmation before any default flip.
    Push from the pi box is NOT possible (no remote credentials: `tools/push_both.sh` returns FAIL/unverified) -- the
    branch must be published by the server, which publishes research/* branches within 15 min.
  - Six further write sites + the dead
    `cp_plastic_mask` remap + deferred compaction + formation labelling, all behind
    `cfg.enforce_plastic_mask_beyond_hebbian` (DEFAULT OFF). Evidence:
    `research/findings/2026-10-31-plastic-mask-residual-the-six-write-sites-the-hebbian-fix-never-reached.md`,
    `research/findings/raw/_pi203_frozen_synapse_residual/verification_record.json` (72 runs; frozen delta 0.0 ON,
    drift OFF; OFF SHA-identical to pre-fix `710883be5` 36/36, recorded in
    `research/findings/raw/_pi203_frozen_synapse_residual/byte_identity_record.json`) and
    `research/findings/raw/_pi203_frozen_synapse_residual/comprehension_no_regression.json` (12 runs; the flag
    changes the comprehension reply ZERO times — reported as a negative). Next: the owner flip decision per
    `research/findings/2026-09-24-plastic-mask-flip-PREREGISTRATION.md`, and pinning WHICH write sites reach the
    comprehension organ's read.
- [ ] Board #203 follow-up: the merged freeze (dc6abf989, flag enforce_plastic_mask_beyond_hebbian) changes the comprehension organ's reply zero times, yet the original drift on that organ was measured (13.8 -> 56.1). Find which write path(s) move the organ's frozen weights that the six fixed sites do not cover: instrument every cp_connections.data write (and any other weight store the organ reads) during the comprehension battery, on dev seeds only, flag ON; report each path with its measured delta before changing code. Then gate the path(s) found behind the same flag, default OFF, with a frozen-flat test and flag-off byte identity, and request a checkpoint merge_review. Research the record first (the 2026-10-31 finding of #203 and the 2026-09-02 read-isolation audit).
