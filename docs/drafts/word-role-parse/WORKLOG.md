# WORKLOG word-role-parse (T20)

## 2026-10-02 07:15 EDT - sections 0-1
- Section 0: worktree research/word-role-parse created from main at PIN=9ea7411eb;
  tests 55 passed in 9.93s (test_word_role_parse + test_s2p_score); pin committed;
  provision already ok (same SHA as T19, all four nodes).
- Section 1: make_jobs sweep -> 8 lines committed; queued 8/8, failures=0 (pool depth 390).
- NEXT: section 2 when all 64 sweep files land in research/findings/raw/_s2p/word-role-parse/9ea7411/sweep/.
2026-10-02 word-role-parse: SELECTED op=0 (dev seeds 1-8); sweep 64/64 + selection committed; O=6/R=12 lines committed. Data + job commits this wake.
2026-10-02 word-role-parse: canary O PASS (pool41, 6 O queued); canary R INCONCLUSIVE x2 (mem_ok 8G refused pool42 + pool41) - 12 R held, NEEDS added. Sections 4-5 blocked.

## 2026-10-05 02:29 -- section 4: R canary PASS recorded; R lines queued

- Fresh attempts per RULING canary-node/canary-auto: 8 launches across wspool+pool40 hit two infra failures (rssh launch false-negative; the line's own mem_ok refused while twin canaries held RAM). The transitive R canary nevertheless ran to completion on wspool: status rc=0 after 1596 s -> recorded PASS (module+revision covers both packets: same runner, same sha).
- Queue adds this wake: transitive ok=12 fail=0, word-role ok=12 fail=0 (12 expected each) with the HANDOFF --checked note.
- fb-spiking-witness canary not re-run: its step R already scored PASS at 40c43d520.
- Trigger: results wake 2026-10-05 01:42 (22 files).

## 2026-10-06 00:5x — section 5 scored

2026-10-06 word-role-parse: O PASS, R PASS at op 0 (real handler). Finding: research/findings/2026-10-06-word-role-parse-gate.md.

- Data: commit 99906f6fc (O jsons, 12 LB shards, verdicts + gate-sanctioned .prov siblings; the scorer does not
  emit verdict sidecars — sibling shape follows the transitive-spiking-reach precedent).
- Scoring: `tools/s2p_score.py O ... --pin 9ea7411eb... --out .../verdicts/O.json` -> PASS op=0 rc=0;
  `tools/s2p_score.py R --tag s2p-R-wr-9ea7411 ... --out .../verdicts/R.json` -> PASS op=0 rc=0.
  O: n_all_pass=6, n_defined=6. R: n_reply_load_bearing=6/6 (regressed on all six seeds), control
  word-role-nofact-ctl 6/6 control-ok, both preconditions ok, at_default=false.
- verify-go (RULING verify-go, two MiMo Pro refuters): lens 1 control-integrity/gate-cheat -> SURVIVES
  (independent re-score byte-identical; control holds; selection op=0 first passing index). Lens 2
  reproducibility/instrument-trust/claim-precision -> SURVIVES (sha256-identical re-score; pin audit
  PINS.tsv+verdicts+shards match; prereg section 6 verbatim; TERMS.md reply-level usage; default-OFF confirmed
  `_WORD_ROLE_PARSE_DEFAULT_ON = False`). Verdict: GO under PROVISIONAL banner (Claude confirm+merge pending).

## 2026-10-06 00:5x — section 5 scored

2026-10-06 word-role-parse: O PASS, R PASS at op 0 (real handler). Finding: research/findings/2026-10-06-word-role-parse-gate.md.

- Data: commit 99906f6fc (O jsons, 12 LB shards, verdicts + gate-sanctioned .prov siblings; the scorer does not
  emit verdict sidecars — sibling shape follows the transitive-spiking-reach precedent).
- Scoring: `tools/s2p_score.py O ... --pin 9ea7411eb... --out .../verdicts/O.json` -> PASS op=0 rc=0;
  `tools/s2p_score.py R --tag s2p-R-wr-9ea7411 ... --out .../verdicts/R.json` -> PASS op=0 rc=0.
  O: n_all_pass=6, n_defined=6. R: n_reply_load_bearing=6/6 (regressed on all six seeds), control
  word-role-nofact-ctl 6/6 control-ok, both preconditions ok, at_default=false.
- verify-go (RULING verify-go, two MiMo Pro refuters): lens 1 control-integrity/gate-cheat -> SURVIVES
  (independent re-score byte-identical; control holds; selection op=0 first passing index). Lens 2
  reproducibility/instrument-trust/claim-precision -> SURVIVES (sha256-identical re-score; pin audit
  PINS.tsv+verdicts+shards match; prereg section 6 verbatim; TERMS.md reply-level usage; default-OFF confirmed
  `_WORD_ROLE_PARSE_DEFAULT_ON = False`). Verdict: GO under PROVISIONAL banner (Claude confirm+merge pending).

## 2026-10-06 00:5x — section 5 scored

2026-10-06 word-role-parse: O PASS, R PASS at op 0 (real handler). Finding: research/findings/2026-10-06-word-role-parse-gate.md.

- Data: commit 99906f6fc (O jsons, 12 LB shards, verdicts + gate-sanctioned .prov siblings; the scorer does not
  emit verdict sidecars — sibling shape follows the transitive-spiking-reach precedent).
- Scoring: `tools/s2p_score.py O ... --pin 9ea7411eb... --out .../verdicts/O.json` -> PASS op=0 rc=0;
  `tools/s2p_score.py R --tag s2p-R-wr-9ea7411 ... --out .../verdicts/R.json` -> PASS op=0 rc=0.
  O: n_all_pass=6, n_defined=6. R: n_reply_load_bearing=6/6 (regressed on all six seeds), control
  word-role-nofact-ctl 6/6 control-ok, both preconditions ok, at_default=false.
- verify-go (RULING verify-go, two MiMo Pro refuters): lens 1 control-integrity/gate-cheat -> SURVIVES
  (independent re-score byte-identical; control holds; selection op=0 first passing index). Lens 2
  reproducibility/instrument-trust/claim-precision -> SURVIVES (sha256-identical re-score; pin audit
  PINS.tsv+verdicts+shards match; prereg section 6 verbatim; TERMS.md reply-level usage; default-OFF confirmed
  `_WORD_ROLE_PARSE_DEFAULT_ON = False`). Verdict: GO under PROVISIONAL banner (Claude confirm+merge pending).
