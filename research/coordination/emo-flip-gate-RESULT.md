# emo-flip-gate RESULT (2026-10-27/28, gate revision F=`3b0b5182a`)

Scored by `python -m research.runners.emo_flip_score --pin <F>` (Amendment 1's combined read);
artifact `research/findings/raw/_emo_bounded_flip/gate_score_3b0b5182aa1819d1804ef21d704154132d75e5bc.json`
(provenance stamped, run_id 1793159784-401057). Data: emo__ matrix 70/70 json (7 seeds x 5 arms x on/off,
incl dev s7), lb shards on+off s42-s44/s100-s102; 72/72 gate lines drained; all paired.

| bar | verdict | reading |
|---|---|---|
| B1 | PASS | n_correct 6 of 6 defined seeds |
| B2 | PASS | lesion_affect n_lesion_effect 6, n_reproduced 6 of 6 (attributable_to 100% each seed) |
| B3 | PASS | n_permuted_tracks 6 of 6 |
| B4 | PASS | 30 ON shards, 150 feel reads: 0 violations, 0 undefined |
| B5 | PASS | OFF intact named categories = 10-26 gate on 6 of 6 seeds, 0 mismatch |
| B6 | PASS | 30 pairs: 240 non-feel whole-record + 150 feel pre-clause compares, 0 diffs |
| B7 | PASS | flip_check 30 pairs (FEEL-NOT-APPEND-ONLY flag not read here: B6's), answer_rate paired OK 6 seeds |
| B8 | **UNDEFINED** | ON row: load-bearing 0 of 6 (need 6/6); OFF beside it 5 of 6. All 6 ON cells fail the pin: sidecars carry stray BRAIN_* env key `BRAIN_EMOTION_CATEGORY_BOUNDED` |

**bars DEFINED: 7 of 8. DEC = UNDEFINED -> STOP with the data. Default NOT flipped (stays OFF).**

## Why B8 is undefined (deterministic, not transient)

`tools/lb_shard.py` `cell_prov_fails()` (line 271) checks `lb.json.prov.json` with
`allow_brain_keys=set()` - NO BRAIN_* env key may appear - and arm sidecars with
`PROV_ARM_ALLOWED_BRAIN_KEYS` (no BOUNDED). But the prereg's B8-ON row *is* the treatment: it must run with
`BRAIN_EMOTION_CATEGORY_BOUNDED=1` in env. Everything else in the pin passes on every cell (git_sha == F in
full, source_kind == git_archive, both source_manifest verifications true, SIM_BACKEND numpy) - the only
failure is the env-key allowlist. The DEC's remedy (re-run each UNDEFINED cell once at the same revision,
then rescore) is deterministic-futile: the same required env reappears, the same refusal returns, a
still-UNDEFINED cell never counts. Fixing the allowlist (e.g. admit BOUNDED for the ON tag's cells) is a
tools/scorer edit = Claude's (section 2); no re-run was spent. `--pin` reading and engagement checks were
never reached on the ON cells (UNDEFINED precedes them); the OFF row reads 5 of 6 load-bearing (one cell
not load-bearing is a data result, not a pin failure - see score json).

Posted as NEEDS `[emo-flip-gate]` RESULT. Prereg: `research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md`.
