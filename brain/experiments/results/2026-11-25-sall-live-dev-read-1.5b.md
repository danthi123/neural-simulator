# Arm S read: live DEV K-P life with `--key S_all --key-bar slept`: KILLED (P2s, collapse guard)

Prereg `brain/research/2026-11-25-sall-live-dev-prereg.md` (a0da1a034 + amendment a3586ecb5). 3 DEV seeds
(319/320/321), AWS g5.xlarge A10G fp16, code a0da1a034, paired with lifeKP (same seeds, job line, AMI). DEV: never a
claim. A verdict on this switch on this life, not on the capability (episodic recall / memory selection).

## Validity
- `brain_kw` = `{sleep_seed, keep_bank: prior, key: S_all, key_bar: slept}` on all 3 seeds: valid.
- `life_costs --dev` G11 setup check PASS on 3/3 (both-off rows match lifeKP). lifeKP lacks torch/cuda provenance: a
  note, per amendment (a).
- `where_ran`: 6/6 files "unknown" (the tool ignores the life JSON `host` field, HANDOFF item 5). The S files record
  host `ip-172-31-*`, `cuda fp16`, `NVIDIA A10G`, code a0da1a034; lifeKP the same AMI and instance type.

## Bars (count.py of the 11-24 read, nights 5-15, trunk)
| | intact told | hippocampus-off told | hurt | helped | intact echo | h-off echo |
|---|---|---|---|---|---|---|
| lifeKP pooled | 123/223 | 138/223 | 41 | 26 | 71/264 | 77/552 |
| arm S pooled | **125/223** | 133/223 | 25 | 17 | 64/264 | **111/552** |
| S s319 / s320 / s321 | 35 / 42 / 48 | 46 / 46 / 41 | 12 / 11 / 2 | 1 / 7 / 9 | | |
| KP s319 / s320 / s321 | 58 / 36 / 29 | 54 / 39 / 45 | 6 / 14 / 21 | 10 / 11 / 5 | | |

- **P1 told** (>= 138 pooled and >= KP on 2/3 seeds): FAIL, 125 < 138 (per seed 2/3 above KP).
- **P2s memory helps**: FAIL on every part. Intact 125 < hippocampus-off 133; helped 17 < hurt 25; intact-frozen
  right-episode fetches 18/223 < lifeKP 45/223 (`frozen_fetch.py`, reproduces lifeKP's 45 exactly).
- **Guards**: H recall 133 >= 115 PASS; intact echo 64 <= 78 PASS; **h-off echo 111 > 84 FAIL**; leak pooled 15/460 all rows,
  15/434 non-refusal (0.035; lifeKP's 17/436 is the non-refusal form) <= 0.069 PASS; lang_guard KL n15 / KP n15 = 0.87 / 1.07 / 0.96 PASS; G7 pooled 35/90 (0.39) <= 0.82 PASS;
  **G8a worst arm per seed FAIL on s319 (intact 0.144 vs KP worst 0.087 + 0.03) and s321 (h-off 0.232 vs 0.137 + 0.03)**
  (the s321 failure is in the hippocampus-off arm, which the key does not touch directly; applied as written);
  **collapse (top reply >= 0.4) seeds-ever: S intact 2 / h-off 2 / frozen 2 / consolidated-off 2 vs KP 1 / 2 / 1 / 0
  FAIL** (`collapse.py`); stuck lived-day runs not computed (the verdict does not depend on it).
- **Decision: KILLED** (P2s fails; the collapse guard fails). Pooled intact 125 is not below K-P's 123, so the third
  kill clause does not fire.

## What happened (report-only)
- **Fetch.** Intact-frozen told rows, right / wrong / none: lifeKP 45 / 141 / 37, arm S 18 / 78 / 127. The offline
  screen 2 predicted this (S_all with the slept bar fetched nothing on 128 told rows); live it fetched nothing on 127.
- **The wrong fetches are mostly one episode.** On intact told rows S fetched the brain's own self-name episode
  ("User: What's your name? / Assistant: My name is Nova") on 24 / 20 / 21 rows (s319/320/321; lifeKP 0 / 0 / 0). On s319 nine
  replies then say the name ("What is my sister's name?" -> "Your sister's name is Nova."; n15 "What is my favourite
  colour?" -> "My name is Nova."). The episode is permanent (strength 4.0 at n10, 5.0 at n15, s319). Two untested hypotheses: the all-layer key
  matches the question form ("What is my ...?"), and told episodes decay below the slept bar. The report-only
  strength tables from `episodes.json` are not computed in this read.
- **Transcript, s319 whole (sonnet reader, all 4828 lines; KP compared at summary level only).**
  - Refusal runs on intact n6 ("I'm not sure." for 9 told items) and n10 ("I'm sorry, I don't know what you do for work").
  - Lived-day loops: L3 6 turns, L9 5 turns, L11 "Oh, I'm sorry. I didn't know that." to 20 tells in a row.
  - The reader's whole-file refusal-phrase count (S 181 vs KP 114) did not reproduce (the skeptic's grep: 258 vs 144),
    so it is not used. Intact-probe abstentions: S 25/894 vs KP 23/894 pooled (s319 24 vs 3).
- **Per-seed swing.** Intact told moved -23 / +6 / +19 against lifeKP on the same seeds, so one-seed moves of this size
  sit inside the life's own variance. Only the pooled bars are read.
- **Scorer notes from the reader**, not acted on here:
  - First-person "My sister's name is Marguerite." is scored wrong (by design: the recall-voice count).
  - Two same-text replies are scored differently across probe kinds (neardup vs sameday).
  - The two points above are worth a separate check of `strict_scorer` consistency.

## Skeptic (sonnet, fresh context, REFUTER-LENS block)
"KILLED stands." Every table number reproduced. Four wording fixes were applied above: the leak denominators, the
question-form and decay hypotheses marked untested, and the refusal count dropped. Lenses: (a) the flag engaged
(`brain_kw`; fetch none 37 -> 127); (b) not a default candidate, abstentions flat pooled; (c) where_ran unknown=6, and
the life JSON host fields are AWS for all 6, so validity does not depend on the host.

## Next method (not a rescue of this one)
Routing ceiling: S_all ranks the right episode first on 168/217 told rows, so the key carries the information. The
gate fails, and it fails in two ways:
- the told episodes decay below the slept bar;
- a permanent identity episode wins on question form.

Candidates for the routing arc:
- exclude permanent and self episodes from the S_all competition, or key them separately;
- gate on rank margin rather than absolute strength.

Each needs its own prereg and the before-build check. Arm S is not rerun.
