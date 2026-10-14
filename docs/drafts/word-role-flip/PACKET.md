# word-role-flip PACKET: BRAIN_WORD_ROLE_PARSE + BRAIN_WORD_ROLE_COMMIT default ON

**DRAFT for Claude review (NEEDS [wrc-flip-packet]); no data exists at filing and none is queued before Claude
approves this packet.** Filed 2026-10-14 on branch `research/wrc-flip-packet`. Follows the validation battery's
flip procedure (`research/findings/2026-10-12-validation-battery-prereg.md`, sections 4-6) as ruled by
wrc-flip-packet-1014: this is the first LIVE SEAM retirement (the seam named in the confirmed-with-caveats
finding `research/findings/2026-10-14-word-role-commit-gate.md`, "Not yet a live seam").

## 1. What flips (defaults READ from source at main, 2026-10-14)

| flag | current default | anchor | reader |
|---|---|---|---|
| BRAIN_WORD_ROLE_PARSE | **OFF** | `research/runners/brain_chat_tui.py::_WORD_ROLE_PARSE_DEFAULT_ON = False` (line 279) | `_word_role_parse_enabled()` (line 283) |
| BRAIN_WORD_ROLE_COMMIT | **OFF** | `research/runners/brain_chat_tui.py::_WORD_ROLE_COMMIT_DEFAULT_ON = False` (line 315); `_WORD_ROLE_COMMIT_DEFAULT_OP = 0` (line 316) | `_word_role_commit_enabled()` (line 319) |

Both flip together (the commit stage is only read when the parse flag is on; RULING wrc-flip-packet-1014 names
both). `BRAIN_WORD_ROLE_OP` and `BRAIN_WORD_ROLE_COMMIT_OP` defaults (0) are unchanged. An explicit `=0` on either
flag remains the byte-identical escape: with both `=0` production parses with the positional BridgeParser and the
count rule, pre-flip behaviour.

## 2. Bars — fixed here, BEFORE any data

The flip unit of the 2026-10-12 battery prereg section 5, adapted to one faculty pair:

- **F1 at the default.** Both LBF rows (`research/runners/lbf_rows/word_role_parse.py`,
  `research/runners/lbf_rows/word_role_commit.py`) at the flipped default (no REQUIRED_ENV override; production
  defaults): **reply-level load-bearing on 6 of 6 seeds** — in every DEFINED cell the intact reply differs from
  that row's lesion reply (PARSE lesion `BRAIN_WORD_ROLE_LESION=1`, COMMIT lesion `BRAIN_WORD_ROLE_COMMIT_LESION=1`;
  reply fields per docs/TERMS.md `load-bearing`). A DEFINED cell that is not reply-load-bearing fails the seed.
  A cell failing engagement (organ route `word_role_commit`, no error, `commit_lesioned` matching the arm) is
  UNDEFINED and never counts. No FLIP-HURTS-ANSWERS (below) may be present.
- **answer_rate, flag-off vs on (FLIP-HURTS-ANSWERS).** `tools/answer_rate.py` on the paired layout
  `--on-tag wrcflip --off-tag wrcflip-alloff --faculty word-role-commit` (and `--faculty word-role-parse`): the
  on arm runs at the flipped default, the alloff arm forces both flags `=0` at the same revision. `flags:
  FLIP-HURTS-ANSWERS` on ANY measured seed is REPORTED beside the verdict, changes no PASS/FAIL by itself, and
  requires a NEEDS line and a Claude ruling before the flip stands. UNMEASURED seeds are listed, never read as
  "no harm".
- **OFF-TARGET CONTROL (target-specificity).** A pre-registered set of questions that skip the word-role parse
  (no agent/action/patient structure for the organ to commit: factual recall and identity turns from the
  existing no-fact probe plus the battery's off-role turns) must reply IDENTICALLY under the COMMIT lesion at the
  ON default — reply-level fields (`answer`/`abstained`/`recalled_svo`) unchanged on **6/6 seeds** with
  `BRAIN_WORD_ROLE_COMMIT_LESION=1` vs intact-at-ON. Any changed off-target reply fails the bar (it means the
  lesion leaks outside the parse path).
- **Dual-role decline rate (reported, never gated).** On the dual-role probe at the default: reads
  right/wrong/declined per seed (right = role layer's assignment, declined = no commit), plus the exclusion-lesion
  co-commit count. Reported beside the verdict; the 17/24-right/0-wrong/7-declined count-mode-era figure in the
  finding is the comparison point.

Dev seeds first (RULING): smoke one line (dev seed 7) for peak RSS + wall time, then the four bars on dev seeds
1-8. Gate seeds 42 43 44 100 101 102 run only through job lines registered against THIS packet after Claude
approves it (gate_seed_guard resolves the gate seeds to the committed prereg; six seeds before any generalization
claim, non-negotiable 6).

## 3. Job shape

Pool lines, `SIM_BACKEND=numpy`, `mem_gb=8`, `max_age_h` stamped, smoke-first per RULING speedup-smoke-first
(mem_gb=ceil(1.3 x smoke peak); do not launch if projected wall > 24 h - one NEEDS with the numbers). The commit
gate lines' shape (the `host_share_bg_seam --run-all ... --commit` lines of `docs/drafts/bg-commit/jobs/`) is the
template for the on/off arms; the flip bars score the landed reply fields, not the seam tracer. One revision,
both tags, sidecars required.

## 4. Failure handling (battery prereg section 6, adapted)

Any bar failing at the gate stage -> both defaults REVERT to OFF with a finding (one flag may stand alone only if
Claude rules it); no retuning on gate seeds; a missing/UNDEFINED cell is re-run once at the same revision, logged
before its result is read, and still-UNDEFINED cells never count. A FLIP-HURTS-ANSWERS flag needs a Claude ruling
before the flip stands even if every other bar passes.

## 5. What this does not claim

- No result: this document registers the run and its bars; every number comes later, in one RESULT filed after
  the shards exist.
- No reduction of the residuals named in the finding ("Not yet a live seam"): at ON the commit read
  (`spiking_commit.py:181-196`; threshold-invariant, not deciding), the slot -> word mapping, the a==v guard, the
  self-alias resolution and ROLE_MIN folded into W_RG all remain and stay named.
- No host-decision-share or ownership claim (docs/plans/2026-09-25 plan is separate); no scale/latency claim;
  honesty boundary (non-negotiable 5) applies to every reply-level read-out.

## 6. Open questions for Claude

1. Off-target set membership: the exact question list for the OFF-TARGET bar is sketched above (no-fact probe +
   off-role battery turns); confirm or amend before gate seeds.
2. Should the flip land as ONE commit turning both anchors True (my plan), or two commits (parse first, commit
   second) so a mid-flip main stays coherent?
3. Does the next battery re-run pick these defaults up automatically (both rows are already in the headline
   registry), or does it need its flip table amended in place?
4. With the parse flag ON by default, non-chat callers of `_neural_question_parse` (if any) inherit the organ
   parse; confirm the blast radius is the chat path only (grep found the reader used in brain_chat_tui only).
