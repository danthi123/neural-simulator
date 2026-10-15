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
1-8 (SUPERSEDED by amendment D below: dev 1-8 is skipped; smoke dev 7 only). Gate seeds 42 43 44 100 101 102
run only through job lines registered against THIS packet after Claude
approves it (gate_seed_guard resolves the gate seeds to the committed prereg; six seeds before any generalization
claim, non-negotiable 6).

## 2b. AMENDMENTS A-D + answers (RULING wrc-packet-review-1014/1014b, wrc-packet-answers-1014, 2026-10-14)

**Order (D), replacing the dev 1-8 stage above: SKIP dev 1-8.** Sequence: smoke at dev seed 7 (peak RSS, wall,
F2/F3 identity, the at-risk list of C, the off-target precondition of A) -> promote this packet to
`research/findings/2026-10-1X-word-role-flip-PREREGISTRATION.md` on main with bar-sanity records -> ASK
[wrc-flip-prereg] -> gate seeds only after that ASK is answered.

**(A) OFF-TARGET set, exact strings, fixed now (10 statement-form turns; >= 8 required):**
"the cat eats fish" / "the cat precedes the ball" / "the ball precedes the worm" / "Sally puts the marble in
the basket." / "Sally leaves the room." / "Anne moves the marble to the box." / "the wolf hunts the deer" /
"the dog and the cat walked in" / "the cat and the dog walked in" / "what do you usually drink?" is NOT in the
set (its parse route engages and declines; it stays the commit packet's no-fact control). Precondition
(measured on the dev-7 smoke, intact-at-ON): NO listed turn may engage the word-role parse (trace route); a
listed turn that does makes the bar UNDEFINED plus a NEEDS, never a silent drop. The bar itself: BOTH lesions
(`BRAIN_WORD_ROLE_LESION=1` PARSE and `BRAIN_WORD_ROLE_COMMIT_LESION=1` COMMIT, run separately) must leave every
listed reply identical to intact-at-ON on **6/6 gate seeds** (reply fields `answer`/`abstained`/`recalled_svo`).

**(B) F2/F3 identity (battery prereg section 5, dev seed 7).** `research/runners/iw1_default_identity.py` and
`research/runners/iw2_default_identity.py` at the flip revision: `default` (no flags set; the flipped anchors)
equals `on:both` (BRAIN_WORD_ROLE_PARSE=1 and BRAIN_WORD_ROLE_COMMIT=1 explicitly) on every script turn (the
default IS the capability), and `off` (both flags `=0`) equals the pre-flip default on every turn (the escape
still reproduces pre-flip behaviour), byte-identical apart from the volatile-key drops those runners already
exclude.

**Clarification (RULING wrc-f3-1015, 2026-10-15):** in this bar `off` means ONLY `BRAIN_WORD_ROLE_PARSE=0` and
`BRAIN_WORD_ROLE_COMMIT=0`, every other flag at its production default — run it as the new `off:word-role` arm
(zeroes only this packet's PACKET_FLAGS; every other flag unset). The runner's all-flags `off` arm is a DIFFERENT
arm: its reply diff at an already-default-ON packet's probe turn (dev-7 seed: iw1 turn 12 via xedge
`_XEDGE_SWM_DEFAULT_ON=True`; iw2 turns 4/9/14/15 via the already-ON false-belief/transitive/neural-selfid
anchors) is expected by construction — report it, never count it. F3 stays: `off:word-role`@F byte-identical to
`default`@P on every script turn.

**(C) Scoped B1.** On the dev-7 smoke, run the battery turn script (iw2 scope/story turns + iw1 turns) ON vs
alloff (both flags `=0`) at the flip revision; every row with a changed reply is AT RISK and is listed here
before any gate seed runs. On the six gate seeds, each at-risk row's n_load_bearing at ON must be >= alloff at
the same revision; a drop reverts the flip (battery prereg section 6). No at-risk rows: B1 is met by the identity
record of (B).

**(Answers) Q2: ONE commit flips both anchors (the commit flag is only read when parse is ON). Q3: the 10-12
battery flip table is NOT amended (pinned at P/R_B); this packet registers its own flip unit, and later battery
re-runs at revisions after the flip inherit the defaults (both rows are in the headline registry). Q4:
confirmed — the only production reader is `brain_chat_tui` (`_neural_question_parse` line 1440, called 1723);
webapp chat goes through ChatBrain, so production chat is the intended blast radius.**

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
