---
type: preregistration
status: preregistration
date: 2026-10-06
packet: word-role-commit
runner: research/runners/word_role_commit_organ.py
scorer: tools/s2p_score.py
---
# word-role-commit: a spiking commit stage decides the word-role parse (PRE-REGISTRATION, 2026-10-06)

Written by Claude (branch `claude-review/spiking-commit`) before any gate-seed run. Plan (mechanism, biology,
application order): `docs/drafts/spiking-commit/PLAN.md`, application A1. HANDOFF: `docs/drafts/spiking-commit/HANDOFF.md`.
Rulings answered: `word-role-verify-go` and `word-role-credit` (credit stays PARTIAL until a spiking all-or-none commit
replaces ROLE_MIN / margin and the per-role argmax).

bar-sanity: docs/drafts/spiking-commit/bar_sanity_record_a1_organ.json
bar-sanity: docs/drafts/spiking-commit/bar_sanity_record_a1_reply.json

## 1. The host decision being moved

With `BRAIN_WORD_ROLE_PARSE=1`, `WordRoleParserOrgan.parse` reads role-ensemble spike counts and a host rule decides:
a word gets role r iff r's count >= ROLE_MIN and >= ROLE_MARGIN x the runner-up (`_rule`), and each role goes to the
word with the highest count for it. That rule is what drops an unheard word (finding
`2026-10-06-word-role-parse-gate.md`, caveat 1).

## 2. The mechanism (code at the pinned revision; default OFF behind `BRAIN_WORD_ROLE_COMMIT`)

The organ is built with `commit=` params (`COMMIT_GRID` row; its role layer is `GRID[BASE_OP]`). Per role, on the
same bridge: a commit gate (12 neurons, driven by the role ensemble, self-recurrent, with an inhibitory pool that
inhibits the other two gates) and a role-slot latch (20 neurons, driven by its gate, self-recurrent strongly enough
to sustain itself). Each word is presented for TEST_STEPS, then HOLD_STEPS of zero input; between words every neuron
is reset except the latches. A latch is ON iff its spike count over the last READ_STEPS of the hold reaches
READ_FRAC x NS x READ_STEPS. A word's role is the latch that turned on during it (one: that role; none: no role;
more: co-commit, no role); a role already ON cannot turn on again (the first word to fill it keeps it).
`_rule` is not called on this path (it is computed for a paired record only). Wiring: `_neural_question_parse` with
`BRAIN_WORD_ROLE_PARSE=1` and `BRAIN_WORD_ROLE_COMMIT=1` calls `parse_from_brain(..., commit_op=, commit_lesion=)`;
the reply's `word_role_trace` records `route: word_role_commit`, `commit_lesioned`, `commit_op` and the commit read.

Lesion `BRAIN_WORD_ROLE_COMMIT_LESION=1`: the role -> commit-gate transmission gate is closed. The role ensembles fire
exactly as in the intact arm; nothing commits; the parse declines and the route abstains.

Declared residuals: training stimuli transcribed from the composer's fact index; self-alias -> 'brain' resolution and
the stop-word strip; the word-form -> assembly dictionary; the latch read (a level read of spike counts, registered
threshold-invariant below); the inter-word reset sparing the latches (a compression of the inter-word interval); the
organ is its own bridge (one-brain debt); all commit weights wired, not learned.

## 3. Step 1: operating point (dev seeds 1-8 only; no verdict)

Every COMMIT_GRID row (0-5) runs the commit battery on dev seeds 1-8 (`--sweep`). Selection by
`tools/s2p_score.py select --packet word-role-commit` only: the FIRST row in grid order with `all_pass` on all 8. None:
the method is banked and NEEDS-CLAUDE follows. Dev record before registration (Claude, numpy on the server, no gate
weight): row 0 8/8, latched minimum 77-84 slot spikes, unlatched maximum 0, every read invariant; rows 1-2 8/8; row 3
7/8 (seed 3 'swapped': the role layer gives "chase" at position 0 six action spikes, which the count rule would also
decline); row 4 7/8; row 5 5/8. Dual-role words cat/fish read right in 18 of 32 dev readings at row 0 (none wrong).

## 4. Step O: the organ gate (gate seeds 42/43/44/100/101/102, the selected row)

Battery (`word_role_commit_organ.CASES`) on BATTERY_FACTS: svo (dog, chase) -> (dog, chase); self (brain, use) ->
(brain, use); adverb (brain, usually, use) -> (brain, use); swapped (chase, dog) -> (dog, chase); novel (zorp, glim)
-> decline; lesion_self (word -> role cut) -> decline; commit_lesion_self (role -> commit cut) -> decline;
latch_lesion_self (latch recurrence cut) -> decline. Per seed `all_pass` = every case right AND every case's read
gives the same parse at READ_FRAC x0.5 and x2. **Bar: PASS iff `all_pass` on 6/6 gate seeds**, cell validity as in
`tools/s2p_score.py` step O (provenance sidecar at the pin, op_index = the selection, prereg = this file).
Reported, never gated: dual-role reads; the exclusion-lesion co-commit count; the count rule on the same run's role
counts (`host_rule`, `diagnostics.host_rule_agrees`); latched minimum and unlatched maximum; co-commit and re-latch
totals.

## 5. Step R: reply-level gate (real handler, LB shards, the same six seeds)

Rows `word-role-commit-selfid` (probe "what do you usually use?": intact states the built-in self fact, 'spikes';
the commit lesion abstains) and the control `word-role-commit-nofact-ctl` ("what do you usually drink?": intact
abstains, the lesion leaves it), in `research/runners/lbf_rows/word_role_commit.py`. Both arms set
`BRAIN_NEURAL_SELFID=1`, `BRAIN_WORD_ROLE_PARSE=1`, `BRAIN_WORD_ROLE_COMMIT=1` and `BRAIN_WORD_ROLE_COMMIT_OP=<selected>`.
**Bar: PASS iff the capability cell is reply-level load-bearing (`regressed`) on 6/6 AND the control is `control-ok`
on 6/6**, under every rule of `tools/s2p_score.py` step R (iw2 S2, degenerate-intact, intact-wrong, lesion-kept,
operating-point sidecar check) AND engagement: every intact and lesion arm's `word_role_trace` has route
`word_role_commit`, no error, and `commit_lesioned` matching the arm (`PACKETS["word-role-commit"]["engagement"]`).
A cell failing engagement is UNDEFINED.

Host-rule baseline (reported, not gated): the word-role-parse R data at 9ea7411 (count mode, same probe and seeds,
PASS 6/6) and each commit-mode reply's `word_role_trace.commit.host_rule_roles`.

## 6. Outcome -> verdict

| step 1 | O | R | verdict |
|---|---|---|---|
| NONE | - | - | method banked (no row passes dev 1-8); NEEDS-CLAUDE |
| SELECTED | PASS | PASS | PASS |
| SELECTED | PASS | FAIL | NO-GO (reply level) |
| SELECTED | FAIL with a failing case | not run | NO-GO (organ level; the case is named) |
| SELECTED | FAIL with every case right but a read not invariant | not run | NO-GO (the read owns part of the decision) |
| any | UNDEFINED | UNDEFINED | no verdict: re-queue the missing cells once, then NEEDS-CLAUDE |

R runs only after O reads PASS.

## 7. What a PASS lets the finding say (fixed sentences)

"With the commit flag on, each word's role in the question is committed by spiking neurons: a role gate that bursts
for at most one role, and a role latch that holds itself after the word ends. The host count rule and argmax no
longer decide the parse; the read of which latch is on does not change when its level is halved or doubled. Cutting
only the commit stage, with the role ensembles firing as before, removes the answer." Credit for word-role-parse
becomes FULL only after a flip packet makes `BRAIN_WORD_ROLE_PARSE` and `BRAIN_WORD_ROLE_COMMIT` the default and its
F1 at the default passes with no FLIP-HURTS-ANSWERS; the residuals in section 2 stay named.

## 8. Seeds, cost, files

Dev seeds 1-8 (step 1); gate seeds 42, 43, 44, 100, 101, 102. Step 1: 8 pool lines (about 2 s per row and seed).
Step O: 6 pool lines. Step R: 12 LB shards, `mem_gb=8`. Outputs under `research/findings/raw/_s2p/word-role-commit/<P7>/`
and `research/findings/raw/_load_bearing/_shards/s2p-R-wc-<P7>/`. Job lines: `docs/drafts/s2p/make_jobs.sh
word-role-commit {sweep|O|R} <PIN> [<OP>]`.
