---
type: preregistration
status: preregistered
date: 2026-09-29
lane: integration (owner 2026-09-29, integration-first; docs/plans/2026-09-29-acceleration-report.md M1)
mechanism: integration wave 1 -- the shared steps of three default-flip packets (d6-hebbian-write, wm-focus-bind,
  xedge-surprise-worldmodel): a pre-flip byte-identity check, a flip applied by a script, and a production-default
  no-regression battery at the flipped revision F against the reference battery B2b, with attribution re-runs
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed in the commit that adds the anchors and rows it governs, before any gate-seed
  job exists. No result is claimed here. Seed 7 (dev) runs are reported in the packet HANDOFFs and carry no weight.
---

# Integration wave 1: shared steps and the no-regression battery (pre-registration)

## Why

The owner's unit of progress (2026-09-29) is a faculty that is on by default and whose lesion changes the reply.
The flip bar (memory `feedback_flip_validated_fixes_without_waiting`, 2026-09-23) has four legs: a 6-seed GO, a
SOUND adversarial review, no regression in a combined battery with the change on, and a production-default
validation run. Each packet prereg covers its own reply-level gate (step R). This document fixes the steps the three
packets share, so they can be run together at one revision and scored by one committed scorer (`tools/iw1_score.py`).

Each packet's candidate, evidence and reply-level probe are in its own prereg:
`2026-09-29-iw1-d6-hebbian-write-flip-PREREGISTRATION.md`, `2026-09-29-iw1-wm-focus-bind-flip-PREREGISTRATION.md`,
`2026-09-29-iw1-xedge-surprise-worldmodel-flip-PREREGISTRATION.md`.

## Revisions

- **P** (pre-flip pin): the head of branch `research/integration-wave1` when step 0 of the shared HANDOFF provisions
  it, and only if `origin/main` is an ancestor of it (the provisioner's stale-source guard). If main has moved, the
  HANDOFF merges `origin/main` into the branch first; P is then that merge commit. Every anchor reads False at P.
- **F** (flip pin): the commit on the same branch made by `tools/integration_wave1_flip.py apply <packets>`, for
  exactly the packets whose steps R and D passed at P. Nothing else changes between P and F.
- After a revert (decision table), **F'** is the next such commit; every F rule then applies to F'.

## Step D: byte-identity at P, one dev seed (seed 7, real handler, `research/runners/iw1_default_identity.py`)

Five fresh-process arms of one 13-turn script (two sessions: the focus intro in both orders): `default`, `off`, `on:d6-hebbian-write`,
`on:wm-focus-bind`, `on:xedge-surprise-worldmodel`. Per-turn sha256 of the response body (timing keys dropped).

- **D0 (byte-identical-off).** `default` equals `off` on all 13 turns. Required for every packet.
- **D_p (the script reaches p's flag).** `on:p` differs from `default` on at least one turn. Required for packet p;
  a vacuous `on:p` (no difference) is UNDEFINED for p, not a pass.

## Step B: the no-regression battery, production default at F

One sharded load-bearing battery at F, `iw1-flip-<F7>`: `tools/lb_shard.py jobs --no-fixes` (the adequate probe set,
no fix flag, no BRAIN_* flag), numpy, six gate seeds, one shard per registry faculty and seed, every job line prefixed
by the guard `tools/assert_iw1_defaults.py --expect-on=<the flipped packets>` (it refuses a revision whose anchors
differ from the expected set, and any BRAIN_* variable in the job environment). At F the lifted rows
(`d6-hebbian-write`, `xedge-surprise-worldmodel`) are in the registry, so the battery also measures them at the
production default.

The comparison is the 2026-09-23 flipdefaults design
(`2026-09-23-flip-validated-fixes-production-default-validation-prereg.md`): the reference is the most recent
COMPLETE production-default battery, B2b (`b2b0924-base` at a308f1e09, 216/216 coverable cells DEFINED), and a drop is
attributed by re-running the dropped faculty at F with the flips forced off. A base battery at P was considered and
not chosen: it doubles the ~270-shard cost, and the attribution run answers the same question for the faculties that
drop. Scored by `python tools/iw1_score.py B ...` and `... X ...` (cell validity = `tools/b2d_score.py` A1.2 rules
1-3 and 7; S0 = the registered trace verdict; S2 = reply level, `answer`/`abstained`/`recalled_svo`, per
docs/TERMS.md). Criteria:

- **B1 (no regression, both levels).** For every faculty in both batteries, its number of load-bearing seeds at F is
  not lower than in B2b, at S0 and at S2 -- OR, for a faculty that is lower, its attribution run
  `iw1-alloff-<F7>` (its six cells at F with every flipped packet's flags `=0`) is also at or below the F count at
  both levels, which means the drop was not caused by the flips (code merged since a308f1e09 is the candidate). A
  faculty present only in B2b is reported, not scored.
- **B2 (completeness).** Every coverable cell of `iw1-flip-<F7>` is DEFINED after the re-run rule below.
- **F1 (the new rows at the production default).** For each flipped packet that has a row
  (`d6-hebbian-write`, `xedge-surprise-worldmodel`), the row is reply-level load-bearing on 6 of 6 seeds in
  `iw1-flip-<F7>`.

## Step F: production-default identity at F, dev seed 7

`iw1_default_identity.py` arms at F: `default`, `off`, `on:<all flipped packets joined by +>`.

- **F2 (the default IS the capability).** `default@F` equals `on:<flipped>@F` on all 13 turns.
- **F3 (the escape still works).** `off@F` equals `default@P` (from step D) on all 13 turns.

## Decision table (applied by the HANDOFF, no judgment)

| outcome | action |
|---|---|
| packet p: R_p PASS and D0, D_p PASS | p enters the flip at F |
| R_p FAIL (a DEFINED seed not reply-load-bearing) | p is not flipped; NO-GO finding for p's reply-level integration |
| R_p or D_p UNDEFINED after the re-run rule | p is not flipped; NEEDS-CLAUDE |
| D0 FAIL | no packet is flipped; NEEDS-CLAUDE (an anchor edit changed the default path) |
| B1, B2, F1, F2, F3 all PASS | the flips stand on the branch; NEEDS-CLAUDE: adversarial review + merge |
| faculty X lower than B2b and its all-off run is higher than F at either level (a flip caused the drop) | bisect: re-run X's six cells at F with ONE flipped packet's flags forced `=0` per run; the packet whose `=0` run is higher than F is reverted (flip script), F' is committed, step B and step F are re-run at F' |
| the bisect shows no single packet, or several | revert every flipped packet; NEEDS-CLAUDE (an interaction) |
| F1 FAIL for p | revert p; re-run step B's flip battery and step F at F' |
| F2 or F3 FAIL | revert every flipped packet; NEEDS-CLAUDE |
| any criterion UNDEFINED after the re-run rule | nothing merges; NEEDS-CLAUDE |

## UNDEFINED and the re-run rule

A cell that is not DEFINED (missing or unreadable arm, sidecar not pinned to the revision, `null-noise`,
`lesion-noise`, an organ error on the probe turn, UNRELIABLE) is re-run once, at the same revision, with the job
line unchanged, and the re-run is logged in `research/packets/integration-wave1/reruns.tsv` before its result is
read. A cell that is still not DEFINED stays UNDEFINED. An UNDEFINED cell never counts as load-bearing and never as
a pass of B1; a criterion that depends on it is UNDEFINED.

## Not measured here

- **Latency.** Reported per turn by step D (`turn_wall_s`), not gated (non-negotiable 4, speed is secondary).
- **The Qwen mouth.** Every arm uses the stub renderer, as every battery does.
- **Long conversations.** The D6 store's read-time view costs grow with the number of taught facts (ceiling between
  50 and 500 on pool CPU, `2026-09-25-d6-capacity-curve-N2000-6seed-CEILING-BETWEEN-50-AND-500.md`); no step here
  teaches more than one fact per session.
- **Process-shared learning.** The C1 world model is one organ per server process; every arm here is a fresh
  process, so cross-session accumulation is not measured.
