---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: A5 (theory of mind: false-belief wire into chat)
mechanism: scoring the seed-7 (dev-seed) smoke + byte-identity-off criteria the 2026-09-24 PREREGISTRATION
  registered for this build lane, against the raw artifacts the lane already produced. No new run.
seeds: [7]
prereg: research/findings/2026-09-24-tom-false-belief-chat-wire-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_tom_false_belief_chat/smoke_seed7.json
  - research/findings/raw/_tom_false_belief_chat/gate_s7_pool_postfix.json
  - research/findings/raw/_tom_false_belief_chat/byte_identity_off.json
  - research/findings/raw/_tom_false_belief_chat/arm_unset.json
  - research/findings/raw/_tom_false_belief_chat/arm_explicit_off.json
  - research/findings/raw/_tom_false_belief_chat/arm_on.json
verdict: Both criteria the PREREGISTRATION fixed for this build lane -- the seed-7 dev-seed smoke of the LIVE
  `FalseBeliefChatOrgan` / `webapp.false_belief_chat` wire, and the byte-identity-off check -- PASS, and PASS
  twice independently (a dirty local dev run and a later clean `git_archive` pool re-run at the post-review-fix
  commit agree on every field). This is a smoke at its registered seed-7 scope only, NOT the 6-seed capability
  gate (explicitly deferred to B2b) and NOT a claim that the A5 wire is `closed`/`integrated` per
  `docs/TERMS.md` -- production stays default-OFF, witnessing/presence is host-parsed, and the belief-location
  read is a host argmax, exactly as the PREREGISTRATION declared up front.
---

# Theory-of-mind false-belief chat wire: seed-7 smoke scored -- both registered criteria PASS

## Completeness and liveness (checked before scoring)

All six raw artifacts named in the task exist in the primary checkout's untracked
`research/findings/raw/_tom_false_belief_chat/` and were copied byte-for-byte (verified with `diff -rq`) into
this worktree, alongside their `.prov.json` sidecars; nothing in the primary checkout was moved or deleted.
`ps -eo etimes,args` over `pool1`, `pool2`, `pool41`, `pool42` (all four reachable, read-only
`ssh -n -F research/queue/.pool_ssh_config`) found no `_tom_false_belief_chat*` or `tom_false_belief_chat_organ`
process on any node -- the closest match was an unrelated `affective-tom` lesion worker on `pool1`. A local
`ps` scan found nothing either. Nothing for this battery is running; it is complete and ripe to score at its
registered seed-7 scope.

## Provenance (why the two production-code artifacts don't carry the "b36669ff9" prefix)

`b36669ff9` (full `b36669ff9a9d4576128ca7f73bf4d4f0fb4a4b3d`) is the PREREGISTRATION commit itself, and its own
message says plainly "No new code yet" -- it fixes the criteria before any evaluation code exists, per the
project's prereg-precedes-evaluation rule. The wire's actual code landed two commits later
(`1d62497bf`, then a review-fix `b5a895471f2f2040fb61fc403aa76f8668c27cd5`, both confirmed descendants of
`b36669ff9` via `git merge-base --is-ancestor`), and that fixed commit is what the byte-identity and
re-verification artifacts were run against:

| artifact | git_sha (full) | git_dirty | source_kind |
|---|---|---|---|
| `smoke_seed7.json` | `b36669ff9` (abbreviated) | true | `null` |
| `gate_s7_pool_postfix.json` | `b5a895471f2f2040fb61fc403aa76f8668c27cd5` | false | `git_archive` |
| `arm_unset.json` / `arm_explicit_off.json` / `arm_on.json` | `b5a895471f2f2040fb61fc403aa76f8668c27cd5` | false | `git_archive` |
| `byte_identity_off.json` | `b5a895471f2f2040fb61fc403aa76f8668c27cd5` | false | `git_archive` |

**`smoke_seed7.json` plainly predates the `git_archive` provenance convention**: it is the original dev run
(local worktree `wf_4703a2bd-4dd-9`, `git_dirty: true`, `source_kind: null`, abbreviated SHA), made while the
runner/organ code was still uncommitted on top of `b36669ff9` -- its own producing commit (`1d62497bf`) landed
two minutes later. It is a dev-seed smoke, not a gate row, so this is scoped as declared, not a defect.
The other five artifacts DO meet the full-revision + `git_archive` bar, at `b5a895471f2f2040fb61fc403aa76f8668c27cd5`
-- a commit that review found a real bug in (`review:A5`: a query's own agent/object was silently discarded) and
fixed. That commit is now merged to `main` (`a14222c993`, confirmed ancestor of `origin/main`), and `git log`/`git
diff` from `a14222c993` to `origin/main` (`154b2ab51`) show **zero** further changes to
`webapp/false_belief_chat.py`, `research/runners/tom_false_belief_chat_organ.py`,
`research/runners/_tom_false_belief_chat_gate.py`, `research/runners/_tom_false_belief_chat_offcheck.py`, or the
false-belief hook region of `webapp/server.py` -- the code these artifacts measured is exactly what `main` runs
today.

## Per-gate table (seed-7 smoke, `_tom_false_belief_chat_gate.py --seed 7 --n-items 8`)

Both runs of this gate (the original dirty dev run,
`research/findings/raw/_tom_false_belief_chat/smoke_seed7.json`, and the later clean `git_archive` pool re-run
at the fixed commit, `research/findings/raw/_tom_false_belief_chat/gate_s7_pool_postfix.json`) report
**identical** values on every field below -- an independent reproduction of the seed-7 result under proper
provenance, not just one run trusted once.

| precondition (runner's own `verdict_block`) | threshold | measured | result |
|---|---|---|---|
| >= 8 items | >=8 | 8 (4 false / 4 true) | PASS |
| dev seed only (not 42/43/44/100/101/102) | seed not in {42,43,44,100,101,102} | seed 7 | PASS |
| false-belief acc vs chance (live organ) | >= 0.85 | 1.0 | PASS |
| reality-baseline FAILS false-belief | <= 0.20 | 0.0 | PASS |
| true-belief control: belief updates when witnessed | >= 0.85 | 1.0 (`true_belief_agree`) | PASS |
| other-lesion collapses the read (control) | treatment vs control both measured, separated | treatment 1.0 / control 0.0 | PASS |
| other-lesion collapsed to <= chance_loc bar | <= 0.45 | 0.0 (attributable_fraction 1.0) | PASS |
| scrambled witnessing collapsed | <= 0.70 | 0.5 (attributable_fraction 0.666667) | PASS |
| **runner's own aggregate verdict** | — | `status: GO`, `go: true`, `undefined_reasons: []` | **PASS** |

No `UNDEFINED` reasons were reported by the runner (masks were non-empty on both false- and true-belief items),
so the "UNDEFINED is never 0" rule has nothing to invoke here -- every precondition resolved to a real
true/false. This is a **dev-seed smoke gate**, not one of the project's 6 validation seeds, so it is not entered
as a gate row anywhere the roadmap counts capability GOs.

## Byte-identity-off criterion

`research/findings/raw/_tom_false_belief_chat/byte_identity_off.json` (runner
`_tom_false_belief_chat_offcheck.py`, same fresh numpy build, `cfg.seed=7`, 10 turns including a full
5-sentence Sally-Anne narration + belief query) reports `byte_identical_off: true`, `off_diffs: []`,
`on_arm_query_populated: true`. I independently re-verified this rather than trusting the summary field:
`research/findings/raw/_tom_false_belief_chat/arm_unset.json` and
`research/findings/raw/_tom_false_belief_chat/arm_explicit_off.json`
(`BRAIN_FALSE_BELIEF_CHAT` unset vs explicitly `"0"`) are byte-identical by direct SHA-256 of the raw files
(`a9d8ec756139ca8c19aa61f7dca756b7d3c79c5684075a9442dccf5cbaf8c2ef` for both), and deep-equal in Python. `arm_on.json`
(flag `"1"`) differs from both, as expected: on the `tom_fb` turn, `arm_unset`/`arm_explicit_off` abstain
("I don't know about that... I haven't learned about sally yet", `false_belief_tom` field absent) while `arm_on`
answers "Sally will look in the basket for the marble." with a populated `false_belief_tom` block
(`belief_location: "basket"`, `reality_location: "box"`) -- the flag is load-bearing on that turn and inert
everywhere else, matching the PREREGISTRATION's byte-identity claim exactly.

## What this does and does not show

**Shows:** the LIVE, chat-facing orchestration (`FalseBeliefChatOrgan` + `webapp/false_belief_chat.py`'s sentence
grammar), not just the already-6/6-seed-GO'd standalone derisk trial runner, reproduces every one of that
derisk's own anti-cheats (reality-baseline failure, other-lesion collapse, scramble-witnessing collapse,
true-belief control) at seed 7, through the actual production code path that is merged to `main` today, with
the default-OFF flag verified byte-identical-inert by direct hash rather than by reading the code.

**Does not show:** a 6-seed capability GO (explicitly out of scope here, deferred to B2b); anything about
generalization beyond `n_items=8` drawn from one dev seed's RNG stream; a spiking replacement for the two
declared host residuals (witnessing/presence is a host regex parse of "leaves"/"returns"; the belief-location
action read is a host `argmax` over late-window firing rate) -- both stay open per the PREREGISTRATION's own
"honest scope" section and are not scored as closed by this document; or that the capability is `wired`+
`on-by-default`+`scaffold-retired` (`docs/TERMS.md` "integrated") -- it is `wired` (reachable from
`webapp/server.py`'s `brain_reply`) but default-OFF, and the LBF row for it is PARKED (merge commit `a14222c993`:
the row's `EXTRA_TURNS` aren't merged into the AG-REG import hook yet, so it isn't in the 38-faculty live
registry).

## Next step (this is a PASS, not a NO-GO, so THE LAW's "bank + new lever" does not apply)

Per the PREREGISTRATION's own plan: the 6-seed capability gate over 42/43/44/100/101/102 runs at a frozen SHA
in B2b -- unchanged by this scoring. Separately, wiring the row's `EXTRA_TURNS` into the AG-REG import hook
(`research/runners/lbf_rows/__init__.py`) would un-park `tom-false-belief` in the load-bearing-fraction battery,
which is currently possible without new experiments (both pieces already exist; only the merge point is
missing).
