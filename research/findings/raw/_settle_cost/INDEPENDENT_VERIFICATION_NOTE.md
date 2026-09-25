<!--derived-->
# Independent re-derivation of the `_settle_cost` VOID verdict (2026-09-25)

Reviewer worktree, branch `research/verify-settle-cost-0925`, cut from `origin/main` @ 0c86f45ba.
Subject: `research/findings/2026-09-25-settle-cost-battery-wrong-instrument-VOID.md`,
scored on branch `research/score--settle-cost-0925` @ `656dd1d0346c79dec10e0a5e30f4a9633cd1c79c`.

This note does not amend, merge, or push to the subject branch or to main. It records an independent check
performed from a *separate* worktree, starting from the raw artifacts alone (copied here byte-identical from
the primary checkout's untracked copies — verified by `sha256sum` before copying, all 9 files match).

## What was independently re-derived (not taken from the scorer's prose)

1. **Criterion L is uncomputable from this data.** `independent_rederive.py` (this directory) opens all four
   raw JSONs directly and checks for the `warm_turn_total_s` key the PREREGISTRATION's own formula
   (`research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md`, "Criterion L") reads.
   Absent from all four. Output: `independent_rederive_output.json`, `independent_verdict: "VOID"`.
2. **Executed the registered comparator against the actual files** (not just read its source): running
   `python -m research.runners._prod_chat_phase_timing --compare-on cupy_on.json --compare-off cupy_off.json`
   from the detached worktree at `656dd1d0` raises `KeyError: 'warm_turn_total_s'` at line 273 — the same
   failure the finding describes, reproduced by actually invoking the registered tool rather than trusting the
   finding's account of it.
3. **Provenance.** All 4 `.prov.json` sidecars: `source_kind: null`, `git_dirty: true`, two distinct
   `git_sha` (`5e571d6c6` cupy pair, `9d83a242a` numpy pair) — fails the git-archive/pinned-revision rule.
   Confirmed the rule predates this battery: `research/runners/__init__.py`'s `source_kind: git_archive` path
   was added by commits `59eded5d6`/`b976f8945` (2026-08-03) and `49958c6a7` (2026-08-04), ~7 weeks before
   these runs (`.prov.json` `started`: 2026-09-24T01:3x-01:4x).
4. **Runner identity.** Every `.prov.json`'s own `runner` field reads
   `research/runners/_settle_turn_cost_probe.py`, not the registered `_prod_chat_phase_timing.py`. Read
   `_settle_turn_cost_probe.py`'s docstring directly: it calls `reader.select_valence()`/`select_arousal()` in
   a loop, never `webapp.server.brain_chat()` — confirmed structurally distinct from the registered instrument,
   not just distinct by filename.
5. **No amendment.** `git log --oneline --all -- research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md`
   shows one commit (`228ba16f0`). Repo-wide grep for `_settle_turn_cost_probe|_settle_cost|production-chat-phase-timing`
   across `research/findings/*.md`, `GAP_CLOSURE_MISSION.md`, `docs/*.md` returns the PREREGISTRATION itself,
   the VOID finding itself, and one **stale** `GAP_CLOSURE_MISSION.md` line (see Issues below) — no amendment.
6. **Multi-seed rule.** All 4 arms are seed 42 (not even the sanctioned seed-7 dev/calibration smoke). The
   PREREGISTRATION explicitly exempts this latency measurement from the 6-seed rule ("no 6-seed rule applies to
   a latency measurement"), and the VOID finding makes no cross-seed or generalization claim, so the 6-seed
   gate does not apply and was correctly not invoked either way.
7. **Lesions.** N/A — the PREREGISTRATION itself states this is "Not a capability measurement. No lesion, no
   null control" by design; the VOID finding does not claim otherwise.
8. **Liveness, re-checked independently** (read-only `ssh -n -F research/queue/.pool_ssh_config`, plus the
   global `~/.ssh/config` entries for pool41/pool42): all four nodes answered (`ip-172-31-47-37`,
   `ip-172-31-47-57`, `node-dl4g243`, `node-dl5d243`, matching the finding's own hostnames exactly), zero
   processes matched `_settle_turn_cost_probe|_prod_chat_phase_timing|test_production_chat_gpu_smoke` on any of
   the four, on local `ps`, or in `research/queue/gpu_queue.log`'s tail / `gpu.queue.running` (which currently
   holds an unrelated `_gap5_dendritic_subunit_blocks_derisk` job).
9. **claim_check + check_docs**, run from a *separate* detached worktree of `656dd1d0` (removed after use, per
   the task's instruction to remove only that one): `tools/claim_check.py` on the VOID finding → `9 cited
   artifacts found, 0 missing; 9/9 measurements traced; exit 0`. `tools/check_docs.py` → W1/W2 both 0
   violations. `docs/TERMS.md`-governed words (`consolidation`, `compositional`, `self-organized`, `closed`,
   `GO`, `fully spiking`, `byte-identical`, `lesion`, `selective`, `works`) do not appear misapplied in the
   finding; the one `NO-GO` occurrence correctly refers to the separate, already-decided full-brain functional
   verdict, not to criterion L.
10. **Byte-identity of the cited raw artifacts**: `sha256sum` of the 9 files in the primary checkout
    (untracked) vs. the score branch's committed copies — all 9 match exactly. No divergence between what was
    scored and what sits in the primary checkout today.

## Issues found (none overturn the VOID verdict)

- `GAP_CLOSURE_MISSION.md:397-398` ("Next flip batch candidate: SETTLE -- cost +0.14 s/turn GPU, +0.23 s CPU")
  still presents the wrong-instrument battery's naive delta as an accepted, decision-relevant number, with no
  pointer to the VOID finding or the instrument mismatch. Predates the VOID commit by ~16h
  (`git blame`: `71b0bd7a93`, 2026-09-24 09:04, vs. the VOID scoring commit at 2026-09-25 13:51), so it is not
  a defect of the scoring commit itself, but it is now a stale board line per CLAUDE.md's same-cycle
  summary-doc-sync rule and should be corrected (strike the number, cite the VOID finding, point to its "Next
  action") the next time the board is touched.

## Verdict

**Agree: VOID.** Criterion L is UNDEFINED for this battery (uncomputable from the data, confirmed by directly
executing the registered comparator, not merely reading it) — it is an instrument-mismatch/methodology defect,
not a NO-GO on SETTLE's production-path cost. All rules and checks that apply here (provenance, no-amendment,
multi-seed exemption, claim_check, check_docs, doc-governed terms, liveness) were independently re-run from raw
data and reproduce the scorer's account with no discrepancy.
