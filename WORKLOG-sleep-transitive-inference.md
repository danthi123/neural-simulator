# WORKLOG — research/sleep-transitive-inference (plan 2026-09-25 step 5)

Build agent log for this branch (worktree `.claude/worktrees/sleep-transitive-inference`). Newest last.

- 2026-09-29 12:15 worktree created from main `f52c6d246`. Research record: `before_you_build.sh` (transitive-inference
  gate 2026-06-27 read: ordinal TI is a different capability; the RANK3 recombination findings of 2026-08-01 read: the
  shared-hub encode was non-selective there; Ecker LTU NO-GO 2026-08-27 read: plastic replay symmetrizes, and its
  no-seed lesion leaves weights unchanged). PubMed abstracts read for the biology binding.
- Design issues of the prioritized-memory design (revision 4) closed in the design as revision 5 (section 10d).
- Runner `research/runners/sleep_transitive_inference.py` + grader `..._grade.py` (selftest PASS). First dev smoke
  (seed 7, 6 laps, short night): the host STDP replica reproduces the engine exactly (max |dw| = 0).
- Commit state: `gates/compute_idle_persistent` blocks non-exempt commits until the pool/GPU are busy (coordinator
  note 2026-09-29). Work is staged here; commits retried only after `tools/parallel_audit.py` stops reading
  UNDER-PARALLELIZED on compute.
- 2026-09-29 13:00-14:00 dev exploration on seed 7 (numpy locally, cupy through `tools/gpu_queue.sh`): SWR events 32.5 ms
  apart spread plastic nights to every assembly; at a 150 ms period the night writes a specific A->D link (numpy replica
  exact); the asymmetric STDP rule erodes B->D under replay; PV feedback inhibition halves the erosion. Logs and scripts
  copied from the RAM scratchpad into `research/findings/raw/_sleep_transitive_inference/dev/` (exploration_seed7.json,
  scripts/).
- Found: the Ecker runners set `cfg.ou_noise_sigma_pa`, a field the engine never reads (it reads `ou_std_current_pA`,
  default 100 pA), so their "40 pA" runs ran at 100 pA. This runner sets the real field at 100 pA. The FAILURE_LOG line
  and a gate for "config attribute the engine never reads" are handed to spawned task task_8f9171ae (adding the line
  here without its gate would block this branch's commits on gates/coverage).
- 14:00 the coordinator's mid-flight audit (7 confirmed issues) applied before registration: budget-matched `scr` + U6
  validity guard + order-only `rev` (STI3b); STI7 premises survive the night (in the STI verdict; PM1 = STI7); `dis`
  replay disruption as the real replay lesion, U2 read off the raster/weights; PM2/PM3 predicted FAIL; minor-detail
  schedule restored to every third lap (the one-lap change would have tuned toward PM2); gate runs refuse non-numpy;
  `cross` crossed-chain arm (STI4 sign flip), `nohub` REPORTED vs its own pre-night store; Z never initiates SWRs.
  Grader selftest 27/27 PASS.
- Final-operating-point dev runs queued on the GPU (seeds 8-11, cupy, every arm); the numpy dev seed 7 is held by
  `tools/mem_ok.sh` (another lane's host_share workers hold ~27 GB).
- 15:xx owner mode change (Claude tokens scarce): finished as an EXECUTION PACKET, `drafts/HANDOFF.md` (commit,
  provision, queue 30 pool jobs = 6 seeds x 5 arm parts, merge + score with the registered scorer, outcome->verdict
  table, stop conditions, NEEDS-CLAUDE list). Dev seed 8 (final constants, cupy): DI 0.32 after the night, PBD 0.41 ->
  0.16 (premise erosion with PV on), `dis` wrote nothing; the budget-matched raster scramble lost the premises (PAB
  0.07), so STI3 now binds on `wperm` (the night's own weight changes permuted; U6 on it), `scr` REPORTED. Plumbing
  smoke of every arm on numpy (dev seed 7, tiny): no error, replica exact. Tests 8/8, grader selftest 28/28.
  Nothing committed (compute-idle gate). GPU dev seeds 9-11 still queued; seed 8 running.
- 2026-09-29 LANDING (Claude): dev cupy seeds 9-11 were killed mid-build (rc=143/137, a GPU pause/game cycle, not a
  code failure); re-queued identically, all 4 (8-11) completed clean (no Traceback). Step 2 operating-point check:
  `--devcheck` on all four prints `all_pass: true` (opcheck.json). Step 3 pre-commit checks all pass (8/8 tests,
  SELFTEST PASS, check_docs OK). claim_check flagged the prereg's own `seedS.json`/`seedS_*.json` naming-template
  prose as missing artifacts (a literal-path false positive); reworded to `seed<S>...` (matches the doc's existing
  `<sha>`/`<PART>`/`<tag>` bracket convention) -- claim_check now clean, no other change. Commit 1 (code+tests+dev)
  812c2af78; commit 2 (prereg+docs) 35e6b85bd; merged origin/main (bfde9a195, no conflicts, prereg unchanged, tests
  re-verified); pushed both remotes. Provisioned pool41/pool42 at bfde9a195; queued 30 gate jobs (6 seeds x 5 arm
  parts), 0 failures. Awaiting the pool; score with `sleep_transitive_inference_grade` per prereg section 8 once all
  30 parts land.
- 2026-10-01 03:50 (local model, Hermes): T3 steps 7-8. All 30 parts present (seed102_b landed 02:53 via sim-sync). Ran the registered scorer (blob 90dd3c5957, byte-identical to gate commit bfde9a195; it is not in S's main tree, so ran the copy at /opt/data/cache/scratch/sleep_ti_grade.py with PYTHONPATH=$P; --selftest PASS). Merged the 6 gate seeds + family -> research/findings/raw/_sleep_transitive_inference/{seed{42,43,44,100,101,102}.json, aggregate.json}. Family verdict UNDEFINED: 4 NO-GO (STI7_premises_survive false on all 4; STI3b_order_necessary also false on 43/44/102) + 2 UNDEFINED via U6 (seeds 42, 101: weight-permutation control lost the premises / flooded the foils -> STI3 has no valid control). pm_verdict UNDEFINED. Necessity computed over the 4 defined seeds. Finding: research/findings/2026-10-01-sleep-transitive-inference-6seed-UNDEFINED.md (PROVISIONAL). U6 -> prereg section 10 "U6 fires" bullet (next control: permute within classes of synapses of equal pre-night weight). Hands back per #8.
- 2026-10-07 20:05 (Hermes): wake re-check of T3. The runbook Start/WORKLOG read mis-fired (the worktree was
  absent, so `tail` on the WORKLOG path printed ABSENT while the packet sat on the branch); worktree created
  (`56da4a4e0`), then found steps 7-8 already landed 2026-10-01 (commit `f18da9f8f`, entry above). Re-ran the
  registered scorer on the same 30/30 parts as an idempotency check: family verdict UNDEFINED, same per-seed
  U6 x2 / NO-GO x4 (only the provenance run_id differs); duplicate 2026-10-07 finding removed, branch data
  restored via `git checkout --`, main's re-stamped prov sidecars restored too. Nothing committed but this line.
