---
type: preregistration
status: live
date: 2026-10-20
lane: integration / load-bearing-fraction (the #1 metric, re-measured at a pinned main)
mechanism: PRE-REGISTRATION of a measurement only: the reply-level load-bearing fraction (B2d design, stub arm) at
  S main f2885df70 on the six gate seeds, with the same job list and scorer as the dev read b2d1018 (seeds 7 and 8).
flip: no - a measurement of the shipped defaults at one pinned revision; it flips no flag and moves no default
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed before any gate-seed shard of this battery exists. No result is claimed here.
runner: research/runners/load_bearing_fraction.py (job lines cut by tools/lb_shard.py as for tag b2d1018; scored by
  tools/b2d_score.py)
artifacts:
  - research/findings/2026-10-12-validation-battery-prereg.md
  - research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md
  - research/findings/raw/_load_bearing/_shards/b2d1018/PIN.txt
  - research/coordination/LOCAL_LLM_RUNBOOK.md
---

bar-sanity: N/A - measurement only; this file registers no pass/fail bar and decides nothing

# The #1 metric at a pinned main, six gate seeds: PRE-REGISTRATION (measurement only)

**Filed 2026-10-20 by Claude, before any gate-seed shard ran.** Owner-approved speed-up (2026-10-18, RULING
`idle-backlog-1018` I1): the reply-level load-bearing fraction was last measured weeks ago; Phase A's exit asks for it
at HEAD. This registers that measurement so the gate seeds can run. It has no pass/fail bar and decides nothing.

## 1. What runs

- **Revision:** S main `f2885df70cfb99a8715c376bca12ac3c2134ef46` (the pin in `research/findings/raw/_load_bearing/_shards/b2d1018/PIN.txt`), the same
  revision as the dev read. No code changes between the dev read and the gate seeds.
- **Job list:** the b2d1018 dev list (49 lines per seed: every coverable faculty's intact A, intact B and lesion arms),
  with only the seed changed, for seeds 42, 43, 44, 100, 101, 102. Outputs:
  `research/findings/raw/_load_bearing/_shards/b2d1018/s<seed>/`.
- **Memory:** each line's mem_ok/memcap is sized from the measured peak (dev smoke PEAK_RSS 34.6 MiB) times 1.3, with
  a 1 GB floor (RULING `pool-stall-1020b`), never a round guess.
- **Pre-flight (gate_preflight):** the dev read is the scorer pre-flight: `tools/b2d_score.py --tag b2d1018 --pin
  f2885df70 --seeds 7 8` gave verdict COMPLETE, 0 fail cells (NEEDS `[lbf-head] RESULT` 2026-10-19 04:22).

## 2. What is reported (all of it, whatever it shows)

`tools/b2d_score.py --tag b2d1018 --pin f2885df70 --seeds 42 43 44 100 101 102`, unmodified:

1. The S0 core fraction (faculties whose lesion changes the reply, over coverable faculties) per seed, its mean and
   standard deviation, and the S2 reading the scorer prints beside it.
2. The per-faculty table: load-bearing on how many of the six seeds, and every cell the scorer marks failed or
   UNDEFINED, by name. An UNDEFINED cell is reported as such, never counted as load-bearing or hollow.
3. The host of every result (`tools/where_ran.py`), since validity must not depend on the node.
4. The dev read beside it (seeds 7 and 8: 33 of 40 = 0.825, sd 0, quoted from NEEDS `[lbf-head] RESULT` 2026-10-19 04:22) <!--derived-->, labelled as dev. The dev number is never cited as
   the result.

## 3. What this does not claim

- **The stub mouth, not the Qwen mouth.** This is the stub arm. The Phase A leg "reply-level fraction at HEAD with the
  mouth" stays open until the mouth arm can run; that arm is blocked on the claim checker (RULING
  `moat-slug-hold-1018`). The report says so in its first line.
- **No capability verdict.** A number below or above any target is a measurement, not a GO or NO-GO. A drop against an
  earlier reading is reported with the faculties that moved.
- **Seeds are one build each.** Each seed is one deterministic build; the six seeds are the replication.

## AMENDMENT 1 (2026-10-20, Claude; filed before any gate-seed shard of this battery exists)

Two operational corrections, both found by the scorer pre-flight (`tools/gate_preflight.py`) before a single gate line
was queued. No arm, job list, output path, report item or claim changes.

1. **Scorer pin.** `tools/b2d_score.py` compares each sidecar's `git_sha` with `--pin` exactly, so the 9-character
   pin written in section 2 reads every cell as MISSING (the first pre-flight read). The registered scorer command
   is therefore `tools/b2d_score.py --tag b2d1018 --pin f2885df70cfb99a8715c376bca12ac3c2134ef46 --seeds 42 43 44
   100 101 102`. With the full pin the dev read reproduces the [lbf-head] numbers: verdict COMPLETE, robust core 33
   of 40 <!--derived-->.
2. **Memory.** The 34.6 MiB in section 1 came from a smoke shard that ran no lesion pair. Three real b2d1018 shards
   printed PEAK_RSS_KB 5910744, 5875500 and 5971268 (5.6 to 5.7 GiB; runbook WORKLOG 10-19) <!--derived-->, so each
   gate line is queued at mem_ok 8 / memcap 8 (the largest peak x 1.3 = 7.4 GiB, rounded up).
