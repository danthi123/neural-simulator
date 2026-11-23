# Stage-2 TEST v2 freeze note (FROZEN)

**Status: FROZEN 2026-11-23** in this one commit (prereg §7): this note, the code it hashes, the arm list and the launch files.
Prereg: `brain/research/2026-11-23-stage2-test2-prereg.md` (sha256 `88897226a9f0781a70123210213a6ffe434563739f677c950c51351243049b5d`,
including Amendments T1, T2 and T3). Written 2026-11-23 on branch `claude/gain-refit-1113`, worktree base HEAD
`c7998f1cb2fe47d6592ff635aeb2aa8bb986af75`. Uncommitted at writing: `brain/eval/life.py`, `brain/tests/test_life.py`,
`tools/aws_brain_batch.sh`, `tools/brain_batch_run.sh` (resume guard, AWS env, AMI pin), the prereg (Amendment T3),
new `brain/research/test2_launch/` (launch plan + 18 job files). The freeze commit must carry exactly the
bytes hashed below. If any file changes before that commit, every hash here is re-computed.

## 1. sha256 of each §7 file (as in the worktree now, `sha256sum`)
| File | sha256 |
|---|---|
| brain/eval/life_bank_test2.py | 5e77577859ea87ceb7d95e0986b8b5a06b99ab4a6888a6a75becb860a5835db5 (equals life_bank_test2.sha256) |
| brain/eval/life_cues_test2.py | 794da243af3f6b26474848be18f5a84669da5f19685fa1d6e8269cc730933a8e (equals life_cues_test2.sha256) |
| brain/eval/life_interview_spec.py | 4b70347f5d56d9f89a0de38e2467a5395d1c30044a3a789fa1d54f27cc2e5a72 (prereg 4b70347f..., equals .sha256) |
| brain/eval/life_heldout.py | adc7d90240c0fca3b3ccb8925484891500ab22d8f79b7d2c98becdc4789077d7 (prereg adc7d902..., equals .sha256) |
| brain/eval/life_read_spec.py | 6cf3690a68e13f05fadcf42df76141b5e2f1d598985ec6065f8a71cdf30ae05a (prereg 6cf3690a..., equals .sha256) |
| brain/eval/life.py | 7c9ce18079e48ad41e59754b819b2cb25273e0fd6f5251ac1a30c492725d095d (uncommitted edit, see header) |
| brain/eval/life_age_ladder.py | 7171270fdfedebaf35e07e10bc2e7c56789efd5640a7cfbd75fae793b1a6888b |
| brain/eval/life_bars.py | ffc2488d4c9bd333551b0981cffc66bfb8e0421614bccf47ae44d0f840a3d4ba |
| brain/eval/life_leak.py | b331f118b1dafd6917eaf88653b93e722887f2e679a60d65530b0702dbd02c32 |
| brain/eval/life_costs.py | 03f77db29010fd7100434bdeb374cc8fc5dad59abcfc845a7b8885e97c26e8d8 |
| brain/eval/life_compare.py | 09727ffd9c3b0f5ecf89bfa4a5eb7b0bcc9bc28eb3a523baed1093992acb082d |
| brain/eval/life_read.py | 4a2f0d6fe9ab208d49287ecf27b0c252247b4f5f2eb78209df0057ead101d858 |
| brain/eval/life_interview_read.py | be61af2ef72deb2b1d5c3c9f1b29edc807fa3b856887905de5b21f19f9652cb2 |
| brain/eval/life_table.py | 0d54a0f4a7cf5025cb97d0410675c38e8670ae204aee0b936785d7fe51cc4fa2 |
| brain/brain.py | 4126b18d1e9e7e0b132e81cdd74395c090b55cd7faefd0e7be205ceb277bb3d7 |
| brain/sleep.py | c4b1abe588659fd4af427d3ca9419573398607ee38fec78ad33d182dda3498df |
| brain/cortex.py | 0952209af05006b38601687609c1b5511c81eecc013117f4ad4da4c2819c264c |

`life.py` `FROZEN_FILES` (line 892) lists the same 17 files plus the two test2 `.sha256` files; every life records
their sha256 in `provenance.frozen_sha256`. Not in §7, recorded for the launch: `tools/aws_brain_batch.sh`
4d94c7ab545e548c1f3482ed13add3b81f5a7ddb43e0c2b44258abfde88fe6fb, `tools/brain_batch_run.sh`
309723156579c730d1ae9372caf553fb73ffc29ce993253713c512d9350a4067. With `AWS_BRAIN_ENV`, `AWS_BRAIN_AMI` and
`AWS_BRAIN_AMI_RECORD` unset, `aws_brain_batch.sh launch` behaves as before (no `env.sh`, newest AMI, no `ami=` state line).

## 2. Reader and extractor models, prompt templates (sha256 of the UTF-8 string constant)
| Role | Model (where from) | Prompt constant (where from) | sha256 |
|---|---|---|---|
| §7 blind reader (G5) | tier `sonnet`: `READER_MODEL_TIER`, life_read_spec.py line 17 | `READER_PROMPT`, life_read_spec.py line 41 | c140bc7fd56c6fe95a88bd235f6a7e68e4fc4e290b40c15f41cc4baf142a68a3 |
| G9 cross-arm pairwise reader | same tier (life_read_spec.py line 17) | `PAIR_PROMPT`, life_read_spec.py line 61 (filled by life_read.py line 303) | 69c01c52039f67b6a157320d78198c88b7a22280c13bd5737adeacbaec218191 |
| answer reader | same tier | `ANSWER_PROMPT`, life_read_spec.py line 77 | 5fc8d6cb15a188d6a3888110e5c7d4d8171571822a4da9122079943d7225ee99 |
| interview claims extractor | tier `sonnet`: `EXTRACTOR_MODEL_TIER`, life_interview_spec.py line 94 | `EXTRACTOR_PROMPT`, life_interview_spec.py line 95 | b435055a0df4eaf6a1a4d8eb8f75dcb2f4fc9752ece283e821764ad915e84931 |
| interview probe (brain input, not a reader) | n/a | `KNOW_TEMPLATE`, life_interview_spec.py line 50 | 6e551c0246b78f9ec8f6b08450fa649d8c0ee5e8a6ce59c776866c4aeacfe482 |

Spec versions: `read-spec-2026-11-21`, `interview-spec-2026-11-20`. The specs name a tier only; the exact model ID is
pinned here. **Every TEST v2 reader and extractor call (§7 reader, G9 pairwise, answer reader, claims extractor) uses
`claude-sonnet-5-5`**, the sonnet-tier model the DEV reads ran on. A read on any other model ID is not a TEST v2 read.

## 3. Arm list (Amendment T2)
B (defaults); K-P unchanged (`--keep-bank prior`); D2-PR2 (`--replay-priority loss_a`; enters per its DEV read,
`results/2026-11-23-d2pr2-pra-dev-read-1.5b.md`, PASSES 3.1-3.3). Six seeds 42/43/44/100/101/102 per arm, 15 days,
the §2 command, AWS g5 CUDA fp16 (§3).

## 4. K-P-f decision
Not built and not run (Amendment T2: the identity-gate read found no variant entering, and its timebox bars a second
bank or mask rule). K-P runs unchanged; the header Expected-fail (G7, G8, G10, G2a, G3) stands.

## 5. Margin rule
Amendment T1: no margin is widened. Device-read values (`results/2026-11-23-test2-b-vs-b-margins.md`) are a reporting
band only; a guard FAIL inside it is labelled so, the verdict does not change.

## 6. §7 gate checklist
| Gate | Status | Evidence |
|---|---|---|
| v2 DEV row check `lifeAL_s322` back and passes v2 §6 | MET | `results/2026-11-23-age-ladder-dev-row-check-s322.md` "Row check PASSES on every condition checkable on DEV" |
| `life_costs.py` reproduces §5 DEV counts; DEV tests with `LIFE_COSTS_REQUIRE_DEV=1`, 0 skipped | MET | 2026-11-23 run, `LIFE_COSTS_REQUIRE_DEV=1 pytest -rs` on test_life_costs, test_life_compare, test_life_read_cross, test_life_read_emit_frozen, test_life_read, test_life: `170 passed, 28 warnings in 74.95s` (0 skipped; rerun after the freeze-review fixes) |
| `life.py` writes the §3 provenance fields | MET | life.py `provenance()` line 940: cuda_precision, torch/CUDA, GPU, both TF32 flags, `code_commit`, `aws_ami_id` (line 952, from `AWS_AMI_ID`, written to `env.sh` by aws_brain_batch.sh when `AWS_BRAIN_AMI_RECORD=1`, which launch_plan.sh sets), `frozen_sha256`, `brain_tree_dirty`; dirty-tree refusal for test2, also on `--resume` (test_registered_guard_also_runs_on_resume) |
| shared same-device check `life_costs.seed_status` used by `life_compare.py` | MET | life_costs.py line 281; life_compare.py lines 71, 432 |
| G9 mode validated | MET | `results/2026-11-23-g9-validation-kp-vs-b.md` Read 2: "G9: VALIDATED (PASS)" (read 1 FAILED its decoy check, stands) |
| D2-PR2 DEV read committed, K-P-f read committed or absent | MET | D2-PR2 read committed (f483ed14e and earlier); K-P-f absent (T2) |
| Reader/extractor model IDs and prompt hashes | MET | prompt hashes and the model ID `claude-sonnet-5-5` in §2 |
| One freeze commit holds note + code + hashes; brain/ clean | NOT MET | `git status --porcelain -- brain/`: `M brain/eval/life.py`, `M brain/research/2026-11-23-stage2-test2-prereg.md`, `M brain/tests/test_life.py`, `?? brain/research/test2_launch/`; this note uncommitted |

## 7. Launch plan
`brain/research/test2_launch/launch_plan.sh {plan|go D1|poll}` (sha256
9cf33275cad684de3c33ca5d2d183b473de6d9177e3bf25d3957d919e262cb00) and 18 job files
`brain/research/test2_launch/jobs_{B,KP,PR}_s{42,43,44,100,101,102}.txt`, both in the freeze commit. Job-list hash,
run from the repo root (stable glob order): `sha256sum brain/research/test2_launch/jobs_*.txt | sha256sum` =
1946ae8250c26f27543fdd09f461fee3deb010edd8bc60b70a104861301c51fc. The plan's log and `.launched_*` markers go to
`_scratch/stage2/test2/` (gitignored) so brain/ stays clean during `go`.
- Schedule (prereg Amendment T3): three UTC days from D1 (planned 12-04 to 12-06; D1 is a parameter, read by 12-10
  stays), at most 6 concurrent, worst case per day $47.47 / $45.11 / $38.72 (also <= $50 with every start 45 min
  late), total about $131 worst, about $102 expected; dead-man B 345, KP 510, PR 355 min. Nothing else on AWS those days.
  The $50 cap is per account.
- AMI: `go` resolves the newest Deep Learning AMI once, logs it, and passes `AWS_BRAIN_AMI` (pin) and
  `AWS_BRAIN_AMI_RECORD=1` (provenance) on every launch, so all 18 lives run one image.
- Never terminates an existing instance: only an instance the current `launch_life` call created (`fresh=1`) and
  verified STOPPED or NOT_STARTED is terminated; an existing state file is logged "existing instance, look by hand".
- Resume: `brain_batch_run.sh` skips a job whose `results/<name>.json` or non-empty `<name>.json.json` exists (life.py
  writes `<out>.json`, refreshed after every day), so a rerun never restarts a life from scratch. **A dead-man-stopped
  TEST life is resumed only by a manual jobs file whose line adds `--resume`, on a g5 (§3), never by the
  `aws_brain_batch.sh resume` action** (on a TEST life that action would skip the part-done job and log "all done";
  the partial home must then be pulled with `AWS_BRAIN_PULL_WORK=1` before any terminate).
