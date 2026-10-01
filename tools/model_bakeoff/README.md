# Model bake-off

Which model should Hermes use for which part of this project's work, at what cost? (Owner, 2026-09-30: replace
most of Claude's weekly work; never use a provider that may train on our data.) This directory runs real project
tasks through the real Hermes harness with each candidate model, grades them, and prices them.

## What it measures

- **Tasks** (`suite.json`) are real pieces of past work with a known right answer, in five roles:
  - **judge**: state a six-seed verdict from its pre-registration;
  - **implement**: fix a real bug so its real test passes;
  - **design**: choose the next lever at a wall;
  - **operate**: run-management steps in a sandbox;
  - **review**: find what history shows was wrong in a change or a finding.
- **Starting point.** Each task starts from a clone that holds only history up to the task's base commit, so the
  answer is not in it.
- **Candidates** (`models.json`): the local Qwen plus OpenRouter models across price tiers. Every OpenRouter request
  carries `provider.data_collection: "deny"`. A model with no endpoint that satisfies it fails the preflight and is
  not run.
- **Grading.**
  - Pass/fail tasks run a command after the model finishes. Files a task protects, such as a given test, are
    restored first.
  - Rubric tasks are scored blind: by Claude by default, or by two OpenRouter judges from different vendors
    (`judge.py`) when `models.json` lists `judges`. Judge disagreements above 0.25 are adjudicated by Claude.
- **Audit.** A run whose tool calls reached the network, or read a path that could hold an answer, is disqualified
  and counted, never dropped silently.
- **Cost.** Estimated from Hermes's token counts at OpenRouter list prices, and checked against the key's own spend
  counter for each batch.
- **Spending as little as possible** (owner, 2026-09-30):
  - The candidates are the local model plus cheap and mid-priced OpenRouter models. There are no frontier models
    by default, because the reference answers are Claude's own past work.
  - Rubric tasks are judged blind by Claude, not by paid judges.
  - `run.py --max-spend` (default $3) starts no new paid run once the batch's real spend passes the cap.
  - The plan is staged: first a small screen (one or two tasks per role, every candidate), then the full suite on
    the two or three best only.

## Isolation

`launch.sh` starts a throwaway container from the same image as Hermes, as the agent's uid. It has 6 CPUs and 10 GB
and is removed on exit. It mounts:

| Read-only | Read-write |
|---|---|
| these tools and the per-task bases | its runs dir |
| a bake-off-only key file (below) | its own `HERMES_HOME` (production config minus `terminal.cwd`) |
| the project's `.venv` / `.python` | |
| the reference library | |

It mounts no workspace, no ssh keys, no RAG index (it holds later findings), and no Hermes data.

**The key file.** A model run can read its own environment and has network access, so it gets a dedicated OpenRouter
key with its own spend limit, never Hermes's `.env` (until 2026-10-01 it got the whole file, so every run could read all
of Hermes's secrets; `research/FAILURE_LOG.md`). `launch.sh` reads `$BAKEOFF_ENV`, default
`/home/dant123/data/sim-bakeoff-secrets/openrouter.env`, and refuses to start a run if the file sets anything but
`OPENROUTER_API_KEY` (plus `HERMES_CUSTOM_192_168_0_68_11434_API_KEY` when the local model is a candidate). Before a
relaunch:

1. Owner: create an OpenRouter key named `sim-bakeoff` with a limit equal to the bake-off budget. `--max-spend` then
   reads that key's own counter, so Hermes's spend no longer counts against the batch cap.
2. Owner, on the server: `sudo install -d -o 99 -g 100 -m 0700 /home/dant123/data/sim-bakeoff-secrets`, then write
   `openrouter.env` there holding only `OPENROUTER_API_KEY=...`, mode 0400, owner 99:100.
3. After the bake-off: delete the key at OpenRouter and `sudo shred -u` the file. Runs use the
production approvals (`HERMES_CRON_SESSION=1`, the deny list, `command_allowlist`) and the production gates
(`core.hooksPath`).

## Running it (Claude, from the workstation)

Everything lives in one Docker volume, `sim-bakeoff`, on Docker's own storage. That is outside `backup.sh` and
sanoid, so about 20 GB of disposable clones never reach the nightly or offsite backups. `launch.sh` is piped over
ssh from a reviewed checkout, never run from the agent-writable workspace.

```bash
L=tools/model_bakeoff/launch.sh
# 1. once: the volume
ssh homelab-server 'bash -s -- init' < $L
# 2. bases at a revision the workspace has (no network; workspace read-only)
ssh homelab-server 'bash -s -- prepare origin/research/model-bakeoff' < $L
# 3. which models are allowed (data_collection=deny)
ssh homelab-server 'bash -s -- preflight' < $L
# 4. a batch (detached), its log, the table
ssh homelab-server 'bash -s -- run screen1 --tasks a,b --models x,y --max-spend 4' < $L
ssh homelab-server 'bash -s -- log screen1' < $L
ssh homelab-server 'bash -s -- report screen1' < $L
# 5. results (no clones) as a tarball, then remove the volume
ssh homelab-server 'bash -s -- fetch screen1' < $L > screen1.tgz
ssh homelab-server 'bash -s -- destroy' < $L
```

The local model shares one slot with Hermes's own driver, so run its batch with `sim-driver` paused (or overnight).
The results worth keeping are committed under `research/findings/raw/_model_bakeoff/`.

Results and the model set they chose: `RESULTS.md`. Run paid batches one at a time: `--max-spend` reads the
OpenRouter key-wide usage counter, so a concurrent batch's spend counts against this one.
