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
| the OpenRouter key file | its own `HERMES_HOME` (production config minus `terminal.cwd`) |
| the project's `.venv` / `.python` | |
| the reference library | |

It mounts no workspace, no ssh keys, no RAG index (it holds later findings), and no Hermes data. Runs use the
production approvals (`HERMES_CRON_SESSION=1`, the deny list, `command_allowlist`) and the production gates
(`core.hooksPath`).

## Running it (Claude, from the workstation)

```bash
# 1. bases (inside Hermes, as the agent; reads the full history, writes only /opt/data/bakeoff)
ssh homelab-server 'docker exec -u 99 -e HOME=/opt/data/home -e BAKEOFF_REV=<rev> hermes bash -c "git -C /home/dant123/Projects/neural-simulator show <rev>:tools/model_bakeoff/prepare.sh > /tmp/prepare.sh && bash /tmp/prepare.sh"'
# 2. which models are allowed (data_collection=deny)
ssh homelab-server 'bash -s -- preflight' < tools/model_bakeoff/launch.sh
# 3. a batch (detached; log at /opt/data/bakeoff/runs/<batch>.log, read as the agent)
ssh homelab-server 'bash -s -- run screen1 --parallel 3' < tools/model_bakeoff/launch.sh
# 4. rubric grading, then the table
ssh homelab-server 'bash -s -- judge screen1'  < tools/model_bakeoff/launch.sh
ssh homelab-server 'bash -s -- report screen1' < tools/model_bakeoff/launch.sh
```

The local model shares one slot with Hermes's own driver: pause `sim-driver` while the local runs go, or run them
overnight. When the bake-off is over, delete `/opt/data/bakeoff` as the agent; the results worth keeping are
committed under `research/findings/raw/_model_bakeoff/`.
