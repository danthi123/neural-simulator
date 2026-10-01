# Weekly Claude review

Hermes drives the project on homelab-server (S); Claude reviews about once a week on a small plan. The owner starts it
when the 3x/day report says "Claude check-in DUE" (a waiting item is 72 h old), or weekly: open Claude Code in
`~/Projects/sim` on the workstation (W) and say "do a Claude check-in". Work down this list in order and stop when the
budget runs low; the items are ranked by what they unblock.

1. **Pause Hermes** so two drivers never edit main at once:
   `ssh homelab-server 'docker exec -u 99 -e HOME=/opt/data/home hermes hermes cron pause sim-driver'`.
2. **Read the waiting list** on S (newest last):
   `ssh homelab-server "sed -n '/^## NEEDS CLAUDE/,\$p' /home/dant123/Projects/neural-simulator/research/coordination/LOCAL_LLM_RUNBOOK.md"`.
3. **Each RESULT or STOP:** check the claim against the files and logs yourself; a RESULT line is a pointer, not
   evidence. Merge what is Claude's (anything `tools/agent_merge.sh` refuses). If a branch's gates are stale, merge
   main into it first. Write rulings as `- <KEY>: <ruling> (<date>)` in runbook section 6, and delete handled lines.
4. **Any PASS or GO:** run the verify-go skill before it touches main or the board.
5. **Draft results (runbook section 5):** promote a good one to a preregistered gate-seed packet, or rule it closed.
6. **Pool provisioning:** the server provisions by itself (`sim-provision`, results in `/opt/host-status/sim-provision/`). Act only on Hermes's NEEDS-CLAUDE: merge main into an old branch, or re-pin the runtime when a change touches `tools/pool/provisioning/` or the pool provisioning helpers (OWNERSHIP-MAP section 11).
7. **Spend:** OpenRouter `usage_daily` against the $6.66/day cap, and the key's $100 total limit.
8. **Resume Hermes** (`hermes cron resume sim-driver`), then bring W's main level with S (below).

**Claude's own commits on S's main** pass the agent backstop in `tools/githooks/pre-commit` (uid 99 on main may only
make routine changes) only with `-e SIM_CLAUDE_REVIEW=1` on the `docker exec`. Prefer committing on W and landing by
fast-forward or cherry-pick, as below.

**Moving commits while the repos are private** (git runs on S only as the agent, uid 99, inside the hermes container):
- W to S: `git bundle create <f> <base>..<branch>`, then `scp <f> homelab-server:/home/dant123/Projects/neural-simulator/_scratch/`.
  Then on S: `docker exec -u 99 -e HOME=/opt/data/home hermes git -C /home/dant123/Projects/neural-simulator fetch _scratch/<f> refs/heads/<branch>:refs/claude/<branch>`,
  then `merge --ff-only`, or `cherry-pick` if S main moved.
- S to W: bundle `<base>..main` on S, scp it back, then `git fetch <f> 'refs/heads/main:refs/remotes/server/main'` and
  `git merge --ff-only refs/remotes/server/main`.

Background: `docs/plans/2026-10-01-hermes-readiness-assessment.md` (why the review is needed) and
`research/coordination/LOCAL_LLM_RUNBOOK.md` sections 2, 5 and 6 (what Hermes may do alone).
