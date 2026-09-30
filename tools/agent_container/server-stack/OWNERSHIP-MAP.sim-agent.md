# OWNERSHIP-MAP.md additions for sim-agent (paste into ~/apps/OWNERSHIP-MAP.md)

Draft text for the owner to paste into `~/apps/OWNERSHIP-MAP.md` on srv-host-0, written in that file's own
style. Two pastes: one line in the Emergency block's local-build list, and one entry in §8 "Bespoke containers &
sidecars". Source: neural-simulator repo, `tools/agent_container/server-stack/OWNERSHIP-MAP.sim-agent.md`; the
full procedure is that repo's `docs/AGENT_HARNESS_SETUP.md`.

## Paste 1: the Emergency block's local-build list

Replace the `#   local builds: ...` comment line with (sim-agent appended):

```bash
#   local builds: proxy, pulsevault, cync-lan, genomeprivate, privdna, proxchat, saythatsh, sim-agent
```

## Paste 2: a new entry at the end of §8 "Bespoke containers & sidecars" (before its closing `---`)

**sim-agent** (the neural-simulator repo's isolated local-model agent harness: a possibly-adversarial model drives
it, so it is fenced off harder than anything else on the box). No web UI, no Caddy route, no published port, never
on `proxy`. LOCAL build from `~/data/sim-agent/src` (a clone of the public GitHub repo, which the container holds
no credential for); the agent's own checkout is `~/data/sim-agent/repo` (`/work/sim`) and its home is
`~/data/sim-agent/home` (`/home/agent`, holds its restricted ssh keys). All three live on
`data/appdata/sim-agent` (refquota-capped). Own bridge `br-simagent` = `apps_sim-agent-net`, `172.30.30.0/24`,
outside the Docker pool. Its firewall is its OWN table, `inet sim_agent` (`/etc/nftables.d/sim-agent.nft`, loaded
by `sim-agent-fw.service` BEFORE docker); the container refuses to start if that table is not dropping its
traffic. Not enrolled in Dockhand auto-update (local build); `update.sh` below rebuilds weekly.
```bash
# State, pause/resume (what the workstation's `agent`/game.sh use, over the restricted ctl key):
docker inspect -f '{{.State.Status}}' sim-agent
docker exec -it sim-agent tmux attach -t agent          # detach: Ctrl-b d
# Rebuild + smoke-test + swap (swaps only if idle and the firewall is up; rolls the tag back on a failed smoke):
bash ~/data/sim-agent/src/tools/agent_container/update.sh
# Update the build source (review first -- the image and the ctl forced command run from it):
git -C ~/data/sim-agent/src fetch origin && git -C ~/data/sim-agent/src log --oneline HEAD..origin/main
git -C ~/data/sim-agent/src merge --ff-only origin/main
# Roll back a bad image (update.sh keeps the previous one as :rollback):
docker image tag sim-agent:rollback sim-agent:local && cd ~/apps && docker compose up -d --no-deps --no-build sim-agent
# Isolation self-check (read-only; all PASS expected):
bash ~/data/sim-agent/src/tools/agent_container/server-stack/verify_isolation.sh
# Firewall: reload after an edit (validate first), and see what the agent tried:
sudo nft -c -f /etc/nftables.d/sim-agent.nft && sudo systemctl reload sim-agent-fw
sudo nft list table inet sim_agent        # counters;  journalctl -k | grep sim-agent-drop
```
- **Never cold-stopped by backup.sh** (it is in `SKIP`): a nightly stop would kill live agent sessions. Rollback is
  the sanoid snapshots of `data/appdata/sim-agent` (`~/data/sim-agent/.zfs/snapshot/`); if the dataset is lost,
  re-clone the repo from Gitea and re-install the keys -- the agent's durable output is what it pushed to Gitea.
- **Keep `nftables.service` disabled** (true box-wide): `/etc/nftables.conf` flushes the whole ruleset, which would
  drop this table (and Docker's) until a reload.
- **Do not `docker compose up -d` the whole project while an agent session is live** if the sim-agent fragment
  changed: it recreates the container. Check first (PATH pinned, absolute binary: the home is model-writable):
  `docker exec -e PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin sim-agent /usr/bin/pgrep -f '/claude$'`
  (empty = idle).
- The fragment in `~/apps/sim-agent/compose.yaml` is a copy of the repo's reviewed one; `update.sh` warns on drift.
  Only `compose.yaml` is tracked there (its own `.gitignore`); secrets are `SIM_AGENT_*` in `~/apps/.env`.
- Gitea: the container pushes only `research/draft-*` on `neural-simulator` with a write deploy key, over
  `192.168.0.10:2222`. Actions must stay OFF for that repo (the act-runner's socket proxy allows POST+EXEC).
