# Agent harness setup on srv-host-0 (owner-applied steps)

This turns `tools/agent_container/` into the daily-driver harness on `homelab-server` (srv-host-0, 192.168.0.10),
deployed the way every other service on that box is: one stack fragment in the `~/apps` compose project, its
secrets in the one `~/apps/.env`, its state on its own ZFS dataset under `${DATA_ROOT}`, its firewall as an
nftables file beside `fw.nft`. Every step is an exact command for the OWNER to run; the agent that built this ran
none of them. **Run them in order**; the harness must not start before step 7 (firewall).

Background: `docs/plans/2026-09-29-local-model-transition-PLAN.md` section 9, and the 2026-09-29 security review
and re-review that hardened this setup. Threat model: the model driving the container may be adversarial; it must
not affect anything beyond its container, its dataset, and the narrow, validated channels below.
`tools/agent_gateway.sh` (workstation) and `tools/agent_harness_ctl.sh` (server) are the two forced-command
scripts the restricted keys point at; their headers say exactly what each allows.

**What changed on 2026-09-30** (branch `research/harness-server-fit`): the first version assumed a generic Docker
host. On srv-host-0 its named volumes landed on `/bulk/docker` (no snapshots, no backup, no cap), `egress.sh`'s
INPUT DROP matched no interface (compose set no bridge name), its 443 rule let the container hairpin into this
box's own `*.dant123.com` services, and the DNS fallback resolver was blocked. This doc replaces the old one
entirely; `egress.sh`, `sim-agent-egress.service`, the standalone `compose.yaml` and the loop-image step are gone.

**Then the 2026-09-30 security review of that refit** (same branch) found that the checks the owner was told to rely
on were computed inside the agent's own container, and fixed or gated each finding: the image PATH no longer starts
with the model-writable home, and nothing that checks the container runs a command from it; the network probes run
from a throwaway pristine container and the host's drop counters are a hard check; `pool-submit` is refused (pool
jobs ran unsandboxed on nodes with passwordless sudo); the container will not start while the hairpin is open, and a
failed WAN refresh alerts; `verify_isolation.sh` fails when Gitea Actions are on, the gitea dataset has no quota, or
the workstation's sshd offers a password; a gate flags agent changes to server and hook code. What is left to the
owner is in the checklist below and in "What the owner must verify" at the end.

**Then the pool-job sandbox** (branch `research/pool-job-sandbox`, 2026-09-30): pool jobs run inside bubblewrap on
the nodes (no network, no-new-privs, the pinned revision read-only, only its output area writable), behind a
`POOL_SANDBOX=auto|on|off` switch on the dispatcher, and the gateway accepts `pool-submit` again only while the live
dispatcher runs with `POOL_SANDBOX=on`. The node steps are in "Pool nodes".

## Where everything lives

| What | On the server | Source of truth (this repo) |
|---|---|---|
| Stack fragment | `~/apps/sim-agent/compose.yaml` (+ `.gitignore`), one `include:` line in `~/apps/compose.yaml` | `tools/agent_container/server-stack/sim-agent/` |
| Build source, ctl script, `update.sh` | `/home/dant123/data/sim-agent/src` (a clone of the GitHub repo; not mounted into the container) | this repo |
| The agent's repo clone | `/home/dant123/data/sim-agent/repo` -> `/work/sim` (uid 61000) | agent-writable |
| The agent's home (Claude Code, keys) | `/home/dant123/data/sim-agent/home` -> `/home/agent` (uid 61000) | agent-writable |
| ZFS dataset for all three | `data/appdata/sim-agent`, refquota 64G, quota 192G, sanoid-snapshotted | step 5 |
| Firewall | `/etc/nftables.d/sim-agent.nft` (table `inet sim_agent`) + `sim-agent-fw.service` | `server-stack/host/etc/...` |
| Own-WAN-address refresher | `/usr/local/sbin/sim-agent-fw-selfwan.sh` + `sim-agent-fw-selfwan.{service,timer}` | `server-stack/host/...` |
| Weekly rebuild | `sim-agent-update.{service,timer}` running `update.sh` | `server-stack/host/etc/systemd/system/` |
| Cloud API keys (optional) | `SIM_AGENT_*` in `~/apps/.env` (0600) | never in git |
| Network | `apps_sim-agent-net`, bridge `br-simagent`, `172.30.30.0/24` (outside the 172.20/16 pool) | the fragment |

**Why the build source is a GitHub clone:** the image, its baked-in guardrails (managed settings, the protected-edit
hook) and the ctl forced command all run from `src/`. The container holds a write key for Gitea, but no credential
at all for GitHub, so nothing the agent does can change what it is built from or what the ctl key executes.
Building from `repo/` (the agent's own checkout) would let it rewrite its own guardrails; never do that.

**What the container can reach** (enforced by `sim-agent.nft`, checked by step 13): DNS at 192.168.0.1 and
1.1.1.1 (the two resolvers pinned in the fragment); the workstation 192.168.0.68 on 22 (restricted keys) and 8080
(the model server); Gitea over ssh at 192.168.0.10:2222; HTTPS (443) to the public internet except this site's own
WAN address. Nothing else: no host listener (sshd, rpcbind, Samba, NFS, upsd, Cockpit), no other container, no
published port, no LAN device, no IPv6, and nothing can connect in. Of the workstation gateway's commands, only
`gpu-submit` (bubblewrap-sandboxed, no network), `status` and `yield` are accepted; `pool-submit` is accepted only
while the pool dispatcher runs with `POOL_SANDBOX=on`, i.e. every pool job runs bubblewrap-sandboxed on the node (see
"Pool nodes").

**Plain-language residual risk (read this before deciding the setup is "done"):** even with every step applied, a
FULLY ROGUE model driving the container could still (1) exfiltrate the repo and any results it can read over the
one egress path that must stay open (HTTPS to the internet: "the Claude Code updater" cannot be told apart from
"any other HTTPS destination" at the IP level this works at); (2) run code on the pool nodes through `pool-submit`,
inside the bubblewrap sandbox (no network, no-new-privs, read-only revision); reaching the LAN or root from there
takes a kernel exploit through unprivileged user namespaces, the same residual as (5); (3) burn CPU/GPU compute and fill its own dataset up to its
refquota (a nuisance, not a compromise); (4) try to
social-engineer the OWNER through what it says in chat/tmux; (5) attempt a kernel or driver exploit to escape the
container: namespaces, cap_drop and seccomp shrink that surface but cannot make it zero; (6) persist code in its
own home (`~/.bashrc`, `~/.local/bin`, `~/.ssh/config`, Claude Code settings) that runs in its NEXT session.
Nothing that CHECKS the container executes from that home (the start guard, the network probes, `update.sh`'s idle
check, the ctl script and `verify_isolation.sh` pin PATH and run only image binaries, and the tmux shell starts
without startup files). The functional smoke test and step 13's gateway `status` call do run home content (the
`claude` binary, the ssh config), inside a container, and are never evidence about isolation. Treat the home as
untrusted; step 11 says how to recreate it from nothing if in doubt.

## The owner checklist (in order)

At a glance (each line is a step below; **no key goes into the container before steps 1, 3 and 10's preconditions
hold**):

1. Workstation sshd: no password logins (today `PasswordAuthentication` is still `yes` there).
2. Generate the four keypairs on the workstation.
3. Gitea: Actions OFF and verified `false` plus a probe push; a ZFS quota on `data/appdata/gitea`; branch and tag
   protection; only then the write deploy key.
4. Retire the 2026-09-29 attempt, including `docker network rm sim-agent_agent-net` (else step 12 fails on the
   172.30.30.0/24 overlap).
5. ZFS dataset `data/appdata/sim-agent`, its two directories, the snapshot policy.
6. Review and merge `research/harness-server-fit`, then clone `src/` from GitHub, so the image that is built and the
   ctl script that runs are the fixed ones.
7. Firewall: install and enable; `self_wan` must hold 1 address (and after every reboot).
8. The stack fragment into `~/apps`.
9. Secrets (optional).
10. Keys into the container's home, host keys pinned, `authorized_keys` lines (preconditions listed there).
11. `backup.sh` SKIP.
12. Build, smoke-test, seed the repo, start.
13. `verify_isolation.sh`: ALL CHECKS PASSED; branch protection probed from the workstation.
14. OWNERSHIP-MAP, weekly rebuild timer.
15. Pool nodes: bubblewrap on every node, the live self-test, then `POOL_SANDBOX=on` on the dispatcher ("Pool
    nodes"); until then the gateway refuses `pool-submit`.

After that, for every agent branch you merge and every `src/` update: "Merging agent branches". Optional site-wide
hardening found on the way: "Optional: site-wide hardening".

### 1. Workstation sshd: no password logins

A restricted forced-command key is meaningless if a password login works instead. In `/etc/ssh/sshd_config` (or a
drop-in under `/etc/ssh/sshd_config.d/`) on the WORKSTATION:

```
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthenticationMethods publickey
```

```bash
sudo systemctl reload sshd
sudo sshd -T | grep -iE '^(passwordauthentication|kbdinteractiveauthentication|authenticationmethods) '
# expect: passwordauthentication no / kbdinteractiveauthentication no / authenticationmethods publickey
```

Keep a working session open and confirm a FRESH key login still works before closing it. (The server already
allows pubkey only, `AllowUsers dant123`.) On 2026-09-30 this workstation still read `passwordauthentication yes`,
and sim-agent.nft lets the container reach 192.168.0.68:22, so until this is applied the account password is an
authentication surface for it. Do this BEFORE step 10. `verify_isolation.sh` (step 13) re-checks it from the
container's network position: it reads the method list the workstation's sshd offers and fails on `password` or
`keyboard-interactive`.

### 2. Generate the four keypairs (on the workstation)

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_container_key    -C "sim-agent container -> workstation gateway"
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_results_sync_key -C "sim-agent container -> results-sync (read-only)"
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_gitea_deploy     -C "sim-agent container -> gitea deploy key"
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_harness_ctl_key  -C "workstation -> sim-agent harness ctl"
```

The first three private halves go INTO the container's home (step 10); the ctl key's private half stays here.

### 3. Gitea: Actions off, a quota, branch protection, THEN the deploy key (web UI, as the admin)

In this order; the deploy key comes last, so no key that can write exists while any of the rest is missing.

1. **MANDATORY (HIGH): turn Actions OFF for `neural-simulator`** (repo Settings, Repository units: untick Actions).
   This box's `[actions] ENABLED = true` instance-wide, and `gitea-act-runner`'s job containers join the shared
   `proxy` network through a docker-socket-proxy that allows POST, EXEC, CONTAINERS, IMAGES, VOLUMES and NETWORKS: a
   workflow file pushed to ANY branch the container can write would run as a container next to every web app, with
   Docker API calls that are root-equivalent on this server. Verify, and re-verify whenever Gitea is upgraded:

   ```bash
   curl -s https://git.dant123.com/api/v1/repos/dant123/neural-simulator | jq .has_actions   # expect: false
   ```

   (On 2026-09-30 this read `true`.) Then prove it with a push: on the workstation, commit a trivial
   `.gitea/workflows/probe.yml` (one `run: echo probe` step) on a throwaway branch `research/draft-actions-probe`,
   push it to `gitea` only, check the repo's Actions page shows no run, then delete that branch on Gitea.
   `verify_isolation.sh` fails while `has_actions` is not `false` (or cannot be read).
2. **MANDATORY: a quota on the gitea dataset.** The deploy key can push unlimited data to unlimited `research/draft-*`
   branches, and branch protection (item 3) stops it deleting them, so only you can clean up; `data/appdata/gitea`
   (2.67G on 2026-09-30, no quota) shares the `data` pool with postgres, immich, clickhouse and every other stack.
   Also cap a single push at 512 MiB:

   ```bash
   sudo zfs set quota=30G data/appdata/gitea
   zfs get -o property,value used,quota data/appdata/gitea
   docker exec -u git gitea git -C /data/git/repositories/dant123/neural-simulator.git config receive.maxInputSize 536870912
   ```

   `verify_isolation.sh` fails without the quota; `update.sh` logs the dataset's use every week and warns past 85%,
   and `ssh homelab-agent-ctl state` reports `gitea_gb` / `gitea_quota_gb` (so does the inspection digest).
3. **Branch protection** (today NO branch on this Gitea instance is protected). Settings, Branches, add rules in
   this priority order: `research/draft-*` first (push allowed to anyone with write access, which includes the
   deploy key; force-push and deletion off), then `*` (push allow-list = your own user only, "allow deploy keys"
   UNticked, force-push off, deletion off). Also protect tags (Settings, Tags: pattern `*`, allowed = your user).
   This is the real enforcement that the container can never touch `main`, any other real branch, or a tag;
   `claude_container_settings.json`'s deny rules are belt-and-braces on top. Step 13 probes it.
4. **Deploy key, last:** Settings, Deploy Keys, add `~/.ssh/sim_agent_gitea_deploy.pub` with **write access** (it
   pushes only `research/draft-*`, per item 3's rules). The container reaches Gitea at `192.168.0.10:2222`: the WAN
   only forwards 80/443, and `git.dant123.com` hairpins through the WAN address, which the firewall drops.

### 4. Retire the 2026-09-29 deployment attempt (on the server)

It sits outside ZFS, git and backup, and its network holds `172.30.30.0/24`: **remove that network before step 12**,
or `docker compose up` fails on the subnet overlap. Nothing references it (no container,
no unit, no `authorized_keys` entry, no cron line). Check, then remove (the volumes hold only a Claude Code install
and an empty repo; look first if you ever copied a key into them):

```bash
docker ps -a --filter network=sim-agent_agent-net                 # expect: no containers
docker run --rm -v sim-agent_sim-agent-home:/h alpine ls -la /h /h/.ssh 2>&1 | head -20
docker network rm sim-agent_agent-net
docker volume rm sim-agent_sim-agent-home sim-agent_sim-agent-repo
rm -rf ~/sim-agent
docker image rm sim-agent:latest                                  # the new image is sim-agent:local
```

### 5. ZFS dataset, directories, snapshot policy

```bash
sudo zfs create -o mountpoint=/home/dant123/data/sim-agent -o refquota=64G -o quota=192G data/appdata/sim-agent
sudo chown dant123:dant123 /home/dant123/data/sim-agent
sudo install -d -o 61000 -g 61000 -m 750 /home/dant123/data/sim-agent/repo
sudo install -d -o 61000 -g 61000 -m 700 /home/dant123/data/sim-agent/home
zfs get -o property,value mountpoint,refquota,quota data/appdata/sim-agent
```

- **refquota=64G** caps LIVE data, sized from the repo as it is today: a fresh clone from Gitea is about 0.5 GB of
  `.git` plus a 1.2 GB checkout; each extra worktree the agent makes adds about 1.2 GB; a full results pull
  (`research/findings/raw` from the workstation) is about 17 GB; a RAG index plus catalog about 1.5 GB; the home
  (Claude Code versions, sessions, caches) 2-5 GB. That is roughly 29 GB, so 64G is about 2x headroom. Past it the
  agent gets ENOSPC inside its own container; nothing else on the pool is affected. Raise it with
  `sudo zfs set refquota=<N>G data/appdata/sim-agent`.
- **quota=192G** (3x) is the backstop against snapshot amplification: refquota does not count snapshots, so an
  agent that writes and deletes tens of GB in a loop would otherwise pin that much pool space per hourly snapshot.
  If it is ever hit, writes fail (EDQUOT) inside the container only, until sanoid prunes or you destroy old
  snapshots of this dataset.
- **uid 61000, not the box's PUID 99:** the image's non-root user (61000, chosen in the 2026-09-29 review so an
  escape never lands as the workstation's uid 1000 or any existing account). A bind mount starts empty (no
  copy-up from the image, unlike a named volume), so these two `install -d` lines are the whole ownership story.
- **Shorter snapshot retention for this dataset** (it churns: worktrees, result pulls, git gc). Append to
  `/etc/sanoid/sanoid.conf`, mirror into `~/apps/host/etc/sanoid/sanoid.conf`, and preview:

  ```ini
  [data/appdata/sim-agent]
  	use_template = production
  	daily = 7
  	monthly = 0
  ```

  ```bash
  sudo sanoid --cron --readonly --verbose 2>&1 | grep -A4 'data/appdata/sim-agent'
  ```

### 6. The build source

Merge `research/harness-server-fit` to `main` first (or add `-b research/harness-server-fit` to trial it): the
2026-09-30 security fixes live there (the image PATH, the absolute interpreters, the PATH-pinned `docker exec`s in
`update.sh`, `agent_harness_ctl.sh` and `verify_isolation.sh`, the pristine-container probes), so the image that gets
built and the ctl script that runs must come from it. Review it like an agent branch ("Merging agent branches"):
it changes every trusted path. A sparse, blobless clone of `tools/` only is all the build and the ctl script need
(about 15 MB):

```bash
git clone --filter=blob:none --sparse https://github.com/danthi123/neural-simulator.git /home/dant123/data/sim-agent/src
git -C /home/dant123/data/sim-agent/src sparse-checkout set tools
git -C /home/dant123/data/sim-agent/src log -1 --oneline
```

Update it only deliberately; the image and the ctl forced command run from it, as you (docker group, sudo) or as
root at boot. Read the trusted paths' diff first, then fast-forward:

```bash
S=/home/dant123/data/sim-agent/src
git -C $S fetch origin && git -C $S log --oneline HEAD..origin/main
git -C $S diff HEAD..origin/main -- tools/agent_container tools/agent_harness_ctl.sh tools/agent_gateway.sh \
    tools/agent_gateway_submit.py tools/agent_gateway_parse.py        # read every line
git -C $S merge --ff-only origin/main
```

`update.sh` refuses to build if this checkout's `tools/agent_container` or `tools/agent_harness_ctl.sh` has any
modified, untracked or ignored file.

### 7. The firewall (before the stack ever starts)

Mirror into `~/apps/host/` (like `fw.nft`), install root-owned from the mirror, validate, enable:

```bash
S=/home/dant123/data/sim-agent/src/tools/agent_container/server-stack/host
cp -rn $S/etc $S/usr ~/apps/host/                      # adds the sim-agent* files only; -n never overwrites
sudo install -o root -g root -m 644 ~/apps/host/etc/nftables.d/sim-agent.nft /etc/nftables.d/sim-agent.nft
sudo install -o root -g root -m 644 ~/apps/host/etc/systemd/system/sim-agent-fw.service \
     ~/apps/host/etc/systemd/system/sim-agent-fw-selfwan.service ~/apps/host/etc/systemd/system/sim-agent-fw-selfwan.timer \
     ~/apps/host/etc/systemd/system/sim-agent-update.service ~/apps/host/etc/systemd/system/sim-agent-update.timer \
     /etc/systemd/system/
sudo install -o root -g root -m 755 ~/apps/host/usr/local/sbin/sim-agent-fw-selfwan.sh /usr/local/sbin/
sudo nft -c -f /etc/nftables.d/sim-agent.nft && echo syntax-ok
sudo systemctl daemon-reload
sudo systemctl enable --now sim-agent-fw.service sim-agent-fw-selfwan.timer
sudo /usr/local/sbin/sim-agent-fw-selfwan.sh           # expect: self_wan set holds 1 address(es)
sudo nft list set inet sim_agent self_wan               # expect: exactly 1 element
systemctl show -p OnFailure --value sim-agent-fw-selfwan.service   # expect: ntfy-notify@...
sudo nft list table inet sim_agent | head -40
```

**After this step and after every reboot**, `sudo nft list set inet sim_agent self_wan` must hold 1 address; after a
DDNS change, check `journalctl -u sim-agent-fw-selfwan` once. A failed refresh now alerts through `ntfy-notify@`,
and the container refuses to start while the WAN address is not dropped (below), so an empty set costs a delayed
start, never an open hairpin.

How it fits the box's firewall:

- **Its own table, `inet sim_agent`**, recreated atomically on every load exactly like `fw.nft` (add, delete,
  redefine); it never touches `inet fw` or Docker's tables and never flushes the ruleset. `nft-fw.service` loads
  only `fw.nft`, so this file has its own unit rather than a glob.
- **Why it beats `fw.nft`'s `iifname "br-*" accept`:** nftables runs every base chain on a hook in priority order,
  and a drop is final while an accept is not. This table's input and forward chains sit at `priority filter - 10`,
  before `fw.nft` (0) and Docker's iptables-nft FORWARD (0), so it can only take access away. No `fw.nft` edit is
  needed. (If you would rather follow the in-file convention, the equivalent is one line in `fw.nft` before its
  L32, `iifname "br-simagent" drop`, plus this file's forward chains, which `fw.nft` does not have.)
- **Boot order:** it matches a fixed bridge NAME (`br-simagent`) and SUBNET, both of which can sit in the ruleset
  before the interface exists, so `sim-agent-fw.service` runs `Before=docker.service`: no window in which the
  container (restart `unless-stopped` + live-restore) runs unfiltered. Docker only *wants* the unit, never requires
  it, so a broken edit can never keep every other stack down. The container covers that case itself:
  `bin/firewall_guard.sh` (run first by the entrypoint) refuses to start unless a probe of this host's sshd through
  the bridge gateway is DROPPED.
- **Hairpin:** every `*.dant123.com` name resolves to the WAN address and the router hairpins it back to Caddy
  with a LAN-looking source, which Caddy's private-range rules treat as a LAN client. The `self_wan` set holds that
  address (resolved at runtime by `sim-agent-fw-selfwan.sh` every 5 minutes; DDNS can change it, and this public
  repo never contains it), and the container's HTTPS rule excludes it. The set is filled best-effort at boot and by
  the timer a minute later, so `bin/firewall_guard.sh` also probes that address on 443 (resolved from
  `SIM_AGENT_FW_WAN_NAME` in the fragment, never printed) and refuses to start the container until it is DROPPED;
  the restart policy retries. `sim-agent-fw-selfwan.service` alerts on failure (`OnFailure=ntfy-notify@%n.service`).
  If a compose-level systemd unit for this stack is ever added, give it `Requires=`/`After=sim-agent-fw-selfwan.service`.
- `/etc/nftables.d/` and `fw.nft` are owned by `dant123`, not root, although root loads them at boot. Because the
  DIRECTORY is dant123's, that account could also replace the root-owned `sim-agent.nft` (unlink + recreate needs
  only write access to the directory). Recommended: `sudo chown root:root /etc/nftables.d /etc/nftables.d/fw.nft`
  (edits then go through the `~/apps/host` mirror + `sudo install`, as for `backup.sh`).
- `nftables.service` must stay disabled: `/etc/nftables.conf` runs `flush ruleset`.

### 8. The stack fragment

```bash
S=/home/dant123/data/sim-agent/src/tools/agent_container/server-stack
mkdir -p ~/apps/sim-agent
install -m 644 $S/sim-agent/compose.yaml $S/sim-agent/.gitignore ~/apps/sim-agent/
$EDITOR ~/apps/compose.yaml          # add "  - sim-agent/compose.yaml" to include:, alphabetically (after searxng)
cd ~/apps && docker compose config --quiet && docker compose config --services | grep -x sim-agent
git -C ~/apps status --short sim-agent/                 # expect only compose.yaml and .gitignore
```

Per `new-service/README.md`: no Caddy route (no web UI), no `proxy` network, and do **not** enroll it in Dockhand
auto-update (local build; not enrolling is how an image is held back). Its Homepage tile (status only) appears
from the labels. `check-runtime-state.sh` needs nothing. The fragment's own `.gitignore` keeps everything except
`compose.yaml` out of the nightly `apps-gitsync` commit: keys and profiles live on the dataset, never here.

### 9. Secrets (optional: only for the cloud mechanical-tier profiles)

In `~/apps/.env` (keep it 0600), under this stack's own prefix; the fragment maps them in with `${VAR:-}` (a
`:?` in an included fragment would break `docker compose` for every stack whenever the key is unset):

```
SIM_AGENT_DEEPSEEK_API_KEY=...
SIM_AGENT_GLM_API_KEY=...
```

Never `env_file:` the root `.env` into this container: that would hand it every other stack's secrets. Before
routing real work to a cloud profile, validate it from the workstation with `tools/local_llm/bakeoff.py` (see
`tools/agent_container/profiles/deepseek-flash.env`'s header); both profiles are `KIND=cloud`, so the gaming yield
skips pausing the harness for them.

### 10. Keys into the container's home, host keys pinned

**Preconditions, all of them, before the first key goes in:** step 1 applied and `sshd -T` confirmed; step 3's
Actions check reads `false`, the probe push produced no run, the gitea quota is set; and the gateway refuses
`pool-submit` unless the pool sandbox is mandatory (confirm on the WORKSTATION):

```bash
cd /home/dant123/Projects/sim
cat research/queue/.pool_sandbox_state      # mode=... as published by the live dispatcher each cycle
env SSH_ORIGINAL_COMMAND='pool-submit --sha aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa --runner research.runners.x --checked y -- --seed 1' AGENT_GATEWAY_LOG=/dev/null bash tools/agent_gateway.sh
# expect, unless mode=on: refused: pool-submit needs the pool sandbox to be mandatory ... (exit status 1)
.venv/bin/python -m pytest -q tests/test_agent_gateway.py tests/test_agent_gateway_submit.py tests/test_pool_job_sandbox.py
```

From the WORKSTATION, streaming each private key straight into place (it never lands anywhere else):

```bash
ssh homelab-server "sudo install -d -o 61000 -g 61000 -m 700 /home/dant123/data/sim-agent/home/.ssh"
ssh homelab-server "sudo sh -c 'umask 077; cat > /home/dant123/data/sim-agent/home/.ssh/id_ed25519 && chown 61000:61000 /home/dant123/data/sim-agent/home/.ssh/id_ed25519'" < ~/.ssh/sim_agent_container_key
ssh homelab-server "sudo sh -c 'umask 077; cat > /home/dant123/data/sim-agent/home/.ssh/id_ed25519_results_sync && chown 61000:61000 /home/dant123/data/sim-agent/home/.ssh/id_ed25519_results_sync'" < ~/.ssh/sim_agent_results_sync_key
ssh homelab-server "sudo sh -c 'umask 077; cat > /home/dant123/data/sim-agent/home/.ssh/id_ed25519_gitea && chown 61000:61000 /home/dant123/data/sim-agent/home/.ssh/id_ed25519_gitea'" < ~/.ssh/sim_agent_gitea_deploy
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub        # note the workstation's host key fingerprint
```

On the SERVER, the ssh config and pinned host keys:

```bash
S=/home/dant123/data/sim-agent/src/tools/agent_container/server-stack
H=/home/dant123/data/sim-agent/home/.ssh
sudo install -o 61000 -g 61000 -m 600 $S/agent-ssh-config $H/config
{ ssh-keyscan -t ed25519 192.168.0.68; ssh-keyscan -t ed25519 -p 2222 192.168.0.10; } 2>/dev/null | sudo tee $H/known_hosts >/dev/null
sudo chown 61000:61000 $H/known_hosts
sudo ssh-keygen -lf $H/known_hosts                       # compare with the two below
sudo ssh-keygen -lf /home/dant123/data/gitea/ssh/ssh_host_ed25519_key.pub   # gitea's host key
```

The `authorized_keys` lines. On the WORKSTATION (`~/.ssh/authorized_keys`), the two container keys, accepted only
from the server's LAN address (container traffic leaves masqueraded as 192.168.0.10):

```
command="bash /home/dant123/Projects/sim/tools/agent_gateway.sh",restrict,from="192.168.0.10" ssh-ed25519 AAAA...container-pubkey... sim-agent container -> workstation gateway
command="rrsync -ro -no-del /home/dant123/Projects/sim/research/findings/raw",restrict,from="192.168.0.10" ssh-ed25519 AAAA...results-sync-pubkey... sim-agent container -> results-sync (read-only)
```

`rrsync` (rsync's restricted wrapper) confines the pull to that one path, read-only, no delete, so no script here
parses rsync's protocol arguments (the 2026-09-29 H3 fix; `agent_gateway.sh` refuses `results-sync`). Symlinks
planted under `research/findings/raw/` are copied as inert links, never followed, as long as the container's pull
never adds `--copy-links`.

On the SERVER (`~/.ssh/authorized_keys`), the ctl key, run from the owner's checkout on the dataset (never from
anything the container can write), accepted only from the workstation:

```
command="bash /home/dant123/data/sim-agent/src/tools/agent_harness_ctl.sh",restrict,from="192.168.0.68" ssh-ed25519 AAAA...ctl-pubkey... workstation -> sim-agent harness ctl
```

And on the WORKSTATION, `~/.ssh/config`:

```
Host homelab-agent-ctl
    HostName 192.168.0.10
    User dant123
    IdentityFile ~/.ssh/sim_agent_harness_ctl_key
    IdentitiesOnly yes
```

### 11. Backups: never cold-stop the agent

`backup.sh` cold-stops every included stack around 01:30, which would kill a live agent session. Add
`sim-agent` to its `SKIP` set (L92), keeping the line's comment; edit the `~/apps/host` copy, then install it:

```bash
sed -i 's/^\(SKIP = set("saythatsh-staging saythatsh-monitor nextcloud\)"/\1 sim-agent"/' ~/apps/host/usr/local/sbin/backup.sh
grep -n '^SKIP' ~/apps/host/usr/local/sbin/backup.sh   # expect: ... nextcloud sim-agent".split())  # saythatsh (prod) ...
sudo install -m 755 ~/apps/host/usr/local/sbin/backup.sh /usr/local/sbin/backup.sh
sudo /usr/local/sbin/backup.sh --dry-run | grep -c 'PLAN sim-agent'   # expect: 0
```

Rollback is the sanoid snapshots of `data/appdata/sim-agent` (hourly/daily, step 5):
`/home/dant123/data/sim-agent/.zfs/snapshot/<snap>/` (browse read-only, copy out). The new `/etc/nftables.d`,
`/etc/systemd/system` and `/usr/local/sbin` files are picked up by `backup.sh`'s system leg automatically.

**What is lost if the dataset itself is destroyed, and why that is acceptable:**

- `repo/`: the agent's clone. Re-clone from Gitea (step 12). Lost: only what the agent had not pushed (uncommitted
  edits, unpushed `research/draft-*` commits, its copy of pulled results, which re-pull from the workstation).
- `home/`: Claude Code (the entrypoint reinstalls it at the next start), the three private keys, ssh config and
  known_hosts (step 10 again, from the workstation's copies), Claude Code's session transcripts and memory, logs.
- `src/`: re-clone from GitHub (step 6).

This is acceptable because the harness is disposable by design: the agent's durable output is what it pushes to
Gitea and the results that live on the workstation, and its local state is untrusted anyway (the threat model
assumes it may be adversarial). Snapshots cover every ordinary mistake; only a loss of the whole `data` pool takes
them with it, and then the rebuild is steps 5, 6, 10 and 12.

### 12. Build, smoke-test, seed the repo, start

```bash
cd ~/apps
docker compose build sim-agent
docker compose run --rm --no-deps sim-agent smoke_test.sh     # expect: ALL PASS/SKIP (import sim SKIPs until seeded)
docker compose run --rm --no-deps sim-agent git clone -o gitea git@gitea-deploy:dant123/neural-simulator.git /work/sim
docker compose up -d sim-agent
docker logs sim-agent 2>&1 | tail -6                          # expect: two "firewall_guard: OK ..." lines, then "ready."
```

The clone's remote is named `gitea`, which the container's settings allow pushing to (`git push origin` is
denied). The smoke test is FUNCTIONAL only (that container mounts the model-writable home, so nothing it reports is
evidence about isolation); the network is probed from a pristine container by step 13 and by every weekly
`update.sh` run. A missing firewall, or an open hairpin, makes the run refuse to start at all.

### 13. Verify the isolation (and again after any firewall, Docker, Gitea or fragment change)

```bash
bash /home/dant123/data/sim-agent/src/tools/agent_container/server-stack/verify_isolation.sh   # expect: ALL CHECKS PASSED
```

Read-only, and nothing in it is computed inside the agent's container. Host side: both firewall units active and
enabled, `sim-agent-fw-selfwan.service` has `OnFailure=`, `nftables.service` not enabled, the table loaded,
`self_wan` non-empty, `br-simagent` present, `apps_sim-agent-net` = `172.30.30.0/24` with IPv6 off, Gitea
`has_actions` = `false` (unreadable fails), `data/appdata/gitea` has a quota, the container on that network only,
no published ports. From `docker inspect` (Docker's record, not the container's): every capability dropped and none
added, not privileged, read-only root, `no-new-privileges`, no host namespace, DNS pinned, exactly the two bind
mounts, no docker socket. Two fresh PATH-pinned execs: uid 61000, `CapEff` 0 and `NoNewPrivs` 1.

Then the network, from a THROWAWAY PRISTINE container (`probe_from_pristine.sh`: the image alone on
`apps_sim-agent-net`, no mounts, read-only, no capabilities), which a rogue process in the agent's container cannot
reach. Each must be DROPPED (a timeout; a connection or a refusal fails): this host via the gateway and via
192.168.0.10 on 22/111/139/445/2049/3493, the published ports 1880/18443/11000/8554, the router's UI, AdGuard at
192.168.0.5, the workstation's 80/445, docker0 and the proxy bridge gateway, both docker-socket-proxies and the gitea
and caddy containers by IP, 8.8.8.8:53, 1.1.1.1:80, and `https://git.dant123.com` (the hairpin, which must time out).
Must work: DNS at both resolvers, 192.168.0.68:22 and :8080 (a refusal is fine if llama-swap is stopped),
192.168.0.10:2222, HTTPS to GitHub and claude.ai. The workstation's sshd must offer no password method, and there
is no routable IPv6. Finally the host's own drop counters for `sim-agent -> host`, `LAN, other containers`,
`everything else` and the hairpin must have GROWN during the probes: the authoritative signal that this table, not
luck, dropped them.

Then the two channels, and branch protection, probed from the WORKSTATION with the deploy key itself (not from
inside the container, whose git config and hooks are model-writable), hooks off:

```bash
ssh homelab-agent-ctl state                              # from the workstation: appdata_gb / gitea_gb / firewall: active
ssh homelab-server "docker exec -e PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin sim-agent /usr/bin/ssh workstation-gateway status"   # the gateway answers
cd /home/dant123/Projects/sim
env GIT_SSH_COMMAND="ssh -i $HOME/.ssh/sim_agent_gitea_deploy -o IdentitiesOnly=yes" git -c core.hooksPath=/dev/null push ssh://git@192.168.0.10:2222/dant123/neural-simulator.git HEAD:refs/heads/research/draft-protection-probe   # expect: accepted
env GIT_SSH_COMMAND="ssh -i $HOME/.ssh/sim_agent_gitea_deploy -o IdentitiesOnly=yes" git -c core.hooksPath=/dev/null push ssh://git@192.168.0.10:2222/dant123/neural-simulator.git HEAD:refs/heads/protection-probe            # expect: REJECTED
env GIT_SSH_COMMAND="ssh -i $HOME/.ssh/sim_agent_gitea_deploy -o IdentitiesOnly=yes" git -c core.hooksPath=/dev/null push ssh://git@192.168.0.10:2222/dant123/neural-simulator.git HEAD:refs/tags/protection-probe-tag         # expect: REJECTED
```

Delete `research/draft-protection-probe` in the Gitea UI afterwards (and anything else that was accepted: then
step 3's rules are wrong, fix them before going further).

### 14. OWNERSHIP-MAP, weekly rebuild

- Paste `tools/agent_container/server-stack/OWNERSHIP-MAP.sim-agent.md` into `~/apps/OWNERSHIP-MAP.md` (it says
  where), then regenerate `docs/SYSTEM-REFERENCE.md` with `docs/gen-system-reference.py` as usual.
- The weekly rebuild (Sunday 04:00, between `backup.timer` and `apps-gitsync.timer`), installed in step 7:
  `sudo systemctl enable --now sim-agent-update.timer`. See "Updates and rollback" below.
- Unrelated stale docs noticed on the way (your call): `docs/SYSTEM-REFERENCE.md` L91/L96 and
  `new-service/README.md` L46 still name `restic-backup.timer` as the nightly backup (it is `backup.timer`), and
  the proxy compose comment gives `nextcloud-aio` as 172.18/16 (it is 172.20.15.0/24).

## Updates and rollback

`tools/agent_container/update.sh` (run weekly by `sim-agent-update.timer`, or by hand as `dant123`) does one cycle:
refuse unless `~/apps` includes the stack and the build source is a clean checkout; warn if the `~/apps` fragment
drifted from the reviewed copy; log disk usage of the dataset and Docker's data-root; tag the current image
`sim-agent:rollback`; `docker compose build --pull sim-agent`; if the image changed, run the (functional) smoke
test against the NEW image and, on failure, point `sim-agent:local` back at the rollback image and stop. Every cycle
it then probes the isolation from a pristine container (`probe_from_pristine.sh`, the targets that do not depend on
the workstation being awake); a failure rolls a changed image back, never swaps, and fails the unit even when the
image did not change, so it is also a weekly regression check of the firewall. It logs the gitea dataset's use
against its quota. It swaps the running container (`docker compose up -d --no-deps --no-build sim-agent`) only when
it is running, no `claude` process is active and `sim-agent-fw.service` is active. A paused harness (a gaming yield froze a session, maybe mid-task) is
deferred, not recreated; a stopped one is never started. It logs to `/home/dant123/data/sim-agent/update.log`;
failures alert through `ntfy-notify@`.

Manual rollback: `docker image tag sim-agent:rollback sim-agent:local && cd ~/apps && docker compose up -d
--no-deps --no-build sim-agent`.

Claude Code itself lives in the home bind mount and is kept current as before: every `agent start` runs
`claude-safe-update.sh` (update, smoke-test against the chosen backend, roll back to the last good version on
failure). Its own auto-updater is never disabled.

Never `docker compose up -d` the whole `~/apps` project while an agent session is live if the sim-agent fragment
changed: it recreates the container. Check first, PATH pinned and the binary absolute (empty = idle):
`docker exec -e PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin sim-agent /usr/bin/pgrep -f '/claude$'`.
Everything you run INTO the container should look like that: the image PATH no longer includes the home, but an
explicit PATH and an absolute binary keep a check from ever depending on it.

## Everyday use (the `agent` fish function, workstation-side)

Already installed at `~/.config/fish/functions/agent.fish`: `agent` attaches, `agent start [profile]` starts a
session, `agent stop`, `agent logs`. `agent start` first records the profile on the SERVER through the ctl key's
`set-active-profile` (so `state`'s gaming-pause decision never trusts anything the container wrote), then types
the start command into the container's tmux session. It builds the remote command as ONE pre-escaped string, so
`ssh` has nothing to re-join:

```fish
case start
    set -l profile $argv[2]
    if test -z "$profile"
        set profile workstation
    end
    ssh homelab-agent-ctl "set-active-profile $profile"
    or echo "agent: warning -- could not record the active profile on the server (state's gaming-pause check may be stale)"
    set -l inner (string escape -- "bash /usr/local/bin/agent-start.sh $profile")
    set -l remote_cmd "docker exec -e PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin sim-agent /usr/bin/tmux send-keys -t agent $inner Enter"
    ssh homelab-server $remote_cmd
    and echo "agent: sent 'agent-start.sh $profile' into the tmux session -- attach with: agent"
```

Update the installed function to match (its `docker exec` pins PATH and names tmux absolutely). Every `docker
exec sim-agent ...` in this doc runs ON THE SERVER; the container does not exist on the workstation. The tmux
session's shell starts with no startup files (`bash --noprofile --norc`, `tmux -f /dev/null`) and has the home's
`~/.local/bin` LAST on its PATH, where `claude` lives.

## Optional: restrict who can reach the workstation's model port

`tools/local_llm/llm.sh` runs llama-swap on `0.0.0.0:8080` on the WORKSTATION (see the comment above
`llama_swap_cmd()` for why one narrower bind cannot serve both loopback and the LAN). `sim-agent.nft` already
confines the CONTAINER, but any other LAN device can still reach 8080 (low severity: no credentials, just
inference and model selection). On the WORKSTATION:

```bash
sudo iptables -I INPUT 1 -i lo -p tcp --dport 8080 -j ACCEPT
sudo iptables -I INPUT 2 -s 192.168.0.10 -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP
```

Verify from another LAN device (`curl -m 3 http://192.168.0.68:8080/health` times out) and from the server (it
answers). These do not survive a reboot unless saved with the workstation's own firewall mechanism.

## Pool nodes (the sandbox `pool-submit` requires)

**Why.** The 2026-09-30 security review found `tools/pool_autodispatch.sh` running every pool job over ssh as the
node's login user, with passwordless sudo and full network: an agent-authored line (`pool-submit`, or the local-model
driver's `tools/pool_queue.sh add`) was a LAN foothold that bypassed everything `sim-agent.nft` confines, next to a
server that trusts the LAN (SMB guest shares, rpcbind/NFS, upsd, published ports).

**What runs now.** For a sandboxed launch the dispatcher sends `tools/pool_job_sandbox.sh` as the remote wrapper. On
the node it builds a bubblewrap sandbox from an empty root, the recipe `gpu-submit` uses: `--unshare-all` (no
network at all, no view of other processes, no host sockets), no-new-privs (sudo cannot work), `--clearenv` with an
allow-list, a fresh `/tmp` and a fresh tmpfs over the node's HOME (so `~/.ssh` and everything else there is absent),
read-only `/usr`, six `/etc` entries, the pinned `~/derisk-pool/revisions/<sha>` (source, `.venv`, `data/corpus`)
and `~/Projects/sim-data`. Writable: only `<revision>/research/findings/raw` (a gateway-built agent line: only its
own `_agent_jobs/<id>`). `memcap.sh` N becomes a `MemoryMax=N G` scope around the whole sandbox, `mem_ok.sh` reads
a snapshot of the node's scopes, and an agent line always gets `MemoryMax=8G`, `TasksMax=512` and a 12 h timeout.
Refusals are rc 126 in `job_status.log`, never an unsandboxed fallback. The file's header has the full recipe.

**The switch** (`POOL_SANDBOX`, read by the dispatcher at start): `auto` (default) sandboxes on every node whose
probe answers `POOL_SANDBOX_READY` and launches unsandboxed, with a loud `⚠ UNSANDBOXED` line per dispatch, on a node
that does not; `on` gives an unready node no work at all (jobs stay queued); `off` is the old launch everywhere.
Whatever the mode, a line carrying `AGENT_JOB_SRC=sim-agent` is never popped for an unsandboxed launch. The live loop
publishes its mode every cycle to `research/queue/.pool_sandbox_state`; `tools/agent_gateway.sh` accepts
`pool-submit` only while that file says `mode=on` and is under 15 minutes old.

**Node readiness found on 2026-09-30** (pool41 and pool42, Ubuntu 22.04.5, kernel 5.15): unprivileged user
namespaces are enabled (`kernel.unprivileged_userns_clone=1`, no AppArmor userns restriction on 22.04, `unshare
--user --map-root-user --net true` succeeds), but bubblewrap is NOT installed. So today every node probes as not
ready and `auto` launches exactly as before.

**Owner steps, in order:**

1. On each node (pool41, pool42, and any node added to `tools/pool_nodes.conf` later), install bubblewrap and
   confirm it starts a sandbox as the job user (expect `ok`):

   ```bash
   ssh pool41 'sudo apt-get update && sudo apt-get install -y bubblewrap'
   ssh pool41 'bwrap --unshare-all --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 --proc /proc --dev /dev -- /bin/sh -c "grep -q \"^NoNewPrivs:.*1\" /proc/self/status && echo ok"'
   ```

   Leave `kernel.unprivileged_userns_clone` at 1. On a node later upgraded to Ubuntu 24.04+, check
   `sysctl kernel.apparmor_restrict_unprivileged_userns`: at 1 it blocks bubblewrap unless an AppArmor profile
   grants `/usr/bin/bwrap` the `userns` permission; the `ok` check above fails loudly if so.
2. From the workstation, the live self-test on each node, against any revision already provisioned there (it runs
   through the exact sandboxed launch, not the queue; exit 0 = every check passed):

   ```bash
   cd /home/dant123/Projects/sim/.claude/worktrees/pool-job-sandbox   # /home/dant123/Projects/sim once merged
   bash tools/pool_autodispatch.sh --sandbox-probe pool41        # POOL_SANDBOX_READY bwrap=...
   bash tools/pool_sandbox_selftest.sh pool41 <provisioned-sha>
   bash tools/pool_sandbox_selftest.sh pool41 <provisioned-sha> --job '<one queued dev-seed-7 line, without #checked>'
   ```

   It checks from inside the sandbox that 192.168.0.10:445, 1.1.1.1:443 and DNS fail, sudo fails with no-new-privs
   set, the revision and `/usr` are read-only, `~/.ssh` and the node's other processes are invisible, then pulls the
   report back with `tools/pool_sync.sh`.
3. Once `research/pool-job-sandbox` is merged, make the sandbox mandatory on the dispatcher and restart it (safe at
   any time: jobs run on the nodes, not under the dispatcher):

   ```bash
   mkdir -p ~/.config/systemd/user/pool-dispatch.service.d
   printf '[Service]\nEnvironment=POOL_SANDBOX=on\n' > ~/.config/systemd/user/pool-dispatch.service.d/sandbox.conf
   systemctl --user daemon-reload && systemctl --user restart pool-dispatch.service
   grep 'POOL_SANDBOX=' research/queue/dispatch.log | tail -1; cat research/queue/.pool_sandbox_state   # mode=on
   ```

   Then queue one dev-seed-7 line with `FRONT=1` and `max_age_h=` in its `--checked` text, and confirm its result
   arrives with the next `tools/pool_sync.sh`, before trusting the backlog to it.
4. Optional defence in depth: a dedicated job user with no sudo on each node (`sudo useradd -m -s /bin/bash
   simjob`), the dispatcher's ssh target switched to it, and the NOPASSWD entry kept for the provisioning account
   only (`sudo grep -rl NOPASSWD /etc/sudoers.d/`). The sandbox already makes sudo unusable from a job; this also
   covers a sandbox escape short of a kernel exploit.

Re-run `tests/test_pool_job_sandbox.py`, `tests/test_agent_gateway.py` and `tests/test_agent_gateway_submit.py`
after any change to the dispatcher, the wrapper or the gateway.

## Merging agent branches (and updating `src/`)

The agent's only route into `main` is a `research/draft-*` branch you merge. What the server runs as you (docker
group, sudo) or as root at boot is built from `src/`, which tracks GitHub `main`; on the workstation,
`core.hooksPath=tools/githooks` and `.claude/settings.json` hooks run `tools/githooks/**`, what those execute, and
`.claude/**` at your next commit or Claude session. So, for every agent branch:

1. Never check it out, or `git worktree add` it, with hooks enabled; review it without a checkout:

   ```bash
   cd /home/dant123/Projects/sim && git fetch gitea
   .venv/bin/python -m tools.gates.agent_trust_paths gitea/research/draft-<topic>    # exit 1 = trusted paths changed
   git diff main...gitea/research/draft-<topic> -- tools/agent_container tools/agent_harness_ctl.sh \
       'tools/agent_gateway*' tools/githooks tools/gates .claude tools/push_both.sh   # read every line
   ```

   The CLI runs `main`'s own copy of the gate, so a branch that edits the gate cannot change this answer.
2. Merge with `--no-ff` (never a fast-forward, a rebase or a cherry-pick: those run no commit hook), acknowledging
   the exact tip you reviewed: `SIM_AGENT_TRUST_ACK=<that sha> git merge --no-ff gitea/research/draft-<topic>`.
   Without the acknowledgement the `agent_trust_paths` gate blocks a merge whose agent commits touch a trusted path.
   It is a tripwire, not the control: a merge already runs the INCOMING hook code, which is why step 1 comes first.
3. On the server, `src/` moves only after the same review (step 6's `git diff HEAD..origin/main -- ...` before
   `merge --ff-only`).

## Optional: site-wide hardening (found on the way, independent of sim-agent)

- **Gitea act-runner:** its job containers join the shared `proxy` network (`container.network: "proxy"` in
  `/home/dant123/data/gitea-act-runner/config.yaml`). Give them a dedicated network, so a workflow in ANY repo is not
  a container next to every web app.
- **homepage-dockerproxy** sits on `proxy` with `CONTAINERS=1`, which exposes every container's environment (every
  stack's secrets) to any compromised container on that network. Move it off `proxy`, or set `CONTAINERS=0` if the
  widgets do not need it.
- **Caddy's LAN privilege:** hairpinned traffic reaches Caddy with a LAN-looking source (usually the router's LAN
  address; confirm in Caddy's access log with a request to a `*.dant123.com` name from a LAN host). Exclude that
  source from the private-range matcher that grants LAN-only access (`not remote_ip <that address>`), so a hairpin
  is never treated as a LAN client, whoever makes it.
- **`/etc/nftables.d` ownership:** `sudo chown root:root /etc/nftables.d /etc/nftables.d/fw.nft` (step 7).

## What the owner must verify (2026-09-30 security review)

Nothing on the server was modified by that review, and `~/apps/.env` was never read (only its mode); the rest of the
evidence came from read-only `sudo -n` reads of nft, exports, samba, `sshd -T`, zfs, sanoid, systemd units, compose
fragments (secret values redacted) and the act-runner `config.yaml`. These are the items only you can close:

1. Before installing the container's gateway key: `pool-submit` is refused by `tools/agent_gateway.sh` (step 10
   shows the check) until the live dispatcher runs with `POOL_SANDBOX=on`, which needs "Pool nodes" steps 1-3 on
   every node; re-run the gateway and sandbox tests after any change. It was the one designed channel that reached
   the LAN.
2. Build the image and run the ctl script from a `src/` that contains the PATH and absolute-interpreter fix
   (`tools/agent_container/Dockerfile`, `entrypoint.sh`, `bin/firewall_guard.sh`) and the `docker exec -e PATH=...`
   change (`update.sh`, `agent_harness_ctl.sh`, `verify_isolation.sh`): merge this branch BEFORE step 6's clone. The
   in-container network probes are gone; they run from a pristine container on `apps_sim-agent-net`.
3. Gitea: Actions unticked and `curl -s https://git.dant123.com/api/v1/repos/dant123/neural-simulator | jq
   .has_actions` printing `false` (it printed `true`), the probe push, and only THEN the write deploy key; the two
   branch rules in the stated priority order and tags `*` protected (step 3).
4. Workstation: sshd step 1 applied (`PasswordAuthentication` was still `yes`), confirmed with
   `sudo sshd -T | grep -iE '^(passwordauthentication|kbdinteractiveauthentication|authenticationmethods) '`.
5. After step 7 and every reboot: `sudo nft list set inet sim_agent self_wan` holds 1 address; the installed
   `sim-agent-fw-selfwan.service` carries `OnFailure=ntfy-notify@%n.service`; `journalctl -u sim-agent-fw-selfwan`
   checked once after a DDNS change.
6. A ZFS quota on `data/appdata/gitea` (20-30G) before the deploy key is added; `receive.maxInputSize` kept.
7. The old network `sim-agent_agent-net` removed (step 4) before step 12, or `docker compose up` fails on the
   172.30.30.0/24 overlap.
8. When merging `research/harness-server-fit` and any later agent branch: the explicit diff of
   `tools/agent_container/**`, `tools/agent_harness_ctl.sh`, `tools/agent_gateway*`, `server-stack/host/**`,
   `tools/githooks/**` and `.claude/**` before the merge and before `git -C src merge --ff-only origin/main`
   ("Merging agent branches"; the `agent_trust_paths` gate backs it up).
9. Optional but recommended site-wide: the act-runner's job containers off `proxy`, and homepage-dockerproxy reduced
   or moved ("Optional: site-wide hardening").
