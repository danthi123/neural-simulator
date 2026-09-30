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
published port, no LAN device, no IPv6, and nothing can connect in.

**Plain-language residual risk (read this before deciding the setup is "done"):** even with every step applied, a
FULLY ROGUE model driving the container could still (1) exfiltrate the repo and any results it can read over the
one egress path that must stay open (HTTPS to the internet: "the Claude Code updater" cannot be told apart from
"any other HTTPS destination" at the IP level this works at); (2) if the RECOMMENDED pool-node sudo fix below is not
applied, get root on the mini-PC pool nodes through a pool job and use that as a foothold on the LAN; (3) burn
CPU/GPU compute and fill its own dataset up to its refquota (a nuisance, not a compromise); (4) try to
social-engineer the OWNER through what it says in chat/tmux; (5) attempt a kernel or driver exploit to escape the
container: namespaces, cap_drop and seccomp shrink that surface but cannot make it zero.

## The owner checklist (in order)

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
allows pubkey only, `AllowUsers dant123`.)

### 2. Generate the four keypairs (on the workstation)

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_container_key    -C "sim-agent container -> workstation gateway"
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_results_sync_key -C "sim-agent container -> results-sync (read-only)"
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_gitea_deploy     -C "sim-agent container -> gitea deploy key"
ssh-keygen -t ed25519 -N "" -f ~/.ssh/sim_agent_harness_ctl_key  -C "workstation -> sim-agent harness ctl"
```

The first three private halves go INTO the container's home (step 10); the ctl key's private half stays here.

### 3. Gitea: Actions off, branch protection, deploy key (web UI, as the admin)

1. **MANDATORY (HIGH): turn Actions OFF for `neural-simulator`** (repo Settings, Repository units: untick Actions).
   This box's `[actions] ENABLED = true` instance-wide, and `gitea-act-runner` drives a docker-socket-proxy that
   allows POST and EXEC: a workflow file pushed to ANY branch the container can write would run arbitrary Docker
   API calls, root-equivalent on this server. Verify:

   ```bash
   curl -s https://git.dant123.com/api/v1/repos/dant123/neural-simulator | jq .has_actions   # expect: false
   ```

   (On 2026-09-30 this read `true`.) Then prove it with a push: on the workstation, commit a trivial
   `.gitea/workflows/probe.yml` (one `run: echo probe` step) on a throwaway branch `research/draft-actions-probe`,
   push it to `gitea` only, check the repo's Actions page shows no run, then delete that branch on Gitea.
2. **Branch protection** (today NO branch on this Gitea instance is protected). Settings, Branches, add rules in
   this priority order: `research/draft-*` first (push allowed to anyone with write access, which includes the
   deploy key; force-push and deletion off), then `*` (push allow-list = your own user only, "allow deploy keys"
   UNticked, force-push off, deletion off). Also protect tags (Settings, Tags: pattern `*`, allowed = your user).
   This is the real enforcement that the container can never touch `main`, any other real branch, or a tag;
   `claude_container_settings.json`'s deny rules are belt-and-braces on top.
3. **Deploy key:** Settings, Deploy Keys, add `~/.ssh/sim_agent_gitea_deploy.pub` with **write access** (it pushes
   only `research/draft-*`, per item 2's rules). The container reaches Gitea at `192.168.0.10:2222`: the WAN only
   forwards 80/443, and `git.dant123.com` hairpins through the WAN address, which the firewall drops.
4. Optional, cheap: cap a single push at 512 MiB (the gitea dataset has no quota):

   ```bash
   docker exec -u git gitea git -C /data/git/repositories/dant123/neural-simulator.git config receive.maxInputSize 536870912
   ```

### 4. Retire the 2026-09-29 deployment attempt (on the server)

It sits outside ZFS, git and backup, and its network holds `172.30.30.0/24`. Nothing references it (no container,
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

After `research/harness-server-fit` is merged to `main` (or add `-b research/harness-server-fit` to trial it
first). A sparse, blobless clone of `tools/` only is all the build and the ctl script need (about 15 MB):

```bash
git clone --filter=blob:none --sparse https://github.com/danthi123/neural-simulator.git /home/dant123/data/sim-agent/src
git -C /home/dant123/data/sim-agent/src sparse-checkout set tools
git -C /home/dant123/data/sim-agent/src log -1 --oneline
```

Update it only deliberately (review what changed first; the image and the ctl forced command run from it):
`git -C /home/dant123/data/sim-agent/src fetch origin && git -C ... log --oneline HEAD..origin/main`, then
`git -C ... merge --ff-only origin/main`. `update.sh` refuses to build if this checkout's `tools/agent_container`
or `tools/agent_harness_ctl.sh` has any modified, untracked or ignored file.

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
sudo nft list table inet sim_agent | head -40
```

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
  repo never contains it), and the container's HTTPS rule excludes it.
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
docker logs sim-agent 2>&1 | tail -5                          # expect: "firewall_guard: OK ..." then "ready."
```

The clone's remote is named `gitea`, which the container's settings allow pushing to (`git push origin` is
denied). The smoke test's network section (section 7) already runs the must-fail/must-pass probes from inside;
a missing firewall makes the run refuse to start at all.

### 13. Verify the isolation (and again after any firewall, Docker or fragment change)

```bash
bash /home/dant123/data/sim-agent/src/tools/agent_container/server-stack/verify_isolation.sh   # expect: ALL CHECKS PASSED
```

Read-only. Host side: both firewall units active and enabled, `nftables.service` not enabled, the table loaded,
`self_wan` non-empty, `br-simagent` present, `apps_sim-agent-net` = `172.30.30.0/24`, the container on that network
only, no published ports. From inside (in parallel, about 5 s), each must be DROPPED (a timeout; a connection or a
refusal fails): this host via the gateway and via 192.168.0.10 on 22/111/139/445/2049/3493, the published ports
1880/18443/11000/8554, the router's UI, AdGuard at 192.168.0.5, the workstation's 80/445, docker0 and the proxy
bridge gateway, both docker-socket-proxies and the gitea and caddy containers by IP, 8.8.8.8:53 (not an allowed
resolver), 1.1.1.1:80, and `https://git.dant123.com` (the hairpin). Must work: DNS at both resolvers,
192.168.0.68:22 and :8080 (a refusal is fine if llama-swap is stopped), 192.168.0.10:2222, HTTPS to GitHub and
claude.ai. Also no IPv6, no docker socket, uid 61000, no effective capabilities, `no_new_privs`, read-only root.
Then the two channels:

```bash
ssh homelab-agent-ctl state                              # from the workstation: appdata_gb / firewall: active lines
docker exec sim-agent ssh workstation-gateway status     # on the server: the gateway answers
docker exec sim-agent git -C /work/sim fetch gitea       # on the server: gitea over 2222 works
# branch protection holds for the deploy key (step 3): a draft push works, anything else is rejected
docker exec sim-agent git -C /work/sim push gitea HEAD:refs/heads/research/draft-protection-probe   # expect: accepted
docker exec sim-agent git -C /work/sim push gitea HEAD:refs/heads/protection-probe                 # expect: REJECTED
```

Delete `research/draft-protection-probe` in the Gitea UI afterwards (and `protection-probe` too, if it was
accepted: then step 3's rules are wrong, fix them before going further).

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
`sim-agent:rollback`; `docker compose build --pull sim-agent`; if the image changed, run the smoke test against
the NEW image and, on failure, point `sim-agent:local` back at the rollback image and stop. It swaps the running
container (`docker compose up -d --no-deps --no-build sim-agent`) only when it is running, no `claude` process is
active and `sim-agent-fw.service` is active. A paused harness (a gaming yield froze a session, maybe mid-task) is
deferred, not recreated; a stopped one is never started. It logs to `/home/dant123/data/sim-agent/update.log`;
failures alert through `ntfy-notify@`.

Manual rollback: `docker image tag sim-agent:rollback sim-agent:local && cd ~/apps && docker compose up -d
--no-deps --no-build sim-agent`.

Claude Code itself lives in the home bind mount and is kept current as before: every `agent start` runs
`claude-safe-update.sh` (update, smoke-test against the chosen backend, roll back to the last good version on
failure). Its own auto-updater is never disabled.

Never `docker compose up -d` the whole `~/apps` project while an agent session is live if the sim-agent fragment
changed: it recreates the container. Check first: `docker exec sim-agent pgrep -f '/claude$'` (empty = idle).

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
    set -l remote_cmd "docker exec sim-agent tmux send-keys -t agent $inner Enter"
    ssh homelab-server $remote_cmd
    and echo "agent: sent 'agent-start.sh $profile' into the tmux session -- attach with: agent"
```

Every `docker exec sim-agent ...` in this doc runs ON THE SERVER; the container does not exist on the workstation.

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

## RECOMMENDED: remove NOPASSWD sudo on the pool nodes

The pool40/41/42 mini-PCs run pool jobs under a NOPASSWD sudoers entry, so a malicious pool job (the agent can
author runner code that is dispatched there) gets root on them for free, and a foothold on the LAN from there.
Pool jobs are not sandboxed the way GPU jobs are (`tools/agent_gateway_submit.py`'s header: they are dedicated,
disposable compute), but disposable does not mean a free root shell on a LAN device. On each node:

```bash
sudo grep -rl NOPASSWD /etc/sudoers.d/            # find the file
sudo visudo -f /etc/sudoers.d/<that-file>         # delete the NOPASSWD line (or require a password)
```

More thorough: run pool jobs as a separate unprivileged user with no sudo at all (`sudo useradd -r -M -s
/usr/sbin/nologin simjob` on each node, then point the dispatcher's remote launch at it), matching the separation
the workstation's GPU jobs already use.
