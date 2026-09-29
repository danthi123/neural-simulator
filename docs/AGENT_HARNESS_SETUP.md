# Agent harness setup (owner-applied steps)

This is the checklist for turning the built `sim-agent` container (`tools/agent_container/`) into the actual
daily-driver harness on `homelab-server` (192.168.0.10). Everything here is security-relevant and is written as
exact copy-paste commands for the OWNER to run. The agent that built this must not run any of it itself.

Background: `docs/plans/2026-09-29-local-model-transition-PLAN.md` section 9, and the 2026-09-29 security
review that hardened this setup (threat model: the model driving the container may be adversarial -- it must
not affect anything beyond its container, its volumes, and the narrow, validated channels below).
`tools/agent_gateway.sh` (on the workstation) and `tools/agent_harness_ctl.sh` (on the server) are the two
forced-command scripts these keys point at -- read their headers for exactly what each allows.

**MANDATORY ORDER -- the harness must NOT be started before step (d\') (the egress firewall) is applied.**
Without it, a compromised container's outbound traffic is unrestricted on the LAN (see step (d\') for why the
old "optional" framing was wrong).

## (a) Generate the two keypairs

One key lets the CONTAINER reach the WORKSTATION (restricted to `tools/agent_gateway.sh`'s five commands). The
other lets the WORKSTATION reach the SERVER's harness (restricted to `tools/agent_harness_ctl.sh`'s four
commands). Generate both on the workstation, ed25519, no passphrase (they are used non-interactively):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/sim_agent_container_key -C "sim-agent container -> workstation gateway" -N ""
ssh-keygen -t ed25519 -f ~/.ssh/sim_agent_harness_ctl_key -C "workstation -> sim-agent harness ctl" -N ""
```

`sim_agent_container_key` (the pair) needs its PRIVATE half copied INTO the container's home volume, and its
PUBLIC half added to the workstation's `authorized_keys` (step b). `sim_agent_harness_ctl_key`'s private half
stays on the workstation; its public half goes into the SERVER's `authorized_keys`.

## (b) The two `authorized_keys` lines

**On the workstation** (`~/.ssh/authorized_keys`, the account this repo lives under), restricting the
container's key to `tools/agent_gateway.sh` only, and only from the server's LAN address:

```
command="bash /home/dant123/Projects/sim/tools/agent_gateway.sh",restrict,from="192.168.0.10" ssh-ed25519 AAAA...container-pubkey... sim-agent container -> workstation gateway
```

(paste the contents of `~/.ssh/sim_agent_container_key.pub` in place of `AAAA...container-pubkey...`).

**On the server** (`homelab-server`'s `~/.ssh/authorized_keys`, whichever account owns `~/sim-agent/`),
restricting the workstation's key to `tools/agent_harness_ctl.sh` only, and only from the workstation's LAN
address:

```
command="bash /home/USERNAME/sim-agent/tools/agent_harness_ctl.sh",restrict,from="192.168.0.68" ssh-ed25519 AAAA...ctl-pubkey... workstation -> sim-agent harness ctl
```

(paste `~/.ssh/sim_agent_harness_ctl_key.pub`; adjust the path to wherever step (e) below deploys the repo copy
on the server, and the workstation's actual LAN IP if it has changed since 2026-09-29's `192.168.0.68`).

Then, on the workstation, point the OUTBOUND direction (workstation -> server ctl) at its target in
`~/.ssh/config`:

```
Host homelab-agent-ctl
    HostName 192.168.0.10
    User USERNAME
    IdentityFile ~/.ssh/sim_agent_harness_ctl_key
    IdentitiesOnly yes
```

**DOC FIX (2026-09-29 security review):** the old version of this doc also added a `workstation-gateway` host
entry HERE, pointed at `127.0.0.1`, "for the owner to test the key from the workstation itself" -- that can
never work: the `authorized_keys` line above restricts the container's key with `from="192.168.0.10"` (the
SERVER's LAN address), so a connection whose actual source address is `127.0.0.1` (loopback, testing from the
workstation itself) is refused by sshd before the forced command ever runs, regardless of which key is used.
There is no correct way to test this key's *authentication* from the workstation itself -- the only real test is
from where the traffic will actually originate: the container (or the server), which the `docker exec` command
below already does correctly.

The container's own `~/.ssh/config` (INSIDE the container, pointing back at the workstation) is written next, in
step (e) below, at the same time as everything else that needs to land inside the container -- **not via `docker
cp`/`docker exec` run from the workstation**, another doc error the old version had: those commands operate on
whatever host they are typed on, and the `sim-agent` container only exists ON THE SERVER, so every `docker
cp`/`docker exec sim-agent ...` command in this file must be run ON THE SERVER (over your own ssh session to
`homelab-server`), never from the workstation, however that looks identical to `docker cp ~/.ssh/... sim-agent:...`
either way -- if the container is not visible to a bare `docker ps`, you are on the wrong host.

Test the direction this key pair CAN be tested from (the workstation, reaching the SERVER's ctl script):

```bash
ssh homelab-agent-ctl state                              # from the workstation
```

The other direction (container -> workstation gateway) is tested once the container has its own key + config
(step (e)), from ON THE SERVER:

```bash
docker exec sim-agent ssh workstation-gateway status      # run ON THE SERVER, not the workstation
```

## (c) Gitea: a scoped bot/deploy key + branch protection

In the gitea web UI (`https://git.dant123.com`), on the `neural-simulator` repo:

1. **Settings -> Branches**: **2026-09-29 security review, M2** -- protect ALL branches, not just `main`:
   add a branch protection rule for `*` (or, if gitea's glob syntax here requires it, one rule per existing
   long-lived branch plus `*`) -- require pull request reviews, disable force-push, disable deletion. THEN add a
   SEPARATE, narrower rule for `research/draft-*` that leaves it unprotected (or only requires status checks,
   never a review) -- this is the ONE namespace the container's deploy key is meant to be able to push to
   directly. Also **protect tags** (Settings -> Branches has a tag-protection section, or Settings -> Tags on
   newer gitea) -- an unprotected tag is otherwise an easy way to plant a ref that some other automation (CI, a
   release process) might trust. This is the REAL enforcement that a compromised container can never touch
   `main` (or any other real branch) or a tag; `claude_container_settings.json`'s own deny rules are
   belt-and-braces on top of it, not instead of it.
2. **Settings -> Actions**: **2026-09-29 security review, M2** -- disable Actions for this repo (repo Settings,
   or org/instance-wide if Actions is not needed anywhere). If Actions and a runner are ever enabled here, a
   pushed workflow file (even to an otherwise-unprotected `research/draft-*` branch) can execute on whatever
   runs the Actions runner -- for a repo a possibly-adversarial container can push to, this is a direct route off
   the container entirely. Disabling Actions closes it regardless of any other setting here.
3. **Settings -> Deploy Keys**, or a dedicated bot account under **Site Administration -> User Accounts** (a
   deploy key is simpler and the container never needs to open PRs itself): add a new deploy key, paste a
   FRESH keypair's public half (`ssh-keygen -t ed25519 -f ~/.ssh/sim_agent_gitea_deploy -N ""`, generated on the
   workstation, private half copied into the container's home volume in step (e) below, the same way as the
   container's workstation-gateway key). Grant it **write access** (deploy keys are read-only by default; the
   container needs to push its own `research/draft-*` branches).
4. Gitea's deploy keys do not support a branch-name push restriction natively as of this writing -- rely on
   step 1's branch protection as the actual backstop (a deploy key can technically push to any UNPROTECTED
   branch, which is exactly the `research/draft-*` namespace this is meant to allow -- everything else is now
   protected). If gitea adds per-key branch restrictions later, apply one limiting this key to `research/draft-*`
   as defense in depth on top of the branch protection, not instead of it.
5. Add the deploy key's HOST entry to the container's ssh config (step (e) below, run ON THE SERVER) and point
   the `gitea` git remote at it, or use gitea's own ssh clone URL with `IdentityFile` pinned to this key.

## (d) A separate, read-only results-sync key (H3, 2026-09-29 security review)

**This replaces the old `results-sync` subcommand on `tools/agent_gateway.sh`.** The old version validated only
the LAST rsync protocol argument, so `. /home/dant123/.ssh/ research/findings/raw/` could read private keys, and
rsync's own `--remove-source-files`/`-L` could delete files or follow attacker-planted symlinks. The fix uses
rsync's own restricted-rsync wrapper (`rrsync`, shipped with rsync -- `/usr/bin/rrsync` on this workstation) as
the ENTIRE forced command, so this script never has to parse rsync's protocol arguments at all:

Generate a THIRD keypair on the workstation:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/sim_agent_results_sync_key -C "sim-agent container -> results-sync (read-only)" -N ""
```

Add its own `authorized_keys` line on the workstation, restricted to `rrsync -ro -no-del <abs path>` (read-only,
no delete) and, same as the other container key, only from the server's LAN address:

```
command="rrsync -ro -no-del /home/dant123/Projects/sim/research/findings/raw",restrict,from="192.168.0.10" ssh-ed25519 AAAA...results-sync-pubkey... sim-agent container -> results-sync (read-only)
```

`rrsync` itself confines the sync to that one absolute path (ignoring/rejecting anything the client's rsync
invocation tries to request outside it) and, with `-ro`, refuses any write direction outright -- there is no
argument-parsing left for this repo's own scripts to get wrong. `-no-del` additionally refuses a delete even
though `-ro` alone already implies no writes. Symlinks: `rrsync` does not need a `-L`/`--copy-links`-style flag
added here to be safe -- omit `--copy-links` on the CONTAINER's own pull command below and rsync's default
behaviour (copy the symlink itself, never follow it) applies, so a pool-job-planted symlink under
`research/findings/raw/` is copied as an inert symlink, never dereferenced into a read of whatever it points at.

Add the container's own `Host` entry pointing at this key (step (e) below writes it inside the container), and
pull from the container as an ordinary read-only rsync:

```bash
rsync -av -e ssh results-sync-gateway: /work/sim/research/findings/raw/
```

## (d') MANDATORY: the egress firewall (M1, 2026-09-29 security review)

**Do not start the harness (`docker compose up -d`) before this is applied.** `tools/agent_container/egress.sh`
replaces this doc's old, OPTIONAL, broken iptables snippet -- read its own header for exactly what was wrong
with the old version (every `-I` inserted at the TOP, so the final `DROP` ended up first and blocked
everything; port 443 was open to this SAME SERVER's own reverse-proxied services and the rest of the LAN, not
just the internet; the gitea rule's `-d 127.0.0.1` could never match; nothing governed the container talking
directly to this host's own listening services).

1. `compose.yaml`'s `agent-net` network now declares a FIXED subnet (`172.30.30.0/24`) so the firewall rules
   below can target it directly and never silently stop matching after a network recreate. Bring the network up
   first (`docker compose -f tools/agent_container/compose.yaml create` is enough to create the network without
   starting the container, if you want to apply the firewall before the container itself ever runs), then find
   the bridge interface name:

   ```bash
   docker network inspect sim-agent_agent-net --format '{{.Id}}'   # -> the bridge is br-<first 12 chars>
   ```

2. Edit the CONFIGURATION block at the top of `tools/agent_container/egress.sh` (subnet, bridge interface,
   workstation IP/port, gitea host/port, your LAN's real DNS resolver), then:

   ```bash
   sudo bash tools/agent_container/egress.sh --dry-run   # review the exact commands first, no root needed
   sudo bash tools/agent_container/egress.sh              # apply (idempotent -- safe to re-run after any edit)
   ```

3. Persist across a reboot -- install the accompanying systemd unit (edit its `ExecStart` path first):

   ```bash
   sudo cp tools/agent_container/sim-agent-egress.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now sim-agent-egress.service
   ```

ONLY once this is applied: `docker compose -f tools/agent_container/compose.yaml up -d`.

## (e) First start: build and seed the repo

On the server, from wherever you keep it (adjust `USERNAME`/path to match step (b)'s `authorized_keys` line):

```bash
mkdir -p ~/sim-agent
# copied from the workstation in the "deploy what's safe" step this build already did (scp), or:
# scp -r dant123@<workstation-ip>:/home/dant123/Projects/sim/.claude/worktrees/agent-container/tools/agent_container/ ~/sim-agent/tools/agent_container/
cd ~/sim-agent/tools/agent_container
docker compose build
# ONLY after (d') has been applied:
docker compose up -d
```

**UID/GID 61000, and why nothing here needs a runtime `chown` (2026-09-29 security review, LOW -- the old
version of this doc had a `docker exec -u root sim-agent chown agent:agent ...` step that could never have
worked: `cap_drop: ALL` drops `CAP_CHOWN` too, so a `chown` run as root INSIDE this container fails regardless of
`-u root`, since capabilities are dropped for the whole container's user namespace, not just its default user).**
The Dockerfile now builds the image with `AGENT_UID=61000`/`AGENT_GID=61000` (not 1000 -- 1000 collides with the
workstation account's own uid) and already creates `/work/sim` (`chown`ed at BUILD time, unrestricted) and
`/home/agent` (via `useradd -m`, also at build time) owned by that uid. Docker copies a NAMED VOLUME's content
-- including ownership -- from the image's own directory the FIRST time it is mounted over one, so both
`sim-agent-repo` and `sim-agent-home` come up correctly owned on first start with **no runtime chown at all**.

If you ever DO need to fix ownership on an already-populated volume (e.g. after manually placing a file into it
with `docker cp`, which preserves the SOURCE file's own uid/gid, not the container's), do it via a throwaway,
UNRESTRICTED container that mounts the same named volume -- never via `docker exec` on the actual `sim-agent`
container, which is deliberately `cap_drop: ALL`:

```bash
docker run --rm -v sim-agent_sim-agent-home:/home/agent alpine chown -R 61000:61000 /home/agent/.ssh
```

Seed the ssh keys into the container's home volume this way too (all of these are `docker cp`/`docker exec`
commands, so run ON THE SERVER, never the workstation -- see step (b)'s note on this same doc error):

```bash
docker cp ~/.ssh/sim_agent_container_key sim-agent:/home/agent/.ssh/id_ed25519
docker cp ~/.ssh/sim_agent_results_sync_key sim-agent:/home/agent/.ssh/id_ed25519_results_sync
docker cp ~/.ssh/sim_agent_gitea_deploy sim-agent:/home/agent/.ssh/id_ed25519_gitea
docker run --rm -v sim-agent_sim-agent-home:/home/agent alpine sh -c \
    'chown 61000:61000 /home/agent/.ssh/id_ed25519* && chmod 600 /home/agent/.ssh/id_ed25519*'
cat <<'EOF' | docker exec -i sim-agent tee -a /home/agent/.ssh/config
Host workstation-gateway
    HostName 192.168.0.68
    User dant123
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host results-sync-gateway
    HostName 192.168.0.68
    User dant123
    IdentityFile ~/.ssh/id_ed25519_results_sync
    IdentitiesOnly yes

Host gitea-deploy
    HostName git.dant123.com
    User git
    IdentityFile ~/.ssh/id_ed25519_gitea
    IdentitiesOnly yes
EOF
```

Seed the repo volume -- EITHER clone straight from gitea (once step (c)'s deploy key is wired):

```bash
docker exec -u agent sim-agent git clone git@gitea-deploy:dant123/neural-simulator.git /work/sim
```

OR, if gitea access is not ready yet, send a bundle from the workstation:

```bash
# on the workstation:
git -C /home/dant123/Projects/sim bundle create /tmp/sim.bundle --all
scp /tmp/sim.bundle homelab-server:/tmp/sim.bundle
# on the server (this docker cp/exec pair, like all such pairs in this doc, runs ON THE SERVER):
docker cp /tmp/sim.bundle sim-agent:/tmp/sim.bundle
docker exec -u agent sim-agent git clone /tmp/sim.bundle /work/sim
```

Then attach and start a session:

```bash
docker exec -it sim-agent tmux attach -t agent
# inside: agent-start.sh workstation      (or whatever `llm list` shows once the bake-off drops a real profile)
```

## Everyday use (the `agent` fish function, workstation-side)

Already installed at `~/.config/fish/functions/agent.fish` by this build (not something you need to write) --
`agent` attaches, `agent start [profile]` starts a session, `agent stop`, `agent logs`.

**Functional fix (2026-09-29 security review):** `agent start [profile]` used to build its remote command as
several SEPARATE arguments to `ssh` (`ssh homelab-server docker exec sim-agent tmux send-keys -t agent "bash
.../agent-start.sh $profile" Enter`) -- but the ssh CLIENT re-joins all of its own trailing command arguments
with plain spaces before sending them to the remote shell, which does not preserve the quoting around the
`"bash .../agent-start.sh $profile"` argument at all; the remote side sees every word as a separate token, so a
`tmux send-keys` call that was meant to type ONE line ends up typing several separate "keys" instead (and any
profile name containing a token tmux treats specially, e.g. a bare `C-c`-shaped word, would be interpreted as a
keystroke rather than literal text, the same class of bug the `say` fix in `tools/agent_harness_ctl.sh`
addresses). The fixed function builds the ENTIRE remote command as ONE pre-escaped string and hands `ssh` only
TWO arguments total (the host, then that one string) -- with nothing left for `ssh` to rejoin, this problem
cannot recur:

```fish
case start
    set -l profile $argv[2]
    if test -z "$profile"
        set profile workstation
    end
    set -l inner (string escape -- "bash /usr/local/bin/agent-start.sh $profile")
    set -l remote_cmd "docker exec sim-agent tmux send-keys -t agent $inner Enter"
    ssh homelab-server $remote_cmd
    and echo "agent: sent 'agent-start.sh $profile' into the tmux session -- attach with: agent"
```

## Optional: a size-capped repo/home volume (M3, 2026-09-29 security review)

homelab-server is SHARED (gitea/nextcloud/home-assistant/etc.) and the two named volumes have no built-in size
cap (compose.yaml's `logging.driver: local` only bounds the container's stdout/stderr, never volume content;
`tools/inspection_digest.py` and `update.sh`'s own disk-usage check both warn when usage looks wrong, but
neither can PREVENT it). If you want a hard ceiling instead of a warning, back one or both named volumes with a
loop-mounted, fixed-size image file instead of the plain directory-backed local driver Docker uses by default --
growth then hits `ENOSPC` INSIDE the loop device, never the shared filesystem:

```bash
fallocate -l 20G /var/lib/sim-agent-repo.img
mkfs.ext4 /var/lib/sim-agent-repo.img
# then either mount it and bind-mount the mountpoint as the volume's host path, or use a `local` volume driver
# with `o: loop` options pointed at the image file -- pick whichever this server's docker version supports; not
# scripted here since it is optional and the exact incantation varies by docker/kernel version.
```

## Scheduling `tools/agent_container/update.sh` (weekly rebuild)

Not installed by this build on purpose (owner directive: don't install a schedule). A weekly cron line, run as
whichever user owns `~/sim-agent/`:

```
0 4 * * 0  /usr/bin/bash ~/sim-agent/tools/agent_container/update.sh >> ~/sim-agent/tools/agent_container/cron.log 2>&1
```

Or a systemd user timer (`~/.config/systemd/user/sim-agent-update.timer` + a matching `.service` running the
same command), enabled with `systemctl --user enable --now sim-agent-update.timer`. Either way the script
itself decides whether it is safe to recreate the container this cycle (see its own header) -- the schedule just
needs to run it periodically.

## Scheduling `tools/agent_container/update.sh` (weekly rebuild)

Not installed by this build on purpose (owner directive: don't install a schedule). A weekly cron line, run as
whichever user owns `~/sim-agent/`:

```
0 4 * * 0  /usr/bin/bash ~/sim-agent/tools/agent_container/update.sh >> ~/sim-agent/tools/agent_container/cron.log 2>&1
```

Or a systemd user timer (`~/.config/systemd/user/sim-agent-update.timer` + a matching `.service` running the
same command), enabled with `systemctl --user enable --now sim-agent-update.timer`. Either way the script
itself decides whether it is safe to recreate the container this cycle (see its own header) -- the schedule just
needs to run it periodically.
