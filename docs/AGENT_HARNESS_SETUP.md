# Agent harness setup (owner-applied steps)

This is the checklist for turning the built `sim-agent` container (`tools/agent_container/`) into the actual
daily-driver harness on `homelab-server` (192.168.0.10). Everything here is security-relevant and is written as
exact copy-paste commands for the OWNER to run. The agent that built this must not run any of it itself.

Background: `docs/plans/2026-09-29-local-model-transition-PLAN.md` section 9, and the 2026-09-29 security
review that hardened this setup (threat model: the model driving the container may be adversarial -- it must
not affect anything beyond its container, its volumes, and the narrow, validated channels below).
`tools/agent_gateway.sh` (on the workstation) and `tools/agent_harness_ctl.sh` (on the server) are the two
forced-command scripts these keys point at -- read their headers for exactly what each allows.

**MANDATORY ORDER -- the harness must NOT be started before step (0) (workstation sshd), step (d\') (the egress
firewall), and the gitea Actions step in (c) are ALL applied.** Without the egress firewall, a compromised
container's outbound traffic is unrestricted on the LAN (see step (d\') for why the old "optional" framing was
wrong); without the sshd hardening, the restricted keys below are not the only way into this account; without
Actions disabled on the gitea repo, a pushed workflow file is a direct route off the container entirely.

**Plain-language residual risk (read this before deciding the setup is "done"):** even with every step below
applied, a FULLY ROGUE model driving the container could still (1) exfiltrate the private repo and any results
it can read, over the one egress path that must stay open for legitimate use (443 to non-LAN destinations --
there is no way to allow "the Claude Code updater" without also allowing "any other HTTPS destination" at the
IP-filtering level this setup works at); (2) if the RECOMMENDED pool-node sudo fix below is not applied, get
root on the mini-PC pool nodes and use that as a foothold onto the rest of the LAN from there (never from the
container's own subnet, which the egress firewall does confine); (3) burn CPU/GPU compute and fill its own two
volumes up to their configured caps (M3) -- a resource-exhaustion nuisance, not a compromise of anything else;
(4) attempt to social-engineer the OWNER through whatever it says in chat/tmux -- no technical control stops a
human from being persuaded, only the owner's own judgment does; (5) attempt a kernel or NVIDIA driver exploit
from inside the sandbox/container to escape it entirely -- namespaces and cap_drop/seccomp reduce this surface
but cannot make it zero; this setup gives an attacker no assistance toward one, but a fully rogue model with
unbounded compute time is still probing the SAME host kernel + driver every other unprivileged process on this
box already can.

## (0) MANDATORY: harden the workstation's sshd first

Before generating or installing any of the restricted keys below, close the one thing that would make them
optional: password authentication. A carefully `restrict`ed, forced-command key is meaningless if an attacker
(or a compromised container that somehow learned or brute-forced a password) can just log in normally instead.
This is a property of the WORKSTATION's sshd as a whole, not of any one key, so it is step 0.

In `/etc/ssh/sshd_config` (or a drop-in under `/etc/ssh/sshd_config.d/`), set:

```
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthenticationMethods publickey
```

Then reload (never a full restart while you might still be connected over the old config -- `reload` applies
the new config to the running daemon without dropping existing sessions):

```bash
sudo systemctl reload sshd
```

Verify the running daemon actually has these in effect (reads sshd's OWN resolved, effective configuration --
not just what the file says, which could differ if another drop-in overrides it later in the read order):

```bash
sudo sshd -T | grep -i passwordauthentication      # expect: passwordauthentication no
sudo sshd -T | grep -i kbdinteractiveauthentication # expect: kbdinteractiveauthentication no
sudo sshd -T | grep -i authenticationmethods        # expect: authenticationmethods publickey
```

Keep at least one already-working ssh-key session open while you do this, and confirm a FRESH connection with
your normal key still works, before closing that session -- a misconfigured `AuthenticationMethods` line can
lock out key-based access too if written wrong.

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
2. **MANDATORY, escalated to HIGH in the 2026-09-29 re-review: Settings -> Actions** -- disable Actions for this
   repo, or (simpler, and covers every repo on this instance at once) instance-wide via `[actions]
   ENABLED=false` in gitea's `app.ini` followed by a gitea restart. **Why this is HIGH, not merely "a pushed
   workflow could run somewhere"**: if this server's gitea has an act-runner registered with a Docker socket (or
   an equivalent POST/EXEC-enabled proxy in front of it, the common way people expose runners without handing
   out the raw socket) reachable from the runner's job containers, a pushed workflow file gets to issue
   arbitrary Docker API calls FROM the runner -- which is root-equivalent on whatever host that socket belongs
   to. For a repo a possibly-adversarial container can push to (`research/draft-*`, even fully branch-protected
   everywhere else), this is a direct path to ROOT ON THE SERVER, not merely "a workflow runs". Disabling
   Actions closes it regardless of any runner configuration detail you have not audited.

   Verify it is actually off:
   ```bash
   # (a) the repo's own Actions tab must be gone/inaccessible in the gitea web UI, and
   # (b) a real test push proves it, not just the UI's own claim:
   git push gitea research/draft-actions-verify-test HEAD:refs/heads/research/draft-actions-verify-test
   # then in the gitea UI: the repo's Actions tab shows NO new run for this push. Delete the test branch after.
   ```
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

**Also (2026-09-29 re-review, L3):** `agent start` now records the profile on the SERVER via
`tools/agent_harness_ctl.sh`'s `set-active-profile` (over the trusted workstation->server ctl key) BEFORE
sending the tmux keystroke -- this is what makes `state`'s gaming-pause-skip decision trustworthy: it reads
this server-local record, never anything the container itself could write.

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

## Optional: restrict who can reach the workstation's local-llm port

`tools/local_llm/llm.sh` runs llama-swap on `0.0.0.0:8080` on the WORKSTATION (192.168.0.68) -- required so
both loopback (`llm.sh` itself, `llm claude`) and the LAN (this harness) can reach it on the same port; see the
comment above `llama_swap_cmd()` in that script for why a single, narrower bind cannot serve both. The egress
firewall above (step (d')) already confines the sim-agent CONTAINER to reaching only `192.168.0.68:8080` in the
first place, but nothing yet stops an unrelated device elsewhere on the home LAN from also reaching that port
directly -- low severity (no credentials, just inference + this workstation's own model selection), but cheap
to close. Optional, on the WORKSTATION itself:

```bash
# accept from loopback and from homelab-server (192.168.0.10, the only real external caller); drop everything else
sudo iptables -I INPUT 1 -i lo -p tcp --dport 8080 -j ACCEPT
sudo iptables -I INPUT 1 -s 192.168.0.10 -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP
```

Verify from another LAN device (should now time out): `curl -m 3 http://192.168.0.68:8080/health`. Verify
homelab-server itself still works: `curl -m 3 http://192.168.0.68:8080/health` FROM 192.168.0.10. These rules
are not persisted across reboot by `iptables` alone -- save them with this distro's usual mechanism (e.g.
`iptables-save`, or a systemd unit / `nftables` equivalent) if you want them to survive one. Revert by deleting
the two ACCEPT lines' effect: `sudo iptables -D INPUT -p tcp --dport 8080 -j DROP` (repeat if it was added more
than once) and the matching `-D` for each `-I` rule above.

## Optional: a cheap cloud model for the mechanical tier

Two ready-made cloud profiles (owner-approved 2026-09-29) let `agent start` route mechanical work to a cheap,
Anthropic-compatible cloud model instead of the workstation's local server: `tools/agent_container/profiles/
deepseek-flash.env` (DeepSeek's `deepseek-flash` model, `https://api.deepseek.com/anthropic`) and `glm-lite.env`
(Z.ai's `glm-4.7-flashx`, `https://api.z.ai/api/anthropic`) -- both files cite the provider's own docs for the
base URL, model id and context window; read either file's header before using it. To use one: create an API
account with that provider (DeepSeek platform / Z.ai platform) and generate a key; put the key in THIS compose
project's own `.env` file on the server (next to `tools/agent_container/compose.yaml`, untracked -- never in
this repo) as `DEEPSEEK_API_KEY=...` or `GLM_API_KEY=...` (compose.yaml has a matching commented example line
to uncomment); then run `agent start deepseek-flash` (or `agent start glm-lite`) from the workstation. Both
profiles are `KIND=cloud`, so `tools/agent_gateway.sh yield` and `tools/game.sh` skip pausing the harness for
them -- mechanical work keeps flowing even while the workstation's GPU is busy or the owner is gaming. Before
routing real work to either one, validate the endpoint + model name actually work with a quick bake-off run
(from the workstation, outside the container -- this needs `claude` and this repo checked out locally, and
costs a little real API spend): `python3 tools/local_llm/bakeoff.py --base-url https://api.deepseek.com/anthropic
--model deepseek-flash --api-key-env DEEPSEEK_API_KEY` (swap in the GLM base URL/model/env var for that profile).
It runs the same three objective agentic tasks (T1 locate / T2 debug / T3 extend) the local-model bake-off used
and prints a PASS/fail verdict per task plus the full result at `tools/local_llm/results/<model>.remote.json`.

## MANDATORY (escalated from optional in the 2026-09-29 re-review, M3): a size-capped repo/home volume

homelab-server is SHARED (gitea/nextcloud/home-assistant/etc.) and the two named volumes have no built-in size
cap (compose.yaml's `logging.driver: local` only bounds the container's stdout/stderr, never volume content;
`tools/inspection_digest.py` and `update.sh`'s own disk-usage check both warn when usage looks wrong, but
neither can PREVENT it) -- a warning is not a ceiling, so this is now required, not optional. Back both named
volumes with a loop-mounted, fixed-size image file instead of the plain directory-backed local driver Docker
uses by default -- growth then hits `ENOSPC` INSIDE the loop device, never the shared filesystem:

```bash
fallocate -l 20G /var/lib/sim-agent-repo.img && mkfs.ext4 /var/lib/sim-agent-repo.img
fallocate -l 10G /var/lib/sim-agent-home.img && mkfs.ext4 /var/lib/sim-agent-home.img
# then either mount each and bind-mount the mountpoint as the volume's host path, or use a `local` volume driver
# with `o: loop` options pointed at the image file -- pick whichever this server's docker version supports.
```

Verify the cap actually holds (from inside the container, or via a throwaway container mounting the same
volume -- writing past the image's own size must fail with ENOSPC, not silently succeed onto the host disk):

```bash
docker exec sim-agent sh -c 'dd if=/dev/zero of=/work/sim/.cap_test bs=1M count=25000 2>&1 | tail -3; rm -f /work/sim/.cap_test'
# expect a "No space left on device" partway through a 20G checkout's cap, not a completed 25000+0 records out
```

Also cap gitea's own accepted push size (a rogue container could otherwise attempt to push an enormous blob to
fill the SERVER's gitea data volume, a separate resource from sim-agent's own two volumes) -- in gitea's
`app.ini`:

```ini
[repository]
; bytes; 500 MiB shown here, adjust to this repo's real needs
MAX_CREATION_LIMIT = -1
[git]
MAX_GIT_DIFF_LINES = 1000
```

Gitea's actual push-size limit setting is `[repository.upload] FILE_MAX_SIZE` (web UI uploads) and the git
transport itself is bounded by the OS/filesystem, not a gitea config key as of this writing -- the loop-image
cap above is what actually bounds it in practice; set `receive.maxInputSize` in the git config gitea's own git
hooks run under (per-repo `.git/config` on the server, or gitea's global git config) as an additional,
git-native ceiling on any single push:

```bash
# on the server, in gitea's git repo storage for this repo:
git config receive.maxInputSize 536870912   # 512 MiB per push
```

## RECOMMENDED: remove NOPASSWD sudo on the pool nodes

The pool40/41/42 mini-PCs run pool-submitted jobs with a NOPASSWD sudoers entry for their own user -- so a
malicious pool job (the agent can author runner code that gets dispatched there, same as GPU jobs) gets ROOT on
those minis for free, and from there a foothold onto the rest of the LAN those nodes sit on. Pool jobs are NOT
sandboxed the way GPU jobs are (see tools/agent_gateway_submit.py's own header on why: they are dedicated,
disposable compute, not shared infrastructure) -- but "disposable" does not mean "a free root shell on a LAN
device", which NOPASSWD sudo hands over unconditionally regardless of what the job actually needed root for.

Either remove the NOPASSWD entry (on each pool node, as whatever admin account manages them):

```bash
sudo visudo -f /etc/sudoers.d/<the-file-granting-it>   # find it first: sudo grep -rl NOPASSWD /etc/sudoers.d/
# delete the NOPASSWD line, or change it to require a password -- either way, a pool job's own user account
# (which never has that password) can no longer sudo at all
```

Or, more thoroughly, run pool jobs as a SEPARATE, unprivileged user with no sudo entry at all (matches the
`simjob`-style separation already used for the workstation's own GPU jobs):

```bash
sudo useradd -r -M -s /usr/sbin/nologin simjob        # on each pool node
# then point tools/pool_provision.sh / the dispatcher's remote launch command at this user (e.g. `sudo -u simjob
# <job>`, run by an account that itself is NOT the one with the NOPASSWD entry) -- a larger change than removing
# NOPASSWD outright, tracked here as the more thorough alternative rather than scripted in this pass.
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
