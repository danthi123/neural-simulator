# Agent harness setup (owner-applied steps)

This is the checklist for turning the built `sim-agent` container (`tools/agent_container/`) into the actual
daily-driver harness on `homelab-server` (192.168.0.10). Everything here is security-relevant and is written as
exact copy-paste commands for the OWNER to run. The agent that built this must not run any of it itself.

Background: `docs/plans/2026-09-29-local-model-transition-PLAN.md` section 9. `tools/agent_gateway.sh` (on the
workstation) and `tools/agent_harness_ctl.sh` (on the server) are the two forced-command scripts these keys
point at -- read their headers for exactly what each allows.

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

Then, on the workstation, point both ends at each other in `~/.ssh/config`:

```
Host workstation-gateway
    HostName 127.0.0.1        # the container reaches this key from ITS side, over the docker network -- this
                               # entry is for the OWNER to test the key from the workstation itself
    User dant123
    IdentityFile ~/.ssh/sim_agent_container_key
    IdentitiesOnly yes

Host homelab-agent-ctl
    HostName 192.168.0.10
    User USERNAME
    IdentityFile ~/.ssh/sim_agent_harness_ctl_key
    IdentitiesOnly yes
```

And copy `~/.ssh/sim_agent_container_key` (the PRIVATE half) into the container's home volume, with an ssh
config pointing back at the workstation:

```bash
docker cp ~/.ssh/sim_agent_container_key sim-agent:/home/agent/.ssh/id_ed25519
docker exec sim-agent chmod 600 /home/agent/.ssh/id_ed25519
docker exec -u root sim-agent chown agent:agent /home/agent/.ssh/id_ed25519
cat <<'EOF' | docker exec -i sim-agent tee -a /home/agent/.ssh/config
Host workstation-gateway
    HostName 192.168.0.68
    User dant123
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF
```

Test both directions:

```bash
ssh homelab-agent-ctl state                              # from the workstation
docker exec sim-agent ssh workstation-gateway status      # from inside the container
```

## (c) Gitea: a scoped bot/deploy key + branch protection

In the gitea web UI (`https://git.dant123.com`), on the `neural-simulator` repo:

1. **Settings -> Branches**: add a branch protection rule for `main` -- require pull request reviews, disable
   force-push, disable deletion. This is the REAL enforcement that a compromised container can never touch
   `main`; `claude_container_settings.json`'s own deny rules are belt-and-braces on top of it, not instead of it.
2. **Settings -> Deploy Keys**, or a dedicated bot account under **Site Administration -> User Accounts** (a
   deploy key is simpler and sufeg the container never needs to open PRs itself): add a new deploy key, paste a
   FRESH keypair's public half (`ssh-keygen -t ed25519 -f ~/.ssh/sim_agent_gitea_deploy -N ""`, generated on the
   workstation, private half copied into the container's home volume the same way as step (b)'s container key).
   Grant it **write access** (deploy keys are read-only by default; the container needs to push its own
   `research/draft-*` branches).
3. Gitea's deploy keys do not support a branch-name push restriction natively as of this writing -- rely on
   step 1's branch protection on `main` as the actual backstop (a deploy key can technically push to any
   UNPROTECTED branch, which is exactly the `research/draft-*` namespace this is meant to allow). If gitea adds
   per-key branch restrictions later, apply one limiting this key to `research/draft-*`.
4. Add the deploy key's HOST entry to the container's ssh config (same `docker exec -i sim-agent tee -a
   /home/agent/.ssh/config` pattern as (b)) and point the `gitea` git remote at it, or use gitea's own ssh clone
   URL with `IdentityFile` pinned to this key.

## (d) OPTIONAL: egress restriction for the container's subnet

`compose.yaml` gives `sim-agent` its own bridge network (`agent-net`), so its egress can be restricted with
`DOCKER-USER` iptables rules on the server without touching any other container's network. Find the bridge's
subnet first:

```bash
docker network inspect sim-agent_agent-net --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}'
```

Then (adjust `<SUBNET>` to that output, and `<WORKSTATION_IP>`/`<POOL_NODE_IPS>` to the real addresses):

```bash
sudo iptables -I DOCKER-USER -s <SUBNET> -d <WORKSTATION_IP> -p tcp --dport 8080 -j ACCEPT   # the model server
sudo iptables -I DOCKER-USER -s <SUBNET> -d <WORKSTATION_IP> -p tcp --dport 22 -j ACCEPT      # the gateway ssh
sudo iptables -I DOCKER-USER -s <SUBNET> -d <POOL_NODE_IPS> -p tcp --dport 22 -j ACCEPT        # pool nodes, if the
                                                                                                 # container ever needs direct ssh to them (today it does not -- pool jobs go through pool-submit)
sudo iptables -I DOCKER-USER -s <SUBNET> -p tcp --dport 2222 -d 127.0.0.1 -j ACCEPT             # gitea (adjust to gitea's real address/port if not local to the server)
sudo iptables -I DOCKER-USER -s <SUBNET> -p tcp --dport 443 -j ACCEPT                            # 443 for cloud APIs + the Claude Code native installer/updater (claude.ai)
sudo iptables -I DOCKER-USER -s <SUBNET> -j DROP                                                 # deny everything else from this subnet
```

These rules are not persisted across a reboot by default -- add them to whatever this server already uses for
persistent iptables rules (`iptables-persistent`, a systemd unit, etc.) if you want them to survive one.

## (e) First start: build and seed the repo

On the server, from wherever you keep it (adjust `USERNAME`/path to match step (b)'s `authorized_keys` line):

```bash
mkdir -p ~/sim-agent
# copied from the workstation in the "deploy what's safe" step this build already did (scp), or:
# scp -r dant123@<workstation-ip>:/home/dant123/Projects/sim/.claude/worktrees/agent-container/tools/agent_container/ ~/sim-agent/tools/agent_container/
cd ~/sim-agent/tools/agent_container
docker compose build
docker compose up -d
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
# on the server:
docker exec -u agent sim-agent git clone /tmp/sim.bundle /work/sim   # needs /tmp/sim.bundle visible inside the
                                                                       # container -- e.g. docker cp it in first:
docker cp /tmp/sim.bundle sim-agent:/tmp/sim.bundle
```

Then attach and start a session:

```bash
docker exec -it sim-agent tmux attach -t agent
# inside: agent-start.sh workstation      (or whatever `llm list` shows once the bake-off drops a real profile)
```

## Everyday use (the `agent` fish function, workstation-side)

Already installed at `~/.config/fish/functions/agent.fish` by this build (not something you need to write) --
`agent` attaches, `agent start [profile]` starts a session, `agent stop`, `agent logs`.

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
