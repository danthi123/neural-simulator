# Home to server: the neural-simulator project moves to srv-host-0

*2026-09-30. Owner: "I want it all done today." The project's home moves from the workstation (192.168.0.68,
CachyOS, fish) to homelab-server (srv-host-0, 192.168.0.10, Ubuntu 26.04). Hermes (official image, `~/apps/hermes`)
becomes the full-time driver, working in the server checkout with its LOCAL terminal backend. The workstation becomes
a compute source (the RTX 3090 queue and local model serving), and Claude Code inspects occasionally. Owner principles:
simple, conventional, the server's own conventions, no impact on existing services, no clutter, reuse the project's
existing mechanisms.*

Short names used below: **S** = `/home/dant123/Projects/neural-simulator` on the server; **W** =
`/home/dant123/Projects/sim` on the workstation.

## 0. Where things run after the move

| What | Where | Mechanism (existing unless marked NEW) |
|---|---|---|
| Canonical checkout, commits, gates | S | git + `tools/githooks` (Hermes in its container, or dant123 on the host) |
| Driver | Hermes, default profile, cwd S | Hermes cron job for continuation (replaces `llm.sh claude --resume`) |
| Pool dispatch | S host, as user `sim` | `/home/sim/runtime/tools/pool_autodispatch.sh` under `sim-pool-dispatch.service` (NEW unit; §10) |
| Pool + GPU result pull, publish `main` | S host, as user `sim` | runtime `pool_sync.sh` x2 + `/usr/local/sbin/sim-publish` under `sim-sync.timer` (NEW unit; §10) |
| Audit, RAG refresh | Hermes | `parallel_audit.py`, `rag/periodic_update.sh`, run by the driver itself (§10: not in a privileged unit) |
| GPU queue + daemon, game mode, driver model | W | `gpu_queue.sh` (`gpu-queue-dispatch`), `game.sh`, `llm.sh` (llama-swap :8080) |
| GPU job submission | S to W | `tools/gpu_remote.sh add` (NEW): per-job worktree on W, then that worktree's own `gpu_add.sh` |
| Pool provisioning | W (for Hermes) or S host | `tools/gpu_remote.sh provision <sha>` runs `pool_provision.sh` on W; the S host can run it directly |
| AWS lane, `aws-guard.timer` | W (unchanged) | `.pool_extra_nodes` is empty, so no AWS pool node is in use |
| Claude inspection | W | Published `main` from Gitea/GitHub; a server branch through Hermes (§10: W holds no unattended credential to S); changes go to S as branches |

## 1. Facts verified today (read-only), including corrections to the survey

- **Pool nodes ARE reachable from the server.** TCP 22 on 192.168.0.41 and .42 connects from srv-host-0; the survey's
  "unreachable" was wrong. Only key authorization is missing.
- **Hermes already has a workstation key.** `/home/dant123/data/hermes/.ssh/workstation_ed25519` ("hermes@srv-host-0 ->
  workstation") is in W's `authorized_keys` with `from="192.168.0.10"`. W's ufw allows 22 and 11434 from .10, and
  does not allow 8080 (llama-swap).
- **The Hermes container** runs Debian 13, python 3.13 at `/opt/hermes/.venv`, uid 99 `hermes` in group 100 `users`,
  umask 0022, passwd home `/opt/data`. It has bash, ssh, flock, git and uv (`/opt/data/bin/uv`). It has **no rsync**,
  no nvidia-smi and no systemd user bus, so `memcap.sh` cannot run inside it. `terminal.home_mode: auto` gives tools
  `HOME=/opt/data/home`.
- **Backups do not cover S.** `backup.sh` backs up only paths under `~/data`, `~/files` and `~/media`, so the Hermes
  bind of `/home/dant123/Projects` is outside it. Sanoid has no stanza for `data/projects`. The nightly 01:30
  cold-stop of the `hermes` stack (for `~/data/hermes`) still happens and lasts a few minutes.
- **The server lacks some tools.** The `acl` package is not installed (setfacl missing). Python is 3.14 only, with no
  pip. dant123's umask is 0002, dant123 is in group `users`, and the dataset uses POSIX ACLs (`acltype=posix`).
- **Server conventions.** `apps-gitsync.service` is the template for a host job: `User=dant123`,
  `OnFailure=ntfy-notify@%n.service`, mirrored in `~/apps/host/etc/systemd/system`. Host daemons exist too
  (`beszel-agent`, `lan-shim`).
- **The workstation already runs GPU jobs outside its checkout.** Some jobs run in `~/derisk-pool/revisions/<sha>`
  copies whose `.venv` and `data` are symlinks to W's; `gpu_queue.log` START lines show this. That is the same layout
  `pool_sync.sh` pulls from a pool node.
- **Remotes allow anonymous fetch.** Gitea and GitHub both answer `git ls-remote` over HTTPS with no credentials.
  Gitea ssh (127.0.0.1:2222) and github.com:22 are reachable from the server.
- **A sparse per-job worktree stays clean.** Tested in a scratch repo: the worktree is created with `--no-checkout`,
  then `sparse-checkout set --no-cone '/*' '!/research/findings/raw/'`, then `reset --hard`. It reads clean to
  `git status --porcelain`, a `.venv` symlink is ignored, and new outputs appear as untracked. Other worktrees are
  unaffected. It sets `extensions.worktreeConfig=true` once in the repo config.
- **`llm.sh yield` records the last `llm.sh claude` session**, from `~/.local/state/sim-local-llm/last_session`
  (present). The queue's post-window restore then runs `llm.sh claude --resume <it>`, which would be a second driver.

## 2. Decisions (simplest conventional option at each fork)

1. **Pool dispatch and sync run as host systemd units.** *(Superseded in part by §10: they run as the unprivileged
   user `sim` from a reviewed checkout, not as dant123 from the workspace.)* They run as `User=dant123`, live in `/etc/systemd/system`,
   are mirrored in `~/apps/host/etc/systemd/system` and alert through `OnFailure=ntfy-notify@%n.service`, which is
   exactly how the box runs its non-container jobs (`apps-gitsync`). The owner manages them with `systemctl`, sees
   them in `systemctl --failed` and gets them in restic's host-layer backup. A compose service would need a custom
   image carrying the project's bash/ssh/rsync toolchain. A linger user unit would be invisible to `systemctl --failed`,
   and would have no backup and no alerts.
2. **Two units plus one timer.** *(§10: the audit and the RAG refresh left the unit; Hermes runs them.)* `sim-pool-dispatch.service` is long-running; `sim-sync.service` is a oneshot run by
   `sim-sync.timer` every 15 min. It pulls pool results, pulls workstation GPU results, runs `parallel_audit.py` (which
   feeds the freshness gates), refreshes the RAG index, then runs `push_both.sh main`. One unit replaces the
   workstation's `pool-sync.timer`, `sim-rag-autoupdate.timer`, the Claude heartbeat's audit and manual pushes.
3. **GPU jobs from S to W reuse `gpu_add.sh`.** `tools/gpu_remote.sh add '<cmd>'` checks that S's HEAD is committed
   and on a branch. It pushes that commit to W's repo as `refs/server-jobs/<sha>` over the same ssh connection (§10:
   W needs no credential to S), then on W creates a sparse worktree
   `~/derisk-pool/revisions/<sha>-<epoch>` without `research/findings/raw` (as `pool_provision.sh` also excludes
   it), symlinks `.venv` and `data` to W's (the existing workstation pattern), and execs **that worktree's**
   `tools/local_llm/gpu_add.sh "cd <worktree> && <cmd>"`. The guard runs where the queue lives, at the job's own
   revision, into W's single queue (`gpu_queue.sh` resolves the shared root through the git common dir). There is one
   worktree per job because `gpu_add.sh` refuses a checkout holding untracked outputs.
4. **GPU results come back through `pool_sync.sh`, not a new daemon.** W presents the pool-node layout: a placeholder
   `~/derisk-pool/sim/research/findings/raw/` plus `~/derisk-pool/revisions/*/research/findings/raw/`. `sim-sync`
   runs `POOL_NODES=workstation bash tools/pool_sync.sh`, with the same allow-list, `--safe-links` and `F-x` safety.
   Excluding raw from the worktrees matters: a checkout's fresh mtimes would make `rsync -au` copy 27.5k tracked raw
   files per worktree and overwrite newer server copies.
5. **Queue files live beside their dispatcher.** Pool-lane and gate state (`pool.queue*`, claims, `pool.running`,
   reservations, canaries, waivers, `dispatch.log`, requeue TSVs) moves to `S/research/queue`. GPU-lane state
   (`gpu.queue*`, `gpu_queue.log`, `GPU_PAUSE`, `GAME_MODE`, `.llm*`) and AWS state stay in `W/research/queue`.
6. **Pool jobs in flight are handed over in this order.** Stop W's dispatcher; running node jobs continue because they
   are `setsid` on the node. Run a final W pull, copy the post-claim queue state, then start S's dispatcher. Claimed
   lines have already left `pool.queue`, so nothing is re-dispatched. S's dispatcher sees the running jobs through its
   node probe (`POOL_JOB_ID` env), and `sim-sync` pulls their results when they finish (`rsync -au` is idempotent).
7. **Data: copy what is slow to rebuild, rebuild what is machine-specific.**
   - Copy: `data/` (corpus 11G, cifar10, mnist), the siblings `sim-data`, `rag_index`, `soma_bundles` and
     `sim-catalog` (resolved as parent-of-common-dir siblings, so they need no code change), and W's uncommitted
     `research/` state.
   - Rebuild: `.venv` and `.venv-rag`, on Python 3.11 via uv, installed under `S/.python` so the same path works on
     the host and in Hermes.
   - Leave on W: `bridges/lmtrain` (25G of GPU training), `~/models`, the HF cache (only MiniLM, about 90 MB, is
     fetched on first use), the 710 worktrees, the 8 stashes and the 42 branch tips that exist on no remote. Nothing
     is deleted.
8. **Git: S is canonical and only S writes `main`.** *(§10: publishing is `sim-publish`, via a sim-owned mirror, not
   `push_both.sh` in the workspace.)* `sim-sync` publishes `main` with `push_both.sh` using a deploy key.
   Fetch URLs stay HTTPS and anonymous; `pushurl`s are ssh aliases. No push credential enters the agent container,
   which mirrors `apps-gitsync`. *(§10: W adds no `server` remote. GPU-job revisions are pushed to W by
   `gpu_remote.sh`, and W's key no longer opens S.)* Claude's changes reach S as branches and are merged there.
9. **Hermes sees S at its host path.** Add the bind `/home/dant123/Projects:/home/dant123/Projects` and set the default
   profile's cwd to S, not `/workspace/neural-simulator`. Venv interpreter links, git worktree gitdirs and
   `pre-commit`'s canonical-venv check are absolute paths shared by host units and the container, and one path
   everywhere is the only way both can use them. `/workspace` stays mounted for everything else.
10. **Hermes's sim model comes from llama-swap on W** (`http://192.168.0.68:8080/v1`, `qwen38-27b-iq4nl-mtp-128k-q4`).
    It is the only server `gpu_queue.sh` coordinates with: jobs are held while `local-llm` is up, and yield and
    restore are built in. An Ollama load is invisible to the queue and can collide with a brain job on the 24 GB card.
11. **Hermes containers are capped** at `mem_limit: 12g` and `cpus: 8`, the numbers the 09-29 sim-agent design used.
    Sim tests now run inside Hermes, so the cap protects the other ~84 containers. Heavy work goes to the pool or GPU.
12. **One driver.** A Hermes cron job is the continuation. W's `last_session` is moved aside (not deleted) so a
    GPU-window restore reloads the model but never resumes a retired local Claude session.
13. **Provisioning.** Hermes provisions pool revisions through W (`gpu_remote.sh provision <sha>` runs that revision's
    own `pool_provision.sh` on W, which has rsync, the node keys and the corpus), because the official Hermes image
    has no rsync. The owner or Claude can equally run `pool_provision.sh` on the S host.
14. **Shared-folder permissions.** Put a default POSIX ACL `g::rwX` on `/home/dant123/Projects`, set
    `core.sharedRepository=group` and add `safe.directory` for both identities. This finishes the dataset's 2775
    setgid intent: files Hermes creates with umask 022 stay writable by dant123's units, and the reverse.

## 3. New code (minimum) and edits

New files:

| File | Why an existing mechanism cannot do it |
|---|---|
| `tools/gpu_remote.sh` | `gpu_add.sh`/`gpu_queue.sh` are single-host by design (queue + singleton lock beside the 3090); nothing creates a checkout of a server revision on W. Subcommands: `add`, `status`, `log [N]`, `yield [--max-hours N]`, `provision <sha>`, and `--dry-run` (prints the remote script). |
| `tests/test_gpu_remote.py` | Project convention (tools are tested). Stub `ssh` on PATH that runs `bash -s` and `git-receive-pack` locally against a temp W repo with no way back to the server, and a stub `gpu_add.sh`; pins the push of the job's revision, base64 job transport (W's login shell is fish), the refusal on a dirty/detached HEAD, and the sparse worktree. |
| `requirements-rag.txt` | `.venv-rag` has no requirements file, so it cannot be rebuilt on S. Generated on W: `.venv-rag/bin/pip freeze \| grep -vE '^(nvidia-\|triton==\|torch==)'` (CPU torch is installed separately on S). |
| `~/apps/host/etc/systemd/system/sim-pool-dispatch.service` | The existing unit is a W user unit with a W path; the server's convention is a mirrored host unit (Appendix A). |
| `~/apps/host/etc/systemd/system/sim-sync.service` | Replaces W's `pool-sync.timer` + `sim-rag-autoupdate.timer` + heartbeat audit + manual push (Appendix A). |
| `~/apps/host/etc/systemd/system/sim-sync.timer` | Its schedule (Appendix A). |

Edits to existing files (small, same branch):

- **Path portability.** Replace the hard-coded `/home/dant123/Projects/sim` with the shared root, meaning the parent of
  `git rev-parse --git-common-dir`, as `gpu_queue.sh` already does. Files: `pool_autodispatch.sh:20`,
  `pool_queue.sh:17`, `pool_canary.sh:51`, `queue_add.sh:19`, `lane_dispatch.sh:22`, `heartbeat_cmd.sh:5`,
  `mem_cap.sh:13`, `parallel_audit.py:33` (and its Claude-project dir, derived from the root),
  `cost_audit.py:23,31`, `audit_pool_fragment_claims.py:60` and `gates/corpus_check_required.py:74`, plus the literal in
  `tests/test_gate_corpus_check_required.py:27`. W-only GPU-training scripts (`gpu_queue_autofill.sh`, `train.sh`,
  `gpu_train_watchdog.sh`, `gap4_*`) stay as they are; they run on W, where the path is unchanged.
- **`parallel_audit.py`:** when `SIM_GPU_HOST` is set, run its two GPU probes (`nvidia-smi` utilization and
  `gpu_queue.sh status`) through `ssh -o BatchMode=yes $SIM_GPU_HOST`, with the remote repo at `~/Projects/sim`.
- **`gpu_queue.sh`:** `add` and `start` refuse on a host with no `nvidia-smi` (unless `GPU_QUEUE_NVIDIA_SMI` is set,
  the selftest seam) and print "use tools/gpu_remote.sh". Without this, a job queued on S would sit in a queue no
  daemon serves.

`gpu_remote.sh` sketch (the remote side is sent as a `bash -s` script; the job travels base64-encoded, like the pool
dispatcher's `JOB_B64`):

```bash
HOST=${SIM_GPU_HOST:-workstation}; WREPO='$HOME/Projects/sim'          # WREPO expands on W
mkwt() { cat <<EOF                                                      # $1 sha, $2 remote worktree path
set -euo pipefail; R=$WREPO; W=$2
git -C "\$R" fetch -q server
git -C "\$R" worktree add -q --no-checkout --detach "\$W" $1
git -C "\$W" sparse-checkout set --no-cone '/*' '!/research/findings/raw/'
git -C "\$W" reset -q --hard
ln -s "\$R/.venv" "\$W/.venv"; ln -s "\$R/data" "\$W/data"
EOF
}
# add: refuse if `git diff --quiet HEAD` fails or `git branch --contains HEAD` is empty, then:
{ mkwt "$sha" "\$HOME/derisk-pool/revisions/$sha-\$(date +%s)"
  echo "exec bash \"\$W/tools/local_llm/gpu_add.sh\" \"cd \$W && \$(printf %s $(printf %s "$cmd" | base64 -w0) | base64 -d)\""
} | ssh -o BatchMode=yes "$HOST" bash -s
# status: bash $WREPO/tools/gpu_queue.sh status      log N: tail -n N $WREPO/research/queue/gpu_queue.log
# yield:  bash $WREPO/tools/local_llm/llm.sh yield --max-hours N
# provision <sha>: mkwt into $HOME/Projects/sim-worktrees/prov-<sha>-<epoch>, then
#                  cd there && bash tools/pool_provision.sh --isolated --revision <sha>
```

## 4. Prep: safe and non-disruptive, can run now

None of this touches W's queues, dispatchers, units or the pool nodes. All W access is read-only (`rsync` from W,
`git worktree` on a new branch).

1. **Build on a branch in a worktree on W.** Use `.claude/worktrees/home-to-server`, branch `research/home-to-server`
   off `main`. Implement section 3's repo files and edits, run the touched tests
   (`tests/test_gpu_remote.py`, `tests/test_gate_corpus_check_required.py`, `tests/test_queue_job_shape_check.py`,
   `tools/gpu_queue.sh --selftest`) and commit through the gates. Publish with
   `bash tools/push_both.sh research/home-to-server`. Do not merge into W's main checkout: its working tree runs the
   live dispatcher.
2. **S, keys.** Generate `ssh-keygen -t ed25519 -N '' -f ~/.ssh/sim_ed25519 -C 'dant123@srv-host-0 sim: pool+workstation'`
   and `ssh-keygen -t ed25519 -N '' -f ~/.ssh/sim_deploy_ed25519 -C 'srv-host-0 neural-simulator deploy'`. Then run
   `docker exec -u 99 hermes ssh-keygen -t ed25519 -N '' -f /opt/data/.ssh/pool_ed25519 -C 'hermes@srv-host-0 -> pool'`.
3. **S, ssh client config.** Append Appendix B's dant123 stanzas to `~/.ssh/config`, and write Appendix B's Hermes
   file as uid 99 with mode 0600 to `/opt/data/.ssh/config`. This is client config only; nothing on a remote changes.
4. **S, uv.** From W, copy the owner's binary: `scp ~/.local/bin/uv homelab-server:.local/bin/uv`.
5. **S, clone.** On the server, run these in order:
   - `git clone --config core.sharedRepository=group https://git.dant123.com/dant123/neural-simulator.git
     /home/dant123/Projects/neural-simulator`
   - `cd` into it, then `git remote rename origin gitea`
   - `git remote add origin https://github.com/danthi123/neural-simulator.git`
   - `git remote set-url --push origin git@github-sim:danthi123/neural-simulator.git`
   - `git remote set-url --push gitea gitea-sim:dant123/neural-simulator.git`
   - `git config core.hooksPath tools/githooks`
   - copy W's `.git/info/exclude` lines and append `/.python/`
   - `git config --global --add safe.directory '/home/dant123/Projects/*'`
   - `git fetch origin`
6. **S, venvs, then the merge.** Build the venvs with uv on Python 3.11; the merge comes last because the git hooks
   need `S/.venv`:
   - `export UV_PYTHON_INSTALL_DIR=$PWD/.python`, then `uv python install 3.11` and `uv venv --seed --python 3.11 .venv`
   - `uv pip install --python .venv/bin/python -r requirements-pool.txt -r requirements-dev.txt`. This is the CPU set
     already proven on the pool nodes plus the dev tools; no cupy is installed on a GPU-less box.
   - `uv venv --seed --python 3.11 .venv-rag`
   - `uv pip install --python .venv-rag/bin/python torch --index-url https://download.pytorch.org/whl/cpu`
   - `uv pip install --python .venv-rag/bin/python -r requirements-rag.txt`
   - `git merge --no-ff gitea/research/home-to-server` into `main`, using the server's existing git identity. From here
     on, `main` is written only on S.
7. **W to S, bulk copy.** Run from W; these are reads there and can be re-run. `-rlt` without `-g`, so the setgid
   dirs assign group `users`:
   - `rsync -rlt --chmod=Dg+rwx,Fg+rw --exclude=/corpus/corpus ~/Projects/sim/data/ homelab-server:Projects/neural-simulator/data/`
     (`corpus/corpus` is a self-symlink to a W path).
   - `rsync -rlt --chmod=Dg+rwx,Fg+rw ~/Projects/sim-data ~/Projects/rag_index ~/Projects/soma_bundles ~/Projects/sim-catalog homelab-server:Projects/`
   - The uncommitted-research pre-pass, which is cutover step 4's command.
8. **S, unit files.** Write Appendix A's three files into `~/apps/host/etc/systemd/system/` without installing them.
   Add a row per unit to `~/apps/host/README.md`'s table, and a short "neural-simulator" section to
   `~/apps/OWNERSHIP-MAP.md`: paths, the two units, `journalctl -u sim-sync`, how to restart, and "the GPU lives on
   the workstation". `apps-gitsync` mirrors these on its own.
9. **S, smoke checks.** Run `bash tools/githooks/pre-commit` (empty index, so only the gates run),
   `.venv/bin/python -m pytest -q -x tests/test_doc_rules.py tests/test_determinism.py`,
   `.venv-rag/bin/python tools/rag/rag_search.py "gap4 wall" 3 --corpus finding` and
   `bash tools/gpu_remote.sh --dry-run status`.
10. **Hermes, workstation profile.** Create it without touching `default`: run `hermes profile create workstation
    --clone`, then set its `terminal:` block to `backend: ssh`, `ssh_host: 192.168.0.68`, `ssh_user: dant123`,
    `ssh_key: /opt/data/.ssh/workstation_ed25519` and `cwd: /home/dant123/Projects` (the file's path is shown by
    `hermes profile show workstation`).

## 5. Owner steps (security, credentials, sudo), in order

*§10 changes steps 11 to 13:* the keys move from dant123 to user `sim`, and the units, `sim-publish`, `backup.sh`
and `sanoid.conf` are installed from `~/apps/host`. The recreation recipe is in `~/apps/OWNERSHIP-MAP.md` §11.
Steps 1 to 10 are unchanged.

Run on the workstation's terminal unless marked `[server]`. Each command is fish- and bash-safe.

1. `[server]` `sudo apt install -y acl`
2. `ssh homelab-server cat .ssh/sim_ed25519.pub | sed 's/^/from="192.168.0.10" /' >> ~/.ssh/authorized_keys`
3. `ssh homelab-server 'cat .ssh/sim_ed25519.pub; sudo cat /home/dant123/data/hermes/.ssh/pool_ed25519.pub' | sed 's/^/from="192.168.0.10" /' | ssh pool41 'cat >> ~/.ssh/authorized_keys'`
4. `ssh homelab-server 'cat .ssh/sim_ed25519.pub; sudo cat /home/dant123/data/hermes/.ssh/pool_ed25519.pub' | sed 's/^/from="192.168.0.10" /' | ssh pool42 'cat >> ~/.ssh/authorized_keys'`
5. `ssh homelab-server cat .ssh/sim_deploy_ed25519.pub > ~/sim_deploy.pub`
6. `gh repo deploy-key add ~/sim_deploy.pub -R danthi123/neural-simulator --allow-write --title srv-host-0-sim`
7. `[browser]` Gitea: dant123/neural-simulator, Settings, Deploy Keys, Add Deploy Key. Paste `~/sim_deploy.pub`, title
   `srv-host-0-sim`, and tick "Enable Write Access".
8. `sudo ufw allow from 192.168.0.10 to any port 8080 proto tcp comment 'llama-swap for Hermes (sim driver)'`
9. `ssh homelab-server 'mkdir -p -m 700 .claude-config/secrets'; scp -p ~/.claude-config/secrets/vikunja.json homelab-server:.claude-config/secrets/vikunja.json`
10. `[server]` `sudo install -D -o 99 -g 100 -m 600 /home/dant123/.claude-config/secrets/vikunja.json /home/dant123/data/hermes/home/.claude-config/secrets/vikunja.json`
11. `[server]` `sudo install -m 644 -t /etc/systemd/system ~/apps/host/etc/systemd/system/sim-pool-dispatch.service ~/apps/host/etc/systemd/system/sim-sync.service ~/apps/host/etc/systemd/system/sim-sync.timer && sudo systemctl daemon-reload`
12. `[server, only at cutover step 6]` `sudo systemctl enable --now sim-pool-dispatch.service sim-sync.timer`
13. `[server, optional, recommended]` `sudoedit /etc/sanoid/sanoid.conf`: add a `[data/projects]` stanza copied from
    `[data/files]`, then `cp /etc/sanoid/sanoid.conf ~/apps/host/etc/sanoid/sanoid.conf`.

Later, not today: to run AWS pool nodes from S, copy `~/.aws` and the aws-train `.pem` to S. That is a credential move,
so it is an owner step too.

## 6. Cutover (pause window of about 20 min, pool lane only; the GPU lane never pauses)

0. **Preconditions.** All must hold:
   - The pool.queue requeue workflow has finished: its report is in, and `flock -n research/queue/pool.queue.lock true`
     succeeds on W.
   - Prep is done and owner steps 1 to 11 are done.
   - From S, `ssh pool41 true`, `ssh pool42 true` and `ssh workstation true` pass, both as sim (§10's `simrun`) and
     through `docker exec -u 99 hermes`.
   - `git push --dry-run origin main` and `git push --dry-run gitea main` pass on S.
   - No driver is committing on W's main checkout.
   - `git -C ~/Projects/sim status --porcelain -- sim webapp tools tests docs experiment` is empty on W; commit any code
     WIP first.
1. **W: stop the pool dispatcher; the pause starts.** `systemctl --user stop pool-dispatch.service`. Node jobs keep
   running.
2. **W: final pull, then stop the pool timers.** `bash -c 'cd ~/Projects/sim && bash tools/pool_sync.sh'`, then
   `systemctl --user stop pool-sync.timer pool-opsweep.timer`. The opsweep service already fails with 203/EXEC.
3. **W to S: pool-lane and gate state.**

   ```bash
   bash -c 'cd ~/Projects/sim && rsync -rlt --chmod=Dg+rwx,Fg+rw \
     --exclude="gpu*" --exclude=".gpu*" --exclude=GPU_PAUSE --exclude=GAME_MODE --exclude=".llm*" \
     --exclude=".local_llm*" --exclude="llm_*" --exclude="*qwen*" --exclude="vllm_*" --exclude="hermes_*" \
     --exclude=".aws_*" --exclude="aws*" --exclude=".pool_ssh_*" --exclude="*.lock" --exclude=".*.lock" \
     --exclude=gnw_congruence_flip_arm3.log --exclude="autofill*" --exclude=.autofill_cycle \
     research/queue/ homelab-server:Projects/neural-simulator/research/queue/'
   ```
4. **W to S: uncommitted research state** (modified and untracked files, overwriting S's), then re-run step 7 of prep:

   ```bash
   bash -c 'cd ~/Projects/sim && git ls-files -z -m -o --exclude-standard -- research ":!research/queue" \
     | rsync -rlt --from0 --files-from=- --ignore-missing-args --chmod=Dg+rwx,Fg+rw ./ \
       homelab-server:Projects/neural-simulator/'
   ```
5. **S: sanity.** If W's main moved since prep, run `git fetch gitea && git merge gitea/main` as the agent
   (§10; before step 11 the container sees S only at `/workspace`:
   `docker exec -u 99 -e HOME=/opt/data/home hermes git -c 'safe.directory=*' -C /workspace/neural-simulator ...`). Check that
   `wc -l research/queue/pool.queue research/queue/pool.running` equals W's, and that
   `bash tools/pool_queue.sh list | tail -3` looks right.
6. **Owner step 12: enable S's units; the pause ends.**
7. **S: verify the handover.** Check `journalctl -u sim-pool-dispatch -n 30` (§10: no dispatch.log on S) for S probing
   pool41 and pool42. The
   per-node `pgrep -fc research.runners` counts should be unchanged across the handover, meaning no duplicate launch.
   Then check `journalctl -u sim-sync -n 40` after the first timer run, which should end in `sim-publish: verified:
   both remotes` (the pull dry-runs are §10's checklist, run as sim, never the workspace's tools as dant123).
8. **W: disable the moved units.** `systemctl --user disable pool-dispatch.service pool-sync.timer pool-opsweep.timer`.
   The unit files stay, for rollback.
9. **W: become a follower and compute node.**
   - (§10) no `server` remote: `gpu_remote.sh` pushes each job's revision to W.
   - `mkdir -p ~/derisk-pool/sim/research/findings/raw`
   - `mv ~/.local/state/sim-local-llm/last_session ~/.local/state/sim-local-llm/last_session.retired-2026-09-30`
10. **S: finish the shared-folder permissions**, after owner step 1 and before Hermes writes:
    `setfacl -R -d -m g::rwX /home/dant123/Projects && chmod -R g+rwX /home/dant123/Projects/neural-simulator`.
11. **Hermes.**
    - Apply Appendix C's compose edit, then `cd ~/apps && docker compose up -d hermes hermes-dashboard`.
    - Point the default profile at S and llama-swap with
      `docker exec -u 99 hermes /opt/hermes/.venv/bin/hermes config set <key> <value>`: `terminal.cwd` =
      `/home/dant123/Projects/neural-simulator`, `model.base_url` = `http://192.168.0.68:8080/v1`, `model.default` =
      `qwen38-27b-iq4nl-mtp-128k-q4`.
    - Set git identity and safe.directory in the tools' HOME:
      `docker exec -u 99 -e HOME=/opt/data/home hermes git config --global user.name 'Daniel Thiberge'`, the same for
      `user.email` (W's `git config user.email`), and `--add safe.directory '/home/dant123/Projects/*'`.
12. **GPU bridge canary** (from S as the agent, `docker exec -u 99 -e HOME=/opt/data/home -w
    /home/dant123/Projects/neural-simulator hermes`, never as dant123, §10; skip it while Hermes's W key is withdrawn):
    `bash tools/gpu_remote.sh add 'mkdir -p research/findings/raw/_migration && .venv/bin/python -c "import json,cupy; json.dump({\"gpus\": cupy.cuda.runtime.getDeviceCount()}, open(\"research/findings/raw/_migration/gpu_bridge_canary.json\",\"w\"))"'`.
    If `local-llm` is up, follow with `bash tools/gpu_remote.sh yield --max-hours 1`. After it runs, the next
    `sim-sync` pull should put the file in S's `research/findings/raw/_migration/`. The queue restores the model
    after the drain.
13. **Hermes smoke turn** in the default profile: `echo $HOME`, `git status`, `bash tools/gpu_remote.sh status`,
    `bash tools/pool_queue.sh list | tail -2`, `ssh pool41 true`, and `bash tools/vikunja.sh list-tasks 2 | head -3`.
14. **Docs** (section 8): commit on S as the agent through the gates (as in step 12, never as dant123: the gates are
    agent-writable hooks); `sim-sync` publishes it. Sync the board (vikunja).
15. **Hermes cron, last:** `docker exec -it -u 99 hermes /opt/hermes/.venv/bin/hermes cron create`, every 30 min in the
    default profile, with the prompt "Continue per CLAUDE.md and research/coordination/LOCAL_LLM_RUNBOOK.md; one task
    per run; this repo's path is /home/dant123/Projects/neural-simulator, never /workspace".
16. **Stragglers.** Pre-cutover GPU jobs that `cd` into W's main checkout write to W's raw. Once `gpu_remote.sh status`
    shows none left, run
    `bash -c 'cd ~/Projects/sim && git ls-files -z -o --exclude-standard -- research/findings/raw | rsync -rlt --from0 --files-from=- --ignore-existing --chmod=Dg+rwx,Fg+rw ./ homelab-server:Projects/neural-simulator/'`.

## 7. Rollback (the workstation stays fully capable until step 8, and after it too)

1. S, owner (sudo): `sudo systemctl disable --now sim-pool-dispatch.service sim-sync.timer`. Pause the Hermes cron job
   (`hermes cron pause <id>`).
2. W: copy the pool state back with
   `rsync -rlt 'homelab-server:Projects/neural-simulator/research/queue/{pool.queue,pool.queue.claims,pool.running,.pool_reservations,pool_canaries.tsv}' ~/Projects/sim/research/queue/`
   (add any `pool.queue.held*` touched since cutover).
3. W: `systemctl --user enable --now pool-dispatch.service pool-sync.timer`.
4. W: to bring back the local Claude driver, rename `last_session.retired-2026-09-30` back.
5. Results made on S in the meantime: `rsync -rlt --ignore-existing` from S's `research/findings/raw/` into W's.
   Nothing on either host is deleted, the GPU lane never changed, and the Hermes compose edit reverts through git in
   `~/apps`.

## 8. Docs to change (in cutover step 14, on S)

- **`CLAUDE.md`:** keep it small.
  - Replace the "Transition to local-model-first" line with a "Where things run" line pointing here: home = S,
    driver = Hermes, GPU and model serving = W.
  - Non-negotiable 8: on S, `sim-sync.timer` runs `sim-publish` (§10); from W, changes go to S as branches.
  - Non-negotiable 9: from S, `tools/gpu_remote.sh add`; `game.sh` and `gpu_queue.sh pause` are W commands.
  - Non-negotiable 10: no heavy jobs on the server. Hermes is capped at 12 GB, `memcap.sh` does not run inside Hermes,
    and heavy work goes to the pool.
  - Non-negotiable 12: the one driver is Hermes; `llm.sh claude` is retired.
  - Workflow essentials: on S, the heartbeat audit runs in `sim-sync`, and a Hermes cron job replaces the Monitor.
- **`research/coordination/LOCAL_LLM_RUNBOOK.md` (v4, Hermes):**
  - `gpu_add.sh` becomes `gpu_remote.sh add`, and `llm.sh yield` becomes `gpu_remote.sh yield`.
  - `push_both.sh` is not run by Hermes (sim-sync does it; check with `git ls-remote origin main`).
  - `memcap.sh` becomes "not available here, use the pool".
  - Provisioning uses `gpu_remote.sh provision <sha>`.
  - Always use S's path, never `/workspace`.
  - `game.sh` is owner-only on W.
- **`GAP_CLOSURE_MISSION.md`:** update CURRENT STATE and the "SESSION START" heartbeat recipe (on S it is `systemctl
  status sim-sync.timer`).
- **`research/findings/AUTONOMOUS_STATE.md` and `ROADMAP.md`:** one line each.
- **`docs/plans/2026-09-29-local-model-transition-PLAN.md`:** a pointer line (the driver is now Hermes on srv-host-0).
- **Board:** one card for the move, done, and one for the follow-ups below.

## 9. Known limits and risks (stated, not hidden)

- **Guardrails.** The Claude Code guardrails (the deny rules in `tools/local_llm/claude_local_settings.json` and the
  PreToolUse hooks) do not apply under Hermes. The enforcement that remains is the pre-commit gates, `gpu_add.sh`
  (through `gpu_remote.sh`), `pool_queue.sh`/`pool_canary.sh` and the Hermes container's own limits. Hermes's
  command-approval settings are the owner's call.
- **Exposure.** llama-swap (:8080, no API key) and Ollama (:11434) on W become reachable from every container on the
  server, since container egress is NATed to .10. This is the same exposure the existing 11434 rule already has.
- **Memory readings.** Inside the container `mem_ok.sh` reads the host's `/proc/meminfo`, not the 12 GB cap. A job over
  the cap is OOM-killed inside the Hermes cgroup.
- **Backups.** S is outside restic and sanoid, and `main` reaches both remotes every 15 min. Untracked results live on
  S (and on the nodes and W until pruned), so commit results promptly, and consider owner step 13.
- **Frozen compute checkout.** W's main checkout is frozen at cutover; GPU jobs use per-job worktrees, so job code is
  always current. Updating W's own GPU tooling (`gpu_queue.sh`, `llm.sh`, `game.sh`) is deliberate:
  `git stash push -m pre-update`, then `git pull --ff-only server main`, then
  `sudo systemctl restart gpu-queue-dispatch` (not done in this workflow). An untracked raw file at a path S has since
  committed blocks that pull until it is moved aside.
- **Worktree growth.** Per-job worktrees accumulate under W's `~/derisk-pool/revisions/`. Remove them by hand
  (`git worktree remove`) once their results are on S; never automatically.
- **Audit gaps.** `parallel_audit.py`'s agent floor reads Claude transcripts, so on S it reports `under_agents`. That
  gate is report-only and never blocks.
- **Nightly stop.** `backup.sh` stops the `hermes` stack nightly for a few minutes. An in-flight Hermes turn dies, and
  the cron job resumes the work. The host units are unaffected.
- **Branch-only commits.** W's 42 unpushed branch tips and 8 stashes stay on W, where nothing deletes them.
  Optionally, once owner step 2 is done, mirror them on S as remote-tracking refs (no remote gets them):
  `git remote add workstation workstation:Projects/sim && git fetch workstation`.

## 10. Trust boundary (2026-09-30 evening; design reviewed by the server-admin session)

**The problem.** The first units (Appendix A draft) ran as dant123, who has passwordless sudo and docker, so is
root-equivalent. They executed scripts from the workspace, which the Hermes agent (uid 99, group `users`) can
rewrite, since the tree is group-writable. And any git command a trusted user runs inside the workspace executes its
agent-controlled `.git/config` (`core.fsmonitor`, `core.sshCommand`, `core.hooksPath`, aliases). So the agent could
have become root on the server. The units were stopped the same evening.

**The model.**

- **The workspace is untrusted data.** No privileged process executes anything from it and none runs git in it. Its
  path reaches the tools only as `SIM_WORKSPACE`.
- **User `sim`** is a system user with no login shell, no sudo and no docker, in group `users`, with home
  `/home/sim` at 0700. It holds the two keys: `sim_ed25519` for the pool nodes and the workstation, and
  `sim_deploy_ed25519`, the GitHub/Gitea write deploy key. Both units run as `User=sim Group=users UMask=0002`.
- **Code runs from outside the agent-writable tree.** `/home/sim/runtime` (0750, sim-owned) is a clone pinned to a
  reviewed `main` commit. It is updated only deliberately: `git fetch`, then `git merge --ff-only <reviewed-sha>`,
  both run as sim. `/usr/local/sbin/sim-publish` is root-owned. Any future privileged unit for a project follows the
  same rule.
- **Publishing.** `sim-publish` fetches the workspace's `main` into `/home/sim/mirror.git`, fast-forward only. It pushes
  that `main` to both remotes with the deploy key and checks both with `ls-remote`. It never forces and pushes no
  tags. `git fetch <path>` runs only git-upload-pack in the workspace, which does not honour its command-bearing
  config. This was verified against a hostile, differently-owned repo: nothing ran, and no `safe.directory` was
  needed. So sim has **no** `safe.directory` for the workspace, and any porcelain git as sim there is refused.
- **Confinement** (systemd, beyond the reviewed design). sim cannot traverse `/home/dant123` (0750). The units therefore
  see an empty tmpfs there, with only `/home/dant123/Projects` bind-mounted at its usual path, and they can write only
  that dataset: `ProtectSystem=strict` and `ProtectHome=read-only`. `sim-sync` can also write its mirror and
  `/var/lib/sim-sync`. The reason is that sim's writes into the workspace follow any symlink the agent plants.
  `fs.protected_symlinks` does not cover group-writable, non-sticky directories, so without the confinement a planted
  `pool.running -> /home/sim/runtime/tools/pool_sync.sh` would let queue text be appended to code that sim runs.
- **Reads that copy content out** are guarded too. In runtime mode the dispatcher copies the queue into a private
  snapshot, and only if the opened file is a regular file on the workspace's own filesystem. A symlink to
  `~/.ssh/sim_ed25519` would otherwise be quarantined line by line into `pool.queue.malformed`, which the agent can
  read.
- **Results.** rsync follows symlinks in its destination path (reproduced: `raw -> /elsewhere` lands there). In runtime
  mode `pool_sync.sh` therefore pulls into `/var/lib/sim-sync/stage`, and `tools/safe_copy_into.py` walks the
  destination from the dataset mount with `O_NOFOLLOW`. It writes a fresh file and renames it into place, and it
  refuses a symlink anywhere on the way. It keeps `rsync -u` semantics, needs only the stdlib and runs under the
  system python3 with no venv.
- **Node answers are untrusted.** The dispatcher's capacity probe passed two node-controlled fields (`cores`,
  `total_gb`) straight into bash arithmetic, where `$(( ROOT[$(cmd)] ))` runs `cmd` on the dispatcher host. This was
  reproduced on the pre-fix script, and every field is now checked as a number first. Node names read from files must
  be plain host aliases (`-oProxyCommand=...` is an ssh option). In runtime mode, the repo-local ssh config, the
  extra-node list, the AWS state and the pause markers are read from the runtime, not the workspace. The runtime has
  none of them, so the AWS lane is off on S.
- **Pool jobs** run with `POOL_SANDBOX=on` (bubblewrap, no network; `research/pool-job-sandbox`). A node that cannot
  sandbox gets no work.
- **Logs** go to the journal (no `research/queue/dispatch.log`). `tools/gates/lane_starvation.py` now also reads
  `pool.running`, which carries the same dispatch record with a date.

**Why the workstation pull said UNREACHABLE.** It was reachable: ssh with `sim_ed25519` works, and the audit's GPU probe
over the same alias succeeded in the same run. The workstation has no `~/derisk-pool/sim/research/findings/raw`, so
rsync's main pull exited 23 ("change_dir failed"). `pool_sync.sh` treated every non-zero rsync exit as UNREACHABLE and
skipped the host's revision pulls with it. rc 23/24 now reads as "reachable, main results dir missing" and the
revisions are still pulled. The workstation also has the placeholder dir now (cutover step 9's `mkdir`, done
2026-09-30). A second defect sat behind the first. The revision listing ran as a bare `ls -d` in the remote login
shell, and on the workstation that is fish with `ls` wrapped in a long, coloured listing. The listing returned
permission, owner and date tokens, so no workstation revision could have been pulled even with the dir in place. It
now runs through `sh -c`. A real dry-run from srv-host-0 lists the three GPU-job revisions (97 files).

**Residual risks (stated).** The agent can still stop its own work: it can corrupt the queue, plant symlinks that make
the dispatcher refuse the queue, or push a `main` that fails to publish. None of that escalates. Hermes's own
workstation key (`workstation_ed25519`, W's `authorized_keys` line with `from=` only) is a full dant123 shell on W, and
that is outside this boundary (but see the re-review below: it loops back to S). What `sim-publish` pushes is whatever
the agent committed on `main`: the gates in the agent's checkout are advisory, and GitHub/Gitea branch protection is the
only remote-side check (it did not exist on 2026-09-30; owner step below).

**Security re-review (2026-09-30 night, server-admin session; verdict UNSOUND until the W chain is broken).**

- **CRITICAL: the boundary was bypassable from W.** Hermes's W key is a full dant123 shell on W. W's dant123 held a
  passphrase-free key (`~/.ssh/homelab_ed25519`) that S's dant123 trusts, and S's dant123 has passwordless sudo: agent
  to root on S in three hops, without touching `sim`. W's dant123 also has passwordless sudo. Fixed in code:
  `gpu_remote.sh` now pushes each job's revision to W (`refs/server-jobs/<sha>`) over Hermes's own connection, so W
  needs no credential to S. Owner steps: S trusts a Bitwarden-held key instead (Bitwarden asks on every use), no
  longer the on-disk key; W's NOPASSWD sudo goes (`train.sh` and `gpu_recover.sh` then ask for a password). The
  on-disk key stays on W for pool40-42 (the agent is root there anyway). Note: the Bitwarden agent's `homelab_ed25519`
  item is a *different* key from the file of that name; S trusted only the file.
  **Still open (owner decision):** Hermes lands as dant123 on W, and every GPU job runs as dant123 there, so
  everything W's dant123 holds is agent-reachable: the owner's `gh` token (`repo` scope, admin on this repo, so it can
  also undo branch protection), `~/.aws` credentials, the Gitea account key (the same on-disk file is dant123's Gitea
  user key "Unraid": git read/write on every repo of the owner's, `srv-host-0` included) and the libsecret Gitea
  credential. The owner's own sessions on W are steerable too (`~/.ssh/config` ControlMaster, a `PATH` shim, shell
  rc), so one owner login from W to S can still be hijacked. The fix is to stop the agent landing in the owner's
  account: a dedicated unprivileged GPU-job user on W, or a `restrict,command=` gate plus bubblewrap-confined GPU jobs.
  Until then the containment is to remove Hermes's W key (the GPU lane from Hermes pauses; the pool lane and sim's
  result pull are unaffected).
- **Pool nodes are agent-controlled.** Hermes's node key (`from=` only) reaches `node`, which has passwordless sudo,
  so `POOL_SANDBOX=on` constrains pool-submit lines, not Hermes. Every byte sim receives from pool41/42 and W is
  hostile: the defences are the client side (node answers checked as numbers, revision paths by regex, rsync
  `--safe-links` + allow-list + `--max-size` + a capped stage, the symlink-refusing copy) and a patched rsync/OpenSSH
  on S (standing requirement; 2026-09-30: rsync 3.4.1, OpenSSH 10.2, unattended-upgrades on). `sim_ed25519` is
  `from=`-only on the nodes (root via `node`'s sudo), so it must never leave `/home/sim`. On W it was a full dant123
  shell (the owner's desktop, passwordless sudo); W's `authorized_keys` now forces it to `rrsync -ro ~/derisk-pool`
  (owner step) and `pool_sync.sh` lists W's revisions with rsync itself (`POOL_SYNC_RRSYNC_NODES=workstation`), so a
  compromised sim can read W's results and nothing else.
- **No remote-side history protection.** GitHub `main` had no protection (404), and the deploy key writes every branch.
  Owner step: a GitHub ruleset on all branches (no force-push, no deletion; repository admins bypass) and a Gitea
  protection rule `**`.
- **Container root on the dataset.** The Hermes container's init runs as uid 0 without a user-namespace remap and binds
  `~/Projects` read-write; `data/projects` had `setuid=on devices=on`. Owner step: `zfs set setuid=off devices=off`
  (venvs need `exec`, which stays on). Longer term: userns-remap or a non-root init for the container.
- **Smaller fixes in code and units.** `--peek-est-gb`/`--min-queued-est-gb` read the queue through the same
  device-checked snapshot as `pop_job`; the stage is bounded (`POOL_SYNC_MAX_FILE_SIZE`, `POOL_SYNC_STAGE_MAX_MB`,
  loud exit 1 at the cap); `sim-pool-dispatch` has `RestartPreventExitStatus=1` and a start limit (5 in 10 min), so a
  refusal or crash loop alerts a bounded number of times; sim's ssh uses `StrictHostKeyChecking yes` (an unknown host
  fails, as a changed one does).
- **Never run the workspace's git or tools as dant123 on S, including during cutover.** Steps 5, 12 and 14 run as the
  agent: `docker exec -u 99 -e HOME=/opt/data/home hermes git -c 'safe.directory=*' -C /workspace/neural-simulator ...`
  (the path that exists before step 11's bind; afterwards `-w /home/dant123/Projects/neural-simulator`). Before the
  W-to-S rsyncs (steps 3, 4, 16), check that `find /home/dant123/Projects/neural-simulator/research -type l` prints
  nothing: rsync as dant123 follows a symlinked destination directory.

## Appendix A: host units (`~/apps/host/etc/systemd/system/`)

*Superseded 2026-09-30 evening by §10.* The authoritative unit files are the ones in `~/apps/host` on the server
(`sim-pool-dispatch.service`, `sim-sync.service`, `sim-sync.timer`, plus `usr/local/sbin/sim-publish`); the text below
is the first draft, which ran as dant123 from the workspace, and is kept only as history.

`sim-pool-dispatch.service`

```ini
# neural-simulator (~/Projects/neural-simulator): pool dispatcher, research/queue/pool.queue -> pool41/pool42 over
# ssh (~/.ssh/sim_ed25519). Moved from the workstation 2026-09-30 (docs/plans/2026-09-30-home-to-server-MIGRATION.md).
# Restart it after changing tools/pool_autodispatch.sh.
[Unit]
Description=neural-simulator pool dispatcher (pool.queue -> pool41/pool42)
After=network-online.target
Wants=network-online.target
OnFailure=ntfy-notify@%n.service

[Service]
Type=simple
User=dant123
Group=users
UMask=0002
Environment=HOME=/home/dant123
WorkingDirectory=/home/dant123/Projects/neural-simulator
ExecStart=/usr/bin/bash /home/dant123/Projects/neural-simulator/tools/pool_autodispatch.sh
Restart=always
RestartSec=30
StandardOutput=append:/home/dant123/Projects/neural-simulator/research/queue/dispatch.log
StandardError=append:/home/dant123/Projects/neural-simulator/research/queue/dispatch.log

[Install]
WantedBy=multi-user.target
```

`sim-sync.service`

```ini
# neural-simulator: pull pool + workstation GPU results, refresh the audit (feeds the freshness gates) and the RAG
# index, then publish main to GitHub + Gitea with the deploy key (~/.ssh/sim_deploy_ed25519). Logs: journalctl -u sim-sync
[Unit]
Description=neural-simulator sync (results in, audit, RAG, push main)
After=network-online.target
Wants=network-online.target
OnFailure=ntfy-notify@%n.service

[Service]
Type=oneshot
User=dant123
Group=users
UMask=0002
Environment=HOME=/home/dant123
Environment=SIM_GPU_HOST=workstation
Environment=SIM_CANONICAL_REPO=/home/dant123/Projects/neural-simulator
WorkingDirectory=/home/dant123/Projects/neural-simulator
TimeoutStartSec=40min
ExecStart=/usr/bin/bash tools/pool_sync.sh
ExecStart=/usr/bin/env POOL_NODES=workstation /usr/bin/bash tools/pool_sync.sh
ExecStart=-/home/dant123/Projects/neural-simulator/.venv/bin/python tools/parallel_audit.py
ExecStart=-/usr/bin/bash tools/rag/periodic_update.sh
ExecStart=/usr/bin/bash tools/push_both.sh main
```

`sim-sync.timer`

```ini
[Unit]
Description=neural-simulator sync every 15 minutes

[Timer]
OnCalendar=*:0/15
Persistent=true

[Install]
WantedBy=timers.target
```

## Appendix B: ssh client config

dant123 on the server (append to `~/.ssh/config`):

```sshconfig
Host pool41
  HostName 192.168.0.41
  User node
Host pool42
  HostName 192.168.0.42
  User node
Host workstation
  HostName 192.168.0.68
  User dant123
Host pool41 pool42 workstation
  IdentityFile ~/.ssh/sim_ed25519
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
Host github-sim
  HostName github.com
  User git
  IdentityFile ~/.ssh/sim_deploy_ed25519
  IdentitiesOnly yes
Host gitea-sim
  HostName 127.0.0.1
  Port 2222
  User git
  IdentityFile ~/.ssh/sim_deploy_ed25519
  IdentitiesOnly yes
```

Hermes (`/opt/data/.ssh/config`, owner uid 99, mode 0600; ssh reads the passwd home `/opt/data`, not `$HOME`):

```sshconfig
Host workstation
  HostName 192.168.0.68
  User dant123
  IdentityFile /opt/data/.ssh/workstation_ed25519
Host pool41
  HostName 192.168.0.41
  User node
  IdentityFile /opt/data/.ssh/pool_ed25519
Host pool42
  HostName 192.168.0.42
  User node
  IdentityFile /opt/data/.ssh/pool_ed25519
Host *
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
```

## Appendix C: Hermes compose edit (`~/apps/hermes/compose.yaml`, the `x-hermes` anchor)

```yaml
x-hermes: &hermes
  image: nousresearch/hermes-agent:latest
  restart: unless-stopped
  security_opt: [ "no-new-privileges:true" ]
  mem_limit: 12g        # neural-simulator driver (2026-09-30): sim tests run in here; the cap protects the other stacks
  cpus: 8
  volumes:
    - ${DATA_ROOT}/hermes:/opt/data
    - /home/dant123/Projects:/workspace
    - /home/dant123/Projects:/home/dant123/Projects   # same path as the host: venvs, git worktrees, hooks are absolute
```

The added bind is outside `backup.sh`'s roots, so the nightly backup plan for the `hermes` stack is unchanged.
