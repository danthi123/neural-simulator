# tools/agent_container/Dockerfile -- the isolated local-model-harness container ("sim-agent"), owner directive
# 2026-09-29 (docs/plans/2026-09-29-local-model-transition-PLAN.md section 9). Runs on homelab-server
# (srv-host-0, 192.168.0.10), a shared box with many other services, as the ~/apps stack `sim-agent`
# (server-stack/sim-agent/compose.yaml). THIS is the one Dockerfile: the stack builds it from the owner's
# checkout at ${DATA_ROOT}/sim-agent/src/tools/agent_container. Deliberately minimal and read-only at runtime
# so a rogue model has nothing to escalate through.
#
# WHAT LIVES WHERE:
#   - this image (read-only at runtime): OS + Node + git/ssh/tmux/jq + the project's own numpy venv (no cupy)
#     + the RAG venv (no GPU torch) -- everything reproducible from the repo's requirements, baked in for speed
#     and so a compromised container can't quietly swap a dependency.
#   - /home/agent, bind-mounted from ${DATA_ROOT}/sim-agent/home (writable, ZFS, refquota-capped): the Claude
#     Code CLI itself, installed by the NATIVE installer at first boot (see entrypoint.sh) so its own built-in
#     auto-updater can keep it current without an image rebuild (owner 2026-09-29). Also: the restricted ssh
#     keys the owner installs, tmux state, Claude Code session state.
#   - /work/sim, bind-mounted from ${DATA_ROOT}/sim-agent/repo (writable): the agent's own git clone.
#   A bind mount starts EMPTY (unlike a named volume, nothing is copied from the image), so the owner creates
#   both directories owned by AGENT_UID:AGENT_GID on the host; nothing here relies on image content under them.
FROM debian:bookworm-slim

ARG AGENT_UID=1000
ARG AGENT_GID=1000
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    AGENT_HOME=/home/agent \
    REPO_DIR=/work/sim \
    SIM_VENV=/opt/venv-sim \
    RAG_VENV=/opt/venv-rag

# --- OS packages -------------------------------------------------------------------------------------------
# python3 here is Debian bookworm's default (3.11), matching the project's pinned .venv (3.11.16, CLAUDE.md /
# tools/local_llm's own docs). build-essential covers the few RAG-venv wheels without prebuilt manylinux
# binaries on some archs; curl+ca-certificates are needed for the Claude Code native installer (run at
# container start, see entrypoint.sh) and for pip's own network access during this build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        git openssh-client tmux jq curl ca-certificates \
        nodejs npm \
        build-essential \
        tini tzdata \
    && rm -rf /var/lib/apt/lists/*

# --- non-root user -------------------------------------------------------------------------------------------
RUN groupadd -g "${AGENT_GID}" agent \
    && useradd -m -u "${AGENT_UID}" -g "${AGENT_GID}" -s /bin/bash -d "${AGENT_HOME}" agent \
    && mkdir -p "${REPO_DIR}" && chown agent:agent "${REPO_DIR}"

# --- project numpy venv (no cupy/torch/GUI -- see requirements-agent.txt's own header) -----------------------
COPY requirements-agent.txt /opt/requirements-agent.txt
RUN python3 -m venv "${SIM_VENV}" \
    && "${SIM_VENV}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${SIM_VENV}/bin/pip" install --no-cache-dir -r /opt/requirements-agent.txt

# --- RAG venv (CPU-only torch: the server has no GPU) ---------------------------------------------------------
COPY requirements-rag.txt /opt/requirements-rag.txt
RUN python3 -m venv "${RAG_VENV}" \
    && "${RAG_VENV}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${RAG_VENV}/bin/pip" install --no-cache-dir -r /opt/requirements-rag.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# --- container-local bin (baked into the read-only image, including firewall_guard.sh, the fail-closed start
# check entrypoint.sh runs first; the CLAUDE CODE BINARY ITSELF is NOT here -- it is installed into the writable
# home bind mount by entrypoint.sh, see the header comment above). There is
# deliberately NO custom `claude` shim: the native installer's own `~/.local/bin/claude` (a symlink to
# ~/.local/share/claude/versions/<version>, verified live 2026-09-29) is already the correct resolution --
# `claude update`/`claude install <version>` manage that symlink themselves. -------------------------------
COPY bin/ /usr/local/lib/agent-container/bin/
# chmod EVERY file (not just *.sh): `llm` has no extension, and the build source is a git clone in which all of
# these are tracked 0644 (the repo runs core.fileMode=false), so nothing may rely on the checkout's own mode bits.
RUN chmod +x /usr/local/lib/agent-container/bin/* \
    && ln -s /usr/local/lib/agent-container/bin/llm /usr/local/bin/llm \
    && ln -s /usr/local/lib/agent-container/bin/agent-start.sh /usr/local/bin/agent-start.sh \
    && ln -s /usr/local/lib/agent-container/bin/claude-safe-update.sh /usr/local/bin/claude-safe-update.sh \
    && ln -s /usr/local/lib/agent-container/bin/smoke_test.sh /usr/local/bin/smoke_test.sh

COPY profiles/ /usr/local/lib/agent-container/profiles/

# --- MANAGED settings + guardrail hook, baked read-only into the image (2026-09-29 security review, LOW:
# "the in-container guardrails can be edited by the model: the hook lives in the writable repo, and the
# .claude settings are writable"). Two fixes, together:
#   1. claude_container_settings.json (this repo's own copy, kept as a build artifact for reference/CI too) is
#      installed as Claude Code's MANAGED settings file -- /etc/claude-code/managed-settings.json on Linux,
#      the highest-precedence, non-overridable settings tier (enterprise policy, not a project/user setting the
#      model's own repo checkout could ever shadow or edit). It lives OUTSIDE both writable bind mounts
#      (/work/sim, /home/agent) and under this image's read-only root filesystem (the stack fragment's
#      `read_only: true`), so the model cannot edit, delete, or bypass it by editing files in its own checkout.
#   2. The PreToolUse hook it points at (hooks/deny_protected_edits.{sh,py} -- vendored copies of tools/
#      local_llm/hooks/deny_protected_edits.{sh,py}, kept byte-identical: tests/test_agent_container_config.py
#      pins this, since the Docker build context is tools/agent_container/ alone and cannot COPY a file from
#      elsewhere in the repo) is ALSO baked into this read-only image, not read from the writable /work/sim
#      checkout -- the OLD config pointed at "$CLAUDE_PROJECT_DIR"/tools/local_llm/hooks/deny_protected_edits.sh,
#      a path inside the model's own writable repo, which the model could simply edit or delete to neuter its
#      own guardrail.
COPY hooks/ /usr/local/lib/agent-container/hooks/
RUN chmod +x /usr/local/lib/agent-container/hooks/*.sh
COPY claude_container_settings.json /usr/local/lib/agent-container/claude_container_settings.json
RUN mkdir -p /etc/claude-code \
    && cp /usr/local/lib/agent-container/claude_container_settings.json /etc/claude-code/managed-settings.json \
    && chmod 444 /etc/claude-code/managed-settings.json /usr/local/lib/agent-container/hooks/*.py

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# tmux needs a runtime dir; /tmp is tmpfs at runtime (the stack fragment). Nothing else under / is written to.
# ~/.local/bin (the native installer's own launcher location, NOT on PATH by default -- its own install output
# says so, verified live 2026-09-29) must be ahead of nothing else named `claude`; there is none, but this is
# also where `llm`/`agent-start.sh`/etc.'s symlinks above would collide if ever renamed, so put it first.
ENV TMUX_TMPDIR=/tmp \
    PATH="/home/agent/.local/bin:${PATH}"

USER agent
WORKDIR /work/sim
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
