"""Static checks on the sim-agent stack fragment (tools/agent_container/server-stack/sim-agent/compose.yaml), the
Dockerfile and claude_container_settings.json. The isolation properties docs/plans/2026-09-29-local-model-
transition-PLAN.md section 9 requires (own network, no docker socket, dropped capabilities, read-only root,
resource limits, no published ports, non-root) are properties of CONFIGURATION, not runtime behaviour a unit test
can otherwise observe -- so this parses the files and asserts the settings are actually there. No docker is
invoked.

2026-09-30 (research/harness-server-fit): the harness moved from a standalone compose project with named volumes
onto srv-host-0's own conventions -- one ~/apps project (`include:`), one ~/apps/.env, per-app ZFS dataset under
${DATA_ROOT}, nftables firewall. Every isolation test below was kept and retargeted at the fragment; the new
tests pin the server-convention layout (bind mounts, no env_file, not `proxy`, fixed bridge name + subnet).
"""
import os
import re

try:
    import yaml
except ImportError:
    yaml = None
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AC = os.path.join(ROOT, "tools", "agent_container")
STACK = os.path.join(AC, "server-stack")
FRAGMENT = os.path.join(STACK, "sim-agent", "compose.yaml")
NFT = os.path.join(STACK, "host", "etc", "nftables.d", "sim-agent.nft")


def _compose_text():
    return open(FRAGMENT).read()


def _compose():
    if yaml is None:
        pytest.skip("pyyaml not installed in this environment")
    return yaml.safe_load(_compose_text())


def _dockerfile_text():
    return open(os.path.join(AC, "Dockerfile")).read()


def _settings():
    return json.load(open(os.path.join(AC, "claude_container_settings.json")))


def test_compose_service_has_no_published_ports():
    svc = _compose()["services"]["sim-agent"]
    assert "ports" not in svc


def test_compose_service_has_its_own_network_only():
    svc = _compose()["services"]["sim-agent"]
    assert svc.get("network_mode") != "host"
    nets = svc.get("networks")
    assert nets, "sim-agent must declare its own network"
    assert list(nets) == ["sim-agent-net"]
    compose = _compose()
    net = compose["networks"]["sim-agent-net"]
    assert net.get("driver") == "bridge"
    assert not net.get("external"), "its own bridge, created by compose -- never an existing shared network"
    assert not net.get("internal"), "needs egress (DNS/HTTPS/workstation/gitea), governed by sim-agent.nft"


def test_compose_service_is_never_on_the_proxy_network_and_has_no_caddy_route():
    # the proxy net is Caddy's front door: joining it would put every web backend on this box one hop away
    svc = _compose()["services"]["sim-agent"]
    assert "proxy" not in (svc.get("networks") or [])
    assert "proxy" not in (_compose().get("networks") or {})
    labels = svc.get("labels") or {}
    assert "homepage.href" not in labels, "no web UI, no route -- a tile link would point at nothing"


def test_compose_service_has_no_docker_socket_mount():
    svc = _compose()["services"]["sim-agent"]
    vols = svc.get("volumes", [])
    for v in vols:
        v_str = v if isinstance(v, str) else str(v)
        assert "docker.sock" not in v_str
    assert "docker.sock" not in _compose_text().split("volumes:", 1)[1]


def test_compose_service_drops_all_capabilities():
    svc = _compose()["services"]["sim-agent"]
    assert svc.get("cap_drop") == ["ALL"]


def test_compose_service_sets_no_new_privileges():
    svc = _compose()["services"]["sim-agent"]
    assert "no-new-privileges:true" in svc.get("security_opt", [])


def test_compose_service_root_fs_is_read_only_with_tmp_tmpfs():
    svc = _compose()["services"]["sim-agent"]
    assert svc.get("read_only") is True
    tmpfs = svc.get("tmpfs", [])
    assert any(str(t).startswith("/tmp") for t in tmpfs)


def test_compose_tmp_tmpfs_is_noexec_nosuid_nodev():
    # 2026-09-29 re-review, L4: /tmp is writable by every process in the container -- without these mount
    # options, a downloaded/generated file there could simply be chmod +x'd and executed directly.
    svc = _compose()["services"]["sim-agent"]
    tmp_entry = next(str(t) for t in svc.get("tmpfs", []) if str(t).startswith("/tmp"))
    opts = tmp_entry.split(":", 1)[1].split(",")
    for flag in ("noexec", "nosuid", "nodev"):
        assert flag in opts, "%s missing from /tmp tmpfs options: %s" % (flag, tmp_entry)


def test_compose_service_persists_exactly_two_bind_mounts_on_the_zfs_dataset():
    # 2026-09-30: named volumes landed on /bulk/docker (no snapshots, no backup, no cap). The site convention is a
    # per-app ZFS dataset under ${DATA_ROOT}; long syntax with create_host_path: false so an unmounted dataset
    # makes the container FAIL to start instead of Docker creating root-owned dirs outside the refquota.
    svc = _compose()["services"]["sim-agent"]
    vols = svc.get("volumes", [])
    assert len(vols) == 2
    by_target = {}
    for v in vols:
        assert isinstance(v, dict), "long syntax only (create_host_path must be explicit): %r" % (v,)
        assert v.get("type") == "bind"
        assert v.get("bind", {}).get("create_host_path") is False
        assert not v.get("read_only")
        by_target[v["target"]] = v["source"]
    assert by_target == {"/work/sim": "${DATA_ROOT}/sim-agent/repo", "/home/agent": "${DATA_ROOT}/sim-agent/home"}
    assert "volumes" not in _compose(), "no top-level named volumes any more"


def test_compose_never_mounts_the_build_source_into_the_container():
    # src/ is what the image and the ctl forced command are built/run from; the agent must never be able to edit it
    svc = _compose()["services"]["sim-agent"]
    assert not any("/src" in str(v.get("source", "")) for v in svc.get("volumes", []))


def test_compose_service_has_resource_limits_sized_for_the_shared_host():
    # plain (non-swarm) resource fields -- see the fragment's own comment on why not `deploy.resources.limits`.
    # srv-host-0: 16 cores / 61 GB RAM with ~23 GB available -> half the cores, 12 GB, no swap beyond it, and the
    # kernel's first OOM victim if the host itself runs short (never another stack's database).
    svc = _compose()["services"]["sim-agent"]
    assert str(svc.get("cpus")) == "8"
    assert str(svc.get("mem_limit")) == "12g"
    assert str(svc.get("memswap_limit")) == str(svc.get("mem_limit"))
    assert svc.get("pids_limit") == 4096
    assert int(svc.get("oom_score_adj", 0)) > 0


def test_compose_service_restart_policy_is_unless_stopped():
    svc = _compose()["services"]["sim-agent"]
    assert svc.get("restart") == "unless-stopped"


def test_dockerfile_creates_a_non_root_user_and_switches_to_it():
    text = _dockerfile_text()
    assert re.search(r"useradd .*agent", text)
    assert re.search(r"^USER agent", text, re.MULTILINE)


def test_dockerfile_does_not_bake_claude_code_into_the_image():
    # owner 2026-09-29: the CLI is installed into the writable home volume at container start (entrypoint.sh),
    # not the read-only image, so its own auto-updater can manage it.
    text = _dockerfile_text()
    assert "install.sh" not in text
    assert "claude.ai" not in text


def test_entrypoint_installs_claude_into_the_home_volume():
    text = open(os.path.join(AC, "entrypoint.sh")).read()
    assert "claude.ai/install.sh" in text
    assert "AGENT_HOME" in text


def test_dockerfile_venvs_exclude_gpu_packages():
    lines = open(os.path.join(AC, "requirements-agent.txt")).read().splitlines()
    pkg_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    assert not any("cupy" in l.lower() for l in pkg_lines)


def test_settings_denies_editing_sim_and_webapp_and_gates():
    allow, deny = _settings()["permissions"]["allow"], _settings()["permissions"]["deny"]
    for needle in ("Edit(sim/**)", "Edit(webapp/**)", "Edit(tools/gates/**)"):
        assert needle in deny


def test_settings_denies_pushing_to_the_github_remote():
    deny = _settings()["permissions"]["deny"]
    assert any("git push origin" in r for r in deny)


def test_settings_wires_the_shared_protected_edit_hook():
    s = _settings()
    cmds = [h["command"] for e in s.get("hooks", {}).get("PreToolUse", []) for h in e.get("hooks", [])]
    assert any("deny_protected_edits.sh" in c for c in cmds)


def test_settings_hook_points_at_the_baked_readonly_path_not_the_writable_repo():
    # 2026-09-29 security review, LOW: the OLD config pointed at "$CLAUDE_PROJECT_DIR"/tools/local_llm/hooks/
    # deny_protected_edits.sh -- a path inside the model's OWN writable repo checkout, which it could simply
    # edit or delete to neuter its own guardrail.
    s = _settings()
    cmds = [h["command"] for e in s.get("hooks", {}).get("PreToolUse", []) for h in e.get("hooks", [])]
    hook_cmds = [c for c in cmds if "deny_protected_edits.sh" in c]
    assert hook_cmds
    for c in hook_cmds:
        assert "CLAUDE_PROJECT_DIR" not in c
        assert c.startswith("/usr/local/lib/agent-container/")


def test_dockerfile_bakes_the_settings_as_managed_settings_on_linux():
    # 2026-09-29 security review, LOW: baked as Claude Code's MANAGED settings (highest-precedence, not a
    # project/user setting a writable repo checkout could ever shadow), read-only, outside both volumes.
    text = _dockerfile_text()
    assert "/etc/claude-code/managed-settings.json" in text
    assert re.search(r"chmod\s+444\s+/etc/claude-code/managed-settings\.json", text) or \
        re.search(r"chmod\s+444\s+[^\n]*managed-settings\.json", text)


def test_dockerfile_bakes_the_hook_scripts_read_only():
    text = _dockerfile_text()
    assert "COPY hooks/" in text
    assert "/usr/local/lib/agent-container/hooks/" in text


def test_vendored_hook_files_are_byte_identical_to_the_local_llm_source():
    # The Docker build context is tools/agent_container/ alone, so it cannot COPY tools/local_llm/hooks/*
    # directly -- these are vendored copies, kept in sync manually. This pins that they have NOT drifted.
    src_dir = os.path.join(ROOT, "tools", "local_llm", "hooks")
    vendored_dir = os.path.join(AC, "hooks")
    for name in ("deny_protected_edits.sh", "deny_protected_edits.py"):
        src = open(os.path.join(src_dir, name), "rb").read()
        vendored = open(os.path.join(vendored_dir, name), "rb").read()
        assert src == vendored, f"{name} has drifted from tools/local_llm/hooks/ -- re-vendor the copy"


def test_compose_service_builds_with_an_unused_uid_not_1000():
    # 2026-09-29 security review, LOW: UID 1000 maps to the WORKSTATION account's own uid (dant123) -- any
    # filesystem escape would land with the same uid as the real user. An unused id has no such collision.
    svc = _compose()["services"]["sim-agent"]
    build_args = svc.get("build", {}).get("args", {})
    assert str(build_args.get("AGENT_UID")) == "61000"
    assert str(build_args.get("AGENT_GID")) == "61000"
    assert str(build_args.get("AGENT_UID")) != "1000"


def test_compose_network_has_a_fixed_subnet_outside_the_docker_address_pool():
    # 2026-09-29 security review, M1: the firewall targets the subnet directly -- it must never be a
    # Docker-assigned one that can change on a recreate. 2026-09-30: and it must sit OUTSIDE daemon.json's
    # default-address-pools (172.20.0.0/16 in /24s) so Docker can never hand it to another stack.
    import ipaddress
    compose = _compose()
    cfg = compose["networks"]["sim-agent-net"].get("ipam", {}).get("config", [])
    assert cfg, "sim-agent-net must declare a fixed ipam subnet"
    net = ipaddress.ip_network(cfg[0]["subnet"])
    assert str(net) == "172.30.30.0/24"
    assert not net.overlaps(ipaddress.ip_network("172.20.0.0/16")), "inside the Docker address pool"
    assert not net.overlaps(ipaddress.ip_network("172.17.0.0/16")), "overlaps docker0"
    assert not net.overlaps(ipaddress.ip_network("192.168.0.0/16")), "overlaps the LAN / WireGuard"
    assert cfg[0].get("gateway") == "172.30.30.1"


def test_compose_network_has_a_fixed_bridge_interface_name():
    # 2026-09-30: without com.docker.network.bridge.name the bridge is br-<hash>, so an interface-matched rule
    # (egress.sh's old `-i br-sim-agent` INPUT DROP) silently matched nothing. Linux caps names at 15 chars.
    opts = _compose()["networks"]["sim-agent-net"].get("driver_opts", {})
    name = opts.get("com.docker.network.bridge.name")
    assert name == "br-simagent"
    assert len(name) <= 15


def test_compose_network_has_ipv6_disabled():
    # 2026-09-29 re-review, LOW: no v6 traffic at all for a missing rule to leave ungoverned (sim-agent.nft
    # additionally drops any v6 from this bridge).
    compose = _compose()
    assert compose["networks"]["sim-agent-net"].get("enable_ipv6") is False


def test_compose_service_bounds_its_log_size():
    # 2026-09-29 security review, M3: an agent-authored job that logs heavily must not fill the shared disk.
    # 2026-09-30: json-file with the site's own 10m x 3 cap (new-service template) -- still bounded (30 MB).
    svc = _compose()["services"]["sim-agent"]
    logging = svc.get("logging", {})
    assert logging.get("driver") in ("json-file", "local")
    opts = logging.get("options", {})
    assert "max-size" in opts and "max-file" in opts


def test_fragment_follows_the_apps_conventions():
    # ~/apps/new-service/compose.yaml.template: container_name, restart, no-new-privileges, TZ from the one .env,
    # bounded json-file logs; and a fragment is INCLUDED into the one `apps` project, so it never names a project.
    compose = _compose()
    assert "name" not in compose, "an included fragment must not set a project name"
    svc = compose["services"]["sim-agent"]
    assert svc.get("container_name") == "sim-agent"
    assert svc.get("restart") == "unless-stopped"
    assert svc.get("environment", {}).get("TZ") == "${TZ}"
    labels = svc.get("labels") or {}
    assert set(labels) <= {"homepage.group", "homepage.name", "homepage.description", "homepage.icon"}
    assert labels.get("homepage.group") == "Infrastructure"


def test_fragment_takes_secrets_only_as_sim_agent_prefixed_interpolations_never_env_file():
    # ONE ~/apps/.env holds every stack's secrets: env_file: on it would hand this container all of them.
    text = "\n".join(l.split("#", 1)[0] for l in _compose_text().splitlines())   # code only, not comments
    svc = _compose()["services"]["sim-agent"]
    assert "env_file" not in svc and "env_file" not in text
    for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", text):
        assert var in ("DATA_ROOT", "TZ") or var.startswith("SIM_AGENT_"), var
    # a `:?` in an INCLUDED fragment aborts `docker compose` for every stack on the box when the var is unset
    assert ":?" not in text


def test_fragment_builds_locally_from_the_owners_checkout_like_caddy():
    # caddy's local build: build + image + pull_policy: missing. The context is the owner's clone on the dataset
    # (never the agent's writable repo/), and it is THIS repo's tools/agent_container -- one Dockerfile.
    svc = _compose()["services"]["sim-agent"]
    build = svc["build"]
    assert build["context"] == "${DATA_ROOT}/sim-agent/src/tools/agent_container"
    assert build.get("dockerfile", "Dockerfile") == "Dockerfile"
    assert svc.get("image") == "sim-agent:local"
    assert svc.get("pull_policy") == "missing"
    assert os.path.isfile(os.path.join(AC, "Dockerfile"))
    others = [os.path.join(d, f) for d, _, fs in os.walk(STACK) for f in fs if f.startswith("Dockerfile")]
    assert not others, "a second Dockerfile would be a second source of truth: %s" % others


def test_fragment_dns_matches_the_firewall_allow_list():
    # Docker's embedded DNS forwards to these FROM the container's netns; sim-agent.nft allows exactly these.
    dns = _compose()["services"]["sim-agent"].get("dns")
    nft = open(NFT).read()
    m = re.search(r"^define DNS_RESOLVERS = \{([^}]*)\}", nft, re.MULTILINE)
    assert m
    assert dns == [a.strip() for a in m.group(1).split(",")]


def test_fragment_requires_the_firewall_and_probes_its_own_gateway():
    env = _compose()["services"]["sim-agent"]["environment"]
    assert env.get("SIM_AGENT_REQUIRE_FIREWALL") == "1"
    gw = _compose()["networks"]["sim-agent-net"]["ipam"]["config"][0]["gateway"]
    assert env.get("SIM_AGENT_FW_PROBE") == gw + ":22"


def test_stack_dir_gitignore_keeps_everything_but_the_fragment_out_of_the_apps_repo():
    # apps-gitsync does `git add -A` nightly; the root .gitignore only covers .env files.
    lines = [l.strip() for l in open(os.path.join(STACK, "sim-agent", ".gitignore")) if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"
    assert set(lines[1:]) == {"!.gitignore", "!compose.yaml"}


def test_superseded_standalone_files_are_gone():
    # one source of truth: the fragment replaced the standalone compose project; sim-agent.nft replaced egress.sh
    for name in ("compose.yaml", "egress.sh", "sim-agent-egress.service"):
        assert not os.path.exists(os.path.join(AC, name)), name


def test_dockerfile_makes_every_bin_file_executable():
    # the build source is a git clone where these are tracked 0644 (core.fileMode=false); `llm` has no .sh suffix
    assert re.search(r"chmod \+x /usr/local/lib/agent-container/bin/\*(\s|$)", _dockerfile_text())


def test_entrypoint_runs_the_firewall_guard_first():
    text = open(os.path.join(AC, "entrypoint.sh")).read()
    guard = text.index("firewall_guard.sh")
    assert guard < text.index("claude.ai/install.sh")
    assert guard < text.index("new-session -d -s agent")


def test_settings_only_allows_the_gateway_subcommands_over_ssh():
    # 2026-09-29 security review, H3: results-sync moved OFF the ssh gateway entirely (a separate, read-only
    # rrsync key) -- it must never appear as a "workstation-gateway results-sync" allow rule any more.
    allow = _settings()["permissions"]["allow"]
    ssh_rules = [r for r in allow if r.startswith("Bash(ssh ")]
    for r in ssh_rules:
        assert "workstation-gateway" in r
    subs = {"pool-submit", "gpu-submit", "status", "yield"}
    covered = {s for s in subs if any(f"workstation-gateway {s}" in r for r in ssh_rules)}
    assert covered == subs
    assert not any("results-sync" in r for r in ssh_rules)


def test_settings_allows_the_dedicated_rrsync_results_pull():
    allow = _settings()["permissions"]["allow"]
    assert any("results-sync-gateway" in r and r.startswith("Bash(rsync") for r in allow)


def test_smoke_test_probes_the_isolation_from_inside_on_the_server():
    # update.sh smoke-tests every new image before the swap, so a firewall regression blocks the swap too
    text = open(os.path.join(AC, "bin", "smoke_test.sh")).read()
    assert 'SIM_AGENT_REQUIRE_FIREWALL:-0}" != "1"' in text
    must_fail = re.search(r'MUST_FAIL="\$\{SMOKE_MUST_FAIL:-([^}]*)\}"', text).group(1).split()
    must_pass = re.search(r'MUST_PASS="\$\{SMOKE_MUST_PASS:-([^}]*)\}"', text).group(1).split()
    for t in ("172.30.30.1:22", "192.168.0.10:22", "192.168.0.10:445", "192.168.0.10:2049", "192.168.0.10:18443"):
        assert t in must_fail, t
    assert "192.168.0.10:2222" in must_pass
    # the workstation may be asleep or gaming during the weekly rebuild: a miss there must never roll an image back
    assert not any(t.startswith("192.168.0.68:") for t in must_pass)


def test_every_shipped_shell_script_parses():
    import subprocess
    scripts = [os.path.join(AC, n) for n in ("entrypoint.sh", "update.sh")]
    scripts += [os.path.join(AC, "bin", n) for n in os.listdir(os.path.join(AC, "bin"))]
    scripts += [os.path.join(STACK, "verify_isolation.sh"),
                os.path.join(STACK, "host", "usr", "local", "sbin", "sim-agent-fw-selfwan.sh"),
                os.path.join(ROOT, "tools", "agent_harness_ctl.sh")]
    for path in scripts:
        r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert r.returncode == 0, "%s: %s" % (path, r.stderr)


# ------------------------------------------------ PATH + interpreters (2026-09-30 security review, HIGH #1)
# The image used to put the agent-writable home bind mount FIRST on PATH, so a model that had run once could plant
# ~/.local/bin/{bash,timeout,...}: the fail-closed guard, the smoke test and every host-side `docker exec` check
# would then run the model's code and report whatever it chose. PATH is now the image's system dirs only; the
# home's ~/.local/bin is appended LAST, in the interactive tmux shell only (where `claude` lives).
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
HOST_SIDE_EXEC_SCRIPTS = [os.path.join(AC, "update.sh"), os.path.join(ROOT, "tools", "agent_harness_ctl.sh"),
                          os.path.join(STACK, "verify_isolation.sh")]


def _code(path):
    return [l for l in open(path).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]


def _bin_files():
    d = os.path.join(AC, "bin")
    return [os.path.join(d, n) for n in sorted(os.listdir(d))]


def test_image_path_never_includes_the_agent_writable_home():
    code = "\n".join(_code(os.path.join(AC, "Dockerfile")))
    values = re.findall(r'\bPATH="?([^"\s\\]+)"?', code)
    assert values, "the Dockerfile must set PATH explicitly"
    for v in values:
        assert "/home" not in v and "AGENT_HOME" not in v and "~" not in v, "image PATH reaches the home: %s" % v
    assert SAFE_PATH in values


def test_container_scripts_use_absolute_interpreters():
    for path in [os.path.join(AC, "entrypoint.sh")] + _bin_files():
        first = open(path).readline().strip()
        assert first.startswith("#!/") and not first.startswith("#!/usr/bin/env"), (path, first)


def test_pre_session_scripts_reset_path_before_any_lookup():
    # entrypoint.sh and firewall_guard.sh run at every start, before (and without) any model session
    for rel in ("entrypoint.sh", "bin/firewall_guard.sh"):
        code = [l.strip() for l in _code(os.path.join(AC, rel))]
        first = next(l for l in code if not l.startswith("set -"))
        assert first == "export PATH=%s" % SAFE_PATH, (rel, first)


def test_guard_and_entrypoint_name_timeout_and_bash_by_absolute_path():
    guard = os.path.join(AC, "bin", "firewall_guard.sh")
    assert '/usr/bin/timeout "$wait_s" /bin/bash -c' in open(guard).read()
    for l in _code(guard):
        stripped = l.replace("/usr/bin/timeout", "").replace("/bin/bash", "")
        assert not re.search(r"(^|[\s;&|(!])(timeout|bash|sh|env)\s", stripped), "relative interpreter: %s" % l
    entry = open(os.path.join(AC, "entrypoint.sh")).read()
    assert "/bin/bash /usr/local/lib/agent-container/bin/firewall_guard.sh" in entry


def test_entrypoint_never_executes_the_home_claude_binary_at_start():
    # the binary lives in the agent-writable home; running it at every start would run planted code before any
    # session. Its version is read from the installer's symlink instead.
    code = "\n".join(_code(os.path.join(AC, "entrypoint.sh")))
    assert not re.search(r'"\$CLAUDE_BIN"\s+--version', code)
    assert "readlink" in code


def test_entrypoint_tmux_shell_skips_home_startup_files_and_puts_home_bin_last():
    text = open(os.path.join(AC, "entrypoint.sh")).read()
    assert "/usr/bin/tmux -f /dev/null new-session" in text
    assert text.count("/bin/bash --noprofile --norc") >= 2      # the first window + default-command for later ones
    assert 'PATH="$SAFE_PATH:$AGENT_HOME/.local/bin"' in text


def test_smoke_test_calls_claude_by_its_install_path_not_path_lookup():
    text = open(os.path.join(AC, "bin", "smoke_test.sh")).read()
    assert '"$AGENT_HOME/.local/bin/claude" --version' in text
    assert "command -v claude" not in text


def test_dockerfile_installs_procps_so_the_absolute_pgrep_exists():
    assert re.search(r"^\s+.*\bprocps\b", _dockerfile_text(), re.MULTILINE)


def test_every_host_side_docker_exec_pins_path_and_calls_an_absolute_binary():
    for path in HOST_SIDE_EXEC_SCRIPTS:
        text = open(path).read()
        assert 'EXEC_ENV_PATH="PATH=%s"' % SAFE_PATH in text, path
        n = 0
        for l in _code(path):
            for m in re.finditer(r'(?:\bdocker|"\$DOCKER")\s+exec\b(.*)', l):
                n += 1
                assert re.match(r'\s+(-i\s+)?-e\s+"\$EXEC_ENV_PATH"\s+"\$CONTAINER"\s+/', m.group(1)), \
                    "%s: %s" % (os.path.basename(path), l.strip())
        assert n or path.endswith("verify_isolation.sh"), path
