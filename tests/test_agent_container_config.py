"""Static checks on tools/agent_container/{compose.yaml,Dockerfile,claude_container_settings.json}: the
isolation properties docs/plans/2026-09-29-local-model-transition-PLAN.md section 9 requires (own network, no
docker socket, dropped capabilities, read-only root, resource limits, no published ports, non-root) are
properties of CONFIGURATION, not runtime behaviour a unit test can otherwise observe -- so this just parses the
files and asserts the settings are actually there. No docker is invoked.
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


def _compose_text():
    return open(os.path.join(AC, "compose.yaml")).read()


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
    assert list(nets) == ["agent-net"]
    compose = _compose()
    assert compose["networks"]["agent-net"].get("driver") == "bridge"


def test_compose_service_has_no_docker_socket_mount():
    svc = _compose()["services"]["sim-agent"]
    vols = svc.get("volumes", [])
    for v in vols:
        v_str = v if isinstance(v, str) else str(v)
        assert "docker.sock" not in v_str


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


def test_compose_service_has_exactly_two_named_volumes_for_persistence():
    svc = _compose()["services"]["sim-agent"]
    vols = svc.get("volumes", [])
    named = [v for v in vols if isinstance(v, str) and ":" in v and not v.startswith("/") and not v.startswith(".")]
    assert len(named) == 2
    targets = {v.split(":")[1] for v in named}
    assert targets == {"/work/sim", "/home/agent"}


def test_compose_service_has_resource_limits():
    svc = _compose()["services"]["sim-agent"]
    # plain (non-swarm) resource fields -- see compose.yaml's own comment on why not `deploy.resources.limits`
    assert str(svc.get("cpus")) == "12"
    assert str(svc.get("mem_limit")) == "24g"
    assert svc.get("pids_limit") == 4096


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


def test_compose_network_has_a_fixed_subnet():
    # 2026-09-29 security review, M1: tools/agent_container/egress.sh's iptables rules target a subnet
    # directly -- it must never be a Docker-assigned one that can change on a network recreate.
    compose = _compose()
    ipam = compose["networks"]["agent-net"].get("ipam", {})
    subnets = [c.get("subnet") for c in ipam.get("config", [])]
    assert subnets, "agent-net must declare a fixed ipam subnet"
    assert subnets[0] == "172.30.30.0/24"


def test_compose_network_has_ipv6_disabled():
    # 2026-09-29 re-review, LOW: egress.sh only governs IPv4 -- rather than build and keep a parallel
    # ip6tables ruleset in sync, IPv6 is disabled outright on this bridge so there is no v6 traffic for a
    # missing rule to ever leave ungoverned.
    compose = _compose()
    assert compose["networks"]["agent-net"].get("enable_ipv6") is False


def test_compose_service_bounds_its_log_size():
    # 2026-09-29 security review, M3: the default json-file driver has no size cap -- an agent-authored job
    # that logs heavily could fill this shared server's disk.
    svc = _compose()["services"]["sim-agent"]
    logging = svc.get("logging", {})
    assert logging.get("driver") == "local"
    assert "max-size" in logging.get("options", {})


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
