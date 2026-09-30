"""The sim-agent host firewall (tools/agent_container/server-stack/host/etc/nftables.d/sim-agent.nft), its systemd
units, the self_wan refresher, and the in-container fail-closed guard. Replaces tests/test_agent_container_egress.py
(2026-09-30): egress.sh's iptables DOCKER-USER rules were only ever checked for insert ORDER against a stubbed
iptables, and on srv-host-0 its INPUT DROP matched no interface at all (compose set no bridge name).

Three layers, cheapest first:
  1. structural -- the file agrees with the stack fragment (bridge name, subnet, DNS), keeps fw.nft's
     recreate-only-its-own-table pattern, runs before fw.nft/Docker, and carries no public address literal;
  2. `nft -c` -- the real nftables parser, unprivileged inside a user+network namespace;
  3. functional -- tests/_agent_fw_netns.py builds a miniature of the server in network namespaces and probes
     from a stand-in container: every must-block destination is reachable WITHOUT this file (control arm, so the
     test is not vacuous) and dropped WITH it; every allowed path stays open.
Layers 2-3 skip where unprivileged user namespaces or nft/ip/nsenter are unavailable; nothing on the machine
running the test is touched.
"""
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

import pytest

try:
    import yaml
except ImportError:
    yaml = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = os.path.join(ROOT, "tools", "agent_container", "server-stack")
NFT = os.path.join(STACK, "host", "etc", "nftables.d", "sim-agent.nft")
UNITS = os.path.join(STACK, "host", "etc", "systemd", "system")
SELFWAN = os.path.join(STACK, "host", "usr", "local", "sbin", "sim-agent-fw-selfwan.sh")
GUARD = os.path.join(ROOT, "tools", "agent_container", "bin", "firewall_guard.sh")
FRAGMENT = os.path.join(STACK, "sim-agent", "compose.yaml")
NETNS_SIM = os.path.join(ROOT, "tests", "_agent_fw_netns.py")


def _nft_text():
    return open(NFT).read()


def _code_lines():
    return [l.split("#", 1)[0] for l in _nft_text().splitlines()]


def _define(name):
    m = re.search(r"^define %s = (.+?)\s*(#.*)?$" % name, _nft_text(), re.MULTILINE)
    assert m, "no `define %s` in sim-agent.nft" % name
    return m.group(1).strip()


def _fragment():
    if yaml is None:
        pytest.skip("pyyaml not installed in this environment")
    return yaml.safe_load(open(FRAGMENT))


def _userns_ok():
    if not all(shutil.which(t) for t in ("unshare", "nft", "ip", "nsenter")):
        return False
    try:
        r = subprocess.run(["unshare", "--user", "--map-root-user", "--net", "nft", "list", "ruleset"],
                           capture_output=True, timeout=10)
    except Exception:
        return False
    return r.returncode == 0


# ------------------------------------------------------------------------------------------ 1. structural

def test_defines_match_the_stack_fragment():
    frag = _fragment()
    net = frag["networks"]["sim-agent-net"]
    assert _define("SIM_IF") == '"%s"' % net["driver_opts"]["com.docker.network.bridge.name"]
    assert _define("SIM_NET") == net["ipam"]["config"][0]["subnet"]
    dns = [a.strip() for a in _define("DNS_RESOLVERS").strip("{} ").split(",")]
    assert dns == frag["services"]["sim-agent"]["dns"]


def test_recreates_only_its_own_table_like_fw_nft():
    code = "\n".join(_code_lines())
    assert "add table inet sim_agent" in code
    assert "delete table inet sim_agent" in code
    assert code.index("add table inet sim_agent") < code.index("delete table inet sim_agent") < \
        code.index("table inet sim_agent {")
    assert "flush ruleset" not in code
    tables = set(re.findall(r"^\s*(?:add |delete )?table (\w+ \w+)", code, re.MULTILINE))
    assert tables == {"inet sim_agent"}, "touches another table: %s" % tables


def test_base_chains_run_before_fw_nft_and_docker():
    # nftables runs every base chain on a hook in priority order; drop is final, accept is not. At filter - 10
    # our drops land before fw.nft's `iifname "br-*" accept` (priority 0) and Docker's FORWARD (priority 0).
    code = "\n".join(_code_lines())
    hooks = re.findall(r"type filter hook (\w+) priority filter - (\d+); policy (\w+);", code)
    assert sorted(h[0] for h in hooks) == ["forward", "input"]
    for hook, prio, policy in hooks:
        assert int(prio) > 0, hook
        assert policy == "accept", "a drop POLICY would also judge unrelated traffic on this shared box"


def test_input_drops_everything_from_the_bridge_by_name_and_by_subnet():
    chain = "\n".join(l for l in _chain("input").splitlines() if not l.strip().startswith("type "))
    assert re.search(r'iifname \$SIM_IF counter drop', chain)
    assert re.search(r'ip saddr \$SIM_NET counter drop', chain)
    assert "accept" not in chain, "the container needs no host listener at all"


def test_from_agent_allow_list_then_private_drop_then_443_then_final_drop():
    chain = _chain("from_agent")
    order = ["meta nfproto ipv6", "ct state established,related accept", "$DNS_RESOLVERS",
             "$WORKSTATION tcp dport $WORKSTATION_TCP accept", "ct original ip daddr $GITEA_ADDR",
             "@self_wan counter drop", "@not_internet4 counter drop", "tcp dport 443 accept",
             'counter drop comment "everything else"']
    idx = [chain.index(o) for o in order]
    assert idx == sorted(idx), "from_agent rule order changed: %s" % list(zip(order, idx))
    # the gitea rule must match what the client DIALED: Docker DNATs :2222 before the forward hook
    assert re.search(r"meta l4proto tcp ct original ip daddr \$GITEA_ADDR ct original proto-dst \$GITEA_SSH_PORT accept",
                     chain)


def test_not_internet_set_covers_every_private_and_special_range():
    m = re.search(r"set not_internet4 \{.*?elements = \{([^}]*)\}", _nft_text(), re.DOTALL)
    nets = [ipaddress.ip_network(e.strip()) for e in m.group(1).split(",")]
    for must in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "127.0.0.0/8",
                 "169.254.0.0/16", "224.0.0.0/4", "240.0.0.0/4"):
        want = ipaddress.ip_network(must)
        assert any(want.subnet_of(n) for n in nets), must
    for addr in ("172.20.0.5", "172.17.0.1", "192.168.0.10", "192.168.3.1", "172.30.30.1"):
        assert any(ipaddress.ip_address(addr) in n for n in nets), addr


def test_to_agent_admits_replies_only():
    chain = _chain("to_agent")
    assert chain.split("\n")[0].strip() == "ct state established,related accept"
    assert chain.strip().endswith("counter drop")


def test_no_public_address_literal_is_committed():
    # this repo is PUBLIC: the site's WAN address is resolved at runtime into the self_wan set, never written
    # here. The only public literal allowed is the documented fallback resolver.
    for addr in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "\n".join(_code_lines())):
        ip = ipaddress.ip_address(addr)
        assert ip.is_private or ip.is_unspecified or ip.is_multicast or ip.is_reserved or \
            ip.is_loopback or ip.is_link_local or str(ip) in ("1.1.1.1",) or \
            ipaddress.ip_network("100.64.0.0/10").supernet_of(ipaddress.ip_network(addr + "/32")), addr


def _chain(name):
    m = re.search(r"chain %s \{\n(.*?)\n\t\}" % name, _nft_text(), re.DOTALL)
    assert m, "no chain %s" % name
    return "\n".join(l.split("#", 1)[0].rstrip() for l in m.group(1).splitlines() if l.split("#", 1)[0].strip())


# ------------------------------------------------------------------------------------- systemd units

def _unit(name):
    return open(os.path.join(UNITS, name)).read()


def test_firewall_unit_loads_before_docker_and_never_holds_docker_hostage():
    u = _unit("sim-agent-fw.service")
    assert re.search(r"^Before=docker\.service$", u, re.MULTILINE)
    assert "ExecStart=/usr/sbin/nft -f /etc/nftables.d/sim-agent.nft" in u
    assert "ExecStop=/usr/sbin/nft delete table inet sim_agent" in u
    assert re.search(r"^WantedBy=.*\bdocker\.service\b", u, re.MULTILINE)
    assert not re.search(r"^(Requires|RequiredBy|BindsTo|PartOf)=", u, re.MULTILINE), \
        "a broken firewall edit must never keep Docker (every other stack) down"
    # a reload recreates the table and so empties self_wan: it must be refilled in the same step
    reloads = re.findall(r"^ExecReload=(.*)$", u, re.MULTILINE)
    assert reloads[0] == "/usr/sbin/nft -f /etc/nftables.d/sim-agent.nft"
    assert reloads[1].endswith("/usr/local/sbin/sim-agent-fw-selfwan.sh")


def test_selfwan_timer_refreshes_periodically():
    t = _unit("sim-agent-fw-selfwan.timer")
    assert "OnBootSec=" in t and "OnUnitActiveSec=" in t
    s = _unit("sim-agent-fw-selfwan.service")
    assert "ExecStart=/usr/local/sbin/sim-agent-fw-selfwan.sh" in s
    assert "Requisite=sim-agent-fw.service" in s


def test_selfwan_refresh_failure_alerts():
    # 2026-09-30 security review (MEDIUM): a failed resolution leaves the set unchanged and exits 1, which used to fail
    # silently; after a DDNS change that left the hairpin open with nobody told
    assert re.search(r"^OnFailure=ntfy-notify@%n\.service$", _unit("sim-agent-fw-selfwan.service"), re.MULTILINE)


def test_firewall_unit_no_longer_claims_nothing_talks_to_the_site_before_a_session():
    # a model can leave code in its persistent home; the guard's WAN probe is what holds the container back now
    u = _unit("sim-agent-fw.service")
    assert "before the owner starts a session" not in u
    assert "firewall_guard.sh" in u


# ------------------------------------------------------------------------------ self_wan refresher script

def _exec(path, body):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def _run_selfwan(tmp, getent_out, *args):
    getent = _exec(os.path.join(tmp, "getent"), "cat <<'EOF'\n%sEOF\n" % getent_out)
    nft_in = os.path.join(tmp, "nft_stdin")
    nft = _exec(os.path.join(tmp, "nft"), 'cat > "%s"\n' % nft_in)
    env = dict(os.environ, SIM_AGENT_SELFWAN_GETENT=getent, SIM_AGENT_SELFWAN_NFT=nft)
    r = subprocess.run(["bash", SELFWAN, *args], env=env, capture_output=True, text=True, timeout=10)
    return r, (open(nft_in).read() if os.path.exists(nft_in) else None)


def test_selfwan_loads_only_public_addresses_atomically_and_deduplicated():
    with tempfile.TemporaryDirectory() as tmp:
        out = ("203.0.113.7     STREAM dant123.com\n203.0.113.7     DGRAM\n192.168.0.10    RAW\n"
               "10.1.2.3        STREAM\n198.51.100.9    STREAM\n")
        r, fed = _run_selfwan(tmp, out)
        assert r.returncode == 0, r.stderr
        assert fed == ("flush set inet sim_agent self_wan\n"
                       "add element inet sim_agent self_wan { 203.0.113.7,198.51.100.9 }\n")
        assert "203.0.113.7" not in r.stdout, "the address is never echoed into logs"


def test_selfwan_leaves_the_set_untouched_when_nothing_public_resolves():
    with tempfile.TemporaryDirectory() as tmp:
        r, fed = _run_selfwan(tmp, "192.168.0.10 STREAM dant123.com\n")
        assert r.returncode != 0
        assert fed is None, "a DNS hiccup must never flush the set"


def test_selfwan_dry_run_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        r, fed = _run_selfwan(tmp, "203.0.113.7 STREAM x\n", "--dry-run")
        assert r.returncode == 0 and fed is None
        assert "DRY-RUN: flush set inet sim_agent self_wan" in r.stdout


# ------------------------------------------------------------------------------------ fail-closed guard

def _guard_copy(tmp, timeout_rc, getent_out=None):
    """The guard names /usr/bin/timeout by absolute path and resets PATH (so nothing on PATH can stand in for the
    probe, which is the point). The stub therefore goes into a COPY of the script, at that absolute path."""
    stub = _exec(os.path.join(tmp, "timeout_stub"), "exit %d\n" % timeout_rc)   # stands in for the probe's outcome
    text = open(GUARD).read()
    assert text.count("/usr/bin/timeout ") >= 1
    text = text.replace("/usr/bin/timeout ", stub + " ")
    if getent_out is not None:
        assert "/usr/bin/getent ahostsv4 " in text
        getent = _exec(os.path.join(tmp, "getent_stub"), "cat <<'EOF'\n%sEOF\n" % getent_out)
        text = text.replace("/usr/bin/getent ahostsv4 ", getent + " ahostsv4 ")
    copy = os.path.join(tmp, "firewall_guard.sh")
    with open(copy, "w") as f:
        f.write(text)
    return copy


def _run_guard(tmp, timeout_rc, extra_env=None, getent_out="203.0.113.7 STREAM example.org\n"):
    env = dict(os.environ, SIM_AGENT_REQUIRE_FIREWALL="1", SIM_AGENT_FW_PROBE="172.30.30.1:22",
               SIM_AGENT_FW_WAN_NAME="example.org")
    env.update(extra_env or {})
    return subprocess.run(["bash", _guard_copy(tmp, timeout_rc, getent_out)], env=env, capture_output=True,
                          text=True, timeout=10)


def test_guard_also_requires_the_sites_own_wan_address_to_be_dropped():
    # 2026-09-30 security review (MEDIUM): self_wan is filled best-effort at boot, by a timer a minute later and every
    # 5 minutes, so the hairpin into this site's own services is open at every boot and after a DDNS change. The
    # container auto-starts with a persistent, model-writable home, so the guard now also refuses to start until a
    # probe of the WAN address on 443 is DROPPED (a timeout), exactly like the gateway's sshd.
    with tempfile.TemporaryDirectory() as tmp:
        r = _run_guard(tmp, 124)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "203.0.113.7" not in r.stdout + r.stderr, "the WAN address is never written to a log"
        # the gateway probe is dropped but the WAN address connects: refuse
        stub = _exec(os.path.join(tmp, "timeout_seq"), 'case "$*" in *" 22"|*"22") exit 124 ;; *) exit 0 ;; esac\n')
        text = open(GUARD).read().replace("/usr/bin/timeout ", stub + " ").replace(
            "/usr/bin/getent ahostsv4 ", _exec(os.path.join(tmp, "g"), "echo '203.0.113.7 STREAM x'\n") + " ahostsv4 ")
        copy = os.path.join(tmp, "guard_seq.sh")
        open(copy, "w").write(text)
        env = dict(os.environ, SIM_AGENT_REQUIRE_FIREWALL="1", SIM_AGENT_FW_PROBE="172.30.30.1:22",
                   SIM_AGENT_FW_WAN_NAME="example.org")
        r = subprocess.run(["bash", copy], env=env, capture_output=True, text=True, timeout=10)
        assert r.returncode == 1 and "WAN" in r.stderr and "REFUSING TO START" in r.stderr, r.stdout + r.stderr


def test_guard_fails_closed_when_the_wan_address_cannot_be_resolved_or_is_not_configured():
    with tempfile.TemporaryDirectory() as tmp:
        r = _run_guard(tmp, 124, getent_out="")
        assert r.returncode == 1 and "REFUSING TO START" in r.stderr
        r = _run_guard(tmp, 124, getent_out="192.168.0.10 STREAM example.org\n")   # private only: not the WAN
        assert r.returncode == 1 and "REFUSING TO START" in r.stderr
        r = _run_guard(tmp, 124, extra_env={"SIM_AGENT_FW_WAN_NAME": ""})
        assert r.returncode == 1 and "SIM_AGENT_FW_WAN_NAME" in r.stderr


def test_the_fragment_names_the_same_wan_name_the_selfwan_refresher_resolves():
    env = _fragment()["services"]["sim-agent"]["environment"]
    default = re.search(r'NAMES="\$\{SIM_AGENT_SELFWAN_NAMES:-([^}]*)\}"', open(SELFWAN).read()).group(1).split()
    assert env.get("SIM_AGENT_FW_WAN_NAME") in default


def test_guard_ignores_a_timeout_planted_on_path():
    # 2026-09-30 security review (HIGH #1): the image PATH used to start with the model-writable home, so a planted
    # `timeout` that always exits 124 made "sim-agent.nft is not loaded" read as "dropped". A real listener
    # answers here; a PATH-first fake timeout claims a drop; the guard must still refuse.
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            fakebin = os.path.join(tmp, "bin")
            os.makedirs(fakebin)
            _exec(os.path.join(fakebin, "timeout"), "exit 124\n")
            env = dict(os.environ, PATH=fakebin + os.pathsep + os.environ["PATH"], SIM_AGENT_REQUIRE_FIREWALL="1",
                       SIM_AGENT_FW_PROBE="127.0.0.1:%d" % s.getsockname()[1], SIM_AGENT_FW_WAN_NAME="localhost")
            r = subprocess.run(["bash", GUARD], env=env, capture_output=True, text=True, timeout=10)
            assert r.returncode == 1 and "ACCEPTED a connection" in r.stderr, r.stdout + r.stderr
    finally:
        s.close()


def test_guard_starts_only_when_the_probe_is_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        assert _run_guard(tmp, 124).returncode == 0            # timeout = dropped by sim-agent.nft
        r = _run_guard(tmp, 0)                                  # connected = firewall not loaded
        assert r.returncode == 1 and "REFUSING TO START" in r.stderr
        r = _run_guard(tmp, 1)                                  # refused/unreachable = answered unfiltered
        assert r.returncode == 1 and "REFUSING TO START" in r.stderr


def test_guard_refuses_a_malformed_probe_and_is_skipped_off_the_server():
    with tempfile.TemporaryDirectory() as tmp:
        r = _run_guard(tmp, 124, {"SIM_AGENT_FW_PROBE": "gateway;rm -rf /:22"})
        assert r.returncode == 1 and "not <ipv4>:<port>" in r.stderr
        r = _run_guard(tmp, 0, {"SIM_AGENT_REQUIRE_FIREWALL": "0"})
        assert r.returncode == 0 and "skipped" in r.stdout


def test_guard_really_connects_and_refuses_when_a_listener_answers():
    # no stub: a real listener on loopback stands in for a host sshd the firewall failed to hide
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        env = dict(os.environ, SIM_AGENT_REQUIRE_FIREWALL="1", SIM_AGENT_FW_PROBE="127.0.0.1:%d" % s.getsockname()[1],
                   SIM_AGENT_FW_WAN_NAME="localhost")
        r = subprocess.run(["bash", GUARD], env=env, capture_output=True, text=True, timeout=10)
        assert r.returncode == 1 and "ACCEPTED a connection" in r.stderr
    finally:
        s.close()


# ------------------------------------------------------------------------ 2. the real nftables parser

@pytest.mark.skipif(not _userns_ok(), reason="needs unprivileged user namespaces + nft/ip/nsenter")
def test_nft_check_accepts_the_file_and_it_loads_twice():
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--net", "nft", "-c", "-f", NFT],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--net", "sh", "-c",
                        'nft -f "$0" && nft -f "$0" && nft list tables', NFT],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "table inet sim_agent"


# ----------------------------------------------------------------- 3. functional (network namespaces)

MUST_PASS = ["dns_router_tcp", "dns_router_udp", "dns_fallback_tcp", "dns_fallback_udp", "workstation_ssh",
             "workstation_model", "gitea_ssh_via_host_2222", "internet_https"]
MUST_BLOCK = ["host_ssh_via_gateway", "host_smb_via_gateway", "host_ssh_lan_addr", "host_smb_lan_addr",
              "host_nfs_lan_addr", "host_rpcbind_lan_addr", "caddy_published_18443", "socket_proxy_direct",
              "gitea_http_direct", "gitea_ssh_direct_not_2222", "workstation_smb", "lan_device_http",
              "router_admin_https", "other_resolver_tcp", "other_resolver_udp", "internet_plain_http",
              "lan_to_agent"]


@pytest.fixture(scope="module")
def netns_result():
    if not _userns_ok():
        pytest.skip("needs unprivileged user namespaces + nft/ip/nsenter")
    r = subprocess.run(["unshare", "--user", "--map-root-user", "--net", sys.executable, NETNS_SIM, NFT],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def test_control_arm_reaches_everything_so_the_test_is_not_vacuous(netns_result):
    # with only fw.nft's `iifname "br-*" accept` + Docker's NAT, the container reaches the host's listeners,
    # the LAN, other stacks and the site's own WAN address -- exactly the gap this file closes
    control = netns_result["control"]
    for key in MUST_PASS + MUST_BLOCK + ["self_wan_https"]:
        assert control[key] == "open", "%s unreachable even without sim-agent.nft: %s" % (key, control[key])


def test_allowed_paths_stay_open_with_the_firewall_loaded(netns_result):
    for arm in ("loaded", "selfwan"):
        for key in MUST_PASS:
            assert netns_result[arm][key] == "open", (arm, key, netns_result[arm][key])


def test_every_other_destination_is_dropped_with_the_firewall_loaded(netns_result):
    for arm in ("loaded", "selfwan"):
        for key in MUST_BLOCK:
            if key in netns_result[arm]:
                assert netns_result[arm][key] == "timeout", (arm, key, netns_result[arm][key])


def test_self_wan_set_closes_the_hairpin(netns_result):
    assert netns_result["loaded"]["self_wan_https"] == "open"       # before the refresher has run
    assert netns_result["selfwan"]["self_wan_https"] == "timeout"   # after


# --------------------------------------------------------------------------- weekly rebuild unit + public repo

def test_update_unit_runs_update_sh_as_the_owner_from_the_dataset_checkout_like_apps_gitsync():
    u = _unit("sim-agent-update.service")
    assert re.search(r"^User=dant123$", u, re.MULTILINE)
    assert "WorkingDirectory=/home/dant123/apps" in u
    assert re.search(r"^ExecStart=/usr/bin/bash /home/dant123/data/sim-agent/src/tools/agent_container/update\.sh$",
                     u, re.MULTILINE)
    assert "OnFailure=ntfy-notify@%n.service" in u
    t = _unit("sim-agent-update.timer")
    assert re.search(r"^OnCalendar=Sun ", t, re.MULTILINE) and "Persistent=true" in t
    # the fragment's build context and the unit agree on where the checkout lives (DATA_ROOT=/home/dant123/data)
    assert "${DATA_ROOT}/sim-agent/src/tools/agent_container" in open(FRAGMENT).read()


def test_no_public_ipv4_literal_anywhere_in_the_server_stack_or_the_setup_doc():
    # this repository is PUBLIC (GitHub + Gitea). The site's WAN address must never be written into it; only these
    # well-known public resolvers are ever named (the fallback resolver, and a must-FAIL probe target).
    allowed = {"1.1.1.1", "8.8.8.8"}
    files = [os.path.join(d, f) for d, _, fs in os.walk(STACK) for f in fs]
    files += [os.path.join(ROOT, "docs", "AGENT_HARNESS_SETUP.md"),
              os.path.join(ROOT, "tools", "agent_container", "bin", "smoke_test.sh")]
    for path in files:
        for addr in re.findall(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])", open(path, errors="replace").read()):
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if ip.is_global and not ip.is_multicast and addr not in allowed:
                pytest.fail("public address %s in %s" % (addr, os.path.relpath(path, ROOT)))
