"""tools/agent_container/server-stack/verify_isolation.sh -- the owner's post-apply check of the sim-agent isolation
on srv-host-0. 2026-09-30 security review:

  HIGH #1  its network probes ran INSIDE the agent container (`docker exec ... bash -s`, mktemp in the shared /tmp,
           PATH starting with the model-writable home), so "ALL CHECKS PASSED" was the model's to choose. They now
           run from a pristine container (probe_from_pristine.sh), and the host-side nft drop counters are a HARD
           check (they used to be printed with `|| true`).
  HIGH #3  Gitea Actions for the repo must be OFF: `has_actions` true (or unreadable) FAILS the run.
  MEDIUM   the gitea dataset must carry a ZFS quota; sim-agent-fw-selfwan.service must alert on failure; the
           workstation sshd must offer no password method (read from the probe container).

Hermetic: every host command (systemctl, sudo, nft, ip, docker, curl, zfs) is a stub first on PATH, answering from
environment knobs; the script runs on the host, where PATH is the owner's own and trusted.
"""
import json
import os
import stat
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFY = os.path.join(ROOT, "tools", "agent_container", "server-stack", "verify_isolation.sh")
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_COMMENTS = ["sim-agent -> host: nothing", "same, by subnet (holds even if the bridge were renamed)",
             "no IPv6 at all", "hairpin to this site's own services", "LAN, other containers, this host",
             "everything else"]


def _nft_json(packets):
    rules = [{"rule": {"family": "inet", "table": "sim_agent", "chain": "input" if i < 2 else "from_agent",
                       "comment": c, "expr": [{"counter": {"packets": packets, "bytes": packets * 60}},
                                              {"drop": None}]}}
             for i, c in enumerate(_COMMENTS)]
    return json.dumps({"nftables": [{"metainfo": {}}] + rules})


INSPECT = {
    "Config": {"User": "agent"},
    "HostConfig": {"CapDrop": ["ALL"], "CapAdd": None, "Privileged": False, "ReadonlyRootfs": True,
                   "SecurityOpt": ["no-new-privileges:true"], "NetworkMode": "apps_sim-agent-net",
                   "Dns": ["192.168.0.1", "1.1.1.1"], "PidMode": "", "IpcMode": "private", "UsernsMode": ""},
    "Mounts": [{"Type": "bind", "Source": "/home/dant123/data/sim-agent/repo", "Destination": "/work/sim"},
               {"Type": "bind", "Source": "/home/dant123/data/sim-agent/home", "Destination": "/home/agent"}],
}


def _stubs(tmp):
    b = os.path.join(tmp, "bin")
    os.makedirs(b)

    def w(name, body):
        p = os.path.join(b, name)
        with open(p, "w") as f:
            f.write("#!/bin/bash\n" + body)
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)

    w("systemctl", r'''echo "systemctl $*" >> "$FAKE_T/calls"
case "$1 $2" in
  "is-active sim-agent-fw.service"|"is-active sim-agent-fw-selfwan.timer") echo active ;;
  "is-enabled sim-agent-fw.service") echo enabled ;;
  "is-enabled nftables.service") echo disabled; exit 1 ;;
  "show -p") echo "$FAKE_ONFAILURE" ;;
  *) exit 1 ;;
esac
''')
    w("sudo", 'exec "$@"\n')
    w("nft", r'''echo "nft $*" >> "$FAKE_T/calls"
case "$*" in
  "list table inet sim_agent") echo "table inet sim_agent {" ;;
  "-j list set inet sim_agent self_wan") echo '{"nftables": [{"set": {"name": "self_wan", "elem": ["203.0.113.7"]}}]}' ;;
  "-j list table inet sim_agent")
    n=$(cat "$FAKE_T/nft_n" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE_T/nft_n"
    if [ "$n" -gt 0 ] && [ "$FAKE_GROW" = 1 ]; then cat "$FAKE_T/nft_after"; else cat "$FAKE_T/nft_before"; fi ;;
  *) exit 1 ;;
esac
''')
    w("ip", "exit 0\n")
    w("curl", r'''echo "curl $*" >> "$FAKE_T/calls"
[ "$FAKE_CURL_RC" = 0 ] || exit "$FAKE_CURL_RC"
echo "{\"id\": 1, \"has_actions\": $FAKE_HAS_ACTIONS, \"name\": \"neural-simulator\"}"
''')
    w("zfs", r'''echo "zfs $*" >> "$FAKE_T/calls"
echo "$FAKE_GITEA_QUOTA"
''')
    w("docker", r'''printf 'docker %s\n' "$*" >> "$FAKE_T/calls"
case "$1" in
  network) case "$*" in *Subnet*) echo 172.30.30.0/24 ;; *bridge.name*) echo br-simagent ;;
                          *EnableIPv6*) echo false ;; esac ;;
  port) exit 0 ;;
  inspect)
    case "$*" in
      *"State.Status"*) echo running ;;
      *'range $k'*) echo "apps_sim-agent-net " ;;
      *IPAddress*) exit 0 ;;
      *) cat "$FAKE_T/inspect.json" ;;
    esac ;;
  exec)
    case "$*" in
      *"/usr/bin/id -u"*) echo 61000 ;;
      *"/proc/self/status"*) printf 'CapEff:\t0000000000000000\nNoNewPrivs:\t1\n' ;;
      *) exit 1 ;;
    esac ;;
  run) echo "  PASS  (stub probe)"; exit "$FAKE_PROBE_RC" ;;
esac
''')
    return b


def _run(**knobs):
    with tempfile.TemporaryDirectory() as tmp:
        b = _stubs(tmp)
        with open(os.path.join(tmp, "nft_before"), "w") as f:
            f.write(_nft_json(5))
        with open(os.path.join(tmp, "nft_after"), "w") as f:
            f.write(_nft_json(40))
        with open(os.path.join(tmp, "inspect.json"), "w") as f:
            json.dump([knobs.pop("inspect", INSPECT)], f)
        env = dict(os.environ, PATH=b + os.pathsep + os.environ["PATH"], FAKE_T=tmp)
        env.update({"FAKE_ONFAILURE": "ntfy-notify@sim-agent-fw-selfwan.service.service", "FAKE_GROW": "1",
                    "FAKE_CURL_RC": "0", "FAKE_HAS_ACTIONS": "false", "FAKE_GITEA_QUOTA": "32212254720",
                    "FAKE_PROBE_RC": "0"})
        env.update({k: str(v) for k, v in knobs.items()})
        r = subprocess.run(["bash", VERIFY], env=env, capture_output=True, text=True, timeout=60)
        calls = open(os.path.join(tmp, "calls")).read() if os.path.exists(os.path.join(tmp, "calls")) else ""
        return r, calls


def test_the_baseline_passes():
    r, calls = _run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout


def test_network_probes_run_from_a_pristine_container_never_inside_the_agent():
    r, calls = _run()
    runs = [l for l in calls.splitlines() if l.startswith("docker run ")]
    assert len(runs) == 1, runs
    line = runs[0]
    assert "--network apps_sim-agent-net" in line and "--read-only" in line and "--cap-drop ALL" in line
    assert " -v " not in line and "--mount" not in line and "--volume" not in line
    assert "net_probe.sh --must-fail" in line
    for flag in ("--hairpin git.dant123.com", "--ssh-pubkey-only dant123@192.168.0.68:22", "--no-ipv6"):
        assert flag in line, flag
    # the only execs into the agent container are fresh, PATH-pinned, absolute, and never run a shell or a probe
    execs = [l for l in calls.splitlines() if l.startswith("docker exec")]
    assert execs
    for l in execs:
        assert l.startswith("docker exec -e PATH=%s sim-agent /" % SAFE_PATH), l
        assert "/bin/bash" not in l and "/dev/tcp" not in l and "mktemp" not in l, l


def test_a_failed_pristine_probe_fails_the_run():
    r, _ = _run(FAKE_PROBE_RC=1)
    assert r.returncode != 0 and "ALL CHECKS PASSED" not in r.stdout


def test_drop_counters_that_did_not_grow_fail_the_run():
    r, _ = _run(FAKE_GROW=0)
    assert r.returncode != 0
    assert "did not grow" in r.stdout


def test_gitea_actions_on_fails_the_run():
    r, calls = _run(FAKE_HAS_ACTIONS="true")
    assert r.returncode != 0
    assert "Actions are ON" in r.stdout
    assert "/api/v1/repos/dant123/neural-simulator" in calls


def test_an_unreadable_gitea_api_fails_rather_than_passes():
    r, _ = _run(FAKE_CURL_RC=7)
    assert r.returncode != 0
    assert "could not read has_actions" in r.stdout


def test_a_gitea_dataset_without_a_quota_fails_the_run():
    r, calls = _run(FAKE_GITEA_QUOTA=0)
    assert r.returncode != 0
    assert "data/appdata/gitea has NO quota" in r.stdout
    assert "zfs get -Hp -o value quota data/appdata/gitea" in calls


def test_a_selfwan_unit_without_onfailure_fails_the_run():
    r, _ = _run(FAKE_ONFAILURE="")
    assert r.returncode != 0
    assert "sim-agent-fw-selfwan.service has no OnFailure" in r.stdout


def test_container_config_is_read_from_docker_inspect():
    bad = json.loads(json.dumps(INSPECT))
    bad["HostConfig"]["CapAdd"] = ["NET_ADMIN"]
    bad["Mounts"].append({"Type": "bind", "Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"})
    bad["HostConfig"]["ReadonlyRootfs"] = False
    r, _ = _run(inspect=bad)
    assert r.returncode != 0
    for text in ("capabilities added", "docker.sock", "root filesystem is writable"):
        assert text in r.stdout, text
