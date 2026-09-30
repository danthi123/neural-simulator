"""The sim-agent network-isolation probes (tools/agent_container/bin/net_probe.sh) and the host-side launcher that
runs them from a throwaway PRISTINE container (tools/agent_container/server-stack/probe_from_pristine.sh).

2026-09-30 security review, HIGH #1: the probes used to run INSIDE the long-lived sim-agent container (smoke_test.sh
section 7, verify_isolation.sh's `docker exec ... bash -s`), where the model owns the home (PATH, rc files) and
shares /tmp with the probe's result files, so "ALL CHECKS PASSED" was the model's to choose. They now run from a
container that starts from the image alone (no mounts) on the same bridge.

Hermetic: real loopback listeners stand in for "accepted"/"refused"; a stub stands in for the one outcome loopback
cannot produce (a silent drop = a timeout), substituted into a COPY of the script at the absolute path it names.
"""
import os
import re
import shlex
import socket
import stat
import subprocess
import tempfile

import pytest

try:
    import yaml
except ImportError:
    yaml = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AC = os.path.join(ROOT, "tools", "agent_container")
PROBE = os.path.join(AC, "bin", "net_probe.sh")
PRISTINE = os.path.join(AC, "server-stack", "probe_from_pristine.sh")
FRAGMENT = os.path.join(AC, "server-stack", "sim-agent", "compose.yaml")


def _exec(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


def _probe_copy(tmp, **stubs):
    """A copy of net_probe.sh with some of its absolute binaries (TIMEOUT/CURL/SSH/GETENT) swapped for stubs."""
    text = open(PROBE).read()
    for var, body in stubs.items():
        line = re.search(r"^%s=(/\S+)$" % var, text, re.MULTILINE)
        assert line, "net_probe.sh must name %s by an absolute path" % var
        text = text.replace(line.group(0), "%s=%s" % (var, _exec(os.path.join(tmp, var.lower() + "_stub"), body)))
    copy = os.path.join(tmp, "net_probe.sh")
    with open(copy, "w") as f:
        f.write(text)
    return copy


def _run(script, *args):
    return subprocess.run(["bash", script, *args], capture_output=True, text=True, timeout=60)


class _Listener:
    def __enter__(self):
        self.s = socket.socket()
        self.s.bind(("127.0.0.1", 0))
        self.s.listen(8)
        self.port = self.s.getsockname()[1]
        return self

    def __exit__(self, *exc):
        self.s.close()


def _closed_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ------------------------------------------------------------------------------------------ net_probe.sh

def test_must_fail_target_that_accepts_or_refuses_fails_the_run():
    with _Listener() as lst:
        r = _run(PROBE, "--must-fail", "127.0.0.1:%d 127.0.0.1:%d" % (lst.port, _closed_port()))
    assert r.returncode == 1
    assert "NOT blocked" in r.stdout and "accepted a connection" in r.stdout
    assert "NOT dropped" in r.stdout and "refusal" in r.stdout
    assert "net_probe: 2 failure(s)" in r.stdout


def test_must_fail_target_passes_only_on_a_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        copy = _probe_copy(tmp, TIMEOUT="exit 124\n")
        r = _run(copy, "--must-fail", "192.0.2.1:22 192.0.2.1:445")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("PASS  blocked:") == 2


def test_must_pass_target_fails_only_on_a_timeout():
    with _Listener() as lst:
        r = _run(PROBE, "--must-pass", "127.0.0.1:%d 127.0.0.1:%d" % (lst.port, _closed_port()))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "reachable: 127.0.0.1:%d" % lst.port in r.stdout
    assert "nothing listening right now" in r.stdout
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(_probe_copy(tmp, TIMEOUT="exit 124\n"), "--must-pass", "192.0.2.1:2222")
    assert r.returncode == 1 and "an allowed path is being dropped" in r.stdout


def test_a_malformed_target_is_a_failure_never_skipped():
    r = _run(PROBE, "--must-fail", "gateway;id:22")
    assert r.returncode == 1 and "malformed" in r.stdout


@pytest.mark.parametrize("curl_rc,ok,text", [(28, True, "hairpin closed"), (0, False, "hairpin OPEN"),
                                             (6, False, "did not resolve"), (7, False, "failed fast")])
def test_hairpin_passes_only_when_the_wan_address_times_out(curl_rc, ok, text):
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(_probe_copy(tmp, CURL="exit %d\n" % curl_rc), "--hairpin", "git.example.org")
    assert (r.returncode == 0) == ok, r.stdout
    assert text in r.stdout


@pytest.mark.parametrize("reply,ok,text", [
    ("dant123@192.168.0.68: Permission denied (publickey).", True, "offers only 'publickey'"),
    ("dant123@192.168.0.68: Permission denied (publickey,password).", False, "a password login is possible"),
    ("dant123@192.168.0.68: Permission denied (publickey,keyboard-interactive).", False, "a password login"),
    ("ssh: connect to host 192.168.0.68 port 22: Connection timed out", False, "the check did not run"),
])
def test_ssh_pubkey_only_reads_the_servers_own_method_list(reply, ok, text):
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "ssh_args")
        copy = _probe_copy(tmp, SSH='printf "%%s\\n" "$*" > %s\necho %s >&2\nexit 255\n'
                                    % (shlex.quote(log), shlex.quote(reply)))
        r = _run(copy, "--ssh-pubkey-only", "dant123@192.168.0.68:22")
        args = open(log).read()
    assert (r.returncode == 0) == ok, r.stdout
    assert text in r.stdout
    # never offers a key or a password, never trusts or writes a known_hosts, reads no ssh config at all
    for opt in ("-F /dev/null", "PreferredAuthentications=none", "PubkeyAuthentication=no", "BatchMode=yes",
                "UserKnownHostsFile=/dev/null"):
        assert opt in args, opt


def test_probe_script_resets_path_and_names_every_binary_absolutely():
    text = open(PROBE).read()
    assert text.startswith("#!/bin/bash\n")
    assert "\nexport PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n" in text
    for var in ("TIMEOUT", "CURL", "SSH", "GETENT"):
        assert re.search(r"^%s=/usr/bin/\w+$" % var, text, re.MULTILINE), var
    assert '/bin/bash -c \'exec 3<>"/dev/tcp/$0/$1"\'' in text


# ------------------------------------------------------------------------- probe_from_pristine.sh

def _pristine_argv(*args):
    with tempfile.TemporaryDirectory() as tmp:
        fakebin = os.path.join(tmp, "bin")
        os.makedirs(fakebin)
        log = os.path.join(tmp, "docker_argv")
        _exec(os.path.join(fakebin, "docker"), 'for a in "$@"; do printf "%%s\\n" "$a"; done > %s\n' % log)
        env = dict(os.environ, PATH=fakebin + os.pathsep + os.environ["PATH"])
        r = subprocess.run(["bash", PRISTINE, *args], env=env, capture_output=True, text=True, timeout=10)
        assert r.returncode == 0, r.stderr
        return open(log).read().splitlines()


def test_the_probe_container_has_no_mounts_and_no_privileges():
    argv = _pristine_argv("--", "--must-fail", "172.30.30.1:22")
    assert argv[:3] == ["run", "--rm", "--pull"] and argv[3] == "never"
    joined = " ".join(argv)
    for flag in ("-v", "--volume", "--mount", "--volumes-from", "--privileged", "--cap-add", "--device"):
        assert flag not in argv, flag
    assert "docker.sock" not in joined
    for pair in (("--network", "apps_sim-agent-net"), ("--cap-drop", "ALL"), ("--user", "61000:61000"),
                 ("--security-opt", "no-new-privileges:true"), ("--entrypoint", "/bin/bash")):
        assert pair[0] in argv and argv[argv.index(pair[0]) + 1] == pair[1], pair
    assert "--read-only" in argv
    tmpfs = argv[argv.index("--tmpfs") + 1]
    assert tmpfs.startswith("/tmp:") and "noexec" in tmpfs
    i = argv.index("sim-agent:local")
    assert argv[i + 1:] == ["/usr/local/lib/agent-container/bin/net_probe.sh", "--must-fail", "172.30.30.1:22"]


@pytest.mark.skipif(yaml is None, reason="needs pyyaml")
def test_the_probe_container_uses_the_fragments_network_and_dns():
    frag = yaml.safe_load(open(FRAGMENT))
    argv = _pristine_argv("--")
    dns = [argv[i + 1] for i, a in enumerate(argv) if a == "--dns"]
    assert dns == frag["services"]["sim-agent"]["dns"]
    net = argv[argv.index("--network") + 1]
    assert net == "apps_" + list(frag["networks"])[0]
    assert _pristine_argv("--image", "sim-agent:rollback", "--")[-2] == "sim-agent:rollback"
