"""tools/audit_pool_fragment_claims.py must find a mid-line fragment claim, map it to its node / executed text /
swallowing probe / exit status, name the full line it came from and that line's later real run -- and must NOT
flag a complete line. Synthetic queue files only; nothing live is read."""
import base64
import datetime as dt
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "audit_pool_fragment_claims.py")

FULL = ("cd ~/derisk-pool/revisions/" + "a" * 40 + " && env SIM_BACKEND=numpy .venv/bin/python -u -m "
        "research.runners.some_runner --seed 42 --out research/findings/raw/_x/s42/out.json  #checked:prereg X")
OTHER = "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.other --seed 1  #checked:reason Y"
FRAG = FULL[FULL.index("s42/out.json"):]          # "s42/out.json  #checked:prereg X" -- a mid-line tail


def _executed(job):
    body = job.split("#checked:", 1)[0].rstrip()
    reason = job.split("#checked:", 1)[1]
    return "POOL_CHECKED_REASON=%s %s" % (reason.replace(" ", "\\ "), body)


def _write(qd, t0):
    claims = [(t0, OTHER), (t0 + 100, FRAG), (t0 + 900, FULL)]
    with open(os.path.join(qd, "pool.queue.claims"), "w") as fh:
        for t, j in claims:
            fh.write("%d\t%s\n" % (t, j))
    open(os.path.join(qd, "pool.queue"), "w").close()
    with open(os.path.join(qd, "pool.running"), "w") as fh:
        for (t, j), node in zip(claims, ("poolA", "poolB", "poolB")):
            fh.write("%s\t%s\t%s\n" % (dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), node,
                                         _executed(j)))
    with open(os.path.join(qd, "dispatch.log"), "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")
        fh.write("[pool-dispatch] 00:00:02 poolA <- %s\n" % _executed(OTHER))
        fh.write("[pool-dispatch] revision %s not provisioned on poolB -- job(s) pinned to it stay queued\n" % ("b" * 40))
        fh.write("[pool-dispatch] 00:01:42 poolB <- %s\n" % _executed(FRAG))
    st = os.path.join(qd, "status")
    os.makedirs(st)
    with open(os.path.join(st, "poolB.job_status.log"), "w") as fh:
        fh.write("v2\t%d\t127\t%s\n" % (t0 + 101, base64.b64encode(_executed(FRAG).encode()).decode()))
        fh.write("v2\t%d\t0\t%s\n" % (t0 + 950, base64.b64encode(_executed(FULL).encode()).decode()))
    return st


def test_fragment_detected_mapped_and_complete_lines_not_flagged(tmp_path):
    qd = str(tmp_path)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    st = _write(qd, t0)
    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status", st,
                        "--json", out], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    assert rep["n_fragments"] == 1, rep
    f = rep["fragments"][0]
    assert f["claim_line"] == 2 and f["fragment"] == FRAG
    assert f["node"] == "poolB" and f["pinned"] is False
    assert f["swallowing_probe"]["node"] == "poolB"
    assert f["classification"]["expect"].startswith("command-not-found")
    assert [s["rc"] for s in f["node_status"]] == [127]
    (parent,) = f["parents"]
    later = parent["occurrences_at_or_after_fragment"]
    assert [(o["line"], o["node"], [s["rc"] for s in o["node_status"]]) for o in later] == [(3, "poolB", [0])]
    assert rep["nonstandard_start_complete_lines"] == []   # OTHER and FULL are ordinary complete lines
