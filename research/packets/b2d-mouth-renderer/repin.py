#!/usr/bin/env python3
"""repin.py -- B2d's re-pin rule, made mechanical (the B2d pre-registration's "Amendment 1" section, A1.4). Run once,
at provisioning time, with the pin P the pool and the workstation will run.

  .venv/bin/python research/packets/b2d-mouth-renderer/repin.py check --pin <P> --tree ~/derisk-pool/revisions/<P>
  .venv/bin/python research/packets/b2d-mouth-renderer/repin.py apply --pin <P> --tree ~/derisk-pool/revisions/<P>

The pre-registration's rule: if origin/main has moved past H, H is replaced by a new pin through an amendment that
regenerates b2d_jobs.txt and PIN.txt, recomputes the registry and names any change, compares the git blob of every
file research/coordination/b2d_decision_fields.json cites and re-audits each row whose cited file changed. Amendment
1 did the re-audit ONCE, by hand, for every cited file that differs between H and the renderer-change revision, and
recorded the audited blobs in the fields file (`cited_blobs_at_pin`, `harness_blobs_at_pin`). This script does the
rest without judgment:

  check -- exit 0 only if ALL hold, else exit 1 and print every difference (a difference is NEEDS-CLAUDE: a cited or
           harness file changed after the audit, or the registry changed, and a human re-audit is due):
    1. P is a full sha, a commit of this repository, and contains origin/main (the provisioner's own guard);
    2. the tree at --tree is an export of P (its .source_revision reads git_sha=P) and its manifest verifies;
    3. the registry fingerprint of the tree (registry_fingerprint.py) equals registry_at_H.json (the registry at H);
    4. every cited file's blob at P equals `cited_blobs_at_pin`, and every harness file's blob at P (the set is
       HARNESS_GLOBS, matched against P's tree, so an added lbf_rows file also counts) equals `harness_blobs_at_pin`.
  apply -- runs check; on exit 0 it rewrites the fields file's `pin` to P (appending the change to `repin_history`),
           regenerates research/coordination/b2d_jobs.txt (+ b2d0929/PIN.txt) and the mouth job file
           research/packets/b2d-mouth-renderer/jobs/mouth_<P7>.txt (+ b2d0929-mouth/PIN.txt) with make_jobs.py, and
           prints the exact paths to commit. It never commits, provisions or queues.
"""
import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIELDS = os.path.join(ROOT, "research", "coordination", "b2d_decision_fields.json")
REGISTRY_AT_H = os.path.join(HERE, "registry_at_H.json")
HARNESS_GLOBS = ("research/runners/load_bearing_fraction.py", "research/runners/onebrain_regression_battery.py",
                 "research/runners/lbf_rows/*.py", "tools/lab.py", "tools/lb_shard.py")


def _git(*args):
    return subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True)


def blob_at(pin, path):
    r = _git("rev-parse", "%s:%s" % (pin, path))
    return r.stdout.strip() if r.returncode == 0 else None


def harness_blobs(pin):
    files = _git("ls-tree", "-r", "--name-only", pin, "research/runners", "tools").stdout.split()
    keep = sorted(f for f in files if any(fnmatch.fnmatchcase(f, g) for g in HARNESS_GLOBS))
    return {f: blob_at(pin, f) for f in keep}


def check(pin, tree, fields=None, python=None):
    fields = fields if fields is not None else json.load(open(FIELDS))
    problems = []
    if not re.fullmatch(r"[0-9a-f]{40}", pin or ""):
        return ["pin %r is not a full 40-hex sha" % pin]
    if _git("cat-file", "-e", pin + "^{commit}").returncode != 0:
        return ["pin %s is not a commit of this repository" % pin]
    _git("fetch", "-q", "origin", "main")
    main = _git("rev-parse", "origin/main").stdout.strip()
    if _git("merge-base", "--is-ancestor", main, pin).returncode != 0:
        problems.append("pin %s does not contain origin/main %s" % (pin, main))
    rev = os.path.join(tree, ".source_revision")
    got = {}
    if os.path.exists(rev):
        for ln in open(rev):
            k, _, v = ln.strip().partition("=")
            got[k] = v
    if got.get("git_sha") != pin:
        problems.append("tree %s is not an export of %s (.source_revision git_sha=%s)" % (tree, pin, got.get("git_sha")))
    else:
        py = python or os.path.join(ROOT, ".venv", "bin", "python")
        v = subprocess.run([py, os.path.join(tree, "tools", "pool", "provisioning", "source_manifest.py"), "verify",
                            "--root", tree, "--manifest", os.path.join(tree, ".source_manifest.sha256"),
                            "--expected-sha256", got.get("source_manifest_sha256", "")], capture_output=True, text=True)
        if v.returncode != 0:
            problems.append("tree manifest does not verify: %s" % (v.stderr.strip() or v.stdout.strip())[:300])
        env = {k: v for k, v in os.environ.items() if not (k.startswith("LB_") or k.startswith("BRAIN_"))}
        env.update(SIM_NO_PROVENANCE="1", SIM_BACKEND="numpy")
        fp = subprocess.run([py, os.path.join(HERE, "registry_fingerprint.py")], cwd=tree, env=env,
                            capture_output=True, text=True)
        if fp.returncode != 0:
            problems.append("registry fingerprint failed at the tree: %s" % fp.stderr.strip()[-300:])
        else:
            want = json.load(open(REGISTRY_AT_H))["fingerprint"]
            have = json.loads(fp.stdout)
            for k in sorted(set(want) | set(have)):
                if want.get(k) != have.get(k):
                    problems.append("registry changed since H: %s" % k)
    if fields.get("tag") != "b2d0929":
        problems.append("fields file tag %r != b2d0929" % fields.get("tag"))
    for f, want in sorted((fields.get("cited_blobs_at_pin") or {}).items()):
        have = blob_at(pin, f)
        if have != want:
            problems.append("cited file changed after the audit: %s (audited %s, at pin %s)" % (f, want, have))
    want_h = fields.get("harness_blobs_at_pin") or {}
    have_h = harness_blobs(pin)
    for f in sorted(set(want_h) | set(have_h)):
        if want_h.get(f) != have_h.get(f):
            problems.append("harness file differs from the audited one: %s (audited %s, at pin %s)"
                            % (f, want_h.get(f), have_h.get(f)))
    return problems


def apply(pin, tree):
    fields = json.load(open(FIELDS))
    problems = check(pin, tree, fields)
    if problems:
        return problems
    old = fields.get("pin")
    if old != pin:
        fields.setdefault("repin_history", []).append(
            {"from": old, "to": pin, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "by": "research/packets/b2d-mouth-renderer/repin.py apply (all checks passed)"})
        fields["pin"] = pin
        with open(FIELDS, "w") as fh:
            json.dump(fields, fh, indent=1)
            fh.write("\n")
    mk = os.path.join(HERE, "make_jobs.py")
    py = sys.executable
    env = dict(os.environ, SIM_NO_PROVENANCE="1")
    subprocess.run([py, mk, "stub", "--pin", pin, "--out", os.path.join(ROOT, "research", "coordination",
                                                                         "b2d_jobs.txt")], check=True, env=env)
    mouth = os.path.join(HERE, "jobs", "mouth_%s.txt" % pin[:7])
    subprocess.run([py, mk, "mouth", "--pin", pin, "--out", mouth], check=True, env=env)
    print("commit these paths (pathspec):")
    for p in ("research/coordination/b2d_decision_fields.json", "research/coordination/b2d_jobs.txt",
              "research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt",
              "research/findings/raw/_load_bearing/_shards/b2d0929-mouth/PIN.txt",
              os.path.relpath(mouth, ROOT)):
        print("  " + p)
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("check", "apply"))
    ap.add_argument("--pin", required=True)
    ap.add_argument("--tree", required=True)
    a = ap.parse_args(argv)
    tree = os.path.expanduser(a.tree)
    problems = check(a.pin, tree) if a.action == "check" else apply(a.pin, tree)
    if problems:
        print("REPIN %s: %d problem(s) -- NEEDS-CLAUDE, do not queue" % (a.action.upper(), len(problems)))
        for p in problems:
            print("  - " + p)
        return 1
    print("REPIN %s OK: pin %s" % (a.action.upper(), a.pin))
    return 0


if __name__ == "__main__":
    sys.exit(main())
