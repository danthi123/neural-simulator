"""0a module contract tests (packet lines 98-100).

Five contracts, each named there:
  1. every module in the 0a family answers --selftest with rc=0;
  2. the stream writer (module 1) is byte-identical across two writes, with the
     H field present in every row;
  3. the replica AND the audit check refuse a stream whose sha256 differs from
     the manifest (both load through _deixis_rule_replica.load_manifests);
  4. no module under webapp/ or any production runner imports a _deixis_* module;
  5. `git diff --stat main -- sim/` is empty (0a touches no sim/ line).
"""
import json
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ENV = dict(os.environ, SIM_BACKEND="numpy", CUDA_VISIBLE_DEVICES="")
FAMILY = ["_deixis_toy_world", "_deixis_hearing_core",
          "_deixis_rule_replica", "_deixis_audit_check",
          "_deixis_calibration", "_deixis_bar_sanity_0a"]


def run_mod(name, *args):
    return subprocess.run(
        [sys.executable, "-m", "research.runners." + name] + list(args),
        cwd=REPO, env=ENV, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)


@pytest.mark.parametrize("name", FAMILY)
def test_module_selftest(name):
    r = run_mod(name, "--selftest")
    assert r.returncode == 0, "%s --selftest rc=%d\n%s" % (
        name, r.returncode, r.stdout[-4000:])


def test_writer_byte_identical_with_h(tmp_path):
    from research.runners import _deixis_toy_world as M1
    outs = []
    for i in (1, 2):
        d = tmp_path / ("write%d" % i)
        man = M1.write_streams("base", [1, 2], 30,
                               str(d / "streams" / "manifest_base.json"))
        outs.append((d, man))
    (d1, m1), (d2, m2) = outs
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    n_rows = 0
    for e in m1["streams"]:
        a, b = d1 / "streams" / e["file"], d2 / "streams" / e["file"]
        ba, bb = a.read_bytes(), b.read_bytes()
        assert ba == bb, "writes differ: %s" % e["file"]
        lines = ba.decode().splitlines()
        for ln in lines[1:]:
            d = json.loads(ln)
            assert "H" in d, "H field missing in %s" % e["file"]
            for k, ent in d["H"]:
                assert 0 <= k < 5 and 0 <= ent < 10
            n_rows += 1
    assert n_rows == 60


def test_replica_and_audit_refuse_sha_mismatch(tmp_path):
    from research.runners import _deixis_toy_world as M1
    from research.runners import _deixis_rule_replica as R4
    from research.runners import _deixis_audit_check as R5
    d = tmp_path / "streams"
    M1.write_streams("base", [1], 20, str(d / "manifest_base.json"))
    target = d / "base" / "s1.jsonl"
    raw = target.read_bytes()
    i = len(raw) // 2
    nb = b"X" if raw[i:i + 1] != b"X" else b"Y"
    target.write_bytes(raw[:i] + nb + raw[i + 1:])
    loaders = [("replica", R4.load_manifests),
               ("audit", R5.R.load_manifests)]
    for label, loader in loaders:
        with pytest.raises(ValueError) as ei:
            loader(str(d))
        assert "sha256 mismatch" in str(ei.value), \
            "%s did not refuse on sha: %s" % (label, ei.value)
        assert "base/s1.jsonl" in str(ei.value)


def test_no_production_module_imports_deixis():
    pat = re.compile(r"\b_deixis_\w+")
    bad = []
    for top in ("webapp", "research/runners"):
        for dirpath, _dirs, files in os.walk(os.path.join(REPO, top)):
            for fn in files:
                if not fn.endswith(".py") or fn.startswith("_deixis_"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    txt = open(p, encoding="utf-8",
                               errors="replace").read()
                except OSError:
                    continue
                if pat.search(txt):
                    bad.append(os.path.relpath(p, REPO))
    assert not bad, "production code importing _deixis_*: %s" % bad


def test_sim_diff_vs_main_empty():
    # Pool revisions are plain tree copies (no .git), so the diff contract can
    # only be asserted where git + the main ref exist (home checkout).
    probe = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           cwd=REPO, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    if probe.returncode != 0:
        pytest.skip("not a git checkout (pool revision): %s"
                    % probe.stdout.strip())
    probe = subprocess.run(["git", "rev-parse", "--verify", "main"],
                           cwd=REPO, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    if probe.returncode != 0:
        pytest.skip("no main ref in this checkout")
    r = subprocess.run(["git", "diff", "--stat", "main", "--", "sim/"],
                       cwd=REPO, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    assert r.returncode == 0, r.stdout
    assert r.stdout.strip() == "", \
        "0a must not touch sim/: git diff main -- sim/ ->\n%s" % r.stdout