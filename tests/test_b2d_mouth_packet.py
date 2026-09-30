"""The B2d execution packet (research/packets/b2d-mouth-renderer/): job files, the re-pin check, the local export and the
wave script agree with the pre-registration's Amendment 1 and with the tools they copy (no brain build, no queueing).
"""
import importlib.util
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(ROOT, "research", "packets", "b2d-mouth-renderer")
SHARDS = os.path.join(ROOT, "research", "findings", "raw", "_load_bearing", "_shards")
PREREG = "research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(K, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _archive_paths(path):
    txt = open(path).read()
    m = re.search(r'git archive "\$(?:SOURCE_)?SHA" (.*?)\| tar -x', txt, re.S)
    assert m, path
    return m.group(1).replace("\\\n", " ").split()


def test_the_local_export_archives_exactly_the_pool_provisioners_paths():
    assert _archive_paths(os.path.join(K, "export_local_revision.sh")) == \
        _archive_paths(os.path.join(ROOT, "tools", "pool_provision.sh"))


def test_the_stub_job_file_is_the_generator_output_at_its_recorded_pin():
    mk = _load("make_jobs")
    pin = open(os.path.join(SHARDS, "b2d0929", "PIN.txt")).read().strip()
    want = open(os.path.join(ROOT, "research", "coordination", "b2d_jobs.txt")).read().splitlines()
    got = mk.stub_lines(pin)
    assert got == want and len(got) == 258
    assert all(ln.startswith("cd ~/derisk-pool/revisions/%s && .venv/bin/python tools/assert_flipped_defaults.py && "
                             % pin) and ln.endswith("/lb.json") and "BRAIN_" not in ln for ln in got)


def test_mouth_lines_differ_from_the_stub_line_only_by_backend_mode_and_memory_guard():
    mk = _load("make_jobs")
    pin = "a" * 40
    mouth = mk.mouth_lines(pin)
    assert len(mouth) == 18
    for ln in mouth:
        assert ln.startswith("cd ~/derisk-pool/revisions/%s && .venv/bin/python tools/assert_flipped_defaults.py && "
                             % pin)
        assert " SIM_BACKEND=cupy " in ln and " LB_ARM_RENDERER=production " in ln and "SIM_BACKEND=numpy" not in ln
        assert ("for _w in $(seq 90); do bash tools/mem_ok.sh 10 4 >/dev/null 2>&1 && break; sleep 60; done; "
                "bash tools/mem_ok.sh 10 4 && bash tools/memcap.sh 16 -- env ") in ln
        assert subprocess.run(["bash", "-n", "-c", ln]).returncode == 0
        assert "/_shards/b2d0929-mouth/s" in ln and ln.endswith("/lb.json") and "BRAIN_" not in ln
        assert " --repeats 2 " in ln
        for flag in ("LB_AFFECT_DRIVE_PROBE=1", "LB_EPISODIC_DRIVE_PROBE=1", "LB_OPEN_ENDED_DISTRIB_PROBE=1"):
            assert flag in ln
    rows = {ln.split("--only ")[1].split(" ")[0] for ln in mouth}
    assert rows == {"affect-coloring", "affect-appraisal-interoceptive", "affect-drives-response"}
    seeds = sorted({int(ln.split("--seed ")[1].split(" ")[0]) for ln in mouth})
    assert seeds == [42, 43, 44, 100, 101, 102]


def test_repin_blob_problems_flags_any_unaudited_change(monkeypatch):
    rp = _load("repin")
    fields = {"tag": "b2d0929", "cited_blobs_at_pin": {"a.py": "1", "b.py": "2"},
              "harness_blobs_at_pin": {"research/runners/load_bearing_fraction.py": "9"}}
    blobs = {"a.py": "1", "b.py": "2"}
    monkeypatch.setattr(rp, "blob_at", lambda pin, f: blobs.get(f))
    monkeypatch.setattr(rp, "harness_blobs", lambda pin: {"research/runners/load_bearing_fraction.py": "9"})
    assert rp.blob_problems("x", fields) == []
    blobs["b.py"] = "3"
    assert any("b.py" in p for p in rp.blob_problems("x", fields))
    blobs["b.py"] = "2"
    monkeypatch.setattr(rp, "harness_blobs", lambda pin: {"research/runners/load_bearing_fraction.py": "9",
                                                          "research/runners/lbf_rows/new_row.py": "5"})
    assert any("new_row.py" in p for p in rp.blob_problems("x", fields))


def test_the_committed_fields_file_matches_the_audit_it_records():
    rp = _load("repin")
    fields = json.load(open(os.path.join(ROOT, "research", "coordination", "b2d_decision_fields.json")))
    audit = fields["repin_audit"]
    at = audit["audited_at"]
    # the audited blobs are the audited commit's blobs, for every cited and harness file
    assert rp.blob_problems(at, fields) == []
    cm = fields["rows"]["comprehension-monitor"]
    assert cm["fields"]["comprehension.comprehended"]["class"] == "N"
    assert "comprehension.comprehended" in cm["kept"] and not cm["sd_only"]


def test_registry_at_h_is_a_static_manifest_with_the_prereg_counts():
    reg = json.load(open(os.path.join(K, "registry_at_H.json")))
    assert reg["execution"] == "not_executed" and reg["pin"].startswith("3c2907b85")
    fp = reg["fingerprint"]
    kinds = {}
    for r in fp["rows"].values():
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    assert fp["n_rows"] == 50 and len(fp["sharded_keys"]) == 43
    assert kinds == {"neural-lesion": 33, "whether-disable": 3, "thin": 6, "mechanism-only": 1, "in-process": 4,
                     "proposed": 3}


def test_the_wave_script_reason_passes_the_gate_seed_guard_and_declares_its_age():
    src = open(os.path.join(ROOT, "research", "coordination", "b2d_queue_next_wave.sh")).read()
    assert subprocess.run(["bash", "-n", "-c", src]).returncode == 0
    m = re.search(r'^REASON="(.*)"$', src, re.M)
    reason = m.group(1).replace("${F:0:9}", "abcdef012")
    # tools/pool_queue.sh's GATE-SEED GUARD: the path is the token after "prereg", up to whitespace
    path = re.search(r"prereg\s+(\S+)", reason).group(1)
    assert path == PREREG and os.path.exists(os.path.join(ROOT, path))
    assert re.search(r"\bmem_gb=8\b", reason) and re.search(r"\bmax_age_h=168\b", reason)
    assert 'TAG=b2d0929\n' in src and 'JOBS="$HERE/b2d_jobs.txt"' in src
