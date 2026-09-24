"""Pins the 2026-09-24 FAILURE_LOG row (perception G0 thread-count mismatch): runner provenance must record the
math-library thread counts, `cpu_count` and every `BRAIN_*` production flag, so two artifacts that differ only in
one of these can be told apart after the fact. The pre-e0a373549 env filter (SIM_/GAP5_/HEBB_/POOL_/GAP4_ only,
no cpu_count) does not satisfy it."""
import json

import research.runners as R


def test_provenance_records_thread_counts_cpu_count_and_brain_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_PROV_DIR", str(tmp_path))
    for k in ("SIM_PROVENANCE_V2", "SIM_PROVENANCE_SCHEMA"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "3")
    monkeypatch.setenv("BRAIN_TEST_PROVENANCE_FLAG", "1")
    monkeypatch.setenv("UNRELATED_SECRETISH_VAR", "x")
    rec = R._record_start()
    env = rec["env"]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert env.get(k) == "3", k
    assert env.get("BRAIN_TEST_PROVENANCE_FLAG") == "1"
    assert "UNRELATED_SECRETISH_VAR" not in env
    assert isinstance(rec.get("cpu_count"), int) and rec["cpu_count"] >= 1
    line = (tmp_path / "runs.jsonl").read_text().strip().splitlines()[-1]
    assert json.loads(line)["env"].get("OMP_NUM_THREADS") == "3"
