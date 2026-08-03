from __future__ import annotations

from tools import lane_check


def test_queue_reader_uses_shared_root_override(tmp_path, monkeypatch):
    queue_dir = tmp_path / "research" / "queue"
    queue_dir.mkdir(parents=True)
    (queue_dir / "gpu.queue").write_text(
        "SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap4_probe\n",
        encoding="utf-8",
    )
    (queue_dir / "pool.queue").write_text(
        "123\tSIM_BACKEND=numpy .venv/bin/python -m "
        "research.runners._curiosity_probe  #checked:prior record read\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SIM_QUEUE_ROOT", str(tmp_path))

    assert lane_check._queue_jobs() == [
        "SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap4_probe",
        "SIM_BACKEND=numpy .venv/bin/python -m research.runners._curiosity_probe",
    ]


def test_explicit_queue_paths_take_precedence(tmp_path, monkeypatch):
    gpu = tmp_path / "custom-gpu.queue"
    pool = tmp_path / "custom-pool.queue"
    gpu.write_text("gpu-job\n", encoding="utf-8")
    pool.write_text("456\tpool-job  #checked:catalog read\n", encoding="utf-8")
    monkeypatch.setenv("SIM_QUEUE_ROOT", str(tmp_path / "unused"))
    monkeypatch.setenv("GPU_QUEUE_PATH", str(gpu))
    monkeypatch.setenv("POOL_QUEUE_PATH", str(pool))

    assert lane_check._queue_jobs() == ["gpu-job", "pool-job"]
