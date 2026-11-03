"""A busy resource ends its idle episode: the next idle stretch alerts at once, not REPEAT_H after the last alert."""
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "attention_watch", os.path.join(os.path.dirname(__file__), "..", "tools", "attention_watch.py"))
aw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aw)


def test_busy_gpu_clears_the_idle_alert(monkeypatch):
    def sh(cmd, timeout=60):
        if "nvidia-smi" in cmd:
            return 0, "100\n"
        if cmd[0] == "ssh":
            return 0, "BUSY\n"
        return 0, "current: some_job\nqueued: 0\n"
    monkeypatch.setattr(aw, "sh", sh)
    monkeypatch.setattr(aw, "aws_checks", lambda: [])
    st = {"seen": {"w-gpu-idle": 1.0, "minipc-idle-pool41": 1.0, "s-other": 1.0}, "gpu_idle_since": None,
          "gpu_log_off": 0, "journal_since": None}
    try:
        aw.w_checks(st, 1000.0)
    except Exception:
        pass                  # later probes (journal, oom) may need the real host; the idle part runs first
    assert "w-gpu-idle" not in st["seen"]
    assert "minipc-idle-pool41" not in st["seen"]
    assert "s-other" in st["seen"]
