"""Guard (2026-09-26): inside pytest, tools/local_llm/llm.sh and tools/gpu_queue.sh must never reach the REAL
local-llm unit. tests/conftest.py defaults LOCAL_LLM_SYSTEMCTL to a stub that reports every unit inactive; a full
suite run without it stopped the owner's running model mid-session and left the unit failed."""
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_llm_systemctl_is_stubbed_for_the_whole_session():
    assert os.environ.get("LOCAL_LLM_SYSTEMCTL", "").endswith("_stub_systemctl_inactive.sh")


def test_llm_off_inside_pytest_cannot_touch_the_real_unit():
    out = subprocess.run(["bash", os.path.join(ROOT, "tools", "local_llm", "llm.sh"), "off"], capture_output=True,
                         text=True, timeout=30)
    assert "not loaded" in out.stdout, out.stdout + out.stderr
