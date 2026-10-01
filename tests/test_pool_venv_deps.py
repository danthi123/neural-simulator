"""The pool venv (requirements-pool.txt) must provide every module the pool runners import.

Incident (2026-09-30): tools/pool_provision.sh installed a hand-kept pip list with no httpx, so every runner that
drives the live /api/brain-chat handler in-process (`from fastapi.testclient import TestClient`, imported lazily
inside a function) died at import on pool41/pool42 -- iw1_default_identity at 049d4bb0b among them -- while the
workstation venv, which has httpx, never showed it. These tests fail if a pool runner's static import closure
reaches a third-party module the pinned pool set does not provide, and pin that pool_provision.sh installs that
set (not a second, hand-kept list).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools.pool.provisioning import import_closure as ic
from tools.pool.provisioning import pool_venv_check

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements-pool.txt"


def _provided(requirements: Path = REQUIREMENTS) -> set[str]:
    try:
        return ic.provided_modules(ic.requirement_names(requirements))
    except ic.RequirementNotInstalled as exc:   # the check resolves dependencies from the running environment
        pytest.skip(f"pool requirement not installed in this environment: {exc}")


# --- the requirements file and the provisioner --------------------------------------------------------------

def test_requirements_pool_is_fully_pinned() -> None:
    pins = pool_venv_check.parse_pins(str(REQUIREMENTS))   # raises SystemExit on any non-`name==version` line
    for needed in ("numpy", "scipy", "fastapi", "starlette", "pydantic", "httpx"):
        assert needed in pins, f"requirements-pool.txt lost {needed}"


def test_provision_installs_requirements_pool_not_a_second_list() -> None:
    provision = (REPO / "tools/pool_provision.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in provision.splitlines() if not line.lstrip().startswith("#"))
    assert "tools/pool_venv_deps.sh" in code, "pool_provision.sh must install the venv through pool_venv_deps.sh"
    assert not re.search(r"pip\S*\s+(?:-q\s+)?install\s+(?!-r\b|--upgrade pip)[\w\\]", code), \
        "pool_provision.sh pip-installs packages directly; the pool set lives in requirements-pool.txt"
    assert 'FAILED_NODES+=("$h:venv-deps")' in code, "a failed venv install must mark the node FAILED"
    deps = (REPO / "tools/pool_venv_deps.sh").read_text(encoding="utf-8")
    assert "-r .venv/pool-requirements.txt" in deps
    assert "pool_venv_check.py" in deps


def test_checker_harness_matches_closure_extras() -> None:
    assert set(pool_venv_check.HARNESS_SUBMODULES) == set(ic.SUBMODULE_EXTRA_IMPORTS)


def test_top_level_modules_do_not_need_top_level_txt(tmp_path: Path) -> None:
    # Python 3.10 (the pool nodes) resolves packages_distributions() from top_level.txt alone, which numpy /
    # scipy / fastapi / httpx wheels do not ship; the first live run of the checker failed every pin that way.
    from importlib import metadata

    info = tmp_path / "fakepkg-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakepkg\nVersion: 1.0\n", encoding="utf-8")
    (info / "RECORD").write_text(
        "fakemod/__init__.py,,\nfakemod/sub/x.py,,\nfakepkg.libs/libz.so,,\nfake_ext.cpython-310-x86_64-linux-gnu.so,,\n"
        "single.py,,\n_private.py,,\n../../bin/tool,,\nfakepkg-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    dist = metadata.PathDistribution(info)
    assert pool_venv_check.top_level_modules(dist) == ["fake_ext", "fakemod", "single"]


# --- the closure analysis catches the incident (negative control) ------------------------------------------

def _harness_tree(root: Path) -> None:
    (root / "research/runners").mkdir(parents=True)
    (root / "research/__init__.py").write_text("", encoding="utf-8")
    (root / "research/runners/__init__.py").write_text("", encoding="utf-8")
    (root / "research/runners/chat_gate.py").write_text(
        "import json\n"
        "try:\n    import psutil\nexcept ImportError:\n    psutil = None\n"
        "def run():\n"
        "    from fastapi.testclient import TestClient\n"
        "    import torch\n"
        "    return TestClient, torch\n",
        encoding="utf-8")


def test_lazy_brain_chat_harness_needs_httpx(tmp_path: Path) -> None:
    _harness_tree(tmp_path)
    closure = ic.import_closure(tmp_path, ["research.runners.chat_gate"])
    third = closure.third_party()
    assert third["httpx"]["eager"] is False and third["httpx"]["guarded_everywhere"] is False
    assert third["httpx"]["alternatives"] == ["httpx", "httpx2"]
    assert third["psutil"]["guarded_everywhere"] is True
    without_httpx = {"fastapi", "starlette", "pydantic", "numpy"}
    gaps = ic.pool_gaps(closure, without_httpx)
    assert set(gaps) == {"httpx"}, gaps          # torch: NOT_ON_POOL + lazy; psutil: guarded
    assert ic.pool_gaps(closure, without_httpx | {"httpx"}) == {}
    assert ic.pool_gaps(closure, without_httpx | {"httpx2"}) == {}   # starlette 1.x's preferred client


def test_eager_not_on_pool_import_is_a_gap(tmp_path: Path) -> None:
    (tmp_path / "research/runners").mkdir(parents=True)
    (tmp_path / "research/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "research/runners/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "research/runners/gpu_gate.py").write_text("from . import helper\n", encoding="utf-8")
    (tmp_path / "research/runners/helper.py").write_text("import torch\n", encoding="utf-8")
    gaps = ic.pool_gaps(ic.import_closure(tmp_path, ["research.runners.gpu_gate"]), set())
    assert set(gaps) == {"torch"} and "cannot start" in str(gaps["torch"]["reason"])


def test_old_hand_kept_list_would_have_failed(tmp_path: Path) -> None:
    old = tmp_path / "old-requirements-pool.txt"
    old.write_text("numpy==2.2.6\nscipy==1.15.3\nh5py==3.16.0\npillow==12.0.0\npyyaml==6.0.3\npytest==8.4.1\n"
                   "fastapi==0.139.1\npydantic==2.13.4\n", encoding="utf-8")
    old_provided = _provided(old)
    _harness_tree(tmp_path / "tree")
    closure = ic.import_closure(tmp_path / "tree", ["research.runners.chat_gate"])
    assert "httpx" in ic.pool_gaps(closure, old_provided)
    assert ic.pool_gaps(closure, _provided()) == {}


# --- the real pool runners present in this tree ------------------------------------------------------------

PRESENT_POOL_RUNNERS = [r for r in ic.POOL_RUNNERS if (REPO / (r.replace(".", "/") + ".py")).is_file()]


def test_at_least_one_pool_runner_is_checked() -> None:
    assert PRESENT_POOL_RUNNERS, "no POOL_RUNNERS present: the coverage test below would be vacuous"


@pytest.mark.parametrize("runner", PRESENT_POOL_RUNNERS)
def test_pool_runner_imports_are_provided_by_the_pool_venv(runner: str) -> None:
    gaps = ic.pool_gaps(ic.import_closure(REPO, [runner]), _provided())
    assert gaps == {}, (
        f"{runner} reaches third-party modules the pool venv lacks: "
        + "; ".join(f"{m} ({g['reason']}; first at {g['first_site']})" for m, g in gaps.items())
        + ". Add the distribution to requirements-pool.txt (pinned), guard the import, or -- GPU/LLM only -- "
          "keep it out of module level.")


def test_brain_chat_server_imports_are_provided_by_the_pool_venv() -> None:
    gaps = ic.pool_gaps(ic.import_closure(REPO, ["webapp.server"]), _provided())
    assert gaps == {}, gaps


def test_cli_reports_gaps_with_exit_status(tmp_path: Path) -> None:
    _harness_tree(tmp_path)
    bare = tmp_path / "bare.txt"
    bare.write_text("numpy==2.2.6\n", encoding="utf-8")
    _provided(bare)
    result = subprocess.run(
        [sys.executable, "-m", "tools.pool.provisioning.import_closure", "--root", str(tmp_path),
         "--requirements", str(bare), "research.runners.chat_gate"],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 1 and '"httpx"' in result.stdout, result.stdout + result.stderr
