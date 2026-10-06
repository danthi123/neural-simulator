"""tools/biology_check.py resolves a source written as ~/Projects/<folder>/... next to the main checkout, whatever
HOME is. 2026-09-30: inside the Hermes container on homelab-server HOME is /opt/data/home, so plain ~ expansion
missed ~/Projects/sim-catalog there and the biology gate blocked every commit (FAILURE_LOG)."""
import importlib.util
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("biology_check_under_test", os.path.join(_ROOT, "tools", "biology_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_projects_sources_resolve_next_to_the_main_checkout_whatever_home_is(monkeypatch, tmp_path):
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    main_checkout = os.path.dirname(common)
    monkeypatch.setenv("HOME", str(tmp_path))          # a HOME with no Projects folder, like the container's
    bc = _load()
    # the main checkout is always a sibling of itself, so this path exists on every machine
    p = "~/Projects/%s/CLAUDE.md" % os.path.basename(main_checkout)
    assert bc._expand(p) == os.path.join(main_checkout, "CLAUDE.md")


def test_other_home_paths_still_expand_against_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    bc = _load()
    assert bc._expand("~/somewhere/else.txt") == os.path.join(str(tmp_path), "somewhere", "else.txt")
