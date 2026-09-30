"""llama-swap replaced a direct `llama-server` launch as the "local-llm" unit's own command (2026-09-29 serving
simplification, research/serving-bakeoff/tools/local_llm/results/summary.md: llama-server still won the bake-off
at 74 tok/s / 3/3 tasks; llama-swap only fronts it so tools/local_llm/profiles.json's other entries share one
config, an idle one can auto-unload, and the frontend can serve the LAN as well as loopback).

THE DRIFT THIS GUARDS (owner requirement: "reuse llm.sh's profile_cmd logic; do not change the flags"). Every
`models.*.cmd` in tools/local_llm/llama-swap.yaml is supposed to be BYTE-IDENTICAL, flag for flag, to what
`llm.sh`'s own profile_cmd() builds for that profile -- the same function `llm on <profile>` always used, before
llama-swap existed and still today (llama-swap.yaml's `cmd:` values were authored by literally running
`LLM_PORT=@@PORT@@ bash tools/local_llm/llm.sh __profile_cmd <name>` and pasting the result). Nothing enforces
that at edit time: a future flag change to profile_cmd() (e.g. the -np/-kvu prompt-cache tuning documented right
in that function) would silently stop matching the YAML, and `llm on` / llama-swap would then serve two
DIFFERENT command lines for what looks like the same profile. This test regenerates the expected command through
profile_cmd() itself (never re-implements the flag list) and fails loudly on any mismatch.

Hermetic: never starts a real llama-server or llama-swap process. `__profile_cmd` only PRINTS the command line;
`-validate` only parses+checks the YAML. Both run against this checkout's own files.
"""
import json
import os
import shlex
import subprocess

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM_SH = os.path.join(ROOT, "tools", "local_llm", "llm.sh")
YAML_PATH = os.path.join(ROOT, "tools", "local_llm", "llama-swap.yaml")
PROFILES_JSON = os.path.join(ROOT, "tools", "local_llm", "profiles.json")
PORT_TOKEN = "@@PORT@@"   # shlex-safe (word chars + '@' only), so profile_cmd()'s shlex.join never quotes it


def _load_yaml():
    return yaml.safe_load(open(YAML_PATH))


def _load_profiles():
    return json.load(open(PROFILES_JSON))


def _expected_cmd(name):
    """The command profile_cmd() builds for `name`, with the real numeric port swapped for llama-swap's own
    ${PORT} macro (the one intentional difference: --port 8080 only made sense when llm.sh launched
    llama-server directly on the shared port; llama-swap now owns 8080 and assigns each backend its own)."""
    out = subprocess.run(["bash", LLM_SH, "__profile_cmd", name],
                         env={**os.environ, "LLM_PORT": PORT_TOKEN}, capture_output=True, text=True, timeout=15)
    assert out.returncode == 0, "profile_cmd(%r) failed: %s" % (name, out.stdout + out.stderr)
    return out.stdout.strip().replace(PORT_TOKEN, "${PORT}")


def _actual_cmd(cfg, name):
    """The YAML's cmd for `name`, with its ${repo_root} macro resolved the same way llm.sh resolves it at
    launch time (LLAMA_SWAP_REPO_ROOT = this checkout's own root, whichever one -- main or a worktree -- is
    running the test), so the comparison is checkout-independent."""
    raw = cfg["models"][name]["cmd"]
    return " ".join(raw.split()).replace("${repo_root}", ROOT)


def test_every_profile_json_entry_has_a_model_and_a_profile_pin():
    cfg = _load_yaml()
    names = [p["name"] for p in _load_profiles()]
    for name in names:
        assert name in cfg["models"], "%r is in profiles.json but missing from llama-swap.yaml's models:" % name
        assert name in cfg["profiles"], "%r has no profiles: entry (needed so `llm on %s` can pin \"local\")" % (name, name)
        assert cfg["profiles"][name]["pins"].get("local") == name, (
            "profiles.%s.pins.local must point at itself: %r" % (name, cfg["profiles"][name]["pins"]))


def test_llama_swap_yaml_has_no_extra_models_beyond_profiles_json():
    # the inverse of the above: nothing in the YAML should silently diverge from profiles.json, the source of
    # truth `llm on <name>` reads. A stale/renamed entry here would be invisible to the owner otherwise.
    cfg = _load_yaml()
    names = {p["name"] for p in _load_profiles()}
    assert set(cfg["models"].keys()) == names, (set(cfg["models"].keys()), names)


def test_each_profile_cmd_is_byte_identical_to_llm_sh_profile_cmd():
    cfg = _load_yaml()
    for p in _load_profiles():
        name = p["name"]
        want = shlex.split(_expected_cmd(name))
        got = shlex.split(_actual_cmd(cfg, name))
        assert got == want, "tools/local_llm/llama-swap.yaml's cmd for %r has drifted from llm.sh's profile_cmd():\n  yaml: %s\n  llm.sh: %s" % (
            name, got, want)


def test_default_profile_matches_the_default_profile_file_and_startup_hook():
    default = open(os.path.join(ROOT, "tools", "local_llm", "default_profile")).read().strip()
    cfg = _load_yaml()
    assert cfg["hooks"]["on_startup"]["profile"] == default, (
        "llama-swap.yaml's hooks.on_startup.profile must track tools/local_llm/default_profile")
    assert default in cfg["models"]


def test_config_validates_with_the_real_llama_swap_binary_if_installed():
    # Best-effort: skip (never fail) when the operator binary is not present on this machine (e.g. a CI runner
    # that never installed it) -- the flag-identity tests above are the real guard and need no binary at all.
    llama_swap_bin = os.environ.get("LLAMA_SWAP_BIN") or os.path.join(
        os.path.expanduser("~"), ".local", "bin", "llama-swap")
    if not os.path.isfile(llama_swap_bin):
        import pytest
        pytest.skip("llama-swap binary not installed at %s" % llama_swap_bin)
    out = subprocess.run([llama_swap_bin, "-config", YAML_PATH, "-validate"],
                         env={**os.environ, "LLAMA_SWAP_REPO_ROOT": ROOT}, capture_output=True, text=True, timeout=15)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "config is valid" in out.stdout, out.stdout


def test_listen_and_ttl_settings_present():
    cfg = _load_yaml()
    assert cfg.get("globalTTL", 0) > 0, "globalTTL must be set (owner 2026-09-29: idle-unload the GPU automatically)"
    assert cfg.get("healthCheckTimeout", 0) >= 120, "a cold 27B GGUF load needs more than the 120s default"
