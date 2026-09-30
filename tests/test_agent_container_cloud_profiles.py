"""Cloud mechanical-tier profiles (owner-approved 2026-09-29: "a cheap cloud model for the mechanical tier",
docs/plans/2026-09-29-local-model-transition-PLAN.md section 9) -- tools/agent_container/profiles/deepseek-flash.env
and glm-lite.env. tools/agent_container/bin/llm (the in-container profile loader) has no schema validation of
its own beyond "BASE_URL and MODEL are non-empty, else refuse at `llm use` time"; this pins the stronger
contract these two cloud profiles specifically must hold, so a future edit cannot silently reintroduce the
2026-09-29 Ollama bake-off crash class (a KIND=cloud profile with no auth configured sending an empty token).

Hermetic: pure file/text parsing, no container, no network, no real API key ever read or required.
"""
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROFILE_DIR = os.path.join(ROOT, "tools", "agent_container", "profiles")
CLOUD_PROFILES = ["deepseek-flash", "glm-lite"]


def _parse_env(path):
    """A minimal, shell-agnostic parser for these files' plain `KEY=value` lines (no quoting/expansion needed
    here -- every value in a profile file is a bare URL, model id, or identifier)."""
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_both_cloud_profiles_exist():
    for name in CLOUD_PROFILES:
        assert os.path.isfile(os.path.join(PROFILE_DIR, name + ".env")), "missing tools/agent_container/profiles/%s.env" % name


def test_cloud_profiles_are_anthropic_endpoints_with_a_named_token_var():
    for name in CLOUD_PROFILES:
        cfg = _parse_env(os.path.join(PROFILE_DIR, name + ".env"))
        assert cfg.get("BASE_URL", "").startswith("https://"), (name, cfg.get("BASE_URL"))
        assert "anthropic" in cfg["BASE_URL"].lower(), "%s's BASE_URL does not look like an Anthropic-compatible endpoint: %r" % (name, cfg["BASE_URL"])
        assert cfg.get("MODEL"), "%s has no MODEL" % name
        assert cfg.get("KIND") == "cloud", "%s must be KIND=cloud (it does not depend on the workstation GPU)" % name
        # A cloud profile with no TOKEN_VAR would send tools/agent_container/bin/llm's dummy "local" token to a
        # real paid API -- always require a named secret var for these two specifically.
        assert cfg.get("TOKEN_VAR"), "%s (KIND=cloud) must name a TOKEN_VAR" % name


def test_cloud_profiles_never_hold_a_literal_secret_value():
    # TOKEN_VAR must be a bare NAME (tools/agent_container/bin/llm reads it as ${!TOKEN_VAR}), never something
    # that looks like an actual key/token pasted into the tracked file.
    for name in CLOUD_PROFILES:
        cfg = _parse_env(os.path.join(PROFILE_DIR, name + ".env"))
        token_var = cfg["TOKEN_VAR"]
        assert re.match(r"^[A-Z_][A-Z0-9_]*$", token_var), "%s's TOKEN_VAR does not look like a bare env var name: %r" % (name, token_var)
        assert len(token_var) < 40 and "-" not in token_var and "." not in token_var


def test_deepseek_and_glm_use_different_token_vars():
    # each provider's key must land in its OWN env var -- sharing one would make it impossible to set both.
    a = _parse_env(os.path.join(PROFILE_DIR, "deepseek-flash.env"))
    b = _parse_env(os.path.join(PROFILE_DIR, "glm-lite.env"))
    assert a["TOKEN_VAR"] != b["TOKEN_VAR"]


def test_compose_yaml_has_a_matching_commented_secret_line_for_each():
    compose = open(os.path.join(ROOT, "tools", "agent_container", "compose.yaml")).read()
    for name in CLOUD_PROFILES:
        cfg = _parse_env(os.path.join(PROFILE_DIR, name + ".env"))
        assert cfg["TOKEN_VAR"] in compose, (
            "compose.yaml has no (commented) example line for %s's TOKEN_VAR=%s -- the owner would have no "
            "template to uncomment when wiring the real key in" % (name, cfg["TOKEN_VAR"]))
