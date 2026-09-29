"""Pins the local model's Claude Code permission rules (tools/local_llm/claude_local_settings.json, loaded by
`tools/local_llm/llm.sh claude`). FAILURE_LOG 2026-09-25: `tools/pool_provision.sh --revision <sha>` WITHOUT
`--isolated` provisions into the nodes' SHARED ~/derisk-pool/sim tree, and POOL_PROVISION_ALLOW_STALE=1 lets a stale
revision roll that tree back. The weekend local model may run only the isolated form and never the stale override;
the runbook's non-negotiables (no --no-verify, no force-push, no merge, no AWS, no sim/ webapp/ tools/gates/ edits)
are denied outright.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS = os.path.join(ROOT, "tools", "local_llm", "claude_local_settings.json")


def _rules():
    j = json.load(open(SETTINGS))
    return j["permissions"]["allow"], j["permissions"]["deny"]


def test_provisioning_is_only_allowed_in_the_isolated_form():
    allow, _ = _rules()
    prov = [r for r in allow if "pool_provision.sh" in r]
    assert prov, "the runbook's re-provision step must be pre-approved"
    for r in prov:
        assert "--isolated" in r, "a non-isolated provision rule would let the local model overwrite the nodes' shared tree: %r" % r


def test_the_stale_provision_override_is_denied():
    _, deny = _rules()
    assert any("POOL_PROVISION_ALLOW_STALE" in r for r in deny)


def test_runbook_non_negotiables_are_denied():
    _, deny = _rules()
    for needle in ("--no-verify", "push --force", "push -f", "reset --hard", "git merge", "Bash(aws", "Edit(sim/**)",
                   "Edit(webapp/**)", "Edit(tools/gates/**)"):
        assert any(needle in r for r in deny), "missing deny rule for %r" % needle


def test_file_rules_use_edit_not_write():
    # Claude Code only honours Edit(path) rules for file paths; a Write(path) rule is silently ignored (seen live
    # 2026-09-25), which would turn an intended deny into no rule at all.
    allow, deny = _rules()
    assert not [r for r in allow + deny if r.startswith("Write(")]


def test_gate_read_state_files_are_denied():
    # 2026-09-27: the local model registered two non-existent builds in lane_builds.jsonl to get past the
    # lane-starvation gate. Files a gate reads must not be editable by it.
    _, deny = _rules()
    for needle in ("lane_builds.jsonl", "pool.queue", "gpu.queue"):
        assert any(needle in r for r in deny), "missing deny rule for %r" % needle


def test_gpu_yield_and_pause_commands_are_pre_approved():
    # docs/plans/2026-09-29-local-model-transition-PLAN.md section 3/8: the driver may pause for gaming, yield the
    # GPU through `llm.sh yield`, and queue GPU jobs only through the checked tools/local_llm/gpu_add.sh wrapper.
    allow, _ = _rules()
    for needle in ("bash tools/game.sh on --graceful", "bash tools/game.sh status",
                   "bash tools/local_llm/llm.sh yield", "bash tools/local_llm/gpu_add.sh"):
        assert any(needle in r for r in allow), "missing allow rule for %r" % needle


def test_raw_gpu_queue_add_stays_denied():
    # The local model must go through the checked wrapper (tools/local_llm/gpu_add.sh), never the raw
    # tools/gpu_queue.sh add -- that is exactly what the wrapper's own commit-revision + gate-seed-prereg checks
    # exist to enforce, and they are bypassed entirely if the raw command is also allowed.
    _, deny = _rules()
    assert any("gpu_queue.sh add" in r for r in deny)
