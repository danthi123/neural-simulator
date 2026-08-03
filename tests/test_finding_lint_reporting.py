from tools.finding_lint import _blocking_gate_status


def test_blocking_gate_status_reads_registry_wrapper():
    wrapped = {
        "name": "example",
        "gate": {"status": "crashed"},
    }
    assert _blocking_gate_status(wrapped) == "crashed"


def test_blocking_gate_status_tolerates_non_registry_blocker():
    assert _blocking_gate_status({"name": "claim-check"}) is None
