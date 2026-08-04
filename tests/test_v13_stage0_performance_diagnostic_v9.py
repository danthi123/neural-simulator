import json
from pathlib import Path

from tools.v13_stage0_performance_diagnostic_v9 import (
    CELL_DEFINITIONS,
    build_run_plan,
    summarize,
)


def test_run_plan_is_deterministic_and_covers_every_cell_rep():
    first = build_run_plan(repetitions=3, order_seed=20260804)
    second = build_run_plan(repetitions=3, order_seed=20260804)
    assert first == second
    assert len(first) == 12
    assert {(row["cell"], row["rep"]) for row in first} == {
        (cell, rep) for cell in CELL_DEFINITIONS for rep in (1, 2, 3)
    }
    assert [row["sequence"] for row in first] == list(range(1, 13))


def test_summary_is_descriptive_and_does_not_promote_sealed_no_go():
    rows = []
    for cell in CELL_DEFINITIONS:
        for rep in (1, 2, 3):
            rows.append({
                "cell": cell,
                "status": "completed",
                "timing": {
                    "wall_seconds": 6.0 if cell in ("A", "C") else 5.8,
                    "cuda_seconds": 5.0,
                    "host_minus_device_seconds": 1.0,
                },
                "structural": {
                    "intrinsic_is_none": True,
                    "external_current_exact_zero": True,
                    "dispatch_preflight": False,
                    "trace": {"megakernel_dispatch_true": 0},
                },
            })
    summary = summarize(rows)
    assert summary["cells"]["A"]["completed_repetitions"] == 3
    assert summary["cells"]["A"]["all_structural_checks_pass"] is True
    assert summary["sealed_v8_boundary"] == "unchanged PERFORMANCE_NO_GO"
    assert summary["stage1_seed_1031"] == "sealed-not-read-or-executed"
    assert "automatic gate verdict" in summary["interpretation_status"]


def test_spec_declares_process_only_and_fixed_boundary():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "research/specs/v13_stage0_performance_diagnostic_v9.json").read_text()
    )
    assert spec["status"] == "preregistered"
    assert spec["mechanism"] == "none-process-only"
    assert spec["execution"]["fixed_v8_gate_is_not_replaced"] is True
    assert spec["execution"]["stage1_seed_1031_remains_sealed"] is True
