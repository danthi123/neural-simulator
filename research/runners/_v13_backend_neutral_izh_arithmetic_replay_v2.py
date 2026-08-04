"""Create-only V13 strict-arithmetic replay v2 after subnormal correction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from research.runners import _v13_backend_neutral_izh_arithmetic_replay as common


ROOT = common.ROOT
BACKENDS = common.BACKENDS
TRAJECTORIES = common.TRAJECTORIES
TOTAL_STEPS = common.TOTAL_STEPS
PROMOTION_VALUE = common.PROMOTION_VALUE
SPEC_RELATIVE_PATH = Path(
    "research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2.json"
)
SPEC_PATH = ROOT / SPEC_RELATIVE_PATH
SPEC_SHA256 = "94d96fb6a67e0d7df3d151e7b1dbbb85a2a9315115e1aa165f6f9355df6f0992"
SCHEMA_CELL = "v13-backend-neutral-izh-arithmetic-replay-cell-v2"
SCHEMA_COMPARISON = "v13-backend-neutral-izh-arithmetic-replay-comparison-v2"
OUTPUT_DIRECTORY = (
    "research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2"
)
_V1_ONLY_AUTHORITIES = {
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-"
    "izh-arithmetic-replay-DIAGNOSTIC-PREREGISTRATION.md",
    "research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic.json",
}
AUTHORITY_SOURCE_PATHS = tuple(
    path for path in common.AUTHORITY_SOURCE_PATHS if path not in _V1_ONLY_AUTHORITIES
) + (
    "research/runners/_v13_backend_neutral_izh_arithmetic_replay_v2.py",
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-"
    "izh-arithmetic-replay-v2-DIAGNOSTIC-PREREGISTRATION.md",
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-backend-neutral-"
    "izh-arithmetic-replay-v1-DIAGNOSTIC-RESULT.md",
    "research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/"
    "comparison.json",
    "research/findings/raw/v13_backend_neutral_izh_arithmetic_replay_diagnostic/"
    "evidence-manifest.json",
    "research/specs/v13_backend_neutral_izh_arithmetic_replay_diagnostic_v2.json",
    "tools/v13_backend_neutral_izh_arithmetic_replay_evidence_v2.py",
)
PROTOCOL = common.ReplayProtocol(
    spec_relative_path=SPEC_RELATIVE_PATH,
    spec_sha256=SPEC_SHA256,
    spec_id="gateB-v13-backend-neutral-izh-arithmetic-replay-diagnostic-v2",
    output_directory=OUTPUT_DIRECTORY,
    diagnostic_schema="v13-backend-neutral-izh-arithmetic-replay-spec-v2",
    runner_module="research.runners._v13_backend_neutral_izh_arithmetic_replay_v2",
    cell_schema=SCHEMA_CELL,
    comparison_schema=SCHEMA_COMPARISON,
    authority_source_paths=AUTHORITY_SOURCE_PATHS,
    enforce_output_directory=True,
)

transplant = common.transplant
execution_receipt = common.execution_receipt
_artifact_digest = common._artifact_digest
_digest_file = common._digest_file
_first_difference = common._first_difference
_forbid_rng_calls = common._forbid_rng_calls
_seal = common._seal


def source_paths(root: Path | None = None) -> tuple[str, ...]:
    return common.source_paths(root, protocol=PROTOCOL)


def load_locked_spec(
    path: Path | None = None, expected_sha256: str | None = None,
) -> dict[str, Any]:
    return common.load_locked_spec(
        path, expected_sha256, protocol=PROTOCOL,
    )


def load_completed_input(spec: dict[str, Any]):
    return common.load_completed_input(spec)


def run_cell(**kwargs):
    return common.run_cell(**kwargs, protocol=PROTOCOL)


def _expected_cell_argv(**kwargs) -> list[str]:
    return common._expected_cell_argv(**kwargs, protocol=PROTOCOL)


def _load_cell(path: Path, receipt_path: Path, backend: str) -> dict[str, Any]:
    return common._load_cell(path, receipt_path, backend, protocol=PROTOCOL)


def compare_cells(**kwargs):
    return common.compare_cells(**kwargs, protocol=PROTOCOL)


def main(argv: list[str] | None = None) -> int:
    return common.main(argv, protocol=PROTOCOL)


if __name__ == "__main__":
    raise SystemExit(main())
