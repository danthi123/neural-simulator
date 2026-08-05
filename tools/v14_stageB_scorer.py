"""Source-bound raw-trace scorer for V14 SNr Stage B."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from tools.v14_stageB_physiology_metrics import peak_conductance, spike_train_metrics
from tools.v14_stageB_scorer_fixtures import StageBFixtureError, score_observation, validate_fixture


SCHEMA = "v14-snr-stageB-raw-observations-v1"
RESULT_SCHEMA = "v14-snr-stageB-score-v1"


class StageBScorerError(ValueError):
    """Raised when raw observations cannot support a Stage B score."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_bound_json(root: Path, declaration: Any, context: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(declaration, Mapping) or set(declaration) != {"path", "sha256"}:
        raise StageBScorerError(f"{context} must declare only path and sha256")
    relative = PurePosixPath(str(declaration.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise StageBScorerError(f"{context} path must be repository-relative")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBScorerError(f"{context} path escapes the repository") from exc
    try:
        data = path.read_bytes()
        value = json.loads(data)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBScorerError(f"cannot load {context}: {exc}") from exc
    if _digest_bytes(data) != declaration.get("sha256"):
        raise StageBScorerError(f"{context} digest does not match")
    if not isinstance(value, dict):
        raise StageBScorerError(f"{context} must contain a JSON object")
    return path, value


def _candidate_echo(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "candidate_id", "candidate_sha256", "effective_parameters"
    }:
        raise StageBScorerError("adaptive_candidate has an invalid shape")
    identifier = value.get("candidate_id")
    digest = value.get("candidate_sha256")
    parameters = value.get("effective_parameters")
    if (not isinstance(identifier, str) or not identifier or not isinstance(digest, str)
            or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(parameters, Mapping)):
        raise StageBScorerError("adaptive_candidate is malformed")
    return {"candidate_id": identifier, "candidate_sha256": digest,
            "effective_parameters": dict(parameters)}


def _spike_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    return spike_train_metrics(
        raw.get("spike_times_s", []),
        **{key: raw[key] for key in (
            "time_unit", "sample_interval_s", "recording_start_s", "recording_end_s",
            "burn_in_start_s", "burn_in_end_s", "window_start_s", "window_end_s",
        )},
    )


def _raw_value(fixture: Mapping[str, Any], raw: Any) -> tuple[float, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise StageBScorerError(f"fixture {fixture['id']!r} has no raw observation")
    kind = raw.get("kind")
    try:
        if kind == "spike_train":
            if fixture["units"] != "spikes/s":
                raise StageBScorerError("spike-train raw data cannot score this fixture unit")
            metrics = _spike_metrics(raw)
            return float(metrics["firing_rate_hz"]), metrics
        if kind == "paired_spike_rate_ratio":
            if fixture["units"] != "dimensionless ratio":
                raise StageBScorerError("paired spike-rate raw data cannot score this fixture unit")
            intact = raw.get("intact")
            lesion = raw.get("lesion")
            if not isinstance(intact, Mapping) or not isinstance(lesion, Mapping):
                raise StageBScorerError("paired spike-rate raw data requires intact and lesion traces")
            protocol_fields = (
                "time_unit", "sample_interval_s", "recording_start_s", "recording_end_s",
                "burn_in_start_s", "burn_in_end_s", "window_start_s", "window_end_s",
            )
            if any(intact.get(field) != lesion.get(field) for field in protocol_fields):
                raise StageBScorerError("intact and lesion spike-rate protocols must match exactly")
            intact_metrics = _spike_metrics(intact)
            lesion_metrics = _spike_metrics(lesion)
            intact_rate = float(intact_metrics["firing_rate_hz"])
            lesion_rate = float(lesion_metrics["firing_rate_hz"])
            ratio = lesion_rate / intact_rate if intact_rate > 0.0 else 0.0
            return ratio, {
                "intact": intact_metrics,
                "lesion": lesion_metrics,
                "lesion_over_intact": ratio,
                "persistent_intact_firing": intact_rate > 0.0,
                "persistent_lesion_firing": lesion_rate > 0.0,
            }
        if kind == "conductance_trace":
            if fixture["units"] != "nS":
                raise StageBScorerError("conductance raw data cannot score this fixture unit")
            metrics = peak_conductance(
                raw.get("time_s", []), raw.get("conductance_nS", []),
                **{key: raw[key] for key in (
                    "time_unit", "conductance_unit", "sample_interval_s", "recording_start_s",
                    "burn_in_start_s", "burn_in_end_s", "window_start_s", "window_end_s",
                )},
            )
            return float(metrics["peak_conductance_nS"]), metrics
    except KeyError as exc:
        raise StageBScorerError(f"fixture {fixture['id']!r} raw protocol is missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise StageBScorerError(f"fixture {fixture['id']!r} raw protocol is invalid: {exc}") from exc
    raise StageBScorerError(f"fixture {fixture['id']!r} has unsupported raw kind {kind!r}")


def score_raw_observations(document: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    """Recompute every bounded fixture observation from digest-bound raw data."""
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise StageBScorerError("raw observation document has the wrong schema")
    root_path = Path(root).resolve()
    candidate = _candidate_echo(document.get("adaptive_candidate"))
    fixture_path, packet = _load_bound_json(root_path, document.get("fixture_packet"), "fixture packet")
    if packet.get("schema") != "v14-snr-stageB-scorer-fixtures-v1":
        raise StageBScorerError("fixture packet has the wrong schema")
    _, target = _load_bound_json(root_path, packet.get("source_target_packet"), "source target packet")
    if target.get("schema") != "v14-snr-stageB-target-packet-v1":
        raise StageBScorerError("source target packet has the wrong schema")

    fixtures = packet.get("fixtures")
    raw_entries = document.get("observations")
    if not isinstance(fixtures, list) or not isinstance(raw_entries, list):
        raise StageBScorerError("fixtures and observations must be lists")
    contracts = [validate_fixture(item) for item in fixtures]
    bounded = {item["id"]: item for item in contracts if item["score_kind"] == "bounded-interval"}
    supplied: dict[str, Any] = {}
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or set(entry) != {"fixture_id", "raw"}:
            raise StageBScorerError("each raw observation must contain only fixture_id and raw")
        fixture_id = entry.get("fixture_id")
        if fixture_id not in bounded or fixture_id in supplied:
            raise StageBScorerError(f"raw observation has unknown or duplicate fixture {fixture_id!r}")
        supplied[str(fixture_id)] = entry["raw"]
    if set(supplied) != set(bounded):
        missing = sorted(set(bounded) - set(supplied))
        raise StageBScorerError(f"raw observations do not cover every bounded fixture: {missing}")

    results = []
    for fixture_id in sorted(bounded):
        fixture = bounded[fixture_id]
        value, metrics = _raw_value(fixture, supplied[fixture_id])
        scored = score_observation(fixture, {
            "cohort": fixture["cohort"], "pathway": fixture["pathway"],
            "metric": fixture["metric"], "units": fixture["units"], "value": value,
        })
        results.append({**scored, "raw_metrics": metrics})
    all_passed = all(item["passed"] is True for item in results)
    return {
        "schema": RESULT_SCHEMA,
        "process_status": "completed",
        "scientific_verdict": "GO" if all_passed else "NO_GO",
        "adaptive_candidate": candidate,
        "fixture_packet": {
            "path": fixture_path.relative_to(root_path).as_posix(),
            "sha256": _digest_bytes(fixture_path.read_bytes()),
        },
        "all_bounded_fixtures_passed": all_passed,
        "results": results,
        "unscored_boundaries": sorted(
            item["id"] for item in contracts if item["score_kind"] != "bounded-interval"
        ),
    }


def score_raw_observation_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Validate one raw artifact and create its result without overwriting evidence."""
    source = Path(input_path)
    destination = Path(output_path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBScorerError(f"cannot load raw observation file: {exc}") from exc
    result = score_raw_observations(document, root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="ascii") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise StageBScorerError(f"refusing to replace existing score: {destination}") from exc
    return result


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw observation JSON")
    parser.add_argument("--output", required=True, help="new score JSON")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    try:
        score_raw_observation_file(args.input, args.output, root=args.root)
    except StageBScorerError as exc:
        parser.exit(2, f"Stage B scorer infrastructure failure: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
