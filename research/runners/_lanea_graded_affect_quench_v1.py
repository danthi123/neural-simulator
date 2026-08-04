"""Locked Lane A graded-affect plus active-clear experiment.

The runner reuses the validated opponent affect circuit and the validated
``quench_fs`` GABA-A limb on one ``EvictionAffectBrain`` bridge. It does not
alter the simulation core. Diagnostic seeds select one recurrent-gain point;
formal seeds are rejected until that create-only diagnostic aggregate exists.

Unit tests exercise only the pure scoring and file-contract helpers. Importing
this module or running its tests does not construct a brain or consume a
reserved scientific seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO / "research" / "specs" / "lanea_graded_affect_quench_v1.json"
SOURCE_ANCHOR_PARENT_SHA = "4c8b0460040f32125064323bb7e30e5627d6f10d"
DIRECT_SOURCE_PATHS = (
    "research/runners/_lanea_graded_affect_quench_v1.py",
    "research/runners/_affect_eviction_derisk.py",
    "research/runners/_affect_state_region_derisk.py",
    "research/specs/lanea_graded_affect_quench_v1.json",
)


class ProtocolError(RuntimeError):
    """The locked execution or artifact contract was not satisfied."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_for(material: str) -> tuple[int, str]:
    prefix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return 3_000_000 + (int(prefix, 16) % 6_000_000), prefix


def load_spec(path: Path = DEFAULT_SPEC) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    spec = json.loads(raw)
    if spec.get("status") != "locked" or spec.get("id") != "laneA-graded-affect-quench-v1":
        raise ProtocolError("spec is not the locked Lane A graded-affect protocol")

    template = spec["seed_derivation"]["material_template"]
    for role in ("diagnostic", "formal"):
        seeds = spec["seeds"][role]
        prefixes = spec["seed_sha256_prefixes"][role]
        if len(seeds) != len(prefixes) or len(set(seeds)) != len(seeds):
            raise ProtocolError(f"invalid {role} seed partition")
        for index, (expected_seed, expected_prefix) in enumerate(zip(seeds, prefixes)):
            seed, prefix = _seed_for(template.format(role=role, index=index))
            if seed != expected_seed or prefix != expected_prefix:
                raise ProtocolError(f"{role} seed {index} does not match its locked derivation")
    if set(spec["seeds"]["diagnostic"]) & set(spec["seeds"]["formal"]):
        raise ProtocolError("diagnostic and formal seed partitions overlap")
    return spec, _sha256_bytes(raw)


def write_create_only(path: Path, value: Any) -> str:
    """Write canonical JSON without ever replacing an existing destination."""
    data = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return _sha256_bytes(data)


def _pearson(a: Iterable[float], b: Iterable[float]) -> float:
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.size < 3 or bv.size != av.size or av.std() < 1e-12 or bv.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def _rates(brain: Any, counts: dict[str, float], steps: int, baseline: float) -> float:
    return float(brain.mood_rate(counts, steps) - baseline)


def _reset_and_baseline(brain: Any, protocol: dict[str, Any]) -> float:
    brain.reset()
    brain.set_quench_drive(0.0)
    brain.step(protocol["settle_ms"])
    n = protocol["baseline_read_ms"]
    return float(brain.mood_rate(brain.step(n), n))


def _level_probe(brain: Any, level: float, protocol: dict[str, Any]) -> dict[str, float]:
    baseline = _reset_and_baseline(brain, protocol)
    vp, vm = max(level, 0.0), max(-level, 0.0)
    drive_ms = protocol["drive_ms"]
    peak = _rates(brain, brain.step(drive_ms, vp=vp, vm=vm, ar=0.5), drive_ms, baseline)
    brain.step(protocol["post_drive_ms"])
    read_ms = protocol["read_ms"]
    held = _rates(brain, brain.step(read_ms), read_ms, baseline)
    oriented_peak = np.sign(level) * peak
    oriented_held = np.sign(level) * held
    retention = oriented_held / oriented_peak if oriented_peak > 1e-9 else 0.0
    return {"level": float(level), "peak": peak, "held": held, "retention": float(retention)}


def _sign_crossing_probe(brain: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    baseline = _reset_and_baseline(brain, protocol)
    levels = np.asarray(protocol["sign_crossing_levels"], dtype=float)
    event_ms = int(protocol["sign_event_ms"])
    moods = []
    for level in levels:
        counts = brain.step(event_ms, vp=max(float(level), 0.0), vm=max(float(-level), 0.0), ar=0.5)
        moods.append(_rates(brain, counts, event_ms, baseline))
    moods_a = np.asarray(moods, dtype=float)
    nonzero = np.abs(levels) > 1e-12
    sign_accuracy = float(np.mean(np.sign(moods_a[nonzero]) == np.sign(levels[nonzero])))
    extreme = float(np.mean(np.abs(moods_a[np.abs(levels) >= 0.75])))
    zero = float(np.mean(np.abs(moods_a[np.abs(levels) < 1e-12])))
    zero_fraction = zero / extreme if extreme > 1e-12 else float("inf")
    dynamic_range = float(np.ptp(moods_a))
    largest_step_fraction = (float(np.max(np.abs(np.diff(moods_a)))) / dynamic_range
                             if dynamic_range > 1e-12 else float("inf"))
    deadband = max(1e-6, 0.1 * extreme)
    signs = [int(np.sign(x)) for x in moods_a if abs(x) > deadband]
    crossings = sum(a != b for a, b in zip(signs, signs[1:]))
    return {
        "levels": levels.tolist(),
        "moods": moods_a.tolist(),
        "pearson": _pearson(levels, moods_a),
        "sign_accuracy": sign_accuracy,
        "crossing_count": int(crossings),
        "zero_band_fraction": float(zero_fraction),
        "largest_step_fraction": float(largest_step_fraction),
        "dynamic_range": dynamic_range,
    }


def _set_quench_lesion(brain: Any, lesion: bool) -> None:
    from research.runners._affect_eviction_derisk import QUENCH_GATE

    brain._bridge.set_transmission_gate(QUENCH_GATE, 0.0 if lesion else 1.0)


def _clear_probe(brain: Any, protocol: dict[str, Any], *, lesion: bool) -> dict[str, float]:
    baseline = _reset_and_baseline(brain, protocol)
    _set_quench_lesion(brain, lesion)
    drive_ms = protocol["drive_ms"]
    brain.step(drive_ms, vp=0.8, ar=0.5)
    brain.step(protocol["post_drive_ms"])
    read_ms = protocol["read_ms"]
    pre = _rates(brain, brain.step(read_ms), read_ms, baseline)

    quench_ms = int(protocol["quench_duration_ms"])
    clear_counts = brain.step_brain_quench(quench_ms)
    fs_during = float(brain.quench_fs_rate(clear_counts, quench_ms))
    read_counts = brain.step(read_ms, record=("affect_vplus", "affect_vminus", "quench_fs"))
    post = _rates(brain, read_counts, read_ms, baseline)
    fs_read = float(brain.quench_fs_rate(read_counts, read_ms))
    drive_at_read = float(brain.quench_drive_conc())

    level = float(protocol["reignite_level"])
    brain.step(drive_ms, vp=max(level, 0.0), vm=max(-level, 0.0), ar=0.5)
    brain.step(protocol["post_drive_ms"])
    reignited = _rates(brain, brain.step(read_ms), read_ms, baseline)
    denom = abs(pre)
    return {
        "pre_clear": pre,
        "post_clear": post,
        "residual_ratio": abs(post) / denom if denom > 1e-9 else float("inf"),
        "reignited": reignited,
        "reignition_ratio": abs(reignited) / denom if denom > 1e-9 else 0.0,
        "quench_fs_rate_during": fs_during,
        "quench_fs_rate_at_read": fs_read,
        "quench_drive_at_read": drive_at_read,
    }


def _new_brain(seed: int, recurrent_weight: float, spec: dict[str, Any], *, nmda_on: bool) -> Any:
    from research.runners._affect_eviction_derisk import EvictionAffectBrain

    bridge = spec["bridge"]
    return EvictionAffectBrain(
        seed,
        nmda_on=nmda_on,
        recur_weight=recurrent_weight,
        ou_pA=bridge["ou_current_pA"],
        enable_gabab=False,
        with_eviction_wiring=False,
        brain_quench=True,
        quench_fs_n=bridge["quench_fs_neurons"],
        quench_gaba_w=bridge["quench_gaba_a_weight"],
        quench_drive_pA=bridge["quench_drive_pA"],
    )


def run_candidate(seed: int, recurrent_weight: float, spec: dict[str, Any]) -> dict[str, Any]:
    protocol = dict(spec["protocol"])
    protocol["quench_duration_ms"] = spec["bridge"]["quench_duration_ms"]
    brain = _new_brain(seed, recurrent_weight, spec, nmda_on=True)

    persistence = [_level_probe(brain, level, protocol) for level in protocol["persistence_levels"]]
    magnitude = [_level_probe(brain, level, protocol) for level in protocol["magnitude_levels"]]
    abs_levels = np.abs([row["level"] for row in magnitude])
    abs_held = np.abs([row["held"] for row in magnitude])
    low = abs_held[abs_levels <= 0.25]
    high = abs_held[abs_levels >= 0.75]
    polarity = float(np.mean([np.sign(row["held"]) == np.sign(row["level"]) for row in magnitude]))

    sign_crossing = _sign_crossing_probe(brain, protocol)
    clear_intact = _clear_probe(brain, protocol, lesion=False)
    clear_lesion = _clear_probe(brain, protocol, lesion=True)

    nmda_off = _new_brain(seed, recurrent_weight, spec, nmda_on=False)
    nmda_off_rows = [_level_probe(nmda_off, level, protocol) for level in protocol["persistence_levels"]]
    metrics = {
        "persistence_each_sign": [row["retention"] for row in persistence],
        "persistence_mean": float(np.mean([row["retention"] for row in persistence])),
        "nmda_off_persistence_max": float(max(row["retention"] for row in nmda_off_rows)),
        "magnitude_pearson": _pearson(abs_levels, abs_held),
        "magnitude_span": float(np.mean(high) - np.mean(low)),
        "magnitude_polarity_accuracy": polarity,
        "sign_crossing_pearson": sign_crossing["pearson"],
        "sign_accuracy": sign_crossing["sign_accuracy"],
        "sign_crossing_count": sign_crossing["crossing_count"],
        "zero_band_fraction": sign_crossing["zero_band_fraction"],
        "largest_step_fraction": sign_crossing["largest_step_fraction"],
        "eviction_ratio": clear_intact["residual_ratio"],
        "reignition_ratio": clear_intact["reignition_ratio"],
        "quench_lesion_residual": clear_lesion["residual_ratio"],
        "quench_lesion_gap": clear_lesion["residual_ratio"] - clear_intact["residual_ratio"],
        "quench_fs_during_rate": min(clear_intact["quench_fs_rate_during"],
                                      clear_lesion["quench_fs_rate_during"]),
        "quench_fs_read_rate": max(clear_intact["quench_fs_rate_at_read"],
                                    clear_lesion["quench_fs_rate_at_read"]),
        "quench_drive_at_read": max(abs(clear_intact["quench_drive_at_read"]),
                                    abs(clear_lesion["quench_drive_at_read"])),
    }
    checks = evaluate_metrics(metrics, spec["thresholds"])
    return {
        "recurrent_weight": float(recurrent_weight),
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "traces": {
            "persistence": persistence,
            "magnitude": magnitude,
            "sign_crossing": sign_crossing,
            "clear_intact": clear_intact,
            "clear_lesion": clear_lesion,
            "nmda_off_persistence": nmda_off_rows,
        },
    }


def evaluate_metrics(m: dict[str, float], t: dict[str, float]) -> dict[str, bool]:
    return {
        "persistence_each_sign": min(m["persistence_each_sign"]) >= t["persistence_min_each_sign"],
        "persistence_mean": m["persistence_mean"] >= t["persistence_mean_min"],
        "nmda_off_loses_persistence": m["nmda_off_persistence_max"] <= t["nmda_off_persistence_max"],
        "magnitude_tracks": m["magnitude_pearson"] >= t["magnitude_pearson_min"],
        "magnitude_spans": m["magnitude_span"] >= t["magnitude_span_min"],
        "magnitude_polarity": m["magnitude_polarity_accuracy"] >= t["magnitude_polarity_accuracy_min"],
        "sign_crossing_tracks": m["sign_crossing_pearson"] >= t["sign_crossing_pearson_min"],
        "sign_accuracy": m["sign_accuracy"] >= t["sign_accuracy_min"],
        "crosses_both_directions": m["sign_crossing_count"] >= t["sign_crossing_count_min"],
        "neutral_near_zero": m["zero_band_fraction"] <= t["zero_band_fraction_max"],
        "no_latch_jump": m["largest_step_fraction"] <= t["largest_step_fraction_max"],
        "active_clear_evicts": m["eviction_ratio"] <= t["eviction_ratio_max"],
        "reignites": m["reignition_ratio"] >= t["reignition_ratio_min"],
        "quench_lesion_restores_state": m["quench_lesion_residual"] >= t["quench_lesion_residual_min"],
        "quench_lesion_is_load_bearing": m["quench_lesion_gap"] >= t["quench_lesion_gap_min"],
        "quench_fs_fires_during_clear": m["quench_fs_during_rate"] >= t["quench_fs_during_rate_min"],
        "quench_fs_quiet_at_read": m["quench_fs_read_rate"] <= t["quench_fs_read_rate_max"],
        "quench_drive_zero_at_read": m["quench_drive_at_read"] <= t["quench_drive_at_read_max"],
    }


def normalized_margin(m: dict[str, Any], t: dict[str, float]) -> float:
    floors = [
        (min(m["persistence_each_sign"]), t["persistence_min_each_sign"]),
        (m["persistence_mean"], t["persistence_mean_min"]),
        (m["magnitude_pearson"], t["magnitude_pearson_min"]),
        (m["magnitude_span"], t["magnitude_span_min"]),
        (m["magnitude_polarity_accuracy"], t["magnitude_polarity_accuracy_min"]),
        (m["sign_crossing_pearson"], t["sign_crossing_pearson_min"]),
        (m["sign_accuracy"], t["sign_accuracy_min"]),
        (m["sign_crossing_count"], t["sign_crossing_count_min"]),
        (m["reignition_ratio"], t["reignition_ratio_min"]),
        (m["quench_lesion_residual"], t["quench_lesion_residual_min"]),
        (m["quench_lesion_gap"], t["quench_lesion_gap_min"]),
        (m["quench_fs_during_rate"], t["quench_fs_during_rate_min"]),
    ]
    caps = [
        (m["nmda_off_persistence_max"], t["nmda_off_persistence_max"]),
        (m["zero_band_fraction"], t["zero_band_fraction_max"]),
        (m["largest_step_fraction"], t["largest_step_fraction_max"]),
        (m["eviction_ratio"], t["eviction_ratio_max"]),
        (m["quench_fs_read_rate"], t["quench_fs_read_rate_max"]),
        (m["quench_drive_at_read"], t["quench_drive_at_read_max"]),
    ]
    margins = [value / threshold - 1.0 for value, threshold in floors]
    margins.extend(1.0 - value / threshold for value, threshold in caps)
    return float(min(margins))


def select_diagnostic(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    expected = set(spec["seeds"]["diagnostic"])
    if {row["seed"] for row in rows} != expected or len(rows) != len(expected):
        raise ProtocolError("diagnostic aggregate requires every unique locked diagnostic seed")
    ladder = spec["diagnostic"]["recurrent_weight_ladder"]
    thresholds = spec["thresholds"]
    eligible = []
    for weight in ladder:
        cells = []
        for row in rows:
            matches = [cell for cell in row["candidates"] if cell["recurrent_weight"] == weight]
            if len(matches) != 1:
                raise ProtocolError(f"diagnostic seed is missing unique weight {weight}")
            cells.append(matches[0])
        if all(cell["pass"] and all(cell["checks"].values()) for cell in cells):
            eligible.append((min(normalized_margin(cell["metrics"], thresholds) for cell in cells), weight))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    return {
        "selected_recurrent_weight": eligible[0][1] if eligible else None,
        "selection_status": "SELECTED_FOR_FORMAL" if eligible else "NO_SELECTION_FORMAL_REMAINS_SEALED",
        "eligible": [{"recurrent_weight": weight, "worst_normalized_margin": margin}
                     for margin, weight in eligible],
    }


def formal_verdict(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    expected = set(spec["seeds"]["formal"])
    if {row["seed"] for row in rows} != expected or len(rows) != len(expected):
        raise ProtocolError("formal aggregate requires every unique locked formal seed")
    selected = {row["selected_recurrent_weight"] for row in rows}
    if len(selected) != 1:
        raise ProtocolError("formal rows do not share one diagnostic-selected recurrent weight")
    passing = sum(bool(row["candidate"]["pass"] and all(row["candidate"]["checks"].values())) for row in rows)
    required = int(spec["formal"]["minimum_passing_seeds"])
    return {
        "verdict": "FORMAL_GO" if passing >= required else "FORMAL_NO_GO",
        "passing_seeds": passing,
        "required_passing_seeds": required,
        "selected_recurrent_weight": next(iter(selected)),
    }


def _git_source_state() -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_ANCHOR_PARENT_SHA, revision],
        cwd=REPO,
        capture_output=True,
    ).returncode != 0:
        raise ProtocolError("scientific source is not descended from the locked anchor")
    sim_paths = subprocess.check_output(
        ["git", "ls-files", "sim"], cwd=REPO, text=True
    ).splitlines()
    source_paths = [*sim_paths, *DIRECT_SOURCE_PATHS]
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal", "--", *source_paths],
        cwd=REPO,
        text=True,
    ).strip()
    if tracked:
        raise ProtocolError("scientific source paths are dirty; execution is undefined")
    identities = {
        relative: _sha256_bytes((REPO / relative).read_bytes()) for relative in source_paths
    }
    return {
        "git_revision": revision,
        "source_anchor_parent_sha": SOURCE_ANCHOR_PARENT_SHA,
        "source_path_count": len(identities),
        "source_identity": identities,
        "source_paths_clean": True,
    }


def _artifact_path(spec: dict[str, Any], phase: str, seed: int | None = None) -> Path:
    root = REPO / spec["output_root"] / phase
    return root / ("aggregate.json" if seed is None else f"seed-{seed}.json")


def _read_phase_rows(spec: dict[str, Any], spec_sha: str, phase: str) -> list[dict[str, Any]]:
    rows = []
    for seed in spec["seeds"][phase]:
        path = _artifact_path(spec, phase, seed)
        if not path.is_file():
            raise ProtocolError(f"missing {phase} artifact for seed {seed}")
        row = json.loads(path.read_text())
        if row.get("phase") != phase or row.get("seed") != seed or row.get("spec_sha256") != spec_sha:
            raise ProtocolError(f"invalid {phase} artifact for seed {seed}")
        rows.append(row)
    return rows


def validate_diagnostic_aggregate(spec: dict[str, Any], spec_sha: str) -> tuple[dict[str, Any], str]:
    path = _artifact_path(spec, "diagnostic")
    if not path.is_file():
        raise ProtocolError("formal phase is sealed until diagnostic aggregate exists")
    aggregate = json.loads(path.read_text())
    if aggregate.get("phase") != "diagnostic" or aggregate.get("spec_sha256") != spec_sha:
        raise ProtocolError("diagnostic aggregate does not match the locked protocol")

    rows = _read_phase_rows(spec, spec_sha, "diagnostic")
    recomputed = select_diagnostic(rows, spec)
    for field in ("selected_recurrent_weight", "selection_status", "eligible"):
        if aggregate.get(field) != recomputed[field]:
            raise ProtocolError(f"diagnostic aggregate field {field} does not match sealed inputs")
    expected_hashes = {
        str(row["seed"]): _sha256_bytes(_artifact_path(spec, "diagnostic", row["seed"]).read_bytes())
        for row in rows
    }
    if aggregate.get("input_artifact_sha256") != expected_hashes:
        raise ProtocolError("diagnostic aggregate input digests do not match sealed inputs")
    return aggregate, _sha256_bytes(path.read_bytes())


def execute_seed(spec: dict[str, Any], spec_sha: str, phase: str, seed: int) -> Path:
    if seed not in spec["seeds"][phase]:
        raise ProtocolError(f"seed {seed} is not assigned to phase {phase}")
    source = _git_source_state()
    if phase == "diagnostic":
        candidates = [run_candidate(seed, weight, spec)
                      for weight in spec["diagnostic"]["recurrent_weight_ladder"]]
        payload = {"schema_version": 1, "phase": phase, "seed": seed, "spec_sha256": spec_sha,
                   "source": source, "promotion_value": "none", "candidates": candidates}
    else:
        diagnostic, diagnostic_sha = validate_diagnostic_aggregate(spec, spec_sha)
        if diagnostic.get("selected_recurrent_weight") is None:
            raise ProtocolError("diagnostic did not select a valid formal operating point")
        weight = float(diagnostic["selected_recurrent_weight"])
        payload = {"schema_version": 1, "phase": phase, "seed": seed, "spec_sha256": spec_sha,
                   "source": source, "selected_recurrent_weight": weight,
                   "diagnostic_aggregate_sha256": diagnostic_sha,
                   "candidate": run_candidate(seed, weight, spec)}
    path = _artifact_path(spec, phase, seed)
    write_create_only(path, payload)
    return path


def aggregate_phase(spec: dict[str, Any], spec_sha: str, phase: str) -> Path:
    source = _git_source_state()
    rows = _read_phase_rows(spec, spec_sha, phase)
    if phase == "diagnostic":
        result = select_diagnostic(rows, spec)
    else:
        result = formal_verdict(rows, spec)
    payload = {"schema_version": 1, "phase": phase, "spec_sha256": spec_sha, "source": source,
               "input_artifact_sha256": {
                   str(row["seed"]): _sha256_bytes(_artifact_path(spec, phase, row["seed"]).read_bytes())
                   for row in rows}, **result}
    path = _artifact_path(spec, phase)
    write_create_only(path, payload)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--phase", required=True, choices=("diagnostic", "formal"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)

    spec, spec_sha = load_spec(args.spec)
    path = (aggregate_phase(spec, spec_sha, args.phase) if args.aggregate
            else execute_seed(spec, spec_sha, args.phase, args.seed))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
