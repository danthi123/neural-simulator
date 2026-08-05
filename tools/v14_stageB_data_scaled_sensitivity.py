"""Data-scaled calibration probes for source-centered Stage B coordinates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


TARGET_STANDARDIZED_RESPONSE = 1.0
INITIAL_STEP = math.sqrt(np.finfo(np.float64).eps)
MAX_BRACKET_EXPANSIONS = 64
MAX_BISECTIONS = 53
_RELATIVE_WIDTH_TOLERANCE = math.sqrt(np.finfo(np.float64).eps)


class DataScaledSensitivityError(ValueError):
    """Raised when a calibration-only sensitivity receipt is malformed."""


class DataScaledProbeController:
    """Bracket one uncertainty-scaled response per source contrast.

    Every round proposes the source baseline and centered +/- probes for all
    coordinates. The controller never sees validation or held-out targets and
    never interprets a probe magnitude as a biological parameter bound.
    """

    def __init__(self, coordinate_ids: Sequence[str], target_uncertainties: Any):
        if (
            isinstance(coordinate_ids, (str, bytes))
            or not isinstance(coordinate_ids, Sequence)
            or not coordinate_ids
            or any(not isinstance(item, str) or not item for item in coordinate_ids)
            or len(set(coordinate_ids)) != len(coordinate_ids)
        ):
            raise DataScaledSensitivityError("coordinate ids must be nonempty and unique")
        uncertainty = np.asarray(target_uncertainties, dtype=np.float64)
        if (
            uncertainty.ndim != 1
            or uncertainty.size == 0
            or not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty <= 0.0)
        ):
            raise DataScaledSensitivityError(
                "target uncertainties must be a nonempty positive finite vector"
            )
        self.coordinate_ids = tuple(coordinate_ids)
        self.target_uncertainties = uncertainty.copy()
        width = len(self.coordinate_ids)
        self.lower = np.zeros(width, dtype=np.float64)
        self.upper = np.full(width, np.nan, dtype=np.float64)
        self.trial = np.full(width, INITIAL_STEP, dtype=np.float64)
        self.expansions = np.zeros(width, dtype=np.int64)
        self.bisections = np.zeros(width, dtype=np.int64)
        self.status = np.full(width, "bracketing", dtype=object)
        self.last_response = np.full(width, np.nan, dtype=np.float64)

    @property
    def complete(self) -> bool:
        return bool(np.all(self.status != "bracketing") & np.all(self.status != "bisecting"))

    def probe_matrix(self) -> np.ndarray:
        """Return source, positive, and negative probes in deterministic order."""

        dimension = len(self.coordinate_ids)
        matrix = np.zeros((1 + 2 * dimension, dimension), dtype=np.float64)
        indices = np.arange(dimension)
        matrix[1 + 2 * indices, indices] = self.trial
        matrix[2 + 2 * indices, indices] = -self.trial
        return matrix

    def observe(self, predictions: Any, valid_mask: Any | None = None) -> np.ndarray:
        """Consume one calibration-only prediction matrix and update brackets."""

        values = np.asarray(predictions, dtype=np.float64)
        expected = (1 + 2 * len(self.coordinate_ids), self.target_uncertainties.size)
        if values.shape != expected:
            raise DataScaledSensitivityError(f"predictions must have shape {expected}")
        if valid_mask is None:
            valid = np.all(np.isfinite(values), axis=1)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != (expected[0],):
                raise DataScaledSensitivityError("valid mask shape is invalid")
            valid &= np.all(np.isfinite(values), axis=1)
        if not valid[0]:
            raise DataScaledSensitivityError("source baseline prediction is invalid")

        plus = values[1::2]
        minus = values[2::2]
        pair_valid = valid[1::2] & valid[2::2]
        centered = (plus - minus) * 0.5
        responses = np.sqrt(
            np.mean((centered / self.target_uncertainties[None, :]) ** 2, axis=1)
        )
        responses[~pair_valid] = np.nan
        self.last_response = responses.copy()

        for index in range(len(self.coordinate_ids)):
            if self.status[index] not in {"bracketing", "bisecting"}:
                continue
            if not pair_valid[index]:
                self.status[index] = "unresolved_invalid_before_target_response"
                continue
            response = responses[index]
            if not math.isfinite(float(response)):
                self.status[index] = "unresolved_nonfinite_response"
                continue
            if math.isnan(self.upper[index]):
                if response < TARGET_STANDARDIZED_RESPONSE:
                    self.lower[index] = self.trial[index]
                    self.expansions[index] += 1
                    if self.expansions[index] >= MAX_BRACKET_EXPANSIONS:
                        self.status[index] = "unresolved_no_target_response"
                    else:
                        self.trial[index] *= 2.0
                    continue
                self.upper[index] = self.trial[index]
                self.status[index] = "bisecting"
            else:
                if response < TARGET_STANDARDIZED_RESPONSE:
                    self.lower[index] = self.trial[index]
                else:
                    self.upper[index] = self.trial[index]
                self.bisections[index] += 1

            upper = self.upper[index]
            lower = self.lower[index]
            relative_width = (upper - lower) / max(upper, np.finfo(np.float64).tiny)
            if (
                relative_width <= _RELATIVE_WIDTH_TOLERANCE
                or self.bisections[index] >= MAX_BISECTIONS
            ):
                self.trial[index] = (lower + upper) * 0.5
                self.status[index] = "data_scaled"
            else:
                self.trial[index] = (lower + upper) * 0.5
        return responses

    def receipt(self) -> dict[str, Any]:
        """Return a JSON-compatible analysis receipt with no scientific verdict."""

        rows = []
        for index, identifier in enumerate(self.coordinate_ids):
            rows.append(
                {
                    "coordinate_id": identifier,
                    "status": str(self.status[index]),
                    "selected_step": (
                        float(self.trial[index]) if self.status[index] == "data_scaled" else None
                    ),
                    "selected_step_is_biological_bound": False,
                    "lower_response_bracket": float(self.lower[index]),
                    "upper_response_bracket": (
                        float(self.upper[index]) if math.isfinite(self.upper[index]) else None
                    ),
                    "expansions": int(self.expansions[index]),
                    "bisections": int(self.bisections[index]),
                    "last_standardized_response": (
                        float(self.last_response[index])
                        if math.isfinite(self.last_response[index]) else None
                    ),
                }
            )
        return {
            "schema": "v14-snr-stageB-data-scaled-sensitivity-receipt-v1",
            "status": "complete" if self.complete else "in_progress",
            "partition": "calibration",
            "target_standardized_response": TARGET_STANDARDIZED_RESPONSE,
            "scientific_verdict": None,
            "optimization_allowed": False,
            "coordinates": rows,
        }


def standardized_jacobian(
    predictions: Any,
    selected_steps: Any,
    target_uncertainties: Any,
) -> np.ndarray:
    """Return the centered, uncertainty-standardized local Jacobian."""

    steps = np.asarray(selected_steps, dtype=np.float64)
    uncertainty = np.asarray(target_uncertainties, dtype=np.float64)
    values = np.asarray(predictions, dtype=np.float64)
    expected = (1 + 2 * steps.size, uncertainty.size)
    if (
        steps.ndim != 1
        or steps.size == 0
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0.0)
        or uncertainty.ndim != 1
        or uncertainty.size == 0
        or not np.all(np.isfinite(uncertainty))
        or np.any(uncertainty <= 0.0)
        or values.shape != expected
        or not np.all(np.isfinite(values))
    ):
        raise DataScaledSensitivityError("Jacobian inputs are invalid")
    derivative = (values[1::2] - values[2::2]) / (2.0 * steps[:, None])
    return derivative / uncertainty[None, :]


def singular_diagnostics(jacobian: Any) -> dict[str, Any]:
    """Expose raw SVD evidence without choosing a scientific rank verdict."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise DataScaledSensitivityError("standardized Jacobian must be a finite matrix")
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    return {
        "schema": "v14-snr-stageB-data-scaled-svd-v1",
        "scientific_verdict": None,
        "optimization_allowed": False,
        "singular_values": singular_values.tolist(),
        "parameter_space_directions": left.T.tolist(),
        "target_space_directions": right.tolist(),
    }
