"""Why did `test_the_tap_does_not_change_what_it_measures` fail on the 3090?

The test builds the comprehension organ TWICE in one process on the same backend: once with the write tap
installed, once without. On numpy the digests match. On cupy they did not (research/findings/raw/
_pi203_frozen_synapse_residual/cupy_frozen_flat_pytest_v6.txt). This runner separates the two possible causes:

  control A vs B  : two untapped organs, same seed, same process  -> if these differ, the organ itself is
                    not reproducible across instances on cupy and the tap is innocent;
  tapped  vs untapped: the assertion under test.

It prints the first item where the replies diverge, so the failure is attributed to a mechanism, not to a hash.
No new claim about the mask: this is an instrument check.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.lab import assert_backend, attributable_to, project_cost  # noqa: E402
from sim.backend import get_backend  # noqa: E402

from research.runners._comprehension_learned_cues_joint_arm import all_items  # noqa: E402
from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, extract_transitive  # noqa: E402
from research.runners import pi203_write_path_tap as tapmod  # noqa: E402


def _replies(tap: bool, seed: int) -> list:
    os.environ.pop("BRAIN_ENFORCE_PLASTIC_MASK", None)
    organ = ComprehensionProductionOrgan(seed=seed)
    organ.ensure_built()
    if tap:
        tapmod.install_tap(organ.comp.bridge)
        blind = tapmod.tap_is_blind(organ.comp.bridge)
    else:
        blind = False
    out = []
    for label, n0, v, n1 in all_items():
        text = f"the {n0} {v} the {n1}"
        tr = extract_transitive(text)
        out.append((label, organ.judge(text), organ.competent(*tr) if tr else None))
    return out, blind


def _margin_sum(replies) -> float:
    return float(sum(r[1].get("margin", 0.0) if isinstance(r[1], dict) else 0.0 for r in replies))


def main() -> int:
    assert_backend("cupy", "the tap-blindness check is only meaningful on the GPU path")
    _xp, resolved = get_backend()
    device = os.environ.get("SIM_BACKEND", resolved)
    t0 = time.time()
    a, _ = _replies(False, 7)
    b, _ = _replies(False, 7)
    c, blind = _replies(True, 7)
    d, _ = _replies(False, 7)

    da, db, dc, dd = (tapmod._digest(x) for x in (a, b, c, d))
    diffs = [i for i in range(len(a)) if a[i] != b[i]]
    tap_diffs = [i for i in range(len(a)) if a[i] != c[i]]
    field_diffs = {}
    for i in tap_diffs:
        for field in (0, 1, 2):
            if a[i][field] != c[i][field]:
                field_diffs.setdefault(field, []).append((a[i][0], a[i][field], c[i][field]))
    result = {
        "runner": "pi203_tap_blindness_cupy_check",
        "backend_requested": "cupy",
        "backend_resolved": resolved,
        "device": device,
        "elapsed_seconds": round(time.time() - t0, 2),
        "organ_seed": 7,
        "untapped_A_digest": da,
        "untapped_B_digest": db,
        "untapped_D_digest": dd,
        "tapped_digest": dc,
        "tap_blind": blind,
        "control_two_untapped_organs_identical": da == db == dd,
        "tapped_equals_untapped": dc == da,
        "first_divergences_control": diffs[:3],
        "first_divergences_tap": tap_diffs[:3],
        "n_items": len(a),
        "n_items_diverging_tap": len(tap_diffs),
        "n_items_diverging_control": len(diffs),
        "divergent_fields": {("reply_dict" if f == 1 else "competent" if f == 2 else "label"): v[:4]
                            for f, v in field_diffs.items()},
        "example_items_where_tap_differs": [list(a[i]) for i in tap_diffs[:3]],
        "example_items_untapped_A": [list(a[i]) for i in diffs[:3]],
    }
    result["reply_margin_sum_untapped"] = _margin_sum(a)
    result["reply_margin_sum_tapped"] = _margin_sum(c)
    result["attributable"] = attributable_to(
        "reply-margin sum: tapped organ vs untapped organ (instrument perturbation)",
        _margin_sum(c), _margin_sum(a),
    )
    project_cost("pi203_tap_blindness_cupy_check", 0, 2, time.time() - t0)
    text = json.dumps(result, indent=2)
    print(text[:2500])
    out = "research/findings/raw/_pi203_frozen_synapse_residual/tap_blindness_cupy_check.json"
    with open(out, "w") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
