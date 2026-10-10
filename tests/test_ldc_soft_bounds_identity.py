"""A2 byte-identical-off (RULING a02-approved): reward_soft_bounds=0 must reproduce the
pre-A2 reference cell bitwise, and soft_bounds=1 must change no BUILD-time structure.

Reference: research/findings/raw/_long_delay_credit_bg/d2_floor_observability/on_s7.json
(credit_immediate seed 7, arm on, 400 trials, run before the A2 edit). The rebuild goes through
the probe's own run_condition with default soft_bounds=0; comparison is the full row minus the
non-deterministic wall_s, as exact JSON (floats round-trip bit-exact; NaN matches NaN as text).
"""
import io
import json
import os

import pytest

REF_PATH = os.path.join(os.path.dirname(__file__), "..",
                        "research/findings/raw/_long_delay_credit_bg/d2_floor_observability/on_s7.json")
if not os.path.exists(REF_PATH):
    # the reference cell lives in the main checkout until the A2 merge carries it
    REF_PATH = ("/home/dant123/Projects/neural-simulator/research/findings/raw/"
                "_long_delay_credit_bg/d2_floor_observability/on_s7.json")
REF = json.load(io.open(REF_PATH, encoding="utf-8")) if os.path.exists(REF_PATH) else None
ROW = (REF[0] if isinstance(REF, list) else (REF.get("rows") or list(REF.values())[0])[0] if REF else None)
EXCLUDE = {"wall_s", "lesion_weight_max_at_test"}


def _canon(d):
    return json.dumps({k: v for k, v in d.items() if k not in EXCLUDE}, sort_keys=True)


@pytest.mark.skipif(ROW is None, reason="reference cell on_s7.json missing")
def test_flag0_rebuild_matches_reference_bitwise():
    from research.runners._ldc_dip_observability import run_condition
    got = run_condition(ROW["schedule"], ROW["arm"], ROW["seed"],
                        n_train=ROW["n_train"], n_test=ROW["n_test"])
    ref_c, got_c = _canon(ROW), _canon(got)
    if ref_c != got_c:
        # report the first differing key for a readable failure
        for k in sorted(set(ROW) | set(got)):
            if k in EXCLUDE:
                continue
            if json.dumps(ROW.get(k), sort_keys=True) != json.dumps(got.get(k), sort_keys=True):
                raise AssertionError("field %r differs between reference and flag-0 rebuild" % k)
        raise AssertionError("rebuild differs from reference (not in compared fields?)")


def test_soft_bounds_flag_lands_on_config_and_builds_identically():
    from research.runners._ldc_credit_omission import CreditParams, build_credit_selector
    b0, _cfg0 = build_credit_selector(7, CreditParams())
    b1, _cfg1 = build_credit_selector(7, CreditParams(soft_bounds=1))
    assert int(getattr(b0.core_config, "reward_soft_bounds", 0)) == 0, "flag must default OFF"
    assert int(getattr(b1.core_config, "reward_soft_bounds", 0)) == 1, "soft_bounds must reach core_config"
    import numpy as np
    from sim.backend import to_host
    assert int(b1.cp_connections.nnz) == int(b0.cp_connections.nnz)
    assert np.array_equal(np.asarray(to_host(b0.cp_connections.data)),
                          np.asarray(to_host(b1.cp_connections.data))), \
        "soft_bounds must change NO build-time weight (it only gates the online update)"
