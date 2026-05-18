from research.runners.constrained_decode_core import (
    cdc_verdict, cdc_scale_confidence, grounded_content_count,
    nonvacuous_answered)


def _seed_ok(**kw):
    d = dict(unconstrained_uer=0.85, constrained_uer=0.0,
             constrained_nonvac_rate=0.9, shuffled_uer=0.85,
             shuffled_nonvac_rate=0.0, bare_moat_abstain_rate=1.0,
             abstain_on_ungrounded_rate=1.0,
             constrained_multitoken_emittable_rate=1.0)
    d.update(kw); return d


def test_grounded_content_count_distinct_on_prop():
    assert grounded_content_count("the max is a dog dog",
                                  "max is a big dog") == 2


def test_nonvacuous_requires_min_distinct_content():
    assert nonvacuous_answered("max dog", "max is a big dog") is True
    assert nonvacuous_answered("max the the", "max is a big dog") is False
    assert nonvacuous_answered("the is a and", "max is a big dog") is False


def test_all_good_seeds_pass():
    v = cdc_verdict({42: _seed_ok(), 43: _seed_ok(), 44: _seed_ok()})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_v1_unmet_unconstrained_not_drifting_is_void():
    v = cdc_verdict({42: _seed_ok(unconstrained_uer=0.10),
                     43: _seed_ok(unconstrained_uer=0.10),
                     44: _seed_ok(unconstrained_uer=0.10)})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False


def test_vacuity_collapse_is_FAIL_not_pass():
    v = cdc_verdict({42: _seed_ok(constrained_nonvac_rate=0.10),
                     43: _seed_ok(constrained_nonvac_rate=0.10),
                     44: _seed_ok(constrained_nonvac_rate=0.10)})
    assert v["GATE"] == "FAIL" and v["instrument_valid"] is True


def test_unconstrained_control_passing_faithful_is_void():
    v = cdc_verdict({42: _seed_ok(unconstrained_uer=0.05),
                     43: _seed_ok(unconstrained_uer=0.05),
                     44: _seed_ok(unconstrained_uer=0.05)})
    assert v["GATE"] == "VOID"


def test_shuffled_control_not_failing_is_void():
    v = cdc_verdict({42: _seed_ok(shuffled_uer=0.0, shuffled_nonvac_rate=0.9),
                     43: _seed_ok(shuffled_uer=0.0, shuffled_nonvac_rate=0.9),
                     44: _seed_ok(shuffled_uer=0.0, shuffled_nonvac_rate=0.9)})
    assert v["GATE"] == "VOID"


def test_no_confab_not_preserved_is_void():
    v = cdc_verdict({42: _seed_ok(abstain_on_ungrounded_rate=0.4),
                     43: _seed_ok(abstain_on_ungrounded_rate=0.4),
                     44: _seed_ok(abstain_on_ungrounded_rate=0.4)})
    assert v["GATE"] == "VOID"


def test_fewer_than_min_seeds_is_void():
    assert cdc_verdict({42: _seed_ok()})["GATE"] == "VOID"


def test_non_numeric_junk_is_void_not_raise():
    bad = dict(_seed_ok()); bad["constrained_nonvac_rate"] = "oops"
    assert cdc_verdict({42: bad, 43: _seed_ok(),
                        44: _seed_ok()})["GATE"] == "VOID"


def test_unorderable_keys_void_not_raise():
    assert cdc_verdict({object(): _seed_ok()})["GATE"] == "VOID"


def _rg(K, g, nv):
    return {"K": K, "verdict": {"GATE": g},
            "constrained_nonvac_rate_mean": nv}


def test_scale_confident_all_pass_nondegrading():
    r = cdc_scale_confidence([_rg(6, "PASS", 0.9), _rg(12, "PASS", 0.9),
                              _rg(24, "PASS", 0.88)])
    assert r["scale_confident"] is True
    assert r["classification"] == "SCALE-CONFIDENT-PASS"


def test_scale_degrades_is_works_small():
    r = cdc_scale_confidence([_rg(6, "PASS", 0.9), _rg(12, "PASS", 0.7),
                              _rg(24, "PASS", 0.55)])
    assert r["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE"


def test_scale_void_and_fail_precedence():
    assert cdc_scale_confidence([_rg(6, "PASS", .9), _rg(12, "VOID", 0.),
        _rg(24, "PASS", .9)])["classification"] == "VOID"
    assert cdc_scale_confidence([_rg(6, "PASS", .9), _rg(12, "FAIL", .5),
        _rg(24, "PASS", .9)])["classification"] == "FAIL"


def test_scale_ladder_tamper_is_void():
    assert cdc_scale_confidence([_rg(6, "PASS", .9), _rg(12, "PASS", .9)]
        )["classification"] == "VOID"


# ---- Fix B: additive frozen instrument-validity floor (multitoken) ----

def test_frozen_cdc_values_byte_unchanged():
    import research.runners.constrained_decode_core as c
    assert c._CDC_FAITHFUL_MAX == 0.20
    assert c._CDC_MIN_GROUNDED_CONTENT == 2
    assert c._CDC_MIN_GROUNDED_ANSWER_RATE == 0.5
    assert c._CDC_MIN_SEEDS == 3
    assert c._CDC_SCALE_LADDER == (6, 12, 24)
    assert c._CDC_SCALE_TOL == 0.10


def test_new_multitoken_floor_constant_is_half():
    import research.runners.constrained_decode_core as c
    assert c._CDC_MIN_MULTITOKEN_EMITTABLE == 0.5


def test_low_multitoken_emittable_any_seed_is_void_cannot_test():
    # subword-defeated regime: instrument cannot express the tested
    # effect -> honest VOID (cannot-test), NOT an ambiguous FAIL.
    v = cdc_verdict({42: _seed_ok(),
                     43: _seed_ok(constrained_multitoken_emittable_rate=0.49),
                     44: _seed_ok()})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False
    assert "subword-defeated" in v["reason"]


def test_at_floor_multitoken_emittable_otherwise_good_passes():
    v = cdc_verdict({42: _seed_ok(constrained_multitoken_emittable_rate=0.5),
                     43: _seed_ok(constrained_multitoken_emittable_rate=0.5),
                     44: _seed_ok(constrained_multitoken_emittable_rate=0.5)})
    assert v["GATE"] == "PASS" and v["instrument_valid"] is True


def test_missing_multitoken_key_is_void_fail_closed():
    bad = dict(_seed_ok())
    bad.pop("constrained_multitoken_emittable_rate")
    v = cdc_verdict({42: bad, 43: _seed_ok(), 44: _seed_ok()})
    assert v["GATE"] == "VOID" and v["instrument_valid"] is False
