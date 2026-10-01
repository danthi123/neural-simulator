"""Scorer-input tests for the long-delay credit benchmark: the section-7 instrument scorer and the registered gate
scorer must read only data artifacts (never the `.prov.json` provenance sidecars the runner door writes next to each
one), and a measured 0.0 must count as a value. Synthetic rows only; not evidence for the capability."""
import json

import pytest

import research.runners.long_delay_credit_bg_instrument as I
import research.runners.long_delay_credit_bg_verdict as V

SCHEDULES = ("credit_immediate", "late_immediate", "sanity_early_ch0")


def _row(schedule, seed, inst=None, **over):
    r = {"schedule": schedule, "arm": "full", "seed": seed, "success": 0.95, "response_rate": 1.0,
         "void_floor_ceiling": False, "da_outcome_rewarded_max_mean": 0.65, "da_outcome_unrewarded_min_mean": 0.40,
         "probe_lr0": {"early_commit_rate": 1.0, "terminal_commit_rate": 1.0, "p_ch0_given_A": 0.5,
                       "p_ch0_given_B": 0.5},
         "instrument": {"interv_commit_rate_probe": 1.0, "decoy_commit_rate_probe": 1.0, "mem_hold_acc_probe": 1.0,
                        "mem_after_decoy_acc_probe": 1.0, "mem_after_clear_hz": 0.0, "da_iti_offset_probe": 0.001}}
    r["instrument"].update(inst or {})
    r.update(over)
    return r


def _fill(d, tag="t", edit=None):
    """9 dev seeds x 3 schedules, each data file with its provenance sidecar beside it (as the pool returns them)."""
    for sch in SCHEDULES:
        for s in I.DEV_SEEDS:
            row = _row(sch, s)
            if edit:
                edit(sch, row)
            p = d / ("%s_%s_s%d.json" % (tag, sch, s))
            p.write_text(json.dumps({"runner_sha256": "h", "rows": [row]}))
            (d / (p.name + ".prov.json")).write_text(json.dumps({"argv": ["-m", "x"], "exit_code": 0}))


def test_instrument_skips_provenance_sidecars(tmp_path):
    _fill(tmp_path)
    c = I.criteria(str(tmp_path), "t")          # before the fix: KeyError 'rows' on the first sidecar
    assert c["n_seeds"] == {s: 9 for s in SCHEDULES}
    assert all(v for k, v in c.items() if k.startswith("c")), c
    with pytest.raises(SystemExit) as e:
        I.main(["--tags", "t", "--dir", str(tmp_path)])
    assert e.value.code == 0


def test_instrument_zero_is_a_measured_value(tmp_path):
    def edit(sch, row):
        row["instrument"]["da_iti_offset_probe"] = 0.0
        if sch == "credit_immediate":
            row["da_outcome_unrewarded_min_mean"] = 0.0     # a full SNc pause at omission is the strongest dip
    _fill(tmp_path, edit=edit)
    assert I.criteria(str(tmp_path), "t")["c3_dopamine"]


def test_instrument_tonic_offset_checked_on_every_probe(tmp_path):
    def edit(sch, row):
        if sch == "late_immediate":
            row["instrument"]["da_iti_offset_probe"] = -0.11   # a critic tail leaking into the ITI (Step 4, strio60)
    _fill(tmp_path, edit=edit)
    assert not I.criteria(str(tmp_path), "t")["c3_dopamine"]


def test_verdict_load_skips_provenance_sidecars(tmp_path):
    p = tmp_path / "credit_full_s42.json"
    p.write_text(json.dumps({"runner_sha256": "h", "rows": [_row("credit", 42)]}))
    side = tmp_path / (p.name + ".prov.json")
    side.write_text(json.dumps({"argv": ["-m", "x"]}))
    cells, hashes = V.load([str(p), str(side)])  # before the fix: KeyError 'rows'
    assert list(cells) == [("credit", "full", 42)] and hashes == {"h"}
