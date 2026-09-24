"""Pin for research/lbf-row-registry-hook (2026-09-24, plan step S08 / lane AG-REG): the import hook in
research/runners/load_bearing_fraction.py that merges research/runners/lbf_rows/*.py EXTRA_LESIONS/EXTRA_PROBES
into FACULTY_LESIONS/FACULTY_PROBES must be a NO-OP when the lbf_rows package is EMPTY (no modules besides
__init__.py -- true as of this commit), and must merge correctly + report collisions once a module IS present.
The failing direction (a merge that silently duplicates rows, or overwrites an existing key) is tested first.
"""
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import load_bearing_fraction as lbf  # noqa: E402
from research.runners import lbf_rows  # noqa: E402


def test_merge_with_no_row_modules_is_a_noop():
    """With NO row modules the merge must change NEITHER registry -- the bar plan step S08 set ('an empty package
    leaves the key sets identical'). The package is no longer empty on main (row modules from other lanes landed
    first), so the empty case is exercised through the test seam rather than by asserting the folder is empty."""
    lesions = dict(lbf.FACULTY_LESIONS)
    probes = list(lbf.FACULTY_PROBES)
    before_lesions, before_probes = dict(lesions), list(probes)

    report = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[])
    assert report["modules"] == [] and report["keys_added"] == [] and report["collisions"] == []
    assert lesions == before_lesions
    assert probes == before_probes


def test_the_live_registry_never_holds_a_probe_whose_turn_is_unregistered():
    """Every FACULTY_PROBES row the battery actually runs must name a turn in PROBE_TURNS; otherwise that faculty
    can never complete and every battery fails its 'no incomplete faculty' criterion (2026-09-24 merge)."""
    from research.runners import onebrain_regression_battery as ob
    turns = {t[0] for t in ob.PROBE_TURNS}
    missing = [row[:2] for row in lbf.FACULTY_PROBES if row[1] not in turns]
    assert missing == [], missing


def test_a_list_of_rows_is_accepted_like_a_dict():
    """FACULTY_PROBES' own shape (a list of 4-tuples) is accepted, and a key listed twice is reported, not merged."""
    lesions = dict(lbf.FACULTY_LESIONS)
    probes = list(lbf.FACULTY_PROBES)

    class _ListMod:
        __name__ = "research.runners.lbf_rows._fake_list_for_test"
        EXTRA_LESIONS = {"synthetic-list-faculty": dict(flag="BRAIN_SYNTHETIC_LIST_LESION", value="1",
                                                         kind="neural-lesion", note="test-only row")}
        EXTRA_PROBES = [("synthetic-list-faculty", "well", ["synthetic.on"], False),
                        ("synthetic-list-faculty", "well", ["synthetic.other"], False)]

    report = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[_ListMod])
    matches = [row for row in probes if row[0] == "synthetic-list-faculty"]
    assert matches == [("synthetic-list-faculty", "well", ["synthetic.on"], False)]
    assert any("more than once" in c for c in report["collisions"])


def test_a_row_whose_probe_turn_is_unregistered_is_parked_not_merged():
    """The failing direction first: a row probing a turn the battery never runs must add NEITHER its lesion NOR its
    probe, and the report must say it was parked."""
    lesions = dict(lbf.FACULTY_LESIONS)
    probes = list(lbf.FACULTY_PROBES)

    class _UnregisteredTurnMod:
        __name__ = "research.runners.lbf_rows._fake_unregistered_turn_for_test"
        EXTRA_LESIONS = {"synthetic-parked-faculty": dict(flag="BRAIN_SYNTHETIC_PARKED_LESION", value="1",
                                                           kind="neural-lesion", note="test-only row")}
        EXTRA_PROBES = [("synthetic-parked-faculty", "no_such_turn_label_anywhere", ["x"], True)]

    report = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[_UnregisteredTurnMod])
    assert "synthetic-parked-faculty" not in lesions
    assert not any(row[0] == "synthetic-parked-faculty" for row in probes)
    assert any("synthetic-parked-faculty" in p for p in report["parked"])


def test_module_already_imported_merge_report_is_idempotent_no_op():
    """load_bearing_fraction.py already ran merge_lbf_rows() once at import time (module-level
    LBF_ROW_MERGE_REPORT). Calling it again in the SAME process (e.g. this test re-importing the module) must not
    re-append any row a second time -- the idempotency guard in lbf_rows.merge_lbf_rows()."""
    assert hasattr(lbf, "LBF_ROW_MERGE_REPORT")
    before_len = len(lbf.FACULTY_PROBES)
    report = lbf_rows.merge_lbf_rows(lbf.FACULTY_LESIONS, lbf.FACULTY_PROBES)
    assert report.get("skipped"), "a second call in-process must report skipped, not re-merge"
    assert len(lbf.FACULTY_PROBES) == before_len


def test_merge_adds_a_well_formed_row_and_keeps_lesions_probes_in_lockstep():
    """A synthetic row module (module-level dicts, not imported through the real package) exercises the real merge
    logic end to end: EXTRA_LESIONS + EXTRA_PROBES for a brand-new faculty key land in fresh copies of the two
    registries, and the merged FACULTY_PROBES row is the exact 4-tuple the module declared."""
    lesions = dict(lbf.FACULTY_LESIONS)
    probes = list(lbf.FACULTY_PROBES)
    assert "synthetic-test-faculty" not in lesions

    class _FakeMod:
        __name__ = "research.runners.lbf_rows._fake_for_test"
        EXTRA_LESIONS = {
            "synthetic-test-faculty": dict(flag="BRAIN_SYNTHETIC_TEST_LESION", value="1", kind="neural-lesion",
                                            note="test-only row, never a real production organ"),
        }
        EXTRA_PROBES = {
            "synthetic-test-faculty": ("synthetic-test-faculty", "well", ["synthetic.on"], False),
        }

    report = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[_FakeMod])

    assert "synthetic-test-faculty" in lesions
    assert lesions["synthetic-test-faculty"]["flag"] == "BRAIN_SYNTHETIC_TEST_LESION"
    matches = [row for row in probes if row[0] == "synthetic-test-faculty"]
    assert len(matches) == 1
    assert matches[0] == ("synthetic-test-faculty", "well", ["synthetic.on"], False)
    assert report["collisions"] == []
    assert "synthetic-test-faculty" in report["keys_added"]


def test_merge_reports_a_collision_instead_of_overwriting_an_existing_row():
    """A row module that claims a faculty_key ALREADY in the base registry must be dropped and reported, never
    silently overwrite the existing (real, reviewed) row."""
    lesions = dict(lbf.FACULTY_LESIONS)
    probes = list(lbf.FACULTY_PROBES)
    existing_key = next(iter(lbf.FACULTY_LESIONS))
    original_spec = dict(lesions[existing_key])

    class _CollidingMod:
        __name__ = "research.runners.lbf_rows._fake_colliding_for_test"
        EXTRA_LESIONS = {existing_key: dict(flag="BRAIN_SHOULD_NOT_LAND_LESION", value="1", kind="neural-lesion", note="")}
        EXTRA_PROBES = {}

    report = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[_CollidingMod])

    assert lesions[existing_key] == original_spec, "a collision must never overwrite the existing row"
    assert any(existing_key in c for c in report["collisions"])
