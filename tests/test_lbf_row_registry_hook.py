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


def test_empty_lbf_rows_package_is_a_noop():
    """The package ships with NO row modules yet (S12-S20 add the first ones later, on separate branches). Calling
    the merge again with the package still empty must change NEITHER registry -- the exact bar plan step S08 sets:
    'A unit test asserts that an empty package leaves the key sets identical.'"""
    before_lesion_keys = set(lbf.FACULTY_LESIONS)
    before_probe_keys = sorted(row[0] for row in lbf.FACULTY_PROBES)
    before_probe_len = len(lbf.FACULTY_PROBES)

    mods, errors = lbf_rows._discover_row_modules()
    assert mods == [], "an EXTRA row module exists in research/runners/lbf_rows/ -- this pin assumes none do yet"
    assert errors == []

    report = lbf_rows.merge_lbf_rows(lbf.FACULTY_LESIONS, lbf.FACULTY_PROBES)
    assert report["modules"] == []
    assert report["keys_added"] == []
    assert report["collisions"] == []

    assert set(lbf.FACULTY_LESIONS) == before_lesion_keys
    assert sorted(row[0] for row in lbf.FACULTY_PROBES) == before_probe_keys
    assert len(lbf.FACULTY_PROBES) == before_probe_len


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
