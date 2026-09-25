"""Affect-marker SURFACE retirement flag (2026-09-25 owner decision, branch research/retire-affect-marker-word;
addendum to research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md, Amendment 3).

Pure env-var-level checks of `webapp.affect_drives_chat.affect_marker_surface_enabled()` — no brain build, no HTTP
call (the webapp-integration coverage, including the ON == OFF+lead byte-identical restore through the real
handler, lives in tests/test_webapp_server.py::test_brain_chat_affect_marker_surface_default_off_records_but_does_
not_surface). Proves:
  * default OFF (`BRAIN_AFFECT_MARKER_SURFACE` unset) -- the new production default: the marker stays an internal
    record only, never surfaced.
  * every documented truthy spelling turns it ON; every documented falsy spelling (including an explicit "0")
    stays OFF -- mirrors every other flag-parsing function in this module (`affect_drives_enabled`,
    `marker_selection_spiking_off`, `congruence_gate_enabled`).
  * an unrecognized value degrades to OFF (the safe default), never raises.
"""
import os

# `webapp.affect_drives_chat` imports `research.runners._graded_affect_attractor_derisk`, which imports the
# `research.runners` package -- the automatic provenance door (research/runners/__init__.py). That import is
# what actually latches the door's `_ENABLED` flag (read once, at first import, from `SIM_NO_PROVENANCE`), so it
# must be disabled BEFORE this module-level import, exactly like tests/test_seam_contracts.py does for its own
# import of the same package -- otherwise this file collected ahead of test_seam_contracts.py (alphabetically,
# "affect" < "seam") leaves the door latched ENABLED in `sys.modules` for the rest of the session, and that
# test's `assert not prov_door._ENABLED` fails even though ITS OWN import guard is correct (a cached module is
# not re-executed, so the guard only protects the FIRST import in the process).
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import pytest  # noqa: E402

from webapp import affect_drives_chat as adc  # noqa: E402

ENV = "BRAIN_AFFECT_MARKER_SURFACE"


@pytest.fixture(autouse=True)
def _clean_env():
    prior = os.environ.pop(ENV, None)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = prior


def test_default_unset_is_off():
    assert ENV not in os.environ
    assert adc.affect_marker_surface_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "on", "ON", "yes", "Yes"])
def test_truthy_spellings_turn_it_on(value):
    os.environ[ENV] = value
    assert adc.affect_marker_surface_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "False", "off", "no", "", "garbage"])
def test_falsy_or_unrecognized_spellings_stay_off(value):
    os.environ[ENV] = value
    assert adc.affect_marker_surface_enabled() is False


def test_flag_is_named_after_the_env_var_used_in_server_wiring():
    # webapp/server.py's two prepend sites read affect_marker_surface_enabled() directly (no separate module-level
    # constant to drift) -- pin the literal env var name so a rename in one place cannot silently orphan the other.
    import inspect
    src = inspect.getsource(adc.affect_marker_surface_enabled)
    assert ENV in src
