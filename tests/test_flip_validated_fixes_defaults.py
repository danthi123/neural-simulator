"""Pin the 2026-09-23 production default-flip of three validated fixes, in BOTH directions.

Owner directive 2026-09-23: validated default-flips are Claude's to make (docs/plans/2026-09-23-autonomous-charter.md
§5). Flipped default-ON: BRAIN_EPISODIC_STORE_VERIFY, BRAIN_PMEM_FACILITATION, BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE.

Each reader must (a) default ON when the variable is UNSET, (b) still honour an EXPLICIT falsy value (the OFF arm stays
reachable -- the 2026-08-27 stale-off-arm bug class, gates/flip_offarm_staleness), and (c) keep an explicit truthy
value ON. Reader-level only: no brain is built here.
"""
from __future__ import annotations

import importlib

import pytest

_CASES = [
    ("research.runners._episodic_dap_dialogue_memory", "_store_verify_enabled", "BRAIN_EPISODIC_STORE_VERIFY",
     "_STORE_VERIFY_DEFAULT_ON"),
    ("research.runners.prospective_memory_production_organ", "pmem_facilitation_enabled", "BRAIN_PMEM_FACILITATION",
     "_PMEM_FACILITATION_DEFAULT_ON"),
    ("research.runners.source_provenance_honesty", "source_prov_abstain_at_tie_enabled",
     "BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE", "_ABSTAIN_AT_TIE_DEFAULT_ON"),
]
_IDS = [c[2] for c in _CASES]


def _reader(modname, fn):
    mod = importlib.import_module(modname)
    return mod, getattr(mod, fn)


@pytest.mark.parametrize("modname,fn,flag,const", _CASES, ids=_IDS)
def test_unset_reads_on(monkeypatch, modname, fn, flag, const):
    mod, reader = _reader(modname, fn)
    monkeypatch.delenv(flag, raising=False)
    assert getattr(mod, const) is True, f"{const} must be the literal True (the ledger anchor reads it)"
    assert reader() is True, f"{flag} unset must now read ON (production default-flip 2026-09-23)"


@pytest.mark.parametrize("off", ["0", "false", "no", "off", "", "FALSE", " 0 "])
@pytest.mark.parametrize("modname,fn,flag,const", _CASES, ids=_IDS)
def test_explicit_falsy_reads_off(monkeypatch, modname, fn, flag, const, off):
    _mod, reader = _reader(modname, fn)
    monkeypatch.setenv(flag, off)
    assert reader() is False, f"{flag}={off!r} must still force the OFF arm (byte-identical escape)"


@pytest.mark.parametrize("on", ["1", "true", "yes", "on", "ON"])
@pytest.mark.parametrize("modname,fn,flag,const", _CASES, ids=_IDS)
def test_explicit_truthy_reads_on(monkeypatch, modname, fn, flag, const, on):
    _mod, reader = _reader(modname, fn)
    monkeypatch.setenv(flag, on)
    assert reader() is True


def test_battery_guard_passes_on_this_revision_and_fails_on_an_override():
    """tools/assert_flipped_defaults.py prefixes every flipdefaults-* battery job; it must pass here with the flags
    unset, and fail as soon as one is set (the battery would then not measure the default)."""
    from tools import assert_flipped_defaults as guard
    assert guard.problems(environ={}) == []
    for flag in guard.FLIPPED:
        probs = guard.problems(environ={flag: "0"})
        assert len(probs) == 1 and flag in probs[0]


def test_flip_offarm_gate_resolves_all_three_defaults_on():
    """The static shadow the OS gate uses must see the flip (else a future pop-based OFF arm would slip through)."""
    import os

    from tools.gates import flip_offarm_staleness as g
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _m, _f, flag, _c in _CASES:
        assert g._default_state(flag, root) == "on", flag
