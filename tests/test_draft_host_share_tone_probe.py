"""CI guard for the host-share instrument, SECOND SEAM (plan section 2, runbook task 2): the tone selection
(`webapp.affect_tone_selection.select_by_mood`). The instrument records the three plan-section-2 measurements on a
dev-seed candidate panel -- (a) DECODING (the host argmin + the candidate valences from the spiking affect organ +
v_held on the same trace), (b) SUBSTITUTION (the N-channel BG selector's race result, the named next rung), and
(c) INTERVENTION (hold the candidate valences + v_held fixed, sweep the host decision, the selected candidate
follows the host's choice -- the host owns the selection at this seam).

numpy/CPU (the probe is cost-routed to CPU). DEV seed only (7); NEVER a gate seed. This is a DEV instrument
guard (structure + the decoding invariant), NOT a 6-seed generalization claim. The N-channel BG selector's
substitution arm is recorded but NOT gated on (the N=5 tonic-firing wall means it does not commit cleanly; that
is a finding, not a pass/fail criterion for this instrument guard).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._draft_host_share_tone_probe import (  # noqa: E402
    _candidate_valences, _congruence_saliences, run_seed,
    CANDIDATE_TEXTS, V_HELD_GRID,
)
from webapp.affect_tone_selection import select_by_mood  # noqa: E402

DEV_SEED = 7  # a dev seed (runbook hard rule: never 42/43/44/100/101/102)
assert DEV_SEED not in (42, 43, 44, 100, 101, 102)


def test_candidate_valences_span_sign():
    """The candidate valences (the neural inputs) must span both signs (the dev candidate texts include positive
    and negative affect words), so the host argmin has a real choice."""
    from research.runners import affect_production_organ as AO
    organ = AO.get_organ(seed=DEV_SEED)
    organ.ensure_built()
    vals = _candidate_valences(organ, CANDIDATE_TEXTS)
    assert len(vals) == len(CANDIDATE_TEXTS)
    assert all(-1.0 <= v <= 1.0 for v in vals), "candidate valences must be in [-1, 1]"
    assert min(vals) < 0.0 < max(vals), "the candidate valences must span both signs (positive + negative)"


def test_decoding_host_tracks_valences():
    """The core claim: the host argmin picks the candidate closest to v_held at each level (the host decision
    faithfully tracks the neural inputs). This is the DECODING invariant."""
    from research.runners import affect_production_organ as AO
    organ = AO.get_organ(seed=DEV_SEED)
    organ.ensure_built()
    vals = _candidate_valences(organ, CANDIDATE_TEXTS)
    for v_held in V_HELD_GRID:
        host_idx = select_by_mood(vals, v_held)
        expected = min(range(len(vals)), key=lambda k: (abs(vals[k] - v_held), k))
        assert host_idx == expected, (
            f"host argmin ({host_idx}) must pick the candidate closest to v_held={v_held} "
            f"(expected {expected}); valences={vals}"
        )


def test_probe_records_three_measurements():
    """The probe records all three plan-section-2 measurements on the dev seed: (a) the host argmin + the
    candidate valences + v_held (decoding), (b) the N-channel BG selector's race result (substitution), and
    (c) the intervention (the selected candidate follows the host's choice)."""
    r = run_seed(DEV_SEED)
    assert r["seed"] == DEV_SEED
    assert r["label"] == "DEV, not a verdict"
    assert len(r["rows"]) == len(V_HELD_GRID)
    for row in r["rows"]:
        # (a) DECODING: the host argmin + the candidate valences + v_held are recorded.
        assert row["host_argmin"] is not None
        assert len(row["candidate_valences"]) == len(CANDIDATE_TEXTS)
        assert row["v_held"] in V_HELD_GRID
        # (b) SUBSTITUTION: the N-channel BG selector's race result is recorded (the winner may be None -- the
        # N=5 wall -- but the race dict is present).
        assert "race_committed" in row
        assert "race_motor_spikes" in row
        # (c) INTERVENTION: the selected candidate follows the host's choice (one row per candidate index).
        assert len(row["intervention"]) == len(CANDIDATE_TEXTS)
        for iv in row["intervention"]:
            assert iv["host_pick"] == iv["selected_candidate"], (
                "the intervention must show the selected candidate follows the host's choice"
            )
    # the decoding invariant holds across the grid.
    assert r["decoding_host_tracks_valences"] is True
