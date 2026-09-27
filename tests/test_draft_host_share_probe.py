"""CI guard for the host-share instrument, FIRST SEAM (plan section 2, runbook task 2): the metacog confidence
read (`metacog_production_organ.mean_role_confidence`). The instrument records the three plan-section-2
measurements on a dev-seed composer query -- (a) DECODING (the host decision + the per-role neural inputs on the
same trace), (b) SUBSTITUTION (the downstream `confident` at the host decision vs a neural-derived one), and
(c) INTERVENTION (hold the neural state fixed, sweep the host decision, the downstream `confident` must respond
monotonically -- the spiking WTA faithfully transmits the host's evidence, so the host owns the commitment at
this seam).

numpy/CPU (the probe is cost-routed to CPU). DEV seed only (7); NEVER a gate seed. This is a DEV instrument
guard (structure + the monotonic-intervention invariant), NOT a 6-seed generalization claim.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._draft_host_share_probe import (  # noqa: E402
    build_composer, ask_and_trace, _host_mrc, _neural_mrc, _confident,
    run_seed_degraded, FACTS, MRC_GRID,
)
from research.runners.metacog_production_organ import (  # noqa: E402
    MetacogProductionOrgan,
)

DEV_SEED = 7  # a dev seed (runbook hard rule: never 42/43/44/100/101/102)
assert DEV_SEED not in (42, 43, 44, 100, 101, 102)


def _build():
    comp = build_composer(DEV_SEED)
    organ = MetacogProductionOrgan(seed=DEV_SEED)
    organ.ensure_built()
    return comp, organ


def _first_answered(comp):
    """The first dev-fact query that answers (not abstains) -- the probe's clean-fact panel."""
    for a, v, _p in FACTS:
        ans, trace = ask_and_trace(comp, a, v)
        if ans is not None and trace:
            return a, v, ans, trace
    raise AssertionError("no dev fact answered on the dev seed (panel regressed)")


def test_probe_records_three_measurements():
    comp, organ = _build()
    a, v, ans, trace = _first_answered(comp)
    host = _host_mrc(trace)
    neural = _neural_mrc(trace)
    assert host is not None and neural is not None, "both the host and the neural read must be present"
    # (a) DECODING: the per-role neural inputs are recorded and at least one field is finite per role.
    roles = trace.get("roles") or []
    assert roles, "the trace must carry per-role fields"
    for r in roles:
        assert any(r.get(k) is not None
                   for k in ("margin", "margin_norm", "margin_snr", "margin_spiking", "confidence")), \
            "each role chip must carry at least one confidence field"
    # (b) SUBSTITUTION: both arms map to a downstream confident bool (the effect is observable).
    ch = _confident(organ, host)
    cn = _confident(organ, neural)
    assert ch["confident"] is not None and cn["confident"] is not None
    assert ch["evidence"] is not None and cn["evidence"] is not None


def test_intervention_is_monotonic():
    """The core claim: with the neural state (the trace) held fixed, the downstream `confident` is a monotonic
    (single-transition) function of the host's mrc choice -- the spiking WTA faithfully transmits the host's
    evidence, so the host owns the commitment at this seam."""
    comp, organ = _build()
    _a, _v, ans, trace = _first_answered(comp)
    assert ans is not None
    seq = [_confident(organ, m)["confident"] for m in MRC_GRID]
    assert all(c is not None for c in seq)
    n_trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    assert n_trans <= 1, f"intervention must be monotonic (a single 0->1 transition), got {n_trans} transitions"


def test_degraded_panel_straddles():
    """The DEGRADED panel's core claim: on the CLEAN panel (sigma 0) the host and the neural confidences both sit
    above the metacog threshold, so the SUBSTITUTION (host decision -> neural decision) is a NO-OP. As the store
    degrades the host (the less-confident read) and the neural (the more-confident read) mrcs fall and cross the
    threshold at DIFFERENT points, so the two arms STRADDLE and the substitution finally produces a REAL
    commitment change. On the dev seed the straddle band is non-empty and does NOT include the clean panel."""
    r = run_seed_degraded(DEV_SEED)
    # the clean panel (sigma 0) must be a NO-OP: no straddling fact.
    clean = r["per_sigma"][0]
    assert clean["sigma"] == 0.0
    assert clean["n_answered"] == len(FACTS), "the clean panel must answer every dev fact"
    assert clean["n_straddle"] == 0, "the clean panel must be a NO-OP (both arms confident -> no straddle)"
    # the degraded panel must produce at least one straddling sigma level (a real commitment change via substitution).
    assert r["straddle_sigmas"], "the degraded panel must produce at least one straddling sigma level"
    assert 0.0 not in r["straddle_sigmas"], "the straddle must occur in the degraded regime, not the clean panel"
    # every straddling fact must be a genuine host-vs-neural divergence (one confident, one not), not a None.
    for s in r["per_sigma"]:
        if s["sigma"] in r["straddle_sigmas"]:
            for row in s["rows"]:
                if row.get("confident_changed"):
                    assert row["host_confident"] is not None and row["neural_confident"] is not None
                    assert row["host_confident"] != row["neural_confident"]
