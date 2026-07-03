"""CI guard for EMERGE-58 RUNG 3 -- folding the EMERGENT-REASONING fluent conversation (EMERGE-51..57) into the
flagship FLUID console under ONE gate-first no-confab MOAT. CPU/numpy, offline.

The load-bearing property is the MOAT across BOTH kinds of question: the renderer must NEVER be invoked on an abstain
(0 renders on abstains). Tests (CPU-safe, template renderer):
  1. ROUTING -- the 'can a X <verb>?' frame is SHARED; disambiguation is by TAXONOMY MEMBERSHIP (member -> reasoner;
     a fluid-known/unknown entity in the same frame -> the fluid path). A fluid knowledge question is not captured.
  2. EMERGE gate-decision ADAPTER FIDELITY -- the per-dimension gate decision matches PerDimensionConsole.ask_can.
  3. MOAT preserved -- unknown / sibling-abstain emit "I don't know" AND the render-call count on abstains is 0.
  4. EMERGE render correct -- inherit (owl->fly), cancel (penguin->walks), per-dimension (robin->breathe).
  5. MEMBERSHIP-AWARE ROUTING (the EMERGE-58 audit remediation) -- a fluid-known entity in the SHARED ability frame
     ('can a dog eat?') is answered by the fluid path (NOT falsely denied), and a genuine unknown abstains gate-first.

The heavier FluidChat build (the no-regression + membership gates) + the GPU fluent render are exercised in
`test_derisk_go` / `test_membership_*` / `test_gpu_render_smoke` (skip-if-no-ckpt), kept separate so the core
routing/moat tests run fast + always.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners._emerge58_unified_fluent_console import (
    UnifiedFluentConsole, emerge_pd_gate_decision, _adapter_matches, _emerge_probes,
    _TemplateEmergeFaculty, emerge_v3, EMERGE_FT_CKPT, _derisk_one,
)
from research.runners._emerge54_per_dimension_cancellation_derisk import _lemma


@pytest.fixture(scope="module")
def reasoner_only():
    """A unified console WITHOUT the heavy FluidChat (routing/adapter/moat only) -- fast, CPU, template renderer."""
    return UnifiedFluentConsole(seed=42, prefer_gpu_render=False, build_fluid=False)


# ---- 1. ROUTING ----------------------------------------------------------------------------------------------------
def test_router_recognises_emerge_ability_frame(reasoner_only):
    con = reasoner_only
    assert con._is_emerge_ability("can an owl fly?") == ("owl", "fly")
    assert con._is_emerge_ability("can a penguin fly?") == ("penguin", "fly")
    assert con._is_emerge_ability("can a robin breathe") == ("robin", "breathe")   # trailing '?' optional


def test_router_does_not_capture_fluid_questions(reasoner_only):
    con = reasoner_only
    # fluid knowledge questions must NOT be routed to the EMERGE reasoner (no cross-talk)
    assert con._is_emerge_ability("what does the dog eat?") is None
    assert con._is_emerge_ability("tell me about the dog") is None
    assert con._is_emerge_ability("does the dog eat meat?") is None
    assert con._is_emerge_ability("compare dog and cat") is None


def test_membership_gates_the_shared_ability_frame(reasoner_only):
    """The 'can a X <verb>?' frame is SHARED. Disambiguation is by TAXONOMY MEMBERSHIP: an observed member routes to
    the reasoner; a fluid-known entity in the SAME frame does NOT (it must go to the fluid path). This is the
    structural guard for the EMERGE-58 audit fix -- 'dog' matches the frame but is NOT a taxonomy member."""
    con = reasoner_only
    # the frame matches for both, but only the taxonomy member is a reasoner route
    assert con._is_emerge_ability("can a dog eat?") == ("dog", "eat")     # frame matches
    assert "dog" not in con.reasoner.member_idx                            # ... but 'dog' is NOT a taxonomy member
    assert "owl" in con.reasoner.member_idx                                # 'owl' IS -> routes to the reasoner


# ---- 2. ADAPTER FIDELITY -------------------------------------------------------------------------------------------
def test_adapter_fidelity_matches_ask_can(reasoner_only):
    """The per-dimension gate decision must equal PerDimensionConsole.ask_can's own decision on every probe."""
    con = reasoner_only
    matches = [_adapter_matches(con.reasoner, m, p) for (m, p, _exp) in _emerge_probes()]
    assert all(matches), f"adapter diverged on {sum(1 for x in matches if not x)}/{len(matches)} probes"


# ---- 3. MOAT (renderer never invoked on an abstain, EITHER kind of abstain) ----------------------------------------
def test_moat_renderer_never_invoked_on_abstain(reasoner_only):
    con = reasoner_only
    fac = con.faculty
    assert isinstance(fac, _TemplateEmergeFaculty)
    for (m, prop, exp) in _emerge_probes():
        if not exp.startswith("moat"):
            continue
        before = fac.render_call_count
        reply = con.turn(f"can a {m} {prop}?")
        after = fac.render_call_count
        assert reply.lower().startswith("i don't know"), reply
        assert after == before, f"renderer INVOKED on abstain for {m!r} (moat breached)"
    assert con.render_calls_on_abstain == 0


def test_moat_unknown_vs_sibling_distinct(reasoner_only):
    con = reasoner_only
    # never-observed -> "I don't know what a zzz is."
    assert con.turn("can a zzz fly?").lower().startswith("i don't know what")
    # sibling branch (owl is a bird, asked a fish ability) -> "I don't know whether an owl can swim."
    assert con.turn("can an owl swim?").lower().startswith("i don't know whether")


# ---- 4. EMERGE render correct (template surface) -------------------------------------------------------------------
def test_emerge_inherit_and_cancel_and_per_dimension(reasoner_only):
    con = reasoner_only
    assert con.turn("can an owl fly?").lower() == "yes, the owl can fly."            # INHERIT
    assert con.turn("can a penguin fly?").lower() == "no, the penguin walks."        # CANCEL (locomotion exception)
    assert con.turn("can a robin breathe?").lower() == "yes, the robin can breathe." # PER-DIMENSION inherit (no leak)


def test_emerge_render_invokes_generator_on_answer(reasoner_only):
    """On an ANSWER the renderer IS invoked (exactly once); on an abstain it is not (the gate-first contract)."""
    con = reasoner_only
    fac = con.faculty
    before = fac.render_call_count
    con.turn("can an owl fly?")
    assert fac.render_call_count == before + 1                        # answer -> renderer invoked once


# ---- 5. inflection fix carried through -----------------------------------------------------------------------------
def test_frame_aware_inflection_no_double_inflect(reasoner_only):
    con = reasoner_only
    reply = con.turn("can a penguin fly?").lower()
    assert "walkses" not in reply and "walks" in reply                # frame-aware: 'walks' stays 'walks'


# ---- adapter unit-level: the four decision branches ----------------------------------------------------------------
def test_gate_decision_branches(reasoner_only):
    con = reasoner_only
    r = con.reasoner
    d_inh = emerge_pd_gate_decision(r, "owl", "fly")
    assert d_inh["gate"] == "ANSWER" and d_inh["polarity"] == "affirm" and d_inh["svo"][2] == "fly"
    d_exc = emerge_pd_gate_decision(r, "penguin", "fly")
    assert d_exc["gate"] == "ANSWER" and d_exc["polarity"] == "negate" and d_exc["svo"][1] == "walks"
    d_sib = emerge_pd_gate_decision(r, "owl", "swim")
    assert d_sib["gate"] == "ABSTAIN" and d_sib["source"] == "moat_sibling"
    d_unk = emerge_pd_gate_decision(r, "zzz", "fly")
    assert d_unk["gate"] == "ABSTAIN" and d_unk["source"] == "moat_unknown"


# ---- 6. FULL de-risk (builds FluidChat -> exercises the no-regression + no-cross-talk gates) ------------------------
@pytest.mark.slow
def test_derisk_go_single_seed():
    """The full single-seed de-risk (builds FluidChat): adapter/render/routing/moat/membership/no-regression/
    no-cross-talk all pass. Slower (builds the flagship fluid console); template renderer keeps it CPU-safe + offline."""
    d = _derisk_one(42, build_fluid=True)
    assert d["adapter_fidelity"] == 1.0
    assert d["emerge_render_correct"] == 1.0
    assert d["members_routed"] is True
    assert d["membership_ok"] is True                                  # audit-remediation gate (no false denial)
    assert d["moat_ok"] is True and d["moat_render_calls_on_abstains"] == 0
    assert d["no_crosstalk"] is True
    assert d["fluid_ok"] is True                                       # NO fluid-path regression


@pytest.mark.slow
def test_membership_aware_routing_no_false_denial():
    """REGRESSION GUARD for the EMERGE-58 audit defect (builds FluidChat): a fluid-known entity in the SHARED ability
    frame must NOT be falsely denied. Pre-fix, the frame-only router routed 'can a dog eat?' to the reasoner, which
    denied 'I don't know what a dog is' -- in the SAME session the fluid path answers 'The dog eats meat.' Post-fix,
    membership-aware routing sends 'dog' to the fluid path (answered), while a genuine unknown still abstains
    gate-first with the fluent generator NOT invoked."""
    con = UnifiedFluentConsole(seed=42, prefer_gpu_render=False, build_fluid=True)
    fac = con.faculty
    # (a) a fluid-known entity in the ability frame is ANSWERED, not falsely denied; the EMERGE 21M is NOT invoked
    before = fac.render_call_count
    dog = con.turn("can a dog eat?").lower()
    assert not dog.startswith("i don't know what a dog"), f"FALSE DENIAL regressed: {dog!r}"
    assert ("meat" in dog or "eat" in dog), f"fluid-known entity not answered: {dog!r}"
    assert fac.render_call_count == before                             # the EMERGE generator was NOT stolen into it
    # (b) consistency: the same fact via the fluid yes/no path agrees (no self-contradiction)
    assert con.turn("does the dog eat meat?").lower().startswith("yes")
    # (c) a genuine unknown still abstains gate-first (fluent generator NOT invoked)
    before = fac.render_call_count
    zzz = con.turn("can a zzz fly?").lower()
    assert zzz.startswith("i don't know")
    assert fac.render_call_count == before                             # the moat holds -- generator NOT invoked


# ---- 7. GPU fluent render smoke (skip-if-no-ckpt/torch) -------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(EMERGE_FT_CKPT), reason="EMERGE-57 re-fine-tuned ckpt absent")
def test_gpu_render_smoke():
    """With the EMERGE-57 ckpt present + torch/CUDA, the fluent 21M renders EMERGE answers behind the SAME gate-first
    loop and the moat holds (0 renders on abstains). If torch/CUDA is missing the console falls back to template ->
    skip (the render-path is exercised by test_emerge57 already)."""
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available")
    from research.runners._emerge58_unified_fluent_console import _gpu_render_smoke
    res = _gpu_render_smoke(42)
    if not res.get("ran"):
        pytest.skip(res.get("note", "GPU faculty unavailable"))
    assert res["render_on_abstain"] == 0                              # the moat held on the REAL model
    # inherit renders a "yes", exception a "no", abstains say "i don't know"
    by = {(r["member"], r["prop"]): r for r in res["records"]}
    assert not by[("zzz", "fly")]["model_invoked"]                    # abstain -> model NOT invoked
