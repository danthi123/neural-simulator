"""FHRR-on-bridge layer (b) b.1: the parallel RF phasor composer does who/what Q&A + abstention (the no-confab moat)
through the bridge's RF complex-synapse bind/bundle/unbind. The de-risk GATE for layer (b). See
docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md.
"""
import pytest

from research.runners.rf_phasor_composer import RFPhasorComposer


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_who_what_abstain(seed):
    """b.1 GATE (multi-seed): store 2 SVO facts; who/what queries retrieve the right role; an absent cue ABSTAINS
    (returns None) -- the no-confab moat preserved on the RF phasor substrate."""
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")

    # who <action> <patient>? -> agent
    assert comp.query_agent("go", "north") == "dog"
    assert comp.query_agent("run", "south") == "cat"
    # what does <agent> <action>? -> patient
    assert comp.query_patient("dog", "go") == "north"
    assert comp.query_patient("cat", "run") == "south"

    # abstention (the no-confab moat): no stored fact matches these cues -> None
    assert comp.query_agent("go", "south") is None        # action=go but patient=south is not a stored pair
    assert comp.query_patient("dog", "run") is None        # agent=dog but action=run is not a stored pair


def test_rf_phasor_composer_store_grows_unseen_relation_word():
    """Regression (2026-08-26, `KeyError: 'confirm'` crash-looped the GNW coincidence-integrator pool lane 25+
    times over 3+ hours). `store()` with an ACTION filler that was never part of the composer's initial vocab
    must NOT crash. Before the 2026-08-12 runtime-growth fix (commit 5b2d1d7c3e), `_filler_phases` indexed
    `self.concepts[filler]` directly -- a fact stored with a brand-new relation word (e.g. the GNW de-risk's
    second-organ relation 'confirm', never in DEFAULT_VOCAB) raised an unhandled KeyError. `_filler_phases` must
    instead allocate a fresh deterministic code for the unseen word (RUNTIME GROWTH) so the fact stores and the
    word is immediately retrievable -- the same mechanism that lets the brain acquire a new word by being told a
    fact containing it. Pins the behavior so a stale/regressed deployment of this file is caught by the suite,
    not discovered by a pool node crash-looping for hours."""
    comp = RFPhasorComposer(seed=42, D=64, period=200)
    assert "confirm" not in comp.concepts                        # genuinely never-seen before this store()
    comp.store("dog", "confirm", "cat")                          # must NOT raise KeyError
    assert "confirm" in comp.concepts                              # grown into the concept dict
    assert "confirm" in comp.words                                 # and joined the cleanup/scan codebook
    assert comp.query_patient("dog", "confirm") == "cat"          # + immediately retrievable
    assert comp.query_agent("confirm", "cat") == "dog"
    assert comp.query_patient("dog", "go") is None                # no-confab moat: unrelated cue still abstains


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_negation_yesno(seed):
    """b.2: a bound AFFIRM/NEGATE polarity tag -> ask_yes_no returns yes/no via the unbound tag; 'unknown'
    (abstention) when no stored fact matches. A 4-role bind (SVO + polarity) through the RF complex synapses."""
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "go", "north", polarity="AFFIRM")
    comp.store("cat", "go", "south", polarity="NEGATE")

    assert comp.ask_yes_no("dog", "go", "north") == "yes"
    assert comp.ask_yes_no("cat", "go", "south") == "no"
    # abstention: no stored fact matches this full SVO cue
    assert comp.ask_yes_no("dog", "go", "south") == "unknown"


def test_rf_trace_exposes_cleanup_margin_and_source_fact():
    comp = RFPhasorComposer(seed=42, D=64, period=200, trace=True)
    comp.store("dog", "go", "north", polarity="AFFIRM")

    assert comp.query_patient("dog", "go") == "north"
    tr = comp.last_trace
    assert tr["matched_fact_index"] == 0
    assert tr["source_fact"]["patient"] == "north"
    patient = next(ch for ch in tr["roles"] if ch["role"] == "patient" and not ch["cue"])
    assert patient["word"] == "north"
    assert patient["confidence"] is not None
    assert patient["runner_word"] != "north"
    assert patient["margin"] >= 0.0
    assert 0.0 <= patient["conflict"] <= 1.0

    assert comp.ask_yes_no("dog", "go", "north") == "yes"
    pol = next(ch for ch in comp.last_trace["roles"] if ch["role"] == "polarity")
    assert pol["word"] in {"AFFIRM", "NEGATE"}
    assert pol["margin"] >= 0.0


def test_rf_trace_margin_norm_is_peak_relative_and_additive():
    """2026-09-01 (board #94 calibration-at-scale): `margin_norm` is an ADDITIVE field on top of `margin` -- the
    raw `top_raw - runner_raw` cosine difference `margin` reports stays UNCHANGED (self_schema_honesty.py and the
    test above depend on it byte-identical), while `margin_norm` = `(top_r - runner_r) / (top_r + eps)` (both
    rectified >= 0 first) is the SAME peak-normalized formula `OneBrainComposer._margin` uses -- the field
    `metacog_production_organ.mean_role_confidence` now prefers so the SAME ROLE_CONF_LO/HI band applies across
    composer types without a per-codebook-size retune (see that function's own docstring)."""
    comp = RFPhasorComposer(seed=42, D=64, period=200, trace=True)
    comp.store("dog", "go", "north", polarity="AFFIRM")

    assert comp.query_patient("dog", "go") == "north"
    tr = comp.last_trace
    patient = next(ch for ch in tr["roles"] if ch["role"] == "patient" and not ch["cue"])
    assert patient["margin_norm"] is not None
    assert 0.0 <= patient["margin_norm"] <= 1.0
    top_r = max(patient["winner_score_raw"], 0.0)
    runner_r = max(patient["runner_score_raw"], 0.0)
    expected = (top_r - runner_r) / (top_r + 1e-9) if top_r > 0.0 else 0.0
    assert patient["margin_norm"] == pytest.approx(expected, abs=1e-9)
    # margin_norm is a DIFFERENT number from the raw margin (not a renamed duplicate) whenever the peak isn't 1.0.
    if top_r not in (0.0, 1.0):
        assert patient["margin_norm"] != pytest.approx(patient["margin"], abs=1e-9)


def test_mean_role_confidence_prefers_margin_norm_over_raw_margin():
    """`metacog_production_organ.mean_role_confidence` must read the peak-normalized `margin_norm` when present
    (an RFPhasorComposer/ShardedPhasorStore-sourced trace) instead of the raw `margin` field -- and stay
    byte-identical (falls through to `margin`) for any role chip that does not carry `margin_norm` (e.g.
    OneBrainComposer's own trace, whose `margin` field IS already the normalized ratio)."""
    from research.runners.metacog_production_organ import mean_role_confidence

    # a role carrying BOTH fields (the RFPhasorComposer/ShardedPhasorStore shape) -> margin_norm wins.
    activity_with_norm = {"roles": [{"role": "patient", "margin": 0.05, "margin_norm": 0.55, "confidence": 0.9}]}
    assert mean_role_confidence(activity_with_norm) == pytest.approx(0.55)

    # a role with only the legacy `margin` field (e.g. OneBrainComposer) -> unchanged fallback behavior.
    activity_legacy = {"roles": [{"role": "patient", "margin": 0.55, "confidence": 0.9}]}
    assert mean_role_confidence(activity_legacy) == pytest.approx(0.55)


def test_mean_role_confidence_prefers_scale_invariant_margin_snr():
    """2026-09-02 (board #94/#108 R3): `margin_snr` -- the composer's scale-INVARIANT winner-vs-bulk z-score --
    is preferred OVER `margin_norm` when a chip carries it, mapped linearly through the 15k reference anchors
    SNR_LO/SNR_HI onto the ROLE_CONF band, so a clean recall reads the same confidence at any vocab scale (the
    100k recalibration: margin_norm keys on the runner-up, which inflates as sqrt(2 ln V) with codebook size, and
    dragged the 100k clean read below the confident floor). The clean-recall anchor SNR_HI maps to ROLE_CONF_HI
    and the degraded-recall anchor SNR_LO maps to ROLE_CONF_LO; a chip WITHOUT margin_snr still falls through to
    margin_norm (backward compatible -- the OneBrainComposer buffer path is byte-identical)."""
    from research.runners.metacog_production_organ import (
        mean_role_confidence, SNR_LO, SNR_HI, ROLE_CONF_LO, ROLE_CONF_HI)

    # snr present -> preferred over margin_norm; the two 15k anchors map exactly to the band edges.
    hi = {"roles": [{"role": "patient", "margin_snr": SNR_HI, "margin_norm": 0.99, "margin": 0.99}]}
    lo = {"roles": [{"role": "patient", "margin_snr": SNR_LO, "margin_norm": 0.99, "margin": 0.99}]}
    assert mean_role_confidence(hi) == pytest.approx(ROLE_CONF_HI)
    assert mean_role_confidence(lo) == pytest.approx(ROLE_CONF_LO)

    # a chip WITHOUT margin_snr still falls through to margin_norm (backward compatible).
    assert mean_role_confidence({"roles": [{"margin_norm": 0.55, "margin": 0.05}]}) == pytest.approx(0.55)


def test_rf_source_monitor_echo_checks_candidate_without_source_fact():
    comp = RFPhasorComposer(
        seed=42,
        D=64,
        period=200,
        trace=True,
        enable_source_monitor=True,
        source_monitor_D=64,
    )
    comp.store("dog", "go", "north", polarity="AFFIRM")

    ok = comp.source_consistency_record(kind="what_does", cue=("dog", "go"), raw_answer="north")
    bad = comp.source_consistency_record(kind="what_does", cue=("dog", "go"), raw_answer="south")

    assert ok["available"] is True
    assert ok["source"] == "rf_independent_source_echo"
    assert ok["source_expected_answer"] == "north"
    assert ok["source_consistent"] is True
    assert bad["source_expected_answer"] == "north"
    assert bad["source_consistent"] is False
    assert "source_fact" not in ok


def test_plastic_source_monitor_requires_experience_and_checks_live_candidate():
    comp = RFPhasorComposer(
        seed=42,
        D=64,
        period=200,
        trace=True,
        enable_plastic_source_monitor=True,
        plastic_source_config={
            "n_banks": 4,
            "proposition_neurons_per_bank": 2048,
            "support_threshold": 0.25,
        },
    )
    comp.store("dog", "go", "north", polarity="AFFIRM")

    before = comp.plastic_source_consistency_record(
        kind="what_does", cue=("dog", "go"), raw_answer="north"
    )
    comp.observe_source_event(kind="what_does", cue=("dog", "go"), candidate="north")
    learned = comp.plastic_source_consistency_record(
        kind="what_does", cue=("dog", "go"), raw_answer="north"
    )
    wrong = comp.plastic_source_consistency_record(
        kind="what_does", cue=("dog", "go"), raw_answer="south"
    )

    assert before["available"] is False
    assert before["source_consistent"] is None
    assert learned["available"] is True
    assert learned["source_consistent"] is True
    assert wrong["source_consistent"] is False
    assert learned["support"] > wrong["support"] + 0.10
    assert "source_expected_answer" not in learned
    assert "matched_source_index" not in learned


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_one_attribute(seed):
    """b.3a: a 1-attribute entity ('big apple') -- the ATTRIBUTE role-tag binding RESOLVES (adjective + noun both
    decoded from the RF unbind). A 4-role bind (agent/action/patient/attribute) through the RF complex synapses.
    (2-attribute is the documented K=5-load BOUNDARY, not asserted here -- carries over from the +-1 composer.)"""
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "look", ("big", "apple"))
    comp.store("cat", "go", "river")

    assert comp.query_patient("dog", "look") == "big apple"   # attribute resolves
    assert comp.query_patient("cat", "go") == "river"           # plain patient still works
    assert comp.query_agent("look", "apple") == "dog"           # query on the noun (patient)
    # abstention
    assert comp.query_patient("cat", "look") is None


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_clause(seed):
    """b.3b: a recursive CLAUSE as a filler ('dog look (cat go north)') -- the clause's bound composite is the
    patient filler; the query renders the nested SVO, decoded from the RF unbind (double-nesting through the RF
    complex synapses). D=128 for the nesting SNR."""
    from research.runners.rf_phasor_composer import Clause
    comp = RFPhasorComposer(seed=seed, D=128, period=200)
    comp.store("dog", "look", Clause("cat", "go", "north"))
    comp.store("river", "stop", "apple")

    assert comp.query_patient("dog", "look") == "cat go north"   # the nested clause renders
    assert comp.query_patient("river", "stop") == "apple"          # a flat patient still works


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_dialogue(seed):
    """b.4: dialogue planning -- elaborate(topic) brings up an on-topic associate via the dlPFC spiking
    content-selection over the association graph built from the RF composer's facts; None when unconnected.
    GPU-only: the reused SpikingSpreadingController (dlPFC) has a numpy-backend IndexError in that component."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("dlPFC SpikingSpreadingController is GPU-validated (numpy-backend IndexError in that component)")
    comp = RFPhasorComposer(seed=seed, D=64, period=200)
    comp.store("dog", "go", "north")
    comp.store("dog", "run", "south")

    assoc = comp.elaborate("dog")
    assert assoc in {"go", "north", "run", "south"}, f"elaborate('dog')={assoc} not an on-topic associate"
    assert comp.elaborate("apple") is None    # apple is in no stored fact -> unconnected (abstention)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_full_matrix_at_scale(seed):
    """Layer (c1): the full capability matrix at a larger fact set (5 facts) -- who/what Q&A, abstention, negation,
    one-attribute, generation -- multi-seed, mirroring the rate-composer's capability bar. D=96 for the 5-fact load."""
    comp = RFPhasorComposer(seed=seed, D=96, period=200)
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")
    comp.store("river", "look", ("big", "apple"))           # one-attribute
    comp.store("dog", "stop", "east", polarity="AFFIRM")
    comp.store("cat", "look", "west", polarity="NEGATE")

    # who/what (action disambiguates the two dog/cat facts)
    assert comp.query_agent("go", "north") == "dog"
    assert comp.query_patient("cat", "run") == "south"
    assert comp.query_patient("river", "look") == "big apple"      # one-attribute resolves at scale
    # generation (river is a unique agent)
    assert comp.render_fact("river") == "river look big apple"
    # negation / yes-no
    assert comp.ask_yes_no("dog", "stop", "east") == "yes"
    assert comp.ask_yes_no("cat", "look", "west") == "no"
    # abstention (the no-confab moat) at scale
    assert comp.query_agent("go", "south") is None
    assert comp.ask_yes_no("dog", "go", "west") == "unknown"
    assert comp.render_fact("apple") is None                       # apple is not an agent


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_production_scale(seed):
    """Layer (c-scale): a larger fact set (10 facts) -- who/what Q&A retrieves the RIGHT fact and the no-confab moat
    does NOT false-match among 10 facts. The key production-scale risk: spurious matches as the KB grows. D=128."""
    comp = RFPhasorComposer(seed=seed, D=128, period=200)
    facts = [
        ("dog", "go", "north"), ("cat", "run", "south"), ("dog", "look", "east"),
        ("river", "stop", "west"), ("apple", "go", "south"), ("cat", "look", "north"),
        ("dog", "run", "west"), ("river", "go", "east"), ("apple", "stop", "north"),
        ("cat", "go", "west"),
    ]
    for a, v, p in facts:
        comp.store(a, v, p)

    # retrieval: the (action, patient) / (agent, action) cue disambiguates the right fact among 10
    assert comp.query_agent("go", "north") == "dog"
    assert comp.query_agent("run", "south") == "cat"
    assert comp.query_patient("river", "stop") == "west"
    assert comp.query_patient("apple", "go") == "south"
    # abstention (the no-confab moat) at 10 facts: cues that match NO stored fact -> None (no false match)
    assert comp.query_agent("stop", "south") is None       # no fact has action=stop AND patient=south
    assert comp.query_patient("dog", "stop") is None         # dog never stops


def test_brain_agent_with_rf_composer():
    """(c2a) end-to-end: BrainConversationalAgent using the FHRR-on-bridge RFPhasorComposer (opt-in injection) --
    the Hebbian parser comprehends sentences and feeds the RF composer; hear -> store, query, abstain. Validates
    the switch works at the AGENT level (the rate-coded composer stays the default; this is the explicit opt-in).
    GPU-only: the Hebbian BridgeParser is GPU-validated (numpy-backend KeyError in the parser bridge)."""
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        pytest.skip("BridgeParser is GPU-validated (numpy-backend KeyError in the parser bridge)")
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    rf = RFPhasorComposer(seed=42, D=96, period=200)
    agent = BrainConversationalAgent(seed=42, composer=rf)
    agent.hear("dog go north")
    agent.hear("cat run south")

    # the parser comprehended each sentence and stored it in the RF composer
    assert agent.composer.query_agent("go", "north") == "dog"
    assert agent.composer.query_patient("cat", "run") == "south"
    assert agent.composer.query_agent("go", "south") is None    # abstention -- the no-confab moat, end to end


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_two_attribute(seed):
    """2-attribute ('big hot apple') -- the documented K=5-load BOUNDARY of the rate-coded +-1 composer (the noun
    degrades at the 5-binding edge ~0.93). The FHRR phasor substrate has better capacity (SNR ~ 2N/M, a D dial):
    at D=256 it RESOLVES multi-seed -- the boundary is LIFTED by the substrate, no F=3 resonator needed."""
    comp = RFPhasorComposer(seed=seed, D=256, period=200)
    comp.store("dog", "look", (("big", "hot"), "apple"))   # two-attribute entity (5-binding fact)
    comp.store("cat", "go", "river")
    assert set(comp.query_patient("dog", "look").split()) == {"big", "hot", "apple"}   # BOTH adjectives + the noun
    assert comp.query_patient("cat", "go") == "river"                                    # flat patient still works
    assert comp.query_patient("dog", "go") is None                                       # abstention (no-confab moat)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_spiking_cleanup_parity(seed):
    """Cheat-B conversion (opt-in): enable_spiking_cleanup routes cleanup through the FULLY-on-bridge spiking path
    -- the matched FILTER is the complex-synapse matvec (the same op as unbind; |c_k| read off the membrane) and the
    SELECTION is a spiking Izhikevich WTA (argmax-over-firing, the NEF-cleanup structure). It must give the SAME
    answers as the numpy-argmax default, AND preserve the no-confab moat (abstention). D=256 (the cleanup codebook
    is dense; the D-dial clears it, as for two-attribute)."""
    cn = RFPhasorComposer(seed=seed, D=256, period=200, enable_spiking_cleanup=False)
    cs = RFPhasorComposer(seed=seed, D=256, period=200, enable_spiking_cleanup=True)
    for a, v, p in [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]:
        cn.store(a, v, p); cs.store(a, v, p)
    for v, p, a in [("go", "north", "dog"), ("run", "south", "cat"), ("look", "apple", "river")]:
        assert cs.query_agent(v, p) == cn.query_agent(v, p) == a            # selection in spikes == numpy
        assert cs.query_patient(a, v) == cn.query_patient(a, v) == p
    assert cs.query_agent("go", "river") is None                            # no-confab moat preserved


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_substrate_store_parity(seed):
    """Cheat-C conversion (opt-in): enable_substrate_store holds each fact's bound composite in per-fact SUBSTRATE
    weights (a trigger->readout complex-synapse bridge, the Crawford-Eliasmith weight-store = Hebb memory-in-weights),
    retrieved by firing the trigger -> phase readout -- NOT a numpy kb array. It must give the SAME answers as the
    numpy-kb default AND preserve the no-confab moat (abstention)."""
    cn = RFPhasorComposer(seed=seed, D=128, period=200, enable_substrate_store=False)
    cs = RFPhasorComposer(seed=seed, D=128, period=200, enable_substrate_store=True)
    for a, v, p in [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]:
        cn.store(a, v, p); cs.store(a, v, p)
    for v, p, a in [("go", "north", "dog"), ("run", "south", "cat"), ("look", "apple", "river")]:
        assert cs.query_agent(v, p) == cn.query_agent(v, p) == a            # memory in the substrate == numpy
        assert cs.query_patient(a, v) == cn.query_patient(a, v) == p
    assert cs.query_agent("go", "river") is None                           # no-confab moat preserved


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_grounded_codes_interface(seed):
    """Cheat-A conversion (opt-in INTERFACE): grounded_codes={word: phases[D]} overrides the random rng.uniform codes
    for those words. The composer must USE the provided codes and still do who/what Q&A + abstention. This guards the
    grounding INTERFACE; the genuine V1-Gabor-grounded validation (codes from REAL sensory features, 6/6 multi-seed)
    is research/findings/raw/_phase3_grounded_codes_derisk.py. The FULL grounding (real object images + abstract-
    concept grounding) is the documented embodied-cognition boundary -- this is the interface, not full semantics."""
    import numpy as np
    words = ["dog", "cat", "go", "run", "north", "south", "river", "apple"]
    g_rng = np.random.default_rng(seed + 4242)
    grounded = {w: g_rng.uniform(0.0, 1.0, 128) for w in words}             # externally-provided codes (de-risk uses V1)
    comp = RFPhasorComposer(seed=seed, D=128, period=200, grounded_codes=grounded)
    for w in words:
        assert np.allclose(comp.concepts[w], grounded[w])                   # the composer USED the provided codes
    comp.store("dog", "go", "north"); comp.store("cat", "run", "south")
    assert comp.query_agent("go", "north") == "dog"                         # Q&A works on the provided codes
    assert comp.query_patient("cat", "run") == "south"
    assert comp.query_agent("go", "south") is None                         # no-confab moat preserved


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_local_reciprocal_unbind_byte_identical(seed):
    """FHRR-B mechanism 1 (opt-in): local_reciprocal_unbind derives the UNBIND synapses from the BIND synapses by a
    one-time LOCAL reciprocal-conjugate wiring rule (a per-synapse quadrature flip) at construction, instead of the
    host computing conj(role) and injecting it per op. It must give answers BYTE-IDENTICAL to the host-conj default
    on the full who/what matrix AND the no-confab abstentions (conj per component IS the per-synapse rule). This is
    the brain-based-purity / neuromorphic-port residual: the unbind structure becomes a host-free device config.
    De-risk: research/findings/2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md."""
    import numpy as np
    from research.runners.rf_phasor_composer import ROLES
    cn = RFPhasorComposer(seed=seed, D=96, period=200, local_reciprocal_unbind=False)   # host-conj (default/legacy)
    cl = RFPhasorComposer(seed=seed, D=96, period=200, local_reciprocal_unbind=True)     # local reciprocal rule

    # (a) the unbind connectivity weights are bit-for-bit conj(bind) for every role.
    for role in ROLES:
        zr_conj = np.conj(cn._to_phasor(cn.roles[role]))
        legacy = [(cn.D + k, k, zr_conj[k]) for k in range(cn.D)]
        local = cl._reciprocal_conjugate(cl._bind_conns(cl.roles[role]))
        assert legacy == local, f"local rule != conj(bind) for role {role}"

    # (b) the full who/what matrix + abstentions are byte-identical.
    facts = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", ("big", "apple")),
             ("dog", "stop", "east"), ("cat", "look", "west")]
    for c in (cn, cl):
        c.store("dog", "go", "north"); c.store("cat", "run", "south")
        c.store("river", "look", ("big", "apple"))
        c.store("dog", "stop", "east", polarity="AFFIRM")
        c.store("cat", "look", "west", polarity="NEGATE")
    assert cl.query_agent("go", "north") == cn.query_agent("go", "north") == "dog"
    assert cl.query_patient("river", "look") == cn.query_patient("river", "look") == "big apple"
    assert cl.render_fact("river") == cn.render_fact("river") == "river look big apple"
    assert cl.ask_yes_no("dog", "stop", "east") == cn.ask_yes_no("dog", "stop", "east") == "yes"
    # the no-confab moat: identical abstentions
    assert cl.query_agent("go", "south") is None and cn.query_agent("go", "south") is None
    assert cl.ask_yes_no("dog", "go", "west") == cn.ask_yes_no("dog", "go", "west") == "unknown"

    # (c) substrate-purity: with the flag ON, the unbind-structure build issues ZERO np.conj calls (the unbind
    #     connectivity comes solely from the local rule over the bind connectivity).
    composite = cl.kb[0][1]
    n_conj = {"n": 0}
    orig = np.conj
    np.conj = lambda x, _c=n_conj, _o=orig: (_c.__setitem__("n", _c["n"] + 1), _o(x))[1]
    try:
        cl._unbind_phases(composite, "agent")
    finally:
        np.conj = orig
    assert n_conj["n"] == 0, "flag ON must not call np.conj in the unbind-structure build"


def test_local_conj_equals_np_conj_bitforbit():
    """FHRR-B mechanism 1, the load-bearing equivalence (CPU, no bridge): _local_conj (the per-component quadrature
    flip used by both RFPhasorComposer and OneBrainComposer's unbind-structure build) == np.conj BIT-FOR-BIT for a
    unit phasor. This is what guarantees the local-rule answer-identity is backend-independent (so the GPU-only
    OneBrainComposer production path is byte-identical by the same argument the CPU smoke shows)."""
    import numpy as np
    for seed in (0, 1, 7):
        z = np.exp(2j * np.pi * np.random.default_rng(seed).uniform(0.0, 1.0, 256))
        lc = RFPhasorComposer._local_conj(z)
        assert np.array_equal(lc, np.conj(z)), "local conj must equal np.conj bit-for-bit for a unit phasor"
        # and _reciprocal_conjugate over a bind conn list equals the legacy conj exactly
        conns = [(256 + k, k, z[k]) for k in range(256)]
        assert RFPhasorComposer._reciprocal_conjugate(conns) == [(256 + k, k, np.conj(z)[k]) for k in range(256)]


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_rf_phasor_composer_cleanup_codebook_local_conj_byte_identical(seed):
    """FHRR-B cleanup-codebook residual (the SAME local rule extended to the cleanup/matched-filter codebook): with
    local_reciprocal_unbind ON, the cleanup codebook is derived from the concept codes by the per-component
    quadrature-flip (_cleanup_conj) instead of the host np.conj. It must give answers BYTE-IDENTICAL to the host-conj
    default on the full who/what matrix AND the no-confab abstentions (conj per component IS the per-synapse rule).
    Combined with Mechanism 1's unbind rule, a FULL store+query build then issues ZERO np.conj calls TOTAL -> the WHOLE
    bind+cleanup structure is host-free. De-risk: research/findings/2026-06-20-FHRR-B-cleanup-codebook-local-conj.md."""
    import numpy as np
    cn = RFPhasorComposer(seed=seed, D=96, period=200, local_reciprocal_unbind=False)   # host-conj (default/legacy)
    cl = RFPhasorComposer(seed=seed, D=96, period=200, local_reciprocal_unbind=True)     # local reciprocal rule

    # (a) the cleanup codebook weights are bit-for-bit conj(concept) for every concept (main vocab + polarity tags).
    for w in list(cn.words) + cn.pol_words:
        legacy = np.conj(cn._to_phasor(cn.concepts[w]))
        local = cl._cleanup_conj(cl._to_phasor(cl.concepts[w]))
        assert np.array_equal(legacy, local), f"local cleanup rule != conj(concept) for '{w}'"

    # (b) the full who/what matrix + abstentions are byte-identical.
    for c in (cn, cl):
        c.store("dog", "go", "north"); c.store("cat", "run", "south")
        c.store("river", "look", ("big", "apple"))
        c.store("dog", "stop", "east", polarity="AFFIRM")
        c.store("cat", "look", "west", polarity="NEGATE")
    assert cl.query_agent("go", "north") == cn.query_agent("go", "north") == "dog"
    assert cl.query_patient("river", "look") == cn.query_patient("river", "look") == "big apple"
    assert cl.render_fact("river") == cn.render_fact("river") == "river look big apple"
    assert cl.ask_yes_no("dog", "stop", "east") == cn.ask_yes_no("dog", "stop", "east") == "yes"
    # the no-confab moat: identical abstentions
    assert cl.query_agent("go", "south") is None and cn.query_agent("go", "south") is None
    assert cl.ask_yes_no("dog", "go", "west") == cn.ask_yes_no("dog", "go", "west") == "unknown"

    # (c) substrate-purity: with the flag ON, a FULL store+query build (unbind STRUCTURE + cleanup CODEBOOK) issues
    #     ZERO np.conj calls TOTAL -> the whole bind+cleanup structure is host-free (the headline).
    cp = RFPhasorComposer(seed=seed, D=64, period=200, local_reciprocal_unbind=True)
    cp.store("dog", "go", "north"); cp.store("cat", "run", "south"); cp.store("river", "look", "apple")
    n_conj = {"n": 0}
    orig = np.conj
    np.conj = lambda x, _c=n_conj, _o=orig: (_c.__setitem__("n", _c["n"] + 1), _o(x))[1]
    try:
        cp.query_agent("go", "north")        # batched scan: _unbind_all_phases + _cleanup_all
        cp.query_patient("river", "look")    # + _render_filler -> unbind -> _cleanup
        cp.query_agent("go", "south")        # the moat (abstains)
    finally:
        np.conj = orig
    assert n_conj["n"] == 0, "flag ON must not call np.conj in a full store+query build (bind+cleanup structure)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
