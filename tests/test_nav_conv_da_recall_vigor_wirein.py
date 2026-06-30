"""CI guard for the PRODUCTION wire-in of the DA-gated RECALL-VIGOR mechanism into MergedNavConvAgent.

The validated mechanism (research/runners/_tier2_6_da_recall_vigor_derisk.py, GPU 6-seed GO on the merged bridge with
the real spiking SNc): a value/salience PRIOR carried by the shared spiking dopamine RE-RANKS WHICH familiarity-cleared
stored fact is RETRIEVED from the conversational composer's cue-match scan -- `score'_i = match_i + beta*(DA-DA_tonic)*value_i`,
argmax wins. It re-ranks ONLY within the familiarity-gated (exact-cue-decode-match) candidate set => MOAT-SAFE BY
CONSTRUCTION (it can NEVER manufacture a match for an unstored cue).

This guard pins the WIRE-IN (the agent's `what_does`/`who_does` routing recall through the value prior behind the
opt-in `enable_da_recall_vigor`), NOT the de-risk class itself (that is pinned by tests/test_tier2_6_da_recall_vigor.py):
  - DEFAULT-OFF == byte-identical: with `enable_da_recall_vigor=False` the agent's recall is the composer's plain
    query (first-match / content-only), value tags ignored;
  - opt-in value-conflicted recall picks the HIGH-value fact at high DA (where the cue alone matches BOTH facts);
  - DA-LESION (DA held at the tonic reference): the value-driven pick COLLAPSES to the first-match (value-independent);
  - the no-confab MOAT (HARD): an UNSTORED cue abstains (-> None) at EVERY DA level, OFF and ON.

CPU/numpy: the MergedNavConvAgent's Hebbian parser/dlPFC are GPU-only, so we exercise the agent's wire-in METHODS
against a manually-attached CPU `RFPhasorComposer` (via object.__new__ + the agent's own value-prior helpers) -- the
EXACT code path the GPU agent runs (the recall methods touch only `self.composer`, `self._fact_values`,
`self.enable_da_recall_vigor`, and the shared-DA read, which the test overrides with a scalar). The decisive multi-seed
claim is the GPU 6-seed on the merged bridge with the REAL shared dopamine (the de-risk runner's default path).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.nav_conv_merged_bridge import MergedNavConvAgent  # noqa: E402


# A value-conflicted who/what cue: two facts SHARE (agent, action) but differ in patient + value. Cueing on
# what_does("dog","go") matches BOTH (both clear familiarity); only the value/DA prior disambiguates which patient is
# recalled. (The de-risk cues on the shared AGENT alone; the production what_does cue is the joint (agent, action), so
# both stored facts share BOTH the agent AND the action here.)
VOCAB = ["dog", "cat", "go", "run", "come", "stop", "north", "south", "east", "west", "river", "look"]
FACT_LO = ("dog", "go", "south")    # stored FIRST -> kb index 0 = the value-INDEPENDENT first-match baseline
FACT_HI = ("dog", "go", "north")    # the intended HIGH-value memory (shares agent=dog AND action=go)
UNSTORED_AGENT = "cat"              # an agent NEVER stored -> the moat probe
DA_TONIC = 0.5
DA_HIGH = 0.84
BETA = 8.0


def _make_agent(enable, da_value, seed=42, D=64):
    """Build a MergedNavConvAgent WITHOUT its GPU __init__ (object.__new__), attach a CPU RFPhasorComposer + the
    minimal wire-in state, and store the two value-conflicted facts (LO then HI). `da_value` is the scalar the agent's
    shared-DA read returns (overrides `_da_recall_dopamine`). Exercises the agent's REAL value-prior recall methods."""
    a = object.__new__(MergedNavConvAgent)
    a.composer = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB)
    a.enable_da_salience_gate = False                 # isolate the recall-vigor wire-in (the salience gate is separate)
    a.enable_da_recall_vigor = bool(enable)
    a._da_recall_beta = float(BETA)
    a._da_recall_baseline = float(DA_TONIC)
    a._da_recall_value_default = 1.0
    a._fact_values = []
    a._da_recall_view = None
    # override the shared-DA read with a constant (the GPU agent reads the live spiking SNc here)
    a._da_recall_dopamine = lambda: float(da_value)
    # store LO first (kb 0) then HI (kb 1), tagging each fact's value via the agent's store hook
    a._store_fact_value(FACT_LO[0], FACT_LO[1], FACT_LO[2], value=0.0)
    a._store_fact_value(FACT_HI[0], FACT_HI[1], FACT_HI[2], value=1.0)
    return a


def test_default_off_is_byte_identical_first_match():
    """DEFAULT-OFF: with enable_da_recall_vigor=False the agent's what_does is the composer's plain query (first-match
    / content-only), regardless of the value tags or DA -- byte-identical to the un-wired recall path."""
    a_off = _make_agent(enable=False, da_value=DA_HIGH)
    # the plain composer query (no value prior): the first stored matching fact's patient
    plain = a_off.composer.query_patient("dog", "go")
    assert a_off.what_does("dog", "go") == plain, "default-off recall must equal the composer's plain query"
    # and the moat is the composer's own abstention (unchanged)
    assert a_off.what_does("cat", "go") is None


def test_value_prior_picks_high_value_at_high_da():
    """OPT-IN, high DA: the value prior re-ranks the (dog,go)-gated candidate set -> the HIGH-value fact's patient
    ('north') is recalled, OVERRIDING the first-match baseline ('south')."""
    a_on = _make_agent(enable=True, da_value=DA_HIGH)
    first_match = a_on.composer.query_patient("dog", "go")       # the value-independent baseline (kb-order = LO)
    assert first_match == FACT_LO[2], f"first-match baseline should be the LO patient, got {first_match!r}"
    assert a_on.what_does("dog", "go") == FACT_HI[2], "high-DA value prior should recall the HI-value patient"


def test_da_lesion_collapses_to_first_match():
    """DA-LESION (DA held at the tonic reference -> beta*(DA-baseline)*value -> 0): the value-driven pick collapses to
    the first-match (value-independent) patient -- the prior is the load-bearing signal."""
    a_lesion = _make_agent(enable=True, da_value=DA_TONIC)
    assert a_lesion.what_does("dog", "go") == a_lesion.composer.query_patient("dog", "go") == FACT_LO[2], \
        "under DA-lesion the recall must collapse to the first-match (value-independent) patient"


def test_moat_holds_off_and_on_at_every_da():
    """The no-confab MOAT (HARD): an UNSTORED cue (cat) abstains (-> None) at EVERY DA level, OFF and ON. The value
    prior re-ranks ONLY within the familiarity-gated set, so an unstored cue has nothing to re-rank (empty set ->
    abstain)."""
    for enable in (False, True):
        for da in (DA_TONIC, DA_HIGH):
            a = _make_agent(enable=enable, da_value=da)
            assert a.what_does(UNSTORED_AGENT, "go") is None, \
                f"moat breach: unstored agent recalled (enable={enable}, DA={da})"
            assert a.who_does("go", "river") is None, \
                f"moat breach: unstored who-cue recalled (enable={enable}, DA={da})"


def test_who_does_value_prior_picks_high_value():
    """who_does is wired symmetrically: a value-conflicted who-cue (two facts share (action, patient) but differ in
    agent + value) recalls the HIGH-value agent at high DA, the first-match agent under lesion."""
    a = object.__new__(MergedNavConvAgent)
    a.composer = RFPhasorComposer(seed=42, D=64, vocab=VOCAB)
    a.enable_da_salience_gate = False
    a.enable_da_recall_vigor = True
    a._da_recall_beta = BETA
    a._da_recall_baseline = DA_TONIC
    a._da_recall_value_default = 1.0
    a._fact_values = []
    a._da_recall_view = None
    a._da_recall_dopamine = lambda: DA_HIGH
    # two facts share (action=go, patient=north) but differ in agent + value
    a._store_fact_value("cat", "go", "north", value=0.0)    # kb 0 = first-match agent
    a._store_fact_value("dog", "go", "north", value=1.0)    # the HIGH-value agent
    assert a.who_does("go", "north") == "dog", "high-DA value prior should recall the HI-value agent"
    a._da_recall_dopamine = lambda: DA_TONIC
    assert a.who_does("go", "north") == a.composer.query_agent("go", "north") == "cat", \
        "under DA-lesion who_does collapses to the first-match agent"


def test_wirein_reuses_the_validated_derisk_class():
    """The wire-in reuses the validated DARecallVigorComposer by composition (reuse-by-import, NO sim/ edit): the
    agent's value-prior view IS a DARecallVigorComposer bound to the agent's own composer."""
    from research.runners._tier2_6_da_recall_vigor_derisk import DARecallVigorComposer
    a = _make_agent(enable=True, da_value=DA_HIGH)
    view = a._da_recall_vigor_view()
    assert isinstance(view, DARecallVigorComposer)
    assert view.comp is a.composer
    assert view.values is a._fact_values
