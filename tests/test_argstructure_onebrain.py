"""CI GUARD for BURNDOWN C4 (the LAST Bucket-A conversion): the TYPED verb-frame argument-structure surface on the
SPIKING one-brain substrate must answer the validated typed cases ANSWER-IDENTICAL to the numpy `ArgStructureComposer`
oracle, with the no-confab moat 0-FA.

C1/C2/C3 brought the word-ordering + the flat-SVO recall/answer onto the spiking `OneBrainComposer`. C4 gives the
spiking substrate a TYPED-ROLE API (typed roles GOAL/THEME/RECIPIENT/LOCATION/... bound + stored via the RF
complex-synapse store, `query_role`, and the verb-frame `render`) -- so the console's `--argstructure` path can run
`--composer onebrain`. Realized by extending `OneBrainComposer` with `typed_roles=(...)` + `store_fact`/`query_role`/
`render` (reuse-by-import; NO sim/ edit). De-risk: research/findings/2026-06-27-burndown-C4-typed-frame-onebrain-GO.md.

The substrate parity is at D=128 (the console brain D). At the test's small D the densest 4-role frames hit a
bundle-SNR boundary; D>=128 clears it (the standard VSA lever). GPU-only (the on-bridge RF store + resonate scan
needs the CuPy substrate); skips gracefully without GPU / when the concept cache is absent.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gpu_backend(),
                                reason="the typed-role OneBrainComposer RF store/scan needs the CuPy/GPU substrate")

from research.runners.argstructure_composer import (  # noqa: E402
    ArgStructureComposer, TYPED_ROLES, FUNCTION_WORDS, reparse_to_fact)
from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402

VOCAB = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase", "send", "run",
         "park", "house", "ball", "bone", "table", "shelf", "river", "hug"]
FACTS = [
    {"agent": "boy", "action": "go", "GOAL": "park"},
    {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
    {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"},
    {"agent": "cat", "action": "chase", "patient": "river"},
]
D = 128  # the console brain D; the bundle-SNR lever clears the densest 4-role frames here


def _build(seed):
    """The spiking typed-role substrate composer + the numpy oracle on identical seed/D/vocab, both stored with the
    same typed facts. use_spiking_cq=False so the render-order parity bar is the substrate CONTENT decode (the
    spiking-CQ order is the separately-validated C1 conversion; the default-on spiking-CQ render is exercised too)."""
    sub = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, typed_roles=TYPED_ROLES, use_spiking_cq=False)
    oracle = ArgStructureComposer(seed=seed, D=D, vocab=VOCAB, use_spiking_cq=False)
    for f in FACTS:
        sub.store_fact(f)
        oracle.store_fact(f)
    return sub, oracle


def test_typed_role_recall_parity():
    """Every typed oblique role recalls on the spiking substrate == the numpy oracle == ground truth."""
    sub, oracle = _build(42)
    cases = [
        ("GOAL", {"agent": "boy", "action": "go"}, "park"),
        ("agent", {"action": "go", "GOAL": "park"}, "boy"),
        ("THEME", {"agent": "girl", "action": "give"}, "ball"),
        ("RECIPIENT", {"agent": "girl", "action": "give"}, "dog"),
        ("THEME", {"agent": "dog", "action": "put"}, "bone"),
        ("LOCATION", {"agent": "dog", "action": "put"}, "table"),
        ("patient", {"agent": "cat", "action": "chase"}, "river"),
    ]
    for role, cue, truth in cases:
        s = sub.query_role(role, **cue)
        o = oracle.query_role(role, **cue)
        assert s == o == truth, f"typed recall {role} {cue}: substrate={s!r} oracle={o!r} truth={truth!r}"


def test_render_parity_boy_goes_to_park():
    """The headline render on the substrate == the oracle == 'the boy goes to the park' (the verb frame's preposition
    'to' + determiner 'the' from the closed-class scaffold), via BOTH use_framecq=False (content+scaffold) AND the
    default spiking-CQ ordering."""
    sub, oracle = _build(42)
    fact = {"agent": "boy", "action": "go", "GOAL": "park"}
    target = "the boy goes to the park"
    assert sub.render(dict(fact), use_framecq=False) == oracle.render(dict(fact), oracle._composite_for(fact),
                                                                      use_framecq=False) == target
    assert sub.render(dict(fact)) == target, "the default spiking-CQ render must match on the canonical frame"


def test_frame_lexicon_coverage_parity():
    """give->THEME+RECIPIENT, put->THEME+LOCATION, default transitive each render on the substrate == the oracle."""
    sub, oracle = _build(44)   # seed 44: 7/7 at D=128 (the de-risk)
    pairs = [
        ({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"}, "the girl gives the ball to the dog"),
        ({"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"}, "the dog puts the bone on the table"),
        ({"agent": "cat", "action": "chase", "patient": "river"}, "the cat chases the river"),
    ]
    for fact, target in pairs:
        s = sub.render(dict(fact), use_framecq=False)
        o = oracle.render(dict(fact), oracle._composite_for(fact), use_framecq=False)
        assert s == o == target, f"render {fact.get('action')}: substrate={s!r} oracle={o!r} target={target!r}"


def test_no_confab_moat_parity():
    """Unstored cues abstain (None) on the substrate == the oracle; 0 false-accepts (the HARD safety gate)."""
    sub, oracle = _build(42)
    moat = [
        ("GOAL", {"agent": "boy", "action": "eat"}),       # unstored verb
        ("GOAL", {"agent": "cat", "action": "go"}),        # unstored (agent,action)
        ("THEME", {"agent": "dog", "action": "give"}),     # wrong agent for give
    ]
    for role, cue in moat:
        s = sub.query_role(role, **cue)
        assert s is None, f"MOAT BREACH: substrate query_role({role}, {cue}) = {s!r}, must abstain (None)"
        assert s == oracle.query_role(role, **cue), "moat abstention must match the oracle"
    # a stored cue still answers (the moat does not over-abstain)
    assert sub.query_role("GOAL", agent="boy", action="go") == "park"


def test_verify_reparse():
    """The substrate-rendered prose re-parses to the stored typed fact (a content mismatch would reject)."""
    sub, _ = _build(44)
    for fact in ({"agent": "boy", "action": "go", "GOAL": "park"},
                 {"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"},
                 {"agent": "dog", "action": "put", "THEME": "bone", "LOCATION": "table"}):
        rendered = sub.render(dict(fact))
        assert rendered is not None and reparse_to_fact(rendered, fact), \
            f"the substrate render {rendered!r} must re-parse to {fact}"


def test_agrammatism_anti_cheat():
    """Ablating the closed-class scaffold collapses to telegraphic 'boy go park' (reproduces Broca's): no function
    words, no tense morpheme, and DIFFERENT from the full render -- the scaffold does real work."""
    sub, _ = _build(42)
    fact = {"agent": "boy", "action": "go", "GOAL": "park"}
    full = sub.render(dict(fact))
    tele = sub.render(dict(fact), ablate_closed_class=True)
    assert tele != full
    assert all(w not in FUNCTION_WORDS for w in tele.split())
    assert "goes" not in tele.split()
    assert tele == "boy go park"


def test_default_path_byte_identical():
    """The additive typed-role wiring must not change the default (no typed_roles) OneBrainComposer: bind_roles stays
    the flat 4-role set, and the inner concept/role codes are byte-identical (the typed roles draw from a DISJOINT
    rng stream, seed+2000). No GPU run needed (only construction + the code arrays)."""
    base = OneBrainComposer(seed=42, D=64, vocab=VOCAB, use_spiking_cq=False)
    typed = OneBrainComposer(seed=42, D=64, vocab=VOCAB, typed_roles=TYPED_ROLES, use_spiking_cq=False)
    assert base.bind_roles == ["agent", "action", "patient", "polarity"], "default bind_roles must be unchanged"
    assert base.typed_roles == ()
    # the inner concept codes (and the core role codes) are byte-identical -> the typed roles did not perturb them
    for w in ("boy", "go", "park"):
        assert np.array_equal(base.comp.concepts[w], typed.comp.concepts[w]), f"concept code for {w!r} drifted"
    for r in ("agent", "action", "patient"):
        assert np.array_equal(base.comp.roles[r], typed.comp.roles[r]), f"core role code for {r!r} drifted"
    # the typed composer has the extra roles + the typed-role API
    assert all(r in typed.bind_roles for r in TYPED_ROLES)
    assert hasattr(typed, "query_role") and hasattr(typed, "store_fact") and hasattr(typed, "render")
