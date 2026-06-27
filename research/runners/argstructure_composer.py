"""Tier 0.1 (verb-frame argument structure) + 0.2 (fixed-capacity WM) PRODUCTION module.

Promoted from the GO de-risk (research/findings/2026-06-27-tier0-argstructure-wm-GO.md;
research/runners/_tier0_argstructure_derisk.py, 6/6 seeds). Extends the deployed RFPhasorComposer
(research/runners/rf_phasor_composer.py) by reuse-by-import with:

  * 0.1 -- a TYPED-ROLE alphabet (GOAL, RECIPIENT, THEME, LOCATION, SOURCE, INSTRUMENT, TIME) beyond
    the bare (agent, action, patient), plus a per-verb-class FRAME LEXICON (MUC-Memory: each verb stores
    its structural frame -- go->GOAL-PP "to X"; give->THEME+RECIPIENT "X to Y"; put->THEME+LOCATION
    "X on Y"; default transitive -> patient). A stored fact is expanded into ordered (content +
    closed-class: determiner / preposition / tense) slots and the CONTENT slots' order is produced by
    the VALIDATED FrameCQ serial-order engine (the seed of syntax, 6/6 GO). Render -> "the boy goes to
    the park". The no-confab moat is the parent's (a query whose cue roles match no stored fact -> None).

  * 0.2 -- the rendered frame's ordered slots live in a FIXED-CAPACITY working-memory buffer
    (FixedCapacityDiscourseWM, a thin wrapper on the in-codebase OrderedPositionWM). The WM neuron-count
    is set by the SLOT COUNT (Cowan ~4+-1 / Lisman-Idiart gamma slots), INDEPENDENT of vocabulary --
    the biologically-correct storage(unbounded, in the codes)/buffer(fixed) split. This is the
    construction that kills the content_selection_spiking.py balloon (n=60*len(vocab)) by design.

Biology: Hagoort MUC (the verb's frame in temporal-cortex Memory; Broca = Unification binds fillers in);
Bock & Levelt functional->positional (the verb lemma projects its argument frame + the closed-class
scaffold; agrammatic Broca's output = a functional structure that never got positional realization);
FrameCQ = Bullock-Rhodes competitive queuing (pre-SMA serial order). Cowan 2001 (~4+-1 WM); Lisman-Idiart
(gamma slots).

NO sim/ edit; NO production-composer edit -- this module SUBCLASSES the composer (reuse-by-import).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer, ROLES as _CORE_ROLES  # noqa: E402
from research.runners.ordered_position_wm import OrderedPositionWM  # noqa: E402


# --- The TYPED-ROLE extension of the composer's alphabet (0.1) ---------------------------------------------------
# Thematic / oblique roles the bare (agent, action, patient) alphabet cannot express. The composer's binding is
# role-AGNOSTIC (rf_phasor_composer.py:262 binds `for r in ROLES if r in fact`), so adding roles costs only more
# codebook entries -- exactly the MUC-Memory "the verb stores its frame; Unification binds the fillers in" story.
TYPED_ROLES = ("GOAL", "RECIPIENT", "THEME", "LOCATION", "SOURCE", "INSTRUMENT", "TIME")
ALL_ROLES = tuple(_CORE_ROLES) + TYPED_ROLES

# --- The per-verb-class FRAME LEXICON (MUC-Memory) ---------------------------------------------------------------
# Each frame is an ordered list of PHRASE UNITS. A unit is (kind, role, lead) where:
#   kind   = "CONTENT" (a filler spelled by the composer) | "TENSE" (the verb, tense-inflected);
#   role   = the role this unit realizes (agent / action / GOAL / THEME / ...);
#   lead   = the CLOSED-CLASS function words (determiner / preposition) that precede this unit's content word
#            (the verb-frame's scaffold). Grouping the scaffold WITH its content unit means an ABSENT role (a
#            partial corpus fact that didn't realize that argument) cleanly drops its leading prep+determiner too,
#            so "boy go" (no GOAL) renders "the boy goes" (not "the boy goes to the [GOAL]").
# Extend by adding a verb -> [units] entry; FRAME_ROLES + the FrameCQ frame-id map are derived from this.
def _U(kind, role, lead=()):
    return (kind, role, tuple(lead))


FRAME_LEXICON = {
    "go":   [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "GOAL", ("to", "the"))],
    "come": [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "GOAL", ("to", "the"))],
    "walk": [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "GOAL", ("to", "the"))],
    "run":  [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "GOAL", ("to", "the"))],
    "give": [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "THEME", ("the",)),
             _U("CONTENT", "RECIPIENT", ("to", "the"))],
    "send": [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "THEME", ("the",)),
             _U("CONTENT", "RECIPIENT", ("to", "the"))],
    "put":  [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "THEME", ("the",)),
             _U("CONTENT", "LOCATION", ("on", "the"))],
    "_default": [_U("CONTENT", "agent", ("the",)), _U("TENSE", "action"), _U("CONTENT", "patient", ("the",))],
}
# Which CONTENT/TENSE roles each verb-frame licenses (the args the extractor may fill) -- derived from the frame.
FRAME_ROLES = {v: [u[1] for u in units] for v, units in FRAME_LEXICON.items()}

# Which PREPOSITION maps to which oblique role for a given verb-frame -- used by the extractor (0.1) to assign a
# corpus PP to the right typed role (e.g. "go to the park": to -> GOAL; "put on the table": on -> LOCATION).
# (verb, preposition) -> role; falls back to the verb's single oblique role if the verb licenses exactly one.
VERB_PREP_ROLE = {
    ("go", "to"): "GOAL", ("come", "to"): "GOAL", ("walk", "to"): "GOAL", ("run", "to"): "GOAL",
    ("give", "to"): "RECIPIENT", ("send", "to"): "RECIPIENT",
    ("put", "on"): "LOCATION", ("put", "in"): "LOCATION",
}

# A present-tense 3sg inflection table (morphology = a legitimate lexical front-end, like the parser's morphology).
# The brain renders the bare verb; this host polish adds the agreement morpheme (a closed-class element).
TENSE_3SG = {
    "go": "goes", "come": "comes", "walk": "walks", "run": "runs", "give": "gives", "send": "sends",
    "put": "puts", "chase": "chases", "eat": "eats", "see": "sees", "like": "likes", "have": "has",
    "make": "makes", "take": "takes", "find": "finds", "look": "looks", "want": "wants",
}

# The closed-class FUNCTION-WORD POOL (determiners + prepositions). Ablating this pool is the agrammatism control.
FUNCTION_WORDS = {"the", "a", "an", "to", "on", "in", "of", "with", "from", "at", "by"}

# Stable per-verb FrameCQ frame-id (distinct frames get distinct learned primacy gradients).
_FRAME_IDS = {v: i for i, v in enumerate(FRAME_LEXICON.keys())}


def frame_for(verb):
    """The verb's stored frame (MUC-Memory). Unknown verbs -> the default transitive frame."""
    return FRAME_LEXICON.get(verb, FRAME_LEXICON["_default"])


def frame_id(verb):
    return _FRAME_IDS.get(verb, _FRAME_IDS["_default"])


def content_slot_count(verb):
    """Number of phrase units in a verb's frame (all units are CONTENT/TENSE -- what FrameCQ orders)."""
    return len(frame_for(verb))


def realized_units(verb, fact):
    """The frame units whose role is PRESENT in `fact` (action + agent always; obliques present-only). A partial
    corpus fact (e.g. 'boy go' with no GOAL) realizes a subset of the frame's units."""
    return [u for u in frame_for(verb) if u[1] in ("action",) or u[1] in fact]


class FrameCQ:
    """The validated frame-conditioned competitive-queuing serial-order generator (== the 6/6-GO
    _phaseB_serial_order_multiframe_derisk.FrameCQ): a per-frame primacy gradient learned from the teacher; emit =
    the choice-WTA read-out in that frame's primacy order (inhibition-of-return). Here it orders the CONTENT slots
    of a verb frame (the argument order). The teacher order is the frame lexicon's canonical content-slot order."""

    def __init__(self, n_frames=None, max_slots=None, lr=0.1, seed=42, wta_noise=0.05, teacher_reps=40):
        n_frames = n_frames if n_frames is not None else len(FRAME_LEXICON)
        max_slots = max_slots if max_slots is not None else max(content_slot_count(v) for v in FRAME_LEXICON)
        self.lr = lr
        self.wta_noise = wta_noise
        self.prim = np.random.default_rng(seed * 13 + 5).standard_normal((n_frames, max_slots)) * 0.01
        self._rng = np.random.default_rng(seed * 71 + 3)
        # teach each frame its canonical content-slot order (identity: slot 0 first) -- the frame lexicon already
        # lists content slots in their canonical argument order.
        for verb in FRAME_LEXICON:
            fid, n = frame_id(verb), content_slot_count(verb)
            for _ in range(teacher_reps):
                for pos in range(n):
                    self.prim[fid][pos] += self.lr * (n - 1 - pos)

    def emit_order(self, fid, unit_indices):
        """Order a set of REALIZED frame-unit indices by the frame's learned primacy gradient (the choice-WTA
        read-out with inhibition-of-return). `unit_indices` are the canonical-frame positions of the units actually
        present (a partial fact realizes a subset). Returns those indices in the frame's learned order. The primacy
        gradient was learned over the FULL frame, so a subset is ordered consistently with the full-frame order."""
        idx = list(unit_indices)
        a = {i: self.prim[fid][i] + self.wta_noise * self._rng.standard_normal() for i in idx}
        avail, order = list(idx), []
        for _ in range(len(idx)):
            best = max(avail, key=lambda i: a[i])
            order.append(best); avail.remove(best)
        return order


class ArgStructureComposer(RFPhasorComposer):
    """RFPhasorComposer extended with TYPED OBLIQUE roles + a per-verb FRAME LEXICON + FrameCQ rendering.

    Typed roles are drawn from a DISJOINT rng stream (seed+2000) so the parent's concept codes stay byte-identical
    (the same disjoint-stream discipline OrderedPositionWM uses). A fact is a dict over {agent, action, <typed
    roles>}; `_encode` (overridden to iterate ALL_ROLES) binds every role present via the parent's spiking RF bind.
    Recall reuses the parent's `unbind`. Render expands the verb's frame into ordered (content + closed-class) slots
    and orders the content slots with FrameCQ. The no-confab moat is the parent's."""

    def __init__(self, seed=42, D=64, vocab=None, grounded_codes=None, framecq_seed=None):
        super().__init__(seed=seed, D=D, vocab=vocab, grounded_codes=grounded_codes)
        prng = np.random.default_rng(seed + 2000)
        for r in TYPED_ROLES:
            self.roles[r] = prng.uniform(0.0, 1.0, self.D)
        self.frame_cq = FrameCQ(seed=seed if framecq_seed is None else framecq_seed)

    # the parent's _encode iterates the module-level ROLES tuple; iterate the EXTENDED role set so typed roles bind.
    def _encode(self, fact):
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in ALL_ROLES if r in fact]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def store_fact(self, fact):
        """Store an argument-structure fact dict, e.g. {'agent':'boy','action':'go','GOAL':'park'}.
        Holds the bound composite (substrate or numpy, per the parent's enable_substrate_store)."""
        comp = self._encode(fact)
        self.kb.append((dict(fact), self._store_substrate(comp) if self.enable_substrate_store else comp))

    def query_role(self, role, **cue_roles):
        """Recall the filler of `role` from the FIRST stored fact whose cue roles ALL match; None = abstain (the
        no-confab moat). Generalizes query_patient/query_agent to ANY typed role."""
        for fact, handle in self.kb:
            comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
            if all(self.unbind(comp, cr) == cv for cr, cv in cue_roles.items()):
                return self.unbind(comp, role)
        return None

    # --- render: expand the verb frame into ordered phrase units, order with FrameCQ, spell each ---------------
    def _decode_unit_word(self, unit, fact, comp):
        """Decode a phrase unit's content word from the RF unbind (NOT the stored label). TENSE -> the inflected
        verb (agreement morpheme); CONTENT -> the role's filler decoded from the composite."""
        kind, role, _lead = unit
        if kind == "TENSE":
            return TENSE_3SG.get(self.unbind(comp, "action"), self.unbind(comp, "action"))
        return self.unbind(comp, role)

    def render(self, fact, comp=None, ablate_closed_class=False, use_framecq=True):
        """Render the fact as prose via its verb frame.

        `ablate_closed_class=True` drops the closed-class scaffold (each unit's lead function words) + the tense
        morphology -> telegraphic agrammatic output (the Broca's anti-cheat -- proves the scaffold does real work).
        `use_framecq=True` orders the REALIZED phrase units by the validated FrameCQ serial-order engine (the
        cognitive ordering is neural). `comp` may be omitted -- it is then recalled from the store (agent+action)."""
        if comp is None:
            comp = self._composite_for(fact)
            if comp is None:
                return None                       # moat: no stored composite -> no fabricated sentence
        verb = fact["action"]
        units = realized_units(verb, fact)        # only the units whose role is present in the fact
        full_frame = frame_for(verb)
        # FrameCQ orders the realized units by their canonical-frame index (the learned per-frame primacy gradient).
        if use_framecq:
            unit_to_idx = {id(u): i for i, u in enumerate(full_frame)}
            realized_idx = [unit_to_idx[id(u)] for u in units]
            order = self.frame_cq.emit_order(frame_id(verb), realized_idx)
            idx_to_unit = {unit_to_idx[id(u)]: u for u in units}
            ordered_units = [idx_to_unit[i] for i in order]
        else:
            ordered_units = units
        toks = []
        for kind, role, lead in ordered_units:
            if not ablate_closed_class:
                toks.extend(lead)                  # the unit's closed-class scaffold (det / prep)
            word = self._decode_unit_word((kind, role, lead), fact, comp)
            if kind == "TENSE" and ablate_closed_class:
                word = self.unbind(comp, "action")     # bare verb (no agreement morpheme) under ablation
            toks.append(word)
        return " ".join(toks)

    def _composite_for(self, fact):
        """The stored composite whose agent (+ action) matches `fact` -- for render(comp=None)."""
        for f, handle in self.kb:
            comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
            if self.unbind(comp, "agent") == fact.get("agent") and self.unbind(comp, "action") == fact.get("action"):
                return comp
        return None


def reparse_to_fact(rendered, fact):
    """VERIFY: strip the closed-class scaffold + tense morphology from the rendered prose and check the residual
    content words match the stored fact's REALIZED fillers (agent, action, + the obliques present in the fact). A
    content mismatch -> False (reject -- the moat on the render)."""
    inv_tense = {v: k for k, v in TENSE_3SG.items()}
    toks = [inv_tense.get(t, t) for t in rendered.split() if t not in FUNCTION_WORDS]
    content_vals = set()
    for kind, role, _lead in realized_units(fact["action"], fact):
        content_vals.add(fact["action"] if role == "action" else fact[role])
    return set(toks) == content_vals


class FixedCapacityDiscourseWM:
    """Tier 0.2 -- a FIXED-CAPACITY working-memory buffer for the discourse / the active verb-frame's slots.

    The biologically-correct storage(unbounded, in the concept codes)/buffer(fixed ~4+-1) split (Cowan 2001;
    Lisman-Idiart gamma slots), realized on the in-codebase OrderedPositionWM (the fixed-slot, vocabulary-
    INDEPENDENT pointer buffer on the spiking RF phasor substrate). The WM substrate's neuron count is set by D +
    the slot count, NOT by vocabulary -- so it does NOT balloon (unlike content_selection_spiking.py's
    SpikingLoopContextBuffer, n=60*len(vocab), which installs one attractor per vocab item). This kills the
    freeze-at-scale by construction.

    Usage: ``hold(items)`` encodes an ordered list of <=n_slots items into the fixed buffer; ``read(k)`` reads slot
    k back (with the no-confab familiarity gate -> None on an empty/unfamiliar slot). The render path can hold the
    verb-frame's ordered content slots here (a frame is ~4-6 slots = exactly WM capacity)."""

    def __init__(self, seed=42, D=128, vocab=None, n_slots=7, cleanup_words=None):
        # the WM holds POINTERS into the concept codes; the vocab is the candidate set it can hold.
        self.wm = OrderedPositionWM(seed=seed, D=D, vocab=vocab, n_slots=n_slots, cleanup_words=cleanup_words)
        self.n_slots = int(n_slots)
        self._held = None

    def hold(self, items):
        """Encode an ordered list of items (<= n_slots) into the fixed buffer; returns the composite phasor."""
        self._held = self.wm.encode_sequence(list(items))
        return self._held

    def read(self, k, gate=True):
        """Read slot k of the currently-held sequence (word or None on abstain). `k` may be an int (-> 'pos{k}')."""
        if self._held is None:
            return None
        slot_key = f"pos{k}" if isinstance(k, int) else k
        word, _strength = self.wm.read_slot(self._held, slot_key, gate=gate)
        return word

    def wm_neuron_count(self):
        """The TOTAL neuron count of the WM's spiking substrate (sum of the lazily-built RF bridges, keyed by
        neuron count = f(D, slots)). Constant as vocab grows -- the balloon is gone (Tier 0.2 verification)."""
        return int(sum(self.wm._bridge_cache.keys()))
