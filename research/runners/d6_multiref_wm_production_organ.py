"""MULTI-REFERENT WORKING MEMORY wired into the PRODUCTION conversational turn (D6, Gate-B shape, 2026-08-12).

The faculty: hold >=2 discourse referents ACROSS a turn/span — "the dog and the cat ... it chased her" needs BOTH
referents live at once. The prior single-attractor anaphora store TIES on 2+ (one bump wins the 1-of-K WTA -> the
superposition ~2-cap). This organ gives the brain a genuinely-SPIKING MULTI-REFERENT buffer: R disjoint slow-NMDA
bistable banks on ONE bridge sharing ONE FS inhibitory pool, each register latching one discourse referent and
SUSTAINING it across the intervening span with zero cross-talk.

It REUSES (does not reinvent) the adversarially-verified de-risk
(`research/runners/_multi_slot_binding_derisk.py` -> the 6-seed GO
`2026-08-11-multi-slot-variable-binding-working-memory-holds-k-bindings-no-crosstalk-ceiling-k5-6seed-GO.md`):
the `MultiSlotHold` spiking core (R banks of the D3 `build_persistent_slot` slow-NMDA slot, ONE shared FS), the
RUNG6c content-agnostic Hebbian fast-weight binder (`HebbianBinder`, each fixed referent -> a stable local slot),
and the role-by-position write MARKER. The k=2 held-out ALL-correct is 1.000 (per-slot [1.0,1.0]), ceiling k=5;
LESION-the-hold (recur=0) collapses k>=2 to 0.000, and the SUPERPOSED-single-slot control collides at ~1/k (the
~2-cap the single-attractor store hits). NOTHING here is reimplemented — the organ imports the de-risked classes.

BRAIN-BASED: the LOAD-BEARING contribution is the spiking HOLD. Every register read is an argmax over per-pool
`cp_firing_states` firing rates under external input ASSERTED zero (the HELD bump, not a re-drive) — the same
read-out-instrument class the affect/comprehension/metacog organs use over their spiking pools. The multi-referent
capability is genuinely carried by the sustained bumps: under the lesion the buffer cannot hold >=2 and the
read-back collapses.

WHAT IT DOES IN A TURN (additive, moat-safe, honest — mirrors is_feel_query / is_expectation_query):
  * MAINTAIN: on an input that introduces >=2 distinct discourse referents (a coordinated NP: "the dog and the cat"),
    LOAD each referent into its own register of the spiking buffer (role-by-position) and HOLD. The buffer persists
    across turns as the session's live discourse-referent state.
  * READ-OUT: on an explicit "who/what are we talking about / what are you keeping in mind" query, READ BACK every
    held referent off the spiking buffer and answer with an honest functional read-out ("I'm holding two: dog and
    cat"). This is what a single-attractor store cannot do (it ties to one).
  * It NEVER manufactures a fact, flips an abstain, or changes WHICH answer the recall produced — it only maintains
    and reads its OWN buffer. Out-of-scope inputs (fewer than 2 referents, no multi-referent query) return None ->
    the turn stays byte-identical.

LESION-LOAD-BEARING: `BRAIN_MULTIREF_LESION=1` builds the buffer with recur=0 (the slow-NMDA recurrence killed). The
bumps die over the span, so a >=2-referent read-back collapses (the de-risk's k>=2 all-correct 1.000 -> 0.000). The
host referent PARSE and the write MARKER are byte-identical with/without the lesion, so the discrimination is caused
by the spiking hold, not the host bookkeeping.

COMPETITIVE FREE-SLOT-WINS ALLOCATION (2026-09-01, additive; FLIPPED DEFAULT-ON 2026-09-02, board #196,
`BRAIN_MULTIREF_COMPETITIVE`): closes the
"register assignment is a role-by-position host MARKER" residual below for the REGISTER dimension (WHICH bank holds a
referent). Before each write, `MultiSlotHold.probe_occupancy()` reads every register's CURRENT band-max firing rate
(a genuine zero-input `cp_firing_states` read, external input asserted zero -- the same read-out-instrument class the
organ's own `read()`/hold-query already uses); the referent is routed to `argmin(occupancy)` -- the register the
substrate itself currently shows as free/least-active -- not to the loop index `r`. An already-occupied register
(elevated sustained rate from an earlier write this call) is measurably avoided by the SAME instrument, so >=2
referents introduced together land in DISTINCT registers by the brain's OWN occupancy read, regardless of which is
mentioned first. `BRAIN_MULTIREF_COMPETITION_LESION=1` ablates ONLY the selection (every referent is routed to
register 0 regardless of the probe), reproducing the already-validated SUPERPOSED-collide regime
(`_multi_slot_binding_derisk.eval_superposed_single`) as a genuine within-register collision -- distinct from
`BRAIN_MULTIREF_LESION` (recur=0, kills the HOLD itself), isolating which piece is load-bearing. See
`research/runners/_d6_wm_competitive_slot_binding_verify.py`. HONEST RESIDUAL, UNCHANGED BY THIS: which TOKENS count
as a referent (extraction) and the referent<->local-slot BIND remain host, as below; and because this substrate has
no background OU noise (`ou_std_current_pA=0`), a probe over an all-baseline bank ties and breaks to the lowest free
index -- a real (not formulaic) tie, but a deterministic one absent prior occupancy.

HONEST RESIDUALS (declared; match the de-risk's named residuals + the task's named open rung):
  * The learned SPIKING WRITE-GATE is the open rung: register allocation is now (default-ON, 2026-09-02) a
    genuine occupancy READ (`argmin(probe_occupancy())`), not a LEARNED gate -- the substrate is READ, not
    trained to choose. `BRAIN_MULTIREF_COMPETITIVE=0` reverts to the pre-existing role-by-position host MARKER
    (referent 0 -> reg0, ...). `739a8867` established even a host position-ORACLE fails to induce role at 6
    seeds -> the residual is CREDIT ASSIGNMENT (gap#4). The learned, emergent, spiking multi-register role-gate
    (a trained selection policy, as opposed to this rung's substrate-READ selection) is un-done.
  * The referent EXTRACTION (which tokens are the discourse referents) is a host parse, bounded by a small referent
    lexicon + a coordinated-NP pattern — the same vocab-ceiling class the comprehension organ declares.
    OPT-IN CONVERSION (2026-09-23, `BRAIN_LEARNED_REFERENT_LEXICON`, default OFF): an off-table word is admitted iff
    the v2 referent (noun-category) detector's coupled spiking WTA calls it a referent — graded drive through
    Hebbian-learned frame->category synapses, reciprocal FSI lateral inhibition, host read-out of the winner
    (`research/runners/lexicon_spiking_frame_category.py`; de-risk `_lexicon_spiking_referent_derisk.py`). The v1
    label-spreading detector (`lexicon_learned_referent.py`) was host-computed + spike-RELAYED and is no longer used
    here. `BRAIN_LEARNED_REFERENT_LESION=1` restores the learned frame->category synapses to their pre-learning
    values (a lesion of that ONE learned edge; the circuit stays). Also reaches the activity-silent WM organ, which
    reuses `extract_referents`.
  * The BIND (referent -> local slot) is the host-numpy RUNG6c binder; the register READ is a host argmax over the
    bank's firing rates (a read-out instrument). Capacity is binder-capped at _K=6 distinct referents (the de-risk's
    valid regime, ceiling k=5).
  * CO-RESIDENT: the buffer runs on ITS OWN `MultiSlotHold` bridge ALONGSIDE the recall composer, not merged onto the
    one recall bridge — rides the one-brain merge (burn-down #1), exactly as the affect/comprehension organs do.

Additive, default-ON, `BRAIN_MULTIREF=0` -> the byte-identical oracle (fully skipped). NO `sim/` edit; uses the
process backend (cupy in production, numpy in tests) via reuse-by-import.
"""
from __future__ import annotations

import os
import re

import numpy as np

# --- the de-risked spiking multi-slot HOLD core (R banks of D3 slow-NMDA, ONE shared FS) ---
from research.runners._multi_slot_binding_derisk import MultiSlotHold
# --- the VERIFIED RUNG6c content-agnostic Hebbian fast-weight binder + barcode mint + slot cap ---
from research.runners._novel_referent_hebbian_fastweight_derisk import HebbianBinder, _mint_codes, _K as _BINDER_K

# The spiking buffer geometry (the de-risk's proven valid regime): R_MAX registers, N_SLOT pools per bank.
# Ceiling is k=5 at 6 seeds; we build R_MAX=5 registers and n_slot=6 (>= the _BINDER_K=6 distinct referents).
R_MAX = 5
N_SLOT = 6

# A small referent lexicon (the declared host parse scope). Common concrete discourse nouns; extended on the fly with
# any capitalized proper name. A coordinated NP ("X and Y") or a re-mention drives the load. This is the vocab ceiling
# residual (like the comprehension cue lexicon) — a real learned referent detector is the next rung.
_REFERENT_NOUNS = {
    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig", "mouse", "rabbit", "fox", "wolf", "bear", "lion",
    "man", "woman", "boy", "girl", "child", "baby", "king", "queen", "doctor", "teacher", "farmer", "friend",
    "car", "ball", "book", "tree", "house", "box", "cup", "table", "chair", "door", "key", "phone",
    "john", "mary", "alice", "bob", "sam", "tom", "anna", "lucy",
}
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with", "by", "for", "as", "is", "are", "was",
    "were", "be", "then", "so", "that", "this", "these", "those", "it", "its", "they", "them", "he", "she", "him",
    "her", "his", "their", "we", "us", "our", "you", "your", "i", "me", "my",
}
_PRONOUNS = {"it", "he", "she", "they", "him", "her", "them", "his", "its", "their"}
# The hold-query's own vocabulary ("what are you keeping in MIND") is never a discourse referent -- only consulted on
# the learned-lexicon path (BRAIN_LEARNED_REFERENT_LEXICON), so the hand path is untouched.
_HOLD_QUERY_WORDS = {"talking", "discussing", "referring", "keeping", "mind", "holding", "remember", "tracking",
                     "referent", "referents"}
_WORD_RE = re.compile(r"[A-Za-z']+")

# "who / what are we talking about", "what are you keeping in mind", "what are you holding" ...
_HOLD_QUERY_RE = re.compile(
    r"\b(who|what)\b.*\b(talking about|discussing|referring to|keeping in mind|holding( in mind)?|"
    r"remember|tracking|referents?)\b",
    re.IGNORECASE,
)


def multiref_enabled() -> bool:
    """Default-ON. `BRAIN_MULTIREF` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_MULTIREF")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def multiref_lesioned() -> bool:
    """`BRAIN_MULTIREF_LESION` in {1,true,yes,on} -> build the buffer with recur=0 (kill the slow-NMDA hold)."""
    v = os.environ.get("BRAIN_MULTIREF_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# 2026-09-02 FLIPPED DEFAULT-ON (board #196, 6-seed GO verified fresh against this exact code:
# research/findings/raw/_d6_wm_competitive_slot_binding/verify_6seed.json / research/findings/2026-09-01-d6-wm-
# competitive-slot-binding-6seed-GO.md): >=2 referents introduced together land in DISTINCT registers via a
# genuine MultiSlotHold.probe_occupancy() argmin read, invariant to mention order; a referent introduced after
# an already-held anchor correctly avoids the anchor's occupied register too; the selection-only lesion
# (BRAIN_MULTIREF_COMPETITION_LESION=1) collapses the separation into the already-validated superposed-collide
# regime on every seed (load-bearing); `BRAIN_MULTIREF_COMPETITIVE=0` reverts BYTE-IDENTICALLY to the
# pre-existing role-by-position path (verified 6/6 seeds).
_MULTIREF_COMPETITIVE_DEFAULT_ON = True


def multiref_lesion_scope() -> str:
    """`BRAIN_MULTIREF_LESION_SCOPE` (default unset -> "organ", byte-identical to before this knob existed).

    "organ" (default): the pre-existing `BRAIN_MULTIREF_LESION` behaviour -- the lesioned buffer is a PRIVATE
    `MultiSlotHold(recur=0)` bridge (never the shared one-brain slice), the one-brain `read_isolation` guard is
    skipped, and the xedge focus (`_own_focus`) / semantic-drop drive are withheld. When the xedge pool is live
    (production default) that lesion therefore changes FOUR things, not one (adversarial review v2:7a3b94367).

    "recur": the EDGE-CONFINED lesion. Only the claimed edge changes -- the slow-NMDA w_k->w_k self-recurrence of this
    organ's register pools. With a shared one-brain slice the lesion zeroes exactly those synapses IN PLACE on the
    shared bridge (the same idiom as the xedge `lesion_cross`), keeps `shared=self._shared`, runs under the same
    `read_isolation` guard, and sets the same `_own_focus`; without a shared slice it is the private recur=0 buffer
    (which already differs from the intact buffer only in that weight). Only read when `BRAIN_MULTIREF_LESION` is on."""
    v = (os.environ.get("BRAIN_MULTIREF_LESION_SCOPE") or "").strip().lower()
    return "recur" if v == "recur" else "organ"


def multiref_competitive_enabled() -> bool:
    """DEFAULT-ON (flipped 2026-09-02, `_MULTIREF_COMPETITIVE_DEFAULT_ON`). `BRAIN_MULTIREF_COMPETITIVE` in
    {0,false,no,off,""} -> an explicit OFF, reverting to the pre-existing role-by-position host MARKER
    (referent i -> register i), byte-identical to before this flag existed. On (the default): register
    ALLOCATION is the EMERGENT free-slot-wins competitive read (`MultiSlotHold.probe_occupancy()` -> argmin) --
    the brain's own current occupancy, not sentence position, decides which register binds a new referent."""
    v = os.environ.get("BRAIN_MULTIREF_COMPETITIVE")
    if _MULTIREF_COMPETITIVE_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def multiref_competition_lesioned() -> bool:
    """`BRAIN_MULTIREF_COMPETITION_LESION` in {1,true,yes,on} -> ablate ONLY the competitive SELECTION (every
    referent is routed to register 0 regardless of the occupancy probe), so >=2 referents collide within one
    register's local competition -- the already-validated SUPERPOSED-collide regime, reproduced here as a
    genuine collision rather than assumed. Distinct from `multiref_lesioned()` (recur=0, kills the HOLD's
    recurrence): this lesions the ALLOCATION decision, isolating which piece is load-bearing. No effect unless
    `multiref_competitive_enabled()` is also True."""
    v = os.environ.get("BRAIN_MULTIREF_COMPETITION_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── SPIKING REFERENT -> FOCUS BINDING + PRONOUN RESOLUTION (2026-09-24, additive, DEFAULT OFF) ─────────────────────
# `BRAIN_MULTIREF_FOCUS_BIND=1`. Pre-registration:
# research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md. The named next mechanism of the
# ordinary-content NO-GO (research/findings/2026-09-24-wm-binding-ordinary-content-probe-6seed-NOGO-held-state-does-
# not-reach-an-ordinary-reply.md): the WM focus stops being the positional `CAND_POOLS[0]` and becomes the register
# whose held bump wins a CUE-DRIVEN RETRIEVAL COMPETITION on the organ's own firing state; an anaphor turn resolves its
# pronoun to that register's referent, so an ordinary answer depends on WHICH referent the buffer holds.
#   * CROSS-TURN HOLD: after a load the organ keeps THIS SESSION's held spiking state (its slice's per-neuron state
#     arrays: membrane, recovery, conductances incl. the slow-NMDA recurrence, firing/refractory). The next turn
#     resumes it and runs a zero-input inter-turn span, so a live bump self-sustains and a dead one stays dead. What
#     is held across turns is read off that state; the host `_slot_of_ref` dict is no longer consulted for WHICH
#     referents are held (it stays only as the binder's forward codebook for the next write).
#   * RETRIEVAL (the anaphor cue): the register pools' own held firing is read over a zero-input window (the SAME
#     occupancy instrument `probe_occupancy` the competitive write uses), and each register's rate drives its own
#     assembly of a FOCUS WTA -- R_MAX excitatory assemblies, each with its own fast-spiking sub-pool that inhibits
#     every OTHER assembly (the N-way lateral-inhibition primitive `_affect_marker_wta_derisk._build_bridge`, reused
#     as the spiking question-route selector reuses it). The anaphor cue supplies a common sub-threshold drive to all
#     assemblies: no cue, no competition; a register with no held bump adds nothing, so a dead buffer leaves every
#     assembly silent. The race between assemblies, not a host comparison, decides the winner. (Measured on dev seed 7
#     before this design: the D6 bank's own shared FS does NOT make a winner -- FS drive of 300/600 pA scales both
#     held bumps down together -- because the bank is built to hold several bumps without cross-talk.)
#   * DECLARED RESIDUALS (named, not hidden): the register->focus projection crosses two bridges as a host
#     spike-rate relay (rate x gain -> current); the final read of WHICH focus assembly won is an argmax over its
#     settled rates with a dead margin (the qroute/affect-marker read-out convention); the winning register's local
#     slot is the organ's existing `read()`-class argmax over its bank; the slot is NAMED by the host RUNG6c binder
#     codebook (`_ref_of_slot`), unchanged; the inter-turn interval is compressed to FOCUS_INTERTURN_STEPS ms; the
#     per-session state stash exists because one shared slice serves every session (a hosting step -- the stash is
#     the substrate state, it carries no referent label); which of two held referents wins an ambiguous competition
#     is set by the pools' intrinsic excitability, not by discourse salience (Centering / recency is the next rung).
# Unset -> every function below is unreachable from production and the organ is byte-identical.
FOCUS_INTERTURN_STEPS = 100       # the inter-turn interval as a zero-input hold (compressed; declared)
FOCUS_READ_STEPS = 40             # zero-input read window of the held registers (the probe_occupancy instrument)
FOCUS_WTA_CUE_PA = 150.0          # the anaphor cue's common drive onto every focus assembly (sub-threshold alone)
FOCUS_WTA_GAIN_PA = 20000.0       # relay gain: pA onto a register's focus assembly per unit held-bump rate
FOCUS_WTA_MIN_RATE = 0.05         # the winning assembly must fire at least this (the cue alone cannot)
FOCUS_WTA_DEAD_MARGIN = 0.05      # winner minus runner-up assembly rate (same convention as the qroute/affect WTAs)
FOCUS_WTA_WASHOUT, FOCUS_WTA_WARMUP, FOCUS_WTA_RUN = 40, 60, 150  # qroute timing, longer race (dev-seed calibration)
# per-neuron state arrays the cross-turn stash carries (only those present with length == num_neurons are kept)
_FOCUS_STATE_ARRAYS = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_conductance_g_e",
                       "cp_conductance_g_i", "cp_conductance_g_gabab", "cp_conductance_g_nmda",
                       "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent",
                       "cp_conductance_g_nmda_recurrent_rise", "cp_firing_states", "cp_prev_firing_states",
                       "cp_refractory_timers")


def multiref_focus_bind_enabled() -> bool:
    """`BRAIN_MULTIREF_FOCUS_BIND` in {1,true,yes,on} -> the spiking referent->focus binding + pronoun resolution
    (see the block comment above). DEFAULT-OFF: unset/anything else -> byte-identical to before this flag existed."""
    v = os.environ.get("BRAIN_MULTIREF_FOCUS_BIND")
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def is_hold_query(text: str) -> bool:
    """An explicit 'who/what are we talking about / what are you keeping in mind' inner-state read-out query."""
    return bool(_HOLD_QUERY_RE.search(text or ""))


def learned_referent_enabled() -> bool:
    """`BRAIN_LEARNED_REFERENT_LEXICON` in {1,true,yes,on} -> extend the referent scope beyond the hand
    `_REFERENT_NOUNS` table to the corpus-learned open-vocab referent (noun-category) detector whose decision is a
    coupled spiking WTA (`research/runners/lexicon_spiking_frame_category.py`). DEFAULT-OFF: unset -> byte-identical
    hand-table path."""
    v = os.environ.get("BRAIN_LEARNED_REFERENT_LEXICON")
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def learned_referent_lesioned() -> bool:
    """`BRAIN_LEARNED_REFERENT_LESION` in {1,true,yes,on} -> restore the detector's Hebbian-learned frame->category
    synapses to their pre-learning values (lesion of the learned edge; the WTA circuit and its drive stay). What the
    scope then becomes is MEASURED by the de-risk, not assumed."""
    v = os.environ.get("BRAIN_LEARNED_REFERENT_LESION")
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def _flag_learned_referent_lexicon():
    """The process-shared deployment lexicon when the flag is on (lesion applied per call), else None."""
    if not learned_referent_enabled():
        return None
    from research.runners.lexicon_spiking_frame_category import get_lexicon
    lex = get_lexicon()
    lex.set_lesion("learned_edge" if learned_referent_lesioned() else None)
    return lex


def extract_referents(text: str, max_refs: int = R_MAX, referent_lexicon=None):
    """Host parse (the declared vocab-ceiling residual): return the ORDERED, de-duplicated discourse referents named in
    `text`. A referent is a lexicon noun OR a capitalized proper name (not sentence-initial-only). Order = order of
    mention (role-by-position marker). Capped at max_refs and at the binder's _K distinct slots.

    `referent_lexicon` (default None -> read `BRAIN_LEARNED_REFERENT_LEXICON`; unset -> byte-identical hand path): an
    object with `is_referent(word) -> bool` (the learned open-vocab detector). A word the hand table lacks is admitted
    iff the learned detector calls it a referent; the hand table always wins first."""
    lexicon = referent_lexicon if referent_lexicon is not None else _flag_learned_referent_lexicon()
    raw = _WORD_RE.findall(text or "")
    refs: list[str] = []
    for i, w in enumerate(raw):
        lw = w.lower()
        is_lex = lw in _REFERENT_NOUNS
        if (not is_lex and lexicon is not None and lw not in _STOP and lw not in _PRONOUNS
                and lw not in _HOLD_QUERY_WORDS):
            is_lex = bool(lexicon.is_referent(lw))
        is_proper = (len(w) > 1 and w[0].isupper() and i > 0 and lw not in _STOP)
        if (is_lex or is_proper) and lw not in _PRONOUNS:
            if lw not in refs:
                refs.append(lw)
        if len(refs) >= max_refs:
            break
    return refs[:min(max_refs, _BINDER_K)]


class MultiReferentWMOrgan:
    """A process-shared spiking multi-referent discourse buffer. Built ONCE (lazily): ONE `MultiSlotHold` (R_MAX banks
    of the D3 slow-NMDA slot, ONE shared FS) plus a content-agnostic referent binder. `load` writes >=1 referents into
    disjoint registers (role-by-position) and HOLDS; `read_all` reads each register's HELD bump off cp_firing_states.
    The read-back is what is surfaced (the brain reports what its SPIKING WM holds), so the lesion is load-bearing."""

    def __init__(self, seed: int = 42, shared=None):
        self.seed = int(seed)
        # ONE-BRAIN MERGE (opt-in, byte-identical when shared is None): a MergedPool whose region slice hosts this
        # buffer's R_MAX banks + shared FS; passed through to the MultiSlotHold core so the HOLD runs on the shared
        # spiking bridge. None -> the organ builds its own bridge exactly as today.
        self._shared = shared
        self._built = False
        self.buf = None          # intact MultiSlotHold (recur>0)
        self.buf_lesion = None   # recur=0 MultiSlotHold (killed hold)
        self.binder = None
        self._ref_of_slot: dict[int, str] = {}   # local slot -> referent string (shared local codebook)
        self._slot_of_ref: dict[str, int] = {}   # referent string -> local slot
        self._codes = None
        self._next_code = 0
        # ONE-BRAIN CROSS-EDGE focus (2026-08-27 cross-session leak fix, research/FAILURE_LOG.md): THIS session's
        # own xedge focus, stored on THIS ORGAN INSTANCE only -- never written onto the shared process-global pool.
        # Each session owns exactly one persistent MultiReferentWMOrgan (webapp/server.py's `_SESSION_MULTIREF`, one
        # per cache_key), so an instance attribute is already correctly session-scoped for free, and is derived
        # ONLY from referents THIS organ has itself loaded (see `load()` / `current_focus()`).
        self._own_focus = None
        # Optional injected referent detector (None -> `extract_referents` reads BRAIN_LEARNED_REFERENT_LEXICON;
        # unset -> the hand table, byte-identical). The de-risk injects a cross-validated lexicon here.
        self.referent_lexicon = None

    def ensure_built(self):
        if self._built:
            return
        self.buf = MultiSlotHold(self.seed, R_MAX, N_SLOT, shared=self._shared)
        self.binder = HebbianBinder()
        # pre-mint a barcode pool (>= _BINDER_K distinct referents; deterministic from the seed)
        self._codes = _mint_codes(np.random.default_rng(self.seed + 7), max(_BINDER_K, N_SLOT))
        self._built = True

    def _lesion_buf(self):
        if self.buf_lesion is None:
            self.buf_lesion = MultiSlotHold(self.seed, R_MAX, N_SLOT, recur=0.0)   # kill the slow-NMDA recurrence
        return self.buf_lesion

    def _recur_masks(self):
        """Boolean masks over the SHARED bridge's `cp_connections.data` selecting this organ's slow-NMDA
        self-recurrence: every synapse whose pre AND post neuron lie in the SAME register pool w_k (row = pre,
        col = post, the orientation the xedge `masks` use). Built once, lazily."""
        if getattr(self, "_recur_mask", None) is None:
            from sim.backend import to_host
            b = self.buf.sb
            coo = b.cp_connections.tocoo()
            row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
            m = np.zeros(row.shape, dtype=bool)
            for k in range(self.buf.K):
                ix = self.buf.idx[k]
                m |= np.isin(row, ix) & np.isin(col, ix)
            self._recur_mask = m
            self._recur_saved = None
        return self._recur_mask

    def _set_recur_lesion(self, on: bool):
        """EDGE-CONFINED lesion on the SHARED slice (`BRAIN_MULTIREF_LESION_SCOPE=recur`): zero (on=True) or restore
        (on=False) exactly the w_k->w_k slow-NMDA synapses, in place. Idempotent; restores the saved values, so an
        intact call after a lesioned one in the same process reads the intact weights."""
        from sim.backend import to_host
        m = self._recur_masks()
        b = self.buf.sb
        lesioned = bool(getattr(self, "_recur_lesioned", False))
        if on == lesioned:
            return
        data = np.asarray(to_host(b.cp_connections.data)).copy()
        if on:
            self._recur_saved = data[m].copy()
            data[m] = 0.0
        else:
            data[m] = self._recur_saved
        xp = getattr(self._shared, "xp", None) or np
        b.cp_connections.data = xp.asarray(data, dtype=b.cp_connections.data.dtype)
        self._recur_lesioned = bool(on)

    def _confined(self, lesion: bool) -> bool:
        """True iff this call is an EDGE-CONFINED recur lesion on a shared slice (see `multiref_lesion_scope`)."""
        return bool(lesion and self._shared is not None and multiref_lesion_scope() == "recur")

    def _local_slot(self, ref: str) -> int:
        """Bind a referent string to a STABLE local slot via the RUNG6c binder (content-agnostic, one-shot Hebbian)."""
        if ref in self._slot_of_ref:
            return self._slot_of_ref[ref]
        if self._next_code >= len(self._codes):
            # binder-capacity ceiling (declared residual): reuse the last code -> a collision, not a WM limit
            code = self._codes[-1]
        else:
            code = self._codes[self._next_code]
            self._next_code += 1
        s = int(self.binder.slot(code))
        self._slot_of_ref[ref] = s
        self._ref_of_slot[s] = ref
        return s

    def load(self, referents, lesion: bool = False, competitive: bool | None = None,
             competition_lesion: bool | None = None, xedge_drop_current=None):
        """LOAD the ordered `referents` into registers of the spiking buffer, holding each across a short intervening
        span, then READ every register BACK off the held bumps. Returns the register -> referent mapping RECOVERED
        FROM THE SPIKING BUFFER (== the input iff the hold carried it; degrades under lesion). `hold_alive` is the
        min per-register bump amplitude read under external input ASSERTED zero.

        `competitive` (default: read `BRAIN_MULTIREF_COMPETITIVE`): False/None -> the pre-existing role-by-position
        MARKER (referent i -> register i). True -> EMERGENT free-slot-wins allocation: before each write, probe
        every register's CURRENT occupancy (a genuine `cp_firing_states` read, zero input) and route the referent to
        `argmin(occupancy)` -- the register the substrate itself shows as free, not the loop index. `competition_lesion`
        (default: `BRAIN_MULTIREF_COMPETITION_LESION`) ablates ONLY that selection (always targets register 0),
        independent of the HOLD lesion above.

        `xedge_drop_current` (default None -> byte-identical to the pre-existing behaviour): an optional
        `(pa, steps)` pair. When given, an EXTRA current is injected directly onto PHYSICAL register 0 (the `w0`
        region the curiosity->d6 cross-edge targets -- see `onebrain_xedge_curiosity_d6_production.py`) AFTER the
        normal write+hold span and BEFORE the final read, so if a referent is currently bound to register 0 the
        read that follows genuinely reflects the post-drive spiking state (a real substrate erasure when `pa` is a
        clear-strength hyperpolarizing pull), not a cosmetic flag on the returned text. Only ever touches register
        0's own band (`MultiSlotHold.apply_register_drive`); every other held register is unaffected."""
        self.ensure_built()
        refs = list(referents)[:min(R_MAX, _BINDER_K)]
        confined = self._confined(lesion)
        if self._shared is not None and multiref_lesion_scope() == "recur":
            self._set_recur_lesion(confined)        # zero / restore ONLY the w_k->w_k synapses on the shared slice
        if confined:
            buf = self.buf                          # SAME shared buffer; only its recurrence differs
            lesion = False                          # every other branch below runs exactly as the intact arm's
        else:
            buf = self._lesion_buf() if lesion else self.buf
        competitive = multiref_competitive_enabled() if competitive is None else bool(competitive)
        competition_lesion = (multiref_competition_lesioned() if competition_lesion is None
                               else bool(competition_lesion))
        # ONE-BRAIN MERGE: keep this buffer's whole-bridge reset+step protocol from leaving a footprint on a
        # co-resident organ's slice (only the buffer's own slice evolves; every other slice is restored at exit).
        import contextlib
        guard = (self._shared.read_isolation("d6_multiref_wm")
                 if (self._shared is not None and not lesion) else contextlib.nullcontext())
        with guard:
            buf.reset()
            locals_ = [self._local_slot(r) for r in refs]
            if competitive:
                # EMERGENT free-slot-wins allocation: each write is routed by a genuine occupancy READ, not by
                # the referent's position in `refs`. An already-written register measurably out-competes a free
                # one on the SAME instrument `read()` uses, so later referents avoid it without any host bookkeeping.
                registers = []
                for loc in locals_:
                    reg = 0 if competition_lesion else int(np.argmin(buf.probe_occupancy()))
                    registers.append(reg)
                    buf.write(reg, loc)
                    buf.hold()
            else:
                # role-by-position WRITE MARKER: referent r -> register r; interleave a HOLD after each write (the
                # intervening span) so the earlier registers must SUSTAIN while later ones load (the durability stress).
                registers = list(range(len(locals_)))
                for r, loc in enumerate(locals_):
                    buf.write(r, loc)
                    buf.hold()
            # an extra held span (the "... it chased her" gap) with input asserted zero
            buf.hold()
            buf.hold()
            # ONE-BRAIN CROSS-EDGE SEMANTIC DROP (2026-09-01, additive, opt-in via xedge_drop_current): a validated,
            # substrate-derived crave-suppression signal (see onebrain_xedge_curiosity_d6_production.py's
            # semantic_drop_current -- scaled by the frozen cross-edge's OWN measured, lesion-controlled weight, ~0
            # when lesioned) is applied HERE, directly on register 0's own band, BEFORE the read below -- so a
            # referent bound to w0 is genuinely dropped from `recovered` by the substrate's own post-drive spiking
            # state, not by a host if-statement on a diagnostic number. None (default) -> no-op, byte-identical.
            if xedge_drop_current is not None and not lesion:
                _pa, _steps = xedge_drop_current
                buf.apply_register_drive(0, _pa, _steps)
            recovered = {}
            alive = []
            for i, reg in enumerate(registers):
                loc, amp = buf.read(reg)
                alive.append(float(amp))
                recovered[i] = self._ref_of_slot.get(loc, None)
            # REFERENT->FOCUS BIND (BRAIN_MULTIREF_FOCUS_BIND, default OFF): AFTER the reads above (so `recovered` /
            # `hold_alive_min` are exactly the flag-off values), run the focus retrieval on the freshly loaded buffer
            # and keep this session's held state for the next turn. Off -> never reached.
            focus_r = None
            if multiref_focus_bind_enabled():
                focus_r = self._retrieve(buf)
                self._stash(buf)
        if multiref_focus_bind_enabled():
            # the focus is the register whose held bump WON the retrieval (None when no bump is live, e.g. under the
            # hold lesion) -- replacing the positional CAND_POOLS[0] below.
            self._last_focus = focus_r
            self._own_focus = None if lesion else self._xedge_focus_pool(focus_r)
            refs_for_positional = []
        else:
            refs_for_positional = refs
        # ONE-BRAIN CROSS-EDGE (opt-in): record the primary held referent's POSITIONAL candidate pool as THIS
        # session's own focus (`self._own_focus`), which the caller (webapp/server.py) later reads via
        # `current_focus()` and passes EXPLICITLY into the comprehension organ's `wm_focus` argument -- so a held
        # WM referent drives the frozen d6->sel cross-edge for the SESSION THAT ACTUALLY HELD IT, and no other.
        # 2026-08-27: this used to write `self._shared.xedge_focus` (a process-global attribute on the ONE shared
        # pool), which every OTHER session's comprehension read consulted too -> a cross-session focus leak
        # (research/FAILURE_LOG.md). Storing it on `self` instead makes leakage structurally impossible: a fresh
        # organ with an empty `_slot_of_ref` never runs this branch, so `current_focus()` stays None regardless of
        # what any other session's organ ever did. Guarded by the xedge pool's OWN marker attr
        # (`xedge_codrive_params`) -> `_own_focus` stays None (byte-identical) when shared is None or not an xedge
        # pool. The register->candidate-pool map is POSITIONAL (declared residual: R3-v3's candidate topology is
        # host-chosen, not a semantic role->pool binding; see onebrain_xedge_production).
        if refs_for_positional and getattr(self._shared, "xedge_codrive_params", None) is not None and not lesion:
            try:
                from research.runners._onebrain_integration_r2_threefactor_selforganized import CAND_POOLS
                self._own_focus = CAND_POOLS[0]
            except Exception:
                pass
        return {
            "n_referents": len(refs),
            "input_order": refs,
            "recovered": recovered,                          # {reg: referent} read off the SPIKING held bumps
            "hold_alive_min": float(min(alive)) if alive else 0.0,
            "zero_input_ok": bool(buf._zero_input_span),
            "all_recovered": bool(len(refs) >= 1 and all(recovered.get(r) == refs[r] for r in range(len(refs)))),
            "registers": registers,                          # which register EACH input-order referent landed in
            "distinct_registers": bool(len(set(registers)) == len(registers)),   # no two referents shared a bank
            "competitive": bool(competitive),
            "competition_lesioned": bool(competitive and competition_lesion),
        } | ({"recur_lesioned": True} if confined else {})

    # ── REFERENT->FOCUS BIND (BRAIN_MULTIREF_FOCUS_BIND, default OFF; reached only with the flag on) ────────────────
    def _slice_idx(self, buf):
        """This organ's own neurons on `buf`'s bridge: every register pool + the shared FS (never another organ's)."""
        parts = [np.asarray(buf.idx[k], dtype=np.int64) for k in range(buf.K)]
        parts.append(np.asarray(buf.fs_idx, dtype=np.int64))
        return np.concatenate(parts)

    def _stash(self, buf):
        """Keep THIS session's held spiking state for the next turn: the organ slice's per-neuron state arrays
        (membrane, recovery, conductances incl. the slow-NMDA recurrence, firing/refractory). No referent label is
        stored; what is held is decided later by reading this state."""
        from sim.backend import to_host
        b = buf.sb
        ix = self._slice_idx(buf)
        n = int(b.core_config.num_neurons)
        st = {}
        for nm in _FOCUS_STATE_ARRAYS:
            arr = getattr(b, nm, None)
            if arr is None or int(arr.shape[0]) != n:
                continue
            st[nm] = np.asarray(to_host(arr))[ix].copy()
        self._focus_stash = {"buf": id(buf), "state": st}

    def _resume(self, buf) -> bool:
        """Write this session's stashed state back onto the organ's slice (another session or organ may have stepped
        or reset the shared bridge since). False when nothing is stashed for `buf`."""
        s = getattr(self, "_focus_stash", None)
        if not s or s.get("buf") != id(buf):
            return False
        b = buf.sb
        ix_dev = buf._from_host(self._slice_idx(buf))
        for nm, vals in s["state"].items():
            arr = getattr(b, nm, None)
            if arr is None:
                continue
            arr[ix_dev] = buf._from_host(vals)
        return True

    def has_held_state(self) -> bool:
        """True iff this session has a stashed buffer state (a >=2-referent load happened under the flag). Says
        nothing about WHICH referents survive -- that is read off the state."""
        return bool(getattr(self, "_focus_stash", None))

    def _focus_wta_rates(self, drive_pa):
        """Build a FRESH quiescent R_MAX-way lateral-inhibition WTA (the `_affect_marker_wta_derisk._build_bridge`
        primitive, reused), drive assembly r with `drive_pa[r]`, and return the settled per-assembly rates. Fresh per
        decision and RNG-isolated on this organ's own private timeline (the spiking question-route selector's
        discipline, reused through its `_isolated`), so the host RNG and every later organ are untouched."""
        from research.runners._affect_marker_wta_derisk import _build_bridge, _pool_rates
        if getattr(self, "_focus_rng", None) is None:
            from research.runners.spiking_qroute_selection_organ import SpikingQRouteSelectorOrgan
            self._focus_rng = SpikingQRouteSelectorOrgan(seed=self.seed + 1009)   # used ONLY for its RNG isolation
        n_pools = len(drive_pa)

        def _run():
            bridge, marker_idx, _fsi = _build_bridge(self.seed + 1009, n_pools, "d6focus")
            return _pool_rates(bridge, marker_idx, np.asarray(drive_pa, dtype=float), warmup=FOCUS_WTA_WARMUP,
                               washout=FOCUS_WTA_WASHOUT, run=FOCUS_WTA_RUN)
        return np.asarray(self._focus_rng._isolated(_run), dtype=float)

    def _retrieve(self, buf) -> dict:
        """THE CUE-DRIVEN RETRIEVAL COMPETITION on the buffer's current state. (1) read every register's held firing
        over a zero-input window (the `probe_occupancy` instrument); (2) relay each register's rate onto its own
        focus-WTA assembly on top of the anaphor cue's common drive and let the lateral-inhibition race run; (3) read
        which assembly won (declared final read-out: argmax with a rate floor + dead margin -> None on a tie or a dead
        buffer). Returns the register rates, the WTA rates and the winner's register / local slot / pool."""
        rates = buf._run(np.zeros(buf.n), FOCUS_READ_STEPS, assert_zero=True)
        bands = np.asarray(rates, dtype=float).reshape(buf.R, buf.n_slot)
        reg_rate = bands.max(axis=1)
        drive = FOCUS_WTA_CUE_PA + FOCUS_WTA_GAIN_PA * reg_rate          # host relay (declared residual)
        wta = self._focus_wta_rates(drive)
        order = np.argsort(-wta, kind="stable")
        w = int(order[0])
        wr = float(wta[w])
        ru = float(wta[order[1]]) if len(wta) > 1 else 0.0
        ok = bool(wr >= FOCUS_WTA_MIN_RATE and (wr - ru) >= FOCUS_WTA_DEAD_MARGIN)
        local = int(np.argmax(bands[w])) if ok else None                 # the organ's read()-class bank argmax
        return {"ok": ok, "register": (w if ok else None), "local": local,
                "pool": ("w%d" % (w * buf.n_slot + local) if ok else None),
                "winner_rate": round(wr, 6), "runner_up_rate": round(ru, 6), "margin": round(wr - ru, 6),
                "register_rates": [round(float(x), 6) for x in reg_rate],
                "wta_rates": [round(float(x), 6) for x in wta]}

    def _xedge_focus_pool(self, r):
        """The xedge focus for comprehension: the WINNING pool, iff it is one the d6->comprehension cross-edge
        spans (CAND_POOLS) and the xedge pool is live; else None (declared: the cross-edge topology covers only
        w0..w2, so a referent held elsewhere has no comprehension drive)."""
        if not r or not r.get("ok") or getattr(self._shared, "xedge_codrive_params", None) is None:
            return None
        try:
            from research.runners._onebrain_integration_r2_threefactor_selforganized import CAND_POOLS
        except Exception:
            return None
        return r["pool"] if r["pool"] in CAND_POOLS else None

    def _focus_buf_and_guard(self, lesion: bool):
        """The buffer + isolation guard a resumed read uses -- the SAME choice `load()` makes (confined recur lesion:
        the shared buffer with its w_k->w_k synapses zeroed; organ-scope lesion: the private recur=0 buffer)."""
        import contextlib
        self.ensure_built()
        confined = self._confined(lesion)
        if self._shared is not None and multiref_lesion_scope() == "recur":
            self._set_recur_lesion(confined)
        if confined or not lesion:
            buf = self.buf
            guard = (self._shared.read_isolation("d6_multiref_wm") if self._shared is not None
                     else contextlib.nullcontext())
        else:
            buf = self._lesion_buf()
            guard = contextlib.nullcontext()
        return buf, guard, confined

    def resolve_anaphor(self, pronoun: str, lesion: bool = False) -> dict | None:
        """An anaphor turn (flag on): resume this session's held state, run the inter-turn span, then the cue-driven
        retrieval. The pronoun resolves to the WINNING register's referent (named by the binder codebook), or to
        None when no held bump wins (dead buffer / tie). None when the flag is off or nothing was ever held."""
        if not multiref_focus_bind_enabled() or not self.has_held_state():
            return None
        buf, guard, confined = self._focus_buf_and_guard(lesion)
        with guard:
            if not self._resume(buf):
                return None
            buf.hold(FOCUS_INTERTURN_STEPS)                # the inter-turn interval: a live bump self-sustains
            r = self._retrieve(buf)
            self._stash(buf)
        resolved = self._ref_of_slot.get(r["local"]) if r["ok"] else None
        self._last_focus = r
        self._own_focus = None if (lesion and not confined) else self._xedge_focus_pool(r)
        out = {"on": True, "kind": "resolve", "pronoun": str(pronoun), "resolved": resolved,
               "resolved_register": r["register"], "resolved_pool": r["pool"], "winner_rate": r["winner_rate"],
               "runner_up_rate": r["runner_up_rate"], "margin": r["margin"], "register_rates": r["register_rates"],
               "wta_rates": r["wta_rates"], "lesioned": bool(lesion), "focus_bind": True,
               "readout_residual": ("host rate relay register->focus WTA; argmax+dead-margin read of the WTA winner; "
                                    "bank argmax for the local slot; binder codebook name")}
        if confined:
            out["lesion_scope"] = "recur"
        return out

    def read_held(self, lesion: bool = False) -> dict | None:
        """A hold-query (flag on): resume this session's held state, run the inter-turn span, and read every register
        off it (zero input). Returns the registers whose bump is alive and their referents -- no re-load from the host
        `_slot_of_ref`."""
        if not multiref_focus_bind_enabled() or not self.has_held_state():
            return None
        buf, guard, confined = self._focus_buf_and_guard(lesion)
        with guard:
            if not self._resume(buf):
                return None
            buf.hold(FOCUS_INTERTURN_STEPS)
            held, amps = [], []
            for reg in range(buf.R):
                loc, amp = buf.read(reg)
                amps.append(float(amp))
                if loc >= 0:
                    held.append((reg, self._ref_of_slot.get(loc)))
            self._stash(buf)
        live = [amp for amp in amps if amp > 1e-6]
        return {"recovered": {str(reg): name for reg, name in held}, "n_referents": len(held),
                "hold_alive_min": float(min(live)) if live else 0.0, "register_amps": [round(a, 6) for a in amps],
                "confined": bool(confined)}

    def judge(self, text: str, lesion: bool = False, xedge_drop_current=None) -> dict | None:
        """Production entry. Returns None when the input is OUT OF SCOPE (fewer than 2 referents AND not a hold-query)
        -> the caller leaves the turn byte-identical. Otherwise a dict with the held referents recovered off the
        spiking buffer and (on a hold-query) an honest functional read-out string.

        `xedge_drop_current` (default None -> byte-identical): forwarded verbatim to `load()` -- see its own
        docstring. The caller (webapp/server.py) only ever supplies this on the hold-query path, and only when the
        curiosity->d6 cross-edge's own validated crave-suppression signal is live."""
        self.ensure_built()
        refs = extract_referents(text, referent_lexicon=self.referent_lexicon)
        query = is_hold_query(text)
        if query and len(refs) < 2 and multiref_focus_bind_enabled():
            # REFERENT->FOCUS BIND: what is held is READ off this session's live resumed state, not re-loaded from the
            # host `_slot_of_ref` (None when nothing was ever held -> out of scope, as before).
            h = self.read_held(lesion=lesion)
            if h is None:
                return None
            names = [h["recovered"][k] for k in sorted(h["recovered"], key=int)]
            out = {"on": True, "lesioned": bool(lesion), "in_scope": True, "composer": "onebrain",
                   "n_referents": h["n_referents"], "input_order": [], "recovered": h["recovered"],
                   "hold_alive_min": h["hold_alive_min"], "zero_input_ok": True,
                   "all_recovered": bool(h["n_referents"] >= 1 and all(names)), "is_hold_query": True,
                   "focus_bind": True, "readout": hold_readout(names)}
            if h["confined"]:
                out["lesion_scope"] = "recur"
            return out
        # SCOPE: only a genuine multi-referent situation (>=2 named referents) or an explicit hold-query while >=2 are
        # already held. A single referent / no referents / a non-query turn is out of scope -> None (byte-identical).
        if len(refs) < 2 and not (query and len(self._slot_of_ref) >= 2):
            return None
        if len(refs) >= 2:
            res = self.load(refs, lesion=lesion, xedge_drop_current=xedge_drop_current)
        else:
            # a hold-query with no new referents: re-materialize + read the currently-known referents
            held = list(self._slot_of_ref.keys())[:R_MAX]
            res = self.load(held, lesion=lesion, xedge_drop_current=xedge_drop_current) if held else {
                "recovered": {}, "n_referents": 0, "hold_alive_min": 0.0, "input_order": [],
                "zero_input_ok": True, "all_recovered": False}
        out = {
            "on": True, "lesioned": bool(lesion), "in_scope": True, "composer": "onebrain",
            "n_referents": res["n_referents"], "input_order": res["input_order"],
            "recovered": {str(k): v for k, v in res["recovered"].items()},
            "hold_alive_min": res["hold_alive_min"], "zero_input_ok": res["zero_input_ok"],
            "all_recovered": res["all_recovered"], "is_hold_query": bool(query),
        }
        if res.get("recur_lesioned"):
            out["lesion_scope"] = "recur"   # EDGE-CONFINED lesion (only present when that knob is on)
        if query:
            out["readout"] = hold_readout([res["recovered"].get(r) for r in range(res["n_referents"])])
        return out

    def current_focus(self):
        """THIS session's own ONE-BRAIN XEDGE focus (the positional candidate pool ITS OWN held referents set),
        or None if this session holds nothing. Derived ONLY from `self._own_focus` (set by `load()`, never by
        another organ instance) -- a brand-new organ with an empty `_slot_of_ref` always returns None here,
        regardless of what any other session's organ has ever held. The caller (webapp/server.py) passes this
        explicitly into the comprehension organ's `wm_focus` argument every turn (2026-08-27 leak fix)."""
        return self._own_focus

    def clear_focus(self):
        """Explicitly forget this session's held xedge focus (turn-start / session-teardown hygiene). Only ever
        touches THIS organ instance's own state -- never the shared process pool, never another session."""
        self._own_focus = None


_ORGAN: MultiReferentWMOrgan | None = None


def get_organ(seed: int = 42) -> MultiReferentWMOrgan:
    """The process-shared multi-referent WM organ (built once on first use). When the ONE-BRAIN WAVE-3 pool flag is
    ON (`BRAIN_ONEBRAIN_WAVE3_POOL`, default-OFF) this organ's R_MAX banks + shared FS co-inhabit the process-shared
    11-organ Wave-3 `merge_organs` pool (`research/runners/onebrain_wave3_pool_production.py`) instead of its own
    private bridge; OFF (default) -> `shared=None`, byte-identical to today. NOTE (honest scope): this module-level
    singleton is NOT what production actually calls -- `webapp/server.py::_get_multiref_organ` builds one
    `MultiReferentWMOrgan` PER SESSION (cache_key-isolated: this organ's held referents are one conversation's, and
    a process singleton would leak one session's referents into another's read-back) with its OWN `shared` resolution
    (currently the xedge pool). So flipping `BRAIN_ONEBRAIN_WAVE3_POOL` has ZERO effect on production today, exactly
    like the other 9 organs' `get_organ()`s the Wave-3 pool module's own docstring already declares -- this wiring
    only readies the DEAD (uncalled) process-shared accessor for a future, separate per-session-safe landing."""
    global _ORGAN
    if _ORGAN is None:
        from research.runners.onebrain_wave3_pool_production import wave3_pool_enabled, get_wave3_pool
        shared = get_wave3_pool(seed) if wave3_pool_enabled() else None
        _ORGAN = MultiReferentWMOrgan(seed=seed, shared=shared)
    return _ORGAN


def resolve_turn(chat, organ, msg, lesion: bool = False):
    """REFERENT->FOCUS BIND production hook (BRAIN_MULTIREF_FOCUS_BIND, default OFF -> returns None, touches nothing).
    On a turn that is not a >=2-referent intro and not a hold-query: if this session's organ holds state, find the
    first anaphor token with the ChatBrain's OWN spiking CA3 detector (`_is_anaphor_token`, the sole detection path),
    resolve it by the organ's cue-driven retrieval, and publish the result for THIS turn's `_resolve_anaphora` as
    `chat._multiref_referent_override` = {question, pronoun, referent (None = unresolved)}. Returns the organ's
    resolution record (the reply's `multiref`, kind "resolve"), or None when out of scope."""
    if not multiref_focus_bind_enabled() or organ is None or not organ.has_held_state():
        return None
    detect = getattr(chat, "_is_anaphor_token", None)
    if detect is None:
        return None
    for tok in (msg or "").split():
        tl = tok.lower().strip(".,!?")
        if tl and detect(tl):
            r = organ.resolve_anaphor(tl, lesion=lesion)
            if r is None:
                return None
            chat._multiref_referent_override = {"question": msg, "pronoun": tl, "referent": r.get("resolved")}
            return r
    return None


def hold_readout(referents) -> str:
    """An honest functional read-out of what the spiking multi-referent buffer currently holds (never a phenomenal
    claim). Reads BACK off the held bumps -> what a single-attractor store cannot report (it ties to one)."""
    refs = [r for r in (referents or []) if r]
    if not refs:
        return "I'm not holding any referent in working memory right now."
    if len(refs) == 1:
        return f"I'm holding one referent in working memory: {refs[0]}."
    joined = ", ".join(refs[:-1]) + f" and {refs[-1]}"
    return f"I'm holding {len(refs)} referents in working memory at once: {joined}."
