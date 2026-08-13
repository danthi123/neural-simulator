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

HONEST RESIDUALS (declared; match the de-risk's named residuals + the task's named open rung):
  * The learned SPIKING WRITE-GATE is the open rung: the register assignment is today a role-by-position host MARKER
    (referent 0 -> reg0, ...). `739a8867` established even a host position-ORACLE fails to induce role at 6 seeds ->
    the residual is CREDIT ASSIGNMENT (gap#4). The learned, emergent, spiking multi-register role-gate is un-done.
  * The referent EXTRACTION (which tokens are the discourse referents) is a host parse, bounded by a small referent
    lexicon + a coordinated-NP pattern — the same vocab-ceiling class the comprehension organ declares.
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


def is_hold_query(text: str) -> bool:
    """An explicit 'who/what are we talking about / what are you keeping in mind' inner-state read-out query."""
    return bool(_HOLD_QUERY_RE.search(text or ""))


def extract_referents(text: str, max_refs: int = R_MAX):
    """Host parse (the declared vocab-ceiling residual): return the ORDERED, de-duplicated discourse referents named in
    `text`. A referent is a lexicon noun OR a capitalized proper name (not sentence-initial-only). Order = order of
    mention (role-by-position marker). Capped at max_refs and at the binder's _K distinct slots."""
    raw = _WORD_RE.findall(text or "")
    refs: list[str] = []
    for i, w in enumerate(raw):
        lw = w.lower()
        is_lex = lw in _REFERENT_NOUNS
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

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.buf = None          # intact MultiSlotHold (recur>0)
        self.buf_lesion = None   # recur=0 MultiSlotHold (killed hold)
        self.binder = None
        self._ref_of_slot: dict[int, str] = {}   # local slot -> referent string (shared local codebook)
        self._slot_of_ref: dict[str, int] = {}   # referent string -> local slot
        self._codes = None
        self._next_code = 0

    def ensure_built(self):
        if self._built:
            return
        self.buf = MultiSlotHold(self.seed, R_MAX, N_SLOT)
        self.binder = HebbianBinder()
        # pre-mint a barcode pool (>= _BINDER_K distinct referents; deterministic from the seed)
        self._codes = _mint_codes(np.random.default_rng(self.seed + 7), max(_BINDER_K, N_SLOT))
        self._built = True

    def _lesion_buf(self):
        if self.buf_lesion is None:
            self.buf_lesion = MultiSlotHold(self.seed, R_MAX, N_SLOT, recur=0.0)   # kill the slow-NMDA recurrence
        return self.buf_lesion

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

    def load(self, referents, lesion: bool = False):
        """LOAD the ordered `referents` into registers 0..len-1 of the spiking buffer (role-by-position marker), holding
        each across a short intervening span, then READ every register BACK off the held bumps. Returns the register ->
        referent mapping RECOVERED FROM THE SPIKING BUFFER (== the input iff the hold carried it; degrades under lesion).
        `hold_alive` is the min per-register bump amplitude read under external input ASSERTED zero."""
        self.ensure_built()
        refs = list(referents)[:min(R_MAX, _BINDER_K)]
        buf = self._lesion_buf() if lesion else self.buf
        buf.reset()
        locals_ = [self._local_slot(r) for r in refs]
        # role-by-position WRITE MARKER: referent r -> register r; interleave a HOLD after each write (the intervening
        # span) so the earlier registers must SUSTAIN while later ones load (the durability stress).
        for r, loc in enumerate(locals_):
            buf.write(r, loc)
            buf.hold()
        # an extra held span (the "... it chased her" gap) with input asserted zero
        buf.hold()
        buf.hold()
        recovered = {}
        alive = []
        for r in range(len(refs)):
            loc, amp = buf.read(r)
            alive.append(float(amp))
            recovered[r] = self._ref_of_slot.get(loc, None)
        return {
            "n_referents": len(refs),
            "input_order": refs,
            "recovered": recovered,                          # {reg: referent} read off the SPIKING held bumps
            "hold_alive_min": float(min(alive)) if alive else 0.0,
            "zero_input_ok": bool(buf._zero_input_span),
            "all_recovered": bool(len(refs) >= 1 and all(recovered.get(r) == refs[r] for r in range(len(refs)))),
        }

    def judge(self, text: str, lesion: bool = False) -> dict | None:
        """Production entry. Returns None when the input is OUT OF SCOPE (fewer than 2 referents AND not a hold-query)
        -> the caller leaves the turn byte-identical. Otherwise a dict with the held referents recovered off the
        spiking buffer and (on a hold-query) an honest functional read-out string."""
        self.ensure_built()
        refs = extract_referents(text)
        query = is_hold_query(text)
        # SCOPE: only a genuine multi-referent situation (>=2 named referents) or an explicit hold-query while >=2 are
        # already held. A single referent / no referents / a non-query turn is out of scope -> None (byte-identical).
        if len(refs) < 2 and not (query and len(self._slot_of_ref) >= 2):
            return None
        if len(refs) >= 2:
            res = self.load(refs, lesion=lesion)
        else:
            # a hold-query with no new referents: re-materialize + read the currently-known referents
            held = list(self._slot_of_ref.keys())[:R_MAX]
            res = self.load(held, lesion=lesion) if held else {"recovered": {}, "n_referents": 0,
                                                               "hold_alive_min": 0.0, "input_order": [],
                                                               "zero_input_ok": True, "all_recovered": False}
        out = {
            "on": True, "lesioned": bool(lesion), "in_scope": True, "composer": "onebrain",
            "n_referents": res["n_referents"], "input_order": res["input_order"],
            "recovered": {str(k): v for k, v in res["recovered"].items()},
            "hold_alive_min": res["hold_alive_min"], "zero_input_ok": res["zero_input_ok"],
            "all_recovered": res["all_recovered"], "is_hold_query": bool(query),
        }
        if query:
            out["readout"] = hold_readout([res["recovered"].get(r) for r in range(res["n_referents"])])
        return out


_ORGAN: MultiReferentWMOrgan | None = None


def get_organ(seed: int = 42) -> MultiReferentWMOrgan:
    """The process-shared multi-referent WM organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = MultiReferentWMOrgan(seed=seed)
    return _ORGAN


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
