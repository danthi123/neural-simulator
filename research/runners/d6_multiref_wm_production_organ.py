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

COMPETITIVE FREE-SLOT-WINS ALLOCATION (2026-09-01, additive, default-OFF `BRAIN_MULTIREF_COMPETITIVE`): closes the
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
  * The learned SPIKING WRITE-GATE is the open rung: the register assignment is today a role-by-position host MARKER
    (referent 0 -> reg0, ...) UNLESS `BRAIN_MULTIREF_COMPETITIVE=1` (above), which substitutes a genuine occupancy
    READ for the position marker. `739a8867` established even a host position-ORACLE fails to induce role at 6 seeds
    -> the residual is CREDIT ASSIGNMENT (gap#4). The learned, emergent, spiking multi-register role-gate is un-done.
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


def multiref_competitive_enabled() -> bool:
    """Default-OFF (additive; 2026-09-01). `BRAIN_MULTIREF_COMPETITIVE` in {1,true,yes,on} switches register
    ALLOCATION from the role-by-position host MARKER (referent i -> register i) to the EMERGENT free-slot-wins
    competitive read (`MultiSlotHold.probe_occupancy()` -> argmin): the brain's own current occupancy, not
    sentence position, decides which register binds a new referent. Off -> byte-identical to the pre-existing
    role-by-position path (the untouched default)."""
    v = os.environ.get("BRAIN_MULTIREF_COMPETITIVE")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


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
        if refs and getattr(self._shared, "xedge_codrive_params", None) is not None and not lesion:
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
        }

    def judge(self, text: str, lesion: bool = False, xedge_drop_current=None) -> dict | None:
        """Production entry. Returns None when the input is OUT OF SCOPE (fewer than 2 referents AND not a hold-query)
        -> the caller leaves the turn byte-identical. Otherwise a dict with the held referents recovered off the
        spiking buffer and (on a hold-query) an honest functional read-out string.

        `xedge_drop_current` (default None -> byte-identical): forwarded verbatim to `load()` -- see its own
        docstring. The caller (webapp/server.py) only ever supplies this on the hold-query path, and only when the
        curiosity->d6 cross-edge's own validated crave-suppression signal is live."""
        self.ensure_built()
        refs = extract_referents(text)
        query = is_hold_query(text)
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
