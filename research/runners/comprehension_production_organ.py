"""COMPREHENSION MEASUREMENT wired into the PRODUCTION conversational turn (Gate-B, 2026-08-12).

The owner's "measurement of understanding of spoken language": before the brain acts on an incoming
transitive assertion, it reads a genuinely-SPIKING signal of whether its role-binding RESOLVED cleanly,
so it can honestly say "my role-binding didn't resolve — I didn't follow that" instead of silently
ingesting an utterance it did not comprehend. This STRENGTHENS the no-confab moat (it never weakens it):
the brain refuses to store / answer on content whose thematic roles its own substrate could not resolve.

It REUSES (does not reinvent) the adversarially-verified D4 faculty
(`research/runners/_spiking_comprehension_monitor_derisk.py`, 6/6 GO, type-2 AUC=1.000, lesion->0.500):
the on-brain `SpikingRoleCompetition`'s two Wong-Wang accumulator pools (`sel_agent`/`sel_patient`, mutual
inhibition) settle to a firing margin `|agentEv_0 - agentEv_1|` read off `bridge.cp_firing_states` when
driven by the SEMANTIC (content: animacy + verbfit) cues only. HIGH when the content decisively separates
the two nouns (a well-formed transitive whose roles resolve), LOW when the content cancels (two-animate /
two-inanimate ambiguity) or is absent (out-of-vocabulary). The host `_semantic_contrast` dot-product (the
shortcut this replaces) is NEVER called — the read is on firing neurons.

BRAIN-BASED: the comprehension scalar is a `cp_firing_states` read (`comp._noun_role_rates`), no host formula.
The DECISION to abstain is a threshold on that spiking margin (calibrated at build from a small battery).

MOAT-SAFE + NON-REGRESSIVE by CONSTRUCTION:
  * SCOPE: fires ONLY on a 3-content-token transitive ASSERTION (agent verb patient) that the monitor is
    COMPETENT to judge — both nouns either toy-cue-COVERED (in the ANIMACY lexicon) or genuinely OOV (not in
    the brain's own vocabulary). Questions (patient is the query -> 2 content tokens), self/identity queries,
    anaphora, open-ended prompts and feel-queries are OUT OF SCOPE -> byte-identical, unchanged.
  * NEVER blocks a KNOWN fact: an asserted triple the brain already holds (`what_does(a,v)==p`) is honored
    (comprehension never abstains on a fact the substrate recalls).
  * Real-but-untabled vocabulary (a word the brain knows that is not in the toy ANIMACY table) is OUT OF the
    monitor's competence -> passed through unchanged (no false abstain). This is the declared RESIDUAL: the
    monitor's competence is bounded by its cue lexicon (the de-risk's 2-noun transitive toy vocab + OOV).

LESION-LOAD-BEARING: zeroing the learned cue->role synaptic weights (`BRAIN_COMPREHENSION_LESION=1`) collapses
the margin to chance (the de-risk's AUC 1.000 -> 0.500), so the well/ill separation — and thus the honest
abstain decision — disappears. The host cue VALUES are byte-identical with/without the lesion, so the
discrimination is caused by the learned spiking competition, not the host cue table.

HONEST RESIDUALS (declared, ride existing burn-down items):
  * CO-RESIDENT: the comprehension monitor runs on ITS OWN `SpikingRoleCompetition` bridge, ALONGSIDE the
    recall composer, not merged onto the ONE recall bridge — rides on the one-brain merge (burn-down #1),
    exactly as the affect organ does.
  * VOCAB CEILING: the cue lexicon (ANIMACY / VERB_SELECTS) is the toy 2-noun transitive scope; out-of-table
    real vocabulary is passed through (not judged). **PARTIALLY CLOSED (2026-08-27,
    `BRAIN_LEARNED_ANIMACY_CUE`, default OFF)**: ANIMACY membership now extends to a corpus-LEARNED,
    spiking-realized open-vocab cue (`_comprehension_learned_animacy_spiking.py`, 6-seed GO) via the single
    `_animacy_of` choke point. VERB_SELECTS stays the hand-coded closed set (declared residual — no GO
    artifact validates an open-vocab verb-selects cue); calibrating on a graded/near-threshold battery is
    the next rung (the de-risk's mapped residual).
  * STRUCTURAL malformedness (no verb / wrong arity) is still a host arity/shape check, not the spiking read.

Additive, default-ON, `BRAIN_COMPREHENSION_GATE=0` -> the byte-identical oracle (fully skipped). NO `sim/` edit;
uses the process backend (cupy in production, numpy in tests) via reuse-by-import.
"""
from __future__ import annotations

import os
import re

import numpy as np

from research.runners._spiking_comprehension_monitor_derisk import (
    _build_comp,
    cue_evidence,
    _agent_evidence_from_spikes,
    semantic_sel_margin,
    SEMANTIC_CUES,
    build_battery,
    ANIMATE,
    INANIM,
    ASYMM_VERBS,
    OOV_NOUNS,
)
from research.runners._phaseB_multicue_competition_spiking_derisk import (
    ANIMACY,
    VERB_SELECTS,
    CUES,
)
from research.runners._comprehension_learned_animacy_spiking import get_lexicon as _get_learned_animacy_lexicon

# Function words stripped to expose the transitive content (agent verb patient). Deliberately minimal so a
# 3-content-token declarative assertion resolves while a WH-question (patient is the query) reduces to <3.
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "that",
    "this", "these", "those", "it", "its", "they", "them", "he", "she", "his", "her", "their",
    "my", "your", "our", "i", "you", "we", "me", "us", "him", "on", "in", "at", "by", "with",
    "for", "as", "so", "then", "now", "just", "please", "does", "do", "did",
}
_WH = {"what", "who", "whom", "whose", "where", "when", "why", "how", "which"}
_WORD_RE = re.compile(r"[a-zA-Z']+")

# Read window for the sel-pool settle (matches the de-risk's default read_steps=60; ~0.1-0.3 s/turn on CPU).
READ_STEPS = 60

# ── OTHER-REPAIR targeting thresholds (T1-6) — calibrated at build from the well-formed pair-max commitment ──
# The repair reads the per-noun agent-evidence (a0, a1) off the SAME spiking sel-pools the D4 margin uses, to
# name WHICH thematic role the substrate could not resolve. Two calibrated scalars, both a fraction of the
# well-formed per-noun role commitment (so they track the substrate's own scale, per seed):
#   * ROLE_FLOOR_FRAC * mean_well_pairmax  -> the commitment-present floor. A covered transitive's roles are
#     ACTIVE when max(|a0|,|a1|) clears it; under the D4 lesion (learned cue->role synapses zeroed) both
#     collapse to ~0 -> below the floor -> NO role target -> the bare abstain (the load-bearing fallback).
#   * lean_margin = mean_well_pairmax      -> the net-lean confidence. sign(a0+a1) names the OVER-subscribed
#     role only when |a0+a1| clears it: a two-inanimate transitive (both nouns lean PATIENT, animacy+verbfit
#     agree) yields a strong negative net lean -> the AGENT slot is the unresolved one; a two-animate
#     transitive (symmetric verb, weak/ambiguous direction) stays within the margin -> a generic role-swap
#     clarification (honest: the substrate cannot confidently say which role is over-subscribed).
ROLE_FLOOR_FRAC = 0.2

# 2026-08-27 CROSS-SESSION xedge_focus LEAK FIX (research/FAILURE_LOG.md): the ONE-BRAIN XEDGE co-drive used to be
# read off `self._shared.xedge_focus`, a single mutable attribute on the ONE process-shared MergedPool -- written
# by WHICHEVER session's d6 organ last held >=2 referents, read by EVERY session's comprehension call thereafter
# (a cross-session WM-focus latch). `wm_focus` below is now an EXPLICIT per-call argument: the caller (webapp/
# server.py) resolves it from the REQUESTING session's own d6 organ (`MultiReferentWMOrgan.current_focus()`, which
# now stores this session's focus on the ORGAN INSTANCE, never on the shared pool) and passes it down. The
# `_WM_FOCUS_UNSET` sentinel keeps the OLD ambient-global read as a fallback ONLY for callers that do not pass the
# kwarg (the offline self-tests below and `research/findings/raw/_onebrain_xedge_session_leak_verify/`), so their
# byte-identical behaviour is untouched; production now always passes an explicit value (even None), so it never
# consults the ambient global.
_WM_FOCUS_UNSET = object()


def comprehension_enabled() -> bool:
    """Default-ON. `BRAIN_COMPREHENSION_GATE` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_COMPREHENSION_GATE")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def comprehension_lesioned() -> bool:
    """`BRAIN_COMPREHENSION_LESION` in {1,true,yes,on} -> zero the learned cue->role weights (load-bearing lesion)."""
    v = os.environ.get("BRAIN_COMPREHENSION_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def learned_animacy_cue_enabled() -> bool:
    """`BRAIN_LEARNED_ANIMACY_CUE` in {1,true,yes,on} -> extend the ANIMACY membership scope beyond the
    ~19-noun hand table to the corpus-LEARNED, spiking-realized open-vocab animacy cue (6-seed GO:
    `research/findings/raw/_comprehension_learned_animacy_cue_6seed.json` -- learned=0.837,
    shuffled-graph=0.504, frequency-only=0.511, gap=+0.333; spiking-realization verify:
    `research/findings/raw/_comprehension_learned_animacy_spiking_verify.json`). Default OFF: unset behaves
    byte-identically to the pre-existing hand-table-only scope."""
    v = os.environ.get("BRAIN_LEARNED_ANIMACY_CUE")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def learned_animacy_cue_lesioned() -> bool:
    """`BRAIN_LEARNED_ANIMACY_LESION` in {1,true,yes,on} -> zero the F_anim/F_inanim coupling (see
    `LearnedAnimacyLexicon.set_lesion`): every open-vocab `classify()` call abstains, so any word the hand
    ANIMACY table does not cover reverts to OUT OF SCOPE -- byte-identical to the flag being off, for that
    word (the load-bearing check: the diff this flag introduces must vanish under this lesion)."""
    v = os.environ.get("BRAIN_LEARNED_ANIMACY_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _animacy_of(n: str):
    """ANIMACY membership + category for noun `n`: "animate" / "inanimate" / None. The hand ANIMACY table is
    the fast path (checked first, always -- byte-identical whether or not the flag below is set). When the
    hand table misses AND `BRAIN_LEARNED_ANIMACY_CUE` is on, falls through to the corpus-learned,
    spiking-realized open-vocab lexicon (`_comprehension_learned_animacy_spiking.LearnedAnimacyLexicon`).
    This is the SINGLE choke point every `n in ANIMACY` membership test below was converted to call."""
    cat = ANIMACY.get(n)
    if cat is not None:
        return cat
    if not learned_animacy_cue_enabled():
        return None
    lex = _get_learned_animacy_lexicon()
    lex.set_lesion(learned_animacy_cue_lesioned())
    return lex.classify(n)


def _evs_for_organ(n0: str, v: str, n1: str):
    """Cue evidence for (n0, v, n1) -- the organ's own `_evs_for`, extended (flag ON) to feed the
    corpus-learned animacy CATEGORY for a noun the hand ANIMACY table lacks into the ALREADY-VALIDATED
    `SpikingRoleCompetition` (D4's AUC=1.000, lesion->0.500 circuit, left untouched). `cue_evidence`'s own
    `permute_map` parameter (built for its permuted-cue anti-cheat) is reused, as designed, to remap such a
    noun to a canonical proxy word of the SAME learned category ("dog" for animate, "ball" for inanimate) --
    so the existing validated competition reads the correct signed animacy (and, transitively, verbfit) vote
    for it, without touching `_phaseB_multicue_competition_spiking_derisk.py`. Byte-identical to
    `_spiking_comprehension_monitor_derisk._evs_for` when the flag is off, or when both nouns are already
    hand-table-covered (permute_map is then empty -> None, the same no-remap call `_evs_for` makes)."""
    pm = {}
    if learned_animacy_cue_enabled():
        for n in (n0, n1):
            if n in ANIMACY:
                continue
            cat = _animacy_of(n)
            if cat is not None:
                pm[n] = "dog" if cat == "animate" else "ball"
    return [
        cue_evidence(n0, 0, 2, v, sent_id=0, clean_cues=True, permute_map=(pm or None)),
        cue_evidence(n1, 1, 2, v, sent_id=0, clean_cues=True, permute_map=(pm or None)),
    ]


def _lemma_verb(v: str) -> str:
    """Map an inflected surface verb to its base form ONLY when the base is a known selectional verb (so a
    real OOV verb stays OOV). 'eats'/'eating'/'chased'/'carries' -> 'eat'/'chase'/'carry'; unknown unchanged."""
    if v in VERB_SELECTS:
        return v
    cands = []
    if v.endswith("ies"):
        cands.append(v[:-3] + "y")                    # carries -> carry
    for suf in ("ing", "es", "ed", "s"):
        if v.endswith(suf):
            base = v[: len(v) - len(suf)]
            cands += [base, base + "e"]
            if len(base) >= 2 and base[-1] == base[-2]:
                cands.append(base[:-1])               # grabbing -> grab (undouble)
    for cand in cands:
        if cand in VERB_SELECTS:
            return cand
    return v


def _lemma_noun(n: str) -> str:
    """Strip a plural 's' ONLY when the singular is a cue-covered noun (so a real OOV noun stays OOV).
    "Cue-covered" is `_animacy_of` -- the hand ANIMACY table, extended (flag ON) to the learned lexicon."""
    if _animacy_of(n) is not None:
        return n
    if n.endswith("s") and _animacy_of(n[:-1]) is not None:
        return n[:-1]
    return n


def extract_transitive(text: str):
    """Return (noun0, verb, noun1) when `text` is a 3-content-token transitive (agent verb patient), else None.
    Strips determiners/aux/pronouns; a WH-question reduces to <3 content tokens -> None (out of scope). Inflected
    surface forms are lemmatized to their base ONLY when the base is a KNOWN cue word (real OOV stays OOV)."""
    toks = [w.lower() for w in _WORD_RE.findall(text or "")]
    if any(t in _WH for t in toks) or "?" in (text or ""):
        return None                                  # a question is not an assertion the monitor scores
    content = [t for t in toks if t not in _FUNCTION_WORDS]
    if len(content) != 3:
        return None
    n0, v, n1 = content
    n0, v, n1 = _lemma_noun(n0), _lemma_verb(v), _lemma_noun(n1)
    if n0 == v or v == n1:                            # degenerate
        return None
    return n0, v, n1


class ComprehensionProductionOrgan:
    """A process-shared spiking comprehension monitor. Built ONCE (lazily): a `SpikingRoleCompetition` with the
    installed cue weights, plus a build-time calibration of the well-vs-ill margin threshold from a small battery.
    Each read drives the SEMANTIC cues for the two nouns, settles the Wong-Wang sel WTA, and reads the spiking
    margin `|agentEv_0 - agentEv_1|` off `cp_firing_states`. Cheap (~0.1-0.3 s/turn CPU)."""

    def __init__(self, seed: int = 42, shared=None):
        self.seed = int(seed)
        # ONE-BRAIN MERGE (opt-in, byte-identical when shared is None): when a MergedPool is injected, the
        # comprehension `SpikingRoleCompetition` runs on the pool's sel_*/sel_FS_*/cue_* SLICE of the SHARED spiking
        # bridge (already built, wired per-region-seamed, and settled-to-rest by the pool). The read is a FROZEN
        # forward pass (plasticity gates 0, weights installed), so nothing mutates a co-resident slice; each read is
        # wrapped in the pool's read_isolation. None -> the organ builds its own comp bridge exactly as today.
        self._shared = shared
        self._built = False
        self.comp = None
        self.comp_lesion = None
        self.threshold = None
        self.role_floor = None   # (T1-6) commitment-present floor for the other-repair role read
        self.lean_margin = None  # (T1-6) net-lean confidence for naming the over-subscribed role
        self.calib = None
        self._rest = {}          # comp-bridge id -> (rest_v, rest_u) resting snapshot for a hard per-turn reset
        # ONE-BRAIN XEDGE — the WM-resolved-role decision (closes the sub-decision caveat): on a content-inconclusive
        # transitive with a WM referent held, the balanced (content-cancelled) read is SIGNED by the cross-edge, so
        # the held referent's LEARNED role becomes the role tiebreaker. `_wm_baseline` is the no-WM balanced margin
        # (cross-edge-independent, computed lazily once); `_wm_resolve_eps` is the min |margin - baseline| that counts
        # as a resolution (so lesioned/no-signal stays inconclusive).
        self._wm_baseline = None
        self._wm_resolve_eps = 0.005

    def _guard(self):
        """The pool's read_isolation on the SHARED path (restores every co-resident organ's slice at the end so
        only comprehension's slice may evolve during a read), else a no-op context."""
        import contextlib
        if self._shared is not None:
            return self._shared.read_isolation("comprehension")
        return contextlib.nullcontext()

    def _snapshot_rest(self, comp):
        """Snapshot a comp's resting state (after a brief settle) so every per-turn read starts IDENTICAL — the
        NMDA-slow Wong-Wang sel pools do not fully quiesce in the internal 8-step soft reset, so without a hard
        reset the margin would depend on the PRIOR turn. Mirrors the surprise de-risk's _hard_reset protocol.

        SHARED path: the pool already settled the WHOLE bridge to a quiescent rest and snapshotted it (`pool.snap`),
        so the rest is that snapshot's (v,u) -- restoring it in _hard_reset returns EVERY slice to rest (co-residents
        included, so no clobber), and no extra settle steps footprint the shared bridge."""
        if self._shared is not None and self._shared.snap is not None:
            snap = self._shared.snap
            self._rest[id(comp)] = (np.asarray(snap["cp_membrane_potential_v"]).copy(),
                                    np.asarray(snap["cp_recovery_variable_u"]).copy())
            return
        b = comp.bridge
        b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            b._run_one_simulation_step()
        self._rest[id(comp)] = (b.cp_membrane_potential_v.copy(), b.cp_recovery_variable_u.copy())

    def _hard_reset(self, comp):
        """Restore the comp bridge to its resting snapshot (v,u) and zero all conductances/firing/refractory/current
        -> the sentence-level read is history-INDEPENDENT (deterministic per input)."""
        b = comp.bridge
        rest = self._rest.get(id(comp))
        if rest is not None:
            b.cp_membrane_potential_v[:] = rest[0]
            b.cp_recovery_variable_u[:] = rest[1]
        for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab",
                     "cp_conductance_g_nmda", "cp_firing_states", "cp_refractory"):
            arr = getattr(b, name, None)
            if arr is not None:
                arr[:] = 0
        b.cp_external_input_current[:] = 0.0

    def _xedge_codrive(self, comp, wm_focus=_WM_FOCUS_UNSET):
        """ONE-BRAIN CROSS-EDGE coupling (opt-in, byte-identical no-op when off). When the CALLER's OWN session
        has a FOCUS d6 candidate pool held (`wm_focus`, resolved by the caller from ITS OWN `MultiReferentWMOrgan`
        — never read off the shared pool's ambient global; see `_WM_FOCUS_UNSET` above), establish that pool's
        self-sustaining slow-NMDA WM bump on the shared bridge BEFORE the cue settle (R3-v3's amb_read protocol:
        drive LOAD_PA for LOAD_STEPS, then HOLD), so the FROZEN w{k}->sel cross-edge transmits the held WM state
        INTO this comprehension read — the `repair_target` net-lean (and the judge margin) then reflect it. The
        slow-NMDA bump self-sustains through the subsequent `_noun_role_rates` soft resets + cue windows (exactly
        as amb_read's held read does). No-op when `wm_focus` is None (this session holds nothing): shared=None
        (standalone), the flag off, or no referent held, is byte-identical. Also a no-op on a bridge that lacks the
        focus region (the standalone D4 cue-lesion twin) — the try/except degrades cleanly."""
        sh = self._shared
        foc = wm_focus
        if foc is _WM_FOCUS_UNSET:      # legacy fallback (self-tests only) -- production always passes explicitly
            foc = getattr(sh, "xedge_focus", None) if sh is not None else None
        if foc is None:
            return
        prm = getattr(sh, "xedge_codrive_params", None) or {}
        load_pa = float(prm.get("load_pa", 400.0))
        load_steps = int(prm.get("load_steps", 30))
        hold_steps = int(prm.get("hold_steps", 6))
        b, xp = comp.bridge, comp.xp
        try:
            idx = xp.asarray(np.asarray(b.region_manager.indices(foc), np.int64))
        except Exception:
            return
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        cur[idx] = xp.float32(load_pa)
        b.cp_external_input_current[:] = cur
        for _ in range(load_steps):
            b._run_one_simulation_step()
        b.cp_external_input_current[:] = 0.0
        for _ in range(hold_steps):
            b._run_one_simulation_step()

    def _wm_resolved_role(self, comp, wm_focus=_WM_FOCUS_UNSET):
        """ONE-BRAIN XEDGE (closes the sub-decision caveat). On a content-inconclusive transitive with a WM referent
        held BY THE CALLING SESSION (`wm_focus`, resolved by the caller from its OWN `MultiReferentWMOrgan` --
        NEVER the shared pool's ambient global; see `_WM_FOCUS_UNSET` above), RESOLVE the role from R3-v3's
        VALIDATED balanced `amb_read` (the F2 instrument the pool binds as `shared.xedge_amb_read`): drive BOTH
        animacy cue directions equally (CONTENT cancels to ~0), hold the focus WM pool, read the signed
        `sel_agent - sel_patient` margin -- the SIGN is the held referent's LEARNED role. role = sign(wm_margin -
        baseline), where baseline holds a NON-candidate control pool (no grown edge). Returns (role, wm_margin), or
        (None, wm_margin) when |delta| < eps (a lesioned / no-signal read stays inconclusive -> no false
        resolution). Byte-identical no-op (None, None) when shared=None / xedge off / no referent held (this
        session's own `wm_focus` is None). Uses the proven pool read rather than a reimplementation (a hand-rolled
        balanced read was NOT actually balanced)."""
        sh = self._shared
        amb = getattr(sh, "xedge_amb_read", None) if sh is not None else None
        foc = wm_focus
        if foc is _WM_FOCUS_UNSET:      # legacy fallback (self-tests only) -- production always passes explicitly
            foc = getattr(sh, "xedge_focus", None) if sh is not None else None
        if amb is None or foc is None:
            return None, None
        cues = getattr(sh, "xedge_balanced_cues", None)
        base_pool = getattr(sh, "xedge_base_pool", None)
        if cues is None or base_pool is None:
            return None, None
        if self._wm_baseline is None:                    # control-hold balanced margin (no grown edge), computed once
            self._wm_baseline = float(amb(base_pool, cues)["margin"])
            self._wm_resolve_eps = max(0.004, 3.0 * abs(self._wm_baseline))
        wm_m = float(amb(foc, cues)["margin"])
        delta = wm_m - float(self._wm_baseline)
        if abs(delta) < self._wm_resolve_eps:
            return None, wm_m
        return ("agent" if delta > 0 else "patient"), wm_m

    def ensure_built(self):
        if self._built:
            return
        # SHARED path: build the comp as a view over the pool slice (installs the cue validities + freezes the cue
        # plasticity gates on the pool's own edges). None -> its own learned cue->role competition, unchanged.
        self.comp = _build_comp(self.seed, shared=self._shared)
        self._snapshot_rest(self.comp)
        # calibrate the well-vs-ill threshold from a small deterministic battery (AUC=1.000 => a clean gap). Each
        # item is read from the SAME hard-reset resting state (as production reads are), so the threshold is on
        # the production read regime, not the de-risk's sequential-read regime.
        items = build_battery(self.seed, n_per_cond=6)
        well, ill = [], []
        # OTHER-REPAIR (T1-6) calibration is folded INTO this SAME loop (no extra reads): `_read_per_noun` performs
        # the byte-identical hard-reset + two `_agent_evidence_from_spikes` calls that `_read`/`semantic_sel_margin`
        # do, so `abs(a0 - a1)` reproduces the D4 margin EXACTLY and consumes the substrate RNG IDENTICALLY — the
        # D4 threshold (and every later production margin read) is unchanged. A SEPARATE read pass would advance the
        # bridge's stochastic state and perturb the production margins, so we must reuse this one.
        # SHARED path: the whole calibration is wrapped in ONE read_isolation so the co-resident slices are restored
        # (the per-noun reads hard-reset comprehension's slice to pool.snap first, so ordering within is immaterial).
        well_pairmax = []
        with self._guard():
            for (lab, _tag, n0, v, n1) in items:
                a0, a1 = self._read_per_noun(self.comp, n0, v, n1)
                m = abs(a0 - a1)                         # == self._read(...) byte-identical (same call sequence)
                (well if lab == 1 else ill).append(m)
                if lab == 1:
                    well_pairmax.append(max(abs(a0), abs(a1)))
        mean_well = float(np.mean(well)) if well else 0.30
        min_well = float(np.min(well)) if well else 0.30
        mean_ill = float(np.mean(ill)) if ill else 0.12
        max_ill = float(np.max(ill)) if ill else 0.12
        # place the threshold in the GAP, biased toward the ill side so a well-formed input reliably passes:
        # midway between the well FLOOR and the ill CEILING when they separate, else the class-mean midpoint.
        self.threshold = 0.5 * (min_well + max_ill) if min_well > max_ill else 0.5 * (mean_well + mean_ill)
        # the repair thresholds set the substrate's own role-evidence scale (a fraction of the well-formed pair-max
        # commitment). Kept OFF `self.calib` (which is echoed in the D4 `comprehension` response) so D4 is byte-identical.
        mean_well_pm = float(np.mean(well_pairmax)) if well_pairmax else 0.15
        self.role_floor = float(ROLE_FLOOR_FRAC * mean_well_pm)
        self.lean_margin = float(mean_well_pm)
        self.calib = {"mean_well": mean_well, "min_well": min_well, "mean_ill": mean_ill,
                      "max_ill": max_ill, "read_steps": READ_STEPS, "n_calib": len(items)}
        self._built = True

    def _lesion_comp(self):
        if self.comp_lesion is None:
            self.comp_lesion = _build_comp(self.seed)
            for c in CUES:
                self.comp_lesion.set_cue_weight(c, 0.0)   # zero the learned cue->role synapses (host cues unchanged)
            self._snapshot_rest(self.comp_lesion)
        return self.comp_lesion

    def _read(self, comp, n0: str, v: str, n1: str, wm_focus=_WM_FOCUS_UNSET) -> float:
        """Hard-reset `comp` to rest, then read the SEMANTIC sel-pool margin off cp_firing_states. `wm_focus` is the
        CALLING session's own xedge focus (or None); see `_xedge_codrive`."""
        self._hard_reset(comp)
        self._xedge_codrive(comp, wm_focus=wm_focus)      # ONE-BRAIN XEDGE (opt-in): co-drive the held WM pool
        return float(semantic_sel_margin(comp, _evs_for_organ(n0, v, n1), READ_STEPS))

    def _read_per_noun(self, comp, n0: str, v: str, n1: str, wm_focus=_WM_FOCUS_UNSET):
        """(T1-6) Hard-reset `comp` to rest, then read the PER-NOUN agent-evidence (a0, a1) = (sel_agent -
        sel_patient) firing for each noun off cp_firing_states, driven by the SEMANTIC (content) cues only —
        the SAME spiking reads whose |a0 - a1| is the D4 comprehension margin. `a_i > 0` = noun i leans AGENT,
        `< 0` = leans PATIENT. Their signs/magnitudes localise WHICH thematic role failed to resolve (the
        repair target). Under the learned-cue lesion both collapse to ~0 (no role activity). `wm_focus` is the
        CALLING session's own xedge focus (or None); see `_xedge_codrive`."""
        self._hard_reset(comp)
        self._xedge_codrive(comp, wm_focus=wm_focus)      # ONE-BRAIN XEDGE (opt-in): co-drive the held WM pool
        evs = _evs_for_organ(n0, v, n1)
        a0 = float(_agent_evidence_from_spikes(comp, evs[0], SEMANTIC_CUES, READ_STEPS))
        a1 = float(_agent_evidence_from_spikes(comp, evs[1], SEMANTIC_CUES, READ_STEPS))
        return a0, a1

    def read_margin(self, n0: str, v: str, n1: str, lesion: bool = False, wm_focus=_WM_FOCUS_UNSET) -> float:
        """The SPIKING comprehension margin for a transitive (n0, v, n1): |agentEv_0 - agentEv_1| from the
        SEMANTIC sel pools, read off cp_firing_states. Hard-resets the comp bridge first (history-independent).
        `lesion` uses the zeroed-cue competition (-> ~chance). `wm_focus` is the CALLING session's own xedge focus
        (None if this session holds nothing / xedge off) -- resolved by the CALLER (never read off a shared
        process-global; closes the 2026-08-27 cross-session xedge_focus leak). Omitting it falls back to the legacy
        ambient-global read for the offline self-tests only (see `_WM_FOCUS_UNSET`)."""
        self.ensure_built()
        comp = self._lesion_comp() if lesion else self.comp
        # SHARED path (intact read): guard the sel-WTA settle so the co-resident slices are restored afterwards.
        # The lesion twin is its OWN bridge (built standalone), so it needs no pool guard.
        if lesion:
            return self._read(comp, n0, v, n1, wm_focus=wm_focus)
        with self._guard():
            return self._read(comp, n0, v, n1, wm_focus=wm_focus)

    def competent(self, n0: str, v: str, n1: str, brain_vocab=None) -> bool:
        """Is the monitor COMPETENT to judge this transitive? It reads a RELIABLE signal only when the cues are
        cleanly resolved: EITHER the input is FULLY cue-covered (verb in VERB_SELECTS AND both nouns in ANIMACY),
        so animacy+verbfit both drive the competition; OR it is FULLY out-of-vocabulary (verb unknown AND both
        nouns unknown to both the cue lexicon and the brain), the genuine 'I heard only noise' case. A PARTIALLY
        recognized input (a covered noun with an unrecognized verb, or a real-but-untabled word the brain knows)
        is OUT of competence -> passed through unchanged (no unreliable read, no false abstain). This is the
        declared vocab-ceiling residual: the monitor's competence is bounded by its cue lexicon, extended
        (flag ON) to the corpus-learned open-vocab animacy cue via `_animacy_of` -- VERB_SELECTS itself stays
        the hand-coded closed set (no GO artifact validates an open-vocab verb-selects cue; declared residual)."""
        bv = brain_vocab or set()
        fully_covered = (v in VERB_SELECTS) and (_animacy_of(n0) is not None) and (_animacy_of(n1) is not None)

        def oov(n):
            return (_animacy_of(n) is None) and (n not in bv)
        fully_oov = (v not in VERB_SELECTS) and oov(n0) and oov(n1)
        return fully_covered or fully_oov

    def judge(self, text: str, brain_vocab=None, lesion: bool = False, wm_focus=_WM_FOCUS_UNSET) -> dict | None:
        """Read the comprehension of `text`. Returns None when the input is OUT OF SCOPE (not a competent 3-token
        transitive) -> the caller leaves the turn byte-identical. Otherwise a dict with the spiking margin, the
        threshold, and `comprehended` (margin >= threshold). `wm_focus` is the CALLING session's own xedge focus
        (see `read_margin`)."""
        tr = extract_transitive(text)
        if tr is None:
            return None
        n0, v, n1 = tr
        if not self.competent(n0, v, n1, brain_vocab=brain_vocab):
            return None                                # out of the monitor's cue-lexicon competence -> unchanged
        self.ensure_built()
        margin = self.read_margin(n0, v, n1, lesion=lesion, wm_focus=wm_focus)
        comprehended = bool(margin >= self.threshold)
        return {
            "on": True, "lesioned": bool(lesion), "in_scope": True,
            "svo": [n0, v, n1], "margin": float(margin), "threshold": float(self.threshold),
            "comprehended": comprehended, "calib": self.calib,
        }

    def repair_target(self, text: str, brain_vocab=None, lesion: bool = False, wm_focus=_WM_FOCUS_UNSET) -> dict | None:
        """(OTHER-REPAIR, T1-6) Given the SAME in-scope transitive the D4 gate abstained on, localise WHICH
        element the substrate could not resolve, so the turn can ask a TARGETED clarification instead of a bare
        abstain. Returns None when nothing can be targeted (-> the caller keeps the bare abstain). `wm_focus` is
        the CALLING session's own xedge focus (None if this session holds no referent / xedge off) -- resolved by
        the CALLER from ITS OWN `MultiReferentWMOrgan`, never read off the shared process pool's ambient global
        (closes the 2026-08-27 cross-session xedge_focus leak; see `_WM_FOCUS_UNSET`). Two branches:

          * OOV (host-lexical scaffold, NOT load-bearing on the spiking read — a declared residual, exactly like
            curiosity's host topic extractor): when a content token is genuinely out of the brain's vocabulary,
            name it. The identity of the unknown word is a lexical fact, not a role-competition read.

          * ROLE (fully spiking, LESION-LOAD-BEARING): for a FULLY cue-covered transitive whose roles did not
            separate, read the per-noun agent-evidence (a0, a1) off cp_firing_states. `max(|a0|,|a1|)` must clear
            the calibrated commitment floor (roles are ACTIVE) — under the D4 lesion both collapse to ~0, so this
            fails -> None -> the bare abstain. When active, the net lean sign(a0+a1) names the OVER-subscribed
            role (so the OTHER role is the unresolved one) when |a0+a1| clears the lean margin; otherwise a
            generic role-swap target (the substrate cannot confidently say which role is over-subscribed)."""
        tr = extract_transitive(text)
        if tr is None:
            return None
        n0, v, n1 = tr
        if not self.competent(n0, v, n1, brain_vocab=brain_vocab):
            return None
        self.ensure_built()
        base = {"on": True, "lesioned": bool(lesion), "svo": [n0, v, n1]}

        # ── OOV branch (host lexical): name the token(s) the brain does not know. ──
        bv = brain_vocab or set()
        oov_tokens = [n for n in (n0, n1) if (_animacy_of(n) is None) and (n not in bv)]
        if v not in VERB_SELECTS and v not in bv:
            oov_tokens.append(v)
        if oov_tokens:
            base.update(kind="oov", oov_tokens=oov_tokens, role=None, word=None,
                        loadbearing="host_lexical")
            return base

        # ── ROLE branch (spiking, load-bearing): the roles are covered but did not separate. ──
        comp = self._lesion_comp() if lesion else self.comp
        if lesion:
            a0, a1 = self._read_per_noun(comp, n0, v, n1, wm_focus=wm_focus)
        else:
            with self._guard():                          # SHARED path: restore co-resident slices after the read
                a0, a1 = self._read_per_noun(comp, n0, v, n1, wm_focus=wm_focus)
        pair_max = max(abs(a0), abs(a1))
        net_lean = a0 + a1
        base.update(a0=float(a0), a1=float(a1), pair_max=float(pair_max), net_lean=float(net_lean),
                    role_floor=float(self.role_floor), lean_margin=float(self.lean_margin),
                    loadbearing="spiking_role_evidence")
        if pair_max < self.role_floor:
            # no role activity at all (the D4 lesion collapses both a_i to ~0) -> nothing to target.
            base.update(kind="none", role=None, word=None)
            return None
        # CONTENT-based role (the pre-xedge behaviour, exact when no WM is held): the net_lean sign names the
        # over-subscribed role; within the lean margin it is a generic role-swap ("either").
        if net_lean < -self.lean_margin:
            content_role = "agent"                        # both nouns over-claim PATIENT -> AGENT slot unresolved
        elif net_lean > self.lean_margin:
            content_role = "patient"                      # both nouns over-claim AGENT -> PATIENT slot unresolved
        else:
            content_role = "either"                       # not confidently one-sided -> generic role-swap
        base.update(kind="role", role=content_role, word=None)
        # ONE-BRAIN XEDGE (closes the sub-decision caveat): this transitive's CONTENT did not resolve which referent
        # plays which role (that is why repair runs). If a WM referent is held, the cross-edge RESOLVES the held
        # referent's role via the balanced (content-cancelled) read -- the discourse-held referent is the tiebreaker
        # the content lacked -- and that becomes the repair role. LOAD-BEARING on the OUTPUT: varying the held
        # referent flips role agent<->patient (-> the clarification wording differs); lesioning the cross-edge
        # collapses the balanced margin to baseline (below eps) -> the content role stands. Byte-identical (content
        # role only) when shared=None / xedge off / no referent held / the WM signal is below eps.
        wm_role, wm_m = (self._wm_resolved_role(comp, wm_focus=wm_focus) if not lesion else (None, None))
        if wm_role is not None:
            base.update(role=wm_role, wm_resolved=True, wm_margin=float(wm_m), content_role=content_role)
        return base


_ORGAN: ComprehensionProductionOrgan | None = None


def get_organ(seed: int = 42) -> ComprehensionProductionOrgan:
    """The process-shared comprehension organ (built once on first use). When the ONE-BRAIN CROSS-EDGE flag is ON
    (`BRAIN_ONEBRAIN_XEDGE`) this returns the cross-edge-grown comprehension organ that co-inhabits the shared
    [d6_multiref_wm + comprehension + da_credit] xedge pool (the frozen w{k}->sel cross-edge lets a HELD WM pool
    drive its role competition); OFF (default) or on any build failure -> its own standalone bridge exactly as
    before (byte-identical). Mirrors the metacog pool-#2 shared-attach template."""
    global _ORGAN
    if _ORGAN is None:
        try:
            from research.runners.onebrain_xedge_production import xedge_enabled, get_xedge_pool
            if xedge_enabled():
                xp = get_xedge_pool(seed)
                if xp is not None and getattr(xp, "comp_organ", None) is not None:
                    _ORGAN = xp.comp_organ
        except Exception:
            _ORGAN = None
        if _ORGAN is None:
            _ORGAN = ComprehensionProductionOrgan(seed=seed)
    return _ORGAN


# The honest functional NOTICE surfaced on a LOW-margin (un-comprehended) transitive assertion.
def didnt_follow_message(svo=None) -> str:
    return ("My role-binding didn't resolve on that — I couldn't tell which word plays which role, "
            "so I didn't follow it. Could you rephrase?")
