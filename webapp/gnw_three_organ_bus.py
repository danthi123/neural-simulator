"""GNW THREE-GENUINELY-DISTINCT-ORGANS coincidence bus for the production `brain_chat` combine — DEFAULT-OFF.

WHAT THIS IS (and how it extends `gnw_two_organ_bus.py`). The production combine is authored by the GNW two-organ
ignition bus (`gnw_two_organ_bus.install_two_organ_gate`, DEFAULT-ON): a turn's recall verdict is committed by the
COINCIDENCE of organ A (spiking RECALL — composer `query_patient`, an FHRR phasor unbind) and organ B (a genuinely
DISTINCT spiking EXPECTATION-VIOLATION monitor — the `SurpriseProductionOrgan`, an Izhikevich predictive-coding
mismatch circuit reading `cp_firing_states[surprise]`). This module adds a THIRD genuinely-distinct organ behind a NEW
default-off flag, so the owner can review before any flip.

  organ A — spiking RECALL (composer):  query_patient(agent, action) -> cand                   [the FHRR read]
  organ B — spiking EXPECTATION-VIOLATION monitor (`SurpriseProductionOrgan`, NON-COMPOSER): confirms `cand` against
            its OWN learned expectation e_B[(agent,action)] by reading `cp_firing_states[surprise]` (a genuinely
            different Izhikevich substrate). CONFIRM (~0 Hz < threshold) -> vote; SURPRISE (> threshold) -> withhold.
  organ C — COMPREHENSION monitor of the RECALLED PROPOSITION `(agent, action, cand)`, reconstructed from the recall
            candidate (the question "what does {agent} {action}?" is a WH-gap, so organ C scores the full proposition
            the brain is about to commit). VETO AUTHORITY = a REAL-VOCAB entity/role COMPETENCE read (2026-08-21 fix,
            replaces the toy cue-lexicon margin that was the false-veto source): a RECALLED fact's thematic roles are
            already RESOLVED by its stored engram (the brain knows dog is the agent / cat the patient BECAUSE it
            stored that fact), so comprehension of a RECALL is "are all its entities/roles KNOWN in the brain's own
            learned vocabulary" (the engram-derived agents/actions/patients the recall composer itself binds), NOT
            "can bottom-up animacy/verbfit cues separate the roles" (the toy-competition question — right for a NOVEL
            incoming assertion, but false-vetoing a known two-animate fact like `dog chase cat`). KNOWN (agent, action
            + the patient head all in the brain's learned vocab) -> organ C CORROBORATES (votes slot(cand)); a content
            entity/role OUTSIDE the learned vocab (genuine non-comprehension) -> the spiking D4 `SpikingRoleCompetition`
            sel-pool WTA (`cp_firing_states` margin |agentEv(agent) - agentEv(cand)|) is consulted as its correct
            "does this UNKNOWN proposition's roles resolve?" instrument: margin >= threshold -> CORROBORATE, else
            organ C WITHHOLDS (a live comprehension VETO). So the spiking read stays load-bearing exactly where it is
            the right tool (an unknown proposition / an ungrounded recalled token), while a legitimately-recalled
            common fact reads HIGH-comprehension by real vocab and is never false-vetoed. HOST RESIDUAL: the
            membership test is host code, but the vocabulary it reads is the brain's OWN learned concept inventory.

THE CONSENSUS (a 3-way AND = a consensus-veto). Each organ writes a SUBTHRESHOLD drive `d_sub` into the shared K-slot
GNW workspace via `norgan_hop`. AGREEING organs ACCUMULATE their drive on slot(cand); `d_sub` is the calibrated
UNANIMITY drive `D_SUB_UNANIMITY[3]` so (N-1)*d_sub < the ignition knee <= N*d_sub (2*1000 does NOT ignite; 3*1000
does). So slot(cand) crosses the ignition knee ONLY when ALL THREE organs vote — the brain commits a recall only when
it RECALLS it (organ A), is NOT surprised by it (organ B), AND actually COMPREHENDED the proposition (organ C). Any one
organ withholding leaves slot(cand) at <= 2*d_sub -> subthreshold -> the workspace ABSTAINS. The AND-over-three-distinct
-organs is the neuronal ignition THRESHOLD (WTA + NMDA sustain), not host control flow. This is the load-bearing new
capability over the 2-organ bus: a NON-COMPREHENDED recall (recall ∧ ¬surprise would commit, but a content entity/role
lies OUTSIDE the brain's learned vocabulary) is VETOED by organ C — a decision the 2-organ bus could not make.

DISCIPLINE (the #1 requirement): ADDITIVE + DEFAULT-OFF + BYTE-IDENTICAL-WHEN-OFF. With `BRAIN_GNW_3ORGAN` unset/off
this module installs NOTHING (the server hook does not even import it), so the turn is byte-identical to today's
production (the DEFAULT-ON 2-organ bus authors it). Only when the flag is ON does the wrapper wrap `chat.gate`.

LESION-LOAD-BEARING (the honest-negative deliverable):
  * `BRAIN_GNW_3ORGAN_ORGANC_LESION=1` — organ C's comprehension VETO is silenced: organ C corroborates
    UNCONDITIONALLY (both its real-vocab read AND its `cp_firing_states` read are bypassed). With organ C always
    voting, the Q=3 consensus reduces to organ A + organ B (the guaranteed organ-C vote + organ A means the deciding
    third vote is organ B's confirmation) — i.e. EXACTLY the 2-organ decision. So a non-comprehension abstain REVERTS
    to the 2-organ commit, attributing the veto to organ C's active participation (a host `if known` outside the organ
    would NOT revert when the organ is lesioned). For the spiking sub-read that decides UNKNOWN propositions, the D4
    SYNAPTIC lesion (`BRAIN_COMPREHENSION_LESION=1`) additionally zeroes organ C's learned cue->role weights and
    collapses the well-vs-ill margin DISCRIMINATION (both read margin ~0), attributing that sub-read's SELECTIVITY to
    the learned spiking competition.
  * The 2-organ bus's own levers still apply on the covered path: `BRAIN_GNW_2ORGAN_WS_LESION=1` (workspace self-
    recurrence zeroed -> even a full consensus cannot sustain -> abstain, while the forward-recall reflex survives)
    and `BRAIN_GNW_2ORGAN_ORGANB_LESION=1` (organ B withholds even on a match -> the consensus collapses).

DE-RISK lever, DEFAULT-OFF, reused from `gnw_two_organ_bus.py` AND NOW EXTENDED HERE to organ C (2026-08-27,
`BRAIN_GNW_ORGANB_LTM_EXEMPT` — deliberately the SAME flag, no second flag was added, so ONE owner switch fixes
the WHOLE stack). Organ B's expectation registry never covers the LTM tier (see that module's LTM-EXEMPTION
docstring), so it withholds on EVERY LTM-sourced recall here too, exactly as in the 2-organ bus. Organ C carried
the MIRROR-IMAGE gap: `_real_vocab_competence` reads `brain_vocab` from the SAME buffer-only `_chat_concepts`, so
an LTM-only entity (e.g. `chelsea_fc`/`country`/`united_kingom`) is "unknown" to organ C too, and its OOV fallback
then consults the D4 comprehension organ's spiking margin — an instrument calibrated on a toy cue lexicon that has
never seen these tokens, so the margin reads exactly 0.0 on every LTM probe, below any real threshold. **THE
FIX**: the SAME argument organ B's exemption already rests on applies verbatim to organ C — a STORED LTM fact IS
comprehended, because its thematic roles are already RESOLVED by its stored engram (the brain knows `chelsea_fc`
is the agent / `country` the relation / `united_kingom` the patient BECAUSE it stored that triple); judging
whether an ALREADY-RECALLED proposition is comprehended is not "can a buffer-calibrated vocabulary-membership
cue see these tokens", so organ C should not consult that instrument for an LTM-sourced recall at all. With the
flag ON, `three_organ_combine` reuses the SAME `ltm_exempt_applied` boolean it computes for organ B (recall
sourced from the stable LTM tier, per `TieredFactStore.query_patient_source`) to also make organ C CORROBORATE
unconditionally for that recall — bypassing both `_real_vocab_competence` and the D4 spiking-margin fallback.
A conversational-buffer recall is completely untouched (organ C reads its real-vocab / spiking-margin instrument
exactly as today); a non-existent fact still abstains at organ A, before organ B or organ C is even consulted.
DEFAULT-ON since 2026-08-27 (flipped with organ B under the ONE shared flag `BRAIN_GNW_ORGANB_LTM_EXEMPT`, per
owner approval, after both de-risks landed 6/6 GO): `ltm_exempt_applied` is True for an LTM-sourced recall, so
organ C corroborates it instead of vetoing on its buffer-only vocab — genuine knowledge-base facts commit through
the whole 3-organ consensus. `BRAIN_GNW_ORGANB_LTM_EXEMPT=0` is the byte-identical escape (reverts organ C's read
to today's production). See `research/findings/2026-08-27-organc-ltm-exempt-derisk-6seed-GO.md` for the verification.

REUSE-BY-IMPORT (NO `sim/` edit, NO re-derivation): the workspace build + ignition-read + organ B (the surprise organ)
+ the calibrated subthreshold drive come from `gnw_two_organ_bus` (which reuses the de-risk parents); the N-organ
consensus hop is `_gnw_norgan_bus_derisk.norgan_hop` (which already supports 3 organs); organ C is the production
`comprehension_production_organ.ComprehensionProductionOrgan` (the 6/6-GO D4 de-risk). This module adds only the glue:
the third organ's read off the live brain + the 3-organ consensus wiring + the gate wrapper.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# reuse-by-import the DEFAULT-ON two-organ bus machinery (organ A read is inline; organ B + the shared warm workspace
# bridge + concept extraction + the backend-discrimination gate all come from the 2-organ module) — NO sim/ edit.
from webapp.gnw_two_organ_bus import (
    _get_organ, _get_bridge, _chat_concepts, _organ_discriminates, _organ_a_recall,
    ws_lesion_on, organb_lesion_on, organb_ltm_exempt_enabled,
)
# organ B's spiking corroboration read (cp_firing_states[surprise]) and the N-organ consensus hop + unanimity drive.
from research.runners._gnw_two_distinct_organs_derisk import organ_b_confirms
from research.runners._gnw_norgan_bus_derisk import norgan_hop, D_SUB_UNANIMITY
from research.runners._gnw_coincidence_integrator_derisk import _pick_decoy
# organ C — the production spiking comprehension monitor (the 6/6-GO D4 faculty; process-shared, built once).
from research.runners.comprehension_production_organ import get_organ as _get_comprehension_organ

_N_ORGANS = 3
_D_SUB_3 = float(D_SUB_UNANIMITY[3])   # the calibrated Q=3 UNANIMITY drive: 2*d_sub < knee <= 3*d_sub


# ── flags (all DEFAULT-OFF) ──────────────────────────────────────────────────────────────────────────────────────
def three_organ_enabled() -> bool:
    """The master switch. `BRAIN_GNW_3ORGAN` in {1,true,on,yes} (THE DEFAULT) enables the THREE-distinct-organs
    consensus combine. DEFAULT-ON since 2026-08-21 (the real-vocab over-veto fix met the flip gate; finding
    2026-08-21-gnw-three-organ-realvocab-flip-GO, ledger on_by_default: YES). `BRAIN_GNW_3ORGAN=0` is the
    byte-identical escape -> installs nothing, the turn delegates to the DEFAULT-ON 2-organ bus.
    NOTE (fixed 2026-08-27): this inner gate was mistakenly left DEFAULT-OFF (`""`) while server.py's OUTER gate +
    the PI-ledger both read default-ON — a HOLLOW two-gate flip: the outer block ran but this inner veto silently
    installed nothing, so the ledger's on_by_default:YES was false in production. Aligned to `"1"` here."""
    return os.environ.get("BRAIN_GNW_3ORGAN", "1").strip().lower() in ("1", "true", "on", "yes")


def organc_lesion_on() -> bool:
    """`BRAIN_GNW_3ORGAN_ORGANC_LESION` truthy -> organ C's comprehension VETO is silenced (it corroborates
    unconditionally), so the Q=3 consensus reduces to the 2-organ (recall ∧ ¬surprise) decision. Load-bearing lever:
    a low-comprehension abstain REVERTS to the 2-organ commit, attributing the veto to organ C's spiking drive."""
    return os.environ.get("BRAIN_GNW_3ORGAN_ORGANC_LESION", "").strip().lower() in ("1", "true", "on", "yes")


# ── organ C: the comprehension read on the RECALLED PROPOSITION (agent, action, cand) ────────────────────────────
def _real_vocab_competence(agent: str, action: str, cand: str, brain_vocab) -> tuple:
    """REAL-VOCAB entity/role competence over the RECALLED PROPOSITION (agent, action, cand): every content
    entity/role is KNOWN when it is in the brain's OWN learned vocabulary — the engram-derived agents/actions/patients
    the recall composer itself binds (`_chat_concepts`), NOT a hand-authored toy lexicon. A multi-word attributed
    patient ('big apple') is reduced to its HEAD entity ('apple') so a learned attribute modifies a known entity and
    never false-vetoes. Returns (known: bool, unknown: list[str]). This is the 2026-08-21 FIX: it replaces the toy
    animacy/verbfit cue-competition margin as the organ-C VETO AUTHORITY (the margin false-vetoed known two-animate /
    verbfit-conflict recalls). HOST RESIDUAL: the membership test is host code, but the vocabulary it reads is the
    brain's own learned concept inventory (the same vocab the recall uses)."""
    vocab = brain_vocab or set()
    cand_str = str(cand).strip()
    toks = cand_str.split()
    cand_head = toks[-1] if toks else cand_str            # entity head; leading tokens are learned attributes
    checks = [str(agent), str(action), cand_head]
    unknown = [t for t in checks if t and t not in vocab]
    return (len(unknown) == 0), unknown


def _comprehension_vote(agent: str, action: str, cand: str, brain_vocab, *, seed: int, lesion: bool,
                        ltm_exempt: bool = False) -> dict:
    """Organ C's vote on committing `cand`. VETO AUTHORITY = a REAL-VOCAB competence read over the recalled
    proposition (agent, action, cand): a RECALLED fact's roles are already RESOLVED by its stored engram, so
    comprehension of a recall is "are all its entities/roles KNOWN in the brain's learned vocab", NOT "can bottom-up
    cues separate the roles" (the toy-competition question that false-vetoed known two-animate recalls). Decision:
      * lesion=True (organc_lesion) -> the whole read is BYPASSED, organ C corroborates unconditionally (veto silenced
        -> the consensus reduces to the 2-organ decision).
      * ltm_exempt=True (the `BRAIN_GNW_ORGANB_LTM_EXEMPT` lever, ALREADY confirmed by the caller that this recall's
        provenance is the stable LTM tier) -> the SAME argument the caller applies to organ B applies here: a stored
        LTM fact's roles are already resolved by its own engram, so treat it as COMPREHENDED without consulting the
        buffer-calibrated real-vocab / D4 spiking-margin instrument at all -> corroborate unconditionally.
      * KNOWN (agent, action + the patient head all in the brain's learned vocab) -> COMPREHENDED -> corroborate.
        [THE FIX: `dog chase cat`, `cat eat fish` are all real-vocab-known -> vote, never false-vetoed.]
      * NOT known (a content entity/role OUTSIDE the learned vocab) -> genuine non-comprehension -> consult the spiking
        D4 `SpikingRoleCompetition` sel-pool WTA (`cp_firing_states` margin, its correct "does this UNKNOWN
        proposition's roles resolve?" instrument): competent AND margin >= threshold -> corroborate; else WITHHOLD
        (the comprehension VETO). Keeps the spiking read load-bearing exactly where it is the right tool."""
    info = {"organ_c_competent": None, "organ_c_margin": None, "organ_c_threshold": None,
            "organ_c_comprehended": None, "organ_c_deferred": False, "organ_c_lesioned": bool(lesion),
            "organ_c_real_vocab_known": None, "organ_c_unknown_tokens": None,
            "organ_c_ltm_exempt_applied": False}
    if lesion:                                        # the veto is silenced -> unconditional corroboration
        info["votes"] = True
        return info
    if ltm_exempt:
        # LTM-EXEMPTION (mirrors organ B's — see the module docstring's DE-RISK section): a STORED LTM fact IS
        # comprehended, its roles resolved by its own stored engram, so corroborate WITHOUT touching the buffer-
        # calibrated real-vocab read or the D4 spiking-margin fallback (neither instrument is consulted here).
        info["organ_c_comprehended"] = True
        info["organ_c_ltm_exempt_applied"] = True
        info["votes"] = True
        return info
    # ── THE FIX: real-vocab competence is the veto authority (replaces the toy cue-lexicon margin). ──────────────
    known, unknown = _real_vocab_competence(agent, action, cand, brain_vocab)
    info["organ_c_real_vocab_known"] = bool(known)
    info["organ_c_unknown_tokens"] = list(unknown)
    if known:
        info["organ_c_comprehended"] = True
        info["votes"] = True
        return info
    # ── NOT real-vocab-known: the proposition carries an entity/role OUTSIDE the brain's learned vocab -> the spiking
    #    D4 monitor's correct domain. Read the cp_firing_states sel-pool margin: CORROBORATE only if the substrate
    #    competently comprehends the (now genuinely unknown) proposition, else VETO (genuine non-comprehension). ────
    corg = _get_comprehension_organ(seed)
    corg.ensure_built()
    info["organ_c_threshold"] = float(corg.threshold)
    competent = bool(corg.competent(str(agent), str(action), str(cand), brain_vocab=brain_vocab))
    info["organ_c_competent"] = competent
    if not competent:                                 # cannot reliably judge an unknown proposition -> genuine veto
        info["organ_c_comprehended"] = False
        info["votes"] = False
        return info
    margin = float(corg.read_margin(str(agent), str(action), str(cand)))   # cp_firing_states sel-pool WTA margin
    comprehended = bool(margin >= corg.threshold)
    info["organ_c_margin"] = margin
    info["organ_c_comprehended"] = comprehended
    info["votes"] = comprehended
    return info


# ── one EVALUATE/COMMIT over the workspace: organ A + (confirmed) organ B + (comprehended) organ C ───────────────
def three_organ_combine(chat, agent: str, action: str, *, seed: int = 42,
                        ws_lesion: bool = False, organb_lesion: bool = False,
                        organc_lesion: bool = False, organb_ltm_exempt: bool = False) -> dict:
    """Route (agent, action) through the THREE-DISTINCT-ORGANS spiking consensus and return the SUBSTRATE's committed
    decision. `committed` is the ignited patient (or None = abstain). NO host `if/else` selects it — organ A (FHRR
    recall), organ B (the surprise monitor's corroboration), and organ C (the comprehension monitor's corroboration)
    each write a subthreshold `d_sub`; only their 3-way COINCIDENCE crosses the ignition knee. Read-only; never
    mutates the answer. (The forward `query_patient` leaves the composer's `last_trace` = the host forward-recall
    trace; organ B / organ C / the workspace use SEPARATE bridges, so the surfaced 'brain activity' stays identical.)

    `organb_ltm_exempt=True` (the `BRAIN_GNW_ORGANB_LTM_EXEMPT` de-risk lever, default off, defined in
    `gnw_two_organ_bus.py`) reuses the SAME organ-B LTM-exemption applied there: a recall sourced from the stable
    cortical LTM tier makes organ B corroborate instead of withhold. AS OF THIS SESSION the identical lever ALSO
    exempts organ C's comprehension veto for the SAME LTM-sourced recall (the mirror-image fix — see this module's
    DE-RISK docstring section and `_comprehension_vote`'s `ltm_exempt` branch): one flag now unblocks BOTH organs
    for genuine long-term-memory knowledge, while a conversational-buffer recall or a non-existent fact is
    unaffected either way."""
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    _a, _v, all_concepts, e_b, stored_patients = _chat_concepts(chat)
    brain_vocab = set(all_concepts)
    info = {"organ_a_recall": None, "organ_b_confirmed": None, "organ_b_surprise_hz": None,
            "organ_c_votes": None, "committed": None, "ignited": False, "n_ignited": 0,
            "d_sub": _D_SUB_3, "n_organs": _N_ORGANS,
            "ws_lesion": bool(ws_lesion), "organb_lesion": bool(organb_lesion), "organc_lesion": bool(organc_lesion),
            "organb_ltm_exempt": bool(organb_ltm_exempt),
            "expected": e_b.get((agent, action))}
    if composer is None:
        info["abstain_reason"] = "no_composer"
        return info

    # organ A — the FHRR recall (the primary organ; a miss abstains by the moat, exactly like the host / 2-organ bus).
    # With the LTM-exemption lever on, also learns WHICH tier answered (buffer vs LTM) so organ B can be exempted
    # below — off (default), this is the identical single `query_patient` call, unaffected.
    cand, recall_source = _organ_a_recall(composer, agent, action, ltm_exempt=bool(organb_ltm_exempt))
    info["organ_a_recall"] = cand
    info["recall_source"] = recall_source
    if cand is None:
        info["abstain_reason"] = "primary_recall_miss"
        return info

    # organ B — the spiking surprise monitor's corroboration of `cand` against its OWN expectation e_B[(agent,action)].
    organ = _get_organ(seed, stored_patients)
    exp = e_b.get((agent, action))
    ltm_exempt_applied = bool(organb_ltm_exempt and recall_source == "ltm")
    if ltm_exempt_applied:
        confirmed_b, hz = True, None            # LTM-EXEMPTION — see gnw_two_organ_bus.two_organ_combine
    else:
        confirmed_b, hz = organ_b_confirms(organ, exp, cand, lesion=bool(organb_lesion))
    info["organ_b_confirmed"] = bool(confirmed_b)
    info["organ_b_surprise_hz"] = (None if (hz is None or np.isnan(hz)) else float(hz))
    info["organ_b_threshold_hz"] = float(getattr(organ, "threshold", float("nan")))
    info["organb_ltm_exempt_applied"] = ltm_exempt_applied

    # organ C — the spiking comprehension monitor's corroboration of the recalled proposition (agent, action, cand).
    # Reuses the SAME `ltm_exempt_applied` boolean organ B was just gated by: when the recall is LTM-sourced AND
    # the flag is on, organ C corroborates unconditionally too (the mirror-image fix — see the module docstring's
    # DE-RISK section). Off (default) or a buffer-sourced recall: `ltm_exempt_applied` is False, so organ C's
    # real-vocab / D4 spiking-margin read runs exactly as before — byte-identical.
    c = _comprehension_vote(agent, action, cand, brain_vocab, seed=seed, lesion=bool(organc_lesion),
                            ltm_exempt=ltm_exempt_applied)
    votes_c = bool(c["votes"])
    for k in ("organ_c_competent", "organ_c_margin", "organ_c_threshold", "organ_c_comprehended",
              "organ_c_deferred", "organ_c_lesioned", "organ_c_real_vocab_known", "organ_c_unknown_tokens",
              "organ_c_ltm_exempt_applied"):
        info[k] = c[k]
    info["organ_c_votes"] = votes_c

    # the shared GNW workspace: organ A drives slot(cand); organ B drives slot(cand) IFF it CONFIRMED; organ C drives
    # slot(cand) IFF it CORROBORATED; a single-vote decoy exercises WTA. Only a >=3-vote slot (all three organs agree)
    # crosses the ignition knee -> a genuine 3-way consensus-ignition commit; anything less IS the abstain (the moat +
    # the comprehension veto). `norgan_hop` already accumulates agreeing organs and WTA-selects via shared inhibition.
    bridge, xp, slots_dev, snap = _get_bridge(seed, bool(ws_lesion))
    decoy = _pick_decoy(list(all_concepts), exclude={cand, exp, agent, action},
                        rng=np.random.default_rng(int(seed) * 991 + 7))
    candidates = [cand, (cand if confirmed_b else None), (cand if votes_c else None)]
    committed, _rates, _winner, n_ign = norgan_hop(
        bridge, xp, slots_dev, snap, candidates, decoy, _D_SUB_3,
        rng=np.random.default_rng(int(seed) * 13 + 3))
    info["committed"] = committed
    info["ignited"] = committed is not None
    info["n_ignited"] = int(n_ign)
    info["decoy"] = decoy
    info["n_votes"] = int(1 + int(bool(confirmed_b)) + int(votes_c))
    if committed is None:
        if not confirmed_b:
            info["abstain_reason"] = "consensus_veto_organ_b_withheld"
        elif not votes_c:
            info["abstain_reason"] = "consensus_veto_organ_c_non_comprehension"
        else:
            info["abstain_reason"] = "no_ignition"
    return info


def three_organ_gate_via(chat, question: str, *, seed: int = 42,
                         ws_lesion: bool = False, organb_lesion: bool = False,
                         organc_lesion: bool = False, organb_ltm_exempt: bool = False):
    """AUTHOR the gate combination with the THREE-DISTINCT-ORGANS spiking consensus for the COVERED routable class,
    mirroring `gnw_two_organ_bus.two_organ_gate_via`'s routing EXACTLY (so extraction + acquisition/open-ended/anaphora
    side effects run ONCE, unchanged) but swapping the covered-class combine from the 2-organ coincidence to the
    3-organ consensus. Returns (svo_or_None, info). Read-only; never mutates the composer/answer. Out-of-scope classes
    (open-ended, acquisition, comprehension-decline, self/identity/short) stay HOST-authored, unchanged."""
    mode_tuple = chat.gate_extract(question)
    mode = mode_tuple[0]

    if mode == 'done':                                        # out-of-scope: open-ended hypothesis / acquisition (host)
        svo = mode_tuple[1]
        _l = (list(svo) if svo is not None else None)
        reason = ("open_ended_generation" if type(svo).__name__ == "HypothesisSVO" else "acquisition_or_abstain")
        return svo, {"routable": False, "reason": reason, "authored_by": "host_out_of_scope",
                     "host_combination_computed": False, "bus_svo": _l}

    if mode == 'route':                                       # the COVERED class -> the SUBSTRATE authors the verdict
        _, q, agent, action, anaphora_used = mode_tuple
        agents_set = getattr(chat, "agents_set", set()) or set()
        info = three_organ_combine(chat, agent, action, seed=seed, ws_lesion=ws_lesion,
                                   organb_lesion=organb_lesion, organc_lesion=organc_lesion,
                                   organb_ltm_exempt=organb_ltm_exempt)
        committed = info.get("committed")
        if committed is not None:
            # preserve gate()'s ONLY covered-class side effect: a concrete patient that is itself an agent becomes the
            # next-turn pronoun referent (the anaphora WM write), so the retirement is behaviour-identical.
            if isinstance(committed, str) and committed in agents_set:
                try:
                    chat._note_referent(committed)
                except Exception:
                    pass
            bus_svo = [agent, action, committed]
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": anaphora_used,
                         "authored_by": "three_organ_bus", "host_combination_computed": False,
                         "bus_svo": list(bus_svo)})
            return bus_svo, info
        # the SUBSTRATE VETOES (no 3-way consensus = the moat / the comprehension veto). Match gate(): abstain unless
        # anaphora, then the host router.
        if not anaphora_used:
            info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": False,
                         "authored_by": "three_organ_bus", "host_combination_computed": False, "bus_svo": None})
            return None, info
        svo = chat._gate_router_combine(q)                    # anaphora-abstain fall-through -> host router (out of scope)
        info.update({"routable": True, "agent": agent, "action": action, "anaphora_used": True,
                     "authored_by": "three_organ_bus_veto_then_host_router", "host_combination_computed": True,
                     "bus_svo": (list(svo) if svo is not None else None)})
        return svo, info

    if mode == 'decline':                                     # parser declined a factual-shaped Q (comprehension abstain)
        _, q, anaphora_used = mode_tuple
        if not anaphora_used:
            return None, {"routable": False, "reason": "parser_decline_abstain", "authored_by": "host_abstain",
                          "host_combination_computed": False, "bus_svo": None}
        svo = chat._gate_router_combine(q)
        _l = (list(svo) if svo is not None else None)
        return svo, {"routable": False, "reason": "parser_decline_anaphora_router", "authored_by": "host_router",
                     "host_combination_computed": True, "bus_svo": _l}

    # mode == 'router': unroutable (self/identity/short) -> the HOST router authors (out of scope, kept)
    q = mode_tuple[1]
    svo = chat._gate_router_combine(q)
    _l = (list(svo) if svo is not None else None)
    return svo, {"routable": False, "reason": "unroutable_self_or_short", "authored_by": "host_router",
                 "host_combination_computed": True, "bus_svo": _l}


def install_three_organ_gate(chat, *, seed: int = 42) -> bool:
    """Idempotently wrap `chat.gate` so the THREE-DISTINCT-ORGANS spiking consensus AUTHORS the organ-combination on
    the covered routable class. Installs ONLY when `BRAIN_GNW_3ORGAN` is truthy AND the organs discriminate on this
    backend (`_organ_discriminates()`: numpy always; cupy iff the surprise organ is built backend-neutral); otherwise
    installs NOTHING and returns False -> the turn is byte-identical to today's production (the DEFAULT-ON 2-organ bus
    authors it). Preserves the wrapped gate as `chat._three_organ_orig_gate` (the 2-organ wrapper when default-on);
    stashes this turn's info on `chat._last_three_organ`. On any per-turn bus exception the wrapper falls back to the
    wrapped gate (a turn never crashes). No `sim/` edit."""
    if not three_organ_enabled():
        return False
    if not _organ_discriminates():
        return False
    if getattr(chat, "_three_organ_installed", False):
        return False

    orig_gate = chat.gate                                     # the DEFAULT-ON 2-organ wrapper (or the host gate)
    chat._three_organ_orig_gate = orig_gate

    def _three_organ_gate(question):
        # RUNTIME re-check: if the flag is toggled off (or the organs would no longer discriminate on this backend)
        # after install, delegate to the WRAPPED gate so the turn is byte-identical to today's production — the
        # wrapper never changes behaviour unless BRAIN_GNW_3ORGAN is actively on. (Belt-and-suspenders with the
        # flag-guarded server hook.)
        if not (three_organ_enabled() and _organ_discriminates()):
            return orig_gate(question)
        try:
            svo, info = three_organ_gate_via(chat, question, seed=seed,
                                             ws_lesion=ws_lesion_on(), organb_lesion=organb_lesion_on(),
                                             organc_lesion=organc_lesion_on(),
                                             organb_ltm_exempt=organb_ltm_exempt_enabled())
        except Exception as e:                                # never let the bus crash a turn -> wrapped gate authors
            host_svo = orig_gate(question)
            chat._last_three_organ = {"routable": False, "reason": f"error:{type(e).__name__}: {e}",
                                      "authored_by": "orig_gate_fallback_on_error",
                                      "host_combination_computed": True,
                                      "bus_svo": (list(host_svo) if host_svo is not None else None)}
            return host_svo
        chat._last_three_organ = info
        return svo

    chat.gate = _three_organ_gate
    chat._three_organ_installed = True
    return True
