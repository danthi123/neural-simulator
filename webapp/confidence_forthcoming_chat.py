"""CONFIDENCE CAPS FORTHCOMINGNESS -- the brain's own spiking decision-margin confidence controls HOW MUCH it
volunteers on the LIVE rich (multi-sentence) `/api/brain-chat` turn, not just whether it appends a hedge phrase
(board #94, 2026-08-27).

WHAT THIS IS. The metacog organ (`metacog_production_organ`, Gate-B/E1, 6/6-seed-GO `nmda_norm` divisive-
normalized NMDA-conductance balance read) already qualifies an answer with an honest hedge when its confidence
reads LOW. This module extends the SAME confidence read to also govern the RichAnswerComposer's
FORTHCOMINGNESS (`max_sentences`/`max_elaborations`): a HIGH-confidence answer is allowed to chain in ONE extra
grounded fact beyond what the mood coupling (#81/#84's `content_plan`) already decided; a LOW-confidence (or
out-of-scope / unread) answer stays at exactly what the mood coupling decided -- no more, no less.

THE DESIGN (owner spec, #94): confidence CAPS forthcomingness, composing with the EXISTING mood coupling as a
FLOOR that is NEVER overridden -- the floor is always whatever `content_plan(level)` (or the composer's own
construction default when the mood organ is off) already set `max_sentences`/`max_elaborations` to for this
turn. Confidence can only ever ADD one elaboration on TOP of that floor (the instrument-confirmed 2-level
confidence signal -- `judge()["confident"]` is already a clean boolean, not a graded scale -- so there is exactly
ONE bonus level to grant, never an unbounded expansion). It NEVER subtracts below the floor: a LOW read (or no
read at all) simply withholds the bonus, it does not shrink what mood already committed to saying. This is the
SAFE direction (only ever elaborates on HIGH, per the owner's build note) -- a miscalibrated confidence read can
at worst make the brain slightly less generous than it could have been, never less generous than the floor the
OTHER faculty (mood) already promised.

THE BUILD-NOTE SUBTLETY (owner-flagged): the metacog confidence read is the answer's OWN mean role-decode
confidence (`mean_role_confidence` off the composer's post-answer trace) -- it is only available AFTER the rich
composer has already gathered + rendered a turn, not before. Two designs were open: (a) a PRE-COMPUTED
primary-recall confidence (probe the direct fact's role-decode confidence via a throwaway pre-flight recall
BEFORE calling `rich.answer()`), or (b) POST-HOC TRUNCATION (let the composer generate up to the FLOOR+1 "reach"
budget, read the SAME post-answer confidence the hedge already computes, and DROP the reach's extra fact when
not confident). This module takes (b), because:
  1. It reuses the EXACT SAME organ read already wired for the E1 hedge (`_metacog_qualify` in webapp/server.py)
     -- ONE spiking confidence read per turn, not two. A pre-flight probe would need its own call into
     `chat.gate`/`_direct_fact`, duplicating the (stochastic, GPU-real) direct-recall gate a second time before
     the composer's OWN `gather()` runs it again -- doubling per-turn latency for no additional evidence, and
     risking the two gate calls disagreeing (a session's discourse-thread state could tick between them).
  2. `_elaboration_facts_neural`/`_chain_facts`/`_dedup(...)[: self.max_sentences]` are deterministic given the
     SAME turn state, so asking for the reach budget (floor+1) and slicing back to the floor afterward yields
     EXACTLY the same head content a floor-only request would have produced -- post-hoc truncation is provably
     equivalent to "never having asked for the extra fact" for everything the user sees when the bonus is
     withheld, so it is not a shortcut relative to (a); it is the SAME answer either design would produce.
  3. It composes cleanly with the mood floor: the reach bump is applied AND undone around the SAME `rich.answer`
     call the mood coupling already wraps in webapp/server.py, so the floor is read directly off
     `rich.max_sentences`/`max_elaborations` at the moment this module's bump is applied -- exactly whatever
     mood (or the construction default) decided, with no separate bookkeeping of "what would the floor have
     been".

DISCOURSE-THREAD HONESTY (a residual the naive truncation would create). `RichAnswerComposer.answer()` marks
EVERY gathered-and-rendered fact (including the reach's withheld extra one) as already "said"
(`self._conversation_said` / `self._said`), so a naive truncation would silently make a later "tell me more"
follow-up skip content the user was NEVER actually shown. `undo_reach_bookkeeping` reverses this: the withheld
fact(s) are removed from both discourse-thread registers, so the brain can still genuinely bring them up on a
later follow-up.

MOAT-SAFE + ADDITIVE BY CONSTRUCTION. This module only ever DROPS a tail fact/sentence from an already-gathered,
already-VERIFIED set (or leaves it as generated) -- it never manufactures content, never changes WHICH fact is
the direct answer (floor_sentences is always >= 1, so `facts[0]` -- the direct recall -- is never touched), and
the honesty filter (`render_paragraph`'s per-sentence VERIFY / claim-level entailment) governs the kept
elaboration exactly as it already does today; truncation only removes verified content, it never adds unverified
content. Default-ON (`BRAIN_CONFIDENCE_FORTHCOMING`; a 2026-08-27 attempt to flip default-ON was reverted the
SAME day once real-traffic verification found it hollow, then RE-FLIPPED 2026-09-01 once the margin-scale
calibration residual that caused the hollowness was closed -- see the constant's own comment below); an
explicit `BRAIN_CONFIDENCE_FORTHCOMING=0` -> the whole block in webapp/server.py is skipped (no bump, no
truncation, no `confidence_forthcoming` key) -> byte-identical to pre-wiring.

LESION (reuses the metacog organ's OWN load-bearing lesion, `BRAIN_METACOG_LESION=1`; no separate lesion flag):
cutting the evidence differential collapses EVERY turn's margin to ~0 -> `confident` reads False unconditionally
-> the bonus is NEVER granted, regardless of the turn's true evidence -> the high-vs-low elaboration-count
DIFFERENCE this coupling produces intact COLLAPSES to zero under the SAME lesion the E1 hedge already uses --
the proof that this coupling rides the SPIKING confidence margin, not a host heuristic.

NO `sim/` edit; reuse-by-import of `metacog_production_organ` (via webapp/server.py's already-wired
`_metacog_qualify`) + `RichAnswerComposer.render_paragraph` (no new spiking machinery -- this module is pure
post-hoc bookkeeping around an existing spiking read).
"""
from __future__ import annotations

import os
from typing import Optional

# the instrument-confirmed 2-level confidence signal (`judge()["confident"]` is already boolean, not graded) ->
# exactly ONE bonus elaboration/sentence to grant on HIGH confidence, never an unbounded expansion.
EXTRA_ELABORATIONS = 1
EXTRA_SENTENCES = 1


# 2026-08-27: ATTEMPTED default-ON (production-integration flip, board #94), then HONESTLY REVERTED the SAME
# day once real-traffic verification found it hollow -- see the module docstring's "PRODUCTION-FLIP ATTEMPT"
# section and research/findings/2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-NOGO.md.
# Two structural bugs found + FIXED en route (kept, independent of this flag's default): (1) TieredFactStore had
# no __setattr__ (research/runners/tiered_fact_store.py), silently nulling `activity` on the +LTM production
# default; (2) RichAnswerComposer._chain_facts's own "probe one hop past a match" pattern clobbered
# OneBrainComposer.last_trace via its unconditional reset (research/runners/rich_answer_composer.py), so even
# with fix (1) `mean_role_confidence` still read None on every real turn. BOTH are now fixed and verified
# byte-identical to the pre-fix chain output. But measured directly against REAL (unpatched, unforced) traffic
# on the shipped tiny-demo brain, `mean_role_confidence` reads a SATURATED 1.0 on every tested topic (both a
# 3-fact and a 2-fact real question) -- the demo's clean, unambiguous SVO facts decode with maximal certainty,
# ABOVE the metacog organ's calibrated HIGH band (`role_conf_hi=0.52`), so `evidence_from_role_conf` clips to
# 1.0 and `confident` reads True unconditionally: no real LOW-confidence turn has been observed to compare
# against, so the coupling's cap/grant behavior has never been exercised on genuine production content -- a
# STILL-hollow flip by a different, more precisely characterized mechanism than before (calibration-band
# saturation on this vocabulary, not a missing signal). RESOLVED 2026-09-01 (below).
#
# 2026-09-01 FLIPPED DEFAULT-ON (owner 2026-09-01 auto-flip policy: validated-GO + load-bearing + moat-safe +
# byte-identical-off -> default-ON, no owner-gate). The saturation residual above is CLOSED by a margin-
# NORMALIZED (scale-invariant) confidence band (`research/runners/metacog_production_organ.py
# mean_role_confidence` now prefers a role chip's `margin_norm` -- the SAME peak-relative `(peak-runner)/peak`
# ratio `OneBrainComposer._margin` already used -- over the RAW, unnormalized `margin` field an LTM-sourced
# `RFPhasorComposer`/`ShardedPhasorStore` trace carries under the same key; see
# research/findings/2026-09-01-confidence-metacog-margin-norm-calibration-at-scale-discriminates-not-byte-id-off.md
# for the residual this closes and research/findings/2026-09-01-confidence-forthcomingness-default-on-flip-GO.md
# for this flip's own re-verification). DISCRIMINATES on the literal shipped `wikidata_core_15k` LTM through the
# real handler, 6/6 seeds (`vary_lesion_all_GO`): a clean KB-relation recall reads `confident=True`, a genuinely
# noise-degraded recall (Gaussian phase jitter on the SAME stored fact, still a correct match, not an abstain)
# reads `confident=False`, `BRAIN_METACOG_LESION=1` collapses both to `confident=False` --
# research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration_6seed.json.
# NOT gated behind this flag (a global calibration fix, not a flag-conditioned band): re-running the
# tiny-demo+LTM isolation harness FRESH after pulling the margin-norm change (the artifact this flag's
# `_CONFIDENCE_FORTHCOMING_DEFAULT_ON` docstring used to cite was STALE -- generated before the fix in this same
# development arc was finalized and never regenerated) shows `byte_identical_off=True` on ALL 6 seeds anyway
# (flags explicitly OFF reproduces the pre-recalibration no-LTM-tier answer verbatim, in n_sentences AND answer
# text) alongside `load_bearing`/`lesion_reverts`/`moat` all GO --
# research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json (freshly regenerated
# 2026-09-01, 1175s, all 4 checks PASS on all 6 seeds). `BRAIN_CONFIDENCE_FORTHCOMING=0` (or false/no/off/"")
# is the byte-identical escape to the pre-flip behavior; `BRAIN_ELABORATE_FROM_LTM_SHARD` (research/runners/
# rich_answer_composer.py `_ELABORATE_FROM_LTM_DEFAULT_ON`) is this coupling's own dependency (without it the
# reach never has LTM content to trim) and was flipped default-ON in the SAME commit.
_CONFIDENCE_FORTHCOMING_DEFAULT_ON = True


def confidence_forthcoming_enabled() -> bool:
    """Default-ON (2026-09-01; see the block above `_CONFIDENCE_FORTHCOMING_DEFAULT_ON` for why). `BRAIN_
    CONFIDENCE_FORTHCOMING` an explicit off (0/false/no/off/"") reverts byte-identically to the pre-flip
    (coupling fully skipped) behavior; unset or any other value keeps the coupling on."""
    v = os.environ.get("BRAIN_CONFIDENCE_FORTHCOMING")
    if _CONFIDENCE_FORTHCOMING_DEFAULT_ON:
        return not (v is not None and v.strip().lower() in ("0", "false", "no", "off", ""))
    return v is not None and v.strip().lower() in ("1", "true", "on", "yes")


def floor_override() -> Optional[tuple]:
    """Testing/tuning affordance (mirrors the #84 mood-INDUCE pattern): `BRAIN_CONFIDENCE_FORTHCOMING_FLOOR=
    "<max_sentences>,<max_elaborations>"` forces the FLOOR directly instead of reading it off whatever the mood
    coupling (or the composer's construction default) currently has set -- useful against a small demo KB whose
    natural content is already exhausted well below the production floor, so the reach bump would otherwise never
    have anything left to grant. Unset (the production path) -> None, meaning the floor is read live off
    `rich.max_sentences`/`max_elaborations` at the moment this module's reach bump is applied -- i.e. exactly
    what the mood coupling decided, per the 'never overriding the floor' design."""
    v = os.environ.get("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR")
    if not v:
        return None
    try:
        s, e = v.split(",")
        return (int(s.strip()), int(e.strip()))
    except Exception:
        return None


def reach_plan(floor_sentences: int, floor_elaborations: int) -> tuple:
    """The temporary REACH budget requested from the composer BEFORE `rich.answer()` runs: the floor plus the
    ONE possible bonus fact. The caller restores `max_sentences`/`max_elaborations` back to the floor in a
    `finally` immediately after the call -- the composer's cached instance must never leak the reach value into a
    later turn."""
    return (int(floor_sentences) + EXTRA_SENTENCES, int(floor_elaborations) + EXTRA_ELABORATIONS)


def undo_reach_bookkeeping(rich, dropped_facts: list) -> None:
    """Reverse `RichAnswerComposer.answer()`'s discourse-thread bookkeeping for facts this module is about to
    WITHHOLD from the surface (a truncation, not a fresh gather) -- so a later 'tell me more' can still honestly
    bring them up, instead of the brain silently believing it already told the user something it never said.
    `answer()` always appends the FULL (reach-sized) `kept` list to `self._said`/`self._conversation_said`
    (`self._said = list(kept)` on a fresh question, `.extend(kept)` on a follow-up) -- `dropped_facts` is exactly
    the tail of that same list, so removing the tail of `_said` and discarding each dropped tuple from
    `_conversation_said` exactly undoes it. Best-effort (never raises -- a residual over-eager 'already said'
    mark on a stray exception path is a minor forthcomingness residual, not a correctness break)."""
    if not dropped_facts:
        return
    try:
        n = len(dropped_facts)
        tail = [list(f) for f in dropped_facts]
        if len(rich._said) >= n and rich._said[-n:] == tail:
            rich._said = rich._said[:-n]
        for f in dropped_facts:
            rich._conversation_said.discard(tuple(f))
    except Exception:
        pass


def apply_cap(rich, r: dict, floor_sentences: int, confident) -> tuple:
    """Decide whether to keep the composer's REACH-sized output (`r`, already generated with `max_sentences` =
    floor+EXTRA_SENTENCES) or truncate it back to the FLOOR. `confident` is `metacog judge()['confident']` for
    THIS turn (True/False), or None when metacog is out of scope / disabled for this turn -- treated the SAME as
    False (only an affirmative HIGH read ever grants the bonus; the safe direction per the owner's build note).

    Returns (r_out, trace): `r_out` is `r` unchanged when the bonus is granted or there was nothing to trim
    (`len(r['facts']) <= floor_sentences`); a NEW dict (a shallow copy of `r` with `answer`/`facts`/`n_sentences`/
    `dropped` replaced by a re-rendered, re-VERIFIED truncation to the floor) when the bonus is withheld and
    there was something to cut. Re-rendering (rather than string-splitting the already-joined paragraph) keeps
    the honesty filter load-bearing on every kept sentence and is safe: the composer's gather is deterministic,
    so the FIRST `floor_sentences` facts a reach-sized gather produces are byte-identical to what a floor-sized
    gather alone would have produced (see the module docstring) -- truncating the tail and re-verifying only
    THOSE facts reproduces exactly the floor-only answer, never a different one. `trace` is the additive
    `confidence_forthcoming` response field (requested/kept counts, confident, granted, reason)."""
    facts = list(r.get("facts") or [])
    n_before = len(facts)
    trace = {"on": True, "floor_sentences": int(floor_sentences), "requested_sentences": n_before,
             "confident": (bool(confident) if confident is not None else None)}
    if bool(confident) and n_before <= (int(floor_sentences) + EXTRA_SENTENCES):
        # the confident read is granted the reach's bonus fact (if the reach actually produced one) -> keep r as-is.
        trace.update({"granted": bool(n_before > floor_sentences), "kept_sentences": n_before,
                      "elaborations_dropped": 0, "reason": "high_confidence"})
        return r, trace
    if n_before <= floor_sentences:
        # nothing beyond the floor was gathered anyway (content-exhausted, not confidence-limited) -> no-op.
        trace.update({"granted": False, "kept_sentences": n_before, "elaborations_dropped": 0,
                      "reason": "nothing_to_cap"})
        return r, trace
    # NOT confident (or unread) AND the reach genuinely produced extra content -> truncate back to the floor.
    new_kept = facts[:floor_sentences]
    dropped_extra = facts[floor_sentences:]
    paragraph, verified_kept, dropped_verify = rich.render_paragraph(new_kept)
    undo_reach_bookkeeping(rich, dropped_extra)
    r_out = dict(r)
    r_out["answer"] = paragraph
    r_out["facts"] = verified_kept
    r_out["n_sentences"] = len(verified_kept)
    r_out["dropped"] = list(r.get("dropped") or []) + list(dropped_verify)
    trace.update({"granted": False, "kept_sentences": len(verified_kept),
                 "elaborations_dropped": len(dropped_extra), "reason": "low_confidence_capped"})
    return r_out, trace
