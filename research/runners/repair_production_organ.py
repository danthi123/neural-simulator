"""OTHER-REPAIR — a TARGETED clarification question on a low-comprehension turn, wired into the PRODUCTION turn
(faculty-map Tier-1 T1-6, 2026-08-12).

Genuine conversation is two-way. Today, on an utterance whose thematic roles the substrate could not resolve, the
D4 comprehension gate emits a bare abstain ("my role-binding didn't resolve — I didn't follow that"). A dead-end
abstain is not repair. This wires conversational OTHER-REPAIR: instead of the bare abstain the brain asks a
TARGETED CLARIFICATION that NAMES what did not resolve — the unresolved thematic ROLE ("...my role-binding didn't
resolve the AGENT — which of them is doing the 'push'?") or the out-of-vocabulary TOKEN ("...I don't know the word
'wug' yet — what does it refer to?"). This is a repair SEQUENCE, not just a flag.

It COMPOSES the already-wired D4 comprehension monitor (`comprehension_production_organ`): the SAME co-resident
`SpikingRoleCompetition` sel-pool reads whose |a0 - a1| margin triggers the abstain also localise the failure. The
organ's `repair_target()` reads the PER-NOUN agent-evidence (a0, a1) off `cp_firing_states` — sign(a0+a1) names the
over-subscribed role (so the OTHER role is unresolved), and the pair-max magnitude confirms the roles are ACTIVE at
all. The repair is TRIGGERED and SHAPED by that spiking read; this module only supplies the host language template.

BRAIN-BASED: the repair DECISION (that comprehension failed) is the D4 spiking abstain; the repair TARGET (which
role failed) is the D4 organ's per-noun spiking sel-pool read. The clarification WORDING here is a fixed host
language scaffold — a QUESTION frame, exactly like curiosity's wh-frame and the body acting on motor output. The
OOV-token branch (naming the specific unknown word) is a HOST-LEXICAL scaffold too, and NOT load-bearing on the
spiking read (a declared residual, like curiosity's host topic extractor) — only the ROLE branch is load-bearing.

MOAT-SAFE: a clarification is unambiguously a QUESTION. It NEVER asserts or confabulates a fact, never flips the
abstain into an answer, never enters the certainty band — the SAME safety class as curiosity's follow-up question
and the bare abstain it replaces. The turn stays an abstain (`abstained=True`); only the surface text changes from
a dead-end notice to a targeted question.

LESION-LOAD-BEARING: the ROLE target is caused by the spiking per-noun evidence, not a host role-parse. Under the
D4 lesion (`BRAIN_COMPREHENSION_LESION=1`, the learned cue->role synapses zeroed) both a_i collapse to ~0, the
commitment floor is not cleared, `repair_target()` returns None -> the turn degrades to the byte-identical bare
abstain. The same covered-ambiguous turn produces a targeted role clarification intact and the bare abstain
lesioned -> the repair is caused by the spiking comprehension signal.

Additive, default-ON, `BRAIN_REPAIR=0` -> the byte-identical bare abstain (fully skipped). NO `sim/` edit;
reuse-by-import of the D4 organ; process backend (cupy in production, numpy in tests).

FUNCTIONAL CORRELATE, NOT phenomenal: this reports a comprehension-repair CORRELATE (a targeted question shaped by
the role-competition read). It makes NO claim of subjective understanding.
"""
from __future__ import annotations

import os


def repair_enabled() -> bool:
    """Default-ON. `BRAIN_REPAIR` in {0,false,no,off} -> the byte-identical bare abstain (fully disabled)."""
    v = os.environ.get("BRAIN_REPAIR")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def clarification_question(target: dict | None) -> str | None:
    """The TARGETED other-repair clarification for a `repair_target()` result. A standalone QUESTION that REPLACES
    the bare abstain and NAMES the unresolved element. Returns None when nothing targetable (-> keep the bare
    abstain). The wording is a fixed host language scaffold; the TARGET (role / token) comes from the spiking /
    lexical read. Unambiguously a question — it never asserts a fact (moat-safe)."""
    if not target:
        return None
    svo = target.get("svo") or [None, None, None]
    n0, v, n1 = (svo + [None, None, None])[:3]
    kind = target.get("kind")

    if kind == "oov":
        toks = [t for t in (target.get("oov_tokens") or []) if t]
        if not toks:
            return None
        if len(toks) == 1:
            return (f"I followed the shape of that, but I don't know the word '{toks[0]}' yet — "
                    f"what does it refer to?")
        return (f"I followed the shape of that, but I don't know the words '{toks[0]}' or '{toks[1]}' yet — "
                f"what do they refer to?")

    if kind == "role" and n0 and v and n1:
        role = target.get("role")
        if role == "agent":
            return (f"I caught the verb '{v}' with the {n0} and the {n1}, but my role-binding didn't resolve the "
                    f"AGENT — which of them is doing the '{v}', the {n0} or the {n1}?")
        if role == "patient":
            return (f"I caught the verb '{v}' with the {n0} and the {n1}, but my role-binding didn't resolve the "
                    f"PATIENT — which of them is the '{v}' done to, the {n0} or the {n1}?")
        # generic role-swap (direction not confidently one-sided)
        return (f"I caught the verb '{v}' with the {n0} and the {n1}, but my role-binding didn't resolve who does "
                f"what — is the {n0} doing the '{v}' to the {n1}, or the other way round?")

    return None
