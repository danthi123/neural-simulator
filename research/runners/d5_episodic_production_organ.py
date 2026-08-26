"""D5-EPISODIC — autobiographical episodic recall of PAST TURNS, wired for the PRODUCTION turn (Gate-B, 2026-08-12).

This is the production-integration glue that gives the brain a genuinely-SPIKING episodic RECALL GATE: on a
referential turn ("earlier you told me about X", "you mentioned a cat"), the brain decides whether topic X was
actually discussed THIS conversation by a hippocampal pattern-completion, NOT a host list scan. It REUSES (does not
reinvent) the standing episodic-dialogue mechanism, which itself reuses the 6/6-GO gap#5 dendritic-dAP readout:

  * the spiking episodic store/recall = `research.runners._episodic_dap_dialogue_memory.EpisodicDapMemory`
    (2026-08-10, kt=8 fix), which composes: emergent-DG membership (n_ca3=2000 sparse detonator) +
    `_build_dap_readout` (two-compartment apical dAP) + `_form_one_assembly` (BTSP one-shot store) +
    `_apical_up_read` (held-cell `cp_v_apical` UP-state completion) + `_held_cue_perm` (cue/held/perm geometry).
    Each spoken TOPIC BTSP-forms a CA3 assembly; a later referential cue COMPLETES it cue-specifically. The read
    is the fraction of held-out cells whose intrinsically-bistable apical latch reaches the UP state after the
    cue volley is driven and the bridge is stepped — a genuine dendritic-state read, NOT a host formula.
    6/6 seeds fire cue-specifically at kt=8 on BOTH numpy and cupy (perm=nocue=lesion(baseline)=0), so the LIVE
    production substrate (numpy in tests, cupy in prod) gets a real spiking recall with no cupy dependency.

WHAT IS SPIKING vs WHAT IS HOST (declared honestly — the honesty boundary is a deliverable, not a caveat):
  * SPIKING (load-bearing): the RECALL GATE — *which topics completed* + *is the referent in episodic memory* is
    decoded from which CA3 assemblies complete via the dendritic dAP read. Lesioning it (restore the UNFORMED
    baseline recurrent weights) collapses every completion to 0 -> the gate falls to "not in memory" -> the teeth.
  * HOST (declared residual scaffolds, ride existing burn-down items):
    - fact CONTENT ("what you told me about X") is still the per-conversation host oracle buffer — the gate is
      spiking, the retrieved sentence content is the next conversion.
    - temporal/recency ORDER (which topic was most recent) is a host store-order index; the WHEN attribute pool
      of the episodic-cue-recall design was scoped but NOT built, so there is NO spiking recency signal yet.
    - the gap#5 converse->sleep->clean-CA3-replay->converse CAPSTONE (a separate 6-seed GO) preserves the
      conversation through an offline replay, but its replay is a place-field TRAJECTORY decoded by a Bayesian
      instrument, not an autobiographical turn re-encode — it is offline consolidation, NOT a per-turn read, so
      it is deliberately NOT on this per-turn path.

THE HONESTY FLOOR is preserved BY CONSTRUCTION: this organ NEVER manufactures a fact and NEVER flips an abstain
into an assert. A topic whose assembly does NOT complete reads "I don't recall discussing X" (a GENUINE spiking
completion failure), never a confabulated recall. Because a spiking mis-fire (build-to-build emergent-size
non-determinism, or a small specificity margin) can at worst mark a discussed topic as not-recalled, a host-oracle
SELF-CONSISTENCY fallback (mirrored from the conversation eval) keeps the reply truthful either way: the gate can
suppress a recall but can never invent one. The moat is therefore only ever tightened, never loosened.

Scope / stateful note: unlike the stateless read organs (affect/metacog/surprise/curiosity, process singletons),
episodic memory ACCUMULATES across a conversation, so this organ is CONVERSATION-SCOPED (one EpisodicDapMemory per
cache_key), built lazily on the first `note_topic`. That is the honest structural difference from the read organs.

Backend: uses the process backend (numpy in tests, cupy in prod) — NO global-backend flip. Latency residual: a
BTSP store is ~seconds on cupy but ~6 min on numpy@2000; production is cupy, and a precompute->.npz cache of
(assembly geometry + formed weights) amortises the store. NO `sim/` edit; additive; default-ON with the
`BRAIN_EPISODIC` env escape (byte-identical host-only oracle) and the `BRAIN_EPISODIC_LESION` load-bearing flag.
"""
from __future__ import annotations

import os
import re

from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory, GRADED_READS

# The graded apical read surfaced as the conversation-visible recall STRENGTH. `depth_hold` = mean-held
# max(cp_v_apical − v_hold, 0) — the substrate's own BTSP instructive signal IS_post, and the 6-seed-validated
# read that rises reliably with learn-through-use where the quantised binary UP-fraction is flat (finding
# 2026-08-20-d5-graded-apical-read-makes-learn-through-use-reliably-conversation-visible).
SURFACED_GRADED_READ = "depth_hold"


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Enable / lesion flags — the exact contract the other Gate-B organs use.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def episodic_enabled() -> bool:
    """Default-ON. `BRAIN_EPISODIC` in {0,false,no,off} -> the byte-identical host-only oracle (organ never built)."""
    v = os.environ.get("BRAIN_EPISODIC")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def episodic_lesioned() -> bool:
    """`BRAIN_EPISODIC_LESION` in {1,true,yes,on} -> read through the UNFORMED baseline recurrent weights, so every
    completion collapses to 0 (the load-bearing teeth: the recall gate must fall to 'not in memory')."""
    v = os.environ.get("BRAIN_EPISODIC_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The conversation-scoped spiking episodic organ. Wraps ONE EpisodicDapMemory (reuse-by-import); tracks store ORDER
# only for the declared host recency residual. A read snapshot/restore-isolates inside EpisodicDapMemory.recall.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class EpisodicRecallOrgan:
    """Per-conversation spiking episodic-dialogue store. `note_topic` BTSP-forms the topic's CA3 assembly (episodic
    WRITE); `recall` drives the referential cue and reads the dendritic dAP completion (episodic READ); `discussed`
    decodes the set of topics whose assemblies complete. Built lazily on the first store (the bridge is expensive)."""

    def __init__(self, seed: int, topics, *, verbose: bool = False, sep_bias: float = 0.0):
        self.seed = int(seed)
        self.topics = sorted(set(str(t).lower() for t in topics))
        self.verbose = bool(verbose)
        self.sep_bias = float(sep_bias)     # D5 pattern-separation set-point (board #73); 0 -> byte-identical to HEAD
        self.mem: EpisodicDapMemory | None = None
        self._store_order: list[str] = []          # host store-order (the declared recency residual — NOT spiking)

    def _ensure_built(self):
        if self.mem is None:
            self.mem = EpisodicDapMemory(self.seed, self.topics, verbose=self.verbose, sep_bias=self.sep_bias)

    # ---- episodic WRITE: a spoken topic BTSP-forms its assembly (the plasticity rule's output is the weight) ----
    def note_topic(self, topic: str) -> bool:
        topic = str(topic).lower()
        if topic not in self.topics:
            return False
        self._ensure_built()
        wrote = bool(self.mem.store(topic))
        if wrote and topic not in self._store_order:
            self._store_order.append(topic)
        return wrote

    # ---- episodic READ: the spiking recall record for a referent (cue/perm/nocue completion + in_memory verdict) --
    def recall(self, topic: str, *, lesion: bool = False) -> dict:
        topic = str(topic).lower()
        if self.mem is None:
            # nothing stored this conversation yet -> genuinely not in memory (no assembly formed)
            return {"topic": topic, "slot": None, "formed": False, "in_memory": False,
                    "apical_cue": 0.0, "apical_perm": 0.0, "apical_nocue": 0.0,
                    "graded_cue": {r: 0.0 for r in GRADED_READS},
                    "graded_perm": {r: 0.0 for r in GRADED_READS},
                    "graded_nocue": {r: 0.0 for r in GRADED_READS}, "reason": "no-store-yet"}
        return self.mem.recall(topic, lesion=lesion)

    def discussed(self, *, lesion: bool = False):
        """(spiking) topics whose CA3 assembly COMPLETES via the dendritic dAP read, in store order; + the per-topic
        recall records. This is the brain's OWN spiking read of what it recalls discussing."""
        if self.mem is None:
            return [], {}
        return self.mem.discussed_topics(lesion=lesion)

    def recency_rank(self, topic: str):
        """DECLARED HOST RESIDUAL — store-order recency (there is no spiking WHEN signal yet). Returns 0 for the most
        recent stored topic, 1 for the next, ...; None if never stored."""
        topic = str(topic).lower()
        if topic not in self._store_order:
            return None
        return len(self._store_order) - 1 - self._store_order.index(topic)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Per-conversation registry (episodic memory is conversation-scoped, NOT a process singleton). Keyed by the same
# cache_key the server uses for _SESSION_MOOD / _SESSION_WORLDVIEW, cleared on reset.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_ORGANS: dict = {}

# The D5 pattern-separation set-point strength (pA) — winner-fatigue intrinsic-excitability bias applied per CA3 cell
# during emergent assembly formation so that memberships stay DISJOINT. 6/6 disjoint + healthy at 1000 (finding
# 2026-08-21-d5-pattern-separation-set-point-sepbias1000-closes-6of6-disjoint-knob1). Retained as the strength IF the
# separator is explicitly armed (pass sep_bias>0 / --sep-bias 1000); NOT armed by default — see _default_sep_bias.
D5_SEP_BIAS = 1000.0


def _default_sep_bias() -> float:
    """sep_bias for a NEW organ: 0.0 (the DG pattern-separation set-point is NOT armed by default -> UNMODIFIED emergent
    assemblies, byte-identical to HEAD formation). RATIONALE (finding 2026-08-21-d5-learn-through-use-flip-GO-per-topic-
    strength-surfacing-the-prior-NO-GO-was-a-surfacing-artifact): the separator was built to close a GRADED-read
    crosstalk, but the recall STRENGTH is now surfaced PER CONSOLIDATED TOPIC (recall_disclosure), so a neighbour's
    graded read is never surfaced and that crosstalk is moot; the separator's winner-fatigue only SHRINKS assemblies (a
    cost), and both sep_bias=0 and sep_bias=1000 soak 5/6 (s102 self-ignites either way — not shrink-caused). The
    mechanism is retained (pass sep_bias>0) but off by default."""
    return 0.0


def get_episodic_organ(cache_key, seed: int, topics, *, verbose: bool = False,
                       sep_bias: float | None = None) -> EpisodicRecallOrgan:
    """The conversation-scoped episodic organ for cache_key (built on first use; topics = the known agent vocabulary
    of this conversation). NOT a process singleton: each conversation accumulates its own stored turns.

    sep_bias=None (the default) reads `_default_sep_bias()` = 0.0 -> the DG pattern-separation set-point is NOT armed by
    default (unmodified emergent assemblies, byte-identical to HEAD formation); no-regression is instead delivered by the
    PER-CONSOLIDATED-TOPIC strength gate in recall_disclosure. Pass an explicit sep_bias>0 to arm the (retained)
    separator (the soak/de-risk pin it directly)."""
    org = _ORGANS.get(cache_key)
    if org is None:
        if sep_bias is None:
            sep_bias = _default_sep_bias()
        org = EpisodicRecallOrgan(seed, topics, verbose=verbose, sep_bias=sep_bias)
        _ORGANS[cache_key] = org
    return org


def reset_episodic_organ(cache_key) -> None:
    """Drop a conversation's episodic memory (called on the server's reset_conversation, alongside _SESSION_MOOD)."""
    _ORGANS.pop(cache_key, None)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Referential-turn detection + referent extraction (host — the environment/parse side, like extract_topic in the
# curiosity organ). The GATE decision that follows is spiking.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_REFERENTIAL_RE = re.compile(
    r"\b(you mentioned|you (told|said)|earlier you|a moment ago|"
    r"we (talk|talked|discuss|discussed)|did we (talk|discuss)|"
    r"what (did|was) (you|we|it|the))\b", re.IGNORECASE)


def is_referential(text: str) -> bool:
    """A turn that refers back to the conversation's own history ('you mentioned X', 'earlier you told me ...')."""
    return bool(_REFERENTIAL_RE.search(text or ""))


def extract_referent(text: str, topics) -> str | None:
    """The topic the referential turn is asking about: the first known topic token present in the message. Host
    (parse-side); the completion that decides whether it is in memory is spiking."""
    toks = set(re.findall(r"[a-zA-Z']+", (text or "").lower()))
    for t in sorted(set(str(x).lower() for x in topics)):
        if t in toks:
            return t
    return None


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The honest disclosure text. The GATE (in_memory) is spiking; `content` (the recalled fact sentence) is the host
# oracle. NEVER asserts content for a topic whose assembly did not complete.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _d5_strength_visible() -> bool:
    """FLAG half of the strength-surfacing gate (the TOPIC half is `_topic_consolidated`). The GRADED recall-strength is
    surfaced ONLY when the D5 learn-through-use consolidation is enabled (`BRAIN_D5_CONSOLIDATE`; default-ON since
    2026-08-21) AND the specific topic has actually been consolidated. This flag half is the byte-identical escape: with
    `BRAIN_D5_CONSOLIDATE=0` nothing consolidates and no strength is ever shown -> the recall reply is byte-identical to
    HEAD. The per-topic half is what keeps the DEFAULT-ON flip no-regression: consolidating one memory surfaces a
    strength for ONLY that memory, never a neighbour. Lazy import avoids any import-order coupling."""
    try:
        from webapp.continuous_engine import d5_consolidate_enabled
        return bool(d5_consolidate_enabled())
    except Exception:
        return False


def _topic_consolidated(cache_key, topic: str) -> bool:
    """True iff `topic` was actually D5-consolidated (learn-through-use) THIS conversation (a per-session set in
    continuous_engine). The recall strength is surfaced ONLY for such a topic, so consolidating one memory changes ONLY
    its own reply — a neighbour that was never consolidated keeps its byte-identical reply (the no-regression property).
    A None cache_key (a read with no session) never surfaces strength. Lazy import (no import-order coupling)."""
    if cache_key is None:
        return False
    try:
        from webapp.continuous_engine import topic_consolidated
        return bool(topic_consolidated(cache_key, topic))
    except Exception:
        return False


def _sleep_replayed_when(cache_key, topic: str):
    """FLAG+topic gate for the offline SLEEP-REPLAY surfacing (#64). Returns (when_rank, batch_size) iff
    `BRAIN_SLEEP_REPLAY` is armed AND `topic` was actually replayed in this session's LAST deep-idle sleep pass, else
    None. With the flag off nothing is ever sleep-replayed (the per-session set is empty), so this returns None for EVERY
    topic -> the recall reply is byte-identical to HEAD. Surfaced only for a replayed topic, so replaying one batch
    changes only the replayed topics' replies (the no-regression property). Lazy import (no import-order coupling)."""
    if cache_key is None:
        return None
    try:
        from webapp.continuous_engine import sleep_replay_enabled, topic_sleep_replayed
        if not sleep_replay_enabled():
            return None
        return topic_sleep_replayed(cache_key, topic)
    except Exception:
        return None


def recall_disclosure(record: dict, content: str | None = None, *, cache_key=None) -> str:
    """Compose the honest recall line from a spiking recall record. in_memory (a spiking completion) gates whether
    any content is surfaced at all — a completion failure is an honest abstain, never a confabulation.

    When the D5 learn-through-use consolidation is armed (`BRAIN_D5_CONSOLIDATE`) AND this specific topic has actually
    been consolidated this conversation (`_topic_consolidated`), the reply ALSO surfaces the GRADED apical magnitude
    (recall STRENGTH, mV above the latch hold) — the conversation-visible number that RISES as a used memory is
    consolidated, where the binary completion fraction saturates. It is surfaced BESIDE the binary gate (which already
    decided in_memory=True), so the moat is unchanged. The strength is gated PER TOPIC (not merely on the flag): a
    neighbour that was recalled but never consolidated keeps a reply byte-identical to HEAD, so consolidating one memory
    can only ever change ITS OWN reply — the no-regression property the default-ON flip needs. With consolidation OFF
    (the default) nothing is ever consolidated, so no topic surfaces a strength and every recall reply is byte-identical
    to HEAD."""
    topic = record.get("topic", "that")
    if record.get("in_memory"):
        cue = float(record.get("apical_cue", 0.0))
        if _d5_strength_visible() and _topic_consolidated(cache_key, topic):
            strength = float((record.get("graded_cue") or {}).get(SURFACED_GRADED_READ, 0.0))
            lead = (f"Earlier you brought up {topic} — my hippocampal readout completes its assembly for it "
                    f"(dendritic dAP completion {cue:.2f}, recall strength {strength:.1f} mV).")
        else:
            lead = (f"Earlier you brought up {topic} — my hippocampal readout completes its assembly for it "
                    f"(dendritic dAP completion {cue:.2f}).")
        # ADDITIVE (#64): if this topic was replayed in the session's last offline SLEEP-REPLAY pass, surface the BATCH
        # retention (its recall reads stronger — the risen graded apical magnitude) + the host store-order WHEN position.
        # Gated PER TOPIC + on BRAIN_SLEEP_REPLAY, so with the flag off (nothing replays) the reply is byte-identical to
        # HEAD. The WHEN-order is the DECLARED host store-order recency residual (EpisodicRecallOrgan.recency_rank), not a
        # spiking recency signal — surfaced honestly as such, never as a neural WHEN code.
        sr = _sleep_replayed_when(cache_key, topic)
        if sr is not None:
            when_rank, bsz = int(sr[0]), int(sr[1])
            strength = float((record.get("graded_cue") or {}).get(SURFACED_GRADED_READ, 0.0))
            ordinal = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th")[when_rank] \
                if 0 <= when_rank < 9 else f"{when_rank + 1}th"
            lead += (f" I also replayed it offline while idle — it was the {ordinal} of {bsz} recent memories I "
                     f"consolidated in store-order during sleep, and its recall reads stronger now "
                     f"(recall strength {strength:.1f} mV).")
        if content:
            return f"{lead} {content}"
        return lead
    return (f"I don't recall us discussing {topic} — no assembly completes for that cue "
            f"(a genuine spiking completion failure, so I won't make something up).")
