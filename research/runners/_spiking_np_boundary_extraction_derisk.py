"""LANE 4 DE-RISK follow-on to the 2026-08-20 spiking-extraction finding
(research/findings/2026-08-20-open-text-spiking-extraction-derisk-reaches-canonical-svo-not-free-np.md):
that de-risk showed BridgeParser (research/runners/brain_conversational_agent.py:28) reaches
artificially-canonical 3-word SVO but FAILS on multi-word noun phrases, copulas, and passives --
the single largest coverage hole for free Qwen prose. Its named "smallest next lever" was:
generalize AttributedBridgeParser's adjective+noun population-binding (research/runners/
attributed_parser.py) so it collapses an ARBITRARY-length determiner-headed NP span into ONE
role-fillable unit before the existing position x voice spiking read-out.

THE GENERALIZATION. `AttributedBridgeParser` (attributed_parser.py:50) binds "S V [adj]* N" via a
(from-start x from-end x voice) conjunction -> 5-role Hebbian ensemble (agent/action/patient/
attribute/attribute2), with fixed caps S_CAP=3, E_CAP=2 -- by its own docstring, scoped to
"1-2 adjectives" (attributed_parser.py:12). `NPHeadBinder` below is the same architecture pattern
(conjunction unit -> Hebbian-trained role ensemble -> firing-rate argmax read-out on a private
Izhikevich SimulationBridge) with the from-START factor DROPPED (irrelevant to "is this word the
head or a modifier?") and the three non-agent/action roles collapsed into TWO: HEAD (the span's
last word) and MODIFIER (every other word). Because only the from-END boundary flag (e==0 vs e>=1)
now decides the role, exactly 2 conjunction units, trained once, cover a span of ANY length --
unlike AttributedBridgeParser's fixed window. This is a genuinely spiking, genuinely Hebbian-learned
mechanism (`enable_hebbian_learning`, `_run_one_simulation_step`); role assignment (HEAD vs
MODIFIER) is decided by which role ensemble fires hardest when a word's boundary-flag conjunction is
driven ALONE, never by host string logic.

NP-BOUNDARY DETECTION vs NP-BINDING (the line this file is careful not to cross). Per the finding's
own framing, "the NP-boundary detection may use a minimal lexical span rule ... but the BINDING of
the span into one unit must be the spiking population-binding mechanism, not a host string join."
`segment_clause()` below is that declared, minimal, host lexical rule: it finds where a determiner-
headed span STARTS and ENDS (scanning for a copula/passive auxiliary or a small verb lexicon) --
exactly the same category of host preprocessing the baseline runner already used (stopword/negator
removal) and the ORIGINAL moat-verifier de-risk used (first-verb-lexicon-match). It never decides
WHICH WORD PLAYS WHICH ROLE. Once a span's boundaries are fixed, `NPHeadBinder.bind()` is what
certifies, in spikes, that every word in it belongs to ONE unit (each word reads out HEAD or
MODIFIER on the SAME two role ensembles) -- and the resulting identity string is built by joining
the words AFTER that spiking certification, exactly the same pass-through BridgeParser/
AttributedBridgeParser already do for single words (`{role: word for ...}` in both classes) -- the
literal surface text of a token was never itself computed by spikes anywhere in this pipeline
family; only which SLOT it belongs to is. The outer subject-NP/verb/object-NP frame is then hand-
ed to the UNCHANGED `BridgeParser.parse()` (its existing position x voice role read-out, including
its existing voice="passive" 0<->2 flip) -- that mechanism is not modified or re-implemented here.

PASSIVE DETECTION. A small lexicon of past-participle verbs (`PARTICIPLES`) plus "was"/"were" plus
a "by <NP>" clause is the passive detector the finding flagged as HOST-ASSUMED, never measured, in
the baseline file. It is exercised here for real: "The Eiffel Tower was built by Gustave Eiffel" ->
voice="passive" is fed to the existing BridgeParser flip. A passive WITHOUT an explicit "by"-agent
("was built in London") has no agent-NP to fill the 3rd slot and is honestly left UNPARSED -- the
same shape the original finding could not parse, reported here as the still-failing residual.

Run: python -m research.runners._spiking_np_boundary_extraction_derisk
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")   # small nets; CPU is plenty, avoids GPU init overhead

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._open_text_moat_verifier_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    Claim, FactStore, classify_claim, split_clauses, is_opinion, STOPWORDS,
)
from research.runners._open_text_spiking_extraction_derisk import (  # noqa: E402  (REUSE UNCHANGED)
    ITEMS, NEGATORS, build_store as build_base_store, extract_claims_spiking, _find_claim,
)
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402  (UNCHANGED)


# ============================================================================
# 1. NPHeadBinder -- the generalized spiking population-binding mechanism.
#    See module docstring for the derivation from AttributedBridgeParser.
# ============================================================================

class NPHeadBinder:
    """(boundary-flag) -> {HEAD, MODIFIER} Hebbian population binder on a private Izhikevich bridge.
    Same architecture family as BridgeParser/AttributedBridgeParser (conjunction unit -> role
    ensemble -> firing-rate read-out, Hebbian-trained co-firing), collapsed to the single factor
    that generalizes to arbitrary span length: is this word the LAST word of the span (HEAD) or not
    (MODIFIER)? `bind()` calls the spiking read-out once per word in a span and returns the ordered
    per-word roles plus the joined identity string (host bookkeeping over an already spiking-decided
    partition, not a decision-making join -- see module docstring)."""

    ROLES = ["HEAD", "MODIFIER"]

    def __init__(self, seed=42, R=40, n_epochs=30, train_steps=120, test_steps=80, drive=2500.0):
        self.R = R; self.test_steps = test_steps; self.drive = drive
        self.conj = [0, 1]                        # 0 = NONEDGE (e>=1) ; 1 = EDGE (e==0, last word)
        self.role_idx = {r: [2 + i * R + j for j in range(R)] for i, r in enumerate(self.ROLES)}
        self.teacher = {0: "MODIFIER", 1: "HEAD"}  # the Hebbian teacher: which role each conjunction drives
        pre, post, w = [], [], []
        for k in self.conj:
            for r in self.ROLES:
                for j in self.role_idx[r]:
                    pre.append(k); post.append(j); w.append(0.5)
        cfg = CoreSimConfig()
        cfg.num_neurons = 2 + len(self.ROLES) * R
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = int(seed); cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0; cfg.num_traits = 1
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = True
        cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
        for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
                  "enable_reward_modulation", "enable_watts_strogatz"):
            setattr(cfg, f, False)
        cfg.ou_std_current_pA = 20.0
        self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                       runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge.inject_explicit_wiring({"npbind": {"pre_indices": pre, "post_indices": post,
                                                        "initial_weights": np.array(w, dtype=np.float32),
                                                        "plastic": True, "conn_type": "E_TO_E", "count": len(pre)}})
        xp = self._bridge_xp()
        self.conj_arr = xp.asarray(self.conj, dtype=xp.int64)
        self.role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in self.role_idx.items()}
        self._n = self.bridge.core_config.num_neurons
        self._train(n_epochs, train_steps)

    def _bridge_xp(self):
        """Bridge's own array module, not the sticky global (see BridgeParser._bridge_xp,
        brain_conversational_agent.py:~100, for why -- same fix, same reasoning)."""
        arr = getattr(self.bridge, "cp_external_input_current", None)
        if arr is not None:
            try:
                import cupy as _cp  # noqa: PLC0415
                return _cp.get_array_module(arr)
            except Exception:
                return np
        xp, _ = get_backend()
        return xp

    def _step_reset(self, reset=20):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset):
            self.bridge._run_one_simulation_step()

    def _train(self, n_epochs, train_steps):
        xp = self._bridge_xp()
        for _ in range(n_epochs):
            for k in self.conj:
                self._step_reset()
                cur = xp.zeros(self._n, dtype=xp.float32)
                cur[self.conj_arr[k]] = self.drive
                cur[self.role_arr[self.teacher[k]]] = self.drive   # teacher-drive the correct role
                self.bridge.cp_external_input_current[:] = cur
                for _ in range(train_steps):
                    self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0

    def role_of_edge(self, is_edge):
        """Drive the boundary-flag conjunction ALONE; read which role ensemble fires hardest."""
        xp = self._bridge_xp()
        k = 1 if is_edge else 0
        self._step_reset()
        cur = xp.zeros(self._n, dtype=xp.float32)
        cur[self.conj_arr[k]] = self.drive
        self.bridge.cp_external_input_current[:] = cur
        rates = {r: 0.0 for r in self.ROLES}
        for _ in range(self.test_steps):
            self.bridge._run_one_simulation_step()
            for r in self.ROLES:
                rates[r] += float(to_host(self.bridge.cp_firing_states[self.role_arr[r]].astype(xp.float64).mean()))
        self.bridge.cp_external_input_current[:] = 0.0
        return max(rates, key=rates.get)

    def bind(self, span_words):
        """Collapse an ordered word span into ONE role-fillable unit. Returns (identity, per_word_roles).
        `identity` is the words joined in original order -- host bookkeeping over a partition that was
        ALREADY decided in spikes (every word's HEAD/MODIFIER tag came from `role_of_edge`), not a
        host decision about span membership (that was `segment_clause`'s job, upstream)."""
        n = len(span_words)
        roles = [self.role_of_edge(idx == n - 1) for idx in range(n)]
        identity = " ".join(span_words)
        return identity, roles


# ============================================================================
# 2. Host NP-boundary / clause-segmentation rule -- minimal, lexical, declared.
#    Decides WHERE spans start/end and which is subject/verb/object; NEVER
#    decides which word plays which grammatical ROLE (that is NPHeadBinder +
#    BridgeParser, both spiking).
# ============================================================================

DETERMINERS = STOPWORDS                              # reused unchanged: {the, a, an, to, that, this, these, those}
PASSIVE_AUX = {"was", "were"}
COPULA_AUX = {"is", "are", "was", "were"}
PARTICIPLES = {"built", "discovered", "designed", "founded", "created", "constructed"}
VERB_LEXICON = {"supports", "produces", "hosts"}       # multi-word-subject fallback verb cues (see segment_clause)


def _strip_det(words):
    return [w for w in words if w not in DETERMINERS]


def segment_clause(tokens):
    """tokens: lowercase word tokens of one clause, determiners/negators STILL PRESENT (needed to find
    boundaries). Returns a dict describing the (subject-span, verb, object-span, voice) frame, or a dict
    with object=None when the span cannot be closed (honest unparsed), or None if nothing matches at all.
    Three lexical passes, in priority order: passive (aux+participle[+by-agent]), copula (aux, no
    participle), plain SVO (exact-3-content-words fast path unchanged from the baseline runner, THEN a
    verb-lexicon fallback for longer spans)."""
    n = len(tokens)

    # -- pass 1: passive (was/were + participle [+ by <agent-NP>]) --
    for i in range(n - 1):
        if tokens[i] in PASSIVE_AUX and tokens[i + 1] in PARTICIPLES:
            verb = tokens[i + 1]
            subj = _strip_det(tokens[:i])
            rest = tokens[i + 2:]
            if "by" in rest:
                bi = rest.index("by")
                agent = _strip_det(rest[bi + 1:])
                if subj and agent:
                    return dict(kind="passive_by", subject=subj, verb=verb, object=agent,
                                voice="passive", negated=False)
            return dict(kind="passive_no_agent", subject=subj, verb=verb, object=None,
                        voice="passive", negated=False)   # no by-agent -> honest unparsed

    # -- pass 2: copula (is/are/was/were, not already claimed by pass 1) --
    for i in range(n):
        if tokens[i] in COPULA_AUX:
            subj = _strip_det(tokens[:i])
            pred = _strip_det(tokens[i + 1:])
            if subj and pred:
                return dict(kind="copula", subject=subj, verb="is", object=pred,
                            voice="active", negated=False)
            return dict(kind="copula_incomplete", subject=subj, verb="is", object=None,
                        voice="active", negated=False)

    # -- pass 3: plain SVO. Exact-3-content-words fast path (byte-identical in spirit to the
    #    baseline runner's zero-lexical-knowledge extractor) first; a small verb-lexicon fallback
    #    only when that does not apply (a genuinely NEW lexical step, needed to locate the verb
    #    inside a >3-content-word span -- documented, used only for segmentation, never for role
    #    assignment). --
    negated = any(w in NEGATORS for w in tokens)
    content = [w for w in tokens if w not in DETERMINERS and w not in NEGATORS]
    if len(content) == 3:
        return dict(kind="plain3", subject=[content[0]], verb=content[1], object=[content[2]],
                    voice="active", negated=negated)
    for i, w in enumerate(content):
        if w in VERB_LEXICON:
            subj, obj = content[:i], content[i + 1:]
            if subj and obj:
                return dict(kind="plainN", subject=subj, verb=w, object=obj,
                            voice="active", negated=negated)
    return None


def extract_svo_npbind(clause, parser, np_binder):
    """The full pipeline: lexical segmentation (host, declared) -> spiking NP-binding of each span
    (NPHeadBinder, spiking) -> spiking position x voice role read-out on the collapsed 3-slot frame
    (BridgeParser, UNCHANGED). Returns ((agent, action, patient, negated), meta) or (None, seg)."""
    tokens = re.findall(r"[a-zA-Z']+", clause.lower())
    seg = segment_clause(tokens)
    if seg is None or seg.get("object") is None:
        return None, seg
    subj_identity, subj_roles = np_binder.bind(seg["subject"])
    obj_identity, obj_roles = np_binder.bind(seg["object"])
    frame = [subj_identity, seg["verb"], obj_identity]
    roles = parser.parse(frame, voice=seg["voice"])   # <-- the EXISTING spiking parser, unchanged
    meta = {"segmentation_kind": seg["kind"], "voice": seg["voice"],
            "subject_span": seg["subject"], "subject_np_roles": subj_roles, "subject_identity": subj_identity,
            "object_span": seg["object"], "object_np_roles": obj_roles, "object_identity": obj_identity}
    return (roles["agent"], roles["action"], roles["patient"], seg.get("negated", False)), meta


def extract_claims_npbind(paragraph, parser, np_binder):
    claims, metas = [], []
    for clause in split_clauses(paragraph):
        lower = clause.lower()
        if is_opinion(lower):
            claims.append(Claim(text=clause, kind="opinion")); metas.append(None)
            continue
        parsed, meta = extract_svo_npbind(clause, parser, np_binder)
        if parsed is None:
            claims.append(Claim(text=clause, kind="unparsed")); metas.append(meta)
            continue
        agent, action, patient, negated = parsed
        claims.append(Claim(text=clause, kind="assertion", agent=agent, action=action,
                             patient=patient, negated=negated))
        metas.append(meta)
    return claims, metas


# ============================================================================
# 3. Fact store: reuse the baseline's UNCHANGED, plus new multi-word-NP /
#    copula / passive facts, keyed EXACTLY as NPHeadBinder.bind()'s identity
#    strings will read (space-joined, determiners stripped, lowercase) --
#    the FactStore is the surrogate for "what the brain actually knows"
#    (same abstraction level the moat-verifier file documents: a plain dict
#    standing in for the spiking composer's SVO memory).
# ============================================================================

def build_store():
    s = build_base_store()
    s.store("great barrier reef", "supports", "coral")
    s.store("great barrier reef", "hosts", "diverse marine life")   # DISTINCT verb from "supports" above --
    # FactStore.store keys on (agent, action) only (one patient per relation, see
    # _open_text_moat_verifier_derisk.FactStore.store); reusing "supports" for a second
    # fact silently overwrote the first (caught by a WRONG-labelled run of this file
    # before this fix -- a real FactStore limitation, not an NP-binding bug, but a test-
    # design collision worth naming so it is not silently rediscovered).
    s.store("great barrier reef", "is", "largest coral reef system in world")
    s.store("amazon rainforest", "produces", "oxygen")
    s.store("eiffel tower", "is", "famous landmark")
    s.store("gustave eiffel", "built", "eiffel tower")
    return s


# ============================================================================
# 4. New harder items: genuine multi-word proper-noun subjects, copula
#    predicate nominals, passives with/without a by-agent. The 2 items in the
#    imported (unchanged) ITEMS list already marked expect_parse=False (the
#    Great Barrier Reef copula sentence and the Eiffel Tower passive-without-
#    agent sentence) are reused AS-IS -- no need to duplicate them here; they
#    are the headline "did the ORIGINAL failing item now parse?" check.
# ============================================================================

NEW_ITEMS = [
    dict(paragraph="The Great Barrier Reef supports coral.",
         clauses=[dict(text="Great Barrier Reef supports coral",
                       gold_triple=("great barrier reef", "supports", "coral"), gold_label=True)]),
    dict(paragraph="The Great Barrier Reef supports cacti.",
         clauses=[dict(text="Great Barrier Reef supports cacti",
                       gold_triple=("great barrier reef", "supports", "cacti"), gold_label=False)]),
    dict(paragraph="The Great Barrier Reef hosts diverse marine life.",
         clauses=[dict(text="Great Barrier Reef hosts diverse marine life",
                       gold_triple=("great barrier reef", "hosts", "diverse marine life"), gold_label=True)]),
    dict(paragraph="The Amazon Rainforest produces oxygen.",
         clauses=[dict(text="Amazon Rainforest produces oxygen",
                       gold_triple=("amazon rainforest", "produces", "oxygen"), gold_label=True)]),
    dict(paragraph="The Eiffel Tower is a famous landmark.",
         clauses=[dict(text="Eiffel Tower is a famous landmark",
                       gold_triple=("eiffel tower", "is", "famous landmark"), gold_label=True)]),
    dict(paragraph="The Eiffel Tower is a natural wonder.",
         clauses=[dict(text="Eiffel Tower is a natural wonder",
                       gold_triple=("eiffel tower", "is", "natural wonder"), gold_label=False)]),
    dict(paragraph="The Eiffel Tower was built by Gustave Eiffel.",
         clauses=[dict(text="Eiffel Tower was built by Gustave Eiffel",
                       gold_triple=("gustave eiffel", "built", "eiffel tower"), gold_label=True)]),
    dict(paragraph="The Eiffel Tower was built by Isambard Kingdom Brunel.",
         clauses=[dict(text="Eiffel Tower was built by Isambard Kingdom Brunel",
                       gold_triple=("isambard kingdom brunel", "built", "eiffel tower"), gold_label=False)]),
]

ALL_ITEMS = ITEMS + NEW_ITEMS
HARD_TEXTS = {c["text"].lower() for it in NEW_ITEMS for c in it["clauses"]} | {
    "the great barrier reef is the largest coral reef system in the world",
    "the eiffel tower was built in london",
}


# ============================================================================
# 5. Scoring harness -- runs BEFORE (plain BridgeParser, extract_claims_spiking,
#    reused unchanged) and AFTER (extract_claims_npbind) over the identical
#    combined item set, per-clause, and diffs them.
# ============================================================================

def _score_pass(extractor_fn, store):
    """extractor_fn(paragraph) -> list[Claim]. Returns per-item rows + aggregate, mirroring the
    baseline runner's scoring loop exactly (kept byte-similar for direct comparability)."""
    per_item, scored_rows = [], []
    n_assertion = n_parsed = n_unparsed = n_opinion = 0
    false_total = false_caught = false_slipped = 0
    row_by_text = {}
    for item in ALL_ITEMS:
        paragraph = item["paragraph"]
        claims = extractor_fn(paragraph)
        clause_rows = []
        for gold in item["clauses"]:
            claim = _find_claim(claims, gold["text"])
            row = {"gold_text": gold["text"], "gold_kind": gold.get("gold_kind", "assertion"),
                   "gold_triple": gold.get("gold_triple"), "gold_label": gold.get("gold_label")}
            if claim is None:
                row["outcome"] = "CLAUSE_SPLIT_MISS"
                clause_rows.append(row); row_by_text[gold["text"].lower()] = row
                continue
            row["matched_claim_text"] = claim.text
            row["matched_kind"] = claim.kind
            if gold.get("gold_kind") == "opinion":
                n_opinion += 1
                row["outcome"] = "opinion_suppressed" if claim.kind == "opinion" else "MISLABELLED_NOT_OPINION"
                clause_rows.append(row); row_by_text[gold["text"].lower()] = row
                continue
            n_assertion += 1
            is_false_gold = (gold["gold_label"] is False)
            if is_false_gold:
                false_total += 1
            if claim.kind == "unparsed":
                n_unparsed += 1
                row["extracted_triple"] = None
                row["predicted_verdict"] = "unparsed_suppressed"
                row["outcome"] = "unparsed_suppressed"
                if is_false_gold:
                    false_slipped += 1
                clause_rows.append(row); row_by_text[gold["text"].lower()] = row
                continue
            n_parsed += 1
            verdict = classify_claim(claim, store)
            predicted_label = (verdict == "grounded")
            row["extracted_triple"] = (claim.agent, claim.action, claim.patient)
            row["predicted_verdict"] = verdict
            row["predicted_label"] = predicted_label
            row["correct"] = (predicted_label == gold["gold_label"])
            row["outcome"] = "CORRECT" if row["correct"] else "WRONG"
            scored_rows.append({"gold_text": gold["text"], "gold_label": gold["gold_label"],
                                 "predicted_label": predicted_label})
            if is_false_gold and not predicted_label:
                false_caught += 1
            clause_rows.append(row); row_by_text[gold["text"].lower()] = row
        per_item.append({"paragraph": paragraph, "clause_results": clause_rows})

    tp = fp = fn = tn = 0
    for r in scored_rows:
        gold_false = (r["gold_label"] is False)
        pred_false = (r["predicted_label"] is False)
        if gold_false and pred_false: tp += 1
        elif (not gold_false) and pred_false: fp += 1
        elif gold_false and (not pred_false): fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (tp + fp) and (tp + fn) and (precision + recall) else float("nan"))
    coverage = n_parsed / n_assertion if n_assertion else float("nan")
    aggregate = {
        "n_assertion_clauses": n_assertion, "n_opinion_clauses": n_opinion,
        "n_parsed": n_parsed, "n_unparsed": n_unparsed, "extraction_coverage": coverage,
        "verifier_on_parsed_subset": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                       "precision": precision, "recall": recall, "f1": f1},
        "false_claim_catch": {"false_claims_total": false_total, "false_claims_caught": false_caught,
                               "false_claims_slipped_unparsed": false_slipped,
                               "catch_rate": (false_caught / false_total) if false_total else float("nan")},
    }
    return per_item, aggregate, row_by_text


def main():
    t0 = time.time()
    store = build_store()
    parser = BridgeParser(seed=42)
    np_binder = NPHeadBinder(seed=42)
    build_s = time.time() - t0

    xp, backend_name = get_backend()

    # sanity self-test: the binder must correctly separate HEAD from MODIFIER on a synthetic
    # 3-word span before we trust it on real items (verified interactively, printed below too).
    _sanity_id, _sanity_roles = np_binder.bind(["alpha", "beta", "gamma"])
    sanity_ok = (_sanity_roles == ["MODIFIER", "MODIFIER", "HEAD"])
    print(f"NPHeadBinder self-test on ['alpha','beta','gamma'] -> {_sanity_roles} "
          f"(expected ['MODIFIER','MODIFIER','HEAD']): {'OK' if sanity_ok else 'FAILED'}")

    per_item_before, agg_before, rows_before = _score_pass(
        lambda p: extract_claims_spiking(p, parser), store)
    per_item_after, agg_after, rows_after = _score_pass(
        lambda p: extract_claims_npbind(p, parser, np_binder)[0], store)

    # hard subset = the 8 new items + the 2 previously-unparsed items reused unchanged from ITEMS
    hard_before = [r for t, r in rows_before.items() if t in HARD_TEXTS and "gold_label" in r
                   and r.get("gold_kind", "assertion") != "opinion"]
    hard_after = [r for t, r in rows_after.items() if t in HARD_TEXTS and "gold_label" in r
                  and r.get("gold_kind", "assertion") != "opinion"]
    hard_n = len(hard_after)
    hard_before_parsed = sum(1 for r in hard_before if r["outcome"] in ("CORRECT", "WRONG"))
    hard_after_parsed = sum(1 for r in hard_after if r["outcome"] in ("CORRECT", "WRONG"))
    still_unparsed = sorted(t for t, r in rows_after.items()
                            if r["outcome"] == "unparsed_suppressed" and t in HARD_TEXTS)

    # newly-parsed subset: unparsed BEFORE, resolved (parsed) AFTER -- computed by diffing the
    # two passes per clause text, not hand-counted, so a coding mistake in the narrative below
    # cannot silently diverge from what the runner actually measured.
    newly_parsed_texts = [t for t in rows_after
                          if rows_before.get(t, {}).get("outcome") == "unparsed_suppressed"
                          and rows_after[t]["outcome"] in ("CORRECT", "WRONG")]
    tp = fp = fn = tn = 0
    nf_total = nf_caught = 0
    for t in newly_parsed_texts:
        r = rows_after[t]
        gold_false = (r["gold_label"] is False)
        pred_false = (r.get("predicted_label") is False)
        if gold_false: nf_total += 1
        if gold_false and pred_false: tp += 1; nf_caught += 1
        elif (not gold_false) and pred_false: fp += 1
        elif gold_false and (not pred_false): fn += 1
        else: tn += 1
    np_precision = tp / (tp + fp) if (tp + fp) else float("nan")
    np_recall = tp / (tp + fn) if (tp + fn) else float("nan")
    np_f1 = (2 * np_precision * np_recall / (np_precision + np_recall)
             if (tp + fp) and (tp + fn) and (np_precision + np_recall) else float("nan"))

    knobs = {
        "seed": 42,
        "parser_class": "BridgeParser (unchanged, reused)",
        "np_binder_class": "NPHeadBinder (new, this file, generalizes AttributedBridgeParser)",
        "backend": backend_name, "sim_backend_env": os.environ.get("SIM_BACKEND"),
        "neuron_model": "IZHIKEVICH",
        "bridgeparser_num_neurons": parser.bridge.core_config.num_neurons,
        "npbinder_num_neurons": np_binder.bridge.core_config.num_neurons,
        "build_train_seconds": build_s,
        "np_binder_sanity_check": {"input": ["alpha", "beta", "gamma"], "roles": _sanity_roles, "ok": sanity_ok},
        "determiners_stripped": sorted(DETERMINERS),
        "negators_set": sorted(NEGATORS),
        "passive_aux": sorted(PASSIVE_AUX), "copula_aux": sorted(COPULA_AUX),
        "participles_lexicon": sorted(PARTICIPLES), "plain_svo_verb_lexicon_fallback": sorted(VERB_LEXICON),
        "n_items_baseline_imported": len(ITEMS), "n_items_new": len(NEW_ITEMS), "n_items_total": len(ALL_ITEMS),
    }

    aggregate = {
        "combined_set": {"before": agg_before, "after": agg_after},
        "hard_subset_multiword_np_copula_passive": {
            "n_clauses": hard_n,
            "before_parsed": hard_before_parsed, "before_coverage": hard_before_parsed / hard_n if hard_n else None,
            "after_parsed": hard_after_parsed, "after_coverage": hard_after_parsed / hard_n if hard_n else None,
            "still_unparsed_texts": still_unparsed,
        },
        "newly_parsed_subset": {
            "n_clauses": len(newly_parsed_texts), "texts": newly_parsed_texts,
            "verifier": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": np_precision, "recall": np_recall, "f1": np_f1},
            "false_claim_catch": {"false_total": nf_total, "false_caught": nf_caught,
                                  "catch_rate": (nf_caught / nf_total) if nf_total else float("nan")},
        },
    }

    print("\n=== BEFORE (plain BridgeParser, extract_claims_spiking, unchanged) ===")
    print(json.dumps(agg_before, indent=2))
    print("\n=== AFTER (NP-boundary spiking binding + BridgeParser) ===")
    print(json.dumps(agg_after, indent=2))
    print("\n=== HARD SUBSET (multi-word-NP / copula / passive) ===")
    print(json.dumps(aggregate["hard_subset_multiword_np_copula_passive"], indent=2))
    print("\n=== NEWLY-PARSED SUBSET verifier ===")
    print(json.dumps(aggregate["newly_parsed_subset"], indent=2))

    out = {"knobs": knobs, "aggregate": aggregate,
           "items_before": per_item_before, "items_after": per_item_after}
    out_path = os.path.join(_REPO, "research", "findings", "raw",
                             "_spiking_np_boundary_extraction_derisk.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")
    return aggregate


if __name__ == "__main__":
    main()
