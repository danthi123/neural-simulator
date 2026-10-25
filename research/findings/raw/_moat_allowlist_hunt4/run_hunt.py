"""General leak-hunt runner for the moat claim checkers (prereg 2026-10-20-moat-allowlist-claim-checker, bar 3, AMENDMENT 2).

Runs every case of a hunt through the production claim gate (`ChatBrain._verify_claim_set`) and the composer entry
(`RichAnswerComposer._verify_rendered`) for three arms: stock (all moat flags off), b2h (`BRAIN_MOAT_SLUG_NORM=1`) and allowlist
(`BRAIN_MOAT_ALLOWLIST=1`), at parser seeds 42, 43, 44, 100, 101, 102 and 1 (one deterministic build per seed: one replicate per
case, not draws). A reply case (round 8 on) also goes through a third entry, `reply` (below).

THE THREE ENTRIES (what each measures):
  gate      `ChatBrain._verify_claim_set(sentence, gathered)`: the claim gate's verdict on the case's `sentence` as one text.
  composer  `RichAnswerComposer._verify_rendered(sentence, None, gathered[0], gathered)`: ONE candidate through the composer's check.
  reply     (cases with "reply": true only; every other case records "n/a" there, never counted, never UNDEFINED) the composer's
            real PARAGRAPH path, `RichAnswerComposer.render_paragraph(gathered)`, for a turn whose gathered facts are exactly the
            case's triples, with a renderer stand-in (`ReplyRenderer`) whose rendering of gathered fact i is `sentences[i]` (asserted
            None, a free-prose mouth; render_svo, render_svo_regen and the batch methods all return it). Nothing else in the
            composer changes: its per-fact candidate loop, its per-candidate check, its build-6 JOINED check (`_join_verified`,
            flag ON) and its prefix cut run unmodified. Run on BOTH paths at every parser seed and arm: `sequential`
            (BRAIN_RICH_BATCH_RENDER unset, a renderer without batch methods) and `batched` (BRAIN_RICH_BATCH_RENDER=1, a renderer
            with render_svo_batch and render_svo_regen_batch). The spiking recall surface (default ON in production) would speak a
            bounded SVO itself instead of the mouth, so the entry pins BRAIN_SPIKING_MOUTH_RECALL=0 and unsets
            BRAIN_TOUCHPOINT_A_FACT_CLAUSE: the reply under test is the MOUTH's text. A reply case carries "sentences", one
            sentence per gathered triple in the same order (k = len(sentences) = len(gathered), 2 <= k <= 4, no repeated
            triple), and its `sentence` is those sentences joined as the composer joins them (the gate and composer entries read
            `sentence` unchanged).
  ORDER: the composer renders and joins the facts in gathered order on both paths, and the stand-in answers BY FACT (the triple
  it is asked to render, never the call index), so a reordering cannot be mis-mapped: it matches no in-order text (UNDEFINED).
  PREFIX RULE: each (arm, seed, path) emission maps to j by `map_reply` (pure): j = k when the full reply is emitted, 1 <= j < k
  when the emitted text is the first j sentences, 0 when the composer emitted nothing. The text is compared EXACTLY against the
  composer's own construction, declared in SPEC (build 6) and in `_join_verified` / `_terminate`: with the allowlist flag ON each
  sentence is `_terminate`d (trailing U+0020/LF/CR stripped, "." added when it has no terminal punctuation) and the sentences are
  joined with one U+0020; flag OFF they are joined with one U+0020 as rendered. No other normalisation (a U+00A0 is not a space).
  The composer's own `kept` facts must give the same text. An in-order NON-prefix selection (correct composer behaviour: a fact
  whose sentence fails alone is dropped before the joined check) is recorded as `subsequence` with its 1-based indices; like a
  prefix it counts through its adjudicated label, and an emitted selection with no label is UNDEFINED.
  UNDEFINED (never "not kept"): text matching no in-order selection (reordered, a partial sentence, extra text, another mouth's
  text, the "!" mood marker), the composer's kept facts disagreeing with the text, an ambiguous match the kept facts cannot decide,
  a renderer error (even one the batched path swallowed), an exception, the wrong path taken, a non-bool claim verdict under the
  allowlist (the composer coerces it) or the allowlist reaching the single-triple `_verify`, an invalid reply case, a reply case
  with no labels or invalid labels, and an emitted selection with no label. Any UNDEFINED reply case makes the arm not VALID.
  COUNTING: per arm, a case's outcome is the set of j (and labelled selections) it emitted over every seed and path; "kept U" =
  the cases that emitted, at ANY seed through EITHER path, a text whose adjudicated label is U (likewise B and F; a case can be in
  more than one), plus the full distribution of j over every (case, seed, path) record (keys '0'..'k'; 'sub:1+3' for a
  non-prefix selection, so sentence 2 alone, 'sub:2', never reads as j = 2).
  LABELS (one key family, Claude ruling 2026-10-21): adjudication.json "subset_labels": {"<case id>": {"1": label, "1+2": label,
  "2": label, "1+3": label, ...}}, a label for EVERY in-order non-empty selection of the reply's sentences (the judges label them
  all: judge_prep.py's default; k <= 4, so at most 15 items per case). Every emitted selection, prefixes included, is looked up
  there by its key (`subset_key`). The full selection's label must equal the case's own label after relabel_to_B (it is that label
  when absent). For compatibility "prefix_labels": {"<case id>": [label of prefix 1, ..., label of prefix k]} (the last the full
  reply's) is still read; where both families label one selection they must agree. adjudicate.py writes both.
  REPLY CONTROLS (over the store's own triples, never case text): RP1 two faithful sentences -> j = k at every seed on both paths;
  RN1 the second sentence role-reversed -> j = 1; RN0 the first role-reversed -> j = 0 or a selection without sentence 1; RNJ the
  same faithful sentence twice (each passes alone, the joined check must cut the repeat) -> j = 1. A failed reply control in the
  allowlist arm sets exit code 5; stock / B2h reply-control outcomes are reported, never failed on.

usage (cwd = worktree): python -m research.findings.raw._moat_allowlist_hunt4.run_hunt <cases_dir> <out.json>
  cases_dir holds cases.json ([{id, gathered, sentence, expect, class, ...}, a reply case adds "reply": true and "sentences"]) and,
  optionally, adjudication.json (relabel_to_B; prefix_labels and subset_labels for reply cases).
  options: --seeds 42,43,..  --limit N (first N cases, smoke only)  --equiv-n N (construction-equivalence sample, default 6)
           --store PATH (facts.json whose vocabulary the brain holds)
exit code: 0 ok; 2 the allowlist arm has an UNDEFINED case (bar 3 cannot be read); 4 the construction-equivalence check failed;
           5 a control failed (below; or a reply control in the allowlist arm); 3 only the stock or B2h comparison column is not
           VALID (void comparison columns).

WHAT ROUND 3 GOT WRONG (the defects this runner removes; diagnosed 2026-10-20 from the production code):
  (1) ChatBrain._verify_claim_set returns (None, None) when the claim verifier is UNBUILDABLE: ClaimEntailmentVerifier.__init__ raises
      AssertionError on a gathered set holding a role-permutation of one of its triples, and `_build_claim_verifier` turns that into
      None (board hole #265: a mutual `shares_border_with` pair). That is the production signal "fall back to the single-triple
      `ChatBrain._verify`". Round 3 stored the raw None and counted it "kept 0". Here the stock/B2h verdict on such a case is the
      FALLBACK's verdict, labelled src="fallback".
  (2) The fallback `_verify(surface, asserted, svo)` was called with the PROSE as `asserted`; production passes the renderer's canonical
      SVO (stub mouth) or None (a free-prose mouth, then `_extract_svo_from_prose` re-reads the prose). `BridgeParser.parse` asserts
      len(words)==3, so every such call raised AssertionError (245 case-seeds stock, 203 B2h). A hunt sentence is free prose, so
      asserted=None here. The round-3 object was also a bare `ChatBrain.__new__` without the vocabulary sets `_verify` reads.
  (3) The composer entry coerced a None allowlist verdict to False (`bool(accepted)`); here the raw gate return is recorded.
Every verdict is a bool; anything else (None, an exception) is recorded per case-seed with its reason and makes that arm UNDEFINED for
that case (AMENDMENT 2), never "kept 0".
INSTRUMENT (build 4, F6): each allowlist-arm record of a reject carries `reason`, the claim gate's reject_reason string (gate entry: the
gate's result; composer entry: what the traced gate returned), and the allowlist arm's summary carries `reject_reason_tags`, per entry the
count of those case-seed reasons by leading tag (`reason_tag`). No verdict and no other summary field depends on them.
"""
import research.runners  # noqa: F401  (the provenance door; stdlib-only, must be first)
import os

os.environ.setdefault("SIM_BACKEND", "numpy")           # explicit: round 3 touched the GPU via CuPy init
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")       # a hunt is CPU work; never contact a GPU another job is using

import contextlib
import json
import re
import socket
import sys
import time
from collections import Counter
from itertools import combinations

SEEDS = (42, 43, 44, 100, 101, 102, 1)
ARMS = {"stock": {}, "b2h": {"BRAIN_MOAT_SLUG_NORM": "1"}, "allowlist": {"BRAIN_MOAT_ALLOWLIST": "1"}}
FLAGS = ("BRAIN_MOAT_SLUG_NORM", "BRAIN_MOAT_ALLOWLIST", "BRAIN_CLAIM_MOAT")
ENTRIES = ("gate", "composer")
LABELS = ("U", "B", "F")
STRICT_ARM = "allowlist"
DEFAULT_STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
VOCAB_ATTRS = ("stored_facts", "agents_set", "actions_set", "patients_set", "inflect")
EXIT_OK, EXIT_ALLOWLIST_UNDEFINED, EXIT_COMPARISON_VOID, EXIT_EQUIV_FAILED, EXIT_CONTROL_FAILED = 0, 2, 3, 4, 5
REPLY_PATHS = ("sequential", "batched")
REPLY_ENV = {"BRAIN_SPIKING_MOUTH_RECALL": "0", "BRAIN_TOUCHPOINT_A_FACT_CLAUSE": None}   # None = unset (see the docstring)
REPLY_MAX_K = 4                     # Claude ruling 2026-10-21: every selection is judged, so at most 2**4 - 1 = 15 items per case
NA = "n/a"

TARGET_POLICY = ("a hunt sentence is free prose and the hunt does not know which gathered fact the mouth was rendering it for; where the "
                 "production path reaches the single-triple `_verify` (a fallback, or a one-fact turn), the sentence is kept if `_verify` "
                 "accepts it for ANY gathered fact as the rendered target (round 3 tried only the first fact). The claim gate and the "
                 "allowlist do not depend on the target.")
VOCAB_NOTE = ("`_verify` re-reads the prose against the brain's own vocabulary sets (agents_set, actions_set, patients_set, inflect). "
              "PRIMARY verdicts: the brain holds the whole store (every triple of --store). In the shipped webapp those sets are read from "
              "the composer's `.kb`, which for a TieredFactStore is the small conversation BUFFER, not the LTM, so a production turn on an "
              "LTM-only entity would extract nothing and reject; the primary vocabulary is therefore generous to stock. Each fallback "
              "record also carries alt_v, the verdict with the vocabulary of the case's gathered triples only (the cleanest extraction "
              "stock can have); the summary reports both.")
PARSER_NOTE = ("The agent's parser is BridgeParser(seed) (a BrainConversationalAgent with an external composer builds exactly that). The "
               "default webapp brain (composer_kind onebrain) parses with the OneBrainComposer's own on-bridge parser, a different object "
               "trained by the same Hebbian rule; building seven of those is ~180 s each and is not done here (as in round 3).")


# ---------------------------------------------------------------------------------------------------------------------------------
# verdict records
# ---------------------------------------------------------------------------------------------------------------------------------
def _rec(v, src, **extra):
    r = {"v": v, "src": src}
    r.update({k: x for k, x in extra.items() if x is not None})
    return r


def _undefined(why, src="undefined"):
    return {"v": None, "src": src, "why": why}


def _error(exc):
    return {"v": None, "src": "error", "why": "%s: %s" % (type(exc).__name__, str(exc)[:200])}


def is_defined(rec):
    """A verdict counts only if it is a real bool (None, a string, a numpy bool, an exception record: all UNDEFINED)."""
    return isinstance(rec, dict) and isinstance(rec.get("v"), bool)


def case_status(recs, key="v"):
    """UNDEFINED if ANY seed's verdict is not a bool (AMENDMENT 2), else 'kept' if any seed kept it, else 'not_kept'.
    key='alt_v' reads the gathered-vocabulary sensitivity verdict where a record carries one (else the primary)."""
    vals = []
    for r in recs:
        if not is_defined(r):
            return "UNDEFINED"
        vals.append(r.get(key, r["v"]) if key != "v" else r["v"])
    if any(not isinstance(x, bool) for x in vals):
        return "UNDEFINED"
    return "kept" if any(vals) else "not_kept"


def reject_reason(r):
    """The claim gate's reject reason from its return `(accepted, result)`: result["reject_reason"] when the result is a dict, else None
    (a keep, or a gate that gave no dict). Recorded on the allowlist arm's records only (field `reason`; a keep carries none)."""
    if isinstance(r, tuple) and len(r) > 1 and isinstance(r[1], dict):
        return r[1].get("reject_reason")
    return None


_REASON_TAG = re.compile(r"^allowlist: ([^:'\"(\[]*)")


def reason_tag(reason):
    """The leading tag of an allowlist reject reason: the words after "allowlist: " up to the first ':', quote, '(' or '[' ("no shape",
    "not gathered", "entity span", "all-caps token", "non-ASCII character", ...); "other" for a reason not in that form."""
    m = _REASON_TAG.match(reason or "")
    return m.group(1).strip() if m and m.group(1).strip() else "other"


# ---------------------------------------------------------------------------------------------------------------------------------
# instrumentation of one chat object: record what the claim gate and the single-triple `_verify` were asked and answered
# ---------------------------------------------------------------------------------------------------------------------------------
class Trace:
    def __init__(self, chat):
        self.events = []
        self.chat = chat
        self._claim, self._single = chat._verify_claim_set, chat._verify
        chat._verify_claim_set, chat._verify = self.claim, self.single

    def claim(self, surface, gated):
        try:
            r = self._claim(surface, gated)
        except BaseException:
            self.events.append(("claim", "raised"))
            raise
        self.events.append(("claim", r[0] if isinstance(r, tuple) and r else r, reject_reason(r)))
        return r

    def single(self, surface, asserted, svo):
        try:
            r = self._single(surface, asserted, svo)
        except BaseException:
            self.events.append(("single", "raised"))
            raise
        self.events.append(("single", r))
        return r


@contextlib.contextmanager
def gathered_vocab(chat, g):
    """Temporarily give the chat the vocabulary of only this case's gathered triples, computed by the REAL `_refresh_facts`
    (the composer's `.kb` is swapped for the duration, then the primary vocabulary is put back by assignment)."""
    holder = chat.inner.composer
    saved_kb = holder.kb
    saved = {k: getattr(chat, k) for k in VOCAB_ATTRS}
    holder.kb = kb_of(g)
    try:
        chat._refresh_facts()
        yield
    finally:
        holder.kb = saved_kb
        for k, v in saved.items():
            setattr(chat, k, v)


def kb_of(triples):
    """The `.kb` shape the composers keep: [(fact_dict, hv)]."""
    return [({"agent": a, "action": v, "patient": p, "polarity": "AFFIRM"}, None) for a, v, p in triples]


def any_target(chat, surface, g):
    """The single-triple `ChatBrain._verify(surface, None, svo)` over every gathered fact that could be the target. `_verify` rebuilds
    `rsvo` from the words it extracted from the prose (reassigned to roles), so a target can match only if its three tokens are the
    extracted three tokens: other facts are False without a call. Returns (kept, extracted, accepted_targets)."""
    from research.runners._grounded_lang_integration_derisk import _extract_svo_from_prose
    ex = _extract_svo_from_prose(surface, chat.agents_set, chat.actions_set, chat.patients_set, chat.inflect)
    seen, cands = set(), []
    for t in g:
        t = tuple(t)
        if t not in seen and ex is not None and sorted(t) == sorted(ex):
            seen.add(t)
            cands.append(t)
    if not cands:
        cands = [tuple(g[0])]                      # extraction failed or matches no fact: the real `_verify` still decides (False)
    hits = [list(t) for t in cands if chat._verify(surface, None, t) is True]
    return bool(hits), ex, hits


def fallback_rec(chat, surface, g, src, why=None, alt_vocab=True, already=None):
    """The production single-triple verdict, labelled, with the gathered-vocabulary sensitivity beside it. `already` = a verdict the
    caller already got for target g[0] (the composer entry); only the other targets are then tried."""
    kept, ex, hits = any_target(chat, surface, g)
    if already is True:
        kept = True
    alt = None
    if alt_vocab:
        with gathered_vocab(chat, g):
            alt, _ex2, _h2 = any_target(chat, surface, g)
    return _rec(kept, src, why=why, extracted=ex, targets=hits or None, alt_v=alt)


def unbuildable_reason(chat, g):
    """Why `_build_claim_verifier` gave None: re-run the verifier's own constructor on the same arguments and quote its error."""
    try:
        from research.runners._moat_claim_entailment_derisk import (ClaimEntailmentVerifier, VERB_SYNONYMS, _build_inflection_map)
        gated = [list(f) for f in g]
        nouns = {t for f in gated for t in (f[0], f[2])}
        verbs = {f[1] for f in gated}
        ClaimEntailmentVerifier(chat.inner, gated, nouns, verbs, VERB_SYNONYMS, _build_inflection_map(sorted(verbs)))
        return "claim verifier unbuildable (constructor raised nothing on a re-run)"
    except AssertionError as e:
        return "claim verifier unbuildable: %s" % str(e)[:160]
    except Exception as e:  # noqa: BLE001
        return "claim verifier unbuildable: %s: %s" % (type(e).__name__, str(e)[:120])


# ---------------------------------------------------------------------------------------------------------------------------------
# the two entries
# ---------------------------------------------------------------------------------------------------------------------------------
def gate_entry(chat, arm, surface, g, alt_vocab=True):
    """`ChatBrain._verify_claim_set`. (None, None) is the production signal 'fall back to the single-triple `_verify`': for stock and
    B2h the record is the fallback's verdict (src='fallback'); for the allowlist arm it is UNDEFINED (the matcher is always buildable)."""
    try:
        acc, _res = chat._verify_claim_set(surface, g)
        if isinstance(acc, bool):
            return _rec(acc, "claim", reason=reject_reason((acc, _res)) if arm == STRICT_ARM else None)
        if acc is None and arm != STRICT_ARM:
            return fallback_rec(chat, surface, g, "fallback", why=unbuildable_reason(chat, g), alt_vocab=alt_vocab)
        return _undefined("claim gate returned %r" % (acc,))
    except Exception as e:  # noqa: BLE001  (recorded, never a pass)
        return _error(e)


def composer_entry(rc, chat, trace, arm, surface, g, alt_vocab=True):
    """`RichAnswerComposer._verify_rendered(surface, asserted=None, svo, gated)`. The Trace says whether the claim gate and/or the
    single-triple `_verify` decided, so a hidden None (the allowlist path's bool() coercion) cannot pass as False."""
    trace.events.clear()
    try:
        v = rc._verify_rendered(surface, None, tuple(g[0]), g)
        ev = list(trace.events)
        claim = [e[1] for e in ev if e[0] == "claim"]
        claim_why = [e[2] for e in ev if e[0] == "claim"]
        single = [e for e in ev if e[0] == "single"]
        if not isinstance(v, bool):
            return _undefined("composer verdict %r is not a bool" % (v,))
        if arm == STRICT_ARM:
            if single:
                return _undefined("the allowlist arm reached the single-triple fallback")
            if not claim:
                return _undefined("the allowlist arm did not consult the claim gate")
            if not isinstance(claim[0], bool):
                return _undefined("claim gate returned %r (the composer coerced it to %r)" % (claim[0], v))
            return _rec(v, "claim", reason=claim_why[0])
        if not single:
            if claim and isinstance(claim[0], bool):
                return _rec(v, "claim")
            return _undefined("composer verdict with no recorded gate verdict (claim=%r)" % (claim,))
        if claim and isinstance(claim[0], bool):
            return _undefined("claim gate answered %r and the composer still reached `_verify`" % (claim[0],))
        why = unbuildable_reason(chat, g) if claim else "one-fact turn: `_verify_rendered` goes straight to the single-triple re-parse"
        return fallback_rec(chat, surface, g, "fallback" if claim else "single", why=why, alt_vocab=alt_vocab, already=v)
    except Exception as e:  # noqa: BLE001
        return _error(e)


# ---------------------------------------------------------------------------------------------------------------------------------
# the reply entry: the composer's paragraph path (sequential and batched) with the build-6 joined check. Pure helpers first.
# ---------------------------------------------------------------------------------------------------------------------------------
def composer_terminate():
    """The composer's own per-candidate normalisation under the allowlist flag (`rich_answer_composer._terminate`, one rule with the
    TUI's `_moat_terminate`)."""
    from research.runners.rich_answer_composer import _terminate
    return _terminate


def is_reply_case(case):
    return isinstance(case, dict) and case.get("reply") is True


def reply_join(sentences, idx, terminate=None):
    """The text the composer emits for the 1-based selection `idx` of `sentences`, in that order: each sentence `terminate`d (flag
    ON; None = as rendered, flag OFF), joined with one U+0020."""
    f = terminate or (lambda s: s)
    return " ".join(f(sentences[i - 1]) for i in idx)


def validate_reply_case(case, terminate=None):
    """None if `case` is a well-formed reply case, else the reason (the reply entry is then UNDEFINED for it, in every arm). Needs:
    "sentences" a list of k non-blank strings, k == len(gathered), 2 <= k <= REPLY_MAX_K; every gathered fact three strings; no
    triple gathered twice (the stand-in answers by fact); `sentence` equal to the sentences joined as the composer joins them (flag
    OFF, or flag ON with `terminate`, default the composer's)."""
    terminate = terminate or composer_terminate()
    s, g = case.get("sentences"), case.get("gathered")
    if not isinstance(s, list) or not all(isinstance(x, str) and x.strip() for x in s):
        return "reply case needs 'sentences', a list of non-blank strings"
    if not isinstance(g, list) or not all(isinstance(f, (list, tuple)) and len(f) == 3 and all(isinstance(t, str) for t in f)
                                          for f in g):
        return "reply case needs 'gathered', a list of three-string triples"
    if len(s) != len(g):
        return "reply case has %d sentences for %d gathered triples (one sentence per triple, same order)" % (len(s), len(g))
    if not 2 <= len(s) <= REPLY_MAX_K:
        return "reply case needs 2 <= k <= %d sentences, has %d" % (REPLY_MAX_K, len(s))
    if len({tuple(f) for f in g}) != len(g):
        return "a triple is gathered twice: the stand-in renders by fact, so its sentences cannot be told apart"
    full = list(range(1, len(s) + 1))
    if case.get("sentence") not in (reply_join(s, full), reply_join(s, full, terminate)):
        return "the case's 'sentence' is not its 'sentences' joined as the composer joins them"
    return None


def map_reply(text, sentences, terminate=None, kept_idx=None):
    """Map one emitted paragraph to its prefix index. Returns {"status": "prefix", "j": j, "subset": [1..j]} (j = 0 for an empty
    paragraph), {"status": "subsequence", "j": None, "subset": [...]} for an in-order non-prefix selection, or {"status":
    "UNDEFINED", "j": None, "subset": None, "why": ...}. The text must EQUAL `reply_join(sentences, selection, terminate)` for an
    in-order selection; `kept_idx` (the composer's kept facts, 1-based) must give that same text, and decides between selections
    whose texts coincide (a repeated sentence); without it a coincidence is UNDEFINED (ambiguous)."""
    def undef(why):
        return {"status": "UNDEFINED", "j": None, "subset": None, "why": why}
    if not isinstance(text, str):
        return undef("emitted %r, not a string" % (text,))
    k = len(sentences)
    if kept_idx is not None and (list(kept_idx) != sorted(set(kept_idx)) or any(not 1 <= i <= k for i in kept_idx)):
        return undef("the composer's kept facts %r are not an in-order selection of the %d gathered facts" % (list(kept_idx), k))
    if text == "":
        if kept_idx:
            return undef("empty paragraph, but the composer kept facts %r" % (list(kept_idx),))
        return {"status": "prefix", "j": 0, "subset": []}
    if k > REPLY_MAX_K:
        return undef("k = %d > %d" % (k, REPLY_MAX_K))
    cands = [list(c) for n in range(1, k + 1) for c in combinations(range(1, k + 1), n) if reply_join(sentences, c, terminate) == text]
    if not cands:
        return undef("the text matches no in-order selection of the case's sentences (reordered, partial, extra text or another "
                     "mouth's text): %r" % (text[:160],))
    if kept_idx is not None:
        if list(kept_idx) not in cands:
            return undef("the composer's kept facts %r do not give the emitted text (it matches %r)" % (list(kept_idx), cands))
        sub = list(kept_idx)
    elif len(cands) > 1:
        return undef("ambiguous: the text matches the selections %r" % (cands,))
    else:
        sub = cands[0]
    if sub == list(range(1, len(sub) + 1)):
        return {"status": "prefix", "j": len(sub), "subset": sub}
    return {"status": "subsequence", "j": None, "subset": sub}


def subset_key(idx):
    """The subset_labels key of a 1-based selection: '1+3' (a single sentence: '2')."""
    return "+".join(str(i) for i in idx)


def outcome_key(rec):
    """'0'..'k' for a prefix, 'sub:1+3' for a non-prefix selection (so 'sub:2', sentence 2 alone, never reads as j = 2), None for an
    UNDEFINED record."""
    if rec.get("status") == "prefix":
        return str(rec["j"])
    if rec.get("status") == "subsequence":
        return "sub:" + subset_key(rec["subset"])
    return None


NO_LABELS = "no subset_labels (or prefix_labels) for this reply case"


def parse_selection(key, k):
    """The 1-based indices of a selection key ("1+3"), or None unless it is the canonical key of an in-order non-empty selection of k."""
    try:
        idx = [int(x) for x in str(key).split("+")]
    except ValueError:
        return None
    if idx != sorted(set(idx)) or not all(1 <= i <= k for i in idx) or subset_key(idx) != key:
        return None
    return idx


def check_prefix_labels(pl, k, own_label):
    """None if `pl` is a valid prefix-label list for a k-sentence reply case whose own (adjudicated) label is `own_label`, else why."""
    if pl is None:
        return "no prefix_labels for this reply case"
    if not isinstance(pl, list) or len(pl) != k:
        return "prefix_labels %r is not a list of %d labels (one per prefix, the full reply last)" % (pl, k)
    bad = [x for x in pl if x not in LABELS]
    if bad:
        return "prefix label %r is not one of %s" % (bad[0], "/".join(LABELS))
    if pl[-1] != own_label:
        return "the full reply's prefix label %r is not the case's own label %r" % (pl[-1], own_label)
    return None


def check_subset_labels(sl, k, own_label=None):
    """None if `sl` (None, or {"1+3": label}) labels only in-order non-empty selections of k sentences (prefixes and the full reply
    included) with U/B/F, and the full selection, if labelled, with `own_label` (when given); else why."""
    if sl is None:
        return None
    if not isinstance(sl, dict):
        return "subset_labels %r is not a dict" % (sl,)
    full = subset_key(range(1, k + 1))
    for key, lab in sl.items():
        if parse_selection(key, k) is None:
            return "subset_labels key %r is not the key of an in-order selection of %d sentences ('1', '1+3', ...)" % (key, k)
        if lab not in LABELS:
            return "subset label %r is not one of %s" % (lab, "/".join(LABELS))
        if key == full and own_label is not None and lab != own_label:
            return "the full reply's subset label %r is not the case's own label %r" % (lab, own_label)
    return None


def reply_label_map(pl, sl, k, own_label):
    """(labels, None) or (None, why): labels = {selection key: label} from `subset_labels` (every selection, prefixes and the full
    reply included) and, for compatibility, `prefix_labels` (prefix j -> key "1+..+j"); where both label one selection they must
    agree. The full selection is the case's own label. Neither family given: (None, NO_LABELS)."""
    if pl is None and sl is None:
        return None, NO_LABELS
    out = {}
    if pl is not None:
        why = check_prefix_labels(pl, k, own_label)
        if why:
            return None, why
        out = {subset_key(range(1, j + 1)): lab for j, lab in enumerate(pl, 1)}
    if sl is not None:
        why = check_subset_labels(sl, k, own_label)
        if why:
            return None, why
        for key, lab in sl.items():
            if key in out and out[key] != lab:
                return None, "prefix_labels and subset_labels disagree on selection %s (%r vs %r)" % (key, out[key], lab)
            out[key] = lab
    out.setdefault(subset_key(range(1, k + 1)), own_label)
    return out, None


def load_reply_labels(adj, cases, label):
    """The reply cases' adjudicated labels from an adjudication dict (`adjudication.json`, or {}):
    {case id: {"prefix": [..] or None, "subset": {..} or None, "why": None or the reason the case's labels are unusable}}."""
    adj = adj or {}
    pls, sls = adj.get("prefix_labels") or {}, adj.get("subset_labels") or {}
    out = {}
    for c in cases:
        if not is_reply_case(c):
            continue
        k = len(c.get("sentences") or [])
        pl, sl = pls.get(c["id"]), sls.get(c["id"])
        _m, why = reply_label_map(pl, sl, k, label[c["id"]])
        out[c["id"]] = {"prefix": None if why else pl, "subset": None if why else sl, "why": why}
    return out


class ReplyRenderer:
    """The mouth stand-in of the reply entry: its rendering of gathered fact i is sentences[i] (asserted None: a free-prose mouth),
    looked up BY FACT, never by call index. Every call is recorded (method, 1-based fact index); a fact outside the case raises and is
    recorded in `errors` (the record is then UNDEFINED even if the batched path swallowed the raise). `batch` adds
    render_svo_batch / render_svo_regen_batch (the batched path); a regenerate returns the same sentence (a deterministic mouth)."""

    name = "hunt reply stand-in (scripted per gathered fact)"

    def __init__(self, gathered, sentences, batch=False):
        self.table = {tuple(f): (i + 1, s) for i, (f, s) in enumerate(zip(gathered, sentences))}
        self.asked, self.errors = [], []
        if batch:
            self.render_svo_batch = lambda svos: ([self._one("render_svo_batch", *s) for s in svos], 0.0)
            self.render_svo_regen_batch = lambda svos: ([self._one("render_svo_regen_batch", *s) for s in svos], 0.0)

    def _one(self, method, a, v, p):
        hit = self.table.get((a, v, p))
        if hit is None:
            self.errors.append("%s asked for %r, not a gathered fact of the case" % (method, (a, v, p)))
            raise KeyError((a, v, p))
        self.asked.append((method, hit[0]))
        return hit[1], None

    def render_svo(self, a, v, p):
        return self._one("render_svo", a, v, p)

    def render_svo_regen(self, a, v, p):
        return self._one("render_svo_regen", a, v, p)


@contextlib.contextmanager
def env_overrides(over):
    """Set (str) or unset (None) each variable for the duration, then put every one back as it was."""
    saved = {k: os.environ.get(k) for k in over}
    try:
        for k, v in over.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def reply_one(rc, chat, trace, case, seed, path):
    """One (seed, path) emission of the composer's paragraph path for a reply case, under the arm's flags already in the
    environment. The record: text, kept_idx, asked (render calls), the `map_reply` outcome, and the gate calls the Trace saw."""
    from research.runners import rich_answer_composer as RAC
    g, sents = case["gathered"], case["sentences"]
    rec = {"seed": seed, "path": path}
    ren = ReplyRenderer(g, sents, batch=(path == "batched"))
    saved = chat.renderer
    trace.events.clear()
    rc._last_reply_rejects = None
    try:
        with env_overrides({**REPLY_ENV, "BRAIN_RICH_BATCH_RENDER": "1" if path == "batched" else None}):
            strict = RAC._moat_allowlist_enabled()
            chat.renderer = ren
            facts = [list(f) for f in g]
            para, kept, _dropped = rc.render_paragraph(facts)
    except Exception as e:  # noqa: BLE001  (recorded, never a pass)
        rec.update(_error(e), status="UNDEFINED", j=None, subset=None, asked=[i for _m, i in ren.asked])
        return rec
    finally:
        chat.renderer = saved
    ev = list(trace.events)
    claim = [e for e in ev if e[0] == "claim"]
    single = [e for e in ev if e[0] == "single"]
    pos = {id(f): i + 1 for i, f in enumerate(facts)}
    kept_idx = [pos.get(id(f)) for f in kept]
    batched_calls = [m for m, _i in ren.asked if m.endswith("_batch")]
    rec.update(text=para, kept_idx=kept_idx, asked=[i for _m, i in ren.asked], n_gate_calls=len(claim), n_single_calls=len(single))
    why = None
    if ren.errors:
        why = "renderer error: %s" % ren.errors[0]
    elif any(e[1] == "raised" for e in ev):
        why = "a traced check raised"
    elif None in kept_idx:
        why = "the composer kept a fact that is not one of the gathered fact objects"
    elif (path == "batched") != bool(batched_calls):
        why = "the %s path was not taken (render calls %r)" % (path, sorted({m for m, _i in ren.asked}))
    elif strict and single:
        why = "the allowlist arm reached the single-triple fallback"
    elif strict and any(not isinstance(e[1], bool) for e in claim):
        why = "claim gate returned %r under the allowlist (the composer coerces it)" % ([e[1] for e in claim if not isinstance(e[1], bool)][0],)
    elif not strict and any(not (isinstance(e[1], bool) or e[1] is None) for e in claim):
        why = "claim gate returned a non-bool, non-None verdict"
    elif not strict and any(not isinstance(e[1], bool) for e in single):
        why = "the single-triple `_verify` returned a non-bool verdict"
    if why:
        rec.update(status="UNDEFINED", j=None, subset=None, why=why)
    else:
        rec.update(map_reply(para, sents, terminate=RAC._terminate if strict else None, kept_idx=kept_idx))
    if strict:
        rec["reasons"] = [e[2] for e in claim if len(e) > 2 and e[2]]
        rec["cut"] = [x.get("reject_reason") for x in (rc._last_reply_rejects or [])]
    return rec


def reply_entry(composers, chats, traces, case, seeds, invalid=None):
    """The reply entry for one case in the current arm: one record per (seed, path), seeds outer. `invalid` (a reason): one UNDEFINED
    record instead of a run."""
    if invalid:
        return [{"seed": None, "path": None, "status": "UNDEFINED", "j": None, "subset": None, "v": None, "src": "invalid", "why": invalid}]
    return [reply_one(composers[sd], chats[sd], traces[sd], case, sd, path) for sd in seeds for path in REPLY_PATHS]


# ---------------------------------------------------------------------------------------------------------------------------------
# the run and its summary (pure functions of the rows: unit-tested without a brain)
# ---------------------------------------------------------------------------------------------------------------------------------
def run_rows(cases, label, seeds, chats, composers, traces, arms=None, alt_vocab=True, progress=None, reply_labels=None,
             entries=ENTRIES):
    """`reply_labels` = `load_reply_labels(...)` (missing: every reply case's labels are missing). `entries` = the gate/composer
    entries to run (the reply controls run none). A reply case also gets the reply entry; every other case records "n/a" there."""
    arms = ARMS if arms is None else arms
    reply_labels = reply_labels or {}
    rows = []
    t0 = time.time()
    for i, c in enumerate(cases):
        g = c["gathered"]
        row = {"id": c["id"], "class": c.get("class"), "label": label[c["id"]], "sentence": c["sentence"],
               "n_gathered": len(g), "gate": {}, "composer": {}, "reply": NA}
        invalid = None
        if is_reply_case(c):
            invalid = validate_reply_case(c)
            lab = reply_labels.get(c["id"]) or {"prefix": None, "subset": None, "why": NO_LABELS}
            row.update(reply={}, sentences=c.get("sentences"), k=len(c.get("sentences") or []), reply_invalid=invalid,
                       prefix_labels=lab["prefix"], subset_labels=lab["subset"], labels_why=lab["why"])
        for arm, env in arms.items():
            for k in FLAGS:
                os.environ.pop(k, None)
            os.environ.update(env)
            try:
                if "gate" in entries:
                    row["gate"][arm] = [gate_entry(chats[sd], arm, c["sentence"], g, alt_vocab) for sd in seeds]
                if "composer" in entries:
                    row["composer"][arm] = [composer_entry(composers[sd], chats[sd], traces[sd], arm, c["sentence"], g, alt_vocab)
                                            for sd in seeds]
                if row["reply"] != NA:
                    row["reply"][arm] = reply_entry(composers, chats, traces, c, seeds, invalid=invalid)
            finally:
                for k in FLAGS:
                    os.environ.pop(k, None)
        rows.append(row)
        if progress and (i + 1) % 10 == 0:
            progress("  %d/%d cases  %.0fs" % (i + 1, len(cases), time.time() - t0))
    return rows


def reply_case_outcome(row, arm):
    """A reply case's outcome in one arm: ("UNDEFINED", why) if any (seed, path) record is UNDEFINED, the case is invalid, its labels
    are missing/invalid (`reply_label_map`), or it emitted a selection (prefix or not) with no label; else ("defined", {"emitted":
    [keys], "labels": {label: [keys]}, "labels_by_path": {path: [labels]}})."""
    recs = row["reply"].get(arm) if isinstance(row.get("reply"), dict) else None
    if not recs:
        return "UNDEFINED", "no reply records for this arm"
    bad = [r for r in recs if r.get("status") not in ("prefix", "subsequence")]
    if bad:
        return "UNDEFINED", bad[0].get("why") or "UNDEFINED record"
    k = row.get("k") or 0
    labmap, why = reply_label_map(row.get("prefix_labels"), row.get("subset_labels"), k, row["label"])
    if why:
        return "UNDEFINED", row.get("labels_why") or why
    labels, by_path = {}, {}
    for r in recs:
        key = outcome_key(r)
        if r["status"] == "prefix" and r["j"] == 0:
            continue
        lab = labmap.get(subset_key(r["subset"]))
        if lab is None:
            return "UNDEFINED", "emitted the selection %s (key %s), which has no adjudicated label" % (key, subset_key(r["subset"]))
        labels.setdefault(lab, set()).add(key)
        by_path.setdefault(r.get("path"), set()).add(lab)
    return "defined", {"emitted": sorted({outcome_key(r) for r in recs}, key=_key_order),
                       "labels": {lab: sorted(v, key=_key_order) for lab, v in labels.items()},
                       "labels_by_path": {p: sorted(v) for p, v in by_path.items()}}


def _key_order(key):
    return (0, int(key), "") if key.isdigit() else (1, 0, key)


def summarise_reply(rows, arm):
    """The reply entry's summary for one arm (reply cases only; "n/a" rows are counted in n_na and nothing else): UNDEFINED cases with
    their reasons; kept U / B / F = the cases that emitted, at ANY seed through EITHER path, a text whose adjudicated label is U / B / F
    (with ids, and per path); the distribution of j over every defined (seed, path) record; per-case emitted sets; and, for the
    allowlist arm, the gate's reject reasons by leading tag (every traced check, and the joined-check cuts)."""
    rr = [r for r in rows if isinstance(r.get("reply"), dict)]
    out = {"n": len(rr), "n_na": sum(1 for r in rows if not isinstance(r.get("reply"), dict)),
           "n_by_case_label": dict(Counter(r["label"] for r in rr)), "UNDEFINED": 0, "undefined_ids": [], "undefined_why": {},
           "kept": {lab: {"n": 0, "ids": []} for lab in LABELS},
           "kept_by_path": {p: {lab: [] for lab in LABELS} for p in REPLY_PATHS},
           "j_distribution": {"all": Counter(), **{p: Counter() for p in REPLY_PATHS}},
           "case_emitted": {}, "full_reply_ids": [], "cut_ids": [], "nothing_emitted_ids": [], "subsequence_ids": [],
           "undefined_record_count": 0}
    for r in rr:
        recs = r["reply"].get(arm) or []
        for x in recs:
            key = outcome_key(x)
            if key is None:
                out["undefined_record_count"] += 1
                continue
            out["j_distribution"]["all"][key] += 1
            if x.get("path") in REPLY_PATHS:
                out["j_distribution"][x["path"]][key] += 1
        status, info = reply_case_outcome(r, arm)
        if status == "UNDEFINED":
            out["UNDEFINED"] += 1
            out["undefined_ids"].append(r["id"])
            out["undefined_why"][r["id"]] = info
            continue
        out["case_emitted"][r["id"]] = info["emitted"]
        for lab in info["labels"]:
            out["kept"][lab]["ids"].append(r["id"])
        for p, labs in info["labels_by_path"].items():
            for lab in labs:
                if p in out["kept_by_path"]:
                    out["kept_by_path"][p][lab].append(r["id"])
        em = info["emitted"]
        k = str(r.get("k"))
        if k in em:
            out["full_reply_ids"].append(r["id"])
        if any(e.isdigit() and 0 < int(e) < int(k) for e in em):
            out["cut_ids"].append(r["id"])
        if em == ["0"]:
            out["nothing_emitted_ids"].append(r["id"])
        if any(not e.isdigit() for e in em):
            out["subsequence_ids"].append(r["id"])
    for lab in LABELS:
        out["kept"][lab]["ids"].sort()
        out["kept"][lab]["n"] = len(out["kept"][lab]["ids"])
    for key in ("undefined_ids", "full_reply_ids", "cut_ids", "nothing_emitted_ids", "subsequence_ids"):
        out[key].sort()
    out["j_distribution"] = {p: dict(sorted(c.items(), key=lambda kv: _key_order(kv[0]))) for p, c in out["j_distribution"].items()}
    if arm == STRICT_ARM:
        # the leading tag with sentence numbers read as N ("sentences N and N both state"), so one rule is one tag
        tags = lambda xs: dict(sorted(Counter(re.sub(r"\d+", "N", reason_tag(x)) for x in xs if x).items(), key=lambda kv: (-kv[1], kv[0])))
        recs = [x for r in rr for x in (r["reply"].get(arm) or [])]
        out["reject_reason_tags"] = {"every_check": tags([y for x in recs for y in x.get("reasons") or []]),
                                     "joined_cut": tags([y for x in recs for y in x.get("cut") or []])}
    return out


def summarise(rows, arms=None):
    """Per arm, per entry, per label: kept / not_kept / UNDEFINED case counts (a case is UNDEFINED for an arm and entry if any seed's
    verdict is not a bool), the kept and UNDEFINED ids, and per arm a VALID flag (zero UNDEFINED over every entry and label).
    The reply entry adds `reply` per arm (`summarise_reply`); its UNDEFINED cases count in n_undefined_case_entries and VALID (with
    no reply case both are exactly as before)."""
    arms = list(ARMS if arms is None else arms)
    out = {}
    for arm in arms:
        a = {"entries": {}, "reasons": {}, "fallback_case_seeds": {}}
        n_undef_cases = 0
        for entry in ENTRIES:
            e = {}
            for lab in LABELS:
                rs = [r for r in rows if r["label"] == lab]
                st = {r["id"]: case_status(r[entry][arm]) for r in rs}
                alt = {r["id"]: case_status(r[entry][arm], key="alt_v") for r in rs}
                e[lab] = {"n": len(rs),
                          "kept": sum(1 for s in st.values() if s == "kept"),
                          "not_kept": sum(1 for s in st.values() if s == "not_kept"),
                          "UNDEFINED": sum(1 for s in st.values() if s == "UNDEFINED"),
                          "kept_ids": sorted(i for i, s in st.items() if s == "kept"),
                          "undefined_ids": sorted(i for i, s in st.items() if s == "UNDEFINED"),
                          "kept_ids_any_seed_before_undefined": sorted(
                              r["id"] for r in rs if st[r["id"]] == "UNDEFINED" and any(x.get("v") is True for x in r[entry][arm])),
                          "gathered_vocab_sensitivity": {"kept": sum(1 for s in alt.values() if s == "kept"),
                                                         "UNDEFINED": sum(1 for s in alt.values() if s == "UNDEFINED"),
                                                         "kept_ids": sorted(i for i, s in alt.items() if s == "kept")}}
                n_undef_cases += e[lab]["UNDEFINED"]
            a["entries"][entry] = e
            reasons, fb = Counter(), Counter()
            for r in rows:
                for x in r[entry][arm]:
                    if not is_defined(x):
                        reasons[(x or {}).get("why") or "non-bool verdict"] += 1
                    elif x.get("src") in ("fallback", "single"):
                        fb[x["src"]] += 1
            a["reasons"][entry] = dict(reasons.most_common())
            a["fallback_case_seeds"][entry] = dict(fb)
        a["reply"] = summarise_reply(rows, arm)
        n_undef_cases += a["reply"]["UNDEFINED"]
        a["n_undefined_case_entries"] = n_undef_cases
        a["VALID"] = n_undef_cases == 0
        if arm == STRICT_ARM:      # the gate's reject reasons, per case-seed record, counted by leading tag (instrument only)
            a["reject_reason_tags"] = {entry: dict(sorted(Counter(reason_tag(x["reason"]) for r in rows for x in r[entry][arm]
                                                                  if isinstance(x, dict) and x.get("reason")).items(),
                                                          key=lambda kv: (-kv[1], kv[0])))
                                       for entry in ENTRIES}
        out[arm] = a
    return out


def exit_code(summary, equiv_ok=True, controls_ok=True, reply_controls_ok=True):
    """Nonzero if the allowlist arm has any UNDEFINED case (bar 3 cannot be read); then if construction equivalence failed; then if a
    control failed (a column that cannot keep a supported sentence, or keeps a reversed one, is dead; `reply_controls_ok` = the
    allowlist arm's reply controls); then if a stock/B2h comparison column is not VALID (void)."""
    if STRICT_ARM in summary and not summary[STRICT_ARM]["VALID"]:
        return EXIT_ALLOWLIST_UNDEFINED
    if not equiv_ok:
        return EXIT_EQUIV_FAILED
    if not controls_ok or not reply_controls_ok:
        return EXIT_CONTROL_FAILED
    if any(not summary[a]["VALID"] for a in summary if a != STRICT_ARM):
        return EXIT_COMPARISON_VOID
    return EXIT_OK


def load_cases(cases_dir):
    cases = json.load(open(os.path.join(cases_dir, "cases.json")))
    relabel = {}
    p = os.path.join(cases_dir, "adjudication.json")
    if os.path.exists(p):
        relabel = json.load(open(p)).get("relabel_to_B", {}) or {}
    label = {c["id"]: ("B" if c["id"] in relabel else c["expect"]) for c in cases}
    return cases, label


def load_adjudication(cases_dir):
    """adjudication.json as a dict ({} when the round has none)."""
    p = os.path.join(cases_dir, "adjudication.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ---------------------------------------------------------------------------------------------------------------------------------
# controls: an arm that can only ever say "not kept" would also read "kept 0". Three sentences over the store's own single-word triples
# (no multi-word slug, so stock's slug-blind clause parser can read them): two supported ones every arm and entry MUST keep, and one
# role-reversed one every arm and entry MUST reject. Run on the same objects AFTER the hunt rows.
# ---------------------------------------------------------------------------------------------------------------------------------
CONTROL_PAIR = [["aug", "follows", "jul"], ["jul", "follows", "june"]]
CONTROL_CASES = [
    {"id": "P1", "gathered": CONTROL_PAIR, "sentence": "Aug follows Jul.", "expect": "P", "class": "control-supported-multifact"},
    {"id": "P2", "gathered": CONTROL_PAIR[:1], "sentence": "Aug follows Jul.", "expect": "P", "class": "control-supported-onefact"},
    {"id": "N1", "gathered": CONTROL_PAIR, "sentence": "Jul follows Aug.", "expect": "N", "class": "control-role-reversed"},
]


def control_report(rows, arms=None):
    """Per arm and entry: every P control kept at every seed and every N control not kept at every seed (a non-bool verdict fails it)."""
    arms = list(ARMS if arms is None else arms)
    ok = {}
    for arm in arms:
        ok[arm] = {}
        for entry in ENTRIES:
            good = True
            for r in rows:
                recs = r[entry][arm]
                if not all(is_defined(x) for x in recs):
                    good = False
                elif r["label"] == "P":
                    good = good and all(x["v"] is True for x in recs)
                else:
                    good = good and all(x["v"] is False for x in recs)
            ok[arm][entry] = good
    return {"cases": [{"id": r["id"], "sentence": r["sentence"], "label": r["label"], "n_gathered": r["n_gathered"]} for r in rows],
            "ok": ok, "ALL_OK": all(v for a in ok.values() for v in a.values())}


# reply controls (the reply entry only), written over the same two store triples, never case text. expect_reply: "full" = j = k at
# every seed on both paths; "first" = j = 1; "not_first" = j = 0 or a selection without sentence 1 (an UNDEFINED record fails).
def _reply_control(cid, sentences, expect, klass):
    return {"id": cid, "gathered": CONTROL_PAIR, "sentences": sentences, "sentence": " ".join(sentences), "reply": True,
            "expect": "P" if expect == "full" else "N", "expect_reply": expect, "class": klass}


REPLY_CONTROL_CASES = [
    _reply_control("RP1", ["Aug follows Jul.", "Jul follows June."], "full", "control-reply-faithful"),
    _reply_control("RN1", ["Aug follows Jul.", "June follows Jul."], "first", "control-reply-second-role-reversed"),
    _reply_control("RN0", ["Jul follows Aug.", "Jul follows June."], "not_first", "control-reply-first-role-reversed"),
    _reply_control("RNJ", ["Aug follows Jul.", "Aug follows Jul."], "first", "control-reply-joined-repeat"),
]


def reply_control_ok(recs, expect, k):
    """Every (seed, path) record is defined and meets the control's expectation (see REPLY_CONTROL_CASES)."""
    if not recs or any(r.get("status") not in ("prefix", "subsequence") for r in recs):
        return False
    if expect == "full":
        return all(r["status"] == "prefix" and r["j"] == k for r in recs)
    if expect == "first":
        return all(r["status"] == "prefix" and r["j"] == 1 for r in recs)
    if expect == "not_first":
        return all((r["status"] == "prefix" and r["j"] == 0) or (r["status"] == "subsequence" and 1 not in r["subset"]) for r in recs)
    raise ValueError("unknown reply-control expectation %r" % (expect,))


def reply_control_report(rows, cases, arms=None):
    """Per arm and reply control: the emitted outcomes over every seed and path and whether the control holds. STRICT_OK reads the
    allowlist arm only (it sets the exit code); the stock / B2h outcomes are reported, never failed on."""
    arms = list(ARMS if arms is None else arms)
    expect = {c["id"]: c["expect_reply"] for c in cases}
    ok, outcomes = {}, {}
    for arm in arms:
        ok[arm], outcomes[arm] = {}, {}
        for r in rows:
            recs = r["reply"].get(arm) if isinstance(r.get("reply"), dict) else None
            ok[arm][r["id"]] = reply_control_ok(recs, expect[r["id"]], r.get("k"))
            outcomes[arm][r["id"]] = sorted({outcome_key(x) or "UNDEFINED" for x in recs or []})
    strict = [v for v in ok.get(STRICT_ARM, {}).values()]
    return {"cases": [{"id": c["id"], "sentences": c["sentences"], "expect_reply": c["expect_reply"], "class": c["class"]}
                      for c in cases],
            "ok": ok, "outcomes": outcomes, "arm_ok": {a: all(v.values()) for a, v in ok.items()},
            "STRICT_OK": (all(strict) if STRICT_ARM in ok else True)}


# ---------------------------------------------------------------------------------------------------------------------------------
# construction (real constructors; the only stand-in is the composer that holds the store's facts)
# ---------------------------------------------------------------------------------------------------------------------------------
class KBHolder:
    """Stand-in for the fact store: what `ChatBrain._refresh_facts` reads from a composer is only `.kb` ([(fact_dict, hv)]) and
    `hasattr(_measure_block_readout)` (absent here). Teaching 78,857 facts into an RFPhasorComposer would not fit this job;
    `check_equivalence` below proves the stand-in changes no verdict."""

    def __init__(self, kb):
        self.kb = kb


def build_objects(seed, kb):
    """BrainConversationalAgent (real; its parser is BridgeParser(seed)), ChatBrain (real __init__, so the vocabulary sets are the real
    `_refresh_facts` of the composer's kb) and RichAnswerComposer (real __init__). `kb` is shared by the seeds, never mutated."""
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    from research.runners.rich_answer_composer import RichAnswerComposer
    agent = BrainConversationalAgent(seed=seed, composer=KBHolder(kb))
    chat = ChatBrain(agent)
    return chat, RichAnswerComposer(chat)


def build_real_objects(seed, triples):
    """The fully constructed comparison object: the default rf composer really TAUGHT the triples through `hear` (the production
    teaching path), then the same ChatBrain / RichAnswerComposer constructors."""
    from research.runners.brain_chat_tui import ChatBrain
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    from research.runners.rich_answer_composer import RichAnswerComposer
    toks = sorted({t for f in triples for t in f})
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in toks})
    for a, v, p in triples:
        agent.hear("%s %s %s" % (a, v, p))
    chat = ChatBrain(agent)
    return chat, RichAnswerComposer(chat)


def pick_equivalence_sample(cases, label, n):
    """A spread of cases: a mutual-pair (unbuildable) one and a buildable one, a one-fact one, and each label."""
    def collides(g):
        s = {tuple(f) for f in g}
        return any((p, v, a) in s and (p, v, a) != (a, v, p) for a, v, p in s)
    want = [is_reply_case,               # first, so a round with reply cases always checks one (matches nothing in rounds 3-7)
            lambda c: collides(c["gathered"]) and label[c["id"]] == "U", lambda c: (not collides(c["gathered"])) and len(c["gathered"]) > 1
            and label[c["id"]] == "U", lambda c: len(c["gathered"]) == 1, lambda c: label[c["id"]] == "F",
            lambda c: label[c["id"]] == "B", lambda c: collides(c["gathered"]) and label[c["id"]] == "F"]
    out, used = [], set()
    for w in want:
        for c in cases:
            if c["id"] not in used and w(c):
                out.append(c)
                used.add(c["id"])
                break
    for c in cases:
        if len(out) >= n:
            break
        if c["id"] not in used:
            out.append(c)
            used.add(c["id"])
    return out[:n]


def check_equivalence(sample, label, seed):
    """Stand-in kb vs really-taught rf composer, same gathered facts, same parser seed: the vocabulary attributes and every verdict
    (3 arms x 2 entries, and for a reply case the reply entry's outcome and text on both paths) must agree."""
    res = {"seed": seed, "cases": [], "ok": True}
    for c in sample:
        g = [tuple(f) for f in c["gathered"]]
        ch_s, rc_s = build_objects(seed, kb_of(g))
        ch_r, rc_r = build_real_objects(seed, g)
        same_attrs = {k: (getattr(ch_s, k) == getattr(ch_r, k)) for k in VOCAB_ATTRS}
        tr_s, tr_r = Trace(ch_s), Trace(ch_r)
        diffs = []
        for arm, env in ARMS.items():
            for k in FLAGS:
                os.environ.pop(k, None)
            os.environ.update(env)
            try:
                for entry, fn in (("gate", lambda ch, rc, tr: gate_entry(ch, arm, c["sentence"], c["gathered"], False)),
                                  ("composer", lambda ch, rc, tr: composer_entry(rc, ch, tr, arm, c["sentence"], c["gathered"], False))):
                    a, b = fn(ch_s, rc_s, tr_s), fn(ch_r, rc_r, tr_r)
                    if (a.get("v"), a.get("src")) != (b.get("v"), b.get("src")):
                        diffs.append({"arm": arm, "entry": entry, "stand_in": [a.get("v"), a.get("src")], "real": [b.get("v"), b.get("src")]})
                if is_reply_case(c) and not validate_reply_case(c):
                    for path in REPLY_PATHS:
                        a, b = reply_one(rc_s, ch_s, tr_s, c, seed, path), reply_one(rc_r, ch_r, tr_r, c, seed, path)
                        pick = lambda x: [x.get("status"), x.get("j"), x.get("subset"), x.get("text")]
                        if pick(a) != pick(b):
                            diffs.append({"arm": arm, "entry": "reply", "path": path, "stand_in": pick(a), "real": pick(b)})
            finally:
                for k in FLAGS:
                    os.environ.pop(k, None)
        ok = all(same_attrs.values()) and not diffs
        res["ok"] = res["ok"] and ok
        res["cases"].append({"id": c["id"], "label": label[c["id"]], "n_gathered": len(g), "vocab_attrs_equal": same_attrs,
                             "verdict_differences": diffs, "ok": ok})
    return res


def backend_fields():
    from sim.backend import get_backend
    xp, name = get_backend()
    return {"sim_backend": os.environ.get("SIM_BACKEND"), "backend": name, "xp": getattr(xp, "__name__", str(xp)),
            "device": "cpu (numpy)" if name == "numpy" else "gpu (%s)" % name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def main(argv=None, objects=None, equivalence=None, control_cases=None, reply_control_cases=None):
    """`objects(seed, kb) -> (chat, composer)`, `equivalence(...)`, `control_cases` and `reply_control_cases` are injectable so a unit
    test needs no brain. `reply_control_cases` defaults to REPLY_CONTROL_CASES when `control_cases` is not injected, else to none
    (an injected control set replaces both)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {"--seeds": None, "--limit": None, "--equiv-n": "6", "--store": DEFAULT_STORE}
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    if len(pos) != 2:
        print(__doc__)
        return 64
    cases_dir, out = pos
    seeds = tuple(int(s) for s in opts["--seeds"].split(",")) if opts["--seeds"] else SEEDS
    cases, label = load_cases(cases_dir)
    reply_labels = load_reply_labels(load_adjudication(cases_dir), cases, label)
    if opts["--limit"]:
        cases = cases[:int(opts["--limit"])]
    t0 = time.time()
    bk = backend_fields()
    if objects is None and bk["backend"] != "numpy":
        raise RuntimeError("the hunt runs on the CPU backend; got %r" % (bk,))
    if objects is None:
        store = json.load(open(opts["--store"]))
        triples = [(x["fact"]["agent"], x["fact"]["action"], x["fact"]["patient"]) for x in store]
        objects = build_objects
        triples_kb = kb_of(triples)
        equivalence = check_equivalence if equivalence is None else equivalence
    else:
        triples, triples_kb = [], []
    chats, composers, traces = {}, {}, {}
    for sd in seeds:
        chats[sd], composers[sd] = objects(sd, triples_kb)
        traces[sd] = Trace(chats[sd])
    print("built %d chat objects in %.0fs (backend %s)" % (len(seeds), time.time() - t0, bk["backend"]), flush=True)
    equiv = None
    if equivalence is not None and int(opts["--equiv-n"]) > 0:
        equiv = equivalence(pick_equivalence_sample(cases, label, int(opts["--equiv-n"])), label, 42)
        print("construction equivalence: %s (%d cases)" % ("OK" if equiv["ok"] else "FAILED", len(equiv["cases"])), flush=True)
    rows = run_rows(cases, label, seeds, chats, composers, traces,
                    alt_vocab=hasattr(chats[seeds[0]], "_refresh_facts"), progress=lambda s: print(s, flush=True),
                    reply_labels=reply_labels)
    summary = summarise(rows)
    if reply_control_cases is None:
        reply_control_cases = REPLY_CONTROL_CASES if control_cases is None else []
    control_cases = CONTROL_CASES if control_cases is None else control_cases
    ctl_rows = run_rows(control_cases, {c["id"]: c["expect"] for c in control_cases}, seeds, chats, composers, traces,
                        alt_vocab=hasattr(chats[seeds[0]], "_refresh_facts"))
    controls = control_report(ctl_rows)
    print("controls: %s %s" % ("OK" if controls["ALL_OK"] else "FAILED", controls["ok"]), flush=True)
    rctl_rows = run_rows(reply_control_cases, {c["id"]: c["expect"] for c in reply_control_cases}, seeds, chats, composers, traces,
                         entries=())
    reply_controls = reply_control_report(rctl_rows, reply_control_cases)
    print("reply controls (allowlist arm): %s %s" % ("OK" if reply_controls["STRICT_OK"] else "FAILED", reply_controls["outcomes"]),
          flush=True)
    code = exit_code(summary, equiv_ok=(equiv is None or equiv["ok"]), controls_ok=controls["ALL_OK"],
                     reply_controls_ok=reply_controls["STRICT_OK"])
    doc = {"_doc": __doc__, "runner": "research/findings/raw/_moat_allowlist_hunt4/run_hunt.py", "argv": sys.argv, "cases_dir": cases_dir,
           "host": socket.gethostname(), "seeds": list(seeds),
           "seeds_note": "parser seeds of one deterministic build: one replicate per case, not draws", **bk,
           "elapsed_seconds": round(time.time() - t0, 1), "exit_code": code,
           "construction": {"real_constructors": "BrainConversationalAgent(seed, composer=KBHolder), ChatBrain(agent), RichAnswerComposer(chat)",
                            "stand_in": "KBHolder holds the store's triples as the composer's .kb (the only thing _refresh_facts reads)",
                            "store": opts["--store"] if triples else None, "store_triples": len(triples),
                            "equivalence": equiv,
                            "asserted_passed_to_verify": "None (a free-prose mouth): `_verify` re-reads the prose with `_extract_svo_from_prose`", "target_policy": TARGET_POLICY,
                            "vocab_note": VOCAB_NOTE, "parser_note": PARSER_NOTE},
           "validity_rule": "an arm that records an error or a non-boolean verdict on a case is UNDEFINED for that case (never 'kept 0'); "
                            "VALID = zero UNDEFINED over every entry and label",
           "reply_entry": {"call": "RichAnswerComposer.render_paragraph(gathered) with chat.renderer = ReplyRenderer (fact i -> sentences[i], "
                                   "asserted None), paths %s, env %r (+ BRAIN_RICH_BATCH_RENDER=1 on the batched path)" % (list(REPLY_PATHS), REPLY_ENV),
                           "n_reply_cases": sum(1 for c in cases if is_reply_case(c)),
                           "prefix_rule": "j = k full reply, 1 <= j < k the first j sentences, 0 nothing emitted; exact match against the "
                                          "composer's own join (flag ON: _terminate each, one U+0020; flag OFF: one U+0020)",
                           "label_rule": "adjudication subset_labels[id][selection key] for every emitted selection, prefixes included "
                                         "(prefix_labels[id][j-1] still read; the two must agree); missing -> UNDEFINED"},
           "summary": summary, "controls": {**controls, "rows": ctl_rows},
           "reply_controls": {**reply_controls, "rows": rctl_rows}, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(doc, open(out, "w"), indent=1)
    research.runners.declare_output(out)
    for arm, s in summary.items():
        print("%-9s VALID=%s" % (arm, s["VALID"]))
        for entry in ENTRIES:
            print("   %-8s %s" % (entry, {lab: {k: s["entries"][entry][lab][k] for k in ("n", "kept", "not_kept", "UNDEFINED")}
                                           for lab in LABELS}))
        rp = s["reply"]
        if rp["n"]:
            print("   %-8s n=%d UNDEFINED=%d kept %s j=%s" % ("reply", rp["n"], rp["UNDEFINED"], {lab: rp["kept"][lab]["n"] for lab in LABELS},
                                                         rp["j_distribution"]["all"]))
    print("exit code", code, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
