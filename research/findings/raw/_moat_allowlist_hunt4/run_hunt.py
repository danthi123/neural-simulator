"""General leak-hunt runner for the moat claim checkers (prereg 2026-10-20-moat-allowlist-claim-checker, bar 3, AMENDMENT 2).

Runs every case of a hunt through the production claim gate (`ChatBrain._verify_claim_set`) and the composer entry
(`RichAnswerComposer._verify_rendered`) for three arms: stock (all moat flags off), b2h (`BRAIN_MOAT_SLUG_NORM=1`) and allowlist
(`BRAIN_MOAT_ALLOWLIST=1`), at parser seeds 42, 43, 44, 100, 101, 102 and 1 (one deterministic build per seed: one replicate per
case, not draws).

usage (cwd = worktree): python -m research.findings.raw._moat_allowlist_hunt4.run_hunt <cases_dir> <out.json>
  cases_dir holds cases.json ([{id, gathered, sentence, expect, class, ...}]) and, optionally, adjudication.json (relabel_to_B).
  options: --seeds 42,43,..  --limit N (first N cases, smoke only)  --equiv-n N (construction-equivalence sample, default 6)
           --store PATH (facts.json whose vocabulary the brain holds)
exit code: 0 ok; 2 the allowlist arm has an UNDEFINED case (bar 3 cannot be read); 4 the construction-equivalence check failed;
           5 a control failed (below); 3 only the stock or B2h comparison column is not VALID (void comparison columns).

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

SEEDS = (42, 43, 44, 100, 101, 102, 1)
ARMS = {"stock": {}, "b2h": {"BRAIN_MOAT_SLUG_NORM": "1"}, "allowlist": {"BRAIN_MOAT_ALLOWLIST": "1"}}
FLAGS = ("BRAIN_MOAT_SLUG_NORM", "BRAIN_MOAT_ALLOWLIST", "BRAIN_CLAIM_MOAT")
ENTRIES = ("gate", "composer")
LABELS = ("U", "B", "F")
STRICT_ARM = "allowlist"
DEFAULT_STORE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k/facts.json"
VOCAB_ATTRS = ("stored_facts", "agents_set", "actions_set", "patients_set", "inflect")
EXIT_OK, EXIT_ALLOWLIST_UNDEFINED, EXIT_COMPARISON_VOID, EXIT_EQUIV_FAILED, EXIT_CONTROL_FAILED = 0, 2, 3, 4, 5

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
# the run and its summary (pure functions of the rows: unit-tested without a brain)
# ---------------------------------------------------------------------------------------------------------------------------------
def run_rows(cases, label, seeds, chats, composers, traces, arms=None, alt_vocab=True, progress=None):
    arms = ARMS if arms is None else arms
    rows = []
    t0 = time.time()
    for i, c in enumerate(cases):
        g = c["gathered"]
        row = {"id": c["id"], "class": c.get("class"), "label": label[c["id"]], "sentence": c["sentence"],
               "n_gathered": len(g), "gate": {}, "composer": {}}
        for arm, env in arms.items():
            for k in FLAGS:
                os.environ.pop(k, None)
            os.environ.update(env)
            try:
                row["gate"][arm] = [gate_entry(chats[sd], arm, c["sentence"], g, alt_vocab) for sd in seeds]
                row["composer"][arm] = [composer_entry(composers[sd], chats[sd], traces[sd], arm, c["sentence"], g, alt_vocab)
                                        for sd in seeds]
            finally:
                for k in FLAGS:
                    os.environ.pop(k, None)
        rows.append(row)
        if progress and (i + 1) % 10 == 0:
            progress("  %d/%d cases  %.0fs" % (i + 1, len(cases), time.time() - t0))
    return rows


def summarise(rows, arms=None):
    """Per arm, per entry, per label: kept / not_kept / UNDEFINED case counts (a case is UNDEFINED for an arm and entry if any seed's
    verdict is not a bool), the kept and UNDEFINED ids, and per arm a VALID flag (zero UNDEFINED over every entry and label)."""
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
        a["n_undefined_case_entries"] = n_undef_cases
        a["VALID"] = n_undef_cases == 0
        if arm == STRICT_ARM:      # the gate's reject reasons, per case-seed record, counted by leading tag (instrument only)
            a["reject_reason_tags"] = {entry: dict(sorted(Counter(reason_tag(x["reason"]) for r in rows for x in r[entry][arm]
                                                                  if isinstance(x, dict) and x.get("reason")).items(),
                                                          key=lambda kv: (-kv[1], kv[0])))
                                       for entry in ENTRIES}
        out[arm] = a
    return out


def exit_code(summary, equiv_ok=True, controls_ok=True):
    """Nonzero if the allowlist arm has any UNDEFINED case (bar 3 cannot be read); then if construction equivalence failed; then if a
    control failed (a column that cannot keep a supported sentence, or keeps a reversed one, is dead); then if a stock/B2h comparison
    column is not VALID (void)."""
    if STRICT_ARM in summary and not summary[STRICT_ARM]["VALID"]:
        return EXIT_ALLOWLIST_UNDEFINED
    if not equiv_ok:
        return EXIT_EQUIV_FAILED
    if not controls_ok:
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
    want = [lambda c: collides(c["gathered"]) and label[c["id"]] == "U", lambda c: (not collides(c["gathered"])) and len(c["gathered"]) > 1
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
    (3 arms x 2 entries) must agree."""
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


def main(argv=None, objects=None, equivalence=None, control_cases=None):
    """`objects(seed, kb) -> (chat, composer)`, `equivalence(...)` and `control_cases` are injectable so a unit test needs no brain."""
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
                    alt_vocab=hasattr(chats[seeds[0]], "_refresh_facts"), progress=lambda s: print(s, flush=True))
    summary = summarise(rows)
    control_cases = CONTROL_CASES if control_cases is None else control_cases
    ctl_rows = run_rows(control_cases, {c["id"]: c["expect"] for c in control_cases}, seeds, chats, composers, traces,
                        alt_vocab=hasattr(chats[seeds[0]], "_refresh_facts"))
    controls = control_report(ctl_rows)
    print("controls: %s %s" % ("OK" if controls["ALL_OK"] else "FAILED", controls["ok"]), flush=True)
    code = exit_code(summary, equiv_ok=(equiv is None or equiv["ok"]), controls_ok=controls["ALL_OK"])
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
           "summary": summary, "controls": {**controls, "rows": ctl_rows}, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(doc, open(out, "w"), indent=1)
    research.runners.declare_output(out)
    for arm, s in summary.items():
        print("%-9s VALID=%s" % (arm, s["VALID"]))
        for entry in ENTRIES:
            print("   %-8s %s" % (entry, {lab: {k: s["entries"][entry][lab][k] for k in ("n", "kept", "not_kept", "UNDEFINED")}
                                           for lab in LABELS}))
    print("exit code", code, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
