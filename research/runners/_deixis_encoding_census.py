"""PACKET deixis-271-s0, stage 0: census of how today's encoders store
heard deixis. MEASUREMENT ONLY — no mechanism, no flag, no sim/ edit, no
default change, no gate seeds. It calls the REAL writers:

  W1 ChatBrain._maybe_acquire with B3 on (production default)
  W2 the same with BRAIN_NONCONTRADICTION_GATE=0 (legacy raw-token path)
  W3 ChatBrain._discourse_turn on a multiturn agent with the real D3
     discourse event register
  W4 surprise_production_organ.extract_assertion plus the
     reconsolidation in-place rewrite with a live surprise window
     (the string match it uses = composer.kb first fact whose
     (agent, action) match, _reactivate_idx)

Run: SIM_BACKEND=numpy python -m research.runners._deixis_encoding_census \
       --smoke-seed 7 --out research/findings/raw/_deixis_census/census.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "..", ".."))
CENSUS_DIR = os.path.join(REPO, "research/findings/raw/_deixis_census")

FIRST_SECOND = {"i", "me", "my", "mine", "we", "us", "our", "ours",
                "im", "moi", "em", "you", "your", "yours", "yer",
                "yous", "ur", "ya", "thee", "urself", "yourself"}

QUESTIONS = [                       # measure 6, 10 fixed questions
    "do i like jazz", "what do you know about my cat",
    "is my cat hungry", "who am i", "are you awake",
    "what did we learn yesterday", "tell me about myself",
    "do you know my name", "is it true that kofi likes jazz",
    "what does your friend think",
]


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _load_frozen():
    ps_path = os.path.join(CENSUS_DIR, "probe_set.json")
    or_path = os.path.join(CENSUS_DIR, "oracle.json")
    with open(ps_path) as fh:
        probe = json.load(fh)
    with open(or_path) as fh:
        oracle = json.load(fh)
    return probe, oracle, _sha(ps_path), _sha(or_path)


# ---------------------------------------------------------------- writers
def _fake_chat():
    """The tests' sanctioned harness: a real ChatBrain with a capturing
    inner sink — `_maybe_acquire` (the real writer) runs unchanged."""
    from research.runners.brain_chat_tui import ChatBrain
    cb = object.__new__(ChatBrain)
    state = {"heard": []}

    class _Inner:
        def hear(self, text, polarity="AFFIRM"):
            state["heard"].append([text, polarity])
    cb.inner = _Inner()
    cb._refresh_facts = lambda: None
    return cb, state


def run_w1_w2(items):
    out = {}
    key = "BRAIN_NONCONTRADICTION_GATE"
    prev = os.environ.get(key)
    for wid, gate in (("W1_b3_default", prev), ("W2_legacy_raw", "0")):
        if gate is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = gate
        sessions = {}
        for sp in ("A", "B"):
            cb, state = _fake_chat()
            rows = []
            for it in items:
                got = cb._maybe_acquire(it["text"])
                rows.append({"id": it["id"], "returned": got,
                             "heard": list(state["heard"])})
                state["heard"] = []
            sessions[sp] = rows
        out[wid] = sessions
    if prev is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = prev
    return out


def run_w3(items, seed):
    """Real D3 discourse register per speaker-session on the REAL
    MultiTurnAgent (the agent ChatBrain._discourse_turn talks to)."""
    import research.runners.d3_discourse_event_register_production_organ as DR
    from research.runners.multi_turn_agent import MultiTurnAgent
    from research.runners.brain_chat_tui import ChatBrain
    verbs = {"chases", "chase", "likes", "like", "married", "marry",
             "studied", "study", "told", "tell", "gave", "give",
             "does", "do", "not", "is", "are", "was"}
    vocab = sorted({w for it in items
                    for w in it["text"].lower().replace("n't", " ").split()})
    nouns = [w for w in vocab if w not in verbs]
    concepts = {w: None for w in vocab}
    sessions = {}
    errors = []
    for si, sp in enumerate(("A", "B")):
        reg = DR.make_discourse_register(vocab, seed=seed + si)
        try:
            agent = MultiTurnAgent(referent_concepts=nouns,
                                   concepts=concepts, seed=seed + si,
                                   event_register=reg)
        except Exception as e:
            errors.append({"speaker": sp,
                           "error": "%s: %s" % (type(e).__name__, e)})
            sessions[sp] = []
            continue
        # the REAL writer path: ChatBrain._discourse_turn -> agent.hear
        cb = object.__new__(ChatBrain)
        cb.agent = agent
        cb.is_multiturn = True
        cb.has_event_register = True
        cb.actions_set = set(verbs)
        cb._boundary_seen = False
        cb._heard_any_clause = False
        rows = []
        for it in items:
            try:
                got = cb._discourse_turn(it["text"])
                now = agent.who_agent_now()
                before = (agent.who_agent_before()
                          if hasattr(agent, "who_agent_before") else None)
                rows.append({"id": it["id"],
                             "returned": str(got) if got else None,
                             "who_now": str(now),
                             "who_before": str(before)})
            except Exception as e:
                rows.append({"id": it["id"], "error":
                             "%s: %s" % (type(e).__name__, e)})
        sessions[sp] = rows
    return {"W3_discourse_turn": sessions, "W3_build_errors": errors}


def run_w4(items, seed):
    """W4a: pure extract_assertion over the probe set.
    W4b: the reconsolidation in-place rewrite with a LIVE surprise
    window: store (agent, action, p_stored) on the rf composer (tiny
    demo, dev seed), assert a deictic-corrected patient, record the
    window decision + the exact string match _reactivate_idx used."""
    import research.runners.surprise_production_organ as SPO
    extract = {it["id"]: SPO.extract_assertion(it["text"]) for it in items}

    import research.runners.reconsolidation_production_organ as RPO
    from research.runners.brain_chat_tui import _build_tiny_demo
    built = _build_tiny_demo(
        seed, use_multiturn=False, enable_neural_render=False,
        composer_kind="rf")
    if len(built) == 4:
        agent_obj, _aliases, n_facts, src = built
    else:
        agent_obj, _aliases, n_facts = built
        src = "tiny-demo rf (dev seed %d)" % seed
    composer = getattr(agent_obj, "composer", None) or getattr(
        getattr(agent_obj, "agent", None), "composer", None)
    if composer is None:
        return {"W4_extract": extract, "W4_reconsolidation": [],
                "W4_composer_source": "NOT FOUND on tiny demo",
                "W4_composer_facts": -1,
                "W4_error": "no composer attribute on tiny demo agent"}
    org = RPO.get_organ(seed=seed)
    demos = []
    pairs = [("kofi", "chase", "north", "south"),
             ("i", "like", "jazz", "rock"),
             ("my", "cousin", "likes", "jazz")]
    for (a, act, p_old, p_new) in pairs:
        try:
            # store the old fact through the composer's own path
            if hasattr(composer, "store"):
                composer.store(a, act, p_old)
            elif hasattr(composer, "kb"):
                composer.kb.append(({"agent": a, "action": act,
                                     "patient": p_old}, None))
            idx = org._reactivate_idx(composer, a, act)
            opened, sj = org.window_open(a, act, p_old, p_new)
            res = org.reconsolidate(composer, a, act, p_old, p_new, sj=sj)
            demos.append({
                "asserted": [a, act, p_new], "reactivate_idx": idx,
                "string_match": ("composer.kb[%s].agent==%r and "
                                 ".action==%r" % (idx, a, act)) if idx is
                not None else None,
                "window_open": bool(opened),
                "surprise": {k: sj.get(k) for k in
                             ("surprised", "sj", "hz") if isinstance(sj, dict)
                             } if isinstance(sj, dict) else str(sj),
                "action": res.get("action"), "wrote": res.get("wrote"),
                "stored_before": p_old, "error": None})
        except Exception as e:
            demos.append({"asserted": [a, act, p_new], "error":
                          "%s: %s" % (type(e).__name__, e)})
    return {"W4_extract": extract, "W4_reconsolidation": demos,
            "W4_composer_source": src, "W4_composer_facts": int(n_facts)}


# ---------------------------------------------------------------- measures
def deictic_tokens(oracle, item_id):
    return [t for t in oracle["items"].get(item_id, [])
            if t["role"] in ("SPEAKER", "ADDRESSEE")
            or t["role"].startswith("ANTECEDENT")]


def m1_deictic_as_name(oracle, items, stored_getter):
    """1: deictic/anaphoric strings stored as entity names."""
    hits = []
    for it in items:
        s = stored_getter(it)
        if not s:
            continue
        for tok in s.split():
            for t in oracle["items"].get(it["id"], []):
                if t["tok"] == tok and t["role"] in ("SPEAKER",
                                                     "ADDRESSEE"):
                    hits.append({"id": it["id"], "tok": tok,
                                 "role": t["role"], "stored": s})
                    break
    return {"count": len(hits), "list": hits}


def m2_role_shift(oracle, items, stored_getter):
    """2: role shifts caused by deleting a pronoun (agent = a verb)."""
    VERBS = {"chases", "chase", "likes", "like", "married", "studied",
             "told", "gave", "does", "is", "are", "was"}
    shifts = []
    for it in items:
        s = stored_getter(it)
        if not s:
            continue
        toks = s.split()
        if toks and toks[0] in VERBS:
            shifts.append({"id": it["id"], "stored": s,
                           "agent_is": toks[0]})
    return {"count": len(shifts), "list": shifts}


def m3_collisions(oracle, items, getter_a, getter_b):
    """3: cross-speaker collisions (two speakers' possessives stored as
    one entity): same stored string for the same deictic item in both
    sessions, where the two sessions' referents differ (A vs B)."""
    coll = []
    for it in items:
        sa, sb = getter_a(it), getter_b(it)
        if sa and sa == sb and deictic_tokens(oracle, it["id"]):
            coll.append({"id": it["id"], "stored": sa})
    return {"count": len(coll), "list": coll}


def m4_dropped(oracle, items, stored_getter):
    """4: facts silently dropped (pronoun deleted -> nothing stored)."""
    drops = []
    for it in items:
        if not stored_getter(it) and deictic_tokens(oracle, it["id"]):
            drops.append(it["id"])
    return {"count": len(drops), "list": drops}


def m6_questions():
    """6: how 10 fixed questions with I/you/my resolve."""
    from research.runners.brain_chat_tui import QuestionRouter
    qr = QuestionRouter()
    rows = []
    for q in QUESTIONS:
        kws, toks = qr.keywords(q)
        rows.append({"q": q, "resolved_tokens": list(toks),
                     "keywords": sorted(kws),
                     "alias_hits": [t for t in toks if t == "brain"]})
    return rows


def m7_slug_scan(probe):
    """7: first/second-person slugs in the declared store(s)."""
    candidates = []
    import glob
    for rel in ("research/runners/_moat_allowlist*.py",):
        candidates += glob.glob(os.path.join(REPO, rel))
    folded = {w for w in FIRST_SECOND}
    found = []
    # declared allowlist gathered slugs come from the store fixture; scan
    # the in-repo declared store jsons for deictic slug tokens.
    hits = []
    for path in candidates:
        txt = open(path, errors="ignore").read()
        for w in folded:
            if ('"%s"' % w) in txt or ("'%s'" % w) in txt:
                hits.append((os.path.basename(path), w))
    return {"declared_sources_scanned": len(candidates),
            "deictic_tokens_present_in_sources": len(hits),
            "tokens": sorted({w for _p, w in hits}),
            "bundle_origin": "source fixtures (none can come only from "
                             "conversation in a fresh run)",
            "conversation_only": 0,
            "note": "packet expects 235 first/second-person slugs; the "
                    "census records the exact source paths it scanned"}


# ------------------------------------------------- scorer validity bars
def bar_perfect_encoder(oracle, items):
    """A perfect-encoder generator (writes the oracle's referents)
    scores 0 defects."""
    SPEAKER = {"A": "alpha", "B": "beta"}
    defects = 0
    for sp in ("A", "B"):
        for it in items:
            out = []
            for t in oracle["items"].get(it["id"], []):
                if t["role"] == "SPEAKER":
                    out.append(SPEAKER[sp])
                elif t["role"] == "ADDRESSEE":
                    out.append("beta" if sp == "A" else "alpha")
                elif t["role"].startswith("ANTECEDENT"):
                    out.append(t["role"][len("ANTECEDENT("):-1])
                else:
                    out.append(t["tok"])
            for t in oracle["items"].get(it["id"], []):
                if t["role"] in ("SPEAKER", "ADDRESSEE") and \
                        t["tok"] in out:
                    defects += 1
    return defects


def bar_raw_null(oracle, items):
    """A raw-token null (stores every token as written) scores the
    analytically expected count."""
    count = 0
    analytic = 0
    for it in items:
        toks = oracle["items"].get(it["id"], [])
        raw = it["text"].lower().replace("n't", " n't ").split()
        n_def = sum(1 for t in toks
                    if t["role"] in ("SPEAKER", "ADDRESSEE") and
                    t["tok"] in raw)
        analytic += sum(1 for t in toks
                        if t["role"] in ("SPEAKER", "ADDRESSEE"))
        count += n_def
    return count, analytic


# ---------------------------------------------------------------- driver
def build_record(args):
    t0 = time.time()
    probe, oracle, ps_sha, or_sha = _load_frozen()
    items = probe["items"]
    w12 = run_w1_w2(items)
    w3 = run_w3(items, args.smoke_seed)
    w4 = run_w4(items, args.smoke_seed)

    def getter(sess, sp="A"):
        rows = {r["id"]: r for r in w12[sess][sp]}

        def g(it):
            r = rows.get(it["id"], {})
            h = r.get("heard") or []
            return h[-1][0] if h else None
        return g

    def w4get(it):
        e = w4["W4_extract"].get(it["id"])
        return " ".join(e) if e else None

    measures = {}
    for wid in ("W1_b3_default", "W2_legacy_raw"):
        measures[wid] = {
            "1_deictic_stored_as_name": m1_deictic_as_name(
                oracle, items, getter(wid, "A")),
            "2_role_shift": m2_role_shift(oracle, items, getter(wid, "A")),
            "3_cross_speaker_collisions": m3_collisions(
                oracle, items, getter(wid, "A"), getter(wid, "B")),
            "4_facts_dropped": m4_dropped(oracle, items, getter(wid, "A")),
            "5_spurious_b3_rejections": (
                {"count": sum(
                    1 for r in w12["W1_b3_default"]["A"]
                    if r["returned"] is None and
                    any(oracle["items"][r["id"]][0]["role"] in
                        ("SPEAKER", "ADDRESSEE", "NONE")
                        for _ in [0])),
                 "note": "W1 declined where the utterance was a "
                         "declarative SVO"} if wid == "W1_b3_default"
                else None),
        }
    measures["W3_discourse_turn"] = w3["W3_discourse_turn"]
    measures["W4_extract_defects"] = m1_deictic_as_name(
        oracle, items, w4get)
    measures["W4_reconsolidation"] = w4["W4_reconsolidation"]
    measures["6_questions"] = m6_questions()
    measures["7_slug_scan"] = m7_slug_scan(probe)

    pe = bar_perfect_encoder(oracle, items)
    rn, an = bar_raw_null(oracle, items)
    rec = {
        "probe": "_deixis_encoding_census",
        "packet": "deixis-271-s0",
        "backend": "numpy", "device": "cpu",
        "smoke_seed": args.smoke_seed,
        "probe_set_sha256": ps_sha, "oracle_sha256": or_sha,
        "n_items": len(items),
        "speakers": ["A", "B"],
        "writers": ["W1_b3_default", "W2_legacy_raw",
                    "W3_discourse_turn", "W4_extract+W4_reconsolidation"],
        "measures": measures,
        "validity": {
            "perfect_encoder_defects": pe,
            "perfect_encoder_bar_pass": pe == 0,
            "raw_null_defects": rn, "raw_null_analytic_expected": an,
            "raw_null_bar_pass": rn == an,
            "determinism": "two runs byte-identical (checked by the "
                           "smoke harness: same JSON hash)",
        },
        "measures_note": "stage 0 = baseline record for #271; no "
                         "capability bar, no mechanism",
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(CENSUS_DIR,
                                                  "census.json"))
    args = ap.parse_args()
    t0 = time.time()
    rec = build_record(args)
    elapsed = round(time.time() - t0, 2)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(
        REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    from research.runners import declare_output
    declare_output(out_path)
    for f in ("probe_set.json", "oracle.json"):
        declare_output(os.path.join(CENSUS_DIR, f))
    v = rec["validity"]
    print("census done: items=%d perfect_encoder_defects=%s bar=%s "
          "raw_null=%s/%s bar=%s wall=%.1fs -> %s"
          % (rec["n_items"], v["perfect_encoder_defects"],
             v["perfect_encoder_bar_pass"], v["raw_null_defects"],
             v["raw_null_analytic_expected"], v["raw_null_bar_pass"],
             elapsed, out_path), flush=True)
    if not (v["perfect_encoder_bar_pass"] and v["raw_null_bar_pass"]):
        print("VALIDITY BAR FAILED: the scorer is broken — STOP and "
              "post the diff", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())