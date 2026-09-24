"""CONTENT PROVENANCE through the production chat path, Qwen mouth ON: is a fact the brain states in a reply one it
LEARNED (a synapse written by a plasticity rule during the conversation), or content the Qwen mouth / a host imprint
supplied? Plus the LEARNED-CONTENT FRACTION of a scripted conversation. Pre-registered in
research/findings/2026-09-24-content-provenance-learned-facts-PREREGISTRATION.md (committed before any run of this).

WHY (owner concern, 2026-09-24). The fear: the brain's knowledge is "fancy RAG the LLM pulls from", not learning. The
same-day audit found (1) the default chat's facts come from a host bulk imprint (wikidata_100k, 78,857 facts, 0 written
by a plasticity rule); (2) Qwen is mouth-only (renders one gated triple; VERIFY + the claim moat reject additions);
(3) D6 learn-through-use (research/runners/d6_hebbian_store.py, BRAIN_D6_HEBBIAN_STORE, default-OFF) proved a synaptic
write carries a later recall at RUNNER level with the LLM disabled (stub renderer, rich=False). The load-bearing battery
marks semantic-recall / in-loop-learning / content-selection / moat-verify as not covered. What was NOT measured: the
same question with the real Qwen mouth rendering the reply on the production default (rich, fluent) turn, on facts
Qwen's own prior would contradict, and a count of how much of what the brain says was learned by plasticity.

THE TEST (per seed; every arm is a FRESH brain in its own subprocess, BRAIN_CHAT_SEED=seed; renderer='qwen' (the warm
QwenRenderer / SpikingQwenFaculty; on CPU when CUDA is hidden, on CUDA via tools/gpu_queue.sh); `rich` OMITTED from
every request so the production default applies (BRAIN_RICH unset -> the fluent multi-sentence RichAnswerComposer
path); the wikidata imprint OFF (BRAIN_LTM_SHIP_DEFAULT=0) in every primary arm; BRAIN_PERSIST_LEARNING=0).
  teach   three assertions: a NONCE fact (Qwen has no prior) and two COUNTERFACTUAL facts (Qwen's prior contradicts):
            "the wug eats the dax" / "the cow eats the moon" / "the bee makes the stone"
          (HEARD arm: the same content words, not an assertion: "the wug and the dax", ... -> nothing to acquire)
  read    "what does the dog chase" / "what does the cat eat"  (build-time facts: the read path is intact)
  probe   "what does the wug eat" / "what does the cow eat" / "what does the bee make"
  untaught  20 questions the brain was never taught (10 near-miss on known agents, 10 fully untaught) -- USE + DEFAULT
  secondary (report-only; USE/FREEZE/DEFAULT, AFTER every primary turn): "the cat eats the moon" then "what does the
            cat eat" -- the owner's own example; it conflicts with the build-time fact (cat eat fish), so it cannot
            separate Qwen's prior from the brain's own imprint and is never scored.
ARMS
  USE      D6 learning path ON (BRAIN_D6_HEBBIAN_STORE=1 + ENGRAM_VOCAB=1 + ENGRAM_READTIME=1, FREEZE=0)
  FREEZE   USE + BRAIN_D6_HEBBIAN_FREEZE=1           (a) learning rule frozen during teaching (eta=0, same activity)
  ABLATE   USE + after the teach turns the experimenter zeroes every taught block's synapses (b)
  HEARD    FREEZE's flags + the exposure sentences   (c) the words only heard, learning off
  DEFAULT  every D6 flag 0 = the production write (a host direct COPY of the composite into the weights)
  DEFAULT_LTM (optional, fraction only, never scored) = the TRUE production default (wikidata LTM attached)
Scoring, instruments and the failing-direction checks: see `score_seed` and the PREREGISTRATION. The scorer is verified
to FAIL in its failing direction both synthetically (`--selftest`) and LIVE on every seed: the real Qwen-alone answer
to each probe/untaught question (the same warm SpikingQwenFaculty, asked directly, no brain) is substituted for the
brain's reply and must be scored as a FAIL / a leak.

LEARNED-CONTENT FRACTION. Over the scripted conversation's recall turns (read + probe + untaught), each factual
proposition a reply STATES (a supporting/recalled SVO whose object word appears in the reply text) is traced to the
store block that holds it, and that block to its LAST writer, logged live by instrumenting the composer's
`_store_composite` / `_write_block`: pre-session imprint (build-time, host-chosen facts) / in-session host direct copy /
in-session local-Hebbian plasticity write (D6, eta>0) / frozen / experimenter ablation / LTM bulk imprint / unsourced.
fraction = (#stated propositions whose block's last writer is an in-session plasticity write) / (#stated propositions).

HONEST SCOPE. The instrument measures WHERE a stated fact's synapses came from, and whether Qwen can inject content;
it does not make the D6 write any less teacher-forced than d6_hebbian_store.py's own addendum states (the Hebbian
weights match the host copy at complex correlation 0.99996 -- a local rule carried by a host-wired instructive
pathway). Host code here is experimenter instrumentation only (wrappers that LOG writes and Qwen calls; the ablation
lesion; the Qwen-alone reference); none of it feeds a reply.

Self-test (no brain):  .venv/bin/python -m research.runners.content_provenance_probe --selftest
One seed, local CPU (Qwen on CPU, brain numpy; ALWAYS memcapped):
  bash tools/mem_ok.sh 10 4 && CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy bash tools/memcap.sh 12 -- \
    .venv/bin/python -u -m research.runners.content_provenance_probe --seeds 7 \
    --arm-dir research/findings/raw/_content_provenance --json research/findings/raw/_content_provenance/cp_s7.json
Score only:  .venv/bin/python -m research.runners.content_provenance_probe --score-only --seeds 42 43 44 100 101 102 \
    --arm-dir research/findings/raw/_content_provenance --json research/findings/raw/_content_provenance/cp_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SESSION = "cprov"
BRAIN = "tiny-demo"
RENDERER = "qwen"
SEEDS6 = [42, 43, 44, 100, 101, 102]
DEV_SEED = 7

# (key, teach assertion, exposure (HEARD), probe, taught object, hand-written prior-answer key)
ITEMS = [
    ("wug", "the wug eats the dax", "the wug and the dax", "what does the wug eat", "dax", ()),
    ("cow", "the cow eats the moon", "the cow and the moon", "what does the cow eat", "moon",
     ("grass", "hay", "grain", "oat", "feed", "plant", "clover", "silage", "corn", "alfalfa")),
    ("bee", "the bee makes the stone", "the bee and the stone", "what does the bee make", "stone",
     ("honey", "wax", "beeswax", "pollen", "nectar", "hive", "honeycomb", "comb", "jelly")),
]
READ_TURNS = [("d1", "what does the dog chase"), ("d2", "what does the cat eat")]
# (key, question, hand-written answer key). 10 near-miss (a known agent, an untaught relation) + 10 fully untaught.
UNTAUGHT = [
    ("u01", "what does the cow drink", ("water", "milk")),
    ("u02", "what does the bee eat", ("nectar", "pollen", "honey", "flower")),
    ("u03", "what does the wug drink", ("water", "milk", "juice")),
    ("u04", "what does the cat drink", ("milk", "water")),
    ("u05", "what does the dog eat", ("meat", "bone", "kibble", "food", "biscuit")),
    ("u06", "what does the bird eat", ("seed", "insect", "berry", "grain", "bug", "fruit")),
    ("u07", "what does the cow make", ("milk", "dairy", "butter", "cheese", "manure")),
    ("u08", "what does the dog drink", ("water", "milk")),
    ("u09", "what does the horse eat", ("hay", "grass", "oat", "apple", "carrot", "grain")),
    ("u10", "what does the bird build", ("nest",)),
    ("u11", "what does the lion eat", ("meat", "zebra", "antelope", "gazelle", "prey", "buffalo", "animal")),
    ("u12", "what does the spider spin", ("web", "silk", "cobweb")),
    ("u13", "what does the farmer grow", ("crop", "wheat", "corn", "vegetable", "fruit", "grain", "rice", "potato")),
    ("u14", "what does the baker bake", ("bread", "cake", "pastry", "cookie", "pie", "muffin", "loaf")),
    ("u15", "what does the monkey eat", ("banana", "fruit", "leaf", "insect", "nut")),
    ("u16", "what does the rabbit eat", ("carrot", "grass", "vegetable", "lettuce", "hay", "clover")),
    ("u17", "what is the capital of france", ("paris",)),
    ("u18", "who wrote hamlet", ("shakespeare", "william")),
    ("u19", "what color is the sky", ("blue", "azure")),
    ("u20", "how many legs does a spider have", ("eight", "8")),
]
SECONDARY = [("s_teach", "the cat eats the moon"), ("s_probe", "what does the cat eat")]
QWEN_ALONE_TEMPLATE = "Answer in one short sentence: {q}?"

# The words the brain itself holds on this protocol (build-time vocabulary + the taught words). A Qwen-alone word in
# this set is NOT counted as a Qwen leak (the brain can legitimately say it from its own store).
BUILD_VOCAB = ("brain", "use", "spikes", "learn", "words", "store", "memory", "dog", "chase", "cat", "eat", "fish",
               "river", "bird", "worm", "ball")
TAUGHT_VOCAB = ("wug", "dax", "cow", "moon", "bee", "make", "stone")
STOP = frozenset("""a an the is are was were be been being am it its they them their this that these those of in on
at to for from with by as and or but not no nor do does did doing what who whom whose how which where when why many much
some any all most more very can could would should will shall may might must has have had having i you he she we me my
your our his her there here than then so if about into over under up down out just only also too each other such
own same s t don doesn didn isn aren wasn weren won wouldn shouldn couldn let lets""".split())
GENERIC = frozenset("""typically usually generally mainly primarily mostly often sometimes commonly various variety
different type types kind kinds known including include includes like well sure sorry answer question information
provide provided specific context depends depend however cannot unable help certain particular called refers refer term
example examples based found curiosity piqued learned tell setting held thread aside fairly think believe seems seem
maybe perhaps guess heard taught remember recall familiar novel uncertain confident monitor reads read know short
sentence one""".split())

D6_ON = {"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_READTIME": "1"}
D6_KEYS = ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME", "BRAIN_D6_HEBBIAN_FREEZE")
# name: (env, teach mode, post-teach step, run untaught block, run secondary block, run Qwen-alone reference, ltm)
ARMS = {
    "USE":     (dict(D6_ON, BRAIN_D6_HEBBIAN_FREEZE="0"), "assert", None, True, True, True, "off"),
    "FREEZE":  (dict(D6_ON, BRAIN_D6_HEBBIAN_FREEZE="1"), "assert", None, False, True, False, "off"),
    "ABLATE":  (dict(D6_ON, BRAIN_D6_HEBBIAN_FREEZE="0"), "assert", "ablate", False, False, False, "off"),
    "HEARD":   (dict(D6_ON, BRAIN_D6_HEBBIAN_FREEZE="1"), "expose", None, False, False, False, "off"),
    "DEFAULT": ({k: "0" for k in D6_KEYS}, "assert", None, True, True, False, "off"),
    "DEFAULT_LTM": ({k: "0" for k in D6_KEYS}, "assert", None, True, False, False, "default"),
}
PRIMARY_ARMS = ("USE", "FREEZE", "ABLATE", "HEARD", "DEFAULT")
LESION_ARMS = ("FREEZE", "ABLATE", "HEARD")
KEEP_KEYS = ("answer", "abstained", "recalled_svo", "supporting_facts", "verified", "renderer", "rich", "n_sentences",
             "derived", "derived_from", "hypothesis", "hypothesis_svo", "followup", "source")


# ── text helpers (the scorer's instrument; host code, never on the reply path) ──────────────────────────────────────
def _stem(w):
    w = str(w).lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and (w.endswith("ches") or w.endswith("shes") or (w.endswith("es") and w[-3] in "sxz")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _stems(text):
    return {_stem(t) for t in _tokens(text)}


def _mentions(text, word):
    """The reply states `word` (a raw word; plural/3sg-insensitive)."""
    return _stem(word) in _stems(text)


def _has_stem(text, stem):
    """The reply carries an already-stemmed word (never re-stemmed: _stem is not idempotent on every word)."""
    return stem in _stems(text)


def _same_verb(a, b):
    a, b = str(a).lower(), str(b).lower()
    return a == b or a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a or _stem(a) == _stem(b)


BRAIN_LEXICON = frozenset(_stem(w) for w in BUILD_VOCAB + TAUGHT_VOCAB)


def qwen_content_stems(qwen_answer, question):
    """Content stems of a Qwen-alone answer that a brain reply must not carry: minus the question's own words, stop and
    generic/template words, and the brain's own lexicon; >= 3 characters (digits kept)."""
    q = _stems(question)
    return sorted({s for s in _stems(qwen_answer)
                   if (len(s) >= 3 or s.isdigit()) and s not in q and s not in STOP and s not in GENERIC
                   and s not in BRAIN_LEXICON})


# ── worker: ONE fresh brain, the scripted conversation through the real handler, fully instrumented ─────────────────
def _buffer_composer(chat):
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is not None and type(comp).__name__ == "TieredFactStore":
        return object.__getattribute__(comp, "buffer"), True
    return comp, False


def _trim(resp):
    out = {k: resp.get(k) for k in KEEP_KEYS if k in resp}
    act = resp.get("activity") or {}
    out["matched_fact_index"] = act.get("matched_fact_index") if isinstance(act, dict) else None
    out["full_sha256"] = hashlib.sha256(json.dumps(resp, sort_keys=True, default=str).encode()).hexdigest()
    return out


def _kb_dump(comp):
    rows = []
    for i, (f, _h) in enumerate(getattr(comp, "kb", []) or []):
        p = f.get("patient")
        rows.append([i, f.get("agent"), f.get("action"), p if isinstance(p, str) else repr(p), f.get("polarity")])
    return rows


def _block_record(comp, agent, patient):
    """The LAST kb block holding (agent, *, patient) and its mean |w| over the D trigger->readout synapses. Read-only."""
    idx = None
    for j, (f, _h) in enumerate(getattr(comp, "kb", []) or []):
        if str(f.get("agent", "")).lower() == agent and str(f.get("patient", "")).lower() == patient:
            idx = j
    if idx is None:
        return {"found": False, "agent": agent, "patient": patient}
    D = comp.D
    ws = [complex(w) for (_p, _q, w) in comp.store_conns[idx * D:(idx + 1) * D]]
    return {"found": True, "agent": agent, "patient": patient, "block": idx,
            "mean_abs_w": round(sum(abs(w) for w in ws) / max(len(ws), 1), 6), "n_syn": len(ws)}


VARIANTS = ("prod", "qwenforced")


def _variant():
    """AMENDMENT A1 (see the PREREGISTRATION's amendment log). 'prod' = the registered configuration (the shipped
    mouth stack: the spiking Broca recall mouth renders bounded SVO recall, Qwen is the renderer behind it);
    'qwenforced' = every arm with BRAIN_SPIKING_MOUTH_RECALL=0, so Qwen phrases every recalled fact."""
    v = os.environ.get("CPROV_VARIANT", "prod").strip().lower()
    return v if v in VARIANTS else "prod"


def _worker(arm, out_path):
    env, teach_mode, post, do_untaught, do_secondary, do_qwen_alone, ltm = ARMS[arm]
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ["BRAIN_PERSIST_LEARNING"] = "0"
    for k, v in env.items():
        os.environ[k] = v                                   # explicit values both directions (never a pop)
    if ltm == "off":
        os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"
    else:
        os.environ.pop("BRAIN_LTM_SHIP_DEFAULT", None)      # DEFAULT_LTM: the true shipped default (LTM attached)
    for k in ("BRAIN_RICH", "BRAIN_CHAT_RENDERER"):
        os.environ.pop(k, None)                             # production default path; renderer set per request
    variant = _variant()
    if variant == "qwenforced":
        # AMENDMENT A1: the spiking Broca recall mouth (BRAIN_SPIKING_MOUTH_RECALL, default-ON since 2026-08-26)
        # renders every bounded SVO recall before Qwen is consulted; OFF -> Qwen phrases every recalled fact.
        os.environ["BRAIN_SPIKING_MOUTH_RECALL"] = "0"
    else:
        os.environ.pop("BRAIN_SPIKING_MOUTH_RECALL", None)  # prod: the shipped default
    from webapp import server as S
    from webapp.server import brain_chat, BrainChatRequest
    from research.runners import d6_hebbian_store as D6
    t0 = time.time()
    out = {"arm": arm, "env": env, "ltm": ltm, "variant": variant, "seed": os.environ.get("BRAIN_CHAT_SEED"),
           "spiking_mouth_recall_env": os.environ.get("BRAIN_SPIKING_MOUTH_RECALL"),
           "backend": os.environ.get("SIM_BACKEND"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "turns": [], "qwen_calls": [], "write_log": [], "taught_blocks_after_teach": {},
           "taught_blocks_at_probe": {}, "ablation": [], "counter_installed": False,
           "store_writes_after_teach_primary": None, "homeostatic_calls_after_teach": None, "qwen_alone": {},
           "errors": []}
    cache_key = (SESSION, BRAIN, RENDERER)
    # ---- build exactly as brain_chat's cache-miss path does (so the logger is in place before the first teach) ----
    chat, source = S._build_chat_brain(BRAIN, RENDERER)
    chat._brain_chat_source = source
    S._BRAIN_CHATS[cache_key] = chat
    S._reload_persisted_learning(cache_key, chat)
    out["build_s"] = round(time.time() - t0, 1)
    out["source"] = source
    rend = getattr(chat, "renderer", None)
    out["renderer_class"] = type(rend).__name__
    fac = getattr(rend, "_fac", None)
    out["qwen_device"] = str(getattr(fac, "device", None))
    comp, ltm_attached = _buffer_composer(chat)
    out["ltm_attached"] = ltm_attached
    out["composer_class"] = type(comp).__name__
    out["n_kb_at_session_start"] = len(getattr(comp, "kb", []) or [])
    out["kb_at_session_start"] = _kb_dump(comp)
    state = {"phase": "session", "turn": None, "in_store": False, "experimenter": None, "count": False,
             "after_teach": [], "homeo": 0}
    # ---- instrument 1: every Qwen generation (the mouth's single choke point: the faculty model's generate) ----
    if fac is not None and getattr(fac, "model", None) is not None:
        tok = fac.tok
        _orig_gen = fac.model.generate

        def _gen_logged(*a, **k):
            ids = k.get("input_ids", a[0] if a else None)
            res = _orig_gen(*a, **k)
            try:
                n_in = ids.shape[1]
                prompts = tok.batch_decode(ids, skip_special_tokens=True)
                comps = tok.batch_decode(res[:, n_in:], skip_special_tokens=True)
            except Exception as e:  # never let the log break a turn
                prompts, comps = ["<decode error %s>" % e], []
            out["qwen_calls"].append({"phase": state["phase"], "turn": state["turn"],
                                      "prompts": [p[-240:] for p in prompts], "completions": comps})
            return res
        fac.model.generate = _gen_logged
    # ---- instrument 2: every store write, with its writer (provenance) ----
    _orig_sc = comp._store_composite

    def _sc_logged(fillers, roles, _o=_orig_sc):
        i = len(comp.kb)
        state["in_store"] = True
        try:
            r = _o(fillers, roles)
        finally:
            state["in_store"] = False
        heb = D6.hebbian_store_enabled()
        enc = getattr(comp, "_d6_last_encode", None) if heb else None
        if heb and enc and int(enc.get("block", -1)) == i:
            method = "hebbian_frozen" if enc.get("frozen") else "hebbian"
        else:
            method = "direct_copy"
        out["write_log"].append({"block": i, "method": method, "via": "store", "phase": state["phase"],
                                 "turn": state["turn"], "encode": dict(enc) if enc else None,
                                 "fillers": [str(x) for x in fillers]})
        return r
    comp._store_composite = _sc_logged
    _orig_wb = comp._write_block

    def _wb_logged(bi, zc, _o=_orig_wb):
        if not state["in_store"]:
            out["write_log"].append({"block": int(bi), "method": state["experimenter"] or "direct_rewrite",
                                     "via": "write_block", "phase": state["phase"], "turn": state["turn"]})
        if state["count"]:
            state["after_teach"].append({"block": int(bi), "phase": state["phase"], "turn": state["turn"]})
        return _o(bi, zc)
    comp._write_block = _wb_logged
    if hasattr(comp, "apply_homeostatic_scaling"):
        _oh = comp.apply_homeostatic_scaling

        def _h_logged(*a, _o=_oh, **k):
            if state["count"]:
                state["homeo"] += 1
            return _o(*a, **k)
        comp.apply_homeostatic_scaling = _h_logged

    def _turn(phase, label, msg):
        state["phase"], state["turn"] = phase, label
        n_q = len(out["qwen_calls"])
        t = time.time()
        try:
            r = brain_chat(BrainChatRequest(session=SESSION, message=msg, brain=BRAIN, renderer=RENDERER))
            rec = _trim(json.loads(r.body))
        except Exception as e:
            rec = {"_error": "%s: %s" % (type(e).__name__, e)}
        rec.update({"phase": phase, "label": label, "message": msg, "elapsed_s": round(time.time() - t, 1),
                    "n_qwen_calls": len(out["qwen_calls"]) - n_q})
        out["turns"].append(rec)
        print("[cprov %s] %-8s %-9s %5.1fs q=%d %r -> %r" % (arm, phase, label, rec["elapsed_s"], rec["n_qwen_calls"],
              msg, str(rec.get("answer", rec.get("_error")))[:140]), flush=True)
        return rec

    # ---- the scripted conversation ----
    for key, teach, expo, _probe, obj, _prior in ITEMS:
        _turn("teach", "t_" + key, teach if teach_mode == "assert" else expo)
    for key, _teach, _expo, _probe, obj, _prior in ITEMS:
        out["taught_blocks_after_teach"][key] = _block_record(comp, key, obj)
    if post == "ablate":                                    # (b) the experimenter zeroes every taught block
        state["experimenter"] = "experimenter_ablation"
        for key, _teach, _expo, _probe, obj, _prior in ITEMS:
            rec = out["taught_blocks_after_teach"][key]
            if rec.get("found"):
                out["ablation"].append(dict(D6.ablate_block(comp, rec["block"]), item=key))
            else:
                out["ablation"].append({"item": key, "error": "taught block not found; ablation not applied"})
        state["experimenter"] = None
    state["count"] = True                                   # every store write from here on is counted
    out["counter_installed"] = True
    for label, q in READ_TURNS:
        _turn("read", label, q)
    for key, _teach, _expo, _probe, obj, _prior in ITEMS:   # the lever, read AT MEASUREMENT TIME
        out["taught_blocks_at_probe"][key] = _block_record(comp, key, obj)
    for key, _teach, _expo, probe, obj, _prior in ITEMS:
        _turn("probe", "p_" + key, probe)
    if do_untaught:
        for key, q, _hk in UNTAUGHT:
            _turn("untaught", key, q)
    out["store_writes_after_teach_primary"] = list(state["after_teach"])
    out["homeostatic_calls_after_teach"] = state["homeo"]
    out["kb_final_primary"] = _kb_dump(comp)
    if do_secondary:
        for label, msg in SECONDARY:
            _turn("secondary", label, msg)
        out["kb_final"] = _kb_dump(comp)
    if do_qwen_alone and fac is not None:                   # the reference: Qwen asked directly, no brain
        state["phase"] = "qwen_alone"
        for label, q in ([("p_" + k, p) for k, _t, _e, p, _o, _pr in ITEMS] + [(k, q) for k, q, _hk in UNTAUGHT]
                         + [("s_probe", SECONDARY[1][1])]):
            state["turn"] = label
            try:
                first, full, secs = fac._generate(QWEN_ALONE_TEMPLATE.format(q=q))
                out["qwen_alone"][label] = {"question": q, "answer": first, "full": full, "secs": secs}
            except Exception as e:
                out["qwen_alone"][label] = {"question": q, "_error": "%s: %s" % (type(e).__name__, e)}
    out["d6_ops"] = dict(getattr(comp, "_d6_ops", {}) or {})
    out["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1, default=str)
    print("[cprov worker] arm=%s -> %s (%.0fs)" % (arm, out_path, out["elapsed_s"]), flush=True)
    return 0


def _arm_path(arm_dir, seed, arm):
    return os.path.join(arm_dir, "s%d_%s.json" % (int(seed), arm))


def _spawn(arm, seed, out_path):
    penv = dict(os.environ)
    penv["BRAIN_CHAT_SEED"] = str(seed)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.content_provenance_probe", "--worker",
                        "--arm", arm, "--out", out_path], env=penv)
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    return json.load(open(out_path))


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


# ── scoring ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _turn_of(arm, label):
    for t in (arm or {}).get("turns") or []:
        if t.get("label") == label:
            return t
    return None


def _prior_stems(item_key, qwen_alone):
    """Words that would mean Qwen's prior answered the probe: the hand key + the Qwen-alone answer's content stems."""
    it = {k: (k, t, e, p, o, pr) for k, t, e, p, o, pr in ITEMS}[item_key]
    _k, _t, _e, probe, obj, prior = it
    s = {_stem(w) for w in prior}
    qa = (qwen_alone or {}).get("p_" + item_key) or {}
    s |= set(qwen_content_stems(qa.get("answer", ""), probe))
    s.discard(_stem(obj))
    return s


def taught_ok(turn, item_key, qwen_alone):
    """CP1 per item: not abstained, the reply STATES the taught object, and states none of Qwen's prior words."""
    obj = {k: o for k, _t, _e, _p, o, _pr in ITEMS}[item_key]
    if not turn or "_error" in turn:
        return False
    ans = turn.get("answer") or ""
    return (turn.get("abstained") is False and _mentions(ans, obj)
            and not any(_has_stem(ans, s) for s in _prior_stems(item_key, qwen_alone)))


def lesion_ok(turn, item_key, qwen_alone):
    """CP2-CP4 per item: the brain abstains, the reply does not state the taught object, and carries no Qwen prior."""
    obj = {k: o for k, _t, _e, _p, o, _pr in ITEMS}[item_key]
    if not turn or "_error" in turn:
        return False
    ans = turn.get("answer") or ""
    return (turn.get("abstained") is True and not _mentions(ans, obj)
            and not any(_has_stem(ans, s) for s in _prior_stems(item_key, qwen_alone)))


def leak_of(turn, ukey, qwen_alone, kb_rows):
    """A leak on an untaught question: the reply states a hand-key answer word, or a content word of Qwen's own answer
    to that question, or a supporting/recalled proposition that is in no store block (unsourced)."""
    q, hk = {k: (q, hk) for k, q, hk in UNTAUGHT}[ukey]
    if not turn or "_error" in turn:
        return {"leak": None, "error": True}
    ans = turn.get("answer") or ""
    hand = sorted(w for w in hk if _mentions(ans, w))
    qa = (qwen_alone or {}).get(ukey) or {}
    qwen = sorted(s for s in qwen_content_stems(qa.get("answer", ""), q) if _has_stem(ans, s))
    unsourced = [p for p in _props(turn) if _kb_match(kb_rows, p) is None]
    return {"leak": bool(hand or qwen or unsourced), "hand_hits": hand, "qwen_hits": qwen, "unsourced": unsourced}


def _props(turn):
    fs = []
    for f in (turn.get("supporting_facts") or []) + ([turn["recalled_svo"]] if turn.get("recalled_svo") else []) \
            + (turn.get("derived_from") or []):
        if isinstance(f, (list, tuple)) and len(f) >= 3:
            key = [str(f[0]).lower(), str(f[1]).lower(), str(f[2]).lower()]
            if key not in fs:
                fs.append(key)
    return fs


def _kb_match(kb_rows, prop):
    a, v, p = prop
    hit = None
    for row in kb_rows or []:
        i, ka, kv, kp = row[0], str(row[1]).lower(), str(row[2]).lower(), str(row[3]).lower()
        if ka == a and kp == p and _same_verb(kv, v):
            hit = i
    return hit


_CLASS = {"hebbian": "in_session_plasticity", "hebbian_frozen": "in_session_frozen",
          "direct_copy": "in_session_host_copy", "direct_rewrite": "in_session_host_copy",
          "experimenter_ablation": "experimenter_ablated"}


def block_classes(arm, exclude_write_phases=("secondary", "qwen_alone")):
    """Each store block's provenance class = its LAST logged writer during the primary script (writes made in the
    report-only secondary block are excluded, so they cannot reclassify a block the primary turns read)."""
    n0 = int(arm.get("n_kb_at_session_start") or 0)
    rows = arm.get("kb_final_primary") or arm.get("kb_final") or []
    cls = {int(r[0]): ("pre_session_imprint" if int(r[0]) < n0 else "in_session_unlogged") for r in rows}
    for w in arm.get("write_log") or []:
        if w.get("phase") in exclude_write_phases:
            continue
        cls[int(w["block"])] = _CLASS.get(w["method"], w["method"])
    return cls


def learned_content_fraction(arm, phases=("read", "probe", "untaught")):
    """The share of STATED factual propositions (over the recall turns) whose store block's last writer is an
    in-session plasticity write. None (UNDEFINED) when no proposition was stated."""
    if not arm:
        return {"fraction": None, "n": 0}
    cls = block_classes(arm)
    rows = arm.get("kb_final_primary") or arm.get("kb_final") or []
    counts, items = {}, []
    for t in arm.get("turns") or []:
        if t.get("phase") not in phases or "_error" in t:
            continue
        for p in _props(t):
            if not _mentions(t.get("answer") or "", p[2]):
                continue                                    # a supporting fact the reply does not state
            idx = _kb_match(rows, p)
            c = cls.get(idx, "in_session_unlogged") if idx is not None else (
                "ltm_imprint" if arm.get("ltm_attached") else "unsourced")
            counts[c] = counts.get(c, 0) + 1
            items.append({"turn": t.get("label"), "prop": p, "block": idx, "class": c})
    n = sum(counts.values())
    return {"fraction": (counts.get("in_session_plasticity", 0) / n) if n else None, "n": n,
            "n_plasticity": counts.get("in_session_plasticity", 0), "by_class": counts, "items": items}


def _arm_void(name, a):
    if a is None:
        return "arm missing / worker failed"
    if a.get("errors"):
        return "worker errors: %s" % a["errors"][:2]
    bad = [t.get("label") for t in a.get("turns") or [] if "_error" in t]
    if bad:
        return "turn errors: %s" % bad
    if a.get("renderer_class") != "QwenRenderer":
        return "the Qwen mouth was not the renderer (%s)" % a.get("renderer_class")
    if name in PRIMARY_ARMS and (a.get("ltm_attached") or "+LTM" in str(a.get("source"))):
        return "the wikidata LTM was attached in a primary arm"
    if any(t.get("rich") is not True for t in a.get("turns") or []):
        return "a turn did not take the production default rich path"
    return None


def score_seed(arms, variant="qwenforced"):
    """Apply the pre-registered criteria (PREREGISTRATION, section 'Criteria', + AMENDMENT A1) to one seed's
    {arm: json|None}. `variant` in VARIANTS; the default is the stricter one (the Qwen-render instrument required)."""
    from tools.lab import void_if
    rec = {"criteria": {}, "void": {}, "go": None, "instrument": {}}
    for name in PRIMARY_ARMS:
        a = arms.get(name)
        if a is not None and a.get("variant", "prod") != variant:
            arms = dict(arms)
            arms[name] = dict(a, errors=["arm variant %r != scored variant %r" % (a.get("variant", "prod"), variant)])
    for name in PRIMARY_ARMS:
        why = _arm_void(name, arms.get(name))
        if void_if(why is not None, "arm %s: %s" % (name, why)):
            rec["void"][name] = why
    if rec["void"]:
        rec["verdict"] = "UNDEFINED (void arms: %s)" % ", ".join(sorted(rec["void"]))
        return rec
    U, F, A, H, DF = (arms[k] for k in PRIMARY_ARMS)
    QA = U.get("qwen_alone") or {}
    keys = [k for k, *_ in ITEMS]
    # ---- instrument checks (a failed instrument -> UNDEFINED, never a pass or a fail) ----
    ins = rec["instrument"]

    def lev(a):
        return {k: (a.get("taught_blocks_at_probe") or {}).get(k) or {} for k in keys}
    lu, lf, la, lh, ld = lev(U), lev(F), lev(A), lev(H), lev(DF)
    ins["lever_USE_written"] = all(r.get("found") and r.get("mean_abs_w", 0) > 0.5 for r in lu.values())
    ins["lever_FREEZE_zero"] = all(r.get("found") and r.get("mean_abs_w") == 0.0 for r in lf.values())
    ins["lever_ABLATE_zero"] = (all(r.get("found") and r.get("mean_abs_w") == 0.0 for r in la.values())
                                and len(A.get("ablation") or []) == len(keys)
                                and all("error" not in x for x in A.get("ablation") or []))
    # NOTE: state["phase"] is only ever "session"/"teach"/"read"/"probe"/"untaught"/"secondary"/"qwen_alone"
    # (see _turn() above) -- "build" is never assigned, because the write_log hook is installed only after
    # the initial brain build completes. This filter is therefore structurally a no-op today (kept as an
    # explicit guard in case a future caller logs a build-phase write into the same list).
    heard_in_session_writes = [w for w in H.get("write_log") or [] if w.get("phase") != "build"]
    ins["lever_HEARD_no_write"] = (not any(r.get("found") for r in lh.values())) and not heard_in_session_writes
    ins["lesion_no_later_writes"] = all(arms[k].get("counter_installed") is True
                                        and arms[k].get("store_writes_after_teach_primary") == []
                                        for k in LESION_ARMS)
    use_probe_qwen = sum((_turn_of(U, "p_" + k) or {}).get("n_qwen_calls", 0) for k in keys)
    ins["qwen_mouth_rendered_USE_probes"] = use_probe_qwen > 0
    ins["qwen_alone_present"] = all(("p_" + k) in QA and "_error" not in QA["p_" + k] for k in keys) and all(
        k in QA and "_error" not in QA[k] for k, _q, _h in UNTAUGHT)
    # FD1 (live failing direction, CP1): Qwen's OWN answer to each probe, substituted for the brain's reply (with the
    # taught fact claimed as its support), must be scored as a FAIL -- unless Qwen alone already states the taught
    # object (then that item cannot separate Qwen from the brain: counterfactual invalid, reported).
    fd1 = {}
    for k in keys:
        qa = QA.get("p_" + k) or {}
        obj = {kk: o for kk, _t, _e, _p, o, _pr in ITEMS}[k]
        fake = {"answer": qa.get("answer", ""), "abstained": False, "supporting_facts": [[k, "x", obj]]}
        fd1[k] = {"qwen_alone_answer": qa.get("answer"), "qwen_alone_states_taught_object": _mentions(
            qa.get("answer", ""), obj), "substituted_scored_fail": not taught_ok(fake, k, QA)}
    ins["FD1"] = fd1
    ins["FD1_ok"] = all(v["substituted_scored_fail"] for v in fd1.values())
    ins["counterfactual_valid"] = {k: not v["qwen_alone_states_taught_object"] for k, v in fd1.items()}
    # FD2 (live failing direction, leak): Qwen's OWN answer to each untaught question, substituted, must be a leak.
    kb_u = U.get("kb_final_primary") or []
    fd2_hits = 0
    fd2_hand = 0
    fd2_items = {}
    for uk, q, hk in UNTAUGHT:
        qa = QA.get(uk) or {}
        lk = leak_of({"answer": qa.get("answer", ""), "abstained": False}, uk, QA, kb_u)
        fd2_items[uk] = {"qwen_alone_answer": qa.get("answer"), "flagged": lk["leak"], "hand_hits": lk["hand_hits"]}
        fd2_hits += bool(lk["leak"])
        fd2_hand += bool(lk["hand_hits"])
    ins["FD2"] = {"flagged": fd2_hits, "hand_key_hits": fd2_hand, "n": len(UNTAUGHT), "items": fd2_items}
    ins["FD2_ok"] = fd2_hits >= 15
    required = ("lever_USE_written", "lever_FREEZE_zero", "lever_ABLATE_zero", "lever_HEARD_no_write",
                "lesion_no_later_writes", "qwen_mouth_rendered_USE_probes", "qwen_alone_present", "FD1_ok", "FD2_ok")
    # AMENDMENT A1: in the 'prod' variant the shipped spiking Broca recall mouth renders bounded SVO recall ahead of
    # Qwen, so "Qwen rendered the probe" is a MEASUREMENT there (reported: rec['use_probe_qwen_calls']), not an
    # instrument; in 'qwenforced' (recall mouth off) it stays a required instrument.
    rec["variant"] = variant
    rec["use_probe_qwen_calls"] = use_probe_qwen
    if variant == "prod":
        required = tuple(k for k in required if k != "qwen_mouth_rendered_USE_probes")
    failed_ins = [k for k in required if not ins.get(k)]
    # ---- the criteria ----
    c = rec["criteria"]
    per = {k: {"USE": taught_ok(_turn_of(U, "p_" + k), k, QA),
               "FREEZE": lesion_ok(_turn_of(F, "p_" + k), k, QA),
               "ABLATE": lesion_ok(_turn_of(A, "p_" + k), k, QA),
               "HEARD": lesion_ok(_turn_of(H, "p_" + k), k, QA)} for k in keys}
    rec["per_item"] = per
    c["CP1_taught_answer"] = all(per[k]["USE"] for k in keys)
    c["CP2_freeze_abstains"] = all(per[k]["FREEZE"] for k in keys)
    c["CP3_zeroed_abstains"] = all(per[k]["ABLATE"] for k in keys)
    c["CP4_heard_only_abstains"] = all(per[k]["HEARD"] for k in keys)
    leaks = {}
    for name, a in (("USE", U), ("DEFAULT", DF)):
        rows = a.get("kb_final_primary") or []
        leaks[name] = {uk: leak_of(_turn_of(a, uk), uk, QA, rows) for uk, _q, _h in UNTAUGHT}
    rec["leaks"] = leaks
    rec["leak_count"] = {n: sum(1 for v in d.values() if v.get("leak")) for n, d in leaks.items()}
    rec["leak_undefined"] = {n: sum(1 for v in d.values() if v.get("leak") is None) for n, d in leaks.items()}
    c["CP5_no_qwen_leak"] = all(rec["leak_count"][n] == 0 and rec["leak_undefined"][n] == 0 for n in leaks)
    # ---- reported, never scored ----
    # ATTRIBUTION: of the items whose reply states the taught object with the write on, what share is absent when
    # ONLY the write is frozen (same input, same host code, eta=0) / when the words were only heard.
    from tools.lab import attributable_to

    def n_states(a):
        return float(sum(1 for k in keys if (lambda t: bool(t) and t.get("abstained") is False and _mentions(
            t.get("answer") or "", {kk: o for kk, _t, _e, _p, o, _pr in ITEMS}[k]))(_turn_of(a, "p_" + k))))
    rec["attributable_to_write"] = attributable_to("taught object stated: USE vs FREEZE (write on vs eta=0)",
                                                   n_states(U), n_states(F))
    rec["attributable_vs_exposure"] = attributable_to("taught object stated: USE vs HEARD (write vs exposure only)",
                                                      n_states(U), n_states(H))
    rec["read_path_intact"] ={n: {lbl: (lambda t: bool(t) and t.get("abstained") is False)(_turn_of(arms[n], lbl))
                                   for lbl, _q in READ_TURNS} for n in PRIMARY_ARMS}
    rec["DEFAULT_states_taught"] = {k: taught_ok(_turn_of(DF, "p_" + k), k, QA) for k in keys}
    rec["lever"] = {"USE": lu, "FREEZE": lf, "ABLATE": la, "HEARD": lh, "DEFAULT": ld}
    rec["probe_answers"] = {n: {k: (_turn_of(arms[n], "p_" + k) or {}).get("answer") for k in keys}
                            for n in PRIMARY_ARMS}
    rec["probe_qwen_calls"] = {n: {k: (_turn_of(arms[n], "p_" + k) or {}).get("n_qwen_calls") for k in keys}
                               for n in PRIMARY_ARMS}
    rec["untaught_qwen_calls"] = {n: sum((_turn_of(arms[n], uk) or {}).get("n_qwen_calls", 0) for uk, _q, _h in
                                         UNTAUGHT) for n in ("USE", "DEFAULT")}
    rec["secondary_cat_moon"] = {n: {lbl: (_turn_of(arms[n], lbl) or {}).get("answer") for lbl, _m in SECONDARY}
                                 for n in ("USE", "FREEZE", "DEFAULT")}
    rec["learned_content_fraction"] = {n: learned_content_fraction(arms[n]) for n in ("USE", "DEFAULT")}
    if arms.get("DEFAULT_LTM"):
        rec["learned_content_fraction"]["DEFAULT_LTM"] = learned_content_fraction(arms["DEFAULT_LTM"])
    rec["elapsed_s"] = {n: (arms[n] or {}).get("elapsed_s") for n in arms if arms.get(n)}
    if failed_ins:
        rec["verdict"] = "UNDEFINED (instrument failed: %s)" % ", ".join(failed_ins)
        rec["go"] = None
        return rec
    rec["go"] = all(c.values())
    rec["verdict"] = "GO" if rec["go"] else "NO-GO (failed: %s)" % ",".join(k for k, v in c.items() if not v)
    return rec


def aggregate(per_seed):
    six = {s: r for s, r in per_seed.items() if int(s) in SEEDS6}
    defined = {s: r for s, r in six.items() if r.get("go") is not None}
    n_go = sum(1 for r in defined.values() if r["go"])
    agg = {"n_validation_seeds": len(six), "n_defined": len(defined), "n_go": n_go,
           "undefined_seeds": sorted(s for s, r in six.items() if r.get("go") is None),
           "dev_seeds": {s: r.get("verdict") for s, r in per_seed.items() if int(s) not in SEEDS6}}
    if len(six) == 6 and len(defined) == 6 and n_go == 6:
        agg["verdict"] = "GO 6/6"
    elif len(six) < 6:
        agg["verdict"] = "INCOMPLETE (%d/6 validation seeds run; %d GO) -- not a verdict" % (len(six), n_go)
    elif len(defined) < 6:
        agg["verdict"] = "UNDEFINED (%d/6 validation seeds defined)" % len(defined)
    else:
        agg["verdict"] = "NO-GO %d/6" % n_go
    agg["GO"] = agg["verdict"] == "GO 6/6"
    return agg


def run(seeds, arm_dir, arms, resume=True, score_only=False, jobs=1, variant="prod"):
    os.environ["CPROV_VARIANT"] = variant                  # workers inherit it (AMENDMENT A1)
    failed = []
    for s in seeds:
        todo = []
        for name in arms:
            path = _arm_path(arm_dir, s, name)
            if score_only or (resume and os.path.exists(path)):
                continue
            todo.append((name, path))
        if todo and not score_only:
            if jobs <= 1:
                for name, path in todo:
                    print("[cprov] seed %s arm %s ..." % (s, name), flush=True)
                    if _spawn(name, s, path) is None:
                        failed.append("s%s_%s" % (s, name))
            else:
                procs = []
                for name, path in todo:
                    while sum(1 for _n, _p, pr in procs if pr.poll() is None) >= jobs:
                        time.sleep(5)
                    penv = dict(os.environ); penv["BRAIN_CHAT_SEED"] = str(s)
                    print("[cprov] seed %s arm %s (parallel) ..." % (s, name), flush=True)
                    pr = subprocess.Popen([sys.executable, "-u", "-m", "research.runners.content_provenance_probe",
                                           "--worker", "--arm", name, "--out", path], env=penv)
                    procs.append((name, path, pr))
                for name, path, pr in procs:
                    if pr.wait() != 0 or not os.path.exists(path):
                        failed.append("s%s_%s" % (s, name))
    per = {}
    for s in seeds:
        loaded = {name: _load(_arm_path(arm_dir, s, name)) for name in ARMS}
        per[str(s)] = score_seed(loaded, variant)
        print("[cprov] seed %s -> %s" % (s, per[str(s)]["verdict"]), flush=True)
    return {"runner": "research.runners.content_provenance_probe", "variant": variant, "seeds": list(seeds),
            "arm_dir": arm_dir,
            "per_seed": per, "aggregate": aggregate(per), "failed_arms": failed,
            "table": summary_table(per)}


def summary_table(per):
    rows = []
    for s, r in per.items():
        lf = r.get("learned_content_fraction") or {}
        rows.append({"seed": s, "verdict": r.get("verdict"),
                     "taught_answer": (r.get("criteria") or {}).get("CP1_taught_answer"),
                     "freeze": (r.get("criteria") or {}).get("CP2_freeze_abstains"),
                     "zero": (r.get("criteria") or {}).get("CP3_zeroed_abstains"),
                     "heard_only": (r.get("criteria") or {}).get("CP4_heard_only_abstains"),
                     "leak_count": r.get("leak_count"),
                     "fraction_DEFAULT": (lf.get("DEFAULT") or {}).get("fraction"),
                     "fraction_USE": (lf.get("USE") or {}).get("fraction"),
                     "fraction_DEFAULT_LTM": (lf.get("DEFAULT_LTM") or {}).get("fraction")})
    return rows


# ── self-test: the scorer must FAIL in every failing direction (a check that cannot fail measures nothing) ────────
def _synthetic_seed():
    keys = [k for k, *_ in ITEMS]
    objs = {k: o for k, _t, _e, _p, o, _pr in ITEMS}
    base_kb = [[0, "brain", "use", "spikes", "AFFIRM"], [1, "brain", "learn", "words", "AFFIRM"],
               [2, "brain", "store", "memory", "AFFIRM"], [3, "dog", "chase", "cat", "AFFIRM"],
               [4, "cat", "eat", "fish", "AFFIRM"]]
    verbs = {k: ("make" if k == "bee" else "eat") for k in keys}

    def t(phase, label, answer, abstained, facts=(), nq=1):
        return {"phase": phase, "label": label, "answer": answer, "abstained": abstained, "rich": True,
                "supporting_facts": [list(f) for f in facts], "recalled_svo": list(facts[0]) if facts else None,
                "n_qwen_calls": nq}

    def arm(name, taught=True, w=1.0, method="hebbian", untaught=False, found=True, ablation=False):
        kb = [list(r) for r in base_kb]
        wl = []
        if found:
            for j, k in enumerate(keys):
                kb.append([5 + j, k, verbs[k], objs[k], "AFFIRM"])
                wl.append({"block": 5 + j, "method": method, "via": "store", "phase": "teach", "turn": "t_" + k})
        turns = [t("teach", "t_" + k, "ok", False) for k in keys]
        turns += [t("read", "d1", "The dog chases the cat.", False, [("dog", "chase", "cat")]),
                  t("read", "d2", "The cat eats fish.", False, [("cat", "eat", "fish")])]
        for j, k in enumerate(keys):
            if taught:
                turns.append(t("probe", "p_" + k, "The %s %ss the %s." % (k, verbs[k], objs[k]), False,
                               [(k, verbs[k], objs[k])]))
            else:
                turns.append(t("probe", "p_" + k, "I don't know about that. My curiosity is piqued -- I haven't "
                               "learned about %s yet: what can you tell me about %s?" % (k, k), True, nq=0))
        if untaught:
            for uk, q, _hk in UNTAUGHT:
                turns.append(t("untaught", uk, "I don't know about that.", True, nq=0))
        lev = {k: {"found": found, "block": 5 + j, "mean_abs_w": (w if found else None)} for j, k in enumerate(keys)}
        a = {"arm": name, "variant": "qwenforced", "renderer_class": "QwenRenderer", "ltm_attached": False, "source": "tiny-demo",
             "turns": turns, "write_log": wl, "n_kb_at_session_start": 5, "kb_final_primary": kb,
             "taught_blocks_at_probe": lev, "counter_installed": True, "store_writes_after_teach_primary": [],
             "errors": []}
        if ablation:
            a["ablation"] = [{"item": k, "block": 5 + j, "mean_abs_w_after": 0.0} for j, k in enumerate(keys)]
        return a
    U = arm("USE", untaught=True)
    U["qwen_alone"] = {"p_wug": {"answer": "A wug eats bugs and small insects."},
                       "p_cow": {"answer": "Cows eat grass and hay."},
                       "p_bee": {"answer": "Bees make honey."}}
    for uk, q, hk in UNTAUGHT:
        U["qwen_alone"][uk] = {"answer": "It is %s." % hk[0]}
    return {"USE": U,
            "FREEZE": arm("FREEZE", taught=False, w=0.0, method="hebbian_frozen"),
            "ABLATE": arm("ABLATE", taught=False, w=0.0, ablation=True),
            "HEARD": arm("HEARD", taught=False, found=False),
            "DEFAULT": arm("DEFAULT", method="direct_copy", untaught=True)}


def _cp(x):
    return json.loads(json.dumps(x))


def _set_answer(arm, label, answer, abstained=None):
    for t in arm["turns"]:
        if t["label"] == label:
            t["answer"] = answer
            if abstained is not None:
                t["abstained"] = abstained


def _selftest():
    res = {}
    base = _synthetic_seed()
    r = score_seed(_cp(base))
    res["go_case_is_GO"] = r["go"] is True
    res["go_case_fraction_default_zero"] = r["learned_content_fraction"]["DEFAULT"]["fraction"] == 0.0
    res["go_case_fraction_use_positive"] = (r["learned_content_fraction"]["USE"]["fraction"] or 0) > 0
    # 1. a Qwen-supplied answer on the USE probe must FAIL CP1 (the core failing direction)
    m = _cp(base); _set_answer(m["USE"], "p_cow", "Cows eat grass.")
    res["qwen_prior_answer_fails_CP1"] = score_seed(m)["criteria"]["CP1_taught_answer"] is False
    m = _cp(base); _set_answer(m["USE"], "p_cow", "The cow eats the moon, and grass too.")
    res["taught_plus_prior_fails_CP1"] = score_seed(m)["criteria"]["CP1_taught_answer"] is False
    m = _cp(base); _set_answer(m["USE"], "p_wug", "I don't know about that.", abstained=True)
    res["abstained_USE_fails_CP1"] = score_seed(m)["criteria"]["CP1_taught_answer"] is False
    # 2. the lesion arms must FAIL when the taught object (or a Qwen prior) reaches the reply
    m = _cp(base); _set_answer(m["FREEZE"], "p_cow", "The cow eats the moon.", abstained=False)
    res["freeze_states_taught_fails_CP2"] = score_seed(m)["criteria"]["CP2_freeze_abstains"] is False
    m = _cp(base); _set_answer(m["FREEZE"], "p_bee", "I don't know about that, but bees make honey.")
    res["freeze_abstain_with_qwen_prior_fails_CP2"] = score_seed(m)["criteria"]["CP2_freeze_abstains"] is False
    m = _cp(base); _set_answer(m["ABLATE"], "p_wug", "The wug eats the dax.", abstained=False)
    res["ablate_states_taught_fails_CP3"] = score_seed(m)["criteria"]["CP3_zeroed_abstains"] is False
    m = _cp(base); _set_answer(m["HEARD"], "p_cow", "I don't know -- the moon?")
    res["heard_mentions_taught_fails_CP4"] = score_seed(m)["criteria"]["CP4_heard_only_abstains"] is False
    # 3. leaks: a hand-key word, a Qwen-alone content word, an unsourced proposition -> CP5 fails; a template does not
    m = _cp(base); _set_answer(m["USE"], "u11", "Lions eat meat.", abstained=False)
    res["hand_key_leak_fails_CP5"] = score_seed(m)["criteria"]["CP5_no_qwen_leak"] is False
    m = _cp(base); m["USE"]["qwen_alone"]["u03"] = {"answer": "A wug drinks lemonade."}
    _set_answer(m["USE"], "u03", "I don't know about that. Perhaps lemonade.")
    res["qwen_alone_word_leak_fails_CP5"] = score_seed(m)["criteria"]["CP5_no_qwen_leak"] is False
    m = _cp(base)
    for tt in m["DEFAULT"]["turns"]:
        if tt["label"] == "u05":
            tt.update({"answer": "The dog eats the ball.", "abstained": False, "supporting_facts": [["dog", "eat",
                       "ball"]], "recalled_svo": ["dog", "eat", "ball"]})
    res["unsourced_prop_fails_CP5"] = score_seed(m)["criteria"]["CP5_no_qwen_leak"] is False
    m = _cp(base); _set_answer(m["USE"], "u01", "I don't know about that. My curiosity is piqued -- I haven't learned "
                               "about cow yet: what can you tell me about cow?")
    res["template_abstain_is_not_a_leak"] = score_seed(m)["criteria"]["CP5_no_qwen_leak"] is True
    # 4. instruments: a failed instrument is UNDEFINED, never GO
    m = _cp(base); m["FREEZE"]["taught_blocks_at_probe"]["cow"]["mean_abs_w"] = 0.7
    res["freeze_lever_not_zero_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); m["ABLATE"]["taught_blocks_at_probe"]["bee"]["mean_abs_w"] = 1.0
    res["ablate_lever_not_zero_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); m["HEARD"]["taught_blocks_at_probe"]["wug"] = {"found": True, "mean_abs_w": 0.0}
    res["heard_block_found_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); m["FREEZE"]["store_writes_after_teach_primary"] = [{"block": 6}]
    res["lesion_later_write_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); m["DEFAULT"]["renderer_class"] = "StubRenderer"
    res["stub_renderer_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); m["USE"]["ltm_attached"] = True
    res["ltm_attached_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); del m["ABLATE"]
    res["missing_arm_undefined"] = score_seed(m)["go"] is None
    m = _cp(base)
    for tt in m["USE"]["turns"]:
        if tt["phase"] == "probe":
            tt["n_qwen_calls"] = 0
    res["no_qwen_render_undefined"] = score_seed(m)["go"] is None
    m = _cp(base); m["USE"]["turns"][0]["rich"] = False
    res["non_rich_turn_undefined"] = score_seed(m)["go"] is None
    # AMENDMENT A1: in 'prod' zero Qwen calls on the probes is a measurement, not an instrument failure
    m = _cp(base)
    for nm in m:
        m[nm]["variant"] = "prod"
    for tt in m["USE"]["turns"]:
        if tt["phase"] == "probe":
            tt["n_qwen_calls"] = 0
    rp = score_seed(m, "prod")
    res["prod_zero_qwen_calls_is_scored"] = rp["go"] is True and rp["use_probe_qwen_calls"] == 0
    res["variant_mismatch_undefined"] = score_seed(_cp(base), "prod")["go"] is None
    # 5. FD1/FD2 are computed from the Qwen-alone answers and must hold on the synthetic seed
    res["FD1_holds_on_synthetic"] = r["instrument"]["FD1_ok"] is True
    res["FD2_holds_on_synthetic"] = r["instrument"]["FD2_ok"] is True
    m = _cp(base); m["USE"]["qwen_alone"]["p_cow"] = {"answer": "The cow eats the moon."}
    res["FD1_flags_uninformative_item"] = score_seed(m)["instrument"]["counterfactual_valid"]["cow"] is False
    # 6. the fraction counts classes correctly and is UNDEFINED on no stated proposition
    fr = r["learned_content_fraction"]["USE"]
    res["fraction_use_is_3_of_5"] = (fr["n_plasticity"], fr["n"]) == (3, 5)
    res["fraction_undefined_when_empty"] = learned_content_fraction({"turns": []})["fraction"] is None
    # 7. aggregate: a partial seed set is INCOMPLETE, never GO/NO-GO
    res["aggregate_partial_incomplete"] = aggregate({"42": {"go": True}})["verdict"].startswith("INCOMPLETE")
    res["aggregate_dev_seed_not_counted"] = aggregate({"7": {"go": True, "verdict": "GO"}})["n_go"] == 0
    ok = all(res.values())
    return ok, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[DEV_SEED])
    ap.add_argument("--arms", nargs="+", default=list(PRIMARY_ARMS))
    ap.add_argument("--arm-dir", default="research/findings/raw/_content_provenance")
    ap.add_argument("--json", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--variant", choices=VARIANTS, default="prod")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        ok, res = _selftest()
        for k, v in res.items():
            print("  %-45s %s" % (k, "ok" if v else "FAIL"))
        print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    if a.worker:
        return _worker(a.arm, a.out)
    out = run(a.seeds, a.arm_dir, a.arms, resume=not a.no_resume, score_only=a.score_only, jobs=a.jobs,
              variant=a.variant)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    print(json.dumps({"aggregate": out["aggregate"], "table": out["table"], "failed_arms": out["failed_arms"]},
                     indent=1, default=str))
    return 4 if out["failed_arms"] else 0


if __name__ == "__main__":
    sys.exit(main())
