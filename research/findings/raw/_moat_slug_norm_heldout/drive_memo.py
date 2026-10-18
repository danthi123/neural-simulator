"""Smallest harness over the probe's own pieces (turn table + the battery's _collect_worker -> webapp.server.brain_chat).
usage (cwd = a worktree; run with that worktree's .venv):
  drive.py --labels a2_c1_mood,a2_c1_q --out OUT.json [--renderer stub|qwen-cpu] [--extra JSON]
 stub     : the battery worker's default stub mode (renderer='stub', numpy).
 qwen-gpu : (added 2026-10-18 for turns 17-29 of the held-out run) the same as qwen-cpu with CUDA VISIBLE: the faculty then
            runs fp16 on the GPU, the production device. Everything else is unchanged.
 replay   : (added 2026-10-18 for the composer-path measurement) the production brain + rich path with the Qwen renderer REPLACED
            by a replay of the recorded Qwen sentences of --replay-corpus (render_svo / render_svo_regen served by (turn, svo,
            method)); a request with no recorded sentence is logged in out['replay_misses'] and gets an empty surface.  No model is
            loaded; every verify/accept/abstain decision is the production code's on the production brain.
 qwen-cpu : LB_ARM_RENDERER=production (renderer omitted = console 'auto') with webapp.server._default_brain_renderer
            forced to 'qwen' (the only GPU-dependent step) and CUDA hidden -> the production QwenRenderer on CPU.
 --extra  : JSON list of [label, message, session, rich] fresh-session turns (reset=True) appended to the turn table.
Trace wrappers (transparent, same args/returns) record ChatBrain.gate / _verify / _verify_claim_set and
RichAnswerComposer._direct_fact / render_paragraph into out['trace'].
"""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--renderer", default="stub", choices=["stub", "qwen-cpu", "qwen-gpu", "replay"])
ap.add_argument("--replay-corpus", default=None, help="replay: corpus.json whose recorded Qwen sentences the replay renderer serves")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--extra", default="[]")
a = ap.parse_args()
sys.path.insert(0, os.getcwd())
os.environ["SIM_BACKEND"] = "numpy"
os.environ["BRAIN_CHAT_SEED"] = str(a.seed)
os.environ["BRAIN_MOUTH_TRACE"] = "1"
if a.renderer in ("qwen-cpu", "qwen-gpu", "replay"):
    if a.renderer in ("qwen-cpu", "replay"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # never touch the GPU
    os.environ["LB_ARM_RENDERER"] = "production"
import research.runners.onebrain_regression_battery as B
import research.runners.b2d_amend2_probe as P
table, order = P.turn_table(B._TURN_BY_LABEL["emo"][1])
for ex in json.loads(a.extra):
    lab, msg, sess, rich = ex[:4]
    reset = ex[4] if len(ex) > 4 else True
    table[lab] = (lab, msg, sess, bool(reset), None, bool(rich))
B._TURN_BY_LABEL.update(table)
labels = a.labels.split(",")
for l in labels:
    assert l in table, l

TRACE = []
CUR = {"label": None}


def _wrap(cls, name, fmt):
    orig = getattr(cls, name)

    def w(self, *args, **kw):
        r = orig(self, *args, **kw)
        try:
            TRACE.append(dict(fmt(args, kw, r), fn=name, turn=CUR["label"]))
        except Exception as e:
            TRACE.append({"fn": name, "trace_error": repr(e)})
        return r
    w.__name__ = name
    w.__wrapped__ = orig
    setattr(cls, name, w)


from research.runners.brain_chat_tui import ChatBrain
from research.runners.rich_answer_composer import RichAnswerComposer
_wrap(ChatBrain, "gate", lambda a_, k, r: {"q": a_[0], "ret": r if r is None else list(r)})
_wrap(ChatBrain, "_verify", lambda a_, k, r: {"surface": a_[0], "svo": list(a_[2]), "ok": r})
_wrap(ChatBrain, "_verify_claim_set",
      lambda a_, k, r: {"surface": a_[0], "gated": [list(f) for f in a_[1]], "ok": r[0],
                        "reject": (r[1] or {}).get("reject_reason"), "trace": (r[1] or {}).get("trace")})
_wrap(RichAnswerComposer, "_direct_fact", lambda a_, k, r: {"q": a_[0], "ret": r if r is None else list(r)})
_wrap(RichAnswerComposer, "render_paragraph",
      lambda a_, k, r: {"n_facts": len(a_[0]), "facts": [list(f) for f in a_[0]], "paragraph": r[0],
                        "kept": [list(f) for f in r[1]], "dropped": [list(f) for f in r[2]]})

# the COMPOSER PATH itself: RichAnswerComposer._verify_rendered records, per rendered candidate, the number of gathered facts, which
# sub-verifier decided (claim gate vs the single-triple chat._verify fallback), and the verdict.  The nested ChatBrain wrappers above
# append to the same record through CURV.
CURV = {"rec": None}
_orig_vr = RichAnswerComposer._verify_rendered
_orig_vc = ChatBrain._verify_claim_set
_orig_v1 = ChatBrain._verify


def _vc(self, surface, gated_facts):
    r = _orig_vc(self, surface, gated_facts)
    if CURV["rec"] is not None:
        CURV["rec"]["claim"] = None if r[0] is None else bool(r[0])
        CURV["rec"]["claim_reject"] = (r[1] or {}).get("reject_reason")
    return r


def _v1(self, surface, asserted, gate_svo):
    r = _orig_v1(self, surface, asserted, gate_svo)
    if CURV["rec"] is not None:
        CURV["rec"]["single_triple"] = bool(r)
        CURV["rec"]["asserted_given"] = asserted is not None
    return r


def _vr(self, surface, asserted, svo, gated):
    rec = {"fn": "_verify_rendered", "turn": CUR["label"], "surface": surface, "svo": list(svo),
           "n_gated": None if gated is None else len(gated), "claim": "not_called", "single_triple": "not_called",
           "vocab": [len(getattr(self.chat, "agents_set", ()) or ()), len(getattr(self.chat, "actions_set", ()) or ()),
                     len(getattr(self.chat, "patients_set", ()) or ())]}
    CURV["rec"] = rec
    try:
        r = _orig_vr(self, surface, asserted, svo, gated)
    finally:
        CURV["rec"] = None
    rec["ok"] = bool(r)
    TRACE.append(rec)
    return r


ChatBrain._verify_claim_set = _vc
ChatBrain._verify = _v1
RichAnswerComposer._verify_rendered = _vr

cap = None
REPLAY_MISSES = []
if a.renderer == "replay":
    assert a.replay_corpus, "--renderer replay needs --replay-corpus"
    import webapp.server as S
    S._default_brain_renderer = lambda: "qwen"
    _rc = json.load(open(a.replay_corpus))
    _served = {}
    for _t in _rc["turns"]:
        for _c in _t.get("calls", []):
            _served.setdefault((_t["label"], tuple(_c["svo"]), _c["method"]), _c["surface"])

    class ReplayRenderer:
        name = "replay of recorded Qwen-0.5B sentences"

        def _serve(self, method, a_, v_, p_):
            k = (CUR["label"].split("|")[0], (a_, v_, p_), method)
            if k in _served:
                return _served[k], None
            REPLAY_MISSES.append({"turn": k[0], "svo": [a_, v_, p_], "method": method})
            return "", None

        def render_svo(self, a_, v_, p_):
            return self._serve("render_svo", a_, v_, p_)

        def render_svo_regen(self, a_, v_, p_):
            return self._serve("render_svo_regen", a_, v_, p_)

    S._get_warm_qwen_renderer = lambda: ReplayRenderer()
if a.renderer in ("qwen-cpu", "qwen-gpu"):
    import webapp.server as S
    S._default_brain_renderer = lambda: "qwen"
    import torch
    assert torch.cuda.is_available() == (a.renderer == "qwen-gpu"), "CUDA visibility does not match --renderer"
    from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
    cap = P.PromptCapture().install(SpikingQwenFaculty)
# learn the current turn from the request the worker builds
_orig_kw = B._arm_request_kwargs


def _kw(mode, session, msg, reset, percept, rich):
    CUR["label"] = "%s|%s|rich=%s" % (session, msg, rich)
    return _orig_kw(mode, session, msg, reset, percept, rich)


B._arm_request_kwargs = _kw
tmp = a.out + ".worker.json"
env = {"BRAIN_MOUTH_TRACE": "1"}
env.update(json.loads(os.environ.get("DRIVE_ENV", "{}")))
rc = B._collect_worker(json.dumps(env), labels, tmp)
if rc != 0:
    raise SystemExit(rc)
resp = json.load(open(tmp))
os.remove(tmp)
import platform, socket
out = {"labels": labels, "renderer": a.renderer, "responses": resp, "trace": TRACE, "host": socket.gethostname(),
       "brain_chat_seed": a.seed, "drive_env": env, "replay_corpus": a.replay_corpus, "replay_misses": REPLAY_MISSES}
if a.renderer == "qwen-gpu":
    out["device"] = "cuda:0 " + torch.cuda.get_device_name(0) + " (Qwen fp16)"
else:
    out["device"] = "cpu (%s)" % (platform.processor() or platform.machine())
if cap is not None:
    out["qwen_calls"] = cap.calls
import webapp.server as S2
out["ltm_bundle"] = S2._resolve_ltm_bundle()
out["shared_ltm_on"] = S2._shared_ltm_on() if hasattr(S2, "_shared_ltm_on") else None
json.dump(out, open(a.out, "w"), indent=1, default=str)
print("[drive] wrote", a.out, flush=True)
