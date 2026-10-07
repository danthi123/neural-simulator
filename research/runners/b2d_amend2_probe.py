"""B2d AMENDMENT 2 PROBE (registered; research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md,
"Amendment 2"): does an affect faculty's lesion change what the production Qwen mouth SAYS, on a turn the mouth
actually speaks while the mood is engaged?

Why a new probe. The registered mouth arm (A1.2) read MOUTH-SILENT on 17 of 18 cells: its turns (`emo`, an abstention
template, and `well`, an assertion echo) never call Qwen, and at the console defaults a bounded transitive recall is
spoken by the spiking recall mouth, not Qwen. The route map (research/packets/b2d-mouth-renderer/QWEN-ROUTE-MAP.md)
shows where Qwen does speak at defaults: a recalled fact the spiking recall mouth cannot frame (a multi-word or
non-alphabetic role, a copula verb, subject == object), which is every long-term-memory (wikidata) fact. So each
candidate here is a two-turn group in ONE session: the battery's own `emo` message (sets the session mood), then a
question about a long-term-memory topic, either single-fact (`rich=False`, what the console sends by default) or rich
(`rich=True`, what the API and the OpenAI shim get when `rich` is omitted). Which candidate is scored is chosen on the
dev seeds by an outcome-blind rule (tools/b2d_amend2_score.py `dev`); every candidate runs in every arm, and all are
reported.

One arm per process (a fresh brain, like every battery arm), through the battery's own worker
(`onebrain_regression_battery._collect_worker`, so LB_ARM_RENDERER=production renders exactly as the mouth arm does:
the console's auto request, the process-wide warm QwenRenderer, the per-turn Qwen call counter). The candidate turns
are added to the worker's turn table in THIS process only; the harness file is not changed. Every arm sets
BRAIN_MOUTH_TRACE=1, which attaches the reply's `mouth_trace` block (each render call, its SVO, the manner clause
appended to the Qwen prompt, the surface, whether it was spoken) and changes no reply text. This process also records,
per turn, the exact prompts the Qwen faculty received (a transparent wrapper on SpikingQwenFaculty._generate /
_generate_batch, installed before the worker's counter; split per turn by the counter's per-turn counts).

  SIM_BACKEND=cupy LB_ARM_RENDERER=production .venv/bin/python -u -m research.runners.b2d_amend2_probe \
      --arm intact_a --seed 1 --pin <40-hex P2> --out research/findings/raw/_b2d_amend2/<P2_7>/dev/s1/intact_a.json

The arm file records the export's .source_revision, whether its manifest verifies, the learned valence map the affect
organ loaded (entries and sha256; the 18 A1.2 cells ran without it), the LTM bundle, and the responses. GPU only in
use: on numpy the worker would resolve the stub renderer, which the scorer reads as class M.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

DEV_SEEDS = tuple(range(1, 9))
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
TRACE_ENV = {"BRAIN_MOUTH_TRACE": "1"}

# The three affect rows of the mouth arm (A1.2) and their registered lesion flags (research/runners/
# load_bearing_fraction.py FACULTY_LESIONS and research/runners/lbf_rows/live_organs.py).
LESIONS = {
    "affect-coloring": ("coloring", {"BRAIN_AFFECT_LESION": "1"}),
    "affect-appraisal-interoceptive": ("appraisal", {"BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION": "1"}),
    "affect-drives-response": ("drives", {"BRAIN_AFFECT_DRIVES_LESION": "1"}),
}


def arms():
    """{arm name: env}. `route` is the route-check arm (intact env; run first on every seed; a third null replicate).
    Each lesion has a replicate (`_rep0`), the lesion floor."""
    a = {"route": {}, "intact_a": {}, "intact_b": {}}
    for _row, (short, env) in LESIONS.items():
        a["les_%s" % short] = dict(env)
        a["les_%s_rep0" % short] = dict(env)
    return {k: dict(TRACE_ENV, **v) for k, v in a.items()}


ARMS = arms()

# Declared before any Amendment-2 output exists. Order matters: the dev rule selects the FIRST eligible candidate.
# The topics are in both shipped bundles (wikidata_100k, the default, and wikidata_core_15k); the per-touchpoint
# measurement (research/findings/raw/_per_touchpoint_qwen_share_shipped_default_AFTER_recall_gate_fix.json) found
# their rich turns rendered entirely by the fluent renderer (spiking_hit_count 0).
CANDIDATES = (
    {"id": "c1", "question": "What is l_quipe_de_france?", "rich": False},
    {"id": "c2", "question": "Tell me about angora_turkey.", "rich": False},
    {"id": "c3", "question": "What is l_quipe_de_france?", "rich": True},
    {"id": "c4", "question": "Tell me about angora_turkey.", "rich": True},
)


def mood_label(cid):
    return "a2_%s_mood" % cid


def q_label(cid):
    return "a2_%s_q" % cid


def turn_table(emo_msg):
    """{label: (label, message, session, reset, percept, rich)} in the worker's tuple shape, and the ordered labels."""
    table, order = {}, []
    for c in CANDIDATES:
        sess = "a2_%s" % c["id"]
        table[mood_label(c["id"])] = (mood_label(c["id"]), emo_msg, sess, True, None, False)
        table[q_label(c["id"])] = (q_label(c["id"]), c["question"], sess, False, None, bool(c["rich"]))
        order += [mood_label(c["id"]), q_label(c["id"])]
    return table, order


class PromptCapture:
    """Records every prompt list the Qwen faculty receives, without changing a call (same arguments, same return)."""

    def __init__(self):
        self.calls = []
        self._orig = {}

    def install(self, cls):
        for name in ("_generate", "_generate_batch"):
            orig = getattr(cls, name, None)
            if orig is None or name in self._orig:
                continue
            self._orig[name] = orig
            setattr(cls, name, self._wrap(name, orig))
        return self

    def _wrap(self, name, orig):
        cap = self

        def _wrapped(*args, **kwargs):
            try:
                msgs = args[1] if len(args) > 1 else kwargs.get("user_msgs", kwargs.get("user_msg"))
                cap.calls.append({"method": name,
                                  "prompts": list(msgs) if name == "_generate_batch" else [msgs]})
            except Exception as e:   # a capture failure is recorded, never raised (the call itself is unchanged)
                cap.calls.append({"method": name, "prompts": None, "error": "%s: %s" % (type(e).__name__, e)})
            return orig(*args, **kwargs)
        _wrapped.__name__ = getattr(orig, "__name__", name)
        _wrapped.__wrapped__ = orig
        return _wrapped

    def uninstall(self, cls):
        for name, orig in self._orig.items():
            setattr(cls, name, orig)
        self._orig = {}


def split_prompts(calls, order, responses, counter_key="_arm_renderer"):
    """Assign the captured calls to turns in order, using each turn's counted Qwen calls. Returns ({label: [call]},
    leftover). A leftover or a shortfall means the capture and the counter disagree (the scorer's ROUTE-MISMATCH)."""
    out, i = {}, 0
    for label in order:
        n = int(((responses.get(label) or {}).get(counter_key) or {}).get("qwen_generate_calls") or 0)
        out[label] = calls[i:i + n]
        i += n
    return out, len(calls) - i


def mouth_attribution(label, intact_a_mouth, intact_b_mouth, lesion_mouth):
    """Whose is a mouth-reply change: the lesion's (treatment: intact_a vs the lesion arm) or the rebuild's (control:
    intact_a vs intact_b). `tools.lab.attributable_to` over the number of (SVO, surface) positions that differ. None
    when neither moved (UNDEFINED, never 0). Used by tools/b2d_amend2.py on every DEFINED cell; its print goes to the
    returned record, not stdout."""
    import contextlib
    import io
    from tools.lab import attributable_to

    def ndiff(x, y):
        n = max(len(x), len(y))
        return sum(1 for i in range(n) if (x[i] if i < len(x) else None) != (y[i] if i < len(y) else None))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        frac = attributable_to(label, ndiff(intact_a_mouth, lesion_mouth), ndiff(intact_a_mouth, intact_b_mouth))
    return {"attributable": frac, "note": buf.getvalue().strip()}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def revision_record(root="."):
    """The export's .source_revision fields and whether its source manifest verifies (the A1.2 rules E1-E3 shape)."""
    rec = {"source_revision_present": False}
    sr = os.path.join(root, ".source_revision")
    if os.path.exists(sr):
        rec["source_revision_present"] = True
        for ln in open(sr):
            if "=" in ln:
                k, v = ln.rstrip("\n").split("=", 1)
                rec[k] = v
        man = os.path.join(root, ".source_manifest.sha256")
        tool = os.path.join(root, "tools", "pool", "provisioning", "source_manifest.py")
        if os.path.exists(man) and os.path.exists(tool) and rec.get("source_manifest_sha256"):
            p = subprocess.run([sys.executable, tool, "verify", "--root", root, "--manifest", man,
                                "--expected-sha256", rec["source_manifest_sha256"]], capture_output=True, text=True)
            rec["manifest_verified"] = p.returncode == 0
    return rec


def valence_map_record():
    from research.runners import affect_production_organ as AO
    path = AO._LEARNED_MAP_PATH
    rec = {"path": os.path.relpath(path), "exists": os.path.exists(path),
           "entries": len(AO._get_learned_valence() or {}), "dr2_enabled": bool(AO.dr2_enabled())}
    if rec["exists"]:
        rec["sha256"] = _sha256(path)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--pin", required=True, help="the 40-hex revision this export is (recorded; the scorer checks it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ["BRAIN_CHAT_SEED"] = str(a.seed)   # the substrate seed (webapp.server._brain_chat_seed)
    import research.runners.onebrain_regression_battery as B
    table, order = turn_table(B._TURN_BY_LABEL["emo"][1])
    B._TURN_BY_LABEL.update(table)
    cap = None
    if B.arm_renderer_mode() == B.ARM_RENDERER_PRODUCTION:
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        cap = PromptCapture().install(SpikingQwenFaculty)   # installed BEFORE the worker's counter (outermost there)
    tmp = a.out + ".worker.json"
    rc = B._collect_worker(json.dumps(ARMS[a.arm]), list(order), tmp)
    if rc != 0:
        return rc
    responses = json.load(open(tmp))
    os.remove(tmp)
    prompts, leftover = split_prompts(cap.calls if cap else [], order, responses)
    try:
        import webapp.server as S
        ltm = S._resolve_ltm_bundle()
    except Exception as e:  # recorded, never fatal
        ltm = "unresolved: %s" % e
    out = {"probe": "b2d-amend2", "arm": a.arm, "arm_env": ARMS[a.arm], "seed": a.seed, "pin": a.pin,
           "renderer_mode": B.arm_renderer_mode(), "backend": os.environ.get("SIM_BACKEND"),
           "revision": revision_record("."), "valence_map": valence_map_record(), "ltm_bundle": ltm,
           "candidates": list(CANDIDATES), "turns": {t: list(table[t][1:]) for t in order},
           "responses": responses, "qwen_prompts": prompts, "qwen_prompts_unassigned": leftover}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("[b2d-amend2] arm=%s seed=%d -> %s" % (a.arm, a.seed, a.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
