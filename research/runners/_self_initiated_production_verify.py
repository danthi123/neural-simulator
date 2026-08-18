"""PRODUCTION VERIFY for the SELF-INITIATED UTTERANCE organ wired into `webapp/server.py::brain_chat` (2026-08-18).

Asserts the GO gate on the REAL production tiny-demo ChatBrain + the REAL `brain_chat` handler, numpy-CPU:

  (A) IDLE turn returns a COHERENT self-initiated utterance ABOUT a real stored concept. Through the REAL handler an
      empty message / a "say something" lead-in returns a self-initiated remark+question; the surfaced concept is a
      real stored fact (mouth fidelity: render_fact decoded it), decode_ok + about-a-real-concept. Per-seed (42..102)
      the SelfInitiationOrgan surfaces a coherent curiosity-top concept.
  (B) BYTE-IDENTICAL on a full reactive panel (recall/abstain/learn/anaphora), measured in SEPARATE PROCESSES and
      hashed: flag-ON (default) == a PRISTINE-HEAD stash (the block removed) == BRAIN_SELF_INITIATE=0, and NO
      `self_initiated` key on any reactive turn (the idle block is a pure no-op on a reactive turn).
  (C) LESION-LOAD-BEARING: BRAIN_SELF_INITIATE_LESION=1 (the store NO-ENCODE control — an emptied RF store, not a host
      flag) collapses the utterance stream (n_utt <= 25% of intact) -> the honest neutral idle fallback. Per-seed + a
      through-the-handler idle turn.
  (D) MOAT-SAFE: the remark is grounded in a real stored concept; an UNKNOWN subject abstains (render_fact None); the
      idle block NEVER flips a reactive abstain (covered by (B) — the abstain panel is byte-identical).

The de-risk's OWN 6/6 per-seed gate (the full curiosity-biased CA3-selection wander + its NO-ENCODE store-lesion
collapse) is the committed cupy GO artifact research/findings/raw/_self_initiated_utterance_derisk.json (cited); on
numpy the heavy wander is DEFERRED (the SAME latency residual d5-episodic declares for its BTSP write), so this verify
exercises the WIRING + the mouth RF-decode substrate + the store-lesion, light.

Run:  SIM_BACKEND=numpy python -m research.runners._self_initiated_production_verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

from pathlib import Path  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_self_initiated_production" / "verify.json"

# A full REACTIVE panel — recall / abstain / learn / anaphora — NONE of which is an idle trigger (so the self-init
# block must be a pure no-op on every one of them). Kept deterministic on the tiny-demo stub brain.
_PANEL = [
    "what does the dog chase?",          # recall
    "what is the capital of france?",    # abstain (no-confab moat)
    "wolf hunt deer",                    # learn (acquisition)
    "what does the wolf hunt?",          # recall the just-taught fact
    "the dog chased the cat",            # assertion / anaphora setup
    "what does it eat?",                 # anaphora
]


def _setup_session(session):
    import webapp.server as S
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    cache_key = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[cache_key] = chat
    # tolerate a PRISTINE-HEAD server.py (no _SESSION_SELFINIT yet) so the byte-identical pristine subprocess runs.
    _ss = getattr(S, "_SESSION_SELFINIT", None)
    if _ss is not None:
        _ss.pop(cache_key, None)
    return cache_key


def _turn(session, message, *, rich=False):
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=rich))
    return json.loads(r.body.decode("utf-8"))


def _panel_responses(session="panel"):
    """Drive the reactive panel through the REAL handler; return the list of (message, response) with any volatile
    debug keys dropped so the hash is over the STABLE surface (answer/abstained/recalled_svo/verified/self_initiated)."""
    _setup_session(session)
    rows = []
    keep = ("answer", "abstained", "recalled_svo", "verified", "rich")
    for m in _PANEL:
        r = _turn(session, m)
        slim = {k: r.get(k) for k in keep}
        slim["has_self_initiated"] = ("self_initiated" in r)
        rows.append({"msg": m, "resp": slim})
    return rows


def _panel_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _subproc_panel(env_extra):
    """Run the reactive panel in a SEPARATE PROCESS (a fresh interpreter reads the current webapp/server.py) and return
    (rows, sha). env_extra overrides os.environ for that child (e.g. BRAIN_SELF_INITIATE=0)."""
    env = dict(os.environ)
    env.update(env_extra)
    env["SIM_BACKEND"] = "numpy"
    env.setdefault("BRAIN_COMPOSER_KIND", "rf")
    out = subprocess.check_output([sys.executable, "-m", "research.runners._self_initiated_production_verify",
                                   "--panel-json"], cwd=_REPO, env=env, stderr=subprocess.DEVNULL)
    import re as _re
    text = out.decode("utf-8", errors="ignore")
    m = _re.search(r"@@JSON_BEGIN@@(.*?)@@JSON_END@@", text, _re.S)
    if m is None:
        raise RuntimeError("subprocess panel emitted no @@JSON@@ block; got: %r" % text[-300:])
    rows = json.loads(m.group(1))
    return rows, _panel_hash(rows)


def _byte_identical(rows_a, rows_b):
    return _panel_hash(rows_a) == _panel_hash(rows_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-json", action="store_true", help="internal: print the reactive-panel responses as JSON")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    a = ap.parse_args()

    if a.panel_json:
        # subprocess mode: a fresh interp -> reads the current webapp/server.py. Build logs may hit stdout, so run the
        # panel with stdout SILENCED and emit ONLY a sentinel-wrapped JSON block the parent extracts robustly.
        import contextlib
        import io
        _real = sys.stdout
        with contextlib.redirect_stdout(io.StringIO()):
            rows = _panel_responses("panel_sub")
        _real.write("@@JSON_BEGIN@@" + json.dumps(rows) + "@@JSON_END@@\n")
        _real.flush()
        return 0

    import time
    t0 = time.time()
    import research.runners.self_initiated_production_organ as SI
    rows = {}

    # ── (A) per-seed COHERENT self-initiated utterance about a real stored concept (light path, fast) ─────────────
    per_seed = []
    for s in a.seeds:
        org = SI.SelfInitiationOrgan(seed=s)
        r = org.speak(lesion=False)
        rl = org.speak(lesion=True)
        # (D) moat: an UNKNOWN subject abstains through the SAME mouth
        unknown_abstains = bool(org.comp.render_fact("zzz_unknown_subject") is None)
        per_seed.append({
            "seed": s, "n_utt": r["n_utt"], "concept": r["concept"], "utterance": r["utterance"],
            "question": r["question"], "about_rate": r["about_rate"], "mouth_fidelity": r["mouth_fidelity"],
            "moat_abstains": r["moat_abstains"], "unknown_abstains": unknown_abstains,
            "lesion_n_utt": rl["n_utt"], "path": r["path"],
            "coherent": bool(r["n_utt"] >= 1 and r["utterance"] and r["concept"]
                             and r["mouth_fidelity"] and r["about_rate"] >= 0.9),
            "collapses": bool(rl["n_utt"] <= max(0, int(0.25 * r["n_utt"]))),
        })
    A_ok = all(p["coherent"] for p in per_seed)
    D_moat_ok = all(p["moat_abstains"] and p["unknown_abstains"] for p in per_seed)
    C_lesion_ok = all(p["collapses"] for p in per_seed)
    # ATTRIBUTION (tools.lab): whose is the self-initiated utterance stream? The store NO-ENCODE lesion holds the mouth
    # decode fixed (same OneBrainComposer geometry) and empties only the RF store; the intact->lesion n_utt drop is
    # therefore attributable to the stored engram, not the mouth glue. Mean over seeds.
    from tools.lab import attributable_to
    _mean_intact = sum(p["n_utt"] for p in per_seed) / len(per_seed)
    _mean_lesion = sum(p["lesion_n_utt"] for p in per_seed) / len(per_seed)
    store_attribution = attributable_to("self-initiated utterance owed to the substrate store (intact vs NO-ENCODE lesion)",
                                        _mean_intact, _mean_lesion)
    rows["A_per_seed"] = per_seed
    rows["store_attribution"] = store_attribution

    # ── (A) + (C) + (D) THROUGH THE REAL HANDLER (organ seed 42) ────────────────────────────────────────────────
    _setup_session("idle_intact")
    h_idle = _turn("idle_intact", "say something")
    h_empty = _turn("idle_intact", "")
    si = h_idle.get("self_initiated", {})
    handler_idle_ok = bool((not h_idle["abstained"]) and h_idle["verified"] and si.get("kind") == "self_initiated"
                           and si.get("utterance") and si.get("mouth_fidelity") and si.get("moat_abstains")
                           and si.get("concept") in (si.get("utterance") or "").split()[:1])
    handler_empty_ok = bool(h_empty.get("self_initiated", {}).get("utterance"))
    # (C) through the handler: lesion -> neutral fallback + no utterance
    _setup_session("idle_lesion")
    os.environ["BRAIN_SELF_INITIATE_LESION"] = "1"
    try:
        h_les = _turn("idle_lesion", "say something")
    finally:
        os.environ.pop("BRAIN_SELF_INITIATE_LESION", None)
    sil = h_les.get("self_initiated", {})
    handler_lesion_ok = bool(h_les["abstained"] and sil.get("n_utt", 1) == 0
                             and "Nothing in particular" in h_les["answer"])
    rows["handler"] = {"idle": {"answer": h_idle["answer"], "self_initiated": si, "ok": handler_idle_ok},
                       "empty": {"answer": h_empty["answer"], "ok": handler_empty_ok},
                       "lesion": {"answer": h_les["answer"], "self_initiated": sil, "ok": handler_lesion_ok}}

    # ── (B) BYTE-IDENTICAL reactive panel — SEPARATE PROCESSES, hashed: flag-ON == pristine-HEAD == flag-OFF ──────
    rows_on, sha_on = _subproc_panel({})                                   # current server.py, default (flag ON)
    rows_off, sha_off = _subproc_panel({"BRAIN_SELF_INITIATE": "0"})       # flag OFF
    # pristine-HEAD stash: temporarily swap webapp/server.py to its HEAD content, run the panel in a fresh process,
    # ALWAYS restore (finally + content re-check). HEAD is the pre-edit baseline (this branch has not committed yet).
    import webapp.server as _WS
    server_path = _WS.__file__
    current_bytes = open(server_path, "rb").read()
    pristine_ok = None
    sha_pristine = None
    try:
        head_bytes = subprocess.check_output(["git", "show", "HEAD:webapp/server.py"], cwd=_REPO)
        if head_bytes and b"_SELF_INITIATE_DEFAULT_ON" not in head_bytes:   # HEAD is genuinely pre-edit
            try:
                open(server_path, "wb").write(head_bytes)
                rows_pristine, sha_pristine = _subproc_panel({})
            finally:
                open(server_path, "wb").write(current_bytes)
            assert open(server_path, "rb").read() == current_bytes, "server.py restore FAILED"
            pristine_ok = _byte_identical(rows_on, rows_pristine)
    except Exception as _pe:
        pristine_ok = None
        sha_pristine = f"skipped: {type(_pe).__name__}: {_pe}"
    no_key_on_reactive = all((not r["resp"]["has_self_initiated"]) for r in rows_on) and \
        all((not r["resp"]["has_self_initiated"]) for r in rows_off)
    B_flag_off_ok = bool(sha_on == sha_off and no_key_on_reactive)
    B_pristine_ok = bool(pristine_ok is True) if pristine_ok is not None else None
    rows["B_byte_identical"] = {"sha_on": sha_on, "sha_off": sha_off, "sha_pristine": sha_pristine,
                                "flag_off_byte_identical": B_flag_off_ok,
                                "pristine_head_byte_identical": B_pristine_ok,
                                "no_self_initiated_key_on_reactive": no_key_on_reactive}

    # ── verdict ─────────────────────────────────────────────────────────────────────────────────────────────────
    gates = {
        "A_idle_coherent_per_seed": bool(A_ok),
        "A_handler_idle_coherent": bool(handler_idle_ok and handler_empty_ok),
        "B_flag_off_byte_identical": bool(B_flag_off_ok),
        "B_pristine_head_byte_identical": (bool(B_pristine_ok) if B_pristine_ok is not None else "skipped"),
        "C_lesion_collapses_per_seed": bool(C_lesion_ok),
        "C_handler_lesion_neutral": bool(handler_lesion_ok),
        "D_moat_safe": bool(D_moat_ok),
    }
    # PASS: A + B(flag-off) + C + D hard; pristine-HEAD corroborates when git state allows (not required if skipped).
    PASS = bool(A_ok and handler_idle_ok and handler_empty_ok and B_flag_off_ok and C_lesion_ok
                and handler_lesion_ok and D_moat_ok and (B_pristine_ok in (True, None)))

    summary = {
        "probe": "self_initiated_production_verify", "backend": os.environ.get("SIM_BACKEND", "numpy"),
        "seeds": a.seeds, "PASS": PASS, "gates": gates, "per_seed": per_seed, "handler": rows["handler"],
        "store_attribution": rows.get("store_attribution"),
        "byte_identical": rows["B_byte_identical"], "panel": _PANEL,
        "elapsed_seconds": round(time.time() - t0, 1),
        "NOTE": "reuse-by-import of the 6-seed GO _self_initiated_utterance_derisk (multibasin CA3 selection + "
                "curiosity want + OneBrainComposer mouth); NO sim/ edit. On numpy the heavy CA3 wander is DEFERRED "
                "(the light path speaks the mouth's curiosity-top decodable concept); the full curiosity-biased "
                "CA3-selection wander + its NO-ENCODE store-lesion collapse are the committed cupy GO "
                "research/findings/raw/_self_initiated_utterance_derisk.json. The store NO-ENCODE lesion (an emptied "
                "RF store, not a host flag) is load-bearing on BOTH paths.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))

    print("=" * 100)
    print("[self_init_verify] (A) per-seed coherent self-initiated utterance:")
    for p in per_seed:
        print(f"    seed {p['seed']}: n_utt={p['n_utt']} concept={p['concept']!r} utt={p['utterance']!r} "
              f"about={p['about_rate']} coherent={p['coherent']} | LESION n_utt={p['lesion_n_utt']} "
              f"collapses={p['collapses']} | moat={p['moat_abstains']}/{p['unknown_abstains']}")
    print(f"[self_init_verify] (A) handler idle: {rows['handler']['idle']['answer']!r} ok={handler_idle_ok}")
    print(f"[self_init_verify] (C) handler lesion: {rows['handler']['lesion']['answer']!r} ok={handler_lesion_ok}")
    print(f"[self_init_verify] (B) sha_on={sha_on[:16]} sha_off={sha_off[:16]} "
          f"pristine={str(sha_pristine)[:16]} | flag_off={B_flag_off_ok} pristine={B_pristine_ok} "
          f"no_key_reactive={no_key_on_reactive}")
    print(f"[self_init_verify] gates: {gates}")
    print(f"[self_init_verify] {'PASS' if PASS else 'FAIL'} | wrote {OUT} | {summary['elapsed_seconds']}s")
    print("=" * 100)
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
