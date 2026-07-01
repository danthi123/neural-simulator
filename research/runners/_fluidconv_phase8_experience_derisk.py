"""Phase-8 DE-RISK: the EXPERIENCE-connection -- the fluid console converses about a PERCEIVED (not taught) object.

The owner's priority: responses "grounded in the brain's OWN knowledge AND EXPERIENCES." Phases 0-7 converse about
TAUGHT facts. This de-risks the conversational handling of a PERCEIVED object: the object's concept code comes from
PERCEPTION (a fixed cortico-cortical projection over a percept feature-vector -> a phasor code, the validated
step-3/Tier-3 grounding `_projection`/`_to_phasor`), NOT from the composer's default code-generation. The fluid
console (RA-fine-tuned 21M + gate + VERIFY + moat) then answers about the perceived object.

Cheap-first (isolates the CONVERSATIONAL layer): a lightweight per-object percept feature-vector stands in for the
live `cortex_it` spiking forward (which the heavier merged nav+conv bridge supplies -- the full embodied
perceive-while-acting -> converse loop is the follow-on). The grounding mechanism (percept -> fixed projection ->
phasor code) is IDENTICAL to the validated live path; only the percept SOURCE is lightweight here.

METRICS (>=3 seeds): (a) CONVERSE = the console answers a question about a perceived-grounded object, RA-rendered,
VERIFY-clean; (b) GROUNDING-LESION = corrupt the perceived object's grounded code AFTER storing -> the recall
COLLAPSES (the answer is load-bearing on the PERCEPT, not a taught label); (c) MOAT = an UN-perceived object ->
abstain (0-FA).

GO = converse (all perceived objects answered grounded) + grounding-lesion collapses + moat 0-FA, >=3 seeds.

Reuse-by-import (grounding projection from step-3; the fluid console pieces); NO `sim/` edit.
Run: python -m research.runners._fluidconv_phase8_experience_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT  # noqa: E402
from research.runners._fluidconv_phase2_ra_qa_eval_derisk import FTFaculty, _v3  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase8_experience.json"
D = 128                    # composer phasor dimension (BrainConversationalAgent default)
N_FEAT = 64                # percept feature-vector length (the "cortex_it rate" stand-in)
# PERCEIVED objects (subjects) + their lived facts (transitive verb the RA generator renders well). The SUBJECT's
# code is grounded from PERCEPTION; rabbit/mouse/worm are the taught patients.
PERCEIVED = [("wolf", "eat", "rabbit"), ("owl", "eat", "mouse"), ("frog", "eat", "worm")]
UNPERCEIVED = "otter"      # in vocab, NEVER perceived/stored -> the moat cue


def _projection(seed):
    """Fixed complex projection percept(N_FEAT) -> D phases (the step-3 grounding: a deterministic function of the
    percept features -> a grounded phasor code, NOT a free random code)."""
    rng = np.random.default_rng(seed * 5077 + 11)
    return (rng.standard_normal((D, N_FEAT)) + 1j * rng.standard_normal((D, N_FEAT))).astype(np.complex128)


def _percept(obj_idx, seed, corrupt=False):
    """A deterministic per-object percept feature-vector (the lightweight stand-in for the live cortex_it rate)."""
    rng = np.random.default_rng(seed * 911 + obj_idx * 17 + 3)
    v = np.abs(rng.standard_normal(N_FEAT)).astype(np.float64)      # a nonneg rate-like feature vector
    if corrupt:
        cr = np.random.default_rng(seed * 333 + obj_idx * 29 + 7)
        v = v + cr.normal(0.0, 1.2, size=v.shape)                  # heavy corruption of the percept
        v = np.maximum(v, 0.0)
    return v


def _to_phase(rate_vec, proj):
    """Ground a percept -> the composer's concept-code format: a REAL phase array angle(proj @ rate) in [-pi, pi]
    (the RFPhasorComposer stores concept codes as PHASES, ingested via np.asarray(., dtype=float)). Deterministic
    function of the percept features -> a grounded code."""
    z = proj @ rate_vec.astype(np.complex128)
    return np.angle(z).astype(np.float64)


def _grounded_codes(seed, corrupt_obj=None):
    proj = _projection(seed)
    codes = {}
    for i, (obj, _v, _p) in enumerate(PERCEIVED):
        codes[obj] = _to_phase(_percept(i, seed, corrupt=(obj == corrupt_obj)), proj)
    return codes


def _answer(agent, faculty, subj, verb, vs):
    agents, actions, patients, inflect, store_keys = vs
    p = agent.what_does(subj, verb)
    if p is None:
        return None, "I don't know."
    ctx = f"the {subj} {_v3(verb)} {p} ."
    ans = faculty.answer(ctx, f"what does the {subj} {verb} ?")
    svos = _extract_all_svos(ans, agents, actions, patients, inflect)
    ung = [s for s in svos if _fact_key(s) not in store_keys]
    verified = bool((([subj, verb, p] in svos) or (p in ans.split())) and not ung)
    return p, (ans if verified else f"The {subj} {_v3(verb)} {p}.")


def run(seed, faculty):
    agents_set = {f[0] for f in PERCEIVED}; patients_set = {f[2] for f in PERCEIVED}; actions_set = {f[1] for f in PERCEIVED}
    inflect = _build_inflection_map(sorted(actions_set))
    vocab = sorted(agents_set | patients_set | actions_set | {UNPERCEIVED})

    # PERCEIVE: ground the perceived objects' codes from percepts (NOT taught), pass as the composer's concept codes.
    gc = _grounded_codes(seed)
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, grounded_codes=gc,
                                     composer_kind="rf", D=D)
    for (a, v, p) in PERCEIVED:
        agent.hear(f"{a} {v} {p}")                     # store the lived fact (subject code = perception-grounded)
    store_keys = {tuple(f) for f in PERCEIVED}
    vs = (agents_set, actions_set, patients_set, inflect, store_keys)

    # (a) CONVERSE about each perceived object
    conv = []
    for (a, v, p) in PERCEIVED:
        got, reply = _answer(agent, faculty, a, v, vs)
        conv.append({"perceived": a, "q": f"what does the {a} {v}?", "reply": reply, "ok": bool(got == p and p in reply.split())})

    # (b) GROUNDING-LESION: corrupt one perceived object's grounded code AFTER storing -> its recall must COLLAPSE
    lesion_obj = PERCEIVED[0][0]
    agent.composer.concepts[lesion_obj] = _grounded_codes(seed, corrupt_obj=lesion_obj)[lesion_obj]
    got_l = agent.what_does(lesion_obj, PERCEIVED[0][1])
    lesion_collapsed = (got_l != PERCEIVED[0][2])       # the corrupted percept no longer recalls the stored patient

    # (c) MOAT: an un-perceived object -> abstain
    moat_gate = agent.what_does(UNPERCEIVED, "eat")
    moat_ok = (moat_gate is None)

    n_conv = sum(c["ok"] for c in conv)
    return {"seed": seed, "converse_ok": n_conv, "converse_total": len(conv), "conv": conv,
            "lesion_obj": lesion_obj, "lesion_collapsed": bool(lesion_collapsed), "lesion_recall": got_l,
            "moat_ok": bool(moat_ok), "moat_gate": moat_gate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT})"); return 2
    t0 = time.time()
    err = None; per_seed = []
    try:
        faculty = FTFaculty()
        print(f"[phase8-exp] loaded RA-fine-tuned ~{faculty.npar:.1f}M (dev={faculty.device})\n", flush=True)
        for s in a.seeds:
            r = run(s, faculty)
            per_seed.append(r)
            print(f"  [seed {s}] converse {r['converse_ok']}/{r['converse_total']} | grounding-lesion collapsed "
                  f"{r['lesion_collapsed']} | moat {r['moat_ok']}", flush=True)
            for c in r["conv"]:
                print(f"      perceived '{c['perceived']}' -> \"{c['reply']}\" (ok={c['ok']})", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        conv_ok = all(r["converse_ok"] == r["converse_total"] for r in per_seed)
        lesion_ok = all(r["lesion_collapsed"] for r in per_seed)
        moat_ok = all(r["moat_ok"] for r in per_seed)
        go = bool(conv_ok and lesion_ok and moat_ok)
        verdict = (("GO -- the fluid console converses about PERCEIVED (not taught) objects: each perceived object's "
                    "code is grounded from a percept (the validated fixed projection), stored, and answered grounded "
                    "(RA-rendered); the GROUNDING-LESION (corrupt the percept) COLLAPSES the recall (load-bearing on "
                    "the experience); the moat holds 0-FA on un-perceived objects. >=3 seeds. The 'experiences' clause "
                    "closed at the conversational layer.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if conv_ok else [f"converse {[r['converse_ok'] for r in per_seed]}/{[r['converse_total'] for r in per_seed]}"]) +
                       ([] if lesion_ok else [f"grounding-lesion {[r['lesion_collapsed'] for r in per_seed]} (not load-bearing)"]) +
                       ([] if moat_ok else [f"moat {[r['moat_ok'] for r in per_seed]}"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase8_experience", "GO": go, "verdict": verdict,
               "resolves": "the fluid console converses about a PERCEIVED (not taught) object -- its code grounded "
                           "from a percept via the validated fixed projection; grounding load-bearing; moat 0-FA.",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "cheap-first: a lightweight per-object percept feature-vector stands in for the live "
                                 "cortex_it spiking forward (same grounding projection). The FULL embodied loop "
                                 "(perceive-while-acting on the merged nav+conv bridge -> converse via the RA console) "
                                 "is the follow-on integration (composes Tier-3 live-and-remember + the RA console)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase8-exp] VERDICT: {verdict}", flush=True)
    print(f"[phase8-exp] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
