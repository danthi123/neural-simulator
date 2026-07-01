"""Phase-1a DE-RISK: FLUID grounded rendering with the ~21M generator -- constrained core + FREE continuation + VERIFY.

Phase 0 proved the 21M TinyStories generator, behind the per-token grounded veto, renders a SINGLE proposition
grounded + non-vacuous (moat-intact) -- but the constrained decode TRADES fluency for faithfulness BY DESIGN (rigid,
one proposition, vetoed to its own words). The owner's north star is FLUID, LLM-like conversation. This de-risk tests
the MIDDLE ground the roadmap names (Phase 1): let the generator be FLUID (free, multi-sentence) but keep the brain's
no-confab moat as a PLUS via POST-HOC VERIFY (per `feedback_moat_not_hard_lossy_memory_ok`), instead of the rigid veto.

THE COMPOSITION (per grounded fact, all validated pieces; NO sim/ edit):
  (i)   GATE            -- the brain (`BrainConversationalAgent`) recalls the stored fact OR abstains (the moat gates
                          FIRST; an untaught cue -> no fact -> the generator is never invoked).
  (ii)  CONSTRAINED CORE -- Phase-0 constrained decode renders the gated proposition as a FAITHFUL grounded opening
                          (grounded-by-construction; == the Phase-0 GO result).
  (iii) FREE CONTINUATION -- the SAME 21M generator, UNCONSTRAINED (veto OFF), continues from that opening for a few
                          more tokens = the FLUID elaboration (the generator's full fluent vocabulary, not vetoed).
  (iv)  VERIFY (moat-as-a-plus) -- re-parse the FULL text (core + continuation) into ALL (known-agent, known-verb,
                          known-patient) triples; EACH must be GROUNDED in the store, else the continuation asserted an
                          ungrounded fact about a known entity = a HALLUCINATION -> FLAG (a real generative model drifts;
                          VERIFY catches exactly that). Narrative color about NON-fact tokens (happy, tasty, ran) is
                          free -- only asserted SVOs over KNOWN entities are policed.

METRICS (multi-seed, GPU):
  (a) FLUID+GROUNDED  -- the core verifies (Phase-0) AND the continuation adds real tokens AND the full text asserts
                        NO ungrounded known-entity SVO. This is the headline: fluid-beyond-one-proposition + still grounded.
  (b) DRIFT-CAUGHT    -- an ADVERSARIAL continuation (primed toward a wrong-but-known fact, e.g. '... Then the cat ate')
                        makes the generator assert an UNGROUNDED SVO; VERIFY FLAGS it (the moat-as-a-plus is load-bearing).
  (c) UNTAUGHT-ABSTAIN -- an untaught cue -> the GATE abstains -> NO generation (the moat gates first).
  (d) CONSTRAINT LOAD-BEARING -- a FULLY-FREE render (no constrained core; free from a bare subject prompt) drifts:
                        it asserts ungrounded known-entity SVOs at a HIGHER rate than the core+continuation path
                        (shows the constrained core + VERIFY are doing work; the free baseline is the foil).

GO = (a) fluid+grounded on the taught set (all core-verified, continuations non-trivial, 0 ungrounded assertions) AND
     (b) every adversarial drift FLAGGED AND (c) untaught abstains AND (d) the free baseline drifts more than the
     grounded path -- >= 3 seeds. HONEST/PARTIAL otherwise (e.g. greedy continuations degenerate/repeat -> needs
     temp-sampling; or the continuation asserts ungrounded facts VERIFY misses -> the extractor/moat needs strengthening).

Reuse-by-import (NO sim/ edit): the brain GATE + VERIFY re-parse machinery from `_grounded_lang_integration_derisk`;
the curriculum from `_grounded_lang_p2_derisk`; the 21M constrained/unconstrained decode from the (additively-
parameterized) `constrained_decode_gate._GroundedConstrainedLM`.

Run: python -m research.runners._fluidconv_phase1_grounded_continuation_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

# brain half = numpy-CPU (portable, no GPU contention with the torch generator); the 21M generator is torch-on-GPU.
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
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402

CKPT = "research/findings/raw/fluidconv/gen_tinystories_20M.ckpt"
BPE = "research/findings/raw/fluidconv/gen_tinystories.bpe.json"
ARCH = dict(d_model=512, n_layer=6, n_head=8, block_size=512, bpe_path=BPE)
OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase1_grounded_continuation.json"


def _extract_all_svos(prose, agents, actions, patients, inflect):
    """Scan a (possibly multi-sentence) prose for ALL (agent, action, patient) SVOs over KNOWN entities, in surface
    order. For every occurrence of a known verb, bind the nearest preceding known agent and the nearest following
    known patient (before the next verb) -> one SVO. Function words / determiners / novel narrative tokens are
    ignored. Returns a list of [a, v, p] canonical triples. This is the multi-fact generalization of the integration
    de-risk's single-SVO extractor -- a real fluent continuation can assert SEVERAL facts, each must be checked."""
    import re
    toks = re.findall(r"[a-z]+", prose.lower())
    # index every token as ('A', word) / ('V', base) / ('P', word) / ('_', ) so we can walk left->right
    marks = []
    for t in toks:
        bv = inflect.get(t)
        if bv in actions:
            marks.append(("V", bv))
        elif t in agents and t in patients:
            marks.append(("AP", t))     # ambiguous (some nouns are both agent+patient in the curriculum)
        elif t in agents:
            marks.append(("A", t))
        elif t in patients:
            marks.append(("P", t))
        else:
            marks.append(("_", t))
    svos = []
    for i, (kind, val) in enumerate(marks):
        if kind != "V":
            continue
        # nearest preceding agent (A or AP), not crossing an earlier verb
        a = None
        for j in range(i - 1, -1, -1):
            if marks[j][0] == "V":
                break
            if marks[j][0] in ("A", "AP"):
                a = marks[j][1]; break
        # nearest following patient (P or AP), before the next verb
        p = None
        for j in range(i + 1, len(marks)):
            if marks[j][0] == "V":
                break
            if marks[j][0] in ("P", "AP"):
                p = marks[j][1]; break
        if a and p:
            svos.append([a, val, p])
    return svos


def _fact_key(svo):
    return (svo[0], svo[1], svo[2])


class Gen21MFaculty:
    """The ~21M TinyStories generator as a fluent faculty: a CONSTRAINED decoder (Phase-0 faithful render of one
    proposition) + an UNCONSTRAINED continuer (free fluid elaboration). Both are the SAME model; two
    `_GroundedConstrainedLM` instances differ only by the veto (mode)."""

    def __init__(self, max_core=28, max_cont=32):
        from research.runners.constrained_decode_gate import _GroundedConstrainedLM
        self.lm_c = _GroundedConstrainedLM(CKPT, mode="constrained", **ARCH)
        self.lm_u = _GroundedConstrainedLM(CKPT, mode="unconstrained", **ARCH)
        self.tok = self.lm_c.tok
        self.device = self.lm_c.device
        self.max_core = int(max_core)
        self.max_cont = int(max_cont)
        self.npar = sum(p.numel() for p in self.lm_c.model.parameters()) / 1e6

    def render_core(self, proposition):
        """CONSTRAINED render of the grounded proposition (Phase-0 faithful; veto = the proposition's own words)."""
        pid = self.tok.encode(proposition)
        gid = self.lm_c.generate_ids(pid, self.max_core)
        return self.tok.decode(gid).strip()

    def continue_free(self, prefix_text, extra_prime=""):
        """UNCONSTRAINED free continuation from prefix_text (+ an optional adversarial extra_prime that steers toward
        a wrong-but-known fact). Returns (continuation_text, full_text)."""
        prompt = (prefix_text + extra_prime).strip()
        pid = self.tok.encode(prompt)
        gid = self.lm_u.generate_ids(pid, self.max_cont)
        cont = self.tok.decode(gid).strip()
        return cont, (prompt + " " + cont).strip()


def _ground_check(svos, store_keys):
    """Return (ungrounded_known_svos): asserted SVOs over known entities NOT in the store. store_keys = set of
    (a,v,p) tuples the brain actually holds. A known-entity SVO absent from the store = a confident hallucination."""
    return [s for s in svos if _fact_key(s) not in store_keys]


def run(cur, vocab, seed, faculty):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)

    facts = cur.get("facts", [])
    agents_set = {f[0] for f in facts}
    patients_set = {f[2] for f in facts}
    actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set))
    store_keys = {tuple(f) for f in facts}

    # --- (a) FLUID+GROUNDED: core (constrained, faithful) + free continuation; VERIFY no ungrounded known SVO ---
    test_facts = [q for q in cur.get("queries_recall", []) if q["type"] == "patient"][:5]
    grounded = []
    for q in test_facts:
        a, v = q["cue"]
        p = agent.what_does(a, v)                              # GATE
        if p is None:
            grounded.append({"cue": q["cue"], "abstained": True, "ok": False}); continue
        core = faculty.render_core(f"{a} {v} {p}")             # CONSTRAINED CORE (Phase-0)
        cont, full = faculty.continue_free(core)               # FREE CONTINUATION (fluid)
        svos = _extract_all_svos(full, agents_set, actions_set, patients_set, inflect)
        ungrounded = _ground_check(svos, store_keys)
        core_has_fact = [a, v, p] in _extract_all_svos(core, agents_set, actions_set, patients_set, inflect)
        n_cont_tokens = len(cont.split())
        ok = bool(core_has_fact and n_cont_tokens >= 3 and len(ungrounded) == 0)
        grounded.append({"cue": q["cue"], "gate": [a, v, p], "core": core, "continuation": cont, "full": full,
                         "svos_found": svos, "ungrounded_asserted": ungrounded, "core_has_fact": core_has_fact,
                         "n_cont_tokens": n_cont_tokens, "abstained": False, "ok": ok})

    # --- (b) DRIFT-CAUGHT: steer the continuation toward a wrong-but-known fact; VERIFY must FLAG the ungrounded SVO ---
    drift = []
    for q in test_facts[:3]:
        a, v = q["cue"]
        p = agent.what_does(a, v)
        if p is None:
            continue
        core = faculty.render_core(f"{a} {v} {p}")
        # pick a wrong-but-KNOWN fact to steer toward: a DIFFERENT agent + its verb but the WRONG patient.
        other_a = next((x for x in sorted(agents_set) if x != a), a)
        steer = f" . then the {other_a} {v}"                   # e.g. '. then the cat eat' -> model likely completes a patient
        cont, full = faculty.continue_free(core, extra_prime=steer)
        svos = _extract_all_svos(full, agents_set, actions_set, patients_set, inflect)
        ungrounded = _ground_check(svos, store_keys)
        # 'caught' = VERIFY surfaced >=1 ungrounded known-entity assertion (so the loop would flag/withhold it)
        caught = len(ungrounded) >= 1
        drift.append({"cue": q["cue"], "steer": steer.strip(), "core": core, "continuation": cont, "full": full,
                      "svos_found": svos, "ungrounded_asserted": ungrounded, "caught": caught})

    # --- (c) UNTAUGHT-ABSTAIN: an untaught cue -> the GATE abstains -> NO generation (moat gates first) ---
    untaught = []
    for q in [x for x in cur.get("queries_moat", []) if x["type"] == "patient"][:3]:
        a, v = q["cue"]
        p = agent.what_does(a, v)
        held = (p is None)                                     # abstained -> generator never invoked
        untaught.append({"cue": q["cue"], "gate_result": p, "held": held})

    # --- (d) CONSTRAINT LOAD-BEARING: a fully-FREE render from a bare subject prompt drifts MORE than the grounded path ---
    free_baseline = []
    for q in test_facts[:5]:
        a, v = q["cue"]
        _cont, full = faculty.continue_free(f"the {a}")        # bare subject, NO grounded core, fully free
        svos = _extract_all_svos(full, agents_set, actions_set, patients_set, inflect)
        ungrounded = _ground_check(svos, store_keys)
        free_baseline.append({"subject": a, "full": full, "svos_found": svos, "ungrounded_asserted": ungrounded,
                              "n_ungrounded": len(ungrounded)})

    n_grounded_ok = sum(r["ok"] for r in grounded)
    n_drift_caught = sum(r["caught"] for r in drift)
    n_untaught_held = sum(r["held"] for r in untaught)
    grounded_path_ungrounded = sum(len(r.get("ungrounded_asserted", [])) for r in grounded)
    free_path_ungrounded = sum(r["n_ungrounded"] for r in free_baseline)
    return {"seed": seed,
            "grounded_ok": n_grounded_ok, "grounded_total": len(grounded),
            "drift_caught": n_drift_caught, "drift_total": len(drift),
            "untaught_held": n_untaught_held, "untaught_total": len(untaught),
            "grounded_path_ungrounded_count": grounded_path_ungrounded,
            "free_baseline_ungrounded_count": free_path_ungrounded,
            "grounded_detail": grounded, "drift_detail": drift, "untaught_detail": untaught,
            "free_baseline_detail": free_baseline}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-core", type=int, default=28)
    ap.add_argument("--max-cont", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not (os.path.exists(CKPT + ".pt") and os.path.exists(BPE)):
        print(f"NOT-RUNNABLE: 21M generator absent ({CKPT}.pt / {BPE})"); return 2

    t0 = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    print(f"[phase1] brain backend={os.environ.get('SIM_BACKEND')} vocab={len(vocab)}; loading the 21M generator "
          f"(constrained core + free continuer)...", flush=True)
    err = None
    per_seed = []
    try:
        faculty = Gen21MFaculty(max_core=a.max_core, max_cont=a.max_cont)
        print(f"[phase1] loaded ~{faculty.npar:.1f}M (dev={faculty.device}); seeds={a.seeds}\n", flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty)
            per_seed.append(r)
            print(f"  [seed {s}] fluid+grounded {r['grounded_ok']}/{r['grounded_total']} | "
                  f"drift-caught {r['drift_caught']}/{r['drift_total']} | untaught-abstain "
                  f"{r['untaught_held']}/{r['untaught_total']} | ungrounded asserted: grounded-path "
                  f"{r['grounded_path_ungrounded_count']} vs free-baseline {r['free_baseline_ungrounded_count']}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        g_ok = all(r["grounded_ok"] == r["grounded_total"] and r["grounded_total"] > 0 for r in per_seed)
        d_ok = all(r["drift_caught"] == r["drift_total"] and r["drift_total"] > 0 for r in per_seed)
        u_ok = all(r["untaught_held"] == r["untaught_total"] and r["untaught_total"] > 0 for r in per_seed)
        # load-bearing: the free baseline asserts strictly MORE ungrounded known-entity facts than the grounded path
        lb_ok = all(r["free_baseline_ungrounded_count"] > r["grounded_path_ungrounded_count"] for r in per_seed)
        go = bool(g_ok and d_ok and u_ok and lb_ok)
        verdict = (("GO -- FLUID grounded rendering: the 21M generator's constrained core + FREE continuation stays "
                    "grounded (VERIFY 0 ungrounded known-entity assertions), adversarial drift is FLAGGED by VERIFY, "
                    "untaught abstains, and the fully-free baseline drifts MORE (constraint+VERIFY load-bearing) -- "
                    ">=3 seeds. Fluid-beyond-one-proposition WITH the moat as a plus.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if g_ok else [f"fluid+grounded {[r['grounded_ok'] for r in per_seed]}/"
                                         f"{[r['grounded_total'] for r in per_seed]} (continuation degenerate or "
                                         "asserted ungrounded known-entity fact)"]) +
                       ([] if d_ok else [f"drift-caught {[r['drift_caught'] for r in per_seed]}/"
                                         f"{[r['drift_total'] for r in per_seed]} (VERIFY missed a steered drift)"]) +
                       ([] if u_ok else [f"untaught-abstain {[r['untaught_held'] for r in per_seed]} (moat leak)"]) +
                       ([] if lb_ok else ["constraint NOT load-bearing (free baseline drifts <= grounded path)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase1_grounded_continuation",
               "resolves": "Phase 1 fluidity: fluid (multi-sentence, free) grounded rendering with the ~21M generator, "
                           "moat kept as a PLUS via post-hoc VERIFY (softer than the Phase-0 per-token veto).",
               "architecture": "GATE (brain recall/abstain) -> CONSTRAINED CORE (Phase-0 faithful render) -> FREE "
                               "CONTINUATION (unconstrained 21M, fluid) -> VERIFY (re-parse full text into all "
                               "known-entity SVOs; each must be grounded, else flag the hallucination).",
               "generator": f"~{faculty.npar:.1f}M TinyStories (d512/L6/H8)" if err is None else "n/a",
               "seeds": a.seeds, "GO": go, "verdict": verdict,
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": ("greedy decoding may make continuations repetitive; temp-sampling + repetition "
                                  "penalty is the fluency follow-on. This tests fluid grounded RENDERING, not yet "
                                  "multi-turn dialogue coherence (the recurrent-state + multi-referent WTA follow-on).")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase1] VERDICT: {verdict}", flush=True)
    print(f"[phase1] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
