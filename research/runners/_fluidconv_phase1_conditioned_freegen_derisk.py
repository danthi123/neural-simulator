"""Phase-1a v3 DE-RISK: FLUID grounded rendering via PROMPT-CONDITIONED FREE generation + post-hoc VERIFY.

v1 (constrained core + free continuation) and v2 (broadened veto + sampling) both NEGATIVE: a per-token grounded veto
is fundamentally incompatible with fluency (the lexicon is too small -> word-salad). The DECISIVE contrast from v2:
the FREE (unvetoed) 21M generation is genuinely FLUENT and NON-HALLUCINATORY (0 false known-entity SVOs) -- it stays
in descriptive-narrative register; it just doesn't render the SPECIFIC grounded fact on command. v3 tests the reframe:
condition the FREE generator with a NATURAL fact-lead (the grounded fact as an opening sentence) and let it continue
fluently; VERIFY re-parses the whole text and rejects any NEW ungrounded assertion. The moat is a post-hoc PLUS (per
`feedback_moat_not_hard_lossy_memory_ok`), NOT a per-token veto -- so fluency is preserved.

Per grounded fact (NO sim/ edit; reuse-by-import):
  (i)   GATE          -- the brain recalls the fact (a, v, p) OR abstains (untaught -> no lead -> no generation).
  (ii)  FACT-LEAD     -- state the grounded fact as a natural opening. The word ORDER is the brain's (agent-verb-
                        patient, the validated neural serial-order render / composer.render_fact); a MINIMAL surface
                        scaffold ("The {a} {v}s {p}.") adds determiner+present-tense. [SCAFFOLD, flagged: morphology
                        is the fluency the fine-tune would learn; here it is a fixed surface for the probe.]
  (iii) FREE CONTINUATION -- the UNVETOED 21M continues from the lead, temperature-sampled (fluent; the v2 free
                        baseline proved this is coherent + non-hallucinatory).
  (iv)  VERIFY (moat-as-a-plus) -- re-parse the FULL text into all known-entity SVOs; the lead contributes the
                        grounded fact, and the continuation must add NO ungrounded known-entity SVO, else FLAG.

METRICS (>=3 seeds, GPU): (a) FLUID+GROUNDED = the full text asserts the gated fact AND the continuation adds real
tokens (>=6 words) AND 0 ungrounded known-entity SVOs AND distinct-ratio >= 0.5; (b) DRIFT-CAUGHT = an adversarial
lead stating a WRONG fact -> VERIFY flags the ungrounded assertion; (c) UNTAUGHT-ABSTAIN = untaught cue -> GATE
abstains -> no generation.

GO = fluid+grounded on the taught set AND drift caught AND untaught abstains, >=3 seeds. HONEST/PARTIAL otherwise
(e.g. the terse lead triggers the canned "Once upon a time..." fallback, or the continuation asserts false facts ->
the retrieval-augmented render fine-tune is the next lever).

Run: python -m research.runners._fluidconv_phase1_conditioned_freegen_derisk --seeds 42 43 44
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

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402
from research.runners._fluidconv_phase1_broadened_veto_derisk import BroadenedVetoFaculty, _distinct_ratio  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase1_conditioned_freegen.json"


def _present(v):
    """Minimal present-tense surface (SCAFFOLD): dog eat meat -> 'eats'. Only the surface morphology; the WORD ORDER
    is the brain's neural render (agent-verb-patient), the grounded CONTENT is the brain's fact."""
    if v.endswith(("s", "sh", "ch", "x", "z")):
        return v + "es"
    return v + "s"


def _lead(a, v, p):
    return f"The {a} {_present(v)} {p}."


def run(cur, vocab, seed, faculty, temperature, rep_penalty):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)
    facts = cur.get("facts", [])
    agents_set = {f[0] for f in facts}
    patients_set = {f[2] for f in facts}
    actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set))
    store_keys = {tuple(f) for f in facts}

    def _svos(text):
        return _extract_all_svos(text, agents_set, actions_set, patients_set, inflect)

    def _ungrounded(text):
        return [s for s in _svos(text) if _fact_key(s) not in store_keys]

    test_facts = [q for q in cur.get("queries_recall", []) if q["type"] == "patient"][:5]

    # --- (a) FLUID + GROUNDED: natural fact-lead + free continuation; VERIFY no NEW ungrounded assertion ---
    grounded = []
    for q in test_facts:
        a, v = q["cue"]
        p = agent.what_does(a, v)                                  # GATE
        if p is None:
            grounded.append({"cue": q["cue"], "abstained": True, "ok": False}); continue
        lead = _lead(a, v, p)
        cont = faculty.sample("", lead, temperature=temperature, rep_penalty=rep_penalty, seed=seed, veto_off=True)
        full = (lead + " " + cont).strip()
        ug = _ungrounded(full)
        gated_asserted = [a, v, p] in _svos(full)
        n_cont = len(cont.split())
        dr = _distinct_ratio(full)
        ok = bool(gated_asserted and len(ug) == 0 and n_cont >= 6 and dr >= 0.5)
        grounded.append({"cue": q["cue"], "gate": [a, v, p], "lead": lead, "continuation": cont, "full": full,
                         "gated_asserted": gated_asserted, "ungrounded": ug, "n_cont": n_cont,
                         "distinct_ratio": round(dr, 3), "abstained": False, "ok": ok})

    # --- (b) DRIFT-CAUGHT: an adversarial lead stating a WRONG fact -> VERIFY flags the ungrounded assertion ---
    drift = []
    for q in test_facts[:3]:
        a, v = q["cue"]
        p = agent.what_does(a, v)
        if p is None:
            continue
        wrong_p = next((x for x in sorted(patients_set) if x != p), p)   # a wrong but known patient
        adv_lead = _lead(a, v, wrong_p)                                  # states an UNGROUNDED fact
        cont = faculty.sample("", adv_lead, temperature=temperature, rep_penalty=rep_penalty, seed=seed, veto_off=True)
        full = (adv_lead + " " + cont).strip()
        ug = _ungrounded(full)
        caught = len(ug) >= 1                                            # VERIFY sees the ungrounded (a,v,wrong_p)
        drift.append({"cue": q["cue"], "wrong_patient": wrong_p, "adv_lead": adv_lead, "full": full,
                      "ungrounded": ug, "caught": caught})

    # --- (c) UNTAUGHT-ABSTAIN: untaught cue -> GATE abstains -> no lead, no generation ---
    untaught = []
    for q in [x for x in cur.get("queries_moat", []) if x["type"] == "patient"][:3]:
        a, v = q["cue"]
        p = agent.what_does(a, v)
        untaught.append({"cue": q["cue"], "gate_result": p, "held": (p is None)})

    n_ok = sum(r["ok"] for r in grounded)
    n_caught = sum(r["caught"] for r in drift)
    n_held = sum(r["held"] for r in untaught)
    return {"seed": seed, "grounded_ok": n_ok, "grounded_total": len(grounded),
            "drift_caught": n_caught, "drift_total": len(drift),
            "untaught_held": n_held, "untaught_total": len(untaught),
            "grounded_detail": grounded, "drift_detail": drift, "untaught_detail": untaught}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not (os.path.exists("research/findings/raw/fluidconv/gen_tinystories_20M.ckpt.pt")):
        print("NOT-RUNNABLE: 21M generator absent"); return 2

    t0 = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    print(f"[phase1-v3] brain={os.environ.get('SIM_BACKEND')} vocab={len(vocab)}; loading 21M (prompt-conditioned "
          f"FREE gen T={a.temperature} rep={a.rep_penalty} + post-hoc VERIFY)...", flush=True)
    err = None; per_seed = []
    try:
        faculty = BroadenedVetoFaculty(max_new=a.max_new)
        print(f"[phase1-v3] loaded ~{faculty.npar:.1f}M (dev={faculty.device})\n", flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty, a.temperature, a.rep_penalty)
            per_seed.append(r)
            print(f"  [seed {s}] fluid+grounded {r['grounded_ok']}/{r['grounded_total']} | drift-caught "
                  f"{r['drift_caught']}/{r['drift_total']} | untaught-abstain {r['untaught_held']}/{r['untaught_total']}",
                  flush=True)
            for d in r["grounded_detail"][:3]:
                if not d.get("abstained"):
                    print(f"      ok={d['ok']} dr={d['distinct_ratio']} ug={len(d['ungrounded'])} :: {d['full']!r}",
                          flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        g_ok = all(r["grounded_ok"] == r["grounded_total"] and r["grounded_total"] > 0 for r in per_seed)
        d_ok = all(r["drift_caught"] == r["drift_total"] and r["drift_total"] > 0 for r in per_seed)
        u_ok = all(r["untaught_held"] == r["untaught_total"] and r["untaught_total"] > 0 for r in per_seed)
        go = bool(g_ok and d_ok and u_ok)
        verdict = (("GO -- FLUID grounded rendering via prompt-conditioned FREE generation: a natural fact-lead + free "
                    "(fluent) continuation asserts the gated fact, adds real narrative, 0 ungrounded assertions; "
                    "adversarial (wrong-fact) leads are FLAGGED by post-hoc VERIFY; untaught abstains -- >=3 seeds. "
                    "Fluid AND grounded, moat as a post-hoc PLUS (no per-token veto), NO fine-tune.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if g_ok else [f"fluid+grounded {[r['grounded_ok'] for r in per_seed]}/"
                                         f"{[r['grounded_total'] for r in per_seed]} (canned fallback, degenerate, or "
                                         "the continuation asserted a false fact) -> the retrieval-augmented render "
                                         "fine-tune is the next lever"]) +
                       ([] if d_ok else [f"drift-caught {[r['drift_caught'] for r in per_seed]} (VERIFY missed a wrong-fact lead)"]) +
                       ([] if u_ok else ["untaught did not abstain (moat leak)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase1_conditioned_freegen",
               "resolves": "Phase 1 fluidity v3: fluid grounded rendering via prompt-conditioned FREE generation "
                           "(natural fact-lead + free continuation) + post-hoc VERIFY (moat-as-a-plus, not a veto).",
               "seeds": a.seeds, "temperature": a.temperature, "rep_penalty": a.rep_penalty,
               "GO": go, "verdict": verdict, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "SCAFFOLD_NOTE": "the fact-lead uses a minimal present-tense surface template ('The {a} {v}s {p}.'); "
                                "the WORD ORDER is the brain's neural render + the CONTENT is the brain's fact. The "
                                "surface morphology is a probe scaffold the render fine-tune would learn.",
               "HONEST_CEILING": "tests fluid RENDERING (fluent grounded statement + narrative), not multi-turn "
                                 "dialogue coherence nor open Q&A (the recurrent-state + fine-tune follow-ons)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase1-v3] VERDICT: {verdict}", flush=True)
    print(f"[phase1-v3] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
