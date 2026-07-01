"""Phase-1a v2 DE-RISK: FLUID multi-fact grounded generation via a BROADENED veto + temperature sampling.

v1 (constrained core + free continuation) was NEGATIVE (2026-07-01-fluid-conversation-phase1-fluidity-mechanism-
findings.md): the Phase-0 veto is INHERENTLY terse (allows only ONE proposition's words -> forced loop past it), and
free continuation hallucinates. The reframe (this v2): broaden the grounded veto from ONE proposition to the queried
subject's ENTIRE retrieved knowledge set (all the brain's facts about it) and SAMPLE with temperature + a repetition
penalty. The generator can then fluidly WEAVE MULTIPLE real facts -- grounded by the broadened veto (only the
subject's real facts' words are legal), fluent by sampling (broad vocabulary, no forced loop). This is the
retrieval-augmentation frame (roadmap GAP B): the brain retrieves the knowledge, the generator is veto-restricted to
exactly that knowledge, abstention is the honest breadth boundary, VERIFY is the post-hoc moat-as-a-plus.

Per queried subject (all validated pieces; NO sim/ edit):
  (i)   GATE (brain-grounded knowledge set) -- for subject S, recall EVERY fact the brain holds about it by querying
        the store: for each known verb V, p = agent.what_does(S, V); collect the non-None (S, V, p). Empty set ->
        ABSTAIN (untaught subject -> nothing grounded to say -> the moat).
  (ii)  BROADENED VETO -- allow_text = the union of all words in S's knowledge set (+ FUNCTION_WORDS). The
        prefix-automaton (reused from the Phase-0 `_GroundedConstrainedLM`) forbids completing any word not in S's
        real facts, so no ungrounded entity can be asserted BY CONSTRUCTION.
  (iii) TEMPERATURE-SAMPLED constrained decode -- sample within the allow-mask with temperature + a repetition
        penalty (fluent; no greedy loop) from a natural lead prompt. (A NEW sampling loop here in the runner reusing
        the LM's automaton methods; the greedy `generate_ids` is byte-unchanged.)
  (iv)  VERIFY -- re-parse the full text into all known-entity SVOs; each must be grounded (in the store). By
        construction of the broadened veto this should be 0 ungrounded; VERIFY confirms + is the moat-as-a-plus.

METRICS (>=3 seeds, GPU): (a) GROUNDED = 0 ungrounded known-entity SVOs; (b) MULTI-FACT FLUENCY = the output weaves
>=2 DISTINCT grounded SVOs AND distinct-token-ratio >= 0.5 (not a degenerate loop) = "fluid beyond one proposition";
(c) MOAT = untaught subject -> empty knowledge -> abstain; (d) LOAD-BEARING = unconstrained sampling (same prompt, NO
veto) asserts ungrounded known-entity SVOs at a strictly higher rate than the broadened-veto path.

GO = grounded (0 ungrounded) AND multi-fact-fluent (>=2 distinct grounded SVOs, distinct-ratio>=0.5) on the taught
subjects AND untaught abstains AND load-bearing, >=3 seeds. HONEST/PARTIAL otherwise (e.g. still stilted -> the small
fact-rendering/dialogue fine-tune is the next lever; or single-fact only -> the weave needs a richer prompt).

Reuse-by-import (NO sim/ edit): brain GATE (`BrainConversationalAgent`) + curriculum (`_grounded_lang_p2_derisk`) +
the automaton machinery + inflection/extractor from v1/`_grounded_lang_integration_derisk`.

Run: python -m research.runners._fluidconv_phase1_broadened_veto_derisk --seeds 42 43 44
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
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM  # noqa: E402
from research.runners._grounded_lang_integration_derisk import _build_inflection_map  # noqa: E402
from research.runners._fluidconv_phase1_grounded_continuation_derisk import _extract_all_svos, _fact_key  # noqa: E402

CKPT = "research/findings/raw/fluidconv/gen_tinystories_20M.ckpt"
BPE = "research/findings/raw/fluidconv/gen_tinystories.bpe.json"
ARCH = dict(d_model=512, n_layer=6, n_head=8, block_size=512, bpe_path=BPE)
OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase1_broadened_veto.json"


def _distinct_ratio(text):
    ws = [w for w in text.lower().split() if any(c.isalpha() for c in w)]
    return (len(set(ws)) / len(ws)) if ws else 0.0


class BroadenedVetoFaculty:
    """The ~21M generator with a BROADENED grounded veto + temperature sampling. Reuses the Phase-0
    `_GroundedConstrainedLM` automaton (prefix-mask over allow_text) but decodes by SAMPLING (temp + repetition
    penalty) instead of greedy -- fluent, not looped. `veto_off=True` = the unconstrained sampling baseline."""

    def __init__(self, max_new=48):
        from research.runners.constrained_decode_gate import _GroundedConstrainedLM
        self.lm = _GroundedConstrainedLM(CKPT, mode="constrained", **ARCH)
        self.tok = self.lm.tok
        self.device = self.lm.device
        self._torch = self.lm._torch
        self.max_new = int(max_new)
        self.npar = sum(p.numel() for p in self.lm.model.parameters()) / 1e6

    def sample(self, allow_text, prompt_text, temperature=0.8, rep_penalty=1.3, seed=42, veto_off=False):
        """Temperature-sampled decode. If not veto_off, restrict each step to the prefix-automaton of allow_text
        (the subject's grounded knowledge). Repetition penalty divides the logit of already-emitted tokens.
        Returns the decoded continuation (excluding the prompt)."""
        torch = self._torch
        lm = self.lm
        auto = lm._allowed_automaton(allow_text) if not veto_off else None
        pid = self.tok.encode(prompt_text)
        seq = list(pid) if pid else [0]
        out, cur = [], []
        g = torch.Generator(device=self.device); g.manual_seed(int(seed))
        with torch.no_grad():
            for _ in range(self.max_new):
                ctx = seq[-lm.block:]
                x = torch.tensor(ctx, dtype=torch.long, device=self.device)[None]
                logits = lm.model(x)[0, -1].float()
                # repetition penalty over tokens already in the sequence (Keskar 2019)
                if rep_penalty and rep_penalty != 1.0:
                    for t in set(seq + out):
                        lv = logits[t]
                        logits[t] = lv / rep_penalty if lv > 0 else lv * rep_penalty
                if auto is not None:
                    V = logits.shape[-1]
                    am = torch.zeros(V, dtype=torch.bool, device=logits.device)
                    for tid in range(V):
                        if lm._token_allowed(auto, cur, tid):
                            am[tid] = True
                    logits = logits.masked_fill(~am, float("-inf"))
                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=g).item())
                seq.append(nxt); out.append(nxt)
                if auto is not None:
                    cur, _done = lm._advance(auto, cur, nxt)
        return self.tok.decode(out).strip()


def _subject_knowledge(agent, subject, verbs):
    """Query the BRAIN's store for every fact it holds about `subject`: for each known verb, what_does(subject, verb).
    Returns the list of grounded [s, v, p] triples (the retrieval-augmentation knowledge set)."""
    facts = []
    for v in sorted(verbs):
        p = agent.what_does(subject, v)
        if p is not None:
            facts.append([subject, v, p])
    return facts


def run(cur, vocab, seed, faculty, temperature, rep_penalty):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)
    facts = cur.get("facts", [])
    agents_set = {f[0] for f in facts}
    patients_set = {f[2] for f in facts}
    actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set))
    store_keys = {tuple(f) for f in facts}

    # subjects with >=2 grounded facts (so a fluid weave of MULTIPLE facts is even possible)
    from collections import Counter
    agent_counts = Counter(f[0] for f in facts)
    rich_subjects = [s for s, c in agent_counts.items() if c >= 2]
    rich_subjects = sorted(rich_subjects)[:4]

    def _ungrounded(text):
        return [s for s in _extract_all_svos(text, agents_set, actions_set, patients_set, inflect)
                if _fact_key(s) not in store_keys]

    def _grounded_svos(text):
        return [s for s in _extract_all_svos(text, agents_set, actions_set, patients_set, inflect)
                if _fact_key(s) in store_keys]

    # --- (a)+(b) GROUNDED + MULTI-FACT FLUENCY on rich subjects ---
    grounded = []
    for s in rich_subjects:
        know = _subject_knowledge(agent, s, actions_set)                 # GATE: the brain's knowledge of s
        allow = " ".join(sorted({w for f in know for w in f}))          # BROADENED veto vocab
        text = faculty.sample(allow, f"the {s}", temperature=temperature, rep_penalty=rep_penalty, seed=seed)
        full = f"the {s} {text}"
        ug = _ungrounded(full)
        gs = _grounded_svos(full)
        distinct_gs = {tuple(x) for x in gs}
        dr = _distinct_ratio(full)
        ok = bool(len(ug) == 0 and len(distinct_gs) >= 2 and dr >= 0.5)
        grounded.append({"subject": s, "knowledge": know, "allow": allow, "text": full,
                         "grounded_svos": gs, "distinct_grounded": len(distinct_gs), "ungrounded": ug,
                         "distinct_ratio": round(dr, 3), "ok": ok})

    # --- (c) MOAT: an untaught subject -> empty knowledge -> abstain (nothing grounded to say) ---
    untaught_subjects = ["lion", "whale", "plane"]
    untaught = []
    for s in untaught_subjects:
        if s not in {w for w in vocab}:
            continue
        know = _subject_knowledge(agent, s, actions_set)
        held = (len(know) == 0)                                          # empty -> abstain
        untaught.append({"subject": s, "knowledge": know, "held": held})

    # --- (d) LOAD-BEARING: unconstrained sampling (same prompt, NO veto) drifts ungrounded ---
    free = []
    for s in rich_subjects:
        text = faculty.sample("", f"the {s}", temperature=temperature, rep_penalty=rep_penalty, seed=seed, veto_off=True)
        full = f"the {s} {text}"
        free.append({"subject": s, "text": full, "ungrounded": _ungrounded(full)})

    n_ok = sum(r["ok"] for r in grounded)
    veto_ungrounded = sum(len(r["ungrounded"]) for r in grounded)
    free_ungrounded = sum(len(r["ungrounded"]) for r in free)
    n_untaught_held = sum(r["held"] for r in untaught)
    return {"seed": seed, "temperature": temperature, "rep_penalty": rep_penalty,
            "grounded_ok": n_ok, "grounded_total": len(grounded),
            "veto_path_ungrounded": veto_ungrounded, "free_path_ungrounded": free_ungrounded,
            "untaught_held": n_untaught_held, "untaught_total": len(untaught),
            "grounded_detail": grounded, "untaught_detail": untaught, "free_detail": free}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not (os.path.exists(CKPT + ".pt") and os.path.exists(BPE)):
        print(f"NOT-RUNNABLE: 21M generator absent"); return 2

    t0 = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    print(f"[phase1-v2] brain={os.environ.get('SIM_BACKEND')} vocab={len(vocab)}; loading 21M (broadened veto + "
          f"temp-sample T={a.temperature} rep={a.rep_penalty})...", flush=True)
    err = None; per_seed = []
    try:
        faculty = BroadenedVetoFaculty(max_new=a.max_new)
        print(f"[phase1-v2] loaded ~{faculty.npar:.1f}M (dev={faculty.device})\n", flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty, a.temperature, a.rep_penalty)
            per_seed.append(r)
            print(f"  [seed {s}] grounded+multifact {r['grounded_ok']}/{r['grounded_total']} | untaught-abstain "
                  f"{r['untaught_held']}/{r['untaught_total']} | ungrounded: veto-path {r['veto_path_ungrounded']} "
                  f"vs free {r['free_path_ungrounded']}", flush=True)
            for d in r["grounded_detail"][:2]:
                print(f"      [{d['subject']}] distinct_grounded={d['distinct_grounded']} dr={d['distinct_ratio']} "
                      f"ok={d['ok']} :: {d['text']!r}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        g_ok = all(r["grounded_ok"] == r["grounded_total"] and r["grounded_total"] > 0 for r in per_seed)
        u_ok = all(r["untaught_held"] == r["untaught_total"] and r["untaught_total"] > 0 for r in per_seed)
        lb_ok = all(r["free_path_ungrounded"] > r["veto_path_ungrounded"] for r in per_seed)
        moat_ok = all(r["veto_path_ungrounded"] == 0 for r in per_seed)
        go = bool(g_ok and u_ok and lb_ok and moat_ok)
        verdict = (("GO -- FLUID multi-fact grounded generation: the broadened veto (subject's whole knowledge) + "
                    "temperature sampling weaves >=2 distinct grounded facts into fluent prose (distinct-ratio>=0.5), "
                    "0 ungrounded assertions (moat by construction), untaught abstains, and free sampling drifts more "
                    "(veto load-bearing) -- >=3 seeds. Fluid-beyond-one-proposition WITH grounding.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if g_ok else [f"grounded+multifact {[r['grounded_ok'] for r in per_seed]}/"
                                         f"{[r['grounded_total'] for r in per_seed]} (single-fact only, degenerate "
                                         "loop, or an ungrounded assertion) -> the small fact-rendering fine-tune "
                                         "may be the next lever"]) +
                       ([] if moat_ok else [f"veto-path asserted ungrounded facts {[r['veto_path_ungrounded'] for r in per_seed]} "
                                            "(the broadened veto/extractor leaked)"]) +
                       ([] if u_ok else ["untaught did not abstain (moat leak)"]) +
                       ([] if lb_ok else ["free baseline did NOT drift more (veto not load-bearing on this set)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase1_broadened_veto_fluid_multifact",
               "resolves": "Phase 1 fluidity v2: fluid multi-fact grounded generation via a broadened veto (the "
                           "subject's whole retrieved knowledge) + temperature sampling; the retrieval-augmentation "
                           "frame, moat by construction + post-hoc VERIFY.",
               "generator": f"~{faculty.npar:.1f}M TinyStories (d512/L6/H8)" if err is None else "n/a",
               "seeds": a.seeds, "temperature": a.temperature, "rep_penalty": a.rep_penalty,
               "GO": go, "verdict": verdict, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": ("if still stilted/single-fact, a small fact-rendering/dialogue fine-tune of the 21M "
                                  "(the brain-train lever) is next -- still a minimized, brain-trained, brain-gated "
                                  "generator, not the Qwen fallback. This tests fluid RENDERING, not multi-turn "
                                  "dialogue coherence (the recurrent-state + multi-referent WTA follow-on).")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase1-v2] VERDICT: {verdict}", flush=True)
    print(f"[phase1-v2] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
