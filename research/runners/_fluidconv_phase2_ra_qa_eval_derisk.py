"""Phase-2 EVAL: the RA-fine-tuned ~21M generator doing FOCUSED conversational Q&A in the grounded-lang loop.

Consumes the fine-tune from `_fluidconv_phase2_ra_finetune` (gen_tinystories_ra_ft.ckpt.pt). Tests whether the
brain-trained generator now ANSWERS questions (focused, grounded) instead of rambling stories (the v3 base-model gap),
in the retrieval-augmented frame with the brain GATE + post-hoc VERIFY.

Per query (NO sim/ edit; reuse-by-import):
  (i)   GATE      -- the brain (`BrainConversationalAgent`) recalls the fact (a, v, p) OR abstains (untaught -> None).
  (ii)  RA-PROMPT -- the SAME format the fine-tune learned: "facts : the {a} {v3} {p} . question : {q} answer :".
                    On a GATE hit the retrieved fact is the context; on an abstain-test the context holds ONLY
                    facts about OTHER subjects (so the model must learn-abstain).
  (iii) ANSWER    -- the fine-tuned model generates the answer (greedy; stops at the learned SEP '*' or first
                    sentence) -- FOCUSED, not a story ramble.
  (iv)  VERIFY    -- re-parse the answer into known-entity SVOs; must match the gated fact / be empty on abstain.

METRICS (>=3 seeds): (a) FOCUSED-GROUNDED = the answer states the correct patient/agent, is FOCUSED (<= ~18 words,
not a ramble), 0 ungrounded SVOs; (b) LEARNED-ABSTAIN = untaught subject (only distractor facts in the prompt) ->
the model says "i do not know" (the learned moat); (c) RA-FAITHFULNESS = when the prompt states a DIFFERENT fact than
the model's base bias, the answer follows the PROVIDED fact (grounded to retrieval, not memorized); (d) FOCUS-vs-v3 =
the answer is much SHORTER than the v3 base-model story ramble (quantify: median answer words).

GO = focused-grounded on the taught set AND learned-abstain AND RA-faithful, >=3 seeds. HONEST/PARTIAL otherwise
(e.g. still rambles -> more fine-tune steps / stronger QA:TS mix; or ignores the retrieval -> RA-faithfulness lever).

Run: python -m research.runners._fluidconv_phase2_ra_qa_eval_derisk --seeds 42 43 44
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
from research.runners._fluidconv_phase2_ra_finetune import VERBS, FT_CKPT, BPE, ARCH, SEP  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_fluidconv_phase2_ra_qa_eval.json"
_V3SG = {b: s for (b, s, _p) in VERBS}          # base verb -> 3rd-person-sg surface (matches the fine-tune format)


def _v3(v):
    if v in _V3SG:
        return _V3SG[v]
    return v + ("es" if v.endswith(("s", "sh", "ch", "x", "z")) else "s")


class FTFaculty:
    """The RA-fine-tuned 21M generator. Prompts in the learned 'facts:...question:...answer:' format and generates
    the answer (greedy; stops at the learned SEP '*' or after ~max_new tokens)."""

    def __init__(self, ckpt=FT_CKPT, max_new=28):
        import torch
        from sim.tiny_transformer import TinyGPT
        from sim.bpe_tokenizer import BPETokenizer
        self._torch = torch
        self.tok = BPETokenizer.load(BPE)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TinyGPT(**ARCH, dropout=0.0).to(self.device)
        st = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(st["model"]); self.model.train(False)
        self.max_new = int(max_new)
        self.block = ARCH["block_size"]
        self._star = self.tok.encode(" * ")[0] if self.tok.encode(" * ") else None   # SEP stop id (57)
        self.npar = sum(p.numel() for p in self.model.parameters()) / 1e6

    def answer(self, facts_ctx, question):
        torch = self._torch
        prompt = f"facts : {facts_ctx} question : {question} answer :"
        ids = self.tok.encode(prompt)
        seq = list(ids); out = []
        with torch.no_grad():
            for _ in range(self.max_new):
                ctx = seq[-self.block:]
                x = torch.tensor(ctx, dtype=torch.long, device=self.device)[None]
                logits = self.model(x)[0, -1]
                nxt = int(torch.argmax(logits).item())               # greedy = focused, deterministic
                if self._star is not None and nxt == self._star:
                    break
                seq.append(nxt); out.append(nxt)
        text = self.tok.decode(out).strip()
        # truncate to the first sentence (focused answer)
        for end in [". ", "! ", "? "]:
            k = text.find(end)
            if k != -1:
                text = text[:k + 1]; break
        return text.strip()


def run(cur, vocab, seed, faculty):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    _teach(agent, cur)
    facts = cur.get("facts", [])
    agents_set = {f[0] for f in facts}; patients_set = {f[2] for f in facts}; actions_set = {f[1] for f in facts}
    inflect = _build_inflection_map(sorted(actions_set))
    store_keys = {tuple(f) for f in facts}

    def _svos(t): return _extract_all_svos(t, agents_set, actions_set, patients_set, inflect)
    def _ung(t): return [s for s in _svos(t) if _fact_key(s) not in store_keys]
    # only test verbs the fine-tune saw in QA format (eat/chase/like/give/make are in VERBS; 'live' is not)
    ft_verbs = {b for (b, _s, _p) in VERBS}
    test_q = [q for q in cur.get("queries_recall", []) if q["type"] == "patient" and q["cue"][1] in ft_verbs][:5]

    # --- (a) FOCUSED-GROUNDED QA ---
    grounded = []
    for q in test_q:
        a, v = q["cue"]; p = agent.what_does(a, v)                    # GATE
        if p is None:
            grounded.append({"cue": q["cue"], "abstained_gate": True, "ok": False}); continue
        ctx = f"the {a} {_v3(v)} {p} ."
        ans = faculty.answer(ctx, f"what does the {a} {v} ?")
        svos = _svos(ans); ung = _ung(ans)
        states_fact = ([a, v, p] in svos) or (p in ans.split())      # names the correct patient
        nwords = len(ans.split())
        focused = nwords <= 18
        ok = bool(states_fact and len(ung) == 0 and focused and nwords >= 2)
        grounded.append({"cue": q["cue"], "gate": [a, v, p], "ctx": ctx, "answer": ans, "svos": svos,
                         "ungrounded": ung, "states_fact": states_fact, "n_words": nwords, "focused": focused, "ok": ok})

    # --- (b) LEARNED-ABSTAIN: untaught subject, prompt holds ONLY distractor facts -> "i do not know" ---
    untaught = []
    distractor_subjects = [f for f in facts][:3]
    for uq in [x for x in cur.get("queries_moat", []) if x["type"] == "patient"][:3]:
        a, v = uq["cue"]
        # build a distractor context about OTHER (taught) subjects; the untaught 'a' is NOT in it
        ctx = " ".join(f"the {df[0]} {_v3(df[1])} {df[2]} ." for df in distractor_subjects if df[0] != a)
        ans = faculty.answer(ctx, f"what does the {a} {v} ?")
        low = ans.lower()
        abstained = ("do not know" in low or "not sure" in low or "can not say" in low or "don't know" in low)
        # also grounded-safe: it must NOT assert a fact about the untaught subject
        asserted_untaught = any(s[0] == a for s in _svos(ans))
        held = bool(abstained and not asserted_untaught)
        untaught.append({"cue": uq["cue"], "ctx": ctx, "answer": ans, "abstained": abstained,
                         "asserted_untaught": asserted_untaught, "held": held})

    # --- (c) RA-FAITHFULNESS: prompt states a DIFFERENT (in-vocab) patient than the taught one -> follow the PROMPT ---
    faithful = []
    for q in test_q[:3]:
        a, v = q["cue"]; true_p = agent.what_does(a, v)
        alt_p = next((o for o in ["ball", "cake", "hat", "toy", "bread"] if o != true_p), "ball")
        ctx = f"the {a} {_v3(v)} {alt_p} ."                          # a DIFFERENT provided fact
        ans = faculty.answer(ctx, f"what does the {a} {v} ?")
        follows_prompt = (alt_p in ans.split())                     # answered with the PROVIDED (alt) patient
        used_bias = (true_p in ans.split()) and not follows_prompt  # ignored retrieval, used its own bias
        faithful.append({"cue": q["cue"], "provided": alt_p, "true": true_p, "answer": ans,
                         "follows_prompt": follows_prompt, "used_bias": used_bias})

    n_ok = sum(r["ok"] for r in grounded)
    n_held = sum(r["held"] for r in untaught)
    n_faithful = sum(r["follows_prompt"] for r in faithful)
    med_words = sorted(r["n_words"] for r in grounded if not r.get("abstained_gate"))
    med_words = med_words[len(med_words) // 2] if med_words else 0
    return {"seed": seed, "grounded_ok": n_ok, "grounded_total": len(grounded),
            "abstain_held": n_held, "abstain_total": len(untaught),
            "faithful": n_faithful, "faithful_total": len(faithful), "median_answer_words": med_words,
            "grounded_detail": grounded, "untaught_detail": untaught, "faithful_detail": faithful}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-new", type=int, default=28)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not os.path.exists(FT_CKPT):
        print(f"NOT-RUNNABLE: fine-tuned ckpt absent ({FT_CKPT}) -- run _fluidconv_phase2_ra_finetune first"); return 2

    t0 = time.time()
    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    print(f"[phase2-eval] loading the RA-fine-tuned 21M ({FT_CKPT})...", flush=True)
    err = None; per_seed = []
    try:
        faculty = FTFaculty(max_new=a.max_new)
        print(f"[phase2-eval] loaded ~{faculty.npar:.1f}M (dev={faculty.device})\n", flush=True)
        for s in a.seeds:
            r = run(cur, vocab, s, faculty)
            per_seed.append(r)
            print(f"  [seed {s}] focused-grounded {r['grounded_ok']}/{r['grounded_total']} | learned-abstain "
                  f"{r['abstain_held']}/{r['abstain_total']} | RA-faithful {r['faithful']}/{r['faithful_total']} | "
                  f"median answer words {r['median_answer_words']}", flush=True)
            for d in r["grounded_detail"][:3]:
                if not d.get("abstained_gate"):
                    print(f"      Q: what does the {d['cue'][0]} {d['cue'][1]}? -> {d['answer']!r} (ok={d['ok']})", flush=True)
            for d in r["untaught_detail"][:1]:
                print(f"      ABSTAIN Q({d['cue']}): {d['answer']!r} held={d['held']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        g_ok = all(r["grounded_ok"] == r["grounded_total"] and r["grounded_total"] > 0 for r in per_seed)
        u_ok = all(r["abstain_held"] == r["abstain_total"] and r["abstain_total"] > 0 for r in per_seed)
        f_ok = all(r["faithful"] == r["faithful_total"] and r["faithful_total"] > 0 for r in per_seed)
        go = bool(g_ok and u_ok and f_ok)
        verdict = (("GO -- the RA-fine-tuned 21M ANSWERS questions FOCUSED + grounded (states the fact, not a story "
                    "ramble), LEARNED-ABSTAINS on untaught subjects ('i do not know'), and is RA-FAITHFUL (follows the "
                    "provided fact over its own bias) -- >=3 seeds. Focused conversational Q&A on a minimized, "
                    "brain-trained, brain-gated generator.") if go else
                   ("HONEST/PARTIAL -- " + "; ".join(
                       ([] if g_ok else [f"focused-grounded {[r['grounded_ok'] for r in per_seed]}/"
                                         f"{[r['grounded_total'] for r in per_seed]} (still rambles/wrong/ungrounded)"]) +
                       ([] if u_ok else [f"learned-abstain {[r['abstain_held'] for r in per_seed]} (moat not learned)"]) +
                       ([] if f_ok else [f"RA-faithful {[r['faithful'] for r in per_seed]} (ignores retrieval, uses bias)"]))))
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "fluidconv_phase2_ra_qa_eval", "GO": go, "verdict": verdict,
               "resolves": "does the RA-fine-tuned 21M do FOCUSED grounded conversational Q&A (vs the v3 base-model "
                           "story ramble), with learned-abstain + RA-faithfulness, in the brain-gated loop?",
               "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per_seed,
               "HONEST_CEILING": "tests focused single-turn grounded Q&A; multi-turn coherence + open breadth are the "
                                 "follow-ons (recurrent state + retrieval-augmentation + abstention)."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[phase2-eval] VERDICT: {verdict}", flush=True)
    print(f"[phase2-eval] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
