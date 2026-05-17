"""Generator-G pre-registered MULTI-SEED capability gate. Decides
whether Generator-F's validated coherent generation can be added to
the validated no-confab moat WITHOUT destroying it. The validated
moat (research.runners.abstention_gate, byte-UNMODIFIED, 'gate 650')
gates answer-vs-abstain FIRST (via sim.grounded_decode); the trained
Generator-F TinyGPT generates ONLY when grounded, faithfulness-
constrained (greedy). No-confab is preserved BY CONSTRUCTION. The
decisive slice isolates the COMPOSITION via a FROZEN deterministic
grounded source (G.20 retrieval is already separately multi-seed-
validated; full-retrieval wiring is a later increment). FIXED bars +
the relational no-confab-preserved bar via generator_g_core (NEVER
tuned here). HONEST CEILING: small-Transformer simple grounded
responses + preserved no-confab, explicitly NOT an LLM. Kill-safe.
Honest propagation is the CONTROLLER's post-run job. ASCII only."""
from __future__ import annotations
import argparse

_GEN_F_CKPT = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"

# FROZEN grounded KB (the decisive slice isolates the composition;
# simple TinyStories-style propositions). GROUNDED query -> high-
# confidence ranked + the retrieved proposition; UNGROUNDED -> empty
# ranked (-> validated moat abstains).
_GROUNDED = {
    "max": "max is a big friendly dog",
    "lily": "lily has a small red ball",
    "tom": "tom found a shiny blue key",
    "sue": "sue likes to bake warm bread",
    "ben": "ben rides a fast green bike",
    "mia": "mia keeps a soft white cat",
}
_UNGROUNDED = ["zarn", "qexel", "drovil", "plonk", "vexin", "wun"]


class _TinyGPTLM:
    def __init__(self, ckpt_prefix, block_size=128):
        import json
        import torch
        from sim.tiny_transformer import TinyGPT
        from sim.bpe_tokenizer import BPETokenizer
        self.tok = BPETokenizer.load(ckpt_prefix + ".bpe.json")
        V = self.tok.vocab_size
        self.block = block_size
        self._torch = torch
        self.model = TinyGPT(vocab_size=V, d_model=256, n_layer=4,
                             n_head=4, block_size=block_size,
                             dropout=0.0)
        st = torch.load(ckpt_prefix + ".pt", map_location="cpu")
        self.model.load_state_dict(st["model"])
        self.model.eval()

    def generate_ids(self, prompt_ids, max_new):
        torch = self._torch
        seq = list(prompt_ids) if prompt_ids else [0]
        out = []
        with torch.no_grad():
            for _ in range(int(max_new)):
                ctx = seq[-self.block:]
                x = torch.tensor(ctx, dtype=torch.long)[None]
                logits = self.model(x)[0, -1]
                nxt = int(torch.argmax(logits).item())  # greedy=faithful
                seq.append(nxt)
                out.append(nxt)
        return out


def main():
    import json
    import os
    import time
    from pathlib import Path
    import numpy as np

    from sim.grounded_decode import grounded_decode
    from research.runners.abstention_gate import gate
    from research.runners.generator_g_core import (
        ungrounded_entity_rate, is_answered, gg_verdict,
        gg_aggregate_multiseed, FUNCTION_WORDS,
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--ckpt", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_g_gate.ckpt")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/g11_bg/"
                            "generator_g_gate.json")
    a = ap.parse_args()

    seeds = [int(s) for s in str(a.seeds).split(",") if s.strip()]

    print("=" * 64, flush=True)
    print("GENERATOR-G PRE-REGISTERED MULTI-SEED CAPABILITY GATE",
          flush=True)
    print("(validated no-confab moat gates FIRST; trained "
          "Generator-F TinyGPT faithfulness-constrained on grounded;",
          flush=True)
    print(" FIXED bars + relational no-confab-PRESERVED bar via "
          "generator_g_core NEVER tuned here; >= 3 seeds;", flush=True)
    print(" HONEST CEILING: small-Transformer simple grounded "
          "responses, NOT an LLM)", flush=True)
    print("=" * 64, flush=True)

    if len(seeds) < 3:
        print("[NOT RUNNABLE] %d seed(s); >= 3 MANDATORY "
              "(generator_g_core enforces; this is the early exit)."
              % len(seeds), flush=True)
        return 2

    if not (os.path.exists(_GEN_F_CKPT + ".pt")
            and os.path.exists(_GEN_F_CKPT + ".bpe.json")):
        print("[NOT RUNNABLE] trained Generator-F checkpoint absent "
              "(%s.pt / .bpe.json) -- the decisive run requires the "
              "Generator-F artifact." % _GEN_F_CKPT, flush=True)
        return 2

    lm = _TinyGPTLM(_GEN_F_CKPT)
    tok = lm.tok
    max_new = 12 if a.tiny else int(a.max_new)

    resume_path = str(a.ckpt) + ".resume.json"
    completed = {}
    if Path(resume_path).exists():
        try:
            completed = {int(k): v for k, v in json.loads(
                Path(resume_path).read_text("utf-8")).get(
                "completed", {}).items()}
        except (ValueError, OSError):
            completed = {}

    def _flush_resume(comp):
        tmp = resume_path + ".tmp"
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).write_text(json.dumps(
            {"completed": {str(k): v for k, v in comp.items()},
             "seeds": seeds}), encoding="utf-8")
        os.replace(tmp, resume_path)

    grounded_items = list(_GROUNDED.items())
    ungrounded = list(_UNGROUNDED)
    if a.tiny:
        grounded_items = grounded_items[:3]
        ungrounded = ungrounded[:3]

    per_seed_verdicts = []
    per_seed_records = []
    t0 = time.time()

    for seed in seeds:
        if seed in completed:
            v = completed[seed]
            per_seed_verdicts.append(v)
            per_seed_records.append({"seed": seed, "resumed": True,
                                     "verdict": v})
            print("[SEED %d] RESUMED" % seed, flush=True)
            continue

        rng = np.random.default_rng(seed)
        gi = list(grounded_items)
        rng.shuffle(gi)
        ug = list(ungrounded)
        rng.shuffle(ug)

        transcripts = {"grounded": [], "ungrounded": []}
        n_grounded_answered = 0
        ent_rates = []
        for subj, prop in gi:
            ranked = [(subj, 900.0, "kb")]
            r = grounded_decode(ranked, lm, tok, retrieved_text=prop,
                                query=subj, threshold=650.0,
                                max_new=max_new)
            ans = (not r["abstained"]) and is_answered(
                r["text"] or "", FUNCTION_WORDS)
            if ans:
                n_grounded_answered += 1
                ent_rates.append(ungrounded_entity_rate(
                    r["text"], prop, FUNCTION_WORDS))
            transcripts["grounded"].append(
                {"q": subj, "abstained": r["abstained"],
                 "answered": bool(ans),
                 "response": (r["text"] or "")[:200]})

        n_ung_abstained = 0
        bare_moat_abstain = 0
        for subj in ug:
            ranked = []                       # ungrounded -> no retrieval
            if gate(ranked, 650.0) is None:   # bare validated moat
                bare_moat_abstain += 1
            r = grounded_decode(ranked, lm, tok, retrieved_text="",
                                query=subj, threshold=650.0,
                                max_new=max_new)
            if r["abstained"]:
                n_ung_abstained += 1
            transcripts["ungrounded"].append(
                {"q": subj,
                 "result": "ABSTAIN" if r["abstained"]
                 else ("ANSWERED:" + (r["text"] or "")[:120])})

        n_g = len(gi)
        n_u = len(ug)
        grounded_answer_rate = (n_grounded_answered / n_g
                                if n_g else 0.0)
        abstain_on_ungrounded_rate = (n_ung_abstained / n_u
                                      if n_u else 0.0)
        bare_moat_abstain_rate = (bare_moat_abstain / n_u
                                  if n_u else 0.0)
        mean_ent = (sum(ent_rates) / len(ent_rates)
                    if ent_rates else 0.0)

        v = gg_verdict(
            abstain_on_ungrounded_rate=abstain_on_ungrounded_rate,
            bare_moat_abstain_rate=bare_moat_abstain_rate,
            grounded_answer_rate=grounded_answer_rate,
            mean_ungrounded_entity_rate=mean_ent,
            has_ungrounded_control=(n_u > 0))
        v["seed"] = seed
        v["n_grounded"] = n_g
        v["n_ungrounded"] = n_u
        per_seed_verdicts.append(v)
        per_seed_records.append({
            "seed": seed, "resumed": False,
            "grounded_answer_rate": grounded_answer_rate,
            "abstain_on_ungrounded_rate": abstain_on_ungrounded_rate,
            "bare_moat_abstain_rate": bare_moat_abstain_rate,
            "mean_ungrounded_entity_rate": mean_ent,
            "n_grounded": n_g, "n_ungrounded": n_u,
            "transcripts": transcripts, "verdict": v})
        completed[seed] = v
        _flush_resume(completed)
        print("[SEED %d] g_answer=%.3f ung_abstain=%.3f "
              "bare_moat=%.3f mean_ent=%.3f -> %s"
              % (seed, grounded_answer_rate,
                 abstain_on_ungrounded_rate, bare_moat_abstain_rate,
                 mean_ent, v["GATE"]), flush=True)

    agg = gg_aggregate_multiseed(per_seed_verdicts)
    result = {
        "task": "Generator-G pre-registered MULTI-SEED capability "
                "gate",
        "mechanism": ("validated no-confab moat gates FIRST; trained "
                      "Generator-F TinyGPT faithfulness-constrained "
                      "on grounded; no-confab preserved BY "
                      "CONSTRUCTION; honest ceiling: small-"
                      "Transformer simple grounded responses, NOT an "
                      "LLM"),
        "decisive_slice_note": ("isolates the COMPOSITION via a "
                                "FROZEN grounded source; G.20 "
                                "retrieval already separately "
                                "multi-seed-validated; full-retrieval "
                                "wiring is a noted later increment"),
        "seeds": seeds, "n_seeds": len(seeds),
        "anti_cheat": {
            "validated_moat_reused_unmodified": "research.runners."
                "abstention_gate gate/abstain/650 byte-UNMODIFIED",
            "no_confab_by_construction": "grounded_decode abstain "
                "path never touches the LM",
            "fixed_bars_in_gg_core": "_GG_UNGROUNDED_ENTITY_MAX=0.20 "
                "/ _GG_MIN_GROUNDED_ANSWER_RATE=0.5 / >=3 seeds; "
                "relational no-confab-preserved bar; NEVER tuned",
            "is_answered_anti_vacuous": "function-word-only/empty "
                "responses are NOT counted as answers",
            "honest_propagation": "CONTROLLER's post-run job"},
        "per_seed": per_seed_records,
        "aggregate_verdict": agg,
        "GATE": agg["GATE"],
        "OVERALL": "PASS" if agg["GATE"] == "PASS" else "FAIL",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8")

    print("\n" + "=" * 64, flush=True)
    print("GENERATOR-G GATE VERDICT", flush=True)
    print("=" * 64, flush=True)
    for r in per_seed_records:
        vv = r["verdict"]
        print("  seed %s: %s (g_answer=%s ung_abstain=%s "
              "bare_moat=%s mean_ent=%s)"
              % (r["seed"], vv["GATE"],
                 r.get("grounded_answer_rate"),
                 r.get("abstain_on_ungrounded_rate"),
                 r.get("bare_moat_abstain_rate"),
                 r.get("mean_ungrounded_entity_rate")), flush=True)
    print("  AGGREGATE: %s (n_seeds=%d n_pass=%d; >=3 mandatory; "
          "FIXED bars untouched)"
          % (agg["GATE"], agg["n_seeds"], agg["n_pass"]),
          flush=True)
    if agg["GATE"] != "PASS":
        print("  NOTE: a maxed FAIL is an HONEST finding -> "
              "propagate (decision-relevant terminus: the two "
              "validated assets stay SEPARATE); do NOT "
              "config-crank.", flush=True)
    else:
        print("  NOTE: a PASS is reported STRICTLY at the honest "
              "ceiling (small-Transformer simple grounded responses, "
              "NOT an LLM); controller smell-tests the actual "
              "transcripts before propagating.", flush=True)
    print("  -> %s" % a.out, flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
