"""Q2 kill-safe gate: validated Generator-F PROPOSES tokens; the
validated no-confab grounded memory VETOES ungrounded next-tokens
PER-TOKEN at decode time. The validated moat (abstention_gate '650',
byte-UNMODIFIED) gates answer-vs-abstain FIRST via grounded_decode
(byte-UNMODIFIED; LM NEVER touched on abstain = no-confab by
construction). Per-token veto makes ungrounded-entity-rate ~0 BY
CONSTRUCTION (MECHANICAL, NOT the discriminator); the DISCRIMINATING
signature is constrained NON-VACUITY (>= _CDC_MIN_GROUNDED_CONTENT
distinct on-prop content words) vs the unconstrained Generator-G drift
regime + shuffled_grounding. Decisive slice isolates the COMPOSITION
via a FROZEN grounded source (G.20 retrieval separately validated;
full-retrieval wiring a later increment). Generator-F inference is
torch (reused validated artifact, INFERENCE ONLY -- no new training/
autograd; inference mode set via model.train(False), which is the
100%-equivalent of putting the module in eval mode). CUDA when
available (the decisive run MUST use the GPU; CPU only if CUDA absent,
logged). FROZEN _CDC_* via constrained_decode_core NEVER tuned.
Honest propagation = the CONTROLLER's post-run job. ASCII."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np

from sim.grounded_decode import grounded_decode
from research.runners.abstention_gate import gate, DEFAULT_THRESHOLD
from research.runners.generator_g_core import ungrounded_entity_rate, \
    FUNCTION_WORDS
from research.runners.constrained_decode_core import (
    cdc_verdict, cdc_scale_confidence, nonvacuous_answered,
    _CDC_SCALE_LADDER)

_GEN_F = "research/findings/raw/g11_bg/generator_f_gate.ckpt.s42.real"

_GROUNDED = {
 "max":"max is a big friendly dog","lily":"lily has a small red ball",
 "tom":"tom found a shiny blue key","sue":"sue likes warm sweet bread",
 "ben":"ben rides a fast green bike","mia":"mia keeps a soft white cat",
 "leo":"leo plants a tall oak tree","ana":"ana paints a bright yellow sun",
 "sam":"sam sails a small wood boat","kai":"kai flies a bright red kite",
 "joy":"joy sings a slow sweet song","rex":"rex digs a deep round hole",
 "ivy":"ivy grows a sweet purple plum","dan":"dan builds a strong stone wall",
 "eve":"eve reads a long old book","gus":"gus cooks a hot tasty soup",
 "pam":"pam sews a warm wool coat","ned":"ned mends a torn blue sail",
 "uma":"uma rows a long thin canoe","ole":"ole carves a small pine duck",
 "fay":"fay bakes a round nut cake","hal":"hal sweeps a wide dusty barn",
 "wes":"wes feeds a tame brown hen","zoe":"zoe ties a tight square knot",
}
_UNGROUNDED = ["zarn","qexel","drovil","plonk","vexin","wun"]


class _GroundedConstrainedLM:
    """Duck-typed IDENTICALLY to generator_g_gate._TinyGPTLM (same
    __init__; same .generate_ids(prompt_ids, max_new)) so it drops into
    grounded_decode BYTE-UNMODIFIED. generate_ids applies the per-token
    grounded VETO: allowed vocab = token ids whose normalized decoded
    surface is empty (punct/space) OR every word is in (allow_words
    UNION FUNCTION_WORDS). allow_words = the prompt's own words (=
    retrieved proposition, since grounded_decode passes prompt_ids =
    tok.encode(retrieved_text)). mode: 'constrained' veto on;
    'unconstrained' veto OFF (= Generator-G regime); 'shuffled' veto
    allow_words from self._shuffle_text (a DIFFERENT proposition)."""
    def __init__(self, ckpt_prefix, mode="constrained", block_size=128):
        import torch
        from sim.tiny_transformer import TinyGPT
        from sim.bpe_tokenizer import BPETokenizer
        self._torch = torch
        self.mode = mode
        self._shuffle_text = None
        self.tok = BPETokenizer.load(ckpt_prefix + ".bpe.json")
        V = self.tok.vocab_size
        self.block = block_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TinyGPT(vocab_size=V, d_model=256, n_layer=4,
                             n_head=4, block_size=block_size,
                             dropout=0.0)
        st = torch.load(ckpt_prefix + ".pt", map_location=self.device)
        self.model.load_state_dict(st["model"])
        self.model.train(False)            # inference mode (eval-equiv)
        self.model.to(self.device)
        self._allow_cache = {}

    def _norm_words(self, s):
        import re
        return [t for t in (re.sub(r"[^\w]", "", w.lower())
                            for w in str(s).split()) if t]

    def _allowed_mask(self, allow_text):
        if allow_text in self._allow_cache:
            return self._allow_cache[allow_text]
        torch = self._torch
        allow = set(self._norm_words(allow_text)) | set(FUNCTION_WORDS)
        V = self.tok.vocab_size
        mask = torch.zeros(V, dtype=torch.bool)
        for tid in range(V):
            surf = self._norm_words(self.tok.decode([tid]))
            if not surf or all(w in allow for w in surf):
                mask[tid] = True
        mask = mask.to(self.device)
        self._allow_cache[allow_text] = mask
        return mask

    def generate_ids(self, prompt_ids, max_new):
        torch = self._torch
        seq = list(prompt_ids) if prompt_ids else [0]
        out = []
        use_veto = self.mode in ("constrained", "shuffled")
        if self.mode == "shuffled" and self._shuffle_text is not None:
            allow_text = self._shuffle_text
        else:
            allow_text = (self.tok.decode(list(prompt_ids))
                          if prompt_ids else "")
        mask = self._allowed_mask(allow_text) if use_veto else None
        with torch.no_grad():
            for _ in range(int(max_new)):
                ctx = seq[-self.block:]
                x = torch.tensor(ctx, dtype=torch.long,
                                 device=self.device)[None]
                logits = self.model(x)[0, -1]
                if use_veto:
                    logits = logits.masked_fill(~mask, float("-inf"))
                nxt = int(torch.argmax(logits).item())
                seq.append(nxt)
                out.append(nxt)
        return out


def _params(tiny):
    if tiny:
        return dict(ladder=(_CDC_SCALE_LADDER[0],), max_new=12,
                    n_ungrounded=3)
    return dict(ladder=_CDC_SCALE_LADDER, max_new=40, n_ungrounded=6)


def _run_rung(K, seeds, lm_c, lm_u, lm_s, max_new, n_ung):
    items = list(_GROUNDED.items())[:K]
    props = [p for _, p in items]
    ung = list(_UNGROUNDED)[:n_ung]
    per_seed = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = list(range(len(items)))
        rng.shuffle(order)
        c_uer, u_uer, s_uer, c_nv, s_nv = [], [], [], [], []
        for idx in order:
            subj, prop = items[idx]
            ranked = [(subj, 900.0, "kb")]
            r = grounded_decode(ranked, lm_c, lm_c.tok,
                                retrieved_text=prop, query=subj,
                                threshold=DEFAULT_THRESHOLD,
                                max_new=max_new)
            ct = r["text"] or ""
            c_uer.append(ungrounded_entity_rate(ct, prop))
            c_nv.append(1.0 if nonvacuous_answered(ct, prop) else 0.0)
            ru = grounded_decode(ranked, lm_u, lm_u.tok,
                                 retrieved_text=prop, query=subj,
                                 threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            u_uer.append(ungrounded_entity_rate(ru["text"] or "", prop))
            lm_s._shuffle_text = props[(idx + 1) % len(props)]
            rs = grounded_decode(ranked, lm_s, lm_s.tok,
                                 retrieved_text=prop, query=subj,
                                 threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            st_ = rs["text"] or ""
            s_uer.append(ungrounded_entity_rate(st_, prop))
            s_nv.append(1.0 if nonvacuous_answered(st_, prop) else 0.0)
        n_abst = bare = 0
        for subj in ung:
            if gate([], DEFAULT_THRESHOLD) is None:
                bare += 1
            ra = grounded_decode([], lm_c, lm_c.tok, retrieved_text="",
                                 query=subj, threshold=DEFAULT_THRESHOLD,
                                 max_new=max_new)
            if ra["abstained"]:
                n_abst += 1
        nu = max(1, len(ung))
        per_seed[seed] = {
            "unconstrained_uer": float(np.mean(u_uer)),
            "constrained_uer": float(np.mean(c_uer)),
            "constrained_nonvac_rate": float(np.mean(c_nv)),
            "shuffled_uer": float(np.mean(s_uer)),
            "shuffled_nonvac_rate": float(np.mean(s_nv)),
            "bare_moat_abstain_rate": bare / nu,
            "abstain_on_ungrounded_rate": n_abst / nu}
    verdict = cdc_verdict(per_seed)
    nv_mean = float(np.mean(
        [per_seed[s]["constrained_nonvac_rate"] for s in per_seed]))
    return {"K": K, "verdict": verdict,
            "constrained_nonvac_rate_mean": nv_mean}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--ckpt", default="research/findings/raw/g11_bg/"
                    "constrained_decode_gate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    if not (os.path.exists(_GEN_F + ".pt")
            and os.path.exists(_GEN_F + ".bpe.json")):
        print("NOT-RUNNABLE: Generator-F artifact absent"); return 2
    P = _params(a.tiny)
    lm_c = _GroundedConstrainedLM(_GEN_F, mode="constrained")
    lm_u = _GroundedConstrainedLM(_GEN_F, mode="unconstrained")
    lm_s = _GroundedConstrainedLM(_GEN_F, mode="shuffled")
    print("DEVICE=%s (CUDA=%s) -- decisive run MUST be cuda"
          % (lm_c.device, lm_c._torch.cuda.is_available()))
    resume = str(a.ckpt) + ".resume.json"
    done = {}
    if Path(resume).exists():
        try:
            done = {int(k): v for k, v in json.loads(
                Path(resume).read_text()).get("done", {}).items()}
        except (ValueError, OSError):
            done = {}
    rungs = []
    t0 = time.time()
    try:
        for K in P["ladder"]:
            if K in done:
                rungs.append(done[K]); continue
            rg = _run_rung(K, a.seeds, lm_c, lm_u, lm_s,
                           P["max_new"], P["n_ungrounded"])
            rungs.append(rg); done[K] = rg
            tmp = resume + ".tmp"
            Path(tmp).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp).write_text(json.dumps(
                {"done": {str(k): v for k, v in done.items()}}))
            os.replace(tmp, resume)
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial resume flushed; resumable")
        return 130
    sc = (cdc_scale_confidence(rungs) if not a.tiny else
          {"scale_confident": False,
           "classification": "TINY (toy; NOT propagated)"})
    out = {"ladder": rungs, "scale_confident": sc["scale_confident"],
           "scale_classification": sc["classification"],
           "scale_reason": sc.get("reason", ""),
           "device": lm_c.device, "tiny": bool(a.tiny),
           "note": ("TINY toy verdict -- NOT propagated" if a.tiny
                    else "multi-rung scale-confidence verdict -- "
                    "recompute from this JSON; no re-run/no tuning"),
           "elapsed_seconds": round(time.time() - t0, 1),
           "HONEST_CEILING": ("scale-confidence PoC: per-token grounded "
               "constrained decoding stays NON-VACUOUSLY faithful by "
               "construction + no scale degradation; NOT open-ended "
               "fluent composition, NOT an LLM, NOT conversation-"
               "solved; constrained decoding TRADES fluency for "
               "faithfulness BY DESIGN")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("SCALE=%s class=%s device=%s"
          % (out["scale_confident"], out["scale_classification"],
             out["device"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
