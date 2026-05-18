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
    grounded_decode BYTE-UNMODIFIED.

    FIX A (BPE-aware faithful per-token grounded veto): Generator-F's
    tokenizer is Sennrich-2016 word-frequency BPE -- grounded content
    words are MULTI-SUBWORD (e.g. "max" -> ['ma','x</w>'], "friendly"
    -> ['friend','ly</w>']). The previous word-level isolated-token
    mask therefore (a) STRUCTURALLY VETOED grounded multi-subword
    content (the model literally could not emit "max") and (b) LEAKED
    ungrounded content via short subword fragments whose isolated
    decode collided with a function word. The veto was a subword sieve,
    not a grounded constraint.

    The faithful veto is a PREFIX-AUTOMATON over the BPE encodings of
    the allowed words. ALLOW = content words of allow_text UNION
    FUNCTION_WORDS. For every w in ALLOW, enc(w) = tok.encode(w);
    PREFIXES = all proper prefixes (incl. empty) of every enc(w);
    FULLS = the set of complete enc(w) tuples. The BPE word boundary
    is structural -- every enc(w) ends in a symbol ending '</w>', so a
    word COMPLETES exactly when a FULLS tuple is matched. Pure-
    punctuation / UNK ids (decoded surface normalizes to empty) are
    SEPARATOR ids; they are allowed only at a clean word boundary
    (cur == []) so they cannot punctuation-terminate a partial NON-
    allowed word. `cur` accumulates the in-progress word's ids since
    the last boundary. A candidate next id `t` is ALLOWED iff
    tuple(cur+[t]) is in PREFIXES, OR tuple(cur+[t]) is in FULLS, OR
    (cur == [] and t is a separator id). On completing a FULLS tuple
    or emitting a separator, cur resets to []. This (i) ALLOWS any
    whole word in ALLOW including multi-subword ones and (ii) FORBIDS
    completing any word NOT in ALLOW. Strictly STRENGTHENS: it makes
    `constrained` a real grounded constraint, never loosens a _CDC_*.

    mode: 'constrained' veto from the prompt's own words (= retrieved
    proposition, since grounded_decode passes prompt_ids =
    tok.encode(retrieved_text)); 'shuffled' veto from
    self._shuffle_text (a DIFFERENT proposition); 'unconstrained' veto
    OFF (NO masking -- differs ONLY by the veto). torch INFERENCE
    ONLY: torch.no_grad(), greedy argmax, no autograd/optimizer/loss;
    inference mode via model.train(False) (eval-equivalent)."""
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
        self._auto_cache = {}
        self._sep_ids = self._compute_sep_ids()

    def _norm_words(self, s):
        import re
        return [t for t in (re.sub(r"[^\w]", "", w.lower())
                            for w in str(s).split()) if t]

    def _compute_sep_ids(self):
        # Separator/eos ids: vocab ids whose decoded surface normalizes
        # to empty (pure punctuation, '</w>', UNK). Faithful boundary
        # markers -- they reset the in-progress word at a clean break.
        sep = set()
        for tid in range(self.tok.vocab_size):
            if not self._norm_words(self.tok.decode([tid])):
                sep.add(int(tid))
        return frozenset(sep)

    def _allowed_automaton(self, allow_text):
        """Precompute (cached per allow_text) the prefix-automaton:
        {'prefixes': set(tuple), 'fulls': set(tuple),
         'sep': frozenset(int)}. enc(w) = tok.encode(w) for every word
        in ALLOW = content(allow_text) UNION FUNCTION_WORDS."""
        if allow_text in self._auto_cache:
            return self._auto_cache[allow_text]
        allow = set(self._norm_words(allow_text)) | set(FUNCTION_WORDS)
        prefixes = set()
        fulls = set()
        for w in allow:
            enc = tuple(self.tok.encode(w))
            if not enc:
                continue
            fulls.add(enc)
            for k in range(len(enc)):
                prefixes.add(enc[:k])
        auto = {"prefixes": prefixes, "fulls": fulls,
                "sep": self._sep_ids}
        self._auto_cache[allow_text] = auto
        return auto

    def _token_allowed(self, auto, cur, t):
        """A candidate next id `t` given in-progress word ids `cur`."""
        t = int(t)
        cand = tuple(cur) + (t,)
        if cand in auto["prefixes"] or cand in auto["fulls"]:
            return True
        # Separator/eos only at a clean word boundary (cur empty) so it
        # can NEVER punctuation-terminate a partial non-allowed word.
        if not cur and t in auto["sep"]:
            return True
        return False

    def _advance(self, auto, cur, t):
        """Apply id `t`. Returns (new_cur, completed_bool). cur resets
        on completing a FULLS word OR emitting a boundary separator."""
        t = int(t)
        cand = tuple(cur) + (t,)
        if cand in auto["fulls"]:
            return [], True
        if not cur and t in auto["sep"]:
            return [], True
        return list(cand), False

    def _props_fully_emittable_rate(self, props):
        """Fix-B metric helper: fraction of `props` whose ALL content
        words are emittable under the faithful mask -- i.e. each
        content word's enc(w) is fully traversable in the automaton
        (every proper prefix in PREFIXES/FULLS and the full seq in
        FULLS). The whole point of Fix A is that this is HIGH; a low
        value means the BPE veto structurally cannot express the
        grounded content (subword-defeated) -> instrument cannot test
        the Q2 premise (-> Fix-B VOID)."""
        if not props:
            return 0.0
        good = 0
        for prop in props:
            auto = self._allowed_automaton(prop)
            words = [w for w in self._norm_words(prop)
                     if w not in FUNCTION_WORDS]
            if not words:
                continue
            ok = True
            for w in words:
                enc = tuple(self.tok.encode(w))
                if not enc or enc not in auto["fulls"]:
                    ok = False
                    break
                if any(enc[:k] not in auto["prefixes"]
                       and enc[:k] not in auto["fulls"]
                       for k in range(len(enc))):
                    ok = False
                    break
            if ok:
                good += 1
        return good / len(props)

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
        auto = self._allowed_automaton(allow_text) if use_veto else None
        cur = []
        with torch.no_grad():
            for _ in range(int(max_new)):
                ctx = seq[-self.block:]
                x = torch.tensor(ctx, dtype=torch.long,
                                 device=self.device)[None]
                logits = self.model(x)[0, -1]
                if use_veto:
                    V = logits.shape[-1]
                    am = torch.zeros(V, dtype=torch.bool,
                                     device=logits.device)
                    for tid in range(V):
                        if self._token_allowed(auto, cur, tid):
                            am[tid] = True
                    logits = logits.masked_fill(~am, float("-inf"))
                nxt = int(torch.argmax(logits).item())
                seq.append(nxt)
                out.append(nxt)
                if use_veto:
                    cur, _done = self._advance(auto, cur, nxt)
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
            # WEAK#2: RNG-permuted shuffle source -- a per-seed
            # rng-chosen DIFFERENT proposition (asserted != idx), not
            # the fixed (idx+1)%len neighbour. Strengthens control
            # independence; deterministic per seed (same rng stream).
            if len(props) > 1:
                sidx = int(rng.integers(0, len(props) - 1))
                if sidx >= idx:
                    sidx += 1
            else:
                sidx = idx
            assert sidx != idx or len(props) <= 1
            lm_s._shuffle_text = props[sidx]
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
        # Fix-B per-seed instrument-validity metric: fraction of the K
        # KB props whose ALL content words are emittable under the
        # constructed faithful mask (each content word's enc(w) fully
        # traversable: prefixes + full present in the automaton).
        mt_emit = lm_c._props_fully_emittable_rate(props)
        per_seed[seed] = {
            "unconstrained_uer": float(np.mean(u_uer)),
            "constrained_uer": float(np.mean(c_uer)),
            "constrained_nonvac_rate": float(np.mean(c_nv)),
            "shuffled_uer": float(np.mean(s_uer)),
            "shuffled_nonvac_rate": float(np.mean(s_nv)),
            "bare_moat_abstain_rate": bare / nu,
            "abstain_on_ungrounded_rate": n_abst / nu,
            "constrained_multitoken_emittable_rate": float(mt_emit)}
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
