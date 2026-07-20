"""WKVFaculty — the spiking-WKV grounded-answer renderer (drop-in for FTFaculty).

Off-bridge numpy forward of the trained WKV/SSM checkpoint, bit-matching the on-bridge
rate-SSM analog in `_emerge_wkv_onbridge_derisk.py` (`rate_ssm_states` recurrence +
`_next_logits` read-out). The on-bridge spiking forward is a parity swap (RF-phase /
fully-synaptic input, gap#1) validated to +/-0.015 nat; this off-bridge numpy path is the
CPU-portable + fast reference the console renders through, and the ceiling/wiring de-risks
use it before spending on-bridge/training compute.

`answer(facts_ctx, question)` matches `FTFaculty.answer` EXACTLY (same signature) so it drops
into `FluidChat` by changing only `self.faculty = WKVFaculty()`. The RAW (un-fine-tuned)
model is a TinyStories *continuation* LM -> it RAMBLES when prompt-conditioned with a fact;
a DATA/format fine-tune (the EMERGE-57 lever = residual-B) makes it answer-not-ramble. This
class carries BOTH: `answer()` (the console interface) + `generate()`/`complete_next()` (the
ceiling probes).

NO `sim/` edit; reuse-by-import of the checkpoint. Word-level vocab (V=4000, `<unk>` trailing).
"""
from __future__ import annotations
import numpy as np

BIG_CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_big_seed42.npz"


class WKVFaculty:
    def __init__(self, ckpt: str = BIG_CKPT, max_new: int = 20, seed: int = 0):
        z = np.load(ckpt, allow_pickle=True)
        self.emb = np.asarray(z["emb.weight"], np.float64)
        self.ln_w = np.asarray(z["ln.weight"], np.float64)
        self.ln_b = np.asarray(z["ln.bias"], np.float64)
        self.Wv = np.asarray(z["Wv.weight"], np.float64)
        self.Wr = np.asarray(z["Wr.weight"], np.float64)
        self.Wo_sp = np.asarray(z["Wo_sp.weight"], np.float64)
        self.head_w = np.asarray(z["head.weight"], np.float64)
        self.head_b = np.asarray(z["head.bias"], np.float64)
        self.decay = float(np.exp(-np.log1p(np.exp(float(np.asarray(z["w"]).ravel()[0])))))
        self.words = [str(w) for w in z["words"]]
        self.V = len(self.words)
        self.D = self.emb.shape[1]
        self.w2i = {w: i for i, w in enumerate(self.words)}
        # <unk> BY NAME, not position: a format fine-tune appends <ans>/<eos> AFTER <unk>, so V-1 no longer
        # points at <unk> (it points at <eos>). The V-1 assumption silently suppressed <eos> as "unk". Index by name.
        self.unk = self.w2i.get("<unk>", self.V - 1)
        self.max_new = int(max_new)
        self.seed = int(seed)
        self.ckpt = ckpt
        self.device = "numpy(cpu)"  # off-bridge reference; the on-bridge spiking forward is the RF-phase parity swap
        self.npar = (self.emb.size + self.head_w.size + self.head_b.size + self.Wv.size + self.Wr.size
                     + self.Wo_sp.size + self.ln_w.size + self.ln_b.size) / 1e6
        self.n_invocations = 0  # gate-first moat instrument: must stay 0 on abstains (faculty never reached)

    # ---- forward (bit-matches _emerge_wkv_onbridge_derisk.rate_ssm_states + _next_logits) ----
    def _ln(self, v):
        return (v - v.mean()) / (v.std() + 1e-5) * self.ln_w + self.ln_b

    def ids(self, words):
        return [self.w2i.get(w, self.unk) for w in words]

    def in_vocab(self, w):
        return w in self.w2i and self.w2i[w] != self.unk

    def _charge(self, tid, ap, an):
        v = self.Wv @ self._ln(self.emb[tid])
        return self.decay * ap + np.maximum(v, 0.0), self.decay * an + np.maximum(-v, 0.0)

    def _logits(self, tid, ap, an):
        rh = 1.0 / (1.0 + np.exp(-(self.Wr @ self._ln(self.emb[tid]))))
        state = np.concatenate([ap, an])
        return self.head_w @ (rh * (self.Wo_sp @ state)) + self.head_b

    def _charge_prompt(self, ids):
        ap = np.zeros(self.D); an = np.zeros(self.D)
        for t in ids:
            ap, an = self._charge(t, ap, an)
        return ap, an

    def generate(self, prompt_words, max_new=None, temp=0.0, no_unk=True, stop_words=None, stop_on_repeat=False):
        """Autoregressive rollout from a WORD prompt. Returns the generated continuation words (excludes the prompt).
        stop_on_repeat: halt on a degenerate A-A / A-B-A-B loop (the WKV's short memory makes a position-dependent
        <eos> unreliable in free-running; no-repeat-ngram is the standard decoding fix, harmless for grounded SVO
        answers where no legitimate immediate repeat exists)."""
        max_new = self.max_new if max_new is None else int(max_new)
        ids = self.ids([w for w in prompt_words if w])
        if not ids:
            ids = [self.w2i.get("the", 0)]
        ap, an = self._charge_prompt(ids)
        gen = list(ids)
        n0 = len(ids)
        rng = np.random.default_rng(self.seed)
        stop = set(stop_words or [])
        for _ in range(max_new):
            lg = self._logits(gen[-1], ap, an)
            if no_unk:
                lg = lg.copy(); lg[self.unk] = -1e30
            if temp > 0.0:
                z = lg / temp; z = z - z.max(); p = np.exp(z); p = p / p.sum()
                nxt = int(rng.choice(len(p), p=p))
            else:
                nxt = int(np.argmax(lg))
            w = self.words[nxt] if 0 <= nxt < self.V else "<unk>"
            if w in stop:
                break
            gen.append(nxt)
            g = gen[n0:]
            if stop_on_repeat and (
                (len(g) >= 2 and g[-1] == g[-2]) or                              # A A
                (len(g) >= 4 and g[-1] == g[-3] and g[-2] == g[-4])):            # A B A B
                gen.pop()                                                        # drop the repeating token
                break
            ap, an = self._charge(nxt, ap, an)
        return [self.words[i] for i in gen[n0:]]

    def next_ranked(self, prompt_words, no_unk=True):
        """The full logit ranking of the next word after a prompt (for fact-completion ceiling probes)."""
        ids = self.ids([w for w in prompt_words if w]) or [self.w2i.get("the", 0)]
        ap, an = self._charge_prompt(ids)
        lg = self._logits(ids[-1], ap, an)
        if no_unk:
            lg = lg.copy(); lg[self.unk] = -1e30
        order = np.argsort(-lg)
        return [(self.words[i], float(lg[i])) for i in order]

    # ---- the FTFaculty-compatible console interface ----
    @property
    def has_markers(self):
        return "<ans>" in self.w2i and "<eos>" in self.w2i

    def answer(self, facts_ctx, question, max_new=None):
        """Match FTFaculty.answer(facts_ctx, question). If the checkpoint was format-fine-tuned (residual-B: the
        <ans>/<eos> markers are in vocab), build the learned frame '{fact words} <ans>' and generate the focused
        restatement until <eos> -- the COPY skill (answer-not-ramble, RA-faithful to the prompt fact). Else (RAW,
        un-fine-tuned) prompt-condition on the natural fact and generate a continuation (the ceiling: it RAMBLES)."""
        self.n_invocations += 1
        prompt = [w for w in facts_ctx.replace(".", " ").split() if self.in_vocab(w)]
        mn = self.max_new if max_new is None else max_new
        if self.has_markers:
            out = self.generate(prompt + ["<ans>"], max_new=mn, temp=0.0, stop_words={"<eos>"}, stop_on_repeat=True)
            out = [w for w in out if w not in ("<ans>", "<eos>")]
            return " ".join(out).strip()
        out = self.generate(prompt, max_new=mn, temp=0.0)
        return " ".join(out).strip()


if __name__ == "__main__":
    f = WKVFaculty()
    print(f"WKVFaculty loaded: V={f.V} D={f.D} decay={f.decay:.4f} ckpt={f.ckpt}")
    print("answer('the dog eats meat .','what does the dog eat ?') ->",
          repr(f.answer("the dog eats meat .", "what does the dog eat ?")))
