# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md (B: learned keys beat fixed keys)
"""Alternative pattern-completion keys for the hippocampus (meaning-based selection), behind a parameter.

The default Hippocampus keys an episode by the cortex's layer-0 matrix-memory delta (lexical). Here the key is one of
several read-outs of the cortex's activity while it reads the experience or the cue:
  S<l> / S_all     matrix-memory (fast synaptic) delta at layer l / all layers (each layer weighted equally)
  att_all/ffn_all  token-shift state (last token's input to each layer's time-mix / channel-mix), all layers
  last<l>/last_all residual stream at the last token, layer l / all layers
  pool<l>/pool_all residual stream averaged over the tokens of the turn (a pooled read), layer l / all layers
  learned          pool_all passed through a learned low-rank correction, key = f + A(B f), trained during sleep by
                   contrastive pairs from replay (an episode vs the brain's own question about it); A starts at zero, so
                   before any sleep the learned key equals pool_all.
Selection is unchanged otherwise: centred cosine against the mean stored key, strength weighting, familiarity floor,
margin. Hippocampus defaults are untouched; KeyedHippocampus(key="S0") reproduces it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from brain.hippocampus import Episode, Hippocampus


@torch.no_grad()
def read_activity(plastic, base, text: str):
    """Read `text` from `base` with the cortex's own math (PlasticCortex, adapters as they are). Returns a dict of
    per-layer read-outs: S (matrix delta), att / ffn (token-shift), last (residual at last token), pool (mean residual
    over the turn's tokens). Lists indexed by layer."""
    pc, z = plastic, plastic.z
    ids = pc.cortex.pipe.encode(text)
    state = [s.clone() for s in base]
    x = z["emb.weight"][ids]
    v_first = torch.empty_like(x)
    last, pool = [], []
    for i in range(pc.L):
        b = f"blocks.{i}."
        xx = F.layer_norm(x, (pc.C,), weight=z[b + "ln1.weight"], bias=z[b + "ln1.bias"])
        xx, state[i * 3], state[i * 3 + 1], v_first = pc._tmix(i, xx, state[i * 3], v_first, state[i * 3 + 1])
        x = x + xx
        xx = F.layer_norm(x, (pc.C,), weight=z[b + "ln2.weight"], bias=z[b + "ln2.bias"])
        xx, state[i * 3 + 2] = pc._cmix(i, xx, state[i * 3 + 2])
        x = x + xx
        last.append(x[-1].clone())
        pool.append(x.mean(0))
    L = pc.L
    return {"S": [(state[3 * l + 1] - base[3 * l + 1]).flatten().float() for l in range(L)],
            "att": [state[3 * l].float() for l in range(L)], "ffn": [state[3 * l + 2].float() for l in range(L)],
            "last": last, "pool": pool}


def _blocks(act, variant: str):
    """The list of per-layer blocks a fixed variant is made of."""
    L = len(act["S"])
    for kind in ("S", "att", "ffn", "last", "pool"):
        if variant == f"{kind}_all":
            return act[kind]
        if variant.startswith(kind) and variant[len(kind):].isdigit():
            return [act[kind][int(variant[len(kind):])]]
        if variant == f"{kind}_mid":
            return [act[kind][L // 2]]
    raise ValueError(variant)


def centred_cos(keys_blocks, q_blocks):
    """keys_blocks: list (per block) of (n, d) tensors; q_blocks: list of (m, d). Each block is centred on the stored
    keys' mean and L2-normalised, and blocks are averaged, i.e. cosine of the concatenation of normalised blocks."""
    sims = 0.0
    for K, Q in zip(keys_blocks, q_blocks):
        mu = K.mean(0, keepdim=True) if K.shape[0] > 1 else torch.zeros_like(K[:1])
        Kc, Qc = K - mu, Q - mu
        sims = sims + F.normalize(Qc, dim=1) @ F.normalize(Kc, dim=1).T
    return sims / len(keys_blocks)


class LearnedKey(torch.nn.Module):
    """key = f + A(B f) on the pooled read of all layers (blocks normalised per layer). A = 0 at birth."""

    def __init__(self, dim: int, rank: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.B = torch.nn.Parameter(torch.randn(rank, dim, generator=g) / dim ** 0.5)
        self.A = torch.nn.Parameter(torch.zeros(dim, rank))
        self.mu = torch.nn.Parameter(torch.zeros(dim), requires_grad=False)

    @staticmethod
    def flat(blocks):
        return torch.cat([F.normalize(b, dim=-1) for b in blocks], -1)

    def forward(self, f):
        f = f - self.mu
        return f + (f @ self.B.T) @ self.A.T

    def train_pairs(self, anchors, positives, *, steps=300, lr=3e-3, tau=0.1, wd=1e-3, seed=0):
        """Contrastive (InfoNCE, both directions): anchors[i] (an episode's read) should be nearest to positives[i] (a
        replayed cue for the same episode) among all positives in the batch. anchors/positives: (n, dim)."""
        torch.manual_seed(seed)
        with torch.no_grad():
            self.mu.copy_(anchors.mean(0))
        opt = torch.optim.Adam([self.A, self.B], lr=lr, weight_decay=wd)
        tgt = torch.arange(anchors.shape[0])
        for _ in range(steps):
            ka, kp = F.normalize(self(anchors), dim=1), F.normalize(self(positives), dim=1)
            logits = ka @ kp.T / tau
            loss = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss)


class KeyedHippocampus(Hippocampus):
    """Hippocampus whose pattern-completion key is `key` (see module doc). Needs a PlasticCortex for the read-outs
    (adapters off or as consolidated). key="S0" is the default Hippocampus."""

    def __init__(self, plastic, key: str = "S0", learned: LearnedKey | None = None):
        super().__init__()
        self.plastic, self.key_name, self.learned = plastic, key, learned

    def _key(self, act):
        if self.key_name == "learned":
            with torch.no_grad():
                return self.learned(LearnedKey.flat(act["pool"]))
        blocks = _blocks(act, self.key_name)
        if len(blocks) == 1:
            return blocks[0]  # raw, as the default Hippocampus (S0 here is exactly its key)
        return torch.cat([F.normalize(b, dim=0) for b in blocks])  # each layer weighted equally

    def store(self, cortex, base, text: str, speaker: str = "User", salience: float = 1.0, permanent: bool = False):
        ep = super().store(cortex, base, text, speaker, salience, permanent)
        act = read_activity(self.plastic, base, f"{speaker}: {text}\n\n")
        ep.activity = act["pool"]  # the encoding-time activity, kept so a key learned in sleep can re-key it
        ep.key = self._key(act)
        return ep

    def rekey(self):
        """After sleep trained the learned key, recompute stored keys from the kept pooled activity."""
        if self.key_name == "learned":
            for ep in self.episodes:
                with torch.no_grad():
                    ep.key = self.learned(LearnedKey.flat(ep.activity))

    def recall(self, cortex, base, cue_text: str, speaker: str = "User", question: bool = True):
        # Same competition as Hippocampus.recall, on this key. Reuse the parent by temporarily swapping key_of.
        act = read_activity(self.plastic, base, f"{speaker}: {cue_text}\n\n")
        q = self._key(act)
        cortex.key_of = lambda delta, layer=0: q  # instance attribute shadows the static method for this call
        try:
            return super().recall(cortex, base, cue_text, speaker, question)
        finally:
            del cortex.key_of
