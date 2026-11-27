# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-27-dg-separation-port-prereg-DRAFT.md; research/findings/2026-08-19-replay-pattern-separation-
#   DG-separator-keeps-similar-memories-discriminable-6seed-GO.md; research/biology/dg-ca3-sparse-index.md;
#   brain/experiments/results/2026-11-25-routing-ceiling-read.md (DG(S0) dead as a fixed map on the layer-0 key)
# Expected-fail: partly. A FIXED random expansion keeps rank order to first order (SimHash), so DG on a dense key cannot
#   reorder much; what a sparse code can change is the SHAPE of the score (a chance floor of f, a widened top-1 margin),
#   which matters for the bar and margin, not for the rank. The spiking, plastic, neurogenesis-fed DG of the biology is not
#   ported (host-rate k-WTA stand-in, declared in brain/SCAFFOLDS.md terms: a code substitute). The routing arc's other
#   wall (strength weighting, hubs) is outside this module.
"""Dentate-gyrus-style pattern separation of the episode key, behind Brain(key="dg") (default OFF).

Mechanism (a code substitute for DG -> CA3, stated as such):
  expansion   the cortex's per-layer matrix-memory change (the same read-out the S_all key uses: each layer's delta
              L2-normalised) goes through a fixed sparse random map (each of E granule units sums `fan_in` input dims
              with random signs; perforant-path divergence). The map is linear, so the projected store can be cached per
              episode and centred on the store mean afterwards (feedforward inhibition against the mean trace).
  k-WTA       only the top `frac` of granule units stay active (PV-basket winner-take-few): a binary sparse code.
  CA3         similarity of a cue code to a stored code is overlap / k. Optional completion (`ca3_beta` > 0): the cue
              code drives a softmax-weighted superposition of the stored codes (a modern-Hopfield / attention step on the
              recurrent collaterals), is sparsified again, and the stored codes are scored against that completed code.
Controls (same code path, one thing changed): `lesion` = no k-WTA, the dense centred projection, cosine (the sparsity
lesion); `shuffle` = every stored code's units are permuted, so overlap carries no content (the shuffled-code control).

Selection is otherwise the default Hippocampus's own: strength weighting, familiarity bar, margin. The scale of
overlap / k is not that of a cosine (chance floor ~ frac), so a `dg` key has NO shipped bar: pass `familiarity` and
`margin`, or set Brain.DG_BAR. Nothing here is a default; Hippocampus and KeyedHippocampus are untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from brain.hippocampus_keys import KeyedHippocampus


def _kwta(h: torch.Tensor, k: int) -> torch.Tensor:
    """Binary code with the k largest entries of each row of h (2-D) set to 1."""
    code = torch.zeros_like(h)
    code.scatter_(1, h.topk(k, dim=1).indices, 1.0)
    return code


def matrix_blocks(d: list) -> dict:
    """{layer: matrix-memory change} from a state-layout list (slot 3l+1 of each layer; token-shift slots are None)."""
    return {l: d[3 * l + 1] for l in range(len(d) // 3) if d[3 * l + 1] is not None}


class DGSeparator:
    """The fixed expansion (E granule units, fan_in inputs each, one map per layer) and its k-WTA. No learning."""

    def __init__(self, E: int = 8192, frac: float = 0.05, fan_in: int = 64, seed: int = 42):
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], not {frac}")
        self.E, self.frac, self.fan_in, self.seed = E, frac, fan_in, seed
        self.k = max(1, round(frac * E))
        self.tag = f"{E}:{fan_in}:{seed}"          # the projection's identity (frac does not change it)
        self._maps: dict = {}

    def _map(self, layer: int, dim: int):
        key = (layer, dim)
        if key not in self._maps:
            g = torch.Generator().manual_seed(self.seed * 100003 + layer * 7919 + dim)
            idx = torch.randint(0, dim, (self.E, self.fan_in), generator=g)
            sign = torch.randint(0, 2, (self.E, self.fan_in), generator=g).float() * 2 - 1
            self._maps[key] = (idx, sign)
        return self._maps[key]

    @torch.no_grad()
    def project(self, blocks: dict) -> torch.Tensor:
        """{layer: delta tensor} -> (E,) projection of the concatenation of the L2-normalised blocks (linear)."""
        h = torch.zeros(self.E)
        for l in sorted(blocks):
            x = F.normalize(blocks[l].flatten().float().cpu(), dim=0)
            idx, sign = self._map(l, x.numel())
            h += (x[idx] * sign).sum(-1)
        return h / max(len(blocks), 1) ** 0.5

    def episode_h(self, ep) -> torch.Tensor:
        """The episode's projection, cached on the episode (never saved). Reads the plain routing change where one is kept,
        as the S_all key does."""
        cache = ep.__dict__.setdefault("_dg_h", {})
        if self.tag not in cache:
            d = getattr(ep, "route_delta", None) or ep.delta
            cache[self.tag] = self.project(matrix_blocks(d))
        return cache[self.tag]


def code_similarity(sep: DGSeparator, H: torch.Tensor, hq: torch.Tensor, lesion: bool = False, shuffle: bool = False,
                    ca3_beta: float = 0.0, ca3_iters: int = 2) -> torch.Tensor:
    """Similarity of a cue projection hq (E,) to every stored projection H (n, E). Centred on the stored mean."""
    n = H.shape[0]
    mu = H.mean(0) if n > 1 else torch.zeros_like(H[0])
    Hc, qc = H - mu, hq - mu
    if lesion:
        return F.normalize(Hc, dim=1) @ F.normalize(qc, dim=0)
    codes = _kwta(Hc, sep.k)
    if shuffle:   # every stored code's units permuted (fixed per position): same sparsity, no content
        g = torch.Generator().manual_seed(sep.seed + 17)
        codes = torch.stack([codes[i][torch.randperm(sep.E, generator=g)] for i in range(n)])
    cq = _kwta(qc[None], sep.k)[0]
    ov = codes @ cq / sep.k
    if ca3_beta > 0:
        for _ in range(ca3_iters):
            a = torch.softmax(ca3_beta * ov, dim=0)
            cq = _kwta((a @ codes)[None], sep.k)[0]
            ov = codes @ cq / sep.k
    return ov


class DGHippocampus(KeyedHippocampus):
    """KeyedHippocampus whose similarity is the DG / CA3 code overlap of the all-layer matrix-memory change. Stores no key
    (read from the episode's delta, like S_all). `familiarity` and `margin` are required: overlap / k has its own scale."""

    def __init__(self, familiarity: float, margin: float, E: int = 8192, frac: float = 0.05, fan_in: int = 64,
                 seed: int = 42, lesion: bool = False, shuffle: bool = False, ca3_beta: float = 0.0, ca3_iters: int = 2,
                 sep: DGSeparator | None = None):
        super().__init__(key="dg", familiarity=familiarity, margin=margin)
        self.sep = sep or DGSeparator(E, frac, fan_in, seed)
        self.lesion, self.shuffle, self.ca3_beta, self.ca3_iters = lesion, shuffle, ca3_beta, ca3_iters

    def _layers(self, cortex):
        return list(range(cortex.n_layers))

    @torch.no_grad()
    def similarities_from_delta(self, dl: list) -> torch.Tensor:
        """Similarities from a cue's already-read matrix change dl (cortex.matrix_delta): lets a harness read a cue once."""
        hq = self.sep.project(matrix_blocks(dl))
        H = torch.stack([self.sep.episode_h(e) for e in self.episodes])
        return code_similarity(self.sep, H, hq, self.lesion, self.shuffle, self.ca3_beta, self.ca3_iters)

    @torch.no_grad()
    def similarities(self, cortex, base, cue_text: str, speaker: str = "User"):
        _, st = cortex.read(f"{speaker}: {cue_text}\n\n", base)
        return self.similarities_from_delta(cortex.matrix_delta(st, base))
