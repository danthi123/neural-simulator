"""DA-GATED SYNAPTIC TAGGING-AND-CAPTURE — the companion process DA-gated encoding was missing (DEFAULT-OFF).

WHY THIS EXISTS (the wall reframe, 2026-09-23). DA-gated encoding read NOT load-bearing on a natural probe
(finding 2026-09-20-da-gated-encoding-not-load-bearing-on-a-natural-probe-honest-negative, branch
research/gap-da-gated-encoding-v2). A salient write (gain g>1) and a unit write recall IDENTICALLY because the store
read is magnitude-invariant. The question CLAUDE.md asks at a wall is "what does the real system run alongside this
that we replaced with a constant?" Here the answer has TWO parts, both measured before this module was written:

  1. PERSISTENCE was a constant (infinite). A stored trace never decays. In the hippocampus the early phase of LTP
     (E-LTP) decays back to baseline within ~1-3 h unless it is CAPTURED into late-phase LTP (L-LTP) by
     plasticity-related proteins (PRPs). D1/D5 dopamine drives PRP synthesis, and a salient or novel event can
     supply PRPs to synapses tagged up to ~1 h before or after it ("behavioral tagging"). Frey & Morris 1997
     (Nature 385:533); Redondo & Morris 2011 (Nat Rev Neurosci 12:17); Moncada & Viola 2007 (J Neurosci 27:7476);
     Wang, Redondo & Morris 2010 (PNAS 107:19537). Bethus, Tse & Morris 2010 (J Neurosci 30:1610) showed D1/D5
     blockade in hippocampus affects the PERSISTENCE of new memories over ~24 h, not their encoding or immediate
     recall.
  2. The synaptic BASELINE was a constant (zero). A store block's synapses hold ONLY the fact's increment, so a
     read has infinite SNR: a pilot on 2026-09-23 (seed 7) scaled a stored block by 1e-6 and it still decoded
     perfectly; only an exact 0 abstained. Real synapses carry a pre-existing strength that the LTP increment rides
     on (a CA1 E-LTP of ~+100% means the increment and the baseline are about the same size). When the increment
     decays, what is left is the baseline, which carries no information about this fact.

WHAT THIS MODULE DOES (opt-in, `BRAIN_DA_TAG_CAPTURE`, default OFF). `TagCaptureLedger` keeps, for each stored fact
block, its baseline synapses b_i (seeded, the same across arms at one seed), its written increment inc_i (the
composer's own `g * zc` write, so the DA gain stays in it), the time it was written, and whether it has been
captured. `observe_da(t, da)` records a PRP-synthesis event when the brain's own DA read reaches the Go boundary.
`advance(comp, t)` rewrites the composer's store synapses as

    w_i(t) = b_i + f_i(t) * inc_i
    f_i(t) = exp(-(t - t_w)/TAU_EARLY)                                 if not captured
           = exp(-(t_c - t_w)/TAU_EARLY) * exp(-(t - t_c)/TAU_LATE)     if captured at t_c

A block is captured when a PRP event falls in [t_w - PRP_WINDOW, t_w + TAG_WINDOW] (at t_c = max(t_w, t_p)).

THE PRP TRIGGER IS THE BRAIN'S OWN DA, THROUGH THE ENCODING EDGE. `prp_da(da)` returns the live DA level unless the
DA->encoding edge is lesioned (`BRAIN_DA_ENCODING_LESION`, the lesion the load-bearing battery already uses for
da-gated-encoding) or the capture sub-edge alone is lesioned (`BRAIN_DA_CAPTURE_LESION`). Either lesion pins the
PRP read to tonic, so nothing is captured. The threshold is the brain's existing Go/FOCUS boundary on its own DA
(`da_mode_drives_chat._DA_NEUTRAL_MAX` = 0.62), imported, not a new constant.

HOST SHORTCUTS (declared, brain-based-only burn-down). The decay, the capture decision and the threshold compare are
host arithmetic on the store synapses, at the same layer as the existing on-store homeostat. The DA they read is the
spiking SNc level driven by the spiking novelty/habituation organ. The next rung is an on-substrate late-phase
variable (the bistable consolidation variable of Clopath, Ziegler, Vasilaki, Busing & Gerstner 2008, PLoS Comput
Biol 4:e1000248) as a synaptic kernel. The world clock (hours between turns) is the environment and is legitimate
host code.

CONTRACT. DEFAULT-OFF. With `BRAIN_DA_TAG_CAPTURE` unset nothing in production constructs a ledger, so the store is
byte-identical (exact compare in the probe's selftest). No `sim/` edit; no edit to `one_brain_composer.py`.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np

# ── pre-registered biology constants (hours). See the PREREG finding 2026-09-23 for the sources and bands. ──────
TAU_EARLY_H = 1.5          # E-LTP decay time constant (Frey & Morris 1997: back to baseline within ~3 h)
TAG_WINDOW_H = 1.5         # synaptic tag lifetime after the write (Frey & Morris 1997; Redondo & Morris 2011)
PRP_WINDOW_H = 1.0         # a PRP event up to 1 h BEFORE the write still captures (Moncada & Viola 2007)
TAU_LATE_H = 30.0 * 24.0   # L-LTP maintenance (Abraham 2003: months in vivo) -- ~no loss over 24 h
BETA_BASELINE = 1.0        # baseline/increment magnitude ratio (CA1 E-LTP ~ +100% -> increment ~ baseline)
_DA_TONIC = 0.5            # the SNc tonic level (== da_encoding_drives_chat._DA_TONIC_BASELINE)


def prp_threshold() -> float:
    """The brain's own Go/FOCUS boundary on its DA read (reused by import; not a new constant)."""
    from webapp.da_mode_drives_chat import _DA_NEUTRAL_MAX
    return float(_DA_NEUTRAL_MAX)


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


def tag_capture_enabled() -> bool:
    """Master flag, DEFAULT OFF. `BRAIN_DA_TAG_CAPTURE` in {1,true,on,yes} arms the companion."""
    return _truthy("BRAIN_DA_TAG_CAPTURE", "0")


def capture_lesioned() -> bool:
    """`BRAIN_DA_CAPTURE_LESION` severs ONLY the DA->PRP (capture) sub-edge; the DA write gain is left intact."""
    return _truthy("BRAIN_DA_CAPTURE_LESION", "0")


def prp_da(da_level: float) -> float:
    """The DA level the capture machinery sees. The existing DA->encoding lesion (`BRAIN_DA_ENCODING_LESION`) and the
    capture-only lesion both pin it to tonic, so no PRP event can fire."""
    from webapp.da_encoding_drives_chat import da_encoding_lesioned
    if da_encoding_lesioned() or capture_lesioned():
        return _DA_TONIC
    return float(da_level)


class TagCaptureLedger:
    """Per-block early/late phase state for a OneBrainComposer store. Times are in HOURS on the world clock."""

    def __init__(self, seed: int, beta: float = BETA_BASELINE, tau_early_h: float = TAU_EARLY_H,
                 tag_window_h: float = TAG_WINDOW_H, prp_window_h: float = PRP_WINDOW_H,
                 tau_late_h: float = TAU_LATE_H, threshold: Optional[float] = None):
        self.seed = int(seed)
        self.beta = float(beta)
        self.tau_early_h = float(tau_early_h)
        self.tag_window_h = float(tag_window_h)
        self.prp_window_h = float(prp_window_h)
        self.tau_late_h = float(tau_late_h)
        self.threshold = prp_threshold() if threshold is None else float(threshold)
        self.blocks: List[dict] = []
        self.prp_events: List[float] = []
        self.da_log: List[tuple] = []

    # ── inputs ──────────────────────────────────────────────────────────────────────────────────────────────
    def observe_da(self, t_h: float, da_level: float) -> bool:
        """Record one turn's DA read (after `prp_da`). Returns True when it is a PRP-synthesis event."""
        d = prp_da(da_level)
        fired = d >= self.threshold
        self.da_log.append((float(t_h), float(da_level), float(d), bool(fired)))
        if fired:
            self.prp_events.append(float(t_h))
        return fired

    def _baseline(self, block_idx: int, D: int) -> np.ndarray:
        """The block's pre-existing synaptic strengths: circular complex Gaussian, E|b|^2 = beta^2, seeded by
        (seed, block index) so every arm at one seed sees the same baseline synapses."""
        rng = np.random.default_rng([self.seed, 7919, int(block_idx)])
        z = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        return (self.beta / math.sqrt(2.0)) * z

    def on_store(self, comp, t_h: float) -> int:
        """Call right after `comp.store(...)`. Registers every block written since the last call (its increment is
        the composer's own DA-gated write) and installs baseline + increment into the store. Returns #new blocks."""
        D = comp.D
        n_blocks = len(comp.store_conns) // D
        new = 0
        for i in range(len(self.blocks), n_blocks):
            sl = comp.store_conns[i * D:(i + 1) * D]
            inc = np.array([complex(w) for (_p, _q, w) in sl], dtype=np.complex128)
            pq = [(p, q) for (p, q, _w) in sl]
            self.blocks.append({"t_w": float(t_h), "inc": inc, "base": self._baseline(i, D), "pq": pq,
                                "t_c": None, "scale": 1.0})
            new += 1
        self._write(comp, t_h)
        return new

    def apply_homeostasis_scales(self, scales) -> None:
        """Keep the ledger consistent after the composer's own Turrigiano pass rescaled store blocks in place."""
        for i, s in enumerate(scales or []):
            if i < len(self.blocks):
                self.blocks[i]["base"] = self.blocks[i]["base"] * complex(s)
                self.blocks[i]["inc"] = self.blocks[i]["inc"] * complex(s)

    # ── dynamics ───────────────────────────────────────────────────────────────────────────────────────────
    def _resolve_capture(self, t_h: float) -> None:
        for blk in self.blocks:
            if blk["t_c"] is not None:
                continue
            t_w = blk["t_w"]
            hits = [tp for tp in self.prp_events
                    if (t_w - self.prp_window_h) <= tp <= (t_w + self.tag_window_h) and tp <= t_h]
            if hits:
                blk["t_c"] = max(t_w, min(hits))

    def factor(self, blk: dict, t_h: float) -> float:
        t_w = blk["t_w"]
        if blk["t_c"] is None:
            return math.exp(-max(0.0, t_h - t_w) / self.tau_early_h)
        t_c = blk["t_c"]
        return math.exp(-max(0.0, t_c - t_w) / self.tau_early_h) * math.exp(-max(0.0, t_h - t_c) / self.tau_late_h)

    def _write(self, comp, t_h: float) -> None:
        D = comp.D
        for i, blk in enumerate(self.blocks):
            f = self.factor(blk, t_h)
            w = blk["base"] + f * blk["inc"]
            comp.store_conns[i * D:(i + 1) * D] = [(p, q, complex(w[k])) for k, (p, q) in enumerate(blk["pq"])]
        # the same cache invalidation the composer's own writers use (_write_block / apply_homeostatic_scaling)
        comp._store_dirty = True
        comp._store_csr = None
        comp._persistent_dirty = True
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
        if getattr(comp, "integrated_loop", False):
            comp._seq_dirty = True
            if getattr(comp, "_fused", False):
                comp._fused_dirty = True

    def advance(self, comp, t_h: float) -> None:
        """Move the world clock to t_h: resolve captures, then rewrite the store synapses."""
        self._resolve_capture(t_h)
        self._write(comp, t_h)

    def summary(self, t_h: float) -> list:
        return [{"block": i, "t_w": b["t_w"], "captured_at": b["t_c"], "factor": self.factor(b, t_h),
                 "inc_mag": float(np.mean(np.abs(b["inc"]))), "base_mag": float(np.mean(np.abs(b["base"])))}
                for i, b in enumerate(self.blocks)]


def maybe_ledger(seed: int, **kw) -> Optional[TagCaptureLedger]:
    """The production-style constructor: None unless `BRAIN_DA_TAG_CAPTURE` is armed (default OFF)."""
    return TagCaptureLedger(seed, **kw) if tag_capture_enabled() else None
