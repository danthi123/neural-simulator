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

⛔ v1/v2 (`TagCaptureLedger`) = A HOST DECISION RULE (adversarial review 2026-09-23): the scalar compare below decides
every 24 h outcome once the DA trace separates at 0.62, so its results must NOT be credited to the brain. It is kept
only so the v1/v2 artifacts reproduce. v3 (`SynapticTagCaptureLedger`, further down) removes the compare: the DA acts
through per-synapse tag / PRP / bistable late-phase state driven by the spiking D1 population's rate.

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


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# v3 (2026-09-23 fix round): DA ACTS THROUGH SYNAPSES. No host threshold compare decides capture.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# WHY. The adversarial review of v2 found that `TagCaptureLedger` decides the 24 h outcome with a host rule: a scalar
# `da >= 0.62` compare marks a block "captured", and a per-block exp() factor then erases everything else. Once the DA
# trace separates at 0.62 (G1), every 24 h result follows arithmetically. That is a host decision, not the brain's.
#
# WHAT CHANGES. `SynapticTagCaptureLedger` replaces the compare and the per-block capture flag with per-synapse state
# variables driven by the brain's own spiking activity, in the tag-trigger-late-phase form of Clopath, Ziegler,
# Vasilaki, Busing & Gerstner 2008 (PLoS Comput Biol 4:e1000248) and the synaptic tagging-and-capture hypothesis
# (Frey & Morris 1997; Redondo & Morris 2011):
#
#   a(t)   D1 activation = the firing-rate excess of the spiking D1 population (the same `write_gain` IZH2007 CA1
#          pyramidal pool whose rate already sets the write gain, research/runners/_da_write_gain_spiking_derisk.py)
#          when the brain's own SNc DA level is broadcast onto it, normalized to that population's own tonic..arousal
#          rate span: a = clip((r(DA) - r(tonic)) / (r(1.24) - r(tonic)), 0, 1). A NEURAL read (spiking, OU noise on).
#   p(t)   cell-wide plasticity-related-protein pool:    dp/dt = -p / tau_p + kappa * a(t)          (kappa = 1/tau_p)
#   h_k(t) synapse-specific tag, set by the local early-LTP amplitude at the write: h_k(t_w) = |inc_k|, decays tau_tag
#   z_k(t) bistable late-phase variable per synapse:   tau_z dz/dt = -z (z - 1/2) (z - 1) + gamma * p * h_k
#   e(t)   early-phase fraction: exp(-(t - t_w) / tau_e)
#   w_k(t) = b_k + inc_k * (e + z_k (1 - e))          (b_k = the pre-existing baseline, as in v1/v2)
#
# A synapse is "captured" only in the sense that its own z_k crossed its own unstable fixed point (1/2) under the drive
# gamma*p*h_k; nothing compares DA to a number. The outcome depends on the TIME COURSE and MAGNITUDE of the brain's DA
# (summation over turns, PRP from an event before the write, the write gain via the tag), so it is not implied by G1.
#
# THE ONE CALIBRATED CONSTANT, gamma, is fixed a priori (no brain data, no gate seed): the minimal gamma at which a
# unit-tag synapse written at t=0 and driven for 5 min (the canonical novelty-exposure duration of Moncada & Viola 2007
# and Wang, Redondo & Morris 2010) at the D1 activation the population produces at the brain's own Go/FOCUS boundary DA
# (0.62, `da_mode_drives_chat._DA_NEUTRAL_MAX`) ends with z > 1/2 at 12 h. So "a 5-min exposure at the Go boundary just
# captures" is the declared operating point; everything else (8-min conversations, graded DA, shared PRP) is left to
# the dynamics.
#
# HOST SHORTCUTS STILL DECLARED. The per-synapse ODEs are host-integrated equations (the same category as every
# plasticity rule in the engine, which are also host arithmetic on synapse state); their constants are pre-registered;
# the rate->activation normalization is host arithmetic on a measured spike count (the same two-anchor calibration the
# write gain already uses). The world clock is environment.

TAU_TAG_H = 1.5            # tag lifetime (Frey & Morris 1997: tag present < 3 h)
TAU_PRP_H = 1.0            # PRP availability (Moncada & Viola 2007: novelty 1 h before a weak event still rescues it)
TAU_Z_H = 0.5              # late-phase switching time constant (declared, a priori)
CAPTURE_PROTOCOL_MIN = 5.0 # canonical novelty exposure (Moncada & Viola 2007; Wang, Redondo & Morris 2010): 5 min
CAPTURE_CHECK_H = 12.0     # the calibration protocol reads z at 12 h (well past tau_p, tau_tag, tau_z)
KERNEL_DT_H = 1.0 / 240.0  # 15 s Euler step for z (tau_z / dt = 120); p integrated exactly per step


def _z_drift(z):
    return -z * (z - 0.5) * (z - 1.0)


def kernel_capture_single(a_level, gamma, tau_p=TAU_PRP_H, tau_tag=TAU_TAG_H, tau_z=TAU_Z_H,
                          dur_h=CAPTURE_PROTOCOL_MIN / 60.0, h0=1.0, t_end=CAPTURE_CHECK_H, dt=KERNEL_DT_H) -> float:
    """z at t_end for ONE synapse (tag h0 at t=0) under a constant D1 activation `a_level` on [0, dur_h). Pure kernel,
    no brain: used only for the a-priori gamma calibration and the band fail-ability precondition."""
    p = z = 0.0
    kappa = 1.0 / tau_p
    dec_p = math.exp(-dt / tau_p)
    n = int(round(t_end / dt))
    for i in range(n):
        t = i * dt
        a = a_level if t < dur_h else 0.0
        p = p * dec_p + kappa * a * tau_p * (1.0 - dec_p)
        h = h0 * math.exp(-t / tau_tag)
        z = z + (dt / tau_z) * (_z_drift(z) + gamma * p * h)
    return z


def calibrate_gamma(a_crit, tau_p=TAU_PRP_H, tau_tag=TAU_TAG_H, tau_z=TAU_Z_H, lo=1e-4, hi=1e3, iters=60) -> float:
    """Minimal gamma at which the 5-min protocol at D1 activation `a_crit` captures (z > 1/2 at 12 h). Bisection on a
    monotone predicate; deterministic; no brain data."""
    if not (a_crit > 0.0):
        raise ValueError("calibrate_gamma: a_crit must be > 0 (got %r)" % (a_crit,))
    if kernel_capture_single(a_crit, hi, tau_p, tau_tag, tau_z) <= 0.5:
        raise ValueError("calibrate_gamma: no capture even at gamma=%g" % hi)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if kernel_capture_single(a_crit, mid, tau_p, tau_tag, tau_z) > 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def critical_activation(gamma, tau_p=TAU_PRP_H, tau_tag=TAU_TAG_H, tau_z=TAU_Z_H, iters=50) -> Optional[float]:
    """The minimal constant D1 activation that captures under the 5-min protocol at these kernel constants (None if
    even a=1 does not capture). Used for the band fail-ability precondition."""
    if kernel_capture_single(1.0, gamma, tau_p, tau_tag, tau_z) <= 0.5:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kernel_capture_single(mid, gamma, tau_p, tau_tag, tau_z) > 0.5:
            hi = mid
        else:
            lo = mid
    return hi


class SpikingD1Activation:
    """a(DA) read off the spiking D1 (`write_gain`) population: set the shared DA concentration, step the pool, count
    spikes, normalize to the pool's own tonic..arousal rate span. The reader is the production one
    (`_da_write_gain_spiking_derisk._get_reader(42, False)`, the population the production write gain reads)."""

    def __init__(self, reader_seed: int = 42, n_cal: Optional[int] = None):
        from research.runners import _da_write_gain_spiking_derisk as W
        self._W = W
        self.r = W._get_reader(int(reader_seed), False)
        n = W.N_CAL_REPEATS if n_cal is None else int(n_cal)
        self.rate_tonic = W._read_rate_hz_repeated(self.r["bridge"], self.r["idx"], _DA_TONIC, self.r["snapshot"], n)
        self.rate_hi = float(self.r["rate_hi"])
        self.rate_go = W._read_rate_hz_repeated(self.r["bridge"], self.r["idx"], prp_threshold(), self.r["snapshot"], n)
        self.a_go = self.activation_of_rate(self.rate_go)

    def activation_of_rate(self, rate: float) -> float:
        span = self.rate_hi - self.rate_tonic
        if span <= 0:
            raise ValueError("D1 population does not discriminate tonic from arousal (span %.3f Hz)" % span)
        return float(min(1.0, max(0.0, (rate - self.rate_tonic) / span)))

    def read(self, da: float):
        W, r = self._W, self.r
        W._restore(r["bridge"], r["snapshot"])
        rate = W._read_rate_hz(r["bridge"], r["idx"], float(da))
        W._restore(r["bridge"], r["snapshot"])
        return self.activation_of_rate(rate), float(rate)


class SynapticTagCaptureLedger:
    """Per-synapse tag / PRP / bistable late-phase state on a OneBrainComposer store (times in HOURS, world clock).

    Usage per turn: `observe_turn(t, dur, da)` schedules the D1 drive of that turn; after a `comp.store(...)` call
    `on_store(comp, t)`; `advance(comp, t)` integrates to t and rewrites the store synapses."""

    def __init__(self, seed: int, gamma: float, d1=None, beta: float = BETA_BASELINE, tau_early_h: float = TAU_EARLY_H,
                 tau_tag_h: float = TAU_TAG_H, tau_p_h: float = TAU_PRP_H, tau_z_h: float = TAU_Z_H,
                 dt_h: float = KERNEL_DT_H, block_offset: int = 0):
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.d1 = d1                       # object with .read(da) -> (a, rate); None -> caller supplies a directly
        self.beta = float(beta)
        self.tau_early_h, self.tau_tag_h = float(tau_early_h), float(tau_tag_h)
        self.tau_p_h, self.tau_z_h, self.dt_h = float(tau_p_h), float(tau_z_h), float(dt_h)
        # store blocks [0, block_offset) are NOT managed (chat wiring: the build-time knowledge written before the
        # ledger existed). 0 = every block managed = the v3 runner's behaviour, byte-for-byte.
        self.block_offset = int(block_offset)
        self.n_external_rescales = 0
        self.n_external_rewrites = 0
        self.t = 0.0
        self.p = 0.0
        self.p_max = 0.0
        self.blocks: List[dict] = []
        self.drive: List[tuple] = []       # (t0, t1, a_effective)
        self.turn_log: List[dict] = []

    # the baseline is the SAME seeded draw as v1/v2 (so beta has the same meaning)
    _baseline = TagCaptureLedger._baseline

    def observe_turn(self, t0_h: float, dur_h: float, da_level: float, a_override: Optional[float] = None) -> float:
        """Schedule one turn's D1 drive. Both lesions act on the EDGE: BRAIN_DA_ENCODING_LESION / BRAIN_DA_CAPTURE_LESION
        pin the DA the D1 pool receives to tonic (`prp_da`); BRAIN_DA_CAPTURE_LESION also zeroes the D1->PRP coupling.
        The pool is still read (it fires at its tonic rate + noise)."""
        d = prp_da(da_level)
        if a_override is not None:
            a, rate = float(a_override), None
        else:
            a, rate = self.d1.read(d)
        coupling = 0.0 if capture_lesioned() else 1.0
        a_eff = a * coupling
        self.drive.append((float(t0_h), float(t0_h) + float(dur_h), a_eff))
        self.turn_log.append({"t_h": float(t0_h), "da": float(da_level), "da_seen_by_d1": float(d),
                              "d1_rate_hz": rate, "a": float(a), "a_eff": float(a_eff)})
        return a_eff

    def _a_at(self, t: float) -> float:
        for (t0, t1, a) in reversed(self.drive):
            if t0 <= t < t1:
                return a
        return 0.0

    def _integrate(self, t_target: float) -> None:
        dt = self.dt_h
        kappa = 1.0 / self.tau_p_h
        while self.t < t_target - 1e-12:
            h_step = min(dt, t_target - self.t)
            dec = math.exp(-h_step / self.tau_p_h)
            a = self._a_at(self.t)
            self.p = self.p * dec + kappa * a * self.tau_p_h * (1.0 - dec)
            self.p_max = max(self.p_max, self.p)
            for blk in self.blocks:
                h = blk["h0"] * math.exp(-(self.t - blk["t_w"]) / self.tau_tag_h)
                z = blk["z"]
                blk["z"] = z + (h_step / self.tau_z_h) * (_z_drift(z) + self.gamma * self.p * h)
            self.t += h_step

    def on_store(self, comp, t_h: float) -> int:
        self._integrate(t_h)
        D = comp.D
        n_blocks = len(comp.store_conns) // D
        new = 0
        for i in range(self.block_offset + len(self.blocks), n_blocks):
            sl = comp.store_conns[i * D:(i + 1) * D]
            inc = np.array([complex(w) for (_p, _q, w) in sl], dtype=np.complex128)
            self.blocks.append({"t_w": float(t_h), "inc": inc, "base": self._baseline(i, D),
                                "pq": [(p, q) for (p, q, _w) in sl], "h0": np.abs(inc).astype(np.float64),
                                "z": np.zeros(D, dtype=np.float64)})
            new += 1
        self._write(comp)
        return new

    apply_homeostasis_scales = TagCaptureLedger.apply_homeostasis_scales

    def sync_from_store(self, comp, t_h: float) -> dict:
        """Chat wiring: other writers act on the SAME store synapses between the ledger's own writes (the composer's
        Turrigiano pass rescales a block in place on the idle tick; reconsolidation rewrites a block in place). Before
        the ledger rewrites the store it reads each managed block back and compares it with what it last wrote:
          * a pure multiplicative change (complex least-squares scale, relative residual < 1e-6) is an external
            rescale -> applied to the block's baseline AND increment (the same bookkeeping as
            `apply_homeostasis_scales`, which the v3 runner called with the scale vector directly);
          * any other change is an external rewrite -> the writer's new weights are taken as a FRESH early-LTP
            increment written now (new tag |inc|, z reset to 0), the baseline kept. DECLARED host bookkeeping.
        Blocks the ledger has not written yet are skipped. Returns counts."""
        D = comp.D
        n_scale = n_rew = 0
        for i, blk in enumerate(self.blocks):
            last = blk.get("last_w")
            j = self.block_offset + i
            sl = comp.store_conns[j * D:(j + 1) * D]
            if last is None or len(sl) != D:
                continue
            cur = np.array([complex(w) for (_p, _q, w) in sl], dtype=np.complex128)
            if np.array_equal(cur, last):
                continue
            den = np.vdot(last, last)
            s = (np.vdot(last, cur) / den) if abs(den) > 0 else 0.0
            nrm = float(np.linalg.norm(cur))
            resid = float(np.linalg.norm(cur - s * last)) / (nrm if nrm > 0 else 1.0)
            if resid < 1e-6:
                blk["base"] = blk["base"] * complex(s)
                blk["inc"] = blk["inc"] * complex(s)
                n_scale += 1
            else:
                blk["inc"] = cur
                blk["pq"] = [(p, q) for (p, q, _w) in sl]
                blk["t_w"] = float(t_h)
                blk["h0"] = np.abs(cur).astype(np.float64)
                blk["z"] = np.zeros(D, dtype=np.float64)
                n_rew += 1
            blk["last_w"] = cur
        self.n_external_rescales += n_scale
        self.n_external_rewrites += n_rew
        return {"rescaled": n_scale, "rewritten": n_rew}

    def weight_factor(self, blk: dict) -> np.ndarray:
        e = math.exp(-max(0.0, self.t - blk["t_w"]) / self.tau_early_h)
        return e + blk["z"] * (1.0 - e)

    def _write(self, comp) -> None:
        D = comp.D
        for i, blk in enumerate(self.blocks):
            w = blk["base"] + self.weight_factor(blk) * blk["inc"]
            j = self.block_offset + i
            comp.store_conns[j * D:(j + 1) * D] = [(p, q, complex(w[k])) for k, (p, q) in enumerate(blk["pq"])]
            blk["last_w"] = np.array([complex(x) for x in w], dtype=np.complex128)
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
        self._integrate(t_h)
        self._write(comp)

    def summary(self) -> list:
        return [{"block": i, "t_w": b["t_w"], "tag0_mean": float(np.mean(b["h0"])),
                 "z_mean": float(np.mean(b["z"])), "z_min": float(np.min(b["z"])), "z_max": float(np.max(b["z"])),
                 "frac_synapses_z_gt_half": float(np.mean(b["z"] > 0.5)),
                 "inc_mag": float(np.mean(np.abs(b["inc"]))), "base_mag": float(np.mean(np.abs(b["base"])))}
                for i, b in enumerate(self.blocks)]
