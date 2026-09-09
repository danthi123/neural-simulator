"""SPIKING HABITUATION as the per-word NOVELTY read — the PRODUCTION WIRE-IN of the 6/6-seed-GO mechanism de-risk
(`research/runners/_spiking_habituation_novelty_derisk.py`, finding
`2026-09-09-spiking-habituation-synaptic-depression-novelty-mechanism-6seed-GO.md`), behind a DEFAULT-OFF flag.

WHAT THIS RETIRES. `webapp/da_mode_drives_chat.py::engagement_of()` computes the per-turn NOVELTY term with BARE HOST
ARITHMETIC — `sum(1 for t in tokens if t not in seen) / len(tokens)`, a permanent per-session Python `set` membership
check that never forgets. That novelty scalar (blended with a host `richness` count) becomes the message ENGAGEMENT
that drives the spiking SNc reward/context afferent, and via the rank-4 shared spiking salience afferent
(`shared_salience_afferent.py`, default-ON) reaches THREE production consumers: da-mode-drives-response (board #79),
da-gated-encoding, da-gated-curiosity-threshold. The novelty judgment ("have I heard this word before, in this
conversation") is a functional judgment over the brain's OWN history — CLAUDE.md's brain-based-only standard places any
such judgment on the neurons/synapses side of the sensation->action boundary — but it is computed today by a host
lookup table.

THE MECHANISM (identical to the de-risk, generalized to an OPEN vocabulary). Kandel PNS 6e Ch 53, Fig 53-2 (Aplysia
gill-withdrawal habituation; Pinsker et al. 1970; Castellucci & Kandel 1974; biology binding
`research/biology/spiking-habituation-novelty.md`): repeated presentation of the same input progressively DEPRESSES its
own synapse (a smaller postsynaptic response each time) "despite no change in the presynaptic action potential"; the
depression RECOVERS over a multi-second time constant during silence, so an old topic revisited after a long gap reads
as freshly novel again — a property the permanent host `set` structurally CANNOT have. Each word gets a dedicated
presynaptic-input -> readout channel wired through dense, fixed-weight, Tsodyks-Markram DEPRESSION-dominant synapses
(`cp_stp_x`, already in `sim/bridge.py`; STP regime `stp_U=0.35`, `stp_tau_d=800ms`, `stp_tau_f=10ms` — the exact
de-risked values, the mirror image of this repo's Mongillo WM facilitation regime). PRESENTING a word = a short
presynaptic BURST (`DRIVE_STEPS`) followed by a silent READOUT window (`READ_STEPS`) whose readout spike count is the
EPSP-amplitude analogue and the sole novelty score — zero host `set`/`dict` membership arithmetic on the score path.
A word's FRESHNESS = its readout response / the fresh-baseline response (measured once at build): a never-seen word
reads ~1.0 (novel); a just-repeated word reads low (habituated); an old word revisited after many intervening
presentations reads high again (recovered). novelty(message) = mean freshness over its content tokens.

OPEN VOCABULARY (recruit-on-demand — the `VocabAgnosticSpikingSampler` pattern). The de-risk fixed 3 channels; a live
conversation has an arbitrary vocabulary. This organ holds a FIXED BANK of `n_channels` block-diagonal word channels
(exactly `VocabAgnosticSpikingSampler`'s "fixed neural bank, words mapped into slots on demand" pattern — no per-word
network is grown) and a word->channel map recruited lazily on first sight of each word. When the bank is full, the
LEAST-RECENTLY-PRESENTED channel is reused for the new word: by the recovery time-constant, the LRU channel is also the
most-recovered (nearest to fresh), so a reused channel starts a new word from a near-fresh synapse WITHOUT an explicit
STP reset — the recovery mechanism and the LRU policy align. (Honest residual: for a very small bank churned faster
than `stp_tau_d`, a reused channel can inherit residual depression — a bounded-capacity INTERFERENCE that is itself
biologically realistic, not a correctness bug; the default bank is sized so eviction is rare in a normal turn.)

CONTRACT (additive, reversible, DEFAULT-OFF). `BRAIN_SPIKING_NOVELTY` truthy (1/true/on/yes) ARMS the spiking novelty
read at `engagement_of()`; UNSET or in {0,false,no,off,''} (the DEFAULT) leaves `engagement_of()`'s host `set` novelty
path UNCHANGED and this organ NEVER BUILT -> byte-identical to pre-wiring. The flip to default-ON is a SEPARATE step,
gated on an integrated `/api/brain-chat` no-regression soak (this wire-in lands default-OFF; see the finding).

LESION (the load-bearing proof). `BRAIN_SPIKING_NOVELTY_LESION=1` builds the bank with `enable_short_term_plasticity=
False` (the de-risk's OWN G5 lesion): with no synaptic depression, every present reads ~fresh regardless of repetition
-> the novelty scalar loses its dependence on word history (reverts to input-independent), even though the words still
vary turn to turn. This is DISTINCT from `BRAIN_DA_DRIVES_LESION` (which silences the downstream SNc nucleus).

REUSE-BY-IMPORT (NO `sim/` edit). The circuit build mirrors the de-risk runner's `_build_bridge`/`HabituationCircuit`
verbatim (a real `SimulationBridge` with the region framework); no `sim/` file is modified. `git diff sim/` is empty.

FUNCTIONAL CORRELATE, NOT phenomenal. This reads + reports a spiking novelty/habituation CORRELATE; it makes no claim
of subjective familiarity or experience.
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

import numpy as np

# ── STP regime + circuit sizing: the EXACT de-risked, 6/6-seed-GO values (research/runners/
#    _spiking_habituation_novelty_derisk.py). Depression-dominant Tsodyks-Markram (high U, long tau_d, negligible
#    tau_f), the opposite of this repo's Mongillo WM-facilitation regime.
STP_U = 0.35
STP_TAU_D = 800.0
STP_TAU_F = 10.0
INPOP = 25                    # presynaptic "word heard" input neurons per channel
POOL = 30                    # postsynaptic readout neurons per channel (the EPSP-amplitude analogue)
DRIVE_STEPS = 8               # a short presynaptic burst — one "utterance" of the word
READ_STEPS = 20               # the postsynaptic INTEGRATION window read AFTER drive stops
WORD_DRIVE = 650.0
PATHWAY_WEIGHT_MEAN = 45.0
PATHWAY_WEIGHT_JITTER = 5.0
_DEFAULT_N_CHANNELS = 64      # the fixed word-channel bank (recruit-on-demand); sized so eviction is rare per turn
_DEFAULT_SEED = 42


def spiking_novelty_enabled() -> bool:
    """The master flag, DEFAULT-OFF. `BRAIN_SPIKING_NOVELTY` truthy (1/true/on/yes) arms the spiking habituation
    novelty read inside `engagement_of()`; UNSET (the default) or in {0,false,no,off,''} leaves the host `set` novelty
    path unchanged and this organ never built -> byte-identical to pre-wiring. Mirrors `da_drives_enabled()`'s
    default-off semantics (this is a NEW retirement landing default-OFF; the flip to default-ON rides a separate
    integrated no-regression soak)."""
    return os.environ.get("BRAIN_SPIKING_NOVELTY", "0").strip().lower() in ("1", "true", "on", "yes")


def spiking_novelty_lesioned() -> bool:
    """`BRAIN_SPIKING_NOVELTY_LESION` truthy -> build the bank with STP OFF (the de-risk's G5 lesion): with no
    synaptic depression the novelty read no longer tracks word-repetition history (reverts to input-independent).
    The load-bearing proof — DISTINCT from `BRAIN_DA_DRIVES_LESION` (which silences the downstream SNc nucleus)."""
    return os.environ.get("BRAIN_SPIKING_NOVELTY_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def _build_bank(seed: int, n_channels: int, stp_on: bool = True):
    """Build ONE SimulationBridge holding `n_channels` block-diagonal (input -> readout) word channels — verbatim the
    de-risk's `_build_bridge`, generalized from 3 to `n_channels` channels. No `sim/` edit. `stp_on=False` is the
    lesion (no synaptic depression anywhere)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    regions, pathways = [], []
    for k in range(n_channels):
        regions.append(BrainRegion(name=f"in{k}", n_neurons=INPOP, exc_fraction=1.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                                   plastic_internal=False))
        regions.append(BrainRegion(name=f"rd{k}", n_neurons=POOL, exc_fraction=1.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                                   plastic_internal=False))
        pathways.append(RegionPathway(from_region=f"in{k}", to_region=f"rd{k}", density=1.0,
                                      weight_mean=PATHWAY_WEIGHT_MEAN, weight_jitter=PATHWAY_WEIGHT_JITTER,
                                      plastic=False))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = int(seed)
    cfg.enable_ou_process = False
    for _flag in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis", "enable_structural_plasticity",
                  "enable_reward_modulation", "enable_input_divisive_norm", "enable_nmda", "enable_bdsp"):
        setattr(cfg, _flag, False)
    cfg.enable_short_term_plasticity = bool(stp_on)      # THE mechanism (== False is the lesion)
    cfg.stp_U = STP_U
    cfg.stp_tau_f = STP_TAU_F
    cfg.stp_tau_d = STP_TAU_D
    cfg.enable_per_type_stp = False
    rt = RuntimeState()
    rt.actual_seed_used = int(seed)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b


class SpikingNoveltyHabituationOrgan:
    """A per-SESSION spiking habituation novelty read (the novelty state is per-conversation, exactly like
    `DaModeDrivesWorkspace.seen` — so this is per-workspace, NOT a process singleton). Holds ONE SimulationBridge
    with a fixed bank of `n_channels` block-diagonal word channels and a word->channel map recruited on demand
    (`VocabAgnosticSpikingSampler`'s fixed-bank/recruit-on-demand pattern). `novelty_of(tokens)` presents each content
    token through its channel and returns the mean freshness (novelty) in [0,1]; presenting also habituates the
    channel, so the read tracks the conversation's word history and recovers during silence."""

    def __init__(self, seed: int = _DEFAULT_SEED, n_channels: int = _DEFAULT_N_CHANNELS, lesion: bool = False):
        self.seed = int(seed)
        self.n_channels = int(n_channels)
        self.lesion = bool(lesion)
        self._bridge = None
        self._in_idx: List[np.ndarray] = []
        self._rd_idx: List[np.ndarray] = []
        self._num = 0
        self._fresh_ref = None            # readout response to a first-ever present of a fresh channel (cached once)
        self._word2ch: dict = {}          # word -> channel index
        self._ch_word: List[Optional[str]] = []   # channel index -> word (None = free)
        self._ch_lastuse: List[int] = []          # channel index -> present-counter at last use (LRU key)
        self._present_count = 0
        self._lock = threading.Lock()

    # ── lazy build ────────────────────────────────────────────────────────────────────────────────────────────
    def _ensure(self):
        if self._bridge is not None:
            return
        b = _build_bank(self.seed, self.n_channels, stp_on=(not self.lesion))
        rm = b.region_manager
        self._bridge = b
        self._in_idx = [np.asarray(list(rm.indices(f"in{k}")), int) for k in range(self.n_channels)]
        self._rd_idx = [np.asarray(list(rm.indices(f"rd{k}")), int) for k in range(self.n_channels)]
        self._num = int(b.core_config.num_neurons)
        self._ch_word = [None] * self.n_channels
        self._ch_lastuse = [-1] * self.n_channels
        # Cache the fresh baseline = the readout response of a fresh channel's first-ever present (the de-risk's
        # `fresh_ref`). Use the LAST channel as a dedicated reference so it never doubles as a word channel.
        self._fresh_ref = max(self._present_channel(self.n_channels - 1), 1e-9)

    # ── the spiking present (verbatim the de-risk's HabituationCircuit.present) ──────────────────────────────────
    def _present_channel(self, k: int) -> float:
        """Present (habituate + read) channel k: a short presynaptic BURST on channel k's own input population only
        (block-diagonal -> depresses ONLY channel k's synapses), then a silent READOUT window whose readout-pool
        spike count is returned. Splitting drive from read is what turns the response into a per-presentation
        EPSP-amplitude analogue instead of a single-present depression cliff (see the de-risk's tuning note)."""
        from sim.backend import from_host, to_host
        cur = np.zeros(self._num, np.float32)
        cur[self._in_idx[k]] = WORD_DRIVE
        self._bridge.cp_external_input_current[:] = from_host(cur)
        for _ in range(DRIVE_STEPS):
            self._bridge._run_one_simulation_step()
        self._bridge.cp_external_input_current[:] = 0.0
        total = 0.0
        for _ in range(READ_STEPS):
            self._bridge._run_one_simulation_step()
            fs = np.asarray(to_host(self._bridge.cp_firing_states)).astype(np.float64)
            total += fs[self._rd_idx[k]].sum()
        return float(total)

    # ── recruit-on-demand word -> channel (LRU eviction, no reset — recovery + LRU align) ───────────────────────
    def _channel_for(self, word: str) -> int:
        ch = self._word2ch.get(word)
        if ch is not None:
            return ch
        # a free channel? (reserve the last channel as the fresh reference — never a word channel)
        for k in range(self.n_channels - 1):
            if self._ch_word[k] is None:
                self._word2ch[word] = k
                self._ch_word[k] = word
                return k
        # full -> reuse the least-recently-presented word channel (the most recovered, by the recovery constant)
        lru = min(range(self.n_channels - 1), key=lambda k: self._ch_lastuse[k])
        old = self._ch_word[lru]
        if old is not None:
            self._word2ch.pop(old, None)
        self._word2ch[word] = lru
        self._ch_word[lru] = word
        return lru

    # ── the read every consumer calls ───────────────────────────────────────────────────────────────────────────
    def novelty_of(self, tokens: List[str]) -> dict:
        """Present each content token through its channel and return the mean FRESHNESS (== novelty) in [0,1] plus
        observability. A never-seen word reads ~1.0 (fresh synapse); a repeated word reads low (depressed); an old
        word revisited after intervening presentations reads high again (recovered). Presenting also habituates the
        channels (state persists across turns)."""
        with self._lock:
            self._ensure()
            per_word = []
            for w in tokens:
                k = self._channel_for(w)
                resp = self._present_channel(k)
                self._present_count += 1
                self._ch_lastuse[k] = self._present_count
                per_word.append(float(np.clip(resp / self._fresh_ref, 0.0, 1.0)))
            novelty = float(np.mean(per_word)) if per_word else 0.0
            return {"novelty": float(np.clip(novelty, 0.0, 1.0)), "n_tokens": len(tokens),
                    "per_word_freshness": per_word, "fresh_ref": float(self._fresh_ref),
                    "n_channels": self.n_channels, "lesioned": self.lesion}
