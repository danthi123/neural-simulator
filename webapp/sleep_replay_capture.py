"""SLEEP-REPLAY-TRIGGERED SYNAPTIC CAPTURE: the route by which an ORDINARY fact told once survives the night when the
DA tag-and-capture ledger is on (DEFAULT-OFF: `BRAIN_SLEEP_REPLAY_CAPTURE`). Branch research/sleep-replay-capture.

WHY (the wall reframe, 2026-09-24). With `BRAIN_DA_TAG_CAPTURE` armed (webapp/da_tag_capture_chat.py), a stored
block's early-phase trace decays (TAU_EARLY_H 1.5 h) unless PRPs arrive while its tag is live, and the ONLY PRP source
is the brain's waking DA crossing the Go boundary. So a plainly told fact is gone by the next day (G3 of the 6/6 GO;
`ordinary_fact_flip_forgetting` on branch research/da-tag-capture-ltm-on). Asked "what does the real system run
alongside tagging-and-capture that we replaced with a constant?", the answer is SLEEP: the hippocampus re-activates the
day's modified synapses in sharp-wave ripples during NREM, and the model had no sleep-phase event at all (the PRP
supply during the night was the constant zero). Biology binding: research/biology/sleep-replay-tag-capture.md.
  * Buzsaki 2006 (Rhythms of the Brain, p.346-347): the pathways modified in the waking brain are replayed during
    sharp waves; "the selective and repeated activation of the same neurons and synapses by the sharp-wave events"
    guides the hours-long molecular cascade back to the synapses that learned, and (note 28) the sharp-wave replay
    mechanism could replace the Frey-Morris tag "or the two processes could work in parallel".
  * Clopath, Ziegler, Vasilaki, Busing & Gerstner 2008 (PLoS Comput Biol 4:e1000248, PMC2596310): PRP synthesis "is
    triggered if the total number of set tags is larger than a critical number"; the phasic dopamine signal that
    co-stimulation releases "is assumed to be proportional to the number of tags"; DA facilitates the trigger.
  * Kandel 6e ch.54: late LTP "recruits the cAMP and PKA signaling pathway"; D1/D5 receptors enhance late LTP
    "through production of cAMP and activation of PKA"; a weak tetanus is captured when "delivered within 2-3 hours"
    of the strong one. Kandel 6e ch.44: sleep "usually begins with a rapid descent into stage N3".

WHAT HAPPENS (only with the flag ON, only on a sleep-depth idle, only for blocks the ledger manages):
  1. SLEEP ONSET. The engine's own sleep-depth criterion (`continuous_engine.SLEEP_IDLE_SEC`, 5 min of idle) after the
     last observed turn starts a sleep episode; its first N3 period carries ONE SWR epoch (sleep "begins with a rapid
     descent into stage N3"). A new observed turn starts a new episode. The ledger is event-driven, so a night that one
     idle tick jumps over is integrated with the epoch placed at its own time.
  2. SWR REACTIVATION (BRAIN-BASED SELECTION). At the epoch the ledger first rewrites the store to its
     value at that time, then EVERY managed block's trigger is driven once on the composer's own resonate-and-fire
     substrate and the store's own cleanup reads it back (`OneBrainComposer._block_role_scores`, the same read the
     metacog hedge uses). R_i = the smallest cleanup decisiveness margin over agent/action/patient: ~0 when the block
     has decayed to its baseline synapses (the reactivation is noise), high when the fact's trace is still expressed.
     No host list or sort picks the facts: every managed trigger is driven identically; what the substrate returns
     decides how strongly each one replays.
  3. RE-TAG (Buzsaki note 28, the replay working in parallel with the tag). Each reactivated block's synapses get a
     replay tag R_i * |inc_k| (the tag is the local early-LTP amplitude, as in v3); the synapse's tag is the larger of
     its write tag and its replay tag, decaying with the same TAU_TAG_H.
  4. SWR-COUPLED PRP (Clopath: phasic DA proportional to the tags set; Kandel: late LTP via D1/D5 -> cAMP -> PKA). The
     SWR bout co-releases DA = tonic + (DA_SWR_FULL - tonic) * min(1, sum_i R_i), read by the SAME spiking D1
     population the waking capture reads (`SpikingD1Activation`), for SWR_BOUT_H split into 30-s reads, feeding the
     SAME cell-wide PRP pool. Both existing lesions act on this edge unchanged: BRAIN_DA_ENCODING_LESION /
     BRAIN_DA_CAPTURE_LESION pin the DA the D1 pool sees to tonic (`prp_da`), and BRAIN_DA_CAPTURE_LESION zeroes the
     D1->PRP coupling. So the DA system stays the gate on every late-phase capture, waking or sleeping.
  5. The existing v3 per-synapse late-phase dynamics then decide capture. Nothing here compares anything to a
     threshold; a block is kept only if its own synapses' z cross 1/2.

r2 (branch research/sleep-replay-capture-r2; pre-registered as Amendment 1 of the sleep-replay-capture prereg):
  * ONE EPOCH PER NIGHT. Before r2 an idle stretch ran one epoch in total, however many nights it spanned. Now night k
    of an idle stretch starts at (end of waking) + sleep onset + k * NIGHT_PERIOD_H (24 h). Any protocol whose recall
    comes within 24 h + sleep onset of the last turn (every one-night group) still runs exactly one epoch.
  * THE END OF WAKING can be an environment mark (`da_tag_capture_chat.mark_awake`, the battery's `awake_*` world
    step): the body stayed awake without conversing, so sleep onset is measured from the end of that interval.
  * SLEEP DOWNSCALING (`BRAIN_SLEEP_DOWNSCALING`, default OFF, read only inside an epoch). After each night's
    reactivation, every managed block's learned increment is multiplied by 1 - SHY_DELTA * (1 - R_i): the slow-wave
    period depresses synapses by ~18 % (de Vivo et al. 2017, Science 355:507, "The axon-spine interface (ASI)
    decreased ~18% after sleep compared with wake", PMC5313037), except in proportion to how strongly the block's own
    reactivation drove the read-out (Gonzalez-Rueda et al. 2018, Neuron 97:1244, "connections contributing to
    postsynaptic spiking are protected against this synaptic weakening", PMC5873548). Under the replay-edge lesion
    R_eff = 0, so nothing is protected. The pre-existing baseline b is NOT downscaled (declared: in this model it
    stands for strength that belongs to other memories; scaling it with the increment would be invisible to the
    magnitude-invariant read). Its host steps: the multiply, the constant, and the choice of R_i (the same cleanup
    margin the replay uses) as the protection read.

THE REPLAY-EDGE LESION. `BRAIN_SLEEP_REPLAY_CAPTURE_LESION=1` severs the reactivation's effect: the substrate reads
still run (same compute, same substrate state) but R_eff = 0, so no replay tag is set and the SWR bout's DA stays
tonic. The D1 pool is still read at tonic (it fires at its tonic rate + noise), exactly as the v3 lesions do.

WHY ONE EPOCH PER NIGHT, NOT ONE PER NREM CYCLE (measured 2026-09-24, fake-substrate test, before any brain run). A
5-cycle variant (90-min cycles, Buzsaki's "four or five non-REM/REM cycles") RESURRECTED noise-level traces: a fact told
8 h before sleep (read-back R = 0.035, at baseline) was captured by cycle 4, because a sub-threshold late-phase z is
expressed in the weight, which raises the next cycle's read-back, which raises the next re-tag -- a runaway with no
brake. The real night runs the brake alongside: sleep's own synaptic downscaling (Kandel 6e ch.44, Tononi & Cirelli:
smaller synapses are "reduced during sleep ... competing weaker ones are removed"). That companion is NOT modeled here,
so only the first-N3 epoch runs. With one epoch the same test captures a fact told <= 2 h before sleep and not one told
>= 3 h before (R 0.30 -> 0.17 across that boundary), matching the 2-3 h capture window above. Modelling the later cycles
needs the downscaling companion first (named next rung).

CONSTANTS (all a priori, none fitted to a gate seed; see the pre-registration):
  SWR_BOUT_H = the v3 CAPTURE_PROTOCOL_MIN (5 min), reused, not new; SWR_SUBREAD_H = 30 s
  (== da_tag_capture_chat.TURN_DRIVE_H); DA_SWR_FULL = 1.24 = the D1 pool's own calibration ceiling
  (`_da_write_gain_spiking_derisk._DA_CAL_HI`, where its activation is 1 by construction), so a fully coherent replay
  drives the pool to the top of its own measured range.

HOST SHORTCUTS (declared, brain-based-only burn-down):
  - the sleep-onset clock is a host timer (the body's sleep/wake clock, the same class as the engine's IDLE_SEC /
    SLEEP_IDLE_SEC scaffolds); one SWR epoch per night stands in for the replay of the first N3 period;
  - the per-block loop that drives every managed trigger is host iteration (it drives ALL of them identically; the
    selection is the substrate's response, not the loop);
  - R_i is the composer's decisiveness margin, (peak - runner_up)/peak of the cleanup membrane scores: host arithmetic
    on a substrate read (the same read the confidence gate and the metacog hedge already use);
  - the replay tag R_i * |inc_k| uses the ledger's stored increment (the same bookkeeping v3 uses for the write tag);
  - the replay-to-DA map tonic + (DA_SWR_FULL - tonic) * min(1, sum R) is a declared operating point standing in for
    the SWR-coupled VTA/SNc burst; the DA level is not produced by the spiking SNc organ during sleep (next rung);
  - the per-synapse ODEs are the v3 host-integrated ones.
This is a synaptic-capture route, not "consolidation" in the docs/TERMS.md sense (no transfer, no source lesion).

CONTRACT. DEFAULT-OFF. With `BRAIN_SLEEP_REPLAY_CAPTURE` unset, `ChatTagCapture._catch_up` never enters its sleep
branch, no replay tag is ever set (the ledger's `h_rep` branch is skipped), and no key is added to the reply: the
tag-and-capture path is byte-identical (tests/test_sleep_replay_capture.py). Inert without `BRAIN_DA_TAG_CAPTURE`
(there is no ledger to act on). No `sim/` edit; no edit to one_brain_composer.py.
"""
from __future__ import annotations

import math
import os
from typing import Callable, List, Optional

import numpy as np

from webapp.da_tag_capture import (_DA_TONIC, CAPTURE_PROTOCOL_MIN, TAU_TAG_H, capture_lesioned, prp_da)

NIGHT_PERIOD_H = 24.0                  # r2: a new night every 24 h of continued idle (the circadian period; host clock)
SHY_DELTA = 0.18                       # r2 downscaling: de Vivo et al. 2017, axon-spine interface ~18% smaller after sleep
SWR_BOUT_H = CAPTURE_PROTOCOL_MIN / 60.0   # the v3 canonical exposure (5 min), reused as the night's SWR bout
SWR_SUBREAD_H = 30.0 / 3600.0          # == da_tag_capture_chat.TURN_DRIVE_H: one D1 read per 30 s of drive
N_SWR_SUBREADS = int(round(SWR_BOUT_H / SWR_SUBREAD_H))   # 10
DA_SWR_FULL = 1.24                     # == _da_write_gain_spiking_derisk._DA_CAL_HI (the D1 pool's own a=1 anchor)
FACT_ROLES = ("agent", "action", "patient")
_K_REACT = 200000                      # private-RNG stream offsets (never collide with the turn reads k=0..n_turns)
_K_SWR_D1 = 300000


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


def replay_capture_enabled() -> bool:
    """Master flag, DEFAULT OFF. `BRAIN_SLEEP_REPLAY_CAPTURE` in {1,true,on,yes} arms the sleep route."""
    return _truthy("BRAIN_SLEEP_REPLAY_CAPTURE")


def replay_capture_lesioned() -> bool:
    """`BRAIN_SLEEP_REPLAY_CAPTURE_LESION` severs the reactivation -> synapse / DA edge (the reads still run)."""
    return _truthy("BRAIN_SLEEP_REPLAY_CAPTURE_LESION")


def downscaling_enabled() -> bool:
    """r2 sub-flag, DEFAULT OFF, only read inside an SWR epoch (so inert without BRAIN_SLEEP_REPLAY_CAPTURE):
    `BRAIN_SLEEP_DOWNSCALING` arms the per-night synaptic downscaling of the managed blocks' learned increments."""
    return _truthy("BRAIN_SLEEP_DOWNSCALING")


def sleep_onset_h() -> float:
    """The engine's own sleep-depth criterion (SLEEP_IDLE_SEC, 5 min of idle), in hours. Imported, not a new constant."""
    from webapp.continuous_engine import SLEEP_IDLE_SEC
    return float(SLEEP_IDLE_SEC) / 3600.0


def swr_da(sum_r: float) -> float:
    """DA co-released by one SWR bout: proportional to the reactivated tag mass (Clopath 2008), from tonic up to the D1
    pool's own full-activation anchor."""
    return float(_DA_TONIC + (DA_SWR_FULL - _DA_TONIC) * min(1.0, max(0.0, float(sum_r))))


def reactivation_strength(comp, block_idx: int) -> Optional[float]:
    """Drive block `block_idx`'s trigger on the composer's substrate and read it back through the store's own cleanup
    (`_block_role_scores`). Returns the smallest decisiveness margin over the fact roles present, in [0, 1], or None when
    the composer has no such read (not a block-structured one-brain store)."""
    fn = getattr(comp, "_block_role_scores", None)
    if fn is None:
        return None
    out = fn(int(block_idx))
    ms = []
    for role in FACT_ROLES:
        if role not in out:
            continue
        rec = out.get(role)
        m = rec[2] if rec else None
        ms.append(0.0 if m is None else float(m))
    if not ms:
        return 0.0
    return float(min(1.0, max(0.0, min(ms))))


class _NullCtx:
    def __init__(self, *a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class SleepReplayCapture:
    """One chat session's sleep episodes. Owned by `ChatTagCapture` (built lazily, flag ON only)."""

    def __init__(self, seed: int, d1, reactivate_fn: Optional[Callable] = None, rng_ctx=None):
        self.seed = int(seed)
        self.d1 = d1                                   # object with .read(da) -> (a, rate), the v3 D1 reader
        self.reactivate_fn = reactivate_fn or reactivation_strength
        self.rng_ctx = rng_ctx or _NullCtx             # (seed, k) -> context manager (da_tag_capture_chat._private_rng)
        self.episode_key = None                        # the observed-turn count that started the current episode
        self.nights_done = 0                           # SWR epochs run in the current idle stretch (one per night)
        self.epochs: List[dict] = []

    @property
    def episode_done(self) -> bool:
        return self.nights_done > 0

    def catch_up(self, ledger, comp, t_last_turn: float, episode_key, t_now: float) -> int:
        """Run every SWR epoch of the current idle stretch that is due by `t_now` (event-driven): night k starts at
        t_last_turn + sleep onset + k * NIGHT_PERIOD_H. Returns #epochs run. (A one-night protocol -- recall within
        24 h of the last turn -- runs exactly one, as before r2.)"""
        if episode_key != self.episode_key:
            self.episode_key = episode_key
            self.nights_done = 0
        onset = sleep_onset_h()
        ran = 0
        while True:
            t_s = float(t_last_turn) + onset + self.nights_done * NIGHT_PERIOD_H
            if t_s > t_now:
                break
            self._epoch(ledger, comp, max(t_s, ledger.t))
            self.nights_done += 1
            ran += 1
        return ran

    def _epoch(self, ledger, comp, t_s: float) -> None:
        # the store as it stands at t_s (decay / capture integrated up to the moment the SWR bout starts)
        ledger.sync_from_store(comp, t_s)
        ledger.on_store(comp, t_s)
        ledger.advance(comp, t_s)
        e_idx = len(self.epochs)
        pre_z = [float(np.mean(b["z"] > 0.5)) for b in ledger.blocks]
        # (2) SWR reactivation of every managed block, read back by the store's own cleanup
        R = []
        for i in range(len(ledger.blocks)):
            with self.rng_ctx(self.seed, _K_REACT + e_idx * 1000 + i):
                r = self.reactivate_fn(comp, ledger.block_offset + i)
            R.append(None if r is None else float(r))
        coupling = 0.0 if replay_capture_lesioned() else 1.0
        R_eff = [coupling * (0.0 if r is None else r) for r in R]
        # (3) re-tag: the replay tag, never below what is left of an earlier replay tag
        for blk, r in zip(ledger.blocks, R_eff):
            h_new = r * np.abs(blk["inc"]).astype(np.float64)
            if blk.get("h_rep") is not None:
                h_new = np.maximum(h_new, blk["h_rep"] * math.exp(-(t_s - blk["t_rep"]) / TAU_TAG_H))
            blk["h_rep"] = h_new
            blk["t_rep"] = float(t_s)
        # (4) SWR-coupled DA onto the spiking D1 pool -> the shared PRP pool (through both existing lesion edges)
        sum_r = float(sum(R_eff))
        da = swr_da(sum_r)
        d_seen = prp_da(da)
        cap_coupling = 0.0 if capture_lesioned() else 1.0
        a_log = []
        for j in range(N_SWR_SUBREADS):
            t0 = t_s + j * SWR_SUBREAD_H
            with self.rng_ctx(self.seed, _K_SWR_D1 + e_idx * 1000 + j):
                a, _rate = self.d1.read(d_seen)
            a_eff = float(a) * cap_coupling
            ledger.drive.append((float(t0), float(t0) + SWR_SUBREAD_H, a_eff))
            a_log.append(a_eff)
        # (6, r2, sub-flag) SLEEP DOWNSCALING: the night's slow-wave activity depresses every managed block's learned
        # increment by SHY_DELTA, except in proportion to how strongly that block's own reactivation drove the read-out
        # (Gonzalez-Rueda et al. 2018: inputs that contribute to postsynaptic spiking in Up states are protected).
        shy = None
        if downscaling_enabled():
            shy = []
            for blk, r in zip(ledger.blocks, R_eff):
                s_i = 1.0 - SHY_DELTA * (1.0 - min(1.0, max(0.0, r)))
                blk["inc"] = blk["inc"] * s_i
                shy.append(round(s_i, 9))
        self.epochs.append({"episode": self.episode_key, "t_h": float(t_s),
                            "R": [None if r is None else round(r, 9) for r in R],
                            "R_eff": [round(r, 9) for r in R_eff], "sum_R_eff": round(sum_r, 9),
                            "da_swr": round(da, 9), "da_seen_by_d1": round(float(d_seen), 9),
                            "a_eff_mean": round(float(np.mean(a_log)) if a_log else 0.0, 9),
                            "pre_frac_z_gt_half": [round(v, 9) for v in pre_z],
                            "replay_lesioned": bool(coupling == 0.0), "capture_lesioned": bool(cap_coupling == 0.0),
                            "no_reader": bool(any(r is None for r in R))})
        if shy is not None:
            self.epochs[-1]["shy_scale"] = shy

    def summary(self) -> dict:
        out = {"on": True, "lesioned": replay_capture_lesioned(), "n_epochs": len(self.epochs),
               "episode_done": self.episode_done, "epochs": list(self.epochs)}
        if downscaling_enabled():
            out["downscaling"] = True
        return out
