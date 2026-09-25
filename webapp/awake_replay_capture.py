"""AWAKE-REST REPLAY FOR THE DA TAG-AND-CAPTURE ROUTE: a fact told hours before sleep is kept capturable by quiet-rest
reactivation during the waking day (DEFAULT-OFF: `BRAIN_AWAKE_REPLAY_CAPTURE`). Branch research/awake-replay-capture.

WHY (the wall, measured on a real build, seed 42, sleep-replay-capture PREREGISTRATION Amendment 1). With
`BRAIN_DA_TAG_CAPTURE` + `BRAIN_SLEEP_REPLAY_CAPTURE` armed, a plainly told fact that is 4 h old at sleep onset is
lost: its early-phase trace has decayed by then (the sleep SWR's read-back was 0.0056 against 0.36-0.47 for a fresh
fact), so the night's replay neither re-tags it nor co-releases DA for it, while today's default (ledger off) recalls
it. Asked "what does the real hippocampus run alongside tagging-and-capture during WAKE that we replaced with a
constant?", the answer is AWAKE SHARP-WAVE-RIPPLE REPLAY during quiet rest: between turns the model's early-phase
trace only decayed (the constant was "nothing reactivates a stored fact while the brain is awake").
Biology binding: research/biology/awake-replay-tag-capture.md.
  * Kandel 6e ch.5 (Fig. 5-2 text): during "immobile or resting behavior" the hippocampus is dominated by sharp-wave
    ripples; they "are prominent during resting periods after recent learning" and replay recent trajectories;
    replay "is thought to represent a form of mental rehearsal".
  * Buzsaki 2006 (Rhythms of the Brain, p.344-349): sharp waves occur in the non-theta states -- "consummatory
    behaviors, such as eating, drinking, and grooming and immobility, non-REM sleep"; "the neuronal pathways used and
    modified in the waking brain can be repeatedly replayed"; the SWR's gain in population excitation "creates
    favorable conditions for synaptic plasticity"; and replay content follows the synaptic weights ("activity spreads
    along the path of the strongest synaptic weights").
  * Sadowski, Jones & Mellor 2016 (Cell Rep 14:1916, PMC4785795): place-cell firing patterns reactivated during SWRs
    in the first 5 min of post-run rest INDUCE LTP at CA3-CA1 synapses (test path x2.2-3.4), only with the
    SWR-associated dendritic depolarization; the induced change scales with the number of LTP-competent spike pairings
    (r = 0.89); quiescence "may enable the connectivity of specific spatial engrams to be enhanced prior to sleep".
  * Jadhav, Kemere, German & Frank 2012 (Science 336:1454, PMC4441285): interrupting AWAKE SWRs impairs learning --
    the lesion this module's `BRAIN_AWAKE_REPLAY_CAPTURE_LESION` mirrors.

WHAT HAPPENS (only with the flag ON, only on an idle tick while the brain is awake, only for ledger-managed blocks):
  1. QUIET WAKE. An awake bout can run on the continuous engine's idle tick (`tick_idle_sessions`, which only ticks a
     session idle >= IDLE_SEC) when the ledger's OWN sleep criterion says the brain is not yet asleep: world-now is
     earlier than (end of waking) + sleep onset, the end of waking being the later of the last observed turn and any
     environment awake mark (`da_tag_capture_chat.mark_awake`). At most one bout per AWAKE_BOUT_H of world time.
     A turn never runs a bout (a turn is not rest); a sleep-depth tick never does (the sleep route owns the night).
  2. REACTIVATION (BRAIN-BASED SELECTION). Exactly the sleep route's read: the ledger first integrates to world-now
     and rewrites the store, then EVERY managed block's trigger is driven once on the composer's own
     resonate-and-fire substrate and read back through the store's own cleanup (`OneBrainComposer._block_role_scores`
     via `sleep_replay_capture.reactivation_strength`). R_i = the smallest cleanup decisiveness margin over
     agent/action/patient. No host list, ranking or threshold picks the fact: every managed trigger is driven
     identically, and what the substrate returns sets how strongly each one is reactivated (a decayed trace reads
     near 0 and is barely touched; a fresh one reads ~0.3-0.47).
  3. REACTIVATION-INDUCED EARLY LTP + A FRESH TAG (Sadowski 2016; the Frey-Morris tag set by any LTP induction). The
     block's early-phase expression e rises by the reactivation's share of the remaining headroom,
     e <- e + R_i * (1 - e) (a saturating Hebbian induction graded by the reactivation, as the pairing count grades
     Sadowski's LTP), and from then decays with the SAME TAU_EARLY_H as a written trace. Its tag is re-set to the
     same level, h_rep = e_new * |inc| (v3's invariant: the write tag is |inc| x the write's E-LTP, both on 1.5 h), the
     larger of that and any live replay tag being kept, exactly as the sleep route combines tags. z (late phase) is
     NOT touched.
  4. NO PRP. An awake bout does not read the D1 pool and adds nothing to the PRP pool (see the next section). A
     re-tagged block becomes late-phase only if PRP arrives while its tag is live: from the brain's own waking DA (a
     later salient / novel turn -- behavioural tagging, already in the v3 ledger) or from the night's SWR-coupled DA
     (webapp/sleep_replay_capture.py), which now finds the fact still readable at sleep onset.

WHY AWAKE REPLAY SUPPLIES NO PRP (decided from the biology, default and only behaviour here). The PRP trigger in this
model is dopamine through D1/D5 (Kandel 6e ch.54; Moncada & Viola 2007, J Neurosci 27:7476: NOVEL, not familiar,
exploration supplies the PRPs, D1/D5-dependently). The one direct measurement of DA-neuron participation in awake
replay (Gomperts, Kloosterman & Wilson 2015, eLife 4:e05360, PMC4695386) found that the VTA cells coordinating with
quiet-wake SWR replay are REWARD-RESPONSIVE cells replaying rewarded (appetitive) experience. A plainly told, familiar
fact carries neither reward nor novelty, so its awake reactivation has no basis for a DA co-release. Giving awake
replay its own PRP would also make "rest" alone permanently store every ordinary fact during the day, with no role left
for DA -- the opposite of Bethus, Tse & Morris 2010 (D1/D5 blockade removes persistence). Declared and not modelled: a
salient/rewarded fact's awake replay may co-activate VTA (Gomperts); such facts are already captured by their waking DA.

THE AWAKE-EDGE LESION. `BRAIN_AWAKE_REPLAY_CAPTURE_LESION=1` severs the reactivation's effect: the substrate reads still
run (same compute, same substrate state, same private RNG streams) but R_eff = 0, so no block's early phase or tag is
touched -- the store is exactly what the flag-off path writes.

CONSTANTS (all a priori; none fitted to a gate seed):
  AWAKE_BOUT_H = the sleep route's SWR_BOUT_H = the v3 CAPTURE_PROTOCOL_MIN (5 min), reused, not new. It is also the
  window Sadowski et al. replayed (the first 5 min of post-run rest, 57 SWRs, which induced full-size LTP), so one bout
  stands for 5 min of quiet-rest SWRs. The induction law has no free constant (the read R_i is the step).

KNOWN PROPERTY, measured on the fake-substrate design sweep before any brain run
(research/runners/_awake_replay_capture_design.py): because R rises with the expressed trace, rest has a positive
feedback; with a bout every 5 min a trace already down to 13 % of its expression (3 h without rest, read ~0.04)
regrows to ~0.92 within one hour of rest. The real companions that would brake it -- competition among many recent
assemblies for the SWR's content (Buzsaki: the most strongly bound assemblies become the "burst initiators") and the
synapse-by-synapse reversal of decaying E-LTP (the ledger shrinks the whole pattern uniformly instead) -- are absent
with one fact in the store. Recorded, not hidden: the pre-registration amendment measures it on the brain as a
REPORTED late-rest arm. On the seed-42 brain smoke (Amendment 5) it did NOT regrow: the composer's read of the
3-h-old trace was 0.008, not the fake curve's 0.043, and an hour of rest held the trace at ~13 % without restoring it.

HOST SHORTCUTS (declared, brain-based-only burn-down):
  - the idle tick and the awake/asleep decision are host clock (the engine's IDLE_SEC tick, the ledger's
    sleep-onset rule, the environment's awake mark: the body's wake/sleep clock, the same class as IDLE_SEC /
    SLEEP_IDLE_SEC); one bout per AWAKE_BOUT_H stands for the SWRs of 5 min of quiet rest;
  - the per-block loop that drives every managed trigger is host iteration (it drives ALL of them identically; the
    selection is the substrate's response, not the loop);
  - R_i is the composer's decisiveness margin (peak - runner_up)/peak: host arithmetic on a substrate read (the same
    read the sleep route, the confidence gate and the metacog hedge use);
  - the induction e <- e + R_i (1 - e) and the tag re-set R-level x |inc| are host arithmetic on the ledger's
    bookkeeping (the stored increment), like the v3 write tag and the sleep route's replay tag;
  - the per-synapse ODEs are the v3 host-integrated ones.
This is reactivation-driven re-potentiation in the same store, not "consolidation" in the docs/TERMS.md sense (no
transfer, no source lesion); the capture itself stays the v3 / sleep-route mechanism.

PATTERN-COMPLETION SUB-FLAG (branch research/awake-replay-completion; default OFF: `BRAIN_AWAKE_REPLAY_COMPLETION`). The
arc family's NO-GO 5/6 (research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md) is a subcritical loop on
a low-margin block: the induction scales with the decode margin R. With the sub-flag armed each bout also runs
webapp/replay_completion.py (a spiking item competition + the substrate re-bind of the reinstated ensemble) and
induces with R_c, the reinstated ensemble's in-phase coherence with the block's increment, instead of R;
`BRAIN_REPLAY_COMPLETION_LESION=1` keeps every read but induces with R (this module's Amendment-4 path). The night's
epoch has its own flag (`BRAIN_SLEEP_REPLAY_COMPLETION`, webapp/sleep_replay_capture.py). Unset -> the branch below is
never entered and nothing is imported.

CONTRACT. DEFAULT-OFF. With `BRAIN_AWAKE_REPLAY_CAPTURE` unset, `ChatTagCapture.tick` never enters its awake branch,
no block ever carries "e_rep" (so `SynapticTagCaptureLedger.early_expression` returns the write's value bit for bit),
no substrate read is made and no key is added to the reply: byte-identical (tests/test_awake_replay_capture.py). Inert
without `BRAIN_DA_TAG_CAPTURE` (no ledger to act on). Independent of `BRAIN_SLEEP_REPLAY_CAPTURE` in code, but without
the sleep route nothing supplies the night's PRP, so the rescue this module exists for needs both. No `sim/` edit; no
edit to one_brain_composer.py.
"""
from __future__ import annotations

import math
import os
from typing import Callable, List, Optional

import numpy as np

from webapp.da_tag_capture import TAU_TAG_H
from webapp.sleep_replay_capture import SWR_BOUT_H, reactivation_strength, sleep_onset_h

AWAKE_BOUT_H = SWR_BOUT_H              # 5 min of quiet-rest SWRs per bout (== v3 CAPTURE_PROTOCOL_MIN; reused, not new)
_K_AWAKE = 400000                      # private-RNG stream offset (never collides with turns, sleep reads 2e5 / 3e5)
_EPS_H = 1e-9


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


def awake_replay_enabled() -> bool:
    """Master flag, DEFAULT OFF. `BRAIN_AWAKE_REPLAY_CAPTURE` in {1,true,on,yes} arms the awake-rest route."""
    return _truthy("BRAIN_AWAKE_REPLAY_CAPTURE")


def awake_replay_lesioned() -> bool:
    """`BRAIN_AWAKE_REPLAY_CAPTURE_LESION` severs the reactivation -> early-LTP / tag edge (the reads still run)."""
    return _truthy("BRAIN_AWAKE_REPLAY_CAPTURE_LESION")


def _completion_enabled() -> bool:
    """`BRAIN_AWAKE_REPLAY_COMPLETION` (default OFF; branch research/awake-replay-completion) -- read here so the
    flag-off bout imports nothing new (webapp/replay_completion.py)."""
    return _truthy("BRAIN_AWAKE_REPLAY_COMPLETION")


def _completion_lesioned() -> bool:
    return _truthy("BRAIN_REPLAY_COMPLETION_LESION")


class _NullCtx:
    def __init__(self, *a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class AwakeReplayCapture:
    """One chat session's quiet-rest reactivation bouts. Owned by `ChatTagCapture` (built lazily, flag ON only)."""

    def __init__(self, seed: int, reactivate_fn: Optional[Callable] = None, rng_ctx=None):
        self.seed = int(seed)
        self.reactivate_fn = reactivate_fn or reactivation_strength
        self.rng_ctx = rng_ctx or _NullCtx             # (seed, k) -> context manager (da_tag_capture_chat._private_rng)
        self.t_last: Optional[float] = None            # world time of the last bout (the AWAKE_BOUT_H rate limit)
        self.bouts: List[dict] = []
        self.n_ticks_asleep = 0                        # idle ticks refused because the ledger's clock says asleep
        self.n_ticks_rate_limited = 0                  # idle ticks refused by the one-bout-per-AWAKE_BOUT_H limit

    @staticmethod
    def awake_at(t_ref: float, t_now: float) -> bool:
        """The ledger's own sleep criterion, read the other way: awake iff world-now is before (end of waking) + sleep
        onset (the sleep route runs its epoch at exactly that time)."""
        return float(t_now) < float(t_ref) + sleep_onset_h() - _EPS_H

    def maybe_bout(self, ledger, comp, t_ref: float, t_now: float) -> bool:
        """Called on an idle tick. Runs one bout at t_now iff the brain is awake and no bout ran in the last
        AWAKE_BOUT_H of world time. Returns whether a bout ran."""
        if not self.awake_at(t_ref, t_now):
            self.n_ticks_asleep += 1
            return False
        if self.t_last is not None and float(t_now) - self.t_last < AWAKE_BOUT_H - _EPS_H:
            self.n_ticks_rate_limited += 1
            return False
        self._bout(ledger, comp, float(t_now))
        return True

    def _bout(self, ledger, comp, t: float) -> None:
        # the store as it stands at t (decay / capture integrated up to the start of the bout)
        ledger.sync_from_store(comp, t)
        ledger.on_store(comp, t)
        ledger.advance(comp, t)
        b_idx = len(self.bouts)
        pre_e = [float(ledger.early_expression(b)) for b in ledger.blocks]
        pre_z = [float(np.mean(b["z"] > 0.5)) for b in ledger.blocks]
        # (2) reactivation of every managed block, read back by the store's own cleanup (the sleep route's read)
        R = []
        comp_rec = None
        if _completion_enabled():
            # PATTERN COMPLETION (default-OFF `BRAIN_AWAKE_REPLAY_COMPLETION`, webapp/replay_completion.py): the same read
            # R, then the spiking item competition + the substrate re-bind of the reinstated ensemble, R_c.
            from webapp import replay_completion as _C
            R, R_read, comp_rec = _C.read_blocks(comp, ledger, self.rng_ctx, self.seed, _K_AWAKE + b_idx * 1000)
        else:
            for i in range(len(ledger.blocks)):
                with self.rng_ctx(self.seed, _K_AWAKE + b_idx * 1000 + i):
                    r = self.reactivate_fn(comp, ledger.block_offset + i)
                R.append(None if r is None else float(r))
            R_read = R
        coupling = 0.0 if awake_replay_lesioned() else 1.0
        R_eff = [coupling * (0.0 if r is None else min(1.0, max(0.0, r))) for r in R_read]
        # (3) reactivation-induced early LTP + a fresh tag at the same level; z untouched; nothing when R_eff == 0
        post_e = []
        for blk, r, e in zip(ledger.blocks, R_eff, pre_e):
            if r > 0.0:
                e_new = e + r * (1.0 - e)
                blk["e_rep"] = float(e_new)
                blk["t_erep"] = float(t)
                h_new = e_new * np.abs(blk["inc"]).astype(np.float64)
                if blk.get("h_rep") is not None:
                    h_new = np.maximum(h_new, blk["h_rep"] * math.exp(-(t - blk["t_rep"]) / TAU_TAG_H))
                blk["h_rep"] = h_new
                blk["t_rep"] = float(t)
            post_e.append(float(ledger.early_expression(blk)))
        ledger._write(comp)                            # express the re-induced early phase in the store synapses
        self.t_last = float(t)
        self.bouts.append({"t_h": float(t), "R": [None if r is None else round(r, 9) for r in R],
                           "R_eff": [round(r, 9) for r in R_eff],
                           "early_before": [round(v, 9) for v in pre_e], "early_after": [round(v, 9) for v in post_e],
                           "pre_frac_z_gt_half": [round(v, 9) for v in pre_z],
                           "p_at_bout": round(float(ledger.p), 12), "n_drive_entries": len(ledger.drive),
                           "lesioned": bool(coupling == 0.0), "no_reader": bool(any(r is None for r in R))})
        if comp_rec is not None:                       # completion record: only ever present with the flag ON
            from webapp import replay_completion as _C
            self.bouts[-1]["completion"] = _C.record(comp_rec)
            self.bouts[-1]["completion_lesioned"] = bool(_completion_lesioned())

    def summary(self) -> dict:
        out = {"on": True, "lesioned": awake_replay_lesioned(), "n_bouts": len(self.bouts),
               "n_ticks_asleep": self.n_ticks_asleep, "n_ticks_rate_limited": self.n_ticks_rate_limited,
               "bout_h": AWAKE_BOUT_H, "bouts": list(self.bouts)}
        if _completion_enabled():                      # only with BRAIN_AWAKE_REPLAY_COMPLETION armed
            out["completion"] = True
            out["completion_lesioned"] = bool(_completion_lesioned())
        return out
