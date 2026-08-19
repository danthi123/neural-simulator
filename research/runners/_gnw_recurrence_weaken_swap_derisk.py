"""GNW RECURRENCE-WEAKEN SWAP — swap one held workspace coalition DIRECTLY for another in a single move, by
composing the two proven halves of the thought-swap:

  (IN-gate, that half works) `build_swap_bridge`'s distributed divisively-normalized workspace already ADMITS a NEW
      coalition: driving the challenger pattern with the ignition current ignites it (the quench-evict NO-GO confirmed
      "B reaches 0.333 whenever driven"). The unsolved half was EVICTING the OLD held coalition.
  (EVICTION, the named fix) Rung-2d (`_gnw_rung2d_weakenable_recurrence_derisk.py`, GO 6/6) proved that short-term
      synaptic DEPRESSION on a coalition's OWN recurrent excitatory (E->E) loop makes the ignited attractor
      dynamically weakenable: sustained self-use depletes the loop's resources x below the sustain knee and the
      attractor SELF-EVICTS to rest — no external brake. The quench-evict / STN-veto / BG-gate NO-GOs all failed
      because a self-sufficient recurrent attractor is INHIBITION-RESISTANT: soma-level feedback inhibition (standing
      or transient, closed- or open-loop) cannot evict a self-sustaining loop; the wall is the RECURRENCE
      self-sufficiency, not the brake shape (2026-08-19-gnw-quench-evict-overwrite-NOGO). The fix is to attack the
      recurrence itself, exactly as Rung-2d named.

THE SINGLE-MOVE SWAP (composition): the incumbent A holds (supra-critical loop, resources x=1). A swap is TRIGGERED
by engaging short-term depression on A's OWN recurrent loop (a transient boost to the per-spike utilization U of A's
E->E synapses) while A is firing: A's own use depletes A's loop below critical, A COLLAPSES to rest. The collapse
(A's private core below rate `VACANCY_THRESH` for `VACANCY_CONFIRM` consecutive steps) OPENS a vacancy-gated IN-gate:
after a short settle the challenger B is admitted ONLY into the freed slot and ignites. Because a real IN-gate does
not fire content into an occupied workspace, the recurrence-weakening is LOAD-BEARING by construction — no depression
-> no collapse -> the gate never opens -> A holds (the inhibition-resistant NO-GO). The ungated `forced` control
(drive B onto the held A regardless) confirms B cannot displace a supra-critical incumbent (0/6), rebutting any
gate-circularity worry. The OLD coalition drops to baseline (beating the 0.333 period-3 residual the quench left).
Because short-term depression is TRANSIENT (x recovers with tau_D once the coalition stops firing), A's recurrence
RECOVERS after the swap, so a THIRD swap works: a two-swap sequence A->B->A brings A back (proof the eviction is not a
permanent lesion). The eviction EFFECTOR is `RecurrenceDepression` (the same Mongillo-Barak-Tsodyks 2008 STD as
Rung-2d's `RecurrentSTD`, already built for THIS substrate in `_gnw_active_overwrite_derisk.py`), here TARGETED to
the incumbent's own loop (`target_units = the incumbent pattern`) so it depletes the coalition being swapped OUT
without poisoning the fresh challenger — the composition the prior overwrite/quench NO-GOs never tried (they used
soma inhibition, or a SHARED-only / GLOBAL depression that has no target on a disjoint incumbent or drains the
challenger too).

Biology: Mongillo, Barak & Tsodyks 2008, Science 319:1543 (synaptic theory of WM; recurrent resources x deplete by
u*x per spike, recover with tau_D). Dehaene & Changeux 2011, Neuron 70:200 (an ignited workspace state must be
destabilizable and spontaneously replaced by another). Compte, Brunel, Goldman-Rakic & Wang 2000, Cereb.Cortex
10:910 (persistent-activity termination).

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed) — per seed, ALL of:
  SWAP        — win_pre=A & n_pre=1, then A DROPS TO BASELINE (old_residual_post NOT ignited, < the 0.333 residual) &
                B ignites & win_post=B & n_post=1 (a genuine single-move content swap, not co-ignition n=2 or a STOP).
  LOAD-BEARING (recurrence-weakening causal) — LESION the STD (boost=0, no depletion) -> A never collapses -> the
                vacancy gate never opens -> A stays ignited on the plateau (old_residual ~0.333), reproducing the
                inhibition-resistant NO-GO. PLUS an UNGATED/forced control (drive B onto the held A regardless) that
                also fails to swap (0/6) -> the incumbent is genuinely un-displaceable without the recurrence route.
  REIGNITE    — the NEW coalition ignites AND HOLDS through an extended free tail (the eviction did not poison it).
  REVERSIBLE  — a two-swap sequence A->B->A: after swapping A out then swapping B out and driving A back, A
                RE-IGNITES (win=A, n=1), proving A's recurrence recovered (STD transient, not a permanent lesion).
  NO-HOST-RESET — the swap headline is a CONTINUOUS run: host_workspace_reset_calls==0 (the only host writes are
                external stimulus drive + the swap-trigger boost = the swap command; NO "clear the workspace" call).
  DETERMINISM — build twice at one seed -> identical substrate hash (heterogeneity from cfg.seed). This is the
                substrate-integrity anti-cheat: the additive_substrate hash is N/A here (NO pool is appended to
                build_swap_bridge; nothing perturbs the base Izhikevich-param RNG draw).

NOT-A-WALL: if the swap is not clean, the residual is QUANTIFIED (old-state rate, whether B ignites, empty window) so
the next mechanism is mapped. NO `sim/` edit; explicit dense frozen wiring; native STP/homeostasis OFF; the STD is
host-computed on the substrate's OWN recurrent synapses (the Rung-2d / active-overwrite pattern).

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_recurrence_weaken_swap_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_recweaken_swap_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_recurrence_weaken_swap_derisk --six-seed \
      --json research/findings/raw/_gnw_recweaken_swap_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim.backend import to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

# reuse-by-import: the IN-gate substrate (build_swap_bridge) + the recurrence-weakening STD effector
# (RecurrenceDepression == the same Mongillo/Tsodyks STD as Rung-2d's RecurrentSTD, built for THIS substrate) +
# the stepping / spiking-read / margin instruments + validated ignition constants.
from research.runners._gnw_active_overwrite_derisk import (
    build_swap_bridge, RecurrenceDepression,
    _ws_step, _drive, _read_private_rates, _instant_private_rate, _margin, _verdict_label,
    _full_restore, _full_snapshot,
    N_PATTERNS, W_SHARED, THAL_TONIC_PA, STRONG_PA, STD_TAU_D, OU_NOISE_PA,
)
from research.runners._gnw_rung1_ignition_curve_derisk import DRIVE_STEPS, FREE_STEPS
from research.runners._gnw_rung2_competitive_access_derisk import _ignited, IGNITE_FRAC, SOLO_PLATEAU

# the incumbent "dropped to baseline" bar: a swapped-OUT coalition reads BELOW the ignition threshold (0.1667). The
# quench NO-GO left old_residual_post pinned at 0.333 (full period-3 plateau); a genuine swap must beat that.
IGNITE_THRESH = IGNITE_FRAC * SOLO_PLATEAU

# swap operating point (swept in --smoke). baseline U=0 => a held coalition NEVER self-depletes (holds indefinitely =
# the inhibition-resistant incumbent, unlike Rung-2d's spontaneous metastability); the swap TRIGGER is a transient
# boost to the incumbent loop's per-spike depletion U (the "swap now" command). w_rec uniform 34 == the quench/active
# supra-critical disjoint headline (the exact inhibition-resistant substrate).
SWAP_BOOST = 0.12            # per-spike utilization boost on the incumbent loop during the evict window (swept: >=0.10
                            # collapses A's loop below the sustain knee; <0.06 leaves A on the plateau)
EVICT_STEPS = 260            # MAX steps the incumbent-loop depression runs (its own firing depletes x below the knee);
                            # the loop exits EARLY the moment the incumbent's collapse (vacancy) is confirmed
REIGNITE_HOLD = 150          # extended free-run after the identity read: the NEW coalition must still be ignited
W_REC = W_SHARED             # 34.0 uniform recurrence (supra-critical disjoint = the inhibition-resistant incumbent)
CHAL_PA = STRONG_PA          # challenger IN-gate drive (pA) = the incumbent's own ignition drive; strong latch into
                            # the VACATED workspace (the vacancy GATE, not the drive strength, enforces the dissociation)
# the IN-gate: the challenger is admitted ONLY into a VACATED workspace. The incumbent's collapse (its private core
# staying below VACANCY_THRESH for VACANCY_CONFIRM consecutive steps) opens the gate; a SETTLE_GAP lets the freed
# workspace quiesce (norm-pool inhibition decays) before the B volley, so a strong short drive latches cleanly.
VACANCY_THRESH = 0.05        # incumbent private-core instant rate below this == "the workspace slot is free"
VACANCY_CONFIRM = 12         # consecutive sub-threshold steps that CONFIRM the incumbent has collapsed (opens the gate)
SETTLE_GAP = 25              # free-run steps after the gate opens, before the challenger volley (norm-pool settles)
B_DRIVE = DRIVE_STEPS        # challenger IN-gate volley length (35 steps @ CHAL_PA) into the vacated workspace

_RESTORE_CALLS = {"n": 0}


def _counted_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


# ── per-pattern recurrence-weakening STD (one RecurrenceDepression per coalition loop; boost only the one swapped out)
class MultiLoopSTD:
    """A set of `RecurrenceDepression` instances, ONE per workspace coalition, each TARGETED to its own pattern's
    recurrent E->E loop (target_units = that pattern). Depression on loop k is driven by coalition k's OWN firing
    (Rung-2d self-eviction); its per-spike utilization is U_baseline + boost[k]. A swap boosts ONLY the incumbent's
    loop, so the coalition being swapped OUT self-evicts while the fresh challenger's loop (boost 0, not yet firing)
    stays at full strength. The loops are disjoint (disjoint patterns), so the instances write disjoint synapse
    slices; applying/updating all each step is exact. Exposes the `std.apply()` / `std.update(fired)` / `std.reset()`
    interface `_ws_step` expects."""

    def __init__(self, bridge, xp, ws_used, patterns_host, U_baseline=0.0, tau_D=STD_TAU_D, dt=1.0):
        self.deps = [RecurrenceDepression(bridge, xp, ws_used, target_units=patterns_host[k],
                                          U=U_baseline, tau_D=tau_D, dt=dt) for k in range(len(patterns_host))]
        self.n_rec_syn = [int(d.n_rec_syn) for d in self.deps]

    def apply(self):
        for d in self.deps:
            d.apply()

    def update(self, fired_host):
        for d in self.deps:
            d.update(fired_host)

    def reset(self):
        for d in self.deps:
            d.reset()

    def set_boost(self, k, val):
        self.deps[k].boost = float(val)

    def clear_boost(self):
        for d in self.deps:
            d.boost = 0.0

    def x_mean(self, k):
        d = self.deps[k]
        return float(d.x[d.target].mean())

    @property
    def n_writes(self):
        return int(sum(d.n_writes for d in self.deps))


# ── build: the IN-gate substrate in its supra-critical disjoint config (the inhibition-resistant incumbent) ─────────
def build(seed=42, w_rec=W_REC, heterogeneity=True, ou_noise_pA=OU_NOISE_PA):
    """build_swap_bridge in its DISJOINT (overlap=0) supra-critical uniform-recurrence config with WTA OFF and
    divisive normalization + tonic thal ON — the exact substrate on which every prior eviction lever hit the
    inhibition-resistance wall. Returns (bridge, xp, patterns_dev, privates_dev, thal_dev, ws_used, snap, handles)."""
    (bridge, xp, patterns_dev, privates_dev, thal_dev, ws_used, _shared_ab, snap, handles) = build_swap_bridge(
        seed=seed, overlap=0, w_shared=float(w_rec), w_priv=float(w_rec), wta_w=0.0,
        heterogeneity=heterogeneity, ou_noise_pA=ou_noise_pA)
    return bridge, xp, patterns_dev, privates_dev, thal_dev, ws_used, snap, handles


# ── determinism hash (substrate-integrity anti-cheat) ──────────────────────────────────────────────────────────────
def _izh_hash(bridge):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest() if parts else ""


# ── one single-move swap (ignite incumbent -> trigger recurrence-weakening + IN-gate -> read -> hold) ──────────────
def run_swap(bridge, xp, patterns, privates, thal, std, snap, *, incumbent=0, challenger=1, boost=SWAP_BOOST,
             evict_steps=EVICT_STEPS, reignite_hold=REIGNITE_HOLD, chal_pa=None, vacancy_confirm=VACANCY_CONFIRM,
             vacancy_thresh=VACANCY_THRESH, settle_gap=SETTLE_GAP, b_drive=B_DRIVE,
             lesion=False, forced=False, isolate=True):
    """Ignite incumbent A (hold); TRIGGER the swap; read identity; hold to check the challenger persists.
    THE SINGLE MOVE = {engage recurrence-weakening on A's OWN loop} + {the vacancy-gated IN-gate}:
      (2) boost short-term depression on A's recurrent loop -> A's own firing depletes x below the sustain knee ->
          A COLLAPSES. The incumbent's private core staying below `vacancy_thresh` for `vacancy_confirm` consecutive
          steps CONFIRMS the collapse and OPENS the IN-gate (vacancy detected); the loop exits early.
      (3) IN-gate: the challenger B is admitted ONLY into a VACATED workspace — after a `settle_gap` (norm-pool
          inhibition decays) a strong B volley (`b_drive` steps @ chal_pa) latches B into the freed slot.
    This makes the recurrence-weakening LOAD-BEARING by construction of the biology (a real IN-gate does not fire
    content into an occupied workspace): lesion=True -> boost 0 -> A never collapses -> the gate never opens -> B is
    never admitted -> A holds on the plateau (the inhibition-resistant NO-GO reproduced).
      forced=True -> the UNGATED control: drive B regardless of vacancy (force the volley onto a possibly-held A) ->
        maps the divisive-norm competition outcome when eviction did NOT clear the slot (the prior overwrite NO-GO
        class). Used to characterise the residual honestly, NOT the headline.
      isolate=False -> a CONTINUOUS run (0 restore calls) = the swap HEADLINE.
    Returns per-phase winners/rates, the old-state residual, the vacancy flag, and A's loop x_min."""
    if chal_pa is None:
        chal_pa = CHAL_PA
    if isolate:
        _counted_restore(bridge, snap)
        std.reset()

    # (1) ignite A alone -> it holds on its supra-critical recurrent loop.
    _drive(bridge, xp, thal, THAL_TONIC_PA, std, [(patterns[incumbent], STRONG_PA)], n=DRIVE_STEPS)
    pre = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_pre, _m_pre, n_pre = _margin(pre)

    # (2) SWAP TRIGGER: boost depression on A's OWN loop; run until the incumbent's collapse is CONFIRMED (vacancy)
    #     or evict_steps is exhausted. The loop drive is ZERO (no re-drive of A) — A depletes through its OWN use.
    eff_boost = 0.0 if lesion else float(boost)
    std.set_boost(incumbent, eff_boost)
    xA_min = 1.0
    a_mid_spikes, a_mid_steps = 0, 0
    inc_p = privates[incumbent]
    inc_pn = int(inc_p.shape[0])
    mid_lo = max(0, evict_steps // 4)
    vacancy = False
    vacancy_step = -1
    low_run = 0
    for t in range(int(evict_steps)):
        _ws_step(bridge, xp, thal, THAL_TONIC_PA, std)
        ra = _instant_private_rate(bridge, xp, privates, incumbent)
        xA_min = min(xA_min, std.x_mean(incumbent))
        if mid_lo <= t < mid_lo + max(1, evict_steps // 4):
            a_mid_spikes += int(to_host(bridge.cp_firing_states[inc_p].astype(xp.float64).sum()))
            a_mid_steps += 1
        low_run = low_run + 1 if ra < vacancy_thresh else 0
        if low_run >= int(vacancy_confirm):
            vacancy = True
            vacancy_step = t
            break
    std.clear_boost()
    old_rate_midevict = (a_mid_spikes / float(a_mid_steps * inc_pn)) if a_mid_steps else -1.0

    # (2b) settle the freed workspace (norm-pool inhibition decays) before the IN-gate volley.
    if vacancy:
        for _ in range(int(settle_gap)):
            _ws_step(bridge, xp, thal, THAL_TONIC_PA, std)

    # (3) IN-gate: admit the challenger ONLY into a vacated workspace (forced=True bypasses the gate = the control).
    b_driven = bool(vacancy or forced)
    if b_driven:
        for _ in range(int(b_drive)):
            _ws_step(bridge, xp, thal, THAL_TONIC_PA, std, drive_map=[(patterns[challenger], chal_pa)])

    # (4) identity read (free-run, no drive, boost off).
    post = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_post, _m_post, n_post = _margin(post)

    # (5) reignition-survival hold: extended free-run -> the challenger must still be ignited (A gone).
    hold = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std, n_free=int(reignite_hold))
    win_hold, _m_hold, n_hold = _margin(hold)

    old_res = float(post[incumbent])
    new_rate = float(post[challenger])
    old_res_hold = float(hold[incumbent])
    new_hold = float(hold[challenger])
    v_pre, _ = _verdict_label(pre)
    v_post, _ = _verdict_label(post)

    swapped = bool(win_pre == incumbent and n_pre == 1 and (not _ignited(old_res))
                   and _ignited(new_rate) and win_post == challenger and n_post == 1)
    reignite_ok = bool(_ignited(new_hold) and win_hold == challenger and n_hold == 1 and (not _ignited(old_res_hold)))
    return {
        "pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
        "hold_rates": [float(r) for r in hold],
        "winner_pre": int(win_pre), "winner_post": int(win_post), "winner_hold": int(win_hold),
        "n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post), "n_ignited_hold": int(n_hold),
        "delivered_pre": v_pre, "delivered_post": v_post,
        "old_residual_post": old_res, "old_residual_hold": old_res_hold,
        "new_rate_post": new_rate, "new_rate_hold": new_hold,
        "old_rate_midevict": float(old_rate_midevict), "xA_min": float(xA_min),
        "vacancy": bool(vacancy), "vacancy_step": int(vacancy_step), "b_driven": b_driven,
        "swapped": swapped, "reignite_ok": reignite_ok,
        "co_ignition": bool(n_pre == 1 and n_post >= 2), "went_empty": bool(n_pre >= 1 and n_post == 0),
        "incumbent_held": bool(win_post == incumbent and n_post >= 1 and not _ignited(new_rate)),
    }


# ── two-swap reversibility: A -> B -> A on ONE continuous substrate (proves A's recurrence RECOVERS) ───────────────
def run_two_swap(bridge, xp, patterns, privates, thal, std, snap, *, a=0, b=1, boost=SWAP_BOOST,
                 evict_steps=EVICT_STEPS, chal_pa=None, recover_steps=None, reignite_hold=REIGNITE_HOLD):
    """A CONTINUOUS run (one restore at the very start, then NO reset): swap A->B, let the substrate free-run so A's
    depleted loop RECOVERS (x -> 1), then swap B->A. If A RE-IGNITES on the second swap, its recurrence recovered
    (short-term depression is transient, not a permanent lesion). recover_steps defaults to ~3*tau_D."""
    _counted_restore(bridge, snap)
    std.reset()
    if recover_steps is None:
        recover_steps = int(3 * STD_TAU_D)

    # swap 1: A -> B (continuous; the incumbent A self-evicts, B ignites).
    s1 = run_swap(bridge, xp, patterns, privates, thal, std, snap, incumbent=a, challenger=b, boost=boost,
                  evict_steps=evict_steps, chal_pa=chal_pa, reignite_hold=reignite_hold, isolate=False)
    xA_after_s1 = std.x_mean(a)

    # recovery window: B holds; A is at rest, so A's loop resources x recover toward 1 (tau_D).
    for _ in range(int(recover_steps)):
        _ws_step(bridge, xp, thal, THAL_TONIC_PA, std)
    xA_recovered = std.x_mean(a)

    # swap 2: B -> A (deplete B's loop; the gate opens on B's collapse; drive A back). A re-igniting == recovered.
    s2 = run_swap(bridge, xp, patterns, privates, thal, std, snap, incumbent=b, challenger=a, boost=boost,
                  evict_steps=evict_steps, chal_pa=chal_pa, reignite_hold=reignite_hold, isolate=False)

    # Reversibility is the SETTLED claim (robust to per-step OU noise in a continuous run): swap1 evicted A; A's
    # depleted loop RECOVERED during the free-run (x -> ~1, so it is a transient depression, not a permanent lesion);
    # and swap2 brought A BACK as the final settled winner (A re-ignites & holds, B evicted). This proves a coalition
    # evicted by recurrence weakening can be re-instated later — the third swap works.
    s1_evicted_A = bool(s1["swapped"])
    recovered = bool(xA_recovered > 0.85)
    s2_brought_A_back = bool(s2["winner_hold"] == a and s2["n_ignited_hold"] == 1
                            and _ignited(s2["new_rate_hold"]) and not _ignited(s2["old_residual_hold"]))
    reversible = bool(s1_evicted_A and recovered and s2_brought_A_back)
    return {"swap1": s1, "swap2": s2, "xA_after_swap1": float(xA_after_s1), "xA_recovered": float(xA_recovered),
            "recover_steps": int(recover_steps), "reversible": reversible,
            "s1_evicted_A": s1_evicted_A, "recovered": recovered, "s2_brought_A_back": s2_brought_A_back}


# ── one seed: headline swap + lesion dissociation + reversibility + determinism ────────────────────────────────────
def evaluate_seed(seed, *, boost=SWAP_BOOST, evict_steps=EVICT_STEPS, chal_pa=CHAL_PA, reignite_hold=REIGNITE_HOLD,
                  w_rec=W_REC, heterogeneity=True, verbose=True):
    b_, xp, pats, privs, thal, ws_used, snap, hh = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    pats_host = [to_host(p).astype(np.int64) for p in pats]

    # ⚠ construct ALL STD instances NOW, on the freshly-built substrate, so each captures the TRUE base recurrence
    # weights. `RecurrenceDepression` snapshots `base` from cp_connections.data at construction; a prior arm leaves
    # those weights depressed, so an STD built AFTER a run would capture wrong (too-low) base -> underpowered loop.
    std = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_les = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_forced = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_rev = MultiLoopSTD(b_, xp, ws_used, pats_host)

    # HEADLINE: single-move swap A->B, CONTINUOUS (0 restore calls in the headline itself).
    restore_before = _RESTORE_CALLS["n"]
    headline = run_swap(b_, xp, pats, privs, thal, std, snap, incumbent=0, challenger=1, boost=boost,
                        evict_steps=evict_steps, chal_pa=chal_pa, reignite_hold=reignite_hold, isolate=False)
    host_workspace_reset_calls = int(_RESTORE_CALLS["n"] - restore_before)

    # LESION (recurrence-weakening OFF: boost=0, no depletion) -> A never collapses -> the IN-gate never opens ->
    # B is never admitted -> A holds on the plateau (the inhibition-resistant NO-GO reproduced).
    lesion = run_swap(b_, xp, pats, privs, thal, std_les, snap, incumbent=0, challenger=1, boost=boost,
                      evict_steps=evict_steps, chal_pa=chal_pa, reignite_hold=reignite_hold, lesion=True, isolate=True)

    # UNGATED CONTROL (honest residual): FORCE the B volley onto the (un-depleted, held) A -> maps the divisive-norm
    # competition when eviction did NOT clear the slot (the prior overwrite NO-GO class: co-ignition / marginal
    # competitive displacement). NOT the headline; characterises what the IN-gate alone (no weakening) can do.
    lesion_forced = run_swap(b_, xp, pats, privs, thal, std_forced, snap, incumbent=0, challenger=1, boost=boost,
                             evict_steps=evict_steps, chal_pa=chal_pa, reignite_hold=reignite_hold,
                             lesion=True, forced=True, isolate=True)

    # REVERSIBILITY: two-swap A->B->A on one continuous substrate.
    two = run_two_swap(b_, xp, pats, privs, thal, std_rev, snap, a=0, b=1, boost=boost, evict_steps=evict_steps,
                       chal_pa=chal_pa, reignite_hold=reignite_hold)

    # ── anti-cheats ──
    swap_ok = bool(headline["swapped"])
    reignite_ok = bool(headline["reignite_ok"])
    # the UNGATED control is the NON-CIRCULAR load-bearing evidence: force the SAME B volley onto the held
    # (un-depleted) incumbent -> it still fails to swap (B locked out). Only A's state (held vs STD-evicted) differs.
    ungated_control_holds = bool(not lesion_forced["swapped"] and _ignited(lesion_forced["old_residual_post"]))
    # LOAD-BEARING: the swap needs the recurrence weakening. Required BOTH ways: (a) the gated lesion never opens the
    # gate (no collapse) AND (b) the UNGATED forced volley cannot displace the held incumbent -> the eviction, not
    # the drive, is what clears the slot (rebuts the gate-circularity worry).
    load_bearing = bool(swap_ok and not lesion["swapped"] and ungated_control_holds)
    # the lesion reproduces the NO-GO: A never vacates (gate stays closed) and holds on the plateau.
    lesion_holds = bool((not lesion["vacancy"]) and _ignited(lesion["old_residual_post"])
                        and not lesion["swapped"])
    reversible = bool(two["reversible"])
    # ATTRIBUTION: whose is the swap? treatment = headline swap; control = the recurrence-weakening lesion.
    swap_attr = attributable_to("single-move swap via incumbent recurrence weakening (headline vs STD-lesion)",
                                float(swap_ok), float(lesion["swapped"]), warn_below=0.0)

    # DETERMINISM (substrate-integrity anti-cheat; additive_substrate hash N/A: no appended pool).
    h1 = _izh_hash(b_)
    b2, xp2, *_2 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    seed_deterministic = bool(_izh_hash(b2) == h1 and h1 != "")

    seed_go = bool(swap_ok and reignite_ok and load_bearing and lesion_holds and reversible
                   and host_workspace_reset_calls == 0 and seed_deterministic)

    v = Verdict("GNW recurrence-weaken swap (seed %d)" % seed)
    v.require("incumbent ignites confidently (n_pre==1, winner A) [precondition]",
              bool(headline["n_ignited_pre"] == 1 and headline["winner_pre"] == 0), expect=True)
    v.require("single-move swap: old drops to baseline & new ignites (n_post==1, winner B)", swap_ok, expect=True)
    v.require("recurrence-weakening LOAD-BEARING (lesion -> no swap, incumbent holds)", load_bearing, expect=True)
    v.require("UNGATED control holds (forced B volley cannot displace the held incumbent, non-circular)",
              ungated_control_holds, expect=True)
    v.require("new coalition re-ignites and HOLDS", reignite_ok, expect=True)
    v.require("REVERSIBLE two-swap A->B->A (incumbent recurrence recovers)", reversible, expect=True)
    v.require("no host workspace reset in the swap headline (continuous run)",
              host_workspace_reset_calls == 0, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating (2026-08-01); STD here targets ONLY the incumbent's "
                   "E->E recurrence, in-runner, with Mongillo params")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"boost": float(boost), "evict_steps": int(evict_steps), "chal_pa": float(chal_pa),
                            "vacancy_confirm": int(VACANCY_CONFIRM), "vacancy_thresh": float(VACANCY_THRESH),
                            "settle_gap": int(SETTLE_GAP), "b_drive": int(B_DRIVE),
                            "reignite_hold": int(reignite_hold),
                            "w_rec": float(w_rec), "U_baseline": 0.0, "tau_D": float(STD_TAU_D),
                            "heterogeneity": bool(heterogeneity)},
        "go_gate": {"swap_ok": swap_ok, "reignite_ok": reignite_ok, "load_bearing": load_bearing,
                    "ungated_control_holds": ungated_control_holds, "lesion_holds": lesion_holds,
                    "reversible": reversible, "no_host_reset": bool(host_workspace_reset_calls == 0),
                    "seed_deterministic": seed_deterministic},
        "anti_cheats": {"recurrence_weakening_load_bearing": load_bearing, "lesion_reproduces_nogo": lesion_holds,
                        "reversible_two_swap": reversible, "no_host_workspace_reset": bool(host_workspace_reset_calls == 0),
                        "seed_deterministic": seed_deterministic,
                        "swap_attributable_fraction": swap_attr},
        "residual": {
            "headline": {"winner_pre": headline["winner_pre"], "winner_post": headline["winner_post"],
                         "n_pre": headline["n_ignited_pre"], "n_post": headline["n_ignited_post"],
                         "old_residual_post": headline["old_residual_post"], "new_ignited": bool(_ignited(headline["new_rate_post"])),
                         "new_rate_post": headline["new_rate_post"], "old_rate_midevict": headline["old_rate_midevict"],
                         "xA_min": headline["xA_min"], "vacancy": headline["vacancy"], "vacancy_step": headline["vacancy_step"],
                         "old_residual_hold": headline["old_residual_hold"], "new_rate_hold": headline["new_rate_hold"]},
            "lesion": {"winner_post": lesion["winner_post"], "n_post": lesion["n_ignited_post"],
                       "old_residual_post": lesion["old_residual_post"], "new_rate_post": lesion["new_rate_post"],
                       "incumbent_held": lesion["incumbent_held"], "vacancy": lesion["vacancy"],
                       "b_driven": lesion["b_driven"], "xA_min": lesion["xA_min"], "swapped": lesion["swapped"]},
            "lesion_forced": {"winner_post": lesion_forced["winner_post"], "n_post": lesion_forced["n_ignited_post"],
                              "old_residual_post": lesion_forced["old_residual_post"],
                              "new_rate_post": lesion_forced["new_rate_post"],
                              "co_ignition": lesion_forced["co_ignition"], "swapped": lesion_forced["swapped"],
                              "b_driven": lesion_forced["b_driven"]},
            "reversibility": {"reversible": two["reversible"], "s1_evicted_A": two["s1_evicted_A"],
                              "recovered": two["recovered"], "s2_brought_A_back": two["s2_brought_A_back"],
                              "xA_after_swap1": two["xA_after_swap1"], "xA_recovered": two["xA_recovered"],
                              "recover_steps": two["recover_steps"], "swap1_swapped": two["swap1"]["swapped"],
                              "swap2_winner_hold": two["swap2"]["winner_hold"],
                              "swap2_n_hold": two["swap2"]["n_ignited_hold"],
                              "swap2_new_rate_hold": two["swap2"]["new_rate_hold"],
                              "swap2_old_residual_hold": two["swap2"]["old_residual_hold"]},
        },
        "host_workspace_reset_calls": int(host_workspace_reset_calls),
        "std_weight_writes": int(std.n_writes), "n_rec_syn_per_loop": std.n_rec_syn,
        "substrate_hash": h1,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        hd = headline
        print(f"[recweaken-swap seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    HEADLINE: win {hd['winner_pre']}->{hd['winner_post']} n {hd['n_ignited_pre']}->{hd['n_ignited_post']}"
              f" | A_midevict={hd['old_rate_midevict']:.3f} xA_min={hd['xA_min']:.3f} old_res={hd['old_residual_post']:.3f}"
              f" new={hd['new_rate_post']:.3f} vac@{hd['vacancy_step']} swapped={hd['swapped']} reignite={hd['reignite_ok']}",
              flush=True)
        print(f"    LESION  : vacancy={lesion['vacancy']} b_driven={lesion['b_driven']} win->{lesion['winner_post']}"
              f" old_res={lesion['old_residual_post']:.3f} held={lesion['incumbent_held']} lesion_holds={lesion_holds}"
              f"  | FORCED: win->{lesion_forced['winner_post']} old_res={lesion_forced['old_residual_post']:.3f}"
              f" new={lesion_forced['new_rate_post']:.3f} co_ign={lesion_forced['co_ignition']} swapped={lesion_forced['swapped']}"
              f"  | load_bearing={load_bearing}", flush=True)
        print(f"    REVERSE : reversible={two['reversible']} xA(after s1)={two['xA_after_swap1']:.3f} "
              f"xA(recovered)={two['xA_recovered']:.3f} s1={two['swap1']['swapped']} s2={two['swap2']['swapped']} "
              f"s2_hold_win={two['swap2']['winner_hold']} | det={seed_deterministic} resets={host_workspace_reset_calls}",
              flush=True)
    return result


# ── smoke: operating-point grid on one seed (find a clean swap point) ──────────────────────────────────────────────
def run_smoke(seed, args):
    print(f"[recweaken-swap smoke] seed={seed} — operating-point grid (boost x evict_steps x chal_pa)", flush=True)
    boost_grid = args.boost_grid if args.boost_grid else [args.boost]
    evict_grid = args.evict_grid if args.evict_grid else [args.evict_steps]
    chal_grid = args.chal_grid if args.chal_grid else [args.chal_pa]
    grid = []
    for bo in boost_grid:
        for ev in evict_grid:
            for ch in chal_grid:
                r = evaluate_seed(seed, boost=float(bo), evict_steps=int(ev), chal_pa=float(ch),
                                  reignite_hold=args.reignite_hold,
                                  w_rec=args.w_rec, heterogeneity=not args.no_heterogeneity, verbose=True)
                grid.append({"boost": float(bo), "evict_steps": int(ev), "chal_pa": float(ch),
                             "seed_go": r["seed_go"], "go_gate": r["go_gate"], "residual": r["residual"]})
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_recurrence_weaken_swap_derisk", "mode": "smoke", "seed": seed, "grid": grid},
                  f, indent=2, default=str)
    print(f"\n[recweaken-swap smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[recweaken-swap six-seed] seeds={seeds} @ boost={args.boost} evict={args.evict_steps} "
          f"chal_pa={args.chal_pa} w_rec={args.w_rec}", flush=True)
    per_seed = []
    for s in seeds:
        r = evaluate_seed(s, boost=args.boost, evict_steps=args.evict_steps, chal_pa=args.chal_pa,
                          reignite_hold=args.reignite_hold,
                          w_rec=args.w_rec, heterogeneity=not args.no_heterogeneity, verbose=True)
        per_seed.append(r)
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_swap = sum(1 for r in per_seed if r["go_gate"]["swap_ok"])
    n_reig = sum(1 for r in per_seed if r["go_gate"]["reignite_ok"])
    n_lb = sum(1 for r in per_seed if r["go_gate"]["load_bearing"])
    n_ungated = sum(1 for r in per_seed if r["go_gate"]["ungated_control_holds"])
    n_les = sum(1 for r in per_seed if r["go_gate"]["lesion_holds"])
    n_rev = sum(1 for r in per_seed if r["go_gate"]["reversible"])
    n_nores = sum(1 for r in per_seed if r["go_gate"]["no_host_reset"])
    n_det = sum(1 for r in per_seed if r["go_gate"]["seed_deterministic"])
    n_forced_swap = sum(1 for r in per_seed if r["residual"]["lesion_forced"]["swapped"])
    pooled_go = bool(n_go >= 5 and n_swap >= 5 and n_reig >= 5 and n_lb >= 5 and n_les == 6 and n_rev >= 5
                     and n_nores == 6 and n_det == 6 and n_ungated == 6 and n_forced_swap == 0)
    verdict = "GO" if pooled_go else ("PARTIAL" if n_swap >= 1 else "NO-GO")

    v = Verdict("GNW recurrence-weaken swap: 6-seed aggregate")
    v.require("single-move swap on >=5/6 (old drops to baseline & new ignites, n=1)", bool(n_swap >= 5), expect=True)
    v.require("recurrence-weakening load-bearing on >=5/6 (lesion -> no swap)", bool(n_lb >= 5), expect=True)
    v.require("UNGATED control holds on 6/6 (forced B cannot displace the held incumbent; non-circular)",
              bool(n_ungated == 6 and n_forced_swap == 0), expect=True)
    v.require("lesion reproduces the inhibition-resistant NO-GO on 6/6 (incumbent holds)", bool(n_les == 6), expect=True)
    v.require("new coalition re-ignites and holds on >=5/6", bool(n_reig >= 5), expect=True)
    v.require("reversible two-swap A->B->A on >=5/6 (recurrence recovers)", bool(n_rev >= 5), expect=True)
    v.require("no host workspace reset on 6/6 (continuous swap)", bool(n_nores == 6), expect=True)
    v.require("determinism: cfg.seed seeds the substrate on 6/6", bool(n_det == 6), expect=True)
    v.disabled("homeostasis", why="frozen base weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating (2026-08-01); STD targets ONLY the incumbent E->E "
                   "recurrence, in-runner, with Mongillo params")
    v.disabled("additive_substrate_hash",
               why="N/A: no pool is appended to build_swap_bridge; determinism (build-twice hash) is the "
                   "substrate-integrity anti-cheat instead (the RNG-prefix property does not hold on this engine)")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_recurrence_weaken_swap_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "swap_ok": n_swap, "reignite_ok": n_reig, "load_bearing": n_lb,
                          "ungated_control_holds": n_ungated, "lesion_holds": n_les, "reversible": n_rev,
                          "no_host_reset": n_nores, "seed_deterministic": n_det,
                          "ungated_forced_swap": n_forced_swap, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[recweaken-swap six-seed] verdict={verdict} seed_go {n_go}/6 swap {n_swap}/6 reignite {n_reig}/6 "
          f"load_bearing {n_lb}/6 ungated_ctrl {n_ungated}/6 lesion_holds {n_les}/6 reversible {n_rev}/6 "
          f"no_reset {n_nores}/6 det {n_det}/6 (ungated-forced-swap {n_forced_swap}/6) -> POOLED_GO={pooled_go}",
          flush=True)
    print(f"[recweaken-swap six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW recurrence-weaken single-move thought-swap de-risk")
    ap.add_argument("--smoke", action="store_true", help="operating-point grid on one seed")
    ap.add_argument("--six-seed", action="store_true", help="42/43/44/100/101/102 at the chosen operating point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boost", type=float, default=SWAP_BOOST, help="incumbent-loop STD boost during the evict window")
    ap.add_argument("--evict-steps", type=int, default=EVICT_STEPS, help="MAX steps the incumbent-loop depression runs")
    ap.add_argument("--chal-pa", type=float, default=CHAL_PA, help="challenger IN-gate drive strength (pA)")
    ap.add_argument("--reignite-hold", type=int, default=REIGNITE_HOLD, help="extended free-run to check B persists")
    ap.add_argument("--w-rec", type=float, default=W_REC, help="uniform recurrence weight (supra-critical)")
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--boost-grid", type=float, nargs="*", default=None, help="smoke: boosts")
    ap.add_argument("--evict-grid", type=int, nargs="*", default=None, help="smoke: evict windows")
    ap.add_argument("--chal-grid", type=float, nargs="*", default=None, help="smoke: challenger drive strengths")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_recweaken_swap.json")
    args = ap.parse_args()

    if args.six_seed:
        return run_six_seed(args)
    if args.smoke:
        return run_smoke(args.seed, args)
    r = evaluate_seed(args.seed, boost=args.boost, evict_steps=args.evict_steps, chal_pa=args.chal_pa,
                      reignite_hold=args.reignite_hold,
                      w_rec=args.w_rec, heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_recurrence_weaken_swap_derisk", "mode": "single", "result": r}, f,
                  indent=2, default=str)
    print(f"[recweaken-swap] wrote {args.json} seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
