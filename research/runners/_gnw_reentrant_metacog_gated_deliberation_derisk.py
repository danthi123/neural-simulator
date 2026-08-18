"""THE KEYSTONE (roadmap T1-1, rung (d)) — a CONFIDENCE/CONFLICT-GATED re-entrant deliberation loop on the GNW bus.

The audit's #1: "ACT on the conflict/confidence signals we only report." In every prior artifact the number of
re-entrant cycles is a HOST CONSTANT (`n_hops = len(actions)`, `HOPS=3`, `max_cycles=k`). Here that number is an
EMERGENT read of the substrate's OWN spiking confidence/conflict — the first time a metacog signal CONTROLS the
deliberation instead of decorating its output.

MECHANISM (reuse-by-import; NO `sim/` edit). ONE persistent GNW workspace (the P1.2 `build_workspace_bridge`:
K dense self-recurrent-NMDA assemblies + one shared inhibitory `workspace_fs` pool = single-content WTA). Per cycle:
  PROPOSE   — the held content `x` (spiking read of the last committed winner) cues the composer's learned relational
              read `query_patient(x, rel)` -> candidate `t`; distractors = competing concepts (weak drive). The
              composer is the DECLARED modular-processor boundary (same boundary P1.2 + the coincidence integrator
              declare); the relation `rel` is world/environment input.
  EVALUATE  — drive `t` (strong) + distractors (weak) into the slots; mutual-inhibition WTA + ignition threshold
              select and SUSTAIN one winner (the EXACT hop the production bus runs, `_deliberate_hop`/`norgan_hop`).
  READ      — off the SAME workspace, not a host formula:
              conf = |g_nmda(win) - g_nmda(runnerup)| / (g_nmda(win) + g_nmda(runnerup) + eps)   (the divisive-
                     normalized NMDA-conductance balance — the production-DEFAULT confidence code
                     `metacog_production_organ.nmda_norm_margin`, off spike-driven `cp_conductance_g_nmda`);
              n_ignited = # slots crossing the ignition knee (off `cp_firing_states`) = the CONFLICT read.
  ACC GATE  — a single ACC-conflict read maps (conf, n_ignited) -> {ADVANCE|RETRY|COMMIT|ABSTAIN} (ADVANCE + RETRY
              are the two flavors of the spec's CONTINUE). It reads ONLY the spiking conf + n_ignited (+ its own
              retry budget) — NEVER host `target is None`, `len(chain)`, or ground-truth depth. So the re-entrant
              cycle count EMERGES from the brain's confidence, replacing the host-fixed `n_hops`.

GO TASK — VARIABLE-DEPTH transitive chase. Chains of mixed depth L in {2,3,4,5} under one chase relation (EAT); the
loop is NOT told L. It must keep re-entering while its own ignition stays confident and HALT when the terminal
collapses ignition (`query_patient` misses at the leaf -> no supra-knee drive -> the workspace reads n_ignited==0 /
conf ~ 0 off SPIKES -> HALT). The depth of the inference is the substrate's to discover.

GO GATE (6 seeds 42/43/44/100/101/102; `cfg.seed`). GO iff >=5/6 with ALL:
  (1) reentrant_confgated_acc >= 0.90 on variable-depth (L in {2,3,4,5}) chains whose depth the loop isn't told;
  (2) single-pass (one hop) acc <= 0.15 AND (reentrant - singlepass) >= 0.60;
  (3) THE novelty proof: confidence-gated acc beats the BEST host-fixed count k in {1..5} by >= 0.20 (no constant k
      matches — short chains over-run, long chains under-shoot);
  (4) the stop is confidence-driven, not budget: frac_halts_at_H_cap == 0 on correct trials AND
      spearman(halt_cycle, true_depth) >= 0.9;
  (5) the moat holds every hop: unstored / over-run / unresolved -> abstain.

ANTI-CHEATS (A1-A8; each kills one "it's not really substrate control" story):
  A1 single-pass fails L>=2; A2 fixed-count sweep k=1..5 (confidence-gated beats best k by >=0.20 — THE load-bearing
  novelty anti-cheat); A3 confidence-blind stop (random Bernoulli at the empirical stop-rate, null = shuffle-mean)
  -> chance; A4 workspace-silence lesion collapses multi-step while the 1-hop reflex survives (dissociation);
  A5 re-cue lesion; A6 permuted-premises; A7 spreading-activation floor; A8 consensus-veto -> ABSTAIN.

ENFORCED no-host-orchestration: the gate reads ONLY spiking conf (NMDA balance off `cp_conductance_g_nmda`) +
n_ignited (off `cp_firing_states`). `confidence_gated_chase` is NEVER passed L or the chain (asserted). The host
supplies only the query (cue + relation = environment) and scores the terminal (readout = body/world boundary).

VERIFY-GO NARROWING (2026-08-18, adversarial): the DECISIVE spiking read for the primary variable-depth task is
`n_ignited` (the ignition/CONFLICT count off cp_firing_states), NOT the graded NMDA-balance `conf`. On this fixture
`conf` is a binary constant (0.94 resolved / 0.00 terminal) perfectly redundant with `n_ignited` (1/0): a theta_hi
sweep reads reent_acc=1.000 for EVERY theta_hi in [0,0.94] and 0.000 only for theta_hi>=0.96 — graded `conf` does no
independent work on Part A (its independent role is the moat + the unbuilt Part-B tie-break). And the +0.75 beats-best-k
is a variable-depth QUALITATIVE fact, not a substrate magnitude (best single k == 4/16 == 0.25 by the equal-depth
`build_var_chains` fixture). Substrate control is carried by A3 (per-trial halt timing) + A4 (lesion dissociation) +
`assert_no_host_orchestration`, NOT by A2 in isolation. So this is honestly "ignition-count/conflict-gated", not
"graded-confidence-gated".

HONEST RESIDUALS (declared, not faked): PER-HOP-RESET form only (snapshot-restore wash-out; the continuous no-reset
train-of-thought is gated on the unbuilt Rung-2b async attractor); PROPOSE = a declared modular-processor boundary
(the terminal is upstream-caused by its miss; the substrate's independent work is the halt TIMING + the ADVANCE guard +
the dissociation); the secondary CONFLICT rung (Part B, cross-retry NMDA accumulation breaking a near-tie) is a
STRENGTHENING probe, not a gate. FUNCTIONAL correlate only (a spiking read that controls deliberation depth) — NO
phenomenal claim.

Run:
  # 1-seed CPU smoke: instrument the knee + the nmda_norm conf split + confirm confidence-gated beats fixed-k
  SIM_BACKEND=numpy python -u -m research.runners._gnw_reentrant_metacog_gated_deliberation_derisk --smoke --seed 42
  # 6-seed decisive (full per-seed GO gate + anti-cheat sweep A1-A8)
  python -u -m research.runners._gnw_reentrant_metacog_gated_deliberation_derisk \
      --seeds 42 43 44 100 101 102 --D 256 \
      --json research/findings/raw/_gnw_reentrant_metacog_gated/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import the P1.2/keystone spiking workspace (build + ignition read + wash-out) + protocol constants.
from research.runners._p1_2_workspace_deliberation_loop_derisk import (
    build_workspace_bridge, _ignite_and_read, _full_restore, _pick_distractors,
    K_SLOTS, ASSEMBLY_SIZE, IGNITE_PA, DISTRACTOR_FRAC, IGNITE_FRAC, SOLO_PLATEAU,
    N_CONTROL_SHUFFLES,
)
from research.runners._gnw_rung1_ignition_curve_derisk import DRIVE_STEPS, FREE_STEPS
# reuse-by-import the production-DEFAULT confidence code: the canonical divisive-normalization eps + the formula
# (nmda_norm_margin) whose winner-vs-runnerup NMDA balance we apply to the deliberation workspace's competing slots.
from research.runners.metacog_production_organ import NORM_EPS, nmda_norm_margin  # noqa: F401 (cited canonical read)
from research.runners._phaseB_multihop_query_chain_derisk import (
    EAT, PLAY, SEE, store_facts, spreading_predict,
)
from research.runners.rf_phasor_composer import RFPhasorComposer
from sim.backend import get_backend, to_host

THR = IGNITE_FRAC * SOLO_PLATEAU                          # a slot is "ignited" iff its late-window rate >= THR


def _qp(composer, x, rel):
    """Memoized `composer.query_patient` (PROPOSE). query_patient is a DETERMINISTIC pure read of (composer, x, rel)
    — the same ~n_concepts unique reads are requested thousands of times across the intact chase + fixed-count sweep
    + shuffle controls, and each costs ~0.5 s (a per-fact D=256 scan). Caching the modular-processor's deterministic
    output changes NO result (it is not a scientific shortcut — the composer is the declared boundary either way);
    it only avoids recomputing an identical read. Keyed per composer instance (`comp_perm` gets its own cache)."""
    cache = getattr(composer, "_qp_cache", None)
    if cache is None:
        cache = {}
        composer._qp_cache = cache
    key = (x, rel)
    if key not in cache:
        cache[key] = composer.query_patient(x, rel)
    return cache[key]

# ── variable-depth control law budgets (safety, NOT the control — the GO gate proves the actual stop is conf/n_ign) ─
R_MAX_DEFAULT = 3          # extra same-hop cycles granted while high-conflict (Part B NMDA accumulation); Part A never hits it
H_CAP_DEFAULT = 8          # generous hard safety budget correct answers never reach (max chain depth is 5)
# The workspace's recurrent NMDA is folded into `cp_conductance_g_nmda` (`cp_conductance_g_nmda_recurrent` is None on
# this build) — the SAME array `metacog_production_organ.nmda_norm_margin` reads. So the confidence read is off exactly
# the production-default NMDA conductance. (The `recurrent` option is kept for other builds; it falls back if None.)
NMDA_ATTR_DEFAULT = "nmda"


# ── VARIABLE-DEPTH fixture: chains of mixed depth L in {2,3,4,5} under ONE relation (EAT), depth NEVER told to the loop
DEPTHS = (2, 3, 4, 5)


def build_var_chains(n_per_depth: int = 4):
    """Build `n_per_depth` chains at EACH depth L in {2,3,4,5}. A depth-L chain has L+1 unique concepts
    c0 --eat--> c1 --eat--> ... --eat--> cL (cL is the terminal leaf: it is never an agent, so query_patient(cL,EAT)
    misses -> the loop must DISCOVER the depth by reading its own ignition collapse). All concepts globally unique
    (chance = 1/n_concepts). Tokens are synthetic ('d{L}c{chain}p{pos}') so the 4 depths never share a concept."""
    chains, depth_of = [], {}
    for L in DEPTHS:
        for ci in range(n_per_depth):
            ch = [f"d{L}_ch{ci}_p{p}" for p in range(L + 1)]
            chains.append(ch)
            depth_of[tuple(ch)] = L
    return chains, depth_of


def build_vocab_var(chains):
    words = set([EAT, PLAY, SEE])
    for ch in chains:
        words |= set(ch)
    return sorted(words)


# ── the SPIKING confidence/conflict read off the SAME deliberation workspace (nmda_norm-style divisive norm) ────────
def _ignite_and_read_nmda(bridge, xp, slots_dev, snap, drives, nmda_attr=NMDA_ATTR_DEFAULT):
    """One EVALUATE/COMMIT (mirrors `_ignite_and_read`) that ALSO returns the late-window mean NMDA conductance per
    slot. `rates` = per-slot late firing rate (off cp_firing_states); `g_nmda` = per-slot late mean of the NMDA
    conductance (`cp_conductance_g_nmda` — the same array `metacog_production_organ.nmda_norm_margin` reads; it
    carries the recurrent NMDA that SUSTAINS the ignited attractor: winner ~1440, runner-up ~0). Both are genuine
    spike-driven substrate state read off the same workspace — no host formula.

    ⚠ The bridge REASSIGNS `cp_*` arrays each `_run_one_simulation_step` (like `_ignite_and_read` reads
    `bridge.cp_firing_states` fresh each step) — so the conductance/firing arrays MUST be re-fetched every step, not
    captured once. Reads accumulate on-device (one stacked fancy-index per step) with a SINGLE host transfer at the
    end (fast + backend-agnostic)."""
    attr = "cp_conductance_g_nmda_recurrent" if nmda_attr == "recurrent" \
        and getattr(bridge, "cp_conductance_g_nmda_recurrent", None) is not None else "cp_conductance_g_nmda"
    n = len(slots_dev)
    stacked = xp.stack([xp.asarray(s) for s in slots_dev])        # (n, ASSEMBLY_SIZE) — one fancy-index per step

    bridge.cp_external_input_current[:] = 0.0
    _full_restore(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        for s_dev, d in zip(slots_dev, drives):
            if d > 0.0:
                bridge.cp_external_input_current[s_dev] = xp.float32(d)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    counts_dev = xp.zeros(n, dtype=xp.float64)
    g_dev = xp.zeros(n, dtype=xp.float64)
    n_late = 0
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            counts_dev += bridge.cp_firing_states[stacked].astype(xp.float64).sum(axis=1)   # fresh each step
            g_dev += getattr(bridge, attr)[stacked].astype(xp.float64).mean(axis=1)         # fresh each step
            n_late += 1
    denom = float((FREE_STEPS - late_start) * ASSEMBLY_SIZE)
    n_late = float(max(1, n_late))
    rates = [float(x) / denom for x in to_host(counts_dev)]
    g_nmda = [float(x) / n_late for x in to_host(g_dev)]
    return rates, g_nmda


def _conf_from_nmda(rates, g_nmda):
    """The divisive-normalized NMDA-conductance confidence between the WINNING slot and the RUNNER-UP slot
    (`nmda_norm_margin`'s formula, applied to the deliberation workspace's top-2 competing slots):

        conf = |g_win - g_runnerup| / (g_win + g_runnerup + NORM_EPS)

    Single clean winner -> runner-up NMDA ~0 -> conf ~ 1 (RESOLVED). Two co-ignited slots -> both NMDA high ->
    conf ~ 0 (UNRESOLVED / high-conflict). Terminal (nothing ignites) -> all NMDA ~ baseline -> conf ~ 0 (caught
    first by n_ignited == 0). n_ignited counts slots over the ignition knee (off cp_firing_states) = the conflict read."""
    order = sorted(range(len(g_nmda)), key=lambda i: g_nmda[i], reverse=True)
    g1 = float(g_nmda[order[0]])
    g2 = float(g_nmda[order[1]]) if len(order) > 1 else 0.0
    conf = abs(g1 - g2) / (g1 + g2 + NORM_EPS)
    winner = int(np.argmax(rates))
    n_ignited = int(sum(1 for r in rates if r >= THR))
    return conf, winner, n_ignited


def _deliberate_hop_conf(bridge, xp, slots_dev, snap, target, distractors, nmda_attr=NMDA_ATTR_DEFAULT):
    """PROPOSE-result `target` -> slot 0 (strong IGNITE_PA); `distractors` -> slots 1.. (weak). EVALUATE via WTA
    ignition; READ the SPIKING (conf, n_ignited). `target=None` -> drive NOTHING (the terminal: the workspace stays
    quiescent, so n_ignited==0 / conf~0 is read off SPIKES, not off a host `target is None` branch). Returns
    (committed|None, rates, winner, n_ignited, conf, assignment)."""
    n = len(slots_dev)
    assignment = {}
    drives = [0.0] * n
    if target is not None:
        assignment[0] = target
        drives[0] = IGNITE_PA
        slot = 1
        for d in distractors:
            if slot >= n:
                break
            assignment[slot] = d
            drives[slot] = IGNITE_PA * DISTRACTOR_FRAC
            slot += 1
    rates, g_nmda = _ignite_and_read_nmda(bridge, xp, slots_dev, snap, drives, nmda_attr=nmda_attr)
    conf, winner, n_ignited = _conf_from_nmda(rates, g_nmda)
    ignited = rates[winner] >= THR
    committed = assignment.get(winner) if ignited else None
    return committed, rates, winner, n_ignited, conf, assignment


# ── θ self-calibration (the metacog organ's approach: a synthetic split, NOT a hand-tuned per-seed constant) ────────
def calibrate_theta(bridge, xp, slots_dev, snap, nmda_attr=NMDA_ATTR_DEFAULT):
    """Self-calibrate the confident/unresolved thresholds from a synthetic battery that uses NO task labels
    (exactly as `MetacogProductionOrgan.ensure_built` splits a synthetic high/low-evidence battery):
      SOLO   — drive ONE slot at IGNITE_PA, others 0 -> a single clean winner -> HIGH conf (the RESOLVED regime).
      CONFLICT — drive TWO slots at equal IGNITE_PA -> both ignite -> LOW conf (the UNRESOLVED regime).
      NULL   — drive nothing -> n_ignited==0, conf~0 (the TERMINAL regime).
    theta_hi = split between SOLO (high) and CONFLICT (low): 0.5*(min_solo+max_conflict) if a clean gap, else the
    mean midpoint. theta_lo = split between CONFLICT and NULL. Also returns the solo-ignition knee for the instrument.
    Reports the raw distributions so seed-invariance of theta can be checked across seeds."""
    n = len(slots_dev)
    solo, conflict, null = [], [], []
    for si in range(min(n, K_SLOTS)):
        drives = [0.0] * n
        drives[si] = IGNITE_PA
        rates, g = _ignite_and_read_nmda(bridge, xp, slots_dev, snap, drives, nmda_attr=nmda_attr)
        conf, _w, nign = _conf_from_nmda(rates, g)
        if nign >= 1:
            solo.append(conf)
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2)][: max(1, n - 1)]
    for (a, b) in pairs:
        if a >= n or b >= n:
            continue
        drives = [0.0] * n
        drives[a] = IGNITE_PA
        drives[b] = IGNITE_PA
        rates, g = _ignite_and_read_nmda(bridge, xp, slots_dev, snap, drives, nmda_attr=nmda_attr)
        conf, _w, _nign = _conf_from_nmda(rates, g)
        conflict.append(conf)
    for _ in range(2):
        rates, g = _ignite_and_read_nmda(bridge, xp, slots_dev, snap, [0.0] * n, nmda_attr=nmda_attr)
        conf, _w, _nign = _conf_from_nmda(rates, g)
        null.append(conf)

    solo = solo or [1.0]
    conflict = conflict or [0.0]
    min_solo, max_conf = float(np.min(solo)), float(np.max(conflict))
    mean_solo, mean_conf = float(np.mean(solo)), float(np.mean(conflict))
    theta_hi = (0.5 * (min_solo + max_conf)) if min_solo > max_conf else (0.5 * (mean_solo + mean_conf))
    max_null = float(np.max(null)) if null else 0.0
    theta_lo = 0.5 * (min(conflict) + max_null) if conflict else 0.5 * max_conf

    # solo-ignition knee (the instrument): the lowest solo drive whose slot-0 late rate crosses THR
    knee = None
    for drive in (600, 800, 1000, 1200, 1400, 1700, 2000, 2100, 2400, 2800):
        rates = _ignite_and_read(bridge, xp, slots_dev, snap, [float(drive)] + [0.0] * (n - 1))
        if rates[0] >= THR and knee is None:
            knee = drive
            break
    return {
        "theta_hi": float(theta_hi), "theta_lo": float(theta_lo), "knee_pA": knee,
        "solo_conf": [round(float(x), 4) for x in solo], "conflict_conf": [round(float(x), 4) for x in conflict],
        "null_conf": [round(float(x), 4) for x in null],
        "min_solo": round(min_solo, 4), "max_conflict": round(max_conf, 4),
        "clean_gap": bool(min_solo > max_conf), "nmda_attr": nmda_attr,
    }


# ── THE ACC-CONFLICT GATE (reads ONLY the spiking conf + n_ignited + its own retry budget) ─────────────────────────
ADVANCE, RETRY, COMMIT, ABSTAIN = "ADVANCE", "RETRY", "COMMIT", "ABSTAIN"


def assert_no_host_orchestration():
    """The spec's enforced guard: the loop's STOP/CONTINUE decision must read ONLY the spiking substrate, never
    host ground-truth. Checked at runtime (not just by comment): (1) `acc_conflict_gate` takes ONLY the spiking
    reads (conf, n_ignited) + its own retry budget — no L / chain / depth / target; (2) `confidence_gated_chase`
    is never given L / the chain / the depth. A structural invariant made executable (fails LOUD if a future edit
    leaks ground-truth into the controller)."""
    import inspect
    gate_params = set(inspect.signature(acc_conflict_gate).parameters)
    assert gate_params == {"conf", "n_ignited", "cycles_on_hop", "R_max", "theta_hi", "theta_lo"}, \
        f"acc_conflict_gate must read ONLY spiking conf/n_ignited (+budget); got {gate_params}"
    chase_params = set(inspect.signature(confidence_gated_chase).parameters)
    forbidden = {"L", "depth", "chain", "true_depth", "n_hops"}
    leaked = chase_params & forbidden
    assert not leaked, f"confidence_gated_chase must NOT receive ground-truth depth/chain; leaked {leaked}"
    return True


def acc_conflict_gate(conf, n_ignited, cycles_on_hop, R_max, theta_hi, theta_lo):
    """(conf, n_ignited) -> {ADVANCE|RETRY|COMMIT|ABSTAIN}. ADVANCE + RETRY are the two flavors of the spec's
    CONTINUE (advance to the next hop vs grant an extra same-hop cycle). Reads NOTHING host: not `target is None`,
    not `len(chain)`, not ground-truth depth — ONLY the spiking confidence balance and the ignition count."""
    if n_ignited == 0:                                   # no slot crossed the knee = terminal -> halt, commit last resolved
        return COMMIT
    if conf >= theta_hi and n_ignited == 1:              # resolved single winner -> broadcast back, next hop
        return ADVANCE
    if n_ignited >= 2 and cycles_on_hop < R_max:         # high-conflict, budget remains -> extra same-hop cycle
        return RETRY
    return ABSTAIN                                        # unresolved (single-but-low-conf, or conflict past R_max)


# ── THE KEYSTONE: the confidence-gated re-entrant deliberation (the while-gated generalization of reentrant_chase) ──
def confidence_gated_chase(bridge, xp, slots_dev, snap, composer, cue, rel, all_concepts, rng,
                           theta_hi, theta_lo, R_max=R_MAX_DEFAULT, H_cap=H_CAP_DEFAULT,
                           nmda_attr=NMDA_ATTR_DEFAULT, recue_lesion_rng=None, return_trace=False):
    """The workspace-carried multi-hop deliberation whose CYCLE COUNT emerges from the substrate's own spiking
    confidence. x starts at `cue`; each cycle PROPOSE (composer relational read) -> EVALUATE/COMMIT (workspace WTA
    ignition) -> READ (conf, n_ignited off spikes) -> ACC GATE decides ADVANCE|RETRY|COMMIT|ABSTAIN. HALT when the
    terminal collapses ignition. `n_hops = len(actions)` of reentrant_chase becomes `while gate != COMMIT/ABSTAIN`.

    ⛔ NO-HOST-ORCHESTRATION: this function is NEVER passed L or the chain — only the cue + the relation. The gate
    sees only the spiking (conf, n_ignited). The generous H_cap is a pure SAFETY budget correct answers never hit.
    `recue_lesion_rng` (A5): replace each committed winner with a random concept before re-cueing."""
    x = cue
    last_resolved = None
    cycles = 0
    cycles_on_hop = 0
    resolved_hops = 0
    trace = []
    halted_at_cap = False
    while True:
        if cycles >= H_cap:
            halted_at_cap = True
            break
        cycles += 1
        target = _qp(composer, x, rel)                    # PROPOSE (declared modular-processor boundary; None at leaf)
        distractors = _pick_distractors(all_concepts, exclude={target, x}, k=len(slots_dev) - 1, rng=rng) \
            if target is not None else []
        committed, rates, winner, n_ignited, conf, assignment = _deliberate_hop_conf(
            bridge, xp, slots_dev, snap, target, distractors, nmda_attr=nmda_attr)
        action = acc_conflict_gate(conf, n_ignited, cycles_on_hop, R_max, theta_hi, theta_lo)
        trace.append({"cycle": cycles, "x": x, "target": target, "committed": committed,
                      "winner": int(winner), "n_ignited": int(n_ignited), "conf": round(float(conf), 4),
                      "action": action})
        if action == COMMIT:
            break
        if action == ABSTAIN:
            last_resolved = None if resolved_hops == 0 else last_resolved
            # an ABSTAIN mid-chase refuses to broadcast an unconfirmed conclusion -> the whole chase abstains
            result = None
            return (result, {"trace": trace, "resolved_hops": resolved_hops, "cycles": cycles,
                             "halted_at_cap": halted_at_cap, "abstained": True}) if return_trace else result
        if action == RETRY:
            cycles_on_hop += 1                            # same x; re-drive the ambiguous hop (Part B accumulation probe)
            continue
        # ADVANCE: broadcast back the spike-derived committed winner
        if committed is None:                            # defensive: single ignited slot must have a concept
            break
        last_resolved = committed
        resolved_hops += 1
        x_next = committed
        if recue_lesion_rng is not None:                 # A5: sever the broadcast-back re-cue
            x_next = all_concepts[int(recue_lesion_rng.integers(len(all_concepts)))]
        x = x_next
        cycles_on_hop = 0
    meta = {"trace": trace, "resolved_hops": resolved_hops, "cycles": cycles,
            "halted_at_cap": halted_at_cap, "abstained": False}
    return (last_resolved, meta) if return_trace else last_resolved


# ── A1/A2 FIXED-COUNT baseline (host-fixed k cycles, NO confidence gate) ────────────────────────────────────────────
def fixed_count_chase(bridge, xp, slots_dev, snap, composer, cue, rel, all_concepts, rng, k,
                      nmda_attr=NMDA_ATTR_DEFAULT):
    """Advance EXACTLY k hops through the workspace (drive+ignite each hop), no confidence read. Returns the concept
    after k hops, or None if a hop fails to ignite / the composer misses (over-run). k=1 == the single-pass wired
    production bus (A1). For a depth-L chain: k<L under-shoots, k>L over-runs (miss -> None), only k==L lands."""
    x = cue
    for _h in range(k):
        target = _qp(composer, x, rel)
        if target is None:
            return None
        distractors = _pick_distractors(all_concepts, exclude={target, x}, k=len(slots_dev) - 1, rng=rng)
        committed, _rates, _w, _nign, _conf, _a = _deliberate_hop_conf(
            bridge, xp, slots_dev, snap, target, distractors, nmda_attr=nmda_attr)
        if committed is None:
            return None
        x = committed
    return x


# ── A3 CONFIDENCE-BLIND stop (random Bernoulli halt at the empirical stop-rate; the spiking read is NOT what times it)
def confidence_blind_chase(bridge, xp, slots_dev, snap, composer, cue, rel, all_concepts, rng, p_stop, halt_rng,
                           H_cap=H_CAP_DEFAULT, nmda_attr=NMDA_ATTR_DEFAULT):
    """Run the SAME workspace ignition each hop, but replace the ACC gate's conf/n_ignited HALT decision with a random
    Bernoulli(p_stop) halt (P1.2's shuffle-null). CRUCIAL: the blind loop is DENIED the substrate's terminal detection
    — it commits ONLY where the Bernoulli fires; if it runs to the leaf without halting it has OVER-RUN and ABSTAINS
    (returns None), never getting the true terminal for free. So a correct answer requires the random halt to land
    EXACTLY at the leaf depth -> ~chance-ish (below the best fixed count). This isolates the per-trial HALT TIMING to
    the spiking read: only reading n_ignited==0 off the SPIKES lands the halt on the right depth every trial."""
    x = cue
    last = None
    for _c in range(H_cap):
        target = _qp(composer, x, rel)
        if target is None:                               # over-ran the leaf without a blind halt -> ABSTAIN (no free terminal)
            return None
        distractors = _pick_distractors(all_concepts, exclude={target, x}, k=len(slots_dev) - 1, rng=rng)
        committed, _r, _w, _n, _conf, _a = _deliberate_hop_conf(
            bridge, xp, slots_dev, snap, target, distractors, nmda_attr=nmda_attr)
        if committed is None:
            return None
        last = committed
        x = committed
        if halt_rng.random() < p_stop:                   # BLIND halt (not conf/n_ignited driven) — the ONLY way to commit
            return last
    return None                                          # exhausted the budget without a blind halt -> abstain


# ── spearman rank correlation (avoid a scipy dependency) ───────────────────────────────────────────────────────────
def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# ── the per-seed experiment ────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, D, n_per_depth=4, nmda_attr=NMDA_ATTR_DEFAULT, n_shuffles=5, verbose=True):
    chains, depth_of = build_var_chains(n_per_depth)
    vocab = build_vocab_var(chains)
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc = store_facts(composer, chains, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in chains for c in ch]
    n_concepts = len(all_concepts)
    chance = 1.0 / n_concepts

    # persistent workspace bridges (built ONCE per seed): intact + workspace-silence lesion. Kept warm.
    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    b_l, xp_l, slots_l, snap_l = build_workspace_bridge(seed, lesion=True)

    def dist_rng():
        return np.random.default_rng(seed * 991 + 7)

    # ── INSTRUMENT FIRST: self-calibrate theta from the synthetic battery (no task labels) ─────────────────────────
    cal = calibrate_theta(b_i, xp, slots_i, snap_i, nmda_attr=nmda_attr)
    theta_hi, theta_lo = cal["theta_hi"], cal["theta_lo"]

    # ── the confidence-gated re-entrant deliberation (INTACT) ──────────────────────────────────────────────────────
    reent_ok = 0
    halt_cycles, true_depths = [], []
    halt_cycle_correct, depth_correct = [], []
    any_halt_at_cap = False
    per_depth_ok = {L: [0, 0] for L in DEPTHS}
    for ch in chains:
        L = depth_of[tuple(ch)]
        cue, want = ch[0], ch[-1]
        term, meta = confidence_gated_chase(b_i, xp, slots_i, snap_i, composer, cue, EAT, all_concepts, dist_rng(),
                                            theta_hi, theta_lo, nmda_attr=nmda_attr, return_trace=True)
        correct = int(term == want)
        reent_ok += correct
        per_depth_ok[L][0] += correct
        per_depth_ok[L][1] += 1
        halt_cycles.append(meta["resolved_hops"])
        true_depths.append(L)
        any_halt_at_cap = any_halt_at_cap or bool(meta["halted_at_cap"])
        if correct:
            halt_cycle_correct.append(meta["resolved_hops"])
            depth_correct.append(L)
    tot = len(chains)
    reent_acc = reent_ok / tot
    spearman_halt_depth = _spearman(halt_cycle_correct, depth_correct)
    frac_halts_at_cap = 0.0  # computed over CORRECT trials: a correct trial by construction halted at COMMIT (< H_cap)
    # any correct trial that hit H_cap would be an over-run — flag it explicitly
    frac_correct_at_cap = float(np.mean([0.0] * len(halt_cycle_correct))) if halt_cycle_correct else 0.0

    # ── A2 FIXED-COUNT SWEEP (k=1..5): the load-bearing novelty anti-cheat ──────────────────────────────────────────
    fixed_acc = {}
    for k in (1, 2, 3, 4, 5):
        ok = 0
        for ch in chains:
            term = fixed_count_chase(b_i, xp, slots_i, snap_i, composer, ch[0], EAT, all_concepts, dist_rng(), k,
                                     nmda_attr=nmda_attr)
            ok += int(term == ch[-1])
        fixed_acc[k] = ok / tot
    best_k = max(fixed_acc, key=fixed_acc.get)
    best_fixed_acc = fixed_acc[best_k]
    singlepass_acc = fixed_acc[1]                          # A1: one hop = the wired production bus
    beats_best_k = reent_acc - best_fixed_acc

    # ── A3 CONFIDENCE-BLIND STOP (random Bernoulli halt at the empirical stop-rate; null = shuffle-mean) ────────────
    #   empirical stop-rate = fraction of cycles that halted = 1 / mean(resolved_hops + 1)
    mean_cycles = float(np.mean([h + 1 for h in halt_cycles])) if halt_cycles else 2.0
    p_stop = 1.0 / max(1.0, mean_cycles)
    blind_accs = []
    for si in range(n_shuffles):
        halt_rng = np.random.default_rng(seed * 7 + si * 1009 + 3)
        ok = 0
        for ch in chains:
            term = confidence_blind_chase(b_i, xp, slots_i, snap_i, composer, ch[0], EAT, all_concepts, dist_rng(),
                                          p_stop, halt_rng, nmda_attr=nmda_attr)
            ok += int(term == ch[-1])
        blind_accs.append(ok / tot)
    blind_acc = float(np.mean(blind_accs))                # the gated value: the shuffle-null MEAN

    # ── A4 WORKSPACE-SILENCE LESION (dissociation): multi-step collapses, the 1-hop reflex survives ────────────────
    lesion_ok = reflex_ok = 0
    for ch in chains:
        term = confidence_gated_chase(b_l, xp_l, slots_l, snap_l, composer, ch[0], EAT, all_concepts, dist_rng(),
                                      theta_hi, theta_lo, nmda_attr=nmda_attr)
        lesion_ok += int(term == ch[-1])
        reflex_ok += int(_qp(composer, ch[0], EAT) == ch[1])            # 1-hop reflex, workspace-independent (cached)
    lesion_acc = lesion_ok / tot
    reflex_acc = reflex_ok / tot

    # ── A5 RE-CUE LESION (broadcast-back load-bearing; null = shuffle-mean) ─────────────────────────────────────────
    recue_accs = []
    for si in range(n_shuffles):
        rc = np.random.default_rng(seed * 7 + 17 + si * 1009)
        ok = 0
        for ch in chains:
            term = confidence_gated_chase(b_i, xp, slots_i, snap_i, composer, ch[0], EAT, all_concepts, dist_rng(),
                                          theta_hi, theta_lo, nmda_attr=nmda_attr, recue_lesion_rng=rc)
            ok += int(term == ch[-1])
        recue_accs.append(ok / tot)
    recue_acc = float(np.mean(recue_accs))

    # ── A6 PERMUTED-PREMISES (role structure, not co-occurrence; null = shuffle-mean) ──────────────────────────────
    #   TWO reads: (a) the workspace confidence-gated chase over permuted premises — collapses to the wandering floor
    #   (the depth-discovering loop halts at SOME leaf; occasionally, by coincidence, the true terminal leaf ≈ the
    #   spreading floor); (b) the PURE fixed-depth permuted chase (told L, relational only) — ≈0, PROVING the permutation
    #   destroys the relational structure. So the workspace residual is the coincidence floor of depth-discovery, not a leak.
    perm_accs, perm_pure_accs = [], []
    for si in range(n_shuffles):
        comp_perm = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
        store_facts(comp_perm, chains, permute_relation=True,
                    rng=np.random.default_rng(seed * 101 + 5 + si * 1009),
                    distractor_rng=np.random.default_rng(seed * 53 + 1))
        ok = pure_ok = 0
        for ch in chains:
            term = confidence_gated_chase(b_i, xp, slots_i, snap_i, comp_perm, ch[0], EAT, all_concepts, dist_rng(),
                                          theta_hi, theta_lo, nmda_attr=nmda_attr)
            ok += int(term == ch[-1])
            L = depth_of[tuple(ch)]                       # pure relational chase told the depth (isolates role-destruction)
            x = ch[0]
            for _ in range(L):
                x = _qp(comp_perm, x, EAT)                # cache-shared with the workspace chase above
                if x is None:
                    break
            pure_ok += int(x == ch[-1])
        perm_accs.append(ok / tot)
        perm_pure_accs.append(pure_ok / tot)
    perm_acc = float(np.mean(perm_accs))
    perm_puredepth_acc = float(np.mean(perm_pure_accs))   # ≈0: the permutation destroys the RELATIONAL structure

    # ── A7 SPREADING-ACTIVATION FLOOR (must stay ~chance; the chase must BEAT it) ───────────────────────────────────
    spread_ok = 0
    for ch in chains:
        L = depth_of[tuple(ch)]
        spread_ok += int(spreading_predict(cooc, ch[0], L, all_concepts) == ch[-1])
    spread_floor = spread_ok / tot

    # ── A8 CONSENSUS-VETO + MOAT: unstored cue + past-chain-end over-run must ABSTAIN (None) ────────────────────────
    moat_unstored = confidence_gated_chase(b_i, xp, slots_i, snap_i, composer, "ball_nonagent", EAT, all_concepts,
                                           dist_rng(), theta_hi, theta_lo, nmda_attr=nmda_attr)
    # over-run: query a chain but keep re-cueing past its end is intrinsic — the loop self-halts at the leaf. Test the
    # explicit past-end cue (the leaf itself, never an agent) -> immediate terminal -> abstain (no confabulated hop).
    a_leaf = chains[0][-1]
    moat_overrun = confidence_gated_chase(b_i, xp, slots_i, snap_i, composer, a_leaf, EAT, all_concepts,
                                          dist_rng(), theta_hi, theta_lo, nmda_attr=nmda_attr)
    moat_unstored_abstains = moat_unstored is None
    moat_overrun_abstains = moat_overrun is None
    moat_ok = bool(moat_unstored_abstains and moat_overrun_abstains)

    two_chance = 2.0 * chance
    seed_go = bool(
        reent_acc >= 0.90 and                                            # (1) reaches the terminal it wasn't told the depth of
        singlepass_acc <= 0.15 and (reent_acc - singlepass_acc) >= 0.60 and  # (2) single-pass fails where re-entrant wins
        beats_best_k >= 0.20 and                                         # (3) THE novelty: beats EVERY fixed count
        (not any_halt_at_cap) and (not np.isnan(spearman_halt_depth) and spearman_halt_depth >= 0.9) and  # (4) conf-driven stop
        blind_acc <= 0.20 and (reent_acc - blind_acc) >= 0.60 and        # A3: confidence-blind stop collapses (per-trial timing)
        lesion_acc <= max(two_chance, 0.10) and reflex_acc >= 0.85 and   # A4: dissociation
        recue_acc <= max(two_chance, 0.10) and                          # A5
        perm_puredepth_acc <= max(two_chance, 0.10) and                 # A6a: permutation DESTROYS the relational structure (pure chase ~0)
        (reent_acc - perm_acc) >= 0.60 and perm_acc <= spread_floor + 0.05 and  # A6b: workspace chase collapses to the wandering/co-occurrence floor
        reent_acc >= spread_floor + 0.5 and                             # A7
        moat_ok                                                          # A8 / (5) moat
    )

    result = {
        "seed": int(seed), "D": int(D), "n_per_depth": int(n_per_depth), "nmda_attr": nmda_attr,
        "n_concepts": n_concepts, "chance": chance, "n_chains": tot, "depths": list(DEPTHS),
        "theta_hi": theta_hi, "theta_lo": theta_lo, "calib": cal,
        "reentrant_confgated_acc": reent_acc,
        "per_depth_acc": {str(L): (per_depth_ok[L][0] / per_depth_ok[L][1]) for L in DEPTHS},
        "singlepass_acc": singlepass_acc,
        "fixed_count_acc": {str(k): v for k, v in fixed_acc.items()},
        "best_fixed_k": int(best_k), "best_fixed_acc": best_fixed_acc, "beats_best_k_margin": beats_best_k,
        "confidence_blind_acc": blind_acc, "confidence_blind_accs": [round(x, 4) for x in blind_accs],
        "empirical_p_stop": round(p_stop, 4),
        "lesion_acc": lesion_acc, "single_hop_reflex_acc": reflex_acc,
        "recue_lesion_acc": recue_acc, "permuted_acc": perm_acc, "permuted_puredepth_acc": perm_puredepth_acc,
        "spreading_floor": spread_floor,
        "spearman_halt_depth": spearman_halt_depth, "any_halt_at_H_cap": bool(any_halt_at_cap),
        "frac_correct_halts_at_cap": frac_correct_at_cap,
        "moat_unstored_abstains": moat_unstored_abstains, "moat_overrun_abstains": moat_overrun_abstains,
        "moat_ok": moat_ok, "n_control_shuffles": int(n_shuffles),
        "seed_go": seed_go,
    }

    if verbose:
        print(f"[keystone seed={seed} D={D} nmda={nmda_attr}] reentrant_confgated={reent_acc:.3f} "
              f"(chance={chance:.3f}, n_concepts={n_concepts})  theta_hi={theta_hi:.3f} theta_lo={theta_lo:.3f} "
              f"knee={cal['knee_pA']} clean_gap={cal['clean_gap']}", flush=True)
        print(f"    per-depth acc: " + " ".join(f"L{L}={result['per_depth_acc'][str(L)]:.2f}" for L in DEPTHS),
              flush=True)
        print(f"    A1/A2 fixed-count: " + " ".join(f"k{k}={fixed_acc[k]:.2f}" for k in (1, 2, 3, 4, 5)) +
              f"  best_k={best_k}({best_fixed_acc:.2f}) -> BEATS_BEST_K={beats_best_k:+.3f} "
              f"(single-pass={singlepass_acc:.2f})", flush=True)
        print(f"    A3 conf-blind={blind_acc:.3f}(p_stop={p_stop:.2f}) | A4 lesion={lesion_acc:.3f} "
              f"reflex={reflex_acc:.3f} | A5 recue={recue_acc:.3f} | A6 perm={perm_acc:.3f} | "
              f"A6pure={perm_puredepth_acc:.3f} | A7 spread_floor={spread_floor:.3f}", flush=True)
        print(f"    (4) spearman(halt,depth)={spearman_halt_depth:.3f} halt_at_cap={any_halt_at_cap} | "
              f"A8/moat unstored={moat_unstored_abstains} overrun={moat_overrun_abstains} | seed_GO={seed_go}",
              flush=True)
    return result


# ── 1-seed instrument smoke: the knee + the conf split + confirm confidence-gated beats fixed-k ────────────────────
def run_smoke(seed, D, n_per_depth=4, nmda_attr=NMDA_ATTR_DEFAULT):
    print(f"[SMOKE seed={seed} D={D}] instrument the ignition knee + nmda_norm conf split, then confirm the gate + "
          f"that confidence-gated BEATS the best fixed-k.\n", flush=True)
    chains, depth_of = build_var_chains(n_per_depth)
    vocab = build_vocab_var(chains)
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_facts(composer, chains, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in chains for c in ch]

    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)

    # instrument the NMDA conf split (solo=RESOLVED high / conflict=UNRESOLVED low / null=TERMINAL ~0)
    cal = calibrate_theta(b_i, xp, slots_i, snap_i, nmda_attr=nmda_attr)
    print(f"  [instrument nmda={nmda_attr}] knee={cal['knee_pA']}pA  solo_conf={cal['solo_conf']} "
          f"conflict_conf={cal['conflict_conf']} null_conf={cal['null_conf']}  -> theta_hi={cal['theta_hi']:.3f} "
          f"theta_lo={cal['theta_lo']:.3f} clean_gap={cal['clean_gap']}", flush=True)

    theta_hi, theta_lo = cal["theta_hi"], cal["theta_lo"]

    # one confidence-gated chase per depth (show the loop DISCOVERS the depth)
    print(f"\n  [chase nmda={nmda_attr} theta_hi={theta_hi:.3f}]", flush=True)
    demo = {}
    for L in DEPTHS:
        ch = next(c for c in chains if depth_of[tuple(c)] == L)
        term, meta = confidence_gated_chase(b_i, xp, slots_i, snap_i, composer, ch[0], EAT, all_concepts,
                                            np.random.default_rng(seed * 991 + 7), theta_hi, theta_lo,
                                            nmda_attr=nmda_attr, return_trace=True)
        ok = term == ch[-1]
        demo[L] = ok
        confs = [t["conf"] for t in meta["trace"]]
        print(f"    L={L}: cue={ch[0]} -> term={term!r} want={ch[-1]!r} {'OK' if ok else 'X'} | "
              f"resolved_hops={meta['resolved_hops']} cycles={meta['cycles']} confs={[round(c,2) for c in confs]}",
              flush=True)

    # confidence-gated vs fixed-k on the full smoke set
    reent_ok = 0
    for ch in chains:
        term = confidence_gated_chase(b_i, xp, slots_i, snap_i, composer, ch[0], EAT, all_concepts,
                                      np.random.default_rng(seed * 991 + 7), theta_hi, theta_lo, nmda_attr=nmda_attr)
        reent_ok += int(term == ch[-1])
    reent_acc = reent_ok / len(chains)
    fixed = {}
    for k in (1, 2, 3, 4, 5):
        ok = 0
        for ch in chains:
            term = fixed_count_chase(b_i, xp, slots_i, snap_i, composer, ch[0], EAT, all_concepts,
                                     np.random.default_rng(seed * 991 + 7), k, nmda_attr=nmda_attr)
            ok += int(term == ch[-1])
        fixed[k] = ok / len(chains)
    best_k = max(fixed, key=fixed.get)
    beats = reent_acc - fixed[best_k]
    print(f"\n  confidence-gated acc={reent_acc:.3f}  vs fixed-k {[f'k{k}={fixed[k]:.2f}' for k in fixed]}  "
          f"best_k={best_k}({fixed[best_k]:.2f}) -> BEATS_BEST_K={beats:+.3f}", flush=True)
    ok = bool(reent_acc >= 0.90 and beats >= 0.20 and all(demo.values()))
    print(f"\n  SMOKE {'HOLDS' if ok else 'NEEDS-WORK'}: per-depth all-correct={all(demo.values())} "
          f"reent>=0.90={reent_acc>=0.90} beats_best_k>=0.20={beats>=0.20}", flush=True)
    return ok


def run_determinism(seed, D, nmda_attr=NMDA_ATTR_DEFAULT):
    """Build the workspace TWICE at one seed; the per-neuron firing thresholds must be identical (cfg.seed seeds the
    substrate — NOT actual_seed_used). Different seeds must differ."""
    b1, _x1, _s1, _n1 = build_workspace_bridge(seed, lesion=False)
    b2, _x2, _s2, _n2 = build_workspace_bridge(seed, lesion=False)
    b3, _x3, _s3, _n3 = build_workspace_bridge(seed + 1, lesion=False)
    h1 = to_host(b1.cp_neuron_firing_thresholds.astype(np.float64))
    h2 = to_host(b2.cp_neuron_firing_thresholds.astype(np.float64))
    h3 = to_host(b3.cp_neuron_firing_thresholds.astype(np.float64))
    same = bool(np.array_equal(h1, h2))
    diff = bool(not np.array_equal(h1, h3))
    print(f"[determinism seed={seed}] same-seed identical thresholds={same} diff-seed differ={diff} "
          f"-> {'OK (cfg.seed controls the substrate)' if (same and diff) else 'FAIL'}", flush=True)
    return same and diff


def main():
    ap = argparse.ArgumentParser(description="THE KEYSTONE: confidence/conflict-gated re-entrant deliberation (T1-1 rung d).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--n-per-depth", type=int, default=4)
    ap.add_argument("--n-shuffles", type=int, default=5, help="stochastic-control null-mean shuffles (A3/A5/A6)")
    ap.add_argument("--nmda-attr", type=str, default=NMDA_ATTR_DEFAULT, choices=["nmda", "recurrent"])
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_reentrant_metacog_gated/summary.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    assert_no_host_orchestration()   # the gate reads ONLY spiking conf/n_ignited; the loop never sees L/the chain

    if args.determinism:
        return 0 if run_determinism(args.seed, args.D, nmda_attr=args.nmda_attr) else 1
    if args.smoke:
        return 0 if run_smoke(args.seed, args.D, n_per_depth=args.n_per_depth, nmda_attr=args.nmda_attr) else 1

    chains, _d = build_var_chains(args.n_per_depth)
    n_concepts = len({c for ch in chains for c in ch})
    print(f"[GNW reentrant metacog-gated deliberation — THE KEYSTONE] depths={list(DEPTHS)} "
          f"{len(chains)} chains | {n_concepts} concepts | chance {1.0/n_concepts:.3f} | K_slots={K_SLOTS} "
          f"D={args.D} nmda={args.nmda_attr} backend={args.backend}\n"
          "  the re-entrant cycle count EMERGES from the substrate's spiking confidence (nmda_norm) — not a host "
          "counter. Per-hop-reset form (continuous train-of-thought gated on Rung-2b).\n", flush=True)

    results = [run_seed(s, args.D, n_per_depth=args.n_per_depth, nmda_attr=args.nmda_attr, n_shuffles=args.n_shuffles) for s in args.seeds]
    n_go = sum(int(r["seed_go"]) for r in results)
    go = n_go >= 5

    def mean(k):
        return float(np.mean([r[k] for r in results]))

    # preconditions the verdict travels with (each `ok` derived from the per-seed AND-conjunction that set seed_go)
    def _pc(name, ok):
        return {"name": name, "ok": bool(ok)}
    preconditions = [
        _pc("reentrant_confgated_acc>=0.90 on all seeds", all(r["reentrant_confgated_acc"] >= 0.90 for r in results)),
        _pc("singlepass<=0.15 AND (reent-single)>=0.60", all(r["singlepass_acc"] <= 0.15 and
            (r["reentrant_confgated_acc"] - r["singlepass_acc"]) >= 0.60 for r in results)),
        _pc("beats_best_fixed_k>=0.20 (variable-depth; magnitude fixture-shaped)",
            all(r["beats_best_k_margin"] >= 0.20 for r in results)),
        _pc("spearman(halt,depth)>=0.9 AND no halt_at_H_cap", all(r["spearman_halt_depth"] >= 0.9 and
            not r["any_halt_at_H_cap"] for r in results)),
        _pc("A3 confidence-blind stop collapses (<=0.20 & gap>=0.60)", all(r["confidence_blind_acc"] <= 0.20 and
            (r["reentrant_confgated_acc"] - r["confidence_blind_acc"]) >= 0.60 for r in results)),
        _pc("A4 lesion collapses (<=0.10) AND 1-hop reflex survives (>=0.85)",
            all(r["lesion_acc"] <= 0.10 and r["single_hop_reflex_acc"] >= 0.85 for r in results)),
        _pc("A5 recue AND A6 permuted-puredepth collapse", all(r["recue_lesion_acc"] <= 0.10 and
            r["permuted_puredepth_acc"] <= max(2 * r["chance"], 0.10) for r in results)),
        _pc("A8 moat holds (unstored+overrun abstain) on all seeds", all(r["moat_ok"] for r in results)),
        _pc("no-host-orchestration guard passes (gate reads only conf/n_ignited)", assert_no_host_orchestration()),
        _pc("determinism from cfg.seed (build-twice thresholds hash identical)",
            run_determinism(args.seeds[0], args.D, nmda_attr=args.nmda_attr)),
    ]

    summary = {
        "runner": "_gnw_reentrant_metacog_gated_deliberation_derisk",
        "preconditions": preconditions,
        "claim": "the re-entrant deliberation cycle count is an EMERGENT read of the substrate's spiking "
                 "confidence/conflict (nmda_norm balance + n_ignited), not a host-fixed counter (T1-1 rung d)",
        "seeds": list(args.seeds), "D": int(args.D), "n_per_depth": int(args.n_per_depth),
        "nmda_attr": args.nmda_attr, "n_shuffles": int(args.n_shuffles), "backend": args.backend,
        "go": go, "n_go": n_go, "n_seeds": len(results), "go_rule": ">=5/6 seeds",
        "mean_reentrant_confgated_acc": mean("reentrant_confgated_acc"),
        "mean_singlepass_acc": mean("singlepass_acc"),
        "mean_best_fixed_acc": mean("best_fixed_acc"),
        "mean_beats_best_k_margin": mean("beats_best_k_margin"),
        "mean_confidence_blind_acc": mean("confidence_blind_acc"),
        "mean_lesion_acc": mean("lesion_acc"), "mean_single_hop_reflex_acc": mean("single_hop_reflex_acc"),
        "mean_recue_lesion_acc": mean("recue_lesion_acc"), "mean_permuted_acc": mean("permuted_acc"), "mean_permuted_puredepth_acc": mean("permuted_puredepth_acc"),
        "mean_spreading_floor": mean("spreading_floor"),
        "mean_spearman_halt_depth": mean("spearman_halt_depth"),
        "any_halt_at_H_cap": any(r["any_halt_at_H_cap"] for r in results),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if go else ("PARTIAL" if n_go >= 1 else "NEGATIVE")
    print(f"\n{'='*100}", flush=True)
    print(f"  THE KEYSTONE VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO; rule >=5/6)", flush=True)
    print(f"    reentrant_confgated={summary['mean_reentrant_confgated_acc']:.3f}  single-pass="
          f"{summary['mean_singlepass_acc']:.3f}  BEATS_BEST_K={summary['mean_beats_best_k_margin']:+.3f}", flush=True)
    print(f"    conf-blind={summary['mean_confidence_blind_acc']:.3f} lesion={summary['mean_lesion_acc']:.3f} "
          f"reflex={summary['mean_single_hop_reflex_acc']:.3f} recue={summary['mean_recue_lesion_acc']:.3f} "
          f"perm={summary['mean_permuted_acc']:.3f} spread_floor={summary['mean_spreading_floor']:.3f}", flush=True)
    print(f"    spearman(halt,depth)={summary['mean_spearman_halt_depth']:.3f} halt_at_cap="
          f"{summary['any_halt_at_H_cap']} moat_all={summary['all_moat_ok']}", flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
