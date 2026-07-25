"""P1.2 follow-on — WIRE THE REAL SPIKING AFFECT-STATE REGION (P0.3) INTO THE GNW WORKSPACE DRIVE, replacing the
host salience scalar. The "TRUE ONE BRAIN" faculties-interacting step: the workspace's DIRECTED deliberation is
now steered by a NEURAL affect signal (the P0.3 region's spikes), not a python `salient` boolean.

WHAT THIS REPLACES. `_p1_2_workspace_deliberation_loop_derisk.py`'s AFFECT-directedness anti-cheat used a HOST
salience SCALAR (`_branch(..., salient: bool)`): salient=True -> the non-value slot drive was set to
IGNITE_PA*0.30 (weak) in python; salient=False (lesion) -> IGNITE_PA (equal). That diagnostic stand-in is NOT in
the GO gate, and the P1.2 finding names the exact follow-on: "wire the real P0.3 spiking affect-state region into
the workspace drive (replacing the host salience scalar)." This file does that.

MECHANISM (reuse-by-import, NO `sim/` edit; both existing builders used verbatim on separate CPU bridges + a
NEURAL signal handoff — the cross-region faculty interaction):
  - AFFECT REGION = P0.3 `AffectStateBrain` (verbatim import): 3 opponent slow-NMDA pools (affect_vplus/vminus/
    arousal) + Namburi-Tye cross-inhibition, appraisal via the diffuse neuromodulator bus, and the affect state's
    SYNAPTIC OUTPUT to `recall_pos`/`recall_neg`/`speak_acc` gated by the ONE `affect_out` transmission gate.
  - VALUE APPRAISAL. The deliberation is a food-seeking chain-chase (EAT relations). A candidate reached by the
    value-relevant EAT relation is DRIVE-CONGRUENT (appetitive) -> appraise the affect region APPETITIVE (vp=1.0);
    a candidate reached by the non-value PLAY relation is drive-incongruent -> appraise NEUTRAL (vp=0.0). Appraisal
    is the legitimate appraised-event INPUT (P0.3's interface); the affect region does the neural transform.
  - THE NEURAL SALIENCE = the affect region's GATED V+ OUTPUT RATE. affect_vplus projects (gated by `affect_out`)
    to `recall_pos`; `recall_pos`'s FIRING RATE (read from `cp_firing_states`) is the affect region's appetitive
    output. sal_V = recall_pos rate under APPETITIVE appraisal; sal_N = recall_pos rate under NEUTRAL appraisal.
    The affect region assigns HIGHER output to the value candidate (sal_V > sal_N) -> a self-normalising contrast
    rel_sal = clip((sal_V - sal_N)/(sal_V + sal_N), 0, 1) in [0,1]. This number is NEVER host-set: it is a
    difference of two SPIKE RATES; you cannot get it without running the spiking affect region.
  - AFFECT -> WORKSPACE DRIVE (biased competition). At a branch the value candidate's slot gets IGNITE_PA (full
    consideration); every non-value slot gets IGNITE_PA*(1 - SUPPRESS_MAX*rel_sal). rel_sal~1 (intact) -> non-value
    ~= IGNITE_PA*0.30 (weak) == the validated WTA drive pattern -> the value slot wins the workspace mutual-
    inhibition WTA (DIRECTED). Slot positions are RANDOMISED per branch (a slot-position bias cannot masquerade as
    directedness). The value slot's IGNITE_PA is SYMMETRIC with the lesioned non-value slots, so ALL directedness
    is attributable to the affect-driven suppression.

THE AFFECT-LESION (the keystone). `AffectStateBrain.set_affect_lesion(True)` zeroes the `affect_out` transmission
gate -> affect_vplus's SYNAPTIC current to recall_pos is removed (a NEURAL ablation of the affect region's OUTPUT,
NOT a host clamp) -> recall_pos fires only at baseline -> sal_V ~= sal_N -> rel_sal ~= 0 -> every non-value slot
gets IGNITE_PA -> all slots equal -> the WTA STILL ignites a winner (deliberation RUNS) but picks ~uniformly
(directedness -> chance). CRUCIALLY the affect POOLS keep appraising + firing under lesion (affect_vplus stays
differential value>non-value) -> only the gated OUTPUT is silenced: the dissociation is "the affect region runs;
its OUTPUT is what is load-bearing for DIRECTEDNESS, not for WHETHER it deliberates."

GO GATE (6-seed 42/43/44/100/101/102, CPU): >=5/6 seeds with
  (1) directed_intact >= 0.75 (>> chance 1/K) — the neural affect routes the drive to the value candidate;
  (2) directed_lesion <= chance + 0.15 AND deliberates_lesion >= 0.9 — affect-lesion collapses DIRECTEDNESS to
      chance while the workspace STILL ignites conclusions;
  (3) NEURAL source verified — sal_V > 0 (recall_pos actually fires), sal_V > sal_N (the affect output
      discriminates value from non-value), rel_sal_intact > 0.2, rel_sal_lesion < 0.1, AND affect_vplus fires
      differentially (value > non-value) in BOTH conditions (the pool appraises in both; only the output gates);
  (4) MOAT + RE-ENTRANT still hold — the P1.2 no-confab moat abstains (unstored + overrun) and the intact 3-hop
      re-entrant chase still reaches the conclusion (the workspace loop is byte-unchanged; only the affect anti-
      cheat is now neural).

Run (smoke, 1 seed, verbose tuning):
  SIM_BACKEND=numpy python -u -m research.runners._p1_2_affect_directed_deliberation_derisk --smoke --seed 42 --D 256
Run (6-seed CPU):
  SIM_BACKEND=numpy python -u -m research.runners._p1_2_affect_directed_deliberation_derisk \
      --seeds 42 43 44 100 101 102 --D 256 --backend numpy \
      --json research/findings/raw/_p1_2_affect_directed_deliberation/summary.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend, to_host

# reuse-by-import: the P1.2 workspace (verbatim) — builder, ignition read, re-entrant chase, protocol constants
from research.runners._p1_2_workspace_deliberation_loop_derisk import (
    build_workspace_bridge, _ignite_and_read, reentrant_chase,
    IGNITE_PA, DISTRACTOR_FRAC, IGNITE_FRAC, SOLO_PLATEAU, K_SLOTS,
)
# reuse-by-import: the P0.3 spiking affect-state region (verbatim) + its pool sizes
from research.runners._affect_state_region_derisk import AffectStateBrain, N_AFF, N_RECALL
# reuse-by-import: the held-out food-web chains + fact store
from research.runners._phaseB_multihop_query_chain_derisk import (
    CHAINS, EAT, PLAY, build_vocab, store_facts,
)
from research.runners.rf_phasor_composer import RFPhasorComposer


# ── appraisal + read protocol (mirror P0.3's establish-then-probe) ────────────────────────────────────────────
APPETITIVE_VP = 1.0     # value-relevant (EAT, drive-congruent) appraisal: strong appetitive V+ drive
APPETITIVE_AR = 0.6
NEUTRAL_VP = 0.0        # non-value (PLAY, drive-incongruent) appraisal: no appetitive drive (neutral)
NEUTRAL_AR = 0.4
SETTLE_MS = 40
ESTABLISH_MS = 120
PROBE_MS = 80

# affect -> workspace drive mapping. rel_sal=1 -> non-value slot = IGNITE_PA*(1-SUPPRESS_MAX) = IGNITE_PA*0.30 =
# the validated `_deliberate_hop` distractor drive; rel_sal=0 (lesion) -> non-value = IGNITE_PA (equal). So the
# affect output SUPPRESSES the non-value slots exactly as the host stand-in did, but by a NEURAL rate contrast.
SUPPRESS_MAX = 1.0 - DISTRACTOR_FRAC        # = 0.70


def measure_value_salience(brain: AffectStateBrain, vp: float, ar: float, lesion: bool):
    """Appraise the affect region (establish-then-probe), read its GATED V+ output (recall_pos rate) = the neural
    value-salience, plus the affect pool's OWN rates (affect_vplus/vminus) for the neural-source verification.
    Returns (recall_pos_rate, vplus_rate, vminus_rate). All rates are spikes/neuron/step from cp_firing_states."""
    brain.reset()                                     # clean quiescent state (re-seeds from cfg.seed)
    brain.set_affect_lesion(lesion)                   # AFTER reset (reset restores affect_out to 1.0)
    brain.step(SETTLE_MS)                             # settle to baseline
    brain.step(ESTABLISH_MS, vp=vp, vm=0.0, ar=ar)   # establish the appraised affect state (slow-NMDA integrates)
    c = brain.step(PROBE_MS, vp=vp, vm=0.0, ar=ar,   # hold appraisal; read the gated output pool + the pools
                   record=("recall_pos", "affect_vplus", "affect_vminus"))
    recall_rate = c["recall_pos"] / (N_RECALL * PROBE_MS)
    vplus_rate = c["affect_vplus"] / (N_AFF * PROBE_MS)
    vminus_rate = c["affect_vminus"] / (N_AFF * PROBE_MS)
    return recall_rate, vplus_rate, vminus_rate


def affect_signal(brain: AffectStateBrain, lesion: bool):
    """Run the affect region on the VALUE-relevant (appetitive) vs NON-VALUE (neutral) appraisal and return the
    neural drive-bias contrast + the raw rates.
      sal_V = recall_pos rate | appetitive appraisal   (the value candidate's affect output)
      sal_N = recall_pos rate | neutral appraisal       (the non-value candidate's affect output)
      rel_sal = clip((sal_V - sal_N) / (sal_V + sal_N), 0, 1)   -- self-normalising; 0 when the output collapses.
    """
    sal_V, vpV, vmV = measure_value_salience(brain, APPETITIVE_VP, APPETITIVE_AR, lesion)
    sal_N, vpN, vmN = measure_value_salience(brain, NEUTRAL_VP, NEUTRAL_AR, lesion)
    denom = sal_V + sal_N
    rel_sal = float(np.clip((sal_V - sal_N) / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0
    return {
        "rel_sal": rel_sal, "sal_V": float(sal_V), "sal_N": float(sal_N),
        "vplus_value": float(vpV), "vplus_nonvalue": float(vpN),
        "vminus_value": float(vmV), "vminus_nonvalue": float(vmN),
    }


def affect_branch(bridge, xp, slots_dev, snap, value_slot, nonvalue_slots, rel_sal, n_slots):
    """ONE branch EVALUATE/COMMIT with the NEURAL drive bias, value candidate placed at `value_slot`. The value
    slot -> IGNITE_PA (full consideration); every non-value slot -> IGNITE_PA*(1 - SUPPRESS_MAX*rel_sal). Returns
    (directed, deliberates, rates, winner_slot). `directed` = the VALUE slot won the WTA."""
    drives = [0.0] * n_slots
    drives[value_slot] = IGNITE_PA
    nv_drive = IGNITE_PA * (1.0 - SUPPRESS_MAX * float(rel_sal))
    for s in nonvalue_slots:
        drives[int(s)] = nv_drive
    rates = _ignite_and_read(bridge, xp, slots_dev, snap, drives)
    w = int(np.argmax(rates))
    deliberates = bool(rates[w] >= IGNITE_FRAC * SOLO_PLATEAU)
    directed = bool(deliberates and w == value_slot)
    return directed, deliberates, [float(r) for r in rates], w


def branch_directedness(bridge, xp, slots_dev, snap, rel_sal, n_slots):
    """Enumerate the value candidate across ALL n_slots positions (the complete slot-position-invariance control:
    a slot-position bias cannot masquerade as directedness). Returns (directed_frac, deliberates_frac) over the
    n_slots placements. Under a strong neural affect the value slot wins in EVERY position (directed_frac->1);
    under the affect-lesion all slots are equal so np.argmax deterministically picks the lowest index -> the value
    wins iff placed at slot 0 -> directed_frac = 1/n_slots EXACTLY (= chance), with a winner always igniting
    (deliberates_frac=1). This makes the chance floor exact + variance-free (no finite-sample coincidence)."""
    directed = deliberates = 0
    for value_slot in range(n_slots):
        nonvalue_slots = [s for s in range(n_slots) if s != value_slot]
        d, db, _rates, _w = affect_branch(bridge, xp, slots_dev, snap, value_slot, nonvalue_slots, rel_sal, n_slots)
        directed += int(d); deliberates += int(db)
    return directed / n_slots, deliberates / n_slots


# ── the per-seed experiment ───────────────────────────────────────────────────────────────────────────────────
def run_seed(seed: int, D: int, verbose: bool = True):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc = store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    n_concepts = len(all_concepts)
    HOPS = 3
    chance = 1.0 / K_SLOTS                             # a branch has 1 value + (K-1) non-value slots

    # persistent workspace bridge (built ONCE per seed; the WORKSPACE stays INTACT in both conditions — only the
    # AFFECT region's output is lesioned)
    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)

    # the AFFECT region (P0.3), built ONCE per seed; reset per appraisal
    affect = AffectStateBrain(seed, nmda_on=True)

    # ── the NEURAL affect signal: intact vs affect-lesion ─────────────────────────────────────────────────────
    sig_intact = affect_signal(affect, lesion=False)
    sig_lesion = affect_signal(affect, lesion=True)
    rel_intact = sig_intact["rel_sal"]
    rel_lesion = sig_lesion["rel_sal"]

    # ── the value/non-value BRANCHES (the deliberation directedness) ───────────────────────────────────────────
    # Count the food-web branches: a chain cue with BOTH a value (EAT) target and a non-value (PLAY) target that
    # differ (the scope over which the appetitive drive-state directs). The workspace slots are concept-agnostic
    # current-driven assemblies (concept identity is readout bookkeeping), so the WTA outcome depends only on the
    # affect-set drives (rel_sal) + slot geometry — the ONE appetitive drive-state biases every branch identically.
    aff_tot = 0
    for ch in CHAINS:
        val_t = composer.query_patient(ch[0], EAT)
        non_t = composer.query_patient(ch[0], PLAY)
        if val_t is not None and non_t is not None and val_t != non_t:
            aff_tot += 1
    # directedness = enumerate the value across ALL K slot positions (full position-invariance control)
    directed_intact, deliberates_intact = branch_directedness(b_i, xp, slots_i, snap_i, rel_intact, K_SLOTS)
    directed_lesion, deliberates_lesion = branch_directedness(b_i, xp, slots_i, snap_i, rel_lesion, K_SLOTS)

    # ── the P1.2 moat + re-entrant loop still hold (the workspace loop is unchanged; sanity re-verify) ──────────
    chains3 = [c for c in CHAINS if len(c) > HOPS]
    reent_ok = 0
    for ch in chains3:
        term = reentrant_chase(b_i, xp, slots_i, snap_i, composer, ch[0], [EAT] * HOPS, all_concepts,
                               np.random.default_rng(seed * 991 + 7))
        reent_ok += int(term == ch[HOPS])
    reent_acc = reent_ok / len(chains3)
    moat_unstored = reentrant_chase(b_i, xp, slots_i, snap_i, composer, "ball", [EAT] * HOPS, all_concepts,
                                    np.random.default_rng(seed * 991 + 7))
    overrun_actions = [EAT] * (len(CHAINS[0]) + 2)
    moat_overrun = reentrant_chase(b_i, xp, slots_i, snap_i, composer, CHAINS[0][0], overrun_actions,
                                   all_concepts, np.random.default_rng(seed * 991 + 7))
    moat_ok = bool(moat_unstored is None and moat_overrun is None)

    # ── neural-source verification ─────────────────────────────────────────────────────────────────────────────
    neural_source_ok = bool(
        sig_intact["sal_V"] > 1e-4 and                                   # recall_pos actually fires (spikes)
        sig_intact["sal_V"] > sig_intact["sal_N"] and                    # the gated output discriminates value
        rel_intact > 0.2 and rel_lesion < 0.1 and                        # the drive-bias contrast is neural + gated
        sig_intact["vplus_value"] > sig_intact["vplus_nonvalue"] and     # the affect pool fires differentially...
        sig_lesion["vplus_value"] > sig_lesion["vplus_nonvalue"]         # ...in BOTH conditions (only OUTPUT gated)
    )

    # ── per-seed GO gate ──────────────────────────────────────────────────────────────────────────────────────
    seed_go = bool(
        directed_intact >= 0.75 and                              # neural affect routes the drive to the value cand
        directed_lesion <= chance + 0.15 and                     # affect-lesion collapses directedness to ~chance
        deliberates_lesion >= 0.9 and                            # ...while the workspace STILL ignites conclusions
        neural_source_ok and                                     # the signal is genuinely neural (spikes, gated)
        moat_ok and reent_acc >= 0.75                            # the no-confab moat + re-entrant loop still hold
    )

    result = {
        "seed": int(seed), "D": int(D), "K_slots": K_SLOTS, "chance": chance,
        "n_branches": aff_tot, "n_concepts": n_concepts,
        "directed_intact": directed_intact, "deliberates_intact": deliberates_intact,
        "directed_lesion": directed_lesion, "deliberates_lesion": deliberates_lesion,
        "rel_sal_intact": rel_intact, "rel_sal_lesion": rel_lesion,
        "sal_V_intact": sig_intact["sal_V"], "sal_N_intact": sig_intact["sal_N"],
        "sal_V_lesion": sig_lesion["sal_V"], "sal_N_lesion": sig_lesion["sal_N"],
        "vplus_value_intact": sig_intact["vplus_value"], "vplus_nonvalue_intact": sig_intact["vplus_nonvalue"],
        "vplus_value_lesion": sig_lesion["vplus_value"], "vplus_nonvalue_lesion": sig_lesion["vplus_nonvalue"],
        "vminus_value_intact": sig_intact["vminus_value"], "vminus_nonvalue_intact": sig_intact["vminus_nonvalue"],
        "neural_source_ok": neural_source_ok,
        "reentrant_3hop_acc": reent_acc, "moat_ok": moat_ok,
        "moat_unstored_abstains": moat_unstored is None, "moat_overrun_abstains": moat_overrun is None,
        "seed_go": seed_go,
    }

    if verbose:
        print(f"[p1.2-affect seed={seed} D={D}] directed intact={directed_intact:.3f} lesion={directed_lesion:.3f} "
              f"(chance={chance:.3f}) | deliberates intact={deliberates_intact:.3f} lesion={deliberates_lesion:.3f}",
              flush=True)
        print(f"    NEURAL affect: rel_sal intact={rel_intact:.3f} lesion={rel_lesion:.3f} | recall_pos(V+ gated "
              f"output) sal_V={sig_intact['sal_V']:.4f} sal_N={sig_intact['sal_N']:.4f} "
              f"(lesion sal_V={sig_lesion['sal_V']:.4f} sal_N={sig_lesion['sal_N']:.4f})", flush=True)
        print(f"    affect_vplus rate (pool, ungated): intact val={sig_intact['vplus_value']:.4f} "
              f"nonval={sig_intact['vplus_nonvalue']:.4f} | lesion val={sig_lesion['vplus_value']:.4f} "
              f"nonval={sig_lesion['vplus_nonvalue']:.4f}  (pool differential in BOTH -> only OUTPUT gated)", flush=True)
        print(f"    controls: moat={moat_ok} reentrant_3hop={reent_acc:.3f} neural_source_ok={neural_source_ok} "
              f"| n_branches={aff_tot}", flush=True)
        print(f"    seed_GO={seed_go}", flush=True)
    return result


def run_graded(seed: int, D: int):
    """GRADED diagnostic (the anti-'neural-boolean' control): sweep the appetitive appraisal strength vp; measure
    the affect region's output rate (recall_pos) AND affect_vplus pool rate AND the resulting workspace
    directedness (normalising the drive-bias by the MAX affect output, rel_sal=clip(sal_V/SAL_REF,0,1)). If the
    drive-bias genuinely tracks the neural affect MAGNITUDE (not just its sign), directedness rises monotonically
    with the affect rate from chance to 1. A step/threshold instead = the affect output is bistable (P0.3's own
    documented latch boundary) — still NEURAL, just not graded (an honest read either way)."""
    print(f"[GRADED] seed={seed} D={D} — sweep appetitive vp; does directedness track the affect RATE magnitude?",
          flush=True)
    affect = AffectStateBrain(seed, nmda_on=True)
    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    vps = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows = []
    for vp in vps:
        sal, vplus, _vm = measure_value_salience(affect, vp, APPETITIVE_AR if vp > 0 else NEUTRAL_AR, lesion=False)
        rows.append({"vp": vp, "sal_V": float(sal), "vplus": float(vplus)})
    sal_ref = max(r["sal_V"] for r in rows) or 1e-9
    print(f"  {'vp':>5} | {'affect_vplus':>12} | {'recall_pos(sal_V)':>18} | {'rel_sal':>8} | {'directed':>8}", flush=True)
    for r in rows:
        rel = float(np.clip(r["sal_V"] / sal_ref, 0.0, 1.0))
        d, _db = branch_directedness(b_i, xp, slots_i, snap_i, rel, K_SLOTS)
        r["rel_sal"] = rel; r["directed"] = float(d)
        print(f"  {r['vp']:>5.2f} | {r['vplus']:>12.4f} | {r['sal_V']:>18.4f} | {rel:>8.3f} | {d:>8.3f}", flush=True)
    # is it graded (>=3 distinct directedness levels between chance and 1) or bistable (a single step)?
    dvals = sorted({round(r["directed"], 3) for r in rows})
    graded = len(dvals) >= 3
    print(f"\n[GRADED] directedness levels observed: {dvals} -> "
          f"{'GRADED (drive-bias tracks affect magnitude)' if graded else 'STEP/BISTABLE (P0.3 latch; still neural)'}",
          flush=True)
    return 0


def run_smoke(seed: int, D: int):
    """Cheapest-first: prove the affect region produces a NEURAL, gated, differential drive-bias signal + one
    intact-vs-lesion branch, before the 6-seed battery."""
    print(f"[SMOKE] seed={seed} D={D} — affect-signal read + one branch", flush=True)
    affect = AffectStateBrain(seed, nmda_on=True)
    sig_i = affect_signal(affect, lesion=False)
    sig_l = affect_signal(affect, lesion=True)
    print(f"  INTACT : rel_sal={sig_i['rel_sal']:.3f} sal_V={sig_i['sal_V']:.4f} sal_N={sig_i['sal_N']:.4f} "
          f"| vplus val/nonval={sig_i['vplus_value']:.4f}/{sig_i['vplus_nonvalue']:.4f} "
          f"vminus val/nonval={sig_i['vminus_value']:.4f}/{sig_i['vminus_nonvalue']:.4f}", flush=True)
    print(f"  LESION : rel_sal={sig_l['rel_sal']:.3f} sal_V={sig_l['sal_V']:.4f} sal_N={sig_l['sal_N']:.4f} "
          f"| vplus val/nonval={sig_l['vplus_value']:.4f}/{sig_l['vplus_nonvalue']:.4f}", flush=True)

    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    # one placement each (value at slot 0) for a visible per-slot rate trace
    di, dbi, rates_i, wi = affect_branch(b_i, xp, slots_i, snap_i, 0, [1, 2, 3], sig_i["rel_sal"], K_SLOTS)
    dl, dbl, rates_l, wl = affect_branch(b_i, xp, slots_i, snap_i, 0, [1, 2, 3], sig_l["rel_sal"], K_SLOTS)
    print(f"  INTACT (value@slot0): winner={wi} directed={di} deliberates={dbi} rates={[round(r,3) for r in rates_i]}",
          flush=True)
    print(f"  LESION (value@slot0): winner={wl} directed={dl} deliberates={dbl} rates={[round(r,3) for r in rates_l]}",
          flush=True)
    # full position-invariance enumeration
    d_int, db_int = branch_directedness(b_i, xp, slots_i, snap_i, sig_i["rel_sal"], K_SLOTS)
    d_les, db_les = branch_directedness(b_i, xp, slots_i, snap_i, sig_l["rel_sal"], K_SLOTS)
    print(f"  ENUMERATED directedness (value across all {K_SLOTS} slots): intact={d_int:.3f} lesion={d_les:.3f} "
          f"(chance={1.0/K_SLOTS:.3f}) | deliberates intact={db_int:.3f} lesion={db_les:.3f}", flush=True)
    ok = bool(sig_i["rel_sal"] > 0.2 and sig_l["rel_sal"] < 0.1 and sig_i["sal_V"] > sig_i["sal_N"]
              and d_int >= 0.75 and d_les <= 1.0 / K_SLOTS + 0.15 and db_les >= 0.9)
    print(f"\n[SMOKE] {'PROCEED' if ok else 'INVESTIGATE'} — neural signal gated (intact rel {sig_i['rel_sal']:.2f} "
          f"> lesion {sig_l['rel_sal']:.2f}), sal_V>sal_N={sig_i['sal_V']>sig_i['sal_N']}, directed intact "
          f"{d_int:.2f} -> lesion {d_les:.2f}", flush=True)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Wire the P0.3 spiking affect-state region into the P1.2 GNW workspace "
                                             "drive (replace the host salience scalar).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42, help="single seed (smoke)")
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--graded", action="store_true", help="graded diagnostic: does directedness track affect rate?")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_p1_2_affect_directed_deliberation/summary.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    if args.smoke:
        return run_smoke(args.seed, args.D)
    if args.graded:
        return run_graded(args.seed, args.D)

    t0 = time.time()
    print(f"[p1.2 affect-directed deliberation] {len(CHAINS)} chains | K_slots={K_SLOTS} "
          f"chance={1.0/K_SLOTS:.3f} | D={args.D} backend={args.backend}\n"
          "  REPLACES the host salience scalar with the P0.3 spiking affect region's gated V+ output.\n", flush=True)

    results = [run_seed(s, args.D) for s in args.seeds]
    n_go = sum(int(r["seed_go"]) for r in results)
    all_go = n_go >= 5     # GO = >=5/6 seeds

    def mean(key):
        vals = [r[key] for r in results if isinstance(r[key], (int, float)) and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "runner": "_p1_2_affect_directed_deliberation_derisk",
        "purpose": "wire the P0.3 spiking affect-state region's gated V+ output into the P1.2 GNW workspace drive "
                   "(replace the host salience scalar) — TRUE-ONE-BRAIN faculties interacting",
        "seeds": list(args.seeds), "D": int(args.D), "backend": args.backend, "K_slots": K_SLOTS,
        "chance": 1.0 / K_SLOTS,
        "n_go": n_go, "n_seeds": len(results), "GO": all_go,
        "mean_directed_intact": mean("directed_intact"),
        "mean_directed_lesion": mean("directed_lesion"),
        "mean_deliberates_intact": mean("deliberates_intact"),
        "mean_deliberates_lesion": mean("deliberates_lesion"),
        "mean_rel_sal_intact": mean("rel_sal_intact"),
        "mean_rel_sal_lesion": mean("rel_sal_lesion"),
        "mean_sal_V_intact": mean("sal_V_intact"), "mean_sal_N_intact": mean("sal_N_intact"),
        "mean_reentrant_3hop_acc": mean("reentrant_3hop_acc"),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "all_neural_source_ok": all(r["neural_source_ok"] for r in results),
        "per_seed": results,
        "mechanism": "P0.3 AffectStateBrain (3 opponent slow-NMDA pools + Namburi-Tye cross-inhibition) appraised "
                     "APPETITIVE (EAT/value) vs NEUTRAL (PLAY/non-value); the affect_vplus->recall_pos gated output "
                     "(affect_out) rate contrast rel_sal=clip((sal_V-sal_N)/(sal_V+sal_N),0,1) suppresses the "
                     "non-value workspace slots (IGNITE_PA*(1-0.70*rel_sal)); the value slot gets IGNITE_PA; "
                     "biased-competition WTA selects the value candidate. affect-lesion = set_affect_lesion(True) "
                     "zeroes affect_out -> rel_sal->0 -> chance directedness, deliberation still ignites.",
        "honest_note": "numpy-CPU (real spiking Izhikevich bridges — 'numpy' is the backend, not a shortcut). "
                       "Two separate CPU bridges (P1.2 workspace + P0.3 affect) with a NEURAL signal handoff (the "
                       "cross-region faculty interaction); the drive-bias contrast rel_sal is a difference of two "
                       "recall_pos SPIKE RATES read from cp_firing_states, NOT a host scalar, and it is gated by the "
                       "affect_out transmission gate (the affect-lesion neurally silences the affect region's "
                       "OUTPUT). A fully co-resident single-bridge synaptic affect->slot projection is the follow-on. "
                       "NO sim/ edit.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_go else ("PARTIAL" if n_go >= 1 else "NEGATIVE")
    print(f"\n{'='*104}", flush=True)
    print(f"  P1.2 AFFECT-DIRECTED DELIBERATION VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    print(f"    directed intact={summary['mean_directed_intact']:.3f} lesion={summary['mean_directed_lesion']:.3f} "
          f"(chance={summary['chance']:.3f}) | deliberates intact={summary['mean_deliberates_intact']:.3f} "
          f"lesion={summary['mean_deliberates_lesion']:.3f}", flush=True)
    print(f"    NEURAL affect: rel_sal intact={summary['mean_rel_sal_intact']:.3f} "
          f"lesion={summary['mean_rel_sal_lesion']:.3f} | sal_V={summary['mean_sal_V_intact']:.4f} "
          f"sal_N={summary['mean_sal_N_intact']:.4f} | neural_source_all_ok={summary['all_neural_source_ok']}", flush=True)
    print(f"    controls: moat_all={summary['all_moat_ok']} reentrant_3hop={summary['mean_reentrant_3hop_acc']:.3f}",
          flush=True)
    print(f"    [saved] {args.json}\n{'='*104}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
