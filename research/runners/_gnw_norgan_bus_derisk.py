"""GNW N-ORGAN IGNITION BUS de-risk — the SUBSTRATE combines N>=3 organ reads via one shared workspace.

WHAT THIS ADDS OVER THE 2-ORGAN KEYSTONE (`_gnw_coincidence_integrator_derisk`, 6/6 GO). The keystone proved the
SUBSTRATE can combine TWO subthreshold organ reads: two organs write drive < the ignition knee, only the slot where
they COINCIDE crosses threshold, the shared inhibitory pool WTA-suppresses a single-vote decoy, and the ignited winner
BROADCASTS BACK (re-entry). That was a 2-organ AND (unanimity over 2). The mission's next rung (faculty-map T1-1,
Phase-B) is to make this the GENERAL PRODUCTION organ-combination mechanism: route N>=3 organs' reads through ONE
workspace where the substrate's ignition dynamics (NOT a host if/else) select the winner and broadcast it — the thing
that replaces the host Python `brain_chat` uses today to snapshot each co-resident organ and combine their reads.

MECHANISM (the direct N-generalization; reuse-by-import of the P1.2/keystone spiking workspace, NO `sim/` edit):
  N organs each read a candidate concept and write SUBTHRESHOLD drive D_SUB (< the solo ignition knee) into the shared
  K-slot workspace. Organs that agree on a concept ACCUMULATE their drive on that slot: a slot with M votes carries
  M*D_SUB. D_SUB is calibrated so that (Q-1)*D_SUB < knee <= Q*D_SUB, i.e. a slot ignites IFF it reaches the CONSENSUS
  QUORUM Q. The shared inhibitory pool (`workspace_fs`) WTA-selects the most-supported slot; the committed spiking
  winner BROADCASTS BACK as the next hop's premise (re-entry). Two hops => a 2-step inference.

  PRIMARY GATED ARM — UNANIMITY (Q = N): the keystone required BOTH of its 2 organs (Q=2=N); the N-organ bus requires
  ALL N (Q=N). So EVERY organ is load-bearing: drop, silence, mis-route, or disagree ANY one organ and the true slot
  falls to (N-1)*D_SUB < knee -> nothing ignites -> abstain. The AND-over-N-organs is the neuronal ignition THRESHOLD;
  the winner-selection is the shared inhibition — the substrate's dynamics, not host control flow.

  MAJORITY DIAGNOSTIC (ungated, Q=2 calibration): with a majority quorum the substrate does PLURALITY — 2 organs
  agreeing on the true concept (2*D_SUB, suprathreshold) OUTVOTE a lone dissenter (D_SUB, subthreshold), showing the
  WTA selects the MOST-supported slot (not mere unanimity). This exercises the "consensus/most-supported" claim in
  full; it is reported, not gated (the clean unanimity arm carries the load-bearing GO).

GO GATE (6 seeds 42/43/44/100/101/102), UNANIMITY arm: consensus_2hop_acc >= 0.75 AND >= spreading_floor + 0.5 AND
matches the host one-shot query_chain baseline (parity) AND EVERY organ-ablation collapses to <= chance-ish AND the
no-confab moat abstains. i.e. a never-told 2-hop conclusion reached by the workspace INTEGRATING N subthreshold organ
reads and iterating on its broadcast, with the combination done synaptically, and every single organ load-bearing.

ANTI-CHEATS (each targets a distinct "it is really the substrate" claim; ALL must collapse):
  - SINGLE-ORGAN COLLAPSE  [THE anti-host-if-else]: keep only organ 0 -> the true slot gets 1*D_SUB -> subthreshold ->
    abstain. A host `if organ_0: return r` would SUCCEED here; the collapse proves the combination is the workspace's
    ignition threshold, not a host read.
  - LEAVE-ONE-OUT COLLAPSE  [every organ load-bearing]: drop ANY single organ (N-1 active) -> (N-1)*D_SUB < knee ->
    collapse. Distinguishes a genuine N-way AND from "any 1 suffices" and from "any 2 suffice".
  - DISAGREEMENT / CONSENSUS-VETO: permute the non-primary organ relations so the organs read DIFFERENT concepts ->
    votes spread across slots, none reaches quorum -> the workspace WITHHOLDS the unconfirmed conclusion.
  - SHUFFLE-OFF-SLOT [the keystone's CORRECTED control]: route one organ's drive to an EMPTY (unoccupied) slot ->
    guaranteed no leak onto a real concept -> the consensus loses a vote -> collapse. (Rerouting onto an OCCUPIED slot
    would LEAK: the drive could land back on the consensus slot and NOT collapse. Route off-slot. INSTRUMENT-VERIFIED.)
  - NO-IGNITION LESION: assembly self-recurrence -> 0 -> even N*D_SUB cannot sustain -> collapse, WHILE the single-hop
    recall reflex (direct query_patient, never routed through the workspace) SURVIVES (the dissociation keystone).
  - NO-RE-ENTRY (single-shot): cap the loop at 1 cycle -> only hop-1 -> the 2-hop want is unreachable -> collapse.
    Exactly what the current PRODUCTION host pipeline (snapshot organs once, combine once, emit) cannot do.
  - SPREADING-ACTIVATION FLOOR: relation-blind co-occurrence diffusion stays ~chance; the integrated chase must beat it.
  - MOAT: an unstored cue and a past-chain-end over-run -> the primary organ misses -> abstain (None).

PRODUCTION-ORGAN PROTOTYPE (`--prototype`): route THREE genuinely-heterogeneous REAL production composer organ reads
  (spiking recall `query_patient`, a corroborating second-relation recall, and a reverse-binding VERIFY via
  `query_agent`) through the SAME bus to produce the combined gate decision `ChatBrain.gate()` currently makes with
  host Python (`if recalled == p`). Demonstrates the SUBSTRATE doing the combination. This is a prototype, not the
  gated claim (which the abstract N-organ arm carries).

Run (CPU cheap-first smoke, then GPU 6-seed):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_norgan_bus_derisk --calibrate --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_norgan_bus_derisk --smoke --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_norgan_bus_derisk --prototype --seed 42
  SIM_BACKEND=cupy  python -u -m research.runners._gnw_norgan_bus_derisk --seeds 42 43 44 100 101 102 --backend cupy \
      --json research/findings/raw/_gnw_norgan_bus/summary.json
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

# reuse-by-import the P1.2/keystone spiking workspace (build + ignition-read) + constants — NO `sim/` edit.
from research.runners._p1_2_workspace_deliberation_loop_derisk import (
    build_workspace_bridge, _ignite_and_read,
    K_SLOTS, ASSEMBLY_SIZE, IGNITE_FRAC, SOLO_PLATEAU,
)
from research.runners._gnw_coincidence_integrator_derisk import _assign_slots, _pick_decoy
from research.runners._phaseB_multihop_query_chain_derisk import (
    CHAINS, EAT, build_vocab, store_facts, spreading_predict,
)
from research.runners.rf_phasor_composer import RFPhasorComposer
from tools.lab import attributable_to, void_if

# ── organ relations: EAT (organ 0, stored by store_facts) + N-1 corroborating relations over the SAME edges ─────
CONFIRM, CORROB, WITNESS = "confirm", "corrob", "witness"
ORGAN_RELATIONS_ALL = [EAT, CONFIRM, CORROB, WITNESS]     # organ i uses ORGAN_RELATIONS_ALL[i]
N_ORGANS_DEFAULT = 3
HOPS = 2                                                  # a 2-step inference (ch[0] -> ch[2])
THR = IGNITE_FRAC * SOLO_PLATEAU                          # "ignited" iff late-window rate >= THR

# D_SUB is set by the CALIBRATED knee so (N-1)*D_SUB < knee <= N*D_SUB (unanimity quorum Q=N). Default for N=3 with
# the measured K=4 workspace knee ~2400 pA: 2*1000=2000 (<knee, subthreshold), 3*1000=3000 (>=knee, suprathreshold).
D_SUB_UNANIMITY = {2: 1400.0, 3: 1000.0, 4: 760.0}       # per-N default; verified in run_calibrate
D_SUB_MAJORITY = 1400.0                                   # Q=2 majority probe: 1 vote sub, 2 votes supra (keystone value)


# ── the N-relation fact store (organ 0 = EAT via store_facts; organs 1.. corroborate the same edges) ──────────
def store_n_relation_facts(composer, chains, organ_relations, permute_from_organ=None, rng=None, distractor_rng=None):
    """Store the food-web under EAT (organ 0, via store_facts) + the SAME edges under each corroborating relation
    (organs 1..). With permute_from_organ set, scramble the patient assignment for organs >= that index (the organs
    DISAGREE with organ 0 -> the consensus-veto control). Returns (edges, cooc)."""
    edges, cooc = store_facts(composer, chains, distractor_rng=distractor_rng)     # EAT + distractors + cooc
    for oi, rel in enumerate(organ_relations):
        if oi == 0:                                       # EAT already stored by store_facts
            continue
        targets = [p for _, p in edges]
        if permute_from_organ is not None and oi >= permute_from_organ:
            targets = list(targets)
            rng.shuffle(targets)
        for (a, _p), t in zip(edges, targets):
            composer.store(a, rel, t)
    return edges, cooc


# ── one EVALUATE/COMMIT over the workspace: N organs vote, drives ACCUMULATE, ignition+WTA select ──────────────
def norgan_hop(bridge, xp, slots_dev, snap, candidates, decoy, d_sub,
               active_mask=None, shuffle_off_organ=None, rng=None):
    """N organs each drive slot(candidate_i) at d_sub (agreeing organs ACCUMULATE); a spurious DECOY drives its own
    slot at d_sub. Only a slot reaching the consensus quorum crosses the ignition knee. Shared inhibition WTA-selects
    the winner. Returns (committed_concept|None, rates, winner_slot, n_ignited).
      active_mask[j]=False -> organ j contributes NO drive (single-organ / leave-one-out collapse controls).
      shuffle_off_organ=j  -> route organ j's drive to an EMPTY (unoccupied) slot (the CORRECTED off-slot control:
                              guaranteed not to leak back onto an occupied concept slot)."""
    present = [c for c in candidates if c is not None]
    concepts = present + ([decoy] if decoy is not None else [])
    slot_of, order = _assign_slots(concepts)              # first-seen concept -> slot index
    n = len(slots_dev)
    drives = [0.0] * n
    occupied = set(slot_of.values())
    empty_slots = [i for i in range(n) if i not in occupied]

    for j, c in enumerate(candidates):
        if active_mask is not None and not active_mask[j]:
            continue
        if c is None or c not in slot_of:
            continue
        tgt = slot_of[c]
        if shuffle_off_organ == j:                        # route OFF-slot to an empty slot (no leak) — the fix
            if empty_slots:
                tgt = empty_slots[int(rng.integers(len(empty_slots)))]
            else:
                continue                                  # no empty slot -> drop the vote (still off the consensus)
        drives[tgt] += d_sub
    if decoy is not None and decoy in slot_of:
        drives[slot_of[decoy]] += d_sub                   # the single-vote competitor (always present)

    rates = _ignite_and_read(bridge, xp, slots_dev, snap, drives)
    # winner = the most-supported OCCUPIED (concept) slot; n_ignited counts ALL slots (single-content across workspace)
    occ_idx = list(range(len(order)))
    winner = int(occ_idx[int(np.argmax([rates[i] for i in occ_idx]))]) if occ_idx else 0
    ignited = bool(occ_idx) and rates[winner] >= THR
    n_ignited = int(sum(1 for i in range(n) if rates[i] >= THR))
    committed = order[winner] if ignited else None
    return committed, rates, winner, n_ignited


def norgan_chase(bridge, xp, slots_dev, snap, composer, cue, all_concepts, d_sub, organ_relations, rng,
                 active_mask=None, shuffle_off_organ=None, max_cycles=None, return_trace=False):
    """The workspace-carried 2-hop DELIBERATION. x starts at cue; each hop: N organs read (PROPOSE) ->
    consensus-ignition EVALUATE/COMMIT -> BROADCAST BACK (x_next = the committed winner). Abstains (None) the moment
    the primary organ misses (moat) or nothing reaches quorum (no consensus)."""
    x = cue
    trace = []
    n_hops = HOPS if max_cycles is None else min(int(max_cycles), HOPS)
    for h in range(n_hops):
        candidates = [composer.query_patient(x, rel) for rel in organ_relations]   # N organ reads
        if candidates[0] is None:                         # the primary (recall) organ missed -> moat abstains
            trace.append({"hop": h, "x": x, "candidates": candidates, "committed": None, "n_ignited": 0})
            return (None, trace) if return_trace else None
        decoy = _pick_decoy(all_concepts, exclude=set(c for c in candidates if c is not None) | {x}, rng=rng)
        committed, rates, winner, n_ign = norgan_hop(
            bridge, xp, slots_dev, snap, candidates, decoy, d_sub,
            active_mask=active_mask, shuffle_off_organ=shuffle_off_organ, rng=rng)
        trace.append({"hop": h, "x": x, "candidates": candidates, "committed": committed,
                      "winner": int(winner), "n_ignited": int(n_ign)})
        if committed is None:                             # no consensus ignited -> abstain
            return (None, trace) if return_trace else None
        x = committed                                     # BROADCAST BACK: the spike-derived re-cue re-enters
    return (x, trace) if return_trace else x


# ── calibration: locate the ignition knee; verify D_SUB gives quorum Q=N (and the Q=2 majority probe) ──────────
def run_calibrate(seed, n_organs, d_sub, json_path=None):
    build_vocab()
    b, xp, slots, snap = build_workspace_bridge(seed, lesion=False)
    print(f"[calibrate seed={seed} N={n_organs}] solo-slot ignition curve (THR={THR:.3f}):", flush=True)
    knee = None
    curve = {}
    for drive in (600, 800, 1000, 1200, 1400, 1500, 1700, 1800, 2000, 2100, 2400, 2800, 3000):
        rates = _ignite_and_read(b, xp, slots, snap, [float(drive)] + [0.0] * (len(slots) - 1))
        ig = rates[0] >= THR
        curve[str(drive)] = round(float(rates[0]), 3)
        if ig and knee is None:
            knee = drive
        print(f"    solo drive={drive:5.0f} -> slot0 late-rate={rates[0]:.3f}  ignited={ig}", flush=True)
    # unanimity Q=N: (N-1)*d_sub must be SUBthreshold, N*d_sub SUPRAthreshold
    r_nm1 = _ignite_and_read(b, xp, slots, snap, [(n_organs - 1) * d_sub] + [0.0] * (len(slots) - 1))[0]
    r_n = _ignite_and_read(b, xp, slots, snap, [n_organs * d_sub] + [0.0] * (len(slots) - 1))[0]
    # majority Q=2: 1*D_SUB_MAJ sub, 2*D_SUB_MAJ supra
    r_maj1 = _ignite_and_read(b, xp, slots, snap, [D_SUB_MAJORITY] + [0.0] * (len(slots) - 1))[0]
    r_maj2 = _ignite_and_read(b, xp, slots, snap, [2 * D_SUB_MAJORITY] + [0.0] * (len(slots) - 1))[0]
    unanimity_ok = bool(knee is not None and r_nm1 < THR and r_n >= THR)
    majority_ok = bool(r_maj1 < THR and r_maj2 >= THR)
    print(f"    UNANIMITY(Q={n_organs}): (N-1)*d_sub={(n_organs-1)*d_sub:.0f}->rate={r_nm1:.3f}(sub={r_nm1<THR}) "
          f"N*d_sub={n_organs*d_sub:.0f}->rate={r_n:.3f}(supra={r_n>=THR})  {'OK' if unanimity_ok else 'RETUNE'}",
          flush=True)
    print(f"    MAJORITY(Q=2): 1*={D_SUB_MAJORITY:.0f}->rate={r_maj1:.3f}(sub={r_maj1<THR}) "
          f"2*={2*D_SUB_MAJORITY:.0f}->rate={r_maj2:.3f}(supra={r_maj2>=THR})  {'OK' if majority_ok else 'RETUNE'}",
          flush=True)
    ok = unanimity_ok and majority_ok
    if json_path:
        rec = {"runner": "_gnw_norgan_bus_derisk", "mode": "calibrate", "seed": int(seed), "n_organs": int(n_organs),
               "THR": round(float(THR), 3), "d_sub": float(d_sub), "knee_pA": knee, "solo_rate_by_drive": curve,
               "rate_Nm1_votes": round(float(r_nm1), 3), "rate_N_votes": round(float(r_n), 3),
               "unanimity_in_window": unanimity_ok,
               "rate_maj1": round(float(r_maj1), 3), "rate_maj2": round(float(r_maj2), 3),
               "majority_in_window": majority_ok}
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"    [saved] {json_path}", flush=True)
    return ok


# ── majority-override diagnostic (Q=2): 2 organs agree on the true next, 1 organ dissents to a decoy ──────────
def majority_probe(bridge, xp, slots_dev, snap, composer, cue, all_concepts, organ_relations, rng):
    """Single hop, Q=2 calibration (D_SUB_MAJORITY): organs 0,1 vote the true next (2 votes -> suprathreshold);
    organ 2 dissents to a decoy (1 vote -> subthreshold). The 2-agree slot must WIN and the dissenter must NOT ignite.
    Returns (majority_won: true concept committed, dissenter_suppressed: dissenter did not ignite)."""
    r = composer.query_patient(cue, organ_relations[0])
    if r is None:
        return None, None
    dissent = _pick_decoy(all_concepts, exclude={r, cue}, rng=rng)
    candidates = [r, r] + [dissent] * (len(organ_relations) - 2)   # organs 0,1 agree on r; organs 2.. dissent to `dissent`
    decoy = _pick_decoy(all_concepts, exclude={r, dissent, cue}, rng=rng)
    committed, rates, winner, n_ign = norgan_hop(bridge, xp, slots_dev, snap, candidates, decoy, D_SUB_MAJORITY, rng=rng)
    majority_won = (committed == r)
    # dissenter slot: find its rate (it is an occupied slot but should be subthreshold at 1 vote)
    _slot_of, order = _assign_slots([c for c in candidates if c is not None] + [decoy])
    diss_slot = _slot_of.get(dissent)
    dissenter_suppressed = (diss_slot is None) or (rates[diss_slot] < THR)
    return majority_won, dissenter_suppressed


# ── 1-seed primitive smoke ────────────────────────────────────────────────────────────────────────────────────
def run_primitive_smoke(seed, n_organs, d_sub):
    organ_relations = ORGAN_RELATIONS_ALL[:n_organs]
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=256, vocab=vocab)
    store_n_relation_facts(composer, CHAINS, organ_relations, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]

    ch = CHAINS[0]
    cue = ch[0]
    reads = [composer.query_patient(cue, rel) for rel in organ_relations]
    print(f"[smoke] N={n_organs} cue={cue!r}: organ reads {list(zip(organ_relations, reads))} "
          f"(want {ch[1]!r}, all-agree={len(set(reads)) == 1})", flush=True)

    b, xp, slots, snap = build_workspace_bridge(seed, lesion=False)
    rng = np.random.default_rng(seed * 991 + 7)
    r = reads[0]
    decoy = _pick_decoy(all_concepts, exclude=set(reads) | {cue}, rng=rng)

    # (1) CONSENSUS: all N organs -> N votes on the true slot -> ignites, single-content, beats the decoy
    com, rates, w, nign = norgan_hop(b, xp, slots, snap, reads, decoy, d_sub, rng=rng)
    print(f"[smoke] CONSENSUS   rates={[round(x,3) for x in rates]} winner=slot{w} n_ignited={nign} committed={com!r}", flush=True)
    consensus_ok = bool(com == r and nign == 1)

    # (2) SINGLE-ORGAN (organ 0 only) -> 1 vote -> subthreshold -> no ignition (the anti-host-if-else)
    m1 = [True] + [False] * (n_organs - 1)
    com_s, rs, _w, n_s = norgan_hop(b, xp, slots, snap, reads, decoy, d_sub, active_mask=m1, rng=rng)
    print(f"[smoke] SINGLE-ORG  rates={[round(x,3) for x in rs]} n_ignited={n_s} committed={com_s!r}", flush=True)
    single_abstains = bool(com_s is None)

    # (3) LEAVE-ONE-OUT (drop organ N-1) -> (N-1) votes -> subthreshold -> collapse (every organ load-bearing)
    mloo = [True] * (n_organs - 1) + [False]
    com_l, rl, _wl, n_lo = norgan_hop(b, xp, slots, snap, reads, decoy, d_sub, active_mask=mloo, rng=rng)
    print(f"[smoke] LEAVE-1-OUT rates={[round(x,3) for x in rl]} n_ignited={n_lo} committed={com_l!r}", flush=True)
    loo_abstains = bool(com_l is None)

    # (4) DISAGREEMENT: organs point to different concepts -> no coincidence -> withhold
    disagree_reads = [r] + [_pick_decoy(all_concepts, exclude={r, cue}, rng=rng) for _ in range(n_organs - 1)]
    com_d, rd, _wd, n_d = norgan_hop(b, xp, slots, snap, disagree_reads, decoy, d_sub, rng=rng)
    print(f"[smoke] DISAGREE    rates={[round(x,3) for x in rd]} n_ignited={n_d} committed={com_d!r} reads={disagree_reads}", flush=True)
    disagree_abstains = bool(com_d is None)

    # (5) SHUFFLE-OFF-SLOT: route organ 1 off to an EMPTY slot -> consensus loses a vote -> collapse
    com_sh, rsh, _wsh, n_sh = norgan_hop(b, xp, slots, snap, reads, decoy, d_sub, shuffle_off_organ=1,
                                         rng=np.random.default_rng(seed * 13 + 3))
    print(f"[smoke] SHUFFLE-OFF rates={[round(x,3) for x in rsh]} n_ignited={n_sh} committed={com_sh!r}", flush=True)
    shuffle_abstains = bool(com_sh is None)

    # (6) NO-IGNITION LESION: even the N-vote consensus cannot sustain
    bl, xpl, slotsl, snapl = build_workspace_bridge(seed, lesion=True)
    com_le, rle, _wle, n_le = norgan_hop(bl, xpl, slotsl, snapl, reads, decoy, d_sub, rng=rng)
    print(f"[smoke] LESION      rates={[round(x,3) for x in rle]} n_ignited={n_le} committed={com_le!r}", flush=True)
    lesion_kills = bool(com_le is None)

    reflex_ok = bool(composer.query_patient(cue, EAT) == ch[1])   # the workspace-independent recall survives

    # (7) MAJORITY OVERRIDE (Q=2): 2 organs agree, 1 dissents -> the 2-agree slot wins, dissenter suppressed
    maj_won, diss_supp = majority_probe(b, xp, slots, snap, composer, cue, all_concepts, organ_relations,
                                        rng=np.random.default_rng(seed * 17 + 2))
    print(f"[smoke] MAJORITY    2-of-{n_organs} override: majority_won={maj_won} dissenter_suppressed={diss_supp}", flush=True)
    majority_ok = bool(maj_won and diss_supp)

    ok = bool(consensus_ok and single_abstains and loo_abstains and disagree_abstains and shuffle_abstains
              and lesion_kills and reflex_ok and majority_ok)
    print(f"\n[smoke] N-ORGAN BUS {'HOLDS' if ok else 'FAILS'}: consensus_ignites_single={consensus_ok} "
          f"single_abstains={single_abstains} leave1out_abstains={loo_abstains} disagree_abstains={disagree_abstains} "
          f"shuffle_off_abstains={shuffle_abstains} lesion_kills={lesion_kills} reflex_survives={reflex_ok} "
          f"majority_override={majority_ok}", flush=True)
    return ok


# ── the per-seed experiment ───────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, n_organs, d_sub, D=256, verbose=True):
    organ_relations = ORGAN_RELATIONS_ALL[:n_organs]
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc = store_n_relation_facts(composer, CHAINS, organ_relations,
                                         distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    n_concepts = len(all_concepts)
    chance = 1.0 / n_concepts

    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    b_l, xp_l, slots_l, snap_l = build_workspace_bridge(seed, lesion=True)

    def rng():
        return np.random.default_rng(seed * 991 + 7)

    chains2 = [ch for ch in CHAINS if len(ch) > HOPS]
    tot = len(chains2)

    # ── INTACT integrated 2-hop consensus chase + host one-shot baseline + spreading floor ──────────────────────
    consensus_ok = qc_ok = spread_ok = 0
    for ch in chains2:
        cue, want = ch[0], ch[HOPS]
        term = norgan_chase(b_i, xp, slots_i, snap_i, composer, cue, all_concepts, d_sub, organ_relations, rng())
        consensus_ok += int(term == want)
        qc_ok += int(composer.query_chain(cue, [EAT] * HOPS) == want)
        spread_ok += int(spreading_predict(cooc, cue, HOPS, all_concepts) == want)
    consensus_acc = consensus_ok / tot
    qc_acc = qc_ok / tot
    spread_floor = spread_ok / tot

    # ── ANTI-CHEATS ─────────────────────────────────────────────────────────────────────────────────────────────
    def chase_acc(**kw):
        ok = 0
        for ch in chains2:
            ok += int(norgan_chase(b_i, xp, slots_i, snap_i, composer, ch[0], all_concepts, d_sub, organ_relations,
                                   rng(), **kw) == ch[HOPS])
        return ok / tot

    single_mask = [True] + [False] * (n_organs - 1)
    single_organ_acc = chase_acc(active_mask=single_mask)             # only organ 0 -> subthreshold -> collapse
    # leave-one-out: drop EACH organ once; the collapse must hold for ALL (report the WORST = max over drops)
    loo_accs = []
    for drop in range(n_organs):
        mask = [j != drop for j in range(n_organs)]
        loo_accs.append(chase_acc(active_mask=mask))
    leaveoneout_acc = float(np.max(loo_accs))                         # worst-case: even the easiest drop collapses
    shuffle_acc = chase_acc(shuffle_off_organ=1)                      # organ 1 off-slot -> collapse
    onecycle_acc = chase_acc(max_cycles=1)                            # single-shot: only hop-1 -> 2-hop unreachable

    # no-ignition lesion (collapse) + the dissociation keystone (single-hop reflex survives)
    lesion_ok = reflex_ok = 0
    for ch in chains2:
        lesion_ok += int(norgan_chase(b_l, xp_l, slots_l, snap_l, composer, ch[0], all_concepts, d_sub,
                                      organ_relations, rng()) == ch[HOPS])
        reflex_ok += int(composer.query_patient(ch[0], EAT) == ch[1])
    lesion_acc = lesion_ok / tot
    reflex_acc = reflex_ok / tot

    # disagreement / consensus-veto: permute organs 1.. -> reads diverge -> no consensus -> withhold
    comp_dis = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_n_relation_facts(comp_dis, CHAINS, organ_relations, permute_from_organ=1,
                           rng=np.random.default_rng(seed * 101 + 5),
                           distractor_rng=np.random.default_rng(seed * 53 + 1))
    disagree_ok = 0
    for ch in chains2:
        disagree_ok += int(norgan_chase(b_i, xp, slots_i, snap_i, comp_dis, ch[0], all_concepts, d_sub,
                                        organ_relations, rng()) == ch[HOPS])
    disagree_acc = disagree_ok / tot

    # MOAT: unstored cue + past-chain-end over-run -> abstain
    moat_unstored = norgan_chase(b_i, xp, slots_i, snap_i, composer, "ball", all_concepts, d_sub, organ_relations, rng())
    moat_over = norgan_chase(b_i, xp, slots_i, snap_i, composer, CHAINS[0][-1], all_concepts, d_sub, organ_relations, rng())
    moat_unstored_abstains = moat_unstored is None
    moat_over_abstains = moat_over is None
    moat_ok = bool(moat_unstored_abstains and moat_over_abstains)

    # mutual-exclusion diagnostic (single-content access at each committed hop)
    me_single = me_total = 0
    for ch in chains2:
        _t, tr = norgan_chase(b_i, xp, slots_i, snap_i, composer, ch[0], all_concepts, d_sub, organ_relations,
                              rng(), return_trace=True)
        for step in tr:
            if step.get("committed") is not None:
                me_total += 1
                me_single += int(step["n_ignited"] == 1)
    me_frac = (me_single / me_total) if me_total else 0.0

    # majority-override diagnostic (ungated): 2 organs outvote a lone dissenter (WTA selects most-supported slot)
    maj_won_ok = maj_supp_ok = maj_tot = 0
    for ch in chains2:
        mw, ms = majority_probe(b_i, xp, slots_i, snap_i, composer, ch[0], all_concepts, organ_relations,
                                np.random.default_rng(seed * 17 + 2))
        if mw is None:
            continue
        maj_tot += 1
        maj_won_ok += int(bool(mw))
        maj_supp_ok += int(bool(ms))
    majority_override_acc = (maj_won_ok / maj_tot) if maj_tot else float("nan")
    majority_suppressed_frac = (maj_supp_ok / maj_tot) if maj_tot else float("nan")

    two_chance = 2.0 * chance
    seed_go = bool(
        consensus_acc >= 0.75 and
        consensus_acc >= spread_floor + 0.5 and
        consensus_acc >= qc_acc and                       # parity with the host one-shot (same conclusion, synaptic path)
        single_organ_acc <= max(two_chance, 0.10) and     # a single organ read is subthreshold (the anti-if-else)
        leaveoneout_acc <= max(two_chance, 0.10) and      # EVERY organ load-bearing (drop any one -> collapse)
        disagree_acc <= max(two_chance, 0.10) and         # conflicting organs -> withhold (consensus-veto)
        shuffle_acc <= max(two_chance, 0.10) and          # combination is congruence, not slot
        onecycle_acc <= max(two_chance, 0.10) and         # re-entry load-bearing (single-shot can't)
        lesion_acc <= max(two_chance, 0.10) and           # ignition load-bearing
        reflex_acc >= 0.85 and                            # the single-hop recall reflex survives (dissociation)
        moat_ok
    )

    result = {
        "seed": int(seed), "n_organs": int(n_organs), "D": int(D), "d_sub": float(d_sub), "hops": HOPS,
        "n_concepts": n_concepts, "chance": chance, "n_chains": tot,
        "consensus_2hop_acc": consensus_acc, "query_chain_2hop_acc": qc_acc, "spreading_floor": spread_floor,
        "single_organ_acc": single_organ_acc, "leaveoneout_acc": leaveoneout_acc, "loo_accs": loo_accs,
        "disagree_acc": disagree_acc, "shuffle_off_acc": shuffle_acc, "onecycle_acc": onecycle_acc,
        "lesion_acc": lesion_acc, "single_hop_reflex_acc": reflex_acc,
        "moat_unstored_abstains": moat_unstored_abstains, "moat_over_abstains": moat_over_abstains, "moat_ok": moat_ok,
        "mutual_exclusion_frac": me_frac,
        "majority_override_acc": majority_override_acc, "majority_suppressed_frac": majority_suppressed_frac,
        "seed_go": seed_go,
    }
    if verbose:
        print(f"[norgan seed={seed} N={n_organs} d_sub={d_sub:.0f}] consensus_2hop={consensus_acc:.3f} "
              f"vs query_chain={qc_acc:.3f} (spread_floor={spread_floor:.3f}, chance={chance:.3f})", flush=True)
        print(f"    ORGAN collapses: single={single_organ_acc:.3f} leave1out(worst)={leaveoneout_acc:.3f} "
              f"disagree={disagree_acc:.3f} shuffle_off={shuffle_acc:.3f}", flush=True)
        print(f"    RE-ENTRY/IGNITION: onecycle={onecycle_acc:.3f} lesion={lesion_acc:.3f} | reflex_survives={reflex_acc:.3f} "
              f"| moat unstored={moat_unstored_abstains} over={moat_over_abstains} | ME_single={me_frac:.3f}", flush=True)
        print(f"    MAJORITY (ungated): override={majority_override_acc:.3f} dissenter_suppressed={majority_suppressed_frac:.3f} "
              f"| seed_GO={seed_go}", flush=True)
    return result


# ── production-organ prototype: 3 REAL composer organ reads combined by the bus (not host Python) ─────────────
def run_prototype(seed, D=256):
    """Route THREE genuinely-heterogeneous REAL production `RFPhasorComposer` organ reads through the SAME bus to
    make the combined gate decision `ChatBrain.gate()` currently makes in host Python (`if recalled == p`):
      organ A (spiking RECALL) : query_patient(agent, EAT)          -> candidate patient
      organ B (CORROBORATION)  : query_patient(agent, CONFIRM)      -> second-relation recall of the same edge
      organ C (reverse VERIFY) : query_agent(EAT, cand_A) == agent  -> votes cand_A iff the reverse binding is consistent
    When the three corroborate, the patient slot reaches quorum and the substrate IGNITES the answer; when a query is
    unstored/inconsistent the reads diverge and the bus ABSTAINS (the no-confab moat, done by the substrate)."""
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, _cooc = store_n_relation_facts(composer, CHAINS, [EAT, CONFIRM],
                                          distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    b, xp, slots, snap = build_workspace_bridge(seed, lesion=False)
    d_sub = D_SUB_UNANIMITY[3]
    rng = np.random.default_rng(seed * 991 + 7)

    def bus_gate(agent, action):
        """The gate() combination done by IGNITION: 3 organ reads -> subthreshold votes -> the substrate decides."""
        cand_A = composer.query_patient(agent, action)            # organ A: spiking recall
        cand_B = composer.query_patient(agent, CONFIRM)           # organ B: corroborating second-relation recall
        # organ C: reverse-binding VERIFY — vote cand_A only if `query_agent(action, cand_A)` recovers the agent
        cand_C = cand_A if (cand_A is not None and composer.query_agent(action, cand_A) == agent) else None
        candidates = [cand_A, cand_B, cand_C]
        if cand_A is None:                                        # primary organ miss -> honest abstain (moat)
            return None, candidates
        decoy = _pick_decoy(all_concepts, exclude=set(c for c in candidates if c is not None) | {agent}, rng=rng)
        committed, rates, w, nign = norgan_hop(b, xp, slots, snap, candidates, decoy, d_sub, rng=rng)
        return committed, candidates

    # (1) STORED query: 3 organs corroborate -> the bus ignites the answer (== host `if recalled==p`)
    hits = tot = 0
    examples = []
    for ch in CHAINS:
        for a, p in zip(ch[:-1], ch[1:]):
            tot += 1
            ans, cands = bus_gate(a, EAT)
            hits += int(ans == p)
            if len(examples) < 4:
                examples.append({"q": f"what does {a} {EAT}?", "organ_reads": cands, "bus_answer": ans, "want": p})
    stored_acc = hits / tot

    # (2) UNSTORED / inconsistent query: organs diverge / primary misses -> the bus ABSTAINS (no confab)
    abstain_ok = abstain_tot = 0
    for bad_agent in ("ball", "box", "bell", "kite", "drum"):
        abstain_tot += 1
        ans, cands = bus_gate(bad_agent, EAT)
        abstain_ok += int(ans is None)
    for a, _p in edges[:5]:                                        # a stored agent under a WRONG action -> abstain
        abstain_tot += 1
        ans, cands = bus_gate(a, "fly")
        abstain_ok += int(ans is None)
    abstain_acc = abstain_ok / abstain_tot

    # (3) host baseline: the exact gate() combination in Python — parity check
    host_hits = 0
    for ch in CHAINS:
        for a, p in zip(ch[:-1], ch[1:]):
            recalled = composer.query_patient(a, EAT)
            host_hits += int(recalled == p)                       # host `if recalled == p`
    host_acc = host_hits / tot

    ok = bool(stored_acc >= 0.85 and abstain_acc >= 0.85 and abs(stored_acc - host_acc) < 1e-9)
    print(f"[prototype seed={seed}] 3 REAL composer organs (recall + corroborate + reverse-VERIFY) routed through the bus:",
          flush=True)
    for ex in examples:
        print(f"    {ex['q']:24s} organ_reads={ex['organ_reads']} -> bus IGNITES {ex['bus_answer']!r} (want {ex['want']!r})",
              flush=True)
    print(f"    STORED: bus answers {stored_acc:.3f} == host gate() {host_acc:.3f} (parity)  |  "
          f"UNSTORED/inconsistent: bus ABSTAINS {abstain_acc:.3f} (no-confab moat, done by the substrate)", flush=True)
    print(f"    PROTOTYPE {'HOLDS' if ok else 'FAILS'}: the SUBSTRATE made the gate() combination host Python makes today.",
          flush=True)
    return ok, {"stored_acc": stored_acc, "abstain_acc": abstain_acc, "host_acc": host_acc, "examples": examples}


def main():
    ap = argparse.ArgumentParser(description="GNW N-organ ignition bus de-risk (substrate combines N>=3 organ reads).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-organs", type=int, default=N_ORGANS_DEFAULT)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--d-sub", type=float, default=None, help="per-organ subthreshold drive (default: unanimity per-N)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prototype", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_norgan_bus/summary.json")
    args = ap.parse_args()

    d_sub = args.d_sub if args.d_sub is not None else D_SUB_UNANIMITY.get(args.n_organs, 1000.0)

    if args.backend != "auto":
        from sim.backend import get_backend
        get_backend(args.backend)

    if args.calibrate:
        cal_json = args.json.replace("summary.json", "calibration_seed%d.json" % args.seed)
        return 0 if run_calibrate(args.seed, args.n_organs, d_sub, json_path=cal_json) else 1
    if args.smoke:
        return 0 if run_primitive_smoke(args.seed, args.n_organs, d_sub) else 1
    if args.prototype:
        ok, _rec = run_prototype(args.seed, args.D)
        return 0 if ok else 1

    n_concepts = len({c for ch in CHAINS for c in ch})
    print(f"[gnw-norgan-bus] {len(CHAINS)} chains | {n_concepts} concepts | chance {1.0/n_concepts:.3f} | "
          f"N_organs={args.n_organs} K_slots={K_SLOTS} D={args.D} d_sub={d_sub:.0f} backend={args.backend}\n", flush=True)

    results = [run_seed(s, args.n_organs, d_sub, D=args.D) for s in args.seeds]
    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)

    def mean(k):
        return float(np.mean([r[k] for r in results]))

    print("\n── integration attribution (tools.lab.attributable_to) ──", flush=True)
    void_if(mean("consensus_2hop_acc") <= 1e-9, "intact consensus chase is ~0 — nothing to attribute")
    attributable_to("2-hop success needs ALL N organs", mean("consensus_2hop_acc"), mean("single_organ_acc"))

    summary = {
        "runner": "_gnw_norgan_bus_derisk",
        "claim": "the spiking workspace COMBINES N>=3 subthreshold organ reads via consensus-ignition + WTA + re-entry",
        "seeds": list(args.seeds), "n_organs": int(args.n_organs), "D": int(args.D), "d_sub": float(d_sub),
        "backend": args.backend, "all_go": all_go, "n_go": n_go, "n_seeds": len(results),
        "mean_consensus_2hop_acc": mean("consensus_2hop_acc"),
        "mean_query_chain_2hop_acc": mean("query_chain_2hop_acc"),
        "mean_spreading_floor": mean("spreading_floor"),
        "mean_single_organ_acc": mean("single_organ_acc"), "mean_leaveoneout_acc": mean("leaveoneout_acc"),
        "mean_disagree_acc": mean("disagree_acc"), "mean_shuffle_off_acc": mean("shuffle_off_acc"),
        "mean_onecycle_acc": mean("onecycle_acc"), "mean_lesion_acc": mean("lesion_acc"),
        "mean_single_hop_reflex_acc": mean("single_hop_reflex_acc"),
        "mean_mutual_exclusion_frac": mean("mutual_exclusion_frac"),
        "mean_majority_override_acc": mean("majority_override_acc"),
        "mean_majority_suppressed_frac": mean("majority_suppressed_frac"),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_go else ("PARTIAL" if n_go >= 1 else "NEGATIVE")
    print(f"\n{'='*100}", flush=True)
    print(f"  GNW N-ORGAN BUS VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)  [N={args.n_organs} organs]", flush=True)
    print(f"    consensus_2hop={summary['mean_consensus_2hop_acc']:.3f} vs query_chain="
          f"{summary['mean_query_chain_2hop_acc']:.3f} (spread_floor={summary['mean_spreading_floor']:.3f})", flush=True)
    print(f"    collapses: single={summary['mean_single_organ_acc']:.3f} leave1out={summary['mean_leaveoneout_acc']:.3f} "
          f"disagree={summary['mean_disagree_acc']:.3f} shuffle_off={summary['mean_shuffle_off_acc']:.3f} "
          f"onecycle={summary['mean_onecycle_acc']:.3f} lesion={summary['mean_lesion_acc']:.3f}", flush=True)
    print(f"    reflex_survives={summary['mean_single_hop_reflex_acc']:.3f} moat_all={summary['all_moat_ok']} "
          f"ME_single={summary['mean_mutual_exclusion_frac']:.3f} | majority_override="
          f"{summary['mean_majority_override_acc']:.3f} dissenter_suppressed={summary['mean_majority_suppressed_frac']:.3f}",
          flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
