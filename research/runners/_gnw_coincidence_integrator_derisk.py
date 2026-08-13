"""GNW coincidence-integrator de-risk — the SUBSTRATE combines >=2 organ reads (the missing rung toward the WIRED bus).

WHAT THIS ADDS OVER P1.2 (the load-bearing distinction). The P1.2 re-entrant loop is 6-seed GO, but its workspace
only ever re-ignites ONE strong candidate: `query_patient` returns a single answer, driven at IGNITE_PA (3.3x the
random distractors), so a SINGLE organ read already crosses the ignition threshold on its own. The workspace there is
a RELAY that re-ignites the composer's answer; the actual combining/reasoning is the HOST composer. The mission's
load-bearing question is DIFFERENT and harder (2026-08-12 faculty-map T1-1): can a spiking workspace take the
SUBTHRESHOLD reads of >=2 organs, and only the slot where the reads COINCIDE cross the ignition threshold — so the
substrate's ignition dynamics ARE the integration (an AND/consensus over organs), not a host if/else? Then broadcast
that ignited consensus and RE-ENTER it as the next premise, to produce a 2-step inference the single-shot pipeline
cannot.

MECHANISM (Dehaene-Changeux ignition as a coincidence/consensus gate + rung-2 mutual inhibition + re-entry; reuse of
the P1.2/rung-1 spiking workspace, NO `sim/` edit):
  Two organs write SUBTHRESHOLD drive (D_SUB pA, < the solo ignition knee) into the shared K-slot workspace:
    Organ R (RECALL, relation EAT)  -> query_patient(x, EAT)     -> candidate r  -> +D_SUB to slot(r)
    Organ C (CONFIRM, relation SEE) -> query_patient(x, CONFIRM) -> candidate c  -> +D_SUB to slot(c)
    plus a DECOY competitor          -> a spurious concept         -> +D_SUB to slot(decoy)   (single-vote rival)
  In a CONSISTENT world r == c (the true chain-next), so slot(r) receives 2*D_SUB -> it alone crosses the ignition
  threshold, the shared inhibitory pool (workspace_fs) WTA-suppresses the single-vote decoy, and the NMDA attractor
  SUSTAINS the ignition. COMMIT = the spiking winner read from late-window rates. BROADCAST BACK: x_next = the committed
  concept RE-ENTERS as the next hop's premise (the loop cursor is the ignited assembly, not a python variable). Two hops
  => the 2-step conclusion ch[0] -> ch[2].

  So the AND-over-organs is computed by the neuronal ignition THRESHOLD (a single read is subthreshold; the coincidence
  of two is suprathreshold) and the WTA by the shared inhibition — the substrate's dynamics, not host control flow.

GO GATE (6 seeds 42/43/44/100/101/102): coincidence_2hop_acc >= 0.75 AND >= spreading_floor + 0.5 AND matches the host
one-shot query_chain baseline (parity) AND EVERY ablation of the synaptic mechanism collapses to <= chance-ish AND the
no-confab moat abstains (unstored cue + chain over-run). i.e. a never-told 2-hop conclusion reached by the workspace
INTEGRATING two subthreshold organ reads and iterating on its broadcast, with the combination done synaptically.

ANTI-CHEATS (the anti-cheats ARE the result — each targets a distinct "it's really the substrate" claim):
  - SINGLE-ORGAN COLLAPSE  [THE anti-host-if-else]: drop organ C's drive -> the true-next slot gets only D_SUB, ties the
    decoy, NEITHER crosses threshold -> abstain -> the chase collapses. Symmetric for organ R alone. Proves ONE read is
    subthreshold: the substrate needs BOTH. A host `if organ_R: return r` would succeed here -> the collapse proves the
    combination is the workspace's ignition, not a host read.
  - DISAGREEMENT / CONSENSUS-VETO: permute the CONFIRM relation so c != r -> two slots at D_SUB, NO coincidence -> nothing
    ignites -> the workspace WITHHOLDS the unconfirmed conclusion (a genuine convergent-evidence gate). Collapse.
  - NO-IGNITION LESION: assembly self-recurrence -> 0 -> even the 2*D_SUB coincidence cannot sustain -> collapse, WHILE
    the single-hop recall reflex (direct query_patient, never routed through the workspace) SURVIVES (the dissociation).
  - NO-RE-ENTRY (single-shot): cap the loop at 1 cycle -> only hop-1 reached -> the 2-hop want is unreachable -> collapse.
    This is exactly what the current PRODUCTION host pipeline (snapshot organs once, combine once, emit) cannot do.
  - SHUFFLE-DRIVE: route organ C's drive to a RANDOM slot instead of slot(c) -> the coincidence lands off-target ->
    collapse. Proves the combination is CONGRUENCE (both organs on the same content), not a fixed slot.
  - SPREADING-ACTIVATION FLOOR: relation-blind co-occurrence diffusion stays ~chance; the integrated chase must beat it.
  - MOAT: an unstored cue and a past-chain-end over-run -> an organ misses -> abstain (None) at that hop.

Run (CPU cheap-first):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_coincidence_integrator_derisk \
      --seeds 42 43 44 100 101 102 --D 256 --backend numpy \
      --json research/findings/raw/_gnw_coincidence_integrator/summary.json
  # calibration + 1-seed primitive smoke first:
  SIM_BACKEND=numpy python -u -m research.runners._gnw_coincidence_integrator_derisk --calibrate --seed 42 --D 256
  SIM_BACKEND=numpy python -u -m research.runners._gnw_coincidence_integrator_derisk --smoke --seed 42 --D 256
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

# reuse-by-import the P1.2 spiking workspace (build + ignition-read) + constants — NO `sim/` edit, NO re-derivation.
from research.runners._p1_2_workspace_deliberation_loop_derisk import (
    build_workspace_bridge, _ignite_and_read,
    K_SLOTS, ASSEMBLY_SIZE, IGNITE_PA, IGNITE_FRAC, SOLO_PLATEAU,
)
from research.runners._phaseB_multihop_query_chain_derisk import (
    CHAINS, EAT, build_vocab, store_facts, spreading_predict,
)
from research.runners.rf_phasor_composer import RFPhasorComposer
from tools.lab import attributable_to, void_if

CONFIRM = "confirm"                    # organ C's relation (a second independent evidence stream over the same chain)
D_SUB_DEFAULT = 1400.0                 # per-organ SUBTHRESHOLD drive: one vote < the ignition knee, two votes (2800) >
HOPS = 2                               # a 2-step inference (ch[0] -> ch[2]); single-shot single-organ can reach neither
THR = IGNITE_FRAC * SOLO_PLATEAU       # "ignited" iff late-window rate >= THR


# ── the two-relation fact store (organ R = EAT, organ C = CONFIRM) ────────────────────────────────────────────
def store_two_relation_facts(composer, chains, permute_confirm=False, rng=None, distractor_rng=None):
    """Store the food-web under EAT (recall organ) + the SAME edges under CONFIRM (the confirm organ) + PLAY/SEE
    distractors (spreading-floor pollution, via store_facts). With permute_confirm the CONFIRM patients are scrambled
    (organ C disagrees with organ R -> the consensus-veto control). Returns (edges, cooc)."""
    edges, cooc = store_facts(composer, chains, distractor_rng=distractor_rng)   # EAT + distractors + cooc
    conf_targets = [p for _, p in edges]
    if permute_confirm:
        rng.shuffle(conf_targets)
    for (a, _p), ct in zip(edges, conf_targets):
        composer.store(a, CONFIRM, ct)
    return edges, cooc


# ── slot bookkeeping: map the (<=3) distinct concepts of a hop to workspace slots ─────────────────────────────
def _assign_slots(concepts):
    """Deterministic concept->slot map (in first-seen order). r==c share ONE slot (their drives ACCUMULATE)."""
    slot_of, order = {}, []
    for c in concepts:
        if c is not None and c not in slot_of:
            slot_of[c] = len(order)
            order.append(c)
    return slot_of, order


def coincidence_hop(bridge, xp, slots_dev, snap, r_cand, c_cand, decoy, d_sub,
                    organ_r=True, organ_c=True, shuffle_rng=None):
    """One EVALUATE/COMMIT over the workspace: organ R drives slot(r_cand), organ C drives slot(c_cand), a spurious
    DECOY drives its own slot — each at d_sub. Only a slot that receives >= 2 votes crosses the ignition knee. WTA +
    ignition select ONE winner (or none -> abstain). Returns (committed_concept|None, rates, winner_slot, n_ignited).
      organ_r/organ_c=False: that organ contributes NO drive (the single-organ collapse controls).
      shuffle_rng: route organ C's drive to a RANDOM slot instead of slot(c_cand) (the shuffle control)."""
    slot_of, order = _assign_slots([r_cand, c_cand, decoy])
    n = len(slots_dev)
    drives = [0.0] * n
    if organ_r and r_cand in slot_of:
        drives[slot_of[r_cand]] += d_sub
    if organ_c and c_cand in slot_of:
        tgt = slot_of[c_cand]
        if shuffle_rng is not None:                       # shuffle: organ C votes for a random slot (off-target)
            tgt = int(shuffle_rng.integers(max(1, len(order))))
        drives[tgt] += d_sub
    if decoy in slot_of:
        drives[slot_of[decoy]] += d_sub                   # the single-vote competitor (always present)

    rates = _ignite_and_read(bridge, xp, slots_dev, snap, drives)
    # restrict to the slots actually in play this hop
    active = list(range(len(order)))
    winner = int(active[int(np.argmax([rates[i] for i in active]))])
    ignited = rates[winner] >= THR
    n_ignited = int(sum(1 for i in active if rates[i] >= THR))
    committed = order[winner] if ignited else None
    return committed, rates, winner, n_ignited


def _pick_decoy(all_concepts, exclude, rng):
    pool = [c for c in all_concepts if c not in exclude]
    return pool[int(rng.integers(len(pool)))] if pool else None


def coincidence_chase(bridge, xp, slots_dev, snap, composer, cue, all_concepts, d_sub, rng,
                      organ_r=True, organ_c=True, shuffle=False, max_cycles=None, return_trace=False):
    """The workspace-carried 2-hop DELIBERATION. x starts at cue; each hop: organ R + organ C read (PROPOSE) ->
    coincidence-ignition EVALUATE/COMMIT -> BROADCAST BACK (x_next = the committed winner). Abstains (None) the moment
    an organ misses (moat) or nothing ignites (no consensus)."""
    x = cue
    trace = []
    n_hops = HOPS if max_cycles is None else min(int(max_cycles), HOPS)
    for h in range(n_hops):
        r = composer.query_patient(x, EAT)                # organ R
        c = composer.query_patient(x, CONFIRM)            # organ C
        if r is None or c is None:                        # an organ missed -> moat abstains
            trace.append({"hop": h, "x": x, "r": r, "c": c, "committed": None, "n_ignited": 0})
            return (None, trace) if return_trace else None
        decoy = _pick_decoy(all_concepts, exclude={r, c, x}, rng=rng)
        sh_rng = rng if shuffle else None
        committed, rates, winner, n_ign = coincidence_hop(
            bridge, xp, slots_dev, snap, r, c, decoy, d_sub,
            organ_r=organ_r, organ_c=organ_c, shuffle_rng=sh_rng)
        trace.append({"hop": h, "x": x, "r": r, "c": c, "committed": committed,
                      "winner": int(winner), "n_ignited": int(n_ign)})
        if committed is None:                             # no consensus ignited -> abstain
            return (None, trace) if return_trace else None
        x = committed                                     # BROADCAST BACK: the spike-derived re-cue re-enters
    return (x, trace) if return_trace else x


# ── calibration: sweep solo drive to locate the ignition knee (D_SUB must sit below it, 2*D_SUB above) ─────────
def run_calibrate(seed, D, d_sub, json_path=None):
    build_vocab()
    b, xp, slots, snap = build_workspace_bridge(seed, lesion=False)
    print(f"[calibrate seed={seed}] solo-slot ignition curve (THR={THR:.3f}):", flush=True)
    knee = None
    curve = {}
    for drive in (600, 900, 1200, 1400, 1500, 1700, 1800, 2100, 2400, 2800):
        rates = _ignite_and_read(b, xp, slots, snap, [float(drive)] + [0.0] * (len(slots) - 1))
        ig = rates[0] >= THR
        curve[str(drive)] = round(float(rates[0]), 3)
        if ig and knee is None:
            knee = drive
        print(f"    solo drive={drive:5.0f} -> slot0 late-rate={rates[0]:.3f}  ignited={ig}", flush=True)
    rc = _ignite_and_read(b, xp, slots, snap, [2 * d_sub] + [0.0] * (len(slots) - 1))
    coincidence_rate = round(float(rc[0]), 3)
    print(f"    2*d_sub={2*d_sub:.0f} -> rate={rc[0]:.3f} ignited={rc[0] >= THR}", flush=True)
    ok = (knee is not None and d_sub < knee and rc[0] >= THR)
    print(f"[calibrate] knee~{knee}  -> D_SUB={d_sub:.0f} "
          f"{'OK (subthreshold solo, suprathreshold doubled)' if ok else 'OUT OF WINDOW — retune'}", flush=True)
    if json_path:
        rec = {"runner": "_gnw_coincidence_integrator_derisk", "mode": "calibrate", "seed": int(seed), "D": int(D),
               "THR": round(float(THR), 3), "d_sub": float(d_sub), "knee_pA": knee,
               "solo_rate_by_drive": curve, "d_sub_solo_rate": curve.get(str(int(d_sub))),
               "coincidence_rate_2xd_sub": coincidence_rate, "in_window": bool(ok)}
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"    [saved] {json_path}", flush=True)
    return ok


# ── 1-seed primitive smoke ────────────────────────────────────────────────────────────────────────────────────
def run_primitive_smoke(seed, D, d_sub):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_two_relation_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]

    ch = CHAINS[0]
    cue = ch[0]
    r = composer.query_patient(cue, EAT)
    c = composer.query_patient(cue, CONFIRM)
    print(f"[smoke] cue={cue!r}: organ_R(EAT)->{r!r}  organ_C(CONFIRM)->{c!r}  (want {ch[1]!r}, agree={r==c})", flush=True)

    b, xp, slots, snap = build_workspace_bridge(seed, lesion=False)
    rng = np.random.default_rng(seed * 991 + 7)
    decoy = _pick_decoy(all_concepts, exclude={r, c, cue}, rng=rng)

    # (1) COINCIDENCE: both organs -> ignites the agreed concept, beats the single-vote decoy
    com, rates, w, nign = coincidence_hop(b, xp, slots, snap, r, c, decoy, d_sub)
    print(f"[smoke] COINCIDENCE rates={[round(x,3) for x in rates[:3]]} winner=slot{w} n_ignited={nign} committed={com!r}", flush=True)
    coincidence_ok = bool(com == r and nign == 1)

    # (2) SINGLE-ORGAN R only -> subthreshold -> no ignition (the anti-host-if-else)
    com_r, rr, _w, n_r = coincidence_hop(b, xp, slots, snap, r, c, decoy, d_sub, organ_c=False)
    print(f"[smoke] R-ONLY       rates={[round(x,3) for x in rr[:3]]} n_ignited={n_r} committed={com_r!r}", flush=True)
    r_only_abstains = bool(com_r is None)

    # (3) SINGLE-ORGAN C only -> subthreshold -> no ignition
    com_c, rcc, _w2, n_c = coincidence_hop(b, xp, slots, snap, r, c, decoy, d_sub, organ_r=False)
    print(f"[smoke] C-ONLY       rates={[round(x,3) for x in rcc[:3]]} n_ignited={n_c} committed={com_c!r}", flush=True)
    c_only_abstains = bool(com_c is None)

    # (4) DISAGREEMENT: organ C points elsewhere -> no coincidence -> withhold
    other = _pick_decoy(all_concepts, exclude={r, c, cue, decoy}, rng=rng)
    com_d, rd, _w3, n_d = coincidence_hop(b, xp, slots, snap, r, other, decoy, d_sub)
    print(f"[smoke] DISAGREE     rates={[round(x,3) for x in rd[:3]]} n_ignited={n_d} committed={com_d!r} (C->{other!r})", flush=True)
    disagree_abstains = bool(com_d is None)

    # (5) NO-IGNITION LESION: even the coincidence cannot sustain
    bl, xpl, slotsl, snapl = build_workspace_bridge(seed, lesion=True)
    com_l, rl, _w4, n_l = coincidence_hop(bl, xpl, slotsl, snapl, r, c, decoy, d_sub)
    print(f"[smoke] LESION       rates={[round(x,3) for x in rl[:3]]} n_ignited={n_l} committed={com_l!r}", flush=True)
    lesion_kills = bool(com_l is None)

    reflex_ok = bool(composer.query_patient(cue, EAT) == ch[1])   # the workspace-independent recall survives

    ok = bool(coincidence_ok and r_only_abstains and c_only_abstains and disagree_abstains and lesion_kills and reflex_ok)
    print(f"\n[smoke] PRIMITIVE {'HOLDS' if ok else 'FAILS'}: coincidence_ignites_single={coincidence_ok} "
          f"R_only_abstains={r_only_abstains} C_only_abstains={c_only_abstains} disagree_abstains={disagree_abstains} "
          f"lesion_kills={lesion_kills} reflex_survives={reflex_ok}", flush=True)
    return ok


# ── the per-seed experiment ───────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, D, d_sub, verbose=True):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc = store_two_relation_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    n_concepts = len(all_concepts)
    chance = 1.0 / n_concepts

    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    b_l, xp_l, slots_l, snap_l = build_workspace_bridge(seed, lesion=True)

    def rng():
        return np.random.default_rng(seed * 991 + 7)

    chains2 = [ch for ch in CHAINS if len(ch) > HOPS]
    tot = len(chains2)

    # ── INTACT integrated 2-hop chase + host one-shot baseline + spreading floor ────────────────────────────────
    coin_ok = qc_ok = spread_ok = 0
    for ch in chains2:
        cue, want = ch[0], ch[HOPS]
        term = coincidence_chase(b_i, xp, slots_i, snap_i, composer, cue, all_concepts, d_sub, rng())
        coin_ok += int(term == want)
        qc_ok += int(composer.query_chain(cue, [EAT] * HOPS) == want)
        spread_ok += int(spreading_predict(cooc, cue, HOPS, all_concepts) == want)
    coin_acc = coin_ok / tot
    qc_acc = qc_ok / tot
    spread_floor = spread_ok / tot

    # ── ANTI-CHEATS ─────────────────────────────────────────────────────────────────────────────────────────────
    def chase_acc(**kw):
        ok = 0
        for ch in chains2:
            ok += int(coincidence_chase(b_i, xp, slots_i, snap_i, composer, ch[0], all_concepts, d_sub, rng(), **kw) == ch[HOPS])
        return ok / tot

    r_only_acc = chase_acc(organ_c=False)                 # single-organ (recall only) -> subthreshold -> collapse
    c_only_acc = chase_acc(organ_r=False)                 # single-organ (confirm only) -> collapse
    shuffle_acc = chase_acc(shuffle=True)                 # organ C off-target -> collapse
    onecycle_acc = chase_acc(max_cycles=1)                # single-shot: only hop-1 -> 2-hop unreachable -> collapse

    # no-ignition lesion (collapse) + the dissociation keystone (single-hop reflex survives)
    lesion_ok = reflex_ok = 0
    for ch in chains2:
        lesion_ok += int(coincidence_chase(b_l, xp_l, slots_l, snap_l, composer, ch[0], all_concepts, d_sub, rng()) == ch[HOPS])
        reflex_ok += int(composer.query_patient(ch[0], EAT) == ch[1])
    lesion_acc = lesion_ok / tot
    reflex_acc = reflex_ok / tot

    # disagreement / consensus-veto: organ C permuted -> no coincidence -> withhold
    comp_perm = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_two_relation_facts(comp_perm, CHAINS, permute_confirm=True,
                             rng=np.random.default_rng(seed * 101 + 5),
                             distractor_rng=np.random.default_rng(seed * 53 + 1))
    disagree_ok = 0
    for ch in chains2:
        disagree_ok += int(coincidence_chase(b_i, xp, slots_i, snap_i, comp_perm, ch[0], all_concepts, d_sub, rng()) == ch[HOPS])
    disagree_acc = disagree_ok / tot

    # MOAT: unstored cue + past-chain-end over-run -> abstain
    moat_unstored = coincidence_chase(b_i, xp, slots_i, snap_i, composer, "ball", all_concepts, d_sub, rng())
    moat_over = coincidence_chase(b_i, xp, slots_i, snap_i, composer, CHAINS[0][-1], all_concepts, d_sub, rng())
    moat_unstored_abstains = moat_unstored is None
    moat_over_abstains = moat_over is None
    moat_ok = bool(moat_unstored_abstains and moat_over_abstains)

    # mutual-exclusion diagnostic (single-content access at each committed hop)
    me_single = me_total = 0
    for ch in chains2:
        _t, tr = coincidence_chase(b_i, xp, slots_i, snap_i, composer, ch[0], all_concepts, d_sub, rng(), return_trace=True)
        for step in tr:
            if step.get("committed") is not None:
                me_total += 1
                me_single += int(step["n_ignited"] == 1)
    me_frac = (me_single / me_total) if me_total else 0.0

    two_chance = 2.0 * chance
    seed_go = bool(
        coin_acc >= 0.75 and
        coin_acc >= spread_floor + 0.5 and
        coin_acc >= qc_acc and                            # parity with the host one-shot (same conclusion, synaptic path)
        r_only_acc <= max(two_chance, 0.10) and           # a single organ read is subthreshold (the anti-if-else)
        c_only_acc <= max(two_chance, 0.10) and
        disagree_acc <= max(two_chance, 0.10) and         # conflicting organs -> withhold (consensus-veto)
        shuffle_acc <= max(two_chance, 0.10) and          # combination is congruence, not slot
        onecycle_acc <= max(two_chance, 0.10) and         # re-entry load-bearing (single-shot can't)
        lesion_acc <= max(two_chance, 0.10) and           # ignition load-bearing
        reflex_acc >= 0.85 and                            # the single-hop recall reflex survives (dissociation)
        moat_ok
    )

    result = {
        "seed": int(seed), "D": int(D), "d_sub": float(d_sub), "hops": HOPS, "n_concepts": n_concepts,
        "chance": chance, "n_chains": tot,
        "coincidence_2hop_acc": coin_acc, "query_chain_2hop_acc": qc_acc, "spreading_floor": spread_floor,
        "r_only_acc": r_only_acc, "c_only_acc": c_only_acc, "disagree_acc": disagree_acc,
        "shuffle_acc": shuffle_acc, "onecycle_acc": onecycle_acc, "lesion_acc": lesion_acc,
        "single_hop_reflex_acc": reflex_acc,
        "moat_unstored_abstains": moat_unstored_abstains, "moat_over_abstains": moat_over_abstains, "moat_ok": moat_ok,
        "mutual_exclusion_frac": me_frac, "seed_go": seed_go,
    }
    if verbose:
        print(f"[coinc seed={seed} D={D} d_sub={d_sub:.0f}] coincidence_2hop={coin_acc:.3f} "
              f"vs query_chain={qc_acc:.3f} (spread_floor={spread_floor:.3f}, chance={chance:.3f})", flush=True)
        print(f"    INTEGRATION collapses: R_only={r_only_acc:.3f} C_only={c_only_acc:.3f} disagree={disagree_acc:.3f} "
              f"shuffle={shuffle_acc:.3f}", flush=True)
        print(f"    RE-ENTRY/IGNITION: onecycle={onecycle_acc:.3f} lesion={lesion_acc:.3f} | reflex_survives={reflex_acc:.3f} "
              f"| moat unstored={moat_unstored_abstains} over={moat_over_abstains} | ME_single={me_frac:.3f}", flush=True)
        print(f"    seed_GO={seed_go}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="GNW coincidence-integrator de-risk (substrate combines >=2 organ reads).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--d-sub", type=float, default=D_SUB_DEFAULT)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_coincidence_integrator/summary.json")
    args = ap.parse_args()

    if args.backend != "auto":
        from sim.backend import get_backend
        get_backend(args.backend)

    if args.calibrate:
        cal_json = args.json.replace("summary.json", "calibration_seed%d.json" % args.seed)
        return 0 if run_calibrate(args.seed, args.D, args.d_sub, json_path=cal_json) else 1
    if args.smoke:
        return 0 if run_primitive_smoke(args.seed, args.D, args.d_sub) else 1

    n_concepts = len({c for ch in CHAINS for c in ch})
    print(f"[gnw-coincidence-integrator] {len(CHAINS)} chains | {n_concepts} concepts | chance {1.0/n_concepts:.3f} | "
          f"K_slots={K_SLOTS} D={args.D} d_sub={args.d_sub:.0f} backend={args.backend}\n", flush=True)

    results = [run_seed(s, args.D, args.d_sub) for s in args.seeds]
    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)

    def mean(k):
        return float(np.mean([r[k] for r in results]))

    # the integration attribution: what fraction of the 2-hop success needs BOTH organs (vs the best single organ)?
    print("\n── integration attribution (tools.lab.attributable_to) ──", flush=True)
    best_single = max(mean("r_only_acc"), mean("c_only_acc"))
    void_if(mean("coincidence_2hop_acc") <= 1e-9, "intact coincidence chase is ~0 — nothing to attribute")
    attributable_to("2-hop success needs BOTH organs", mean("coincidence_2hop_acc"), best_single)

    summary = {
        "runner": "_gnw_coincidence_integrator_derisk",
        "claim": "the spiking workspace COMBINES two subthreshold organ reads via coincidence-ignition + re-entry",
        "seeds": list(args.seeds), "D": int(args.D), "d_sub": float(args.d_sub), "backend": args.backend,
        "all_go": all_go, "n_go": n_go, "n_seeds": len(results),
        "mean_coincidence_2hop_acc": mean("coincidence_2hop_acc"),
        "mean_query_chain_2hop_acc": mean("query_chain_2hop_acc"),
        "mean_spreading_floor": mean("spreading_floor"),
        "mean_r_only_acc": mean("r_only_acc"), "mean_c_only_acc": mean("c_only_acc"),
        "mean_disagree_acc": mean("disagree_acc"), "mean_shuffle_acc": mean("shuffle_acc"),
        "mean_onecycle_acc": mean("onecycle_acc"), "mean_lesion_acc": mean("lesion_acc"),
        "mean_single_hop_reflex_acc": mean("single_hop_reflex_acc"),
        "mean_mutual_exclusion_frac": mean("mutual_exclusion_frac"),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_go else ("PARTIAL" if n_go >= 1 else "NEGATIVE")
    print(f"\n{'='*100}", flush=True)
    print(f"  GNW COINCIDENCE-INTEGRATOR VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    print(f"    coincidence_2hop={summary['mean_coincidence_2hop_acc']:.3f} vs query_chain="
          f"{summary['mean_query_chain_2hop_acc']:.3f} (spread_floor={summary['mean_spreading_floor']:.3f})", flush=True)
    print(f"    collapses: R_only={summary['mean_r_only_acc']:.3f} C_only={summary['mean_c_only_acc']:.3f} "
          f"disagree={summary['mean_disagree_acc']:.3f} shuffle={summary['mean_shuffle_acc']:.3f} "
          f"onecycle={summary['mean_onecycle_acc']:.3f} lesion={summary['mean_lesion_acc']:.3f}", flush=True)
    print(f"    reflex_survives={summary['mean_single_hop_reflex_acc']:.3f} moat_all={summary['all_moat_ok']} "
          f"ME_single={summary['mean_mutual_exclusion_frac']:.3f}", flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
