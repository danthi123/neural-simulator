"""GNW coincidence-integrator with TWO GENUINELY DIFFERENT ORGANS — closing the parent's caveat #1.

WHAT THIS CLOSES. `research/findings/2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md`
is a 6/6 GO: the spiking workspace COMBINES two SUBTHRESHOLD organ reads via coincidence-ignition + 2-hop re-entry
(an AND, not a host if/else). Its NAMED honest-scope caveat #1: "Both organ reads come from the composer (recall
organ under two RELATIONS = two evidence streams). This is a deliberate simplification ... A genuinely distinct
second organ (a spiking surprise/familiarity monitor, or the P0.3 affect/value organ) is the immediate next rung."
This runner is that rung, in the SAME 2-organ / 2-hop coincidence-integrator FRAME (not the 3-organ quorum of
`_gnw_bus_heterogeneous_organ_derisk`, and not single-hop): the second subthreshold read now comes from a
GENUINELY DIFFERENT SPIKING ORGAN.

  organ A — spiking RECALL (composer):  query_patient(x, EAT) -> candidate a          [FHRR phasor VSA unbind]
  organ B — spiking EXPECTATION-VIOLATION monitor (NON-COMPOSER): the production `SurpriseProductionOrgan`, a
            predictive-coding mismatch circuit (Izhikevich cue -> FS/PV GABA_A subtractive prediction -> surprise
            RS pool). Organ B holds its OWN independent expectation e_B(x) of the world (an associative next-map,
            the environment/teacher boundary — exactly as the parent's organ C stored its own CONFIRM edges) and
            CONFIRMS the recall candidate against it by reading `cp_firing_states[surprise]`: a matches e_B(x) ->
            the asserted block is INHIBITED by the learned prediction -> CONFIRM (~0 Hz, below the organ's own
            calibrated threshold) -> organ B casts its subthreshold vote for slot(a); a violates e_B(x) -> the
            asserted block is un-inhibited -> the surprise pool FIRES (> threshold) -> organ B WITHHOLDS.
            The read is a genuine SPIKING rate off a DIFFERENT mechanism (NEVER `query_patient`); its lesion
            (zero the prediction edges) collapses the confirm/contradict separation (load-bearing).

THE UPGRADE OVER THE PARENT (the caveat the finding named). The parent's organ C was `query_patient(x, CONFIRM)`
— the SAME FHRR composer under a second relation, so the "two organ reads" were two relations of one substrate.
Here the two reads come from TWO DISTINCT SPIKING SUBSTRATES: the FHRR phasor composer (VSA unbind) and an
Izhikevich predictive-coding mismatch circuit (`cp_firing_states[surprise]`). The coincidence AND is now a genuine
CONVERGENT-EVIDENCE gate across two different organs (Dehaene-Changeux: ignition needs convergent drive), which is
exactly what caveat #1 asked for.

MECHANISM (reuse of the P1.2/rung-1 spiking workspace; NO `sim/` edit). Each organ writes D_SUB=1400 pA (below the
measured solo ignition knee ~2400 pA) into the shared K-slot workspace: organ A -> slot(a); organ B -> slot(a) IFF
its spiking read CONFIRMS a. In a consistent world a == e_B(x), so slot(a) receives 2*D_SUB=2800 pA -> it alone
crosses the knee, the shared inhibitory pool (`workspace_fs`) WTA-suppresses the single-vote decoy, the NMDA
attractor sustains it, and the committed spiking winner BROADCASTS BACK as the next hop's premise. Two hops -> a
2-step conclusion ch[0]->ch[2]. The AND-over-organs is the neuronal ignition THRESHOLD; the WTA is the shared
inhibition — the substrate's dynamics, not host control flow.

GO GATE (6 seeds 42/43/44/100/101/102): coincidence_2hop_acc >= 0.75 AND >= spreading_floor + 0.5 AND >= the host
one-shot query_chain baseline (parity) AND EVERY ablation of the synaptic mechanism collapses to <= chance-ish AND
organ B's own spiking read DISCRIMINATES (confirm Hz < threshold <= contradict Hz) AND the no-confab moat abstains.

ANTI-CHEATS (the parent's exact battery, each targeting a distinct "it's really the substrate combining two DISTINCT
organs" claim; UNDEFINED not NO-GO if an instrument precondition fails):
  - A-ONLY  [the anti-host-if-else]: drop organ B's drive -> slot(a) gets only D_SUB -> subthreshold -> abstain.
    A host `if organ_A: return a` would succeed; the collapse proves the combination is the workspace ignition.
  - B-ONLY: drop organ A's drive -> slot(a) gets only organ B's D_SUB -> subthreshold -> abstain. One organ read
    (composer OR the spiking surprise organ) is subthreshold on its own; the substrate needs BOTH.
  - DISAGREE / CONSENSUS-VETO: PERMUTE organ B's independent expectation e_B -> it reads the recall candidate as
    SURPRISING (contradict, high Hz) -> withholds its vote -> slot(a) falls to D_SUB -> abstain. The disagreement
    is a genuine SPIKING read (surprise Hz over threshold), not a host flip.
  - SHUFFLE: route organ B's confirm vote to an EMPTY slot -> no coincidence forms -> abstain (congruence, not
    content; the parent's corrected off-slot control).
  - ONECYCLE: cap the loop at 1 cycle -> only hop-1 reached -> the 2-hop conclusion is unreachable -> abstain.
    This is exactly what the current PRODUCTION host pipeline (snapshot organs once, combine once, emit) cannot do.
  - WORKSPACE LESION: zero the assembly self-recurrence -> even 2*D_SUB cannot sustain -> abstain, WHILE the
    single-hop composer recall REFLEX (direct query_patient, never routed through the workspace) SURVIVES.
  - ORGAN-B LESION [organ B's OWN spiking prediction is load-bearing]: zero the surprise circuit's
    patient_expected->surprise prediction edges -> CONFIRM fires as high as CONTRADICT -> organ B can no longer
    recognize the familiar continuation -> it withholds even on a genuine match -> abstain. Proves the second vote
    is caused by the learned SPIKING prediction, not a fixed input artifact.
  - MOAT: an unstored cue + a past-chain-end over-run -> organ A misses -> abstain.

DISCIPLINE: reuse-by-import (the P1.2 workspace + the parent coincidence machinery + the production surprise organ).
NO `sim/` edit. Deterministic per seed. Run (CPU numpy cheap-first; the effect is a subthreshold-vs-suprathreshold
bifurcation, not GPU-scale-dependent):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_two_distinct_organs_derisk --calibrate --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_two_distinct_organs_derisk --smoke --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_two_distinct_organs_derisk --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_two_distinct_organs/summary.json
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
    build_workspace_bridge, _ignite_and_read, K_SLOTS,
)
# reuse-by-import the parent coincidence-integrator's slot machinery + calibrated drive + calibration — same frame.
from research.runners._gnw_coincidence_integrator_derisk import (
    _assign_slots, _pick_decoy, THR, D_SUB_DEFAULT, run_calibrate,
)
# reuse-by-import the held-out CHAINS fixtures + fact store + spreading-activation floor + relation.
from research.runners._phaseB_multihop_query_chain_derisk import (
    CHAINS, EAT, build_vocab, store_facts, spreading_predict,
)
from research.runners.rf_phasor_composer import RFPhasorComposer
# reuse-by-import the PRODUCTION spiking expectation-violation / familiarity monitor (the NON-COMPOSER organ B).
from research.runners.surprise_production_organ import SurpriseProductionOrgan
from tools.lab import attributable_to, void_if

HOPS = 2                               # a 2-step inference (ch[0] -> ch[2]); single-shot single-organ reaches neither
# Organ B circuit size. The 2-hop chase's DISTINCT expected concepts are ch[1] & ch[2] over 8 chains = 16 -> a
# 16-block cue-addressable range gives each its OWN trained block (cue_i -> patient_i learned distinctly), avoiding
# the round-robin block-wrap the heterogeneous-bus runner deliberately dodged with its 8 first-edges; the 8 novel
# blocks hold the mismatched-asserted/decoy patients. (Both configurable; --n-trained/--n-novel.)
N_TRAINED_DEFAULT = 16
N_NOVEL_DEFAULT = 8


# ── organ B: the genuinely-distinct SPIKING expectation-violation organ, sized for the 2-hop chase ────────────
class _ExpandedSurpriseOrgan(SurpriseProductionOrgan):
    """`SurpriseProductionOrgan` with a configurable cue-addressable block count (the stock organ hardcodes
    n_trained=8). Only `_build_one` (the standalone, non-merged path) is overridden; everything else — the Hebbian
    `train_expectation`, the homeostatic prediction-gain equalizer, the confirm/contradict threshold calibration,
    `read_surprise` (the `cp_firing_states[surprise]` read), and the load-bearing lesion — is the PRODUCTION organ,
    unchanged."""

    def __init__(self, seed: int = 42, n_trained: int = N_TRAINED_DEFAULT, n_novel: int = N_NOVEL_DEFAULT, **kw):
        super().__init__(seed=seed, **kw)
        self._n_trained = int(n_trained)
        self._n_novel = int(n_novel)

    def _build_one(self, lesion=False):
        from sim.backend import get_backend
        from research.runners._spiking_expectation_rpe_derisk import (
            build_expectation_circuit, train_expectation, _idx, _install_block_diagonal,
        )
        xp, _ = get_backend()
        bridge, cfg, meta = build_expectation_circuit(
            self.seed, n_trained=self._n_trained, n_novel=self._n_novel, blk=24, cue_blk=24,
            cue_to_expected_weight=self.cue_w)
        bridge._blk = meta["blk"]
        regions = ("cue", "patient_expected", "patient_asserted", "surprise")
        idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}
        train_expectation(bridge, cfg, idx_map, meta, xp, n_reps=self.n_reps)
        cfg.enable_hebbian_learning = False
        if lesion:
            _install_block_diagonal(bridge, "patient_expected", "surprise", meta["blk"], 0.0)  # remove prediction
        return bridge, cfg, meta, xp, idx_map


def build_organ_b(seed, n_trained=N_TRAINED_DEFAULT, n_novel=N_NOVEL_DEFAULT):
    organ = _ExpandedSurpriseOrgan(seed=seed, n_trained=n_trained, n_novel=n_novel)
    organ.ensure_built()
    return organ


def pre_register_expected(organ, concepts):
    """Give each concept that can be organ B's EXPECTATION a STABLE cue-addressable block ([0, n_trained)), in a
    deterministic order, BEFORE any read — so mismatched-asserted/decoy patients (cue_addressable=False) route to
    the novel range and cannot masquerade as a confirm. (No effect on the mechanism; only stabilizes the block map.)"""
    for c in concepts:
        organ._block_for(str(c), cue_addressable=True)


def organ_b_confirms(organ, exp, cand, lesion=False):
    """Organ B's SPIKING corroboration of recall's candidate `cand` against organ B's own expectation `exp`:
    read `cp_firing_states[surprise]` (Hz) for asserting `cand` when expecting `exp`; CONFIRM iff the rate is below
    the organ's OWN calibrated threshold. Returns (confirmed: bool, surprise_hz: float). exp is None -> organ B
    cannot form an expectation for this cue -> no vote."""
    if exp is None:
        return False, float("nan")
    hz = float(organ.read_surprise(str(exp), str(cand), lesion=lesion))
    return bool(hz < organ.threshold), hz


# ── organ B's OWN independent world-model: the associative next-map (the environment/teacher boundary) ────────
def build_e_next(chains, permute=False, rng=None):
    """Organ B's INDEPENDENT expectation e_B(x) = the next concept it expects to follow x (built from the world's
    food-web edges, exactly as the parent's organ C stored its own CONFIRM edges — an independent second memory,
    NOT organ A's live recall). With `permute`, the expected patients are SCRAMBLED (organ B disagrees with the
    world -> the consensus-veto control)."""
    edges = [(a, p) for ch in chains for a, p in zip(ch[:-1], ch[1:])]
    patients = [p for _, p in edges]
    if permute:
        rng.shuffle(patients)
    return {a: patients[i] for i, (a, _p) in enumerate(edges)}


# ── one EVALUATE/COMMIT over the workspace: organ A + (confirmed) organ B + a single-vote decoy ───────────────
def coincidence_hop(bridge, xp, slots_dev, snap, a_cand, b_cand, decoy, d_sub,
                    organ_a=True, organ_b=True, shuffle_rng=None):
    """organ A drives slot(a_cand); organ B drives slot(b_cand) (== a_cand when it CONFIRMED, else None = withheld);
    a spurious DECOY drives its own slot — each at d_sub. Only a slot that receives >= 2 votes crosses the ignition
    knee. WTA + ignition select ONE winner (or none -> abstain). Returns (committed|None, rates, winner, n_ignited,
    b_slot).
      organ_a/organ_b=False: that organ contributes NO drive (the single-organ collapse controls).
      shuffle_rng: route organ B's drive to an EMPTY slot instead of slot(b_cand) (the off-slot congruence control)."""
    slot_of, order = _assign_slots([a_cand, b_cand, decoy])
    n = len(slots_dev)
    drives = [0.0] * n
    if organ_a and a_cand in slot_of:
        drives[slot_of[a_cand]] += d_sub
    b_slot = None
    if organ_b and b_cand is not None and b_cand in slot_of:
        tgt = slot_of[b_cand]
        if shuffle_rng is not None:                       # off-slot: route organ B's (correct-content) vote to an
            empty = [i for i in range(len(order), n)]     # EMPTY slot -> it cannot coincide with organ A at slot(a)
            tgt = int(empty[int(shuffle_rng.integers(len(empty)))]) if empty else tgt
        drives[tgt] += d_sub
        b_slot = tgt
    if decoy in slot_of:
        drives[slot_of[decoy]] += d_sub                   # the single-vote competitor (always present) exercises WTA

    rates = _ignite_and_read(bridge, xp, slots_dev, snap, drives)
    active = list(range(len(order)))
    winner = int(active[int(np.argmax([rates[i] for i in active]))])
    ignited = rates[winner] >= THR
    n_ignited = int(sum(1 for i in active if rates[i] >= THR))
    committed = order[winner] if ignited else None
    return committed, rates, winner, n_ignited, b_slot


def coincidence_chase(bridge, xp, slots_dev, snap, composer, organ, e_next, cue, all_concepts, d_sub, rng,
                      organ_a=True, organ_b=True, shuffle=False, het_lesion=False, max_cycles=None,
                      return_trace=False):
    """The workspace-carried 2-hop DELIBERATION. x starts at cue; each hop: organ A recalls a; organ B reads its
    SPIKING confirmation of a against its own expectation e_B(x); coincidence-ignition EVALUATE/COMMIT; BROADCAST
    BACK (x_next = the committed winner). Abstains (None) the moment organ A misses (moat) or nothing ignites."""
    x = cue
    trace = []
    n_hops = HOPS if max_cycles is None else min(int(max_cycles), HOPS)
    for h in range(n_hops):
        a = composer.query_patient(x, EAT)                # organ A: FHRR recall
        if a is None:                                     # organ A missed -> moat abstains
            trace.append({"hop": h, "x": x, "a": None, "committed": None, "n_ignited": 0})
            return (None, trace) if return_trace else None
        exp = e_next.get(x)                               # organ B's OWN independent expectation for cue x
        confirmed, b_hz = (False, float("nan"))
        if organ_b:
            confirmed, b_hz = organ_b_confirms(organ, exp, a, lesion=het_lesion)   # cp_firing_states[surprise]
        decoy = _pick_decoy(all_concepts, exclude={a, exp, x}, rng=rng)
        sh_rng = rng if shuffle else None
        committed, rates, winner, n_ign, b_slot = coincidence_hop(
            bridge, xp, slots_dev, snap, a, (a if confirmed else None), decoy, d_sub,
            organ_a=organ_a, organ_b=(organ_b and confirmed), shuffle_rng=sh_rng)
        trace.append({"hop": h, "x": x, "a": a, "exp": exp, "b_confirmed": bool(confirmed),
                      "b_surprise_hz": (None if np.isnan(b_hz) else float(b_hz)),
                      "committed": committed, "winner": int(winner), "n_ignited": int(n_ign)})
        if committed is None:                             # no coincidence ignited -> abstain
            return (None, trace) if return_trace else None
        x = committed                                     # BROADCAST BACK: the spike-derived re-cue re-enters
    return (x, trace) if return_trace else x


# ── 1-seed primitive smoke (the whole organ-B path + the coincidence bifurcation, verified in code) ──────────
def run_primitive_smoke(seed, D, d_sub, n_trained, n_novel):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    e_next = build_e_next(CHAINS)
    expected_concepts = sorted({p for _a, p in e_next.items()})

    organ = build_organ_b(seed, n_trained=n_trained, n_novel=n_novel)
    pre_register_expected(organ, expected_concepts)
    calib = organ.calib
    print(f"[smoke] organ B (SurpriseProductionOrgan) calib: confirm_hz={calib['confirm_hz']:.3f} "
          f"threshold={organ.threshold:.3f} contradict_hz={calib['contradict_hz']:.3f} novel_hz={calib['novel_hz']:.3f}",
          flush=True)

    # (1) ORGAN B IS A GENUINE SPIKING READ THAT DISCRIMINATES — assert confirm << threshold << contradict over the
    #     ACTUAL 2-hop (cue, expected, recalled) triples, not the organ's internal calibration blocks.
    agree_hz, disagree_hz = [], []
    for ch in CHAINS:
        for x, want in ((ch[0], ch[1]), (ch[1], ch[2])):
            a = composer.query_patient(x, EAT)
            exp = e_next.get(x)
            if a is None or exp is None:
                continue
            _c1, hz_agree = organ_b_confirms(organ, exp, a)              # a == exp -> CONFIRM (low)
            wrong = _pick_decoy(all_concepts, exclude={a, exp, x}, rng=np.random.default_rng(seed + 1))
            _c2, hz_dis = organ_b_confirms(organ, exp, wrong)           # wrong != exp -> CONTRADICT (high)
            agree_hz.append(hz_agree)
            disagree_hz.append(hz_dis)
    mean_agree = float(np.mean(agree_hz)); mean_disagree = float(np.mean(disagree_hz))
    discriminates = bool(mean_agree < organ.threshold <= mean_disagree)
    print(f"[smoke] organ B SPIKING discrimination over the 2-hop triples: agree_hz={mean_agree:.3f} < "
          f"thr={organ.threshold:.3f} <= disagree_hz={mean_disagree:.3f}  discriminates={discriminates}", flush=True)

    # (2) organ B's SOLO drive is SUBTHRESHOLD (reuse the parent calibration: d_sub solo below the knee) ─────────
    b_i, xp, slots, snap = build_workspace_bridge(seed, lesion=False)
    solo = _ignite_and_read(b_i, xp, slots, snap, [float(d_sub)] + [0.0] * (len(slots) - 1))
    solo_subthreshold = bool(solo[0] < THR)
    coinc = _ignite_and_read(b_i, xp, slots, snap, [2.0 * float(d_sub)] + [0.0] * (len(slots) - 1))
    coinc_ignites = bool(coinc[0] >= THR)
    print(f"[smoke] d_sub solo late-rate={solo[0]:.3f} (subthreshold={solo_subthreshold}); "
          f"2*d_sub={2*d_sub:.0f} late-rate={coinc[0]:.3f} (ignites={coinc_ignites})", flush=True)

    # (3) COINCIDENCE hop: both organs agree -> ignite; A-only / B-only / disagree / lesion -> collapse ──────────
    ch = CHAINS[0]
    x, want = ch[0], ch[1]
    a = composer.query_patient(x, EAT)
    exp = e_next.get(x)
    confirmed, _hz = organ_b_confirms(organ, exp, a)
    decoy = _pick_decoy(all_concepts, exclude={a, exp, x}, rng=np.random.default_rng(seed * 991 + 7))
    com, rates, w, nign, _bs = coincidence_hop(b_i, xp, slots, snap, a, (a if confirmed else None), decoy, d_sub)
    coincidence_ok = bool(com == a and nign == 1 and confirmed)
    com_a, _r, _w, _n, _b = coincidence_hop(b_i, xp, slots, snap, a, a, decoy, d_sub, organ_b=False)   # A-only
    com_b, _r2, _w2, _n2, _b2 = coincidence_hop(b_i, xp, slots, snap, a, a, decoy, d_sub, organ_a=False)  # B-only
    # disagree: organ B's expectation permuted -> it should contradict the true recall candidate -> withhold
    e_perm = build_e_next(CHAINS, permute=True, rng=np.random.default_rng(seed * 101 + 5))
    pre_register_expected(organ, sorted({p for _a, p in e_perm.items()}))
    conf_d, _hzd = organ_b_confirms(organ, e_perm.get(x), a)
    com_d, _rd, _wd, _nd, _bd = coincidence_hop(b_i, xp, slots, snap, a, (a if conf_d else None), decoy, d_sub)
    b_l, xpl, slotsl, snapl = build_workspace_bridge(seed, lesion=True)
    com_l, _rl, _wl, _nl, _bl = coincidence_hop(b_l, xpl, slotsl, snapl, a, a, decoy, d_sub)               # lesion
    print(f"[smoke] COINCIDENCE committed={com!r} (want {a!r}) n_ignited={nign} | A-only={com_a!r} B-only={com_b!r} "
          f"disagree={com_d!r} lesion={com_l!r}", flush=True)

    reflex_ok = bool(composer.query_patient(x, EAT) == want)
    ok = bool(coincidence_ok and discriminates and solo_subthreshold and coinc_ignites and
              com_a is None and com_b is None and com_d is None and com_l is None and reflex_ok)
    print(f"\n[smoke] PRIMITIVE {'HOLDS' if ok else 'FAILS'}: coincidence_ignites_single={coincidence_ok} "
          f"organB_discriminates={discriminates} solo_subthreshold={solo_subthreshold} 2xd_sub_ignites={coinc_ignites} "
          f"A_only_abstains={com_a is None} B_only_abstains={com_b is None} disagree_abstains={com_d is None} "
          f"lesion_kills={com_l is None} reflex_survives={reflex_ok}", flush=True)
    return ok


# ── the per-seed experiment ───────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, D, d_sub, n_trained, n_novel, verbose=True):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    n_concepts = len(all_concepts)
    chance = 1.0 / n_concepts
    chance_ish = max(2.0 * chance, 0.10)

    e_next = build_e_next(CHAINS)                                          # organ B's OWN expectation
    e_perm = build_e_next(CHAINS, permute=True, rng=np.random.default_rng(seed * 101 + 5))
    expected_concepts = sorted({p for _a, p in e_next.items()} | {p for _a, p in e_perm.items()})

    organ = build_organ_b(seed, n_trained=n_trained, n_novel=n_novel)
    pre_register_expected(organ, expected_concepts)

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
        term = coincidence_chase(b_i, xp, slots_i, snap_i, composer, organ, e_next, cue, all_concepts, d_sub, rng())
        coin_ok += int(term == want)
        qc_ok += int(composer.query_chain(cue, [EAT] * HOPS) == want)
        spread_ok += int(spreading_predict(_cooc(seed), cue, HOPS, all_concepts) == want)
    coin_acc = coin_ok / tot
    qc_acc = qc_ok / tot
    spread_floor = spread_ok / tot

    # ── ANTI-CHEATS ─────────────────────────────────────────────────────────────────────────────────────────────
    def chase_acc(e_map=e_next, **kw):
        ok = 0
        for ch in chains2:
            ok += int(coincidence_chase(b_i, xp, slots_i, snap_i, composer, organ, e_map, ch[0], all_concepts,
                                        d_sub, rng(), **kw) == ch[HOPS])
        return ok / tot

    a_only_acc = chase_acc(organ_b=False)                 # single-organ (recall only) -> subthreshold -> collapse
    b_only_acc = chase_acc(organ_a=False)                 # single-organ (surprise-confirm only) -> collapse
    disagree_acc = chase_acc(e_map=e_perm)                # organ B's expectation permuted -> consensus-veto
    shuffle_acc = chase_acc(shuffle=True)                 # organ B off-target -> collapse
    onecycle_acc = chase_acc(max_cycles=1)                # single-shot: only hop-1 -> 2-hop unreachable -> collapse
    het_lesion_acc = chase_acc(het_lesion=True)           # organ B's OWN spiking prediction zeroed -> collapse

    # no-ignition workspace lesion (collapse) + the dissociation keystone (single-hop reflex survives)
    lesion_ok = reflex_ok = 0
    for ch in chains2:
        lesion_ok += int(coincidence_chase(b_l, xp_l, slots_l, snap_l, composer, organ, e_next, ch[0],
                                           all_concepts, d_sub, rng()) == ch[HOPS])
        reflex_ok += int(composer.query_patient(ch[0], EAT) == ch[1])
    lesion_acc = lesion_ok / tot
    reflex_acc = reflex_ok / tot

    # MOAT: unstored cue + past-chain-end over-run -> organ A misses -> abstain
    moat_unstored = coincidence_chase(b_i, xp, slots_i, snap_i, composer, organ, e_next, "ball", all_concepts,
                                      d_sub, rng())
    moat_over = coincidence_chase(b_i, xp, slots_i, snap_i, composer, organ, e_next, CHAINS[0][-1], all_concepts,
                                  d_sub, rng())
    moat_unstored_abstains = moat_unstored is None
    moat_over_abstains = moat_over is None
    moat_ok = bool(moat_unstored_abstains and moat_over_abstains)

    # organ B's SPIKING read discriminates over the ACTUAL 2-hop triples (the instrument precondition) + ME diag
    agree_hz, disagree_hz = [], []
    me_single = me_total = 0
    for ch in chains2:
        _t, tr = coincidence_chase(b_i, xp, slots_i, snap_i, composer, organ, e_next, ch[0], all_concepts, d_sub,
                                   rng(), return_trace=True)
        for step in tr:
            if step.get("committed") is not None:
                me_total += 1
                me_single += int(step["n_ignited"] == 1)
            if step.get("b_surprise_hz") is not None:
                agree_hz.append(step["b_surprise_hz"])
        # a matched contradict read per chain (organ B's expectation is correct, assert a WRONG continuation)
        a0 = composer.query_patient(ch[0], EAT)
        exp0 = e_next.get(ch[0])
        if a0 is not None and exp0 is not None:
            wrong = _pick_decoy(all_concepts, exclude={a0, exp0, ch[0]}, rng=np.random.default_rng(seed + 2))
            _cd, hzd = organ_b_confirms(organ, exp0, wrong)
            disagree_hz.append(hzd)
    me_frac = (me_single / me_total) if me_total else 0.0
    mean_agree_hz = float(np.mean(agree_hz)) if agree_hz else float("nan")
    mean_disagree_hz = float(np.mean(disagree_hz)) if disagree_hz else float("nan")
    organ_b_discriminates = bool(mean_agree_hz < organ.threshold <= mean_disagree_hz)

    # instrument preconditions: organ B's read works AND d_sub sits in the coincidence window
    solo = _ignite_and_read(b_i, xp, slots_i, snap_i, [float(d_sub)] + [0.0] * (len(slots_i) - 1))
    coinc = _ignite_and_read(b_i, xp, slots_i, snap_i, [2.0 * float(d_sub)] + [0.0] * (len(slots_i) - 1))
    d_sub_in_window = bool(solo[0] < THR <= coinc[0])
    precondition_ok = bool(organ_b_discriminates and d_sub_in_window)

    seed_go = bool(
        precondition_ok and
        coin_acc >= 0.75 and
        coin_acc >= spread_floor + 0.5 and
        coin_acc >= qc_acc and                            # parity with the host one-shot (same conclusion, synaptic path)
        a_only_acc <= chance_ish and                      # a single organ read is subthreshold (the anti-if-else)
        b_only_acc <= chance_ish and
        disagree_acc <= chance_ish and                    # conflicting organs -> withhold (consensus-veto)
        shuffle_acc <= chance_ish and                     # combination is congruence, not slot
        onecycle_acc <= chance_ish and                    # re-entry load-bearing (single-shot can't)
        het_lesion_acc <= chance_ish and                  # organ B's OWN spiking prediction load-bearing
        lesion_acc <= chance_ish and                      # workspace ignition load-bearing
        reflex_acc >= 0.85 and                            # the single-hop recall reflex survives (dissociation)
        moat_ok
    )

    result = {
        "seed": int(seed), "D": int(D), "d_sub": float(d_sub), "hops": HOPS, "n_concepts": n_concepts,
        "chance": chance, "n_chains": tot, "n_trained": int(n_trained), "n_novel": int(n_novel),
        "organ_a": "rf_phasor_composer.query_patient(EAT)",
        "organ_b": "surprise_production_organ.SurpriseProductionOrgan.read_surprise(cp_firing_states[surprise])",
        "coincidence_2hop_acc": coin_acc, "query_chain_2hop_acc": qc_acc, "spreading_floor": spread_floor,
        "a_only_acc": a_only_acc, "b_only_acc": b_only_acc, "disagree_acc": disagree_acc,
        "shuffle_acc": shuffle_acc, "onecycle_acc": onecycle_acc, "organ_b_lesion_acc": het_lesion_acc,
        "workspace_lesion_acc": lesion_acc, "single_hop_reflex_acc": reflex_acc,
        "moat_unstored_abstains": moat_unstored_abstains, "moat_over_abstains": moat_over_abstains, "moat_ok": moat_ok,
        "mutual_exclusion_frac": me_frac,
        "organ_b_threshold_hz": float(organ.threshold), "organ_b_mean_agree_hz": mean_agree_hz,
        "organ_b_mean_disagree_hz": mean_disagree_hz, "organ_b_discriminates": organ_b_discriminates,
        "organ_b_calib": organ.calib,
        "d_sub_solo_rate": float(solo[0]), "d_sub_coincidence_rate": float(coinc[0]), "d_sub_in_window": d_sub_in_window,
        "precondition_ok": precondition_ok, "seed_go": seed_go,
    }
    if verbose:
        print(f"[2organ seed={seed} D={D} d_sub={d_sub:.0f}] coincidence_2hop={coin_acc:.3f} "
              f"vs query_chain={qc_acc:.3f} (spread_floor={spread_floor:.3f}, chance={chance:.3f})", flush=True)
        print(f"    organ B SPIKING read (NON-COMPOSER, cp_firing_states[surprise]): agree_hz={mean_agree_hz:.3f} "
              f"< thr={organ.threshold:.3f} <= disagree_hz={mean_disagree_hz:.3f} discriminates={organ_b_discriminates} "
              f"| d_sub_in_window={d_sub_in_window}", flush=True)
        print(f"    INTEGRATION collapses: A_only={a_only_acc:.3f} B_only={b_only_acc:.3f} disagree={disagree_acc:.3f} "
              f"shuffle={shuffle_acc:.3f} organ_b_lesion={het_lesion_acc:.3f}", flush=True)
        print(f"    RE-ENTRY/IGNITION: onecycle={onecycle_acc:.3f} workspace_lesion={lesion_acc:.3f} | "
              f"reflex_survives={reflex_acc:.3f} | moat unstored={moat_unstored_abstains} over={moat_over_abstains} "
              f"| ME_single={me_frac:.3f}", flush=True)
        print(f"    precondition_ok={precondition_ok} seed_GO={seed_go}", flush=True)
    return result


# cooc cache (the spreading floor is computed from the SAME distractor-polluted co-occurrence graph as the parent)
_COOC_CACHE = {}


def _cooc(seed):
    if seed not in _COOC_CACHE:
        vocab = build_vocab()
        comp = RFPhasorComposer(seed=seed, D=64, vocab=vocab)
        _edges, cooc = store_facts(comp, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
        _COOC_CACHE[seed] = cooc
    return _COOC_CACHE[seed]


def main():
    ap = argparse.ArgumentParser(description="GNW coincidence-integrator with TWO GENUINELY DIFFERENT organs "
                                             "(composer recall + spiking surprise/expectation-violation monitor).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--d-sub", type=float, default=D_SUB_DEFAULT)
    ap.add_argument("--n-trained", type=int, default=N_TRAINED_DEFAULT)
    ap.add_argument("--n-novel", type=int, default=N_NOVEL_DEFAULT)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_two_distinct_organs/summary.json")
    args = ap.parse_args()

    if args.backend != "auto":
        from sim.backend import get_backend
        get_backend(args.backend)

    if args.calibrate:
        build_vocab()
        cal_json = args.json.replace("summary.json", "calibration_seed%d.json" % args.seed)
        return 0 if run_calibrate(args.seed, args.D, args.d_sub, json_path=cal_json) else 1
    if args.smoke:
        return 0 if run_primitive_smoke(args.seed, args.D, args.d_sub, args.n_trained, args.n_novel) else 1

    n_concepts = len({c for ch in CHAINS for c in ch})
    print(f"[gnw-two-distinct-organs] {len(CHAINS)} chains | {n_concepts} concepts | chance {1.0/n_concepts:.3f} | "
          f"K_slots={K_SLOTS} D={args.D} d_sub={args.d_sub:.0f} organ_B(n_trained={args.n_trained},"
          f"n_novel={args.n_novel}) backend={args.backend}\n"
          "  organ A = FHRR composer recall; organ B = spiking expectation-violation monitor "
          "(cp_firing_states[surprise], NON-COMPOSER).\n", flush=True)

    results = [run_seed(s, args.D, args.d_sub, args.n_trained, args.n_novel) for s in args.seeds]
    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)
    all_precond = all(r["precondition_ok"] for r in results)

    def mean(k):
        return float(np.mean([r[k] for r in results]))

    # the integration attribution: what fraction of the 2-hop success needs BOTH organs (vs the best single organ)?
    print("\n── integration attribution (tools.lab.attributable_to) ──", flush=True)
    best_single = max(mean("a_only_acc"), mean("b_only_acc"))
    void_if(mean("coincidence_2hop_acc") <= 1e-9, "intact coincidence chase is ~0 — nothing to attribute")
    attributable_to("2-hop success needs BOTH distinct organs", mean("coincidence_2hop_acc"), best_single)

    summary = {
        "runner": "_gnw_two_distinct_organs_derisk",
        "claim": ("the spiking workspace COMBINES two subthreshold reads from TWO GENUINELY DIFFERENT organs "
                  "(FHRR composer recall + a spiking predictive-coding expectation-violation monitor) via "
                  "coincidence-ignition + 2-hop re-entry — closing the parent finding's caveat #1"),
        "seeds": list(args.seeds), "D": int(args.D), "d_sub": float(args.d_sub),
        "n_trained": int(args.n_trained), "n_novel": int(args.n_novel), "backend": args.backend,
        "all_go": all_go, "n_go": n_go, "n_seeds": len(results), "all_precondition_ok": all_precond,
        "mean_coincidence_2hop_acc": mean("coincidence_2hop_acc"),
        "mean_query_chain_2hop_acc": mean("query_chain_2hop_acc"),
        "mean_spreading_floor": mean("spreading_floor"),
        "mean_a_only_acc": mean("a_only_acc"), "mean_b_only_acc": mean("b_only_acc"),
        "mean_disagree_acc": mean("disagree_acc"), "mean_shuffle_acc": mean("shuffle_acc"),
        "mean_onecycle_acc": mean("onecycle_acc"), "mean_organ_b_lesion_acc": mean("organ_b_lesion_acc"),
        "mean_workspace_lesion_acc": mean("workspace_lesion_acc"),
        "mean_single_hop_reflex_acc": mean("single_hop_reflex_acc"),
        "mean_mutual_exclusion_frac": mean("mutual_exclusion_frac"),
        "mean_organ_b_mean_agree_hz": mean("organ_b_mean_agree_hz"),
        "mean_organ_b_mean_disagree_hz": mean("organ_b_mean_disagree_hz"),
        "all_organ_b_discriminate": all(r["organ_b_discriminates"] for r in results),
        "all_d_sub_in_window": all(r["d_sub_in_window"] for r in results),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "per_seed": results,
    }
    out_dir = os.path.dirname(os.path.abspath(args.json))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    for r in results:                                     # per-seed artifacts alongside the aggregate
        with open(os.path.join(out_dir, "seed%d.json" % r["seed"]), "w") as f:
            json.dump(r, f, indent=2, default=str)

    if all_go:
        verdict = "GO"
    elif not all_precond:
        verdict = "UNDEFINED (instrument precondition failed)"
    elif n_go >= 1:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"
    print(f"\n{'='*100}", flush=True)
    print(f"  GNW TWO-DISTINCT-ORGANS VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    print(f"    coincidence_2hop={summary['mean_coincidence_2hop_acc']:.3f} vs query_chain="
          f"{summary['mean_query_chain_2hop_acc']:.3f} (spread_floor={summary['mean_spreading_floor']:.3f})", flush=True)
    print(f"    organ B SPIKING (cp_firing_states[surprise]): agree_hz={summary['mean_organ_b_mean_agree_hz']:.3f} "
          f"disagree_hz={summary['mean_organ_b_mean_disagree_hz']:.3f} discriminate_all="
          f"{summary['all_organ_b_discriminate']} | d_sub_in_window_all={summary['all_d_sub_in_window']}", flush=True)
    print(f"    collapses: A_only={summary['mean_a_only_acc']:.3f} B_only={summary['mean_b_only_acc']:.3f} "
          f"disagree={summary['mean_disagree_acc']:.3f} shuffle={summary['mean_shuffle_acc']:.3f} "
          f"onecycle={summary['mean_onecycle_acc']:.3f} organ_b_lesion={summary['mean_organ_b_lesion_acc']:.3f} "
          f"workspace_lesion={summary['mean_workspace_lesion_acc']:.3f}", flush=True)
    print(f"    reflex_survives={summary['mean_single_hop_reflex_acc']:.3f} moat_all={summary['all_moat_ok']} "
          f"ME_single={summary['mean_mutual_exclusion_frac']:.3f}", flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
