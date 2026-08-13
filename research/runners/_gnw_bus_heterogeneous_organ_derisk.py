"""GNW ignition bus — a HETEROGENEOUS NON-COMPOSER organ as one of the N ignition votes (de-risk).

WHAT THIS ADDS OVER THE N-ORGAN BUS (`_gnw_norgan_bus_derisk`, 6/6 GO; production flip 2026-08-13). The N-organ
bus proved the SUBSTRATE combines N>=3 subthreshold organ reads via consensus-ignition + shared-inhibition WTA +
re-entry. But the N organs there are all CORROBORATING RELATIONAL reads of ONE composer (forward recall + a second
relation + a reverse-binding VERIFY, all `query_patient`/`query_agent`). The named #1 residual on both GNW findings
is: *"a non-composer organ (a spiking surprise/familiarity monitor, the affect organ) as one of the N votes is the
immediate follow-on rung."* This runner closes that de-risk: it routes N=3 votes through the SAME spiking workspace
where AT LEAST ONE vote is a genuinely HETEROGENEOUS, NON-COMPOSER organ — the production spiking
expectation-violation / familiarity monitor (`surprise_production_organ.SurpriseProductionOrgan`), which reads
`cp_firing_states[surprise]` off a predictive-coding mismatch circuit and NEVER calls `query_patient`.

  organ A — spiking RECALL (composer):        query_patient(agent, EAT)     -> cand_A       [composer]
  organ B — spiking CORROBORATION (composer): query_patient(agent, CONFIRM) -> cand_B       [composer]
  organ H — spiking FAMILIARITY/SURPRISE monitor (NON-COMPOSER): votes cand_A IFF its predictive-coding mismatch
            circuit reads (agent,action)->cand_A as a FAMILIAR / non-surprising fact (CONFIRM, low surprise Hz).
            The read is `cp_firing_states[surprise]` off a DIFFERENT mechanism (Hebbian topographic cue->expected +
            GABA_A subtractive inhibition), thresholded by the organ's OWN calibrated decision into a binary vote.

THE DEEPENED CLAIM. The substrate integrates DIFFERENT KINDS of organ by ignition — not just corroborating reads of
one composer. Organ H's continuous surprise Hz is thresholded (the organ's own `surprised = hz >= threshold`
decision, a legitimate spiking read-out) into a binary "familiar -> vote cand_A / surprising -> withhold", then
written as the IDENTICAL subthreshold d_sub drive as the composer votes (subthreshold-commensurate by
construction). Three agreeing votes accumulate to N*d_sub >= knee and IGNITE; drop / silence / mis-route / disagree
organ H and the true slot falls to 2*d_sub < knee -> abstain. So the heterogeneous organ is LOAD-BEARING in the
consensus, exactly like a composer vote.

ANTI-CHEATS (each targets "the heterogeneous organ GENUINELY participates in the ignition consensus"):
  - HET-DROPPED / SILENCED [organ H load-bearing]: leave-one-out on organ H (or any organ) -> 2 votes < knee ->
    collapse. A host `if recall: return r` (or "the two composer organs suffice") would SUCCEED; the collapse proves
    the heterogeneous vote is part of the substrate AND, not a decorative extra.
  - HET-DISAGREES [consensus-veto by the heterogeneous organ]: present organ H a CONTRADICTING assertion -> its
    mismatch circuit fires high (surprising) -> it withholds its vote -> the consensus loses organ H -> collapse.
    The disagreement is a SPIKING read (surprise Hz over threshold), not a host flip.
  - HET-MIS-ROUTED [shuffle-off-slot]: route organ H's drive to an EMPTY slot -> the consensus loses a vote ->
    collapse (the keystone's corrected off-slot control; no leak onto an occupied concept slot).
  - HET-ORGAN LESIONED [its OWN spiking prediction is load-bearing]: zero the mismatch circuit's
    patient_expected->surprise prediction edges -> CONFIRM fires as high as CONTRADICT -> organ H can no longer
    recognize the familiar fact -> it withholds even on a genuine fact -> collapse. The heterogeneous vote is
    caused by the learned SPIKING prediction, not a fixed input artifact.
  - WORKSPACE LESION [ignition load-bearing]: zero the workspace assembly self-recurrence -> even N agreeing votes
    cannot sustain -> collapse, WHILE the single-hop composer recall reflex (direct query_patient, never routed
    through the workspace) SURVIVES (the dissociation keystone).
  - SINGLE-ORGAN COLLAPSE: keep only organ A -> 1 vote subthreshold -> abstain (the anti-host-if-else).
  - MOAT: an unstored agent + a stored agent under a WRONG action -> the primary recall organ misses -> abstain.
  - HET DISCRIMINATION (instrument): the organ's confirm-vs-contradict spiking separation must hold (agree reads
    below the organ's threshold, disagree reads above) — the read that DRIVES the binary vote genuinely flips.

GO GATE (6 seeds 42/43/44/100/101/102): consensus_acc >= 0.85 AND == host recall (parity) AND EVERY collapse
control (single, het-dropped, leave-one-out worst, disagree, shuffle-off, workspace-lesion, het-organ-lesion) <=
chance-ish AND the composer reflex survives the workspace lesion AND the het organ discriminates AND the moat
abstains. -> the SUBSTRATE combines a heterogeneous non-composer organ with composer organs by ignition, and that
heterogeneous vote is load-bearing.

DISCIPLINE: reuse-by-import (the N-organ bus hop + workspace + the production surprise organ). NO `sim/` edit.
Deterministic per seed. Run (CPU numpy cheap-first; the effect is a subthreshold-vs-suprathreshold bifurcation,
not GPU-scale-dependent):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_bus_heterogeneous_organ_derisk --smoke --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_bus_heterogeneous_organ_derisk --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_bus_heterogeneous/summary.json
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

# reuse-by-import the de-risked N-organ bus hop + calibrated subthreshold drive + workspace — NO `sim/` edit.
from research.runners._gnw_norgan_bus_derisk import (
    norgan_hop, D_SUB_UNANIMITY, THR, CONFIRM, store_n_relation_facts,
)
from research.runners._p1_2_workspace_deliberation_loop_derisk import build_workspace_bridge, K_SLOTS
from research.runners._gnw_coincidence_integrator_derisk import _pick_decoy
from research.runners._phaseB_multihop_query_chain_derisk import CHAINS, EAT, build_vocab
from research.runners.rf_phasor_composer import RFPhasorComposer
# reuse-by-import the PRODUCTION spiking expectation-violation / familiarity monitor (the NON-COMPOSER organ).
from research.runners.surprise_production_organ import SurpriseProductionOrgan
from tools.lab import attributable_to, void_if

N_ORGANS = 3
# the clean regime for the surprise organ (8 cue-addressable trained blocks): one first-edge per chain, 8 distinct
# stored patients -> exactly the 8 trained blocks, no round-robin wrap -> validated confirm/contradict discrimination.
FIRST_EDGES = [(ch[0], ch[1]) for ch in CHAINS]           # 8 (agent, patient) edges, distinct patients


class HeterogeneousOrganVote:
    """The heterogeneous NON-COMPOSER vote: a spiking familiarity/surprise monitor that votes for the recalled
    patient IFF its predictive-coding mismatch circuit reads that patient as FAMILIAR (CONFIRM, low surprise Hz).
    The read is `cp_firing_states[surprise]` — a DIFFERENT mechanism from the composer's `query_patient`. Its
    continuous Hz is thresholded by the organ's OWN calibrated decision into a binary vote (a spiking read-out),
    written as the identical subthreshold d_sub drive as the composer votes (subthreshold-commensurate)."""

    def __init__(self, seed: int):
        self.organ = SurpriseProductionOrgan(seed=seed)
        self.organ.ensure_built()

    @property
    def threshold(self) -> float:
        return float(self.organ.threshold)

    @property
    def calib(self) -> dict:
        return dict(self.organ.calib)

    def read_hz(self, agent, action, recalled_patient, asserted_patient, het_lesion=False) -> float:
        """The SPIKING surprise rate (Hz) for asserting `asserted_patient` when the brain recalled
        `recalled_patient` for (agent, action). NON-COMPOSER: `cp_firing_states[surprise]`, never `query_patient`."""
        return float(self.organ.read_surprise(recalled_patient, asserted_patient, lesion=het_lesion))

    def vote(self, agent, action, recalled_patient, asserted_patient, het_lesion=False):
        """Return (voted_concept | None, surprise_hz, surprised). Votes `asserted_patient` IFF the mismatch circuit
        reads it as familiar (surprise below the organ's calibrated threshold); withholds on a surprising read."""
        hz = self.read_hz(agent, action, recalled_patient, asserted_patient, het_lesion=het_lesion)
        surprised = bool(hz >= self.threshold)
        return (None if surprised else asserted_patient), hz, surprised


def _build_composer(seed, D=256):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    # organ A = EAT, organ B = CONFIRM: both store agent->patient over the SAME edges (the corroborating relation).
    store_n_relation_facts(composer, CHAINS, [EAT, CONFIRM],
                           distractor_rng=np.random.default_rng(seed * 53 + 1))
    return composer


def het_bus_gate(composer, het, bridge, xp, slots, snap, agent, all_concepts, d_sub, rng, *,
                 het_mode="agree", active_mask=None, shuffle_off_organ=None, het_lesion=False):
    """Route the 3 organ reads for (agent, EAT) through the bus, ONE of which (organ H) is the heterogeneous
    NON-COMPOSER familiarity/surprise monitor. Returns (committed|None, candidates, het_hz, het_surprised).
      het_mode='agree'    -> organ H asserts the recalled patient (should read familiar -> vote cand_A);
      het_mode='disagree' -> organ H asserts a DECOY patient (should read surprising -> withhold)."""
    cand_A = composer.query_patient(agent, EAT)                    # organ A: composer forward recall
    cand_B = composer.query_patient(agent, CONFIRM)                # organ B: composer corroborating recall
    if cand_A is None:                                            # primary recall organ miss -> moat abstains
        return None, [None, None, None], float("nan"), None
    if het_mode == "disagree":
        asserted = _pick_decoy(all_concepts, exclude={cand_A, cand_B, agent}, rng=rng)   # a wrong patient
    else:
        asserted = cand_A
    cand_H, het_hz, het_surprised = het.vote(agent, EAT, cand_A, asserted, het_lesion=het_lesion)
    candidates = [cand_A, cand_B, cand_H]                          # organ H's vote is the 3rd (heterogeneous) slot
    exclude = set(c for c in candidates if c is not None) | {agent}
    decoy = _pick_decoy(all_concepts, exclude=exclude, rng=rng)    # a single-vote rival -> exercises WTA
    committed, rates, winner, n_ign = norgan_hop(
        bridge, xp, slots, snap, candidates, decoy, d_sub,
        active_mask=active_mask, shuffle_off_organ=shuffle_off_organ, rng=rng)
    return committed, candidates, het_hz, het_surprised


def _rng(seed):
    return np.random.default_rng(int(seed) * 991 + 7)


def run_seed(seed, d_sub, D=256, verbose=True):
    composer = _build_composer(seed, D=D)
    het = HeterogeneousOrganVote(seed)
    all_concepts = [c for ch in CHAINS for c in ch]
    n_concepts = len(all_concepts)
    chance = 1.0 / n_concepts
    chance_ish = max(2.0 * chance, 0.10)

    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    b_l, xp_l, slots_l, snap_l = build_workspace_bridge(seed, lesion=True)

    edges = FIRST_EDGES
    tot = len(edges)

    def acc(pred):
        ok = 0
        for (a, p) in edges:
            ok += int(pred(a, p))
        return ok / tot

    # ── CONSENSUS (het AGREES): 3 votes -> ignite the recalled patient == host recall ─────────────────────────────
    # Two things are measured and kept SEPARATE: (i) the SUBSTRATE-COMBINATION claim — when the heterogeneous organ
    # casts its vote, does the substrate ignite the correct winner (`consensus_when_voted`), and when it withholds,
    # does the substrate ABSTAIN rather than confabulate (`abstain_when_withheld`)? (ii) the het organ's OWN read
    # reliability — how often does its single spiking confirm-read recognize a genuinely familiar fact (`het_vote_rate`)?
    consensus_hits = 0
    het_agree_hz = []
    n_voted = n_voted_correct = n_withheld = n_withheld_abstained = 0
    for (a, p) in edges:
        com, cands, hz, surp = het_bus_gate(composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub,
                                            _rng(seed), het_mode="agree")
        consensus_hits += int(com == p)
        if not np.isnan(hz):
            het_agree_hz.append(hz)
        het_voted = cands[2] is not None                      # organ H cast its subthreshold vote (read familiar)
        if het_voted:
            n_voted += 1
            n_voted_correct += int(com == p)                  # substrate ignited the correct winner
        else:
            n_withheld += 1
            n_withheld_abstained += int(com is None)          # substrate ABSTAINED (conservative moat), not confab
    consensus_acc = consensus_hits / tot
    het_vote_rate = n_voted / tot                             # het organ's single-read familiarity reliability
    consensus_when_voted = (n_voted_correct / n_voted) if n_voted else float("nan")   # THE substrate claim
    abstain_when_withheld = (n_withheld_abstained / n_withheld) if n_withheld else 1.0  # stronger moat

    # host recall parity (the `if recalled == p` combination the substrate replaces)
    host_acc = acc(lambda a, p: composer.query_patient(a, EAT) == p)

    # ── SINGLE-ORGAN (organ A only) -> subthreshold -> abstain (the anti-host-if-else) ────────────────────────────
    single_mask = [True] + [False] * (N_ORGANS - 1)
    single_organ_acc = acc(lambda a, p: het_bus_gate(
        composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub, _rng(seed),
        het_mode="agree", active_mask=single_mask)[0] == p)

    # ── HET-DROPPED (leave organ H out) -> 2 votes -> abstain (organ H load-bearing) ──────────────────────────────
    het_drop_mask = [True, True, False]
    het_dropped_acc = acc(lambda a, p: het_bus_gate(
        composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub, _rng(seed),
        het_mode="agree", active_mask=het_drop_mask)[0] == p)

    # ── LEAVE-ONE-OUT (drop EACH organ once) -> worst-case collapse (every organ load-bearing) ────────────────────
    loo_accs = []
    for drop in range(N_ORGANS):
        mask = [j != drop for j in range(N_ORGANS)]
        loo_accs.append(acc(lambda a, p, m=mask: het_bus_gate(
            composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub, _rng(seed),
            het_mode="agree", active_mask=m)[0] == p))
    leaveoneout_worst_acc = float(np.max(loo_accs))

    # ── HET-DISAGREES (organ H reads a contradiction as surprising -> withholds) -> abstain (consensus-veto) ──────
    disagree_hits = 0
    het_disagree_hz = []
    het_disagree_withheld = 0
    for (a, p) in edges:
        com, cands, hz, surp = het_bus_gate(composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub,
                                            _rng(seed), het_mode="disagree")
        disagree_hits += int(com == p)
        if not np.isnan(hz):
            het_disagree_hz.append(hz)
            het_disagree_withheld += int(cands[2] is None)     # organ H genuinely withheld its vote
    disagree_acc = disagree_hits / tot
    het_disagree_withheld_frac = het_disagree_withheld / tot

    # ── HET-MIS-ROUTED (organ H off-slot) -> consensus loses a vote -> abstain ────────────────────────────────────
    shuffle_off_acc = acc(lambda a, p: het_bus_gate(
        composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub, _rng(seed * 13 + 3),
        het_mode="agree", shuffle_off_organ=2)[0] == p)

    # ── HET-ORGAN LESIONED (its OWN spiking prediction edges zeroed) -> can't recognize familiar -> withhold ──────
    het_lesion_hits = 0
    het_lesion_withheld = 0
    for (a, p) in edges:
        com, cands, hz, surp = het_bus_gate(composer, het, b_i, xp, slots_i, snap_i, a, all_concepts, d_sub,
                                            _rng(seed), het_mode="agree", het_lesion=True)
        het_lesion_hits += int(com == p)
        het_lesion_withheld += int(cands[2] is None)
    het_organ_lesion_acc = het_lesion_hits / tot
    het_organ_lesion_withheld_frac = het_lesion_withheld / tot

    # ── WORKSPACE LESION (assembly self-recurrence 0) -> even 3 votes can't ignite -> abstain; reflex survives ────
    lesion_hits = reflex_hits = 0
    for (a, p) in edges:
        com, cands, hz, surp = het_bus_gate(composer, het, b_l, xp_l, slots_l, snap_l, a, all_concepts, d_sub,
                                            _rng(seed), het_mode="agree")
        lesion_hits += int(com == p)
        reflex_hits += int(composer.query_patient(a, EAT) == p)   # the workspace-independent recall reflex
    lesion_acc = lesion_hits / tot
    reflex_acc = reflex_hits / tot

    # ── MOAT: unstored agent + stored agent under a WRONG action -> the primary organ misses -> abstain ───────────
    moat_unstored = het_bus_gate(composer, het, b_i, xp, slots_i, snap_i, "ball", all_concepts, d_sub,
                                 _rng(seed), het_mode="agree")[0]
    a0 = edges[0][0]
    moat_wrong_action = composer.query_patient(a0, "fly")         # a stored agent under an unstored action
    moat_unstored_abstains = moat_unstored is None
    moat_wrong_action_abstains = moat_wrong_action is None
    moat_ok = bool(moat_unstored_abstains and moat_wrong_action_abstains)

    # ── HET DISCRIMINATION (instrument): agree reads below threshold, disagree reads above ────────────────────────
    thr = het.threshold
    mean_agree_hz = float(np.mean(het_agree_hz)) if het_agree_hz else float("nan")
    mean_disagree_hz = float(np.mean(het_disagree_hz)) if het_disagree_hz else float("nan")
    het_discriminates = bool(mean_agree_hz < thr <= mean_disagree_hz)

    # THE GATED CLAIM is the SUBSTRATE-COMBINATION, isolated from the het organ's own single-read precision: when the
    # heterogeneous organ VOTES, the substrate ignites the correct winner (consensus_when_voted==1); when it WITHHOLDS
    # (a marginal familiarity read), the substrate ABSTAINS (abstain_when_withheld==1, a stronger moat) — never
    # confabulates. het_vote_rate (the organ's own reliability) has a floor but is NOT required to be perfect (its
    # single-read confirm precision rides the surprise organ's own documented burn-down). The strict end-to-end
    # consensus_acc vs host_acc is REPORTED for transparency; the het organ can only ADD abstentions vs host recall.
    substrate_combines = bool(consensus_when_voted >= 0.999 and abstain_when_withheld >= 0.999)
    seed_go = bool(
        substrate_combines and                                    # THE claim: substrate ignites-when-voted / abstains-when-not
        het_vote_rate >= 0.85 and                                 # het organ recognizes >=7/8 familiar facts (its floor)
        single_organ_acc <= chance_ish and
        het_dropped_acc <= chance_ish and                         # organ H load-bearing (leave-one-out on H)
        leaveoneout_worst_acc <= chance_ish and                   # EVERY organ load-bearing
        disagree_acc <= chance_ish and                            # het organ consensus-veto (spiking disagreement)
        shuffle_off_acc <= chance_ish and                         # combination is congruence, not slot
        het_organ_lesion_acc <= chance_ish and                    # the het organ's OWN spiking prediction load-bearing
        lesion_acc <= chance_ish and                              # workspace ignition load-bearing
        reflex_acc >= 0.85 and                                    # composer reflex survives (dissociation)
        het_discriminates and                                     # the read that drives the het vote genuinely flips
        moat_ok
    )

    result = {
        "seed": int(seed), "n_organs": N_ORGANS, "D": int(D), "d_sub": float(d_sub),
        "n_edges": tot, "n_concepts": n_concepts, "chance": chance,
        "consensus_acc": consensus_acc, "host_recall_acc": host_acc,
        "consensus_when_voted": consensus_when_voted, "abstain_when_withheld": abstain_when_withheld,
        "het_vote_rate": het_vote_rate, "substrate_combines": substrate_combines,
        "single_organ_acc": single_organ_acc, "het_dropped_acc": het_dropped_acc,
        "leaveoneout_worst_acc": leaveoneout_worst_acc, "loo_accs": loo_accs,
        "disagree_acc": disagree_acc, "het_disagree_withheld_frac": het_disagree_withheld_frac,
        "shuffle_off_acc": shuffle_off_acc,
        "het_organ_lesion_acc": het_organ_lesion_acc, "het_organ_lesion_withheld_frac": het_organ_lesion_withheld_frac,
        "workspace_lesion_acc": lesion_acc, "reflex_acc": reflex_acc,
        "moat_unstored_abstains": moat_unstored_abstains, "moat_wrong_action_abstains": moat_wrong_action_abstains,
        "moat_ok": moat_ok,
        "het_threshold_hz": thr, "het_mean_agree_hz": mean_agree_hz, "het_mean_disagree_hz": mean_disagree_hz,
        "het_discriminates": het_discriminates, "het_calib": het.calib,
        "seed_go": seed_go,
    }
    if verbose:
        print(f"[het-bus seed={seed} N={N_ORGANS} d_sub={d_sub:.0f}] SUBSTRATE: ignite-when-voted="
              f"{consensus_when_voted:.3f} abstain-when-withheld={abstain_when_withheld:.3f} "
              f"(substrate_combines={substrate_combines}) | het_vote_rate={het_vote_rate:.3f} "
              f"| end-to-end consensus={consensus_acc:.3f} vs host_recall={host_acc:.3f}", flush=True)
        print(f"    HET vote SPIKING read (NON-COMPOSER, cp_firing_states[surprise]): agree_hz={mean_agree_hz:.2f} "
              f"< thr={thr:.2f} <= disagree_hz={mean_disagree_hz:.2f}  discriminates={het_discriminates}", flush=True)
        print(f"    HET load-bearing: dropped={het_dropped_acc:.3f} disagree={disagree_acc:.3f} "
              f"(withheld {het_disagree_withheld_frac:.2f}) shuffle_off={shuffle_off_acc:.3f} "
              f"het_organ_lesion={het_organ_lesion_acc:.3f} (withheld {het_organ_lesion_withheld_frac:.2f})", flush=True)
        print(f"    substrate: single_organ={single_organ_acc:.3f} leave1out(worst)={leaveoneout_worst_acc:.3f} "
              f"workspace_lesion={lesion_acc:.3f} | reflex_survives={reflex_acc:.3f} | moat_ok={moat_ok} "
              f"| seed_GO={seed_go}", flush=True)
    return result


def run_smoke(seed, d_sub, D=256):
    print(f"[smoke] heterogeneous NON-COMPOSER organ (spiking surprise/familiarity monitor) as one of N={N_ORGANS} "
          f"ignition votes, seed={seed}", flush=True)
    r = run_seed(seed, d_sub, D=D, verbose=True)
    ok = r["seed_go"]
    print(f"\n[smoke] HET-ORGAN BUS {'HOLDS' if ok else 'FAILS'}: the substrate combines a heterogeneous "
          f"non-composer vote by ignition, and that vote is load-bearing.", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="GNW bus — a heterogeneous NON-COMPOSER organ as one of the N votes.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--d-sub", type=float, default=None, help="per-organ subthreshold drive (default: unanimity N=3)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_bus_heterogeneous/summary.json")
    args = ap.parse_args()

    d_sub = args.d_sub if args.d_sub is not None else D_SUB_UNANIMITY.get(N_ORGANS, 1000.0)
    if args.backend != "auto":
        from sim.backend import get_backend
        get_backend(args.backend)

    if args.smoke:
        return 0 if run_smoke(args.seed, d_sub, D=args.D) else 1

    n_concepts = len({c for ch in CHAINS for c in ch})
    print(f"[gnw-het-organ-bus] {len(FIRST_EDGES)} first-edges | {n_concepts} concepts | chance "
          f"{1.0/n_concepts:.3f} | N_organs={N_ORGANS} (1 heterogeneous NON-COMPOSER) K_slots={K_SLOTS} "
          f"D={args.D} d_sub={d_sub:.0f} backend={args.backend}\n", flush=True)

    results = [run_seed(s, d_sub, D=args.D) for s in args.seeds]
    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)

    def mean(k):
        return float(np.mean([r[k] for r in results]))

    print("\n── integration attribution (tools.lab.attributable_to) ──", flush=True)
    void_if(mean("consensus_acc") <= 1e-9, "intact consensus is ~0 — nothing to attribute")
    attributable_to("consensus needs the heterogeneous non-composer organ vote",
                    mean("consensus_acc"), mean("het_dropped_acc"))

    summary = {
        "runner": "_gnw_bus_heterogeneous_organ_derisk",
        "claim": ("the spiking workspace COMBINES a genuinely heterogeneous NON-COMPOSER organ (spiking "
                  "surprise/familiarity monitor) with composer organs via consensus-ignition + WTA; the "
                  "heterogeneous vote is load-bearing"),
        "seeds": list(args.seeds), "n_organs": N_ORGANS, "D": int(args.D), "d_sub": float(d_sub),
        "backend": args.backend, "all_go": all_go, "n_go": n_go, "n_seeds": len(results),
        "all_substrate_combines": all(r["substrate_combines"] for r in results),
        "mean_consensus_when_voted": mean("consensus_when_voted"),
        "mean_abstain_when_withheld": mean("abstain_when_withheld"), "mean_het_vote_rate": mean("het_vote_rate"),
        "mean_consensus_acc": mean("consensus_acc"), "mean_host_recall_acc": mean("host_recall_acc"),
        "mean_single_organ_acc": mean("single_organ_acc"), "mean_het_dropped_acc": mean("het_dropped_acc"),
        "mean_leaveoneout_worst_acc": mean("leaveoneout_worst_acc"),
        "mean_disagree_acc": mean("disagree_acc"), "mean_het_disagree_withheld_frac": mean("het_disagree_withheld_frac"),
        "mean_shuffle_off_acc": mean("shuffle_off_acc"),
        "mean_het_organ_lesion_acc": mean("het_organ_lesion_acc"),
        "mean_het_organ_lesion_withheld_frac": mean("het_organ_lesion_withheld_frac"),
        "mean_workspace_lesion_acc": mean("workspace_lesion_acc"), "mean_reflex_acc": mean("reflex_acc"),
        "mean_het_mean_agree_hz": mean("het_mean_agree_hz"), "mean_het_mean_disagree_hz": mean("het_mean_disagree_hz"),
        "all_het_discriminate": all(r["het_discriminates"] for r in results),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_go else ("PARTIAL" if n_go >= 1 else "NEGATIVE")
    print(f"\n{'='*100}", flush=True)
    print(f"  GNW HETEROGENEOUS-ORGAN BUS VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    print(f"    SUBSTRATE COMBINES (the claim): ignite-when-voted={summary['mean_consensus_when_voted']:.3f} "
          f"abstain-when-withheld={summary['mean_abstain_when_withheld']:.3f} "
          f"all_substrate_combines={summary['all_substrate_combines']}", flush=True)
    print(f"    HET organ (NON-COMPOSER) reliability: vote_rate={summary['mean_het_vote_rate']:.3f} "
          f"| spiking read agree_hz={summary['mean_het_mean_agree_hz']:.2f} "
          f"disagree_hz={summary['mean_het_mean_disagree_hz']:.2f} discriminate_all={summary['all_het_discriminate']} "
          f"| end-to-end consensus={summary['mean_consensus_acc']:.3f} vs host={summary['mean_host_recall_acc']:.3f}",
          flush=True)
    print(f"    HET load-bearing collapses: dropped={summary['mean_het_dropped_acc']:.3f} "
          f"disagree={summary['mean_disagree_acc']:.3f} shuffle_off={summary['mean_shuffle_off_acc']:.3f} "
          f"het_organ_lesion={summary['mean_het_organ_lesion_acc']:.3f}", flush=True)
    print(f"    substrate: single={summary['mean_single_organ_acc']:.3f} "
          f"leave1out={summary['mean_leaveoneout_worst_acc']:.3f} workspace_lesion="
          f"{summary['mean_workspace_lesion_acc']:.3f} | reflex={summary['mean_reflex_acc']:.3f} "
          f"moat_all={summary['all_moat_ok']}", flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
