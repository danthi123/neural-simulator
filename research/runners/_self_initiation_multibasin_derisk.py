"""SELF-INITIATION, MULTIPLE BALANCED BASINS — the internally-driven wander SELECTS among stored concepts.

2026-08-13. The self-initiated-spontaneous-thought de-risk (2026-08-13-self-initiated-spontaneous-thought-GO.md,
runner _self_initiated_spontaneous_thought_derisk.py) is 6-seed GO but used n_mem=2, which reliably reactivates ONE
dominant basin per seed — so it de-risked STEERING (identity-controlled novel-vs-familiar on the SAME thought), NOT
SELECTION (choosing among several equally-reactivatable concepts). That finding's named next rung #1 is EXACTLY this:
"Multiple balanced basins so the wander SELECTS among concepts (curiosity biasing which of several equally-storable
thoughts wins the noise-seeded race) — pattern-separated encoding to equalise basin strength."

THE MECHANISM (the surpass = PATTERN SEPARATION for balance, on top of the two validated organs):
  (1) BALANCED MULTI-BASIN STORE. The gap#5 RANK-1 spontaneous-reactivation substrate draws its assemblies with
      independent random `rng.choice` — they OVERLAP (~29 shared cells for two 240-cell assemblies at n_ca3=2000),
      and the strongest overlapping structure wins -> one dominant basin (the n_mem=2 limit the GO finding reported).
      THE SURPASS: PATTERN-SEPARATED (DISJOINT) encoding — partition a random permutation of CA3 cells into n_mem
      NON-overlapping equal-size assemblies (dentate-gyrus pattern separation: orthogonal engrams; McNaughton &
      Morris 1987; Leutgeb et al. 2007 Science "Pattern separation in DG"; Bakker et al. 2008). Equal size + no shared
      cells + identical BTSP one-shot encode -> n_mem INDEPENDENT, equally-strong attractor basins, no host thumb on
      the scale. This is the ONLY change from the RANK-1 substrate (`_prepare_balanced` below); NO `sim/` edit.
  (2) THE WANDER SELECTS. Under weak NON-SPECIFIC Poisson background (NO cue, 0 external CONTENT drive — the SAME
      operating point as the GO finding), each discrete noise-seeded volley ignites WHICHEVER balanced basin its
      coincidental within-window overlap favours, then the bistable KIR down-state returns the net to silence before
      the next event. Over a long spontaneous session the wander therefore VISITS DIFFERENT stored concepts (Bouhadjar
      et al. 2023 PLoS Comput Biol "Coherent noise enables probabilistic sequence replay"; the biased-competition /
      divisive-normalisation WTA our D3 attractor work uses). The "WHICH concept" is the substrate's attractor
      competition + noise — 0 host content-draws (no random.choice over concepts; asserted).
  (3) CURIOSITY BIASES WHICH. Each concept's NOVELTY (the ENVIRONMENT) maps through the production CURIOSITY organ's
      SPIKING ASK-pool want to a proportional NEUROMODULATORY RECURRENT GAIN on that engram (McNamara 2014;
      Ambrose/Pfeiffer/Foster 2016; Mattar & Daw 2018 need x gain). A more-novel concept's basin completes from a
      smaller coincidental volley -> it wins MORE of the noise-seeded races -> the interesting thought surfaces more.
      The gain is GRADED (not a hard tag), so it BIASES the selection without collapsing it to one basin.

THE QUESTION (SELECTION, not steering): over a spontaneous session (no cue), does the wander VISIT multiple distinct
stored concepts (not stuck on one), each visit COHERENT (a real stored assembly, member >> chance), with CURIOSITY
biasing the visit distribution (novel concepts visited more than a shuffle control)?

SUBSTRATE vs HOST (honesty boundary is a deliverable):
  * SPIKING (load-bearing): the reactivation + selection itself (CA3 dendritic-plateau attractor competition under
    noise), the silence between events, the steering VALUE (curiosity ASK-pool want read off cp_firing_states).
  * HOST (declared, rides existing burn-downs): (i) the per-concept NOVELTY levels are the ENVIRONMENT; (ii) the
    PROJECTION of the spiking want onto the CA3 engram as a recurrent-gain factor is a host-parameterised
    neuromodulatory projection scaling (named next rung: release the `curiosity` modulator onto CA3 on ONE bridge).
    The DISJOINT partition is a wiring choice (pattern separation), NOT a per-event content-draw — the WHICH-basin
    selection at run time is entirely the spiking competition + noise (0 host random.choice over concepts; asserted).

FUNCTIONAL CORRELATE, NOT phenomenal: measures + reports a self-initiation SELECTION correlate. No claim of experience.

THE ANTI-CHEATS (each VERIFIED, not asserted):
  (a) BALANCED (no host thumb): CURIOSITY-OFF (all gains == 1) -> the wander visits multiple basins ~UNIFORMLY (high
      entropy, low top-1 share). If uniform gain already collapsed to one basin, the basins are not balanced.
  (b) SELECTS: >= min_visit distinct concepts visited over the session, each COHERENT (member >= min_frac, >> random).
  (c) CURIOSITY-BIASED (identity-controlled): PRIMARY = the WITHIN-CONCEPT DOSE-RESPONSE — each concept is seen at
      three gains {uniform 1.0, its curiosity gain, its REVERSED gain}; demeaning share within each concept removes
      intrinsic basin strength, so the residual gain->share correlation is the gain's causal effect. SECONDARY = the
      SAME novel concepts surface MORE under HIGH gain (curiosity-on) than LOW gain (REVERSED: novelty->gain inverted)
      -> the bias is the curiosity VALUE, not the basin identity. attributable_to(novel-share on vs reversed).
  (d) INTERNALLY-GENERATED: NO-NOISE (gains on, noise off) -> SILENT (the ignition is genuinely noise-seeded).
  (e) STORE-LESION: NO-ENCODE (same noise + gains, store skipped) -> coherence collapses (member -> chance).
  (f) plasticity byte-FROZEN during the session (verified cp_connections.data unchanged).

CPU-smoke:  SIM_BACKEND=numpy python -u -m research.runners._self_initiation_multibasin_derisk --seeds 42 --n-mem 4 --rest-steps 1800 --gain-scale 1.0 --smoke
Full (GPU): SIM_BACKEND=cupy  python -u -m research.runners._self_initiation_multibasin_derisk --seeds 42 43 44 100 101 102 --n-mem 4 --rest-steps 8000 --gain-scale 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

# reuse-by-import the VALIDATED gap#5 RANK-1 substrate primitives (6-seed GO building blocks)
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    GO_CFG, _extract_ca3ca3_vec,
)
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates  # noqa: E402
# reuse-by-import the SELF-INITIATION (steering) runner's rest/steering/readout machinery (6-seed GO)
from research.runners._self_initiated_spontaneous_thought_derisk import (  # noqa: E402
    _scale_within_assembly, _steered_rest, _surfacing, _assembly_stats, _curiosity_wants, _pearson,
)
from tools.lab import attributable_to, void_if  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_self_initiation_multibasin_derisk.json"

# per-concept NOVELTY levels (the ENVIRONMENT: concepts differ in how novel/interesting they are). DESCENDING so
# concept 0 is the MOST curious. The curiosity organ maps each novelty -> a graded spiking ASK-pool want (Hz).
NOV_BY_NMEM = {
    4: [0.95, 0.65, 0.35, 0.15],
    5: [0.95, 0.75, 0.55, 0.35, 0.15],
    6: [0.95, 0.79, 0.63, 0.47, 0.31, 0.15],
    8: [0.95, 0.84, 0.72, 0.61, 0.49, 0.38, 0.26, 0.15],
}


def _prepare_balanced(seed, cfg, do_encode=True):
    """Reproduce the gap#5 RANK-1 CLOSED-completion bridge EXACTLY (build + BTSP one-shot encode + structural-sep +
    recall-k + selective-inhib), with the ONE surpass this lane adds: the stored assemblies are PATTERN-SEPARATED
    (DISJOINT, equal-size) instead of overlapping random draws, so each concept gets an INDEPENDENT, equally-strong
    basin (no shared cells -> no single dominant basin). Everything else is byte-identical to
    `_gap5_spontaneous_reactivation_derisk._prepare` (copied, not edited, to keep that GO finding untouched).
    do_encode=False = the STORE-LESION (NO-ENCODE) anti-cheat."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    n_ca3 = int(cfg["n_ca3"])
    _init_ca3w = float(cfg["encode_ca3w"]) if cfg.get("encode_btsp") else 6.0
    bridge = _build(seed, n_ca3=n_ca3, ca3w=_init_ca3w, ca3_density=cfg["ca3_density"],
                    coincidence=True, two_comp=True, nmda_recurrent=False, nmda_tau=100.0, nmda_ratio=1.0,
                    apical_R=cfg["apical_R"], apical_gc=cfg["apical_gc"], k_thresh=cfg["k_thresh"],
                    plateau_strength=cfg["plateau_strength"], train=True, hebb_max=cfg["hebb_max"], hebb_rate=True,
                    ca3_fb_inhib=cfg["ca3_fb_inhib"], coact_thresh=cfg["coact_thresh"], hebb_lr=None, enable_ou=False,
                    plateau_self_regen=cfg["plateau_self_regen"], plateau_v_hold=cfg["plateau_v_hold"],
                    apical_kir_g=cfg["apical_kir_g"], apical_gc_read=cfg["apical_gc_read"], ca1_ff_inhib=None)
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3"))
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    rng = np.random.default_rng(seed * 17 + 3)
    n_assy = max(6, int(cfg["assembly_frac"] * n_ca3))
    n_mem = int(cfg["n_mem"])
    # --- PATTERN SEPARATION (the surpass): DISJOINT equal-size assemblies from a random permutation of CA3 cells ---
    assert n_mem * n_assy <= len(ca3_idx), (
        f"disjoint requires n_mem*n_assy <= n_ca3: {n_mem}*{n_assy}={n_mem * n_assy} > {len(ca3_idx)}")
    perm = rng.permutation(np.asarray(ca3_idx, dtype=np.int64))
    assemblies = [np.asarray(sorted(perm[i * n_assy:(i + 1) * n_assy]), dtype=np.int64) for i in range(n_mem)]

    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    conn = bridge.cp_connections

    _set_gates(bridge, 1.0)
    if do_encode and cfg.get("encode_btsp"):
        train_events = int(cfg["train_events"]); reset_steps = int(cfg["reset_steps"])
        drive_steps = int(cfg["drive_steps"]); encode_drive = float(cfg["encode_drive"])
        cfg_b = bridge.core_config
        cfg_b.enable_hebbian_learning = False
        cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
        cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
        cfg_b.coincidence_plateau_v_hold = float(cfg["plateau_v_hold"]); cfg_b.apical_kir_g = float(cfg["apical_kir_g"])
        cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(cfg["btsp_lr"])
        cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(cfg["hebb_max"])
        cfg_b.btsp_hetero_dep = float(cfg["encode_btsp_hetero"])
        bridge.cp_bdsp_apical_drive = cp.zeros(int(cfg_b.num_neurons), dtype=cp.float32)
        for m, assy in enumerate(assemblies):
            assy_arr = cp.asarray(assy, dtype=cp.int64)
            plateau_vec = cp.full(len(assy), float(cfg["encode_plateau_pA"]), dtype=cp.float32)
            for ev in range(train_events):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_bdsp_apical_drive[:] = 0.0
                for _ in range(reset_steps):
                    bridge._run_one_simulation_step()
                for _st in range(drive_steps):
                    bridge.cp_external_input_current[:] = 0.0
                    bridge.cp_external_input_current[assy_arr] = encode_drive
                    bridge.cp_bdsp_apical_drive[:] = 0.0
                    bridge.cp_bdsp_apical_drive[assy_arr] = plateau_vec
                    bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
        cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None
    _set_gates(bridge, 0.0)

    n_ca3_loc = len(ca3_idx)
    member_local = np.zeros(n_ca3_loc, dtype=bool)
    member_local[np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)] = True
    pre_mem = member_local[pre_l_h]; post_mem = member_local[post_l_h]
    within = pre_mem & post_mem
    within_flat = flat_h[within].astype(np.int64)

    d = np.asarray(to_host(conn.data))
    w_within = float(np.mean(d[within_flat])) if within_flat.size else 0.0

    if int(cfg["structural_sep"]) >= 1:
        zsel = post_mem & (~pre_mem)
        if zsel.any():
            idxs = cp.asarray(flat_h[zsel], dtype=cp.int64)
            conn.data[idxs] = cp.zeros(int(zsel.sum()), dtype=conn.data.dtype)

    if cfg.get("recall_k_thresh") is not None:
        bridge.core_config.coincidence_k_threshold = float(cfg["recall_k_thresh"])

    if cfg["selective_inhib"]:
        n_all = int(bridge.core_config.num_neurons)
        bask_bool = np.zeros(n_all, dtype=bool); bask_bool[np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)] = True
        assy_bool = np.zeros(n_all, dtype=bool); assy_bool[np.asarray(sorted(int(g) for a in assemblies for g in a), dtype=np.int64)] = True
        conn2 = bridge.cp_connections; nnz = int(conn2.nnz)
        indptr = np.asarray(to_host(conn2.indptr)); indices = np.asarray(to_host(conn2.indices))
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        spare = bask_bool[pre_of] & assy_bool[indices[:nnz]]
        if spare.any():
            idxs = cp.asarray(np.nonzero(spare)[0], dtype=cp.int64)
            conn2.data[idxs] = cp.full(int(spare.sum()), float(cfg["sel_inhib_spare"]), dtype=conn2.data.dtype)

    assembly_local = np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)
    assemblies_local = [np.asarray(sorted(ca3_pos[int(g)] for g in a), dtype=np.int64) for a in assemblies]
    ca3_arr_host = np.asarray(ca3_idx, dtype=np.int64)
    try:
        ca3_inh = set(int(g) for g in rm.inhibitory_indices("ca3"))
    except Exception:
        ca3_inh = set()
    ca3_exc_local = np.asarray([i for i, g in enumerate(ca3_idx) if int(g) not in ca3_inh], dtype=np.int64)
    # OVERLAP diagnostic (must be 0 for disjoint) — the anti-cheat that the basins really are pattern-separated
    union = sorted(int(g) for a in assemblies for g in a)
    max_overlap = int(max((len(set(a.tolist()) & set(b.tolist())) for i, a in enumerate(assemblies)
                           for b in assemblies[i + 1:]), default=0))
    return dict(bridge=bridge, ca3_idx=ca3_idx, ca3_arr_host=ca3_arr_host, assemblies=assemblies,
                assembly_local=assembly_local, assemblies_local=assemblies_local, ca3_exc_local=ca3_exc_local,
                within_flat=within_flat, w_within=w_within, n_assy=n_assy, max_pair_overlap=max_overlap,
                n_union=len(union))


def _run_condition(seed, cfg, rest_steps, noise_on, *, gains=None, do_encode=True):
    """A condition = a FRESH deterministic bridge (fresh-per-condition is MANDATORY — _hard_silence does not fully
    reset the bistable/dendritic state, so reusing a bridge leaks; and the gains multiply conn.data destructively).
    Same seed -> byte-identical substrate + DISJOINT assemblies + encode + the identical Poisson noise stream; the
    ONLY thing that differs across conditions is the per-assembly curiosity recurrent-gain -> clean attribution."""
    n_mem = int(cfg["n_mem"])
    prep = _prepare_balanced(seed, cfg, do_encode=do_encode)
    if gains is not None:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(gains[i]))
    F, diag = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=noise_on)
    return F, prep, diag


def _selection(F, al, seed, min_frac):
    """SELECTION read-out on a session firing tensor F [T, n_ca3]: which distinct stored concepts the wander VISITED,
    how the visits are DISTRIBUTED (entropy / top-1 share -> stuck-on-one vs selects-among), and per-concept COHERENCE
    (member vs random). A concept is VISITED when it owns winner-take-all dwell > 0; COHERENT when its surfaced steps
    overlap its stored assembly at member >= min_frac and >> random."""
    s = _surfacing(F, al, seed, min_frac)
    n_mem = len(al)
    dwell = np.asarray(s["dwell"], dtype=float)
    stats = [_assembly_stats(F, al, i, seed, min_frac) for i in range(n_mem)]
    visited = dwell > 0
    coherent = np.asarray([(stats[i]["member"] >= min_frac and stats[i]["member"] > 2.0 * (stats[i]["random"] + 1e-6))
                           for i in range(n_mem)], dtype=bool)
    visited_coherent = visited & coherent
    total = float(dwell.sum())
    share = (dwell / total) if total > 0 else np.zeros(n_mem)
    top1 = float(share.max()) if total > 0 else 0.0
    p = share[share > 0]
    entropy = float(-(p * np.log(p)).sum() / np.log(n_mem)) if (len(p) > 1 and n_mem > 1) else 0.0
    vc = [i for i in range(n_mem) if visited[i]]
    pooled_member = float(np.mean([stats[i]["member"] for i in vc])) if vc else 0.0
    pooled_random = float(np.mean([stats[i]["random"] for i in vc])) if vc else 0.0
    return dict(dwell=dwell.tolist(), share=share.tolist(), visited=visited.tolist(),
                coherent=coherent.tolist(), visited_coherent=visited_coherent.tolist(),
                n_visited=int(visited.sum()), n_visited_coherent=int(visited_coherent.sum()),
                top1_share=top1, entropy=entropy, pooled_member=pooled_member, pooled_random=pooled_random,
                per_events=[int(st["n_events"]) for st in stats], per_member=[float(st["member"]) for st in stats],
                per_mass=[float(st["mass_net"]) for st in stats], total_dwell=total)


def _dose_response(gain_by_cond, share_by_cond):
    """WITHIN-CONCEPT DOSE-RESPONSE (the identity-controlled curiosity metric). Each concept is observed at SEVERAL
    gains (one per condition: uniform 1.0, its curiosity gain, its reversed gain). Demeaning the visit share WITHIN
    each concept (across conditions) REMOVES that concept's intrinsic basin strength — the per-concept mean — so the
    residual correlation of gain vs share is the gain's CAUSAL effect, not the basin identity (this is the gap#5
    97%-clamp check made structural: the intrinsic term that ran in every arm is subtracted out). Returns Pearson r
    over all (concept, condition) demeaned points."""
    G = np.asarray(gain_by_cond, dtype=float)          # [C, n_mem]
    S = np.asarray(share_by_cond, dtype=float)          # [C, n_mem]
    Gd = (G - G.mean(axis=0, keepdims=True)).ravel()    # within-concept (across-condition) deviations
    Sd = (S - S.mean(axis=0, keepdims=True)).ravel()
    if Gd.std() < 1e-9 or Sd.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(Gd, Sd)[0, 1])


def one_seed(seed, n_mem, rest_steps, gain_scale, min_frac, acid_steps, min_visit, top1_max, ent_min):
    """SELECTION among balanced basins: store n_mem DISJOINT equal concepts; run the noise-driven wander under
    CURIOSITY-OFF (all gains 1 -> the balance/no-thumb test), CURIOSITY-ON (graded novelty gains -> the biased
    production) and a REVERSED anti-curiosity control (same gain magnitudes assigned INVERSELY to novelty); plus
    NO-NOISE and NO-ENCODE acids. Curiosity bias is read identity-controlled (within-concept dose-response)."""
    t0 = time.time()
    out = {"seed": seed, "n_mem": n_mem}
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(n_mem)

    # NOVELTY assigned to a RANDOM permutation of concepts (decouple novelty from the disjoint-partition index, so
    # intrinsic basin strength is uncorrelated with novelty). GRADED novelty -> graded curiosity want -> graded gain.
    nov_rng = np.random.default_rng(seed * 7919 + 1)
    novelties = [float(v) for v in nov_rng.permutation(np.asarray(NOV_BY_NMEM[n_mem], dtype=float))]
    wants, cur_meta = _curiosity_wants(seed, novelties)
    wmax = max(wants) if wants else 1.0
    gains_on = [1.0 + gain_scale * (w / wmax if wmax > 1e-9 else 0.0) for w in wants]     # monotone in novelty
    gains_off = [1.0] * n_mem
    # REVERSED (anti-curiosity control): the SAME gain multiset assigned INVERSELY to novelty (most-novel concept ->
    # the SMALLEST gain). A clean identity-controlled contrast — the novel concepts are HIGH-gain in curiosity-on and
    # LOW-gain here, so a surfacing difference on the SAME concepts is the curiosity VALUE, not the basin identity.
    nov = np.asarray(novelties, dtype=float)
    order = [int(i) for i in np.argsort(-nov)]           # concepts most-novel -> least-novel
    gvals = sorted(gains_on, reverse=True)               # gain magnitudes, largest first
    gains_reversed = [0.0] * n_mem
    for k, ci in enumerate(order):
        gains_reversed[ci] = gvals[n_mem - 1 - k]        # most-novel concept receives the SMALLEST gain magnitude
    out["novelties"] = novelties; out["wants_hz"] = wants; out["gains_on"] = gains_on
    out["gains_reversed"] = gains_reversed; out["novel_order"] = order; out["curiosity"] = cur_meta
    print(f"  [seed {seed}] novelty {[round(v, 2) for v in novelties]} -> want(Hz) {[round(w, 1) for w in wants]} -> "
          f"gains_on {[round(g, 2) for g in gains_on]} reversed {[round(g, 2) for g in gains_reversed]} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- CURIOSITY-OFF (balanced, all gains 1): the pure selection + no-host-thumb test --
    F_bal, prep_b, d_bal = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_off)
    out["w_within"] = prep_b["w_within"]; out["max_pair_overlap"] = int(prep_b["max_pair_overlap"])
    sel_bal = _selection(F_bal, prep_b["assemblies_local"], seed, min_frac)
    out["balanced"] = sel_bal
    print(f"  [seed {seed}] BALANCED (gains=1): visited {sel_bal['n_visited']} (coherent {sel_bal['n_visited_coherent']}) "
          f"top1 {sel_bal['top1_share']:.2f} entropy {sel_bal['entropy']:.2f} member {sel_bal['pooled_member']:.2f} "
          f"vs rand {sel_bal['pooled_random']:.2f} dwell {[int(x) for x in sel_bal['dwell']]} overlap {prep_b['max_pair_overlap']}",
          flush=True)

    # -- CURIOSITY-ON (graded novelty gains): the biased production wander --
    F_cur, prep_c, d_cur = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_on)
    sel_cur = _selection(F_cur, prep_c["assemblies_local"], seed, min_frac)
    out["curiosity_on"] = sel_cur; out["weights_frozen"] = bool(d_cur["weights_frozen"])
    out["apical_rest_max"] = d_cur["apical_rest_max"]
    print(f"  [seed {seed}] CURIOSITY-ON:      visited {sel_cur['n_visited']} (coherent {sel_cur['n_visited_coherent']}) "
          f"top1 {sel_cur['top1_share']:.2f} entropy {sel_cur['entropy']:.2f} member {sel_cur['pooled_member']:.2f} "
          f"vs rand {sel_cur['pooled_random']:.2f} share {[round(x, 2) for x in sel_cur['share']]}", flush=True)

    # -- REVERSED control (same gain magnitudes, assigned INVERSELY to novelty): isolates the curiosity VALUE --
    F_rv, prep_rv, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_reversed)
    sel_rv = _selection(F_rv, prep_rv["assemblies_local"], seed, min_frac)
    out["reversed"] = sel_rv

    # -- CURIOSITY BIAS (identity-controlled). PRIMARY = the WITHIN-CONCEPT DOSE-RESPONSE: across the three conditions
    #    each concept sees a DIFFERENT gain {uniform 1.0, curiosity gain, reversed gain}; demeaning share within each
    #    concept removes intrinsic basin strength, so the residual gain->share correlation is the gain's causal effect.
    #    SECONDARY = the novel concepts' pooled visit share HIGH-gain (curiosity-on) vs LOW-gain (reversed): SAME
    #    concepts, opposite gains -> the surfacing difference is the curiosity VALUE, not the basin identity. --
    share_bal = np.asarray(sel_bal["share"], dtype=float)
    share_on = np.asarray(sel_cur["share"], dtype=float)
    share_rv = np.asarray(sel_rv["share"], dtype=float)
    gain_bal = np.ones(n_mem); gain_on = np.asarray(gains_on, dtype=float); gain_rv = np.asarray(gains_reversed, dtype=float)
    r_within = _dose_response([gain_bal, gain_on, gain_rv], [share_bal, share_on, share_rv])
    novel_set = np.asarray(order[:max(1, n_mem // 2)], dtype=int)
    novel_share_on = float(share_on[novel_set].sum())
    novel_share_bal = float(share_bal[novel_set].sum())
    novel_share_rv = float(share_rv[novel_set].sum())
    # cross-concept novelty-vs-share correlations (report only — confounded by intrinsic strength; the within-concept r is primary)
    r_on_cc = _pearson(nov, share_on); r_rv_cc = _pearson(nov, share_rv)
    bias_attr = attributable_to("curiosity-gain @ novel-concept visit share (on vs reversed)", novel_share_on, novel_share_rv)
    out["bias"] = dict(r_within=r_within, r_on_crossconcept=r_on_cc, r_reversed_crossconcept=r_rv_cc,
                       novel_share_on=novel_share_on, novel_share_balanced=novel_share_bal,
                       novel_share_reversed=novel_share_rv, uniform_expectation=float(len(novel_set) / n_mem),
                       attributable=bias_attr)
    print(f"  [seed {seed}] BIAS within-concept dose r={r_within:+.2f} | novel-share on={novel_share_on:.2f} "
          f"bal={novel_share_bal:.2f} reversed={novel_share_rv:.2f} (uniform {len(novel_set)/n_mem:.2f}) "
          f"attributable={('%.0f%%' % (100 * bias_attr)) if bias_attr is not None else 'UNDEF'}", flush=True)

    # -- ACIDS: NO-NOISE (gains on, noise off) must be SILENT --
    F_nn, pnn, d_nn = _run_condition(seed, cfg, acid_steps, noise_on=False, gains=gains_on)
    s_nn = _selection(F_nn, pnn["assemblies_local"], seed, min_frac)
    out["no_noise"] = {"total_dwell": float(s_nn["total_dwell"]), "n_visited": int(s_nn["n_visited"]),
                       "max_member": float(max(s_nn["per_member"]) if s_nn["per_member"] else 0.0),
                       "apical_rest_max": d_nn["apical_rest_max"]}
    # -- STORE-LESION (NO-ENCODE, gains on, noise on): coherence must collapse --
    F_sl, psl, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gains_on, do_encode=False)
    s_sl = _selection(F_sl, psl["assemblies_local"], seed, min_frac)
    out["store_lesion"] = {"pooled_member": float(s_sl["pooled_member"]), "pooled_random": float(s_sl["pooled_random"]),
                           "n_visited_coherent": int(s_sl["n_visited_coherent"]), "w_within": float(psl["w_within"])}
    print(f"  [seed {seed}] ACID no_noise dwell={out['no_noise']['total_dwell']:.0f} visited={out['no_noise']['n_visited']} "
          f"| STORE-LESION member {s_sl['pooled_member']:.2f} vs rand {s_sl['pooled_random']:.2f} "
          f"coherent-visits {s_sl['n_visited_coherent']}", flush=True)

    # ---- per-seed GO gate ----
    disjoint_ok = bool(out["max_pair_overlap"] == 0)
    # (a) BALANCED (SELECTION CAPACITY + no host thumb): under UNIFORM gain the wander visits >= min_visit distinct
    #     COHERENT basins near-uniformly (high entropy, no single basin monopolises) -> the n_mem basins are genuinely
    #     equally-selectable (identical disjoint encode, no thumb on the scale). This is the substrate's selection
    #     capacity among balanced basins.
    balanced = bool(sel_bal["n_visited_coherent"] >= min_visit and sel_bal["top1_share"] <= top1_max
                    and sel_bal["entropy"] >= ent_min)
    # (b) PRODUCTION SELECTS: the curiosity-on production wander visits MULTIPLE (>= 2) distinct COHERENT concepts and
    #     is NOT collapsed onto a single fixed thought (top1 <= 0.85) -> curiosity biases WHICH surfaces more but the
    #     wander still visits several over the session (the mission's "visits multiple, not stuck on one").
    production_selects = bool(sel_cur["n_visited_coherent"] >= 2 and sel_cur["top1_share"] <= 0.85)
    # (b') COHERENT: surfaced concepts overlap their stored assembly well above chance
    coherent = bool(sel_cur["pooled_member"] >= min_frac and sel_cur["pooled_member"] > 2.0 * (sel_cur["pooled_random"] + 1e-6))
    # (c) CURIOSITY-BIASED (identity-controlled). PRIMARY: the SAME novel concepts surface MATERIALLY more under HIGH
    #     gain (curiosity-on) than under LOW gain (reversed) -> the curiosity VALUE steers WHICH thought surfaces, not
    #     the basin identity (intrinsic strength cancels — same concepts). CORROBORATOR: the within-concept
    #     dose-response is positive. (The `on vs balanced` contrast can ceiling when random novelty assignment lands
    #     novel concepts on the intrinsically-strong basins; the reversed contrast avoids that, so it is primary.)
    curiosity_biased = bool(novel_share_on >= novel_share_rv + 0.10 and r_within >= 0.2)
    # (d) INTERNALLY-GENERATED: NO-NOISE acid silent
    internally_generated = bool(out["no_noise"]["total_dwell"] <= 2 and out["no_noise"]["max_member"] < min_frac
                                and out["weights_frozen"]
                                and (out["apical_rest_max"] is None
                                     or out["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3))
    # (e) STORE-LESION: NO-ENCODE collapses coherence
    store_lesion_ok = bool(s_sl["n_visited_coherent"] == 0
                           or s_sl["pooled_member"] < 0.5 * sel_cur["pooled_member"]
                           or s_sl["pooled_member"] < 2.0 * (s_sl["pooled_random"] + 1e-6))

    checks = dict(disjoint_ok=disjoint_ok, balanced=balanced, production_selects=production_selects, coherent=coherent,
                  curiosity_biased=curiosity_biased, internally_generated=internally_generated,
                  store_lesion_load_bearing=store_lesion_ok)
    seed_go = bool(all(checks.values()))
    out["checks"] = checks; out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={checks}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=4, choices=[4, 5, 6, 8])
    ap.add_argument("--rest-steps", type=int, default=8000, help="session rest steps for the wander (long -> visits multiple)")
    ap.add_argument("--acid-steps", type=int, default=1200, help="rest steps for the NO-NOISE acid test")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="curiosity recurrent-gain scale (novel gain = 1+scale; GRADED down in novelty)")
    ap.add_argument("--min-frac", type=float, default=0.30, help="assembly-active fraction to count a surfaced step")
    ap.add_argument("--min-visit", type=int, default=3, help="distinct COHERENT concepts the wander must visit (of n_mem)")
    ap.add_argument("--top1-max", type=float, default=0.70, help="max allowed top-1 visit share (>this = stuck on one basin)")
    ap.add_argument("--ent-min", type=float, default=0.55, help="min normalised visit entropy in the balanced condition")
    ap.add_argument("--smoke", action="store_true", help="smoke: >=50%% seeds GO; full gate is >=5/6")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    print(f"[multibasin] n_mem={a.n_mem} rest_steps={a.rest_steps} gain_scale={a.gain_scale} min_visit={a.min_visit} "
          f"top1_max={a.top1_max} ent_min={a.ent_min} seeds={a.seeds} backend={os.environ.get('SIM_BACKEND','auto')}",
          flush=True)
    t0 = time.time(); per = []; err = None
    partial_path = Path(a.out).with_suffix(".partial.json")
    try:
        for s in a.seeds:
            per.append(one_seed(s, a.n_mem, a.rest_steps, a.gain_scale, a.min_frac, a.acid_steps,
                                a.min_visit, a.top1_max, a.ent_min))
            # incremental checkpoint: a kill mid-run cannot erase completed seeds
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(json.dumps({"partial": True, "seeds_done": [p["seed"] for p in per],
                                                "per_seed": per}, indent=2, default=str))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    attribution = None; preconditions = []
    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        thresh = max(1, (len(per) + 1) // 2) if a.smoke else max(1, (5 * len(per) + 5) // 6)
        go = n_go >= thresh
        m_visit_cur = float(np.mean([p["curiosity_on"]["n_visited_coherent"] for p in per]))
        m_visit_bal = float(np.mean([p["balanced"]["n_visited_coherent"] for p in per]))
        m_top1_cur = float(np.mean([p["curiosity_on"]["top1_share"] for p in per]))
        m_ent_bal = float(np.mean([p["balanced"]["entropy"] for p in per]))
        m_member = float(np.mean([p["curiosity_on"]["pooled_member"] for p in per]))
        m_random = float(np.mean([p["curiosity_on"]["pooled_random"] for p in per]))
        m_r_within = float(np.mean([p["bias"]["r_within"] for p in per]))
        m_novel_on = float(np.mean([p["bias"]["novel_share_on"] for p in per]))
        m_novel_bal = float(np.mean([p["bias"]["novel_share_balanced"] for p in per]))
        m_novel_rv = float(np.mean([p["bias"]["novel_share_reversed"] for p in per]))
        # ATTRIBUTION (the gap#5 clamp check): the novel concepts' visit share HIGH-gain (curiosity-on) vs the SAME
        # concepts LOW-gain (reversed). Identity-controlled -> the fraction owned by the curiosity gain, not identity.
        attribution = attributable_to("curiosity-gain @ novel-concept visit share (6-seed, on vs reversed)", m_novel_on, m_novel_rv)

        vd = Verdict("self-initiation multibasin selection (6-seed)", chance=m_random)
        vd.require("seeds passing all anti-cheats >= threshold", n_go, expect=lambda x, t=thresh: x >= t)
        vd.require("balanced basins are DISJOINT (max pairwise overlap == 0) every seed",
                   all(p["max_pair_overlap"] == 0 for p in per), expect=True)
        vd.require("balanced (uniform-gain) wander visits >= min_visit distinct COHERENT basins (mean) -- selection capacity",
                   m_visit_bal, expect=lambda x, mv=a.min_visit: x >= mv)
        vd.require("production (curiosity-on) wander visits MULTIPLE (mean >= 2) distinct COHERENT concepts",
                   m_visit_cur, expect=lambda x: x >= 2.0)
        vd.control("curiosity bias: novel visit share HIGH-gain (on) vs LOW-gain (reversed), same concepts",
                   m_novel_on, m_novel_rv, min_separation=0.05)
        vd.control("coherent: surfaced member vs random floor", m_member, m_random, min_separation=0.15)
        vd.floor("coherence member above random", m_member, floor=m_random)
        vd.require("internally-generated: NO-NOISE acid silent every seed",
                   all(p["no_noise"]["total_dwell"] <= 2 for p in per), expect=True)
        vd.require("store-lesion collapses coherence every seed",
                   all(p["checks"]["store_lesion_load_bearing"] for p in per), expect=True)
        vd.require("plasticity byte-frozen during the session every seed",
                   all(p["weights_frozen"] for p in per), expect=True)
        vd.require("within-concept dose-response positive (corroborator, mean r_within >= 0.25)",
                   m_r_within, expect=lambda x: x >= 0.25)
        vd.disabled("hebbian/BTSP plasticity during the session", "the wander measures noise-seeded completion on a frozen store")
        decided = vd.decide(go)
        preconditions = decided["preconditions"]

        verdict = (f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{len(per)} -- {a.n_mem} DISJOINT balanced CA3 basins; "
                   f"the noise-driven wander "
                   f"{'SELECTS among balanced basins (uniform-gain visits mean %.1f of %d coherent, entropy %.2f) and the production wander (curiosity-on) visits MULTIPLE (mean %.1f, top1 %.2f) with CURIOSITY biasing WHICH surfaces (within-concept dose r=%.2f; novel visit-share HIGH-gain %.2f vs LOW-gain %.2f vs uniform %.2f)' % (m_visit_bal, a.n_mem, m_ent_bal, m_visit_cur, m_top1_cur, m_r_within, m_novel_on, m_novel_rv, m_novel_bal) if go else 'did NOT cleanly SELECT among balanced basins (see per-seed checks)'}; "
                   f"coherence member {m_member:.2f} vs random {m_random:.2f}"
                   f"{'; %.0f%% of the novel-concept surfacing attributable to the curiosity gain' % (100 * attribution) if attribution is not None else ''}. "
                   f"{'=> self-initiation now SELECTS its own thoughts among several balanced concepts (a step toward internally-driven ideation).' if go else 'Per THE LAW: tune gain_scale / rest_steps / min_frac / basin balance; not a stop.'}")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        vd = Verdict("self-initiation multibasin selection (6-seed)")
        vd.require("run completed without error", err is None, expect=True)
        preconditions = vd.decide(False)["preconditions"]

    summary = {"probe": "self_initiation_multibasin", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_mem": a.n_mem, "rest_steps": a.rest_steps, "gain_scale": a.gain_scale,
               "min_visit": a.min_visit, "top1_max": a.top1_max, "ent_min": a.ent_min,
               "curiosity_bias_attribution": attribution, "preconditions": preconditions,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[multibasin] VERDICT: {verdict}\n[multibasin] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
