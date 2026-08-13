"""SELF-INITIATED / SPONTANEOUS THOUGHT — a DEFAULT-MODE-NETWORK correlate: the spiking substrate generating its
OWN internally-driven content WITHOUT an external prompt, STEERED by curiosity so the interesting thought surfaces.

2026-08-13. A genuinely-conversing brain does not only REACT to prompts — it has internally-generated thought
(mind-wandering / DMN / spontaneous replay-driven ideation) that seeds curiosity-driven questions and self-initiated
conversation (Christoff et al. 2016 Nat Rev Neurosci "Mind-wandering as spontaneous thought: a dynamic framework"
— the DMN *generates* the sequence, the frontoparietal control / salience system STEERS it; Buckner 2008 "internal
train of thought"; our own read-only cluster review 2026-06-27-conv-thinking-research-reasoning-thinking.md §2.6
flagged this as `missing` and its wall #5 as the hardest to anti-cheat). Today the sim is prompt-driven. This
de-risk composes EXISTING validated organs into the FIRST self-initiation correlate; NO `sim/` edit; reuse-by-import.

THE MECHANISM (two validated pieces composed, plus one biological projection):
  (1) INTERNALLY-GENERATED + COHERENT = the gap#5 RANK-1 spontaneous-reactivation substrate (6-seed GO,
      2026-07-22-gap5-RANK1-spontaneous-reactivation-6seed-GO.md), reused-by-import
      (`_gap5_spontaneous_reactivation_derisk._prepare/_hard_silence/_detect_events`). A CLOSED bistable CA3 store
      (dendritic-plateau two-compartment neurons + committed BTSP one-shot encode) holds SEVERAL stored concepts as
      attractor basins. Under WEAK NON-SPECIFIC background (Poisson volleys to random CA3-exc cells — NO cue, NO
      recall_drive, 0 external CONTENT drive), a stored assembly SPONTANEOUSLY + basin-SELECTIVELY reactivates as a
      discrete event and the net rests silent between events. That reactivated assembly IS a coherent
      internally-driven "thought" (it lands on real stored content, not noise) that the brain could then speak.
  (2) CURIOSITY STEERING = the production CURIOSITY organ (`curiosity_production_organ.CuriosityProductionOrgan`,
      Gate-B, reused-by-import), which reads a genuinely-SPIKING ASK-pool WANT (Hz) that tracks a concept's
      epistemic-gap/interest (novel -> ~128 Hz, familiar -> ~7 Hz; the `from_novelty` neuromodulator drives an
      excitability_drive on group:ask, read off cp_firing_states). Each stored concept's curiosity WANT tags its CA3
      assembly with a proportional NEUROMODULATORY RECURRENT GAIN (a transient multiplicative scaling of that
      engram's within-assembly recurrent synapses, applied ONCE before rest, plasticity frozen) — the biology of
      DA/ACh tagging salient/novel memories for PREFERENTIAL offline reactivation (McNamara et al. 2014 Nat
      Neurosci "Dopaminergic neurons promote hippocampal reactivation"; Ambrose/Pfeiffer/Foster 2016 Neuron "Reward
      enhances reverse replay"; the Mattar & Daw 2018 need x gain prioritisation). A stronger recurrent basin
      completes from a smaller coincidental noise volley -> the tagged thought reactivates MORE often. The gain only
      amplifies RECURRENCE, so it is verified subthreshold (GAIN-ONLY, noise off -> the net stays SILENT): the
      ignition is genuinely noise-SEEDED, only STEERED. (The tonic-excitability method was tried first and MEASURED
      too weak/non-monotone on this substrate — events 5/2/4 across bias 0/60/120 pA; per THE LAW the failing method
      is banked and the recurrent-gain method is the surpass.)

SUBSTRATE vs HOST (honesty boundary is a deliverable, not a caveat):
  * SPIKING (load-bearing): the reactivation itself (CA3 dendritic-plateau attractor completion), the SILENCE
    between events, AND the steering VALUE (the ASK-pool WANT is read off cp_firing_states in the curiosity organ).
  * HOST (declared, ride existing burn-downs): (i) the per-concept NOVELTY levels are the ENVIRONMENT (concepts
    genuinely differ in how novel they are — a host boundary exactly as the curiosity organ declares its novelty
    derivation and the surprise organ its sensory encoding); (ii) the PROJECTION of the spiking WANT onto the CA3
    engram as a recurrent-gain factor is a host-parameterised neuromodulatory projection scaling — the SAME class
    of boundary as the curiosity organ's novelty->drive mapping. NAMED NEXT RUNG: wire the `curiosity` neuromodulator
    to RELEASE onto the CA3 store on ONE bridge (the one-brain merge), so the gain is set BY the spiking modulator
    instead of a host scalar — exactly the co-resident-merge rung the affect / surprise / episodic organs each carry.

FUNCTIONAL CORRELATE, NOT phenomenal: this measures + reports a self-initiation CORRELATE (a coherent,
curiosity-steered, internally-driven reactivation with no prompt). It makes NO claim of subjective experience.

THE FOUR ANTI-CHEATS (each VERIFIED, not asserted; the mission's bar + our own wall #5 skepticism):
  (a) INTERNALLY-GENERATED: 0 external CONTENT drive (only non-specific Poisson to random CA3-exc cells; no cue, no
      recall_drive). NO-NOISE (acid) -> SILENT; GAIN-ONLY (steering on, noise off) -> SILENT (the gain amplifies
      recurrence only, so it cannot manufacture a thought — it only steers a noise-seeded one). Plasticity byte-frozen.
  (b) COHERENT: the reactivated pattern OVERLAPS a STORED assembly (member_frac >> random_frac); STORE-LESION
      (NO-ENCODE, same noise+gain) -> no coherent events (incoherent noise), so the content is the learned store.
  (c) CURIOSITY-STEERED (identity-controlled): the SAME stored thought reactivates MORE (more events + dwell) when
      tagged NOVEL (high curiosity gain) than when tagged FAMILIAR (near-unity gain) — the surfacing tracks the
      curiosity VALUE, not the content identity (the FAMILIAR tag is the mismatched-value control on the same
      content, so intrinsic basin dominance is cancelled).
  (d) STEERING LESION-LOAD-BEARING: remove the curiosity tag (no-gain baseline) -> surfacing drops below the
      novel-tag level — the curiosity gain is what boosts WHICH internally-driven thought surfaces.

CPU-smoke:  SIM_BACKEND=numpy python -u -m research.runners._self_initiated_spontaneous_thought_derisk --seeds 42 --n-mem 2 --rest-steps 1500 --gain-scale 2 --smoke
Full (GPU): SIM_BACKEND=cupy  python -u -m research.runners._self_initiated_spontaneous_thought_derisk --seeds 42 43 44 100 101 102 --n-mem 2 --rest-steps 6000 --gain-scale 2
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

# reuse-by-import the VALIDATED gap#5 RANK-1 spontaneous-reactivation building blocks (6-seed GO substrate)
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    GO_CFG, _prepare, _hard_silence, _configure_ou, _detect_events,
)
# reuse-by-import the production CURIOSITY organ (spiking ASK-pool want read)
from research.runners.curiosity_production_organ import CuriosityProductionOrgan  # noqa: E402
# attribution (whose the difference is) + a verdict that carries its preconditions into the artifact
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_self_initiated_spontaneous_thought_derisk.json"

# The RANK-1 6-seed GO Poisson operating point (2026-07-22 finding): rate 0.015, 1500 pA, dur 10, rest 1500.
NOISE_RATE = 0.015
NOISE_PA = 1500.0
NOISE_DUR = 10

# per-concept NOVELTY levels (the ENVIRONMENT: concepts differ in how novel/interesting they are). Descending so
# concept 0 is the MOST curious. Curiosity organ maps novelty -> a graded spiking ASK-pool want (7..128 Hz).
NOV_BY_NMEM = {
    2: [0.95, 0.15],
    3: [0.95, 0.55, 0.15],
    4: [0.95, 0.65, 0.35, 0.15],
}


def _pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _curiosity_wants(seed, novelties):
    """The per-concept curiosity VALUE = the production organ's SPIKING ASK-pool want (Hz) at each concept's
    novelty. Reuse-by-import; deterministic given seed. Returns a list of want_hz aligned to `novelties`."""
    org = CuriosityProductionOrgan(seed=seed)
    org.ensure_built()
    wants = [float(org.judge(novelty=float(v))["want_hz"]) for v in novelties]
    return wants, {"threshold_hz": float(org.threshold), "calib": org.calib}


def _bias_vector(prep, bias_per_assembly):
    """Build the tonic per-CELL neuromodulatory excitability bias (pA) over ALL neurons: assembly i's cells get
    bias_per_assembly[i]. A device array (cupy in prod, numpy in tests). NOTE: assemblies may share cells (random
    draws); a shared cell takes the MAX tag so an isolation run (one assembly tagged) is unaffected by the others."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    n_all = int(prep["bridge"].core_config.num_neurons)
    bias = np.zeros(n_all, dtype=np.float32)
    for i, assy in enumerate(prep["assemblies"]):
        idx = np.asarray(assy, dtype=np.int64)
        bias[idx] = np.maximum(bias[idx], float(bias_per_assembly[i]))
    return cp.asarray(bias, dtype=cp.float32)




def _scale_within_assembly(prep, i, factor):
    """NEUROMODULATORY RECURRENT-GAIN steering: multiply assembly i's WITHIN-assembly recurrent (CA3->CA3) synaptic
    weights by `factor` (a transient neuromodulatory gain applied ONCE before rest — DA/ACh scaling the tagged
    engram's recurrence: McNamara 2014, Ambrose 2016; the Mattar-Daw need x gain prioritisation of reactivation).
    This is NOT a plasticity update — plasticity stays frozen during rest (verified byte-unchanged), so the
    reactivation dynamics remain fully spiking. A stronger recurrent basin completes from a smaller coincidental
    noise volley -> the tagged thought reactivates more often. Returns the number of edges scaled."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    if abs(factor - 1.0) < 1e-9:
        return 0
    bridge = prep["bridge"]; conn = bridge.cp_connections
    n_all = int(bridge.core_config.num_neurons); nnz = int(conn.nnz)
    memb = np.zeros(n_all, dtype=bool)
    memb[np.asarray(prep["assemblies"][i], dtype=np.int64)] = True
    indptr = np.asarray(to_host(conn.indptr)); indices = np.asarray(to_host(conn.indices))
    pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
    within = memb[pre_of] & memb[indices[:nnz]]
    if not within.any():
        return 0
    idxs = cp.asarray(np.nonzero(within)[0], dtype=cp.int64)
    conn.data[idxs] = conn.data[idxs] * cp.asarray(float(factor), dtype=conn.data.dtype)
    return int(within.sum())


def _steered_rest(prep, bias_per_assembly, rest_steps, seed, *, noise_on=True):
    """Freeze plasticity, hard-silence (verify dendritic reset), apply the tonic per-assembly excitability BIAS,
    optionally add weak NON-SPECIFIC Poisson background (no cue / no recall_drive), run REST, record CA3 firing.
    The Poisson RNG stream is seeded ONLY by `seed` (identical across bias conditions) so the ONLY thing that
    differs between STEERED / SHUFFLE / STEER-LESION is the bias -> causal attribution of any surfacing difference
    to the steering. Returns F [T, n_ca3] bool + diagnostics."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = prep["bridge"]
    bridge.core_config.enable_hebbian_learning = False
    assert bridge.core_config.enable_hebbian_learning is False

    _hard_silence(bridge)
    apical_max = None; n_latched = 0
    if getattr(bridge, "cp_v_apical", None) is not None:
        va = np.asarray(to_host(bridge.cp_v_apical))[prep["ca3_arr_host"][prep["assembly_local"]]]
        apical_max = float(np.max(va)); n_latched = int((va > float(GO_CFG["plateau_v_hold"])).sum())

    _configure_ou(bridge, None, seed)                      # OU disabled; we drive Poisson explicitly
    bias_dev = _bias_vector(prep, bias_per_assembly)
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()

    # Poisson non-specific background (CA3-EXC-targeted; deterministic host RNG). Same setup as the RANK-1 GO.
    exc_glob = prep["ca3_arr_host"][prep["ca3_exc_local"]]
    exc_dev = cp.asarray(exc_glob, dtype=cp.int64)
    prng = np.random.default_rng(int(seed) * 100003 + 11)
    countdown = np.zeros(len(exc_glob), dtype=np.int32)

    ca3_arr_host = prep["ca3_arr_host"]
    F = np.zeros((rest_steps, len(ca3_arr_host)), dtype=bool)
    for t in range(rest_steps):
        bridge.cp_external_input_current[:] = bias_dev     # tonic neuromodulatory excitability bias (steering)
        if noise_on:
            new = prng.random(len(exc_glob)) < NOISE_RATE
            countdown[new] = NOISE_DUR
            active = countdown > 0
            if active.any():
                idx = cp.asarray(np.nonzero(active)[0], dtype=cp.int64)
                bridge.cp_external_input_current[exc_dev[idx]] += NOISE_PA   # ADD the pulse on top of the bias
            countdown[active] -= 1
        bridge._run_one_simulation_step()
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[ca3_arr_host].astype(bool)

    w_after = np.asarray(to_host(bridge.cp_connections.data))
    weights_frozen = bool(np.array_equal(w_before, w_after))
    return F, dict(apical_rest_max=apical_max, apical_n_latched=n_latched, weights_frozen=weights_frozen)


def _surfacing(F, assemblies_local, seed, min_frac=0.30):
    """From a rest-phase CA3 firing tensor F [T, n_ca3], compute per-concept SURFACING: dwell (winner-take-all
    active steps), discrete events, and coherence (member vs random). A step 'surfaces' concept i when its
    per-step assembly-active fraction is >= min_frac AND is the max across concepts. Random control = mean over
    permuted (non-member) same-size sets, per concept, so member >> random == coherent (lands on stored content)."""
    T, nca3 = F.shape
    n_mem = len(assemblies_local)
    A = np.stack([F[:, np.asarray(a, dtype=np.int64)].mean(1) for a in assemblies_local], axis=0)  # [n_mem, T]
    active = A >= min_frac                                   # [n_mem, T]
    any_active = active.any(0)
    winner = np.argmax(A, axis=0)                            # [T]
    dwell = np.zeros(n_mem, dtype=float)
    for i in range(n_mem):
        dwell[i] = float(np.sum(active[i] & (winner == i) & any_active))

    # coherence: member_frac (during that concept's surfaced steps) vs a random non-member set of the same size
    rng = np.random.default_rng(seed * 991 + 5)
    union = set(int(g) for a in assemblies_local for g in a)
    nonmember = np.asarray([i for i in range(nca3) if i not in union], dtype=np.int64)
    member_frac = np.zeros(n_mem, dtype=float); random_frac = np.zeros(n_mem, dtype=float)
    for i in range(n_mem):
        sel = active[i] & (winner == i) & any_active
        if sel.any():
            member_frac[i] = float(A[i][sel].mean())
            asize = len(assemblies_local[i])
            rsets = [rng.choice(nonmember, asize, replace=False) for _ in range(6)] if len(nonmember) >= asize else []
            random_frac[i] = float(np.mean([F[sel][:, rs].sum(1).mean() / asize for rs in rsets])) if rsets else 0.0

    total_dwell = float(dwell.sum())
    share = (dwell / total_dwell) if total_dwell > 0 else np.full(n_mem, np.nan)
    return dict(dwell=dwell, share=share, total_dwell=total_dwell, winner_active_steps=int(any_active.sum()),
                member_frac=member_frac, random_frac=random_frac,
                pooled_member=float(member_frac[dwell > 0].mean()) if (dwell > 0).any() else 0.0,
                pooled_random=float(random_frac[dwell > 0].mean()) if (dwell > 0).any() else 0.0)


def _assembly_stats(F, al, i, seed, min_frac):
    """Assembly i's surfacing on rest tensor F: DISCRETE events (the base's proven MAD-threshold detector) +
    winner-take-all dwell + coherence (member vs random). `other_local` = the union of the OTHER stored assemblies,
    so a competing basin is excluded from the random control."""
    others = None
    if len(al) > 1:
        others = np.concatenate([np.asarray(al[j], dtype=np.int64) for j in range(len(al)) if j != i])
    ev = _detect_events(F, al[i], seed, other_local=others, min_frac=min_frac)
    s = _surfacing(F, al, seed, min_frac)
    # reactivation MASS = mean assembly-active fraction over the WHOLE rest (frequency x strength x duration in one
    # continuous, threshold-free number -> sensitive to graded steering) minus a random non-member floor.
    T, nca3 = F.shape
    A_i = np.asarray(al[i], dtype=np.int64)
    mass = float(F[:, A_i].mean())
    union = set(int(g) for a in al for g in a)
    nonmember = np.asarray([k for k in range(nca3) if k not in union], dtype=np.int64)
    rng = np.random.default_rng(seed * 131 + i)
    rsets = [rng.choice(nonmember, len(A_i), replace=False) for _ in range(6)] if len(nonmember) >= len(A_i) else []
    mass_rand = float(np.mean([F[:, rs].mean() for rs in rsets])) if rsets else 0.0
    return dict(n_events=int(ev["n_events"]), n_specific=int(ev["n_specific"]), member=float(ev["member_frac"]),
                random=float(ev["random_frac"]), spec=float(ev["specificity"]), duty=float(ev["duty_cycle"]),
                dwell=float(s["dwell"][i]), mass=mass, mass_rand=mass_rand, mass_net=float(mass - mass_rand))


def _run_condition(seed, cfg, rest_steps, noise_on, *, gains=None, do_encode=True):
    """A condition = a FRESH deterministic bridge (same seed -> byte-identical substrate + assemblies + encode) run
    for ONE rest phase. Fresh-per-condition is MANDATORY: _hard_silence does NOT fully reset the bistable/dendritic
    state, so reusing one bridge across conditions leaks state (verified: 1st rest after encode != 2nd). Every
    condition therefore sees the identical 'first rest after encode' substrate + the identical Poisson noise stream
    (seeded by `seed`); the ONLY thing that differs is the STEERING (`gains`, a per-assembly neuromodulatory
    recurrent-gain factor) -> a clean causal attribution. No tonic bias is used (the tonic-excitability method was
    measured too weak/non-monotone on this substrate; the recurrent-gain method is the surpass)."""
    n_mem = int(cfg["n_mem"])
    prep = _prepare(seed, cfg, do_encode=do_encode)
    if gains is not None:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(gains[i]))
    F, diag = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=noise_on)
    return F, prep, diag


def one_seed(seed, n_mem, rest_steps, gain_scale, min_frac, acid_steps):
    """IDENTITY-CONTROLLED steering: each stored concept-assembly is tested UNDER ITS OWN identity with a NOVEL
    curiosity-tag vs a FAMILIAR curiosity-tag vs NO tag (isolation — only that assembly is tagged). Because the
    SAME assembly serves as its own control, any surfacing difference is caused by the curiosity VALUE of the tag,
    not by intrinsic basin dominance. The tag = a NEUROMODULATORY RECURRENT-GAIN on that assembly's engram, scaled
    by the production curiosity organ's SPIKING want at a NOVEL vs a FAMILIAR novelty (a genuinely-novel concept
    drives ~128 Hz -> the higher recurrent gain -> that thought reactivates more; McNamara 2014 / Ambrose 2016 /
    Mattar-Daw need x gain)."""
    t0 = time.time()
    out = {"seed": seed, "n_mem": n_mem}
    cfg = dict(GO_CFG); cfg["n_ca3"] = 2000; cfg["n_mem"] = int(n_mem)

    # -- the curiosity VALUE (SPIKING ASK-pool want) at a NOVEL vs FAMILIAR concept -> the recurrent-gain tag --
    (want_novel, want_familiar), cur_meta = _curiosity_wants(seed, [0.95, 0.15])
    gain_novel = 1.0 + gain_scale                                     # novel concept: full curiosity gain
    gain_familiar = 1.0 + gain_scale * (want_familiar / want_novel if want_novel > 1e-9 else 0.0)   # familiar: ~1.0
    out["want_novel_hz"] = want_novel; out["want_familiar_hz"] = want_familiar
    out["gain_novel"] = gain_novel; out["gain_familiar"] = gain_familiar; out["curiosity"] = cur_meta
    print(f"  [seed {seed}] curiosity want novel={want_novel:.1f}Hz familiar={want_familiar:.1f}Hz -> recurrent gain "
          f"novel=x{gain_novel:.2f} familiar=x{gain_familiar:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # -- BASELINE (no tag, noise on): each assembly's INTRINSIC spontaneous reactivation (fresh bridge) --
    F_base, prep_b, d_base = _run_condition(seed, cfg, rest_steps, noise_on=True)
    out["w_within_prepare"] = prep_b["w_within"]
    base = [_assembly_stats(F_base, prep_b["assemblies_local"], i, seed, min_frac) for i in range(n_mem)]

    # -- ISOLATION steering: per assembly, tag it NOVEL vs FAMILIAR on a FRESH bridge (others untagged). Same
    #    substrate + noise; only the curiosity recurrent-gain differs -> the surfacing difference IS the steering. --
    novel = []; familiar = []
    for i in range(n_mem):
        gn = [1.0] * n_mem; gn[i] = gain_novel
        gf = [1.0] * n_mem; gf[i] = gain_familiar
        Fn, pn, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gn)
        novel.append(_assembly_stats(Fn, pn["assemblies_local"], i, seed, min_frac))
        Ff, pf, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=gf)
        familiar.append(_assembly_stats(Ff, pf["assemblies_local"], i, seed, min_frac))
    out["baseline"] = base; out["novel_tag"] = novel; out["familiar_tag"] = familiar

    # -- ACIDS (fresh bridges): gain-only (ALL novel-tagged, noise OFF -> worst-case self-ignition) must be SILENT --
    F_bo, pbo, d_bo = _run_condition(seed, cfg, acid_steps, noise_on=False, gains=[gain_novel] * n_mem)
    s_bo = _surfacing(F_bo, pbo["assemblies_local"], seed, min_frac)
    out["gain_only"] = {"total_dwell": float(s_bo["total_dwell"]),
                        "max_member": float(np.nan_to_num(np.nanmax(np.append(s_bo["member_frac"], 0.0)))),
                        "apical_rest_max": d_bo["apical_rest_max"]}
    # NO-NOISE (no tag, noise off) must be SILENT
    F_nn, pnn, d_nn = _run_condition(seed, cfg, acid_steps, noise_on=False)
    s_nn = _surfacing(F_nn, pnn["assemblies_local"], seed, min_frac)
    out["no_noise"] = {"total_dwell": float(s_nn["total_dwell"]),
                       "max_member": float(np.nan_to_num(np.nanmax(np.append(s_nn["member_frac"], 0.0))))}
    out["weights_frozen"] = bool(d_base["weights_frozen"]); out["apical_rest_max"] = d_base["apical_rest_max"]

    # -- STORE-LESION (NO-ENCODE, novel gain on all, noise on): coherence must collapse (incoherent noise) --
    F_sl, psl, _ = _run_condition(seed, cfg, rest_steps, noise_on=True, gains=[gain_novel] * n_mem, do_encode=False)
    s_sl = _surfacing(F_sl, psl["assemblies_local"], seed, min_frac)
    out["store_lesion"] = {"total_dwell": float(s_sl["total_dwell"]), "pooled_member": float(s_sl["pooled_member"]),
                           "pooled_random": float(s_sl["pooled_random"])}
    out["w_within_noencode"] = psl["w_within"]

    # ---- steering scalars: PRIMARY = reactivation MASS (net of random floor, summed over assemblies); event count
    #      + dwell are reported as secondary (recurrent gain boosts completeness/duration more than raw count). ----
    En = float(sum(a["n_events"] for a in novel)); Ef = float(sum(a["n_events"] for a in familiar))
    Eb = float(sum(a["n_events"] for a in base))
    Dn = float(sum(a["dwell"] for a in novel)); Df = float(sum(a["dwell"] for a in familiar))
    Db = float(sum(a["dwell"] for a in base))
    Mn = float(sum(max(a["mass_net"], 0.0) for a in novel)); Mf = float(sum(max(a["mass_net"], 0.0) for a in familiar))
    Mb = float(sum(max(a["mass_net"], 0.0) for a in base))
    # coherence is a property of the thoughts that GENUINELY surface (winner-take-all dwell>0), NOT of the MAD
    # detector's weak noise-blip "events" on a non-igniting basin (an assembly with dwell==0 never surfaces, so its
    # near-chance member_frac must not dilute the coherence of the thought that DID surface).
    novel_member = [a["member"] for a in novel if a["dwell"] > 0]
    novel_random = [a["random"] for a in novel if a["dwell"] > 0]
    pooled_member = float(np.mean(novel_member)) if novel_member else 0.0
    pooled_random = float(np.mean(novel_random)) if novel_random else 0.0
    steer_ratio = float(Mn / Mf) if Mf > 1e-9 else (float("inf") if Mn > 0 else float("nan"))
    out["steering"] = dict(mass_novel=Mn, mass_familiar=Mf, mass_baseline=Mb, steer_ratio_mass=steer_ratio,
                           events_novel=En, events_familiar=Ef, events_baseline=Eb,
                           dwell_novel=Dn, dwell_familiar=Df, dwell_baseline=Db,
                           pooled_member=pooled_member, pooled_random=pooled_random,
                           per_assembly_novel_mass=[a["mass_net"] for a in novel],
                           per_assembly_familiar_mass=[a["mass_net"] for a in familiar],
                           per_assembly_baseline_mass=[a["mass_net"] for a in base],
                           per_assembly_novel_events=[a["n_events"] for a in novel],
                           per_assembly_novel_dwell=[a["dwell"] for a in novel])

    # ---- per-seed GO gate ----
    def _silent(rec):
        return bool(rec["total_dwell"] <= 2 and rec["max_member"] < min_frac)
    internally_generated = bool(_silent(out["no_noise"]) and _silent(out["gain_only"]) and out["weights_frozen"]
                                and (out["apical_rest_max"] is None
                                     or out["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3))
    coherent = bool(En > 0 and pooled_member >= min_frac and pooled_member > 2.0 * (pooled_random + 1e-6))
    # steered: the NOVEL curiosity-tag surfaces the SAME thought MORE (reactivation mass) than the FAMILIAR tag AND
    # the no-tag baseline -> the surfacing tracks the curiosity VALUE, not the content identity.
    steered = bool(Mn > 0 and Mn >= 1.25 * max(Mf, 1e-9) and Mn >= 1.25 * max(Mb, 1e-9) and Dn >= max(Df, Db))
    store_lesion_ok = bool(out["store_lesion"]["total_dwell"] == 0
                           or out["store_lesion"]["pooled_member"] < 0.5 * pooled_member
                           or out["store_lesion"]["pooled_member"] < 2.0 * (out["store_lesion"]["pooled_random"] + 1e-6))
    # steering lesion-load-bearing: removing the curiosity tag (baseline) drops surfacing mass below the novel-tag.
    steer_lesion_load = bool(Mb < 0.8 * Mn)

    checks = dict(internally_generated=internally_generated, coherent=coherent, steered=steered,
                  store_lesion_load_bearing=store_lesion_ok, steer_lesion_load_bearing=steer_lesion_load)
    seed_go = bool(all(checks.values()))
    out["checks"] = checks; out["seed_go"] = seed_go
    print(f"  [seed {seed}] NOVEL  mass_net={[round(a['mass_net'],4) for a in novel]} (sum {Mn:.4f}) events {En:.0f} dwell {Dn:.0f} member {pooled_member:.2f} vs rand {pooled_random:.2f}", flush=True)
    print(f"  [seed {seed}] FAMIL  mass_net sum {Mf:.4f} events {Ef:.0f} dwell {Df:.0f} | BASE mass_net sum {Mb:.4f} events {Eb:.0f} dwell {Db:.0f} | ratio novel/fam(mass)={steer_ratio:.2f}", flush=True)
    print(f"  [seed {seed}] ACID no_noise dwell={out['no_noise']['total_dwell']:.0f} gain_only dwell={out['gain_only']['total_dwell']:.0f} | STORE-LESION member {out['store_lesion']['pooled_member']:.2f} vs rand {out['store_lesion']['pooled_random']:.2f}", flush=True)
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={checks}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--rest-steps", type=int, default=6000, help="rest steps for the tag/baseline/store-lesion conditions")
    ap.add_argument("--acid-steps", type=int, default=1000, help="rest steps for the NO-NOISE / GAIN-ONLY acid tests")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="curiosity recurrent-gain scale: novel concept gain = 1+scale (subthreshold: GAIN-ONLY noise-off stays silent)")
    ap.add_argument("--min-frac", type=float, default=0.30, help="assembly-active fraction to count a surfaced step")
    ap.add_argument("--smoke", action="store_true", help="smoke: >=50%% seeds GO; full gate is >=5/6")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    print(f"[self-init] n_mem={a.n_mem} rest_steps={a.rest_steps} gain_scale={a.gain_scale} noise=Poisson(r={NOISE_RATE},pA={NOISE_PA},dur={NOISE_DUR}) "
          f"seeds={a.seeds} backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a.n_mem, a.rest_steps, a.gain_scale, a.min_frac, a.acid_steps))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    attribution = None; preconditions = []
    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        thresh = max(1, (len(per) + 1) // 2) if a.smoke else max(1, (5 * len(per) + 5) // 6)
        go = n_go >= thresh
        mMn = float(np.mean([p["steering"]["mass_novel"] for p in per]))
        mMf = float(np.mean([p["steering"]["mass_familiar"] for p in per]))
        mMb = float(np.mean([p["steering"]["mass_baseline"] for p in per]))
        mmemb = float(np.nanmean([p["steering"]["pooled_member"] for p in per]))
        mrand = float(np.nanmean([p["steering"]["pooled_random"] for p in per]))

        # ATTRIBUTION (whose the surfacing is): what FRACTION of the novel-tag surfacing mass is OWNED by the
        # curiosity recurrent-gain vs the intrinsic no-tag baseline that runs in BOTH arms. This is the gap#5
        # 97%-clamp check applied here: if most of the surfacing were already in the baseline, the steering would
        # be cosmetic. (treatment = novel-tag mass; control = no-tag baseline mass, same substrate + noise.)
        attribution = attributable_to("curiosity-gain @ surfacing mass", mMn, mMb)

        # EARN the verdict: register the four anti-cheats as preconditions so the artifact CARRIES what earned it.
        vd = Verdict("self-initiated spontaneous thought (6-seed)", chance=mrand)
        vd.require("seeds passing all four anti-cheats >= threshold", n_go, expect=lambda x, t=thresh: x >= t)
        vd.control("curiosity-steered: novel-tag vs no-tag baseline (mass)", mMn, mMb, min_separation=0.05)
        vd.control("coherent: surfaced member vs random floor", mmemb, mrand, min_separation=0.15)
        vd.floor("coherence member above the random floor", mmemb, floor=mrand)
        vd.require("internally-generated: NO-NOISE + GAIN-ONLY acids silent every seed",
                   all(p["no_noise"]["total_dwell"] == 0 and p["gain_only"]["total_dwell"] == 0 for p in per),
                   expect=True)
        vd.require("store-lesion collapses coherence (NO-ENCODE member == 0) every seed",
                   all(p["store_lesion"]["pooled_member"] == 0.0 for p in per), expect=True)
        vd.require("plasticity byte-frozen during rest every seed",
                   all(p["weights_frozen"] for p in per), expect=True)
        vd.disabled("hebbian/BTSP plasticity during REST", "rest measures noise-seeded completion on a frozen store")
        decided = vd.decide(go)
        preconditions = decided["preconditions"]

        verdict = (f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{len(per)} -- a stored bistable CA3 store "
                   f"{'SPONTANEOUSLY reactivates coherent content under non-specific noise (no prompt) and a NOVEL curiosity-tag surfaces that same thought MORE than a FAMILIAR tag' if go else 'did NOT cleanly self-initiate a curiosity-steered thought'}: "
                   f"mean reactivation-mass novel-tag {mMn:.4f} vs familiar-tag {mMf:.4f} vs no-tag baseline {mMb:.4f} "
                   f"({100 * attribution:.0f}% of the surfacing attributable to the curiosity gain); "
                   f"coherence member {mmemb:.2f} vs random {mrand:.2f}. "
                   f"{'=> the FIRST self-initiated/spontaneous-thought correlate (internally-generated + coherent + curiosity-steered + lesion-load-bearing) is de-risked.' if go else 'Per THE LAW: tune gain_scale / noise / rest_steps / min_frac; not a stop.'}")
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        vd = Verdict("self-initiated spontaneous thought (6-seed)")
        vd.require("run completed without error", err is None, expect=True)
        preconditions = vd.decide(False)["preconditions"]

    summary = {"probe": "self_initiated_spontaneous_thought", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_mem": a.n_mem, "rest_steps": a.rest_steps, "gain_scale": a.gain_scale,
               "noise": {"rate": NOISE_RATE, "pa": NOISE_PA, "dur": NOISE_DUR},
               "curiosity_gain_attribution": attribution, "preconditions": preconditions,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110 + f"\n[self-init] VERDICT: {verdict}\n[self-init] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
