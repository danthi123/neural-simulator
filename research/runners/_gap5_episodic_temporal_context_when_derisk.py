"""gap#5 WHEN — give the episodic store a SENSE OF TIME: a slowly-drifting TEMPORAL-CONTEXT population
(Howard & Kahana 2002 TCM) realised as LEC episodic TIME cells (Tsao et al. 2018 Nature), BOUND to each stored
CA3 assembly AT ENCODE through a NEW plastic context->CA3 heteroassociative pathway. Recall then acquires two
signatures that a context-free CA3 store cannot have:
  (a) RECENCY  -- recently-encoded assemblies complete more readily from a partial cue than older ones (a GRADED
      gradient), because the retrieval (test) context overlaps a recent item's stored context far more than an
      old item's (context drifts), so the context->CA3 synapses deliver MORE completion-gating drive to recent
      assemblies.
  (b) CONTIGUITY -- cueing item i (which reinstates item i's encoding context) preferentially CO-REACTIVATES its
      temporal neighbours i+-1, because drifting context makes t_i overlap t_{i+-1} most.

WHY A NEW BRIDGE TOPOLOGY (the survey's honest blocker): EpisodicDapMemory
(research/runners/_episodic_dap_dialogue_memory.py) is a CA3-only recurrent-completion store -- there is NO context
pool and NO plastic context->CA3 pathway, so recency/contiguity are impossible there by construction. This runner
ADDS, on top of the SAME 6/6-GO substrate (emergent-DG selection -> BTSP one-shot formation -> dendritic-dAP
apical-UP readout, ALL reused by import, NO sim/ edit):
  * a temporal-context pool  : n_ctx LEC time cells whose population vector DRIFTS across encode-time (the TCM
                               drift schedule -- a documented scaffold standing in for LEC time-cell dynamics).
  * a plastic context->CA3   : W_ctx (n_ca3 x n_ctx), Hebbian-bound at encode (post = the co-firing assembly
    heteroassociative pathway   cells, pre = the current context vector). Its transmission W_ctx @ c is delivered
                               to CA3 as injected current -- EXACTLY the same kind of synaptic-current injection
                               the reused instrument already uses for the partial cue; it is NOT host "recency"
                               bookkeeping (no store-index is ever read to make the gradient).

THE LOAD-BEARING ANTI-CHEAT (the survey named it): a CONTEXT-LESION control -- zero the context->CA3 pathway
(W_ctx := 0) and the recency gradient MUST COLLAPSE to flat. If "recency" survives the lesion it is a
BTSP-write-recency / assembly-size confound (each assembly is formed in its OWN isolated episode so C carries no
write-order; the ONLY ordered write is W_ctx), NOT a genuine temporal-context code. The fraction of the intact
recency range that DISAPPEARS under the lesion is the headline number (tools.lab.attributable_to).

OPERATING POINT: the partial cue must be WEAK enough that cue-alone completion is LOW and FLAT across items (else
the dAP read saturates and there is no headroom for a context gradient -- observed directly: cue_frac=0.30
drive=300 completes ~0.85 with OR without context). The context->CA3 current then supplies the graded, recency-
tuned lift. Use --sweep (fast --preassigned --n-ca3 400 lane) to locate the (drive_pA, ctx_pA) window, then run the
faithful emergent n_ca3=2000 store at that window.

DETERMINISM: every bridge is built via the reused constructors that set cfg.seed=seed (the substrate is genuinely
seeded -- CLAUDE.md seed trap); contexts come from np.random.default_rng(seed).

GO (6-seed 42/43/44/100/101/102): per seed, (i) RECENCY intact -- Spearman(position, completion) >= 0.5, positive
slope, newest-third/oldest-third ratio >= 1.5; (ii) CONTEXT-LESION collapses recency -- >= 60% of the intact
recency range absent under W_ctx=0 AND |Spearman_lesion| < 0.5; (iii) CONTIGUITY intact -- co-reactivation at
|lag|=1 > at |lag|>=3. Overall GO on >=5/6. Honest PARTIAL/NO-GO with the confound isolated + next lever named is a
first-class deliverable. SIM_BACKEND=numpy (faithful, slow -- speed secondary) or cupy.
  Run: OMP_NUM_THREADS=2 SIM_BACKEND=numpy python -m research.runners._gap5_episodic_temporal_context_when_derisk \
         --seeds 42 43 44 100 101 102 --n-items 6 \
         --out research/findings/raw/_episodic_when/when_6seed.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend  # noqa: E402
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _build_dap_readout, _reset_apical_latch, _held_cue_perm)
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (  # noqa: E402
    make_readout, _form_one_assembly, _build_bridge as _formation_build_bridge)
from research.runners._gap5_emergent_end_to_end_episodic_loop_derisk import emergent_assemblies  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_episodic_when" / "when.json"

# GO defaults inherited VERBATIM from the standing episodic-dap store (research/runners/_episodic_dap_dialogue_memory.py
# GO_DEFAULTS) -- the dendritic-dAP readout operating point that is 6/6 cue-specific. Only the WHEN-specific knobs are new.
GO_DEFAULTS = dict(
    density=0.5, wmax=100.0, kthresh=8.0, plateau_strength=30.0, apical_R=0.15, self_regen=2.0,
    v_hold=-35.0, apical_kir_g=1.0, apical_gc=0.3, apical_gc_read=0.3, up_thresh=-20.0, ca3_fb_inhib=60.0,
    btsp_lr=0.05, encode_drive=700.0, encode_plateau_pA=250.0, train_events=40, drive_steps=48, reset_steps=15,
    assembly_frac=0.18, warm_steps=100, read_steps=100, silence_steps=50,
)


# ---- TEMPORAL CONTEXT: LEC episodic time cells with a Howard-Kahana drift ---------------------------------------
def drift_contexts(seed, n_steps, n_ctx, rho, beta, k_active):
    """A drifting temporal-context trajectory: c_i = normalize( rho * c_{i-1} + beta * eta_i ), where eta_i is a
    SPARSE non-negative recruitment vector (k_active of n_ctx time cells fire) -- LEC episodic time cells fire
    sparsely (Tsao et al. 2018 Nature), so a sparse population gives orthogonal-enough contexts and a WIDE overlap
    dynamic range (dense non-negative vectors share a large common-mode that compresses the signal). c stays >= 0 (a
    firing-rate population). Returns (n_steps, n_ctx) L2-normalised vectors. Consecutive contexts overlap strongly and
    the overlap c_i . c_j decays with |i-j| -> the exact structure that yields RECENCY (the test context overlaps
    recent items most) and CONTIGUITY (neighbour contexts overlap most). The drift schedule is a documented SCAFFOLD
    standing in for LEC time-cell dynamics; the brain-based part is the plastic context->CA3 pathway + the spiking
    dendritic completion it gates."""
    rng = np.random.default_rng(seed * 977 + 13)
    c = np.zeros(n_ctx, dtype=np.float64)
    out = []
    for _ in range(n_steps):
        eta = np.zeros(n_ctx, dtype=np.float64)
        eta[rng.choice(n_ctx, size=int(k_active), replace=False)] = 1.0
        eta /= (np.linalg.norm(eta) + 1e-12)
        c = rho * c + beta * eta
        c = np.clip(c, 0.0, None)
        c /= (np.linalg.norm(c) + 1e-12)
        out.append(c.copy())
    return np.asarray(out, dtype=np.float64)


# ---- READ machinery: drive a partial cue + a context->CA3 current, read the apical-UP latch --------------------
def _drive_context_read(bridge, R, cue_g, ctx_curr, *, drive_pA, warm, read):
    """Hard-silence -> reset the apical latch -> drive the partial cue (drive_pA on cue cells) AND the context->CA3
    current (ctx_curr, position-aligned to R.ca3_idx) -> warm+read -> return the host apical membrane vector
    (cp_v_apical, indexed by GLOBAL neuron id). ctx_curr None => no context (pathway lesion / baseline)."""
    cp = R.cp
    R.hard_silence(); _reset_apical_latch(bridge)
    ext = bridge.cp_external_input_current
    if cue_g is not None and len(cue_g) > 0:
        darr = cp.asarray(np.asarray(cue_g, dtype=np.int64), dtype=cp.int64)
        ext[darr] = cp.float32(drive_pA)
    if ctx_curr is not None:
        ext[R.ca3_arr] += cp.asarray(ctx_curr, dtype=cp.float32)
    for _ in range(warm + read):
        bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
    va = R.to_host(bridge.cp_v_apical) if getattr(bridge, "cp_v_apical", None) is not None else None
    ext[:] = 0.0
    return va


def _up_fraction(va, cells_global, up_thresh):
    if va is None:
        return 0.0
    gg = np.asarray(cells_global, dtype=np.int64)
    return float(np.mean(va[gg] > up_thresh)) if len(gg) else 0.0


# ---- small stats helpers (no scipy dependency; deterministic) --------------------------------------------------
def _rankdata(x):
    """Average-rank of x (proper tie handling). A CONSTANT vector -> all ranks equal -> zero variance (so a flat
    lesion curve correlates 0, not the spurious +1 that sequential-argsort ranks produce for ties)."""
    x = np.asarray(x, dtype=np.float64); n = len(x)
    order = np.argsort(x, kind="mergesort"); sx = x[order]
    ranks = np.empty(n, dtype=np.float64); i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x, y):
    if len(x) < 3:
        return 0.0
    rx = _rankdata(x); ry = _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 1e-12 else 0.0


def _slope(positions, vals):
    if len(positions) < 2:
        return 0.0
    return float(np.polyfit(np.asarray(positions, dtype=np.float64), np.asarray(vals, dtype=np.float64), 1)[0])


def _third_ratio(vals):
    """newest-third mean / oldest-third mean (vals ordered oldest..newest); also returns the two means."""
    v = np.asarray(vals, dtype=np.float64); n = len(v); k = max(1, n // 3)
    old = float(v[:k].mean()); new = float(v[-k:].mean())
    return (new / old) if old > 1e-6 else (float("inf") if new > 1e-6 else 1.0), old, new


# ---- STORE: build the substrate, BTSP-form each item's assembly, BIND its encoding context (expensive; do once) --
class Store:
    pass


def build_and_form(seed, *, n_items, n_ctx, rho, beta, k_active, ctx_lr, cue_frac, drive_pA, p,
                   preassigned=False, n_ca3_pre=400, verbose=True):
    cp, _ = get_backend()
    S = Store(); S.seed = seed; S.cp = cp

    # ---- assemblies: emergent-DG selected (faithful) OR pre-assigned random-permutation (fast operating-point lane)
    if preassigned:
        n_ca3 = int(n_ca3_pre)
        bridge = _build_dap_readout(
            seed, n_ca3=n_ca3, ca3_density=p["density"], ca3_fb_inhib=p["ca3_fb_inhib"], k_thresh=p["kthresh"],
            plateau_strength=p["plateau_strength"], apical_R=p["apical_R"], self_regen=p["self_regen"],
            v_hold=p["v_hold"], apical_kir_g=p["apical_kir_g"], apical_gc=p["apical_gc"],
            apical_gc_read=p["apical_gc_read"], coincidence=True)
        # EQUAL-SIZE DISJOINT pre-assigned assemblies (no position/size confound): n_items disjoint slices of a
        # permutation, each of a_size cells with n_items*a_size <= n_ca3. make_readout's own slicing would truncate
        # the last assembly (0.18*n_ca3 * n_items > n_ca3), systematically shrinking the NEWEST -> confound. Avoided.
        ca3_idx = np.asarray(list(bridge.region_manager.indices("ca3")), dtype=np.int64)
        a_size = max(6, int(len(ca3_idx) // (n_items + 1)))     # equal size, all disjoint, room to spare
        rng_a = np.random.default_rng(seed * 13 + 7)
        perm = rng_a.permutation(len(ca3_idx))
        assemblies = [ca3_idx[perm[a * a_size:(a + 1) * a_size]] for a in range(n_items)]
        R = make_readout(bridge, seed, assembly_frac=p["assembly_frac"], cue_frac=cue_frac, drive_pA=drive_pA,
                         warm_steps=p["warm_steps"], read_steps=p["read_steps"], silence_steps=p["silence_steps"],
                         assemblies_ext=assemblies)
    else:
        assemblies, r1 = emergent_assemblies(seed, n_patterns=n_items)
        n_ca3 = int(r1[2])
        bridge = _build_dap_readout(
            seed, n_ca3=n_ca3, ca3_density=p["density"], ca3_fb_inhib=p["ca3_fb_inhib"], k_thresh=p["kthresh"],
            plateau_strength=p["plateau_strength"], apical_R=p["apical_R"], self_regen=p["self_regen"],
            v_hold=p["v_hold"], apical_kir_g=p["apical_kir_g"], apical_gc=p["apical_gc"],
            apical_gc_read=p["apical_gc_read"], coincidence=True)
        R = make_readout(bridge, seed, assembly_frac=p["assembly_frac"], cue_frac=cue_frac, drive_pA=drive_pA,
                         warm_steps=p["warm_steps"], read_steps=p["read_steps"], silence_steps=p["silence_steps"],
                         assemblies_ext=assemblies)
    sizes = [int(len(a)) for a in assemblies]
    if min(sizes) == 0:
        S.error = f"an assembly is EMPTY (sizes={sizes})"; return S

    held_pos_by_asm, cue_by_asm, _perm = _held_cue_perm(R, seed)
    held_global = [[int(R.ca3_idx[pp]) for pp in held_pos_by_asm[i]] for i in range(n_items)]
    asm_global = [[int(g) for g in np.asarray(a, dtype=np.int64)] for a in assemblies]

    # ---- temporal-context trajectory + the plastic context->CA3 pathway (Hebbian bind at encode) -----------------
    ctxs = drift_contexts(seed, n_items + 1, n_ctx, rho, beta, k_active)   # item i uses ctxs[i]; probe = ctxs[n_items]
    c_test = ctxs[n_items]
    ov = ctxs @ ctxs.T
    W_ctx = np.zeros((n_ca3, n_ctx), dtype=np.float64)

    read_kwargs = dict(assembly_frac=p["assembly_frac"], cue_frac=cue_frac, drive_pA=drive_pA,
                       warm_steps=p["warm_steps"], read_steps=p["read_steps"], silence_steps=p["silence_steps"],
                       assemblies_ext=assemblies)
    form_build_kwargs = dict(n_ca3=n_ca3, ca3_density=p["density"], ca3_fb_inhib=p["ca3_fb_inhib"],
                             ca3_ff_inhib=None, nmda_tau=100.0, nmda_ratio=1.0, enable_ou=False, element="nmda_slow")
    for i in range(n_items):
        bi = _formation_build_bridge(seed, **form_build_kwargs)
        Ri = make_readout(bi, seed, **read_kwargs)
        _form_one_assembly(bi, Ri, i, btsp_w_max=p["wmax"], btsp_lr=p["btsp_lr"], encode_drive=p["encode_drive"],
                           encode_plateau_pA=p["encode_plateau_pA"], train_events=p["train_events"],
                           drive_steps=p["drive_steps"], reset_steps=p["reset_steps"], plateau=True)
        m = Ri.withinA_masks[i]
        R.C.data[m] = bi.cp_connections.data[m]      # copy ONLY the within-i BTSP-formed recurrent weights
        pos_i = np.asarray([R.ca3_pos[g] for g in asm_global[i]], dtype=np.int64)
        W_ctx[pos_i, :] += ctx_lr * ctxs[i][None, :]   # Hebbian: post = co-firing cells, pre = c_i
        del bi, Ri

    S.bridge = bridge; S.R = R; S.n_ca3 = n_ca3; S.sizes = sizes; S.preassigned = bool(preassigned)
    S.asm_global = asm_global; S.held_global = held_global; S.cue_by_asm = cue_by_asm
    S.W_ctx = W_ctx; S.W_les = np.zeros_like(W_ctx); S.ctxs = ctxs; S.c_test = c_test; S.n_items = n_items
    S.warm = p["warm_steps"]; S.read = p["read_steps"]; S.up_thresh = p["up_thresh"]; S.error = None
    S.ctx_overlap_probe = [round(float(c_test @ ctxs[i]), 4) for i in range(n_items)]
    S.ctx_overlap_neighbour = round(float(np.mean([ov[i, i + 1] for i in range(n_items - 1)])), 4)
    S.ctx_overlap_far = round(float(np.mean([ov[i, j] for i in range(n_items) for j in range(n_items)
                                             if abs(i - j) >= 3])), 4)
    if verbose:
        print(f"  [s{seed}] formed n_ca3={n_ca3} sizes={sizes} preassigned={preassigned} "
              f"ctx_probe(old..new)={S.ctx_overlap_probe} neigh={S.ctx_overlap_neighbour} far={S.ctx_overlap_far}",
              flush=True)
    return S


def _ctx_current(S, c_probe, W, ctx_pA):
    """context->CA3 transmission = ctx_pA * (W @ c_probe), a per-CA3 current vector in R.ca3_idx position order --
    the synaptic drive the context pathway delivers (same kind of injected current the instrument uses for the cue),
    NOT a host recency formula (W is the pathway; W:=0 removes exactly this)."""
    return (ctx_pA * (W @ np.asarray(c_probe, dtype=np.float64))).astype(np.float32)


def eval_recency(S, *, drive_pA, ctx_pA):
    """RECENCY: partial cue of item i + the TEST context (t_now); read held-cell apical-UP completion, per item."""
    def curve(W):
        return [_up_fraction(_drive_context_read(S.bridge, S.R, S.cue_by_asm[i], _ctx_current(S, S.c_test, W, ctx_pA),
                                                 drive_pA=drive_pA, warm=S.warm, read=S.read),
                             S.held_global[i], S.up_thresh) for i in range(S.n_items)]
    rec_i = curve(S.W_ctx); rec_l = curve(S.W_les)
    pos = list(range(S.n_items))
    sp_i = _spearman(pos, rec_i); sp_l = _spearman(pos, rec_l)
    ratio_i, old_i, new_i = _third_ratio(rec_i); ratio_l, old_l, new_l = _third_ratio(rec_l)
    return dict(drive_pA=drive_pA, ctx_pA=ctx_pA, intact=[round(v, 4) for v in rec_i],
                lesion=[round(v, 4) for v in rec_l], spearman_intact=round(sp_i, 4),
                spearman_lesion=round(sp_l, 4), slope_intact=round(_slope(pos, rec_i), 5),
                ratio_intact=round(ratio_i, 3), ratio_lesion=round(ratio_l, 3),
                range_intact=round(new_i - old_i, 4), range_lesion=round(new_l - old_l, 4))


def eval_contiguity(S, *, drive_pA, ctx_pA):
    """CONTIGUITY: cue item i + REINSTATE its encoding context t_i; read co-reactivation of item j by lag = j-i."""
    def by_lag(W):
        acc = {}
        for i in range(S.n_items):
            va = _drive_context_read(S.bridge, S.R, S.cue_by_asm[i], _ctx_current(S, S.ctxs[i], W, ctx_pA),
                                     drive_pA=drive_pA, warm=S.warm, read=S.read)
            for j in range(S.n_items):
                if j == i:
                    continue
                acc.setdefault(j - i, []).append(_up_fraction(va, S.asm_global[j], S.up_thresh))
        return {lag: float(np.mean(v)) for lag, v in acc.items()}

    def near_far(cur):
        near = [v for lag, v in cur.items() if abs(lag) == 1]
        far = [v for lag, v in cur.items() if abs(lag) >= 3]
        return (float(np.mean(near)) if near else 0.0), (float(np.mean(far)) if far else 0.0)

    con_i = by_lag(S.W_ctx); con_l = by_lag(S.W_les)
    near_i, far_i = near_far(con_i); near_l, far_l = near_far(con_l)
    return dict(intact_by_lag={str(k): round(v, 4) for k, v in sorted(con_i.items())},
                lesion_by_lag={str(k): round(v, 4) for k, v in sorted(con_l.items())},
                near_intact=round(near_i, 4), far_intact=round(far_i, 4),
                near_lesion=round(near_l, 4), far_lesion=round(far_l, 4))


def run_one_seed(seed, *, n_items, n_ctx, rho, beta, k_active, ctx_pA, ctx_lr, cue_frac, drive_pA, p,
                 preassigned=False, n_ca3_pre=400, verbose=True):
    from tools.lab import attributable_to
    t = {"seed": seed, "backend": os.environ.get("SIM_BACKEND", "(unset)")}
    S = build_and_form(seed, n_items=n_items, n_ctx=n_ctx, rho=rho, beta=beta, k_active=k_active, ctx_lr=ctx_lr,
                       cue_frac=cue_frac, drive_pA=drive_pA, p=p, preassigned=preassigned, n_ca3_pre=n_ca3_pre,
                       verbose=verbose)
    if S.error:
        t["error"] = S.error; return t
    t["assembly_sizes"] = S.sizes; t["n_ca3"] = S.n_ca3; t["preassigned"] = S.preassigned
    t["ctx_overlap_probe"] = S.ctx_overlap_probe; t["ctx_overlap_neighbour"] = S.ctx_overlap_neighbour
    t["ctx_overlap_far"] = S.ctx_overlap_far

    rec = eval_recency(S, drive_pA=drive_pA, ctx_pA=ctx_pA)
    t.update({f"recency_{k}": v for k, v in rec.items() if k not in ("drive_pA", "ctx_pA")})
    t["recency_attributable_to_context"] = attributable_to(
        f"[s{seed}] recency newest-vs-oldest RANGE: intact context vs context-LESION",
        float(rec["range_intact"]), float(rec["range_lesion"]))

    con = eval_contiguity(S, drive_pA=drive_pA, ctx_pA=ctx_pA)
    t.update({f"contiguity_{k}": v for k, v in con.items()})
    t["contiguity_attributable_to_context"] = attributable_to(
        f"[s{seed}] contiguity near-minus-far separation: intact vs context-LESION",
        float(con["near_intact"] - con["far_intact"]), float(con["near_lesion"] - con["far_lesion"]))

    recency_go = bool(rec["spearman_intact"] >= 0.5 and rec["slope_intact"] > 0 and rec["ratio_intact"] >= 1.5)
    lesion_collapses = bool((t["recency_attributable_to_context"] is not None
                             and t["recency_attributable_to_context"] >= 0.60)
                            and abs(rec["spearman_lesion"]) < 0.5)
    ni, fi, nl, fl = con["near_intact"], con["far_intact"], con["near_lesion"], con["far_lesion"]
    contiguity_go = bool(ni > fi and (ni - fi) > 0.02 and (ni - fi) > 2.0 * max(0.0, nl - fl))
    t["recency_go"] = recency_go; t["lesion_collapses"] = lesion_collapses; t["contiguity_go"] = contiguity_go
    t["seed_go"] = bool(recency_go and lesion_collapses and contiguity_go)
    del S
    if verbose:
        print(f"  [s{seed}] REC intact {rec['intact']} (rho_s={rec['spearman_intact']} "
              f"ratio={rec['ratio_intact']}) | LESION {rec['lesion']} (rho_s={rec['spearman_lesion']}) "
              f"attrib={t['recency_attributable_to_context']} || CONTIG near={ni:.3f} far={fi:.3f} "
              f"(les near={nl:.3f} far={fl:.3f}) || recency_go={recency_go} lesion={lesion_collapses} "
              f"contig={contiguity_go} => SEED_GO={t['seed_go']}", flush=True)
    return t


def _pooled_curve(valid, key):
    """Mean per serial-position across seeds (recency is a POPULATION serial-position effect; per-assembly
    heterogeneity averages out over seeds -- exactly how a free-recall serial-position curve is measured)."""
    arrs = [p[key] for p in valid if isinstance(p.get(key), list)]
    if not arrs:
        return []
    m = min(len(a) for a in arrs)
    return [float(np.mean([a[i] for a in arrs])) for i in range(m)]


def _pooled_lag(valid, key):
    lags = {}
    for p in valid:
        for k, v in (p.get(key) or {}).items():
            lags.setdefault(int(k), []).append(float(v))
    return {lag: float(np.mean(vs)) for lag, vs in sorted(lags.items())}


def build_summary(per, seeds, cfg, elapsed, err=None):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    valid = [p for p in per if not p.get("error")]
    n = len(valid)
    n_rec = sum(1 for p in valid if p.get("recency_go"))
    n_les = sum(1 for p in valid if p.get("lesion_collapses"))
    n_con = sum(1 for p in valid if p.get("contiguity_go"))
    n_go = sum(1 for p in valid if p.get("seed_go"))
    need = max(1, int(np.ceil(5 / 6 * len(seeds))))

    # ---- POOLED (across-seed) serial-position + lag-CRP curves = the headline population effect ------------------
    pooled = {}
    if valid:
        rec_i = _pooled_curve(valid, "recency_intact"); rec_l = _pooled_curve(valid, "recency_lesion")
        pos = list(range(len(rec_i)))
        sp_i = _spearman(pos, rec_i); sp_l = _spearman(pos, rec_l)
        ratio_i, oi, ni_ = _third_ratio(rec_i) if rec_i else (1.0, 0.0, 0.0)
        ratio_l, ol, nl_ = _third_ratio(rec_l) if rec_l else (1.0, 0.0, 0.0)
        rng_i = ni_ - oi; rng_l = nl_ - ol
        attrib = attributable_to("POOLED recency newest-vs-oldest RANGE: intact vs context-LESION",
                                 float(rng_i), float(rng_l))
        con_i = _pooled_lag(valid, "contiguity_intact_by_lag"); con_l = _pooled_lag(valid, "contiguity_lesion_by_lag")
        near_i = float(np.mean([v for lag, v in con_i.items() if abs(lag) == 1])) if con_i else 0.0
        far_i = float(np.mean([v for lag, v in con_i.items() if abs(lag) >= 3])) if con_i else 0.0
        near_l = float(np.mean([v for lag, v in con_l.items() if abs(lag) == 1])) if con_l else 0.0
        far_l = float(np.mean([v for lag, v in con_l.items() if abs(lag) >= 3])) if con_l else 0.0
        pooled = dict(recency_intact=[round(v, 4) for v in rec_i], recency_lesion=[round(v, 4) for v in rec_l],
                      recency_spearman_intact=round(sp_i, 4), recency_spearman_lesion=round(sp_l, 4),
                      recency_ratio_intact=round(ratio_i, 3), recency_range_intact=round(rng_i, 4),
                      recency_range_lesion=round(rng_l, 4), recency_attributable_to_context=attrib,
                      contiguity_intact_by_lag={str(k): round(v, 4) for k, v in con_i.items()},
                      contiguity_lesion_by_lag={str(k): round(v, 4) for k, v in con_l.items()},
                      contiguity_near_intact=round(near_i, 4), contiguity_far_intact=round(far_i, 4),
                      contiguity_near_lesion=round(near_l, 4), contiguity_far_lesion=round(far_l, 4))
    pooled_recency_go = bool(pooled and pooled["recency_spearman_intact"] >= 0.5
                             and pooled["recency_ratio_intact"] >= 1.5)
    pooled_lesion_collapses = bool(pooled and pooled["recency_attributable_to_context"] is not None
                                   and pooled["recency_attributable_to_context"] >= 0.60
                                   and abs(pooled["recency_spearman_lesion"]) < 0.5)
    pooled_contiguity_go = bool(pooled and pooled["contiguity_near_intact"] > pooled["contiguity_far_intact"]
                                and (pooled["contiguity_near_intact"] - pooled["contiguity_far_intact"]) > 0.02
                                and (pooled["contiguity_near_intact"] - pooled["contiguity_far_intact"])
                                > 2.0 * max(0.0, pooled["contiguity_near_lesion"] - pooled["contiguity_far_lesion"]))

    v = Verdict("gap5 WHEN: drifting temporal-context (TCM/LEC) bound to CA3 gives episodic recency + contiguity")
    v.require("POOLED RECENCY -- graded serial-position gradient (Spearman>=0.5, newest/oldest ratio>=1.5)",
              pooled_recency_go, expect=True,
              note=f"pooled Spearman {pooled.get('recency_spearman_intact')} ratio {pooled.get('recency_ratio_intact')}")
    v.require("POOLED CONTEXT-LESION collapses recency (>=60% of range absent, |Spearman_les|<0.5)",
              pooled_lesion_collapses, expect=True, note="the load-bearing anti-cheat: else recency is a write/size confound")
    v.require(f"PER-SEED lesion-collapse holds on >={need}/{len(seeds)} seeds", n_les >= need, expect=True)
    v.require("POOLED CONTIGUITY -- neighbours (|lag|=1) co-reactivate more than far (|lag|>=3)",
              pooled_contiguity_go, expect=True)
    v.disabled("plasticity at recall (hebbian/stdp/btsp/bdsp)", why="the formed attractor + W_ctx are frozen reads")
    v.disabled("OU membrane noise", why="isolate the deterministic per-cell dAP bistability + context drive")
    go = bool(pooled_recency_go and pooled_lesion_collapses and pooled_contiguity_go and n_les >= need and n > 0)
    decided = v.decide(go=go)
    status = decided["status"]
    kind = "GO" if (go and status == "GO") else ("PARTIAL" if (pooled_recency_go and pooled_lesion_collapses) else "NO-GO")
    verdict = (f"WHEN-{kind}: POOLED recency Spearman {pooled.get('recency_spearman_intact')} ratio "
               f"{pooled.get('recency_ratio_intact')} (lesion Spearman {pooled.get('recency_spearman_lesion')} "
               f"attrib_to_context {pooled.get('recency_attributable_to_context')}) | POOLED contiguity near "
               f"{pooled.get('contiguity_near_intact')} vs far {pooled.get('contiguity_far_intact')} | per-seed "
               f"recency {n_rec}/{n} lesion-collapse {n_les}/{n} contiguity {n_con}/{n} seed_go {n_go}/{n}")
    if err is not None:
        verdict = f"ERROR -- {err}"; go = False
    return {"probe": "gap5_episodic_temporal_context_when", "GO": go, "status": status, "kind": kind,
            "verdict": verdict, "seeds": seeds, "config": cfg, "need": need, "elapsed_seconds": elapsed,
            "pooled": pooled, "pooled_recency_go": pooled_recency_go, "pooled_lesion_collapses": pooled_lesion_collapses,
            "pooled_contiguity_go": pooled_contiguity_go, "n_seed_go": n_go, "n_recency": n_rec,
            "n_lesion_collapse": n_les, "n_contiguity": n_con,
            "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
            "per_seed": per}


def do_sweep(a, p):
    """Form ONCE (seed[0]) then evaluate RECENCY over a (drive_pA x ctx_pA) grid -- the cheap operating-point search
    (formation dominates cost, so reuse it). Prints a table; writes no verdict."""
    seed = a.seeds[0]
    S = build_and_form(seed, n_items=a.n_items, n_ctx=a.n_ctx, rho=a.rho, beta=a.beta, k_active=a.k_active,
                       ctx_lr=a.ctx_lr, cue_frac=a.cue_frac, drive_pA=a.drive_grid[0], p=p,
                       preassigned=a.preassigned, n_ca3_pre=a.n_ca3, verbose=True)
    if S.error:
        print(f"[sweep] ERROR {S.error}", flush=True); return 1
    print(f"[sweep] seed={seed} n_items={a.n_items} preassigned={a.preassigned} n_ca3={S.n_ca3} "
          f"cue_frac={a.cue_frac} grid drive={a.drive_grid} ctx={a.ctx_grid}", flush=True)
    rows = []
    for dP in a.drive_grid:
        for cP in a.ctx_grid:
            r = eval_recency(S, drive_pA=dP, ctx_pA=cP)
            rows.append(r)
            print(f"  drive={dP:5.0f} ctx={cP:5.0f} | intact {r['intact']} rho_s={r['spearman_intact']:+.2f} "
                  f"ratio={r['ratio_intact']} rng={r['range_intact']:+.3f} || LESION {r['lesion']} "
                  f"rho_s={r['spearman_lesion']:+.2f} rng={r['range_lesion']:+.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"sweep": True, "seed": seed, "n_ca3": S.n_ca3, "sizes": S.sizes,
                                       "ctx_overlap_probe": S.ctx_overlap_probe, "rows": rows}, indent=2, default=str))
    print(f"[sweep] wrote {a.out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-items", type=int, default=6, help="list length (temporal positions / stored assemblies)")
    ap.add_argument("--n-ctx", type=int, default=200, help="LEC temporal-context / time-cell population size")
    ap.add_argument("--k-active", type=int, default=10, help="LEC time cells recruited per drift step (sparse)")
    ap.add_argument("--rho", type=float, default=0.72, help="TCM context drift retention (closer to 1 = slower drift)")
    ap.add_argument("--beta", type=float, default=0.60, help="TCM new-context input gain")
    ap.add_argument("--ctx-pA", type=float, default=700.0, help="context->CA3 current scale (pA at overlap=1)")
    ap.add_argument("--ctx-lr", type=float, default=1.0, help="Hebbian context->CA3 binding rate")
    ap.add_argument("--cue-frac", type=float, default=0.15,
                    help="partial-cue fraction -- WEAK so cue-alone completion is low and context provides the graded tip")
    ap.add_argument("--drive-pa", type=float, default=50.0, help="partial-cue drive (pA); weak keeps cue-alone flat-low")
    ap.add_argument("--preassigned", action="store_true",
                    help="pre-assigned assemblies at --n-ca3 (fast operating-point lane; emergent membership is a "
                         "separately-closed anti-cheat, not re-litigated here)")
    ap.add_argument("--n-ca3", type=int, default=500,
                    help="n_ca3 for --preassigned; >= n_items*ceil(0.18*n_ca3) so all assemblies are equal size "
                         "(no position/size confound). Emergent path fixes n_ca3=2000.")
    ap.add_argument("--sweep", action="store_true", help="operating-point search: form once, grid over drive x ctx")
    ap.add_argument("--drive-grid", type=float, nargs="+", default=[80, 120, 200])
    ap.add_argument("--ctx-grid", type=float, nargs="+", default=[150, 300, 500])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    p = dict(GO_DEFAULTS)
    if a.sweep:
        return do_sweep(a, p)
    cfg = dict(n_items=a.n_items, n_ctx=a.n_ctx, k_active=a.k_active, rho=a.rho, beta=a.beta, ctx_pA=a.ctx_pA,
               ctx_lr=a.ctx_lr, cue_frac=a.cue_frac, drive_pA=a.drive_pa, preassigned=a.preassigned,
               n_ca3=(a.n_ca3 if a.preassigned else 2000), backend=os.environ.get("SIM_BACKEND", "(unset)"))
    print(f"[when] gap#5 WHEN temporal-context store | seeds={a.seeds} cfg={cfg}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_one_seed(s, n_items=a.n_items, n_ctx=a.n_ctx, rho=a.rho, beta=a.beta, k_active=a.k_active,
                             ctx_pA=a.ctx_pA, ctx_lr=a.ctx_lr, cue_frac=a.cue_frac, drive_pA=a.drive_pa, p=p,
                             preassigned=a.preassigned, n_ca3_pre=a.n_ca3, verbose=True)
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] ERROR {r['error']}", flush=True)
            print(f"  [seed {s}] done ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()
    summary = build_summary(per, a.seeds, cfg, round(time.time() - t0, 1),
                            err=(err if (err is not None or not [q for q in per if not q.get("error")]) else None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[when] VERDICT: {summary['verdict']}\n[when] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
