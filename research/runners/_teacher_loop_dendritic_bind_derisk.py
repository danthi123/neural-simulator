"""TEACHER-LOOP DENDRITIC-BIND DE-RISK (2026-08-09): make the conjunction a REAL SPIKING DENDRITIC COINCIDENCE.

WHERE THE ARC STANDS. The biologized composer has BOTH ops on spikes: BUNDLE (superposition) + BIND (conjunction).
BIND = _teacher_loop_conjunctive_binding_derisk.py's `binding` arm: a per-primitive spiking-reservoir readout produces
two factor vectors fA[a], fB[b] (d_p each) and the runtime conjunction is their HOST elementwise product fA (o) fB (a
functional dendritic-AND done as numpy `*`). It recovers zero-shot held-out composition where additive superposition
BREAKS (max mixing s=1.0: additive -> ~chance; readout-product bind -> 0.77-0.83, measured in-run here as the target).

THE RESIDUAL THIS DE-RISK CLOSES. Replace the HOST `*` with a REAL DENDRITIC COINCIDENCE. The substrate has a
two-compartment pyramidal (sim/dendritic_neuron.py: basal + apical compartments, Larkum BAC). Route primitive-a drive
to the APICAL compartment and primitive-b drive to the BASAL compartment of a dendritic unit; the apical x basal
coincidence NONLINEARITY (dendritic plateau / AND) computes the conjunction IN THE NEURON -- not a runner `*`. Does the
dendritic bind RECOVER zero-shot held-out composition at high mixing (s=0.75/1.0) like the readout-product bind, vs the
additive floor? An HONEST NEGATIVE (the plateau nonlinearity is too lossy to carry the bilinear factors) is first-class.

THE MECHANISM (sim/dendritic_neuron.py:DendriticLayer.apical_basal_coincidence -- an ADDITIVE / DEFAULT-OFF guarded
addition; step() is byte-identical; git-diff asserted additions-only). The neuron computes soma = phi(basal)*phi(apical)
where phi is the NON-NEGATIVE saturating dendritic plateau (Michaelis-Menten / finite NMDA-Ca conductance; phi(0)=0 ->
a genuine coincidence AND: no output unless BOTH compartments are engaged). The MULTIPLICATION is the dendritic unit's
intrinsic sigma-pi operation (Larkum 2013 BAC firing; Mel/Poirazi two-layer; catalog G.02+J.08), NOT a host product of
two precomputed answers. Signed factors are carried by biological PUSH-PULL: separate ON/OFF drive channels
aP=relu(fA), aN=relu(-fA), bP=relu(fB), bN=relu(-fB) (all >=0, the rate a real spiking population carries), combined at
the soma by excitatory (same-sign) + inhibitory (opposite-sign) dendritic branches:
    dendritic_bind(a,b)_c = g_out * ( coinc(bP,aP)_c + coinc(bN,aN)_c - coinc(bN,aP)_c - coinc(bP,aN)_c )
with coinc(x,y)_c = phi(x_c)*phi(y_c) the per-channel dendritic coincidence (identity/labeled-line wiring
W_basal=B_apical=I so channel c's apical synapse carries fA[.,c] and its basal synapse carries fB[.,c]). At the
small-signal limit phi(z)~z -> the four terms sum to the exact product; the plateau's saturation is the honest source
of any degradation. The runner NEVER computes fA (o) fB -- the only `*` in the conjunction path is the neuron's soma.

ARMS (all on the ONE frozen spiking Izhikevich reservoir; readout-only; de-clamped bdsp_wmax=1e9):
  * dendritic_bind (TREATMENT): additive part + the DENDRITIC coincidence of the two spiking-readout factor outputs.
  * readout_product_bind (IN-RUN BASELINE = the target to match): the conjunctive-binding runner's `binding` -- SAME
    factors, combined by the host `*`. Measured here so dendritic-vs-product is like-for-like on the SAME reservoir.
  * additive (FLOOR): neural superposition (VSA bundling) -- breaks at high s (~chance).

CALIBRATION (homeostatic, taught-only, ruler-free): z0_basal/z0_apical = median |compartment drive| over TAUGHT cells
(the plateau half-saturation -> the operating point genuinely bends, not a linear pass-through); g_out = one scalar so
the dendritic conjunction term's RMS matches the host-product term's RMS over TAUGHT cells. NO held-out / ruler read.

ANTI-CHEATS (each a real assertion / witness in the output):
  * the conjunction is a REAL DENDRITIC COINCIDENCE, NOT a host `*`: computed by DendriticLayer.apical_basal_coincidence
    (soma = phi(basal)*phi(apical)); the AND anchor coinc(x,0)=coinc(0,y)=0 holds (no output without both compartments);
    the coincidence is genuinely NONLINEAR (mean |soma - best-linear-fit(product)| / RMS(soma) > 0 -> not the linear
    product); reservoir mean spikes > 0 (the factors come from spiking readouts).
  * genuinely ZERO-SHOT (disjoint taught/held-out; every held-out primitive seen in >=1 taught combo; NO leakage).
  * cfg.seed byte-identical substrate; de-clamped bdsp_wmax=1e9; git diff main -- sim/ is ADDITIONS-ONLY in
    dendritic_neuron.py (the extension is additive/default-off -> step() byte-identical when unused); backend recorded.

GO (per grid, per seed): at s in {0.75,1.0} -- dendritic_bind held-out recall >= additive + 0.30 AND >= 0.5 (recovers
the break), AND >= readout_product_bind - 0.10 (matches the target within tolerance); at s <= 0.5 -- dendritic_bind >=
additive - 0.05 (no low-s cost); all anti-cheats hold; sim additions-only; byte-identical. HONEST NEGATIVE if the
dendritic coincidence does NOT recover (naming WHY: plateau saturation distorts the bilinear factors; or it costs low-s).

DISCIPLINE: reuse-by-import (the conjunctive-binding runner's BindingGenerator + mixed world + reservoir + the
zero-shot split/floors/asserts). Guarded/default-off sim edit only. SIM_BACKEND=numpy.

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_dendritic_bind_derisk --seed 42 \
      --grids 7x7 --s-values 0.0 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_dendritic_bind_s42.json
  MULTI-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_dendritic_bind_derisk --seeds 42 43 44 \
      --grids 7x7 8x8 --s-values 0.0 0.25 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_dendritic_bind.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402

# reuse-by-import: the conjunctive-binding runner (BindingGenerator = the readout-product bind + additive; the mixed
# world; the reservoir; the offsets) + the zero-shot split/floors/asserts + the byte-identical assert. NO NEW sim edit
# beyond the guarded default-off DendriticLayer.apical_basal_coincidence.
from research.runners._teacher_loop_conjunctive_binding_derisk import (  # noqa: E402
    BindingGenerator, _Reservoir, _make_mixed_env, _additive_nonadditivity_witness,
    _A_OFF, _B_OFF, _JA_OFF, _JB_OFF, _MULA_OFF, _MULB_OFF,
)
from research.runners._teacher_loop_generative_replay_derisk import _cos  # noqa: E402
from research.runners._teacher_loop_compositional_generator_derisk import _grid_facts  # noqa: E402
from research.runners._teacher_loop_zeroshot_composition_derisk import (  # noqa: E402
    _heldout_split, _nearest_proto, _recall_fraction,
)
from research.runners._teacher_loop_scaling_derisk import _corrective_batch, N_ACT  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import _assert_byte_identical_substrate  # noqa: E402
from sim.dendritic_neuron import DendriticLayer  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_dendritic_bind.json"


# ============================ the DENDRITIC coincidence bind ============================
class DendriticBind:
    """Wraps a BindingGenerator's spiking factor readouts (fA[a]=_mul_factor_a, fB[b]=_mul_factor_b) and combines them
    with a REAL two-compartment DENDRITIC COINCIDENCE (sim/dendritic_neuron.py) instead of the host `*`. Per channel c a
    dendritic unit receives fA[.,c] at its APICAL synapse and fB[.,c] at its BASAL synapse; the neuron's plateau
    coincidence soma = phi(basal)*phi(apical) IS the conjunction. Signed factors -> biological ON/OFF push-pull."""

    def __init__(self, bg: BindingGenerator, seed):
        self.bg = bg
        self.d_p = bg.d_p
        # ONE dendritic layer, labeled-line (identity) wiring so channel c is a self-contained coincidence unit. The
        # apical feedback is normally FIXED-RANDOM (feedback alignment); here the layer is used purely as a coincidence
        # COMPUTER (no plasticity, no credit assignment), so identity routing is the correct labeled-line wiring.
        self.layer = DendriticLayer(self.d_p, self.d_p, self.d_p, seed=int(seed) + 777)
        self.layer.W_basal = np.eye(self.d_p)
        self.layer.B_apical = np.eye(self.d_p)
        self.z0_b = 1.0
        self.z0_a = 1.0
        self.g_out = 1.0
        self._nl_witness = 0.0     # coincidence-is-nonlinear witness
        self._and_ok = False       # coinc(x,0)=coinc(0,y)=0 (the AND anchor)

    # --- the dendritic coincidence of two signed factor vectors (fa=apical, fb=basal) ---
    def _coincidence(self, fa, fb):
        fa = np.asarray(fa, float); fb = np.asarray(fb, float)
        aP = np.maximum(fa, 0.0); aN = np.maximum(-fa, 0.0)     # ON/OFF apical drive channels (>=0)
        bP = np.maximum(fb, 0.0); bN = np.maximum(-fb, 0.0)     # ON/OFF basal drive channels (>=0)
        # each coinc(basal, apical) = the neuron's plateau product phi(basal)*phi(apical) (per channel; identity wiring)
        cPP = self.layer.apical_basal_coincidence(bP, aP, self.z0_b, self.z0_a)["soma"]   # same sign -> +
        cNN = self.layer.apical_basal_coincidence(bN, aN, self.z0_b, self.z0_a)["soma"]   # same sign -> +
        cPN = self.layer.apical_basal_coincidence(bN, aP, self.z0_b, self.z0_a)["soma"]   # opposite -> -
        cNP = self.layer.apical_basal_coincidence(bP, aN, self.z0_b, self.z0_a)["soma"]   # opposite -> -
        return cPP + cNN - cPN - cNP                                                       # excit + inhib soma

    def calibrate(self, taught_idx, attrs):
        """Homeostatic, taught-only, ruler-free: set the plateau half-saturation to the median compartment drive (so it
        genuinely bends) and the output gain so the dendritic term matches the host-product term's RMS."""
        fa_list = []; fb_list = []; prod_list = []
        for j in taught_idx:
            a, b = attrs[j]
            fa = self.bg._mul_factor_a(a); fb = self.bg._mul_factor_b(b)
            fa_list.append(fa); fb_list.append(fb); prod_list.append(fa * fb)   # host product = the target term's scale
        FA = np.abs(np.stack(fa_list)); FB = np.abs(np.stack(fb_list))
        # ON/OFF split has non-negative drives whose magnitudes equal |f|; half-saturate at their median (>0 -> bends).
        self.z0_a = float(np.median(FA[FA > 1e-9])) if np.any(FA > 1e-9) else 1.0
        self.z0_b = float(np.median(FB[FB > 1e-9])) if np.any(FB > 1e-9) else 1.0
        # output gain: match dendritic-term RMS to host-product-term RMS over taught cells (one scalar)
        dend = np.stack([self._coincidence(fa, fb) for fa, fb in zip(fa_list, fb_list)])
        dend_rms = float(np.sqrt(np.mean(dend ** 2))) + 1e-12
        prod_rms = float(np.sqrt(np.mean(np.stack(prod_list) ** 2)))
        self.g_out = prod_rms / dend_rms
        # anti-cheat witnesses -------------------------------------------------
        # (1) AND anchor: no output when either compartment is silent.
        z = np.zeros(self.d_p)
        and_a = float(np.max(np.abs(self._coincidence(fa_list[0], z))))     # basal silent
        and_b = float(np.max(np.abs(self._coincidence(z, fb_list[0]))))     # apical silent
        self._and_ok = bool(and_a < 1e-12 and and_b < 1e-12)
        # (2) genuinely NONLINEAR: the dendritic soma is NOT the best-scalar-linear multiple of the exact product.
        d_flat = (self.g_out * dend).ravel()
        p_flat = np.stack(prod_list).ravel()
        alpha = float((d_flat @ p_flat) / (p_flat @ p_flat + 1e-12))        # best linear (scalar) fit of the product
        resid = d_flat - alpha * p_flat
        self._nl_witness = float(np.sqrt(np.mean(resid ** 2)) / (np.sqrt(np.mean(d_flat ** 2)) + 1e-12))

    def dendritic(self, a, b):
        """TREATMENT: co-adapted additive part (same as the readout-product bind) + the DENDRITIC coincidence term."""
        add = self.bg.gj + self.bg._jadd_a(a) + self.bg._jadd_b(b)
        return add + self.g_out * self._coincidence(self.bg._mul_factor_a(a), self.bg._mul_factor_b(b))


# ============================ git guard: sim edit is ADDITIONS-ONLY (guarded/default-off) ============================
def _git_sim_additions_only():
    """The sim edit is legitimate ONLY if it is additive/default-off: assert git diff main -- sim/ touches ONLY
    dendritic_neuron.py and contains NO deletions (existing lines -- incl. step() -- untouched => byte-identical when
    the new coincidence method is never called)."""
    try:
        out = subprocess.run(["git", "diff", "main", "--", "sim/"], cwd=str(_REPO),
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return False, "git diff failed", []
        body = out.stdout
        files = [ln.split(" b/")[-1] for ln in body.splitlines() if ln.startswith("diff --git")]
        deletions = [ln for ln in body.splitlines()
                     if ln.startswith("-") and not ln.startswith("---")]
        only_dend = bool(files) and all(f.endswith("sim/dendritic_neuron.py") for f in files)
        additions_only = (len(deletions) == 0)
        return bool(only_dend and additions_only), body[:600], files
    except Exception as e:
        return False, f"exc {e}", []


# ============================ per (grid, s) driver ============================
def _run_grid_s(seed, K1, K2, m, s, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                bdsp_wmax, conv_tol, conv_max_epochs, n_draws, bind_gate):
    N = K1 * K2
    d_p = int(d_a) + int(d_b)
    n_in = d_p + N_ACT
    chance = 1.0 / N
    referents, attrs = _grid_facts(K1, K2)

    taught_idx, held_idx = _heldout_split(K1, K2, m, seed)
    taught_set, held_set = set(taught_idx), set(held_idx)
    disjoint = bool(len(taught_set & held_set) == 0 and len(held_set) > 0)
    trained_a = {attrs[j][0] for j in taught_idx}; trained_b = {attrs[j][1] for j in taught_idx}
    coverage_ok = bool(all(attrs[j][0] in trained_a and attrs[j][1] in trained_b for j in held_idx))
    assert disjoint, "taught and held-out sets must be disjoint and held-out non-empty"
    assert coverage_ok, "every held-out primitive (a AND b) must appear in >= 1 taught combo"

    env = _make_mixed_env(seed, K1, K2, d_a, d_b, noise, s, mix_scale, referents, attrs)
    protos = np.stack([env.proto(referents[j]) for j in range(N)]).astype(np.float64)   # (N,d_p) test-time RULER only
    witness = _additive_nonadditivity_witness(protos, attrs, K1, K2)

    all_addrs = ([_A_OFF + a for a in range(K1)] + [_B_OFF + b for b in range(K2)]
                 + [_JA_OFF + a for a in range(K1)] + [_JB_OFF + b for b in range(K2)]
                 + [_MULA_OFF + a for a in range(K1)] + [_MULB_OFF + b for b in range(K2)])
    res = _Reservoir(gen_k, n_in, gen_hidden, seed, gen_settle, gen_lr, w_clip, bdsp_wmax, all_addrs)

    engrams = {}; fed = []
    for j in taught_idx:
        Xj, _yj = _corrective_batch(env, referents[j], j, n_draws)
        engrams[j] = np.asarray(Xj, dtype=np.float64).mean(axis=0)
        fed.append(j)
    no_leakage = bool(not (set(fed) & held_set))
    assert no_leakage, "a held-out fact index leaked into a training path"

    # readout-product bind + additive (the in-run baseline + floor), on ONE reservoir
    bg = BindingGenerator(res, d_a, d_b, K1, K2, gen_lr, conv_tol, conv_max_epochs, gate=bind_gate)
    bg.fit(taught_idx, attrs, engrams)

    # dendritic bind: SAME spiking factor readouts, combined by the REAL dendritic coincidence
    db = DendriticBind(bg, seed)
    db.calibrate(taught_idx, attrs)

    def add_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.additive(a, b)[:d_p], protos)

    def prod_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.binding(a, b)[:d_p], protos)

    def dend_pred(j):
        a, b = attrs[j]; return _nearest_proto(db.dendritic(a, b)[:d_p], protos)

    out = {
        "K1": K1, "K2": K2, "N": N, "s": s, "mix_scale": mix_scale, "chance": chance,
        "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "taught_heldout_disjoint": disjoint, "every_heldout_primitive_seen_in_taught": coverage_ok,
        "no_leakage_heldout_never_trained": no_leakage,
        "nonadditivity_witness": witness,
        # HEADLINE: held-out (zero-shot) recall -- the three arms
        "dendritic_bind_heldout_recall": _recall_fraction(held_idx, dend_pred, protos),
        "readout_product_bind_heldout_recall": _recall_fraction(held_idx, prod_pred, protos),
        "additive_heldout_recall": _recall_fraction(held_idx, add_pred, protos),
        # taught (seen) recall = sanity
        "dendritic_bind_seen_recall": _recall_fraction(taught_idx, dend_pred, protos),
        "readout_product_bind_seen_recall": _recall_fraction(taught_idx, prod_pred, protos),
        # dendritic anti-cheat witnesses
        "reservoir_mean_spikes": res.mean_spikes(),
        "dend_and_anchor_ok": bool(db._and_ok),                 # coinc(x,0)=coinc(0,y)=0 (real AND)
        "dend_nonlinearity_witness": db._nl_witness,            # soma != linear multiple of the product
        "dend_z0_basal": db.z0_b, "dend_z0_apical": db.z0_a, "dend_g_out": db.g_out,
    }
    out["dendritic_bind_heldout_cos"] = float(np.mean([_cos(db.dendritic(*attrs[j])[:d_p], protos[j])
                                                       for j in held_idx]))
    out["readout_product_bind_heldout_cos"] = float(np.mean([_cos(bg.binding(*attrs[j])[:d_p], protos[j])
                                                             for j in held_idx]))
    return out


# ============================ verdict ============================
def _verdict(result, sim_additions_only, sim_diff_head, sim_files):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    grids = result["per_grid"]
    gkeys = sorted(grids, key=lambda g: grids[g]["N"])
    v = Verdict("teacher-loop DENDRITIC BIND (a REAL apical x basal coincidence recovers the conjunction on spikes)",
                chance=None)

    per_grid_summary = {}
    all_go = True
    for g in gkeys:
        gr = grids[g]
        rows = gr["rows"]
        svals = sorted(rows, key=lambda x: float(x))
        N = gr["N"]
        high_s = [sv for sv in svals if float(sv) >= 0.75]
        low_s = [sv for sv in svals if float(sv) <= 0.5]

        for sv in svals:
            r = rows[sv]
            attributable_to(f"[{g} s={sv}] dendritic bind vs additive superposition (held-out)",
                            r["dendritic_bind_heldout_recall"], r["additive_heldout_recall"])

        recover_ok = True
        match_ok = True
        top_s = svals[-1]
        for sv in high_s:
            r = rows[sv]
            margin_ok = bool(r["dendritic_bind_heldout_recall"] >= r["additive_heldout_recall"] + 0.30)
            abs_ok = bool(r["dendritic_bind_heldout_recall"] >= 0.5) if sv == top_s else True
            mok = bool(r["dendritic_bind_heldout_recall"] >= r["readout_product_bind_heldout_recall"] - 0.10)
            ok = bool(margin_ok and abs_ok)
            recover_ok = recover_ok and ok
            match_ok = match_ok and mok
            v.require(f"[{g} s={sv}] dendritic RECOVERS held-out (>= additive+0.30"
                      + (" AND >= 0.5)" if sv == top_s else ")"), ok, expect=True,
                      note=f"dend {r['dendritic_bind_heldout_recall']:.2f} vs add "
                           f"{r['additive_heldout_recall']:.2f} (chance {r['chance']:.3f})")
            v.require(f"[{g} s={sv}] dendritic MATCHES readout-product (>= product-0.10)", mok, expect=True,
                      note=f"dend {r['dendritic_bind_heldout_recall']:.2f} vs prod "
                           f"{r['readout_product_bind_heldout_recall']:.2f}")
        nocost_ok = True
        for sv in low_s:
            r = rows[sv]
            ok = bool(r["dendritic_bind_heldout_recall"] >= r["additive_heldout_recall"] - 0.05)
            nocost_ok = nocost_ok and ok
            v.require(f"[{g} s={sv}] no low-s cost (dendritic >= additive-0.05)", ok, expect=True,
                      note=f"dend {r['dendritic_bind_heldout_recall']:.2f} vs add {r['additive_heldout_recall']:.2f}")
        seen_min = min(rows[sv]["dendritic_bind_seen_recall"] for sv in svals)
        seen_ok = bool(seen_min >= 0.85)
        v.require(f"[{g}] dendritic taught (seen) recall >= 0.85 (min over s)", seen_ok, expect=True,
                  note=f"min-seen {seen_min:.2f}")
        rt = rows[top_s]
        neural_ok = bool(rt["reservoir_mean_spikes"] > 0.0)
        and_ok = bool(rt["dend_and_anchor_ok"])
        nl_ok = bool(rt["dend_nonlinearity_witness"] > 1e-3)
        witness_ok = bool(rt["nonadditivity_witness"] > 0.02)
        zshot_ok = bool(rt["taught_heldout_disjoint"] and rt["every_heldout_primitive_seen_in_taught"]
                        and rt["no_leakage_heldout_never_trained"])
        v.require(f"[{g}] factors are NEURAL (reservoir spikes > 0)", neural_ok, expect=True,
                  note=f"mean spikes {rt['reservoir_mean_spikes']:.1f}")
        v.require(f"[{g}] bind is a REAL dendritic COINCIDENCE (AND anchor: coinc(x,0)=coinc(0,y)=0)", and_ok,
                  expect=True)
        v.require(f"[{g}] coincidence genuinely NONLINEAR (soma != linear multiple of the product)", nl_ok,
                  expect=True, note=f"nl-witness {rt['dend_nonlinearity_witness']:.3f}")
        v.require(f"[{g}] world carries a real CONJUNCTION at high s (witness > 0.02)", witness_ok, expect=True,
                  note=f"witness {rt['nonadditivity_witness']:.3f}")
        v.require(f"[{g}] genuinely ZERO-SHOT (disjoint + coverage + no-leakage)", zshot_ok, expect=True)

        grid_go = bool(recover_ok and match_ok and nocost_ok and seen_ok and neural_ok and and_ok and nl_ok
                       and witness_ok and zshot_ok)
        all_go = all_go and grid_go
        per_grid_summary[g] = {
            "N": N, "held_out_n": gr["held_out_n"],
            "by_s": {sv: {
                "dendritic_bind_heldout_recall": rows[sv]["dendritic_bind_heldout_recall"],
                "readout_product_bind_heldout_recall": rows[sv]["readout_product_bind_heldout_recall"],
                "additive_heldout_recall": rows[sv]["additive_heldout_recall"],
                "dendritic_bind_seen_recall": rows[sv]["dendritic_bind_seen_recall"],
                "dendritic_minus_additive": float(rows[sv]["dendritic_bind_heldout_recall"]
                                                  - rows[sv]["additive_heldout_recall"]),
                "dendritic_minus_product": float(rows[sv]["dendritic_bind_heldout_recall"]
                                                 - rows[sv]["readout_product_bind_heldout_recall"]),
                "nonadditivity_witness": rows[sv]["nonadditivity_witness"],
                "dend_nonlinearity_witness": rows[sv]["dend_nonlinearity_witness"],
                "chance": rows[sv]["chance"],
            } for sv in svals},
            "recover_ok": recover_ok, "match_product_ok": match_ok, "nocost_ok": nocost_ok, "grid_go": grid_go,
        }

    v.require("(seed) substrate byte-identical", bool(result["substrate_byte_identical"]), expect=True,
              note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) edit is ADDITIONS-ONLY in dendritic_neuron.py (guarded/default-off)", bool(sim_additions_only),
              expect=True, note=f"files={sim_files}")

    go = bool(all_go and result["substrate_byte_identical"] and sim_additions_only)
    decision = v.decide(go=go)
    return {"grids": gkeys, "per_grid": per_grid_summary,
            "substrate_byte_identical": result["substrate_byte_identical"],
            "sim_additions_only": sim_additions_only, "sim_diff_head": sim_diff_head, "sim_files": sim_files,
            **decision}


# ============================ orchestration ============================
def run(seed, grids, held_out, s_values, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
        bdsp_wmax, conv_tol, conv_max_epochs, n_draws, bind_gate):
    n_in = int(d_a) + int(d_b) + N_ACT
    Kbig = max(k1 * k2 for k1, k2 in grids)
    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, Kbig, seed, max(120, 6 * Kbig), 20,
                                                               0.5, w_clip, bdsp_wmax)
    per_grid = {}
    for (K1, K2) in grids:
        m = held_out.get(f"{K1}x{K2}", max(1, min(K1, K2)))
        rows = {}
        print(f"\n{'=' * 96}\n# SEED {seed}  GRID {K1}x{K2} (N={K1*K2}, held_out={m})  s-sweep {s_values}\n{'=' * 96}",
              flush=True)
        for s in s_values:
            r = _run_grid_s(seed, K1, K2, m, s, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr,
                            w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws, bind_gate)
            rows[str(s)] = r
            print(f"  [s={s:.2f}] held-out: dend {r['dendritic_bind_heldout_recall']:.2f} | prod "
                  f"{r['readout_product_bind_heldout_recall']:.2f} | add {r['additive_heldout_recall']:.2f} "
                  f"(chance {r['chance']:.3f}) | seen(dend) {r['dendritic_bind_seen_recall']:.2f} | witness "
                  f"{r['nonadditivity_witness']:.3f} | nl {r['dend_nonlinearity_witness']:.3f} | AND "
                  f"{r['dend_and_anchor_ok']} | spikes {r['reservoir_mean_spikes']:.0f}", flush=True)
        per_grid[f"{K1}x{K2}"] = {"K1": K1, "K2": K2, "N": K1 * K2, "held_out_n": rows[str(s_values[0])]["held_out_n"],
                                  "rows": rows}
    return {"seed": seed, "grids": [f"{k1}x{k2}" for k1, k2 in grids], "s_values": s_values, "mix_scale": mix_scale,
            "d_a": d_a, "d_b": d_b, "n_in": n_in,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "config": {"d_a": d_a, "d_b": d_b, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "n_draws": n_draws,
                       "held_out": held_out, "s_values": s_values, "mix_scale": mix_scale, "frozen_hidden": True},
            "per_grid": per_grid}


def _parse_grid(s):
    a, b = s.lower().split("x"); return (int(a), int(b))


def _one_seed(a, seed, grids, held_out):
    result = run(seed, grids, held_out, a.s_values, a.mix_scale, a.d_a, a.d_b, a.noise, a.gen_hidden, a.gen_k,
                 a.gen_settle, a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.n_draws, a.bind_gate)
    sim_ok, sim_head, sim_files = _git_sim_additions_only()
    return result, _verdict(result, sim_ok, sim_head, sim_files)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop DENDRITIC BIND: a REAL apical x basal dendritic coincidence "
                                             "(sim/dendritic_neuron.py) computes the conjunction in spikes, replacing "
                                             "the host `*`; recover zero-shot composition at high mixing.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--grids", nargs="+", default=["7x7", "8x8"])
    ap.add_argument("--held-out", nargs="+", default=["7x7:7", "8x8:8"])
    ap.add_argument("--s-values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--mix-scale", type=float, default=0.4)
    ap.add_argument("--bind-gate", type=float, default=0.25)
    ap.add_argument("--d-a", type=int, default=10)
    ap.add_argument("--d-b", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--gen-hidden", type=int, default=96)
    ap.add_argument("--gen-k", type=int, default=64)
    ap.add_argument("--gen-settle", type=int, default=15)
    ap.add_argument("--gen-lr", type=float, default=0.8)
    ap.add_argument("--conv-tol", type=float, default=0.02)
    ap.add_argument("--conv-max-epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grids = [_parse_grid(g) for g in a.grids]
    held_out = {}
    for spec in a.held_out:
        k, mm = spec.split(":"); held_out[k.lower()] = int(mm)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  grids={a.grids} held_out={a.held_out} s={a.s_values} "
              f"mix_scale={a.mix_scale}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, grids, held_out)
        summary = {"probe": "teacher_loop_dendritic_bind", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "grids": a.grids, "held_out": a.held_out, "s_values": a.s_values, "mix_scale": a.mix_scale,
                   "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        rv = verdict
        print("\n" + "=" * 100, flush=True)
        for g in rv["grids"]:
            pg = rv["per_grid"][g]
            for sv, row in pg["by_s"].items():
                print(f"[dend] seed {s} {g} s={sv}: N={pg['N']} | HELD-OUT dend "
                      f"{row['dendritic_bind_heldout_recall']:.2f} vs prod "
                      f"{row['readout_product_bind_heldout_recall']:.2f} vs add {row['additive_heldout_recall']:.2f} "
                      f"(d_add {row['dendritic_minus_additive']:+.2f} d_prod {row['dendritic_minus_product']:+.2f}) | "
                      f"witness {row['nonadditivity_witness']:.3f} nl {row['dend_nonlinearity_witness']:.3f}",
                      flush=True)
            print(f"[dend] seed {s} {g}: recover {pg['recover_ok']} match-prod {pg['match_product_ok']} no-cost "
                  f"{pg['nocost_ok']} | GO {pg['grid_go']}", flush=True)
        print(f"[dend] seed {s} byte-id {rv['substrate_byte_identical']} sim-additions-only "
              f"{rv['sim_additions_only']} | VERDICT {rv['status']}", flush=True)
        print(f"[dend] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        agg = {"probe": "teacher_loop_dendritic_bind_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "grids": a.grids, "held_out": a.held_out,
               "s_values": a.s_values, "mix_scale": a.mix_scale, "go_count": go_n, "n_seeds": len(seeds),
               "per_grid_s_means": {}, "per_seed": per_seed}
        for g in per_seed[0]["verdict"]["grids"]:
            agg["per_grid_s_means"][g] = {}
            svs = list(per_seed[0]["verdict"]["per_grid"][g]["by_s"].keys())
            for sv in svs:
                dend = [p["verdict"]["per_grid"][g]["by_s"][sv]["dendritic_bind_heldout_recall"] for p in per_seed]
                prod = [p["verdict"]["per_grid"][g]["by_s"][sv]["readout_product_bind_heldout_recall"] for p in per_seed]
                add = [p["verdict"]["per_grid"][g]["by_s"][sv]["additive_heldout_recall"] for p in per_seed]
                seen = [p["verdict"]["per_grid"][g]["by_s"][sv]["dendritic_bind_seen_recall"] for p in per_seed]
                wit = [p["verdict"]["per_grid"][g]["by_s"][sv]["nonadditivity_witness"] for p in per_seed]
                nl = [p["verdict"]["per_grid"][g]["by_s"][sv]["dend_nonlinearity_witness"] for p in per_seed]
                agg["per_grid_s_means"][g][sv] = {
                    "N": per_seed[0]["verdict"]["per_grid"][g]["N"],
                    "chance": per_seed[0]["verdict"]["per_grid"][g]["by_s"][sv]["chance"],
                    "dendritic_bind_heldout_recall_mean": float(np.nanmean(dend)),
                    "dendritic_bind_heldout_recall_per_seed": [float(x) for x in dend],
                    "readout_product_bind_heldout_recall_mean": float(np.nanmean(prod)),
                    "readout_product_bind_heldout_recall_per_seed": [float(x) for x in prod],
                    "additive_heldout_recall_mean": float(np.nanmean(add)),
                    "dendritic_bind_seen_recall_mean": float(np.nanmean(seen)),
                    "nonadditivity_witness_mean": float(np.nanmean(wit)),
                    "dend_nonlinearity_witness_mean": float(np.nanmean(nl)),
                }
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[dend AGG] GO {go_n}/{len(seeds)}", flush=True)
        for g, svd in agg["per_grid_s_means"].items():
            for sv, mm in svd.items():
                print(f"   {g} s={sv}: N={mm['N']} | HELD-OUT dend {mm['dendritic_bind_heldout_recall_mean']:.2f} vs "
                      f"prod {mm['readout_product_bind_heldout_recall_mean']:.2f} vs add "
                      f"{mm['additive_heldout_recall_mean']:.2f} (chance {mm['chance']:.3f}) | dend/seed "
                      f"{mm['dendritic_bind_heldout_recall_per_seed']}", flush=True)
        print(f"[dend AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
