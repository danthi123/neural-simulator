"""TEACHER-LOOP FAITHFUL SPIKING-DENDRITE BIND (2026-08-09): the conjunction as a REAL TEMPORAL SPIKING COINCIDENCE.

THE RESIDUAL THIS CLOSES. _teacher_loop_dendritic_bind_derisk.py landed a PARTIAL: a more-biological RATE sigma-pi
(sim/dendritic_neuron.py:apical_basal_coincidence -> soma = phi(basal)*phi(apical)) recovered zero-shot composition,
but the 2026-08-09 adversarial verify CAUGHT that it is still ONE HOST MULTIPLY -- no time, no spikes, step()'s
membrane dynamics bypassed. This de-risk replaces that static product with the FAITHFUL mechanism: route the two
primitive-factor drives through a genuinely TEMPORAL SPIKING two-compartment BAC unit
(sim/dendritic_neuron.py:bac_spiking_coincidence) so the coincidence is computed by MEMBRANE DYNAMICS OVER TIME and
read out as SPIKE COUNTS -- basal depolarization must reach the soma WHILE an apical Ca2+ plateau is active. NO host
product anywhere in the conjunction path.

THE MECHANISM (bac_spiking_coincidence; Larkum 2013 BAC firing; Larkum/Zhu/Sakmann 1999 Ca2+ plateau; catalog
G.02 + J.08). Per channel c a dendritic unit: the basal drive (saturating plateau phi -> bounded) leaky-integrates
the SOMA (tau_m); a supra-threshold APICAL drive IGNITES a regenerative Ca2+ plateau (graded, self-sustaining,
decays plateau_tau) that injects a SUSTAINED depolarizing current into the soma across a temporal WINDOW; a somatic
SPIKE (HARD threshold theta + reset + refractory) fires ONLY when basal coincides IN TIME with the plateau. The AND
is the HARD SPIKE THRESHOLD acting on two individually SUB-THRESHOLD inputs -- the conjunction a SOFT (sigmoid) soma
cannot form. Signed factors -> biological ON/OFF push-pull (aP/aN/bP/bN >=0), combined by excit(same-sign) +
inhib(opposite-sign) branches: bind_c = g*(coinc(bP,aP)+coinc(bN,aN)-coinc(bN,aP)-coinc(bP,aN)), each coinc a
per-channel SPIKE COUNT. theta is set HOMEOSTATICALLY (taught-only, ruler-free): between the single-input membrane
peak and the coincident sum, so neither compartment alone crosses (the AND anchor holds BY MEMBRANE, not by phi(0)=0).

ARMS (all on the ONE frozen spiking Izhikevich reservoir; readout-only; de-clamped bdsp_wmax=1e9):
  * spiking_dendritic (TREATMENT, the faithful target): additive part + the SPIKE-COUNT temporal BAC coincidence.
  * rate_dendritic (the PARTIAL, for like-for-like): additive part + the STATIC phi*phi (apical_basal_coincidence).
  * readout_product (host `*` baseline): the conjunctive-binding runner's `binding`.
  * additive (FLOOR): neural superposition (VSA bundling) -- breaks at high mixing s.

FAITHFULNESS WITNESSES (each a real assertion / number in the output -- this is what the PARTIAL could NOT show):
  * SPIKE-BASED: the coincidence is integer SPIKE COUNTS (total spikes > 0); the output is not a continuous product.
  * TEMPORAL: delaying the basal drive PAST the plateau window (basal_onset large) COLLAPSES the coincidence toward
    the AND floor (temporal_collapse = 1 - recall_delayed/recall_overlap >> 0). A static product has NO time -> a
    delay would not change it; a large collapse PROVES the coincidence is genuinely temporal, not phi*phi.
  * AND anchor: coinc(x,0)=coinc(0,x)=0 spikes (no somatic spike unless BOTH compartments engage).
  * NOT a host product: the ONLY multiply in the conjunction path is inside the membrane update (leaky decay); the
    per-channel bind value is a COUNT of hard threshold crossings.
  * genuinely ZERO-SHOT (disjoint taught/held-out; every held-out primitive in >=1 taught combo; NO leakage).
  * cfg.seed byte-identical substrate; de-clamped bdsp_wmax=1e9; git diff main -- sim/ ADDITIONS-ONLY.

GO (per grid, per seed): at s in {0.75,1.0} -- spiking_dendritic held-out recall >= additive + 0.30 AND >= 0.5 at top
s (recovers the break), AND >= readout_product - 0.15 (matches the target within the spiking-quantization tolerance);
at s <= 0.5 -- >= additive - 0.05 (no low-s cost); TEMPORAL witness temporal_collapse >= 0.30 (the coincidence really
is temporal); AND anchor holds; total spikes > 0; sim additions-only; byte-identical. HONEST NEGATIVE if the spiking
quantization is too coarse to carry the bilinear factors (naming WHY) -- first-class either way (it maps what a real
spiking coincidence can/can't carry).

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_dendritic_bind_spiking_derisk --seed 42 \
      --grids 7x7 --s-values 0.0 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_dendritic_bind_spiking_s42.json
  MULTI-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_dendritic_bind_spiking_derisk --seeds 42 43 44 \
      --grids 7x7 8x8 --s-values 0.0 0.25 0.5 0.75 1.0 \
      --out research/findings/raw/teacher_loop_dendritic_bind_spiking.json
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

from research.runners._teacher_loop_conjunctive_binding_derisk import (  # noqa: E402
    BindingGenerator, _Reservoir, _make_mixed_env, _additive_nonadditivity_witness,
    _A_OFF, _B_OFF, _JA_OFF, _JB_OFF, _MULA_OFF, _MULB_OFF,
)
from research.runners._teacher_loop_dendritic_bind_derisk import DendriticBind  # noqa: E402
from research.runners._teacher_loop_generative_replay_derisk import _cos  # noqa: E402
from research.runners._teacher_loop_compositional_generator_derisk import _grid_facts  # noqa: E402
from research.runners._teacher_loop_zeroshot_composition_derisk import (  # noqa: E402
    _heldout_split, _nearest_proto, _recall_fraction,
)
from research.runners._teacher_loop_scaling_derisk import _corrective_batch, N_ACT  # noqa: E402
from research.runners._teacher_loop_cls_two_store_derisk import _assert_byte_identical_substrate  # noqa: E402
from sim.dendritic_neuron import DendriticLayer  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_dendritic_bind_spiking.json"


# ============================ the FAITHFUL SPIKING dendritic coincidence bind ============================
class SpikingDendriticBind:
    """Wraps a BindingGenerator's spiking factor readouts (fA[a], fB[b]) and combines them with a REAL TEMPORAL
    SPIKING two-compartment BAC coincidence (sim/dendritic_neuron.py:bac_spiking_coincidence) -- SPIKE COUNTS over
    time, NOT the static phi*phi. Per channel c, fA[.,c] drives the APICAL compartment, fB[.,c] the BASAL; the
    somatic spike count over a window IS the conjunction. Signed factors -> ON/OFF push-pull. theta homeostatic."""

    def __init__(self, bg: BindingGenerator, seed, T, tau_m, plateau_tau, plateau_thresh, apical_gain,
                 plateau_onset, refractory, and_margin, delay_gap):
        self.bg = bg
        self.d_p = bg.d_p
        self.layer = DendriticLayer(self.d_p, self.d_p, self.d_p, seed=int(seed) + 991)
        self.layer.W_basal = np.eye(self.d_p)
        self.layer.B_apical = np.eye(self.d_p)
        self.T = int(T); self.tau_m = float(tau_m); self.plateau_tau = float(plateau_tau)
        self.plateau_thresh = float(plateau_thresh); self.apical_gain = float(apical_gain)
        self.plateau_onset = int(plateau_onset); self.refractory = int(refractory)
        self.and_margin = float(and_margin); self.delay_gap = int(delay_gap)
        self.z0_b = 1.0; self.z0_a = 1.0; self.theta = 1.0; self.g_out = 1.0
        self._and_ok = False; self._and_max = 0.0
        self._temporal_collapse = 0.0; self._mean_spikes = 0.0

    def _spk(self, x_basal, x_apical, basal_onset=0):
        """one per-channel spike-count coincidence from the FAITHFUL temporal spiking BAC unit."""
        return self.layer.bac_spiking_coincidence(
            x_basal, x_apical, self.theta, z0_basal=self.z0_b, z0_apical=self.z0_a, T=self.T,
            tau_m=self.tau_m, plateau_tau=self.plateau_tau, plateau_thresh=self.plateau_thresh,
            apical_gain=self.apical_gain, basal_onset=basal_onset, plateau_onset=self.plateau_onset,
            refractory=self.refractory)

    def _coincidence(self, fa, fb, basal_onset=0):
        fa = np.asarray(fa, float); fb = np.asarray(fb, float)
        aP = np.maximum(fa, 0.0); aN = np.maximum(-fa, 0.0)     # ON/OFF apical (>=0)
        bP = np.maximum(fb, 0.0); bN = np.maximum(-fb, 0.0)     # ON/OFF basal (>=0)
        cPP = self._spk(bP, aP, basal_onset)                    # same sign -> +
        cNN = self._spk(bN, aN, basal_onset)                    # same sign -> +
        cPN = self._spk(bN, aP, basal_onset)                    # opposite -> -
        cNP = self._spk(bP, aN, basal_onset)                    # opposite -> -
        return cPP + cNN - cPN - cNP

    def calibrate(self, taught_idx, attrs):
        """Homeostatic, taught-only, ruler-free. (1) operating points z0 = median |compartment drive|; (2) theta set
        BETWEEN the single-input membrane peak and the coincident sum (so neither compartment alone spikes -- the AND
        anchor holds by MEMBRANE); (3) g_out matches the spiking-term RMS to the host-product-term RMS."""
        fa_list = []; fb_list = []; prod_list = []
        for j in taught_idx:
            a, b = attrs[j]
            fa = self.bg._mul_factor_a(a); fb = self.bg._mul_factor_b(b)
            fa_list.append(fa); fb_list.append(fb); prod_list.append(fa * fb)
        FA = np.abs(np.stack(fa_list)); FB = np.abs(np.stack(fb_list))
        self.z0_a = float(np.median(FA[FA > 1e-9])) if np.any(FA > 1e-9) else 1.0
        self.z0_b = float(np.median(FB[FB > 1e-9])) if np.any(FB > 1e-9) else 1.0

        # --- set theta homeostatically from membrane peaks (theta itself is used inside bac_spiking_coincidence for
        # the spike test, but for the PEAK measurement we pass a huge theta so no spike/reset masks the true peak) ---
        BIG = 1e9
        single_peaks = []; sum_peaks = []
        for fa, fb in zip(fa_list, fb_list):
            aP = np.maximum(fa, 0.0); aN = np.maximum(-fa, 0.0)
            bP = np.maximum(fb, 0.0); bN = np.maximum(-fb, 0.0)
            z = np.zeros(self.d_p)
            for (bb, aa) in [(bP, aP), (bN, aN), (bP, aN), (bN, aP)]:
                # basal-only and apical-only membrane peaks (the single-input drives)
                pb = self.layer.bac_spiking_coincidence(bb, z, BIG, z0_basal=self.z0_b, z0_apical=self.z0_a,
                                                        T=self.T, tau_m=self.tau_m, plateau_tau=self.plateau_tau,
                                                        plateau_thresh=self.plateau_thresh,
                                                        apical_gain=self.apical_gain, plateau_onset=self.plateau_onset,
                                                        refractory=self.refractory, return_traces=True)["v_peak"]
                pa = self.layer.bac_spiking_coincidence(z, aa, BIG, z0_basal=self.z0_b, z0_apical=self.z0_a,
                                                        T=self.T, tau_m=self.tau_m, plateau_tau=self.plateau_tau,
                                                        plateau_thresh=self.plateau_thresh,
                                                        apical_gain=self.apical_gain, plateau_onset=self.plateau_onset,
                                                        refractory=self.refractory, return_traces=True)["v_peak"]
                psum = self.layer.bac_spiking_coincidence(bb, aa, BIG, z0_basal=self.z0_b, z0_apical=self.z0_a,
                                                        T=self.T, tau_m=self.tau_m, plateau_tau=self.plateau_tau,
                                                        plateau_thresh=self.plateau_thresh,
                                                        apical_gain=self.apical_gain, plateau_onset=self.plateau_onset,
                                                        refractory=self.refractory, return_traces=True)["v_peak"]
                single_peaks.append(np.maximum(pb, pa)); sum_peaks.append(psum)
        single_peaks = np.stack(single_peaks); sum_peaks = np.stack(sum_peaks)
        # theta ABOVE the (near-)max single-input peak (AND: neither alone crosses) but below the typical sum.
        hi_single = float(np.percentile(single_peaks, 97))
        med_sum = float(np.percentile(sum_peaks, 60))
        theta = hi_single * (1.0 + self.and_margin)
        # keep theta strictly below the coincident sum so the conjunction can fire; if the window is too tight,
        # sit just under the sum (the honest source of any AND-leak -> flagged by the anchor witness).
        theta = min(theta, max(med_sum * 0.95, hi_single * (1.0 + 0.5 * self.and_margin)))
        self.theta = float(theta)

        # --- output gain: match spiking-coincidence RMS to host-product RMS over taught cells ---
        dend = np.stack([self._coincidence(fa, fb) for fa, fb in zip(fa_list, fb_list)])
        dend_rms = float(np.sqrt(np.mean(dend ** 2))) + 1e-12
        prod_rms = float(np.sqrt(np.mean(np.stack(prod_list) ** 2)))
        self.g_out = prod_rms / dend_rms
        self._mean_spikes = float(np.mean(np.abs(dend)))

        # --- anti-cheat witnesses ---
        # (1) AND anchor: no somatic spikes when a compartment is silent.
        z = np.zeros(self.d_p)
        and_a = float(np.max(np.abs(self._coincidence(fa_list[0], z))))    # basal silent
        and_b = float(np.max(np.abs(self._coincidence(z, fb_list[0]))))    # apical silent
        self._and_max = max(and_a, and_b)
        self._and_ok = bool(self._and_max < 1e-9)
        # (2) TEMPORAL witness: delaying basal past the plateau collapses the coincidence toward the AND floor.
        delayed_onset = self.plateau_onset + self.delay_gap
        overlap = np.stack([np.abs(self._coincidence(fa, fb, basal_onset=0)) for fa, fb in zip(fa_list, fb_list)])
        delayed = np.stack([np.abs(self._coincidence(fa, fb, basal_onset=delayed_onset))
                            for fa, fb in zip(fa_list, fb_list)])
        ov = float(np.sum(overlap)); dl = float(np.sum(delayed))
        self._temporal_collapse = float(1.0 - dl / ov) if ov > 1e-12 else 0.0

    def spiking_dendritic(self, a, b):
        """TREATMENT: co-adapted additive part (same as the readout-product bind) + the SPIKING BAC coincidence term."""
        add = self.bg.gj + self.bg._jadd_a(a) + self.bg._jadd_b(b)
        return add + self.g_out * self._coincidence(self.bg._mul_factor_a(a), self.bg._mul_factor_b(b))


# ============================ git guard: sim edit is ADDITIONS-ONLY ============================
def _git_sim_additions_only():
    try:
        out = subprocess.run(["git", "diff", "main", "--", "sim/"], cwd=str(_REPO),
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return False, "git diff failed", []
        body = out.stdout
        files = [ln.split(" b/")[-1] for ln in body.splitlines() if ln.startswith("diff --git")]
        deletions = [ln for ln in body.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        only_dend = bool(files) and all(f.endswith("sim/dendritic_neuron.py") for f in files)
        additions_only = (len(deletions) == 0)
        return bool(only_dend and additions_only), body[:600], files
    except Exception as e:
        return False, f"exc {e}", []


# ============================ per (grid, s) driver ============================
def _run_grid_s(seed, K1, K2, m, s, mix_scale, d_a, d_b, noise, gen_hidden, gen_k, gen_settle, gen_lr, w_clip,
                bdsp_wmax, conv_tol, conv_max_epochs, n_draws, bind_gate, spk_cfg):
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
    protos = np.stack([env.proto(referents[j]) for j in range(N)]).astype(np.float64)
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

    bg = BindingGenerator(res, d_a, d_b, K1, K2, gen_lr, conv_tol, conv_max_epochs, gate=bind_gate)
    bg.fit(taught_idx, attrs, engrams)

    # rate dendritic (the PARTIAL) + spiking dendritic (the faithful treatment), SAME factor readouts
    rd = DendriticBind(bg, seed); rd.calibrate(taught_idx, attrs)
    sd = SpikingDendriticBind(bg, seed, **spk_cfg); sd.calibrate(taught_idx, attrs)

    def add_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.additive(a, b)[:d_p], protos)

    def prod_pred(j):
        a, b = attrs[j]; return _nearest_proto(bg.binding(a, b)[:d_p], protos)

    def rate_pred(j):
        a, b = attrs[j]; return _nearest_proto(rd.dendritic(a, b)[:d_p], protos)

    def spk_pred(j):
        a, b = attrs[j]; return _nearest_proto(sd.spiking_dendritic(a, b)[:d_p], protos)

    out = {
        "K1": K1, "K2": K2, "N": N, "s": s, "mix_scale": mix_scale, "chance": chance,
        "held_out_n": len(held_idx), "taught_n": len(taught_idx),
        "taught_heldout_disjoint": disjoint, "every_heldout_primitive_seen_in_taught": coverage_ok,
        "no_leakage_heldout_never_trained": no_leakage, "nonadditivity_witness": witness,
        # HEADLINE: held-out (zero-shot) recall -- the four arms
        "spiking_dendritic_heldout_recall": _recall_fraction(held_idx, spk_pred, protos),
        "rate_dendritic_heldout_recall": _recall_fraction(held_idx, rate_pred, protos),
        "readout_product_heldout_recall": _recall_fraction(held_idx, prod_pred, protos),
        "additive_heldout_recall": _recall_fraction(held_idx, add_pred, protos),
        "spiking_dendritic_seen_recall": _recall_fraction(taught_idx, spk_pred, protos),
        # faithfulness witnesses (what the PARTIAL could not show)
        "reservoir_mean_spikes": res.mean_spikes(),
        "spk_and_anchor_ok": bool(sd._and_ok), "spk_and_max": sd._and_max,
        "spk_temporal_collapse": sd._temporal_collapse,     # delayed basal -> AND floor (proves TEMPORAL)
        "spk_mean_coincidence_spikes": sd._mean_spikes,     # integer spike counts (proves SPIKE-BASED)
        "spk_theta": sd.theta, "spk_z0_basal": sd.z0_b, "spk_z0_apical": sd.z0_a, "spk_g_out": sd.g_out,
    }
    out["spiking_dendritic_heldout_cos"] = float(np.mean([_cos(sd.spiking_dendritic(*attrs[j])[:d_p], protos[j])
                                                          for j in held_idx]))
    return out


# ============================ verdict ============================
def _verdict(result, sim_additions_only, sim_diff_head, sim_files, prod_tol, collapse_min):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    grids = result["per_grid"]
    gkeys = sorted(grids, key=lambda g: grids[g]["N"])
    v = Verdict("teacher-loop FAITHFUL SPIKING-DENDRITE BIND (a TEMPORAL spiking BAC coincidence recovers the "
                "conjunction as spike counts)", chance=None)

    per_grid_summary = {}
    all_go = True
    for g in gkeys:
        gr = grids[g]; rows = gr["rows"]
        svals = sorted(rows, key=lambda x: float(x)); N = gr["N"]
        high_s = [sv for sv in svals if float(sv) >= 0.75]
        low_s = [sv for sv in svals if float(sv) <= 0.5]
        top_s = svals[-1]

        for sv in svals:
            r = rows[sv]
            attributable_to(f"[{g} s={sv}] spiking dendritic bind vs additive superposition (held-out)",
                            r["spiking_dendritic_heldout_recall"], r["additive_heldout_recall"])

        recover_ok = True; match_ok = True
        for sv in high_s:
            r = rows[sv]
            margin_ok = bool(r["spiking_dendritic_heldout_recall"] >= r["additive_heldout_recall"] + 0.30)
            abs_ok = bool(r["spiking_dendritic_heldout_recall"] >= 0.5) if sv == top_s else True
            mok = bool(r["spiking_dendritic_heldout_recall"] >= r["readout_product_heldout_recall"] - prod_tol)
            ok = bool(margin_ok and abs_ok)
            recover_ok = recover_ok and ok; match_ok = match_ok and mok
            v.require(f"[{g} s={sv}] spiking RECOVERS held-out (>= additive+0.30"
                      + (" AND >= 0.5)" if sv == top_s else ")"), ok, expect=True,
                      note=f"spk {r['spiking_dendritic_heldout_recall']:.2f} vs add "
                           f"{r['additive_heldout_recall']:.2f} (chance {r['chance']:.3f})")
            v.require(f"[{g} s={sv}] spiking MATCHES readout-product (>= product-{prod_tol:.2f})", mok, expect=True,
                      note=f"spk {r['spiking_dendritic_heldout_recall']:.2f} vs prod "
                           f"{r['readout_product_heldout_recall']:.2f}")
        nocost_ok = True
        for sv in low_s:
            r = rows[sv]
            ok = bool(r["spiking_dendritic_heldout_recall"] >= r["additive_heldout_recall"] - 0.05)
            nocost_ok = nocost_ok and ok
            v.require(f"[{g} s={sv}] no low-s cost (spiking >= additive-0.05)", ok, expect=True,
                      note=f"spk {r['spiking_dendritic_heldout_recall']:.2f} vs add {r['additive_heldout_recall']:.2f}")
        seen_min = min(rows[sv]["spiking_dendritic_seen_recall"] for sv in svals)
        seen_ok = bool(seen_min >= 0.85)
        v.require(f"[{g}] spiking taught (seen) recall >= 0.85 (min over s)", seen_ok, expect=True,
                  note=f"min-seen {seen_min:.2f}")
        rt = rows[top_s]
        neural_ok = bool(rt["reservoir_mean_spikes"] > 0.0)
        spk_ok = bool(rt["spk_mean_coincidence_spikes"] > 0.0)       # the coincidence is SPIKE COUNTS
        and_ok = bool(rt["spk_and_anchor_ok"])
        temporal_ok = bool(rt["spk_temporal_collapse"] >= collapse_min)  # delayed basal -> AND floor
        witness_ok = bool(rt["nonadditivity_witness"] > 0.02)
        zshot_ok = bool(rt["taught_heldout_disjoint"] and rt["every_heldout_primitive_seen_in_taught"]
                        and rt["no_leakage_heldout_never_trained"])
        v.require(f"[{g}] factors are NEURAL (reservoir spikes > 0)", neural_ok, expect=True,
                  note=f"mean spikes {rt['reservoir_mean_spikes']:.1f}")
        v.require(f"[{g}] coincidence is SPIKE-BASED (mean coincidence spikes > 0)", spk_ok, expect=True,
                  note=f"mean coinc spikes {rt['spk_mean_coincidence_spikes']:.2f}")
        v.require(f"[{g}] bind is a REAL BAC coincidence (AND anchor: coinc(x,0)=coinc(0,x)=0 spikes)", and_ok,
                  expect=True, note=f"and-max {rt['spk_and_max']:.3f}")
        v.require(f"[{g}] coincidence is TEMPORAL (delayed basal collapses to AND floor, collapse >= {collapse_min})",
                  temporal_ok, expect=True, note=f"temporal-collapse {rt['spk_temporal_collapse']:.3f}")
        v.require(f"[{g}] world carries a real CONJUNCTION at high s (witness > 0.02)", witness_ok, expect=True,
                  note=f"witness {rt['nonadditivity_witness']:.3f}")
        v.require(f"[{g}] genuinely ZERO-SHOT (disjoint + coverage + no-leakage)", zshot_ok, expect=True)

        grid_go = bool(recover_ok and match_ok and nocost_ok and seen_ok and neural_ok and spk_ok and and_ok
                       and temporal_ok and witness_ok and zshot_ok)
        all_go = all_go and grid_go
        per_grid_summary[g] = {
            "N": N, "held_out_n": gr["held_out_n"],
            "by_s": {sv: {
                "spiking_dendritic_heldout_recall": rows[sv]["spiking_dendritic_heldout_recall"],
                "rate_dendritic_heldout_recall": rows[sv]["rate_dendritic_heldout_recall"],
                "readout_product_heldout_recall": rows[sv]["readout_product_heldout_recall"],
                "additive_heldout_recall": rows[sv]["additive_heldout_recall"],
                "spiking_dendritic_seen_recall": rows[sv]["spiking_dendritic_seen_recall"],
                "spiking_minus_additive": float(rows[sv]["spiking_dendritic_heldout_recall"]
                                                - rows[sv]["additive_heldout_recall"]),
                "spiking_minus_product": float(rows[sv]["spiking_dendritic_heldout_recall"]
                                               - rows[sv]["readout_product_heldout_recall"]),
                "spk_temporal_collapse": rows[sv]["spk_temporal_collapse"],
                "spk_and_max": rows[sv]["spk_and_max"],
                "spk_mean_coincidence_spikes": rows[sv]["spk_mean_coincidence_spikes"],
                "spk_theta": rows[sv]["spk_theta"],
                "nonadditivity_witness": rows[sv]["nonadditivity_witness"], "chance": rows[sv]["chance"],
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
        bdsp_wmax, conv_tol, conv_max_epochs, n_draws, bind_gate, spk_cfg):
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
                            w_clip, bdsp_wmax, conv_tol, conv_max_epochs, n_draws, bind_gate, spk_cfg)
            rows[str(s)] = r
            print(f"  [s={s:.2f}] held-out: spk {r['spiking_dendritic_heldout_recall']:.2f} | rate "
                  f"{r['rate_dendritic_heldout_recall']:.2f} | prod {r['readout_product_heldout_recall']:.2f} | add "
                  f"{r['additive_heldout_recall']:.2f} (chance {r['chance']:.3f}) | seen(spk) "
                  f"{r['spiking_dendritic_seen_recall']:.2f} | collapse {r['spk_temporal_collapse']:.2f} | AND "
                  f"{r['spk_and_anchor_ok']}({r['spk_and_max']:.2f}) | coincSpk {r['spk_mean_coincidence_spikes']:.2f} "
                  f"| theta {r['spk_theta']:.2f}", flush=True)
        per_grid[f"{K1}x{K2}"] = {"K1": K1, "K2": K2, "N": K1 * K2, "held_out_n": rows[str(s_values[0])]["held_out_n"],
                                  "rows": rows}
    return {"seed": seed, "grids": [f"{k1}x{k2}" for k1, k2 in grids], "s_values": s_values, "mix_scale": mix_scale,
            "d_a": d_a, "d_b": d_b, "n_in": n_in,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "config": {"d_a": d_a, "d_b": d_b, "noise": noise, "gen_hidden": gen_hidden, "gen_k": gen_k,
                       "gen_settle": gen_settle, "gen_lr": gen_lr, "w_clip": w_clip, "bdsp_wmax": bdsp_wmax,
                       "conv_tol": conv_tol, "conv_max_epochs": conv_max_epochs, "n_draws": n_draws,
                       "held_out": held_out, "s_values": s_values, "mix_scale": mix_scale, "frozen_hidden": True,
                       "spk_cfg": spk_cfg},
            "per_grid": per_grid}


def _parse_grid(s):
    a, b = s.lower().split("x"); return (int(a), int(b))


def _one_seed(a, seed, grids, held_out, spk_cfg):
    result = run(seed, grids, held_out, a.s_values, a.mix_scale, a.d_a, a.d_b, a.noise, a.gen_hidden, a.gen_k,
                 a.gen_settle, a.gen_lr, a.w_clip, a.bdsp_wmax, a.conv_tol, a.conv_max_epochs, a.n_draws, a.bind_gate,
                 spk_cfg)
    sim_ok, sim_head, sim_files = _git_sim_additions_only()
    return result, _verdict(result, sim_ok, sim_head, sim_files, a.prod_tol, a.collapse_min)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop FAITHFUL SPIKING-DENDRITE BIND: a temporal spiking BAC "
                                             "coincidence (sim/dendritic_neuron.py:bac_spiking_coincidence) computes "
                                             "the conjunction as SPIKE COUNTS, replacing the PARTIAL's static phi*phi.")
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
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9)
    ap.add_argument("--n-draws", type=int, default=16)
    # spiking BAC unit
    ap.add_argument("--spk-T", type=int, default=40)
    ap.add_argument("--spk-tau-m", type=float, default=3.0)
    ap.add_argument("--spk-plateau-tau", type=float, default=18.0)
    ap.add_argument("--spk-plateau-thresh", type=float, default=0.3)
    ap.add_argument("--spk-apical-gain", type=float, default=0.6)
    ap.add_argument("--spk-plateau-onset", type=int, default=6)
    ap.add_argument("--spk-refractory", type=int, default=2)
    ap.add_argument("--spk-and-margin", type=float, default=0.15)
    ap.add_argument("--spk-delay-gap", type=int, default=20, help="basal_onset delay past plateau (temporal witness)")
    ap.add_argument("--prod-tol", type=float, default=0.15, help="spiking may trail product by this (quantization)")
    ap.add_argument("--collapse-min", type=float, default=0.30, help="min temporal collapse for the TEMPORAL witness")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    spk_cfg = {"T": a.spk_T, "tau_m": a.spk_tau_m, "plateau_tau": a.spk_plateau_tau,
               "plateau_thresh": a.spk_plateau_thresh, "apical_gain": a.spk_apical_gain,
               "plateau_onset": a.spk_plateau_onset, "refractory": a.spk_refractory,
               "and_margin": a.spk_and_margin, "delay_gap": a.spk_delay_gap}
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
              f"mix_scale={a.mix_scale} spk_cfg={spk_cfg}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, grids, held_out, spk_cfg)
        summary = {"probe": "teacher_loop_dendritic_bind_spiking", "seed": s,
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
                print(f"[spk] seed {s} {g} s={sv}: N={pg['N']} | HELD-OUT spk "
                      f"{row['spiking_dendritic_heldout_recall']:.2f} vs rate "
                      f"{row['rate_dendritic_heldout_recall']:.2f} vs prod "
                      f"{row['readout_product_heldout_recall']:.2f} vs add {row['additive_heldout_recall']:.2f} "
                      f"(d_add {row['spiking_minus_additive']:+.2f} d_prod {row['spiking_minus_product']:+.2f}) | "
                      f"collapse {row['spk_temporal_collapse']:.2f} coincSpk {row['spk_mean_coincidence_spikes']:.2f}",
                      flush=True)
            print(f"[spk] seed {s} {g}: recover {pg['recover_ok']} match-prod {pg['match_product_ok']} no-cost "
                  f"{pg['nocost_ok']} | GO {pg['grid_go']}", flush=True)
        print(f"[spk] seed {s} byte-id {rv['substrate_byte_identical']} sim-additions-only "
              f"{rv['sim_additions_only']} | VERDICT {rv['status']}", flush=True)
        print(f"[spk] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        agg = {"probe": "teacher_loop_dendritic_bind_spiking_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "grids": a.grids, "held_out": a.held_out,
               "s_values": a.s_values, "mix_scale": a.mix_scale, "go_count": go_n, "n_seeds": len(seeds),
               "per_grid_s_means": {}, "per_seed": per_seed}
        for g in per_seed[0]["verdict"]["grids"]:
            agg["per_grid_s_means"][g] = {}
            svs = list(per_seed[0]["verdict"]["per_grid"][g]["by_s"].keys())
            for sv in svs:
                spk = [p["verdict"]["per_grid"][g]["by_s"][sv]["spiking_dendritic_heldout_recall"] for p in per_seed]
                rate = [p["verdict"]["per_grid"][g]["by_s"][sv]["rate_dendritic_heldout_recall"] for p in per_seed]
                prod = [p["verdict"]["per_grid"][g]["by_s"][sv]["readout_product_heldout_recall"] for p in per_seed]
                add = [p["verdict"]["per_grid"][g]["by_s"][sv]["additive_heldout_recall"] for p in per_seed]
                seen = [p["verdict"]["per_grid"][g]["by_s"][sv]["spiking_dendritic_seen_recall"] for p in per_seed]
                col = [p["verdict"]["per_grid"][g]["by_s"][sv]["spk_temporal_collapse"] for p in per_seed]
                cspk = [p["verdict"]["per_grid"][g]["by_s"][sv]["spk_mean_coincidence_spikes"] for p in per_seed]
                agg["per_grid_s_means"][g][sv] = {
                    "N": per_seed[0]["verdict"]["per_grid"][g]["N"],
                    "chance": per_seed[0]["verdict"]["per_grid"][g]["by_s"][sv]["chance"],
                    "spiking_dendritic_heldout_recall_mean": float(np.nanmean(spk)),
                    "spiking_dendritic_heldout_recall_per_seed": [float(x) for x in spk],
                    "rate_dendritic_heldout_recall_mean": float(np.nanmean(rate)),
                    "readout_product_heldout_recall_mean": float(np.nanmean(prod)),
                    "readout_product_heldout_recall_per_seed": [float(x) for x in prod],
                    "additive_heldout_recall_mean": float(np.nanmean(add)),
                    "spiking_dendritic_seen_recall_mean": float(np.nanmean(seen)),
                    "spk_temporal_collapse_mean": float(np.nanmean(col)),
                    "spk_mean_coincidence_spikes_mean": float(np.nanmean(cspk)),
                }
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[spk AGG] GO {go_n}/{len(seeds)}", flush=True)
        for g, svd in agg["per_grid_s_means"].items():
            for sv, mm in svd.items():
                print(f"   {g} s={sv}: N={mm['N']} | HELD-OUT spk {mm['spiking_dendritic_heldout_recall_mean']:.2f} "
                      f"vs rate {mm['rate_dendritic_heldout_recall_mean']:.2f} vs prod "
                      f"{mm['readout_product_heldout_recall_mean']:.2f} vs add {mm['additive_heldout_recall_mean']:.2f} "
                      f"(chance {mm['chance']:.3f}) collapse {mm['spk_temporal_collapse_mean']:.2f} | spk/seed "
                      f"{mm['spiking_dendritic_heldout_recall_per_seed']}", flush=True)
        print(f"[spk AGG] wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
