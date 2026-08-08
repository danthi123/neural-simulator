"""SEAM verification probe for the Stage-A FULL single-bridge live integration -- proves each APPENDED-LAST
co-resident seam (A: forward-model reservoir; C: graded-affect ladder) is DEFAULT-OFF byte-identical, engages when
ON, and is MOAT/FM4-safe by construction. Reuse-by-import of `_stageA_full_integration_derisk.build_one_brain`
(+ the seam helpers); NO `sim/` edit.

Per-seam checks (the task's byte-identity oracle, on numpy/CPU for the LITERAL byte claim):
  (1) BYTE-IDENTITY: build the merged bridge WITH the seam flag OFF vs ON. The pre-existing neurons' firing
      thresholds + the frozen conversational cp_connections weights + the composer cp_rf_w_* magnitudes are
      bit-identical (the seam slice appends LAST with zero out-edges -> index bases unchanged, the Probe-1 mechanism).
  (3) AT-REST: flag ON but no drive -> the seam is silent -> g_eff == g0 floor / affect differential ~0 -> no
      content injected / neutral tone (identical to OFF).
  (2/engage) the seam ENGAGES: A -> fm_reservoir active + content decodes + certainty band tightens g_eff
      (tightening-only, always >= g0). C -> graded ladder monotone staircase read NEURALLY (in the faculty smoke).
  (4) MASKED-RESET ISOLATION (seam A): after an fm read+wash, co-resident nav/conv v/u are byte-untouched.

Run:
  SIM_BACKEND=numpy python -m research.runners._stageA_seam_integration_probe --seam A --seed 42
  SIM_BACKEND=numpy python -m research.runners._stageA_seam_integration_probe --seam C --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners import _stageA_full_integration_derisk as S  # noqa: E402


def _sha(arr):
    return hashlib.sha256(np.asarray(to_host(arr), dtype=np.float64).tobytes()).hexdigest()


def _conn_block_sha(bridge, n):
    """sha of the frozen conversational connectivity restricted to the pre-existing [:n, :n] block of the
    cp_connections CSR (the seam's own synapses live at indices >= n, so they are excluded). Byte-identity of this
    block proves the appended-LAST seam did not perturb any pre-existing synapse. Returns None if absent."""
    import scipy.sparse as sp
    w = getattr(bridge, "cp_connections", None)
    if w is None:
        return None
    if sp.issparse(w):
        sub = w.tocsr()[:n, :n].tocsr()
        sub.sort_indices()
        blob = (np.asarray(sub.indptr, np.int64).tobytes() + np.asarray(sub.indices, np.int64).tobytes()
                + np.asarray(sub.data, np.float64).tobytes())
        return hashlib.sha256(blob).hexdigest()
    wh = np.asarray(to_host(w), dtype=np.float64)
    return hashlib.sha256(wh[:n, :n].tobytes() if wh.ndim == 2 else wh[:n].tobytes()).hexdigest()


def _rf_w_sha(bridge):
    """sha of the composer's complex RF phasor weights (the no-confab MOAT store; array-disjoint from cp_connections;
    must be byte-identical). Returns the sha over whatever rf-weight/store arrays are present."""
    parts = []
    for name in ("cp_rf_w_re", "cp_rf_w_im", "cp_rf_w_dense", "cp_rf_store_re", "cp_rf_store_im",
                 "cp_rf_store_dense"):
        a = getattr(bridge, name, None)
        if a is not None:
            try:
                parts.append(name + ":" + _sha(a))
            except Exception:
                pass
    return "|".join(parts) if parts else None


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SEAM A -- forward-model reservoir
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _byte_identity_a(seed):
    bo, _c, _i, _s = S.build_one_brain(seed, with_faculties=True, co_resident_forward_model=False)
    n_off = int(bo.core_config.num_neurons)
    off_thr = _sha(bo.cp_neuron_firing_thresholds[:n_off])
    off_w = _conn_block_sha(bo, n_off)
    off_rf = _rf_w_sha(bo)
    bn, _c2, _i2, _s2 = S.build_one_brain(seed, with_faculties=True, co_resident_forward_model=True)
    n_on = int(bn.core_config.num_neurons)
    on_thr = _sha(bn.cp_neuron_firing_thresholds[:n_off])
    on_w = _conn_block_sha(bn, n_off)
    on_rf = _rf_w_sha(bn)
    return {
        "n_off": n_off, "n_on": n_on, "fm_slice_size": n_on - n_off,
        "appended_last": bool(n_on == n_off + S.FM_N_POOL),
        "threshold_prefix_identical": bool(off_thr == on_thr),
        "conn_weight_prefix_identical": bool(off_w == on_w) if off_w is not None else None,
        "rf_w_identical": bool(off_rf == on_rf) if off_rf is not None else None,
        "byte_identical": bool(n_on == n_off + S.FM_N_POOL and off_thr == on_thr
                               and (off_w is None or off_w == on_w) and (off_rf is None or off_rf == on_rf)),
    }


def _at_rest_a(seed, in_dim=16):
    xp, _ = get_backend()
    b, comp, idx, snap = S.build_one_brain(seed, with_faculties=True, co_resident_forward_model=True)
    W_in = S.make_fm_projection(seed, S.FM_N_POOL, in_dim)
    U = [np.zeros(in_dim) for _ in range(4)]
    counts = S.read_forward_model(b, xp, idx, snap, W_in, U, silence=True)
    silent = float(np.max(counts))
    g0 = S.FM_G0
    g_eff_rest = S.fm_tighten_g_eff(g0, None)          # silent reservoir -> margin None -> untouched
    return {"silent_max_spikecount": silent, "reservoir_silent": bool(silent < 1e-6),
            "g_eff_at_rest": float(g_eff_rest), "g_eff_equals_floor": bool(abs(g_eff_rest - g0) < 1e-12),
            "at_rest_neutral": bool(silent < 1e-6 and abs(g_eff_rest - g0) < 1e-12)}


def _engage_a(seed, in_dim=16, n_classes=4, seq_len=5, n_train=12, n_test=6):
    """The fm_reservoir carries decodable content on the SHARED bridge: distinct (s,a) inputs -> distinguishable
    spike-count features -> a ridge read-out (the DECLARED host shortcut) decodes class + a top1-top2 margin. Then
    the certainty band tightens g_eff (tightening-only). The reservoir SPIKES are the brain-based content."""
    xp, _ = get_backend()
    b, comp, idx, snap = S.build_one_brain(seed, with_faculties=True, co_resident_forward_model=True)
    W_in = S.make_fm_projection(seed, S.FM_N_POOL, in_dim)
    rng = np.random.default_rng(seed * 131 + 7)
    protos = [rng.normal(0, 1, in_dim) for _ in range(n_classes)]

    def _seq(c, noise):
        base = protos[c]
        return [base + noise * rng.normal(0, 1, in_dim) for _ in range(seq_len)]

    X, Y = [], []
    for _ in range(n_train):
        c = int(rng.integers(0, n_classes))
        sc = S.read_forward_model(b, xp, idx, snap, W_in, _seq(c, 0.25))
        X.append(np.concatenate([sc, [1.0]])); Y.append(c)
    X = np.asarray(X); Y = np.asarray(Y)
    Yoh = np.eye(n_classes)[Y]
    lam = 1.0
    Ws = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Yoh)
    # train acc + mean activity
    mean_spikes = float(np.mean([x[:-1].mean() for x in X]))
    train_hit = 0
    for x, y in zip(X, Y):
        if int(np.argmax(x @ Ws)) == int(y):
            train_hit += 1
    train_acc = train_hit / len(Y)
    # test decode + margins + g_eff tightening (moat-safe)
    margins, g_effs, decoded_ok = [], [], 0
    for _ in range(n_test):
        c = int(rng.integers(0, n_classes))
        sc = S.read_forward_model(b, xp, idx, snap, W_in, _seq(c, 0.25))
        pred, margin, _logits = S.fm_decode(sc, Ws)
        margins.append(margin)
        g_effs.append(S.fm_tighten_g_eff(S.FM_G0, margin))
        decoded_ok += int(pred == c)
    test_acc = decoded_ok / n_test
    # tightening-only demonstration: a LOW margin (0.0) tightens strongly; a HIGH margin (1.0) -> g0 floor; both >= g0
    g_low = S.fm_tighten_g_eff(S.FM_G0, 0.0)
    g_high = S.fm_tighten_g_eff(S.FM_G0, 1.0)
    return {
        "reservoir_mean_spikecount": mean_spikes, "reservoir_active": bool(mean_spikes > 0.0),
        "ridge_train_acc": float(train_acc), "ridge_test_acc": float(test_acc),
        "chance": 1.0 / n_classes, "decodes_above_chance": bool(test_acc >= 0.5),
        "mean_test_margin": float(np.mean(margins)),
        "g_eff_low_margin": float(g_low), "g_eff_high_margin": float(g_high), "g0_floor": S.FM_G0,
        "tightening_only": bool(g_low >= S.FM_G0 - 1e-12 and g_high >= S.FM_G0 - 1e-12
                                and g_low >= g_high - 1e-12 and abs(g_high - S.FM_G0) < 1e-9),
        "engages": bool(mean_spikes > 0.0 and test_acc >= 0.5),
    }


def _masked_reset_isolation_a(seed, in_dim=16):
    """After an fm read+wash, the co-resident nav/conv v/u are byte-untouched (the wash restores the baseline
    snapshot; non-fm indices return byte-identical)."""
    xp, _ = get_backend()
    b, comp, idx, snap = S.build_one_brain(seed, with_faculties=True, co_resident_forward_model=True)
    n = int(b.core_config.num_neurons)
    fm = set(int(i) for i in idx["fm"])
    non_fm = np.asarray([i for i in range(n) if i not in fm], dtype=np.int64)
    v_before = np.asarray(to_host(b.cp_membrane_potential_v))[non_fm].copy()
    u_before = np.asarray(to_host(b.cp_recovery_variable_u))[non_fm].copy()
    W_in = S.make_fm_projection(seed, S.FM_N_POOL, in_dim)
    rng = np.random.default_rng(seed * 17 + 1)
    _ = S.read_forward_model(b, xp, idx, snap, W_in, [rng.normal(0, 1, in_dim) for _ in range(5)])
    v_after = np.asarray(to_host(b.cp_membrane_potential_v))[non_fm]
    u_after = np.asarray(to_host(b.cp_recovery_variable_u))[non_fm]
    return {"n_non_fm": int(len(non_fm)),
            "v_byte_untouched": bool(np.array_equal(v_before, v_after)),
            "u_byte_untouched": bool(np.array_equal(u_before, u_after)),
            "navconv_isolated": bool(np.array_equal(v_before, v_after) and np.array_equal(u_before, u_after))}


def verify_seam_a(seed):
    print(f"[seam-A] byte-identity ...", flush=True)
    bid = _byte_identity_a(seed)
    print(f"   byte_identical={bid['byte_identical']} (n_off={bid['n_off']} -> n_on={bid['n_on']}, "
          f"fm_slice={bid['fm_slice_size']}; thr={bid['threshold_prefix_identical']} "
          f"conn={bid['conn_weight_prefix_identical']} rf_w={bid['rf_w_identical']})", flush=True)
    print(f"[seam-A] at-rest neutrality ...", flush=True)
    rest = _at_rest_a(seed)
    print(f"   reservoir_silent={rest['reservoir_silent']} g_eff@rest={rest['g_eff_at_rest']:.4f} "
          f"(floor {S.FM_G0}) at_rest_neutral={rest['at_rest_neutral']}", flush=True)
    print(f"[seam-A] engagement (fm active + content decode + certainty-band tighten) ...", flush=True)
    eng = _engage_a(seed)
    print(f"   reservoir_active={eng['reservoir_active']} (mean_spk {eng['reservoir_mean_spikecount']:.3f}) "
          f"decode test={eng['ridge_test_acc']:.2f} (chance {eng['chance']:.2f}) tightening_only={eng['tightening_only']} "
          f"g_eff(low={eng['g_eff_low_margin']:.3f} high={eng['g_eff_high_margin']:.3f}) engages={eng['engages']}",
          flush=True)
    print(f"[seam-A] masked-reset isolation (nav/conv v/u byte-untouched) ...", flush=True)
    iso = _masked_reset_isolation_a(seed)
    print(f"   navconv_isolated={iso['navconv_isolated']} (v={iso['v_byte_untouched']} u={iso['u_byte_untouched']})",
          flush=True)
    ok = bool(bid["byte_identical"] and rest["at_rest_neutral"] and eng["engages"]
              and eng["tightening_only"] and iso["navconv_isolated"])
    return {"seam": "A", "seed": int(seed), "byte_identity": bid, "at_rest": rest, "engagement": eng,
            "masked_reset_isolation": iso, "seam_ok": ok}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SEAM C -- graded-affect ladder (filled in when seam C is wired)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _byte_identity_c(seed):
    bo, _c, _i, _s = S.build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=False)
    n_off = int(bo.core_config.num_neurons)
    off_thr = _sha(bo.cp_neuron_firing_thresholds[:n_off])
    off_w = _conn_block_sha(bo, n_off)
    off_rf = _rf_w_sha(bo)
    bn, _c2, _i2, _s2 = S.build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=True)
    n_on = int(bn.core_config.num_neurons)
    on_thr = _sha(bn.cp_neuron_firing_thresholds[:n_off])
    on_w = _conn_block_sha(bn, n_off)
    on_rf = _rf_w_sha(bn)
    return {
        "n_off": n_off, "n_on": n_on, "ladder_slice_size": n_on - n_off,
        "threshold_prefix_identical": bool(off_thr == on_thr),
        "conn_weight_prefix_identical": bool(off_w == on_w) if off_w is not None else None,
        "rf_w_identical": bool(off_rf == on_rf) if off_rf is not None else None,
        "byte_identical": bool(n_on > n_off and off_thr == on_thr
                               and (off_w is None or off_w == on_w) and (off_rf is None or off_rf == on_rf)),
    }


def _at_rest_c(seed):
    """Flag ON at neutral appraisal -> balanced ladders -> differential ~0 -> neutral tone. Reads the ladder
    differential off the shared cp_firing_states via S.read_affect_ladder."""
    xp, _ = get_backend()
    b, comp, idx, snap = S.build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=True)
    neutral = S.read_affect_ladder(b, xp, idx, snap, appraisal=0.0)
    return {"neutral_differential": float(neutral["differential"]),
            "neutral_tone_zero": bool(abs(neutral["differential"]) < S.LADDER_NEUTRAL_TOL),
            "at_rest_neutral": bool(abs(neutral["differential"]) < S.LADDER_NEUTRAL_TOL)}


def _engage_c(seed, levels=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Graded appraisal -> a MONOTONE staircase in the ladder differential (read NEURALLY through affect_out).
    FM4-safe: affect_out is array-disjoint from g_eff + the moat gate, so tone colors within the decided band."""
    xp, _ = get_backend()
    b, comp, idx, snap = S.build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=True)
    held = []
    for m in levels:
        r = S.read_affect_ladder(b, xp, idx, snap, appraisal=float(m))
        held.append(r["differential"])
    # Spearman rank correlation between appraisal level and held differential
    lv = np.asarray(levels, float)
    hv = np.asarray(held, float)
    rx = np.argsort(np.argsort(lv)).astype(float)
    ry = np.argsort(np.argsort(hv)).astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 1e-9 and ry.std() > 1e-9 else 0.0
    rng = float(max(held) - min(held))
    # FM4 structural disjointness: affect_out indices vs g_eff/moat path (the composer rf slice)
    lesion = S.read_affect_ladder(b, xp, idx, snap, appraisal=1.0, lesion=True)
    return {"levels": list(levels), "held": [float(x) for x in held], "spearman": rho, "range": rng,
            "lesion_differential": float(lesion["differential"]),
            "lesion_collapses": bool(abs(lesion["differential"]) < 0.5 * max(abs(hv).max(), 1e-9)),
            "monotone_staircase": bool(rho >= 0.8 and rng >= S.LADDER_RANGE_BAR),
            "engages": bool(rho >= 0.8 and rng >= S.LADDER_RANGE_BAR)}


def verify_seam_c(seed):
    print(f"[seam-C] byte-identity ...", flush=True)
    bid = _byte_identity_c(seed)
    print(f"   byte_identical={bid['byte_identical']} (n_off={bid['n_off']} -> n_on={bid['n_on']}, "
          f"ladder_slice={bid['ladder_slice_size']}; thr={bid['threshold_prefix_identical']} "
          f"conn={bid['conn_weight_prefix_identical']} rf_w={bid['rf_w_identical']})", flush=True)
    print(f"[seam-C] at-rest neutrality ...", flush=True)
    rest = _at_rest_c(seed)
    print(f"   neutral_diff={rest['neutral_differential']:.4f} at_rest_neutral={rest['at_rest_neutral']}", flush=True)
    print(f"[seam-C] engagement (graded staircase, read neurally through affect_out) ...", flush=True)
    eng = _engage_c(seed)
    print(f"   staircase rho={eng['spearman']:.2f} range={eng['range']:.4f} lesion_collapses={eng['lesion_collapses']} "
          f"engages={eng['engages']}", flush=True)
    ok = bool(bid["byte_identical"] and rest["at_rest_neutral"] and eng["engages"])
    return {"seam": "C", "seed": int(seed), "byte_identity": bid, "at_rest": rest, "engagement": eng, "seam_ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", choices=["A", "C"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    get_backend("numpy")
    res = verify_seam_a(a.seed) if a.seam == "A" else verify_seam_c(a.seed)
    print(f"\n[seam-{a.seam}] === seam_ok={res['seam_ok']} ===", flush=True)
    out = a.out or f"research/findings/raw/lanes/stageA/seam_{a.seam}_probe_s{a.seed}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[seam-{a.seam}] wrote {out}", flush=True)
    return 0 if res["seam_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
