"""GPU 6-seed confirmation of the gap#5 field-quality headline.

WHY THIS IS OWED, not invented: the headline `place-specific circ 0.565 = 65% of oracle` is currently **GPU at 3
seeds** (numpy agrees to within 5% at 6 seeds). The project standard is 6 seeds on the production backend, and the
findings record this as an explicit remaining item. Nothing else here is new -- same config, same metrics, same
controls, just the seed count on cupy.

Config is the VALIDATED one (do not tune): robust firing w0=600/drive=8000, elig_tau=1000 (biological default,
shorter is worse), hetero_dep=0.2 (lowers the pedestal), elig_exp=4.0 (de-fragments). Reports width and the
RANDSET null alongside circ, because circ alone rewards concentration regardless of place structure.
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.INFO)
import importlib
B = importlib.import_module("research.runners._gap5_btsp_place_field_derisk")

SEEDS = (42, 43, 44, 100, 101, 102)


def stats(dW):
    pk, wd = [], []
    for row in dW:
        r = np.maximum(row, 0.0)
        if r.max() <= 0:
            continue
        ab = r > 0.5 * r.max()
        runs, prev = 0, ab[-1]
        for a_ in ab:
            if a_ and not prev:
                runs += 1
            prev = a_
        pk.append(max(runs, 1)); wd.append(int(ab.sum()))
    return (float(np.mean(pk)), float(np.mean(wd))) if pk else (float("nan"), float("nan"))


def main():
    print("GPU 6-SEED field quality (numpy 6-seed ref: circ 0.6853, width 16.1, randset 0.0906, place-spec +0.5947)")
    print("  %-8s %-12s %-12s %-13s %-12s %-9s" % ("seed", "peaks/cell", "WIDTH /60", "circ(dW)", "randset", "sat"))
    rows = []
    for s in SEEDS:
        M0, M1, nr, npl, ap, *_ = B.run(s, 0.0, True, 0.002, 2500.0, laps=1, dwell=30, drive=8000.0,
                                    w0=600.0, elig_tau_ms=1000.0, hetero_dep=0.2, elig_exp=4.0)
        dW = np.maximum(M1 - M0, 0.0); p, w = stats(M1 - M0)
        c = float(np.mean([B.circ_resultant(r) for r in dW]))
        sat = float((M1 >= 2500.0 * 0.98).mean())
        M0r, M1r, *_ = B.run(s, 0.0, True, 0.002, 2500.0, laps=1, dwell=30, drive=8000.0,
                                  w0=600.0, elig_tau_ms=1000.0, hetero_dep=0.2, elig_exp=4.0, randset=True)
        cr = float(np.mean([B.circ_resultant(np.maximum(r, 0.0)) for r in (M1r - M0r)]))
        # POSITION-ONLY PERMUTATION TEST (2026-07-31). The randset control above changes the DRIVE; this one
        # holds the observed increments' magnitude AND concentration exactly fixed and permutes only POSITION,
        # so it isolates spatial contiguity alone. It matters here because at the TUNED operating point
        # (w_max=150, dwell=180, density=0.25) circ_dW 0.6572 turned out to be reproduced by a position shuffle
        # to within 1.3% (ratio 1.013, p=0.42) -- i.e. all concentration, no position -- while the sigma=5
        # oracle gives ratio 4.525, p=0.0025. THIS config reports a 5.4x randset ratio, so it is the one place
        # place-specificity may genuinely live; the permutation test is what decides that.
        # Costs no extra simulation: it reshuffles increments already computed.
        perm = B.permuted_increment_circ_dW_null(M0, M1, s, n_perm=400)
        rows.append(dict(seed=s, peaks=p, width=w, circ=c, randset=cr, sat=sat,
                         perm_obs=perm["obs"], perm_null_mean=perm["null_mean"],
                         perm_null_p95=perm["null_p95"], perm_p=perm["p_value"]))
        print("  %-8d %-12.2f %-12.1f %-13.4f %-12.4f %-9.3f  perm: null %.4f ratio %.2fx p=%.4f%s"
              % (s, p, w, c, cr, sat, perm["null_mean"],
                 perm["obs"] / perm["null_mean"] if perm["null_mean"] else float("nan"), perm["p_value"],
                 "  ⛔ SATURATED" if sat > 0.2 else ""))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print("  " + "-" * 70)
    print("  GPU MEAN peaks %.2f | width %.1f | circ %.4f | randset %.4f -> place-specific %+.4f"
          % (m("peaks"), m("width"), m("circ"), m("randset"), m("circ") - m("randset")))
    print("  numpy 6-seed ref: place-specific +0.5947 | GPU 3-seed was +0.5647")
    _pr = m("perm_obs") / m("perm_null_mean") if m("perm_null_mean") else float("nan")
    _pm = float(np.median([r["perm_p"] for r in rows]))
    print("  POSITION-ONLY PERMUTATION: obs %.4f vs null %.4f = %.2fx, median p=%.4f  => %s"
          % (m("perm_obs"), m("perm_null_mean"), _pr, _pm,
             "PLACE-SPECIFIC above a concentration-matched null"
             if (_pm < 0.05 and _pr > 1.0) else
             "⛔ NOT place-specific once concentration is held fixed (the randset gap is concentration)"))
    print("     reference points: sigma=5 oracle 4.53x p=0.0025 | tuned operating point 1.01x p=0.42")
    # The default path holds the BANKED GPU 6-seed GO artifact. A CPU/numpy verification run must NOT
    # clobber it -- overwriting a banked result with a differently-produced one destroys the record that
    # the GO rests on. GAP5_FQ_OUT lets a check write elsewhere; a backup of the banked file lives at
    # fieldquality_gpu6.BANKED-GPU-6seed.json.
    out = os.environ.get("GAP5_FQ_OUT", "research/findings/raw/gap5_reader/fieldquality_gpu6.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
