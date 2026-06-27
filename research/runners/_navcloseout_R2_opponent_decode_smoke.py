"""R2 SC opponent-axis decode — CPU smoke (well-formedness + higher margin-SNR).

NO GPU, NO bridge. Pure-numpy replica of the SC orienting decode geometry
(install_spiking_sc_wiring stage-3: sc_map -> cortex_{N,E,S,W}) to prove the
R2 opponent-axis decode is (1) WELL-FORMED (returns the correct cardinal for an
eccentric synthetic SC bump) and (2) HIGHER margin-SNR than the R1-a
independent-population pop-vector decode it augments.

The two decodes (both read the SAME synthetic sc_map firing-rate field):
  - INDEPENDENT pop-vector (the R1-a baseline): each cardinal a gets the SUM over
    sc_map sites of rate(site) * max(0, u_hat_site . axis_a)  -- the cosine-tuned
    weight from install_spiking_sc_wiring(popvector=True). Four INDEPENDENT pool
    drives; the winner is argmax. The opposing pool (e.g. cortex_W for an East
    bump) still gets a LARGE positive drive from any off-axis (N/S) bump mass and
    from the bump's own spread -> the E-vs-W MARGIN is small.
  - OPPONENT push-pull (R2): each AXIS is the signed DIFFERENCE of its two opposing
    half-plane sums. drive_E_net = popvec_E - popvec_W ; drive_W_net = popvec_W -
    popvec_E (and N/S likewise). A symmetric common-mode (equal E & W mass) CANCELS;
    an eccentric bump yields a sharp single winner -> the margin is the full
    DIFFERENCE, not a small lead inside two large correlated sums. Biology:
    center-surround / ON-OFF push-pull opponency (catalog E.05/E.06) + the SC motor
    map's opponent organization (H.25). Realized on the substrate as opposing-
    half-plane sc_map sites INHIBITING the cortex pool through its cortex_FS
    interneuron (so cortex_E net = E-excitation - W-inhibition).

MARGIN-SNR metric (the headline): for a decode that produces per-cardinal drives
d[N,E,S,W], the winner's normalized margin is
    (d_win - d_runnerup) / (|d_win| + |d_runnerup| + eps)
A larger normalized margin = the position-bearing direction stands out more
decisively from its competitors = fewer ties for the stochastic tie-break to coin-
flip = a tighter spiking orienting (the 1.91x-shrink lever). We also report the
RAW margin and the common-mode (mean drive) the opponent decode rejects.

This is a DECODE-GEOMETRY smoke (does the math sharpen the margin?), NOT the nav
score (that is the controller's GPU eval). It deliberately mirrors the exact
weight formula in install_spiking_sc_wiring so a GO here means the on-substrate
opponent wiring will inject a sharper cortex margin.
"""
import math
import numpy as np

ACTION_NAMES = ["N", "E", "S", "W"]
# +sx = East, +sy = North (matches g11_bg_runner install_spiking_sc_wiring).
_CARD_AXIS = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def synthetic_sc_field(SCN, bump_cell, sigma=2.0, bg=0.05, peak=1.0):
    """A synthetic sc_map firing-rate field: a Gaussian bump centred at bump_cell
    (sx, sy in sc-sheet coords) + a uniform background. Mirrors a real SC bump
    (a localized peak at the goal's retinotopic site over a low baseline)."""
    field = np.full((SCN, SCN), bg, dtype=np.float64)
    bx, by = bump_cell
    for sy in range(SCN):
        for sx in range(SCN):
            r2 = (sx - bx) ** 2 + (sy - by) ** 2
            field[sy, sx] += peak * math.exp(-r2 / (2 * sigma * sigma))
    return field


def popvec_pool_drives(field, SCN):
    """The R1-a INDEPENDENT pop-vector drives: per cardinal, sum over sites of
    rate * max(0, u_hat_site . axis). Returns {cardinal: drive}. Exact replica of
    install_spiking_sc_wiring(popvector=True) stage-3 weight * presynaptic rate."""
    sc_center = (SCN - 1) / 2.0
    drives = {a: 0.0 for a in ACTION_NAMES}
    for sy in range(SCN):
        for sx in range(SCN):
            rate = float(field[sy, sx])
            if rate <= 0.0:
                continue
            ddx, ddy = sx - sc_center, sy - sc_center
            mag = math.hypot(ddx, ddy)
            if mag <= 0.0:
                continue
            ux, uy = ddx / mag, ddy / mag
            for a in ACTION_NAMES:
                ax, ay = _CARD_AXIS[a]
                w = max(0.0, ux * ax + uy * ay)  # cosine in [0,1]
                drives[a] += w * rate
    return drives


def opponent_axis_drives(popvec):
    """The R2 OPPONENT push-pull net drives: each cardinal's net = its own pop-vec
    half-plane sum MINUS its axis-opposite half-plane sum (rectified is applied at
    the read; here we return the SIGNED net so the margin math is transparent).
    A symmetric common-mode cancels; an eccentric bump => one sharp winner."""
    return {a: popvec[a] - popvec[_OPP[a]] for a in ACTION_NAMES}


def winner(drives):
    return max(ACTION_NAMES, key=lambda a: drives[a])


def normalized_margin(drives):
    """winner-vs-runnerup normalized margin in [-1, 1] (1 = a decisive single
    winner; ~0 = a near-tie the tie-break must coin-flip)."""
    vals = sorted((drives[a] for a in ACTION_NAMES), reverse=True)
    top, second = vals[0], vals[1]
    denom = abs(top) + abs(second) + 1e-9
    return (top - second) / denom, (top - second)


def expected_cardinal(bump_cell, SCN):
    """The ground-truth cardinal for a bump at bump_cell (which half-plane it lies
    in relative to the foveal centre)."""
    sc_center = (SCN - 1) / 2.0
    ddx, ddy = bump_cell[0] - sc_center, bump_cell[1] - sc_center
    if abs(ddx) >= abs(ddy):
        return "E" if ddx > 0 else "W"
    return "N" if ddy > 0 else "S"


def main():
    SCN = 16  # the deployed sc_map sheet side (image_size 32 // 2)
    sc_center = (SCN - 1) / 2.0
    # A spread of eccentric goals across all 4 cardinals + diagonals + a NEAR
    # (weak-margin) case (the phases where R1-a random-walks). (sx, sy) sc-sheet.
    cases = [
        ("far-E",   (14, 8)),
        ("far-W",   (1, 8)),
        ("far-N",   (8, 14)),
        ("far-S",   (8, 1)),
        ("NE-diag", (12, 12)),
        ("SW-diag", (3, 3)),
        ("near-E",  (10, 8)),   # weak margin (close to centre)
        ("near-N",  (8, 10)),
    ]
    print(f"[R2-smoke] SC opponent-axis decode well-formedness + margin-SNR "
          f"(SCN={SCN}, centre={sc_center:.1f})")
    print(f"[R2-smoke] {'case':9} {'truth':5} | {'pv-win':6} {'pv-nmarg':8} "
          f"{'pv-raw':7} | {'opp-win':7} {'opp-nmarg':9} {'opp-raw':7} | "
          f"{'nmarg-x':7} better?")
    n_wellformed_pv = 0
    n_wellformed_opp = 0
    n_opp_higher = 0
    ratios = []
    for name, bump in cases:
        truth = expected_cardinal(bump, SCN)
        field = synthetic_sc_field(SCN, bump, sigma=2.0)
        pv = popvec_pool_drives(field, SCN)
        opp = opponent_axis_drives(pv)
        pv_win, opp_win = winner(pv), winner(opp)
        pv_nm, pv_raw = normalized_margin(pv)
        opp_nm, opp_raw = normalized_margin(opp)
        if pv_win == truth:
            n_wellformed_pv += 1
        if opp_win == truth:
            n_wellformed_opp += 1
        if opp_nm > pv_nm + 1e-9:
            n_opp_higher += 1
        ratio = (opp_nm / pv_nm) if pv_nm > 1e-9 else float("inf")
        ratios.append(ratio if math.isfinite(ratio) else 0.0)
        better = "YES" if opp_nm > pv_nm + 1e-9 else "no"
        print(f"[R2-smoke] {name:9} {truth:5} | {pv_win:6} {pv_nm:8.4f} "
              f"{pv_raw:7.2f} | {opp_win:7} {opp_nm:9.4f} {opp_raw:7.2f} | "
              f"{ratio:7.2f} {better}")
    n = len(cases)
    mean_ratio = float(np.mean([r for r in ratios if r > 0])) if ratios else 0.0
    print(f"\n[R2-smoke] well-formed (decode == truth): pop-vec {n_wellformed_pv}/{n}"
          f"  opponent {n_wellformed_opp}/{n}")
    print(f"[R2-smoke] opponent normalized-margin HIGHER than pop-vec: "
          f"{n_opp_higher}/{n} cases  (mean nmarg ratio {mean_ratio:.2f}x)")
    # Common-mode rejection illustration on a SYMMETRIC (ambiguous E-W) bump:
    # equal mass at far-E and far-W -> pop-vec gives BOTH E and W a large drive
    # (small margin); the opponent net cancels to ~0 on the E-W axis (correctly
    # reports "no E-W preference") -> demonstrates the common-mode rejection.
    f_e = synthetic_sc_field(SCN, (14, 8), sigma=2.0, peak=1.0)
    f_w = synthetic_sc_field(SCN, (1, 8), sigma=2.0, peak=1.0)
    f_sym = f_e + f_w - np.full((SCN, SCN), 0.05)  # add both, subtract one bg
    pv_sym = popvec_pool_drives(f_sym, SCN)
    opp_sym = opponent_axis_drives(pv_sym)
    print(f"\n[R2-smoke] common-mode (symmetric E+W bump) rejection:")
    print(f"[R2-smoke]   pop-vec E={pv_sym['E']:.2f} W={pv_sym['W']:.2f} "
          f"(|E-W|={abs(pv_sym['E']-pv_sym['W']):.3f}, both large = small EW margin)")
    print(f"[R2-smoke]   opponent E={opp_sym['E']:.3f} W={opp_sym['W']:.3f} "
          f"(net ~0 = common mode REJECTED on the E-W axis)")

    # VERDICT: the smoke passes iff (a) the opponent decode is well-formed
    # (returns the correct cardinal on every eccentric case it's defined for) AND
    # (b) it gives a HIGHER normalized margin than the independent pop-vec on the
    # majority of cases (the margin-SNR improvement = the 1.91x-shrink lever).
    ok_wellformed = (n_wellformed_opp == n)
    ok_higher = (n_opp_higher >= (n + 1) // 2) and (mean_ratio > 1.0)
    verdict = "PASS" if (ok_wellformed and ok_higher) else "CHECK"
    print(f"\n[R2-smoke] VERDICT: {verdict}  "
          f"(well-formed={ok_wellformed}, higher-margin={ok_higher})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
