"""BTSP place-field formation with between-reader soft-WTA — the method BOTH the corpus query and the
adversarial workflow converged on, after three mechanisms were refuted and the metric itself was voided.

WHY THIS DESIGN (each point earned by a specific failure tonight):
  * BTSP, not Hebbian/STDP: Bittner-Magee 2017 -- CA1 fields form in ONE lap via plateau-gated potentiation of
    inputs arriving SECONDS before/after a complex spike. Plain pairwise Hebbian is not how place fields form.
  * FAITHFUL TIMESCALE: btsp_elig_tau_ms=1000, so a 60 ms lap (my earlier probe) leaves the eligibility trace
    undecayed across an entire traversal -- every synapse equally eligible, no rule can differentiate. Here a
    field crossing takes ~30 ms x 60 = 1.8 s per lap.
  * BETWEEN-READER soft-WTA: all three refuted mechanisms had competition only WITHIN each reader's afferents,
    so N readers seeing identical population drive cannot differentiate by phase even in principle.
  * btsp_w_max ABOVE the design weight: the default 5.0 against a 250 pA weight is the standing bound trap that
    has now bitten four rules.
  * METRICS VALIDATED BEFORE USE: peak/mean is PERMUTATION-INVARIANT over place index (contiguous vs scattered
    score identically to 15 dp), so it cannot detect a spatial field. This file uses circular-resultant and
    best-contiguous-window mass, and PROVES they are permutation-SENSITIVE plus scores a sigma=5 ORACLE ceiling,
    before any mechanism runs.
  * lr=0 ARM + spikes>0 ASSERT: two claims were retracted tonight for reading a structural head-start as learning.
"""
import argparse, json, os, sys
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.INFO)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim import SimulationBridge
from sim.backend import to_host

NPLACE, NREAD, NFS = 60, 12, 12
PLACE_READ_DENSITY = float(os.environ.get("GAP5_PLACE_READ_DENSITY", "1.0"))
# The eligibility timescale sets the kernel WIDTH in place indices (tau_eff = btsp_elig_tau_ms / dwell),
# which is what the sigma=5 oracle is a width over. Exposed by env for the same reason as the density:
# the signature already carries 13 optional kwargs and a sweep does not need a 14th.
ELIG_TAU_MS = os.environ.get("GAP5_ELIG_TAU_MS")
W0, DESIGN = 250.0, 250.0


# ---------------------------------------------------------------- metrics (validated below before use)
def circ_resultant(w):
    """|sum w_i e^(i theta_i)| / sum w_i over place index on a circle. PERMUTATION-SENSITIVE by construction."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if w.sum() <= 0: return 0.0
    th = 2 * np.pi * np.arange(len(w)) / len(w)
    return float(abs((w * np.exp(1j * th)).sum()) / w.sum())


def best_window_mass(w, width=11):
    """Max fraction of total mass inside any CONTIGUOUS (circular) window of `width` place indices."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if w.sum() <= 0: return 0.0
    d = np.concatenate([w, w[:width]])
    return float(max(d[i:i + width].sum() for i in range(len(w))) / w.sum())


def peak_over_mean(w):
    """The VOIDED metric, kept ONLY to demonstrate its permutation-invariance in the validation step."""
    w = np.asarray(w, dtype=float)
    return float(w.max() / (w.mean() + 1e-12))


def validate_metrics(seed=0):
    """RULE (2026-07-29): prove the metric can detect the effect BEFORE trusting any score."""
    rng = np.random.default_rng(seed)
    x = np.arange(NPLACE)
    oracle = np.exp(-0.5 * ((x - NPLACE / 2) / 5.0) ** 2) * 100.0        # the ideal sigma=5 field
    uniform = np.full(NPLACE, 100.0 / NPLACE)
    scattered = np.zeros(NPLACE); scattered[rng.permutation(NPLACE)[:11]] = 100.0 / 11
    contig = np.zeros(NPLACE); contig[20:31] = 100.0 / 11
    perm_of_oracle = oracle[rng.permutation(NPLACE)]
    rows = []
    for name, v in (("sigma=5 ORACLE", oracle), ("uniform (null)", uniform),
                    ("11 contiguous", contig), ("11 scattered", scattered),
                    ("ORACLE permuted", perm_of_oracle)):
        rows.append((name, circ_resultant(v), best_window_mass(v), peak_over_mean(v)))
    return rows


# ---------------------------------------------------------------- bridge
def build(seed, w_inh, btsp, w_max, lat_kind="soft", w0=None, thr_scale=None, elig_tau_ms=None, hetero_dep=None, hetero_theta=None, elig_thresh=None, elig_exp=None, pgate=None):
    R = [BrainRegion(name="place", n_neurons=NPLACE, exc_fraction=1.0, internal_density=0.0),
         BrainRegion(name="read", n_neurons=NREAD, exc_fraction=1.0, internal_density=0.0)]
    # coincidence_detector=True: clustered place input -> dendritic PLATEAU, which is what gates BTSP.
    # OPT-0 ARM B (2026-07-30). density was HARDCODED 1.0, and that single constant is why the instructive
    # signal is a DC clamp: at density=1.0 every reader connects to ALL 60 place cells, so the sigma=5 bump
    # delivers the SAME ~12 coincident inputs to every reader at every position, c_count is position-INVARIANT,
    # and is_post saturates flat (it is already flat for c_count>=4). No plateau parameter can fix that --
    # k_thresh only chooses "always on" or "always off". The DC-ness is a property of the WIRING.
    # Below 1.0 each reader sees a random SUBSET, so c_count varies with position and the plateau can become an
    # EVENT. Settable by env so the sweep needs no signature threading; RECORDED in the artifact below, because
    # a knob that only exists in a filename is not provenance (the lesson that cost this project a forensic
    # reconstruction from synapse counts).
    P = [RegionPathway(from_region="place", to_region="read", density=PLACE_READ_DENSITY,
                       weight_mean=(W0 if w0 is None else w0), weight_jitter=0.3, plastic=True,
                       coincidence_detector=(True if btsp else False),
                       # plasticity_gate MUST be tagged for cp_plasticity_rate_gain to be ALLOCATED --
                       # without it the array is None and any per-synapse gating silently no-ops (caught by
                       # an engagement counter reading 'gated steps 0', not by the result looking wrong).
                       plasticity_gate=pgate)]
    if w_inh > 0:
        # BETWEEN-READER competition: readers drive a shared FS pool which inhibits them all back.
        # Recruited in proportion to reader activity => the best-driven reader survives (soft k-WTA),
        # unlike the uniform inhibition that produced no differentiation.
        R.append(BrainRegion(name="fs", n_neurons=NFS, exc_fraction=0.0, internal_density=0.0))
        P += [RegionPathway(from_region="read", to_region="fs", density=1.0, weight_mean=300.0,
                            weight_jitter=0.0, plastic=False),
              RegionPathway(from_region="fs", to_region="read", density=1.0, weight_mean=w_inh,
                            weight_jitter=0.0, plastic=False)]
    kw = {}
    if btsp:
        # BTSP REQUIRES AN APICAL COMPARTMENT. bridge.py:8067 gates the whole block on
        # `self.cp_v_apical is not None`, and is_post = max(v_apical - v_hold, 0) -- the PLATEAU above hold is
        # the instructive signal (bridge.py:8088). cp_v_apical is allocated only under
        # enable_two_compartment_dap (bridge.py:7157-7160). Without it, enable_btsp=True is a NO-OP and every
        # arm returns delta EXACTLY 0.0000 -- which is UNDEFINED, not a negative. (Caught that way on first run.)
        # THE FULL DEPENDENCY CHAIN (found the hard way, each link verified in bridge.py):
        #   enable_coincidence_detection -> the block that computes clustered drive + the plateau
        #   enable_two_compartment_dap   -> allocates cp_v_apical INSIDE that block (bridge.py:7157-7160)
        #   enable_btsp                  -> the rule, gated on `cp_v_apical is not None` (bridge.py:8067)
        #   pathway coincidence_detector -> supplies the clustered input that triggers the plateau
        # Miss any link and enable_btsp is a silent NO-OP: every delta reads EXACTLY 0.0000, which is
        # UNDEFINED, not a negative. The apical_max assert below is what caught it twice.
        kw.update(enable_btsp=True, btsp_w_max=w_max, btsp_w_min=0.0,
                  enable_two_compartment_dap=True, enable_coincidence_detection=True,
                  coincidence_k_threshold=4.0)
        # btsp_elig_tau_ms defaults to 1000 ms. The ratio that governs FIELD WIDTH is
        # elig_tau / field-crossing-time: biology ~1000/1000 = 1, but this probe's crossing is
        # dwell ms (30), giving ~33 -- an eligibility window spanning 33 field-widths.
        if elig_tau_ms is not None:
            kw["btsp_elig_tau_ms"] = elig_tau_ms
        # btsp_hetero_dep: HETEROSYNAPTIC COMPETITION. The engine's own comment says it "lowers the
        # pedestal without lowering the peak" -- and a near-global pedestal (width 51/60 of afferents
        # potentiated) is precisely this arc's defect. Default 0.0 => inert, so it must be set explicitly.
        if hetero_dep is not None:
            kw["btsp_hetero_dep"] = hetero_dep
        if hetero_theta is not None:
            kw["btsp_hetero_theta"] = hetero_theta
        # These two act in the AFFERENT (place-index) dimension, which is where the fragmentation lives:
        #   btsp_elig_hard_thresh -- k-WTA gate ON PRESYNAPTIC ELIGIBILITY: only afferents whose eligibility
        #     clears the bar may potentiate. Eligibility tracks recent presynaptic activity and the sweep's
        #     bump activates CONTIGUOUS place cells, so a threshold should select a contiguous block.
        #   btsp_elig_exponent -- supralinear eligibility, sharpening high-vs-low without a hard cut.
        if elig_thresh is not None:
            kw["btsp_elig_hard_thresh"] = elig_thresh
        if elig_exp is not None:
            kw["btsp_elig_exponent"] = elig_exp
    cfg = CoreSimConfig(seed=seed, dt_ms=1.0, enable_brain_region_framework=True, brain_regions=R,
                        region_pathways=P, enable_hebbian_learning=False, enable_stdp=False,
                        enable_homeostasis=False, enable_structural_plasticity=False,
                        enable_ou_process=False, **kw)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def wmat(b):
    rm = b.region_manager
    pl = np.asarray(rm.indices("place")); rd = np.asarray(rm.indices("read"))
    C = b.cp_connections
    ip = to_host(C.indptr); idx = to_host(C.indices); dat = to_host(C.data)
    ps = {int(p): i for i, p in enumerate(pl)}; rs = {int(r): i for i, r in enumerate(rd)}
    M = np.zeros((NREAD, NPLACE))
    for pre in pl:
        pre = int(pre)
        for k in range(int(ip[pre]), int(ip[pre + 1])):
            po = int(idx[k])
            if po in rs: M[rs[po], ps[pre]] += float(dat[k])
    return M


def run(seed, w_inh, btsp, lr, w_max, laps=5, dwell=30, drive=3000.0, width=5.0, randset=False, w0=None, thr_scale=None, elig_tau_ms=None, hetero_dep=None, hetero_theta=None, elig_thresh=None, elig_exp=None, pgate=None):
    """randset=True -> each step drives a RANDOM SET of place cells of the same size/intensity as the bump,
    so total drive and total activity are matched but there is NO moving bump and NO place manifold at all.
    This is the exact control that refuted the k-WTA gate (+1.217 of its +1.272 survived it). If the BTSP gain
    survives randset, it is generic potentiation, NOT place-field formation."""
    b = build(seed, w_inh, btsp, w_max, w0=w0, thr_scale=thr_scale, elig_tau_ms=elig_tau_ms,
              hetero_dep=hetero_dep, hetero_theta=hetero_theta,
              elig_thresh=elig_thresh, elig_exp=elig_exp, pgate=pgate)
    if btsp:
        b.core_config.btsp_learning_rate = lr
    rm = b.region_manager; pl = rm.indices("place"); rd = rm.indices("read")
    if thr_scale is not None and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
        _rd = np.asarray(rm.indices("read"))
        b.cp_neuron_firing_thresholds[_rd] = b.cp_neuron_firing_thresholds[_rd] * thr_scale
    M0 = wmat(b); nread = 0; nplace = 0
    x = np.arange(NPLACE)
    rs = np.random.default_rng(seed * 31337)
    for lap in range(laps):
        for c in range(NPLACE):
            if randset:
                # match the bump's total mass, but scatter it over random indices (no locality, no travel)
                bump = np.exp(-0.5 * ((x - c) / width) ** 2)
                p = np.zeros(NPLACE); p[rs.permutation(NPLACE)[:int(round(bump.sum()))]] = 1.0
            else:
                p = np.exp(-0.5 * ((x - c) / width) ** 2)
            for _ in range(dwell):                       # ~30 ms per field crossing => 1.8 s per lap
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[pl] = drive * p
                b._run_one_simulation_step()
                b.runtime_state.current_time_ms += b.core_config.dt_ms   # the clock bug, avoided
                nread += int(to_host(b.cp_firing_states[rd]).sum())
                nplace += int(to_host(b.cp_firing_states[pl]).sum())
    M1 = wmat(b)
    # ENGAGEMENT on the MECHANISM, not just on firing: was there any apical plateau above hold at all?
    # OPT-0 ARM A (2026-07-30). `apical_max` ALONE CANNOT DISTINGUISH "on once" FROM "on always" -- it is a single
    # scalar max over the whole run, so a plateau that fires briefly at one position and a plateau clamped high
    # for every reader at every position produce the SAME number. Adversarial review (4/4 refuted the fix I was
    # about to build) derived from sim/kernels.py:325-345 + the apical ODE that is_post should be PINNED at
    # ~34.82 mV for every reader, because density=1.0 gives every reader ~12 coincident inputs at every step and
    # is_post is flat for c_count>=4. That is an inference from arithmetic, NOT a measurement -- this arc has
    # never instrumented its own instructive signal. These four stats measure it.
    # THE TELL: if the spread ACROSS READERS is ~0 while the max is high, the instructive signal is a DC clamp
    # and the BTSP write is uniform by construction (dw = eta*elig_pre*is_post*(w_max-w) with is_post constant),
    # which makes the field width a necessary output rather than a sharpening deficit. If instead readers differ,
    # the DC-clamp diagnosis is REFUTED and three planned builds were aimed at the wrong term.
    if getattr(b, "cp_v_apical", None) is not None:
        _va = to_host(b.cp_v_apical[rd])
        apical_max = float(_va.max())
        apical_stats = {"max": apical_max, "min": float(_va.min()),
                        "mean": float(_va.mean()), "std_across_readers": float(_va.std())}
    else:
        apical_max = float("nan")
        apical_stats = {"max": float("nan"), "min": float("nan"),
                        "mean": float("nan"), "std_across_readers": float("nan")}
    return M0, M1, nread, nplace, apical_max, apical_stats


def permuted_increment_null(M0, M1, seed):
    """THE CORRECT MATCHED NULL: permute the ACTUAL trained increments across place indices.

    Holds magnitude AND concentration/kurtosis EXACTLY fixed; randomises only POSITION. So it isolates the one
    property under test -- SPATIAL CONTIGUITY.

    ⚠️ THIS REPLACES a Dirichlet-based "sharpening-matched null" that was MIS-SPECIFIED and nearly produced a
    false negative (2026-07-29). That version split the dW budget by a Dirichlet, which CONCENTRATES mass on a
    few random indices -- and circ_resultant rewards concentration -- so it scored 57% of the real gain and the
    criterion printed "NOT place-specific". It differed from the treatment in TWO properties (position AND
    concentration), so it could isolate neither. RULE: a null must differ from the treatment in EXACTLY ONE
    property, the one being tested.
    """
    rng = np.random.default_rng(seed * 977)
    dW = M1 - M0
    Mp = M0 + np.array([dW[j][rng.permutation(dW.shape[1])] for j in range(dW.shape[0])])
    return float(np.mean([circ_resultant(r) for r in Mp]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--dwell", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--w-max", type=float, default=2500.0)
    ap.add_argument("--w-inh", type=float, default=150.0)
    ap.add_argument("--out", default="research/findings/raw/gap5_reader/btsp_place_field.json")
    a = ap.parse_args()

    print("=" * 96)
    print("STEP A -- VALIDATE THE METRICS BEFORE ANY MECHANISM RUNS (the 2026-07-29 rule)")
    print("  %-18s %-14s %-16s %-14s" % ("pattern", "circ_result", "window_mass", "peak/mean(VOID)"))
    rows = validate_metrics()
    for n, c, wm, pm in rows:
        print("  %-18s %-14.4f %-16.4f %-14.4f" % (n, c, wm, pm))
    orc, orw = rows[0][1], rows[0][2]
    ctg, sct = rows[2], rows[3]
    permo = rows[4]
    ok_perm = abs(permo[1] - orc) > 0.05          # permuting the oracle MUST change a valid metric
    ok_contig = (ctg[1] - sct[1]) > 0.05          # contiguous MUST beat scattered
    ok_void = abs(ctg[3] - sct[3]) < 1e-9         # and peak/mean MUST be identical (proving it is blind)
    print("  CHECK permuting the oracle changes circ_resultant: %s (%.4f -> %.4f)"
          % ("PASS" if ok_perm else "FAIL", orc, permo[1]))
    print("  CHECK contiguous beats scattered:                   %s (%.4f vs %.4f)"
          % ("PASS" if ok_contig else "FAIL", ctg[1], sct[1]))
    print("  CHECK peak/mean is BLIND to that difference:        %s (%.6f vs %.6f)"
          % ("CONFIRMED-BLIND" if ok_void else "unexpected", ctg[3], sct[3]))
    if not (ok_perm and ok_contig):
        print("  ABORT: the metric failed its own validity check; no mechanism number would be meaningful.")
        return 1
    print("  ORACLE CEILING: circ_resultant=%.4f  window_mass=%.4f  <- the number to beat" % (orc, orw))

    print()
    print("STEP B -- MECHANISM ARMS (lr=0 is an ARM; spikes>0 asserted)")
    arms = [("lr0_btsp",      dict(w_inh=0.0,      btsp=True,  lr=0.0)),
            ("btsp",          dict(w_inh=0.0,      btsp=True,  lr=a.lr)),
            ("lr0_btsp_wta",  dict(w_inh=a.w_inh,  btsp=True,  lr=0.0)),
            ("btsp_wta",      dict(w_inh=a.w_inh,  btsp=True,  lr=a.lr))]
    res = {}
    print("  %-14s %-7s %-11s %-11s %-11s %-9s" % ("arm", "seed", "circ", "window", "d_circ", "read_spk"))
    for name, kw in arms:
        res[name] = []
        for s in a.seeds:
            M0, M1, nread, nplace, apmax, apst = run(s, kw["w_inh"], kw["btsp"], kw["lr"], a.w_max,
                                        laps=a.laps, dwell=a.dwell,
                                        elig_tau_ms=(float(ELIG_TAU_MS) if ELIG_TAU_MS else None))
            c1 = float(np.mean([circ_resultant(r) for r in M1]))
            w1 = float(np.mean([best_window_mass(r) for r in M1]))
            c0 = float(np.mean([circ_resultant(r) for r in M0]))
            # circ ON THE WEIGHT CHANGE, which is the quantity the 6-seed headline actually reports.
            # `circ` above is computed on FINAL weights and is dominated by the random initial structure -- with
            # lr=0 the final weights simply ARE the init, and a sparse random vector over 60 place indices has a
            # high circular resultant BY CONSTRUCTION. Reading the density sweep off `circ` therefore said
            # nothing about learning, and I briefly (and wrongly) concluded from it that BTSP was damaging the
            # field. Recording circ_dW alongside makes the headline's quantity available in the artifact instead
            # of being recomputed by hand from two columns of a markdown table.
            cdW = float(np.mean([circ_resultant(r) for r in (M1 - M0)]))
            res[name].append(dict(seed=s, circ=c1, circ_dW=cdW, window=w1, circ_init=c0, d_circ=c1 - c0,
                                  read_spikes=nread, place_spikes=nplace, apical_max=apmax,
                                  apical=apst,          # OPT-0 arm A: the instructive signal, actually measured
                                  dW=float(np.abs(M1 - M0).mean())))
            print("  %-14s %-7d %-11.4f %-11.4f %-+11.4f %-9d%s" % (
                name, s, c1, w1, c1 - c0, nread,
                ("  ⛔ READ SILENT" if nread == 0 else
                 ("  ⛔ NO APICAL COMPARTMENT (btsp is a NO-OP => UNDEFINED)" if apmax != apmax else ""))))
    print()
    m = lambda n, k: float(np.mean([r[k] for r in res[n]]))
    print("  %-14s %-11s %-11s %-11s %-11s" % ("arm", "circ", "window", "dW", "read_spk"))
    for n, _ in arms:
        print("  %-14s %-11.4f %-11.4f %-11.4f %-11.0f" % (n, m(n, "circ"), m(n, "window"), m(n, "dW"), m(n, "read_spikes")))
    print()
    print("  LEARNING (vs its OWN lr=0 arm, the only comparison that counts):")
    print("    BTSP alone      : circ %+.4f   window %+.4f" % (m("btsp", "circ") - m("lr0_btsp", "circ"),
                                                               m("btsp", "window") - m("lr0_btsp", "window")))
    print("    BTSP + softWTA  : circ %+.4f   window %+.4f" % (m("btsp_wta", "circ") - m("lr0_btsp_wta", "circ"),
                                                               m("btsp_wta", "window") - m("lr0_btsp_wta", "window")))
    print("    vs ORACLE ceiling circ %.4f / window %.4f" % (orc, orw))

    print()
    print("STEP C -- THE CONTROLS THAT REFUTED THE PREVIOUS THREE MECHANISMS")
    ctrl = {}
    for label, kw in (("btsp", dict(w_inh=0.0, btsp=True, lr=a.lr)),
                      ("btsp_wta", dict(w_inh=a.w_inh, btsp=True, lr=a.lr))):
        rc, rp = [], []
        for s_ in a.seeds:
            # (i) RANDSET: same activity, no place manifold. Gain must COLLAPSE.
            M0r, M1r, nr_, np_, ap_, _apst_r = run(s_, kw["w_inh"], kw["btsp"], kw["lr"], a.w_max,
                                          laps=a.laps, dwell=a.dwell, randset=True)
            rc.append(float(np.mean([circ_resultant(r) for r in M1r])) - float(np.mean([circ_resultant(r) for r in M0r])))
            # (ii) SHARPENING-MATCHED NULL: same total |dW|, no place structure
            # (ii) PERMUTED-INCREMENT NULL on the REAL place-sweep arms (same magnitudes + concentration,
            #      positions shuffled). Recompute the sweep arm so the increments are the genuine ones.
            M0p, M1p, _, _, _, _ = run(s_, kw["w_inh"], kw["btsp"], kw["lr"], a.w_max,
                                    laps=a.laps, dwell=a.dwell, randset=False)
            base = float(np.mean([circ_resultant(r) for r in M0p]))
            rp.append(permuted_increment_null(M0p, M1p, s_) - base)
        ctrl[label] = dict(randset_d_circ=float(np.mean(rc)), permuted_increment_d_circ=float(np.mean(rp)))
        real = m(label, "circ") - m("lr0_" + label if label == "btsp" else "lr0_btsp_wta", "circ")
        print("  %-10s  place-sweep gain %+.4f | RANDSET (no place) %+.4f | PERM-INCREMENTS %+.4f  => %s"
              % (label, real, ctrl[label]["randset_d_circ"], ctrl[label]["permuted_increment_d_circ"],
                 "PLACE-SPECIFIC" if (real > 2 * max(ctrl[label]["randset_d_circ"], ctrl[label]["permuted_increment_d_circ"]))
                 else "⛔ NOT place-specific (generic potentiation)"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(metric_validation=[list(r) for r in rows], oracle=dict(circ=orc, window=orw),
                   laps=a.laps, dwell=a.dwell, lr=a.lr, w_max=a.w_max, w_inh=a.w_inh, arms=res,
                   # RECORD THE KNOBS. place_read_density is the OPT-0 arm-B lever and backend decides whether
                   # the run is decisive at all -- both were previously unrecoverable from the artifact, which
                   # is exactly how a numpy "6-seed GO" got banked across four lanes today.
                   place_read_density=PLACE_READ_DENSITY,
                   elig_tau_ms=(float(ELIG_TAU_MS) if ELIG_TAU_MS else None),
                   backend=os.environ.get("SIM_BACKEND", "unset"),
                   controls=ctrl),
              open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
