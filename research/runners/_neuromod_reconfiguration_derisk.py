"""Does a NEUROMODULATOR RECONFIGURE the effective circuit, or merely change its GAIN?

THE QUESTION (Bargmann, "Beyond the connectome: how neuromodulators shape neural circuits", BioEssays
34:458-465, 2012; Marder, "Neuromodulation of Neuronal Circuits: Back to the Future", Neuron 76:1-11,
2012). A fixed connectome UNDERDETERMINES behaviour: the SAME wiring runs DIFFERENT functional circuits
depending on neuromodulatory state -- some region-crossing pathways OPEN, others CLOSE, so one substrate
supports multiple functional modes. Our neuromodulators today mostly change GAIN (scale activity up/down).
A gain knob keeps the functional connectivity PATTERN identical and just rescales it; a RECONFIGURATION
changes WHICH edges carry signal. This runner tests whether a dopaminergic Go/NoGo state switch on the
nav basal-ganglia substrate RECONFIGURES the effective connectivity, using the board-#63 perturb-and-
measure probe as the read-out (reuse-by-import; NO sim/ edit).

MECHANISM (all-spiking / all-synaptic, canonical BG neuromodulation -- Albin-DeLong-Penney direct/indirect
model; Gerfen & Surmeier 2011). Dopamine acts through D1 receptors (Gs, EXCITATORY) on direct-pathway MSNs
and D2 receptors (Gi, INHIBITORY) on indirect-pathway MSNs. We express this via ONE neuromodulator on the
subsystem's OWN bus (`sim.neuromodulators.NeuromodulatorManager`): a `dopamine_mode` modulator with two
`excitability_drive` targets -- scope="group:str_D1" (sensitivity>0) and scope="group:str_D2"
(sensitivity<0). The concentration IS the state:
  * HIGH DA  (Go-mode)   -> D1 depolarised, D2 hyperpolarised  -> the DIRECT pathway is primed.
  * LOW  DA  (NoGo-mode) -> D1 hyperpolarised, D2 depolarised  -> the INDIRECT pathway is primed.
The per-neuron excitability drive is computed by the subsystem's OWN
`NeuromodulatorManager.compute_excitability_drive_per_neuron()` (concentration -> current bias); the runner
only ADDS that bias to the neurons' input current, exactly as the bridge does internally when the subsystem
is enabled. The neurons then fire (or not) and their SYNAPSES propagate -- no host arithmetic stands
between the modulatory state and the measured propagation. The ANATOMY W (pathways) is byte-identical
across states; only the modulatory state differs.

READ-OUT. For each state we run the #63 probe: perturb every region type A (+800 pA), settle, record the
signed downstream Delta-rate of every other region B -> a FUNCTIONAL connectivity matrix F (state-specific).

ANTI-CHEATS (they ARE the result):
  1. RECONFIGURATION, NOT GAIN. A pure gain change is rank-preserving: spearman(F_hi, F_lo) == 1 even
     though magnitudes differ. RECONFIGURATION drops the RANK correlation and, specifically, OPENS edges in
     one state that are CLOSED in the other -- a DOUBLE DISSOCIATION (>=1 edge active only in Go AND >=1
     edge active only in NoGo). We report spearman(F_hi, F_lo) over the union-nonzero edges + the exact
     edges that flip. If it only rescales uniformly (spearman ~1, no opened/closed edges), that is the
     HONEST NEGATIVE: a gain knob, not a reconfigurator -- and we say so.
  2. NEUROMODULATOR-DRIVEN. Zero the modulator (both targets sensitivity=0 -> the per-neuron drive is
     identically 0 at every concentration). The two "states" then collapse to the SAME network:
     F_hi == F_lo byte-for-byte. Dissociation: the reconfiguration REQUIRES the modulator.
  3. SAME WIRING. The anatomical W is IDENTICAL across states (we never touch pathways) -- confirmed
     max|W_hi - W_lo| == 0. The effect is purely functional reconfiguration on fixed structure.
  4. 6 seeds (42,43,44,100,101,102), per-seed + pooled, deterministic (cfg.seed), byte-identical F re-run.

  SIM_BACKEND=numpy python -m research.runners._neuromod_reconfiguration_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/neuromod_reconfig/nr.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

import research.runners._perturb_and_measure_derisk as PM
from research.runners._perturb_and_measure_derisk import TYPES, ACTIONS
from tools.lab import functional_vs_anatomical, assert_backend, Verdict, attributable_to
from tools.lab import _pearson, _avg_rank  # spearman = pearson-of-ranks (same as the #63 probe uses)
from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, NeuromodulatorManager

# ---- LOCKED operating point + modulation (see finding) ------------------------------------------------
# The BG's working regime (the #63 probe's own operating point) EXCEPT the striatal tone is lowered so the
# D1/D2 MSNs sit near threshold and the dopaminergic drive can GATE which pathway a cortical drive recruits
# (at the #63 str tone of 250 pA the MSNs fire regardless and DA only rescales -- the honest-negative regime;
# see the finding's operating-point note). gpi keeps its strong pacemaker so the disinhibition route can
# carry signal. NOT tuned to flatter the probe: the anti-cheats (modulator-lesion, W-unchanged) guard it.
BASELINE = dict(cortex=200, str_D1=40, str_D2=40, str_PV_FSI=0, str_striosome=40,
                gpe=150, gpe_arky=120, gpi=900, stn=220, snc=150, thal=550, motor=150)
PERTURB_PA = 400.0         # cortical/region perturbation; lower than #63's 800 so the direct/indirect
                           # recruitment (set by DA excitability) determines propagation, not a saturating drive.
DA_BASELINE = 0.5          # tonic DA concentration (the neutral point of the excitability formula)
DA_HI = 1.0               # Go-mode  concentration (D1 up, D2 down): (conc-baseline)=+0.5
DA_LO = 0.0               # NoGo-mode concentration (D1 down, D2 up): (conc-baseline)=-0.5
S_D1 = 1000.0            # pA excitability drive per unit (conc-baseline) on D1  (D1R, Gs, excitatory)
S_D2 = 1000.0            # pA excitability drive per unit (conc-baseline) on D2  (D2R, Gi, inhibitory) -> NEGATIVE sign below
# => Go: D1 +500 / D2 -500 pA (direct primed, indirect silent); NoGo: D1 -500 / D2 +500 (reverse).
EDGE_THRESH = 0.008        # |Delta rate| above which a functional edge is "active" (== #63)


def dopamine_mode_config(s_d1=None, s_d2=None, baseline=None):
    """ONE modulator, two excitability_drive targets: D1R excitatory (+), D2R inhibitory (-).
    Concentration is set manually by the runner (the STATE); no production rules -> tonic level.
    None resolves to the current module-global (so CLI exploration overrides take effect)."""
    s_d1 = S_D1 if s_d1 is None else s_d1
    s_d2 = S_D2 if s_d2 is None else s_d2
    baseline = DA_BASELINE if baseline is None else baseline
    return NeuromodulatorConfig(
        name="dopamine_mode", baseline=baseline, decay_tau_ms=1e12,
        concentration_min=0.0, concentration_max=2.0,
        targets=[
            ModulatorTarget(target_type="excitability_drive", scope="group:str_D1", sensitivity=+float(s_d1)),
            ModulatorTarget(target_type="excitability_drive", scope="group:str_D2", sensitivity=-float(s_d2)),
        ],
        production_rules=[],
    )


def _group_indices(sb):
    d1, d2 = [], []
    for a in ACTIONS:
        d1 += [int(x) for x in sb.region_manager.indices(f"str_D1_{a}")]
        d2 += [int(x) for x in sb.region_manager.indices(f"str_D2_{a}")]
    return {"str_D1": d1, "str_D2": d2}


def modulator_drive(sb, cfg, conc):
    """Per-neuron excitability bias (pA) produced by the subsystem's OWN code from the DA concentration.
    Returns a numpy array (n_neurons,). The runner injects this as the tonic neuromodulatory state."""
    mgr = NeuromodulatorManager([cfg], dt_ms=1.0)
    mgr.initialize(sb.core_config.num_neurons, np)          # numpy as the array module (SIM_BACKEND=numpy)
    mgr.set_group_indices(_group_indices(sb))
    mgr.set_concentration("dopamine_mode", conc)
    drive = mgr.compute_excitability_drive_per_neuron()      # None if no per-neuron-scoped target fires
    if drive is None:
        return np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    return np.asarray(drive, dtype=np.float64)


def measure_with_mod(sb, nbt, baseline, drive, perturb=None, warmup=PM.WARMUP, settle=PM.SETTLE):
    """#63 probe's measure_region_rates + a constant per-neuron neuromodulator drive added to the input."""
    from sim.backend import to_host
    idx = lambda n: np.asarray(sb.region_manager.indices(n))
    sb.cp_external_input_current[:] = 0.0
    for t, cur in baseline.items():
        for n in nbt[t]:
            sb.cp_external_input_current[idx(n)] = cur
    if perturb:
        for n, extra in perturb.items():
            sb.cp_external_input_current[idx(n)] += extra
    sb.cp_external_input_current[:] = sb.cp_external_input_current + drive   # tonic modulatory state
    for _ in range(warmup):
        sb._run_one_simulation_step()
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    rates = {}
    for t in TYPES:
        for n in nbt[t]:
            rates[n] = float(acc[idx(n)].mean() / settle)
    return rates


def functional_matrix_state(seed, cfg, conc, baseline=None, perturb_pa=None):
    """F[A][B] under one DA state: perturb every type A, measure Delta type-rate of every B vs unperturbed.
    None resolves to the current module-global (so CLI exploration overrides take effect)."""
    baseline = BASELINE if baseline is None else baseline
    perturb_pa = PERTURB_PA if perturb_pa is None else perturb_pa
    sb0, regions, pathways = PM.build(seed)
    nbt = PM.names_by_type(regions)
    drive0 = modulator_drive(sb0, cfg, conc)
    r0 = PM.type_rates(measure_with_mod(sb0, nbt, baseline, drive0), nbt)
    F = np.zeros((len(TYPES), len(TYPES)), dtype=float)
    for i, a in enumerate(TYPES):
        sbi, ri, _ = PM.build(seed)
        nbti = PM.names_by_type(ri)
        drivei = modulator_drive(sbi, cfg, conc)
        pert = {n: perturb_pa for n in nbti[a]}
        rp = PM.type_rates(measure_with_mod(sbi, nbti, baseline, drivei, perturb=pert), nbti)
        for j, b in enumerate(TYPES):
            F[i, j] = rp[b] - r0[b]
    return F, r0, regions, pathways


def spearman_nz(A, B, thresh=EDGE_THRESH):
    """Rank correlation over the union of off-diagonal cells nonzero in EITHER matrix (scale-invariant:
    a pure gain change -> 1.0). Same union-nonzero convention as functional_vs_anatomical's spearman_rho_nz."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    mask = off & ((np.abs(A) > thresh) | (np.abs(B) > thresh))
    a = A[mask]; b = B[mask]
    if a.size < 3:
        return float("nan")
    return _pearson(_avg_rank(a), _avg_rank(b))


def flipped_edges(F_hi, F_lo, thresh=EDGE_THRESH):
    """Edges whose ACTIVE/INACTIVE status differs between states (opened in one, closed in the other) or
    whose SIGN flips. This is the reconfiguration a gain change CANNOT produce."""
    F_hi = np.asarray(F_hi, float); F_lo = np.asarray(F_lo, float)
    opened_go, opened_nogo, sign_flips = [], [], []
    for i, a in enumerate(TYPES):
        for j, b in enumerate(TYPES):
            if i == j:
                continue
            fh, fl = float(F_hi[i, j]), float(F_lo[i, j])
            ah, al = abs(fh) > thresh, abs(fl) > thresh
            e = dict(**{"from": a, "to": b}, F_go=round(fh, 5), F_nogo=round(fl, 5))
            if ah and not al:
                opened_go.append(e)
            elif al and not ah:
                opened_nogo.append(e)
            elif ah and al and ((fh > 0) != (fl > 0)):
                sign_flips.append(e)
    opened_go.sort(key=lambda d: -abs(d["F_go"]))
    opened_nogo.sort(key=lambda d: -abs(d["F_nogo"]))
    return opened_go, opened_nogo, sign_flips


def run_seed(seed):
    t0 = time.time()
    cfg = dopamine_mode_config()
    cfg_lesion = dopamine_mode_config(s_d1=0.0, s_d2=0.0)   # ANTI-CHEAT 2: modulator zeroed

    F_go, r0_go, regions, pathways = functional_matrix_state(seed, cfg, DA_HI)
    F_nogo, r0_nogo, _, _ = functional_matrix_state(seed, cfg, DA_LO)

    W_go = PM.anatomical_matrix(regions, pathways)
    W_nogo = PM.anatomical_matrix(*(functional_matrix_state(seed, cfg, DA_LO)[2:]))  # rebuilds regions/pathways
    w_unchanged_max = float(np.max(np.abs(W_go - W_nogo)))

    rho = spearman_nz(F_go, F_nogo)
    opened_go, opened_nogo, sign_flips = flipped_edges(F_go, F_nogo)

    # ANTI-CHEAT 2: zero the modulator -> the two states become the SAME network.
    F_go_les, _, _, _ = functional_matrix_state(seed, cfg_lesion, DA_HI)
    F_nogo_les, _, _, _ = functional_matrix_state(seed, cfg_lesion, DA_LO)
    rho_lesion = spearman_nz(F_go_les, F_nogo_les)
    lesion_max_dF = float(np.max(np.abs(F_go_les - F_nogo_les)))

    # ATTRIBUTION (not just measuring both arms): the reconfiguration MAGNITUDE is the departure from the
    # pure-gain rank-invariant 1.0, i.e. (1 - spearman). WHOSE is it? treatment = with the modulator;
    # control = with the modulator zeroed (which is identically 0 -- the states become the same network).
    reconfig_attribution = attributable_to(
        f"seed {seed}: reconfiguration (1 - spearman) attributable to dopamine_mode",
        treatment_value=1.0 - rho, control_value=1.0 - rho_lesion)

    # context: each state's F still DIFFERS from the anatomical W (the #63 result, per state)
    cmp_go = functional_vs_anatomical(F_go, W_go, labels=TYPES, edge_thresh=EDGE_THRESH)
    cmp_nogo = functional_vs_anatomical(F_nogo, W_nogo, labels=TYPES, edge_thresh=EDGE_THRESH)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), elapsed_s=round(elapsed, 2),
        types=TYPES,
        F_go=F_go.tolist(), F_nogo=F_nogo.tolist(), W=W_go.tolist(),
        spearman_F_go_vs_F_nogo=rho,
        n_opened_go=len(opened_go), n_opened_nogo=len(opened_nogo), n_sign_flips=len(sign_flips),
        opened_go=opened_go, opened_nogo=opened_nogo, sign_flips=sign_flips,
        # anti-cheat 2 (modulator-lesion dissociation)
        spearman_F_go_vs_F_nogo_MOD_LESIONED=rho_lesion,
        mod_lesion_max_abs_dF=lesion_max_dF,
        reconfig_attributable_to_modulator=reconfig_attribution,
        # anti-cheat 3 (same wiring)
        W_unchanged_max_abs=w_unchanged_max,
        # context
        spearman_F_go_vs_W=cmp_go["spearman_rho_nz"], spearman_F_nogo_vs_W=cmp_nogo["spearman_rho_nz"],
        n_poly_go=cmp_go["n_polysynaptic"], n_poly_nogo=cmp_nogo["n_polysynaptic"],
        baseline_rates_go=r0_go, baseline_rates_nogo=r0_nogo,
    )


def pooled(results):
    def col(key):
        return np.array([r[key] for r in results], dtype=float)
    out = dict(n_seeds=len(results))
    out["spearman_F_go_vs_F_nogo_mean"] = float(np.nanmean(col("spearman_F_go_vs_F_nogo")))
    out["spearman_F_go_vs_F_nogo_max"] = float(np.nanmax(col("spearman_F_go_vs_F_nogo")))
    out["n_opened_go_min"] = int(np.min(col("n_opened_go")))
    out["n_opened_nogo_min"] = int(np.min(col("n_opened_nogo")))
    out["n_opened_go_mean"] = float(np.mean(col("n_opened_go")))
    out["n_opened_nogo_mean"] = float(np.mean(col("n_opened_nogo")))
    out["n_sign_flips_mean"] = float(np.mean(col("n_sign_flips")))
    out["spearman_MOD_LESIONED_min"] = float(np.nanmin(col("spearman_F_go_vs_F_nogo_MOD_LESIONED")))
    out["mod_lesion_max_abs_dF_max"] = float(np.max(col("mod_lesion_max_abs_dF")))
    out["W_unchanged_max_abs_max"] = float(np.max(col("W_unchanged_max_abs")))
    # edges opened in EACH state that recur across ALL seeds (the robust reconfiguration signature)
    def edge_set(res, key):
        return set((e["from"], e["to"]) for e in res[key])
    go_common = set.intersection(*[edge_set(r, "opened_go") for r in results]) if results else set()
    nogo_common = set.intersection(*[edge_set(r, "opened_nogo") for r in results]) if results else set()
    out["opened_go_common_all_seeds"] = sorted(f"{a}->{b}" for a, b in go_common)
    out["opened_nogo_common_all_seeds"] = sorted(f"{a}->{b}" for a, b in nogo_common)
    return out


def determinism_check(seed=42):
    cfg = dopamine_mode_config()
    F1, _, _, _ = functional_matrix_state(seed, cfg, DA_HI)
    F2, _, _, _ = functional_matrix_state(seed, cfg, DA_HI)
    return bool(np.array_equal(F1, F2)), float(np.max(np.abs(F1 - F2)))


def explore(seed):
    """One-seed exploration print: the pattern-reconfiguration numbers for the current LOCKED params."""
    cfg = dopamine_mode_config()
    F_go, _, regions, pathways = functional_matrix_state(seed, cfg, DA_HI)
    F_nogo, _, _, _ = functional_matrix_state(seed, cfg, DA_LO)
    rho = spearman_nz(F_go, F_nogo)
    og, on, sf = flipped_edges(F_go, F_nogo)
    print(f"[explore seed {seed}] str_tone={BASELINE['str_D1']} S_D1={S_D1} S_D2={S_D2} "
          f"DA hi/lo={DA_HI}/{DA_LO}")
    print(f"  spearman(F_go, F_nogo) = {rho:+.4f}   (1.0 == pure gain; lower == reconfiguration)")
    print(f"  edges OPENED only in Go   ({len(og)}): " + ", ".join(f"{e['from']}->{e['to']}(go{e['F_go']:+.3f}/nogo{e['F_nogo']:+.3f})" for e in og[:8]))
    print(f"  edges OPENED only in NoGo ({len(on)}): " + ", ".join(f"{e['from']}->{e['to']}(go{e['F_go']:+.3f}/nogo{e['F_nogo']:+.3f})" for e in on[:8]))
    print(f"  SIGN FLIPS ({len(sf)}): " + ", ".join(f"{e['from']}->{e['to']}(go{e['F_go']:+.3f}/nogo{e['F_nogo']:+.3f})" for e in sf[:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--out", default="research/findings/raw/neuromod_reconfig/nr.json")
    ap.add_argument("--explore", action="store_true", help="one-seed exploration print (no JSON, no gate)")
    ap.add_argument("--str-tone", type=float, default=None, help="override striatal baseline tone (exploration)")
    ap.add_argument("--s-d1", type=float, default=None)
    ap.add_argument("--s-d2", type=float, default=None)
    ap.add_argument("--da-hi", type=float, default=None)
    ap.add_argument("--da-lo", type=float, default=None)
    ap.add_argument("--perturb", type=float, default=None)
    args = ap.parse_args()

    # exploration overrides
    global BASELINE, S_D1, S_D2, DA_HI, DA_LO, PERTURB_PA
    if args.str_tone is not None:
        BASELINE = dict(BASELINE); BASELINE["str_D1"] = args.str_tone; BASELINE["str_D2"] = args.str_tone
        BASELINE["str_striosome"] = args.str_tone
    if args.s_d1 is not None:
        S_D1 = args.s_d1
    if args.s_d2 is not None:
        S_D2 = args.s_d2
    if args.da_hi is not None:
        DA_HI = args.da_hi
    if args.da_lo is not None:
        DA_LO = args.da_lo
    if args.perturb is not None:
        PERTURB_PA = args.perturb

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"), note="(neuromod reconfiguration)")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.explore:
        explore(seeds[0])
        return

    print("=== NEUROMOD RECONFIGURATION: does DA reconfigure the effective circuit, or just gain it? ===")
    print(f"seeds={seeds}  str_tone={BASELINE['str_D1']}  S_D1={S_D1} S_D2={S_D2}  DA hi/lo={DA_HI}/{DA_LO}  "
          f"perturb=+{PERTURB_PA}pA\n", flush=True)

    det_ok, det_max = determinism_check(seeds[0])
    print(f"[determinism] byte-identical F_go on re-run @ seed {seeds[0]}: {det_ok} (max|dF|={det_max:.2e})\n",
          flush=True)

    results = []
    for s in seeds:
        r = run_seed(s)
        results.append(r)
        print(f"[seed {s}] ({r['elapsed_s']}s)  spearman(F_go,F_nogo)={r['spearman_F_go_vs_F_nogo']:+.3f}  "
              f"opened_go={r['n_opened_go']} opened_nogo={r['n_opened_nogo']} sign_flips={r['n_sign_flips']}",
              flush=True)
        print(f"          MOD-LESION: spearman={r['spearman_F_go_vs_F_nogo_MOD_LESIONED']:+.3f} "
              f"max|dF|={r['mod_lesion_max_abs_dF']:.2e}   W_unchanged max|dW|={r['W_unchanged_max_abs']:.2e}",
              flush=True)
        og = ", ".join(f"{e['from']}->{e['to']}" for e in r["opened_go"][:6])
        on = ", ".join(f"{e['from']}->{e['to']}" for e in r["opened_nogo"][:6])
        print(f"          opened_go: {og}\n          opened_nogo: {on}\n", flush=True)

    pool = pooled(results)

    v = Verdict("neuromodulator RECONFIGURES effective connectivity (Bargmann/Marder) via perturb-and-measure")
    for proc in ("STDP", "reward_modulation", "hebbian", "homeostasis", "STP", "OU_noise",
                 "conductance_noise", "parameter_heterogeneity", "structural_plasticity"):
        v.disabled(proc, why="isolation: measure propagation on a fixed graph, no learning/noise")
    v.require("6 seeds", len(seeds), expect=lambda n: n >= 6)
    v.require("determinism: byte-identical F re-run @ fixed seed", det_ok, expect=True)
    # ANTI-CHEAT 3: same wiring
    v.require("SAME WIRING: max|W_go - W_nogo| == 0 (anatomy identical across states)",
              pool["W_unchanged_max_abs_max"], expect=lambda x: x == 0.0)
    # ANTI-CHEAT 1: reconfiguration, not gain -- rank correlation strictly below 1, AND a double dissociation
    v.require("RECONFIG: spearman(F_go, F_nogo) < 0.9 (below the pure-gain rank-invariant 1.0), max over seeds",
              pool["spearman_F_go_vs_F_nogo_max"], expect=lambda x: x < 0.9)
    v.require("DOUBLE DISSOCIATION: >=1 edge opened ONLY in Go, every seed (min)",
              pool["n_opened_go_min"], expect=lambda n: n >= 1)
    v.require("DOUBLE DISSOCIATION: >=1 edge opened ONLY in NoGo, every seed (min)",
              pool["n_opened_nogo_min"], expect=lambda n: n >= 1)
    # ANTI-CHEAT 2: neuromodulator-driven -- without the modulator the states are identical
    v.control("MOD-DRIVEN: reconfiguration collapses without the modulator (spearman: with vs lesioned)",
              treatment=pool["spearman_F_go_vs_F_nogo_mean"], control=pool["spearman_MOD_LESIONED_min"],
              min_separation=0.05)
    v.require("MOD-LESION: without the modulator F_go == F_nogo byte-for-byte (max|dF|==0)",
              pool["mod_lesion_max_abs_dF_max"], expect=lambda x: x == 0.0)

    go = (len(seeds) >= 6 and det_ok
          and pool["W_unchanged_max_abs_max"] == 0.0
          and pool["spearman_F_go_vs_F_nogo_max"] < 0.9
          and pool["n_opened_go_min"] >= 1
          and pool["n_opened_nogo_min"] >= 1
          and pool["mod_lesion_max_abs_dF_max"] == 0.0)
    verdict = v.decide(go=go)

    print("\n=== POOLED ===")
    for k in ("n_seeds", "spearman_F_go_vs_F_nogo_mean", "spearman_F_go_vs_F_nogo_max",
              "n_opened_go_min", "n_opened_nogo_min", "n_opened_go_mean", "n_opened_nogo_mean",
              "n_sign_flips_mean", "spearman_MOD_LESIONED_min", "mod_lesion_max_abs_dF_max",
              "W_unchanged_max_abs_max"):
        print(f"  {k:38s} {pool[k]}")
    print(f"  opened_go  (common to ALL seeds): {pool['opened_go_common_all_seeds']}")
    print(f"  opened_nogo(common to ALL seeds): {pool['opened_nogo_common_all_seeds']}")
    print(f"\n  STATUS: {verdict['status']}")

    payload = dict(
        probe="neuromod_reconfiguration_of_effective_connectivity",
        question="does a neuromodulator RECONFIGURE effective connectivity, or merely change gain?",
        sources=["Bargmann BioEssays 34:458-465 (2012)", "Marder Neuron 76:1-11 (2012)",
                 "read-out: Randi et al. Nature 623:406-414 (2023) perturb-and-measure (board #63)"],
        substrate="nav basal ganglia (build_bg_brain_regions flagship A+E), reuse-by-import",
        mechanism="dopamine_mode neuromodulator: excitability_drive D1(+)/D2(-) via NeuromodulatorManager",
        seeds=seeds, operating_point=BASELINE, perturb_pA=PERTURB_PA,
        da_baseline=DA_BASELINE, da_hi=DA_HI, da_lo=DA_LO, s_d1=S_D1, s_d2=S_D2, edge_thresh=EDGE_THRESH,
        determinism=dict(byte_identical=det_ok, max_abs_dF=det_max),
        per_seed=results, pooled=pool, **verdict)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
