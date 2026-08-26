"""Does a SPIKING dopamine NUCLEUS decide the brain's mode from reward/context — self-driven?

THE QUESTION (board task #76, closing the honest limit of board #64). Board #64 showed that ONE
dopaminergic modulator RECONFIGURES the effective circuit on fixed wiring (high DA -> DIRECT/Go pathway,
low DA -> INDIRECT/NoGo pathway; `2026-08-19-neuromod-reconfiguration-GO.md`). Its honest limit: the DA
LEVEL was HOST-SET (a manual concentration = the state). This runner removes that host knob. The dopamine
concentration is now SET BY A SPIKING DA NUCLEUS (the substrate's own `snc` population, IZH2007_DOPAMINE),
whose firing rate the neuromodulator bus reads each step; the nucleus's firing is itself driven by a
reward/context afferent. So the whole loop is:

    reward/context afferent  ->  spiking SNc nucleus fires  ->  DA concentration (bus)  ->  D1(+)/D2(-)
    excitability  ->  DIRECT/INDIRECT effective-circuit reconfiguration          (NO host DA knob)

MECHANISM (all-spiking / all-synaptic; on the neuromodulator bus, reuse-by-import; NO sim/ edit).
  * The DA nucleus is the substrate's `snc` region (Izhikevich DOPAMINE neurons). It has NO outgoing
    synapses in this substrate (its afferents are gpi->snc, str_striosome->snc), so it is a genuine
    READ-OUT nucleus: its firing rate is what the bus transduces into the tonic DA level -- there is no
    synaptic short-circuit from snc to D1/D2.
  * ONE modulator on the subsystem's OWN bus (`sim.neuromodulators.NeuromodulatorManager`),
    `dopamine_mode`, carries BOTH halves:
      - production_rule `from_region_firing_signed` on source_regions=["snc"]: reads the SNc mean firing
        fraction each step (EMA over window_ms), threshold at the neutral/tonic SNc rate, and drives the
        concentration ABOVE baseline when SNc bursts (rate>threshold) / BELOW baseline when SNc is sub-tonic
        (rate<threshold). This is the Schultz-1998 SIGNED dopamine code, already in the bus for the
        spiking-SNc actor-critic (`from_region_firing_signed`, docs/plans/2026-06-08-spiking-snc-...).
      - targets `excitability_drive` scope="group:str_D1" (sensitivity>0, D1R Gs excitatory) and
        scope="group:str_D2" (sensitivity<0, D2R Gi inhibitory) -- the EXACT #64 reconfiguration target.
    The concentration is NEVER set by the runner (grep: no set_concentration for dopamine_mode). It is
    produced entirely by SNc firing through the bus's own `step()`.
  * REWARD/CONTEXT afferent: appetitive context = a strong depolarising current to the SNc nucleus (SNc
    bursts -> DA rises -> Go); aversive context = a weak/zero current to the SNc nucleus (SNc sub-tonic
    -> DA falls -> NoGo). This afferent is the environment's reward/context signal reaching the DA nucleus
    (the honest residual: the reward/context SCALAR is still environmental -- computing it from the brain's
    own sensory stream is a SEPARATE faculty; what is closed HERE is that the DA LEVEL is now brain-derived,
    set by SNc spikes, not a host concentration knob).

READ-OUT (board #63 perturb-and-measure, reuse-by-import of `_perturb_and_measure_derisk`). For each
context we run the live loop (SNc drives the DA level every step) and, for each region TYPE A, perturb A
(+PERTURB pA), settle, record the signed downstream Delta firing-rate of every other TYPE B -> a state-
specific FUNCTIONAL matrix F. The DA level used is whatever the SNc nucleus produced; the loop is never
frozen.

ANTI-CHEATS (they ARE the result):
  1. SELF-DRIVEN SWITCH. Changing ONLY the reward/context afferent drives the spiking SNc nucleus, which
     drives the mode: F_appetitive vs F_aversive reconfigures (spearman well below 1, matching the #64
     signature -- DIRECT edges open in appetitive/Go, INDIRECT edges open in aversive/NoGo). No host
     concentration knob is in the loop (grep-verify: this file calls no set_concentration).
  2. DA-NUCLEUS IS LOAD-BEARING (lesion dissociation). SILENCE the spiking SNc nucleus (clamp its input to a
     fixed sub-firing current, context-independent). The reward/context change can no longer reach the DA
     level, so it no longer switches the mode: F_appetitive == F_aversive byte-for-byte (max|dF|==0). The
     mode switch REQUIRES the nucleus.
  3. SAME WIRING + SAME #64 SIGNATURE. Anatomy identical across contexts: max|W_app - W_ave|==0 (pathways
     never touched). The direct/indirect double dissociation reproduces the #64 opened-edge signature.
     Operating point keeps the MSNs near threshold (str tone 40, the #64 reconfiguring regime; at tone 250
     DA degrades to gain-only -- reported, not hidden).
  4. 6 seeds (42,43,44,100,101,102), per-seed + pooled, deterministic (cfg.seed), byte-identical F re-run.

  SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -m research.runners._neuromod_spiking_da_mode_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/neuromod_spiking_da/nsd.json
  # exploration (one seed, tune the SNc->DA operating point):
  #   ... --explore --seeds 42 --app-snc 800 --ave-snc 0 --da-threshold 0.07 --da-sens 70
"""
import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

import research.runners._perturb_and_measure_derisk as PM
import research.runners._neuromod_reconfiguration_derisk as NR
from research.runners._perturb_and_measure_derisk import TYPES
from tools.lab import assert_backend, Verdict, attributable_to
from sim.neuromodulators import (NeuromodulatorConfig, ModulatorTarget, ProductionRule,
                                 NeuromodulatorManager)
from sim.backend import to_host

# ---- LOCKED operating point (see finding) --------------------------------------------------------------
# Same reconfiguring regime as board #64 (str tone 40 so D1/D2 MSNs sit near threshold and the DA drive
# GATES which pathway a cortical drive recruits). snc is set by the CONTEXT afferent, not this baseline.
BASELINE = dict(cortex=200, str_D1=40, str_D2=40, str_PV_FSI=0, str_striosome=40,
                gpe=150, gpe_arky=120, gpi=900, stn=220, snc=150, thal=550, motor=150)
PERTURB_PA = 400.0          # cortical/region perturbation (== #64: recruitment set by DA excitability)
EDGE_THRESH = 0.008         # |Delta rate| above which a functional edge is "active" (== #63/#64)

# ---- reward/context afferent to the SNc nucleus + the SNc->DA transduction ----------------------------
APP_SNC = 800.0             # appetitive-context afferent current to snc (pA): SNc bursts  -> DA up  -> Go
AVE_SNC = 0.0               # aversive-context  afferent current to snc (pA): SNc sub-tonic -> DA down -> NoGo
SNC_SILENCE_CLAMP = -500.0  # anti-cheat 2: lesion = clamp snc input context-independently (nucleus silenced)

DA_BASELINE = 0.5           # tonic DA (neutral point of the excitability formula; == #64)
CONC_MAX = 2.0
DA_THRESHOLD = 0.07         # from_region_firing_signed threshold = the NEUTRAL/tonic SNc firing fraction:
                            # SNc rate above it raises DA (Go), below it lowers DA (NoGo).
DA_SENS = 70.0              # production sensitivity (SNc rate deviation -> concentration); tuned so the
                            # SNc-established level spans ~ the #64 (conc-baseline)=+-0.5 reconfiguring band.
DA_TAU = 100.0             # DA decay tau (ms): tonic-ish level, slow vs the perturbation transient.
DA_WINDOW = 50.0           # SNc rate EMA tau (ms): fast enough to track the tonic rate within warmup.
S_D1 = 1000.0              # pA excitability per unit (conc-baseline) on D1 (D1R Gs, +) -- == #64
S_D2 = 1000.0              # pA excitability per unit (conc-baseline) on D2 (D2R Gi, -> NEGATIVE below) == #64

WARMUP = 200               # long enough for the SNc-driven concentration to reach its context equilibrium
SETTLE = 120               # == #63/#64 firing-rate accumulation window


def da_nucleus_config():
    """ONE `dopamine_mode` modulator: SNc firing DRIVES the concentration (from_region_firing_signed on
    ["snc"]); the concentration DRIVES D1(+)/D2(-) excitability. No manual concentration anywhere."""
    return NeuromodulatorConfig(
        name="dopamine_mode", baseline=DA_BASELINE, decay_tau_ms=DA_TAU,
        concentration_min=0.0, concentration_max=CONC_MAX,
        targets=[
            ModulatorTarget(target_type="excitability_drive", scope="group:str_D1", sensitivity=+float(S_D1)),
            ModulatorTarget(target_type="excitability_drive", scope="group:str_D2", sensitivity=-float(S_D2)),
        ],
        production_rules=[
            ProductionRule(rule_type="from_region_firing_signed", sensitivity=float(DA_SENS),
                           threshold=float(DA_THRESHOLD), window_ms=float(DA_WINDOW),
                           source_regions=["snc"]),
        ],
    )


def _bridge_xp(sb):
    """The array module SB's OWN `cp_*` state actually lives on -- NOT the process-global `SIM_BACKEND` /
    `get_backend()` (a sticky cache; `sim.bridge` binds its module-level `cp` at IMPORT time, so a substrate
    built under `SIM_BACKEND=cupy` (the production `/api/brain-chat` path) has cupy `cp_*` arrays regardless
    of what `get_backend()` reports later). Deriving `xp` from the substrate's own array keeps every array this
    module allocates on the SAME device as `sb`, so `sb.cp_external_input_current[:] = <this-module's array>`
    never mixes host numpy into a device cupy fill. Same pattern + same root cause as the 2026-06-24
    `brain_conversational_agent._bridge_xp` webapp fix; this module hardcoded `np` instead (2026-08-25 root
    cause of `da_drives.reason = "error:ValueError: non-scalar numpy.ndarray cannot be used for fill"` --
    board #76's `make_manager`/`_base_current`/`measure_self_driven` are NUMPY-only, so on the cupy webapp
    every DA-mode read silently threw and was swallowed by `da_mode_drives_chat.observe`'s bare `except`).
    Falls back to numpy when cupy is not installed (a numpy-only box can only ever have a numpy substrate)."""
    try:
        import cupy as _cp  # noqa: PLC0415
        return _cp.get_array_module(sb.cp_external_input_current)
    except Exception:
        return np


def make_manager(sb):
    xp = _bridge_xp(sb)                                      # device-correct: matches SB's own cp_* arrays
    mgr = NeuromodulatorManager([da_nucleus_config()], dt_ms=1.0)
    mgr.initialize(sb.core_config.num_neurons, xp)           # numpy on a numpy substrate, cupy on a cupy one
    mgr.set_group_indices(NR._group_indices(sb))             # group:str_D1 / group:str_D2 for the targets
    return mgr


def _base_current(sb, nbt, baseline, context_snc, perturb, silence_snc):
    """STATIC input template: baseline tones + the reward/context afferent to snc (or the lesion clamp) +
    perturbation. Only the neuromodulatory drive changes step-to-step, so this is built ONCE per measure.
    Built on SB's own array module (`_bridge_xp`) so the result can be assigned straight into
    `sb.cp_external_input_current` on either backend -- byte-identical to the old numpy-only version whenever
    SB itself is numpy-backed (the board-#76 GO path, `SIM_BACKEND=numpy`)."""
    xp = _bridge_xp(sb)
    idx = lambda n: xp.asarray(sb.region_manager.indices(n))
    v = xp.zeros(sb.core_config.num_neurons, dtype=xp.float64)
    for t, cur in baseline.items():
        for n in nbt[t]:
            v[idx(n)] = cur
    snc_val = SNC_SILENCE_CLAMP if silence_snc else context_snc      # anti-cheat 2: context-independent
    for n in nbt["snc"]:
        v[idx(n)] = snc_val
    if perturb:
        for n, extra in perturb.items():
            v[idx(n)] += extra
    return v


def measure_self_driven(sb, mgr, nbt, baseline, context_snc, perturb=None,
                        warmup=WARMUP, settle=SETTLE, silence_snc=False):
    """LIVE loop: each step add the SNc-produced DA excitability drive to the static base current, step the
    substrate, then let the bus read SNc firing to update the DA concentration. Returns per-region rates +
    the established DA concentration + the SNc firing fraction (means over the settle window).

    `base` and `drive` are kept on SB's OWN array module (`_bridge_xp`) right up to the assignment into
    `sb.cp_external_input_current[:]` -- on a cupy substrate `mgr` is now also cupy-backed (`make_manager`
    fix), so `drive` is already a cupy array; casting through `np.asarray` here (the old code) would either
    force an illegal device->host->device round trip or, for `base` alone, assign a host numpy array into a
    device cupy slice -- the exact `ValueError: non-scalar numpy.ndarray cannot be used for fill` this fixes.
    `rates`/`snc_firing` stay host-numpy via `to_host` below, unaffected (that half was already correct)."""
    xp = _bridge_xp(sb)
    base = _base_current(sb, nbt, baseline, context_snc, perturb, silence_snc)
    snc_idx = np.asarray(sb.region_manager.indices("snc"))
    idx = lambda n: np.asarray(sb.region_manager.indices(n))
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    conc_acc = 0.0
    snc_acc = 0.0
    for step_i in range(warmup + settle):
        drive = mgr.compute_excitability_drive_per_neuron()          # subsystem's OWN conc->pA mapping
        if drive is None:
            sb.cp_external_input_current[:] = base
        else:
            sb.cp_external_input_current[:] = base + xp.asarray(drive, dtype=xp.float64)
        sb._run_one_simulation_step()
        mgr.step(sb)                                                 # SNc firing -> DA concentration (bus)
        if step_i >= warmup:
            fs = to_host(sb.cp_firing_states).astype(np.float64)
            acc += fs
            conc_acc += mgr.get_concentration("dopamine_mode")
            snc_acc += float(fs[snc_idx].mean())
    rates = {}
    for t in TYPES:
        for n in nbt[t]:
            rates[n] = float(acc[idx(n)].mean() / settle)
    return rates, conc_acc / settle, snc_acc / settle


def functional_matrix_context(seed, context_snc, baseline=None, perturb_pa=None, silence_snc=False):
    """F[A][B] under one reward/context: perturb every type A, measure Delta type-rate of every B vs the
    unperturbed baseline. The DA level is set by the spiking SNc nucleus every step (self-driven)."""
    baseline = BASELINE if baseline is None else baseline
    perturb_pa = PERTURB_PA if perturb_pa is None else perturb_pa
    sb0, regions, pathways = PM.build(seed)
    nbt = PM.names_by_type(regions)
    r0_rates, conc0, snc0 = measure_self_driven(sb0, make_manager(sb0), nbt, baseline, context_snc,
                                                perturb=None, silence_snc=silence_snc)
    r0 = PM.type_rates(r0_rates, nbt)
    F = np.zeros((len(TYPES), len(TYPES)), dtype=float)
    concs, sncs = [conc0], [snc0]
    for i, a in enumerate(TYPES):
        sbi, ri, _ = PM.build(seed)
        nbti = PM.names_by_type(ri)
        pert = {n: perturb_pa for n in nbti[a]}
        rp_rates, conci, snci = measure_self_driven(sbi, make_manager(sbi), nbti, baseline, context_snc,
                                                    perturb=pert, silence_snc=silence_snc)
        rp = PM.type_rates(rp_rates, nbti)
        for j, b in enumerate(TYPES):
            F[i, j] = rp[b] - r0[b]
        concs.append(conci); sncs.append(snci)
    return F, r0, float(np.mean(concs)), float(np.mean(sncs)), regions, pathways


SNC_I = TYPES.index("snc")


def spearman_excl_snc(F_app, F_ave):
    """ROBUSTNESS: rank correlation with the snc ROW+COL removed, so no snc->x / x->snc edge (the nucleus
    self-perturbation) can contribute. Isolates the genuine D1/D2 direct/indirect reconfiguration."""
    keep = [i for i in range(len(TYPES)) if i != SNC_I]
    A = np.asarray(F_app)[np.ix_(keep, keep)]
    B = np.asarray(F_ave)[np.ix_(keep, keep)]
    return NR.spearman_nz(A, B)


def flipped_nonsnc(F_app, F_ave, thresh=EDGE_THRESH):
    """opened edges EXCLUDING any snc->x or x->snc edge -> the genuine (non-nucleus) double dissociation."""
    og, on = [], []
    for i, a in enumerate(TYPES):
        for j, b in enumerate(TYPES):
            if i == j or a == "snc" or b == "snc":
                continue
            fh = float(F_app[i][j]); fl = float(F_ave[i][j])
            ah = abs(fh) > thresh; al = abs(fl) > thresh
            if ah and not al:
                og.append(f"{a}->{b}")
            elif al and not ah:
                on.append(f"{a}->{b}")
    return og, on


def run_seed(seed):
    t0 = time.time()
    # --- self-driven: context sets SNc firing sets DA level sets the mode ---
    F_app, r0_app, conc_app, snc_app, regions, pathways = functional_matrix_context(seed, APP_SNC)
    F_ave, r0_ave, conc_ave, snc_ave, _, _ = functional_matrix_context(seed, AVE_SNC)

    W_app = PM.anatomical_matrix(regions, pathways)
    _, reg2, pth2 = PM.build(seed)                       # independent rebuild: context is input-only, W is untouched
    W_ave = PM.anatomical_matrix(reg2, pth2)
    w_unchanged_max = float(np.max(np.abs(W_app - W_ave)))

    rho = NR.spearman_nz(F_app, F_ave)
    opened_app, opened_ave, sign_flips = NR.flipped_edges(F_app, F_ave)   # app==Go, ave==NoGo
    rho_excl_snc = spearman_excl_snc(F_app, F_ave)                        # robustness (no nucleus self-perturb)
    opened_go_ns, opened_nogo_ns = flipped_nonsnc(F_app, F_ave)

    # --- ANTI-CHEAT 2: silence the SNc nucleus -> context can't reach the DA level -> no switch ---
    F_app_les, _, conc_app_les, snc_app_les, _, _ = functional_matrix_context(seed, APP_SNC, silence_snc=True)
    F_ave_les, _, conc_ave_les, snc_ave_les, _, _ = functional_matrix_context(seed, AVE_SNC, silence_snc=True)
    rho_lesion = NR.spearman_nz(F_app_les, F_ave_les)
    lesion_max_dF = float(np.max(np.abs(F_app_les - F_ave_les)))

    # ATTRIBUTION: reconfiguration magnitude (1 - spearman) attributable to the spiking DA nucleus.
    reconfig_attribution = attributable_to(
        f"seed {seed}: reconfiguration (1 - spearman) attributable to the spiking SNc DA nucleus",
        treatment_value=1.0 - rho, control_value=1.0 - rho_lesion)

    elapsed = time.time() - t0
    return dict(
        seed=int(seed), elapsed_s=round(elapsed, 2), types=TYPES,
        F_appetitive=F_app.tolist(), F_aversive=F_ave.tolist(), W=W_app.tolist(),
        # self-driven DA level (produced by the SNc nucleus, NOT set by the runner)
        da_conc_appetitive=conc_app, da_conc_aversive=conc_ave,
        snc_firing_appetitive=snc_app, snc_firing_aversive=snc_ave,
        # reconfiguration signature
        spearman_F_app_vs_F_ave=rho,
        n_opened_go=len(opened_app), n_opened_nogo=len(opened_ave), n_sign_flips=len(sign_flips),
        opened_go=opened_app, opened_nogo=opened_ave, sign_flips=sign_flips,
        # robustness: reconfiguration with the snc nucleus self-perturbation removed
        spearman_F_app_vs_F_ave_excl_snc=rho_excl_snc,
        opened_go_nonsnc=opened_go_ns, opened_nogo_nonsnc=opened_nogo_ns,
        # anti-cheat 2 (DA-nucleus lesion dissociation)
        spearman_F_app_vs_F_ave_SNC_SILENCED=rho_lesion,
        snc_lesion_max_abs_dF=lesion_max_dF,
        da_conc_appetitive_SNC_SILENCED=conc_app_les, da_conc_aversive_SNC_SILENCED=conc_ave_les,
        snc_firing_appetitive_SNC_SILENCED=snc_app_les, snc_firing_aversive_SNC_SILENCED=snc_ave_les,
        reconfig_attributable_to_da_nucleus=reconfig_attribution,
        # anti-cheat 3 (same wiring)
        W_unchanged_max_abs=w_unchanged_max,
        baseline_rates_appetitive=r0_app, baseline_rates_aversive=r0_ave,
    )


def pooled(results):
    def col(key):
        return np.array([r[key] for r in results], dtype=float)
    out = dict(n_seeds=len(results))
    out["spearman_F_app_vs_F_ave_mean"] = float(np.nanmean(col("spearman_F_app_vs_F_ave")))
    out["spearman_F_app_vs_F_ave_max"] = float(np.nanmax(col("spearman_F_app_vs_F_ave")))
    out["spearman_excl_snc_mean"] = float(np.nanmean(col("spearman_F_app_vs_F_ave_excl_snc")))
    out["spearman_excl_snc_max"] = float(np.nanmax(col("spearman_F_app_vs_F_ave_excl_snc")))
    out["n_opened_go_min"] = int(np.min(col("n_opened_go")))
    out["n_opened_nogo_min"] = int(np.min(col("n_opened_nogo")))
    out["n_opened_go_mean"] = float(np.mean(col("n_opened_go")))
    out["n_opened_nogo_mean"] = float(np.mean(col("n_opened_nogo")))
    out["n_sign_flips_mean"] = float(np.mean(col("n_sign_flips")))
    out["spearman_SNC_SILENCED_min"] = float(np.nanmin(col("spearman_F_app_vs_F_ave_SNC_SILENCED")))
    out["snc_lesion_max_abs_dF_max"] = float(np.max(col("snc_lesion_max_abs_dF")))
    out["W_unchanged_max_abs_max"] = float(np.max(col("W_unchanged_max_abs")))
    # self-driven DA level (SNc-produced): appetitive must be ABOVE aversive on every seed
    out["da_conc_appetitive_mean"] = float(np.mean(col("da_conc_appetitive")))
    out["da_conc_aversive_mean"] = float(np.mean(col("da_conc_aversive")))
    out["da_conc_gap_min"] = float(np.min(col("da_conc_appetitive") - col("da_conc_aversive")))
    out["snc_firing_appetitive_mean"] = float(np.mean(col("snc_firing_appetitive")))
    out["snc_firing_aversive_mean"] = float(np.mean(col("snc_firing_aversive")))
    out["snc_firing_gap_min"] = float(np.min(col("snc_firing_appetitive") - col("snc_firing_aversive")))

    def edge_set(res, key):
        return set((e["from"], e["to"]) for e in res[key])
    go_common = set.intersection(*[edge_set(r, "opened_go") for r in results]) if results else set()
    nogo_common = set.intersection(*[edge_set(r, "opened_nogo") for r in results]) if results else set()
    out["opened_go_common_all_seeds"] = sorted(f"{a}->{b}" for a, b in go_common)
    out["opened_nogo_common_all_seeds"] = sorted(f"{a}->{b}" for a, b in nogo_common)
    go_ns = set.intersection(*[set(r["opened_go_nonsnc"]) for r in results]) if results else set()
    no_ns = set.intersection(*[set(r["opened_nogo_nonsnc"]) for r in results]) if results else set()
    out["opened_go_nonsnc_common_all_seeds"] = sorted(go_ns)
    out["opened_nogo_nonsnc_common_all_seeds"] = sorted(no_ns)
    return out


def determinism_check(seed=42):
    F1, _, _, _, _, _ = functional_matrix_context(seed, APP_SNC)
    F2, _, _, _, _, _ = functional_matrix_context(seed, APP_SNC)
    return bool(np.array_equal(F1, F2)), float(np.max(np.abs(F1 - F2)))


def explore(seed):
    F_app, _, conc_app, snc_app, _, _ = functional_matrix_context(seed, APP_SNC)
    F_ave, _, conc_ave, snc_ave, _, _ = functional_matrix_context(seed, AVE_SNC)
    rho = NR.spearman_nz(F_app, F_ave)
    og, on, sf = NR.flipped_edges(F_app, F_ave)
    print(f"[explore seed {seed}] str_tone={BASELINE['str_D1']} app_snc={APP_SNC} ave_snc={AVE_SNC} "
          f"thr={DA_THRESHOLD} sens={DA_SENS} tau={DA_TAU} win={DA_WINDOW} S={S_D1}")
    print(f"  SELF-DRIVEN DA level: appetitive conc={conc_app:.3f} (snc fire={snc_app:.4f})  "
          f"aversive conc={conc_ave:.3f} (snc fire={snc_ave:.4f})   gap={conc_app-conc_ave:+.3f}")
    print(f"  spearman(F_app, F_ave) = {rho:+.4f}   (1.0 == same pattern; lower == reconfiguration)")
    print(f"  edges OPENED only in appetitive/Go   ({len(og)}): " +
          ", ".join(f"{e['from']}->{e['to']}" for e in og[:10]))
    print(f"  edges OPENED only in aversive/NoGo   ({len(on)}): " +
          ", ".join(f"{e['from']}->{e['to']}" for e in on[:10]))
    print(f"  SIGN FLIPS ({len(sf)}): " + ", ".join(f"{e['from']}->{e['to']}" for e in sf[:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--out", default="research/findings/raw/neuromod_spiking_da/nsd.json")
    ap.add_argument("--explore", action="store_true", help="one-seed exploration print (no JSON, no gate)")
    ap.add_argument("--app-snc", type=float, default=None)
    ap.add_argument("--ave-snc", type=float, default=None)
    ap.add_argument("--da-threshold", type=float, default=None)
    ap.add_argument("--da-sens", type=float, default=None)
    ap.add_argument("--da-tau", type=float, default=None)
    ap.add_argument("--da-window", type=float, default=None)
    ap.add_argument("--str-tone", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    args = ap.parse_args()

    global APP_SNC, AVE_SNC, DA_THRESHOLD, DA_SENS, DA_TAU, DA_WINDOW, BASELINE, WARMUP
    if args.app_snc is not None: APP_SNC = args.app_snc
    if args.ave_snc is not None: AVE_SNC = args.ave_snc
    if args.da_threshold is not None: DA_THRESHOLD = args.da_threshold
    if args.da_sens is not None: DA_SENS = args.da_sens
    if args.da_tau is not None: DA_TAU = args.da_tau
    if args.da_window is not None: DA_WINDOW = args.da_window
    if args.warmup is not None: WARMUP = args.warmup
    if args.str_tone is not None:
        BASELINE = dict(BASELINE); BASELINE["str_D1"] = args.str_tone; BASELINE["str_D2"] = args.str_tone
        BASELINE["str_striosome"] = args.str_tone

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"), note="(neuromod spiking DA mode)")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.explore:
        explore(seeds[0])
        return

    print("=== SPIKING DA NUCLEUS decides the mode: reward/context -> SNc spikes -> DA level -> reconfig ===")
    print(f"seeds={seeds}  str_tone={BASELINE['str_D1']}  app_snc={APP_SNC} ave_snc={AVE_SNC}  "
          f"thr={DA_THRESHOLD} sens={DA_SENS} tau={DA_TAU} win={DA_WINDOW}  S={S_D1}  "
          f"perturb=+{PERTURB_PA}pA warmup={WARMUP} settle={SETTLE}\n", flush=True)

    det_ok, det_max = determinism_check(seeds[0])
    print(f"[determinism] byte-identical F_app on re-run @ seed {seeds[0]}: {det_ok} (max|dF|={det_max:.2e})\n",
          flush=True)

    results = []
    for s in seeds:
        r = run_seed(s)
        results.append(r)
        print(f"[seed {s}] ({r['elapsed_s']}s)  DA level app={r['da_conc_appetitive']:.3f}"
              f"/ave={r['da_conc_aversive']:.3f} (snc {r['snc_firing_appetitive']:.4f}/"
              f"{r['snc_firing_aversive']:.4f})", flush=True)
        print(f"          spearman(F_app,F_ave)={r['spearman_F_app_vs_F_ave']:+.3f}  "
              f"opened_go={r['n_opened_go']} opened_nogo={r['n_opened_nogo']} sign_flips={r['n_sign_flips']}",
              flush=True)
        print(f"          SNC-SILENCED: spearman={r['spearman_F_app_vs_F_ave_SNC_SILENCED']:+.3f} "
              f"max|dF|={r['snc_lesion_max_abs_dF']:.2e}  (DA level app={r['da_conc_appetitive_SNC_SILENCED']:.3f}"
              f"/ave={r['da_conc_aversive_SNC_SILENCED']:.3f})   W_unchanged max|dW|={r['W_unchanged_max_abs']:.2e}",
              flush=True)
        og = ", ".join(f"{e['from']}->{e['to']}" for e in r["opened_go"][:6])
        on = ", ".join(f"{e['from']}->{e['to']}" for e in r["opened_nogo"][:6])
        print(f"          opened_go(direct): {og}\n          opened_nogo(indirect): {on}\n", flush=True)

    pool = pooled(results)

    v = Verdict("spiking DA nucleus SELF-DRIVES the Go/NoGo effective-circuit mode from reward/context")
    for proc in ("STDP", "reward_modulation", "hebbian", "homeostasis", "STP", "OU_noise",
                 "conductance_noise", "parameter_heterogeneity", "structural_plasticity"):
        v.disabled(proc, why="isolation: measure propagation on a fixed graph, no learning/noise")
    v.require("6 seeds", len(seeds), expect=lambda n: n >= 6)
    v.require("determinism: byte-identical F re-run @ fixed seed", det_ok, expect=True)
    # SELF-DRIVEN LEVEL: the SNc nucleus produces a HIGHER DA level in appetitive than aversive, every seed
    v.require("SELF-DRIVEN: SNc fires more in appetitive than aversive (gap>0, min over seeds)",
              pool["snc_firing_gap_min"], expect=lambda x: x > 0.0)
    v.require("SELF-DRIVEN: DA level appetitive > aversive (gap>0, min over seeds)",
              pool["da_conc_gap_min"], expect=lambda x: x > 0.0)
    # ANTI-CHEAT 3: same wiring
    v.require("SAME WIRING: max|W_app - W_ave| == 0 (anatomy identical across contexts)",
              pool["W_unchanged_max_abs_max"], expect=lambda x: x == 0.0)
    # ANTI-CHEAT 1: reconfiguration, not gain -- rank correlation below 1 AND a double dissociation
    v.require("RECONFIG: spearman(F_app, F_ave) < 0.9 (below the same-pattern 1.0), max over seeds",
              pool["spearman_F_app_vs_F_ave_max"], expect=lambda x: x < 0.9)
    v.require("DOUBLE DISSOCIATION: >=1 edge opened ONLY in appetitive/Go, every seed (min)",
              pool["n_opened_go_min"], expect=lambda n: n >= 1)
    v.require("DOUBLE DISSOCIATION: >=1 edge opened ONLY in aversive/NoGo, every seed (min)",
              pool["n_opened_nogo_min"], expect=lambda n: n >= 1)
    # ROBUSTNESS: the reconfiguration survives removing the snc nucleus self-perturbation (row+col)
    v.require("ROBUST: spearman(F_app,F_ave) EXCL snc row/col still < 0.9 (genuine D1/D2 reconfig), max over seeds",
              pool["spearman_excl_snc_max"], expect=lambda x: x < 0.9)
    # ANTI-CHEAT 2: DA-nucleus-driven -- silencing the SNc nucleus collapses the switch
    v.control("DA-NUCLEUS-DRIVEN: reconfiguration collapses when the SNc nucleus is silenced",
              treatment=pool["spearman_F_app_vs_F_ave_mean"], control=pool["spearman_SNC_SILENCED_min"],
              min_separation=0.05)
    v.require("SNC-LESION: silenced nucleus -> F_app == F_ave byte-for-byte (max|dF|==0)",
              pool["snc_lesion_max_abs_dF_max"], expect=lambda x: x == 0.0)

    go = (len(seeds) >= 6 and det_ok
          and pool["snc_firing_gap_min"] > 0.0
          and pool["da_conc_gap_min"] > 0.0
          and pool["W_unchanged_max_abs_max"] == 0.0
          and pool["spearman_F_app_vs_F_ave_max"] < 0.9
          and pool["spearman_excl_snc_max"] < 0.9
          and pool["n_opened_go_min"] >= 1
          and pool["n_opened_nogo_min"] >= 1
          and pool["snc_lesion_max_abs_dF_max"] == 0.0)
    verdict = v.decide(go=go)

    print("\n=== POOLED ===")
    for k in ("n_seeds", "da_conc_appetitive_mean", "da_conc_aversive_mean", "da_conc_gap_min",
              "snc_firing_appetitive_mean", "snc_firing_aversive_mean", "snc_firing_gap_min",
              "spearman_F_app_vs_F_ave_mean", "spearman_F_app_vs_F_ave_max",
              "spearman_excl_snc_mean", "spearman_excl_snc_max",
              "n_opened_go_min", "n_opened_nogo_min", "n_opened_go_mean", "n_opened_nogo_mean",
              "n_sign_flips_mean", "spearman_SNC_SILENCED_min", "snc_lesion_max_abs_dF_max",
              "W_unchanged_max_abs_max"):
        print(f"  {k:38s} {pool[k]}")
    print(f"  opened_go  (common to ALL seeds): {pool['opened_go_common_all_seeds']}")
    print(f"  opened_nogo(common to ALL seeds): {pool['opened_nogo_common_all_seeds']}")
    print(f"  opened_go  non-snc (common): {pool['opened_go_nonsnc_common_all_seeds']}")
    print(f"  opened_nogo non-snc (common): {pool['opened_nogo_nonsnc_common_all_seeds']}")
    print(f"\n  STATUS: {verdict['status']}")

    payload = dict(
        probe="spiking_da_nucleus_self_drives_the_mode",
        question="does a SPIKING dopamine nucleus decide the Go/NoGo effective-circuit mode from reward/context?",
        sources=["Bargmann BioEssays 34:458-465 (2012)", "Marder Neuron 76:1-11 (2012)",
                 "Schultz Science 275:1593-1599 (1997) / J Neurophysiol 80 (1998) signed DA RPE",
                 "read-out: Randi et al. Nature 623:406-414 (2023) perturb-and-measure (board #63)",
                 "base capability: board #64 neuromod reconfiguration GO"],
        substrate="nav basal ganglia (build_bg_brain_regions flagship A+E); DA nucleus = snc (IZH2007_DOPAMINE)",
        mechanism="dopamine_mode: from_region_firing_signed on [snc] -> concentration; "
                  "excitability_drive D1(+)/D2(-); reward/context = afferent current to snc. No set_concentration.",
        seeds=seeds, operating_point=BASELINE, perturb_pA=PERTURB_PA,
        app_snc=APP_SNC, ave_snc=AVE_SNC, snc_silence_clamp=SNC_SILENCE_CLAMP,
        da_baseline=DA_BASELINE, da_threshold=DA_THRESHOLD, da_sens=DA_SENS, da_tau=DA_TAU,
        da_window=DA_WINDOW, s_d1=S_D1, s_d2=S_D2, edge_thresh=EDGE_THRESH, warmup=WARMUP, settle=SETTLE,
        determinism=dict(byte_identical=det_ok, max_abs_dF=det_max),
        per_seed=results, pooled=pool, **verdict)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
