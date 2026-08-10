"""Gap #5 — does a SOMATIC slow-NMDA REVERBERATORY recurrent (Wang 2002) give a ROBUST bistable + cue-SPECIFIC
CA3 completion where the point-neuron AMPA recurrent CANNOT (NEW runner, NO sim/ edit)?

Frontier (2026-08-10): `_gap5_ca3_selective_attractor_sweep_derisk.py` established an UPPER BOUND on the point-neuron
recurrent path — even a HAND-INSTALLED PERFECT pattern-selective potentiation on the *fast/AMPA* ca3->ca3 recurrent
FAILS the teeth (as within-assembly weight W rises, completion climbs but the PERMUTED cue overtakes it AND the net
self-ignites at rest). The 2026-07-18 research gate
(`2026-07-18-gap5-bistable-completion-mechanism-research-gate.md`) argues that is the WRONG recurrent ELEMENT: a
dendritic-dAP *coincidence readout* / an instantaneous AMPA drive standing in for Wang's *somatic slow-NMDA
reverberatory excitation*. Slow NMDA (tau_decay~100 ms, Mg2+ self-limiting) is claimed to give (i) a stable LOW (silent)
state that does NOT self-ignite — the horn AMPA cannot hold — and (ii) TEMPORAL INTEGRATION that rejects a transient /
non-specific (permuted) cue. Whether that reaches the GO bar on a DISTRIBUTED sparse assembly is the OPEN frontier — the
retracted 2026-07-18 "Wang seed-42" result only tested a Hebbian-grown WEAK attractor (w~49, frozen-dead), never a
proper reverberatory operating point, and the 2026-08-10 upper bound used the AMPA recurrent, NOT this slow-NMDA mode.

Method (mirrors the clean 2026-08-10 upper-bound de-risk — hand-install to sidestep the Hebbian-collapse confound; the
ONLY variable changed is the recurrent ELEMENT: fast/AMPA -> `exc_receptor="nmda_slow"` reverberatory recurrent, plus a
LONG read window >= 2.5*tau_NMDA so the slow conductance can build):
  1. Build CA3 with a WEAK uniform baseline ca3->ca3 recurrent tagged `exc_receptor="nmda_slow"` + `enable_nmda_recurrent`
     (reuse `_riii_ca3_coincidence_completion_derisk._build(nmda_recurrent=True, coincidence=False, train=False)`), plus
     the shared FS basket (`ca3_fb_inhib`) for the E/I working point. Plasticity FROZEN throughout.
  2. HAND-INSTALL a perfect within-assembly potentiation W on the nmda_slow recurrent synapses (the idealized outcome a
     perfect recurrent LTP would produce). Sweep W.
  3. Recall: HARD-SILENCE (clear v/u/firing + ALL conductances incl. `cp_conductance_g_nmda_recurrent(_rise)` — the D3
     lesson: the tau=100 ms slow conductance does NOT decay away in a short reset, so a latched high state leaks into the
     next condition and fakes a self-sustaining / non-silent read), then drive a condition and read the held-out members'
     SETTLED firing over the LAST half of a long window.

Mandatory anti-cheats / teeth (a completion test that omits these has produced 3 retractions here):
  - CORRECT 50% cue -> held-out members REACTIVATE (held_cue).
  - PERMUTED cue (random NON-assembly cells, same count) -> held-out members do NOT (held_perm; GO needs cue >= 3x perm).
  - NO-CUE / silent rest -> held-out silent (held_nocue <= 0.10; no always-on limit cycle / self-ignition).
  - NO-ENCODING (weak baseline, no install) -> held_cue collapses (the attractor is load-bearing).
  - RECURRENCE-ZERO (zero the ca3->ca3 recurrent) -> held_cue collapses (completion is the reverberation, not cue re-drive).
  - OU noise CONTROLLABLE: report OU-OFF (deterministic bistability) AND OU-ON. If cue ~ perm ~ 0.5 with OU on, that is
    pure noise, NOT completion.
  - `cfg.seed=seed` set explicitly (NOT actual_seed_used); build-twice-hash-`cp_neuron_firing_thresholds` determinism check.

GO bar (6-seed 42/43/44/100/101/102): held_cue >= 0.20 AND held_cue >= 3*held_perm AND held_cue >= 3*held_nocue AND
held_nocue <= 0.10, on all/most seeds. Report per-seed. A PARTIAL (n/6) is an honest, first-class result. SIM_BACKEND=cupy.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
import numpy as np

from research.runners._riii_ca3_coincidence_completion_derisk import _build


def _threshold_hash(bridge):
    from sim.backend import to_host
    arr = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if arr is None:
        return "none"
    return hashlib.sha1(np.asarray(to_host(arr)).tobytes()).hexdigest()[:12]


def _csr_row_col(cp, C):
    n = C.shape[0]
    rows = cp.repeat(cp.arange(n, dtype=cp.int64), cp.diff(C.indptr))
    cols = C.indices.astype(cp.int64)
    return rows, cols


def run_seed(seed, n_ca3=800, ca3_density=0.10, weights=(60, 150, 400, 1000, 2500),
             n_assembly=3, assembly_frac=0.18, cue_frac=0.5, base_w=1.5, ca3_fb_inhib=20.0,
             nmda_tau=100.0, nmda_ratio=1.0, drive_pA=250.0, warm_steps=180, read_steps=170,
             silence_steps=50, enable_ou=False, element="nmda_slow", verbose=True):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    # ---- build: ca3->ca3 recurrent element. element="nmda_slow" = SOMATIC slow-NMDA reverberatory (Wang 2002);
    #      element="ampa" = the DEFAULT fast AMPA(+standard NMDA) recurrent = ATTRIBUTION CONTROL (same FS basket,
    #      same hand-install, same long read -> isolates the recurrent ELEMENT as the load-bearing variable). FROZEN. ----
    _nmda_rec = (element == "nmda_slow")
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=ca3_density, coincidence=False, two_comp=False,
                    nmda_recurrent=_nmda_rec, nmda_tau=nmda_tau, nmda_ratio=nmda_ratio,
                    ca3_fb_inhib=ca3_fb_inhib, train=False, enable_ou=enable_ou)
    cfg = bridge.core_config
    cfg.enable_hebbian_learning = False
    cfg.enable_stdp = False
    cfg.enable_structural_plasticity = False

    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}

    C = bridge.cp_connections
    if not hasattr(C, "indptr"):
        C = C.tocsr(); bridge.cp_connections = C
    rows, cols = _csr_row_col(cp, C)
    baseline_data = C.data.copy()

    n = C.shape[0]
    is_ca3 = cp.zeros(n, dtype=cp.bool_); is_ca3[ca3_arr] = True
    rec_mask = is_ca3[rows] & is_ca3[cols]                 # all ca3->ca3 (excitatory recurrent) synapses
    n_rec = int(to_host(cp.sum(rec_mask)))

    # K disjoint sparse assemblies (random subsets of CA3)
    rng = np.random.default_rng(seed)
    perm_idx = rng.permutation(len(ca3_idx))
    a_size = max(6, int(assembly_frac * len(ca3_idx)))
    assemblies = [ca3_idx[perm_idx[a * a_size:(a + 1) * a_size]] for a in range(n_assembly)]
    withinA_masks = []
    for A in assemblies:
        is_A = cp.zeros(n, dtype=cp.bool_); is_A[cp.asarray(A)] = True
        withinA_masks.append(is_A[rows] & is_A[cols])

    def restore_baseline():
        C.data[:] = baseline_data

    def install_selective(W):
        restore_baseline()
        for m in withinA_masks:
            C.data[m] = cp.float32(W)

    def zero_recurrents():
        restore_baseline()
        C.data[rec_mask] = cp.float32(0.0)

    # ---- hard-silence: clear v/u/firing + ALL conductances (incl. the slow nmda_recurrent which does NOT
    #      decay away in a short reset), then settle silent (a genuine down state for the no-cue/permuted teeth). ----
    def hard_silence():
        if getattr(bridge, "cp_izh_c_reset", None) is not None:
            bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
        else:
            bridge.cp_membrane_potential_v[:] = -65.0
        bridge.cp_recovery_variable_u[:] = 0.0
        if getattr(bridge, "cp_firing_states", None) is not None:
            bridge.cp_firing_states[:] = False
        for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
                   "cp_conductance_g_e", "cp_conductance_g_i",
                   "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
            _arr = getattr(bridge, _a, None)
            if _arr is not None:
                _arr[:] = 0.0
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(silence_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    def drive_read(drive_indices, sustain=False):
        """hard-silence -> drive -> warm (let slow NMDA build) -> accumulate CA3 firing over the LAST window.
        If sustain: after the cue-on read window, RELEASE the cue and read a further window (the high state must
        PERSIST with the cue OFF for a genuine BISTABLE attractor). Returns per-CA3 duty vector; if sustain, a tuple
        (cue_on_vec, cue_off_vec)."""
        hard_silence()
        if drive_indices is not None and len(drive_indices) > 0:
            darr = cp.asarray(np.asarray(drive_indices, dtype=np.int64), dtype=cp.int64)
            bridge.cp_external_input_current[darr] = cp.float32(drive_pA)
        else:
            darr = None
        for _ in range(warm_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        cnt = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for _ in range(read_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            cnt += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        v_on = to_host(cnt) / float(read_steps)
        if not sustain:
            if darr is not None:
                bridge.cp_external_input_current[darr] = 0.0
            return v_on
        # RELEASE the cue; the reverberatory high state must sustain WITHOUT external drive (bistability).
        if darr is not None:
            bridge.cp_external_input_current[darr] = 0.0
        cnt2 = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for _ in range(read_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            cnt2 += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        return v_on, to_host(cnt2) / float(read_steps)

    def eval_assembly(A):
        se = np.asarray(A, dtype=np.int64)
        r = np.random.default_rng(seed * 131 + int(se[0]))
        se = se[r.permutation(len(se))]
        n_cue = max(2, int(cue_frac * len(se)))
        cue, held = se[:n_cue], se[n_cue:]
        held_pos = [ca3_pos[int(g)] for g in held]
        member = set(int(g) for g in se)
        nonmember_pos = [i for g, i in ca3_pos.items() if g not in member]

        v_cue, v_sustain = drive_read(cue, sustain=True)
        nonA = np.asarray([g for g in ca3_idx if int(g) not in member], dtype=np.int64)
        perm_cue = r.choice(nonA, size=len(cue), replace=False)
        v_perm = drive_read(perm_cue)

        held_cue = float(np.mean(v_cue[held_pos])) if held_pos else 0.0
        held_sustain = float(np.mean(v_sustain[held_pos])) if held_pos else 0.0
        held_perm = float(np.mean(v_perm[held_pos])) if held_pos else 0.0
        nonmember_act = float(np.mean(v_cue[nonmember_pos])) if nonmember_pos else 0.0
        return {"held_cue": held_cue, "held_sustain": held_sustain, "held_perm": held_perm,
                "nonmember_act": nonmember_act, "n_held": len(held_pos), "n_cue": n_cue}

    def no_cue_rest():
        v = drive_read(None)
        # held-out members of assembly 0 as the reference silent-rest cells
        se0 = np.asarray(assemblies[0], dtype=np.int64)
        r = np.random.default_rng(seed * 131 + int(se0[0]))
        se0 = se0[r.permutation(len(se0))]
        held0 = se0[max(2, int(cue_frac * len(se0))):]
        held0_pos = [ca3_pos[int(g)] for g in held0]
        return float(np.mean(v[held0_pos])) if held0_pos else float(np.mean(v))

    # ---- CONTROL: no-encoding (weak baseline, no install) ----
    restore_baseline()
    unt = [eval_assembly(A) for A in assemblies]
    unt_held_cue = float(np.mean([e["held_cue"] for e in unt]))

    # ---- CONTROL: recurrence-zero ----
    zero_recurrents()
    zr = [eval_assembly(A) for A in assemblies]
    zr_held_cue = float(np.mean([e["held_cue"] for e in zr]))

    from tools.lab import attributable_to
    grid = []
    for W in weights:
        install_selective(W)
        rest = no_cue_rest()
        evals = [eval_assembly(A) for A in assemblies]
        held_cue = float(np.mean([e["held_cue"] for e in evals]))
        held_sustain = float(np.mean([e["held_sustain"] for e in evals]))
        held_perm = float(np.mean([e["held_perm"] for e in evals]))
        nonmem = float(np.mean([e["nonmember_act"] for e in evals]))
        # attribution: completion vs the two teeth that carry the verdict.
        attributable_to(f"[s{seed} ou{int(enable_ou)} W{W:.0f}] completion: correct-cue vs NO-ENCODING (attractor load-bearing)",
                        held_cue, unt_held_cue)
        attributable_to(f"[s{seed} ou{int(enable_ou)} W{W:.0f}] SPECIFICITY: correct-cue vs PERMUTED-cue held-out reactivation",
                        held_cue, held_perm)
        row = {
            "seed": seed, "element": element, "enable_ou": bool(enable_ou), "ca3_density": ca3_density, "W": float(W),
            "n_ca3": n_ca3, "n_rec_syn": n_rec, "assembly_size": a_size,
            "held_cue": held_cue, "held_sustain": held_sustain, "held_perm": held_perm, "held_nocue": rest,
            "nonmember_act": nonmem, "no_encoding_held_cue": unt_held_cue,
            "recurrence_zero_held_cue": zr_held_cue,
            "cue_over_perm": held_cue / (held_perm + 1e-6), "cue_over_nocue": held_cue / (rest + 1e-6),
        }
        row["GO"] = bool(held_cue >= 0.20 and held_cue >= 3.0 * (held_perm + 1e-6)
                         and held_cue >= 3.0 * (rest + 1e-6) and rest <= 0.10)
        grid.append(row)
        if verbose:
            print(f"  [s{seed} ou{int(enable_ou)} d{ca3_density} W{W:>6.0f}] cue={held_cue:.3f} sustain={held_sustain:.3f} "
                  f"perm={held_perm:.3f} nocue={rest:.3f} (noenc={unt_held_cue:.3f} reczero={zr_held_cue:.3f} "
                  f"nonmem={nonmem:.3f}) cue/perm={row['cue_over_perm']:5.2f} {'GO' if row['GO'] else '--'}", flush=True)
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-ca3", type=int, default=800)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--assembly-frac", type=float, default=0.18)
    ap.add_argument("--weights", default="60,150,400,1000,2500")
    ap.add_argument("--fb-inhib", type=float, default=20.0)
    ap.add_argument("--nmda-tau", type=float, default=100.0)
    ap.add_argument("--nmda-ratio", type=float, default=1.0)
    ap.add_argument("--drive-pa", type=float, default=250.0)
    ap.add_argument("--warm-steps", type=int, default=180)
    ap.add_argument("--read-steps", type=int, default=170)
    ap.add_argument("--ou", action="store_true", help="OU noise ON (default OFF = deterministic bistability)")
    ap.add_argument("--both-ou", action="store_true", help="run BOTH OU-off and OU-on")
    ap.add_argument("--element", default="nmda_slow", choices=["nmda_slow", "ampa"],
                    help="ca3->ca3 recurrent element; 'ampa' = attribution control (same basket/install/read)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    weights = [float(x) for x in a.weights.replace(",", " ").split()]
    ou_modes = [False, True] if a.both_ou else [bool(a.ou)]

    # determinism check: build twice at seeds[0], hash firing thresholds (NOT actual_seed_used).
    _nr = (a.element == "nmda_slow")
    b1 = _build(seeds[0], n_ca3=a.n_ca3, ca3_density=a.density, coincidence=False, two_comp=False,
                nmda_recurrent=_nr, nmda_tau=a.nmda_tau, ca3_fb_inhib=a.fb_inhib, train=False, enable_ou=False)
    b2 = _build(seeds[0], n_ca3=a.n_ca3, ca3_density=a.density, coincidence=False, two_comp=False,
                nmda_recurrent=_nr, nmda_tau=a.nmda_tau, ca3_fb_inhib=a.fb_inhib, train=False, enable_ou=False)
    h1, h2 = _threshold_hash(b1), _threshold_hash(b2)
    print(f"[determinism] threshold-hash build1={h1} build2={h2} -> {'SEEDED' if h1 == h2 else 'UNSEEDED-BUG'}", flush=True)
    del b1, b2

    print(f"[gap5 nmda_slow reverberatory] seeds={seeds} n_ca3={a.n_ca3} density={a.density} "
          f"assembly_frac={a.assembly_frac} weights={weights} fb_inhib={a.fb_inhib} nmda_tau={a.nmda_tau} "
          f"drive={a.drive_pa} read={a.warm_steps}+{a.read_steps} ou_modes={ou_modes}", flush=True)
    print("  GO gate (per seed): held_cue>=0.20 & cue>=3x perm & cue>=3x nocue & nocue<=0.10", flush=True)
    all_rows = []
    for ou in ou_modes:
        ngo = 0
        for s in seeds:
            t0 = time.time()
            grid = run_seed(s, n_ca3=a.n_ca3, ca3_density=a.density, weights=weights,
                            assembly_frac=a.assembly_frac, ca3_fb_inhib=a.fb_inhib, nmda_tau=a.nmda_tau,
                            nmda_ratio=a.nmda_ratio, drive_pA=a.drive_pa, warm_steps=a.warm_steps,
                            read_steps=a.read_steps, enable_ou=ou, element=a.element)
            all_rows.extend(grid)
            seed_go = any(r["GO"] for r in grid)
            ngo += int(seed_go)
            best = max(grid, key=lambda r: (r["GO"], r["held_cue"]))
            print(f"    (seed {s} ou{int(ou)}: {time.time()-t0:.0f}s) BEST W={best['W']:.0f} cue={best['held_cue']:.3f} "
                  f"perm={best['held_perm']:.3f} nocue={best['held_nocue']:.3f} seedGO={seed_go}", flush=True)
        print(f"  RESULT ou{int(ou)}: {ngo}/{len(seeds)} seeds have >=1 GO working point", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump(all_rows, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
