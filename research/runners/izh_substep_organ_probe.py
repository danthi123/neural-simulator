"""T2 organ validation for cfg.izh_membrane_substeps (research/izh-substeps; finding
research/findings/2026-10-26-izh-membrane-substeps-derisk.md). Observe-only: no sim/ code is wrapped except to
READ state after each step and to attribute rule-(a) silent resets to neurons.

Builds the BG neural-seam organ exactly as the step-1 grid cell does (get_neural_organ, commit op from
BRAIN_BG_COMMIT_OP, preset-centred cells + bounded conductance + tonic drive via the organ's env switches) for ONE
dev seed and ONE arm, runs the build and the four hsb select_once events in order, and writes one JSON:
  * cfg read-back from bridge.core_config (not the CLI) and the env it ran under;
  * engagement counters from the bridge: substep_calls, refractory_model_resets (after build and at the end);
  * membrane: whole-run min/max, per-step network min/max, breach cells (v < -150 or > +100 mV on any step)
    with idx, region, a/b/C/d, first breach step, v_min/v_max, spikes;
  * per region: n, min C, a-priori linearised subthreshold multipliers (see mult(); worst cell of the region) for
    this arm's n and for n = 1: the per-sub-step |1 + h Q'(v)/C| at v = vr and v = E_i, and the full-step
    multiplier with g = 0 and in the strong-conductance limit of the bounded current (held over the sub-steps);
    spikes, 1-ms-ISI spikes (fired on two consecutive steps: what rule (b) newly allows), rule-(a) silent resets;
  * per hsb event: winner (None = abstention), match to the expected [1, 0, 1, 0], commit winner, v_min/v_max,
    step range, tonic_rates_hz.

Arms: fix_off | fix_a (n 4, model_reset) | fix_b (n 4, no_engine_timer) | fix_a_unbounded | fix_a_nopreset |
fix_a_n2. Gate seeds 42/43/44/100/101/102 are refused.

Usage: SIM_NO_PROVENANCE=1 python -m research.runners.izh_substep_organ_probe --seed 72 --arm fix_a --out X.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

GATE_SEEDS = {42, 43, 44, 100, 101, 102}
EXPECTED = [1, 0, 1, 0]
EVENTS = (("hsb_empty1", "..."), ("hsb_q", "what does the dog chase"), ("hsb_empty2", "?"), ("hsb_hi", "hi"))
A_SUB = {"BRAIN_BG_IZH_SUBSTEPS": "4", "BRAIN_BG_IZH_SUBSTEP_RULE": "model_reset"}
ARMS = {
    "fix_off": {},
    "fix_a": dict(A_SUB),
    "fix_b": {"BRAIN_BG_IZH_SUBSTEPS": "4", "BRAIN_BG_IZH_SUBSTEP_RULE": "no_engine_timer"},
    "fix_a_unbounded": dict(A_SUB, BRAIN_BG_BOUNDED_COND=""),
    "fix_a_nopreset": dict(A_SUB, BRAIN_BG_PRESET_CELLS=""),
    "fix_a_n2": {"BRAIN_BG_IZH_SUBSTEPS": "2", "BRAIN_BG_IZH_SUBSTEP_RULE": "model_reset"},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpe", default="100")
    ap.add_argument("--stn", default="100")
    args = ap.parse_args(argv)
    if args.seed in GATE_SEEDS:
        sys.exit(f"refused: {args.seed} is a gate seed")

    env = {"SIM_BACKEND": "numpy", "CUDA_VISIBLE_DEVICES": "", "BRAIN_BG_NEURAL_SEAM": "1", "BRAIN_BG_COMMIT": "1",
           "BRAIN_BG_COMMIT_OP": "", "BRAIN_BG_COMMIT_LESION": "", "BRAIN_BG_NEURAL_SEAM_LESION": "",
           "BRAIN_CHAT_SEED": str(args.seed), "BRAIN_BG_PRESET_CELLS": "1", "BRAIN_BG_BOUNDED_COND": "1",
           "BRAIN_BG_TONIC_DRIVE": "1", "BRAIN_BG_TONIC_GPE_PA": args.gpe, "BRAIN_BG_TONIC_STN_PA": args.stn,
           "BRAIN_BG_IZH_SUBSTEPS": "", "BRAIN_BG_IZH_SUBSTEP_RULE": ""}
    env.update(ARMS[args.arm])
    os.environ.update(env)

    import numpy as np
    import sim.bridge as SB
    from sim.kernels import izhikevich2007_dynamics_update
    from research.runners import bg_action_selection_production_organ as BG
    from research.runners import _vocal_action_selector_gate as VG
    from tools.lab import lever

    st = {"vmin": None, "vmax": None, "first": None, "spk": None, "isi1": None, "prev": None, "step_min": [],
          "step_max": [], "silent": None, "n": 0}

    _orig_sub = SB.izhikevich2007_substep_update

    def sub_w(v, u, C, k, vr, vt, a, b, I, dt, n, thr, c, d, refr=None, *, backend_neutral_arithmetic=False):
        out = _orig_sub(v, u, C, k, vr, vt, a, b, I, dt, n, thr, c, d, refr,
                        backend_neutral_arithmetic=backend_neutral_arithmetic)
        if refr is not None:            # replica of the rule-(a) loop, for per-neuron attribution only
            vv, uu = np.asarray(v), np.asarray(u)
            done = np.zeros(vv.shape, bool)
            per = np.zeros(vv.shape, np.int64)
            for _ in range(int(n)):
                v2, u2 = izhikevich2007_dynamics_update(vv, uu, C, k, vr, vt, a, b, I, dt / int(n),
                                                        backend_neutral_arithmetic=backend_neutral_arithmetic)
                vv, uu = np.where(done, vv, v2), np.where(done, uu, u2)
                cr = (vv >= thr) & ~done
                sl = cr & refr
                vv, uu = np.where(sl, c, vv), np.where(sl, uu + d, uu)
                per += sl
                done |= cr & ~refr
            assert np.array_equal(vv, np.asarray(out[0])) and np.array_equal(uu, np.asarray(out[1]))
            assert int(per.sum()) == int(np.asarray(out[2]))
            st["silent"] = per if st["silent"] is None else st["silent"] + per
        return out

    SB.izhikevich2007_substep_update = sub_w
    _orig_step = SB.SimulationBridge._run_one_simulation_step

    def step_w(self):
        _orig_step(self)
        v = np.asarray(self.cp_membrane_potential_v, dtype=np.float64)
        f = np.asarray(self.cp_firing_states, dtype=bool)
        t = st["n"]
        if st["vmin"] is None:
            st.update(vmin=v.copy(), vmax=v.copy(), first=np.full(v.shape, -1, np.int64),
                      spk=np.zeros(v.shape, np.int64), isi1=np.zeros(v.shape, np.int64),
                      prev=np.zeros(v.shape, bool))
        np.minimum(st["vmin"], v, out=st["vmin"])
        np.maximum(st["vmax"], v, out=st["vmax"])
        bad = ((v < -150.0) | (v > 100.0)) & (st["first"] < 0)
        st["first"][bad] = t
        st["spk"] += f
        st["isi1"] += f & st["prev"]
        st["prev"] = f.copy()
        st["step_min"].append(float(v.min()))
        st["step_max"].append(float(v.max()))
        st["n"] = t + 1

    SB.SimulationBridge._run_one_simulation_step = step_w

    t0 = time.time()
    organ = BG.get_neural_organ(seed=args.seed, lesion=None, commit_op=BG.bg_commit_op(), commit_lesion=None)
    organ.ensure_built()
    br = organ._bridge
    cfg = br.core_config
    n_build = st["n"]
    counters_build = {"substep_calls": int(br.substep_calls), "refractory_model_resets": int(br.refractory_model_resets)}
    # the arm's manipulation must reach the engine: the cfg read-back and the engagement counter move off their
    # defaults in every fix-on arm (a no-op arm would make every cross-arm comparison void)
    fix_on = args.arm != "fix_off"
    lever("izh_membrane_substeps read-back", 1, int(cfg.izh_membrane_substeps), required=fix_on)
    lever("substep_calls after build", 0, int(br.substep_calls), required=fix_on)
    events = []
    for i, (label, msg) in enumerate(EVENTS):
        s0 = st["n"]
        r = organ.select_once(msg)
        w = r["winner"]
        events.append({"label": label, "msg": msg, "winner": w, "abstention": w is None, "expected": EXPECTED[i],
                       "match": w == EXPECTED[i], "commit_winner": (r.get("commit") or {}).get("winner"),
                       "v_min": r["v_min"], "v_max": r["v_max"], "step_range": [s0, st["n"]],
                       "tonic_rates_hz": r.get("tonic_rates_hz")})

    regions = [rg.name for rg in cfg.brain_regions]
    reg_of = np.full(int(cfg.num_neurons), -1, np.int64)
    for i, nm in enumerate(regions):
        reg_of[np.asarray(br.region_manager.indices(nm), dtype=np.int64)] = i
    par = {k: np.asarray(getattr(br, "cp_izh_" + k), dtype=np.float64)
           for k in ("a", "b", "C", "k", "vr", "vt", "vpeak", "c_reset", "d_increment")}
    e_i = br.cp_syn_reversal_potential_i_per_neuron
    e_i = (np.asarray(e_i, dtype=np.float64) if e_i is not None
           else np.full(int(cfg.num_neurons), float(cfg.syn_reversal_potential_i)))
    nsub = int(cfg.izh_membrane_substeps)

    def mult(n):
        """Linearised subthreshold multipliers (u held; worst = largest), n sub-steps of h = dt/n, Q'(v) = k(2v-vr-vt):
        substep_*  = |r|, r = 1 + h Q'(v)/C  (the per-sub-step factor |1 + h Q'(v)/C|);
        step_*_g0  = |r^n|  (one full step, no conductance);
        step_Ei_ginf = |r^n - (1/n) sum_{j<n} r^j| at v = E_i: one full step under the strong-conductance limit of
                   the bounded current, which is computed once from v at step start and held over the sub-steps.
        A full step is linearly stable where the step_* value is < 1."""
        h = float(cfg.dt_ms) / n
        C, k, vr, vt = par["C"], par["k"], par["vr"], par["vt"]
        r_rest = 1.0 + h * k * (vr - vt) / C
        r_ei = 1.0 + h * k * (2.0 * e_i - vr - vt) / C
        geo = sum(r_ei ** j for j in range(n)) / n
        return {"substep_rest": np.abs(r_rest), "substep_Ei": np.abs(r_ei),
                "step_rest_g0": np.abs(r_rest ** n), "step_Ei_g0": np.abs(r_ei ** n),
                "step_Ei_ginf": np.abs(r_ei ** n - geo)}

    m_arm, m_dt = mult(nsub), mult(1)
    silent = st["silent"] if st["silent"] is not None else np.zeros(int(cfg.num_neurons), np.int64)
    per_region = {}
    for i, nm in enumerate(regions):
        idx = np.where(reg_of == i)[0]
        if not len(idx):
            continue
        per_region[nm] = {
            "n": int(len(idx)), "min_C": float(par["C"][idx].min()),
            "argmin_C_idx": int(idx[np.argmin(par["C"][idx])]),
            "E_i": [float(e_i[idx].min()), float(e_i[idx].max())],
            "mult_n_arm": {k: round(float(v[idx].max()), 4) for k, v in m_arm.items()},
            "mult_n1": {k: round(float(v[idx].max()), 4) for k, v in m_dt.items()},
            "v_min": float(st["vmin"][idx].min()), "v_max": float(st["vmax"][idx].max()),
            "spikes": int(st["spk"][idx].sum()), "isi1_spikes": int(st["isi1"][idx].sum()),
            "silent_model_resets": int(silent[idx].sum()),
            "n_breach": int((st["first"][idx] >= 0).sum()),
        }
    bad = np.where(st["first"] >= 0)[0]
    breach_cells = [{"idx": int(j), "region": regions[reg_of[j]], "first_step": int(st["first"][j]),
                     "v_min": float(st["vmin"][j]), "v_max": float(st["vmax"][j]), "spikes": int(st["spk"][j]),
                     "a": float(par["a"][j]), "b": float(par["b"][j]), "C": float(par["C"][j]),
                     "d": float(par["d_increment"][j])} for j in bad]
    out = {
        "instrument": "research.runners.izh_substep_organ_probe", "seed": args.seed, "arm": args.arm,
        "env": env, "argv": sys.argv,
        "cfg_readback": {
            "seed": int(cfg.seed), "izh_membrane_substeps": nsub,
            "izh_substep_refractory_rule": cfg.izh_substep_refractory_rule,
            "enable_bounded_conductance_integration": bool(cfg.enable_bounded_conductance_integration),
            "preset_centred_parameter_heterogeneity": bool(cfg.preset_centred_parameter_heterogeneity),
            "enable_parameter_heterogeneity": bool(cfg.enable_parameter_heterogeneity),
            "refractory_period_steps": int(cfg.refractory_period_steps), "dt_ms": float(cfg.dt_ms),
            "fast_spike_reset": bool(cfg.fast_spike_reset), "num_neurons": int(cfg.num_neurons),
            "commit_op": BG.bg_commit_op(), "tonic_pA": list(VG.tonic_drive_pA(organ.config)),
        },
        "counters_after_build": counters_build,
        "counters_end": {"substep_calls": int(br.substep_calls),
                         "refractory_model_resets": int(br.refractory_model_resets)},
        "n_steps": int(st["n"]), "n_build_steps": int(n_build),
        "membrane": {"v_min": float(st["vmin"].min()), "v_max": float(st["vmax"].max()),
                     "in_range_every_step": bool(len(bad) == 0), "n_breach_cells": int(len(bad)),
                     "steps_out_of_range": int(sum(1 for a, b in zip(st["step_min"], st["step_max"])
                                                   if a < -150.0 or b > 100.0))},
        "breach_cells": breach_cells, "per_region": per_region, "events": events,
        "winners": [e["winner"] for e in events], "n_match": int(sum(e["match"] for e in events)),
        "n_abstain": int(sum(e["abstention"] for e in events)), "wall_s": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({"seed": args.seed, "arm": args.arm, "n_breach": out["membrane"]["n_breach_cells"],
                      "v": [out["membrane"]["v_min"], out["membrane"]["v_max"]], "winners": out["winners"],
                      "counters": out["counters_end"], "wall_s": out["wall_s"]}))


if __name__ == "__main__":
    main()
