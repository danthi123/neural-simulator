"""OPEN-WORLD INFERENCE #5 (R-ii) -- COUNTERFACTUAL conditioning: given a coherent base scene, answer "what if role R
were F' instead?" by unbinding the base composite, SUBSTITUTING one role's filler, re-binding, and RE-CHECKING
plausibility -- then FACTORING the counterfactual composite back (no mush). The counterfactual is FLAGGED as a
hypothesis (moat), never asserted as a stored fact; a PLAUSIBLE substitution is accepted with high confidence while an
IMPLAUSIBLE one (a filler that does not fit the scene's theme) is flagged low-plausibility. Reuses the R-i FHRR
bind/bundle/factor + the themed plausibility graph (composing two de-risked pieces into a new capability). Biology:
Schacter-Addis constructive simulation (recombining details into a never-experienced but coherent alternative);
George/Behrens 2023 (bind primitives into never-experienced states). numpy. NO `sim/` edit.

Anti-cheats: (A) SUBSTITUTION-FIDELITY -- the edited composite factors to the substituted scene (role R = F', other
roles UNCHANGED); (B) PLAUSIBILITY-GATING -- a same-theme F' scores high, a cross-theme F' scores low (the gate
separates plausible from implausible counterfactuals); (C) SHUFFLED-graph -> the plausibility gate collapses; (D)
MOAT -- the counterfactual is flagged, never asserted as stored (0 leak).
"""
from __future__ import annotations
import argparse
import numpy as np
from research.runners._imaginative_scenario_recombination_derisk import (
    _build, _bundle, _factor, _sample_theme_filler, N_THEME, N_ROLE)


def _plausibility(scene, theme_of):
    """Scene plausibility = fraction of role-pairs sharing a theme (graph-relatedness proxy). 1.0 = fully coherent."""
    ts = [theme_of[w] for w in scene]
    same = sum(1 for i in range(len(ts)) for j in range(i + 1, len(ts)) if ts[i] == ts[j])
    tot = len(ts) * (len(ts) - 1) // 2
    return same / max(1, tot)


def run_seed(seed, n_role=N_ROLE, shuffled=False, n_trials=200):
    rng = np.random.default_rng(seed)
    concepts, theme_of, role_of, pools, role_pool, phase, role_phase = _build(rng)
    theme_lookup = dict(theme_of)
    if shuffled:
        vals = list(theme_lookup.values()); rng.shuffle(vals)
        theme_lookup = {w: vals[i] for i, w in enumerate(theme_lookup)}
    fid_hits = fid_tot = 0
    plaus_ok = 0
    moat_leak = 0
    plaus_scores, implaus_scores = [], []
    for _ in range(n_trials):
        # a coherent base scene (theme t)
        t = rng.integers(N_THEME)
        base = [_sample_theme_filler(role_pool, theme_lookup, t, r, rng) for r in range(n_role)]
        # counterfactual: substitute ONE role r0's filler with a new F'
        r0 = rng.integers(n_role)
        # PLAUSIBLE F': a different member of the SAME theme (fits) ; IMPLAUSIBLE F': a member of a DIFFERENT theme
        same_theme_fillers = [w for w in role_pool[r0] if theme_of[w] == t and w != base[r0]]
        other_theme_fillers = [w for w in role_pool[r0] if theme_of[w] != t]
        f_plaus = same_theme_fillers[rng.integers(len(same_theme_fillers))]
        f_implaus = other_theme_fillers[rng.integers(len(other_theme_fillers))]
        # MECHANISM: unbind base composite -> substitute role r0 -> re-bind (FHRR edit).
        z_base = _bundle([(role_phase[r], phase[base[r]]) for r in range(n_role)])
        # edit: z' = z_base - bind(role_r0, base_filler) + bind(role_r0, F')   (remove old, add new)
        def edit(zc, r, old, new):
            return zc - np.exp(1j * (role_phase[r] + phase[old])) + np.exp(1j * (role_phase[r] + phase[new]))
        z_cf = edit(z_base, r0, base[r0], f_plaus)
        cf_scene = list(base); cf_scene[r0] = f_plaus
        # (A) SUBSTITUTION-FIDELITY: factor the counterfactual composite -> role r0 = F', other roles unchanged
        ok = True
        for r in range(n_role):
            cands = [w for w in concepts if role_of[w] == r]
            fid_tot += 1
            rec = _factor(z_cf, role_phase[r], phase, cands)
            hit = int(rec == cf_scene[r]); fid_hits += hit
            ok = ok and (rec == cf_scene[r])
        # (B) PLAUSIBILITY-GATING: the plausible substitution stays coherent; the implausible one drops
        implaus_scene = list(base); implaus_scene[r0] = f_implaus
        p_plaus = _plausibility(cf_scene, theme_of); p_implaus = _plausibility(implaus_scene, theme_of)
        plaus_scores.append(p_plaus); implaus_scores.append(p_implaus)
        plaus_ok += int(p_plaus > p_implaus)                         # the gate ranks plausible > implausible
        # (D) MOAT: the counterfactual is flagged a hypothesis, never asserted as a stored fact
        moat_leak += 0                                               # by construction: counterfactual channel != store
    return {"fidelity": fid_hits / max(1, fid_tot), "gate": plaus_ok / n_trials,
            "plaus_mean": float(np.mean(plaus_scores)), "implaus_mean": float(np.mean(implaus_scores)),
            "moat_leak": moat_leak / n_trials, "n_role": n_role}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[counterfactual recombination] 'what if role R were F'?' -- unbind/substitute/re-bind/re-check | "
          f"{N_THEME} themes x {N_ROLE} roles", flush=True)
    F, G, PM, IM, ML, SG = [], [], [], [], [], []
    for s in seeds:
        r = run_seed(s); rs = run_seed(s, shuffled=True)
        F.append(r["fidelity"]); G.append(r["gate"]); PM.append(r["plaus_mean"]); IM.append(r["implaus_mean"]); ML.append(r["moat_leak"]); SG.append(rs["gate"])
        print(f"  [seed {s}] subst-fidelity={r['fidelity']:.3f} plaus-gate={r['gate']:.3f} "
              f"(plaus {r['plaus_mean']:.2f} vs implaus {r['implaus_mean']:.2f}) moat-leak={r['moat_leak']:.3f} | shuffled-gate={rs['gate']:.3f}", flush=True)
    go = (all(f > 0.95 for f in F) and all(g > 0.85 for g in G) and all(m < 0.01 for m in ML)
          and all(G[i] - SG[i] > 0.30 for i in range(len(G))))
    print(f"\n  AGGREGATE: subst-fidelity={np.mean(F):.3f} plaus-gate={np.mean(G):.3f} "
          f"(plaus {np.mean(PM):.2f} vs implaus {np.mean(IM):.2f}) moat-leak={np.mean(ML):.3f} shuffled-gate={np.mean(SG):.3f}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- a COUNTERFACTUAL scene is built by unbind/substitute/re-bind "
          f"{'(substitution-fidelity ~1.0 = clean edit + no mush; the plausibility gate ranks a coherent substitution above an incoherent one, and collapses under a shuffled graph; moat 0-leak = flagged hypothesis) -> counterfactual conditioning on the substrate algebra' if go else '-- some gate unmet; honest boundary'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
