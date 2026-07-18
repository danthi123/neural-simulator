"""Gap #3 residual A1 (spiking) — the feature-compatibility that decides WHICH referent to bias is computed by
SPIKING neurons (a coincidence over learned animacy-feature detectors), replacing the host `content_bias_target`.

Mechanism (cheap-first GO `2026-07-18-gap3-A1-learned-feature-compatibility-cheap-first-GO.md`): concept ANIMACY +
verb SELECTION emerge from corpus co-occurrence (joint EM). This realizes the COMPATIBILITY READOUT on a real
`SimulationBridge`: two feature-detector pools (F_anim, F_inanim); the query VERB drives its learned selection pool;
each CANDIDATE drives its learned animacy pool; the COMPATIBLE candidate co-drives the verb's selection pool (a
coincidence) → its shared-pool firing is highest → it is the bias target. The learned SIGNS are the offline-EM
scaffold (like the concept codes); the DECISION (compatibility) is spiking. GO: spiking bias-target == host
`content_bias_target` on resolvable cases, 6-seed; PERMUTED-CORPUS anti-cheat (learn features from a shuffled corpus)
collapses the readout (the corpus→features→spiking pipeline is load-bearing, not a host smuggle).
"""
import os, sys
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.biased_competition_buffer import ANIMACY, VERB_SELECTS, content_bias_target
from research.runners._gap3_learned_feature_compat_derisk import make_corpus, learn_features, CONCEPTS, ANIMATE, INANIM, VERBS


def _build(seed, n_feat=40):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    regions = [BrainRegion(name=nm, n_neurons=n_feat, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
               for nm in ("F_anim", "F_inanim")]
    # an inert pathway (weight 0) so the connectivity generation has a profile (a minimal-config bridge init edge case)
    pathways = [RegionPathway(from_region="F_anim", to_region="F_inanim", density=1.0,
                              weight_mean=0.0, weight_jitter=0.0, plastic=False)]
    cfg = CoreSimConfig(); cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.enable_brain_region_framework = True; cfg.brain_regions = regions; cfg.region_pathways = pathways
    cfg.enable_stdp = False; cfg.enable_homeostasis = False; cfg.enable_short_term_plasticity = False
    cfg.enable_ou_process = False; cfg.enable_structural_plasticity = False; cfg.fast_spike_reset = True
    cfg.enable_hebbian_learning = False
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def run_seed(seed, permute=False):
    from sim.backend import to_host, from_host
    facts = make_corpus(seed, permute=permute)                     # anti-cheat: permuted corpus -> wrong learned features
    ca, vs = learn_features(facts)                                  # learned signs (offline-EM scaffold)
    agree = np.mean([np.sign(ca.get(c, 0.0)) == (1 if ANIMACY[c] == "animate" else -1)
                     for c in CONCEPTS if ca.get(c, 0.0) != 0])
    sign = 1.0 if agree >= 0.5 else -1.0
    b = _build(seed); n = b.core_config.num_neurons
    f_anim = np.asarray(list(b.region_manager.indices("F_anim")), int)
    f_inan = np.asarray(list(b.region_manager.indices("F_inanim")), int)

    def _feat_pool(anim_sign):                                      # which detector pool a +/- animacy sign drives
        return f_anim if anim_sign > 0 else f_inan

    def _reset():
        if getattr(b, "cp_izh_c_reset", None) is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for _a in ("cp_conductance_g_e", "cp_conductance_g_i"):
            _arr = getattr(b, _a, None)
            if _arr is not None:
                _arr[:] = 0.0

    def _coincidence(verb, cand, steps=25, drive=500.0):
        """Spiking coincidence: drive the verb's SELECTION pool + the candidate's ANIMACY pool; read the firing of the
        VERB's selection pool. The COMPATIBLE candidate (same feature) co-drives it -> higher firing."""
        vsel = np.sign(vs.get(verb, 0.0) * sign)
        casign = np.sign(ca.get(cand, 0.0) * sign)
        if vsel == 0:
            return 0.0
        vpool = _feat_pool(vsel); cpool = _feat_pool(casign)
        _reset()
        cur = np.zeros(n)
        cur[vpool] += drive                                        # verb -> its selection detector (coincidence amplify)
        cur[cpool] += drive                                        # candidate -> its animacy detector
        dev = from_host(cur.astype(np.float64)); rate = 0.0
        for _ in range(steps):
            b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
            rate += float(np.asarray(to_host(b.cp_firing_states))[vpool].mean())   # firing of the VERB's selection pool
        return rate

    rng = np.random.default_rng(seed * 3 + 1); ok = ntot = 0
    for v in VERBS:
        for _ in range(6):
            a = rng.choice(ANIMATE); i = rng.choice(INANIM); cands = [a, i]; rng.shuffle(cands)
            host = content_bias_target(cands, v)
            if host is None:
                continue
            scores = {c: _coincidence(v, c) for c in cands}
            top = max(scores, key=scores.get)
            spk = top if (scores[top] > 1e-6 and list(scores.values()).count(scores[top]) == 1) else None
            ok += int(spk == host); ntot += 1
    return ok / ntot if ntot else 0.0


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = (42, 43, 44, 100, 101, 102)
    acc = [run_seed(s) for s in seeds]
    les = [run_seed(s, permute=True) for s in seeds]
    ma, ml = float(np.mean(acc)), float(np.mean(les))
    print(f"[gap#3 A1 SPIKING feature-compatibility] F_anim/F_inanim detector pools, coincidence readout")
    for s, a, l in zip(seeds, acc, les):
        print(f"  [seed {s}] spiking==host {a:.2f} | permuted-corpus {l:.2f}")
    go = ma >= 0.80 and ml <= 0.60
    print(f"  MEAN(6): spiking==host {ma:.2f} | permuted-corpus {ml:.2f} (must collapse) -> {'GO' if go else 'NO'}")


if __name__ == "__main__":
    main()
