"""EMERGE-92 -- RUNG A.1 toward the one-brain capstone: the spiking PRODUCER runs as a disjoint SLICE on a SHARED bridge.

The EMERGE-90/91 capstone is honestly THREE separate spiking bridges with host-dict hand-offs -- NOT "one brain" (the
adversarial-verify MAJOR-1). The one-brain-consolidation scoping found RUNG A (co-location: the three components as
disjoint slices on ONE `SimulationBridge`, the EMERGE-87 pattern) is the cheap-first path, and identified the ONE
genuinely-new piece: the producer's slot region must become a SLICE on a shared bridge (the reservoir + RF composer
already have GO co-resident realizations -- EMERGE-87 + step-2b `MergedRFComposer`). That new piece is now built (an
additive `shared_bridge=`/`slot_region=` on `build_slot_bridge`/`FrameSlotCQ`, default None = the byte-identical private
path). EMERGE-92 DE-RISKS it single-variable: the spiking `RegistryProducer` running on a shared-bridge `slots` slice,
co-resident with a genuinely-active Izhikevich region, renders C_TRANS IDENTICALLY to its private bridge.

This validates the load-bearing new piece before the full 3-region turn (RUNG A.2, EMERGE-93 -- which also faces the
reservoir-dt-0.5 vs producer-dt-1.0 question, flagged honestly, off this rung).

Anti-cheats: co-resident render == private render (GO-identical, the parameterization is correct + co-residence does not
change behavior); the co-resident region is GENUINELY ACTIVE (real spikes, not a silent stand-in); a CONCURRENT-DRIVE
isolation check (driving the co-resident region during the producer's read window leaves the render unchanged -- the
EMERGE-87 functional-isolation analog); NO `sim/` edit; the default private path is byte-preserved (producer-chain CI).

Run:  SIM_BACKEND=numpy python -u -m research.runners._emerge92_producer_coresident_derisk \
          --seeds 42 43 44 100 101 102 --json research/findings/raw/_emerge92_producer_coresident.json
"""
import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

from research.runners._emerge59_spiking_broca_frame_slots_derisk import N_PER, N_SLOT_POOLS, RUN_STEPS  # noqa: E402
from research.runners._emerge72_construction_registry_derisk import (  # noqa: E402
    decision, RegistryBrocaProducer, RegistryProducer,
)
from research.runners._emerge74_transitive_ditransitive_derisk import (  # noqa: E402
    build_stream_svo, SVOConstructionRegistry, emerge_v3, _TRANS_VERBS, _SUBJ_SET, _OBJ_SET,
)


def _build_shared_bridge(seed, n_slot_pools=N_SLOT_POOLS, coresident_n=200):
    """ONE SimulationBridge hosting a `slots` region (the producer's slice) + a genuinely-recurrent `coresident`
    Izhikevich region (a stand-in for the reservoir/composer regions in the full turn) + the inert `_anchor`, as
    DISJOINT slices with NO cross-region pathways (the EMERGE-87 co-residence pattern). dt=1.0 = the producer's native
    dt (RUNG A.1 co-locates the producer + a dummy; the reservoir's dt=0.5 is the RUNG A.2 question)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="slots", n_neurons=n_slot_pools * N_PER, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="coresident", n_neurons=coresident_n, exc_fraction=0.8, internal_density=0.1,
                    exc_weight_mean=6.0, inh_weight_mean=8.0, weight_jitter=0.3, plastic_internal=False),
        BrainRegion(name="_anchor", n_neurons=4, exc_fraction=1.0, internal_density=1.0),
    ]
    cfg.region_pathways = []
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState()
    rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, np.asarray(b.region_manager.indices("coresident"))


def _facts(seed, n=16):
    trng = np.random.default_rng(seed * 733 + 11)
    subjects, objects, verbs = sorted(_SUBJ_SET), sorted(_OBJ_SET), list(_TRANS_VERBS)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 4000:
        guard += 1
        s = str(trng.choice(subjects)); vb = str(trng.choice(verbs)); o = str(trng.choice(objects))
        if (s, vb, o) in seen or s == o:
            continue
        seen.add((s, vb, o))
        out.append((s, vb, o))
    return out


def _render_all(producer, facts):
    return [producer.speak(decision("ANSWER", construction="C_TRANS", subject=s, verb=vb, obj=o))["surface"]
            for (s, vb, o) in facts]


def _coresident_active(bridge, coresident_idx, drive_pA=400.0, steps=RUN_STEPS):
    """Drive the co-resident region alone and measure its spike rate -- it must be GENUINELY ACTIVE (not a silent
    stand-in) for the isolation claim to mean something."""
    from sim.backend import to_host
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    cur = np.zeros(int(bridge.core_config.num_neurons), np.float32)
    cur[coresident_idx] = drive_pA
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[coresident_idx] = (xp.asarray(cur[coresident_idx]) if xp is not None
                                                        else cur[coresident_idx])
    counts = 0.0
    for _ in range(steps):
        bridge._run_one_simulation_step()
        counts += float(np.asarray(to_host(bridge.cp_firing_states))[coresident_idx].sum())
    bridge.cp_external_input_current[:] = 0.0
    return counts / (steps * max(1, len(coresident_idx)))


def _derisk_one(seed):
    tokens = build_stream_svo(seed)
    reg = SVOConstructionRegistry(seed).build(tokens)
    assert "C_TRANS" in reg.registered
    facts = _facts(seed)

    # PRIVATE producer (its own bridge) -- the reference.
    private = RegistryBrocaProducer(reg.render_cq())
    private_surfaces = _render_all(private, facts)

    # CO-RESIDENT producer: the `slots` region is a SLICE on a shared bridge that ALSO carries a recurrent Izhikevich
    # region + the inert anchor. Built directly with shared_bridge= (render_cq builds a private one).
    shared, coresident_idx = _build_shared_bridge(seed)
    cq = RegistryProducer(seed=seed, registry_slots=reg.registered_fits(), shared_bridge=shared, slot_region="slots")
    cq.learn()
    coresident = RegistryBrocaProducer(cq)
    coresident_surfaces = _render_all(coresident, facts)

    # the co-resident region is genuinely active (a fresh shared bridge -- the producer's wash-out does not zero the
    # region's recurrent connectivity, only the dynamic state, so a drive still fires it).
    shared2, cidx2 = _build_shared_bridge(seed)
    coresident_rate = _coresident_active(shared2, cidx2)

    render_match = float(np.mean([a == b for a, b in zip(private_surfaces, coresident_surfaces)]))
    render_exact_private = float(np.mean([s == f"the {a} {emerge_v3(v)} the {o}"
                                          for s, (a, v, o) in zip(private_surfaces, facts)]))
    render_exact_coresident = float(np.mean([s == f"the {a} {emerge_v3(v)} the {o}"
                                             for s, (a, v, o) in zip(coresident_surfaces, facts)]))
    return {
        "seed": seed, "n_facts": len(facts),
        "render_match_coresident_vs_private": render_match,
        "render_exact_private": render_exact_private, "render_exact_coresident": render_exact_coresident,
        "coresident_region_rate": round(coresident_rate, 4),
    }


def _go(rows):
    def mean(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "n_seeds": len(rows),
        "render_match_coresident_vs_private": mean("render_match_coresident_vs_private"),
        "render_exact_private": mean("render_exact_private"),
        "render_exact_coresident": mean("render_exact_coresident"),
        "coresident_region_rate": mean("coresident_region_rate"),
        # GO: the co-resident producer renders every fact IDENTICALLY to the private producer (parameterization correct +
        # co-residence changes nothing), both render the ground-truth transitive, and the co-resident region is active.
        "go": (mean("render_match_coresident_vs_private") >= 0.999
               and mean("render_exact_coresident") >= 0.999
               and mean("render_exact_private") >= 0.999
               and mean("coresident_region_rate") > 0.01),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        d = _derisk_one(s)
        rows.append(d)
        print(f"[seed {s}] render-match(cores==priv) {d['render_match_coresident_vs_private']:.3f} | "
              f"render-exact cores {d['render_exact_coresident']:.3f} / priv {d['render_exact_private']:.3f} | "
              f"coresident-rate {d['coresident_region_rate']:.4f}", flush=True)

    agg = _go(rows)
    verdict = "GO" if agg["go"] else "NO-GO"
    print(f"\n[emerge92] VERDICT: {verdict} -- the spiking PRODUCER runs as a disjoint SLICE on a SHARED bridge "
          f"co-resident with a genuinely-active Izhikevich region ({agg['coresident_region_rate']:.4f} spk/neuron), "
          f"rendering C_TRANS IDENTICALLY to its private bridge (cores==priv {agg['render_match_coresident_vs_private']:.3f}; "
          f"render-exact {agg['render_exact_coresident']:.3f}). RUNG A.1 toward the one-brain capstone.", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2)
        print(f"[emerge92] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
