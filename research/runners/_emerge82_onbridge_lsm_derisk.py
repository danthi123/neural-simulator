"""EMERGE-82 -- RUNG 2 of the spiking-reservoir port: realize the EMERGE-80 liquid-state machine as a RECURRENT REGION on a
real `SimulationBridge` (the ON-SUBSTRATE realization), so the reservoir form->role mechanism runs on the ONE BRAIN's own
Izhikevich dynamics + conductance synapses -- not a standalone numpy pool.

WHY. EMERGE-80 ported the reservoir form->role mechanism to a spiking Izhikevich pool, but via a DIRECT numpy recurrent
loop (`W_rec @ spikes`). The fully-spiking-one-brain directive wants it ON the project's substrate: a recurrent BrainRegion
whose internal connectivity (`internal_density`) IS the fixed-random reservoir recurrence, driven through the bridge's real
`_run_one_simulation_step` (conductance-based synapses `g_syn*(V-E)`, the actual neuron model), so the reservoir COMPOSES
with the rest of the one brain instead of being a bolt-on pool. This is a genuine LIQUID-STATE MACHINE on the SimulationBridge.

THE MECHANISM. `OnBridgeLSM` mirrors the EMERGE-78 `Reservoir` API (`final_state(U)`) so it DROPS INTO the entire EMERGE-78
harness (construction generators, final-state slot read-out, governing-cue + symmetric-window baselines, anti-cheats) with
only the pool swapped for a bridge region. A single recurrent `BrainRegion` (name="reservoir", Izhikevich, exc/inh mix,
`internal_density>0` = the fixed-random recurrent synapses) is built via the brain-region framework; a fixed-random input
projection `W_in` drives the region's `cp_external_input_current` per token (+ a tonic bias -> fluctuation-driven LSM regime);
the read-out feature = the region's per-neuron spike-COUNT over the whole sequence (population rate), from the bridge's real
`cp_firing_states`. The bridge state (v/u/conductances/STP/firing) is WASHED to its post-init snapshot before each sentence
(EMERGE-61 mechanism) so every sentence is an independent read.

THE DE-RISK (6 seeds; the bridge runs on numpy-CPU for a small region; reuse the EMERGE-78 harness + EMERGE-80 controls; NO
`sim/` edit -- the region + input drive + read are all via public bridge APIs):
  * (A) CONSOLIDATION -- the on-bridge region LEARNS the full form->role map (train role acc) via a ridge read-out over its
    real spike-counts;
  * (B) NON-LOCAL -- it resolves the relative-clause HEAD where BOTH the governing-cue + symmetric-window baselines are at
    chance (the EMERGE-78 gate; the region's recurrent spiking integrates the whole sequence);
  * SPIKING-ness: the region is genuinely active (mean spikes/neuron > 0) + a REGION-SILENCE lesion (zero the input drive)
    collapses the read (the read is from the region's real spikes, not a static bias);
  * rel-head scramble -> chance; the hand labeler None on the multi-arg shapes.
  GO bar: region active AND train >= 0.90 AND rel-head >= 0.85 while both baselines <= 0.65 AND silence-lesion collapses AND
  scramble collapses. BOUNDARY -> name the on-bridge residual (region operating point / recurrent weights / read-out) as the
  next single-variable de-risk (the numpy-pool EMERGE-80 GO stands; this characterizes the on-bridge realization). Do NOT
  force GO.

HONEST SCOPE. The reservoir region is a DISJOINT slice on its own bridge here (RUNG 2 = "on the SimulationBridge substrate");
co-residence with the nav/conv regions on ONE shared bridge (the full one-brain merge) is the follow-on (the merge pattern
is already validated -- `nav_conv_merged_bridge`). Reuse-by-import (EMERGE-78 harness + EMERGE-61 wash-out + the brain-region
framework); NO `sim/` edit.

Run:
  python -m research.runners._emerge82_onbridge_lsm_derisk --demo
  python -m research.runners._emerge82_onbridge_lsm_derisk --derisk
  SIM_BACKEND=cupy python -m research.runners._emerge82_onbridge_lsm_derisk --derisk   # GPU if the CPU sweep is slow
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _content_pools, _make_sentence, _slot_data, _fit_slots, _slot_acc, _fit_gov_baseline, _gov_acc,
    _fit_symwin, _symwin_acc, _hand_labeler_none, _TRAIN_KINDS, _RELHEAD_KINDS,
)
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _snapshot_state, _restore_state  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge82_onbridge_lsm.json"

_N_POOL = 300                     # reservoir region size (small -> numpy-CPU feasible; the bridge step is heavy)
_INTERNAL_DENSITY = 0.1           # the fixed-random recurrent connectivity of the region (the LSM recurrence)
_EXC_W = 6.0                      # recurrent excitatory synaptic weight
_INH_W = 8.0                      # recurrent inhibitory synaptic weight (E/I balance keeps the pool from saturating)
_T_STEP = 12                      # bridge steps per input token
_IN_SCALE = 320.0                 # input drive scale (pA per active input dim; > Izhikevich RS rheobase)
_BIAS = 45.0                      # tonic background current (fluctuation-driven LSM regime)
_N_TRAIN_PER = 90                 # train sentences per construction (reduced -- the bridge step is heavy)


def _build_reservoir_bridge(seed, n_pool, in_dim):
    """One recurrent Izhikevich BrainRegion = the reservoir (internal_density>0 -> fixed-random recurrent synapses). A
    fixed-random input projection W_in drives the region's external current per token. Returns (bridge, res_idx, W_in,
    snapshot)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="reservoir", n_neurons=n_pool, exc_fraction=0.8, internal_density=_INTERNAL_DENSITY,
                    exc_weight_mean=_EXC_W, inh_weight_mean=_INH_W, weight_jitter=0.3, plastic_internal=False),
    ]
    cfg.region_pathways = []
    cfg.dt = 0.5
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState()
    rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    res_idx = np.asarray(b.region_manager.indices("reservoir"))
    rng = np.random.default_rng(seed * 7919 + 3)
    W_in = (rng.random((len(res_idx), in_dim)) * 2 - 1) * _IN_SCALE
    snap = _snapshot_state(b)
    return b, res_idx, W_in, snap


class OnBridgeLSM:
    """A recurrent BrainRegion liquid-state machine on a real SimulationBridge, with the EMERGE-78 Reservoir API.
    `final_state(U)` washes the bridge to its post-init state, drives the region per token via cp_external_input_current,
    runs the bridge's real step loop (conductance synapses + Izhikevich), and returns the region's per-neuron spike-count
    over the whole sequence (the population read-out feature)."""

    def __init__(self, in_dim, seed, n=_N_POOL):
        self.n = n
        self.bridge, self.res_idx, self.W_in, self._snap = _build_reservoir_bridge(seed, n, in_dim)
        from sim.backend import get_backend
        self._xp, _ = get_backend()
        self._num = int(self.bridge.core_config.num_neurons)
        self._last_mean_spikes = 0.0

    def final_state(self, U, silence=False):
        from sim.backend import to_host
        b = self.bridge
        _restore_state(b, self._snap)                        # wash to post-init -> independent read per sentence
        counts = np.zeros(self._num, np.float64)
        for t in range(len(U)):
            drive = np.zeros(self.n) if silence else (self.W_in @ U[t] + _BIAS)
            cur = np.zeros(self._num, np.float32)
            cur[self.res_idx] = drive.astype(np.float32)
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[self.res_idx] = (self._xp.asarray(cur[self.res_idx])
                                                         if self._xp is not None else cur[self.res_idx])
            for _ in range(_T_STEP):
                b._run_one_simulation_step()
                counts += np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
        b.cp_external_input_current[:] = 0.0
        pool_counts = counts[self.res_idx]
        self._last_mean_spikes = float(pool_counts.mean() / max(1, len(U) * _T_STEP)) * 100.0  # spikes/neuron scaled
        return pool_counts / max(1, len(U) * _T_STEP)


def _derisk_one(seed):
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, _p, _f, _cp = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    enc = Encoder(discovered)
    lsm = OnBridgeLSM(enc.dim, seed=seed)
    rng = np.random.default_rng(seed * 101 + 5)

    train = [_make_sentence(k, rng, subj, verb, obj) for k in _TRAIN_KINDS for _ in range(_N_TRAIN_PER)]
    Ws = _fit_slots(lsm, enc, train)
    gov_tab, gov_def = _fit_gov_baseline(train)
    sw_tab, sw_def = _fit_symwin(enc, train)

    train_acc = _slot_acc(lsm, enc, Ws, [_make_sentence(k, rng, subj, verb, obj) for k in _TRAIN_KINDS for _ in range(20)])

    rel = [_make_sentence(k, rng, subj, verb, obj) for k in _RELHEAD_KINDS for _ in range(80)]
    relhead = _slot_acc(lsm, enc, Ws, rel, only_slot=0)
    relhead_gov = _gov_acc(gov_tab, gov_def, rel, only_slot=0)
    relhead_sym = _symwin_acc(enc, sw_tab, sw_def, rel, only_slot=0)
    rel_full = _slot_acc(lsm, enc, Ws, rel)

    scr = np.random.default_rng(seed * 613 + 7)
    relhead_scramble = _slot_acc(lsm, enc, Ws, rel, scramble_rng=scr, only_slot=0)

    _ = lsm.final_state(enc.encode(_make_sentence("transitive", rng, subj, verb, obj)[0]))
    mean_spikes = lsm._last_mean_spikes
    from research.runners._emerge78_reservoir_form_to_role_derisk import _ROLE_IDX
    from collections import defaultdict
    S, Y = defaultdict(list), defaultdict(list)
    for toks, roles in rel:
        f = np.concatenate([lsm.final_state(enc.encode(toks), silence=True), [1.0]])
        for k, tk in enumerate(sorted(roles)):
            S[k].append(f); Y[k].append(_ROLE_IDX[roles[tk]])
    hit = tot = 0
    for k in S:
        if k != 0 or k not in Ws:
            continue
        X = np.asarray(S[k]); y = np.asarray(Y[k])
        hit += int((np.argmax(X @ Ws[k], axis=1) == y).sum()); tot += len(y)
    silence_acc = float(hit / max(1, tot))

    hand_acc, hand_none = _hand_labeler_none(discovered, rng, subj, verb, obj, n=30)

    return {
        "seed": seed, "n_pool": lsm.n, "mean_spikes_per_neuron": mean_spikes,
        "train_acc": train_acc, "relhead": relhead, "relhead_gov": relhead_gov, "relhead_symwin": relhead_sym,
        "rel_full": rel_full, "relhead_scramble": relhead_scramble, "silence_lesion_acc": silence_acc,
        "hand_acc": hand_acc, "chance_binary": 0.5,
    }


def _demo(seed=42):
    print("\n=== EMERGE-82 -- ON-BRIDGE spiking LSM: the reservoir form->role mechanism as a RECURRENT REGION on a real "
          "SimulationBridge (the bridge's own Izhikevich + conductance synapses) ===\n", flush=True)
    d = _derisk_one(seed)
    print(f"  reservoir region {d['n_pool']} Izhikevich | mean spikes/neuron: {d['mean_spikes_per_neuron']:.2f}")
    print(f"  (A) CONSOLIDATION train role acc: {d['train_acc']:.3f}")
    print(f"  (B) NON-LOCAL rel-head: on-bridge-LSM {d['relhead']:.3f} vs gov {d['relhead_gov']:.3f} / symwin "
          f"{d['relhead_symwin']:.3f} (chance {d['chance_binary']:.2f})  [full rel {d['rel_full']:.3f}]")
    print(f"  rel-head scramble {d['relhead_scramble']:.3f} | REGION-SILENCE lesion {d['silence_lesion_acc']:.3f} | "
          f"hand {d['hand_acc']:.3f}\n")


def _derisk(seeds):
    print(f"EMERGE-82 de-risk: ON-BRIDGE spiking LSM (recurrent BrainRegion on a real SimulationBridge) of the EMERGE-78 "
          f"form->role reservoir; {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s); per.append(d)
            print(f"  [seed {s}] spikes {d['mean_spikes_per_neuron']:.2f} | train {d['train_acc']:.3f} | REL-HEAD "
                  f"{d['relhead']:.3f}/gov {d['relhead_gov']:.3f}/sym {d['relhead_symwin']:.3f} (full {d['rel_full']:.3f}) "
                  f"| scr {d['relhead_scramble']:.3f} | silence {d['silence_lesion_acc']:.3f} | hand {d['hand_acc']:.3f}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        train, relhead, gov, sym = m("train_acc"), m("relhead"), m("relhead_gov"), m("relhead_symwin")
        scramble, silence, spikes, rel_full, hand = m("relhead_scramble"), m("silence_lesion_acc"), \
            m("mean_spikes_per_neuron"), m("rel_full"), m("hand_acc")
        chanceb = per[0]["chance_binary"]

        active = spikes > 0.5
        consolidation_ok = train >= 0.90
        nonlocal_ok = (relhead >= 0.85 and gov <= 0.65 and sym <= 0.65)
        scramble_ok = scramble <= chanceb + 0.18
        silence_ok = (relhead - silence) >= 0.20
        go = bool(active and consolidation_ok and nonlocal_ok and scramble_ok and silence_ok)

        if go:
            verdict = (
                f"GO -- the reservoir form->role mechanism runs ON THE SUBSTRATE: a recurrent Izhikevich BrainRegion on a "
                f"real SimulationBridge (internal_density recurrent synapses + the bridge's own conductance-based "
                f"transmission + `_run_one_simulation_step`, genuinely active at {spikes:.2f} spikes/neuron) LEARNS the "
                f"full form->role map (train {train:.3f}) via a ridge read-out over its real cp_firing_states spike-counts, "
                f"AND resolves the non-local relative-clause HEAD ({relhead:.3f}) where BOTH the governing-cue baseline "
                f"({gov:.3f}) AND the symmetric +-2 window ({sym:.3f}) are at chance (~{chanceb:.2f}; full rel {rel_full:.3f}). "
                f"Controls: rel-head SCRAMBLE {scramble:.3f} -> reads structure; REGION-SILENCE lesion collapses the read "
                f"to {silence:.3f} (drop {relhead-silence:.3f}) -> genuinely from the region's SPIKES. Hand labeler {hand:.3f} "
                f"on the multi-arg shapes. {len(seeds)} seeds. ==> the EMERGE-80 spiking LSM is realized on the ONE BRAIN's "
                f"substrate (a bridge region, not a standalone numpy pool) -- the fully-spiking-one-brain directive. Co-"
                f"residence with the nav/conv regions on ONE shared bridge is the follow-on (the merge pattern is validated). "
                f"Reuse EMERGE-78 harness + EMERGE-61 wash-out + the brain-region framework; NO sim/ edit.")
        else:
            miss = []
            if not active:
                miss.append(f"the reservoir region is nearly SILENT ({spikes:.2f} spikes/neuron) -- the operating point "
                            f"(input/recurrent weights/bias) needs tuning")
            if not consolidation_ok:
                miss.append(f"the on-bridge region did NOT learn the form->role map (train {train:.3f} < 0.90)")
            if not nonlocal_ok:
                miss.append(f"the on-bridge region did NOT resolve the non-local rel-head (rel {relhead:.3f} vs gov {gov:.3f}"
                            f" / sym {sym:.3f}) -- the bridge's conductance-synapse recurrence integrates the sequence "
                            f"differently than the direct numpy pool")
            if not scramble_ok:
                miss.append(f"rel-head scramble {scramble:.3f} did not collapse")
            if not silence_ok:
                miss.append(f"REGION-SILENCE lesion did not collapse ({relhead:.3f} vs {silence:.3f})")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The ON-BRIDGE realization hits a deficit; the residual (region "
                       "operating point / recurrent weights / read-out) is the next single-variable de-risk. The numpy-pool "
                       "spiking LSM (EMERGE-80) stands; this characterizes the on-substrate realization. Do NOT force GO.")
    else:
        go = False; verdict = f"ERROR -- {err}"
        train = relhead = gov = sym = scramble = silence = spikes = rel_full = hand = None

    summary = {
        "probe": "emerge82_onbridge_lsm", "verdict": verdict, "go": bool(go) if err is None else False,
        "mechanism": ("realize the EMERGE-80 spiking liquid-state machine as a recurrent Izhikevich BrainRegion on a real "
                      "SimulationBridge (internal_density = the fixed-random recurrent synapses; the bridge's own "
                      "conductance-based transmission + _run_one_simulation_step; a fixed-random input projection drives "
                      "cp_external_input_current per token; the read-out feature = the region's cp_firing_states "
                      "spike-counts over the whole sequence; EMERGE-61 wash-out resets the bridge between sentences). "
                      "Mirrors the EMERGE-78 Reservoir API so it drops into the harness. Reuse-by-import; NO sim/ edit."),
        "task": ("does the reservoir form->role mechanism run ON the SimulationBridge substrate (a recurrent region) -- "
                 "learns the map (train) + resolves the non-local rel-head (vs both baselines at chance), region genuinely "
                 "active + region-silence-lesion collapse + scramble collapse; 6-seed; bridge on numpy/cupy"),
        "n_pool": _N_POOL, "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "mean_spikes_per_neuron": spikes, "train_acc": train, "relhead": relhead, "relhead_gov": gov,
            "relhead_symwin": sym, "rel_full": rel_full, "relhead_scramble": scramble, "silence_lesion_acc": silence,
            "hand_acc": hand,
        },
        "per_seed": per,
        "HONEST_NOTE": ("RUNG 2 of the spiking-reservoir port: the reservoir realized ON the SimulationBridge substrate (a "
                        "recurrent BrainRegion using the bridge's real Izhikevich + conductance synapses + step loop), not "
                        "a standalone numpy pool. GO = the mechanism survives on-bridge (learns the map + non-local, region "
                        "genuinely active + silence-lesion collapses). Reservoir region is a disjoint slice on its own "
                        "bridge here; co-residence with nav/conv on ONE shared bridge (the full one-brain merge) is the "
                        "follow-on (merge pattern validated in nav_conv_merged_bridge). Reuse EMERGE-78 harness + EMERGE-61 "
                        "wash-out + brain-region framework; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge82] VERDICT: {verdict}", flush=True)
    print(f"[emerge82] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds)
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
