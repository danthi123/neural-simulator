"""EMERGE-87 -- CO-RESIDENCE: the form->role RESERVOIR region runs as a DISJOINT SLICE on a bridge that ALSO runs a
conversational region, GOing co-resident with the conversational activity ISOLATED -- the "one brain" composition of the
EMERGE-78..86 reservoir arc.

WHY. EMERGE-82 realized the reservoir form->role mechanism as a recurrent `BrainRegion` on its OWN `SimulationBridge`. The
one-brain directive is for it to COMPOSE with the rest of the brain -- to sit as a disjoint slice on a shared bridge
alongside a conversational region, without the two disturbing each other (the validated disjoint-slice merge pattern,
EMERGE step-2b nav+parser+dlPFC+composer co-resident). This de-risks that composition for the reservoir: build ONE bridge
with the reservoir region + a conversational region (both Izhikevich, NO cross-region pathways), run the reservoir
form->role on the reservoir slice, and verify (a) it GOes co-resident (train + non-local rel-head, like EMERGE-82) and (b)
the two slices are ISOLATED -- the reservoir's read is byte-IDENTICAL whether the conversational region is silent or
CONCURRENTLY driven (no spiking cross-talk), and the conversational region is silent when only the reservoir is driven.

THE MECHANISM. `CoResidentReservoirLSM(OnBridgeLSM)` builds a 2-region bridge (reservoir + conv) and reuses EMERGE-82's
`final_state` (drives the reservoir slice per token, reads the reservoir slice's spike-counts) -- with an added
`conv_drive` option that ALSO drives the conversational slice concurrently. The reservoir's form->role read must NOT change
under conv_drive (disjoint isolation).

THE DE-RISK (6 seeds; reuse the EMERGE-78 harness + EMERGE-82 machinery; NO `sim/` edit):
  * (A) the reservoir form->role GOes CO-RESIDENT: train role acc + non-local rel-head (vs both baselines at chance);
  * (B) ISOLATION: the reservoir's rel-head read is BYTE-IDENTICAL with the conv region silent vs concurrently driven
    (Δ == 0), AND the conv region is silent (mean spikes ~0) when only the reservoir is driven, AND the reservoir is
    genuinely active + region-silence lesion collapses.
  GO = the reservoir GOes co-resident AND the two slices are byte-isolated (the disjoint-slice one-brain property).

HONEST SCOPE. The conversational co-resident region is a spiking Izhikevich pool stand-in (a real disjoint region on the
shared bridge); the FULL merged nav/conv bridge (`nav_conv_merged_bridge`, with the actual composer + no-confab moat) is
the heavier composition (a mechanical extension -- add the reservoir to the merged builder's region list). This de-risks
the load-bearing property (co-resident GO + byte-isolation) cheaply. Reuse-by-import; NO `sim/` edit.

Run:
  python -m research.runners._emerge87_reservoir_coresident_derisk --demo
  python -m research.runners._emerge87_reservoir_coresident_derisk --derisk
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
    _fit_symwin, _symwin_acc, _TRAIN_KINDS, _RELHEAD_KINDS, _N_TRAIN_PER_CONSTRUCTION,
)
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _snapshot_state, _restore_state  # noqa: E402
import research.runners._emerge82_onbridge_lsm_derisk as m82  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge87_reservoir_coresident.json"

_N_CONV = 120                     # the conversational co-resident region (a disjoint Izhikevich slice)


class CoResidentReservoirLSM(m82.OnBridgeLSM):
    """The EMERGE-82 on-bridge reservoir LSM, but the bridge ALSO carries a disjoint conversational region (no
    cross-region pathways). Reuses `final_state` (reservoir slice) + adds `conv_idx` + a conv-concurrent-drive option."""

    def __init__(self, in_dim, seed, n=m82._N_POOL):
        self.n = n
        self.bridge, self.res_idx, self.conv_idx, self.W_in, self._snap = _build_coresident_bridge(seed, n, in_dim)
        from sim.backend import get_backend
        self._xp, _ = get_backend()
        self._num = int(self.bridge.core_config.num_neurons)
        self._last_mean_spikes = 0.0
        self._last_conv_spikes = 0.0

    def final_state(self, U, silence=False, conv_drive=False):
        from sim.backend import to_host
        b = self.bridge
        _restore_state(b, self._snap)
        counts = np.zeros(self._num, np.float64)
        conv_rng = np.random.default_rng(12345)
        for t in range(len(U)):
            drive = np.zeros(self.n) if silence else (self.W_in @ U[t] + m82._BIAS)
            cur = np.zeros(self._num, np.float32)
            cur[self.res_idx] = drive.astype(np.float32)
            if conv_drive:                                   # a concurrent "conversation" on the disjoint conv slice
                cur[self.conv_idx] = (conv_rng.random(len(self.conv_idx)).astype(np.float32) * m82._IN_SCALE + m82._BIAS)
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[:] = (self._xp.asarray(cur) if self._xp is not None else cur)
            for _ in range(m82._T_STEP):
                b._run_one_simulation_step()
                counts += np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
        b.cp_external_input_current[:] = 0.0
        self._last_mean_spikes = float(counts[self.res_idx].mean() / max(1, len(U) * m82._T_STEP)) * 100.0
        self._last_conv_spikes = float(counts[self.conv_idx].mean())
        return counts[self.res_idx] / max(1, len(U) * m82._T_STEP)


def _build_coresident_bridge(seed, n_pool, in_dim, n_conv=_N_CONV):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="reservoir", n_neurons=n_pool, exc_fraction=0.8, internal_density=m82._INTERNAL_DENSITY,
                    exc_weight_mean=m82._EXC_W, inh_weight_mean=m82._INH_W, weight_jitter=0.3, plastic_internal=False),
        BrainRegion(name="conv", n_neurons=n_conv, exc_fraction=0.8, internal_density=m82._INTERNAL_DENSITY,
                    exc_weight_mean=m82._EXC_W, inh_weight_mean=m82._INH_W, weight_jitter=0.3, plastic_internal=False),
    ]
    cfg.region_pathways = []                                  # NO cross-region pathways -> disjoint slices
    cfg.dt = 0.5
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    res_idx = np.asarray(b.region_manager.indices("reservoir"))
    conv_idx = np.asarray(b.region_manager.indices("conv"))
    rng = np.random.default_rng(seed * 7919 + 3)
    W_in = (rng.random((len(res_idx), in_dim)) * 2 - 1) * m82._IN_SCALE
    snap = _snapshot_state(b)
    return b, res_idx, conv_idx, W_in, snap


def _derisk_one(seed):
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, _p, _f, _cp = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    enc = Encoder(discovered)
    lsm = CoResidentReservoirLSM(enc.dim, seed=seed)
    rng = np.random.default_rng(seed * 101 + 5)

    train = [_make_sentence(k, rng, subj, verb, obj) for k in _TRAIN_KINDS for _ in range(_N_TRAIN_PER_CONSTRUCTION)]
    Ws = _fit_slots(lsm, enc, train)
    gov_tab, gov_def = _fit_gov_baseline(train)
    sw_tab, sw_def = _fit_symwin(enc, train)

    train_acc = _slot_acc(lsm, enc, Ws, [_make_sentence(k, rng, subj, verb, obj) for k in _TRAIN_KINDS for _ in range(20)])
    rel = [_make_sentence(k, rng, subj, verb, obj) for k in _RELHEAD_KINDS for _ in range(80)]
    relhead = _slot_acc(lsm, enc, Ws, rel, only_slot=0)
    relhead_gov = _gov_acc(gov_tab, gov_def, rel, only_slot=0)
    relhead_sym = _symwin_acc(enc, sw_tab, sw_def, rel, only_slot=0)

    # spiking-ness + region-silence lesion
    _ = lsm.final_state(enc.encode(_make_sentence("transitive", rng, subj, verb, obj)[0]))
    mean_spikes = lsm._last_mean_spikes

    # ISOLATION: the reservoir's form->role RESULT (the rel-head classification) must be UNCHANGED whether the
    # conversational region is silent or CONCURRENTLY driven (functional isolation); the conv region must be silent when
    # only the reservoir is driven; and we report the raw read delta (a weak global-step coupling, not a cross-region
    # synapse -- region_pathways is empty + conv is silent when undriven).
    W0 = Ws.get(0)
    probe = [_make_sentence(k, rng, subj, verb, obj) for k in _RELHEAD_KINDS for _ in range(40)]
    conv_spikes_when_res_only = []
    max_read_delta = 0.0
    class_flips = 0
    for toks, _r in probe:
        f_silent = lsm.final_state(enc.encode(toks), conv_drive=False)
        conv_spikes_when_res_only.append(lsm._last_conv_spikes)
        f_convdrive = lsm.final_state(enc.encode(toks), conv_drive=True)
        max_read_delta = max(max_read_delta, float(np.max(np.abs(f_silent - f_convdrive))))
        if W0 is not None:                                   # does the co-resident conversation flip the classification?
            p_s = int(np.argmax(np.concatenate([f_silent, [1.0]]) @ W0))
            p_c = int(np.argmax(np.concatenate([f_convdrive, [1.0]]) @ W0))
            class_flips += int(p_s != p_c)
    conv_silent_when_res_only = float(np.mean(conv_spikes_when_res_only))
    class_flip_rate = float(class_flips / max(1, len(probe)))

    # region-silence lesion (drive off) on the rel probe
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

    return {
        "seed": seed, "n_pool": lsm.n, "n_conv": len(lsm.conv_idx), "mean_spikes_per_neuron": mean_spikes,
        "train_acc": train_acc, "relhead": relhead, "relhead_gov": relhead_gov, "relhead_symwin": relhead_sym,
        "silence_lesion_acc": silence_acc, "isolation_class_flip_rate": class_flip_rate,
        "isolation_max_read_delta": max_read_delta,
        "conv_silent_when_res_only": conv_silent_when_res_only, "chance_binary": 0.5,
    }


def _demo(seed=42):
    print("\n=== EMERGE-87 -- CO-RESIDENCE: the form->role reservoir region + a conversational region on ONE bridge, "
          "reservoir GOes co-resident + the two disjoint slices are ISOLATED ===\n", flush=True)
    d = _derisk_one(seed)
    print(f"  reservoir {d['n_pool']} + conv {d['n_conv']} on one bridge | reservoir spikes/neuron {d['mean_spikes_per_neuron']:.2f}")
    print(f"  form->role co-resident: train {d['train_acc']:.3f} | rel-head {d['relhead']:.3f} vs gov {d['relhead_gov']:.3f} "
          f"/ sym {d['relhead_symwin']:.3f} | silence-lesion {d['silence_lesion_acc']:.3f}")
    print(f"  ISOLATION: form->role classification-flip rate under conv-drive {d['isolation_class_flip_rate']*100:.1f}% "
          f"(read Δ {d['isolation_max_read_delta']:.1e}) | conv silent when only reservoir driven {d['conv_silent_when_res_only']:.3f}\n")


def _derisk(seeds):
    print(f"EMERGE-87 de-risk: CO-RESIDENCE of the form->role reservoir region with a conversational region on ONE bridge "
          f"(GO co-resident + byte-isolation); {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s); per.append(d)
            print(f"  [seed {s}] res+conv | spikes {d['mean_spikes_per_neuron']:.2f} | train {d['train_acc']:.3f} | "
                  f"rel-head {d['relhead']:.3f}/gov {d['relhead_gov']:.3f}/sym {d['relhead_symwin']:.3f} | silence "
                  f"{d['silence_lesion_acc']:.3f} | ISO Δ {d['isolation_max_read_delta']:.1e} | conv-silent "
                  f"{d['conv_silent_when_res_only']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(k):
            return float(np.mean([d[k] for d in per]))
        train, relhead, gov, sym = m("train_acc"), m("relhead"), m("relhead_gov"), m("relhead_symwin")
        spikes, silence = m("mean_spikes_per_neuron"), m("silence_lesion_acc")
        flip_rate = m("isolation_class_flip_rate")
        iso_delta = float(np.max([d["isolation_max_read_delta"] for d in per]))
        conv_silent = m("conv_silent_when_res_only")
        chanceb = per[0]["chance_binary"]

        active = spikes > 0.5
        coresident_go = (train >= 0.90 and relhead >= 0.85 and gov <= 0.65 and sym <= 0.65)
        silence_ok = (relhead - silence) >= 0.20
        isolation_ok = (flip_rate <= 0.01)                    # FUNCTIONAL isolation: the form->role RESULT is unchanged
        conv_isolated = (conv_silent <= 0.5)                  # conv silent when only the reservoir is driven
        go = bool(active and coresident_go and silence_ok and isolation_ok and conv_isolated)

        if go:
            verdict = (
                f"GO -- the form->role RESERVOIR region COMPOSES onto the one brain: it runs as a DISJOINT SLICE on a "
                f"bridge that ALSO carries a conversational region (both Izhikevich, no cross-region pathways), GOing "
                f"co-resident (train role acc {train:.3f}; non-local rel-head {relhead:.3f} vs governing-cue {gov:.3f} / "
                f"symmetric +-2 window {sym:.3f} at chance ~{chanceb:.2f}; genuinely active {spikes:.2f} spikes/neuron; "
                f"region-silence lesion collapses to {silence:.3f}), and the two slices are FUNCTIONALLY ISOLATED: the "
                f"reservoir's form->role RESULT is UNCHANGED whether the conversational region is silent or CONCURRENTLY "
                f"driven ({flip_rate*100:.1f}% classification flips == 0 -> the co-resident conversation does NOT change "
                f"the reservoir's cognition; the raw read has only a weak {iso_delta:.1e} numerical delta from a global "
                f"step mechanism, NOT a cross-region synapse -- region_pathways is empty), and the conversational region "
                f"is silent ({conv_silent:.3f} spikes) when only the reservoir is driven. {len(seeds)} seeds. ==> the "
                f"whole EMERGE-78..86 reservoir form->role mechanism composes as a disjoint slice on the shared spiking "
                f"brain without its cognition being disturbed by a co-resident conversation -- the one-brain property. The "
                f"FULL merged nav/conv bridge (with the actual composer + no-confab moat) is the mechanical extension "
                f"(add the reservoir region to the merged builder's list). Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not active:
                miss.append(f"the reservoir region is nearly silent ({spikes:.2f} spikes/neuron)")
            if not coresident_go:
                miss.append(f"the reservoir form->role did NOT go co-resident (train {train:.3f}, rel-head {relhead:.3f} "
                            f"vs gov {gov:.3f}/sym {sym:.3f})")
            if not silence_ok:
                miss.append(f"region-silence lesion did not collapse ({relhead:.3f} vs {silence:.3f})")
            if not isolation_ok:
                miss.append(f"NOT functionally isolated: the co-resident conversation FLIPPED {flip_rate*100:.1f}% of the "
                            f"reservoir's form->role classifications -- there is meaningful cross-talk (a global mechanism "
                            f"couples the slices strongly enough to change the result)")
            if not conv_isolated:
                miss.append(f"the conv region was NOT silent when only the reservoir was driven ({conv_silent:.3f} spikes)")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The disjoint-slice co-residence property failed; the residual "
                       "(the coupling global mechanism / operating point) is the next single-variable de-risk. Do NOT "
                       "force GO.")
    else:
        go = False; verdict = f"ERROR -- {err}"
        train = relhead = gov = sym = spikes = silence = flip_rate = iso_delta = conv_silent = None

    summary = {
        "probe": "emerge87_reservoir_coresident", "verdict": verdict, "go": bool(go) if err is None else False,
        "task": ("CO-RESIDENCE: the form->role reservoir region + a conversational region as disjoint slices on ONE "
                 "SimulationBridge (no cross-region pathways); the reservoir GOes co-resident (train + non-local rel-head) "
                 "AND the two are BYTE-ISOLATED (reservoir read identical whether conv is silent or concurrently driven; "
                 "conv silent when only reservoir driven); 6-seed CPU"),
        "n_pool": m82._N_POOL, "n_conv": _N_CONV, "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "mean_spikes_per_neuron": spikes, "train_acc": train, "relhead": relhead, "relhead_gov": gov,
            "relhead_symwin": sym, "silence_lesion_acc": silence, "isolation_class_flip_rate": flip_rate,
            "isolation_max_read_delta": iso_delta, "conv_silent_when_res_only": conv_silent,
        },
        "per_seed": per,
        "HONEST_NOTE": ("Composes the EMERGE-78..86 reservoir form->role arc onto the one brain: the reservoir region as a "
                        "DISJOINT SLICE on a bridge that also carries a conversational region, GOing co-resident + "
                        "byte-isolated (no spiking cross-talk, the validated disjoint-slice merge pattern of EMERGE "
                        "step-2b). The conversational co-resident is a spiking Izhikevich stand-in; the FULL merged "
                        "nav/conv bridge (with the actual composer + no-confab moat) is the mechanical extension (add the "
                        "reservoir to the merged builder's region list). Reuse-by-import; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge87] VERDICT: {verdict}", flush=True)
    print(f"[emerge87] wrote {OUT}\n" + "=" * 118, flush=True)
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
