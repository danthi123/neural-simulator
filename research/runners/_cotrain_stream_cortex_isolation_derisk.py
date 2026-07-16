"""2026-07-16 — the forward plan's named LONGEST POLE, cheap-first: SIMULTANEOUS stream-cortex co-training WITHOUT
cross-talk on ONE `SimulationBridge`. The plasticity-isolation gates are validated for FROZEN slices; the open frontier
(`docs/plans/2026-07-15-months-scale-plan-...` line 46) is whether TWO ACTIVELY-LEARNING stream cortices, time-shared
(interleaved) on one bridge, each learn their OWN co-occurrence structure without the other's global Hebbian plasticity
corrupting it. This is the one-brain-that-LEARNS integration step (directive-central, not scale-gated).

DESIGN (reuse-by-import of the validated CYCLE-95/96 on-bridge stream cortex):
  * ONE bridge, FOUR disjoint regions: hub_A, target_A (learner A's vocab half) + hub_B, target_B (learner B's other
    half), two disjoint plastic pathways hub_A->target_A, hub_B->target_B. Global rate-Hebbian ON (the shared rule).
  * INTERLEAVE the two streams: window from corpus-half A -> co-activate (hub_A, target_A); next window from half B ->
    (hub_B, target_B); alternating. Each pathway learns ONLY its own half's co-occurrence by coincidence.
  * GATE (6-seed): the CO-TRAINED code + corr(M,C) for EACH learner ≈ its SEPARATE-BRIDGE baseline (no degradation from
    co-residence), AND the cross-contamination corr(M_A, C_B) ≈ 0 (A did not learn B's structure).
  * CROSS-TALK POSITIVE CONTROL: a `--shared-target` variant where A and B write into the SAME target region -> they
    SHOULD interfere (corr drops / cross-contamination rises) -> proves the isolation metric can DETECT cross-talk.

numpy-CPU smoke / CuPy for scale; NO `sim/` edit (reuse the region framework + per-region plastic pathways).
Run: SIM_BACKEND=numpy python -u -m research.runners._cotrain_stream_cortex_isolation_derisk --seeds 42 --max-windows 4000
"""
import os, sys, time, json, argparse
import numpy as np
from collections import Counter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._phaseB_onbridge_stream_cortex_derisk import (
    load_token_stream, double_center, _cos_sim, _pearson_vs_Strue, heldout_generalization, STOPLIST, WINDOW,
    taxonomy_to_vocab_categories, TAXONOMY_8x8,
)
from sim.backend import to_host


def build_cotrain_bridge(NtA, NtB, n_hub, n_per, seed, shared_target=False, decay=0.00001, homeostasis=True):
    """ONE bridge with hub_A/target_A + hub_B/target_B (or a SHARED target if shared_target). Two plastic pathways."""
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    if shared_target:
        Nt = max(NtA, NtB)
        cfg.brain_regions = [
            BrainRegion(name="hubA", n_neurons=n_hub * n_per, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="hubB", n_neurons=n_hub * n_per, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="target", n_neurons=Nt * n_per, exc_fraction=1.0, internal_density=0.0),
        ]
        cfg.region_pathways = [
            RegionPathway(from_region="hubA", to_region="target", density=1.0, weight_mean=0.05, weight_jitter=0.0, plastic=True,
                          plasticity_gate="learnA"),
            RegionPathway(from_region="hubB", to_region="target", density=1.0, weight_mean=0.05, weight_jitter=0.0, plastic=True,
                          plasticity_gate="learnB"),
        ]
    else:
        cfg.brain_regions = [
            BrainRegion(name="hubA", n_neurons=n_hub * n_per, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="targetA", n_neurons=NtA * n_per, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="hubB", n_neurons=n_hub * n_per, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="targetB", n_neurons=NtB * n_per, exc_fraction=1.0, internal_density=0.0),
        ]
        cfg.region_pathways = [
            RegionPathway(from_region="hubA", to_region="targetA", density=1.0, weight_mean=0.05, weight_jitter=0.0, plastic=True,
                          plasticity_gate="learnA"),
            RegionPathway(from_region="hubB", to_region="targetB", density=1.0, weight_mean=0.05, weight_jitter=0.0, plastic=True,
                          plasticity_gate="learnB"),
        ]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_homeostasis = homeostasis
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.03; cfg.hebbian_max_weight = 5.0; cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = decay
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    idx = {r: np.asarray(bridge.region_manager.indices(r)) for r in [rg.name for rg in cfg.brain_regions]}
    return bridge, idx


def _learn_code(bridge, hub_region, tgt_region, n_hub, n_per, Nt):
    """READ: population block-mean of the learned hub->target weights -> M[target, hub], log-double-centred code."""
    W = np.asarray(to_host(bridge.cp_connections.todense())).astype(np.float64)
    blk = W[np.ix_(hub_region, tgt_region)].reshape(n_hub, n_per, Nt, n_per).mean(axis=(1, 3))
    M = blk.T
    return M, double_center(np.log1p(M * 100.0))


def run_seed(seed, stories, vocab, cat_ids, a, mode="cotrain"):
    """mode: 'cotrain' (A+B interleaved on one bridge) / 'separateA' / 'separateB' (baseline) / 'shared' (cross-talk +ctrl)."""
    rng = np.random.RandomState(seed)
    targets = list(vocab); cat = np.asarray(cat_ids)
    # split vocab into two disjoint halves BY CATEGORY (A = even categories, B = odd) -> two distinct co-occurrence blocks
    cats = sorted(set(cat_ids)); catA = set(cats[0::2]);
    isA = np.array([c in catA for c in cat_ids])
    tA = [w for w, m in zip(targets, isA) if m]; tB = [w for w, m in zip(targets, isA) if not m]
    catA_ids = [c for c, m in zip(cat_ids, isA) if m]; catB_ids = [c for c, m in zip(cat_ids, isA) if not m]
    NtA, NtB = len(tA), len(tB); n_hub, n_per = a.n_hub, a.n_per
    rowA = {w: i for i, w in enumerate(tA)}; rowB = {w: i for i, w in enumerate(tB)}
    setA, setB = set(tA), set(tB)
    gfreq = Counter()
    for toks in stories: gfreq.update(toks)
    hubs = [w for w, _ in gfreq.most_common() if w not in STOPLIST and w not in setA and w not in setB][:n_hub]
    hubidx = {w: i for i, w in enumerate(hubs)}; keep = setA | setB | set(hubs)

    shared = (mode == "shared")
    bridge, idx = build_cotrain_bridge(NtA, NtB, n_hub, n_per, seed, shared_target=shared, decay=a.hebbian_decay, homeostasis=a.homeostasis)
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    tgtA_name = "target" if shared else "targetA"; tgtB_name = "target" if shared else "targetB"
    hubA_r, tgtA_r = idx["hubA"], idx[tgtA_name]; hubB_r, tgtB_r = idx["hubB"], idx[tgtB_name]
    NtT = max(NtA, NtB) if shared else None

    def present(hub_region, tgt_region, n_tgt_total, hub_ids, tgt_ids):
        hub_full = np.zeros(n_hub * n_per, np.float32); tgt_full = np.zeros(n_tgt_total, np.float32)
        for h in hub_ids: hub_full[h * n_per:(h + 1) * n_per] = a.hub_scale
        for t in tgt_ids: tgt_full[t * n_per:(t + 1) * n_per] = a.tgt_scale
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[hub_region] = xp.asarray(hub_full) if xp is not None else hub_full
        bridge.cp_external_input_current[tgt_region] = xp.asarray(tgt_full) if xp is not None else tgt_full
        for _ in range(a.window_steps): bridge._run_one_simulation_step()

    CA = np.zeros((NtA, n_hub)); CB = np.zeros((NtB, n_hub))
    nA = shared and NtT * n_per or NtA * n_per; nB = shared and NtT * n_per or NtB * n_per
    story_order = rng.permutation(len(stories)); nwin = 0; t0 = time.time()
    # build the interleaved window list: (which, tgt_ids, hub_ids)
    wins = []
    for si in story_order:
        kept = [t for t in stories[si] if t in keep]
        for c in range(len(kept)):
            lo, hi = max(0, c - WINDOW), min(len(kept), c + WINDOW + 1); win = kept[lo:hi]
            hub_ids = [hubidx[w] for w in win if w in hubidx]
            aids = [rowA[w] for w in win if w in setA]; bids = [rowB[w] for w in win if w in setB]
            if hub_ids and aids: wins.append(("A", aids, hub_ids))
            if hub_ids and bids: wins.append(("B", bids, hub_ids))
    rng.shuffle(wins)                                   # interleave A/B streams
    # PER-LEARNER window budget (each learner gets max_windows) so co-training and separate give each the SAME
    # data budget -> the co-vs-sep comparison isolates CROSS-TALK, not a halved data budget (the design-confound fix).
    # --gate-plasticity: freeze the IDLE learner's pathway during the other's window (attentional / neuromodulatory
    # gating of plasticity to the attended stream). Tests the SPURIOUS-LTP cross-talk vector that the decay and
    # homeostasis probes could not: present() zeroes external input but STILL steps the WHOLE bridge, so during a
    # B-window the global Hebbian rule keeps running on hubA->targetA against A's residual (membrane/conductance/NMDA)
    # activity -- pairing A's synapses with content that has nothing to do with them. That asymmetry exists ONLY in the
    # co-trained arm (separate arms present their own windows back-to-back).
    # NOTE the control arms are UNAFFECTED by construction: in separateA only A-windows are presented, so learnA is
    # held at 1.0 throughout == ungated. So this flag isolates the co-trained arm -- a clean single variable.
    _gate = bool(getattr(a, "gate_plasticity", 0))
    def set_gates(active):
        # tolerate a gate that never registered: a pathway with no synapses (e.g. a degenerate tiny-vocab config
        # where one half's target region is empty) declares no gate, and set_plasticity_gate would KeyError.
        if not _gate: return
        for nm, act in (("learnA", "A"), ("learnB", "B")):
            try:
                bridge.set_plasticity_gate(nm, 1.0 if active == act else 0.0)
            except KeyError:
                pass

    # --idle-match: TIMING-MATCH the separate baseline to the co-trained arm.
    # THE CONTROL CONFOUND (found by reading this loop, 2026-07-16): in `separateA` the B-windows are SKIPPED
    # ENTIRELY -- no steps run -- so A's windows are BACK-TO-BACK. In `cotrain` every A-window is separated by
    # window_steps of B-window, during which A's neurons decay with ZERO external input. So a co-trained learner
    # starts each of its windows from a COLDER (more-decayed) membrane/conductance state than its own baseline ever
    # does. That is a TIMING asymmetry between treatment and control, not cross-talk between the learners --
    # and it is the same CLASS of confound as the already-fixed data-budget bug (a baseline that does not match
    # the treatment arm). It also explains BOTH prior refutations: it is not weight decay (so --hebbian-decay 0
    # could not touch it) and not threshold drift (so --homeostasis 0 cannot either).
    # With --idle-match 1 the separate arm runs the SAME number of idle steps the co-trained arm would have spent
    # on the other learner -- zero input, nothing presented, nothing counted, no window budget consumed.
    # If the gap closes => the residual was a baseline artifact and co-training cross-talk is ~FREE.
    _idle = bool(getattr(a, "idle_match", 0))
    def idle_steps():
        if not _idle: return
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(a.window_steps): bridge._run_one_simulation_step()

    nwinA = nwinB = 0
    for which, tgt_ids, hub_ids in wins:
        if nwinA >= a.max_windows and nwinB >= a.max_windows: break
        # in a SEPARATE arm, the other learner's window is where the co-trained arm would have spent idle steps
        if _idle and mode == "separateA" and which == "B" and nwinB < a.max_windows:
            idle_steps(); nwinB += 1; continue
        if _idle and mode == "separateB" and which == "A" and nwinA < a.max_windows:
            idle_steps(); nwinA += 1; continue
        if which == "A" and nwinA < a.max_windows and mode in ("cotrain", "separateA", "shared"):
            set_gates("A")
            present(hubA_r, tgtA_r, nA, hub_ids, tgt_ids)
            for t in tgt_ids:
                for h in hub_ids: CA[t, h] += 1.0
            nwinA += 1
        elif which == "B" and nwinB < a.max_windows and mode in ("cotrain", "separateB", "shared"):
            set_gates("B")
            present(hubB_r, tgtB_r, nB, hub_ids, tgt_ids)
            for t in tgt_ids:
                for h in hub_ids: CB[t, h] += 1.0
            nwinB += 1
    nwin = nwinA + nwinB
    bridge.cp_external_input_current[:] = 0.0

    def metrics(hub_r, tgt_r, Nt, C, catids):
        Nt_region = len(tgt_r) // n_per                       # actual region target count (shared: max(NtA,NtB))
        M, code = _learn_code(bridge, hub_r, tgt_r, n_hub, n_per, Nt_region)
        M = M[:Nt]; code = code[:Nt]                          # slice to this learner's own targets (no-op when disjoint)
        S = (np.asarray(catids)[:, None] == np.asarray(catids)[None, :]).astype(np.float64)
        p = _pearson_vs_Strue(_cos_sim(code), S)
        mc = float(np.corrcoef(M.flatten(), C.flatten())[0, 1]) if M.std() > 0 else 0.0
        return M, p, mc
    out = {"seed": seed, "mode": mode, "n_windows": nwin, "NtA": NtA, "NtB": NtB}
    if mode in ("cotrain", "separateA", "shared"):
        MA, pA, mcA = metrics(hubA_r, tgtA_r, NtA, CA, catA_ids); out.update(codeA=round(pA, 4), corrA=round(mcA, 4))
    if mode in ("cotrain", "separateB", "shared"):
        MB, pB, mcB = metrics(hubB_r, tgtB_r, NtB, CB, catB_ids); out.update(codeB=round(pB, 4), corrB=round(mcB, 4))
    print(f"[cotrain {mode} s{seed}] {nwin}w ({time.time()-t0:.0f}s) NtA={NtA} NtB={NtB} | "
          + " ".join(f"{k}={out[k]}" for k in out if k in ("codeA", "corrA", "codeB", "corrB")), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()   # defaults MATCH the validated phaseB stream cortex (corr(M,C) +0.705)
    p.add_argument("--seeds", default="42"); p.add_argument("--n-hub", type=int, default=300)
    p.add_argument("--n-per", type=int, default=16); p.add_argument("--window-steps", type=int, default=2)
    p.add_argument("--hub-scale", type=float, default=250.0); p.add_argument("--tgt-scale", type=float, default=1200.0)
    p.add_argument("--max-windows", type=int, default=24000); p.add_argument("--max-vocab", type=int, default=64)
    p.add_argument("--hebbian-decay", type=float, default=0.00001)
    p.add_argument("--homeostasis", type=int, default=1)
    p.add_argument("--gate-plasticity", type=int, default=0, dest="gate_plasticity",
                   help="freeze the IDLE learner's pathway during the other's window (spurious-LTP probe); 0 = off (byte-identical)")
    p.add_argument("--idle-match", type=int, default=0, dest="idle_match",
                   help="TIMING-MATCH the separate baseline: run the same idle steps the co-trained arm spends on the "
                        "other learner (zero input, nothing presented/counted). Tests whether the residual is a "
                        "baseline artifact rather than cross-talk. 0 = off (byte-identical)")
    p.add_argument("--corpus", default=None); p.add_argument("--out", default="research/findings/raw/_cotrain_stream_isolation.json")
    a = p.parse_args()
    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_8x8)
    stories = load_token_stream(a.corpus)
    if len(vocab) > a.max_vocab:
        vocab = list(vocab)[:a.max_vocab]; cat_ids = list(cat_ids)[:a.max_vocab]
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        co = run_seed(s, stories, vocab, cat_ids, a, mode="cotrain")
        sepA = run_seed(s, stories, vocab, cat_ids, a, mode="separateA")
        sepB = run_seed(s, stories, vocab, cat_ids, a, mode="separateB")
        shr = run_seed(s, stories, vocab, cat_ids, a, mode="shared")
        # GO: co-trained corr ~= separate-bridge corr (no degradation from co-residence) for BOTH learners,
        #     AND the shared-target cross-talk +control DEGRADES (proves the metric detects cross-talk when regions overlap).
        dA = co["corrA"] - sepA["corrA"]; dB = co["corrB"] - sepB["corrB"]
        shared_degrades = (shr["corrA"] < sepA["corrA"] - 0.05) or (shr["corrB"] < sepB["corrB"] - 0.05)
        go = (dA > -0.08) and (dB > -0.08) and shared_degrades
        row = {"seed": s, "cotrain": co, "sepA": sepA, "sepB": sepB, "shared": shr,
               "dA_vs_sep": round(dA, 4), "dB_vs_sep": round(dB, 4), "shared_degrades": bool(shared_degrades), "GO": bool(go)}
        rows.append(row)
        print(f"  ==> seed {s}: co corrA/B {co['corrA']}/{co['corrB']} vs sep {sepA['corrA']}/{sepB['corrB']} "
              f"(dA {dA:+.3f} dB {dB:+.3f}) | shared-ctrl corrA/B {shr['corrA']}/{shr['corrB']} degrades={shared_degrades} | GO={go}", flush=True)
    print(f"[cotrain] {sum(r['GO'] for r in rows)}/{len(rows)} GO (simultaneous stream-cortex co-training WITHOUT cross-talk)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
