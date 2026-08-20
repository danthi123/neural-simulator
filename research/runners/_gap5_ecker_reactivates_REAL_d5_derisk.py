"""STEP 1 of the banked D5-integration plan: close the proxy->real gap. Does the Ecker AdEx CA3 SWR replay
REACTIVATE a GENUINELY-STORED D5 assembly by riding its REAL BTSP-grown within-assembly recurrence?

CONTEXT (read research/findings/2026-08-20-ecker-replay-into-D5-integration-FEASIBLE-by-composition-not-replacement.md):
  The feasibility probe (_gap5_ecker_reactivates_d5_stored_assembly_derisk.py) used a STRONG-RECURRENCE PROXY store --
  a fresh AdEx `build_store` with contiguous 20-cell assemblies and a hand-set uniform within-weight (w_stored=60,
  w_unstored=8). It showed the Ecker AdEx substrate supplies the DISCRETE, self-terminating, assembly-SPECIFIC,
  recurrence-riding reactivation D5's persistent dendritic-dAP latch could not. THE RESIDUAL it left open: is that a
  property of the PROXY store, or does it TRANSFER to the REAL organ? This runner closes that gap.

WHAT IS DIFFERENT HERE (the whole point): the store is a GENUINE production D5 `EpisodicDapMemory` (n_ca3=2000). We
  * STORE 'dog' the real way -- its BTSP one-shot encode (`mem.store('dog')`), forming a ~13-cell emergent DG-selected
    assembly with HETEROGENEOUS BTSP-grown within-recurrence (mean ~84, range ~49-100 vs the never-formed baseline ~1.5);
  * identify 'cat' as a NEVER-formed control (its within-recurrence stays at the baseline ~1.5);
  * EXTRACT dog's real membership (which CA3 cells) + its real per-synapse potentiated within-recurrence weights, and
    cat's membership + its real never-formed baseline weights;
  * MAP that membership + those weights onto an AdEx `ADEX_ECKER_CA3_PC` bridge as an EXACT COPY (same within-assembly
    connectivity graph, same per-synapse weights -- NOT a fresh `build_store`, NOT a uniform hand-set weight, NOT tuned
    to make it work). Cells are relabeled to a compact index space (an isomorphic copy: relabeling graph nodes changes
    no biology), and the AdEx substrate + SWR envelope are byte-identical to the feasibility probe's `reactivate`.

Then run the SAME reactivation probe (partial cue under the SWR envelope) on the REAL dog assembly vs the never-formed
cat, measuring the four properties + the recurrence-lesion teeth.

THE VERDICT (per the scoping finding, full held-out COMPLETION at 20-cell sparsity is EXPECTED-weak and is NOT the bar;
the load-bearing question for learn-through-use / BTSP is whether the real stored members CO-FIRE discretely + specifically):
  * CO-FIRE   : the STORED dog assembly, partial-cued under the SWR envelope, drives its HELD-OUT members to fire
                (held co-firing >= COFIRE_MIN) -- BTSP potentiation needs the members to co-activate.
  * DISCRETE  : the reactivation SELF-TERMINATES (tail/peak population activity low) -- AdEx adaptation gives the SWR
                transient D5's persistent KIR latch could not.
  * SPECIFIC  : the never-formed cat assembly, IDENTICAL cue dose, does NOT co-fire its held-out members (dog >> cat).
  * TEETH     : weaken dog's REAL within-recurrence to its never-formed baseline -> co-firing collapses (proves it is
                carried by the real stored recurrence, not raw cue excitability).
  * NOCUE-SILENT: envelope + OU only (no cue) -> dog does not co-fire (the drive is a partial cue, not free noise).
GO = CO-FIRE and DISCRETE and SPECIFIC and TEETH and NOCUE-SILENT. Honest NEGATIVE if the REAL store does not reactivate
(that localizes the next residual: the weight-scale interface between D5's readout and the AdEx substrate).

Reuse-by-import: `reactivate` + `_global_cells` (the feasibility probe's AdEx SWR reactivation instrument, byte-identical
substrate) and `_smooth`. NO `sim/` edit. GPU-preferred (small AdEx net -> fast; the D5 build is ~min-scale).

  Run:    SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_reactivates_REAL_d5_derisk \
              --seed 42 --out research/findings/raw/_ecker_reactivates_real_d5/seed42.json
  6-seed: SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_ecker_reactivates_REAL_d5_derisk \
              --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from sim.enums import NeuronModel, NeuronType  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402
# the AdEx SWR reactivation instrument (byte-identical substrate to the feasibility probe):
from research.runners._gap5_ecker_reactivates_d5_stored_assembly_derisk import reactivate, _global_cells  # noqa: E402
from research.runners._episodic_dap_dialogue_memory import EpisodicDapMemory  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_ecker_reactivates_real_d5" / "seed42.json"

# --- GO thresholds. CO-FIRE (not full completion) is the bar per the scoping finding. -------------------------------
COFIRE_MIN = 0.15          # held-out member co-firing FRACTION that counts as "reactivated" (BTSP needs co-activation)
DOG_OVER_CAT = 3.0         # specificity: dog co-firing >= 3x cat co-firing
TERMINATE_MAX = 0.30       # tail population activity (last 20%) / peak -- discreteness / self-termination


# ----------------------------------------------------------------------------------------------------------------------
# EXTRACT: build a genuine D5 store, store 'dog', pull dog's real membership + potentiated within-recurrence and cat's
# never-formed baseline. rows=pre, cols=post (CSR convention; sim applies connections.T @ drive so col=post).
# ----------------------------------------------------------------------------------------------------------------------
def extract_real_d5(seed, *, verbose=False):
    mem = EpisodicDapMemory(seed, topics=["cat", "dog"], verbose=verbose)
    stored = mem.store("dog")           # BTSP one-shot encode -- the REAL store write
    if not stored:
        raise RuntimeError("mem.store('dog') returned False -- dog was not BTSP-formed")
    rec_dog = mem.recall("dog"); rec_cat = mem.recall("cat")

    dog_slot = mem.topic_slot["dog"]; cat_slot = mem.topic_slot["cat"]
    dog_cells = np.asarray(mem.assemblies[dog_slot], dtype=np.int64)
    cat_cells = np.asarray(mem.assemblies[cat_slot], dtype=np.int64)

    R = mem.R
    rows = np.asarray(to_host(R.rows), dtype=np.int64)      # pre
    cols = np.asarray(to_host(R.cols), dtype=np.int64)      # post
    data = np.asarray(to_host(R.C.data), dtype=np.float64)  # CURRENT weights (dog within = BTSP-formed)
    baseline = np.asarray(to_host(mem.baseline_weights), dtype=np.float64)  # NEVER-formed weights (lesion + cat target)
    dog_mask = np.asarray(to_host(R.withinA_masks[dog_slot])).astype(bool)
    cat_mask = np.asarray(to_host(R.withinA_masks[cat_slot])).astype(bool)

    # Overlap handling: emergent assemblies can share a cell. Keep the ASSEMBLY-UNDER-TEST (dog) byte-exact; drop any
    # shared cell from the CAT control (and cat synapses touching it) so membership is disjoint (the reuse bookkeeping
    # + `_per_assembly_within_positions` require each cell in exactly one assembly). Cat stays a never-formed control.
    shared = np.intersect1d(dog_cells, cat_cells)
    cat_cells_final = np.array([c for c in cat_cells if c not in set(dog_cells.tolist())], dtype=np.int64)

    dog_set = set(dog_cells.tolist()); cat_set = set(cat_cells_final.tolist())
    # dog within synapses: REAL formed weights (an exact per-synapse copy).
    dpos = np.nonzero(dog_mask)[0]
    dog_syn = [(int(rows[p]), int(cols[p]), float(data[p]), float(baseline[p])) for p in dpos]     # (pre,post,formed,baseline)
    # cat within synapses among the FINAL (disjoint) cat cells: REAL never-formed BASELINE weights.
    cpos = np.nonzero(cat_mask)[0]
    cat_syn = [(int(rows[p]), int(cols[p]), float(baseline[p])) for p in cpos
               if int(rows[p]) in cat_set and int(cols[p]) in cat_set]

    info = dict(
        n_ca3=int(mem.n_ca3), dog_slot=dog_slot, cat_slot=cat_slot,
        dog_size=int(dog_cells.size), cat_size_raw=int(cat_cells.size), cat_size_final=int(cat_cells_final.size),
        shared_cells=int(shared.size), n_dog_within_syn=int(len(dog_syn)), n_cat_within_syn=int(len(cat_syn)),
        dog_formed_w_mean=float(np.mean([s[2] for s in dog_syn])),
        dog_formed_w_min=float(np.min([s[2] for s in dog_syn])),
        dog_formed_w_max=float(np.max([s[2] for s in dog_syn])),
        dog_baseline_w_mean=float(np.mean([s[3] for s in dog_syn])),
        cat_baseline_w_mean=float(np.mean([s[2] for s in cat_syn])) if cat_syn else 0.0,
        d5_recall_dog=rec_dog, d5_recall_cat=rec_cat,
    )
    del mem
    return dict(dog_cells=dog_cells, cat_cells=cat_cells_final, dog_syn=dog_syn, cat_syn=cat_syn, info=info)


# ----------------------------------------------------------------------------------------------------------------------
# MAP: build an AdEx ADEX_ECKER_CA3_PC bridge whose within-assembly recurrence is the EXACT COPY of the real D5 store's
# dog (formed) + cat (baseline). Config mirrors the feasibility probe's `build_store` (same neuron type / b / OU / dt).
# Cells relabeled to a compact 0..K-1 space (isomorphic copy). Returns a `store` dict compatible with `reactivate`.
# ----------------------------------------------------------------------------------------------------------------------
def build_adex_from_real(extract, seed, *, b_override, ou_sigma, dt):
    cp, _ = get_backend()
    dog_cells = extract["dog_cells"]; cat_cells = extract["cat_cells"]
    dog_syn = extract["dog_syn"]; cat_syn = extract["cat_syn"]

    # compact index space: dog cells first (0..len(dog)-1), then cat cells -- disjoint by construction.
    order = list(dog_cells.tolist()) + list(cat_cells.tolist())
    g2l = {int(g): i for i, g in enumerate(order)}
    K = len(order)
    dog_local = np.arange(0, dog_cells.size, dtype=np.int64)
    cat_local = np.arange(dog_cells.size, dog_cells.size + cat_cells.size, dtype=np.int64)

    regions = [BrainRegion(name="pc", n_neurons=K, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
    cfg.dt_ms = float(dt); cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    for f in ("enable_homeostasis", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity", "enable_inhibitory_stdp", "enable_reward_modulation"):
        setattr(cfg, f, False)
    cfg.enable_stdp = True   # allocate cp_last_spike_time through init, then flip OFF (mirrors build_store)
    cfg.stdp_a_plus = 0.0; cfg.stdp_a_minus = 0.0
    cfg.stdp_tau_plus_ms = 20.0; cfg.stdp_tau_minus_ms = 20.0
    cfg.stdp_w_min = 0.0; cfg.stdp_w_max = 900.0
    cfg.enable_ou_process = ou_sigma > 0; cfg.ou_noise_sigma_pa = float(ou_sigma)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    b.core_config.enable_stdp = False
    if b_override is not None:
        b.core_config.adex_b = float(b_override)
    pc = np.asarray(b.region_manager.indices("pc"), int)   # global AdEx indices of the pc region

    # inject the WITHIN wiring (frozen). pre/post relabeled to compact local -> global-pc index.
    win_pre, win_post, win_w = [], [], []
    for (pre_g, post_g, w_formed, _w_base) in dog_syn:
        win_pre.append(int(pc[g2l[pre_g]])); win_post.append(int(pc[g2l[post_g]])); win_w.append(float(w_formed))
    n_dog = len(win_w)
    for (pre_g, post_g, w_base) in cat_syn:
        win_pre.append(int(pc[g2l[pre_g]])); win_post.append(int(pc[g2l[post_g]])); win_w.append(float(w_base))
    # a MINIMAL plastic between-group is injected only so the per-synapse plastic mask is built (STDP is OFF; a_plus=0).
    # A single weak self-consistent forward edge dog[0]->cat[0] at weight 0 is skipped by the injector, so use a tiny
    # positive weight on ONE edge and record it; it does not couple the assemblies at reactivation drive levels and is
    # verified inert by the specificity + nocue controls. (Kept minimal to avoid any dog<->cat coupling.)
    bet_pre = [int(pc[dog_local[0]])]; bet_post = [int(pc[cat_local[0]])]; bet_w = [1e-6]

    wiring = {
        "within": {"pre_indices": win_pre, "post_indices": win_post, "initial_weights": win_w,
                   "plastic": False, "conn_type": "ff"},
        "between": {"pre_indices": bet_pre, "post_indices": bet_post, "initial_weights": bet_w,
                    "plastic": True, "conn_type": "ff"},
    }
    b.inject_explicit_wiring(wiring)

    # COO order == CSR .data order: locate dog's within .data positions + verify the EXACT-COPY alignment.
    coo = b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row), int); col = np.asarray(to_host(coo.col), int)
    dog_g = set(int(pc[l]) for l in dog_local)
    cat_g = set(int(pc[l]) for l in cat_local)
    asm_of = np.full(b.core_config.num_neurons, -1, int)
    asm_of[pc[dog_local]] = 0; asm_of[pc[cat_local]] = 1
    a_pre = asm_of[row]; a_post = asm_of[col]
    within_dog_pos = np.nonzero((a_pre == 0) & (a_post == 0))[0]
    within_cat_pos = np.nonzero((a_pre == 1) & (a_post == 1))[0]

    data0 = np.asarray(to_host(b.cp_connections.data))
    # exact-copy check: the multiset of dog within weights on the AdEx bridge == the extracted formed weights.
    formed_sorted = np.sort([s[2] for s in dog_syn])
    adex_dog_sorted = np.sort(data0[within_dog_pos])
    copy_ok = bool(within_dog_pos.size == n_dog and np.allclose(formed_sorted, adex_dog_sorted, atol=1e-3))

    # build a per-dog-synapse (formed, baseline) aligned to the .data positions, for a FAITHFUL per-synapse lesion.
    base_by_pair = {(pr, po): bw for (pr, po, _wf, bw) in
                    [(int(pc[g2l[a]]), int(pc[g2l[bb]]), wf, bwv) for (a, bb, wf, bwv) in dog_syn]}
    formed_by_pair = {(int(pc[g2l[a]]), int(pc[g2l[bb]])): wf for (a, bb, wf, _bwv) in dog_syn}
    dog_baseline_aligned = np.array([base_by_pair[(int(row[p]), int(col[p]))] for p in within_dog_pos], dtype=np.float32)
    dog_formed_aligned = np.array([formed_by_pair[(int(row[p]), int(col[p]))] for p in within_dog_pos], dtype=np.float32)

    store = dict(bridge=b, cp=cp, pc=pc, asm_local=[dog_local, cat_local], m_asm=2,
                 asm_size=int(dog_cells.size), pre_post=(row, col),
                 within_dog_pos=within_dog_pos, within_cat_pos=within_cat_pos,
                 dog_baseline_aligned=dog_baseline_aligned, dog_formed_aligned=dog_formed_aligned,
                 copy_ok=copy_ok, n_dog_within=int(within_dog_pos.size), n_cat_within=int(within_cat_pos.size),
                 adex_dog_w_mean=float(data0[within_dog_pos].mean()) if within_dog_pos.size else 0.0,
                 adex_cat_w_mean=float(data0[within_cat_pos].mean()) if within_cat_pos.size else 0.0)
    return store


def _lesion_dog(store, to_baseline):
    """Write dog's within .data positions to their real never-formed baseline (lesion) or back to formed (restore)."""
    cp = store["cp"]
    pos = cp.asarray(store["within_dog_pos"])
    vals = store["dog_baseline_aligned"] if to_baseline else store["dog_formed_aligned"]
    store["bridge"].cp_connections.data[pos] = cp.asarray(vals)


def _sweep_seed(seed, a, backend, out_dir):
    """OP-POINT ROBUSTNESS of the NO-GO: build the real store, sweep cue_pa x cue_frac on the dog assembly, bank the
    grid to sweep_s<seed>.json. Substantiates the 'holds across op-points' claim with an ON-DISK artifact instead of an
    unbanked reproduction. Does NOT write the single-op-point verdict file (seed<seed>.json stays the verified draw);
    store('dog') is a fresh draw here, and the negative holds for any draw (that is the point)."""
    print("\n" + "=" * 110)
    print(f"[ecker->REAL-d5 SWEEP] seed={seed} backend={backend} -- cue_pa x cue_frac op-point robustness of the NO-GO",
          flush=True)
    extract = extract_real_d5(seed, verbose=False)
    store = build_adex_from_real(extract, seed, b_override=a.b_override, ou_sigma=a.ou_sigma, dt=a.dt)
    cue_pas = [3000.0, 9000.0, 20000.0, 40000.0, 70000.0, 100000.0, 130000.0, 150000.0]
    cue_fracs = [0.3, 0.5, 0.7]
    grid = []
    max_dog = 0.0
    for cp_pa in cue_pas:
        for cf in cue_fracs:
            rk = dict(cue_frac=cf, cue_pa=cp_pa, cue_steps=a.cue_steps, window_steps=a.window_steps)
            d = reactivate(store, 0, seed, **rk)
            dc = float(d["completion"])
            grid.append(dict(cue_pa=cp_pa, cue_frac=cf, dog_cofire=round(dc, 4),
                             terminate_ratio=round(float(d["terminate_ratio"]), 4)))
            max_dog = max(max_dog, dc)
    # specificity across the sweep: cat control at the STRONGEST op-point must also stay silent.
    cat = reactivate(store, 1, seed, cue_frac=cue_fracs[-1], cue_pa=cue_pas[-1],
                     cue_steps=a.cue_steps, window_steps=a.window_steps)
    out = dict(seed=seed, backend=backend, params=vars(a), extract=extract["info"],
               map=dict(copy_ok=store["copy_ok"], n_dog_within=store["n_dog_within"],
                        adex_dog_w_mean=store["adex_dog_w_mean"], adex_cat_w_mean=store["adex_cat_w_mean"]),
               n_oppoints=len(grid), cue_pas=cue_pas, cue_fracs=cue_fracs,
               max_dog_cofire=round(max_dog, 4), cofire_min_bar=COFIRE_MIN,
               cat_cofire_at_max=round(float(cat["completion"]), 4),
               robust_nogo=bool(max_dog < COFIRE_MIN), grid=grid)
    sweep_path = out_dir / f"oppoint_sweep_s{seed}.json"   # NOT 'sweep_*' -> committable (gitignore excludes sweep_*.json)
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[ecker->REAL-d5 SWEEP] seed={seed} n_oppoints={len(grid)} max_dog_cofire={max_dog:.4f} (bar {COFIRE_MIN}) "
          f"robust_nogo={out['robust_nogo']} cat_at_max={out['cat_cofire_at_max']:.4f}", flush=True)
    print(f"[ecker->REAL-d5 SWEEP] wrote {sweep_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--b-override", type=float, default=120.0, help="AdEx spike-triggered adaptation (SWR self-termination)")
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--cue-frac", type=float, default=0.5)
    ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--window-steps", type=int, default=400)
    ap.add_argument("--sweep", action="store_true",
                    help="op-point robustness: sweep cue_pa x cue_frac on the dog store and bank sweep_s<seed>.json. "
                         "Does NOT write the single-op-point verdict file -- leaves the verified seed<seed>.json intact "
                         "(store('dog') is a fresh draw here; the negative holds for any draw).")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()

    if a.sweep:
        for seed in seeds:
            _sweep_seed(seed, a, backend, Path(a.out).parent)
        return 0
    all_results = {}
    go_flags = []
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1
                    else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        go_flags.append(bool(res.get("verdict_status") == "GO"))

    if len(seeds) > 1:
        summ_go = int(sum(go_flags)); n = len(seeds)
        print("\n" + "#" * 110)
        print(f"[ecker->REAL-d5] 6-SEED SUMMARY: {summ_go}/{n} GO  seeds={seeds}  go_flags={go_flags}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} "
                  f"dog_cofire={c.get('dog_cofire')} cat_cofire={c.get('cat_cofire')} "
                  f"term={c.get('dog_terminate_ratio')} lesion={c.get('lesion_cofire')} nocue={c.get('nocue_cofire')}")
        print("#" * 110)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": summ_go, "go_flags": go_flags, "backend": backend,
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[ecker->REAL-d5] wrote {summ_path}")
    return 0 if all(go_flags) else 1


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 110)
    print(f"[ecker->REAL-d5] seed={seed} backend={backend} -- build genuine D5 store, extract dog's REAL recurrence, "
          f"map onto AdEx, reactivate", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    try:
        extract = extract_real_d5(seed, verbose=True)
        result["extract"] = extract["info"]
        print(f"[ecker->REAL-d5] extracted: dog={extract['info']['dog_size']} cells "
              f"({extract['info']['n_dog_within_syn']} within syn, formed w mean={extract['info']['dog_formed_w_mean']:.1f} "
              f"[{extract['info']['dog_formed_w_min']:.1f}-{extract['info']['dog_formed_w_max']:.1f}]); "
              f"cat={extract['info']['cat_size_final']} cells (never-formed, baseline w mean="
              f"{extract['info']['cat_baseline_w_mean']:.2f}); shared_dropped={extract['info']['shared_cells']}", flush=True)

        store = build_adex_from_real(extract, seed, b_override=a.b_override, ou_sigma=a.ou_sigma, dt=a.dt)
        result["map"] = dict(copy_ok=store["copy_ok"], n_dog_within=store["n_dog_within"],
                             n_cat_within=store["n_cat_within"], adex_dog_w_mean=store["adex_dog_w_mean"],
                             adex_cat_w_mean=store["adex_cat_w_mean"], K=int(len(store["pc"])))
        print(f"[ecker->REAL-d5] mapped onto AdEx: copy_ok={store['copy_ok']} "
              f"dog_within={store['n_dog_within']} (w~{store['adex_dog_w_mean']:.1f}) "
              f"cat_within={store['n_cat_within']} (w~{store['adex_cat_w_mean']:.2f})", flush=True)

        rk = dict(cue_frac=a.cue_frac, cue_pa=a.cue_pa, cue_steps=a.cue_steps, window_steps=a.window_steps)
        # reactivate the REAL dog assembly (index 0) and the never-formed cat (index 1), identical cue dose.
        dog_react = reactivate(store, 0, seed, **rk)
        cat_react = reactivate(store, 1, seed, **rk)
        nocue = reactivate(store, 0, seed, cue_on=False, **rk)
        # RECURRENCE TEETH: weaken dog's REAL within-recurrence to its never-formed baseline -> re-cue -> should collapse.
        _lesion_dog(store, to_baseline=True)
        lesion = reactivate(store, 0, seed, **rk)
        _lesion_dog(store, to_baseline=False)   # restore formed

        # co-firing = held-out member firing fraction (probe's `completion` = peak smoothed held-out firing FRACTION).
        dog_cofire = float(dog_react["completion"])
        cat_cofire = float(cat_react["completion"])
        dog_term = float(dog_react["terminate_ratio"])
        lesion_cofire = float(lesion["completion"])
        nocue_cofire = float(nocue["completion"])

        result["dog_react"] = dog_react
        result["cat_react"] = cat_react
        result["nocue"] = nocue
        result["lesion"] = lesion

        COFIRE = bool(dog_cofire >= COFIRE_MIN)
        DISCRETE = bool(dog_term <= TERMINATE_MAX)
        SPECIFIC = bool(dog_cofire >= DOG_OVER_CAT * (cat_cofire + 1e-6))
        TEETH = bool(lesion_cofire <= max(0.5 * dog_cofire, COFIRE_MIN * 0.5))
        NOCUE_SILENT = bool(nocue_cofire <= COFIRE_MIN * 0.5)

        # attribute the dog-specific reactivation to the REAL stored recurrence (dog vs cat, and dog vs lesion).
        attr_vs_cat = attributable_to(f"[s{seed}] dog reactivation NOT in the never-formed cat control",
                                      dog_cofire, cat_cofire)
        attr_vs_lesion = attributable_to(f"[s{seed}] dog reactivation carried by the REAL stored recurrence (vs baseline-lesion)",
                                         dog_cofire, lesion_cofire)
        result["attributable"] = {"vs_cat": attr_vs_cat, "vs_lesion": attr_vs_lesion}

        # -------- earned verdict --------
        # PRECONDITIONS = INSTRUMENT VALIDITY only (things that must hold for the GO/NO-GO to be interpretable). A
        # FAILED validity precondition -> UNDEFINED. The OUTCOME measures (cofire/discrete/specific/teeth) are what
        # decide(go=...) evaluates: with a VALID instrument, go=False is a genuine NO-GO (an honest negative), NOT
        # UNDEFINED. (Do not register an outcome as a precondition -- that would misreport a real negative as an
        # instrument failure.)
        v = Verdict(f"REAL-D5 dog assembly reactivates under Ecker AdEx SWR (seed {seed})")
        v.disabled("D5 dendritic-dAP READ path", "reactivation is on the AdEx SWR substrate; D5's own read is byte-untouched")
        v.disabled("inter-assembly band", "single-assembly arc-1 test (no dog<->cat coupling); NOCUE + specificity verify inert")
        v.require("d5-store-actually-formed-dog", extract["info"]["d5_recall_dog"]["in_memory"], expect=True,
                  note="the genuine D5 store recalled dog (in_memory) -- the thing we transferred is a REAL memory")
        v.require("exact-copy-onto-adex", store["copy_ok"], expect=True,
                  note="dog within-weight multiset on the AdEx bridge == extracted real BTSP weights (a copy, not tuned)")
        v.control("dog-recurrence-strong-vs-catbaseline", store["adex_dog_w_mean"], store["adex_cat_w_mean"],
                  min_separation=5.0, note="the mapped dog recurrence is the potentiated store; cat is never-formed baseline")
        v.reaches("lesion-moves-dog-recurrence", store["adex_dog_w_mean"], float(store["dog_baseline_aligned"].mean()),
                  note="the teeth lesion actually weakens dog's within-recurrence to the never-formed baseline")
        v.require("nocue-silent", nocue_cofire, expect=lambda x: x <= COFIRE_MIN * 0.5,
                  note="validity: no cue -> no co-firing (the drive is a partial cue, not free noise)")
        go = COFIRE and DISCRETE and SPECIFIC and TEETH
        decided = v.decide(go=go)
        result["verdict"] = decided
        result["verdict_status"] = decided["status"]

        checks = dict(COFIRE=COFIRE, DISCRETE=DISCRETE, SPECIFIC=SPECIFIC, TEETH=TEETH, NOCUE_SILENT=NOCUE_SILENT,
                      dog_cofire=round(dog_cofire, 3), cat_cofire=round(cat_cofire, 3),
                      dog_terminate_ratio=round(dog_term, 3), lesion_cofire=round(lesion_cofire, 3),
                      nocue_cofire=round(nocue_cofire, 3), dog_other_fire=round(float(dog_react["other_fire_mean"]), 4))
        result["checks"] = checks
        print(f"[ecker->REAL-d5] checks={checks}", flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["verdict_status"] = "ERROR"
        traceback.print_exc()

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    status = result.get("verdict_status")
    print("=" * 110)
    print(f"[ecker->REAL-d5] seed={seed} VERDICT: {status}")
    if "checks" in result:
        c = result["checks"]
        print(f"    dog_cofire={c['dog_cofire']} cat_cofire={c['cat_cofire']} term={c['dog_terminate_ratio']} "
              f"lesion={c['lesion_cofire']} nocue={c['nocue_cofire']}")
    print(f"[ecker->REAL-d5] wrote {out_path}")
    print("=" * 110)
    return result


if __name__ == "__main__":
    sys.exit(main())
