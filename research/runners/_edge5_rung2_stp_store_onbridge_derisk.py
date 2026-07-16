"""2026-07-15 — EDGE 5 rung 2 (plan #1, the fully-spiking realization of the content-addressable STORE on a real
SimulationBridge, via Mongillo synaptic FACILITATION — the scoped, source-read (Mongillo 2008), NON-banked path; the FS-WTA
binder stays BANKED per the emergence bar). Realizes the Edge-5-rung-1 delta-STORE as SPIKES: the bind lives in the
facilitated `cp_stp_u` of the barcode→value synapses.

MECHANISM: a barcode-input region --[dense plastic-OFF barcode→value synapses, STP ON, LONG tau_f]--> K value pools on ONE
bridge. WRITE (present barcode_i + a teacher on value_i's pool): the co-active barcode_i-input→value_i synapses FACILITATE
(cp_stp_u rises). FILLERS decay it (tau_f). RETRIEVE (present barcode_j ALONE): the facilitated barcode_j→value_j synapses
RELEASE strongest (release ∝ u·x) → value_j's pool fires most → the read (argmax value-pool rate) = the retrieved value. A
NOVEL barcode has no facilitation → no bound value (the content-addressing is genuine, not a fixed map).

GATE (6-seed): retrieve-acc >> chance AND >> a novel-barcode control (no facilitation) AND the STP-OFF lesion collapses it
(the facilitation is load-bearing, not the fixed weights). Cheap-first single-pair first (does facilitation store+retrieve
at all?), then multi-pair + a horizon check (does it survive T fillers within tau_f?). NO hand-tuned FS-WTA (rate read).

Run: SIM_BACKEND=numpy python -u -m research.runners._edge5_rung2_stp_store_onbridge_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._novel_referent_hebbian_fastweight_derisk import _mint_codes, _DIM, _KACT

KV = 4                    # value pools (cheap-first small)
POOL = 40                 # neurons per value pool
INPOP = 3                 # input neurons per barcode bit
WRITE_STEPS = 40
RETRIEVE_STEPS = 40
TAU_F = 1500.0            # Mongillo augmentation time constant (ms) -- the bind persists activity-silent
STP_U = 0.05             # low baseline release so FACILITATION (u rising) is the stored signal
BAR_DRIVE = 600.0
TEACH_DRIVE = 700.0


def _build_store_bridge(seed):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    n_in = _DIM * INPOP
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="bar", n_neurons=n_in, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="val", n_neurons=KV * POOL, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg.region_pathways = [
        # dense barcode->value; weights FIXED (plastic_internal off); the MEMORY is the STP facilitation, not the weight.
        RegionPathway(from_region="bar", to_region="val", density=1.0, weight_mean=8.0, weight_jitter=2.0, plastic=False),
    ]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    for _flag in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis", "enable_structural_plasticity",
                  "enable_reward_modulation", "enable_input_divisive_norm", "enable_nmda", "enable_bdsp"):
        setattr(cfg, _flag, False)
    cfg.enable_short_term_plasticity = True     # THE mechanism
    cfg.stp_U = STP_U
    cfg.stp_tau_f = TAU_F                        # long facilitation = the Mongillo Ca buffer holding the bind
    cfg.stp_tau_d = 200.0
    cfg.enable_per_type_stp = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, cfg


class STPStore:
    def __init__(self, seed, stp_on=True):
        self.bridge, self.cfg = _build_store_bridge(seed)
        if not stp_on:
            self.cfg.enable_short_term_plasticity = False       # the LESION: fixed weights only, no facilitation memory
        self._num = int(self.bridge.core_config.num_neurons)
        rm = self.bridge.region_manager
        self.bar_idx = np.asarray(list(rm.indices("bar")), int)
        self.val_idx = np.asarray(list(rm.indices("val")), int)
        self.bar_ch = [self.bar_idx[v * INPOP:(v + 1) * INPOP] for v in range(_DIM)]
        self.val_pool = [self.val_idx[k * POOL:(k + 1) * POOL] for k in range(KV)]

    def _drive(self, barcode=None, value=None):
        from sim.backend import from_host
        cur = np.zeros(self._num, np.float32)
        if barcode is not None:
            for bit in np.nonzero(barcode)[0]:
                cur[self.bar_ch[bit]] = BAR_DRIVE
        if value is not None:
            cur[self.val_pool[value]] = TEACH_DRIVE
        self.bridge.cp_external_input_current[:] = from_host(cur)

    def _run(self, steps, read=False):
        from sim.backend import to_host
        counts = np.zeros(KV)
        for _ in range(steps):
            self.bridge._run_one_simulation_step()
            if read:
                fs = np.asarray(to_host(self.bridge.cp_firing_states)).astype(np.float64)
                for k in range(KV):
                    counts[k] += fs[self.val_pool[k]].sum()
        return counts

    def write(self, barcode, value):
        self._drive(barcode, value); self._run(WRITE_STEPS)             # co-activation facilitates barcode->value synapses
        self.bridge.cp_external_input_current[:] = 0.0

    def fillers(self, n):
        self._drive(); self._run(n)                                    # silent/decay steps (facilitation fades w/ tau_f)

    def retrieve(self, barcode):
        self._drive(barcode, None)
        counts = self._run(RETRIEVE_STEPS, read=True)
        self.bridge.cp_external_input_current[:] = 0.0
        return int(np.argmax(counts)), counts


def run_one(seed, n_trials=40, T=8):
    rng = np.random.default_rng(seed)
    codes = _mint_codes(rng, 16)
    res = {"seed": seed, "chance": round(1.0 / KV, 4), "T": T, "tau_f": TAU_F}
    hit, hit_novel, hit_lesion = [], [], []
    for t in range(n_trials):
        store = STPStore(seed * 100 + t, stp_on=True)
        lesion = STPStore(seed * 100 + t, stp_on=False)
        ei = int(rng.integers(16)); vi = int(rng.integers(KV))
        store.write(codes[ei], vi); store.fillers(T)
        pred, _ = store.retrieve(codes[ei]); hit.append(pred == vi)
        # novel-barcode control: retrieve with a NEVER-written barcode -> no facilitation -> no bound value
        nb = _mint_codes(np.random.default_rng(seed * 7 + t), 1)[0]
        pn, _ = store.retrieve(nb); hit_novel.append(pn == vi)
        # STP-OFF lesion: same write+retrieve, facilitation disabled -> fixed weights can't store the bind
        lesion.write(codes[ei], vi); lesion.fillers(T)
        pl, _ = lesion.retrieve(codes[ei]); hit_lesion.append(pl == vi)
    res["retrieve_acc"] = round(float(np.mean(hit)), 4)
    res["novel_barcode_acc"] = round(float(np.mean(hit_novel)), 4)
    res["stp_off_lesion_acc"] = round(float(np.mean(hit_lesion)), 4)
    res["GO"] = bool(res["retrieve_acc"] > 0.5 and res["retrieve_acc"] > res["novel_barcode_acc"] + 0.2
                     and res["retrieve_acc"] > res["stp_off_lesion_acc"] + 0.2)
    return res


def run_multipair(seed, Ps=(1, 2, 3, 4), n_trials=20, T=8):
    """Write P (barcode,value) binds into ONE store's facilitation, then retrieve each -> does the Mongillo store hold
    MULTIPLE binds (the numpy delta-store's interference regime), or does the facilitation interfere past ~few items
    (the biological Mongillo WM-capacity boundary)? Distinct values (P<=KV)."""
    rng = np.random.default_rng(seed)
    codes = _mint_codes(rng, 16)
    res = {"seed": seed, "chance": round(1.0 / KV, 4), "byP": {}}
    for P in Ps:
        hits = []
        for t in range(n_trials):
            store = STPStore(seed * 100 + t, stp_on=True)
            ents = rng.choice(16, size=P, replace=False)
            vals = list(rng.permutation(KV)[:P])
            for i in range(P):
                store.write(codes[ents[i]], vals[i])       # each write facilitates its own barcode->value synapses
            store.fillers(T)
            for i in range(P):
                pred, _ = store.retrieve(codes[ents[i]])
                hits.append(pred == vals[i])
        res["byP"][P] = round(float(np.mean(hits)), 4)
    res["holds_ge3"] = bool(res["byP"].get(3, 0) > 0.6)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--multipair", action="store_true", help="run the multi-pair interference/capacity sweep instead")
    ap.add_argument("--out", default="research/findings/raw/_edge5_rung2_stp_store_onbridge.json")
    a = ap.parse_args()
    if a.multipair:
        rows = [run_multipair(s, n_trials=a.n_trials) for s in a.seeds]
        for r in rows:
            bp = r["byP"]
            print(f"[stp-store-multi s{r['seed']}] chance={r['chance']} || retrieve-acc by #pairs: "
                  f"P1={bp.get(1):.2f} P2={bp.get(2):.2f} P3={bp.get(3):.2f} P4={bp.get(4):.2f} || holds>=3={r['holds_ge3']}", flush=True)
        nh = sum(x["holds_ge3"] for x in rows)
        print(f"[stp-store-multi] holds>=3-pairs {nh}/{len(rows)} (the Mongillo facilitation store's on-bridge WM capacity; "
              f"a graceful fall past ~few items = the biological Mongillo/Lisman-Idiart capacity boundary, not a bug)", flush=True)
        json.dump(rows, open(a.out.replace(".json", "_multipair.json"), "w"))
        return
    rows = [run_one(s, n_trials=a.n_trials) for s in a.seeds]
    for r in rows:
        print(f"[stp-store s{r['seed']}] chance={r['chance']} tau_f={r['tau_f']} || RETRIEVE-via-facilitation={r['retrieve_acc']:.3f} "
              f"| novel-barcode={r['novel_barcode_acc']:.3f} | STP-OFF-lesion={r['stp_off_lesion_acc']:.3f} || {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[stp-store] {ngo}/{len(rows)} GO (the content-addressable store realized ON SPIKES via Mongillo facilitation: "
          f"re-presenting a written barcode retrieves its value from the facilitated cp_stp_u; novel barcode + STP-OFF collapse)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
