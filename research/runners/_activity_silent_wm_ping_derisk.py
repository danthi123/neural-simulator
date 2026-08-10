"""2026-08-10 — ACTIVITY-SILENT WORKING MEMORY via Mongillo (2008) short-term FACILITATION, the SIGNATURE protocol:
a NONSPECIFIC ping reactivates a SILENTLY-held item (memory lives in residual presynaptic Ca2+ / cp_stp_u, NOT in
persistent firing). This is DISTINCT from the already-GO edge5-rung2 content-addressable STORE
(2026-07-15-edge5-rung2-STP-store-onbridge-6seed-GO.md): that used a SPECIFIC barcode cue for retrieval and had ACTIVE
fillers during the delay. Here the two Mongillo hallmarks are the whole point:
   (1) the delay is VERIFIED SILENT (assembly firing ~0 -- no persistent-activity attractor), and
   (2) the reactivating cue is NONSPECIFIC (a uniform ping to the WHOLE population, carrying NO item identity).

MECHANISM: K isolated excitatory assemblies, each with WITHIN-assembly recurrent E->E synapses (STP ON, long tau_f,
plastic OFF -- the recurrent WEIGHT is fixed; the MEMORY is the facilitation cp_stp_u). LOAD item A: transiently drive
assembly A -> its co-active recurrent synapses FACILITATE (u rises). DELAY: zero drive; the recurrent weight is
sub-self-sustaining so the assembly falls SILENT (verified), but u decays only with tau_f (1500ms) >> delay. PING:
uniform external drive to ALL neurons (identical to every assembly). Because assembly A's recurrent synapses are
facilitated (higher u -> higher effective release u*x), A's neurons get extra recurrent reverberation -> A reactivates
PREFERENTIALLY. Read = per-assembly firing rate in the ping window; argmax should be the loaded assembly.

CONTROLS (all three):
  (1) TAU_F-MINIMAL control (the FAIR, excitability-MATCHED lesion): keep enable_short_term_plasticity=True (so the
      u*x effective-weight multiplier and thus the network excitability are IDENTICAL) but set stp_tau_f to a tiny
      value (5ms << delay). The facilitation then DECAYS AWAY during the delay -> at ping time u~baseline for the
      loaded assembly too -> the ping reactivates nothing selectively (~chance). This isolates the MEMORY (the
      tau_f-bridged facilitation) from wiring/excitability. NB: the cruder enable_short_term_plasticity=False toggle
      is NOT fair here -- it removes the u*x multiplier, jumping effective recurrence from ~w*U to full w, which turns
      the net into a Wang-2002 PERSISTENT-FIRING attractor (delay no longer silent); we record it as a diagnostic but
      the GO gate uses the tau_f-minimal control.
  (2) SPECIFICITY: the loaded assembly is randomized per trial; high argmax-accuracy then means the ping reactivates
      WHICHEVER was loaded (not a structural favorite -- a fixed favorite would score ~chance since loaded varies).
  (3) SILENT-DELAY: mean assembly firing during the delay is reported; it must be ~0 (the memory is activity-silent,
      not a persistent-firing attractor).

GATE (single-seed smoke, cheapest decisive): reactivation-acc >> chance(1/K) AND >> STP-OFF-lesion AND delay-firing~0.

Run: SIM_BACKEND=numpy python -u -m research.runners._activity_silent_wm_ping_derisk --seeds 42
NO sim/ edit -- reuse-by-import the committed STP path (enable_short_term_plasticity + stp_tau_f + cp_stp_u).
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

K = 4                     # assemblies (chance = 1/K = 0.25)
POOL = 40                 # neurons per assembly
W_REC = 60.0              # within-assembly recurrent E->E weight (FIXED; sub-self-sustaining -> silent delay). Must be
                          # strong enough that the FACILITATED recurrence is a DECISIVE share of the ping-window firing
                          # drive (at W_REC=6 the recurrent boost is dwarfed by the direct ping -> no selectivity).
STP_U = 0.05              # low baseline release so FACILITATION (u rising) is the stored signal (Mongillo regime)
TAU_F = 1500.0            # Mongillo augmentation time constant (ms) -- the held item lives here, activity-silent
TAU_D = 200.0            # depression recovery (ms) -- recovers over the delay so facilitation dominates the ping
LOAD_DRIVE = 600.0
LOAD_STEPS = 40
DELAY_STEPS = 400         # tau_d(200) < delay < tau_f(1500): depression x RECOVERS while facilitation u PERSISTS, so at
                          # ping-time only the facilitation carries the memory (the core Mongillo time-constant window).
PING_DRIVE = 200.0        # uniform, NONSPECIFIC ping -- carries NO item identity; the facilitated assembly amplifies it
PING_STEPS = 40
TAU_F_LESION = 5.0        # the FAIR control: STP stays ON (excitability matched) but facilitation can't bridge the delay


def _build_assemblies(seed, stp_on=True, w_rec=W_REC, stp_u=STP_U, tau_f=TAU_F):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    # K isolated assemblies, each with dense WITHIN-assembly recurrent E->E (the substrate of the facilitation memory).
    cfg.brain_regions = [
        BrainRegion(name=f"A{k}", n_neurons=POOL, exc_fraction=1.0, internal_density=1.0,
                    exc_weight_mean=w_rec, inh_weight_mean=0.0, weight_jitter=0.5, plastic_internal=False)
        for k in range(K)
    ]
    cfg.region_pathways = []          # NO cross-assembly wiring -- the ping is external & uniform, not routed
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    for _flag in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis", "enable_structural_plasticity",
                  "enable_reward_modulation", "enable_input_divisive_norm", "enable_nmda", "enable_bdsp"):
        setattr(cfg, _flag, False)
    cfg.enable_short_term_plasticity = bool(stp_on)     # the crude (excitability-confounded) OFF diagnostic flips this
    cfg.stp_U = stp_u
    cfg.stp_tau_f = tau_f                               # the FAIR control shrinks THIS (facilitation can't bridge delay)
    cfg.stp_tau_d = TAU_D
    cfg.enable_per_type_stp = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, cfg


class ActivitySilentWM:
    def __init__(self, seed, stp_on=True, w_rec=W_REC, stp_u=STP_U, delay_steps=DELAY_STEPS, ping_drive=PING_DRIVE,
                 ping_steps=PING_STEPS, tau_f=TAU_F):
        self.delay_steps = delay_steps
        self.ping_drive = ping_drive
        self.ping_steps = ping_steps
        self.bridge, self.cfg = _build_assemblies(seed, stp_on=stp_on, w_rec=w_rec, stp_u=stp_u, tau_f=tau_f)
        self._num = int(self.bridge.core_config.num_neurons)
        rm = self.bridge.region_manager
        self.pool = [np.asarray(list(rm.indices(f"A{k}")), int) for k in range(K)]

    def _drive_all(self, val):
        from sim.backend import from_host
        cur = np.zeros(self._num, np.float32)
        cur[:] = val
        self.bridge.cp_external_input_current[:] = from_host(cur)

    def _drive_one(self, k, val):
        from sim.backend import from_host
        cur = np.zeros(self._num, np.float32)
        cur[self.pool[k]] = val
        self.bridge.cp_external_input_current[:] = from_host(cur)

    def _run(self, steps, read=False):
        from sim.backend import to_host
        counts = np.zeros(K)
        total = 0.0
        for _ in range(steps):
            self.bridge._run_one_simulation_step()
            fs = np.asarray(to_host(self.bridge.cp_firing_states)).astype(np.float64)
            total += fs.sum()
            if read:
                for k in range(K):
                    counts[k] += fs[self.pool[k]].sum()
        return counts, total

    def load(self, k):
        self._drive_one(k, LOAD_DRIVE); self._run(LOAD_STEPS)
        self.bridge.cp_external_input_current[:] = 0.0

    def delay(self):
        self.bridge.cp_external_input_current[:] = 0.0
        _, total = self._run(self.delay_steps)
        return total / (self.delay_steps * self._num)  # mean firing prob per neuron per step during the SILENT delay

    def ping(self):
        self._drive_all(self.ping_drive)
        counts, _ = self._run(self.ping_steps, read=True)
        self.bridge.cp_external_input_current[:] = 0.0
        return counts


def run_one(seed, n_trials=40, w_rec=W_REC, stp_u=STP_U, delay_steps=DELAY_STEPS, ping_drive=PING_DRIVE,
            ping_steps=PING_STEPS):
    rng = np.random.default_rng(seed)
    res = {"seed": seed, "chance": round(1.0 / K, 4), "K": K, "delay_steps": delay_steps, "tau_f": TAU_F,
           "tau_d": TAU_D, "w_rec": w_rec, "stp_u": stp_u, "ping_drive": ping_drive, "ping_steps": ping_steps}
    hit, hit_ctrl, hit_off = [], [], []
    delay_fire, delay_fire_ctrl, delay_fire_off = [], [], []
    margin = []
    for t in range(n_trials):
        loaded = int(rng.integers(K))
        # --- TEST: full facilitation (tau_f=1500) holds the item silently ---
        wm = ActivitySilentWM(seed * 100 + t, stp_on=True, w_rec=w_rec, stp_u=stp_u, tau_f=TAU_F,
                              delay_steps=delay_steps, ping_drive=ping_drive, ping_steps=ping_steps)
        wm.load(loaded)
        df = wm.delay(); delay_fire.append(df)
        counts = wm.ping()
        hit.append(int(np.argmax(counts)) == loaded)
        others = np.mean([counts[k] for k in range(K) if k != loaded])
        margin.append(counts[loaded] - others)
        # --- CONTROL (fair, excitability-matched): STP ON but tau_f tiny -> facilitation can't bridge the delay ---
        ctrl = ActivitySilentWM(seed * 100 + t, stp_on=True, w_rec=w_rec, stp_u=stp_u, tau_f=TAU_F_LESION,
                                delay_steps=delay_steps, ping_drive=ping_drive, ping_steps=ping_steps)
        ctrl.load(loaded)
        dfc = ctrl.delay(); delay_fire_ctrl.append(dfc)
        cc = ctrl.ping()
        hit_ctrl.append(int(np.argmax(cc)) == loaded)
        # --- DIAGNOSTIC (excitability-CONFOUNDED): enable_short_term_plasticity=False -> full-weight recurrence ---
        off = ActivitySilentWM(seed * 100 + t, stp_on=False, w_rec=w_rec, stp_u=stp_u, tau_f=TAU_F,
                               delay_steps=delay_steps, ping_drive=ping_drive, ping_steps=ping_steps)
        off.load(loaded)
        dfo = off.delay(); delay_fire_off.append(dfo)
        co = off.ping()
        hit_off.append(int(np.argmax(co)) == loaded)
    res["reactivation_acc"] = round(float(np.mean(hit)), 4)
    res["taufmin_control_acc"] = round(float(np.mean(hit_ctrl)), 4)     # the FAIR control (must be ~chance)
    res["stp_off_diag_acc"] = round(float(np.mean(hit_off)), 4)          # excitability-confounded diagnostic
    res["delay_firing_mean"] = round(float(np.mean(delay_fire)), 6)
    res["delay_firing_ctrl_mean"] = round(float(np.mean(delay_fire_ctrl)), 6)
    res["delay_firing_off_diag_mean"] = round(float(np.mean(delay_fire_off)), 6)
    res["reactivation_margin_mean"] = round(float(np.mean(margin)), 3)
    res["silent_delay"] = bool(res["delay_firing_mean"] < 0.01)   # ~0 firing during delay -> activity-silent
    res["GO"] = bool(res["reactivation_acc"] > 0.5
                     and res["reactivation_acc"] > res["taufmin_control_acc"] + 0.2
                     and res["silent_delay"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--w-rec", type=float, default=W_REC)
    ap.add_argument("--stp-u", type=float, default=STP_U)
    ap.add_argument("--delay-steps", type=int, default=DELAY_STEPS)
    ap.add_argument("--ping-drive", type=float, default=PING_DRIVE)
    ap.add_argument("--ping-steps", type=int, default=PING_STEPS)
    ap.add_argument("--out", default="research/findings/raw/_activity_silent_wm_ping.json")
    a = ap.parse_args()
    rows = [run_one(s, n_trials=a.n_trials, w_rec=a.w_rec, stp_u=a.stp_u, delay_steps=a.delay_steps,
                    ping_drive=a.ping_drive, ping_steps=a.ping_steps) for s in a.seeds]
    for r in rows:
        print(f"[act-silent-wm s{r['seed']}] chance={r['chance']} tau_f={r['tau_f']} w_rec={r['w_rec']} || "
              f"REACTIVATION-via-nonspecific-ping={r['reactivation_acc']:.3f} (margin={r['reactivation_margin_mean']:+.2f}) "
              f"| FAIR-ctrl(tau_f={TAU_F_LESION:g})={r['taufmin_control_acc']:.3f} | delay-firing(test/ctrl)="
              f"{r['delay_firing_mean']:.5f}/{r['delay_firing_ctrl_mean']:.5f} silent={r['silent_delay']} || "
              f"[diag stp-OFF acc={r['stp_off_diag_acc']:.3f} delayfire={r['delay_firing_off_diag_mean']:.5f} "
              f"= persistent-attractor confound] || {'GO' if r['GO'] else 'no'}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[act-silent-wm] {ngo}/{len(rows)} GO (a NONSPECIFIC ping reactivates a SILENTLY-held item from the "
          f"facilitated cp_stp_u; STP-OFF collapses it to chance; delay verified silent -- Mongillo activity-silent WM)",
          flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
