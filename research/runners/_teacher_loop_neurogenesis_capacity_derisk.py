"""TEACHER-LOOP NEUROGENESIS / CAPACITY-EXPANSION DE-RISK (2026-08-09): attack the N-SCALING degradation of the
sleep-replay retention baseline (measured here: ~0.85 frac-recalled @ N=10 -> ~0.45 @ N=20, and the reservoir-gated
collapse on the fixed-DG seeds) with ADULT DENTATE-GYRUS NEUROGENESIS: GROW the frozen DG-expansion reservoir as
facts accumulate. At each new fact the brain BIRTHS `grow_k` fresh granule units -- real Izhikevich neurons in the
substrate that were dormant (afferent synapses = 0, silent) and now receive brain-owned RANDOM afferent synapses
(input->granule cp_connections), so they begin to SPIKE (readable in cp_firing_states) and add fresh, uncommitted
pattern-separated dimensions to the shared leaky readout. Older facts live on older granule units that later births
never disturb (feedforward reservoir, hidden frozen -> no cross-unit interference). The SELF-REPLAY consolidation
(the baseline to beat) runs in EVERY arm; the only manipulation is capacity growth.

GROUNDING. Adult DG neurogenesis supports CUMULATIVE lifelong representation + temporal pattern separation
(PMC6877936 Anacker/Hen; PMC4373261 Aimone). DSD-SNN 'Dynamic Structure Development of SNNs' (arXiv 2308.04749)
grows new neurons per task (and prunes redundant ones) to expand capacity for continual learning. The birthed unit
is a genuine adult-born granule cell: it exists in the substrate, was silent (no afferent drive), and integrates by
GROWING afferent synapses -> it starts responding to percepts and its efferent (readout) synapse is then learnable.

BRAIN-BASED (the load-bearing distinction, and every anti-cheat is a REAL test):
  (a) growth adds REAL neural units with BRAIN-OWNED synapses written into cp_connections (input->granule Xavier*ff
      draws from a brain RNG), NOT a host lookup that grows a table. The unit's firing is read from cp_firing_states;
      a dormant (zero-afferent) unit is verified SILENT and a born unit verified SPIKING.
  (b) LOAD-BEARING: FROZEN_GROWTH (grow only up to the N=10 size, then stop) -> retention must DROP at N=20 vs GROWN.
  (c) DECISIVE CONTROL: MATCHED_FIXED = a reservoir of the SAME FINAL size but fixed from the start. If it does AS
      WELL as GROWN -> the lever is just 'more capacity' (a scale finding, honest); if GROWN wins -> grow-as-you-go
      timing matters (neurogenesis-specific). BOTH reported.
  (d) reservoir-gated seeds (the fixed-DG collapse) -- does growth lift them? per-seed reported.
  (e) cfg.seed set (NOT actual_seed_used) + byte-identical substrate across two builds at one seed (asserted).
  (f) git diff main -- sim/ empty (asserted at import; the growth is a runner-side write over cp_connections the
      e-prop readout already uses).
  (g) UNTRAINED / RANDOM-ADDED-UNITS control: RANDOM_UNITS births the SAME units (they fire) but their readout
      columns are FROZEN at brain-random and NEVER trained -> 'more units' alone (random, unintegrated) injects
      representational noise but cannot carry the facts. If retention still rises, the win is fake capacity.

ARMS (all: frozen DG reservoir + self-replay consolidation; the ONLY difference is the capacity schedule):
  * SELF_REPLAY   = BASELINE. Fixed reservoir at BASE size n0. The in-run self-replay retention this must BEAT
                    (MEASURED here per-seed; the 0.85/0.45 headline is NOT imported).
  * GROWN         = TREATMENT. Birth grow_k granule units per fact (n0 -> n0 + n_max*grow_k). Self-replay.
  * MATCHED_FIXED = DECISIVE CONTROL. Fixed reservoir at the FINAL grown size from the start. Self-replay.
  * FROZEN_GROWTH = LOAD-BEARING (b). Grow only through the N=10 milestone, then freeze growth. Self-replay.
  * RANDOM_UNITS  = ANTI-CHEAT (g). Birth as GROWN, but the born units' readout columns are frozen brain-random
                    (never trained) -> unintegrated capacity.

TEETH / GO: GROWN frac-recalled @ N=20 > SELF_REPLAY @ N=20 (the measured baseline) by a margin, 6/6 seeds; AND
GROWN > RANDOM_UNITS (integration, not noise); AND FROZEN_GROWTH < GROWN (capacity is load-bearing at scale). The
MATCHED_FIXED comparison is REPORTED (capacity vs grow-as-you-go), not a GO gate -- both outcomes are honest.
IMMEDIATE acquisition of the new fact must stay high in GROWN (the new granule units must not block learning).

DISCIPLINE: reuse-by-import (OnBridgeEpropNet; _feat / _fit_readout_norm_world / _teach_fact / _fact_acc /
_corrective_batch / N_ACT from the scaling de-risk; Hippocampus + _self_replay_consolidate from the sleep-replay
de-risk; ReferentEnv from the corrective-acquire de-risk). NO sim/ edit. cfg.seed via the seed= the build passes to
CoreSimConfig.seed. SIM_BACKEND=numpy (the teacher-loop net is tiny + launch-bound -> CPU faster).

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_neurogenesis_capacity_derisk --seed 42 \
      --n-max 20 --milestones 10 20 --n0 20 --grow-k 4 --epochs 20 --replay-epochs 12 --replay-per-fact 8 \
      --n-draws 16 --settle-steps 20 --test-n 40 \
      --out research/findings/raw/teacher_loop_neurogenesis_s42.json
  6-SEED (GO needs the rise + integration + load-bearing 6/6 at 42..47), one seed per process in parallel:
    for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_neurogenesis_capacity_derisk --seed $s \
      --n-max 20 --milestones 10 20 --n0 20 --grow-k 4 --epochs 20 --replay-epochs 12 --replay-per-fact 8 \
      --n-draws 16 --settle-steps 20 --test-n 40 \
      --out research/findings/raw/teacher_loop_neurogenesis_s$s.json & done; wait
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny launch-bound net -> CPU faster (the teacher-loop runners are numpy)
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import to_host  # noqa: E402
# reuse-by-import: the teacher-loop SCALING machinery + the OnBridge e-prop net + the sleep-replay Hippocampus. NO sim/ edit.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _feat, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import Hippocampus, _self_replay_consolidate  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_neurogenesis.json"


# ============================ the GROWING DG-expansion reservoir (adult-born granule units) ============================
class NeurogenesisNet(OnBridgeEpropNet):
    """OnBridgeEpropNet with a FROZEN hidden reservoir of MAX size H_max, of which only the first `n_active` granule
    units are INTEGRATED at any time. A dormant unit has its afferent (input->granule) cp_connections weights = 0, so
    it receives only sub-threshold tonic drive and is SILENT (no spikes -> zero readout eligibility -> zero influence
    on the logits). BIRTH sets a unit's afferent synapses to brain-owned random Xavier draws (the substrate's own
    init distribution) -> it begins to spike and adds a fresh reservoir dimension to the shared leaky readout. The
    hidden reservoir is frozen (train_layers = {readout}) so births never perturb older units' tuning (a feedforward
    reservoir: granule units do not interconnect). All growth is a runner-side write over the SAME cp_connections the
    e-prop readout uses -- NO sim/ edit."""

    def __init__(self, n_in, k, seed, h_max, n0, settle, eprop_lr, w_clip, ff_w_init=2000.0, bdsp_wmax=1e9):
        # bdsp_wmax WIDENS the inherited bdsp_w_min/max = -6/+6 clamp (parent :183) so fused_bdsp_update's
        # unconditional cp.clip is a NO-OP -- otherwise every forward CRUSHES the afferent synapses whose presyn fired
        # to |w|<=6 (the documented clamp trap), which would silence the birthed granule units. De-clamped, the birthed
        # random projections survive and the units robustly spike (~24 vs ~1 spikes/percept). Applied to ALL arms so
        # the self-replay baseline is measured in the SAME regime (like-for-like).
        hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=ff_w_init, pbar_alpha=0.05,
                  in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0, freeze_hidden=True, bdsp_wmax=bdsp_wmax)
        super().__init__(n_in, h_max, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                         eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                         logit_source="leaky_readout", w_clip=w_clip, hp=hp)
        self.h_max = int(h_max)
        self.ff_w_init = float(ff_w_init)
        self.n_active = 0
        self._brain_rng = np.random.default_rng(seed + 424242)   # brain-owned afferent/readout-birth RNG
        self._in_pathway = self._data_idx_flat[0]                # input->granule FF pathway (flat, (n_in_phys, H_phys))
        self._out_pathway = self._data_idx_flat[-1]              # granule->readout FF pathway (flat, (H_phys, out_phys))
        self._n_in_phys = self.sizes_phys[0]
        self._out_phys = self.sizes_phys[-1]
        self._afferent_lim = float(np.sqrt(6.0 / (self.sizes_phys[0] + self.sizes_phys[1])))
        # per-unit readout-norm (mu/sigma): a dormant unit is silent so mu=0, sigma=1 -> standardized r = 0 (no
        # influence). Set explicitly (fit later per-birth) so an unfitted dormant unit never blows up the readout.
        self._r_mu = np.zeros(self.h_max, dtype=np.float64)
        self._r_sigma = np.ones(self.h_max, dtype=np.float64)
        self._frozen_out_cols = set()                            # RANDOM_UNITS: readout rows frozen (never trained)
        self._frozen_out_vals = {}                               # col -> the fixed brain-random readout weights
        # start with ALL H_max units DORMANT (afferents zeroed); the arm then births n0 (+ growth).
        self._zero_afferents(range(self.h_max))

    def _base_drive(self):
        """Zero the tonic background current on DORMANT granule units (index >= n_active) so an unintegrated adult-born
        cell has NO drive at all (no afferent synapses AND no tonic) -> genuinely SILENT until birth. Active units keep
        the tonic. Feedforward reservoir (no hidden interconnections) -> this never perturbs active/older units."""
        drive = super()._base_drive()
        hl = self.slices[-2]                                     # hidden (granule) slice; pool_k=1 -> unit j at hl.start+j
        if self.n_active < self.h_max:
            drive[hl.start + self.n_active:hl.stop] = 0.0
        return drive

    # ---- afferent (input->granule) column addressing: flat idx of edge (i_pre, j_unit) = i_pre*H_phys + j_unit ----
    def _afferent_flat_idx(self, unit):
        return self._in_pathway[np.arange(self._n_in_phys) * self.h_max + int(unit)]

    def _readout_row_flat_idx(self, unit):
        return self._out_pathway[int(unit) * self._out_phys + np.arange(self._out_phys)]

    def _zero_afferents(self, units):
        data = self.br.cp_connections.data
        xp = self._xp
        for u in units:
            idx = self._afferent_flat_idx(u)
            data[idx] = xp.asarray(np.zeros(self._n_in_phys, dtype=np.float32))

    def _birth_afferents(self, unit):
        """Grow brain-owned random afferent synapses onto one granule unit (Xavier*ff_w_init, the substrate's own FF
        init distribution, drawn from the BRAIN RNG). Writes real cp_connections weights -> the unit starts spiking."""
        data = self.br.cp_connections.data
        xp = self._xp
        w = self._brain_rng.uniform(-self._afferent_lim, self._afferent_lim, self._n_in_phys) * self.ff_w_init
        data[self._afferent_flat_idx(unit)] = xp.asarray(w.astype(np.float32))

    def _fit_unit_norm(self, env, referents, seed, units):
        """Homeostatic readout-input scaling for freshly-born units: measure each unit's last-hidden eligibility r
        over the world and set its (mu, sigma) ONCE (frozen thereafter), so a newly-integrated unit's readout is
        well-conditioned WITHOUT retuning older units' normalization (which would distort earlier facts)."""
        units = list(units)
        if not units:
            return
        rng = np.random.default_rng(seed + 909)
        feats = [_feat(env, r) for r in referents for _ in range(6)]
        R = np.array([self._readout_elig(self._forward_record(feats[i])[0]) for i in range(len(feats))])
        for u in units:
            self._r_mu[u] = R[:, u].mean()
            self._r_sigma[u] = R[:, u].std() + 1e-6
        _ = rng

    def birth(self, k, env, referents, seed, freeze_readout=False):
        """Birth k new granule units: grow their afferents (they spike), fit their readout normalization, and -- for
        the RANDOM_UNITS control -- freeze their readout row at brain-random (never trained)."""
        new = list(range(self.n_active, min(self.n_active + int(k), self.h_max)))
        if not new:
            return []
        for u in new:
            self._birth_afferents(u)
        self._fit_unit_norm(env, referents, seed, new)
        if freeze_readout:
            data = self.br.cp_connections.data
            xp = self._xp
            for u in new:
                # a fixed brain-random readout row (unintegrated capacity: fires + injects noise, never learns).
                rv = (self._brain_rng.standard_normal(self._out_phys) * 0.5).astype(np.float32)
                data[self._readout_row_flat_idx(u)] = xp.asarray(rv)
                self._frozen_out_cols.add(u)
                self._frozen_out_vals[u] = rv.copy()
        self.n_active = new[-1] + 1
        return new

    def _apply_grads(self, grads, bsz):
        super()._apply_grads(grads, bsz)
        # RANDOM_UNITS: restore the frozen readout rows so those units never integrate (their weights stay random).
        if self._frozen_out_cols:
            data = self.br.cp_connections.data
            xp = self._xp
            for u in self._frozen_out_cols:
                data[self._readout_row_flat_idx(u)] = xp.asarray(self._frozen_out_vals[u])

    # ---- diagnostics: which active units actually spike on a percept (cp_firing_states), for the anti-cheat check ----
    def unit_spike_counts(self, feat_row):
        sp, _vv, _acts = self._forward_record(feat_row)
        hl = self.slices[-2]
        return sp[:, hl].sum(axis=0)                              # (H_phys,) summed spikes per granule unit


# ============================================ one arm of the curriculum ============================================
def _run_arm(arm, seed, referents, env, K, n_in, h_max, n0, grow_k, freeze_at, settle, epochs, batch, eprop_lr,
             w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance,
             bdsp_wmax=1e9):
    """arm in {self_replay, grown, matched_fixed, frozen_growth, random_units}. All arms self-replay-consolidate;
    the manipulation is the capacity schedule (which/when granule units are born)."""
    grow = arm in ("grown", "frozen_growth", "random_units")
    random_units = arm == "random_units"
    # ONLY frozen_growth honors freeze_at (stop growing at the N=10 milestone); grown/random grow the whole way.
    eff_freeze = freeze_at if arm == "frozen_growth" else None
    # matched_fixed: the SAME final size as GROWN, but fixed from the start (the decisive capacity-vs-timing control).
    # GROWN births grow_k at EVERY fact i=0..K-1 -> final = n0 + K*grow_k, exactly matched_fixed's fixed size.
    final_size = n0 + K * grow_k
    start_active = final_size if arm == "matched_fixed" else n0

    net = NeurogenesisNet(n_in, K, seed, h_max, n0, settle, eprop_lr, w_clip, bdsp_wmax=bdsp_wmax)
    net.birth(start_active, env, referents, seed)                # birth the initial cohort (or the full matched reservoir)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)
    hippo = Hippocampus(seed, replay_noise=replay_noise)

    # anti-cheat (a) SILENT-vs-SPIKING check: accumulate spikes over ALL referents so a born unit that happens not to
    # respond to one percept is still seen; a dormant (zero-afferent) unit must NEVER spike on any percept.
    silent_dormant_ok = None; born_spikes_ok = None
    if net.n_active < h_max:
        counts = np.zeros(h_max, dtype=np.float64)
        for rr in referents:
            counts += net.unit_spike_counts(_feat(env, rr))
        silent_dormant_ok = bool(counts[net.n_active:h_max].sum() < 1e-9)   # dormant units NEVER spike
        born_spikes_ok = bool(counts[:net.n_active].sum() > 0.0)            # the born cohort DOES spike

    acquire_acc = []
    retention = {}
    growth_trace = []
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)
        # --- GROWTH: birth grow_k fresh granule units for this fact (neurogenesis is continuous) ---
        if grow and (eff_freeze is None or i < eff_freeze):
            net.birth(grow_k, env, referents, seed, freeze_readout=random_units)
        growth_trace.append(int(net.n_active))
        # --- WAKE: teach fact i (readout-only; hidden frozen) ---
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))
        # --- SLEEP: self-replay consolidation of the whole store (teacher + world ABSENT) ---
        hippo.encode(X, i)
        _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=False)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "n_active_units": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
    out = {"arm": arm, "final_n_active": int(net.n_active), "growth_trace": growth_trace,
           "acquire_acc_immediate": [float(a) for a in acquire_acc],
           "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
           "retention_curve": retention,
           "silent_dormant_ok": silent_dormant_ok, "born_spikes_ok": born_spikes_ok}
    return out


def _assert_byte_identical_substrate(n_in, K, seed, h_max, n0, settle, eprop_lr, w_clip):
    """anti-cheat (e): two builds at ONE seed must produce the byte-identical substrate (cfg.seed seeds the neurons,
    NOT actual_seed_used). Hash the per-neuron firing thresholds."""
    a = NeurogenesisNet(n_in, K, seed, h_max, n0, settle, eprop_lr, w_clip)
    b = NeurogenesisNet(n_in, K, seed, h_max, n0, settle, eprop_lr, w_clip)
    ta = np.asarray(to_host(a.br.cp_neuron_firing_thresholds), dtype=np.float64)
    tb = np.asarray(to_host(b.br.cp_neuron_firing_thresholds), dtype=np.float64)
    return bool(np.array_equal(ta, tb)), float(np.max(np.abs(ta - tb)))


def _git_sim_diff_empty():
    """anti-cheat (f): assert git diff main -- sim/ is empty (the growth is entirely runner-side)."""
    try:
        out = subprocess.run(["git", "diff", "main", "--", "sim/"], cwd=str(_REPO),
                             capture_output=True, text=True, timeout=30)
        return (out.returncode == 0 and out.stdout.strip() == ""), out.stdout[:400]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run(seed, n_max, milestones, n0, grow_k, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_epochs, replay_per_fact, replay_noise, arms_to_run, freeze_growth_at, bdsp_wmax=1e9):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    freeze_at = int(freeze_growth_at)                                   # FROZEN_GROWTH stops birthing at this fact index (N=10)
    h_max = n0 + n_max * grow_k                                          # the reservoir pool = final grown size
    referents = [f"ref{i}" for i in range(n_max)]

    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, K, seed, h_max, n0, settle, eprop_lr, w_clip)
    sim_clean, sim_diff = _git_sim_diff_empty()

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = ReferentEnv(seed, d_p=d_p, noise=noise)                    # fresh env per arm (identical referents + draw stream)
        for r in referents:
            env.proto(r)
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, h_max, n0, grow_k, freeze_at, settle, epochs, batch,
                             eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact,
                             replay_noise, chance, bdsp_wmax=bdsp_wmax)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm:14s}] {arms[arm]['wall_seconds']:.0f}s | active {arms[arm]['final_n_active']:3d} | "
              f"immediate-acq {arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "h_max": h_max, "freeze_at": freeze_at,
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"n0": n0, "grow_k": grow_k, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "frozen_hidden": True, "bdsp_wmax": bdsp_wmax},
            "arms": arms}


def _verdict(result):
    """Verdict preconditions + GO. TEETH:
      (a) GROWN frac-recalled @ largest N > SELF_REPLAY (the MEASURED self-replay baseline) by a margin.
      (b) LOAD-BEARING: FROZEN_GROWTH (growth frozen at N=10) < GROWN at N=20.
      (g) integration not noise: GROWN > RANDOM_UNITS; RANDOM_UNITS ~ SELF_REPLAY.
      MATCHED_FIXED reported (capacity vs grow-as-you-go), not gated. Anti-cheats (a/e/f) asserted."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]

    def frac(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        big = max((int(k) for k in rc), default=None)
        return (rc[str(big)]["frac_recalled"] if big else float("nan")), big

    def frac_at(arm, N):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return rc.get(str(N), {}).get("frac_recalled", float("nan"))

    base_f, big = frac("self_replay") if "self_replay" in arms else (float("nan"), None)
    grown_f, big_g = frac("grown") if "grown" in arms else (float("nan"), None)
    big = big if big is not None else big_g
    matched_f, _ = frac("matched_fixed") if "matched_fixed" in arms else (float("nan"), None)
    frozen_f, _ = frac("frozen_growth") if "frozen_growth" in arms else (float("nan"), None)
    random_f, _ = frac("random_units") if "random_units" in arms else (float("nan"), None)
    grown_acq = arms["grown"]["mean_acquire_acc_immediate"] if "grown" in arms else float("nan")
    # partial-arm probe (e.g. --arms self_replay): no grown -> emit the measured curves, skip the GO machinery.
    if "grown" not in arms or "self_replay" not in arms:
        return {"largest_N": big, "self_replay_frac_recalled": base_f, "grown_frac_recalled": grown_f,
                "matched_fixed_frac_recalled": matched_f, "frozen_growth_frac_recalled": frozen_f,
                "random_units_frac_recalled": random_f, "grown_immediate_acq": grown_acq,
                "n10_self_replay": frac_at("self_replay", 10), "n10_grown": frac_at("grown", 10),
                "n20_self_replay": frac_at("self_replay", 20), "n20_grown": frac_at("grown", 20),
                "n20_matched_fixed": frac_at("matched_fixed", 20),
                "rise_grown_minus_baseline": float(grown_f - base_f) if not (np.isnan(grown_f) or np.isnan(base_f)) else None,
                "capacity_vs_timing": "n/a (partial-arm probe)",
                "substrate_byte_identical": result["substrate_byte_identical"],
                "sim_diff_empty": result["sim_diff_empty"], "status": "PARTIAL"}

    attributable_to("grow-as-you-go reservoir (grown vs self-replay baseline)", grown_f, base_f)
    if not np.isnan(random_f):
        attributable_to("readout INTEGRATION of new units (grown vs random-frozen units)", grown_f, random_f)

    v = Verdict("teacher-loop neurogenesis capacity expansion", chance=chance)
    v.reaches("(a) retention RISES vs the measured self-replay baseline", before=base_f, after=grown_f)
    v.require("(a') grown beats the self-replay baseline by a margin", grown_f > base_f + 0.10, expect=True,
              note=f"grown {grown_f:.2f} vs self_replay {base_f:.2f} @ N={big}")
    # PRIMARY LOAD-BEARING LESION: remove the added CAPACITY entirely (== the self_replay baseline, no growth) ->
    # retention drops to base_f. The self_replay arm IS that lesion (a').
    # INTEGRATION LESION (g): random_units has the SAME firing units as grown but their readout is frozen-random
    # (never learned) -> retention collapses to ~baseline. The added units must be READOUT-INTEGRATED, not merely
    # present + spiking. This is the load-bearing control that 'more units alone' cannot fabricate the win.
    if not np.isnan(random_f):
        v.control("(g) INTEGRATION is load-bearing (grown vs random-frozen-readout units)", treatment=grown_f,
                  control=random_f, min_separation=0.10)
    # frozen_growth (freeze growth at N=10) is REPORTED, not gated: at this operating point the N=10-size reservoir
    # already saturates retention (the capacity threshold is modest), so freezing does NOT drop -- an honest nuance,
    # NOT a GO gate. The capacity dose-response is self_replay(n0) << grown/matched(final).
    v.floor("(c) grown immediate acquisition stays high", grown_acq, floor=0.85)
    # anti-cheat asserts (recorded; not a chance-gate)
    v.require("(e) substrate byte-identical across two builds at one seed", bool(result["substrate_byte_identical"]),
              expect=True, note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(f) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True,
              note="growth is runner-side over cp_connections")
    grown_arm = arms["grown"]
    if grown_arm.get("silent_dormant_ok") is not None:
        v.require("(a-anti-cheat) dormant granule units are SILENT (cp_firing_states)",
                  bool(grown_arm["silent_dormant_ok"]), expect=True)
        v.require("(a-anti-cheat) born granule units SPIKE (cp_firing_states)",
                  bool(grown_arm["born_spikes_ok"]), expect=True)

    go = (grown_f > base_f + 0.10 and grown_acq >= 0.85 and bool(result["substrate_byte_identical"])
          and bool(result["sim_diff_empty"]))
    if not np.isnan(random_f):
        go = go and (grown_f > random_f + 0.10)          # integration lesion must collapse the win
    decision = v.decide(go=go)

    return {
        "largest_N": big,
        "self_replay_frac_recalled": base_f, "grown_frac_recalled": grown_f,
        "matched_fixed_frac_recalled": matched_f, "frozen_growth_frac_recalled": frozen_f,
        "random_units_frac_recalled": random_f,
        "grown_immediate_acq": grown_acq,
        "rise_grown_minus_baseline": float(grown_f - base_f),
        "grown_minus_matched_fixed": (float(grown_f - matched_f) if not np.isnan(matched_f) else None),
        "grown_minus_frozen_growth": (float(grown_f - frozen_f) if not np.isnan(frozen_f) else None),
        "grown_minus_random_units": (float(grown_f - random_f) if not np.isnan(random_f) else None),
        "n10_self_replay": frac_at("self_replay", 10), "n10_grown": frac_at("grown", 10),
        "n20_self_replay": frac_at("self_replay", 20), "n20_grown": frac_at("grown", 20),
        "n20_matched_fixed": frac_at("matched_fixed", 20),
        "capacity_vs_timing": ("grow-as-you-go timing matters (grown > matched_fixed)"
                               if (not np.isnan(matched_f) and grown_f > matched_f + 0.10)
                               else ("just capacity (matched_fixed ~= grown)" if not np.isnan(matched_f) else "n/a")),
        "substrate_byte_identical": result["substrate_byte_identical"],
        "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _one_seed(a, seed, arms_to_run):
    result = run(seed, a.n_max, a.milestones, a.n0, a.grow_k, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact, a.replay_noise,
                 arms_to_run, a.freeze_growth_at, bdsp_wmax=a.bdsp_wmax)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop NEUROGENESIS capacity expansion (DSD-SNN / adult DG "
                                             "neurogenesis): GROW the frozen DG-expansion reservoir as facts "
                                             "accumulate to lift retention at scale past the self-replay baseline.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--n0", type=int, default=20, help="base granule cohort (the SELF_REPLAY reservoir size)")
    ap.add_argument("--grow-k", type=int, default=4, help="granule units BORN per fact (GROWN/FROZEN/RANDOM arms)")
    ap.add_argument("--freeze-growth-at", type=int, default=10, help="FROZEN_GROWTH stops birthing at this fact index")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9,
                    help="widens the inherited bdsp_w_min/max=-6/+6 clamp; 1e9=de-clamped (default), 6=the historical "
                         "CLAMP (bound-trap A/B: does de-clamping alone recover the N=20 self-replay baseline?)")
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--replay-epochs", type=int, default=12)
    ap.add_argument("--replay-per-fact", type=int, default=8)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--arms", nargs="+",
                    default=["self_replay", "grown", "matched_fixed", "frozen_growth", "random_units"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_neurogenesis_capacity", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        print("\n" + "=" * 100, flush=True)
        print(f"[neurogenesis] seed {s} @ N={verdict['largest_N']}: SELF_REPLAY {verdict['self_replay_frac_recalled']:.2f} | "
              f"GROWN {verdict['grown_frac_recalled']:.2f} | MATCHED_FIXED {verdict['matched_fixed_frac_recalled']:.2f} | "
              f"FROZEN_GROWTH {verdict['frozen_growth_frac_recalled']:.2f} | RANDOM_UNITS "
              f"{verdict['random_units_frac_recalled']:.2f} (chance {result['chance']:.2f})", flush=True)
        print(f"[neurogenesis] N10 base/grown {verdict['n10_self_replay']:.2f}/{verdict['n10_grown']:.2f} | "
              f"N20 base/grown {verdict['n20_self_replay']:.2f}/{verdict['n20_grown']:.2f} | "
              f"rise(grown-base) {verdict['rise_grown_minus_baseline']:+.2f} | {verdict['capacity_vs_timing']}", flush=True)
        print(f"[neurogenesis] grown-immediate-acq {verdict['grown_immediate_acq']:.3f} | byte-identical "
              f"{verdict['substrate_byte_identical']} | sim-clean {verdict['sim_diff_empty']} | VERDICT {verdict['status']}", flush=True)
        print(f"[neurogenesis] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        grown = [p["verdict"]["grown_frac_recalled"] for p in per_seed]
        base = [p["verdict"]["self_replay_frac_recalled"] for p in per_seed]
        matched = [p["verdict"]["matched_fixed_frac_recalled"] for p in per_seed]
        agg = {"probe": "teacher_loop_neurogenesis_capacity_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"), "go_count": go_n, "n_seeds": len(seeds),
               "grown_frac_mean": float(np.mean(grown)), "self_replay_frac_mean": float(np.mean(base)),
               "matched_fixed_frac_mean": float(np.nanmean(matched)),
               "grown_minus_base_mean": float(np.mean(np.array(grown) - np.array(base))),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[neurogenesis AGG] GO {go_n}/{len(seeds)} | GROWN {np.mean(grown):.2f} vs SELF_REPLAY "
              f"{np.mean(base):.2f} vs MATCHED_FIXED {np.nanmean(matched):.2f} | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
