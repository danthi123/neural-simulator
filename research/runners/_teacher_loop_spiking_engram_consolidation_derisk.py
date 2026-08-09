"""TEACHER-LOOP SLEEP-REPLAY with a SPIKING PATTERN-COMPLETING ENGRAM STORE (2026-08-09): biologize the engram
store the sleep-replay consolidation loop replays from. The baseline
(research/runners/_teacher_loop_sleep_replay_consolidation_derisk.py, finding
2026-08-08-...self-replay-beats-catastrophic-forgetting) SELF-GENERATES its replay + is robust (replay > no-replay
every seed; SCRAMBLE -> ~0) but UNDER-CONSOLIDATES: 6-seed replay frac_recalled mean 0.55 (range 0.20-0.90),
no-replay 0.13, ceiling = interleaved 8/10 = 0.8. The declared likely cause: the replayed engram is a LOSSY HOST
MEAN-VECTOR (`Hippocampus.encode` does `X.mean(axis=0)` -> a Python list; replay = engram + host noise), NOT the
brain's own neural attractor. This runner REPLACES that host mean-vector store with a SPIKING PATTERN-COMPLETING
ATTRACTOR and asks: does a higher-fidelity NEURAL engram raise replay retention toward the 0.8 ceiling?

THE SPIKING ENGRAM STORE (`SpikingAttractorHippocampus`, this file). A minimal Hopfield/CA3-style spiking
attractor -- the completion is done by NEURONS + RECURRENT SYNAPSES, not a host mean:
  * INDEX ATTRACTOR (M binary-spiking units): each fact i is a sparse random binary ASSEMBLY xi_i (a place/engram
    code). At encode the RECURRENT synapses W_rec are potentiated ONLINE by the sparse-covariance Hebbian rule
    (Tsodyks-Feigelman) so each stored assembly becomes a fixed-point attractor. Sparsity is held by feedback
    inhibition (E%-max / k-active, de Almeida-Idiart-Lisman) -- an inhibitory interneuron pool, not a host argmax
    over content.
  * ENGRAM READOUT SYNAPSES (W_if, n_in x M): at encode, each experienced percept f_t co-active with fact i's
    assembly potentiates the assembly-units' readout synapses ONLINE (Hebbian outer product W_if[:, xi_i] += eta*f_t;
    a per-unit synaptic-count gain c_j normalises at readout). NO `.mean(axis=0)` anywhere -- the engram lives in
    synaptic weights, read out by the SPIKING completed assembly.
REPLAY (sleep, teacher + world ABSENT): cue the index attractor with a DEGRADED pattern (partial xi_i + spurious
units borrowed from OTHER facts' assemblies + brain-owned flip noise), let the SPIKING RECURRENT dynamics
PATTERN-COMPLETE it (restore the dropped members AND reject the spurious ones), then read the engram out through
W_if via the completed assembly + brain-owned variability. That reconstruction is the replay draw fed to e-prop --
higher fidelity than the lossy mean-vector because completion DENOISES + de-contaminates the cue.

FIVE ARMS, same net/seed/epochs (the ONLY difference is the store + the replay content):
  * host_noreplay = the scaling baseline (no consolidation) -> the wall (~0.13).
  * host_replay   = the 2026-08-08 HOST MEAN-VECTOR store, replay ON -> the 0.55 baseline to beat.
  * spk_replay    = the SPIKING pattern-completing store, replay ON -> the treatment.
  * spk_scramble  = spiking store, replay compute IDENTICAL but assembly->label map SHUFFLED (content lesion;
                    self-generation teeth: forgetting must return).
  * spk_lesion    = spiking store, replay ON but the ATTRACTOR RECURRENTS LESIONED (no completion; the cue's
                    spurious units contaminate the readout -> fidelity + retention must drop => the neural
                    engram/completion is load-bearing).

TEETH (aggregate over seeds; single-seed SMOKE + 6-seed command below):
  (a) RETENTION RISES vs the host store: spk_replay frac_recalled > host_replay (same net/seed/epochs), toward 0.8.
  (b) COMPLETION LOAD-BEARING (neural engram): spk_lesion < spk_replay AND reconstruction fidelity
      (cosine to the true engram) drops when the recurrents are lesioned.
  (c) SELF-GENERATED: spk_scramble ~ host_noreplay (content-lesioned store forgets; the rise is the STORED content).
  (d) immediate acquisition stays perfect in spk_replay (>= 0.9).
  An HONEST NEGATIVE with teeth (spk_replay does NOT beat host_replay, or completion does not lift fidelity) is a
  first-class deliverable -- it maps that store-FIDELITY is NOT the consolidation bottleneck.
  grep-verify TEACHER/WORLD ABSENT + NO host mean-vector on the replay path:
      grep -n 'def generate_replay' research/runners/_teacher_loop_spiking_engram_consolidation_derisk.py
      -> no `env`, no `.mean(axis=0)` in its body (the true engram is captured MEASUREMENT-ONLY in the harness).

DISCIPLINE: reuse-by-import (_mk_net / _feat / _fit_readout_norm_world / _teach_fact / _fact_acc / _corrective_batch
+ ReferentEnv from the scaling + corrective-acquire de-risks; the HOST Hippocampus + _self_replay_consolidate from
the sleep-replay de-risk, unchanged). NO sim/ edit. cfg.seed via the seed= the net passes to CoreSimConfig.seed
(NOT actual_seed_used). SIM_BACKEND=cupy (3090) by default; numpy for a plumbing smoke. tools.lab attribution +
a Verdict preconditions block.

RUN (single-seed smoke as instructed):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_spiking_engram_consolidation_derisk --seeds 42 \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
      --out research/findings/raw/teacher_loop_spiking_engram_s42.json
  6-SEED (the decisive rise vs the 0.55 host-store mean), one process:
    SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_spiking_engram_consolidation_derisk --seeds 42 43 44 45 46 47 \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
      --out research/findings/raw/teacher_loop_spiking_engram_6seed.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")   # the 3090 by default; caller may set numpy for a plumbing smoke
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
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _mk_net, _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
# reuse the HOST mean-vector store + the offline self-replay consolidation UNCHANGED (the like-for-like baseline).
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import (  # noqa: E402
    Hippocampus, _self_replay_consolidate,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_spiking_engram.json"


# ===================== the SPIKING PATTERN-COMPLETING ENGRAM STORE (the biologized hippocampus) =====================
class SpikingAttractorHippocampus:
    """A minimal Hopfield/CA3-style SPIKING pattern-completing attractor that stores the brain's engrams in
    RECURRENT + READOUT SYNAPSES (not a host mean-vector). Interface matches the host `Hippocampus`
    (encode / generate_replay) so it is a drop-in for `_self_replay_consolidate`.

    INDEX ATTRACTOR: M binary-spiking units. Fact i owns a sparse assembly xi_i (a_active units). Recurrent
    synapses W_rec store the assemblies by the sparse-covariance Hebbian rule; feedback inhibition (k-active,
    E%-max) holds sparsity. A partial/noisy cue completes to the nearest stored assembly (spiking dynamics).
    ENGRAM READOUT: W_if (n_in x M) holds, per assembly unit, the Hebbian sum of the percepts experienced while
    that unit was active; a per-unit count c_j gives the readout gain. The completed spiking assembly reads the
    engram out through W_if. No `.mean(axis=0)` -- the engram is synaptic, reconstructed by the neurons."""

    def __init__(self, seed, n_in, M=400, sparsity=0.08, replay_noise=0.10,
                 cue_keep=0.5, cue_spurious=0.5, complete_steps=12, eta=1.0, lesion_recurrents=False):
        self.rng = np.random.default_rng(seed + 5150)   # a BRAIN-owned RNG (distinct from env's; self-generated)
        self.M = int(M)
        self.a = max(4, int(round(sparsity * M)))        # active units per assembly (E%-max target sparsity)
        self.n_in = int(n_in)
        self.replay_noise = float(replay_noise)
        self.cue_keep = float(cue_keep)                  # fraction of an assembly delivered in the cue (partial)
        self.cue_spurious = float(cue_spurious)          # spurious active units (from OTHER facts), as a fraction of a
        self.T = int(complete_steps)
        self.eta = float(eta)
        self.lesion = bool(lesion_recurrents)
        # synaptic stores (the engram lives HERE, not in a Python mean-vector)
        self.W_rec = np.zeros((self.M, self.M), dtype=np.float64)   # index-attractor recurrent synapses
        self.W_if = np.zeros((self.n_in, self.M), dtype=np.float64) # engram readout synapses (assembly -> features)
        self.c = np.zeros(self.M, dtype=np.float64)                 # per-unit synaptic count (readout gain)
        self.assemblies = []       # fact i -> np.array of its assembly unit indices (the place code)
        self.labels = []           # fact i -> class label bound to the assembly
        self._true_engrams = []    # MEASUREMENT ONLY (fidelity reference); NEVER read on the replay path

    # ---- assembly allocation: disjoint-ish sparse random codes (structural pattern separation) ----
    def _new_assembly(self):
        used = set(int(u) for a in self.assemblies for u in a)
        free = np.array([u for u in range(self.M) if u not in used], dtype=np.int64)
        if len(free) < self.a:                     # fall back to (rare) overlap if the store is near-full
            free = np.arange(self.M, dtype=np.int64)
        return np.sort(self.rng.choice(free, self.a, replace=False))

    def encode(self, X_experienced, cls):
        """Wake capture: allocate fact's assembly, imprint it into the RECURRENT synapses (sparse-covariance
        Hebbian) and imprint the experienced percepts into the READOUT synapses (Hebbian outer product, online).
        No mean is taken -- W_if accumulates a synaptic sum, c the presynaptic count."""
        X = np.asarray(X_experienced, dtype=np.float64)
        xi_idx = self._new_assembly()
        fid = len(self.assemblies)
        self.assemblies.append(xi_idx); self.labels.append(int(cls))
        # recurrent Hopfield imprint (sparse-covariance rule): W_rec += (x-p)(x-p)^T / (M p (1-p)), diagonal zero.
        p = self.a / self.M
        x = np.zeros(self.M, dtype=np.float64); x[xi_idx] = 1.0
        dv = (x - p)
        self.W_rec += np.outer(dv, dv) / (self.M * p * (1.0 - p))
        np.fill_diagonal(self.W_rec, 0.0)
        # engram readout imprint: each active assembly unit accumulates the experienced percepts (Hebbian).
        fsum = X.sum(axis=0)                        # synaptic accumulation (a SUM of pre*post, not a host mean)
        self.W_if[:, xi_idx] += self.eta * fsum[:, None]
        self.c[xi_idx] += X.shape[0]
        # MEASUREMENT-ONLY fidelity reference (NOT used by generate_replay / the replay path):
        self._true_engrams.append(fsum / max(1, X.shape[0]))
        return fid

    # ---- the spiking pattern-completion dynamics (k-active feedback inhibition = E%-max) ----
    def _complete(self, s0):
        """s0: binary M-vector cue. Iterate binary-spiking recurrent dynamics with k-active feedback inhibition;
        return the completed binary assembly. Lesion (recurrents zeroed) => no completion => the cue is returned."""
        if self.lesion:
            return s0.copy()                        # no recurrent attractor: feedforward cue only (contaminated)
        s = s0.astype(np.float64).copy()
        for _ in range(self.T):
            drive = self.W_rec @ s                  # recurrent synaptic drive
            if not np.any(drive > 0):
                break
            thr = np.partition(drive, self.M - self.a)[self.M - self.a]   # k-active inhibition (E%-max threshold)
            s_new = (drive >= thr).astype(np.float64)
            if s_new.sum() > self.a:                # break ties down to exactly a spikes (sharpest units win)
                keep = np.argsort(-drive)[:self.a]
                s_new = np.zeros(self.M); s_new[keep] = 1.0
            if np.array_equal(s_new, s):
                break
            s = s_new
        return s

    def _cue_for(self, fid):
        """Build a DEGRADED cue for fact fid: keep a fraction of its assembly, drop the rest, and add spurious
        units borrowed from OTHER facts' assemblies (brain-owned flip noise). The attractor must clean this."""
        xi = self.assemblies[fid]
        n_keep = max(1, int(round(self.cue_keep * len(xi))))
        keep = self.rng.choice(xi, n_keep, replace=False)
        s = np.zeros(self.M, dtype=np.float64); s[keep] = 1.0
        others = np.array([u for j, a in enumerate(self.assemblies) if j != fid for u in a], dtype=np.int64)
        n_spur = int(round(self.cue_spurious * len(xi)))
        if n_spur > 0 and len(others) > 0:
            spur = self.rng.choice(others, min(n_spur, len(others)), replace=False)
            s[spur] = 1.0
        return s

    def _readout(self, s_complete):
        """Read the engram out of W_if via the completed spiking assembly, normalised by the synaptic count gain."""
        active = np.where(s_complete > 0.5)[0]
        if len(active) == 0:
            return np.zeros(self.n_in, dtype=np.float64)
        num = self.W_if[:, active].sum(axis=1)
        den = self.eta * self.c[active].sum()
        return num / den if den > 0 else num

    def reconstruct(self, fid):
        """Deterministic (no brain-noise) reconstruction of fact fid's engram via cue -> completion -> readout.
        Used for the fidelity teeth (compared to the MEASUREMENT-ONLY _true_engrams)."""
        s = self._complete(self._cue_for(fid))
        return self._readout(s)

    def generate_replay(self, per_fact, scramble_labels=False):
        """Sleep self-generation: for each stored fact, cue the index attractor with a degraded pattern, let the
        SPIKING recurrents PATTERN-COMPLETE it, read the engram out of the readout synapses, and add brain-owned
        variability -> per_fact replay draws. NO env, NO teacher, NO host mean-vector: the pattern is reconstructed
        from the neural store. scramble_labels (the content lesion) shuffles which class each assembly is replayed
        AS while the compute is identical."""
        if not self.assemblies:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
        labels = list(self.labels)
        if scramble_labels:
            labels = list(self.rng.permutation(labels))
        Xs, ys = [], []
        for fid in range(len(self.assemblies)):
            for _ in range(per_fact):
                s = self._complete(self._cue_for(fid))          # spiking pattern completion (per-draw fresh cue)
                engram = self._readout(s)                        # synaptic engram readout via the completed assembly
                draw = engram + self.replay_noise * self.rng.standard_normal(self.n_in)
                Xs.append(draw)
                ys.append(int(labels[fid]))
        return np.asarray(Xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)

    # ---- fidelity teeth (measurement only): mean cosine( reconstruct(fid), true_engram(fid) ) ----
    def mean_fidelity(self):
        cs = []
        for fid in range(len(self.assemblies)):
            r = self.reconstruct(fid); t = self._true_engrams[fid]
            nr = np.linalg.norm(r); nt = np.linalg.norm(t)
            if nr > 1e-9 and nt > 1e-9:
                cs.append(float(np.dot(r, t) / (nr * nt)))
        return float(np.mean(cs)) if cs else float("nan")


# ================================= one arm of the sequential curriculum =================================
def _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws,
             milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance,
             spk_M, spk_sparsity, spk_cue_keep, spk_cue_spurious, spk_steps):
    """Teach the referents SEQUENTIALLY into ONE brain. arm in
    {host_noreplay, host_replay, spk_replay, spk_scramble, spk_lesion}."""
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)
    is_spk = arm.startswith("spk")
    if is_spk:
        hippo = SpikingAttractorHippocampus(
            seed, n_in, M=spk_M, sparsity=spk_sparsity, replay_noise=replay_noise,
            cue_keep=spk_cue_keep, cue_spurious=spk_cue_spurious, complete_steps=spk_steps,
            lesion_recurrents=(arm == "spk_lesion"))
    else:
        hippo = Hippocampus(seed, replay_noise=replay_noise)
    do_replay = arm != "host_noreplay"
    scramble = arm == "spk_scramble"

    acquire_acc = []
    retention = {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acq = _fact_acc(net, env, r, i, n=test_n)
        acquire_acc.append(acq)
        hippo.encode(X, i)
        if do_replay:
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=scramble)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
    out = {
        "arm": arm,
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
    }
    if is_spk:
        out["mean_reconstruction_fidelity"] = hippo.mean_fidelity()   # completion fidelity (teeth b)
        out["store_kind"] = "spiking_attractor"
        out["spk_a_active"] = hippo.a
    else:
        out["store_kind"] = "host_mean_vector"
    return out


ARMS = ("host_noreplay", "host_replay", "spk_replay", "spk_scramble", "spk_lesion")


def run_seed(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
             test_n, replay_epochs, replay_per_fact, replay_noise, spk_M, spk_sparsity, spk_cue_keep,
             spk_cue_spurious, spk_steps):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)
    arms = {}
    for arm in ARMS:
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)      # same teaching percepts across arms (like-for-like)
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip,
                             n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance,
                             spk_M, spk_sparsity, spk_cue_keep, spk_cue_spurious, spk_steps)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        fid = arms[arm].get("mean_reconstruction_fidelity")
        fids = f" | fidelity {fid:.3f}" if fid is not None else ""
        print(f"  [seed {seed}] arm {arm:14s} {arms[arm]['wall_seconds']:5.0f}s | immediate-acq "
              f"{arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}{fids}", flush=True)
    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones, "arms": arms}


def _frac(seed_result, arm):
    rc = seed_result["arms"][arm]["retention_curve"]
    big = max((int(k) for k in rc), default=None)
    return rc[str(big)]["frac_recalled"] if big else float("nan")


def _verdict(seed_results):
    """Aggregate over seeds. TEETH:
      (a) spk_replay frac > host_replay frac (rise vs the 0.55 host-store baseline, same net/seed/epochs).
      (b) spk_lesion < spk_replay AND fidelity(replay) > fidelity(lesion) (completion/neural-engram load-bearing).
      (c) spk_scramble ~ host_noreplay (content-lesioned store forgets -> the rise is the STORED content).
      (d) spk_replay immediate acquisition >= 0.9."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict

    def mean_frac(arm):
        return float(np.mean([_frac(r, arm) for r in seed_results]))

    def mean_key(arm, key):
        return float(np.mean([r["arms"][arm][key] for r in seed_results if key in r["arms"][arm]]))

    host_nore = mean_frac("host_noreplay")
    host_rep = mean_frac("host_replay")
    spk_rep = mean_frac("spk_replay")
    spk_scr = mean_frac("spk_scramble")
    spk_les = mean_frac("spk_lesion")
    spk_acq = mean_key("spk_replay", "mean_acquire_acc_immediate")
    fid_rep = mean_key("spk_replay", "mean_reconstruction_fidelity")
    fid_les = mean_key("spk_lesion", "mean_reconstruction_fidelity")
    chance = seed_results[0]["chance"]

    # attribution: the rise is the SPIKING store (spk_replay vs host_replay) and the STORED content (vs scramble).
    attributable_to("spiking store vs host mean-vector (spk_replay vs host_replay)", spk_rep, host_rep)
    attributable_to("stored content, self-generated (spk_replay vs spk_scramble)", spk_rep, spk_scr)
    attributable_to("attractor completion (spk_replay vs spk_lesion)", spk_rep, spk_les)

    # (a) demands a MEANINGFUL rise, not an epsilon. A bare `spk_rep > host_rep` passes on +0.017 (3.1%
    # attributable, 96.9% in the control) -- that is a TIE, and reporting it as a rise would be exactly the
    # threshold-overclaim verify-go forbids. RISE_MARGIN is the real-effect bar for the headline hypothesis.
    RISE_MARGIN = 0.10
    # The PRECONDITIONS registered below are INSTRUMENT-VALIDITY checks: they prove the store is live, neural,
    # load-bearing and self-generated, so that a headline miss is an EARNED NEGATIVE (NO-GO), not an UNDEFINED
    # instrument failure. Teeth (a) -- the headline hypothesis -- drives ONLY the `go` boolean (a real rise over
    # the host store), NOT a precondition; registering it as a require would collapse an honest negative to
    # UNDEFINED. This is the b/c/d-pass-but-a-fails shape the de-risk is built to detect.
    v = Verdict("spiking pattern-completing engram store for sleep-replay consolidation", chance=chance)
    v.reaches("(a) retention vs the HOST mean-vector store (raw before/after; the headline)",
              before=host_rep, after=spk_rep,
              note=f"rise {spk_rep-host_rep:+.3f} vs the >= {RISE_MARGIN} real-effect bar")
    v.floor("(a') spk_replay beats the no-replay wall (instrument produces retention)", spk_rep,
            floor=host_nore + 1e-9)
    v.control("(b) attractor completion load-bearing (spk_replay vs spk_lesion)", treatment=spk_rep,
              control=spk_les, min_separation=0.0)
    v.require("(b') completion raises reconstruction fidelity (replay > lesion => engram is neural)",
              fid_rep > fid_les, expect=True, note=f"fidelity replay {fid_rep:.3f} vs lesion {fid_les:.3f}")
    v.require("(c) scramble forgets like no-replay (self-generated content)", spk_scr <= host_nore + 0.10,
              expect=True, note=f"scramble {spk_scr:.2f} vs host_noreplay {host_nore:.2f}")
    v.floor("(d) immediate acquisition stays perfect (spk_replay)", spk_acq, floor=0.9)

    # HEADLINE GO = teeth (a): a REAL rise over the host mean-vector store (margin, not epsilon). The mechanism
    # can be genuine + load-bearing + self-generated (preconditions pass) yet NOT beat the mean (go=False) -> a
    # clean, EARNED honest negative.
    headline_rise = spk_rep - host_rep
    go = (headline_rise >= RISE_MARGIN)
    decision = v.decide(go=go)
    return {
        "host_noreplay_frac": host_nore, "host_replay_frac": host_rep,
        "spk_replay_frac": spk_rep, "spk_scramble_frac": spk_scr, "spk_lesion_frac": spk_les,
        "spk_replay_immediate_acq": spk_acq,
        "fidelity_replay": fid_rep, "fidelity_lesion": fid_les,
        "rise_spk_minus_host": float(spk_rep - host_rep),
        "completion_margin_replay_minus_lesion": float(spk_rep - spk_les),
        "self_gen_margin_replay_minus_scramble": float(spk_rep - spk_scr),
        **decision,
    }


def main():
    ap = argparse.ArgumentParser(description="Sleep-replay consolidation with a SPIKING pattern-completing engram "
                                             "store replacing the host mean-vector; measure the retention rise.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--replay-epochs", type=int, default=24)
    ap.add_argument("--replay-per-fact", type=int, default=16)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    # spiking attractor store hyperparameters
    ap.add_argument("--spk-M", type=int, default=400, help="index-attractor units")
    ap.add_argument("--spk-sparsity", type=float, default=0.08, help="assembly active fraction (E%%-max target)")
    ap.add_argument("--spk-cue-keep", type=float, default=0.5, help="fraction of the assembly delivered in the cue")
    ap.add_argument("--spk-cue-spurious", type=float, default=0.5, help="spurious cue units (frac of a) from OTHER facts")
    ap.add_argument("--spk-steps", type=int, default=12, help="pattern-completion recurrent steps")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for s in a.seeds:
        print(f"[spiking-engram] seed {s} ...", flush=True)
        seed_results.append(run_seed(
            s, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr, a.w_clip,
            a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact, a.replay_noise,
            a.spk_M, a.spk_sparsity, a.spk_cue_keep, a.spk_cue_spurious, a.spk_steps))
    verdict = _verdict(seed_results)
    summary = {"probe": "teacher_loop_spiking_engram_consolidation", "seeds": a.seeds,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": len(a.seeds) == 1,
               "elapsed_seconds": round(time.time() - t0, 1),
               "config": {"n_max": a.n_max, "epochs": a.epochs, "replay_epochs": a.replay_epochs,
                          "replay_per_fact": a.replay_per_fact, "replay_noise": a.replay_noise, "n_draws": a.n_draws,
                          "spk_M": a.spk_M, "spk_sparsity": a.spk_sparsity, "spk_cue_keep": a.spk_cue_keep,
                          "spk_cue_spurious": a.spk_cue_spurious, "spk_steps": a.spk_steps},
               "seed_results": seed_results, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print(f"[spiking-engram] seeds {a.seeds} @ N={a.n_max}  (chance {seed_results[0]['chance']:.2f})", flush=True)
    print(f"  HOST no-replay {verdict['host_noreplay_frac']:.2f} | HOST replay {verdict['host_replay_frac']:.2f} "
          f"(the 0.55 baseline) | SPK replay {verdict['spk_replay_frac']:.2f} | "
          f"SPK scramble {verdict['spk_scramble_frac']:.2f} | SPK lesion {verdict['spk_lesion_frac']:.2f}", flush=True)
    print(f"  rise(spk-host) {verdict['rise_spk_minus_host']:+.2f} | completion margin(spk-lesion) "
          f"{verdict['completion_margin_replay_minus_lesion']:+.2f} | self-gen margin(spk-scramble) "
          f"{verdict['self_gen_margin_replay_minus_scramble']:+.2f}", flush=True)
    print(f"  fidelity replay {verdict['fidelity_replay']:.3f} vs lesion {verdict['fidelity_lesion']:.3f} | "
          f"SPK immediate-acq {verdict['spk_replay_immediate_acq']:.3f} | VERDICT {verdict['status']}", flush=True)
    print(f"[spiking-engram] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
