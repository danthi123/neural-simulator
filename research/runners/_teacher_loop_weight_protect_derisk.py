"""TEACHER-LOOP WEIGHT-PROTECTION DE-RISK (2026-08-09): raise sequential-teacher retention PAST the sleep-replay
plateau (~0.55 frac-recalled at extreme sequential interference; finding 8d2510d3, 2026-06-30-100M-C2-scaleup)
with a PROVEN continual-learning mechanism, NOT more replay (replay MAGNITUDE is REFUTED as the lever, main 8d2510d3).

THE MECHANISM (two proven, biology-faithful components layered ON TOP of the brain's own self-replay):
  (1) FIXED ORTHOGONAL READOUT TARGETS  == the dentate-gyrus DECORRELATION / pattern-separation job. Each fact gets
      a FIXED, mutually-ORTHOGONAL distributed target code (rows of a random ORTHONORMAL matrix, t_i . t_j = 0),
      PREDEFINED once and NEVER learned -- the readout regresses to these stable centres (MSE) and classifies by
      nearest centre, instead of a drifting learnable softmax where teaching fact i pulls every other class prototype.
      (PS-SNN, Hu et al. 2026 Sci.Rep. -- replace the drifting classifier with predefined orthogonal class centres.)
  (2) SYNAPTIC-IMPORTANCE-GATED PLASTICITY (EWC diagonal Fisher / Zenke SI / the project Phase-1.4 gate-freeze that
      got 103% retention). After each fact consolidates, the brain measures a per-readout-synapse IMPORTANCE from its
      OWN replayed activity -- the diagonal Fisher F_jk = E[(r_j * delta_k)^2] over the hippocampal store (the exact
      curvature of the readout loss in each synapse, computed on the brain's OWN synapses, NO host freeze-list). The
      NEXT fact's plasticity is GATED per-synapse: gain_jk = 1/(1 + lambda * Fhat_jk). High-importance (consolidated)
      synapses RESIST change; new-fact plasticity flows to the LOW-importance synapses. This is metaplasticity /
      consolidated-synapse resistance (EWC Kirkpatrick 2017; Zenke SI 2017; Phase-1.4 gate-freeze).

BRAIN-BASED / one-substrate discipline. The readout WEIGHTS are the substrate's OWN synapses (cp_connections.data at
the H_last->out edges -- the Bellec leaky-readout the port already trains); we read them, move them by the readout
gradient, and WRITE them back into cp_connections each fact, exactly as the port's e-prop does. The hidden layers are
the FIXED spiking reservoir (the port's own framing: "a linear readout on a fixed random spiking reservoir"), so ALL
interference and ALL protection live on the readout synapses -- the SPEC's target ("protect the readout synapses that
carry earlier facts"). The importance is the brain's own per-synapse Fisher; the gate is a per-synapse plasticity
rate. NO sim/ edit, NO port edit (reuse-by-import). cfg.seed via the seed= the net passes to CoreSimConfig.seed
(NOT actual_seed_used). Self-replay is teacher/world-ABSENT (the hippocampus self-generates from stored engrams).

THE MECHANISM = softmax readout + self-replay + synaptic-importance-gated plasticity (EWC/SI on the brain's own
readout synapses). FOUR ARMS, one net-build / seed / epoch-budget / replay-schedule -- the ONLY difference is the gate
(and, for the ortho variant, the fixed target code):
  * replay     = softmax readout + self-replay, NO protection. THE BASELINE (the sleep-replay regime, ~0.55 6-seed mean).
  * protect    = + synaptic-importance-gated plasticity (lambda>0). THE MECHANISM.
  * scramble   = SAME protection AMOUNT, importance PERMUTED across synapses (wrong synapses). Structure control.
  * orthoprot  = fixed orthogonal PS-SNN targets + protection. Reported for the record (does orthogonality add anything).

TEETH (single-seed SMOKE; 6-seed command below -- the 0.55 is a 6-SEED MEAN, so the decisive read is the seed mean):
  (a) RETENTION RISES: protect.frac_recalled@N > replay.frac_recalled@N (same net/seed/epochs), toward 0.8+.
  (b1) PROTECTION LOAD-BEARING: removing the gate IS the replay arm -> protect.frac > replay.frac by a margin.
  (b2) IMPORTANCE STRUCTURE LOAD-BEARING: protect.frac > scramble.frac -- the benefit needs WHICH synapses are
       protected (the Fisher structure), not merely a global plasticity brake (scramble applies the same amount, shuffled).
  (c) IMMEDIATE ACQUISITION STAYS HIGH + the tradeoff is MEASURED: protect's mean immediate-acquisition (accuracy of
      each fact right after teaching, BEFORE consolidation) stays high (>=0.85); protect-vs-replay immediate acq is
      reported so the EWC "slightly compromises new learning" tradeoff is quantified, not hidden.
  (d) CAPACITY HOLDS: mean_retained_acc does not collapse.
  Importance is the brain's OWN synapses' Fisher (grep: `update_importance` reads r + W, no label freeze-list; the
  self-replay `_self_replay_features` has no `env` param; the readout weights are committed back into cp_connections).

RUN (3090, single-seed smoke as instructed):
  SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_weight_protect_derisk --seed 42 \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --n-draws 32 --lam 8.0 \
      --out research/findings/raw/teacher_loop_weight_protect_s42.json
  6-SEED (GO needs the protect>replay rise + the structure control 6/6 at 42..47), one seed per process in parallel:
    for s in 42 43 44 45 46 47; do SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_weight_protect_derisk --seed $s \
      --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --n-draws 32 --lam 8.0 \
      --out research/findings/raw/teacher_loop_weight_protect_s$s.json & done; wait
  PLUMBING SMOKE (fast, numpy): ... SIM_BACKEND=numpy ... --n-max 3 --milestones 1 3 --epochs 8 \
      --replay-epochs 6 --n-draws 12 --settle-steps 12 --test-n 20
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
# reuse-by-import: the teacher-loop SCALING net-build/world helpers + the corrective-acquire world + the sleep-replay
# HIPPOCAMPUS (the brain's own engram store). NO sim/ edit, NO re-derivation of the baseline.
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _mk_net, _feat, _fit_readout_norm_world, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import Hippocampus  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_weight_protect.json"


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ======================= the READOUT as the substrate's own H_last->out synapses =======================
class ProtectedReadout:
    """The Bellec leaky-readout the port trains, but with (1) an optional FIXED ORTHOGONAL target code and (2) an
    optional per-synapse IMPORTANCE gate. The weights ARE the substrate's cp_connections H_last->out edges: read at
    build, moved by the readout gradient, WRITTEN BACK into cp_connections each fact. The hidden reservoir is FIXED,
    so these readout synapses carry every fact and are the sole site of interference AND of protection."""

    def __init__(self, net, K, ortho, seed, lr=0.5, w_clip=4000.0, logit_temp=1.0):
        from sim.backend import to_host
        self.net = net
        self.K = int(K)
        self.ortho = bool(ortho)
        self.lr = float(lr)
        self.w_clip = float(w_clip)
        self.temp = float(logit_temp)
        self._idx = net._data_idx_flat[-1]                          # flat indices of the readout edges in cp data
        self.n_h = int(net.sizes_phys[-2])                          # last-hidden width (readout fan-in)
        # authoritative host copy of the substrate's readout synapses (port zero-inits them); mirrored back each fact.
        W0 = np.asarray(to_host(net.br.cp_connections.data[self._idx]), dtype=np.float64).reshape(self.n_h, self.K)
        self.W = W0.copy()
        # FIXED orthogonal distributed target codes (the DG-decorrelation / PS-SNN centres): rows of a random
        # ORTHONORMAL K x K matrix, PREDEFINED once, NEVER learned. (ortho OFF -> one-hot == the drifting softmax.)
        rng = np.random.default_rng(seed + 2027)
        Q, _ = np.linalg.qr(rng.standard_normal((self.K, self.K)))  # orthonormal rows: Q Q^T = I
        self.C = Q.astype(np.float64) if self.ortho else np.eye(self.K, dtype=np.float64)
        # per-synapse importance (diagonal Fisher, accumulated online over the store) and its plasticity gate.
        self.F = np.zeros((self.n_h, self.K), dtype=np.float64)
        self.gain = np.ones((self.n_h, self.K), dtype=np.float64)   # 1.0 = full plasticity (protect OFF default)

    def features(self, X):
        """Forward each input percept through the FIXED spiking reservoir once -> standardized last-hidden
        eligibility r (the readout feature the port uses). Teacher/world absent here: pure substrate forward."""
        net = self.net
        return np.asarray([net._readout_feature(net._forward_record(X[i])[0]) for i in range(len(X))],
                          dtype=np.float64)                          # (n, n_h)

    def _out(self, R):
        return R @ self.W                                           # (n, K) readout activation

    def train(self, R, y, epochs, batch, rng, apply_gain=True):
        """Move the readout synapses by the (gated) gradient. ortho -> MSE to the fixed code C[y]; else -> softmax
        cross-entropy (the drifting readout). gain multiplies the gradient PER SYNAPSE: high-importance synapses
        (low gain) resist change -> new-fact plasticity flows to the low-importance synapses. apply_gain=False ->
        FULL plasticity (used during SLEEP replay: metaplasticity gates the NEW WAKE encoding, while replay is the
        consolidation phase that must be free to reconsolidate every fact)."""
        y = np.asarray(y, dtype=np.int64)
        gain = self.gain if apply_gain else 1.0
        n = len(R)
        for _ in range(int(epochs)):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                Rb = R[b]; yb = y[b]
                out = Rb @ self.W                                   # (bsz, K)
                if self.ortho:
                    D = out - self.C[yb]                            # MSE gradient to the FIXED orthogonal centre
                    # NLMS/Widrow-Hoff: normalize by the input second moment so the MSE step is stable at the same
                    # nominal lr the softmax path uses (a delta rule with input normalization -- the ill-conditioned
                    # raw MSE step blows the readout weights to the clip; the softmax delta self-limits, MSE does not).
                    step = self.lr / ((Rb ** 2).sum(axis=1).mean() + 1e-6)
                else:
                    p = _softmax(out / self.temp)
                    D = p.copy(); D[np.arange(len(yb)), yb] -= 1.0   # softmax cross-entropy delta (drifting readout)
                    step = self.lr
                grad = (Rb.T @ D) / max(1, len(b))                  # (n_h, K)
                self.W = np.clip(self.W - step * gain * grad, -self.w_clip, self.w_clip)

    def predict(self, R):
        out = R @ self.W
        if self.ortho:
            return np.argmax(out @ self.C.T, axis=1)                # nearest fixed orthogonal centre
        return np.argmax(out, axis=1)

    def acc(self, X, cls):
        R = self.features(X)
        return float(np.mean(self.predict(R) == int(cls)))

    def commit_to_substrate(self):
        """Write the readout synapses BACK into cp_connections -- the weights live in the substrate, as the port's
        e-prop leaves them (one-substrate faithfulness)."""
        xp = self.net._xp
        self.net.br.cp_connections.data[self._idx] = xp.asarray(self.W.ravel().astype(np.float32))

    def update_importance(self, R, y, lam, scramble_rng=None):
        """Diagonal FISHER over the store's replayed activity (the brain's OWN synapses): F_jk = E[(r_j*delta_k)^2],
        the readout-loss curvature per synapse. gain_jk = 1/(1 + lam*Fhat_jk) protects the important ones. lam=0 ->
        gain stays all-ones (protection ablated). Uses ONLY r (substrate features) and W -- no host label freeze-list.
        scramble_rng (the STRUCTURE control): permute the SAME gain VALUES across synapses -> identical amount of
        protection, applied to the WRONG synapses. If protection needs the importance STRUCTURE (which synapses),
        scrambling it removes the benefit; if it were merely a global plasticity brake, scrambling would not matter."""
        y = np.asarray(y, dtype=np.int64)
        out = R @ self.W
        if self.ortho:
            D = out - self.C[y]
        else:
            p = _softmax(out / self.temp)
            D = p.copy(); D[np.arange(len(y)), y] -= 1.0
        # F_jk = mean_n (R_nj * D_nk)^2 = curvature of the readout loss in synapse (j->k), on the store's own activity.
        self.F = ((R ** 2).T @ (D ** 2)) / max(1, len(R))          # (n_h, K)
        if lam > 0:
            Fhat = self.F / (self.F.mean() + 1e-8)
            g = 1.0 / (1.0 + float(lam) * Fhat)
            if scramble_rng is not None:
                g = scramble_rng.permutation(g.ravel()).reshape(g.shape)   # same values, wrong synapses
            self.gain = g
        else:
            self.gain = np.ones_like(self.F)


def _self_replay_features(readout, hippocampus, per_fact):
    """SLEEP self-generation: the hippocampus reactivates its stored engrams and generates replay percepts from its
    OWN variability (no env, no teacher -- this function has no `env` param); the FIXED reservoir turns them into
    readout features. The brain replaying its own memory, not the world re-presenting it."""
    Xr, yr = hippocampus.generate_replay(per_fact, scramble_labels=False)
    if len(Xr) == 0:
        return np.zeros((0, readout.n_h)), np.zeros((0,), dtype=np.int64)
    return readout.features(Xr), yr


# ================================= one arm of the sequential curriculum =================================
def _run_arm(arm, ortho, lam, scramble, seed, referents, env, K, n_in, hidden, settle, epochs, batch, readout_lr,
             w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance):
    net = _mk_net(n_in, K, seed, hidden, settle, readout_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)             # readout-norm fit ONCE over the world (as baseline)
    readout = ProtectedReadout(net, K, ortho=ortho, seed=seed, lr=readout_lr, w_clip=w_clip,
                               logit_temp=net.logit_temp)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)                 # brain-owned RNG for consolidation shuffling
    scr_rng = np.random.default_rng(seed + 999) if scramble else None
    hippo = Hippocampus(seed, replay_noise=replay_noise)

    acquire_acc = []
    retention = {}
    for i, r in enumerate(referents):
        # --- WAKE: the teacher teaches fact i from the world; protection gate (from facts 0..i-1) is IN FORCE ---
        X, y = _corrective_batch(env, r, i, n_draws)
        Rw = readout.features(X)
        readout.train(Rw, y, epochs, batch, teach_rng)
        acquire_acc.append(readout.acc([_feat(env, r) for _ in range(test_n)], i))   # immediate, BEFORE replay
        hippo.encode(X, i)
        # --- SLEEP: offline self-replay consolidation (teacher + world ABSENT); protection gate still in force ---
        if replay_per_fact > 0:
            Rr, yr = _self_replay_features(readout, hippo, replay_per_fact)
            if len(Rr):
                readout.train(Rr, yr, replay_epochs, batch, brain_rng, apply_gain=False)  # replay = free reconsolidation
        readout.commit_to_substrate()                             # weights live in the substrate's cp_connections
        # --- recompute per-synapse importance from the store (the brain's own Fisher) for the NEXT fact's gate ---
        Rs, ys = _self_replay_features(readout, hippo, replay_per_fact)
        if len(Rs):
            readout.update_importance(Rs, ys, lam, scramble_rng=scr_rng)
        N = i + 1
        if N in milestones:
            accs = [readout.acc([_feat(env, referents[j]) for _ in range(test_n)], j) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
                "importance_gain_min": float(readout.gain.min()), "importance_gain_mean": float(readout.gain.mean()),
            }
    return {
        "arm": arm, "ortho": bool(ortho), "lam": float(lam), "scramble": bool(scramble),
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
    }


def run(seed, n_max, milestones, hidden, settle, epochs, batch, readout_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_epochs, replay_per_fact, replay_noise, lam):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))

    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    # (ortho, lam, scramble) toggles per arm; SAME net-build/seed/epoch-budget/replay-schedule across arms -- the ONLY
    # difference is the per-synapse importance gate (and, for the ortho variant, the fixed orthogonal target code).
    arm_cfg = {
        "replay":     (False, 0.0, False),   # softmax readout + self-replay, NO protection -> THE BASELINE
        "protect":    (False, lam, False),   # + synaptic-importance-gated plasticity (EWC/SI) -> THE MECHANISM
        "scramble":   (False, lam, True),    # SAME protection amount, importance permuted across synapses (structure control)
        "orthoprot":  (True,  lam, False),   # fixed orthogonal targets (PS-SNN) + protection -> the ortho variant
    }
    arms = {}
    for arm, (ortho, arm_lam, scr) in arm_cfg.items():
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)               # SAME teaching percepts across arms (like-for-like)
        arms[arm] = _run_arm(arm, ortho, arm_lam, scr, seed, referents, env, K, n_in, hidden, settle, epochs, batch,
                             readout_lr, w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact,
                             replay_noise, chance)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm:9s}] {arms[arm]['wall_seconds']:.0f}s | immediate-acq "
              f"{arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "readout_lr": readout_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "lam": lam},
            "arms": arms}


def _verdict(result):
    """Verdict preconditions block + GO. THE MECHANISM = softmax readout + synaptic-importance-gated plasticity
    (EWC/SI on the brain's own readout synapses). TEETH:
      (a) retention rises: protect.frac > replay.frac (toward 0.8+).
      (b1) protection load-bearing (remove the gate == replay): protect.frac > replay.frac by a margin.
      (b2) importance STRUCTURE load-bearing: protect.frac > scramble.frac (same protection amount, wrong synapses).
      (c) immediate acquisition stays high in protect (>=0.85); the EWC new-learning tradeoff (protect vs replay) reported.
      (d) capacity holds: protect.mean_retained_acc not collapsed.
    orthoprot (fixed orthogonal PS-SNN targets + protection) is reported for the record -- an honest read on whether
    orthogonal distributed codes add anything in this regime."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    rc = {a: result["arms"][a]["retention_curve"] for a in result["arms"]}
    big = max((int(k) for k in rc["protect"]), default=None)
    key = str(big)
    fr = {a: rc[a][key]["frac_recalled"] for a in rc}
    protect_acq = result["arms"]["protect"]["mean_acquire_acc_immediate"]
    replay_acq = result["arms"]["replay"]["mean_acquire_acc_immediate"]
    protect_meanret = rc["protect"][key]["mean_retained_acc"]
    chance = result["chance"]

    # the rise is attributable to the importance gate (protect vs replay) and to its STRUCTURE (protect vs scramble),
    # NOT to extra compute (all arms match net-build/epochs/replay-schedule).
    attributable_to("importance gate (protect vs replay)", fr["protect"], fr["replay"])
    attributable_to("importance STRUCTURE (protect vs scramble)", fr["protect"], fr["scramble"])

    v = Verdict("teacher-loop weight-protection continual learning", chance=chance)
    v.reaches("(a) retention RISES vs replay-only baseline", before=fr["replay"], after=fr["protect"])
    v.require("(a') protect reaches toward 0.8+", fr["protect"] >= 0.8, expect=True,
              note=f"protect frac_recalled@N={big} = {fr['protect']:.2f}")
    v.control("(b1) importance gate load-bearing (protect vs replay)", treatment=fr["protect"],
              control=fr["replay"], min_separation=0.10)
    v.control("(b2) importance STRUCTURE load-bearing (protect vs scramble)", treatment=fr["protect"],
              control=fr["scramble"], min_separation=0.10)
    v.floor("(c) immediate acquisition stays high (protect)", protect_acq, floor=0.85)
    v.floor("(d) capacity holds (protect mean retained acc)", protect_meanret, floor=max(0.5, chance + 0.15))
    go = (fr["protect"] > fr["replay"] + 0.10 and fr["protect"] >= 0.8 and fr["protect"] > fr["scramble"] + 0.10
          and protect_acq >= 0.85 and protect_meanret >= max(0.5, chance + 0.15))
    decision = v.decide(go=go)
    return {
        "largest_N": big,
        "frac_recalled": fr,
        "protect_immediate_acq": protect_acq, "replay_immediate_acq": replay_acq,
        "ewc_new_learning_tradeoff_protect_minus_replay": float(protect_acq - replay_acq),
        "protect_mean_retained_acc": protect_meanret,
        "rise_protect_minus_replay": float(fr["protect"] - fr["replay"]),
        "structure_load_bearing_protect_minus_scramble": float(fr["protect"] - fr["scramble"]),
        "orthoprot_minus_protect": float(fr["orthoprot"] - fr["protect"]),
        **decision,
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop WEIGHT PROTECTION: fixed orthogonal readout targets + "
                                             "synaptic-importance-gated plasticity (EWC/SI/Phase-1.4 gate-freeze) on "
                                             "the brain's own readout synapses to beat sequential forgetting past the "
                                             "sleep-replay ~0.55 plateau.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40, help="per-fact WAKE teaching epochs")
    ap.add_argument("--replay-epochs", type=int, default=24, help="offline SLEEP consolidation epochs over the store")
    ap.add_argument("--replay-per-fact", type=int, default=16, help="self-generated replay draws per stored engram")
    ap.add_argument("--replay-noise", type=float, default=0.10, help="brain-owned variability on the replayed engram")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--readout-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--lam", type=float, default=8.0, help="EWC/SI importance-gate strength (0 disables protection)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.readout_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact, a.replay_noise,
                 a.lam)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_weight_protect", "seed": a.seed, "backend": os.environ.get("SIM_BACKEND"),
               "single_seed_smoke": True, "elapsed_seconds": round(time.time() - t0, 1),
               "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    fr = verdict["frac_recalled"]
    print("\n" + "=" * 100, flush=True)
    print(f"[weight-protect] seed {a.seed} @ N={verdict['largest_N']} (chance {result['chance']:.2f}): "
          f"replay {fr['replay']:.2f} | PROTECT {fr['protect']:.2f} | scramble {fr['scramble']:.2f} | "
          f"orthoprot {fr['orthoprot']:.2f}", flush=True)
    print(f"[weight-protect] rise(protect-replay) {verdict['rise_protect_minus_replay']:+.2f} | "
          f"structure-load-bearing(protect-scramble) {verdict['structure_load_bearing_protect_minus_scramble']:+.2f} | "
          f"orthoprot-minus-protect {verdict['orthoprot_minus_protect']:+.2f}", flush=True)
    print(f"[weight-protect] protect immediate-acq {verdict['protect_immediate_acq']:.3f} | "
          f"EWC new-learn tradeoff(protect-replay) {verdict['ewc_new_learning_tradeoff_protect_minus_replay']:+.3f} | "
          f"protect mean-retained {verdict['protect_mean_retained_acc']:.3f} | VERDICT {verdict['status']}", flush=True)
    for arm in ("replay", "protect", "scramble", "orthoprot"):
        rcm = result["arms"][arm]["retention_curve"]
        line = " ".join(f"N={k}:{rcm[k]['n_recalled']}/{k}({rcm[k]['frac_recalled']:.2f})" for k in sorted(rcm, key=int))
        print(f"    {arm:9s}: {line}", flush=True)
    print(f"[weight-protect] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
