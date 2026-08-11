"""TEACHER-LOOP METAPLASTIC ACQUISITION DE-RISK (2026-08-11): does per-synapse METAPLASTICITY defeat the
ACQUISITION-AT-SCALE forgetting that caps continual fact-learning at frac_recalled ~ 1/N?

THE ISOLATED BOTTLENECK (named by the prior arc, NOT re-derived here). Teaching N distinct facts SEQUENTIALLY
into ONE brain whose facts are acquired by the transport-free e-prop WEIGHT CHANGE (OnBridgeEpropNet, the a1-GO
rule) forgets to frac_recalled ~ 1/N as N grows: later facts overwrite earlier ones in the SHARED readout/hidden
weights (interleaved-GR finding 2026-08-11 isolated it -- teaching the 50th fact alone into a crowded shared
readout barely registers, newest-fact immediate-acq -> chance at N=50). The de-clamp finding
(2026-08-11-sleep-replay-bdsp-declamp-...) retired the ~0.55 "cap" as a bdsp_wmax=6 CLAMP artifact; with the
clamp WIDENED the residual forgetting is exactly this acquisition-at-scale overwrite. So this de-risk holds the
de-clamp CONSTANT across all arms (it is NOT the lever) and tests the NAMED next mechanism: METAPLASTICITY.

THE MECHANISM (biologically cited; a per-synapse consolidation state that gates plasticity).
  Fusi, Drew & Abbott 2005 (Neuron, "Cascade models of synaptically stored memories"): each synapse carries a
  cascade of metaplastic states of decreasing plasticity; repeated potentiation moves a synapse to a DEEPER,
  MORE STABLE, LESS-PLASTIC state, so already-stored memories resist being overwritten -> power-law (not
  exponential) forgetting and greatly extended capacity.
  Benna & Fusi 2016 (Nat Neurosci, "Computational principles of synaptic memory consolidation"): the synapse's
  visible efficacy is coupled to slow HIDDEN variables that integrate its history and feed back to STABILIZE the
  efficacy; multi-timescale consolidation extends memory lifetime by orders of magnitude over a single-variable
  synapse.
  IMPLEMENTED (a continuous-state cascade): each trained FF synapse ij carries a hidden consolidation variable
  c_ij >= 0 (init 0). Its EFFECTIVE plasticity is gated -- lr_eff_ij = lr / (1 + g * c_ij) -- so a consolidated
  synapse resists change (Fusi's deeper state / Benna-Fusi's stabilized efficacy). After teaching each fact, the
  synapses POTENTIATED for that fact deepen: c_ij += rate * (|Δw_ij during the fact| / max|Δw|), i.e. the fact's
  most-used synapses accrue the most consolidation and are protected from the NEXT fact. The e-prop weight w
  (drives the spiking forward) is the visible efficacy; c is the slow hidden variable. This is a per-SYNAPSE
  local state (a synaptic-tagging/consolidation variable, CaMKII-autophosphorylation-like), NOT host cognition --
  brain-based-only: the synapse gates its own learning rate from its own local potentiation history.

FOUR ARMS, one world / seed / schedule / de-clamp (the ONLY difference is the metaplastic gate):
  * vanilla       = e-prop, metaplasticity OFF -> the CONTROL. Expect frac_recalled ~ 1/N (the collapse).
  * metaplastic   = e-prop + the per-synapse consolidation gate -> the TREATMENT. GO = frac_recalled stays HIGH
                    (flat), not 1/N, as N grows, while vanilla collapses.
  * meta_lesion   = metaplastic machinery PRESENT but its per-synapse STATE frozen at 0 (consolidate step
                    skipped) -> gate == 1 always -> MUST collapse back to ~vanilla. THE EARNED TOOTH: the
                    per-synapse consolidation STATE is load-bearing, not the code path.
  * meta_permute  = consolidate normally, but at gate time PERMUTE c across synapses so protection lands on the
                    WRONG synapses -> attribution anti-cheat: if protecting random synapses does not help, the
                    SPECIFIC per-synapse targeting (which synapse carries which memory) is the drive.

ANTI-CHEATS (executed via tools.lab, not asserted in prose):
  (a) load-bearing lesion: meta_lesion frac_recalled ~= vanilla, and metaplastic >> both.
  (b) attributable_to(metaplastic vs meta_permute) and (metaplastic vs meta_lesion): the fraction of the effect
      NOT present in the wrong-synapse / no-state controls -> the specific consolidation is the drive.
  (c) no-confab moat: N/A here by construction -- the fact readout is a FORCED-CHOICE k-way argmax (every cue
      gets a class; there is no abstain/novelty gate to false-accept). Stated, not silently skipped.
  (d) the lever MOVED (one flag != one variable): the gate summary confirms vanilla's effective-lr scale == 1.0
      everywhere (no protection) while metaplastic's is < 1.0 (protection active). lever() prints both.

IMMEDIATE-ACQUISITION (the plasticity side of stability-plasticity). Over-protecting old synapses can block NEW
learning. We report mean immediate-acq for every arm; the treatment must keep acquiring (floor 0.6) while
retaining. A treatment that flattens frac_recalled ONLY by refusing to learn new facts is a NEGATIVE, and the
immediate-acq column exposes it.

DISCIPLINE: reuse-by-import of ALL substantive machinery (world/teach/held-out-acc/corrective-batch from
_teacher_loop_scaling_derisk; ReferentEnv from _teacher_loop_corrective_acquire_derisk; OnBridgeEpropNet from the
a1-GO port). The metaplastic state is a runner-side per-synapse numpy array + an override of _apply_grads on a
SUBCLASS -- NO sim/ edit. cfg.seed via the seed= arg the net passes to CoreSimConfig.seed. Additive; the net is
byte-identical to OnBridgeEpropNet when meta_enabled=False (the vanilla arm proves it).

RUN (net is ~48 neurons; per the de-clamp finding numpy avoids cupy launch overhead at this size -- MEASURED in
the smoke, reported for the coordinator to choose):
  single-seed SMOKE:
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python -m research.runners._teacher_loop_metaplastic_acquisition_derisk --seed 42 \
        --n-max 16 --milestones 4 8 16 --epochs 30 --settle-steps 20 --test-n 24 --n-draws 32 \
        --out research/findings/raw/metaplastic_acq_s42.json
  6-SEED sweep command is in the finding / returned to the coordinator (one seed per process, N-sweep in one run
  via --n-max 100 --milestones 16 32 50 100).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # ~48-neuron net; numpy avoids cupy launch overhead at this size
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
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
# reuse-by-import: world / teach / held-out acc / corrective batch / readout-norm-over-world / action width.
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "metaplastic_acq.json"
ARMS = ("vanilla", "metaplastic", "meta_lesion", "meta_permute")


# ============================================================================================================
# The metaplastic e-prop net: OnBridgeEpropNet (transport-free e-prop, the sole learner) + a per-synapse
# Fusi/Benna-Fusi consolidation state that gates each synapse's effective learning rate. NO sim/ edit.
# ============================================================================================================
class MetaplasticEpropNet(OnBridgeEpropNet):
    def __init__(self, *args, meta_enabled=False, meta_gain=8.0, meta_consol_rate=1.0,
                 meta_lesion_freeze=False, meta_permute=False, meta_permute_seed=0, **kw):
        super().__init__(*args, **kw)
        L = len(self.sizes) - 1
        # per-synapse consolidation state c_ij >= 0 for every trained FF pathway (aligned to _data_idx_flat[li]).
        self.meta_c = [np.zeros(int(self._data_idx_flat[li].shape[0]), dtype=np.float64) for li in range(L)]
        self._fact_dw_abs = [np.zeros_like(c) for c in self.meta_c]   # per-fact |Δw| accumulator (drives consolidation)
        self.meta_enabled = bool(meta_enabled)
        self.meta_gain = float(meta_gain)
        self.meta_consol_rate = float(meta_consol_rate)
        self.meta_lesion_freeze = bool(meta_lesion_freeze)            # keep the machinery, freeze the STATE at 0 (lesion)
        self.meta_permute = bool(meta_permute)                       # protect the WRONG synapses (attribution control)
        prng = np.random.default_rng(int(meta_permute_seed) + 20260811)
        self._meta_perm = [prng.permutation(int(c.shape[0])) for c in self.meta_c]

    # ---- override the FF weight-write to insert the per-synapse metaplastic gate + accumulate |Δw| per fact ----
    def _apply_grads(self, grads, bsz):
        xp = self._xp
        data = self.br.cp_connections.data
        L = len(grads)
        for li in range(L):
            if self.train_layers is not None and li not in self.train_layers:
                continue                                    # frozen pathway (unused here; kept for parity)
            idx = self._data_idx_flat[li]
            lr_li = self.eprop_lr * (1.0 if li == L - 1 else self.hidden_lr_scale)
            dw = (lr_li * (grads[li] / max(1, bsz))).astype(np.float64).ravel()   # float64 for the gate
            if self.meta_enabled:
                # METAPLASTIC GATE (Fusi/Benna-Fusi): consolidated synapses resist change. lr_eff = lr/(1+g*c).
                c = self.meta_c[li]
                if self.meta_permute:
                    c = c[self._meta_perm[li]]              # decouple protection from the memory-carrying synapse
                dw = dw / (1.0 + self.meta_gain * c)
            self._fact_dw_abs[li] += np.abs(dw)             # the ACTUAL applied |Δw| this batch (for consolidation)
            dw32 = dw.astype(np.float32)
            cur = data[idx]
            new = xp.clip(cur - xp.asarray(dw32), -self.w_clip, self.w_clip)   # GD: w -= lr_eff * grad
            if self.br.cp_synapse_plastic_mask is not None:
                pm = self.br.cp_synapse_plastic_mask[idx]
                new = xp.where(pm, new, cur)
            if self.br.cp_plasticity_rate_gain is not None:
                gain = self.br.cp_plasticity_rate_gain[idx]
                new = cur + (new - cur) * gain
            data[idx] = new

    def consolidate_fact(self):
        """Cascade step (Fusi 2005 / Benna-Fusi 2016): synapses potentiated for the just-taught fact move to a
        deeper, more stable, LESS-plastic metaplastic state. Called ONCE after each fact's teaching. In the
        frozen-state LESION the deepening is skipped so meta_c stays 0 -> the machinery runs but carries no
        information -> the treatment MUST collapse (the per-synapse STATE is the load-bearing thing)."""
        for li in range(len(self.meta_c)):
            u = self._fact_dw_abs[li]
            if self.meta_enabled and not self.meta_lesion_freeze:
                m = float(u.max())
                if m > 0:
                    self.meta_c[li] += self.meta_consol_rate * (u / m)    # relative usage in [0,1]; top-changed ~1
            self._fact_dw_abs[li][...] = 0.0                              # reset the per-fact accumulator

    def gate_summary(self):
        """The effective-lr scale currently applied per pathway. mean_gate==1.0 => NO protection (meta off / c=0);
        <1.0 => protection active. This is the (d) 'the lever moved' read-out."""
        out = {}
        for li in range(len(self.meta_c)):
            c = self.meta_c[li]
            if self.meta_enabled:
                cc = c[self._meta_perm[li]] if self.meta_permute else c
                gate = 1.0 / (1.0 + self.meta_gain * cc)
            else:
                gate = np.ones_like(c)
            out[str(li)] = {"mean_gate": float(gate.mean()), "min_gate": float(gate.min()),
                            "frac_protected": float(np.mean(gate < 0.99)),
                            "mean_c": float(c.mean()), "max_c": float(c.max())}
        return out

    def global_min_gate(self):
        g = 1.0
        for li in range(len(self.meta_c)):
            g = min(g, self.gate_summary()[str(li)]["min_gate"])
        return g


def _mk_meta_net(n_in, k, seed, hidden, settle, eprop_lr, w_clip, declamp_wmax, meta_kwargs):
    """Same a1-GO OnBridgeEpropNet build the scaling/declamp siblings use, held de-clamped, wrapped as
    MetaplasticEpropNet with the per-synapse consolidation gate configured by meta_kwargs."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    if declamp_wmax is not None:
        hp["bdsp_wmax"] = float(declamp_wmax)                # de-clamp held CONSTANT across arms (NOT the lever)
    return MetaplasticEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                               eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                               logit_source="leaky_readout", w_clip=w_clip, hp=hp, meta_permute_seed=seed,
                               **meta_kwargs)


def _run_arm(arm, meta_kwargs, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip,
             n_draws, milestones, test_n, chance, declamp_wmax):
    """Teach the referents SEQUENTIALLY into ONE brain; consolidate after each fact; record retention at each
    milestone. The ONLY cross-arm difference is meta_kwargs (the gate)."""
    net = _mk_meta_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, declamp_wmax, meta_kwargs)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    acquire_acc, retention, gate_curve = [], {}, {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)             # WAKE: teacher draws from the world (legitimate)
        _teach_fact(net, X, y, epochs, batch, teach_rng)          # brain acquires the fact by e-prop weight change
        net.consolidate_fact()                                    # metaplastic deepening AFTER the fact (lesion skips)
        acq = _fact_acc(net, env, r, i, n=test_n)                 # immediate held-out acquisition (plasticity side)
        acquire_acc.append(acq)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {"frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                                 "one_over_N": float(1.0 / N),
                                 "mean_retained_acc": float(np.mean(accs)),
                                 "oldest_fact_acc": float(accs[0]), "newest_fact_acc": float(accs[-1]),
                                 "per_fact_acc": [float(a) for a in accs]}
            gate_curve[str(N)] = net.gate_summary()
    return {"arm": arm, "meta_kwargs": {k: (v if not isinstance(v, bool) else bool(v)) for k, v in meta_kwargs.items()},
            "acquire_acc_immediate": [float(a) for a in acquire_acc],
            "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
            "final_gate": net.gate_summary(), "gate_curve": gate_curve, "retention_curve": retention}


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise, test_n,
        declamp_wmax, meta_gain, meta_consol_rate):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    G, R = float(meta_gain), float(meta_consol_rate)
    specs = {
        "vanilla":      dict(meta_enabled=False),
        "metaplastic":  dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R),
        "meta_lesion":  dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R, meta_lesion_freeze=True),
        "meta_permute": dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R, meta_permute=True),
    }
    arms = {}
    for name in ARMS:
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)               # identical teaching percepts across arms (like-for-like)
        arms[name] = _run_arm(name, specs[name], seed, referents, env, K, n_in, hidden, settle, epochs, batch,
                              eprop_lr, w_clip, n_draws, milestones, test_n, chance, declamp_wmax)
        arms[name]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[name]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        gmin = min((arms[name]["final_gate"][li]["min_gate"] for li in arms[name]["final_gate"]), default=1.0)
        print(f"[arm {name:13s}] {arms[name]['wall_seconds']:6.0f}s | immediate-acq "
              f"{arms[name]['mean_acquire_acc_immediate']:.3f} | min-gate {gmin:.3f} | "
              f"frac-recalled@N={big}: {fr:.3f} (1/N={1.0/big:.3f})", flush=True)
    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "declamp_wmax": declamp_wmax, "meta_gain": G, "meta_consol_rate": R,
                       "backend": os.environ.get("SIM_BACKEND")},
            "arms": arms}


def _verdict(result):
    from tools.lab import lever, attributable_to, assert_backend  # noqa: F401
    from tools.verdict import Verdict
    rc = {a: result["arms"][a]["retention_curve"] for a in result["arms"]}
    big = max((int(k) for k in rc["metaplastic"]), default=None)
    key = str(big)
    f = {a: rc[a][key]["frac_recalled"] for a in rc}
    acq = {a: result["arms"][a]["mean_acquire_acc_immediate"] for a in result["arms"]}
    chance = result["chance"]
    one_over_N = 1.0 / big

    # (d) THE LEVER MOVED: vanilla gate == 1.0 everywhere (no protection); metaplastic gate < 1.0 (protection).
    van_min = min(result["arms"]["vanilla"]["final_gate"][li]["min_gate"] for li in result["arms"]["vanilla"]["final_gate"])
    met_min = min(result["arms"]["metaplastic"]["final_gate"][li]["min_gate"] for li in result["arms"]["metaplastic"]["final_gate"])
    lever("metaplastic gate (min effective-lr scale)", round(van_min, 4), round(met_min, 4))

    # (a)/(b) attribution: the fraction of the retention effect NOT present in the no-state / wrong-synapse controls.
    attributable_to("metaplastic vs vanilla (frac_recalled@Nmax)", f["metaplastic"], f["vanilla"])
    attributable_to("metaplastic vs meta_lesion (state frozen at 0)", f["metaplastic"], f["meta_lesion"])
    attributable_to("metaplastic vs meta_permute (wrong-synapse protection)", f["metaplastic"], f["meta_permute"])

    v = Verdict("teacher-loop metaplastic acquisition", chance=chance)
    v.reaches("(1) metaplastic beats vanilla (acquisition-at-scale)", before=f["vanilla"], after=f["metaplastic"])
    v.reaches("(2) lesion (state frozen) collapses to ~vanilla", before=f["metaplastic"], after=f["meta_lesion"])
    v.reaches("(3) permute (wrong synapses) collapses to ~vanilla", before=f["metaplastic"], after=f["meta_permute"])
    v.floor("(4) metaplastic keeps acquiring new facts (immediate-acq)", acq["metaplastic"], floor=0.6)
    # GO: treatment clearly beats vanilla AND both controls; lesion+permute stay near vanilla (load-bearing state);
    #     treatment still acquires. Margins are 6-seed-confirmable; single-seed prints SMOKE, never a 6-seed GO.
    go = (f["metaplastic"] > f["vanilla"] + 0.15
          and f["metaplastic"] > f["meta_lesion"] + 0.15
          and f["metaplastic"] > f["meta_permute"] + 0.15
          and f["meta_lesion"] <= f["vanilla"] + 0.12
          and acq["metaplastic"] >= 0.6)
    decision = v.decide(go=go)
    return {"largest_N": big, "one_over_N": one_over_N, "frac_recalled": f, "immediate_acq": acq,
            "vanilla_min_gate": float(van_min), "metaplastic_min_gate": float(met_min),
            "meta_beats_vanilla": float(f["metaplastic"] - f["vanilla"]),
            "lesion_collapse_margin": float(f["metaplastic"] - f["meta_lesion"]),
            "permute_collapse_margin": float(f["metaplastic"] - f["meta_permute"]),
            "lesion_matches_vanilla": bool(abs(f["meta_lesion"] - f["vanilla"]) <= 0.12), **decision}


def _aggregate(paths):
    """6-seed roll-up. GO = every seed: metaplastic > vanilla+0.15 AND > lesion+0.15 AND > permute+0.15 AND
    lesion ~= vanilla (state load-bearing) AND metaplastic immediate-acq >= 0.6."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        vd = d["verdict"]
        f = vd["frac_recalled"]; acq = vd["immediate_acq"]
        seed_go = bool(f["metaplastic"] > f["vanilla"] + 0.15 and f["metaplastic"] > f["meta_lesion"] + 0.15
                       and f["metaplastic"] > f["meta_permute"] + 0.15 and f["meta_lesion"] <= f["vanilla"] + 0.12
                       and acq["metaplastic"] >= 0.6)
        rows.append({"seed": d["seed"], "N": vd["largest_N"], **{a: f[a] for a in ARMS},
                     "acq_meta": acq["metaplastic"], "seed_go": seed_go})
    means = {a: float(np.mean([r[a] for r in rows])) for a in ARMS}
    n_go = sum(r["seed_go"] for r in rows)
    go = n_go == len(rows) and len(rows) >= 6
    print("\n" + "=" * 104)
    print(f"[AGG] {len(rows)} seeds | GO needs metaplastic>vanilla+.15 & >lesion+.15 & >permute+.15 & lesion~=vanilla, all seeds")
    print(f"{'seed':>5} {'N':>4} " + " ".join(f"{a[:11]:>12}" for a in ARMS) + f" {'acqMeta':>8} {'GO':>4}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>5} {r['N']:>4} " + " ".join(f"{r[a]:>12.3f}" for a in ARMS) + f" {r['acq_meta']:>8.3f} {str(r['seed_go']):>4}")
    print(f"{'mean':>5} {'':>4} " + " ".join(f"{means[a]:>12.3f}" for a in ARMS))
    print(f"[AGG] metaplastic {means['metaplastic']:.3f} vs vanilla {means['vanilla']:.3f} vs "
          f"lesion {means['meta_lesion']:.3f} vs permute {means['meta_permute']:.3f} | seeds GO {n_go}/{len(rows)} "
          f"| VERDICT {'GO' if go else 'NO-GO'}")
    print("=" * 104)
    return 0 if go else 1


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop metaplastic acquisition: does a per-synapse "
                                             "Fusi/Benna-Fusi consolidation gate defeat the frac_recalled~1/N "
                                             "acquisition-at-scale forgetting?")
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-seed JSONs -> 6-seed GO roll-up")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=16)
    ap.add_argument("--milestones", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=24)
    ap.add_argument("--declamp-wmax", type=float, default=1e9,
                    help="bdsp_w_max for ALL arms (de-clamp held constant; NOT the lever). None keeps the +-6 default.")
    ap.add_argument("--meta-gain", type=float, default=8.0, help="metaplastic gain g in lr_eff=lr/(1+g*c)")
    ap.add_argument("--meta-consol-rate", type=float, default=1.0, help="per-fact consolidation increment rate")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.aggregate:
        return _aggregate(a.aggregate)
    if a.declamp_wmax is not None and a.declamp_wmax < 0:
        a.declamp_wmax = None
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.declamp_wmax, a.meta_gain, a.meta_consol_rate)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_metaplastic_acquisition", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    f = verdict["frac_recalled"]; acq = verdict["immediate_acq"]
    print("\n" + "=" * 104, flush=True)
    print(f"[metaplastic-acq] seed {a.seed} @ N={verdict['largest_N']} (1/N={verdict['one_over_N']:.3f}, "
          f"chance {result['chance']:.3f}):", flush=True)
    for arm in ARMS:
        print(f"    {arm:13s}: frac-recalled {f[arm]:.3f} | immediate-acq {acq[arm]:.3f}", flush=True)
    print(f"[metaplastic-acq] meta-beats-vanilla {verdict['meta_beats_vanilla']:+.3f} | lesion-collapse "
          f"{verdict['lesion_collapse_margin']:+.3f} | permute-collapse {verdict['permute_collapse_margin']:+.3f} "
          f"| lesion~=vanilla {verdict['lesion_matches_vanilla']} | VERDICT {verdict['status']}", flush=True)
    print(f"[metaplastic-acq] wrote {a.out}\n" + "=" * 104, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
