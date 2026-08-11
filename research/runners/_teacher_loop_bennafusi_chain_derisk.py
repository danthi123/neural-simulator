"""TEACHER-LOOP BENNA-FUSI CONSOLIDATION-CHAIN DE-RISK (2026-08-11): does a TRUE multi-timescale
Benna-Fusi consolidation CHAIN protect the OLDEST facts that the SINGLE-variable metaplastic gate could not?

WHY THIS EXISTS (the named next mechanism, not re-derived here). The single-variable metaplastic de-risk
(`_teacher_loop_metaplastic_acquisition_derisk.py`, finding
`2026-08-11-metaplastic-acquisition-continual-learning-6seed-NOGO-mechanism-real-but-subthreshold.md`)
showed a per-synapse consolidation variable `c` gating lr_eff = lr/(1+g*c) moves the acquisition-at-scale
forgetting the RIGHT way (+0.137 mean frac_recalled, load-bearing freeze-lesion, correctly attributed, no
acquisition cost) but is SUB-THRESHOLD at 6 seeds: the single fast variable lifts the MIDDLE of the retention
curve yet the very-OLDEST fact (fact 0) is still overwritten at N=100. The finding NAMED the surpass: a TRUE
multi-timescale Benna-Fusi consolidation CHAIN whose SLOW variables reach the oldest memories the single fast
variable cannot. This de-risk builds + smoke-tests that chain.

THE MECHANISM (biologically cited).
  Benna, M. K. & Fusi, S. (2016). "Computational principles of synaptic memory consolidation."
  Nature Neuroscience 19(12):1697-1706. Each synapse's VISIBLE efficacy is coupled to a cascade of HIDDEN
  variables u_1..u_K at geometrically-increasing timescales; a plasticity event injects into the fastest
  variable, which then diffuses down the chain (a bidirectional, conservative cascade). The slow variables
  integrate the synapse's history over ever-longer windows and feed back to STABILIZE the efficacy, extending
  memory lifetime by orders of magnitude over a single-variable synapse (power-law, not exponential, forgetting).
  Antecedent cascade: Fusi, Drew & Abbott 2005 (Neuron), discrete metaplastic states of decreasing plasticity.

IMPLEMENTED (a continuous-state bidirectional cascade, LR-gate form — per the named-surpass spec).
  Each trained FF synapse ij carries a chain u_ij[0..K-1] (>= 0, init 0; u[0] = fastest/most-plastic tracker,
  u[K-1] = slowest). After teaching each fact, the fact's relative usage (|Δw|/max, in [0,1] — identical signal
  to the single-var arm) is INJECTED into u[0], then one CASCADE tick runs: for each bond k a conservative
  exchange transfer = f_k*(u[k]-u[k+1]) with geometric flow rates f_k = 0.5*ratio^-k, so bond k equilibrates on
  a timescale ~ ratio^k (geometrically increasing; reflecting boundaries => total mass conserved). The effective
  plasticity is gated by the CHAIN STATE: lr_eff_ij = lr / (1 + g * C(u_ij)) with C(u) = Σ_k w_k u[k], where the
  gate weights w_k are DEEP-EMPHASIZED (raw = base + slope*k) and NORMALIZED TO MEAN 1. Normalizing to mean 1 is
  the whole experimental control: the chain's total protection BUDGET is matched to the single-var arm's (same
  injection, conserved mass, same meta_gain g); only the AGE-DISTRIBUTION of protection differs. A synapse whose
  usage still sits in u[0] (a JUST-taught fact) gets w_0<1 => LESS protection => stays plastic (immediate
  acquisition preserved); a synapse whose usage has propagated to the deep variables (an OLD fact) gets w_{K-1}>1
  => MORE protection => resists overwrite. So the chain HARDENS with age exactly where the single flat
  accumulator could not. This is brain-based-only: a per-synapse local state gating its own learning rate from
  its own local potentiation history (synaptic tagging / consolidation), NOT host cognition. The e-prop weight w
  (drives the spiking forward) is the visible efficacy; the chain is the slow hidden cascade. NO sim/ edit.

FIVE ARMS, one world / seed / schedule / de-clamp (the ONLY difference is the consolidation gate):
  * vanilla       = e-prop, no gate -> the collapse control (frac_recalled ~ 1/N).
  * single_var    = the PRIOR single-c metaplastic gate (the arm to BEAT — more timescales must help, not
                    just having any metaplasticity). Same g, same injection, same de-clamp.
  * chain         = the Benna-Fusi multi-timescale cascade -> the TREATMENT. GO = beats vanilla by the strict
                    +0.15 margin AND beats single_var AND protects the OLDEST fact (fact 0 @ N_max) where the
                    single var failed, while still acquiring new facts.
  * chain_lesion  = chain machinery PRESENT but its per-synapse STATE frozen at 0 (inject+diffuse skipped) ->
                    gate == 1 always -> MUST collapse to ~vanilla. The load-bearing tooth: the chain STATE, not
                    the code path, does the work.
  * chain_permute = cascade runs normally, but at gate time PERMUTE C across synapses so protection lands on the
                    WRONG synapses -> attribution: if protecting random synapses does not help, the SPECIFIC
                    per-synapse targeting is the drive.

ANTI-CHEATS (executed via tools.lab + tools.verdict.Verdict, not asserted in prose):
  (a) load-bearing: chain_lesion (state frozen) frac_recalled ~= vanilla, and chain >> both.
  (b) attributable: attributable_to(chain vs chain_permute) and (chain vs chain_lesion) — the fraction of the
      effect NOT present in the wrong-synapse / no-state controls.
  (c) no acquisition cost: chain mean immediate-acq >= 0.6 (the plasticity side of stability-plasticity).
  (d) the lever MOVED + de-clamp held CONSTANT: gate min < 1 for chain, == 1 for vanilla; bdsp_wmax identical
      across ALL arms (reported) so the de-clamp is not the lever.
  (e) chain BEATS single_var (not just vanilla): the whole point of the extra timescales. Reported as a margin
      and gated; and per-fact-AGE retention (oldest / middle / newest) makes the oldest-fact claim explicit.

DISCIPLINE: reuse-by-import of ALL substantive machinery — MetaplasticEpropNet (which itself wraps the a1-GO
OnBridgeEpropNet) + the teacher-loop world/teach/held-out-acc/corrective-batch. The chain is an ADDITIVE subclass
(BennaFusiEpropNet); with chain_enabled=False it is the single_var arm, with meta_enabled=False it is vanilla,
byte-for-byte. cfg.seed via the seed= arg the net passes to CoreSimConfig.seed. NO sim/ edit. SIM_BACKEND=numpy
(this ~48-neuron net is launch-bound; numpy avoids cupy launch overhead — MEASURED, reported).

RUN:
  single-seed SMOKE (chain beats single_var + lesion bites, small N):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python -m research.runners._teacher_loop_bennafusi_chain_derisk --seed 42 \
        --n-max 32 --milestones 8 16 32 --epochs 30 --settle-steps 20 --test-n 24 --n-draws 32 \
        --out research/findings/raw/bennafusi_chain_s42.json
  6-SEED sweep command is returned to the coordinator (one seed per process; N-sweep in one run via
  --n-max 100 --milestones 16 32 50 100).
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
# reuse-by-import: the single-var metaplastic net (which wraps OnBridgeEpropNet) + the teacher-loop machinery.
from research.runners._teacher_loop_metaplastic_acquisition_derisk import MetaplasticEpropNet  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "bennafusi_chain.json"
ARMS = ("vanilla", "single_var", "chain", "chain_lesion", "chain_permute")


# ============================================================================================================
# The Benna-Fusi chain net: MetaplasticEpropNet (single-var gate + a1-GO e-prop) EXTENDED with a per-synapse
# multi-timescale consolidation CASCADE that gates each synapse's effective learning rate. NO sim/ edit.
# chain_enabled=False -> the single_var arm; meta_enabled=False -> vanilla. Additive, default-off.
# ============================================================================================================
class BennaFusiEpropNet(MetaplasticEpropNet):
    def __init__(self, *args, chain_enabled=False, chain_depth=6, chain_ratio=2.0, gate_base=1.0,
                 gate_slope=1.0, chain_consol_rate=1.0, chain_lesion_freeze=False, chain_permute=False, **kw):
        super().__init__(*args, **kw)
        self.chain_enabled = bool(chain_enabled)
        self.chain_depth = int(chain_depth)
        self.chain_ratio = float(chain_ratio)
        self.chain_consol_rate = float(chain_consol_rate)
        self.chain_lesion_freeze = bool(chain_lesion_freeze)     # keep the machinery, freeze the STATE at 0 (lesion)
        self.chain_permute = bool(chain_permute)                 # protect the WRONG synapses (attribution control)
        L = len(self.sizes) - 1
        K = self.chain_depth
        # per-synapse cascade u[:, 0..K-1] (u[0] fastest .. u[K-1] slowest), aligned to _data_idx_flat[li].
        self.bf_u = [np.zeros((int(self._data_idx_flat[li].shape[0]), K), dtype=np.float64) for li in range(L)]
        # geometric bond flow rates f_k = 0.5*ratio^-k  => bond k equilibrates on timescale ~ ratio^k (geometric).
        self._flow = np.array([0.5 * (self.chain_ratio ** (-k)) for k in range(max(1, K - 1))], dtype=np.float64)
        # DEEP-EMPHASIZED gate weights, NORMALIZED TO MEAN 1: matches the single-var arm's total protection budget
        # (same meta_gain g), so ONLY the age-distribution of protection differs -> isolates the multi-timescale bet.
        raw = gate_base + gate_slope * np.arange(K, dtype=np.float64)
        self._gate_w = (raw / raw.mean()).astype(np.float64)

    def _consol_divisor(self, li):
        """(1 + g*C) per synapse (the lr_eff denominator), or None for the ungated (vanilla) arm.
        chain arm: C = Σ_k w_k u[k] (deep-emphasized, mean-1 weights). single_var arm: C = c (parent state)."""
        if not self.meta_enabled:
            return None
        if self.chain_enabled:
            C = self.bf_u[li].dot(self._gate_w)              # (n_syn,)
            if self.chain_permute:
                C = C[self._meta_perm[li]]                    # protection lands on the WRONG synapses
            return 1.0 + self.meta_gain * C
        c = self.meta_c[li]                                  # single-var: exactly the parent metaplastic gate
        if self.meta_permute:
            c = c[self._meta_perm[li]]
        return 1.0 + self.meta_gain * c

    def _apply_grads(self, grads, bsz):
        """FF weight-write with the per-synapse consolidation gate inserted + |Δw| accumulated per fact.
        Mirrors MetaplasticEpropNet._apply_grads but routes the gate through _consol_divisor (chain or single)."""
        xp = self._xp
        data = self.br.cp_connections.data
        L = len(grads)
        for li in range(L):
            if self.train_layers is not None and li not in self.train_layers:
                continue
            idx = self._data_idx_flat[li]
            lr_li = self.eprop_lr * (1.0 if li == L - 1 else self.hidden_lr_scale)
            dw = (lr_li * (grads[li] / max(1, bsz))).astype(np.float64).ravel()   # float64 for the gate
            div = self._consol_divisor(li)
            if div is not None:
                dw = dw / div                                # lr_eff = lr/(1+g*C): consolidated synapses resist change
            self._fact_dw_abs[li] += np.abs(dw)              # the ACTUAL applied |Δw| this batch (for consolidation)
            dw32 = dw.astype(np.float32)
            cur = data[idx]
            new = xp.clip(cur - xp.asarray(dw32), -self.w_clip, self.w_clip)      # GD: w -= lr_eff * grad
            if self.br.cp_synapse_plastic_mask is not None:
                pm = self.br.cp_synapse_plastic_mask[idx]
                new = xp.where(pm, new, cur)
            if self.br.cp_plasticity_rate_gain is not None:
                gain = self.br.cp_plasticity_rate_gain[idx]
                new = cur + (new - cur) * gain
            data[idx] = new

    def _diffuse(self, li):
        """One Benna-Fusi cascade tick: conservative bidirectional exchange along the chain (reflecting ends)."""
        u = self.bf_u[li]
        for k in range(self.chain_depth - 1):
            t = self._flow[k] * (u[:, k] - u[:, k + 1])
            u[:, k]     -= t
            u[:, k + 1] += t

    def consolidate_fact(self):
        """Called ONCE after each fact. chain arm: inject the fact's relative usage into u[0], then one cascade
        tick (usage migrates toward the slow variables over subsequent facts). Frozen-state LESION skips it so the
        chain stays 0 -> gate==1 -> MUST collapse. Non-chain arms defer to the parent (single-var / vanilla)."""
        if not self.chain_enabled:
            return super().consolidate_fact()
        for li in range(len(self.bf_u)):
            u_signal = self._fact_dw_abs[li]
            if self.meta_enabled and not self.chain_lesion_freeze:
                m = float(u_signal.max())
                if m > 0:
                    self.bf_u[li][:, 0] += self.chain_consol_rate * (u_signal / m)   # relative usage in [0,1] -> fastest
                    self._diffuse(li)
            self._fact_dw_abs[li][...] = 0.0                                          # reset the per-fact accumulator

    def gate_summary(self):
        """Effective-lr scale currently applied per pathway + chain occupancy per level (fast..slow). mean_gate==1
        => NO protection; <1 => protection active. level_mean_occupancy shows mass migrating to the slow variables."""
        if not self.chain_enabled:
            return super().gate_summary()
        out = {}
        for li in range(len(self.bf_u)):
            u = self.bf_u[li]
            C = u.dot(self._gate_w)
            if self.chain_permute:
                C = C[self._meta_perm[li]]
            gate = 1.0 / (1.0 + self.meta_gain * C) if self.meta_enabled else np.ones(u.shape[0])
            out[str(li)] = {"mean_gate": float(gate.mean()), "min_gate": float(gate.min()),
                            "frac_protected": float(np.mean(gate < 0.99)),
                            "mean_C": float(C.mean()), "max_C": float(C.max()),
                            "level_mean_occupancy": [float(u[:, k].mean()) for k in range(self.chain_depth)]}
        return out


def _mk_bf_net(n_in, k, seed, hidden, settle, eprop_lr, w_clip, declamp_wmax, kwargs):
    """Same a1-GO OnBridgeEpropNet build the metaplastic/scaling siblings use, held de-clamped, wrapped as
    BennaFusiEpropNet with the cascade configured by kwargs (identical hp across arms -> de-clamp not the lever)."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    if declamp_wmax is not None:
        hp["bdsp_wmax"] = float(declamp_wmax)                # de-clamp held CONSTANT across arms (NOT the lever)
    return BennaFusiEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                             eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                             logit_source="leaky_readout", w_clip=w_clip, hp=hp, meta_permute_seed=seed, **kwargs)


def _age_buckets(accs):
    """Oldest / middle / newest retention over the acquired set (fact 0 = oldest). Returns per-position anchors
    (fact 0, N//2, N-1) AND third-bucket means, so 'protects the oldest' is an explicit, reported number."""
    N = len(accs)
    a = np.asarray(accs, dtype=np.float64)
    third = max(1, N // 3)
    return {"oldest_fact_acc": float(a[0]), "mid_fact_acc": float(a[N // 2]), "newest_fact_acc": float(a[-1]),
            "oldest_third_mean": float(a[:third].mean()), "middle_third_mean": float(a[third:2 * third].mean()),
            "newest_third_mean": float(a[2 * third:].mean())}


def _run_arm(arm, kwargs, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip,
             n_draws, milestones, test_n, chance, declamp_wmax):
    """Teach the referents SEQUENTIALLY into ONE brain; consolidate after each fact; record retention + per-age
    retention at each milestone. The ONLY cross-arm difference is kwargs (the consolidation gate)."""
    net = _mk_bf_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, declamp_wmax, kwargs)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    acquire_acc, retention, gate_curve = [], {}, {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)             # WAKE: teacher draws from the world (legitimate)
        _teach_fact(net, X, y, epochs, batch, teach_rng)          # brain acquires the fact by e-prop weight change
        net.consolidate_fact()                                    # cascade tick AFTER the fact (lesion skips)
        acq = _fact_acc(net, env, r, i, n=test_n)                 # immediate held-out acquisition (plasticity side)
        acquire_acc.append(acq)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {"frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                                 "one_over_N": float(1.0 / N), "mean_retained_acc": float(np.mean(accs)),
                                 "per_fact_acc": [float(a) for a in accs], **_age_buckets(accs)}
            gate_curve[str(N)] = net.gate_summary()
    return {"arm": arm, "kwargs": {k: (bool(v) if isinstance(v, bool) else v) for k, v in kwargs.items()},
            "acquire_acc_immediate": [float(a) for a in acquire_acc],
            "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
            "final_gate": net.gate_summary(), "gate_curve": gate_curve, "retention_curve": retention}


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise, test_n,
        declamp_wmax, meta_gain, meta_consol_rate, chain_depth, chain_ratio, gate_base, gate_slope):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    G, R = float(meta_gain), float(meta_consol_rate)
    chain_kw = dict(meta_enabled=True, meta_gain=G, chain_enabled=True, chain_depth=chain_depth,
                    chain_ratio=chain_ratio, chain_consol_rate=R, gate_base=gate_base, gate_slope=gate_slope)
    specs = {
        "vanilla":       dict(meta_enabled=False),
        "single_var":    dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R),       # the arm to BEAT
        "chain":         dict(chain_kw),                                                 # the treatment
        "chain_lesion":  dict(chain_kw, chain_lesion_freeze=True),                       # state frozen 0 -> vanilla
        "chain_permute": dict(chain_kw, chain_permute=True),                             # wrong-synapse protection
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
        row = rc[str(big)] if big else {}
        fr = row.get("frac_recalled", float("nan"))
        old = row.get("oldest_fact_acc", float("nan"))
        gmin = min((arms[name]["final_gate"][li]["min_gate"] for li in arms[name]["final_gate"]), default=1.0)
        print(f"[arm {name:14s}] {arms[name]['wall_seconds']:6.0f}s | immediate-acq "
              f"{arms[name]['mean_acquire_acc_immediate']:.3f} | min-gate {gmin:.3f} | "
              f"frac-recalled@N={big}: {fr:.3f} (1/N={1.0/big:.3f}) | oldest-fact {old:.3f}", flush=True)
    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "declamp_wmax": declamp_wmax, "meta_gain": G, "meta_consol_rate": R,
                       "chain_depth": chain_depth, "chain_ratio": chain_ratio, "gate_base": gate_base,
                       "gate_slope": gate_slope, "gate_weights_norm_mean1": True,
                       "backend": os.environ.get("SIM_BACKEND")},
            "arms": arms}


def _verdict(result):
    from tools.lab import lever, attributable_to, assert_backend
    from tools.verdict import Verdict
    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"), note="(bennafusi chain de-risk)")
    rc = {a: result["arms"][a]["retention_curve"] for a in result["arms"]}
    big = max((int(k) for k in rc["chain"]), default=None)
    key = str(big)
    f = {a: rc[a][key]["frac_recalled"] for a in rc}
    old = {a: rc[a][key]["oldest_fact_acc"] for a in rc}          # the crux: oldest fact (fact 0) @ N_max
    mid = {a: rc[a][key]["mid_fact_acc"] for a in rc}
    new = {a: rc[a][key]["newest_fact_acc"] for a in rc}
    acq = {a: result["arms"][a]["mean_acquire_acc_immediate"] for a in result["arms"]}
    chance = result["chance"]
    one_over_N = 1.0 / big

    # (d) THE LEVER MOVED: vanilla gate == 1.0 everywhere (no protection); chain gate < 1.0 (protection active).
    van_min = min(result["arms"]["vanilla"]["final_gate"][li]["min_gate"] for li in result["arms"]["vanilla"]["final_gate"])
    chn_min = min(result["arms"]["chain"]["final_gate"][li]["min_gate"] for li in result["arms"]["chain"]["final_gate"])
    lever("chain gate (min effective-lr scale)", round(van_min, 4), round(chn_min, 4))

    # (a)/(b)/(e) attribution: fraction of the effect NOT present in the no-state / wrong-synapse / single-var controls.
    attributable_to("chain vs vanilla (frac_recalled@Nmax)", f["chain"], f["vanilla"])
    attributable_to("chain vs single_var (extra timescales)", f["chain"], f["single_var"])
    attributable_to("chain vs chain_lesion (state frozen 0)", f["chain"], f["chain_lesion"])
    attributable_to("chain vs chain_permute (wrong-synapse)", f["chain"], f["chain_permute"])
    # the crux, on the OLDEST fact: chain must protect fact 0 where the single var could not.
    attributable_to("chain vs single_var (OLDEST-fact acc @Nmax)", old["chain"], old["single_var"], warn_below=0.0)

    v = Verdict("teacher-loop benna-fusi consolidation chain", chance=chance)
    v.reaches("(1) chain beats vanilla (acquisition-at-scale)", before=f["vanilla"], after=f["chain"])
    v.reaches("(2) chain beats single_var (extra timescales help)", before=f["single_var"], after=f["chain"])
    v.reaches("(3) chain protects OLDEST fact vs single_var", before=old["single_var"], after=old["chain"])
    v.reaches("(4) lesion (state frozen) collapses to ~vanilla", before=f["chain"], after=f["chain_lesion"])
    v.reaches("(5) permute (wrong synapses) collapses to ~vanilla", before=f["chain"], after=f["chain_permute"])
    v.floor("(6) chain keeps acquiring new facts (immediate-acq)", acq["chain"], floor=0.6)
    # GO: chain clears the strict +0.15 vanilla margin, BEATS single_var, PROTECTS the oldest fact, both controls
    #     collapse near vanilla, and it still acquires. 6-seed-confirmable; single-seed prints SMOKE, never a GO.
    go = (f["chain"] > f["vanilla"] + 0.15
          and f["chain"] > f["single_var"] + 0.05
          and old["chain"] > old["single_var"] + 0.10
          and f["chain"] > f["chain_lesion"] + 0.15
          and f["chain"] > f["chain_permute"] + 0.10
          and f["chain_lesion"] <= f["vanilla"] + 0.12
          and acq["chain"] >= 0.6)
    decision = v.decide(go=go)
    return {"largest_N": big, "one_over_N": one_over_N, "backend": backend,
            "frac_recalled": f, "oldest_fact_acc": old, "mid_fact_acc": mid, "newest_fact_acc": new,
            "immediate_acq": acq, "vanilla_min_gate": float(van_min), "chain_min_gate": float(chn_min),
            "chain_beats_vanilla": float(f["chain"] - f["vanilla"]),
            "chain_beats_single_var": float(f["chain"] - f["single_var"]),
            "chain_oldest_beats_single_var": float(old["chain"] - old["single_var"]),
            "lesion_collapse_margin": float(f["chain"] - f["chain_lesion"]),
            "permute_collapse_margin": float(f["chain"] - f["chain_permute"]),
            "lesion_matches_vanilla": bool(abs(f["chain_lesion"] - f["vanilla"]) <= 0.12), **decision}


def _aggregate(paths):
    """6-seed roll-up. GO = every seed: chain > vanilla+0.15 AND > single_var+0.05 AND oldest-fact > single_var+0.10
    AND > chain_lesion+0.15 AND > chain_permute+0.10 AND chain_lesion ~= vanilla AND chain immediate-acq >= 0.6."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        vd = d["verdict"]
        f = vd["frac_recalled"]; old = vd["oldest_fact_acc"]; acq = vd["immediate_acq"]
        seed_go = bool(f["chain"] > f["vanilla"] + 0.15 and f["chain"] > f["single_var"] + 0.05
                       and old["chain"] > old["single_var"] + 0.10 and f["chain"] > f["chain_lesion"] + 0.15
                       and f["chain"] > f["chain_permute"] + 0.10 and f["chain_lesion"] <= f["vanilla"] + 0.12
                       and acq["chain"] >= 0.6)
        rows.append({"seed": d["seed"], "N": vd["largest_N"], **{a: f[a] for a in ARMS},
                     "old_chain": old["chain"], "old_single": old["single_var"],
                     "acq_chain": acq["chain"], "seed_go": seed_go})
    means = {a: float(np.mean([r[a] for r in rows])) for a in ARMS}
    old_c = float(np.mean([r["old_chain"] for r in rows])); old_s = float(np.mean([r["old_single"] for r in rows]))
    n_go = sum(r["seed_go"] for r in rows)
    go = n_go == len(rows) and len(rows) >= 6
    print("\n" + "=" * 118)
    print(f"[AGG] {len(rows)} seeds | GO needs chain>vanilla+.15 & >single_var+.05 & oldest>single_var+.10 & "
          f">lesion+.15 & >permute+.10 & lesion~=vanilla, ALL seeds")
    print(f"{'seed':>5} {'N':>4} " + " ".join(f"{a[:12]:>13}" for a in ARMS) +
          f" {'oldChain':>9} {'oldSingle':>10} {'acqCh':>7} {'GO':>4}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>5} {r['N']:>4} " + " ".join(f"{r[a]:>13.3f}" for a in ARMS) +
              f" {r['old_chain']:>9.3f} {r['old_single']:>10.3f} {r['acq_chain']:>7.3f} {str(r['seed_go']):>4}")
    print(f"{'mean':>5} {'':>4} " + " ".join(f"{means[a]:>13.3f}" for a in ARMS) +
          f" {old_c:>9.3f} {old_s:>10.3f}")
    print(f"[AGG] chain {means['chain']:.3f} vs single_var {means['single_var']:.3f} vs vanilla {means['vanilla']:.3f} "
          f"| oldest chain {old_c:.3f} vs single {old_s:.3f} | seeds GO {n_go}/{len(rows)} "
          f"| VERDICT {'GO' if go else 'NO-GO'}")
    print("=" * 118)
    return 0 if go else 1


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop Benna-Fusi consolidation chain: does a multi-timescale "
                                             "cascade protect the OLDEST facts the single-var metaplastic gate could "
                                             "not, and beat it on frac_recalled?")
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-seed JSONs -> 6-seed GO roll-up")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--milestones", type=int, nargs="+", default=[8, 16, 32])
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
                    help="bdsp_w_max for ALL arms (de-clamp held constant; NOT the lever). <0 keeps the +-6 default.")
    ap.add_argument("--meta-gain", type=float, default=8.0, help="gain g in lr_eff=lr/(1+g*C)")
    ap.add_argument("--meta-consol-rate", type=float, default=1.0, help="per-fact consolidation increment rate")
    ap.add_argument("--chain-depth", type=int, default=6, help="number of Benna-Fusi cascade variables per synapse")
    ap.add_argument("--chain-ratio", type=float, default=2.0, help="geometric timescale ratio between chain levels")
    ap.add_argument("--gate-base", type=float, default=1.0, help="gate-weight base (level 0); weights normed to mean 1")
    ap.add_argument("--gate-slope", type=float, default=1.0, help="gate-weight slope per level (deep-emphasis)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.aggregate:
        return _aggregate(a.aggregate)
    if a.declamp_wmax is not None and a.declamp_wmax < 0:
        a.declamp_wmax = None
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.declamp_wmax, a.meta_gain, a.meta_consol_rate,
                 a.chain_depth, a.chain_ratio, a.gate_base, a.gate_slope)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_bennafusi_chain", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    f = verdict["frac_recalled"]; old = verdict["oldest_fact_acc"]; acq = verdict["immediate_acq"]
    print("\n" + "=" * 118, flush=True)
    print(f"[bennafusi-chain] seed {a.seed} @ N={verdict['largest_N']} (1/N={verdict['one_over_N']:.3f}, "
          f"chance {result['chance']:.3f}):", flush=True)
    for arm in ARMS:
        print(f"    {arm:14s}: frac-recalled {f[arm]:.3f} | oldest-fact {old[arm]:.3f} | "
              f"immediate-acq {acq[arm]:.3f}", flush=True)
    print(f"[bennafusi-chain] chain-beats-vanilla {verdict['chain_beats_vanilla']:+.3f} | "
          f"chain-beats-single_var {verdict['chain_beats_single_var']:+.3f} | "
          f"oldest chain-vs-single {verdict['chain_oldest_beats_single_var']:+.3f} | "
          f"lesion-collapse {verdict['lesion_collapse_margin']:+.3f} | "
          f"permute-collapse {verdict['permute_collapse_margin']:+.3f} | VERDICT {verdict['status']}", flush=True)
    print(f"[bennafusi-chain] wrote {a.out}\n" + "=" * 118, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
