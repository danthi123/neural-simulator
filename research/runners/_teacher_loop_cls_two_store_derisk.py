"""TEACHER-LOOP CLS TWO-STORE DE-RISK (2026-08-09): the BREADTH crux was resolved by SIZING THE RESERVOIR to the
fact count (flat capacity, 0.967 @ N=20). But a single flat store that GROWS with N means per-step consolidation
compute grows with everything ever learned -- a wall for a YEAR of real-world learning. The biology's answer is
Complementary Learning Systems (McClelland/McNaughton/O'Reilly 1995): a BOUNDED fast hippocampal store (the recent
working set) + a slow distributed cortical store, with systems consolidation moving memories fast->slow via
interleaved replay, after which the hippocampal index DECAYS. The fast store never grows with lifetime -- it stays
BOUNDED. This de-risk tests whether that gives a BOUNDED working set on our substrate: retention DECOUPLED from
fast-store size (and thus from per-step consolidation cost).

THE HYPOTHESIS. A FIXED-SIZE fast store F (holds only the ~F most-recent facts' engrams) + a slow cortical readout,
consolidated by interleaved SELF-REPLAY (fast->slow) with older facts EVICTED from the fast store after
consolidation, RETAINS N facts even when N >> F -- so retention decouples from fast-store size (and per-step cost).
Contrast with the flat store, which needs an O(N) replay set to retain N.

THE TWO STORES (both brain-based; host code ONLY for world/body):
  * FAST STORE = the hippocampal engram buffer (`BoundedHippocampus`): a fixed-capacity list of compressed engrams
    (the brain's own lossy wake traces). After each SLEEP its index DECAYS -- engrams older than the F most-recent
    are EVICTED, so the store NEVER grows with lifetime N. The FLAT baseline is the SAME store UNBOUNDED (it grows
    to N and replays all N every sleep).
  * SLOW STORE = the shared leaky e-prop readout over a FIXED large reservoir (the cortex). Sized ONCE to the
    corpus (born fully at start, NEVER grown per-fact -> NOT the flat growing reservoir in disguise; asserted). It
    is moved ONLY by e-prop -- during WAKE (the new fact) and during SLEEP (self-replay of the fast store).

CONSOLIDATION IS NEURAL SELF-REPLAY, not a host copy. `_self_replay_consolidate` (reused from the sleep-replay
de-risk) takes ONLY the fast store: the hippocampus GENERATES replay patterns from its stored engrams (engram +
brain-owned variability, a separate brain RNG; teacher + world ABSENT -- the function has NO `env` param) and the
SAME e-prop rule (`_teach_fact` -> `net.train_batch`) moves the slow readout. Fast-store-generated spikes ->
transport-free e-prop on the slow readout. There is NO host weight/input copy fast->slow anywhere.

THREE ARMS (same net build / seed / wake budget / slow reservoir; the ONLY difference is the fast-store schedule):
  * two_store   = TREATMENT. Bounded fast store (capacity F). Encode -> SLEEP self-replay of the <=F buffer ->
                  EVICT to F (index decays). Per-step replay set is O(F), CONSTANT in N.
  * flat        = BASELINE, MEASURED in-run (the capacity result, de-clamped). UNBOUNDED fast store: encode ->
                  SLEEP self-replay of ALL N -> NO eviction. Per-step replay set is O(N).
  * no_consol   = CONTROL (consolidation load-bearing). Bounded fast store F, but SLEEP self-replay DISABLED. The
                  slow readout gets ONLY wake training -> it forgets, and evicted facts have no lasting store. If
                  the two_store retention were really "the big slow reservoir memorizing during wake", no_consol
                  (same reservoir, same wake) would retain too. It must FORGET the older facts.

THE KEY MEASUREMENT: retention as N grows with F FIXED (F=5; N=10, 20). Does two_store retention hold while the
fast store stays F (decoupling cost from lifetime)? two_store ~ flat with |fast|=F<<N is the GO; two_store ~
no_consol (forgets what was evicted) is the honest NEGATIVE with teeth (report WHAT consolidation fails to
transfer).

ANTI-CHEATS (each a REAL assertion in the output):
  * fast store TRULY bounded: two_store's active fast-store size is F at BOTH N=10 and N=20 (does NOT grow with N);
    flat's is N. Reported per milestone + the max replay-set size.
  * consolidation NEURAL: fast-store-GENERATED replay -> e-prop (grep: `_self_replay_consolidate` has no `env`;
    it calls `hippo.generate_replay` then `_teach_fact`). No host weight/input copy.
  * the slow store is NOT the flat growing reservoir in disguise: the slow reservoir n_active is CONSTANT across
    the whole curriculum (born fully at start; asserted equal at N=10 and N=20).
  * cfg.seed byte-identical substrate across two builds at one seed (NOT actual_seed_used); de-clamped
    bdsp_wmax=1e9 (the -6/+6 clamp silences the reservoir -- see 8ca014ff2); git diff main -- sim/ empty; backend
    recorded.

DISCIPLINE: reuse-by-import (NeurogenesisNet for the fixed de-clamped slow reservoir; _feat/_teach_fact/_fact_acc/
_corrective_batch/N_ACT from the scaling de-risk; Hippocampus/_self_replay_consolidate from the sleep-replay
de-risk; ReferentEnv from the corrective-acquire de-risk). NO sim/ edit. SIM_BACKEND=numpy (tiny launch-bound net).

RUN (single-seed smoke, numpy):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_cls_two_store_derisk --seed 42 \
      --n-max 20 --milestones 10 20 --capacity 5 --slow-hidden 100 \
      --epochs 20 --replay-epochs 12 --replay-per-fact 8 --n-draws 16 --settle-steps 20 --test-n 40 \
      --out research/findings/raw/teacher_loop_cls_two_store_s42.json
  6-SEED (self-sweep, one aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_cls_two_store_derisk --seeds 42 43 44 45 46 47 \
      --n-max 20 --milestones 10 20 --capacity 5 --slow-hidden 100 \
      --epochs 20 --replay-epochs 12 --replay-per-fact 8 --n-draws 16 --settle-steps 20 --test-n 40 \
      --out research/findings/raw/teacher_loop_cls_two_store.json
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
# reuse-by-import: the fixed de-clamped slow reservoir (NeurogenesisNet, born fully at start = a matched_fixed
# cortex), the scaling teacher machinery, the sleep-replay hippocampus + neural self-replay. NO sim/ edit.
from research.runners._teacher_loop_neurogenesis_capacity_derisk import NeurogenesisNet  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import (  # noqa: E402
    Hippocampus, _self_replay_consolidate,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_cls_two_store.json"


# ============================== the BOUNDED FAST STORE: a fixed-capacity hippocampal buffer ==============================
class BoundedHippocampus(Hippocampus):
    """The fast store. Inherits the sleep-replay Hippocampus (wake `encode` of a compressed engram; sleep
    `generate_replay` from the brain's OWN engrams + brain-owned variability). ADDS a fixed capacity F: after each
    consolidation the index DECAYS -- engrams older than the F most-recent are EVICTED, so the store NEVER grows
    with lifetime N. capacity <= 0 => UNBOUNDED (the flat baseline: grows to N, replays all N).

    This is systems consolidation's bounded working set: a fact is captured, self-replayed into cortex for the F
    rounds it stays in the buffer, then dropped (the hippocampal index decays; McClelland/McNaughton/O'Reilly 1995).
    The replay pattern is self-GENERATED from the stored engram -- never the world/teacher re-presenting."""

    def __init__(self, seed, capacity, replay_noise=0.10):
        super().__init__(seed, replay_noise=replay_noise)
        self.capacity = int(capacity)          # F; <=0 => unbounded (flat)
        self.max_replay_set = 0                # the largest replay set ever consolidated (bound witness)

    def note_replay_size(self):
        self.max_replay_set = max(self.max_replay_set, len(self.engrams))

    def evict(self):
        """Index decay: keep only the F most-recent engrams (older EVICTED). No-op if unbounded."""
        if self.capacity > 0 and len(self.engrams) > self.capacity:
            self.engrams = self.engrams[-self.capacity:]

    @property
    def active_size(self):
        return len(self.engrams)


# ================================= build the FIXED large slow cortical reservoir =================================
def _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax, env, referents):
    """The SLOW store = a NeurogenesisNet whose reservoir is born FULLY at start (a matched_fixed cortex): a fixed
    large leaky e-prop readout, de-clamped (bdsp_wmax=1e9), NEVER grown per-fact. Returns (net, n_active)."""
    net = NeurogenesisNet(n_in, K, seed, slow_hidden, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax=bdsp_wmax)
    net.birth(slow_hidden, env, referents, seed)   # birth the WHOLE reservoir once (fixed cortex; no per-fact growth)
    return net, int(net.n_active)


# ====================================== one arm of the sequential curriculum ======================================
def _run_arm(arm, seed, referents, env, K, n_in, slow_hidden, capacity, settle, epochs, batch, eprop_lr, w_clip,
             n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance, bdsp_wmax):
    """arm in {two_store, flat, no_consol}. All arms: same fixed slow reservoir + same wake budget. The manipulation
    is the FAST-STORE schedule: two_store = bounded F + consolidate + evict; flat = unbounded + consolidate;
    no_consol = bounded F + NO consolidation (the slow store gets wake only)."""
    consolidate = arm in ("two_store", "flat")
    cap = 0 if arm == "flat" else int(capacity)     # flat => unbounded fast store (grows to N)

    net, slow_active0 = _build_slow_cortex(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax,
                                           env, referents)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)   # brain-owned RNG for the interleaved consolidation shuffle
    hippo = BoundedHippocampus(seed, capacity=cap, replay_noise=replay_noise)

    acquire_acc = []
    retention = {}
    fast_size_trace = []          # active fast-store size after eviction, per fact (the boundedness witness)
    slow_active_trace = []        # slow reservoir n_active per fact (must stay CONSTANT: not a growing reservoir)
    for i, r in enumerate(referents):
        # --- WAKE: teacher teaches fact i from the world (env draws = the legitimate sensory environment) ---
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))     # IMMEDIATE acquisition, before consolidation
        # --- the FAST store captures the engram of this episode ---
        hippo.encode(X, i)
        # --- SLEEP: neural self-replay of the CURRENT fast buffer into the slow readout (teacher + world ABSENT) ---
        if consolidate:
            hippo.note_replay_size()
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=False)
        # --- index DECAY: evict engrams older than the F most-recent (two_store); no-op for flat/unbounded ---
        hippo.evict()
        fast_size_trace.append(hippo.active_size)
        slow_active_trace.append(int(net.n_active))
        N = i + 1
        if N in milestones:
            # RECALL reads the SLOW cortical store (the neural readout). A fact still in the fast buffer is captured
            # here too because it was just self-replayed into the slow store -- no host argmax over the buffer.
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "fast_store_active_size": int(hippo.active_size),        # F for two_store/no_consol; N for flat
                "slow_reservoir_active": int(net.n_active),
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                # the EVICTED set (facts no longer in the fast buffer): their retention is what consolidation must carry.
                "evicted_frac_recalled": (float(np.mean([a >= max(0.5, chance + 0.15) for a in accs[:max(0, N - hippo.active_size)]]))
                                          if N - hippo.active_size > 0 else None),
                "per_fact_acc": [float(a) for a in accs],
            }
    return {
        "arm": arm, "fast_capacity": cap, "consolidate": consolidate,
        "slow_reservoir_active_start": slow_active0,
        "slow_reservoir_active_constant": bool(len(set(slow_active_trace)) == 1),
        "max_replay_set_size": int(hippo.max_replay_set),
        "fast_size_trace": fast_size_trace,
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
    }


# ================================================= anti-cheat asserts =================================================
def _assert_byte_identical_substrate(n_in, K, seed, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax):
    a = NeurogenesisNet(n_in, K, seed, slow_hidden, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax=bdsp_wmax)
    b = NeurogenesisNet(n_in, K, seed, slow_hidden, slow_hidden, settle, eprop_lr, w_clip, bdsp_wmax=bdsp_wmax)
    ta = np.asarray(to_host(a.br.cp_neuron_firing_thresholds), dtype=np.float64)
    tb = np.asarray(to_host(b.br.cp_neuron_firing_thresholds), dtype=np.float64)
    return bool(np.array_equal(ta, tb)), float(np.max(np.abs(ta - tb)))


def _git_sim_diff_empty():
    try:
        out = subprocess.run(["git", "diff", "main", "--", "sim/"], cwd=str(_REPO),
                             capture_output=True, text=True, timeout=30)
        return (out.returncode == 0 and out.stdout.strip() == ""), out.stdout[:400]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run(seed, n_max, milestones, capacity, slow_hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p,
        noise, test_n, replay_epochs, replay_per_fact, replay_noise, arms_to_run, bdsp_wmax):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    referents = [f"ref{i}" for i in range(n_max)]

    seeded_ok, seed_maxdiff = _assert_byte_identical_substrate(n_in, K, seed, slow_hidden, settle, eprop_lr,
                                                               w_clip, bdsp_wmax)
    sim_clean, sim_diff = _git_sim_diff_empty()

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        env = ReferentEnv(seed, d_p=d_p, noise=noise)          # fresh env per arm: identical referents + draw stream
        for r in referents:
            env.proto(r)
        env.rng = np.random.default_rng(seed + 101)            # reset the draw-stream so each arm sees the SAME percepts
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, slow_hidden, capacity, settle, epochs, batch,
                             eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact,
                             replay_noise, chance, bdsp_wmax)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        fs = rc[str(big)]["fast_store_active_size"] if big else -1
        print(f"[arm {arm:10s}] {arms[arm]['wall_seconds']:.0f}s | fast|store|@N={big}: {fs:3d} | "
              f"immediate-acq {arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}",
              flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "capacity_F": int(capacity), "slow_hidden": int(slow_hidden),
            "substrate_byte_identical": seeded_ok, "substrate_seed_maxdiff": seed_maxdiff,
            "sim_diff_empty": sim_clean, "sim_diff_head": sim_diff,
            "config": {"capacity_F": capacity, "slow_hidden": slow_hidden, "settle_steps": settle, "epochs": epochs,
                       "batch": batch, "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p,
                       "noise": noise, "test_n": test_n, "replay_epochs": replay_epochs,
                       "replay_per_fact": replay_per_fact, "replay_noise": replay_noise, "bdsp_wmax": bdsp_wmax,
                       "frozen_hidden": True},
            "arms": arms}


def _verdict(result):
    """Verdict + GO. TEETH:
      (KEY) two_store retention HOLDS as N grows with F FIXED: two_store frac @ largest N is near the flat baseline
            (>= flat - 0.15) AND well above chance, while its fast store stays F (<< N).
      (bounded) two_store fast-store size is F at every milestone (does NOT grow with N); flat's is N.
      (load-bearing) no_consol (consolidation lesioned, SAME slow reservoir + wake) FORGETS: two_store - no_consol
            > 0.20 -> the retention is carried by consolidation, not the big slow reservoir memorizing during wake.
      Anti-cheats: byte-identical substrate, sim/ clean, slow reservoir constant (not a growing reservoir)."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]
    F = result["capacity_F"]

    def frac_at(arm, N):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return rc.get(str(N), {}).get("frac_recalled", float("nan"))

    def fast_at(arm, N):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return rc.get(str(N), {}).get("fast_store_active_size", None)

    def big_of(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        return max((int(k) for k in rc), default=None)

    big = big_of("two_store") or big_of("flat")
    two_f = frac_at("two_store", big)
    flat_f = frac_at("flat", big)
    noc_f = frac_at("no_consol", big)
    two_acq = arms.get("two_store", {}).get("mean_acquire_acc_immediate", float("nan"))

    # partial-arm probe: if the treatment or the load-bearing control is missing, emit measured curves, skip GO.
    if "two_store" not in arms or "no_consol" not in arms:
        return {"largest_N": big, "two_store_frac": two_f, "flat_frac": flat_f, "no_consol_frac": noc_f,
                "status": "PARTIAL"}

    # boundedness: two_store fast-store size is F at every milestone; flat is N.
    ms = result["milestones"]
    two_fast_sizes = {str(N): fast_at("two_store", N) for N in ms}
    flat_fast_sizes = {str(N): fast_at("flat", N) for N in ms} if "flat" in arms else {}
    two_bounded = all((two_fast_sizes[str(N)] is not None and two_fast_sizes[str(N)] <= F) for N in ms)
    two_no_grow = len(set(v for v in two_fast_sizes.values() if v is not None)) == 1   # SAME size at every N
    flat_grows = (all((flat_fast_sizes.get(str(N)) == N) for N in ms) if flat_fast_sizes else None)
    slow_constant = bool(arms["two_store"].get("slow_reservoir_active_constant"))

    # the retention must be carried by CONSOLIDATION, not the slow reservoir memorizing during wake.
    attributable_to("consolidation (two_store vs no_consol, same slow reservoir + wake)", two_f, noc_f)
    if "flat" in arms:
        attributable_to("bounded fast store retains what flat's O(N) store does (two_store vs flat)", two_f, flat_f)

    v = Verdict("teacher-loop CLS two-store (bounded fast + slow cortex)", chance=chance)
    # KEY: does bounded retention hold near the flat baseline while |fast|=F<<N?
    if not np.isnan(flat_f):
        v.reaches("(KEY) bounded fast store retains near the flat O(N) baseline", before=noc_f, after=two_f)
        v.require("(KEY') two_store within 0.15 of flat AND above 0.5", (two_f >= flat_f - 0.15 and two_f >= 0.5),
                  expect=True, note=f"two_store {two_f:.2f} vs flat {flat_f:.2f} @ N={big}")
    v.control("(load-bearing) consolidation carries it (two_store vs no_consol)", treatment=two_f, control=noc_f,
              min_separation=0.20)
    v.require("(bounded) two_store fast store <= F at every milestone", two_bounded, expect=True,
              note=f"sizes {two_fast_sizes} (F={F})")
    v.require("(bounded') two_store fast store does NOT grow with N", two_no_grow, expect=True,
              note=f"sizes {two_fast_sizes}")
    v.require("(decoupled) slow reservoir is CONSTANT across the curriculum (not a growing reservoir)",
              slow_constant, expect=True)
    v.floor("(acq) two_store immediate acquisition stays high", two_acq, floor=0.85)
    v.require("(seed) substrate byte-identical across two builds at one seed", bool(result["substrate_byte_identical"]),
              expect=True, note=f"max threshold diff {result['substrate_seed_maxdiff']:.2e}")
    v.require("(sim) git diff main -- sim/ empty", bool(result["sim_diff_empty"]), expect=True)

    go = (two_bounded and two_no_grow and slow_constant and two_acq >= 0.85
          and (two_f - noc_f) > 0.20 and bool(result["substrate_byte_identical"]) and bool(result["sim_diff_empty"]))
    if not np.isnan(flat_f):
        go = go and (two_f >= flat_f - 0.15) and (two_f >= 0.5)
    decision = v.decide(go=go)

    return {
        "largest_N": big, "capacity_F": F,
        "two_store_frac_recalled": two_f, "flat_frac_recalled": flat_f, "no_consol_frac_recalled": noc_f,
        "two_store_immediate_acq": two_acq,
        "two_store_minus_no_consol": float(two_f - noc_f),
        "two_store_minus_flat": (float(two_f - flat_f) if not np.isnan(flat_f) else None),
        "retention_vs_N": {str(N): {"two_store": frac_at("two_store", N), "flat": frac_at("flat", N),
                                    "no_consol": frac_at("no_consol", N),
                                    "two_store_fast_size": fast_at("two_store", N),
                                    "flat_fast_size": fast_at("flat", N)} for N in ms},
        "evicted_frac_recalled_at_bigN": arms["two_store"]["retention_curve"].get(str(big), {}).get("evicted_frac_recalled"),
        "two_store_fast_bounded": two_bounded, "two_store_fast_no_grow": two_no_grow, "flat_fast_grows": flat_grows,
        "max_replay_set_two_store": arms["two_store"].get("max_replay_set_size"),
        "max_replay_set_flat": arms.get("flat", {}).get("max_replay_set_size"),
        "slow_reservoir_constant": slow_constant,
        "substrate_byte_identical": result["substrate_byte_identical"], "sim_diff_empty": result["sim_diff_empty"],
        **decision,
    }


def _one_seed(a, seed, arms_to_run):
    result = run(seed, a.n_max, a.milestones, a.capacity, a.slow_hidden, a.settle_steps, a.epochs, a.batch,
                 a.eprop_lr, a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact,
                 a.replay_noise, arms_to_run, a.bdsp_wmax)
    return result, _verdict(result)


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop CLS TWO-STORE: a BOUNDED fast hippocampal store + a slow "
                                             "cortical readout, consolidated by neural self-replay with eviction, to "
                                             "decouple retention (and per-step cost) from lifetime N.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--capacity", type=int, default=5, help="F = the FIXED fast-store capacity (bounded working set)")
    ap.add_argument("--slow-hidden", type=int, default=100, help="the FIXED slow cortical reservoir size (born fully "
                                                                 "at start; sized to the corpus, NEVER grown per-fact)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20, help="per-fact WAKE teaching epochs")
    ap.add_argument("--replay-epochs", type=int, default=12, help="SLEEP self-replay epochs over the fast buffer")
    ap.add_argument("--replay-per-fact", type=int, default=8, help="self-generated replay draws per stored engram")
    ap.add_argument("--replay-noise", type=float, default=0.10, help="brain-owned variability on the replayed engram")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--bdsp-wmax", type=float, default=1e9, help="1e9 = de-clamped (required; 6 = the CLAMP that "
                                                                "silences the reservoir, bound-trap 8ca014ff2)")
    ap.add_argument("--n-draws", type=int, default=16)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["two_store", "flat", "no_consol"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arms_to_run = list(a.arms)

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}  (F={a.capacity}, slow_hidden={a.slow_hidden}, n_max={a.n_max})\n"
              + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_cls_two_store", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        print("\n" + "=" * 100, flush=True)
        rv = verdict
        print(f"[cls] seed {s} @ N={rv.get('largest_N')}: two_store {rv.get('two_store_frac_recalled', float('nan')):.2f} "
              f"| flat {rv.get('flat_frac_recalled', float('nan')):.2f} | no_consol "
              f"{rv.get('no_consol_frac_recalled', float('nan')):.2f} (chance {result['chance']:.2f})", flush=True)
        rvn = rv.get("retention_vs_N", {})
        for N in result["milestones"]:
            d = rvn.get(str(N), {})
            print(f"    N={N:3d}: two_store {d.get('two_store', float('nan')):.2f} (|fast|={d.get('two_store_fast_size')}) | "
                  f"flat {d.get('flat', float('nan')):.2f} (|fast|={d.get('flat_fast_size')}) | "
                  f"no_consol {d.get('no_consol', float('nan')):.2f}", flush=True)
        print(f"[cls] two_store-no_consol {rv.get('two_store_minus_no_consol', float('nan')):+.2f} | "
              f"two_store-flat {rv.get('two_store_minus_flat')} | fast-bounded {rv.get('two_store_fast_bounded')} "
              f"| no-grow {rv.get('two_store_fast_no_grow')} | slow-constant {rv.get('slow_reservoir_constant')} "
              f"| immediate-acq {rv.get('two_store_immediate_acq', float('nan')):.3f} | VERDICT {rv.get('status')}",
              flush=True)
        print(f"[cls] byte-identical {rv.get('substrate_byte_identical')} | sim-clean {rv.get('sim_diff_empty')} | "
              f"max-replay two/flat {rv.get('max_replay_set_two_store')}/{rv.get('max_replay_set_flat')} | "
              f"wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        two = [p["verdict"].get("two_store_frac_recalled", float("nan")) for p in per_seed]
        flat = [p["verdict"].get("flat_frac_recalled", float("nan")) for p in per_seed]
        noc = [p["verdict"].get("no_consol_frac_recalled", float("nan")) for p in per_seed]
        agg = {"probe": "teacher_loop_cls_two_store_AGG", "seeds": seeds, "backend": os.environ.get("SIM_BACKEND"),
               "capacity_F": a.capacity, "slow_hidden": a.slow_hidden, "n_max": a.n_max,
               "go_count": go_n, "n_seeds": len(seeds),
               "two_store_frac_mean": float(np.nanmean(two)), "flat_frac_mean": float(np.nanmean(flat)),
               "no_consol_frac_mean": float(np.nanmean(noc)),
               "two_store_minus_no_consol_mean": float(np.nanmean(np.array(two) - np.array(noc))),
               "two_store_minus_flat_mean": float(np.nanmean(np.array(two) - np.array(flat))),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[cls AGG] GO {go_n}/{len(seeds)} | two_store {np.nanmean(two):.2f} vs flat {np.nanmean(flat):.2f} "
              f"vs no_consol {np.nanmean(noc):.2f} (F={a.capacity}, slow={a.slow_hidden}) | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
