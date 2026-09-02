"""INTEGRATED-BRAIN SCALE PROBE -- the resource cost of the episodic/ideation spiking organ at n_ca3=400 vs 2000.

WHAT THIS MEASURES (owner-requested de-risk of experiment "B", 2026-09-02). The live production chat brain
(`webapp/server.py:/api/brain-chat`) runs its spiking organs on the process backend, which DEFAULTS TO CUPY/GPU
whenever a CUDA GPU is present (`webapp/server.py:45-80`: `setdefault SIM_BACKEND=cupy` before any sim import;
numpy is only the GPU-less fallback or an explicit CPU override). Its episodic/ideation D5 organ has TWO scales:

  * REDUCED  = n_ca3=400 PRE-ASSIGNED (random-permutation) assembly membership -- the de-risk STAND-IN scale the
    on-substrate mechanism was validated at (`_generative_attractor_wander_onsubstrate_derisk.P`, the numpy-ideation
    stand-in's on-substrate port before it lifts to production).
  * PRODUCTION = n_ca3=2000 EMERGENT DG-selected membership -- what `emergent_assemblies` (R1 sparse-detonator,
    n_ca3=2000) actually selects, the SAME membership the production D5 organ `_episodic_dap_dialogue_memory.
    EpisodicDapMemory` / `d5_episodic_production_organ` stores concepts into, and what `build_production_store`
    (the ONE call `webapp/continuous_engine.py`'s live ideation wiring makes) builds.

BECAUSE production is cupy/GPU and BOTH the spiking substrate AND the Qwen mouth live on the GPU, the resource that
this scale knob actually stresses is GPU VRAM (the "does the 2000 organ + the mouth fit 24 GB" question) plus cupy
per-turn wall-clock. CPU RSS is captured for completeness but is NOT the headline on the production backend.

REUSE-BY-IMPORT, NOT A TOY. The organ arms call the EXACT production build+BTSP-form path:
  * 2000 arm -> `build_production_store(seed, n_mem)`  (emergent=True -> `emergent_assemblies` n_ca3=2000 +
    `_build_and_form`; the live webapp `_spiking_ideate_store` calls this verbatim).
  * 400 arm  -> `_build_and_form(seed, n_mem, P400, emergent=False)`  (the pre-assigned de-risk stand-in scale).
Both compose the SAME dendritic-dAP CA3 readout (`_build_dap_readout`) + `make_readout` + `form_btsp_multi`/
`_form_one_assembly` the D5 episodic organ uses. The per-turn "representative read" is one `blend_settle_production`
(the live per-idle-tick ideation read). `--with-integrated` ADDITIONALLY builds the full `build_one_brain` co-resident
bridge + the converted spiking-Qwen MOUTH and drives ONE real HUMAN_TURN through `run_conversation` (the turing-test
turn driver) so the arm reports the TOTAL integrated VRAM (mouth + main bridge + organ) and a real per-turn latency;
the mouth + main bridge are scale-INVARIANT in n_ca3, so total@scale = integrated_baseline + organ(scale).

ANTI-CHEAT (mandatory, per the task): each arm records the ACTUAL substrate CA3 cell count (`len(R.ca3_idx)`) and the
formed assembly sizes; across arms the probe ASSERTS the count actually grew by the requested factor (2000/400 = 5).
If two arms report the same n_ca3 the knob did NOT engage and the probe prints INVALID loudly rather than a bogus
"no difference". VRAM is read from the cupy memory pool (this-process, clean) with an nvidia-smi whole-GPU sample for
context; on the numpy backend VRAM is reported 0/NA HONESTLY (the substrate is on CPU there), never faked.

DISCIPLINE: no `sim/` edit; additive; reuse-by-import; uses the PROCESS backend (SIM_BACKEND) -- does NOT force one,
so the controller runs it on cupy/GPU for the production-faithful numbers and the correctness smoke runs it on numpy.

  # correctness smoke (CPU, fast -- two SMALL pre-assigned scales so the anti-cheat cross-arm assert exercises):
  SIM_BACKEND=numpy PYTHONPATH=$PWD .venv/bin/python -m research.runners._integrated_brain_scale_probe \
      --smoke --n-ca3 200,400 --n-mem 2 --seed 42 --out research/findings/raw/_integrated_brain_scale/smoke.json

  # FULL production-faithful measurement (GPU -- queue via tools/gpu_queue.sh, needs the GPU):
  SIM_BACKEND=cupy PYTHONPATH=$PWD .venv/bin/python -m research.runners._integrated_brain_scale_probe \
      --n-ca3 400,2000 --n-mem 3 --with-integrated --seed 42 \
      --out research/findings/raw/_integrated_brain_scale/production_400_vs_2000_s42.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Resource sampling helpers (no psutil dependency -- resource.getrusage + /proc for RSS, cupy pool + nvidia-smi
# for VRAM). Each returns None/0 HONESTLY when the resource does not apply (e.g. VRAM on the numpy backend).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _rss_peak_mb() -> float:
    """Process-lifetime PEAK resident set size (MB). ru_maxrss is in KiB on Linux."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _rss_now_mb():
    """Current RSS (MB) from /proc/self/status VmRSS -- a per-checkpoint sample (ru_maxrss is a lifetime peak)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _cupy_pool_mb():
    """(used_mb, total_mb) of THIS process's default cupy memory pool, or (None, None) if cupy is not the backend.
    used_bytes = live allocations; total_bytes = pool reserved from the driver (the real VRAM footprint)."""
    try:
        import cupy  # noqa: PLC0415
        mp = cupy.get_default_memory_pool()
        return mp.used_bytes() / 2 ** 20, mp.total_bytes() / 2 ** 20
    except Exception:
        return None, None


def _cupy_free_pool():
    try:
        import cupy  # noqa: PLC0415
        cupy.get_default_memory_pool().free_all_blocks()
        try:
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    except Exception:
        pass


def _torch_reserved_mb():
    """(reserved_mb, max_reserved_mb) of THIS process's torch CUDA caching allocator, or (None, None). The converted
    spiking-Qwen MOUTH runs on torch, whose VRAM lives in torch's OWN allocator -- NOT the cupy pool -- so the
    integrated-brain total must add this to the cupy-pool substrate footprint."""
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return None, None
        return (torch.cuda.memory_reserved() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20)
    except Exception:
        return None, None


def _nvsmi_used_mb():
    """Whole-GPU used VRAM (MB) for device 0 from nvidia-smi -- CONTEXT only (includes other processes); the clean
    this-process number is the cupy pool + torch reserved. None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        vals = [float(x) for x in out.stdout.splitlines() if x.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


class _PeakTracker:
    """Accumulate the MAX cupy-pool-total and nvidia-smi-used across the checkpoints of ONE arm."""

    def __init__(self):
        self.cupy_used = 0.0
        self.cupy_total = 0.0
        self.torch_reserved = 0.0
        self.nvsmi = 0.0
        self.rss_now = 0.0
        self.have_cupy = False
        self.have_torch = False

    def sample(self):
        u, t = _cupy_pool_mb()
        if u is not None:
            self.have_cupy = True
            self.cupy_used = max(self.cupy_used, u)
            self.cupy_total = max(self.cupy_total, t)
        tr, trmax = _torch_reserved_mb()
        if tr is not None:
            self.have_torch = True
            self.torch_reserved = max(self.torch_reserved, trmax if trmax else tr)
        n = _nvsmi_used_mb()
        if n is not None:
            self.nvsmi = max(self.nvsmi, n)
        r = _rss_now_mb()
        if r is not None:
            self.rss_now = max(self.rss_now, r)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _smoke_params(p: dict) -> dict:
    """Reduce the BTSP-formation + drive/read step budgets so a correctness smoke completes in seconds on the numpy
    CPU backend (RSS < ~4 GB). This is a PLUMBING smoke -- it validates that every field is measured and the
    anti-cheat neuron-count assert fires; it does NOT reproduce the GO physics (which need the full GO_DEFAULTS)."""
    p = dict(p)
    p.update(train_events=6, drive_steps=10, reset_steps=6, warm_steps=20, read_steps=20,
             silence_steps=10, hold_steps=20)
    return p


def _measure_organ_arm(arm_n_ca3: int, *, seed: int, n_mem: int, emergent: bool, smoke: bool, track: _PeakTracker):
    """Build + measure ONE organ scale arm. Returns a dict of the measured fields for this arm.

    2000/emergent -> `build_production_store` (emergent DG-selected membership; n_ca3 lifts to the R1 selection's
    own 2000). Any other n_ca3 (or --no-emergent) -> `_build_and_form(emergent=False)` at PRE-ASSIGNED membership at
    that n_ca3 (the de-risk stand-in scale). Both share the ONE build+BTSP-form code path."""
    from research.runners._generative_attractor_wander_onsubstrate_derisk import (
        P, _build_and_form, _build_readout, _population_up, blend_settle_production)
    from sim.backend import get_backend, to_host

    cp, backend_name = get_backend()
    p = dict(P)
    p["n_ca3"] = int(arm_n_ca3)
    if smoke:
        p = _smoke_params(p)

    rec: dict = {"arm_n_ca3": int(arm_n_ca3), "emergent": bool(emergent), "smoke": bool(smoke),
                 "backend_substrate": backend_name}

    # ── (1) ORGAN BUILD + BTSP FORMATION (the scale-sensitive cost) ──────────────────────────────────────────
    t0 = time.time()
    built = _build_and_form(seed, n_mem, p, emergent=emergent)
    rec["organ_build_s"] = round(time.time() - t0, 3)
    track.sample()
    p_eff = built["p"]                                  # for emergent, p_eff["n_ca3"] is now the selected 2000
    rec["genuine_formation"] = bool(built["genuine_formation"])
    rec["actual_assembly_sizes"] = [int(s) for s in built["sizes"]]
    rec["formation_diag"] = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                             for k, v in built["diag"].items() if k in ("w_within", "cross_dw", "nonmem_dw")}

    # ── ANTI-CHEAT: read the ACTUAL substrate CA3 cell count + total neuron count off a fresh readout harness ──
    # (`_build_and_form` deletes its own bridge; `_build_readout` is the cheap ~0.5s harness the reads use.)
    bridge, R = _build_readout(seed, p_eff, n_mem, assemblies_ext=built["assemblies_ext"])
    rec["actual_n_ca3"] = int(len(R.ca3_idx))
    try:
        rec["organ_bridge_n_neurons"] = int(bridge.core_config.num_neurons)
    except Exception:
        rec["organ_bridge_n_neurons"] = None
    track.sample()
    del bridge, R

    # assemble the store dict `blend_settle_production` consumes (same shape `build_production_store` returns)
    store = dict(seed=int(seed), n_mem=int(n_mem), p=p_eff, assemblies_ext=built["assemblies_ext"],
                 assemblies=built["assemblies"], sizes=built["sizes"],
                 baseline_weights=built["baseline_weights"], formed_weights=built["formed_weights"],
                 diag=built["diag"], genuine_formation=built["genuine_formation"])

    # ── (2) PER-TURN REPRESENTATIVE READ (one live ideation blend-settle read) ──────────────────────────────
    t0 = time.time()
    read_result = blend_settle_production(store, 0, 1) if n_mem >= 2 else None
    rec["organ_recall_read_s"] = round(time.time() - t0, 3)
    rec["blend_read_result"] = read_result
    track.sample()

    # `blend_settle_production` returns None when formation was not genuine (e.g. reduced smoke budgets). To still
    # exercise + TIME the drive/read path (the per-turn substrate op) in that case, run one direct fresh-bridge read.
    if read_result is None:
        t0 = time.time()
        b2, r2 = _build_readout(seed, p_eff, n_mem, assemblies_ext=built["assemblies_ext"])
        r2.C.data[:] = built["formed_weights"]
        cue = None
        try:
            import numpy as _np
            asm0 = built["assemblies"][0]
            cue = _np.asarray(asm0[: max(2, int(0.5 * len(asm0)))], dtype=_np.int64)
        except Exception:
            cue = None
        _population_up(b2, r2, cue, cp=cp, to_host=to_host, drive_pA=p_eff["drive_pA"],
                       warm_steps=p_eff["warm_steps"], read_steps=p_eff["read_steps"],
                       up_thresh=p_eff["up_thresh"], hold_steps=0)
        rec["organ_direct_read_s"] = round(time.time() - t0, 3)
        rec["per_turn_read_path"] = "direct_population_up (formation not genuine -> blend read short-circuited)"
        del b2, r2
        track.sample()
    else:
        rec["per_turn_read_path"] = "blend_settle_production"

    return rec, p_eff


def _measure_integrated(arm_n_ca3: int, *, seed: int, device: str, track: _PeakTracker):
    """OPTIONAL: build the full production integrated brain (build_one_brain co-resident bridge + converted
    spiking-Qwen MOUTH + fm world-model) and drive ONE real HUMAN_TURN through the turing-test turn driver. Reports
    the TOTAL integrated build time + a real per-turn latency + the peak VRAM WITH the mouth loaded. The mouth + main
    bridge are scale-INVARIANT in n_ca3, so this is the FIXED baseline; total@scale = this + the organ(scale)."""
    from research.runners import _stageA_full_integration_derisk as SA
    from research.runners import _conversation_turing_test_derisk as TT
    from research.runners._stageA_foundation_honesty_arbiter_derisk import FacultyRNG
    from sim.backend import get_backend

    xp, backend_name = get_backend()
    out: dict = {"backend_mouth": device}
    t0 = time.time()
    bridge, comp, idx, baseline_snap = SA.build_one_brain(
        seed, with_faculties=True, co_resident_forward_model=True, co_resident_affect_ladder=True,
        co_resident_certainty_opponent=True)
    vocab, facts = SA._store_facts(comp)
    emb = SA._word_embedding(seed, vocab)
    W_in = SA.make_fm_projection(seed, SA.FM_N_POOL, SA.FM_LOOP_IN_DIM)
    fm = SA.build_fm_world_model(bridge, xp, idx, baseline_snap, comp, facts, emb, W_in, seed)
    track.sample()
    mouth = SA._load_generator_mouth(seed, facts, device=device)
    out["integrated_build_s"] = round(time.time() - t0, 3)
    out["integrated_n_neurons"] = int(bridge.core_config.num_neurons)
    out["mouth_spiking_ops_enabled"] = bool(mouth["spiking_ops_enabled"])
    track.sample()

    # ONE representative real chat turn: drive the in-domain turn 3 ("Tell me about the dog") through the REAL
    # `run_conversation` driver (a single-turn slice) so the latency reflects the real integrated per-turn cost.
    faculty_rng = FacultyRNG(seed, ["moat", "honesty", "arbiter", "affect", "curiosity"])
    one_turn = [TT.HUMAN_TURNS[2]]                                    # ("Let's talk about ... the dog.", ...)
    orig = TT.HUMAN_TURNS
    try:
        TT.HUMAN_TURNS = one_turn
        t0 = time.time()
        TT.run_conversation(bridge, xp, idx, baseline_snap, comp, facts, fm, mouth, faculty_rng, episodic_mem=None)
        out["per_turn_latency_s"] = round(time.time() - t0, 3)
    finally:
        TT.HUMAN_TURNS = orig
    track.sample()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Integrated-brain episodic/ideation organ scale probe (n_ca3 400 vs 2000).")
    ap.add_argument("--n-ca3", type=str, default="400,2000",
                    help="comma list of arm scales. An arm == 2000 uses EMERGENT DG-selected membership (production); "
                         "any other value uses PRE-ASSIGNED membership at that n_ca3 (the de-risk stand-in scale).")
    ap.add_argument("--n-mem", type=int, default=3, help="number of stored assemblies/topics (>=2 for the blend read).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emergent-at", type=int, default=2000, help="the n_ca3 value that triggers the emergent path.")
    ap.add_argument("--no-emergent", action="store_true", help="force PRE-ASSIGNED membership for every arm (smoke).")
    ap.add_argument("--with-integrated", action="store_true",
                    help="ALSO build build_one_brain + the spiking-Qwen mouth and drive one real HUMAN_TURN (GPU).")
    ap.add_argument("--device", type=str, default="cuda", help="the mouth device for --with-integrated.")
    ap.add_argument("--smoke", action="store_true", help="reduce BTSP/step budgets for a fast CPU correctness smoke.")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_integrated_brain_scale/scale_probe.json")
    args = ap.parse_args()

    from sim.backend import get_backend
    _, backend_name = get_backend()
    arms = [int(x) for x in str(args.n_ca3).split(",") if x.strip()]
    print(f"[scale-probe] backend={backend_name} arms(n_ca3)={arms} n_mem={args.n_mem} seed={args.seed} "
          f"with_integrated={args.with_integrated} smoke={args.smoke}", flush=True)

    results = []
    for arm in arms:
        emergent = (arm == args.emergent_at) and (not args.no_emergent)
        print(f"\n[scale-probe] === ARM n_ca3={arm} emergent={emergent} ===", flush=True)
        track = _PeakTracker()
        track.sample()
        rec, _p_eff = _measure_organ_arm(arm, seed=args.seed, n_mem=args.n_mem, emergent=emergent,
                                         smoke=args.smoke, track=track)
        if args.with_integrated:
            try:
                rec.update(_measure_integrated(arm, seed=args.seed, device=args.device, track=track))
            except Exception as e:
                rec["integrated_error"] = repr(e)
                print(f"[scale-probe] WARNING integrated build/turn FAILED: {e!r}", flush=True)
        else:
            rec["backend_mouth"] = "(not built; --with-integrated off)"
            rec["per_turn_latency_s"] = None

        # ── resource fields (VRAM headline on cupy; CPU RSS for completeness; 0/NA HONESTLY when not applicable) ──
        rec["peak_rss_mb"] = round(_rss_peak_mb(), 1)               # process-lifetime peak (whole process)
        rec["arm_rss_now_mb"] = round(track.rss_now, 1) if track.rss_now else None
        rec["cupy_pool_total_mb"] = round(track.cupy_total, 1) if track.have_cupy else None    # substrate VRAM
        rec["cupy_pool_used_mb"] = round(track.cupy_used, 1) if track.have_cupy else None
        rec["torch_reserved_mb"] = round(track.torch_reserved, 1) if track.have_torch else None  # mouth VRAM
        # HEADLINE this-process VRAM = cupy substrate pool + torch (mouth) allocator. 0.0 on the numpy substrate with
        # no mouth (substrate on CPU -> no VRAM), reported HONESTLY. nvidia-smi is whole-GPU context (other procs).
        rec["peak_vram_mb"] = round(track.cupy_total + track.torch_reserved, 1)
        rec["nvidia_smi_used_mb"] = round(track.nvsmi, 1) if track.nvsmi else None   # whole-GPU context
        results.append(rec)
        print(f"[scale-probe] arm n_ca3={arm}: actual_n_ca3={rec['actual_n_ca3']} "
              f"organ_build_s={rec['organ_build_s']} read_s={rec['organ_recall_read_s']} "
              f"peak_vram_mb={rec['peak_vram_mb']} peak_rss_mb={rec['peak_rss_mb']}", flush=True)
        _cupy_free_pool()

    # ── ANTI-CHEAT: the scale knob must ACTUALLY have changed the substrate CA3 count across arms ─────────────
    anti = {"valid": True, "notes": []}
    if len(results) >= 2:
        counts = [r["actual_n_ca3"] for r in results]
        req = [r["arm_n_ca3"] for r in results]
        if len(set(counts)) < len(counts):
            anti["valid"] = False
            anti["notes"].append(
                f"INVALID: arms report identical actual_n_ca3 {counts} for requested {req} -- the scale knob did NOT "
                f"engage the substrate; any 'no resource difference' would be an ARTIFACT, not a result.")
        else:
            got = counts[-1] / max(1, counts[0])
            want = req[-1] / max(1, req[0])
            anti["actual_n_ca3_ratio"] = round(got, 3)
            anti["requested_ratio"] = round(want, 3)
            if abs(got - want) / max(want, 1e-9) > 0.25:
                anti["notes"].append(
                    f"WARNING: actual CA3-count ratio {got:.2f} deviates >25% from the requested {want:.2f} "
                    f"(emergent selection may return a different n_ca3 than the label -- inspect actual_n_ca3).")
            else:
                anti["notes"].append(
                    f"OK: substrate CA3 count grew {counts[0]}->{counts[-1]} (ratio {got:.2f} ~ requested {want:.2f}).")
    else:
        anti["notes"].append("single arm -- cross-arm anti-cheat not applicable (need >=2 arms).")

    payload = {
        "runner": "research/runners/_integrated_brain_scale_probe.py",
        "backend_substrate": backend_name,
        "seed": int(args.seed),
        "n_mem": int(args.n_mem),
        "with_integrated": bool(args.with_integrated),
        "smoke": bool(args.smoke),
        "arms": results,
        "anti_cheat": anti,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n[scale-probe] anti_cheat: valid={anti['valid']} {anti['notes']}", flush=True)
    print(f"[scale-probe] wrote {args.out}", flush=True)
    if not anti["valid"]:
        print("[scale-probe] ⛔ ANTI-CHEAT FAILED -- the scale knob did not change the substrate; results are INVALID.",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
