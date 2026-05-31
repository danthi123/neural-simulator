"""In-substrate spiking BIND/UNBIND on real substrate concept codes -- the decisive
demonstration that the validated VSA composition runs IN the spiking substrate (not
numpy algebra on captured codes). This is the design doc's cheap-first gate for the
full build (docs/plans/2026-05-31-spiking-composition-integration-design.md).

Architecture (one bridge, 8D neurons; D = projected concept-code dim):
  role_ON  [0,D)    role_OFF [D,2D)    fill_ON [2D,3D)   fill_OFF [3D,4D)   <- driven sources
  coincA [4D,5D) = AND(role_ON , fill_ON )   -> bound_ON term
  coincB [5D,6D) = AND(role_OFF, fill_OFF)   -> bound_ON term
  coincC [6D,7D) = AND(role_ON , fill_OFF)   -> bound_OFF term
  coincD [7D,8D) = AND(role_OFF, fill_ON )   -> bound_OFF term
Each coinc neuron has 2 synaptic inputs (weight w) + a tonic bias (the validated
coincidence operating point). bound_ON[i]=rate(A[i])+rate(B[i]); bound_OFF=rate(C)+rate(D).
This EXACTLY computes (role (x) filler)_ON/_OFF in spiking rates (verified algebra).

BIND: drive role_ON/OFF from role r (binary +-1), fill_ON/OFF from concept c (graded
ON/OFF) -> read bound_ON/OFF. UNBIND: reuse the SAME coincidence layer with role := query
q and fill := the stored bound rates -> est_ON/OFF; est = est_ON - est_OFF = q (x) bound.
For q = r, est = c (recovered). SUPERPOSITION: sum bound rates across K pairs (linear
memory), then unbind each role.

CLEANUP: argmax_k concept_k . est (the substrate is ID-separable; needs no near-ortho).

FROZEN bar: spiking recovery (correct query -> right concept) >= 0.80 multi-pair, AND the
no-binding control (wrong query) at chance (~1/V). A pure-numpy reference at the same D is
the ceiling (isolates projection/cleanup from spiking loss). Three-state:
RESOLVES (>=0.80 + control chance) / BOUNDARY (works K=1 but degrades) / DOES-NOT-RESOLVE.

GPU/CuPy real run. Reuses the project bridge by import; no protected-module modification.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
# coincidence operating point (validated in _insubstrate_coincidence_probe / _graded_gating)
W_COINC = 320.0
COINC_BIAS = -1000.0
ROLE_DRIVE = 2500.0      # binary role source drive (active bit)
FILL_DRIVE = 2500.0      # max graded fill source drive (scaled by concept magnitude)
RESET_STEPS = 20
RUN_STEPS = 60


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed, D, rng):
    """Load substrate concept codes, project to D (random Gaussian; preserves cosines),
    mean-center + unit-normalize. Returns (words, codes[V,D])."""
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)  # [V, 3200]
    P = rng.standard_normal((raw.shape[1], D)) / np.sqrt(raw.shape[1])
    proj = raw @ P
    codes = np.stack([_center(proj[i]) for i in range(proj.shape[0])])
    return ws, codes


def make_roles(R, D, rng):
    r = rng.choice([-1.0, 1.0], size=(R, D))
    return r   # +-1 distributed roles (ON/OFF realizable)


def onoff(vec):
    """signed vector -> (ON, OFF) non-negative parts."""
    return np.maximum(vec, 0.0), np.maximum(-vec, 0.0)


def build(seed, D, xp):
    cfg = CoreSimConfig()
    cfg.num_neurons = 8 * D
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    role_on = np.arange(0, D); role_off = np.arange(D, 2 * D)
    fill_on = np.arange(2 * D, 3 * D); fill_off = np.arange(3 * D, 4 * D)
    A = np.arange(4 * D, 5 * D); B = np.arange(5 * D, 6 * D)
    C = np.arange(6 * D, 7 * D); Dd = np.arange(7 * D, 8 * D)
    pre, post = [], []
    # A=AND(role_on,fill_on); B=AND(role_off,fill_off); C=AND(role_on,fill_off); D=AND(role_off,fill_on)
    for src1, src2, dst in ((role_on, fill_on, A), (role_off, fill_off, B),
                            (role_on, fill_off, C), (role_off, fill_on, Dd)):
        for i in range(D):
            pre.append(int(src1[i])); post.append(int(dst[i]))
            pre.append(int(src2[i])); post.append(int(dst[i]))
    w = np.full(len(pre), W_COINC, dtype=np.float32)
    plan = {"bind": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                     "plastic": False, "conn_type": "E_TO_E", "count": len(pre)}}
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.inject_explicit_wiring(plan)
    idx = dict(role_on=role_on, role_off=role_off, fill_on=fill_on, fill_off=fill_off,
               A=A, B=B, C=C, D=Dd)
    return bridge, {k: xp.asarray(v, dtype=xp.int64) for k, v in idx.items()}


def hadamard_spiking(bridge, idx, role_vec, fill_on_cur, fill_off_cur, D, xp):
    """One spiking (x) operation: drive role (binary +-1) + fill (graded ON/OFF currents);
    read coincidence banks. Returns (out_on, out_off) D-vectors of coinc rates."""
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(8 * D, dtype=xp.float32)
    r_on = xp.asarray((role_vec > 0).astype(np.float32) * ROLE_DRIVE)
    r_off = xp.asarray((role_vec < 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["role_on"]] = r_on
    cur[idx["role_off"]] = r_off
    cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = COINC_BIAS
    bridge.cp_external_input_current[:] = cur
    acc = {b: xp.zeros(D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
    for _ in range(RUN_STEPS):
        bridge._run_one_simulation_step()
        for b in ("A", "B", "C", "D"):
            acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    rates = {b: to_host(acc[b]) / RUN_STEPS for b in ("A", "B", "C", "D")}
    out_on = rates["A"] + rates["B"]     # bound_ON / est_ON
    out_off = rates["C"] + rates["D"]    # bound_OFF / est_OFF
    return out_on, out_off


def _scale_to_current(on, off, drive):
    """map non-negative ON/OFF arrays to source currents with max -> drive."""
    m = max(on.max(), off.max(), 1e-9)
    return on / m * drive, off / m * drive


def _wrong_role(roles, true_i, rng):
    """pick a role index != true_i (rigorous wrong-query control)."""
    j = rng.integers(roles.shape[0])
    while j == true_i:
        j = rng.integers(roles.shape[0])
    return roles[j]


def numpy_reference(codes, roles, K, V, rng, n_trials):
    """pure-numpy bind/unbind/cleanup ceiling at this D."""
    correct = total = ctrl = 0
    for _ in range(n_trials):
        fi = rng.choice(V, size=K, replace=False)
        ri = rng.choice(roles.shape[0], size=K, replace=False)
        S = np.zeros(codes.shape[1])
        for k in range(K):
            S = S + roles[ri[k]] * codes[fi[k]]
        for k in range(K):
            est = S * roles[ri[k]]
            correct += int(int(np.argmax(codes @ est)) == fi[k]); total += 1
            wrong = _wrong_role(roles, ri[k], rng)
            ctrl += int(int(np.argmax(codes @ (S * wrong))) == fi[k])
    return correct / total, ctrl / total


def run_spiking(bridge, idx, codes, roles, K, V, D, xp, rng, n_trials, opponency=True):
    correct = total = ctrl = 0
    for _ in range(n_trials):
        fi = rng.choice(V, size=K, replace=False)
        ri = rng.choice(roles.shape[0], size=K, replace=False)
        # BIND each pair in spiking, accumulate bound rates (superposition = linear sum)
        bound_on = np.zeros(D); bound_off = np.zeros(D)
        for k in range(K):
            c_on, c_off = onoff(codes[fi[k]])
            fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
            b_on, b_off = hadamard_spiking(bridge, idx, roles[ri[k]], fon, foff, D, xp)
            bound_on += b_on; bound_off += b_off
        # ON/OFF opponency (lateral inhibition / common-mode removal -- biological): the
        # readout is the signed DIFFERENCE, so re-canonicalize the superposed bound to its
        # signed form before unbind, freeing dynamic range from the redundant common-mode
        # (retinal/thalamic ON-OFF opponency; the project's mean-centering motif). Linear,
        # in-substrate-realizable (mutual inhibition bound_ON[i]<->bound_OFF[i]).
        if opponency:
            bsig = bound_on - bound_off
            bound_on, bound_off = onoff(bsig)
        # UNBIND each role from the superposed bound (reuse same coincidence layer)
        fon, foff = _scale_to_current(bound_on, bound_off, FILL_DRIVE)
        for k in range(K):
            e_on, e_off = hadamard_spiking(bridge, idx, roles[ri[k]], fon, foff, D, xp)
            est = e_on - e_off
            correct += int(int(np.argmax(codes @ est)) == fi[k]); total += 1
            wrong = _wrong_role(roles, ri[k], rng)
            e_on2, e_off2 = hadamard_spiking(bridge, idx, wrong, fon, foff, D, xp)
            ctrl += int(int(np.argmax(codes @ (e_on2 - e_off2)) == fi[k]))
    return correct / total, ctrl / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--ks", type=str, default="1,2,3")
    ap.add_argument("--no-opponency", action="store_true",
                    help="disable ON/OFF common-mode removal before unbind (raw superposition)")
    a = ap.parse_args()
    opponency = not a.no_opponency
    if not os.path.exists(CACHE % a.seed):
        print("CANNOT-CONCLUDE (no cache)"); return
    xp, backend = get_backend()
    rng = np.random.default_rng(a.seed)
    D = a.proj_dim
    words, codes = load_concepts(a.seed, D, rng)
    V = len(words)
    roles = make_roles(8, D, rng)
    print(f"=== in-substrate spiking BIND/UNBIND (backend={backend}, seed={a.seed}, "
          f"D={D}, V={V}, 8D={8*D} neurons, opponency={opponency}) ===", flush=True)
    print(f"chance = 1/{V} = {1.0/V:.3f}; FROZEN bar: spiking recovery >= 0.80, control ~chance", flush=True)
    bridge, idx = build(a.seed, D, xp)
    ks = [int(x) for x in a.ks.split(",")]
    rows = []
    for K in ks:
        nref, cref = numpy_reference(codes, roles, K, V, np.random.default_rng(1000 + K), a.n_trials)
        nspk, cspk = run_spiking(bridge, idx, codes, roles, K, V, D, xp,
                                 np.random.default_rng(2000 + K), a.n_trials, opponency=opponency)
        rows.append((K, nref, nspk, cspk))
        print(f"  K={K} | numpy-ref recovery={nref:.3f} | SPIKING recovery={nspk:.3f} "
              f"control(wrong-query)={cspk:.3f}", flush=True)
    print("\nREAD: spiking recovery >= 0.80 AND control ~ chance -> the spiking bind/unbind "
          "reproduces the validated algebra IN the substrate.", flush=True)
    multi = [r for r in rows if r[0] >= 2]
    ok = all(r[2] >= 0.80 for r in multi) and all(r[3] <= 2.0 / V + 0.1 for r in multi) and len(multi) > 0
    k1 = [r for r in rows if r[0] == 1]
    if ok:
        print("VERDICT: RESOLVES -- spiking bind/unbind recovers >=0.80 multi-pair, control at chance.")
    elif k1 and k1[0][2] >= 0.80:
        print("VERDICT: BOUNDARY -- works at K=1 but degrades with load; characterize SNR/scaling.")
    else:
        print("VERDICT: needs tuning/DOES-NOT-RESOLVE -- inspect numpy-ref (ceiling) vs spiking gap.")


if __name__ == "__main__":
    main()
