"""PAST-RESERVOIR Rung 4a (the spiking-realization ladder, step 1): the per-neuron selective diagonal SSM realized with a
BIOLOGICAL CONDUCTANCE-BASED input-modulated leak (shunting), not the abstract sigmoid gate — does the selective advantage
(Rung 2) SURVIVE the conductance-shunt dynamics that a real spiking neuron uses? This is the cheap-first bridge-FAITHFUL
rung before wiring the actual `SimulationBridge` (Rung 4b): a conductance-based leaky integrator's effective leak is set by
its total conductance, and an INPUT-DRIVEN SHUNTING conductance (g_syn to rest) is exactly the input-modulated leak = the
selective lambda. NO `sim/` edit; self-contained numpy.

MECHANISM (bridge-faithful; the mapping derived + verified):
  conductance-based leaky integrator:  V_{t,i} = V_{t-1,i}*(1 - k*(1 + g_shunt_{t,i})) + k*inj_{t,i}
  input-driven shunt (the gate):        g_shunt_{t,i} = softplus(w_i . u_t + c_i)   (>=0 conductance; more input -> more
                                                                                     leak -> LOWER retention lambda_eff)
  effective retention:                  lambda_eff_{t,i} = 1 - k*(1 + g_shunt_{t,i})   (clipped to [0,1))
  EXACT forward-mode eligibility (local; softplus' = sigmoid):
     dlam/dw_i = -k * sigmoid(a_i) * u_t ,   e^w = lambda_eff*e^w + (dlam/dw)*(V_prev - inj) ,   Δtheta ∝ -delta*e
  bias init so lambda_eff starts ~0.9 (hold): a small g_shunt at init (c chosen so softplus(c)~small).

TASK + ARMS: identical to Rung 2 ([KEY, filler×12, QUERY]->rule[KEY,QUERY]); selective (conductance-shunt gate TRAINED) vs
fixed_res (fixed leak) vs detached (shunt gate UNTRAINED) vs randgate (shunt gate on a RANDOM input). GO iff selective
beats all three + chance on >=5/6 -- i.e. the selective long-range conjunction SURVIVES the biological conductance-based
leak, green-lighting the on-bridge Rung 4b.

Run: python -m research.runners._reslm_rung4_conductance_shunt_ssm_derisk --seeds 42
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np

K = 6
D_IN = 10
N_HID = 64
DEPTH = 12
N_SEQ = 900
EPOCHS = 10
LR_RO = 0.05
LR_GATE = 0.4                 # the gate gradient is K_LEAK-scaled (small) -> compensate with a larger gate LR (the leak
                              # sensitivity dlam/da = -K_LEAK*sigmoid is ~K_LEAK smaller than the abstract-gate form)
K_LEAK = 0.06                 # base leak rate (dt/tau); small -> lambda_eff ~0.92 at init so the state HOLDS (forget-bias
                              # equivalent); the input-driven SHUNT can INCREASE the leak (release)
C_INIT = -1.2                 # gate bias -> softplus(-1.2)~0.26 shunt at init -> lambda_eff ~ 1 - 0.06*1.26 ~ 0.92 (hold)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)      # numerically-stable softplus


def _embed(seed):
    return np.random.default_rng(seed * 3 + 1).standard_normal((K + 1, D_IN)) * 0.8


def _make_seqs(seed):
    rng = np.random.default_rng(seed * 11 + 5)
    rule = rng.integers(0, K, (K, K))
    seqs = []
    for _ in range(N_SEQ):
        k = int(rng.integers(0, K)); q = int(rng.integers(0, K))
        seqs.append(([k] + [K] * DEPTH + [q], int(rule[k, q])))       # [KEY, filler×DEPTH, QUERY] -> rule[KEY,QUERY]
    return seqs


def _params(seed):
    rng = np.random.default_rng(seed * 7 + 2)
    Win = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    w = rng.standard_normal((N_HID, D_IN)) / np.sqrt(D_IN)
    c = np.full(N_HID, C_INIT)
    fixed_gsh = _softplus(rng.standard_normal(N_HID) * 0.3 + C_INIT)              # fixed per-neuron shunt (leaky ESN)
    return Win, w, c, fixed_gsh


def _run_arm(seed, arm):
    E = _embed(seed); Win, w, c, fixed_gsh = _params(seed)
    seqs = _make_seqs(seed)
    permrng = np.random.default_rng(seed * 101 + 7)
    train_gate = (arm in ("selective", "randgate"))
    Wro = np.zeros((K, N_HID))
    for _ep in range(EPOCHS):
        for (toks, y) in seqs[:int(0.7 * len(seqs))]:
            V = np.zeros(N_HID); ew = np.zeros((N_HID, D_IN)); ec = np.zeros(N_HID)
            for t, tok in enumerate(toks):
                u = E[tok]; inj = Win @ u
                ug = E[int(permrng.integers(K + 1))] if arm == "randgate" else u
                if arm == "fixed_res":
                    gsh = fixed_gsh; a = None
                else:
                    a = w @ ug + c; gsh = _softplus(a)
                lam = np.clip(1.0 - K_LEAK * (1.0 + gsh), 0.0, 0.999)               # conductance-based effective retention
                V_prev = V; V = lam * V_prev + (1.0 - lam) * inj
                if train_gate:
                    dlam_da = -K_LEAK * _sigmoid(a)                                  # d lambda_eff / d a  (softplus'=sigmoid)
                    base = (V_prev - inj) * dlam_da
                    ew = lam[:, None] * ew + base[:, None] * ug[None, :]
                    ec = lam * ec + base
                z = Wro @ V; z -= z.max(); p = np.exp(z); p /= p.sum()
                err = p.copy(); err[y] -= 1.0 if t == len(toks) - 1 else 0.0      # target only at the read (query) step
                if t == len(toks) - 1:
                    delta = Wro.T @ err
                    Wro -= LR_RO * np.outer(err, V)
                    if train_gate:
                        w -= LR_GATE * (delta[:, None] * ew)
                        c -= LR_GATE * (delta * ec)
    cor = tot = 0
    for (toks, y) in seqs[int(0.7 * len(seqs)):]:
        V = np.zeros(N_HID)
        for t, tok in enumerate(toks):
            u = E[tok]; inj = Win @ u
            ug = E[int(permrng.integers(K + 1))] if arm == "randgate" else u
            gsh = fixed_gsh if arm == "fixed_res" else _softplus(w @ ug + c)
            lam = np.clip(1.0 - K_LEAK * (1.0 + gsh), 0.0, 0.999)
            V = lam * V + (1.0 - lam) * inj
        cor += int(np.argmax(Wro @ V) == y); tot += 1
    return cor / tot


def run(seed):
    acc = {a: _run_arm(seed, a) for a in ("selective", "fixed_res", "detached", "randgate")}
    chance = 1.0 / K
    go = bool(acc["selective"] > acc["fixed_res"] + 0.08 and acc["selective"] > acc["detached"] + 0.05
              and acc["selective"] > acc["randgate"] + 0.05 and acc["selective"] > chance + 0.12)
    print(f"[rung4 seed={seed}] selective={acc['selective']:.3f} fixed_res={acc['fixed_res']:.3f} "
          f"detached={acc['detached']:.3f} randgate={acc['randgate']:.3f} (chance={chance:.3f}) "
          f"-> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, **acc, "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s) for s in a.seeds]
    print(f"[rung4] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
