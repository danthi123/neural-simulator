"""gap#2 Rank-1 MECHANISM — the LEARNED (local-Hebbian/delta) fast-weight binder, realized ON THE SPIKING RF substrate,
multi-bind retrieve@P over the 788 correlated stream-cortex phasor codes. The emergence-bar close: replace the
composer's FIXED-FHRR exact-inverse bind with a J WRITTEN by a LOCAL outer-product rule (no backprop, no transport),
and READ it via the RF resonate loop (the committed spiking dynamics) — the gate's AKOrN energy-binder primitive on
`cp_rf_w_re/im`.

Per-fact fast-weight W (D x D complex), written by a LOCAL rule over (role-key, filler-value) phasor pairs:
  delta (error-correcting):  W += (v - W k) k^H     (erase-then-write)
  additive (Hebbian):        W += v k^H             (accumulates crosstalk -> the P>=2 collapse)
W is installed as the FULL RF coupling (out[k] <- in[j] : W[k,j]); kick the role key on the input block, run the RF
resonate loop, read the output-block phases -> cleanup (nearest concept). ONE VARIABLE: write-rule delta ON vs OFF
(OFF=additive) vs the fixed-FHRR baseline. GATE: delta reaches the ceiling (retrieve>=0.80 at P>=3) on SPIKES AND beats
additive (delta load-bearing). Anti-cheats: permuted-role -> chance; decorrelated-code control (op works there too);
the additive arm must REPRODUCE the P>=2 collapse. `--seeds`, `--pmax`, `--n-facts`.
"""
import argparse
import glob
import numpy as np

from sim.backend import to_host
from research.runners.rf_phasor_composer import _build_rf_bridge


def load_phasor_codes(cap=None):
    path = sorted(glob.glob("bridges/developed/scale787/day_*/grounded_codes.npz"))[-1]
    z = np.load(path, allow_pickle=True)
    keys = [k for k in z.keys() if k.startswith("g:")]
    theta = np.stack([np.asarray(z[k], np.float64) for k in keys])
    if cap:
        theta = theta[:cap]
    return np.exp(2j * np.pi * theta)                     # (N, D) unit phasors


def build_W(role_ph, filler_ph, roleids, fillerids, D, delta):
    """Per-fact fast-weight W (D x D) written by a LOCAL outer-product rule over the (role,filler) phasor pairs."""
    W = np.zeros((D, D), complex)
    for r, f in zip(roleids, fillerids):
        k = role_ph[r]; v = filler_ph[f]
        if delta:
            # error-correcting erase-then-write; normalize by <k,k>=D (the keys are D-dim UNIT PHASORS with norm^2=D,
            # NOT unit-norm) so the step does not overshoot 128x. Local: pre=k, post=(v-Wk)/D.
            W = W + np.outer(v - W @ k, k.conj()) / D
        else:
            W = W + np.outer(v, k.conj()) / D             # plain Hebbian outer product (same 1/D scale)
    return W


def spiking_read(bridge, W, key, D):
    """Install W as the FULL RF coupling (out[k]<-in[j]:W[k,j]), kick `key` on the input block, resonate, read the
    output-block phases -> the recovered phasor (its phase)."""
    conns = [(D + k, j, complex(W[k, j])) for k in range(D) for j in range(D) if W[k, j] != 0]
    bridge.rf_set_complex_weights(conns)
    kick = np.zeros(2 * D, complex); kick[:D] = key
    bridge.rf_kick(kick, period=200, lam=0.0)
    bridge.rf_resonate_steps(208)
    ph = np.asarray(to_host(bridge.rf_read_phases()))
    return np.exp(2j * np.pi * ph[D:2 * D])               # output block -> recovered phasor


def cleanup(u, Z):
    return int(np.argmax(np.abs(Z.conj() @ u)))


def retrieve(Z, bridge, seed, P, n_facts, D, delta, permute=False, decorrelate=False):
    rng = np.random.default_rng(seed * 137 + P)
    N = Z.shape[0]
    ZB = np.exp(2j * np.pi * rng.random((N, D))) if decorrelate else Z
    roles = np.exp(2j * np.pi * rng.random((P, D)))
    n_ok = n = 0
    for _ in range(n_facts):
        fids = rng.choice(N, P, replace=False)
        W = build_W(roles, ZB, list(range(P)), list(fids), D, delta)
        for i in range(P):
            key = roles[(i + 1) % P] if (permute and P > 1) else roles[i]
            rec = spiking_read(bridge, W, key, D)
            n_ok += int(cleanup(rec, ZB) == fids[i]); n += 1
    return n_ok / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--pmax", type=int, default=4)
    ap.add_argument("--n-facts", type=int, default=20)
    ap.add_argument("--cap", type=int, default=256)
    args = ap.parse_args()

    Z = load_phasor_codes(cap=args.cap); N, D = Z.shape
    bridge = _build_rf_bridge(2 * D, seed=args.seeds[0])
    print(f"[gap2 SPIKING learned binder] codes={N} D={D} on RF substrate (2D={2*D} neurons) | seeds={args.seeds}")
    for P in range(1, args.pmax + 1):
        de = np.mean([retrieve(Z, bridge, s, P, args.n_facts, D, delta=True) for s in args.seeds])
        ad = np.mean([retrieve(Z, bridge, s, P, args.n_facts, D, delta=False) for s in args.seeds])
        pm = np.mean([retrieve(Z, bridge, s, P, args.n_facts, D, delta=True, permute=True) for s in args.seeds]) if P > 1 else 0.0
        dc = np.mean([retrieve(Z, bridge, s, P, args.n_facts, D, delta=True, decorrelate=True) for s in args.seeds])
        print(f"  P={P}: DELTA {de:.3f} | additive {ad:.3f} | permuted-role {pm:.3f} | decorrelated-ctrl {dc:.3f}"
              f"{'  <-- gate: delta>=0.80 & delta>additive' if P >= 3 else ''}")
    print("  => the LEARNED local-J binder READ on the SPIKING RF resonate loop; delta reaching the ceiling at P>=3 "
          "while additive collapses = the emergence-bar spiking multi-bind close.")


if __name__ == "__main__":
    main()
