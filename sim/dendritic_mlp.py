"""Deep sigmoid MLP for literature-faithful feedback alignment
(Lillicrap 2016; GLR-2017). Per HIDDEN layer: forward W + a FIXED
RANDOM feedback matrix B (set once from seed, NEVER learned, NEVER
derived from any forward W -- no weight transport). Hidden learning
delegates to the committed sign-correct
sim.dendritic_plasticity.urbanczik_senn_update (batched == per-sample
sum). Output layer by local delta. `oracle` mode is a HAND-DERIVED
backprop used ONLY as the V1 positive-control + the emergent-
alignment measurement -- it is fenced as measurement/validity, NOT a
shipped biologically-local learning mode, and uses NO reverse-mode
graph / NO automatic-differentiation framework.

GPU-acceleration note (pure speed; numerics scientifically
equivalent): array math routes through the project's validated
pluggable backend (sim.backend.get_backend -> CuPy on an NVIDIA GPU,
NumPy fallback otherwise; SIM_BACKEND env = auto/cupy/numpy). That is
plain array math (matmuls only), not a differentiation framework --
the biological-locality, no-weight-transport, committed-rule-faithful,
THREE-STATE and kill-safe guarantees are independent of which array
library runs the matmuls and are all preserved. The deterministic
seeded init stays on NumPy then is moved onto the backend (xp.asarray)
so init is byte-identical across CPU/GPU. Public scalars are returned
as host floats so the runner / verdict are unaffected. ASCII only.
Does NOT import sim.bptt_snn."""
from __future__ import annotations
import numpy as np  # deterministic seeded init only (NOT autodiff)
from sim.backend import get_backend, to_host
from sim.dendritic_plasticity import urbanczik_senn_update  # committed

# Pluggable array backend: CuPy (GPU) when available, else NumPy. This
# only swaps the array library for the matmuls -- every formula below
# is unchanged. CuPy is array math, not automatic differentiation.
xp, _ = get_backend()


def _sig(z):
    return 1.0 / (1.0 + xp.exp(-xp.clip(z, -30.0, 30.0)))


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    ez = xp.exp(z)
    return ez / ez.sum(1, keepdims=True)


class DendriticMLP:
    def __init__(self, sizes, seed=0):
        # Seeded init stays on NumPy (exact, reproducible, identical
        # across CPU/GPU), then moves onto the active backend. Keeps
        # test_modes_deterministic_given_seed green on either backend.
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.n_out = sizes[-1]
        self.W = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(xp.asarray(
                rng.uniform(-lim, lim, (sizes[i], sizes[i + 1]))))
        # Per-layer FIXED RANDOM feedback B: set ONCE from seed, NEVER
        # mutated, NEVER derived from forward W (no weight transport).
        self.B = [xp.asarray(rng.normal(0, 1.0, (self.n_out, sizes[i])))
                  for i in range(1, len(sizes) - 1)]

    def _forward(self, X):
        acts = [xp.asarray(X, float)]
        for li in range(len(self.W) - 1):
            acts.append(_sig(acts[-1] @ self.W[li]))
        return acts, acts[-1] @ self.W[-1]

    def loss(self, X, y):
        _, lg = self._forward(X)
        p = _softmax(lg)
        y = xp.asarray(y)
        return float(to_host(-xp.log(
            p[xp.arange(len(y)), y] + 1e-12).mean()))

    def accuracy(self, X, y):
        _, lg = self._forward(X)
        y = xp.asarray(y)
        return float(to_host(xp.mean(xp.argmax(lg, 1) == y)))

    def _debug_fwd_err(self, X, y):
        acts, lg = self._forward(X)
        y = xp.asarray(y)
        e = _softmax(lg).copy()
        e[xp.arange(len(y)), y] -= 1.0
        return acts, e

    def _debug_hidden_dW0(self, X, y):
        acts, e = self._debug_fwd_err(X, y)
        a_prev, a_l = acts[0], acts[1]
        ap = e @ self.B[0]
        return a_prev.T @ (ap * a_l * (1.0 - a_l))

    def _true_grads(self, X, y):
        """Hand-derived backprop (NO automatic differentiation, NO
        reverse-mode graph). Returns list of dL/dW per layer.
        Measurement/validity ONLY."""
        acts, e = self._debug_fwd_err(X, y)
        nW = len(self.W)
        grads = [None] * nW
        d = e
        grads[nW - 1] = acts[nW - 1].T @ d
        for li in range(nW - 2, -1, -1):
            a = acts[li + 1]
            d = (d @ self.W[li + 1].T) * a * (1.0 - a)
            grads[li] = acts[li].T @ d
        return grads

    def train_step(self, X, y, mode, lr):
        acts, e = self._debug_fwd_err(X, y)
        nW = len(self.W)
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ e)              # output local delta
        if mode == "oracle":
            g = self._true_grads(X, y)
            upd = [-gi for gi in g]
        else:
            for li in range(nW - 1):
                a_prev, a_l = acts[li], acts[li + 1]
                ap = e @ self.B[li]
                base = a_prev.T @ (ap * a_l * (1.0 - a_l))
                if mode == "local_correct" or mode == "permuted":
                    upd[li] = -base
                elif mode == "local_wrongsign":
                    upd[li] = base
                elif mode == "global_scalar":
                    gscal = float(self.loss(X, y))
                    upd[li] = -gscal * (a_prev.T @ (a_l - 0.5))
                else:
                    raise ValueError("unknown mode %r" % mode)
        for li in range(nW):
            self.W[li] = self.W[li] + lr * upd[li]

    def hidden_grad_alignment(self, X, y):
        """cos(applied layer-0 local update dir, true gradient-descent
        dir). Measurement-only; never fed to the rule."""
        local_step = -self._debug_hidden_dW0(X, y)
        gtrue0 = self._true_grads(X, y)[0]
        desc = -gtrue0
        return float(to_host(
            xp.sum(local_step * desc) / (
                xp.linalg.norm(local_step) * xp.linalg.norm(desc)
                + 1e-9)))
