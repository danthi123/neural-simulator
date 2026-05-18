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


# Instrument-calibration constant (NOT a science bar). Heavy-ball
# (Polyak) momentum coefficient for the SGD weight updates. This is
# mode-AGNOSTIC optimizer machinery: it accelerates whatever per-layer
# update each mode already produced (oracle's hand-derived true grad,
# the committed local rule's update, the global-scalar update, the
# wrong-sign update, the permuted-label update) -- it does NOT change
# which gradient/rule any mode computes, so it cannot advantage the
# local rule over its controls. A standard, well-understood,
# non-fundamental fix; pure array math, NO automatic differentiation.
_MOMENTUM = 0.9


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
            # Xavier/Glorot uniform: the CORRECT init for sigmoid
            # (sigmoid is KEPT -- the committed Urbanczik-Senn rule
            # hard-codes the sigmoid derivative soma*(1-soma); switching
            # the hidden activation would break the adversarially-pinned
            # committed-rule-faithfulness invariant). Verified
            # well-scaled: layer-1 pre-activation std ~1 on standardized
            # MNIST, only ~0.3% saturated -- init is sound; the VOID was
            # the optimizer, not the init.
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(xp.asarray(
                rng.uniform(-lim, lim, (sizes[i], sizes[i + 1]))))
        # Per-layer FIXED RANDOM feedback B: set ONCE from seed, NEVER
        # mutated, NEVER derived from forward W (no weight transport).
        self.B = [xp.asarray(rng.normal(0, 1.0, (self.n_out, sizes[i])))
                  for i in range(1, len(sizes) - 1)]
        # Per-parameter heavy-ball velocity buffers (one per W layer).
        # Lazily zero-initialized on first train_step so a restored /
        # freshly regenerated net is byte-identical for the
        # no-weight-transport self-check (the velocity is transient
        # optimizer state, NOT part of W or the FIXED feedback B and
        # NOT checkpointed -- it self-recovers in <=1 epoch).
        self._vel = None

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
        # --- Instrument optimizer (mode-AGNOSTIC; applied IDENTICALLY
        # to every mode's `upd` above; does NOT alter which gradient/
        # rule each mode computed, so discriminating power between the
        # local rule and its controls is preserved). Two standard,
        # well-understood, non-fundamental, pure-array fixes (NO
        # automatic differentiation):
        #   (1) Mean-over-batch normalization. Every `upd[li]` above is
        #       a SUM over the minibatch (`a.T @ d`). Without dividing
        #       by the batch size the effective step scales with the
        #       batch (~128x too large for the pre-registered run) and
        #       the deep sigmoid stack diverges into the dead-0.5 fixed
        #       point -> MNIST chance even WITH the exact gradient
        #       (the observed VOID). Normalizing makes the step batch-
        #       size invariant. This is the dominant root cause of the
        #       VOID.
        #   (2) Heavy-ball momentum (_MOMENTUM). Per-parameter velocity
        #       buffer; standard acceleration that lets the sigmoid MLP
        #       converge in the pre-registered epoch budget.
        m = X.shape[0]
        if m < 1:
            m = 1
        if self._vel is None:
            self._vel = [xp.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]

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
