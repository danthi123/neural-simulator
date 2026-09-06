"""PredictiveContinualSubstrate (PCS) — the AGI-fork's ONE shared learning core.

Charter: FORK.md (relaxed-constraint branch). Design:
docs/plans/2026-09-06-agi-fork-first-move-design.md section (a)+(g)#2.

WHAT THIS IS
------------
ONE shared leaky-integrator recurrent core — "the whole brain between sensation
and action" — trained ONLINE by a single next-experience predictive objective.
Faculties (place / object / permanence / value) are NOT built in; they are read
off the same population `h_t` post-hoc and must be shown load-bearing on
behavior (the anti-hollow bar, feedback_faculties_must_drive_not_observe.md).

    h_t = (1-alpha) h_{t-1} + alpha * phi( W_h h_{t-1} + W_e e_t + W_a a_{t-1} + W_d d_t + b_h )

    e_t     encoded egocentric view (learned or fixed linear compression of fixed Gabor V1)
    a_{t-1} efference copy of the last action (one-hot) — the input that makes path-integration learnable
    d_t     interoceptive drive afferent (TwoPoolDrive)

Three thin heads read the SAME h_t:
    H1  next-latent JEPA predictor    ehat_t ~ sg(EncEMA(view_{t+1}))   (predict-in-latent, NOT pixels)
    H2  reward predictor              rhat_t ~ r_t
    H3  action logits                 pi(h_t) over the moves            (trained by policy-gradient, not BPTT)

Predictive objective (H1+H2), trained by online truncated BPTT (T~16-20):
    L = mean_t || ehat_t - sg(EncEMA(view_{t+1})) ||^2  +  beta * mean_t (rhat_t - r_t)^2  +  var_lambda * L_var(e)
L_var is a VICReg-style per-dimension variance floor on e — a provable anti-collapse
guard for the learned-encoder+EMA-target (JEPA) path (a learned encoder EMA'd toward
its own prediction can collapse to a constant; the reward term opposes this weakly, the
variance floor opposes it directly).

THE --units DIAL (the fork's headline hypothesis, made measurable not asserted)
------------------------------------------------------------------------------
`units="rate"`  : phi = tanh (saturating), analytic derivative 1 - g^2.  (the primary
                  EMERGENCE arm + the Day-1 smoke.)
`units="spike"` : phi = Heaviside forward + hard reset (LIF, mirrors sim/bptt_snn.py),
                  atan_surrogate backward (sim/surrogate_grad.py's formula, reimplemented
                  backend-agnostically here because that module hard-imports cupy which
                  breaks the numpy smoke path). Heads read a low-pass rate trace of the
                  spikes so decoders get a graded signal. (the Day-5 FORK-THESIS arm.)

BRAIN-BASED / HONESTY NOTE
--------------------------
This is the RELAXED fork: rate units + truncated BPTT + EMA are ALLOWED here (FORK.md),
traded for function-first speed. Every self-report remains a FUNCTIONAL instrument
reading; nothing here asserts felt/phenomenal experience.

SEEDING (heed CLAUDE.md's trap: actual_seed_used seeds NOTHING — set the real seed).
This module owns its own RNG; `PCSConfig.seed` deterministically seeds every weight
init and the action sampler. `selftest()` builds twice at one seed and hashes W_h to
prove determinism.

Self-checks (no world needed):
    python -m sim.pcs_substrate --gradcheck   # finite-difference the manual BPTT (rate AND spike)
    python -m sim.pcs_substrate --selftest    # determinism: same seed -> identical weights
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

from sim.backend import get_backend, from_host, to_host


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PCSConfig:
    # dimensions
    n_hidden: int = 512          # shared recurrent core size (~512-1024)
    feat_dim: int = 512          # fixed Gabor-V1 feature dim fed to the encoder (set by the world)
    n_latent: int = 128          # encoded view e_t dim
    n_actions: int = 4           # 4-move body
    n_drive: int = 4             # interoceptive drive afferent dim
    # core dynamics
    alpha: float = 0.2           # leak of the rate-trace / integrator
    units: str = "rate"          # "rate" (tanh) or "spike" (LIF + surrogate)
    threshold: float = 1.0       # spike threshold (spike mode)
    leak: float = 0.95           # membrane leak exp(-dt/tau) (spike mode; matches bptt_snn)
    # objective
    tbptt_T: int = 18            # truncated-BPTT window length
    beta_reward: float = 1.0     # weight of the reward-prediction term
    var_lambda: float = 1.0      # weight of the VICReg variance floor on e (0 disables)
    var_gamma: float = 1.0       # target per-dim std for the variance floor
    # encoder
    encoder: str = "learned_ema"  # "learned_ema" (JEPA: learned enc + EMA target) or "fixed" (collapse-proof)
    ema_rate: float = 0.99       # EMA decay of the target encoder (learned_ema only)
    # optimizer
    lr: float = 1e-3             # Adam lr for the predictive objective
    grad_clip: float = 5.0       # global-norm clip on the predictive grads
    # policy (H3) — REINFORCE with a running-mean baseline, online per step
    lr_policy: float = 5e-3
    curiosity_beta: float = 0.5  # weight of learning-progress in the intrinsic reward
    entropy_beta: float = 0.01   # entropy bonus (exploration)
    baseline_decay: float = 0.99
    lp_fast: float = 0.1         # learning-progress fast/slow loss EMAs (LP = relu(slow - fast))
    lp_slow: float = 0.01
    # misc
    weight_scale: float = 0.5    # init scale for recurrent/input weights (1/sqrt fan-in applied on top)
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# small helpers (backend-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def _atan_surrogate(xp, v_minus_threshold, alpha: float = 2.0):
    """ATan surrogate gradient (Zenke 2018 / SuperSpike), reimplemented on `xp`.

    Mirrors sim/surrogate_grad.atan_surrogate exactly:
        dspike/dv = (1/pi) * (1 / (1 + (alpha*(v-thr))^2))
    That module hard-imports cupy; this backend-agnostic copy keeps the numpy
    smoke path working. Same formula, so byte-for-byte on cupy.
    """
    return (1.0 / np.pi) * (1.0 / (1.0 + (alpha * v_minus_threshold) ** 2))


class _Adam:
    """Minimal Adam over a dict of backend arrays (backend-agnostic)."""

    def __init__(self, params: Dict[str, Any], xp, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.xp = xp
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {k: xp.zeros_like(v) for k, v in params.items()}
        self.v = {k: xp.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, Any], grads: Dict[str, Any], clip: float = 0.0):
        xp = self.xp
        if clip and clip > 0:
            total = 0.0
            for k in grads:
                total = total + float((grads[k] ** 2).sum())
            norm = np.sqrt(total) + 1e-12
            if norm > clip:
                scale = clip / norm
                for k in grads:
                    grads[k] = grads[k] * scale
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            params[k] = params[k] - self.lr * mhat / (xp.sqrt(vhat) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# The substrate
# ─────────────────────────────────────────────────────────────────────────────
class PredictiveContinualSubstrate:
    """One shared predictive recurrent core + 3 heads, trained online by TBPTT.

    Online loop (one env step):
        h = sub.observe(v1feat, a_prev_idx, d)   # forward one step, returns the shared state (lesion-aware)
        a = sub.act(h)                            # sample action from pi(h) (records logprob for REINFORCE)
        ... env applies a -> reward r, next v1feat/d ...
        sub.learn(r)                              # register reward; TBPTT update at window boundary; policy step

    Probing (frozen): sub.freeze(); roll observe() without learn(); read the returned h_t.
    Lesion: sub.set_lesion_mask(bool_mask over hidden units) — zeros those units in h_t
            everywhere they are read (heads, recurrence, policy), so a lesion tests BEHAVIOR.
    """

    def __init__(self, cfg: PCSConfig):
        self.cfg = cfg
        self.xp, self.backend_name = get_backend()
        xp = self.xp
        if cfg.units not in ("rate", "spike"):
            raise ValueError(f"units must be 'rate' or 'spike', got {cfg.units!r}")
        if cfg.encoder not in ("learned_ema", "fixed"):
            raise ValueError(f"encoder must be 'learned_ema' or 'fixed', got {cfg.encoder!r}")

        # ---- deterministic init (own RNG, seeded from cfg.seed) ----
        rng = np.random.default_rng(cfg.seed)
        H, F, E, A, D = cfg.n_hidden, cfg.feat_dim, cfg.n_latent, cfg.n_actions, cfg.n_drive

        def w(shape, fan_in):
            s = cfg.weight_scale / np.sqrt(fan_in)
            return from_host(rng.standard_normal(shape).astype(np.float32) * s)

        # predictive-objective params (updated by TBPTT + Adam)
        self.P: Dict[str, Any] = {
            "W_h": w((H, H), H),
            "W_e": w((H, E), E),
            "W_a": w((H, A), A),
            "W_d": w((H, D), D),
            "b_h": xp.zeros((H,), dtype=xp.float32),
            "W_pred": w((E, H), H),   # JEPA head: h -> ehat (predict next latent)
            "b_pred": xp.zeros((E,), dtype=xp.float32),
            "w_r": w((H,), H),        # reward head: h -> rhat (scalar)
            "b_r": xp.zeros((1,), dtype=xp.float32),
        }
        # encoder (learned in learned_ema mode; fixed transducer in fixed mode)
        self.W_enc = w((E, F), F)
        self.b_enc = xp.zeros((E,), dtype=xp.float32)
        if cfg.encoder == "learned_ema":
            self.P["W_enc"] = self.W_enc
            self.P["b_enc"] = self.b_enc
            self.W_enc_ema = self.W_enc.copy()   # EMA target encoder (stop-grad target)
            self.b_enc_ema = self.b_enc.copy()
        else:  # fixed: encoder never updates, target == online (still stop-grad)
            self.W_enc_ema = self.W_enc
            self.b_enc_ema = self.b_enc

        # policy head (H3) — trained by REINFORCE, kept OUT of self.P (separate update)
        self.W_pi = w((A, H), H)
        self.b_pi = xp.zeros((A,), dtype=xp.float32)

        self.opt = _Adam(self.P, xp, lr=cfg.lr)

        # ---- runtime state ----
        self.h = xp.zeros((H,), dtype=xp.float32)          # shared state (rate-trace)
        self.v = xp.zeros((H,), dtype=xp.float32)          # membrane (spike mode)
        self.s = xp.zeros((H,), dtype=xp.float32)          # spikes (spike mode)
        self._lesion = None                                # bool mask over hidden units, or None
        self._frozen = False

        # TBPTT tape: list of per-step dicts holding the raw inputs + cached forward tensors
        self._tape: List[dict] = []

        # policy bookkeeping
        self._sampler = np.random.default_rng(cfg.seed + 10_000)
        self._last_logits = None
        self._last_probs = None
        self._last_action = None
        self._last_h_for_pi = None
        self._baseline = 0.0
        # learning-progress (LP) EMAs of the predictive loss
        self._loss_fast = None
        self._loss_slow = None
        self._last_lp = 0.0

        # diagnostics
        self.last_pred_loss = None
        self.n_updates = 0
        self.n_steps = 0

    # ── lesion / freeze API ────────────────────────────────────────────────
    def set_lesion_mask(self, mask):
        """mask: bool array (n_hidden,), True = lesioned (zeroed). Applied to h_t
        everywhere it is read — heads, recurrence, and policy — so the lesion is
        load-bearing on behavior, not decorative."""
        if mask is None:
            self._lesion = None
            return
        m = np.asarray(mask, dtype=bool)
        assert m.shape == (self.cfg.n_hidden,), m.shape
        self._lesion = from_host(m.astype(np.float32))   # 1.0 where lesioned

    def clear_lesion(self):
        self._lesion = None

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def reset_state(self):
        """Zero the recurrent state and clear the TBPTT tape (episode/probe boundary)."""
        xp = self.xp
        self.h = xp.zeros((self.cfg.n_hidden,), dtype=xp.float32)
        self.v = xp.zeros((self.cfg.n_hidden,), dtype=xp.float32)
        self.s = xp.zeros((self.cfg.n_hidden,), dtype=xp.float32)
        self._tape = []

    def _apply_lesion(self, h):
        if self._lesion is None:
            return h
        return h * (1.0 - self._lesion)

    # ── forward primitives (shared by online loop AND gradcheck) ────────────
    def _encode(self, v1feat, ema=False):
        W = self.W_enc_ema if ema else self.W_enc
        b = self.b_enc_ema if ema else self.b_enc
        return W @ v1feat + b

    def _core_forward_rate(self, h_prev, e_t, a_prev, d_t, P):
        """One rate step. Returns dict of intermediates. h_prev is ALREADY lesion-masked."""
        pre = P["W_h"] @ h_prev + P["W_e"] @ e_t + P["W_a"] @ a_prev + P["W_d"] @ d_t + P["b_h"]
        g = self.xp.tanh(pre)
        h_raw = (1.0 - self.cfg.alpha) * h_prev + self.cfg.alpha * g
        return {"pre": pre, "g": g, "h_raw": h_raw}

    def _core_forward_spike(self, v_prev, s_prev, h_prev_trace, e_t, a_prev, d_t, P):
        """One spike (LIF) step, mirroring sim/bptt_snn.py's hard-reset LIF with a
        recurrent W_h on spikes, plus a low-pass rate trace the heads read.

        v_t = leak * v_{t-1} * (1 - s_{t-1}) + (W_h s_{t-1} + W_e e_t + W_a a + W_d d + b_h)
        s_t = Heaviside(v_t - threshold)
        h_t = (1-alpha) h_{t-1} + alpha * s_t          # rate trace (what heads read)
        """
        inp = P["W_h"] @ s_prev + P["W_e"] @ e_t + P["W_a"] @ a_prev + P["W_d"] @ d_t + P["b_h"]
        v = self.cfg.leak * v_prev * (1.0 - s_prev) + inp
        s = (v >= self.cfg.threshold).astype(self.xp.float32)
        h_raw = (1.0 - self.cfg.alpha) * h_prev_trace + self.cfg.alpha * s
        return {"v": v, "s": s, "g": s, "h_raw": h_raw}

    def _heads_forward(self, h_t, P):
        ehat = P["W_pred"] @ h_t + P["b_pred"]
        rhat = P["w_r"] @ h_t + P["b_r"][0]
        return ehat, rhat

    # ── online step: observe / act / learn ──────────────────────────────────
    def observe(self, v1feat, a_prev_idx: int, d):
        """Advance the shared state one step and return h_t (lesion-aware).

        v1feat : (feat_dim,) fixed Gabor-V1 features of the CURRENT egocentric view
        a_prev_idx : int index of the last action (efference copy); -1 for "none"
        d : (n_drive,) interoceptive drive afferent
        """
        xp = self.xp
        # backend-agnostic input coercion (accepts numpy or cupy arrays / lists)
        v1feat = xp.asarray(np.asarray(to_host(v1feat), dtype=np.float32))
        d = xp.asarray(np.asarray(to_host(d), dtype=np.float32))
        a_prev = xp.zeros((self.cfg.n_actions,), dtype=xp.float32)
        if a_prev_idx is not None and a_prev_idx >= 0:
            a_prev[a_prev_idx] = 1.0

        e_t = self._encode(v1feat, ema=False)
        h_prev_masked = self._apply_lesion(self.h)
        v_prev_carry, s_prev_carry = None, None

        if self.cfg.units == "rate":
            fwd = self._core_forward_rate(h_prev_masked, e_t, a_prev, d, self.P)
            h_raw = fwd["h_raw"]
        else:
            v_prev_carry, s_prev_carry = self.v, self.s
            fwd = self._core_forward_spike(self.v, self.s, h_prev_masked, e_t, a_prev, d, self.P)
            self.v = fwd["v"]
            self.s = fwd["s"]
            h_raw = fwd["h_raw"]

        h_t = self._apply_lesion(h_raw)
        self.h = h_raw            # keep unmasked raw trace as the carried state; lesion re-applied on read
        self.n_steps += 1

        # record for TBPTT (store backend arrays; cheap references). v_prev/s_prev are the
        # (detached) membrane carried INTO this step, so the window recompute matches online exactly.
        step = {
            "v1feat": v1feat, "a_prev": a_prev, "d": d,
            "e_t": e_t, "h_prev": h_prev_masked, "h_raw": h_raw, "h_t": h_t,
            "pre": fwd.get("pre"), "g": fwd["g"],
            "v": fwd.get("v"), "s": fwd.get("s"),
            "v_prev": v_prev_carry, "s_prev": s_prev_carry,
            "reward": None,
        }
        self._tape.append(step)
        return h_t

    def act(self, h_t=None, greedy: bool = False) -> int:
        """Sample an action from pi(h_t). Records logits/probs/action for the REINFORCE update."""
        xp = self.xp
        if h_t is None:
            h_t = self._apply_lesion(self.h)
        logits = self.W_pi @ h_t + self.b_pi
        z = logits - logits.max()
        ez = xp.exp(z)
        probs = ez / ez.sum()
        probs_host = np.asarray(to_host(probs), dtype=np.float64)
        probs_host = np.clip(probs_host, 1e-8, None)
        probs_host = probs_host / probs_host.sum()
        if greedy:
            a = int(np.argmax(probs_host))
        else:
            a = int(self._sampler.choice(self.cfg.n_actions, p=probs_host))
        self._last_logits = logits
        self._last_probs = probs
        self._last_action = a
        self._last_h_for_pi = h_t
        return a

    def learn(self, reward: float):
        """Register the reward for the just-observed step; run the TBPTT predictive
        update at the window boundary; run the online policy (REINFORCE) update."""
        if self._tape:
            self._tape[-1]["reward"] = float(reward)

        # ---- predictive objective: TBPTT at the window boundary ----
        if not self._frozen and len(self._tape) >= self.cfg.tbptt_T:
            loss = self._tbptt_update(self._tape)
            self.last_pred_loss = loss
            self._update_learning_progress(loss)
            # truncate: keep the running state self.h/v/s (already carried), drop the tape
            self._tape = []

        # ---- policy: REINFORCE with running-mean baseline + learning-progress curiosity ----
        if not self._frozen and self._last_action is not None:
            r_int = float(reward) + self.cfg.curiosity_beta * self._last_lp
            adv = r_int - self._baseline
            self._baseline = self.cfg.baseline_decay * self._baseline + (1 - self.cfg.baseline_decay) * r_int
            self._policy_update(adv)
        self._last_action = None

    def _update_learning_progress(self, loss: float):
        cfg = self.cfg
        if self._loss_fast is None:
            self._loss_fast = loss
            self._loss_slow = loss
        else:
            self._loss_fast += cfg.lp_fast * (loss - self._loss_fast)
            self._loss_slow += cfg.lp_slow * (loss - self._loss_slow)
        # learning progress = the loss is DROPPING (slow > fast). Noisy-TV has high loss but ~0 LP.
        self._last_lp = max(0.0, float(self._loss_slow - self._loss_fast))

    def _policy_update(self, adv: float):
        xp = self.xp
        h = self._last_h_for_pi
        p = self._last_probs
        onehot = xp.zeros((self.cfg.n_actions,), dtype=xp.float32)
        onehot[self._last_action] = 1.0
        # d(-logp_a)/dlogits = p - onehot ; policy-gradient ascent on adv * logp_a
        dlogits = adv * (p - onehot)
        # entropy bonus: encourage exploration (dH/dlogits = -p*(logp - sum p logp))
        logp = xp.log(xp.clip(p, 1e-8, 1.0))
        ent_grad = -p * (logp - (p * logp).sum())
        dlogits = dlogits - self.cfg.entropy_beta * ent_grad
        gW = xp.outer(dlogits, h)
        gb = dlogits
        self.W_pi = self.W_pi - self.cfg.lr_policy * gW
        self.b_pi = self.b_pi - self.cfg.lr_policy * gb

    # ── TBPTT: forward-recompute + backward over a window (pure over params) ──
    def _window_forward(self, tape, P, W_enc, b_enc):
        """Recompute the window forward from tape[0]['h_prev'] with the given params.
        Returns (loss, cache) where cache holds per-step tensors for the backward.
        Uses EMA encoder (stop-grad) for the JEPA target.
        """
        xp = self.xp
        cfg = self.cfg
        T = len(tape)
        # h_prev of the FIRST step is the (detached) carry from the previous window
        h_prev = tape[0]["h_prev"]
        v_prev = s_prev = None
        if cfg.units == "spike":
            # start the window's membrane from the recorded (detached) pre-window carry;
            # zeros at a true truncation start.
            v_prev = tape[0].get("v_prev")
            s_prev = tape[0].get("s_prev")
            if v_prev is None:
                v_prev = xp.zeros((cfg.n_hidden,), dtype=xp.float32)
            if s_prev is None:
                s_prev = xp.zeros((cfg.n_hidden,), dtype=xp.float32)

        e_list, h_list, g_list, pre_list, v_list, s_list = [], [], [], [], [], []
        ehat_list, rhat_list = [], []
        hp = h_prev
        for t in range(T):
            e_t = W_enc @ tape[t]["v1feat"] + b_enc
            if cfg.units == "rate":
                fwd = self._core_forward_rate(hp, e_t, tape[t]["a_prev"], tape[t]["d"], P)
                h_raw = fwd["h_raw"]
                pre_list.append(fwd["pre"]); g_list.append(fwd["g"])
            else:
                fwd = self._core_forward_spike(v_prev, s_prev, hp, e_t, tape[t]["a_prev"], tape[t]["d"], P)
                h_raw = fwd["h_raw"]
                v_prev, s_prev = fwd["v"], fwd["s"]
                v_list.append(fwd["v"]); s_list.append(fwd["s"]); g_list.append(fwd["g"]); pre_list.append(None)
            ehat, rhat = self._heads_forward(h_raw, P)
            e_list.append(e_t); h_list.append(h_raw); ehat_list.append(ehat); rhat_list.append(rhat)
            hp = h_raw

        # loss: JEPA over t=0..T-2 (predict next-view EMA encoding), reward over all t with a reward
        loss = 0.0
        n_jepa = max(1, T - 1)
        z_targets = []
        for t in range(T):
            if t < T - 1:
                z = self.W_enc_ema @ tape[t + 1]["v1feat"] + self.b_enc_ema   # stop-grad target
                z_targets.append(z)
            else:
                z_targets.append(None)
        jl = 0.0
        for t in range(T - 1):
            diff = ehat_list[t] - z_targets[t]
            jl = jl + float((diff * diff).sum())
        jl = jl / n_jepa
        rl = 0.0
        n_r = 0
        for t in range(T):
            if tape[t]["reward"] is not None:
                d = rhat_list[t] - tape[t]["reward"]
                rl = rl + float(d * d)
                n_r += 1
        rl = (rl / n_r) if n_r > 0 else 0.0
        # VICReg variance floor on e (learned_ema only; provable anti-collapse)
        vl = 0.0
        if cfg.encoder == "learned_ema" and cfg.var_lambda > 0:
            E = xp.stack(e_list, axis=0)               # (T, E)
            mean = E.mean(axis=0, keepdims=True)
            std = xp.sqrt(E.var(axis=0) + 1e-6)        # (E,)
            hinge = xp.clip(cfg.var_gamma - std, 0.0, None)
            vl = float(hinge.mean()) * cfg.var_lambda

        loss = jl + cfg.beta_reward * rl + vl
        cache = {
            "e_list": e_list, "h_list": h_list, "g_list": g_list, "pre_list": pre_list,
            "v_list": v_list, "s_list": s_list, "ehat_list": ehat_list, "rhat_list": rhat_list,
            "z_targets": z_targets, "h_prev": h_prev, "v_prev0": v_prev, "n_jepa": n_jepa, "n_r": n_r,
        }
        return loss, cache

    def _window_backward(self, tape, P, cache):
        """Analytic BPTT grads of the predictive loss w.r.t. every param in P (+encoder if learned)."""
        xp = self.xp
        cfg = self.cfg
        T = len(tape)
        H = cfg.n_hidden
        grads = {k: xp.zeros_like(v) for k, v in P.items()}
        e_list = cache["e_list"]; h_list = cache["h_list"]; g_list = cache["g_list"]
        pre_list = cache["pre_list"]; v_list = cache["v_list"]; s_list = cache["s_list"]
        ehat_list = cache["ehat_list"]; rhat_list = cache["rhat_list"]; z_targets = cache["z_targets"]
        n_jepa = cache["n_jepa"]; n_r = cache["n_r"]

        # upstream grad on each h_t from the heads
        dh_head = [xp.zeros((H,), dtype=xp.float32) for _ in range(T)]
        for t in range(T):
            # JEPA head (t predicts t+1), only for t < T-1
            if t < T - 1:
                g_ehat = (2.0 / n_jepa) * (ehat_list[t] - z_targets[t])   # dL/dehat_t
                grads["W_pred"] += xp.outer(g_ehat, h_list[t])
                grads["b_pred"] += g_ehat
                dh_head[t] = dh_head[t] + P["W_pred"].T @ g_ehat
            # reward head
            if tape[t]["reward"] is not None and n_r > 0:
                g_rhat = (2.0 * cfg.beta_reward / n_r) * (rhat_list[t] - tape[t]["reward"])
                grads["w_r"] += g_rhat * h_list[t]
                grads["b_r"] += xp.asarray([g_rhat], dtype=xp.float32)
                dh_head[t] = dh_head[t] + g_rhat * P["w_r"]

        # VICReg variance grad on e (adds to de_t during the loop)
        de_var = [None] * T
        if cfg.encoder == "learned_ema" and cfg.var_lambda > 0:
            E = xp.stack(e_list, axis=0)               # (T, E)
            mean = E.mean(axis=0, keepdims=True)
            var = E.var(axis=0) + 1e-6
            std = xp.sqrt(var)                          # (E,)
            gate = (std < cfg.var_gamma).astype(xp.float32)   # 1 where below floor
            # L_var = (var_lambda / E_dim) * sum_d relu(gamma - std_d)
            Edim = e_list[0].shape[0]
            coef = -cfg.var_lambda / Edim
            # dstd_d/de[t,d] = (e[t,d]-mean_d)/(T*std_d)
            centered = E - mean                          # (T,E)
            dvar = coef * gate                           # (E,) per-dim gradient on std, times chain to e
            for t in range(T):
                de_var[t] = dvar * (centered[t] / (T * std))

        # ---- BPTT through the recurrence ----
        grad_h = [xp.zeros((H,), dtype=xp.float32) for _ in range(T + 1)]  # grad_h[t] = dL/dh_t; index shift: grad_h[t] for h at step t
        for t in range(T):
            grad_h[t + 1] = grad_h[t + 1] + dh_head[t]   # h at step t stored at index t+1 (grad_h[0]=pre-window carry)

        if cfg.units == "rate":
            alpha = cfg.alpha
            for t in range(T - 1, -1, -1):
                dh_t = grad_h[t + 1]
                gt = g_list[t]
                phi_prime = 1.0 - gt * gt                 # tanh'
                dpre = alpha * phi_prime * dh_t
                h_prev_t = tape[0]["h_prev"] if t == 0 else h_list[t - 1]
                grads["W_h"] += xp.outer(dpre, h_prev_t)
                grads["W_e"] += xp.outer(dpre, e_list[t])
                grads["W_a"] += xp.outer(dpre, tape[t]["a_prev"])
                grads["W_d"] += xp.outer(dpre, tape[t]["d"])
                grads["b_h"] += dpre
                de_t = P["W_e"].T @ dpre
                if de_var[t] is not None:
                    de_t = de_t + de_var[t]
                if "W_enc" in P:
                    grads["W_enc"] += xp.outer(de_t, tape[t]["v1feat"])
                    grads["b_enc"] += de_t
                # propagate to h_{t-1}: direct leak term + through W_h
                grad_h[t] = grad_h[t] + (1.0 - alpha) * dh_t + P["W_h"].T @ dpre
        else:
            # spike mode: h_raw = (1-a) h_prev + a s ; s = Heaviside(v-thr) [surrogate];
            # v = leak*v_prev*(1-s_prev) + (W_h s_prev + ...). Backprop through h-trace, s, v.
            alpha = cfg.alpha
            grad_v_next = xp.zeros((H,), dtype=xp.float32)   # dL/dv_{t+1} carried back
            grad_s_next = xp.zeros((H,), dtype=xp.float32)   # dL/ds_{t+1} via v_{t+1}'s reset & W_h recurrence
            for t in range(T - 1, -1, -1):
                dh_t = grad_h[t + 1]
                # h_raw_t = (1-a) h_{t-1} + a s_t
                ds_from_htrace = alpha * dh_t
                # trace recurrence to h_{t-1}
                grad_h[t] = grad_h[t] + (1.0 - alpha) * dh_t
                # s_t also feeds v_{t+1} (reset factor and W_h). grad_s_next holds that.
                ds_t = ds_from_htrace + grad_s_next
                v_t = v_list[t]
                surro = _atan_surrogate(xp, v_t - cfg.threshold)
                dv_t = ds_t * surro + grad_v_next            # v feeds s (surrogate) and next-step reset (grad_v_next)
                # params via v_t = leak*v_{t-1}*(1-s_{t-1}) + (W_h s_{t-1} + W_e e_t + W_a a + W_d d + b_h)
                s_prev = s_list[t - 1] if t > 0 else (tape[0].get("s_prev") if tape[0].get("s_prev") is not None else xp.zeros((H,), dtype=xp.float32))
                grads["W_h"] += xp.outer(dv_t, s_prev)
                grads["W_e"] += xp.outer(dv_t, e_list[t])
                grads["W_a"] += xp.outer(dv_t, tape[t]["a_prev"])
                grads["W_d"] += xp.outer(dv_t, tape[t]["d"])
                grads["b_h"] += dv_t
                de_t = P["W_e"].T @ dv_t
                if de_var[t] is not None:
                    de_t = de_t + de_var[t]
                if "W_enc" in P:
                    grads["W_enc"] += xp.outer(de_t, tape[t]["v1feat"])
                    grads["b_enc"] += de_t
                # carries to t-1:
                if t > 0:
                    v_prev = v_list[t - 1]
                    # dv_t/dv_{t-1} = leak*(1 - s_{t-1})
                    grad_v_next = dv_t * cfg.leak * (1.0 - s_prev)
                    # dv_t/ds_{t-1} = -leak*v_{t-1}  (reset)  +  W_h^T dv_t (recurrent drive)
                    grad_s_next = -dv_t * cfg.leak * v_prev + P["W_h"].T @ dv_t
                else:
                    grad_v_next = xp.zeros((H,), dtype=xp.float32)
                    grad_s_next = xp.zeros((H,), dtype=xp.float32)
        return grads

    def _tbptt_update(self, tape) -> float:
        loss, cache = self._window_forward(tape, self.P, self.W_enc, self.b_enc)
        grads = self._window_backward(tape, self.P, cache)
        self.opt.step(self.P, grads, clip=self.cfg.grad_clip)
        # keep the encoder refs in sync (Adam replaced the array objects in self.P)
        if self.cfg.encoder == "learned_ema":
            self.W_enc = self.P["W_enc"]
            self.b_enc = self.P["b_enc"]
            # EMA target update
            r = self.cfg.ema_rate
            self.W_enc_ema = r * self.W_enc_ema + (1.0 - r) * self.W_enc
            self.b_enc_ema = r * self.b_enc_ema + (1.0 - r) * self.b_enc
        self.n_updates += 1
        return float(loss)

    # ── read-outs for probes ────────────────────────────────────────────────
    def state(self):
        """Return the current shared state h_t as a host numpy vector (lesion-aware)."""
        return np.asarray(to_host(self._apply_lesion(self.h)), dtype=np.float32)

    def weight_hash(self) -> str:
        """Stable hash of W_h — determinism check (CLAUDE.md seeding trap)."""
        import hashlib
        a = np.asarray(to_host(self.P["W_h"]), dtype=np.float32)
        return hashlib.sha1(a.tobytes()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# self-checks
# ─────────────────────────────────────────────────────────────────────────────
def _make_tiny(units="rate", seed=1):
    cfg = PCSConfig(n_hidden=7, feat_dim=6, n_latent=4, n_actions=3, n_drive=2,
                    tbptt_T=5, units=units, var_lambda=0.5, encoder="learned_ema", seed=seed)
    return PredictiveContinualSubstrate(cfg)


def _fill_tape(sub, T, rng):
    """Drive `sub` for T steps with deterministic random inputs, register rewards."""
    sub.reset_state()
    a_prev = -1
    for t in range(T):
        v1 = rng.standard_normal(sub.cfg.feat_dim).astype(np.float32)
        d = rng.standard_normal(sub.cfg.n_drive).astype(np.float32)
        sub.observe(v1, a_prev, d)
        sub._tape[-1]["reward"] = float(rng.standard_normal())
        a_prev = int(rng.integers(sub.cfg.n_actions))
    return sub._tape


def gradcheck(units="rate", tol=2e-2):
    """Finite-difference the analytic BPTT grads for a tiny substrate."""
    xp_is_numpy = True
    sub = _make_tiny(units=units, seed=3)
    rng = np.random.default_rng(7)
    tape = _fill_tape(sub, sub.cfg.tbptt_T, rng)
    P = sub.P
    loss0, cache = sub._window_forward(tape, P, sub.W_enc, sub.b_enc)
    grads = sub._window_backward(tape, P, cache)

    eps = 1e-4
    max_rel = 0.0
    worst = None
    checked = 0
    for name in ["W_h", "W_e", "W_a", "W_d", "b_h", "W_pred", "b_pred", "w_r", "b_r", "W_enc", "b_enc"]:
        if name not in P:
            continue
        arr = np.asarray(to_host(P[name]), dtype=np.float64)
        flat = arr.ravel()
        gflat = np.asarray(to_host(grads[name]), dtype=np.float64).ravel()
        # sample up to 6 entries per param
        idxs = np.linspace(0, flat.size - 1, min(6, flat.size)).astype(int)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            P[name] = from_host(flat.reshape(arr.shape).astype(np.float32))
            if name in ("W_enc", "b_enc"):
                Wenc, benc = (P["W_enc"], P["b_enc"])
            else:
                Wenc, benc = sub.W_enc, sub.b_enc
            lp, _ = sub._window_forward(tape, P, Wenc, benc)
            flat[i] = orig - eps
            P[name] = from_host(flat.reshape(arr.shape).astype(np.float32))
            if name in ("W_enc", "b_enc"):
                Wenc, benc = (P["W_enc"], P["b_enc"])
            else:
                Wenc, benc = sub.W_enc, sub.b_enc
            lm, _ = sub._window_forward(tape, P, Wenc, benc)
            flat[i] = orig
            P[name] = from_host(flat.reshape(arr.shape).astype(np.float32))
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1.0, abs(num), abs(ana))
            rel = abs(num - ana) / denom
            checked += 1
            if rel > max_rel:
                max_rel = rel
                worst = (name, int(i), float(num), float(ana))
    ok = max_rel < tol
    print(f"[gradcheck units={units}] checked={checked} max_rel_err={max_rel:.2e} "
          f"{'OK' if ok else 'FAIL'}  worst={worst}")
    return ok


def selftest():
    """Determinism: two builds at one seed hash W_h identically; different seeds differ."""
    a = PredictiveContinualSubstrate(PCSConfig(seed=42, n_hidden=16, feat_dim=8))
    b = PredictiveContinualSubstrate(PCSConfig(seed=42, n_hidden=16, feat_dim=8))
    c = PredictiveContinualSubstrate(PCSConfig(seed=43, n_hidden=16, feat_dim=8))
    ha, hb, hc = a.weight_hash(), b.weight_hash(), c.weight_hash()
    ok = (ha == hb) and (ha != hc)
    print(f"[selftest] seed42={ha} seed42={hb} seed43={hc}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PCS substrate self-checks")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    all_ok = True
    if args.selftest or not (args.gradcheck or args.selftest):
        all_ok &= selftest()
    if args.gradcheck or not (args.gradcheck or args.selftest):
        all_ok &= gradcheck("rate")
        all_ok &= gradcheck("spike", tol=5e-2)
    raise SystemExit(0 if all_ok else 1)
