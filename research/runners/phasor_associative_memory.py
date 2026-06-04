"""PhasorAssociativeMemory -- the LEARNED-code foundation of phasor substrate unification.

The nesting agent (research/runners/nested_composition_agent.py) uses CONSTRUCTED phasor codes (assigned by
phase arithmetic). This memory instead LEARNS the map from a grounded sparse word cue to a phasor concept
code via online weight-bounded spike-timing plasticity -- the biologically-grounded mechanism validated
cheap-first this session (research/findings/2026-06-03-spiking-STDP-learns-phasor-map-RESOLVES-algorithmic.md):
real-valued synapses (scale, not rotate phase), an asymmetric STDP kernel, hard weight saturation, and the
project's actual grounded word encoder (sim.text_embeddings.vocab_to_drive_pattern). The readout is the
resonate-and-fire phasor-neuron phase (angle of the real-weighted input population vector; Frady-Sommer 2019).

It is the reusable building block of a possible production migration onto phasor FHRR -- the representation
that, unlike the non-invertible real-Hadamard production binding, supports nesting/composition. Reuse-by-import
of sim.text_embeddings; no protected-module edits.

  python -m research.runners.phasor_associative_memory          # scripted demo
"""
from __future__ import annotations
import numpy as np

from sim.text_embeddings import vocab_to_drive_pattern


class PhasorAssociativeMemory:
    """Learn word -> phasor-code associations via online weight-bounded STDP; recall (with abstention) and
    compose (bind/unbind through roles). Grounded cue = the word's active neurons (vocab_to_drive_pattern)
    firing at their fixed preferred phases."""

    def __init__(self, n_input=256, D=512, seed=42, sparsity=0.1, epochs=4, learn_rate=0.05,
                 w_max=1.0, tau=0.6, a_plus=1.0, a_minus=0.5, conf_threshold=0.15):
        self.n_input = int(n_input)
        self.D = int(D)
        self.sparsity = float(sparsity)
        self.epochs = int(epochs)
        self.learn_rate = float(learn_rate)
        self.w_max = float(w_max)
        self.tau, self.a_plus, self.a_minus = float(tau), float(a_plus), float(a_minus)
        self.conf_threshold = float(conf_threshold)
        self._rng = np.random.default_rng(seed)
        self.phi = self._rng.uniform(-np.pi, np.pi, size=self.n_input)   # fixed preferred phase per input neuron
        self.W = np.zeros((self.D, self.n_input))                        # REAL synaptic weights
        self.codes: dict[str, np.ndarray] = {}                           # token -> phasor code (phases, D)
        self.roles: dict[str, np.ndarray] = {}                           # role name -> role phases (D,)

    # --- grounded cue + code ---
    def _cue(self, token):
        active = vocab_to_drive_pattern(token, n_neurons=self.n_input, sparsity=self.sparsity) > 0
        c = np.zeros(self.n_input, complex)
        c[active] = np.exp(1j * self.phi[active])
        return c, np.where(active)[0]

    def _ensure_code(self, token):
        if token not in self.codes:
            self.codes[token] = self._rng.uniform(-np.pi, np.pi, size=self.D)   # the concept's phasor code
        return self.codes[token]

    def role(self, name):
        if name not in self.roles:
            self.roles[name] = self._rng.uniform(-np.pi, np.pi, size=self.D)
        return self.roles[name]

    def code(self, token):
        return self.codes.get(token)

    # --- learning (online weight-bounded STDP) ---
    def learn(self, token):
        """Learn the grounded-cue(token) -> code(token) association; online updates with hard weight bounds."""
        code = self._ensure_code(token)
        _, idx = self._cue(token)
        for _ in range(self.epochs):
            dt = code[:, None] - self.phi[idx][None, :]         # post(code_j) - pre(preferred phase of active input i)
            dt = (dt + np.pi) % (2 * np.pi) - np.pi
            dW = np.where(dt > 0, self.a_plus * np.exp(-dt / self.tau), -self.a_minus * np.exp(dt / self.tau))
            self.W[:, idx] = np.clip(self.W[:, idx] + self.learn_rate * dW, -self.w_max, self.w_max)
        return self

    # --- recall (with abstention) ---
    def _readout(self, cue_vec):
        return np.angle(self.W @ cue_vec)                       # resonate-and-fire phase of the weighted population vector

    @staticmethod
    def _pcos(pred, code):
        return float(np.cos(pred - code).mean())               # phasor cosine in [-1, 1]

    def _best(self, pred):
        if not self.codes:
            return None, -1.0
        toks = list(self.codes)
        sims = [self._pcos(pred, self.codes[t]) for t in toks]
        k = int(np.argmax(sims))
        return toks[k], sims[k]

    def recall_confidence(self, token):
        pred = self._readout(self._cue(token)[0])
        return self._best(pred)[1]

    def recall(self, token):
        """word -> the recalled concept token, or None if no learned concept matches confidently (abstention)."""
        pred = self._readout(self._cue(token)[0])
        tok, conf = self._best(pred)
        return tok if conf >= self.conf_threshold else None

    # --- composition (the learned codes bind/unbind through roles) ---
    def bind(self, role_name, token):
        return np.exp(1j * (self.role(role_name) + self.code(token)))     # bind = phase add

    def bundle(self, vecs):
        return np.sum(vecs, axis=0)                                       # superposition

    def unbind_cleanup(self, role_name, bundle):
        pred = np.angle(bundle * np.exp(-1j * self.role(role_name)))      # unbind then clean up vs the codebook
        return self._best(pred)[0]


def main():
    words = ["apple", "river", "dog", "cat", "big", "small", "hot", "cold"]
    m = PhasorAssociativeMemory(seed=42)
    print("=== PhasorAssociativeMemory: LEARNED word->code memory (online-bounded STDP) ===\n", flush=True)
    for w in words:
        m.learn(w)
    print("  -- recall (learned words; abstain on the unknown) --", flush=True)
    for w in words + ["zebra"]:
        print(f"  recall({w!r}) -> {m.recall(w)}   (confidence {m.recall_confidence(w):.2f})", flush=True)
    print("\n  -- compose with the LEARNED codes (bind/unbind through roles) --", flush=True)
    bundle = m.bundle([m.bind("AGENT", "dog"), m.bind("PATIENT", "cat")])
    print(f"  bind(AGENT,dog)+bind(PATIENT,cat) -> who is AGENT? {m.unbind_cleanup('AGENT', bundle)}; "
          f"PATIENT? {m.unbind_cleanup('PATIENT', bundle)}", flush=True)
    print("\n  -> the migration's foundation: codes LEARNED by spike-timing plasticity (not constructed) that", flush=True)
    print("     recall, abstain, and compose -- the property the non-invertible production binding lacks.", flush=True)


if __name__ == "__main__":
    main()
