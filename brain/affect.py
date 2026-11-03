"""Affect: appraisal of events -> a slow mood variable -> a diffuse valence bias on the cortex. Default OFF.

Design and biology: brain/research/2026-11-03-mood-state-design.md.

How it works (all on the cortex's own code; no words are ever added to what the brain hears or says):
1. Innate primaries (declared scaffold, the "body"): a handful of bodily events with an innate sign (warmth, food,
   gentle touch vs pain, cold, sickness). The cortex reads each one; the difference between the mean positive and the
   mean negative fast-memory trace is the VALENCE AXIS in the cortex's own state space (an amygdala-like innate hook
   onto cortical codes; old track DR-2b: valence self-organizes from ~10 innate primary reinforcers).
2. Appraisal: each conversation event is read by the cortex; its trace is projected onto the valence axis (per-layer
   normalized, centred on neutral statements) -> a signed appraisal v in (-1, 1). Arousal = a fast leaky |v|.
3. Mood: a slow leaky integrator of appraisals (time constant TAU_MOOD events): slow mood vs fast emotion.
4. Modulation: before the cortex reads a turn, GAIN * mood * axis is added to its matrix memory (a low-dimensional,
   diffuse, signed bias, one scalar broadcast through a fixed projection pattern). Lesion = modulation off while the
   mood is still computed and read out.
5. Read-out: `readout()` returns the numbers (functional read-out; never a claim of experience; never turned into
   marker words by host code).
"""
from __future__ import annotations

import math
import os

import torch

# Scaffold (SCAFFOLDS.md): innate primary reinforcers, delivered by the BODY channel: bodily (warmth, food, pain,
# cold) and innate social ones (a smile, being held vs being shouted at, separation). 13 per sign; a seed draws a
# subset (the genome's choice should not matter).
POSITIVE_PRIMARIES = ["warm and safe in a soft blanket", "a gentle hug", "eating sweet fresh fruit when hungry",
                      "resting in the warm sun", "a kind hand on the shoulder", "a full belly after a good meal",
                      "cool water when thirsty", "a soft cozy bed after a long day", "being held close and calm",
                      "a pleasant warm bath", "a friend smiling at me", "laughing together with friends",
                      "being cuddled by someone who loves me"]
NEGATIVE_PRIMARIES = ["sharp pain in the hand", "freezing cold and shivering", "hungry for days with nothing to eat",
                      "a deep cut that bleeds", "feeling sick and dizzy", "a hard fall onto the ground",
                      "a burn from a hot stove", "a loud bang and a sudden shock", "a painful bite",
                      "choking and unable to breathe", "being shouted at and pushed away",
                      "left all alone in the dark", "someone I love leaving forever"]
# Neutral statements in the conversation frame: the appraisal's zero point (an amygdala baseline).
NEUTRAL_REFERENCE = ["The table is next to the window.", "There are seven days in a week.",
                     "The bus leaves at nine in the morning.", "The book has two hundred pages.",
                     "We put the cups in the cupboard.", "The road goes past the station."]

DEFAULT_ON = os.environ.get("BRAIN_AFFECT", "0") == "1"


def _flat_unit(delta, layers):
    """Per-layer unit-normalized matrix-memory trace, concatenated (each layer weighs equally in the projection)."""
    parts = []
    for i in layers:
        d = delta[3 * i + 1].flatten().float()
        parts.append(d / d.norm().clamp_min(1e-8))
    return torch.cat(parts) / math.sqrt(len(layers))


class Affect:
    TAU_MOOD = 4.0      # events: slow mood
    TAU_AROUSAL = 1.5   # events: fast arousal
    GAIN = 16.0         # modulation gain (dev questions, seed 42: 8 weak, 32 intrudes primary content; design doc)

    def __init__(self, enabled: bool | None = None):
        self.enabled = DEFAULT_ON if enabled is None else enabled   # the modulation switch (lesion = False)
        self.mood = 0.0
        self.arousal = 0.0
        self.last_appraisal = 0.0
        self.axis_unit = None     # projection axis (concatenated per-layer unit traces)
        self.axis_delta = None    # modulation pattern, per-layer matrix-memory (raw scale, half the pos-neg gap)
        self.zero, self.scale = 0.0, 1.0
        self.layers = None

    # ---- birth: hook the innate primaries onto the cortex's code ----
    def _trace(self, cortex, base, text, speaker):
        _, st = cortex.read(f"{speaker}: {text}\n\n", base)
        return cortex.matrix_delta(st, base)

    def grow(self, cortex, base, positives=None, negatives=None, neutral=None):
        positives = positives or POSITIVE_PRIMARIES
        negatives = negatives or NEGATIVE_PRIMARIES
        neutral = neutral or NEUTRAL_REFERENCE
        self.layers = list(range(len(base) // 3))
        pos = [self._trace(cortex, base, t, "Body") for t in positives]
        neg = [self._trace(cortex, base, t, "Body") for t in negatives]
        mean = lambda ds, i: torch.stack([d[i] for d in ds]).mean(0)  # noqa: E731
        self.axis_delta = [None if i % 3 != 1 else ((mean(pos, i) - mean(neg, i)) / 2.0) for i in range(len(base))]
        up = torch.stack([_flat_unit(d, self.layers) for d in pos]).mean(0)
        un = torch.stack([_flat_unit(d, self.layers) for d in neg]).mean(0)
        self.axis_unit = (up - un) / (up - un).norm().clamp_min(1e-8)
        # zero point: neutral conversation statements; scale: half the primaries' separation on the axis
        self.zero = 0.0
        z = [self.project(cortex, base, t) for t in neutral]
        self.zero = sum(z) / len(z)
        p = sum(float(_flat_unit(d, self.layers) @ self.axis_unit) for d in pos) / len(pos)
        n = sum(float(_flat_unit(d, self.layers) @ self.axis_unit) for d in neg) / len(neg)
        self.scale = max((p - n) / 2.0, 1e-6)
        return self

    def project(self, cortex, base, text, speaker="User"):
        return float(_flat_unit(self._trace(cortex, base, text, speaker), self.layers) @ self.axis_unit) - self.zero

    # ---- appraisal -> emotion (fast) -> mood (slow) ----
    def appraise(self, cortex, base, text, speaker="User"):
        v = math.tanh(self.project(cortex, base, text, speaker) / self.scale)
        self.update(v)
        return v

    def update(self, v: float):
        self.last_appraisal = v
        self.mood += (v - self.mood) / self.TAU_MOOD
        self.arousal += (abs(v) - self.arousal) / self.TAU_AROUSAL

    def reset(self):
        self.mood = self.arousal = self.last_appraisal = 0.0

    # ---- action on the cortex ----
    def modulate(self, cortex, state):
        """Diffuse valence bias on the cortex's fast memory. Off (lesion) -> the state is returned untouched."""
        if not self.enabled or self.axis_delta is None or self.mood == 0.0:
            return state
        return cortex.add_delta(state, self.axis_delta, self.GAIN * self.mood)

    # ---- honest read-out ----
    def readout(self) -> dict:
        """Functional read-out of the affect variables (numbers only; not a claim of experience)."""
        return {"mood": round(self.mood, 4), "arousal": round(self.arousal, 4),
                "last_appraisal": round(self.last_appraisal, 4), "modulation_on": bool(self.enabled)}
