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

# Scaffold (SCAFFOLDS.md): innate primary reinforcers, delivered by the BODY channel. Balanced 11-03 (iteration 2):
# 8 bodily + 8 social per sign (8 bodily + 10 bodily-negative available); a seed draws 4 bodily + 4 social per sign
# (the genome's choice should not matter). Iteration 1 drew 8 of 13 at random from unbalanced lists (7+6 vs 10+3).
POSITIVE_BODILY = ["warm and safe in a soft blanket", "eating sweet fresh fruit when hungry", "resting in the warm sun",
                   "a full belly after a good meal", "cool water when thirsty", "a soft cozy bed after a long day",
                   "a pleasant warm bath", "a gentle breeze on a hot day"]
POSITIVE_SOCIAL = ["a gentle hug", "a kind hand on the shoulder", "being held close and calm", "a friend smiling at me",
                   "laughing together with friends", "being cuddled by someone who loves me",
                   "someone softly stroking my hair", "a child running into my arms"]
NEGATIVE_BODILY = ["sharp pain in the hand", "freezing cold and shivering", "hungry for days with nothing to eat",
                   "a deep cut that bleeds", "feeling sick and dizzy", "a hard fall onto the ground",
                   "a burn from a hot stove", "a loud bang and a sudden shock", "a painful bite",
                   "choking and unable to breathe"]
NEGATIVE_SOCIAL = ["being shouted at and pushed away", "left all alone in the dark", "someone I love leaving forever",
                   "being hit by someone bigger", "a stranger glaring and growling at me",
                   "being shoved out of the group", "nobody answering when I call for help",
                   "being grabbed roughly by the arm"]
POSITIVE_PRIMARIES = POSITIVE_BODILY + POSITIVE_SOCIAL
NEGATIVE_PRIMARIES = NEGATIVE_BODILY + NEGATIVE_SOCIAL
# Neutral BODY states: each opponent pole's own zero (the poles are centred here before they are combined).
BODY_NEUTRAL = ["standing still in a quiet room", "sitting on a wooden chair", "walking along a flat path",
                "breathing in and out slowly", "holding a small stone", "looking at a white wall"]


def draw_primaries(rng, per_kind: int = 4):
    """A seed's innate set: per_kind bodily + per_kind social primaries per sign (stratified, balanced)."""
    return (rng.sample(POSITIVE_BODILY, per_kind) + rng.sample(POSITIVE_SOCIAL, per_kind),
            rng.sample(NEGATIVE_BODILY, per_kind) + rng.sample(NEGATIVE_SOCIAL, per_kind))


# Iteration 5 (11-18, design doc section 17; SCAFFOLDS.md): 48 matched body-scaffold pairs. Each pair shares its
# situation and differs only in the sign of its valence, so the topic cancels in the pair difference. A seed draws
# 32 (`draw_pairs`). Written before any iteration-5 run; content words are disjoint from the lexical instrument, the
# question sets and every event set (tests/test_mood5.py).
PRIMARY_PAIRS = [
    ('warm and snug in a soft blanket', 'cold and shaking beneath a thin blanket'),
    ('a mug of hot tea held in cupped hands', 'a mug of hot tea tipped over onto my hands'),
    ('walking barefoot on soft green grass', 'walking barefoot on sharp splintered glass'),
    ('a sweet peach dripping with nectar', 'a rotten peach that tastes of mold'),
    ('stretching tall after a deep restful sleep', 'stumbling about after hours with no sleep'),
    ('a cool shower after a hot dusty hike', 'a freezing shower with no hot water'),
    ('a clerk holding the door open for me', 'a clerk slamming the door in my face'),
    ('my uncle carrying me gently to bed', 'my uncle dragging me roughly out of bed'),
    ('a soft pillow beneath my head', 'a cold brick beneath my head'),
    ('fresh bread baking in the oven', 'burnt bread smoking in the oven'),
    ('a steady hand helping me up the stairs', 'a shove from behind on the stairs'),
    ('my hands wrapped in thick wool gloves', 'my hands numb and cracked in the wind'),
    ('a dry towel ready after a swim', 'no towel after a swim in icy water'),
    ('singing along with my daughter in the car', 'my daughter screaming at me in the car'),
    ('a mother humming while she brushes my hair', 'a mother yanking my hair while she brushes it'),
    ('an open skylight with a sweet floral smell', 'an open skylight with a smell of rot'),
    ('a dog wagging its tail at the kerb', 'a dog lunging and snapping at the kerb'),
    ('a quiet study with soft lamp light', 'a quiet study with a flickering bulb and a strange hum'),
    ('sipping cool clear water by the river', 'swallowing muddy water by the river'),
    ('my feet resting on a heated tile', 'my feet pressed on a frozen tile'),
    ('a postman waving as I pass', 'a postman sneering as I pass'),
    ('a teacher nodding at my answer', 'a teacher frowning at my answer'),
    ('a smooth warm stone in my palm', 'a jagged sharp stone cutting my palm'),
    ('lying on a sandy shore in strong sun', 'lying on a rocky shore in freezing rain'),
    ('my aunt sharing her sandwich with me', 'my aunt grabbing my sandwich from me'),
    ('a gentle massage on my stiff shoulders', 'a hard pinch on my stiff shoulders'),
    ('a juicy orange peeled and shared', 'a bitter orange spat onto the pavement'),
    ('a granny wrapping me in her scarf', 'a granny snatching her scarf off me'),
    ('snow falling softly on my cheeks by the fire', 'sleet stinging my cheeks far from any fire'),
    ('a hot meal ready on the table', 'a bare plate ready on the table'),
    ('a crowd giving me space on the bench', 'a crowd pushing me off the bench'),
    ('the smell of cut grass after rain', 'the smell of smoke after a blaze'),
    ('a cat purring against my chest', 'a cat hissing and scratching my chest'),
    ('my spine resting against a sun-warmed fence', 'my spine pressed against a wet icy fence'),
    ('a clear light dawn with birds singing', 'a gray heavy dawn with crows screaming'),
    ('a firm friendly handshake from a colleague', 'a crushing hostile handshake from a colleague'),
    ('a waiter pulling out a chair for me', 'a waiter pulling a chair out from behind me'),
    ('bare toes dipped in a warm shallow pool', 'bare toes dipped in a filthy shallow pool'),
    ('a sore tooth soothed by medicine', 'a sore tooth throbbing for lack of medicine'),
    ('my nephew tucking a shawl around me', 'my nephew tearing a shawl off me'),
    ('a slow deep breath of clean mountain air', 'a slow deep breath of thick burning smoke'),
    ('eyes resting in a soft patch of sunlight', 'eyes watering in the harsh glare of floodlights'),
    ('a trainer clapping me on the shoulder', 'a trainer slapping the side of my head'),
    ('feet dry in sturdy boots on a wet trail', 'feet soaked in worn boots on a wet trail'),
    ('an old classmate calling me by name', 'an old classmate walking by with a blank stare'),
    ('a thick sweater fresh from the dryer', 'a thin sweater soaked by the rain'),
    ('a table full of guests turning to listen to me', 'a table full of guests turning their backs to me'),
    ('a cool damp cloth on my forehead', 'a scratchy rough cloth wrapped tight on my forehead'),
]
N_PAIRS_DRAWN = 32


def draw_pairs(rng, n: int = N_PAIRS_DRAWN):
    """A seed's matched pairs: n of the 48 (the genome's choice should not matter)."""
    return rng.sample(PRIMARY_PAIRS, n)


def band_for(n_layers: int, band=(8, 16)):
    """The modulation / read band for a model: the pre-registered (8, 16) when the model has >= 16 layers, else the
    middle third (0.1B has 12 layers -> (4, 8)). Returned as a half-open (lo, hi)."""
    if n_layers >= band[1]:
        return tuple(band)
    return (n_layers // 3, 2 * n_layers // 3)


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
    """readout: "twopole_deep" (iteration 2, default): appraisal reads the deeper half of the layers (association
    cortex -> amygdala), each opponent pole is centred on neutral body states before the axis is formed.
    "lin_all" (iteration 1): one axis over all layers, primaries not centred."""
    TAU_MOOD = 4.0      # events: slow mood
    TAU_AROUSAL = 1.5   # events: fast arousal
    GAIN = 16.0         # modulation gain (dev questions, seed 42: 8 weak, 32 intrudes primary content; design doc)

    def __init__(self, enabled: bool | None = None, readout: str = "twopole_deep", mod_band=(8, 16),
                 alpha: float = 0.0, placement: str = "pre"):
        assert readout in ("twopole_deep", "lin_all", "pairs")
        assert placement in ("pre", "post")
        self.readout_kind = readout
        # iteration 5 (readout="pairs" only; ignored by the other readouts): band, push size as a fraction of the
        # layer's matrix-memory norm, and where the push lands ("pre": before the question is read, "post": after)
        self.mod_band, self.alpha, self.placement = tuple(mod_band), alpha, placement
        self.band_used = None
        self.layer_norm = None    # {layer: mean Frobenius norm of the matrix memory over neutral reads}
        self.topic_delta = None   # modulation pattern of the topic-only control (norm-matched per layer)
        self.topic_axis = None    # appraisal axis of the topic-only control: {"unit", "zero", "scale"}
        self.read_layers = None
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

    def grow(self, cortex, base, positives=None, negatives=None, neutral=None, body_neutral=None, pairs=None):
        if self.readout_kind == "pairs":
            return self._grow_pairs(cortex, base, pairs or PRIMARY_PAIRS[:N_PAIRS_DRAWN], neutral, body_neutral)
        positives = positives or POSITIVE_PRIMARIES
        negatives = negatives or NEGATIVE_PRIMARIES
        neutral = neutral or NEUTRAL_REFERENCE
        body_neutral = body_neutral or BODY_NEUTRAL
        n_layers = len(base) // 3
        self.layers = list(range(n_layers))
        self.read_layers = self.layers if self.readout_kind == "lin_all" else list(range(n_layers // 2, n_layers))
        pos = [self._trace(cortex, base, t, "Body") for t in positives]
        neg = [self._trace(cortex, base, t, "Body") for t in negatives]
        mean = lambda ds, i: torch.stack([d[i] for d in ds]).mean(0)  # noqa: E731
        # modulation pattern (all layers, raw scale): half the positive - negative primary gap (unchanged from it. 1)
        self.axis_delta = [None if i % 3 != 1 else ((mean(pos, i) - mean(neg, i)) / 2.0) for i in range(len(base))]
        flat = lambda ds: torch.stack([_flat_unit(d, self.read_layers) for d in ds]).mean(0)  # noqa: E731
        up, un = flat(pos), flat(neg)
        if self.readout_kind == "lin_all":
            ax = up - un
        else:   # opponent poles (BLA positive / negative populations), each centred on neutral body states
            bn = flat([self._trace(cortex, base, t, "Body") for t in body_neutral])
            P, N = up - bn, un - bn
            ax = P / P.norm().clamp_min(1e-8) - N / N.norm().clamp_min(1e-8)
        self.axis_unit = ax / ax.norm().clamp_min(1e-8)
        # zero point: neutral conversation statements; scale: half the primaries' separation on the axis
        self.zero = 0.0
        z = [self.project(cortex, base, t) for t in neutral]
        self.zero = sum(z) / len(z)
        self.scale = max(float((up - un) @ self.axis_unit) / 2.0, 1e-6)
        return self

    # ---- iteration 5: matched-pair axis (readout="pairs") ----
    def _grow_pairs(self, cortex, base, pairs, neutral=None, body_neutral=None):
        """Axis = mean over pairs of the (positive - negative) fast-memory trace, per layer of the band (the topic cancels
        in each difference). Appraisal reads the band, zero = midpoint of the two poles read as "User", scale = half the
        pole gap. Push = per-layer unit direction x the layer's mean matrix-memory norm over neutral reads.
        Streaming: only the band layers of each pair are kept (a 1.5B trace of all layers is ~12 MB)."""
        neutral = neutral or NEUTRAL_REFERENCE
        body_neutral = body_neutral or BODY_NEUTRAL
        n_layers = len(base) // 3
        self.layers = list(range(n_layers))
        lo, hi = band_for(n_layers, self.mod_band)
        self.band_used = (lo, hi)
        band = list(range(lo, hi))
        self.read_layers = band
        unit = lambda d: d / d.flatten().norm().clamp_min(1e-8)  # noqa: E731
        # layer norms: matrix memory (the full state, not the change) after neutral conversation statements
        norms = {l: [] for l in band}
        for t in neutral:
            _, st = cortex.read(f"User: {t}\n\n", base)
            for l in band:
                norms[l].append(float(st[3 * l + 1].float().norm()))
        self.layer_norm = {l: sum(v) / len(v) for l, v in norms.items()}
        n = len(pairs)
        self._pair_d, self._pair_fd = [], []
        topic_sum = {l: 0 for l in band}
        fp_sum = fn_sum = 0
        pu_sum = nu_sum = 0
        for ptxt, ntxt in pairs:
            p, q = self._trace(cortex, base, ptxt, "Body"), self._trace(cortex, base, ntxt, "Body")
            self._pair_d.append({l: (p[3 * l + 1] - q[3 * l + 1]).clone() for l in band})
            fp, fn = _flat_unit(p, band), _flat_unit(q, band)
            self._pair_fd.append(fp - fn)
            fp_sum, fn_sum = fp_sum + fp, fn_sum + fn
            for l in band:
                topic_sum[l] = topic_sum[l] + (p[3 * l + 1] + q[3 * l + 1]) / 2
            pu_sum = pu_sum + _flat_unit(self._trace(cortex, base, ptxt, "User"), band)
            nu_sum = nu_sum + _flat_unit(self._trace(cortex, base, ntxt, "User"), band)
            del p, q
        bn = [self._trace(cortex, base, t, "Body") for t in body_neutral]
        bn_mat = {l: torch.stack([b[3 * l + 1] for b in bn]).mean(0) for l in band}
        fb = torch.stack([_flat_unit(b, band) for b in bn]).mean(0)
        del bn
        mk = lambda per_layer: [None if (i % 3 != 1 or (i // 3) not in self.layer_norm) else  # noqa: E731
                                unit(per_layer[i // 3]) * self.layer_norm[i // 3] for i in range(len(base))]
        self.axis_delta = mk({l: torch.stack([d[l] for d in self._pair_d]).mean(0) for l in band})
        self.topic_delta = mk({l: topic_sum[l] / n - bn_mat[l] for l in band})
        # appraisal axes (flat per-layer-unit space of the band)
        ax = torch.stack(self._pair_fd).mean(0)
        self.axis_unit = ax / ax.norm().clamp_min(1e-8)
        t = (fp_sum + fn_sum) / (2 * n) - fb
        self.topic_axis = {"unit": t / t.norm().clamp_min(1e-8)}
        # poles read as "User" (the speaker of conversation events) -> the midpoint zero; the same rule for every axis
        self._pole_u = (pu_sum / n, nu_sum / n)
        self.zero, self.scale = self._midpoint(self.axis_unit)
        self.topic_axis["zero"], self.topic_axis["scale"] = self._midpoint(self.topic_axis["unit"])
        return self

    def _midpoint(self, unit_axis):
        pu, nu = self._pole_u
        zero = float(((pu + nu) / 2) @ unit_axis)
        gap = float((pu - nu) @ unit_axis)
        return zero, max(abs(gap) / 2.0, 1e-6)

    def shuffled(self, rng):
        """Sign-shuffled control: each pair difference is sign-flipped (exactly half of them, a random half), so the
        shared valence component cancels while the per-pair content stays. Returns (modulation delta norm-matched per
        layer, appraisal axis dict). Needs readout="pairs"."""
        n = len(self._pair_d)
        signs = [1.0] * (n // 2) + [-1.0] * (n - n // 2)
        rng.shuffle(signs)
        unit = lambda d: d / d.flatten().norm().clamp_min(1e-8)  # noqa: E731
        per = {l: sum(s * d[l] for s, d in zip(signs, self._pair_d)) / n for l in self.read_layers}
        delta = [None if (i % 3 != 1 or (i // 3) not in self.layer_norm) else unit(per[i // 3]) * self.layer_norm[i // 3]
                 for i in range(len(self.axis_delta))]
        ax = sum(s * f for s, f in zip(signs, self._pair_fd)) / n
        u = ax / ax.norm().clamp_min(1e-8)
        zero, scale = self._midpoint(u)
        return delta, {"unit": u, "zero": zero, "scale": scale, "signs": signs}

    def shuffled_delta(self, rng):
        return self.shuffled(rng)[0]

    def project_pure(self, cortex, base, text, speaker="User", axis=None):
        """Centred projection of one read on an axis ({"unit","zero","scale"}; default the pair axis). Touches no
        affect state (no mood, arousal or last appraisal update). Returns (projection - zero, tanh(that / scale))."""
        ax = axis or {"unit": self.axis_unit, "zero": self.zero, "scale": self.scale}
        x = _flat_unit(self._trace(cortex, base, text, speaker), self.read_layers)
        raw = float(x @ ax["unit"]) - ax["zero"]
        return raw, math.tanh(raw / ax["scale"])

    def project(self, cortex, base, text, speaker="User"):
        x = _flat_unit(self._trace(cortex, base, text, speaker), self.read_layers)
        return float(x @ self.axis_unit) - self.zero

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
        if self.readout_kind == "pairs":
            return state if self.alpha == 0.0 else cortex.add_delta(state, self.axis_delta, self.alpha * self.mood)
        return cortex.add_delta(state, self.axis_delta, self.GAIN * self.mood)

    # ---- honest read-out ----
    def readout(self) -> dict:
        """Functional read-out of the affect variables (numbers only; not a claim of experience)."""
        return {"mood": round(self.mood, 4), "arousal": round(self.arousal, 4),
                "last_appraisal": round(self.last_appraisal, 4), "modulation_on": bool(self.enabled)}
