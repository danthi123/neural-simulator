"""AFFECT-CONDITIONED GENERATION for the Qwen articulation mouth (D5 "feel" over the OPEN reply) — DEFAULT-OFF.

WHY (research/findings/2026-09-23-affect-tone-neural-coupling-6seed-subdelta-NOGO.md + its external-research
section): two DECODE-POINT methods (an additive top-margin logit bias; a neuromodulator nudge at the decode race)
are NO-GO over the open reply. The proven class in the literature conditions the GENERATION REPRESENTATION on a
continuous valence/arousal signal — Affect-LM (Ghosh et al., ACL 2017) and continuous VAD-conditioning (Guo, Xu &
Chua, arXiv:2111.04730) — not the decode. The Qwen mouth is the owner-ratified PERMANENT conditioned-articulation
scaffold (2026-09-19), and the brain ALREADY conditions it through one channel: `build_prompt(StateContext)` —
the retrieved KNOWLEDGE lines + a MOOD line built from the live spiking affect organ's valence.

THE MEASURED DEFECT IN THAT CHANNEL (this module's reason to exist, quantified in
research/runners/_lbf_affect_conditioned_mouth_derisk.py --magnitude-audit over the committed NO-GO arm data):
`_mood_phrase` has a DEAD ZONE at |valence| < 0.25, and the live valence is `clip(4 * differential)` with the
organ's held differential at +0.035..+0.041 (positive priming) and -0.018..-0.045 (negative priming) — i.e.
valence +0.14..+0.16 / -0.07..-0.18. So on 12/12 pos/neg arms of both NO-GO runs the Qwen MOOD line read
"even and steady": the brain's affect never reached the mouth's conditioning through words at all, only through
a two-decimal number. This module removes that dead zone and adds the representation-level method.

TWO MODES (env `BRAIN_OPEN_ENDED_AFFECT_CONDITIONED`; unset/0/off -> this module is never imported by
`open_ended_chat.answer_turn` -> byte-identical to before it existed):

  prompt : GRADED, dead-zone-free conditioning in the EXISTING channel. The MOOD line is rewritten from the
           normalized conditioning scalar c (below) with a monotone intensity grade; c == 0 exactly (the
           BRAIN_AFFECT_LESION arm) reproduces the production "even and steady" wording.
  resid  : REPRESENTATION conditioning (the Affect-LM / VAD-conditioning class, realized as contrastive activation
           addition — Rimsky et al. 2023 "Steering Llama 2 via Contrastive Activation Addition"; Turner et al.
           2023 "Activation Addition"). The prompt is left EXACTLY as production builds it; during generation a
           forward hook adds  c * K * u  to the residual stream at decoder layer L, where u is the mouth's OWN
           affect axis (difference of mean layer-L activations of Qwen on a fixed set of positive vs negative
           WARRINER-word sentences — disjoint from the independent scoring lexicon, enforced by the runner) and c
           is the BRAIN's live valence. c == 0 -> no hook is registered at all (exact no-op).

THE CONDITIONING SCALAR c (brain-derived, not a host sentiment formula): c = clip(valence / VALENCE_FS, -1, 1)
where `valence` is exactly what brain_chat already passes the mouth (the spiking affect organ's held
differential, `valence_from_affect`) and VALENCE_FS is the organ's own FULL-SCALE valence (4 x the held
differential at |appraisal| = 1, measured by `--calibrate-organ`; env `BRAIN_AFFECT_COND_VALENCE_FS`
overrides). This is a READOUT GAIN NORMALIZATION — the downstream reader scaled to the upstream population's
operating range (divisive normalization, Carandini & Heeger 2012). Done here as one host division: a NAMED
SHORTCUT on the scaffold boundary, not a brain computation. The organ's sign-asymmetry (V- weaker than V+) is
deliberately NOT normalized away — one constant for both signs, so the mouth can only express what the organ holds.

HONESTY: functional read-out only. Reply tone tracking the organ's valence is a functional coupling; nothing here
claims felt experience. No sim/ edit.
"""
from __future__ import annotations

import contextlib
import os
import threading

# ── preregistered operating point (fixed BEFORE any tone result existed; see the runner's GO-gate block) ──────
# Qwen2.5-0.5B has 24 decoder layers; mid-depth is the standard CAA steering site.
RESID_LAYER = 12
# CAA multiplier applied to the RAW difference-of-means vector at full-scale c=1 (Rimsky et al. report coherent
# steering with multipliers in the low single digits). Fixed at 4.0; the runner's fluency precondition
# (salad_frac <= 0.16) is the tripwire if this breaks form. NOT tuned on tone.
RESID_K = 4.0
# default full-scale valence (4 x the organ's held differential at |appraisal|=1). Overridden by the env var; the
# runner measures it per seed with --calibrate-organ and passes the measured value explicitly.
DEFAULT_VALENCE_FS = 0.5

# contrast sentences for the mouth's own affect axis. Built ONLY from WARRINER-normed words (the boosted set the
# independent scoring lexicon is DISJOINT from) + stop/neutral words, so the steering axis cannot be derived from
# any word the tone ruler scores. The runner's selftest enforces the disjointness.
CONTRAST_POS = [
    "I feel happy and glad today.",
    "Everything is wonderful and I love it.",
    "What a good day, I am so pleased and delighted.",
    "I am cheerful, hopeful and full of joy.",
    "This is nice, fun and good, and it makes me smile and laugh.",
    "I feel safe, warm and happy with my kind friends.",
]
CONTRAST_NEG = [
    "I feel sad and unhappy today.",
    "Everything is terrible and I hate it.",
    "What an awful day, I am so upset and miserable.",
    "I am afraid, hopeless and full of grief.",
    "This is horrible, ugly and bad, and it makes me cry.",
    "I feel lonely, sick and angry without my friends.",
]

_AXIS_CACHE: dict = {}
_AXIS_LOCK = threading.Lock()
LAST_TRACE: dict = {}          # the most recent turn's conditioning trace (read in-process by the runner)


def affect_conditioned_mode() -> str:
    """'' (off, the default), 'prompt', or 'resid'. Unknown values read as off."""
    v = os.environ.get("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", "").strip().lower()
    return v if v in ("prompt", "resid") else ""


def valence_fs() -> float:
    try:
        fs = float(os.environ.get("BRAIN_AFFECT_COND_VALENCE_FS", "") or DEFAULT_VALENCE_FS)
    except ValueError:
        fs = DEFAULT_VALENCE_FS
    return fs if fs > 0 else DEFAULT_VALENCE_FS


def conditioning_scalar(valence: float) -> float:
    """c = clip(valence / VALENCE_FS, -1, 1). valence == 0 (lesion) -> exactly 0.0."""
    v = float(valence)
    if v == 0.0:
        return 0.0
    return max(-1.0, min(1.0, v / valence_fs()))


# ── mode 'prompt': graded, dead-zone-free MOOD line ─────────────────────────────────────────────────────────────
_GRADES = ((0.2, "faintly"), (0.4, "slightly"), (0.6, "moderately"), (0.8, "clearly"), (1.01, "intensely"))


def graded_mood_phrase(c: float, arousal: float) -> str:
    """Monotone in |c|, no dead zone: any nonzero c gets a signed descriptor. c == 0 -> the production neutral
    wording. Descriptors follow the production `_mood_phrase` vocabulary EXCEPT 'upbeat' -> 'buoyant': 'upbeat' is a
    scored word of the independent tone lexicon, and a conditioning word the mouth could echo must not be one the
    ruler scores (anti-circularity; the runner's selftest enforces disjointness of every descriptor)."""
    if c == 0.0:
        base = "even and steady"
    else:
        adv = next(a for (hi, a) in _GRADES if abs(c) <= hi)
        base = ("%s warm, buoyant and engaged" if c > 0 else "%s subdued, curt and low-energy") % adv
    return f"{base} (mood intensity {c:+.2f}, arousal {float(arousal):.2f})"


def condition_prompt(system: str, valence: float, arousal: float) -> str:
    """Rewrite ONLY the MOOD line of the production system prompt with the graded phrase. Every other line
    (SELF/KNOWLEDGE/FAMILIARITY/CURIOSITY) is untouched, so content conditioning is identical across arms."""
    c = conditioning_scalar(valence)
    out = []
    replaced = False
    for line in system.split("\n"):
        if line.startswith("MOOD: ") and not replaced:
            line = f"MOOD: you feel {graded_mood_phrase(c, arousal)}."
            replaced = True
        out.append(line)
    return "\n".join(out)


# ── mode 'resid': representation conditioning via the mouth's own affect axis ──────────────────────────────────
def _hidden_mean(fac, text, layer):
    torch = fac._torch
    ids = fac.tok(text, return_tensors="pt").to(fac.device)
    with torch.no_grad():
        out = fac.model(**ids, output_hidden_states=True)
    # hidden_states[0] = embeddings; hidden_states[layer+1] = output of decoder layer `layer`
    return out.hidden_states[layer + 1][0].float().mean(dim=0)


def affect_axis(fac, layer=RESID_LAYER):
    """u = mean_L(pos contrast) - mean_L(neg contrast), computed ONCE per process from the mouth itself."""
    key = (id(fac), int(layer))
    if key in _AXIS_CACHE:
        return _AXIS_CACHE[key]
    with _AXIS_LOCK:
        if key not in _AXIS_CACHE:
            torch = fac._torch
            mp = torch.stack([_hidden_mean(fac, t, layer) for t in CONTRAST_POS]).mean(dim=0)
            mn = torch.stack([_hidden_mean(fac, t, layer) for t in CONTRAST_NEG]).mean(dim=0)
            _AXIS_CACHE[key] = (mp - mn)
    return _AXIS_CACHE[key]


@contextlib.contextmanager
def resid_conditioning(fac, valence: float, layer=RESID_LAYER, k=RESID_K):
    """Add c*K*u to decoder layer `layer`'s output for the duration of the block. c == 0 -> no hook at all."""
    c = conditioning_scalar(valence)
    LAST_TRACE.clear()
    LAST_TRACE.update({"mode": "resid", "valence_in": float(valence), "c": c, "valence_fs": valence_fs(),
                       "layer": int(layer), "k": float(k), "hook_registered": False, "axis_norm": None})
    if c == 0.0:
        yield LAST_TRACE
        return
    u = affect_axis(fac, layer)
    LAST_TRACE["axis_norm"] = float(u.norm().item())
    model_dtype = next(fac.model.parameters()).dtype
    vec = (float(c) * float(k) * u).to(dtype=model_dtype, device=fac.device)
    calls = {"n": 0}

    def _hook(_mod, _inp, output):
        calls["n"] += 1
        if isinstance(output, tuple):
            return (output[0] + vec,) + tuple(output[1:])
        return output + vec

    h = fac.model.model.layers[int(layer)].register_forward_hook(_hook)
    LAST_TRACE["hook_registered"] = True
    try:
        yield LAST_TRACE
    finally:
        h.remove()
        LAST_TRACE["hook_calls"] = calls["n"]


def generate_conditioned(gen, system: str, user: str, valence: float, arousal: float, *, seed: int,
                         max_new_tokens: int):
    """The one entry point `answer_turn` calls when the mode is on and the Qwen one-shot path is taken."""
    mode = affect_conditioned_mode()
    if mode == "prompt":
        system2 = condition_prompt(system, valence, arousal)
        LAST_TRACE.clear()
        LAST_TRACE.update({"mode": "prompt", "valence_in": float(valence), "c": conditioning_scalar(valence),
                           "valence_fs": valence_fs(), "mood_line": next(
                               (ln for ln in system2.split("\n") if ln.startswith("MOOD: ")), None)})
        return gen.generate(system2, user, seed=seed, max_new_tokens=max_new_tokens)
    if mode == "resid":
        with resid_conditioning(gen.fac, valence):
            return gen.generate(system, user, seed=seed, max_new_tokens=max_new_tokens)
    LAST_TRACE.clear()
    return gen.generate(system, user, seed=seed, max_new_tokens=max_new_tokens)
