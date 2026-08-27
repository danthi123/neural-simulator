"""CONFIRM + FIX-VERIFY: the affect-tone-coloring / spiking-mouth-recall cross-faculty regression (2026-08-27).

BACKGROUND. `spiking-mouth-recall` (`BRAIN_SPIKING_MOUTH_RECALL`, default-ON 2026-08-26) and `affect-coloring`'s
manner effect (`BRAIN_AFFECT`'s `MoodConditionedRenderer`, default-ON 2026-08-12) are structurally incompatible:
`ChatBrain.spiking_recall_surface` (`brain_chat_tui.py`) returns the brain-native spiking-Broca surface EARLY,
BEFORE `self.renderer.render_svo` (where `MoodConditionedRenderer` injects the mood manner clause) is ever
reached. For every bounded-transitive recall the spiking mouth handles, the affect-coloring manner effect never
runs -- a validated cross-faculty coupling made inert by a later flip (the "faculties must DRIVE not observe"
failure mode).

PART A -- CONFIRM (crossed A/B: mood {positive, negative} x BRAIN_SPIKING_MOUTH_RECALL {1, 0}). Runs the REAL
`ChatBrain.render` / `spiking_recall_surface` / `spiking_mouth_recall_prod.recall_mouth_enabled` /
`affect_production_organ.MoodConditionedRenderer` / `manner_template_for` UNMODIFIED. Two things are faked for
CPU-cheapness (no GPU / Qwen model load / heavy sim, per the "no heavy sims" operating constraint):
  (1) the fluent-mouth base the MoodConditionedRenderer wraps (a deterministic stand-in exposing the same
      `_fac._generate` / `CONSTRAIN_TEMPLATE` shape MoodConditionedRenderer._fac() requires; its text visibly
      depends on the manner clause, mirroring the real Qwen mouth's behaviour in the 2026-08-12 GateB finding);
  (2) `ChatBrain._verify` is monkeypatched to always accept, isolating the ROUTING/dispatch question (does the
      manner-colored renderer ever get CALLED?) from the MOAT question (already adversarially verified
      elsewhere) -- both arms funnel through the SAME patched call, so it cannot manufacture the asymmetry
      measured (call-count / text-identity).

PART B -- FIX VERIFY (6 seeds, the REAL unmodified `_verify`, no monkeypatch). The fix threads the SAME Gate-B
mood LEVEL onto the spiking mouth's OWN surface via a NEW `ChatBrain._mood_tone_level` attribute (set by
webapp.server's existing affect block, alongside `wrapped.manner`) and a tiny 2-pool spiking tone-read
(`spiking_mouth_recall_prod.mouth_tone_marker` / `mouth_mood_enabled`). Per seed in {42,43,44,100,101,102}:
  (1) NEUTRAL byte-identity -- `_mood_tone_level` unset/0 (every caller that never sets it: the TUI, unit tests,
      BRAIN_AFFECT=0) -> surface UNCHANGED from the pre-fix spiking-mouth surface.
  (2) LOAD-BEARING -- level=+2 vs level=-2 -> surfaces DIFFER, sign-correct ('!' warm / '.' curt).
  (3) LESION -- `BRAIN_SPIKING_MOUTH_MOOD=0` -> both signs REVERT to the exact neutral surface.
Deliberately on the lightweight 'rf' composer (a few thousand neurons total), NOT the heavy ~46-region/4180-
neuron 'onebrain' composer `_spiking_mouth_recall_soak.py` normally exercises -- the machine was under severe
memory pressure during this arc (39/46 GiB swap in use from concurrent runs) and a heavier soak was avoided per
the explicit "no heavy sims" constraint on this task.

Run: SIM_BACKEND=numpy python -m research.runners._affect_tone_spiking_mouth_fix_verify \
       --out research/findings/raw/_affect_tone_spiking_mouth_fix_verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.verdict import Verdict  # noqa: E402

_SEEDS = [42, 43, 44, 100, 101, 102]
_FACTS = [("dog", "chase", "cat"), ("cat", "eat", "fish")]


def _build_chat(seed):
    from research.runners.brain_chat_tui import ChatBrain, DEFAULT_SELF_ALIASES
    from research.runners.multi_turn_agent import MultiTurnAgent
    vocab = sorted({w for f in _FACTS for w in f})
    actions = {v for _a, v, _p in _FACTS}
    referents = [w for w in vocab if w not in actions]
    agent = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in vocab}, seed=seed,
                           enable_neural_render=False, composer_kind="rf",
                           enable_biased_competition=False, defer_planner=True, event_register=None)
    inner = getattr(agent, "agent", agent)
    for a, v, p in _FACTS:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    # renderer must be non-None (spiking_recall_surface early-returns on `self.renderer is None`, the raw-mode
    # guard); the real object is installed by each part below.
    return ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=object())


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# PART A -- CONFIRM the regression (crossed A/B, seed 42, real routing code, faked GPU-free fluent mouth).
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class _FakeQwenLikeFac:
    """Stands in for SpikingQwenFaculty. Same `_generate`/`CONSTRAIN_TEMPLATE` shape MoodConditionedRenderer._fac()
    requires; the returned prose visibly encodes whether a warm/curt manner clause was present in the prompt
    (mirrors the real Qwen mouth's behaviour in the 2026-08-12 GateB finding), at zero GPU/model-load cost."""
    CONSTRAIN_TEMPLATE = "SVO: {a} {v} {p}."

    def __init__(self):
        self.calls = 0

    def _generate(self, prompt):
        self.calls += 1
        if "warm" in prompt.lower():
            surface = "The dog happily chases the cat!"
        elif "blunt" in prompt.lower():
            surface = "The dog chases the cat."
        else:
            surface = "The dog chases the cat (neutral)."
        return surface, surface, 0.001


class _FakeBaseRenderer:
    name = "fake-fluent-mouth (probe stand-in for off-bridge Qwen; CPU, no GPU/model load)"

    def __init__(self):
        self._fac = _FakeQwenLikeFac()

    def render_svo(self, a, v, p):
        return f"The {a} {v}s the {p} (fake base, no manner).", None


def part_a_confirm():
    from research.runners.affect_production_organ import MoodConditionedRenderer, manner_template_for
    from research.runners.spiking_mouth_recall_prod import recall_mouth_enabled

    chat = _build_chat(42)
    chat._verify = lambda surface, asserted, gate_svo: True   # isolate ROUTING from the (separately-proven) moat
    base = _FakeBaseRenderer()
    chat.renderer = base

    def server_affect_block(level):
        """Mirrors webapp/server.py's actual Gate-B block (both branches set from the SAME `level`)."""
        chat._mood_tone_level = 0
        manner_tmpl = manner_template_for(level)
        base_r = getattr(chat, "renderer", None)
        if base_r is not None and not isinstance(base_r, MoodConditionedRenderer):
            chat.renderer = MoodConditionedRenderer(base_r)
        wrapped = getattr(chat, "renderer", None)
        if isinstance(wrapped, MoodConditionedRenderer):
            wrapped.manner = manner_tmpl
        chat._mood_tone_level = int(level)

    def run(flag_on, level, tag):
        os.environ["BRAIN_SPIKING_MOUTH_RECALL"] = "1" if flag_on else "0"
        server_affect_block(level)
        base._fac.calls = 0
        surface = chat.render(["dog", "chase", "cat"])
        return {"label": tag, "flag_on": flag_on, "mood_level": level,
                "recall_mouth_enabled": recall_mouth_enabled(), "qwen_like_fac_calls": base._fac.calls,
                "surface": surface}

    rows = [run(flag_on, level, tag)
            for flag_on in (False, True)
            for level, tag in ((+2, "positive"), (-2, "negative"))]

    off = {r["label"]: r for r in rows if not r["flag_on"]}
    on = {r["label"]: r for r in rows if r["flag_on"]}
    # NOTE: this repo state IS post-fix, so `on_differs_post_fix` below is expected True. The regression itself
    # (flag-ON rows IDENTICAL regardless of mood) was separately confirmed by the identical crossed-A/B probe run
    # BEFORE the fix landed -- see the finding doc for that pre-fix transcript.
    return {
        "rows": rows,
        "off_differs": off["positive"]["surface"] != off["negative"]["surface"],
        "off_manner_reached": off["positive"]["qwen_like_fac_calls"] >= 1 and off["negative"]["qwen_like_fac_calls"] >= 1,
        "on_differs_post_fix": on["positive"]["surface"] != on["negative"]["surface"],
        "on_never_reaches_qwen": on["positive"]["qwen_like_fac_calls"] == 0 and on["negative"]["qwen_like_fac_calls"] == 0,
    }


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# PART B -- FIX VERIFY (6 seeds, real unmodified `_verify`, the lightweight 'rf' composer).
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def part_b_fix_verify():
    rows = []
    for seed in _SEEDS:
        os.environ["BRAIN_SPIKING_MOUTH_RECALL"] = "1"
        chat = _build_chat(seed)

        neutral = chat.spiking_recall_surface("dog", "chase", "cat")

        os.environ.pop("BRAIN_SPIKING_MOUTH_MOOD", None)
        chat._mood_tone_level = 2
        surf_pos = chat.spiking_recall_surface("dog", "chase", "cat")
        chat._mood_tone_level = -2
        surf_neg = chat.spiking_recall_surface("dog", "chase", "cat")

        os.environ["BRAIN_SPIKING_MOUTH_MOOD"] = "0"
        chat._mood_tone_level = 2
        surf_pos_lesion = chat.spiking_recall_surface("dog", "chase", "cat")
        chat._mood_tone_level = -2
        surf_neg_lesion = chat.spiking_recall_surface("dog", "chase", "cat")
        os.environ.pop("BRAIN_SPIKING_MOUTH_MOOD", None)

        rows.append({
            "seed": seed, "neutral": neutral, "pos": surf_pos, "neg": surf_neg,
            "pos_lesion": surf_pos_lesion, "neg_lesion": surf_neg_lesion,
            "content_ok": bool(neutral and neutral.startswith("the dog chase") and "cat" in neutral),
            "load_bearing_ok": bool(surf_pos != surf_neg and neutral and
                                    surf_pos == neutral + "!" and surf_neg == neutral + "."),
            "lesion_ok": bool(surf_pos_lesion == neutral and surf_neg_lesion == neutral),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# PART C -- the `_TONE_DEAD_MARGIN` TUNING measurement (the smallest-magnitude case, |level|==1, across all 6
# seeds), so the constant chosen in `spiking_mouth_recall_prod._TONE_DEAD_MARGIN` is traceable to a cited number
# rather than asserted in prose.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def part_c_dead_margin_tuning():
    from research.runners._emerge59_spiking_broca_frame_slots_derisk import (
        build_slot_bridge, slot_pool_rates, PRIMACY_pA, EQUAL_pA)
    from research.runners.spiking_mouth_recall_prod import _TONE_WARM, _TONE_CURT, _TONE_N_POOLS, _TONE_DEAD_MARGIN

    rows = []
    for seed in _SEEDS:
        bridge, idx = build_slot_bridge(seed, n_slot_pools=_TONE_N_POOLS)
        for lvl in (1, -1):
            mag = min(1.0, abs(lvl) / 3.0)
            driven_pA = EQUAL_pA + mag * (PRIMACY_pA[0] - EQUAL_pA)
            if lvl > 0:
                drive = {_TONE_WARM: driven_pA, _TONE_CURT: EQUAL_pA}
            else:
                drive = {_TONE_WARM: EQUAL_pA, _TONE_CURT: driven_pA}
            rate = slot_pool_rates(bridge, idx, drive, n_slot_pools=_TONE_N_POOLS)
            warm_r, curt_r = float(rate[_TONE_WARM]), float(rate[_TONE_CURT])
            sep = abs(warm_r - curt_r)
            rows.append({"seed": seed, "level": lvl, "warm_rate": warm_r, "curt_rate": curt_r,
                        "abs_separation": sep, "sign_correct": (warm_r > curt_r) == (lvl > 0),
                        "clears_dead_margin": sep > _TONE_DEAD_MARGIN})
    min_sep = min(r["abs_separation"] for r in rows)
    max_sep = max(r["abs_separation"] for r in rows)
    return {"dead_margin_used": _TONE_DEAD_MARGIN, "rows": rows,
           "min_abs_separation_at_|level|=1": min_sep, "max_abs_separation_at_|level|=1": max_sep,
           "all_sign_correct": all(r["sign_correct"] for r in rows),
           "all_clear_dead_margin": all(r["clears_dead_margin"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        _REPO, "research", "findings", "raw", "_affect_tone_spiking_mouth_fix_verify.json"))
    args = ap.parse_args()

    a = part_a_confirm()
    b_rows = part_b_fix_verify()
    c = part_c_dead_margin_tuning()

    v = Verdict("affect-tone-coloring restored on the spiking recall mouth (2026-08-27 fix)")
    v.require("PART A: flag-OFF (Qwen path) mood-manner reaches the renderer", a["off_manner_reached"], expect=True)
    v.require("PART A: flag-OFF (Qwen path) positive vs negative differ", a["off_differs"], expect=True)
    v.require("PART A: flag-ON (spiking mouth, post-fix) positive vs negative differ", a["on_differs_post_fix"],
              expect=True)
    v.require("PART A: flag-ON never falls back to the Qwen-like mouth", a["on_never_reaches_qwen"], expect=True)
    for row in b_rows:
        v.require(f"PART B seed={row['seed']}: neutral content recovers the recalled SVO", row["content_ok"],
                  expect=True)
        v.require(f"PART B seed={row['seed']}: mood is load-bearing on the spiking mouth's own surface",
                  row["load_bearing_ok"], expect=True)
        v.require(f"PART B seed={row['seed']}: BRAIN_SPIKING_MOUTH_MOOD=0 lesion reverts to byte-identical neutral",
                  row["lesion_ok"], expect=True)
    v.require("PART C: the tone read is sign-correct on every seed at the smallest drive (|level|=1)",
              c["all_sign_correct"], expect=True)
    v.require("PART C: _TONE_DEAD_MARGIN clears on every seed at the smallest drive (|level|=1)",
              c["all_clear_dead_margin"], expect=True)

    go = (a["off_differs"] and a["off_manner_reached"] and a["on_differs_post_fix"] and a["on_never_reaches_qwen"]
          and all(r["content_ok"] and r["load_bearing_ok"] and r["lesion_ok"] for r in b_rows)
          and c["all_sign_correct"] and c["all_clear_dead_margin"])
    result = v.decide(go=go)
    result["part_a_confirm"] = a
    result["part_b_fix_verify"] = b_rows
    result["part_c_dead_margin_tuning"] = c
    result["seeds"] = _SEEDS

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nwrote %s" % args.out)
    return result["status"]


if __name__ == "__main__":
    status = main()
    raise SystemExit(0 if status == "GO" else 1)
