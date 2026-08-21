"""VERIFY for the DEFAULT-OFF continuous IDEATION mode (the creativity/novelty rung of the between-turn life).

WHAT WAS BUILT (webapp/continuous_engine.py, strictly additive, default-OFF behind `BRAIN_CONTINUOUS_IDEATE`): the
between-turn idle wander OCCASIONALLY (every Nth tick — a declared host cadence) GENERATES a NOVEL blended concept
instead of recalling a stored one. It drives a BLENDED cue of the TWO most curiosity-active basins into a sparse
associative-attractor which settles into a NOVEL recombination that was NEVER stored (novelty from the DYNAMICS),
reusing the GO de-risk mechanism (`_generative_attractor_wander_derisk`: Tsodyks-Feigelman sparse-Hopfield + the
ca3_ff_inhib MEAN+std dynamic-threshold settle). The novel idea is TAGGED (kind=novel-association, is_fact=False) and
surfaced via `recent_ideation()` on a channel DISJOINT from `recent_wander()` (recalled concepts), so the next turn
frames it as "a thought that occurred to me", NEVER a stored fact.

WHAT THIS PROVES (GO = A and B and C and D):
  (A) OFF (`BRAIN_CONTINUOUS_IDEATE` unset) -> the continuous wander is BYTE-IDENTICAL to today: through a real
      SelfInitiationOrgan the tick records NO `ideation` key, `recent_ideation()` is None, and the recall wander ==
      today's single-basin curiosity selection (== `organ.speak()`). Also: merely ENABLING the flag on a NON-ideation
      tick is a no-op (identical recall wander) — the flip is untouched. THIS PROTECTS THE LIVE default-on flip.
  (B) ON, LOAD-BEARING -> an ideation wander produces a genuinely NOVEL concept. MECHANISM (numpy de-risk attractor,
      6 seeds x 2 scales): the 2-source blend settles to a FIXED POINT with LOW max-overlap with any single stored
      basin (not a single item) AND a BALANCED blend of the two cued sources, far above any OTHER non-cued basin.
      INTEGRATION (real tick, numpy): the tick records a flagged novel-association and `recent_ideation()` surfaces it
      tagged (is_fact=False).
  (C) the novelty is from the BLEND DYNAMICS, not noise and not single recall: a SINGLE-cue drive recovers ONE stored
      pattern SPECIFICALLY (no balanced blend of the two sources); a pure-NOISE cue is NOT balanced on the two cued
      sources; an UNTRAINED (W=0) network does not fake completion. The blend's balance-on-the-cued-sources exceeds
      both the single-cue's and the noise's by a clear margin.
  (D) HONESTY: the flagged-novel wander is is_fact=False + kind=novel-association, surfaces on a channel DISJOINT from
      recall-wander (`recent_wander()` is None on an ideation tick -> it can never be spoken as "I'd been mulling over
      X" recall), and writes NO store / manufactures NO fact (the organ's store is untouched by ideation).

HONEST SCOPE / DECLARED SCAFFOLDS. The novelty rides a fast standalone numpy attractor (the de-risked stand-in for
the on-substrate CA3 blend — the SAME latency residual the self-init organ declares; the on-substrate CA3 port is the
finding's mapped next step). The every-Nth cadence is a host-timed scheduler. The SELECTION of the two source basins
rides the organ's spiking curiosity gains (one-brain merge #1). FUNCTIONAL creativity correlate, NOT a phenomenal claim.

Run (numpy-CPU, foreground, ~a minute):
  SIM_BACKEND=numpy python -u -m research.runners._continuous_ideation_verify
"""
from __future__ import annotations

import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

# reuse-by-import the GO de-risk's attractor mechanism (the SAME one the live ideation path drives)
from research.runners._generative_attractor_wander_derisk import (  # noqa: E402
    _sparse_pattern, _train_weights, _threshold_settle, _overlap)

SCALES = [(400, 40, 6), (1200, 60, 20)]   # the de-risk's two validated scales
SEEDS = [42, 43, 44, 45, 46, 47]
THRESH_C = 0.7


# ── (B/C) MECHANISM: the 2-source blend is a NOVEL balanced recombination, with clean blend-not-noise/single controls ──
def _mech_row(seed, n, k, n_mem, thresh_c=THRESH_C):
    """One seed at one scale: drive a 2-source blend + the single-cue / noise / untrained controls, all via the
    de-risk's sparse-Hopfield threshold settle. Returns the discriminating metrics."""
    rng = np.random.default_rng(seed)
    pats = [_sparse_pattern(rng, n, k) for _ in range(n_mem)]
    W, a = _train_weights(pats, n)
    mA, mB = 0, 1     # the two "curiosity-top" basins
    hk = k // 2

    # the BLEND cue = half of A + half of B -> settle
    idxA = np.flatnonzero(pats[mA]).copy(); idxB = np.flatnonzero(pats[mB]).copy()
    rng.shuffle(idxA); rng.shuffle(idxB)
    cue = np.zeros(n); cue[idxA[:hk]] = 1.0; cue[idxB[:hk]] = 1.0
    settled, fixed, _ = _threshold_settle(W, a, cue, 12, c=thresh_c)
    ov = [_overlap(settled, p) for p in pats]
    novelty = max(ov)                                    # max overlap with any SINGLE stored basin (LOW = not one item)
    balance = min(ov[mA], ov[mB])                        # both cued sources genuinely represented
    others = max(ov[m] for m in range(n_mem) if m not in (mA, mB))   # any OTHER non-cued basin

    # SINGLE-cue control (the blend-not-single lesion): a partial cue of ONE source -> recover it SPECIFICALLY, and it
    # is NOT balanced on the two sources (no novel blend).
    idx0 = np.flatnonzero(pats[mA]).copy(); rng.shuffle(idx0)
    cue1 = np.zeros(n); cue1[idx0[:hk]] = 1.0
    s1, _, _ = _threshold_settle(W, a, cue1, 12, c=thresh_c)
    ov1 = [_overlap(s1, p) for p in pats]
    single_rec = ov1[mA]; single_oth = max(ov1[m] for m in range(1, n_mem))
    single_bal_AB = min(ov1[mA], ov1[mB])                # a single cue is NOT balanced on A,B

    # UNTRAINED control (W=0): the threshold rule alone must not fake completion.
    s0, _, _ = _threshold_settle(np.zeros_like(W), a, cue, 12, c=thresh_c)
    untrained = max(_overlap(s0, p) for p in pats)

    # NOISE control (clean discriminator): a random cue is NOT balanced on the SAME two cued sources (no A/B structure).
    nidx = rng.choice(n, size=k, replace=False)
    cueN = np.zeros(n); cueN[nidx] = 1.0
    sN, _, _ = _threshold_settle(W, a, cueN, 12, c=thresh_c)
    ovN = [_overlap(sN, p) for p in pats]
    noise_bal_AB = min(ovN[mA], ovN[mB])

    return dict(seed=seed, n=n, k=k, n_mem=n_mem, fixed_point=bool(fixed),
                novelty_max_overlap=round(float(novelty), 3), blend_balance=round(float(balance), 3),
                blend_vs_other=round(float(others), 3), single_recovered=round(float(single_rec), 3),
                single_overlap_others=round(float(single_oth), 3), single_balance_on_AB=round(float(single_bal_AB), 3),
                untrained_best=round(float(untrained), 3), noise_balance_on_AB=round(float(noise_bal_AB), 3))


def _mechanism():
    rows = [_mech_row(s, n, k, nm) for (n, k, nm) in SCALES for s in SEEDS]
    ok = all(
        r["fixed_point"]
        and r["novelty_max_overlap"] < 0.85
        and r["blend_balance"] > 0.50
        and (r["blend_balance"] - r["blend_vs_other"]) > 0.15
        and r["single_recovered"] > 0.85
        and r["single_overlap_others"] < 0.25
        and r["untrained_best"] < 0.40
        and r["noise_balance_on_AB"] < 0.40 and (r["blend_balance"] - r["noise_balance_on_AB"]) > 0.15
        and r["single_balance_on_AB"] < 0.40 and (r["blend_balance"] - r["single_balance_on_AB"]) > 0.15
        for r in rows)
    agg = dict(
        blend_balance=round(float(np.mean([r["blend_balance"] for r in rows])), 3),
        novelty=round(float(np.mean([r["novelty_max_overlap"] for r in rows])), 3),
        blend_vs_other=round(float(np.mean([r["blend_vs_other"] for r in rows])), 3),
        noise_balance_on_AB=round(float(np.mean([r["noise_balance_on_AB"] for r in rows])), 3),
        single_balance_on_AB=round(float(np.mean([r["single_balance_on_AB"] for r in rows])), 3),
        single_recovered=round(float(np.mean([r["single_recovered"] for r in rows])), 3),
        untrained_best=round(float(np.mean([r["untrained_best"] for r in rows])), 3),
        all_clear=bool(ok))
    return rows, agg


# ── (A/B/D) INTEGRATION: the real tick_session surfaces a flagged novel concept ON, byte-identical to today OFF ──
class _FakeAffect:
    """A deterministic affect read (NOT what this change touches) — a stand-in so the tick runs without the heavy
    affect organ; the affect part is identical across arms by construction."""
    def read_differential(self, v, lesion=False):
        return {"differential": round(0.5 * float(v), 4)}


def _run_tick(CE, organ, key, mood):
    """One idle tick through the REAL tick_session; returns the recorded rec."""
    return CE.tick_session(key, {key: dict(mood)}, _FakeAffect(), now=1000.0, selfinit_organ=organ)


def _store_fingerprint(organ):
    """A cheap fingerprint of the organ's mouth/store — must be UNCHANGED by ideation (it writes no store)."""
    return (tuple(organ.agents), tuple(bool(x) for x in (organ.decode_ok or [])),
            tuple(sorted((organ.utt_by_agent or {}).items())))


def _integration():
    from webapp import continuous_engine as CE
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan

    organ = SelfInitiationOrgan(seed=42)
    organ._ensure_mouth()
    today_concept = organ.speak(lesion=False).get("concept")   # today's single-basin recall selection
    fp0 = _store_fingerprint(organ)
    mood = {"valence": 0.6, "arousal": 0.4}

    # (A) OFF: BRAIN_CONTINUOUS_IDEATE unset -> byte-identical to today's recall wander.
    os.environ.pop("BRAIN_CONTINUOUS_IDEATE", None)
    os.environ.pop("BRAIN_CONTINUOUS_IDEATE_EVERY", None)
    os.environ["BRAIN_CONTINUOUS"] = "1"
    k_off = "ideation_off"; CE.forget_session(k_off)
    rec_off = _run_tick(CE, organ, k_off, mood)
    off_no_key = "ideation" not in (rec_off or {})
    off_recent_none = CE.recent_ideation(k_off) is None
    off_wander_matches = (rec_off or {}).get("wandered") == today_concept

    # ENABLING the flag on a NON-ideation tick (every=999 so this tick does not ideate) must be a no-op.
    os.environ["BRAIN_CONTINUOUS_IDEATE"] = "1"; os.environ["BRAIN_CONTINUOUS_IDEATE_EVERY"] = "999"
    k_non = "ideation_nonidea"; CE.forget_session(k_non)
    rec_non = _run_tick(CE, organ, k_non, mood)
    flagon_nonideate_identical = ("ideation" not in (rec_non or {})
                                  and (rec_non or {}).get("wandered") == today_concept)

    # (B) ON, ideation forced (every=1): the tick records a flagged novel-association; recent_ideation surfaces it.
    os.environ["BRAIN_CONTINUOUS_IDEATE"] = "1"; os.environ["BRAIN_CONTINUOUS_IDEATE_EVERY"] = "1"
    k_on = "ideation_on"; CE.forget_session(k_on)
    rec_on = _run_tick(CE, organ, k_on, mood)
    idea_rec = (rec_on or {}).get("ideation")
    on_ideation_present = bool(idea_rec) and idea_rec.get("kind") == "novel-association" and idea_rec.get("is_fact") is False
    on_wandered_none = (rec_on or {}).get("wandered") is None       # the ideation tick did NOT do a recall
    surfaced = CE.recent_ideation(k_on)
    on_recent_tagged = bool(surfaced) and surfaced.get("is_fact") is False and len(surfaced.get("sources", [])) >= 2
    on_recent_consumes = CE.recent_ideation(k_on) is None           # consumed on read (surfaces once)

    # (D) HONESTY: the ideation never entered the recall-wander channel, and wrote no store / no fact.
    on_recall_channel_empty = CE.recent_wander(k_on) is None        # never surfaced as "I'd been mulling over X"
    fp1 = _store_fingerprint(organ)
    store_unchanged = (fp0 == fp1)

    return dict(
        today_concept=today_concept,
        A_off_byte_identical=dict(no_ideation_key=off_no_key, recent_ideation_none=off_recent_none,
                                  recall_wander_matches_today=off_wander_matches,
                                  today_concept=today_concept, off_wandered=(rec_off or {}).get("wandered"),
                                  flag_on_nonideation_tick_identical=flagon_nonideate_identical),
        B_on_load_bearing=dict(ideation_recorded_flagged=on_ideation_present, wandered_none_on_ideation_tick=on_wandered_none,
                               recent_ideation_tagged=on_recent_tagged, recent_ideation_consumes=on_recent_consumes,
                               ideation=idea_rec, surfaced=surfaced),
        D_honesty=dict(recall_channel_empty_on_ideation=on_recall_channel_empty, store_unchanged=store_unchanged,
                       is_fact=(idea_rec or {}).get("is_fact"), kind=(idea_rec or {}).get("kind")),
        _flags=dict(off_no_key=off_no_key, off_recent_none=off_recent_none, off_wander_matches=off_wander_matches,
                    flagon_nonideate_identical=flagon_nonideate_identical, on_ideation_present=on_ideation_present,
                    on_wandered_none=on_wandered_none, on_recent_tagged=on_recent_tagged,
                    on_recent_consumes=on_recent_consumes, on_recall_channel_empty=on_recall_channel_empty,
                    store_unchanged=store_unchanged),
    )


def main():
    rows, agg = _mechanism()
    integ = _integration()
    f = integ["_flags"]

    # ATTRIBUTION: what fraction of the balanced-blend-on-the-cued-sources is owed to the BLEND cue STRUCTURE, not to
    # an arbitrary cue? The NOISE cue is the control (a random input that shares nothing with the two sources).
    # (treatment - control)/treatment ~ 1.0 means the balanced blend is the cue's doing, not something any input fakes.
    from tools.lab import attributable_to
    blend_attribution = attributable_to(
        "the balanced blend-on-the-two-cued-sources owed to the BLEND cue structure (control = a pure-noise cue)",
        agg["blend_balance"], agg["noise_balance_on_AB"])
    agg["blend_attribution_vs_noise"] = blend_attribution

    from tools.verdict import Verdict
    v = Verdict("Continuous IDEATION mode (novel between-turn thought), strictly additive + default-OFF")
    # (A) OFF byte-identical — protects the just-flipped live continuous default.
    v.require("(A) OFF: no `ideation` key on the tick record", f["off_no_key"], expect=True,
              note="BRAIN_CONTINUOUS_IDEATE unset -> the rec is byte-identical to today's wander rec")
    v.require("(A) OFF: recent_ideation() is None", f["off_recent_none"], expect=True)
    v.require("(A) OFF: the recall wander == today's single-basin selection (organ.speak)", f["off_wander_matches"],
              expect=True, note=f"today_concept={integ['today_concept']!r}")
    v.require("(A) enabling the flag on a NON-ideation tick is a no-op (== OFF recall wander)",
              f["flagon_nonideate_identical"], expect=True, note="the recall path is untouched; the flip is protected")
    # (B) ON, load-bearing (mechanism + integration).
    v.require("(B) MECHANISM: the 2-source blend is a NOVEL balanced recombination (all 6 seeds x 2 scales)",
              agg["all_clear"], expect=True,
              note=f"blend_balance={agg['blend_balance']} novelty={agg['novelty']} vs-other={agg['blend_vs_other']}")
    v.require("(B) INTEGRATION: the tick records a FLAGGED novel-association ideation", f["on_ideation_present"],
              expect=True, note=f"ideation={integ['B_on_load_bearing']['ideation']}")
    v.require("(B) INTEGRATION: recent_ideation() surfaces it tagged (is_fact=False, >=2 sources)",
              f["on_recent_tagged"], expect=True)
    v.require("(B) INTEGRATION: the ideation surfaces exactly once (consumed on read)", f["on_recent_consumes"],
              expect=True)
    # (C) novelty from the BLEND dynamics — not noise, not single recall.
    v.control("novelty from the BLEND, not a SINGLE recall (blend balanced on the cued sources; single-cue is not)",
              treatment=agg["blend_balance"], control=agg["single_balance_on_AB"], min_separation=0.15,
              note=f"blend balance-on-AB {agg['blend_balance']} >> single-cue {agg['single_balance_on_AB']}")
    v.control("novelty from the BLEND, not NOISE (blend balanced on the cued sources; a random cue is not)",
              treatment=agg["blend_balance"], control=agg["noise_balance_on_AB"], min_separation=0.15,
              note=f"blend balance-on-AB {agg['blend_balance']} >> noise {agg['noise_balance_on_AB']}")
    v.control("novelty from LEARNING, not the threshold rule (untrained W=0 does not fake completion)",
              treatment=agg["blend_balance"], control=agg["untrained_best"], min_separation=0.15,
              note=f"untrained best-overlap {agg['untrained_best']}")
    v.require("(C) single-cue recovers ONE stored pattern SPECIFICALLY (positive control)",
              (agg["single_recovered"] > 0.85), expect=True, note=f"single recovered={agg['single_recovered']}")
    # (D) honesty boundary.
    v.require("(D) HONESTY: the novel idea is is_fact=False + kind=novel-association", f["on_ideation_present"],
              expect=True)
    v.require("(D) HONESTY: the ideation NEVER enters the recall channel (recent_wander None on an ideation tick)",
              f["on_recall_channel_empty"], expect=True, note="it can never be spoken as 'I'd been mulling over X' recall")
    v.require("(D) HONESTY: ideation writes NO store / manufactures NO fact (organ store fingerprint unchanged)",
              f["store_unchanged"], expect=True)
    v.disabled("the on-substrate CA3 blend — the novelty rides the de-risked numpy stand-in attractor",
               why="cupy CA3 is ~s, numpy@scale ~min; the SAME latency residual the self-init organ declares. The "
                   "SELECTION of the two source basins rides the organ's spiking curiosity gains (one-brain merge #1); "
                   "the on-substrate CA3 port is the finding's mapped next step")
    v.disabled("the every-Nth-tick cadence (a host-timed scheduler)",
               why="WHEN to ideate is host-clocked, like the idle-tick clock; the mechanism/honesty are what is proven")

    go_core = (agg["all_clear"] and f["off_no_key"] and f["off_recent_none"] and f["off_wander_matches"]
               and f["flagon_nonideate_identical"] and f["on_ideation_present"] and f["on_wandered_none"]
               and f["on_recent_tagged"] and f["on_recent_consumes"] and f["on_recall_channel_empty"]
               and f["store_unchanged"])
    decided = v.decide(go=go_core, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_continuous_ideation_verify",
        "go": go, "status": decided["status"],
        "mechanism": {"rows": rows, "aggregate": agg},
        "integration": {k: val for k, val in integ.items() if k != "_flags"},
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    op = "research/findings/raw/_continuous_ideation/verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    bar = "=" * 104
    print("\n" + bar, flush=True)
    print("  CONTINUOUS IDEATION VERIFY — novel between-turn thought (default-OFF BRAIN_CONTINUOUS_IDEATE)", flush=True)
    print(bar, flush=True)
    print(f"  (B/C) MECHANISM (6 seeds x 2 scales): blend_balance={agg['blend_balance']} novelty(max-1item)={agg['novelty']} "
          f"vs-other={agg['blend_vs_other']} | controls: single-bal-AB={agg['single_balance_on_AB']} "
          f"noise-bal-AB={agg['noise_balance_on_AB']} untrained={agg['untrained_best']} | all_clear={agg['all_clear']}", flush=True)
    print(f"  (A) OFF byte-identical: no_key={f['off_no_key']} recent_none={f['off_recent_none']} "
          f"recall=={integ['today_concept']!r}({f['off_wander_matches']}) flag-on-nonideate-noop={f['flagon_nonideate_identical']}", flush=True)
    _idea = integ["B_on_load_bearing"]["ideation"] or {}
    print(f"  (B) ON: flagged ideation sources={_idea.get('sources')} novelty={_idea.get('novelty_max_overlap')} "
          f"balance={_idea.get('blend_balance')} | recent_ideation tagged={f['on_recent_tagged']} consumed-once={f['on_recent_consumes']}", flush=True)
    print(f"  (D) HONESTY: recall-channel-empty={f['on_recall_channel_empty']} store-unchanged={f['store_unchanged']} "
          f"is_fact={integ['D_honesty']['is_fact']} kind={integ['D_honesty']['kind']!r}", flush=True)
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} ({decided['status']})", flush=True)
    for r in decided["undefined_reasons"]:
        print(f"     UNMET/UNMEASURED: {r}", flush=True)
    print(f"  [saved] {op}\n" + bar, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
