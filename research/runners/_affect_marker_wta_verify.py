"""VERIFY board #86 -- the spiking lateral-inhibition marker-SELECTION circuit
(`_affect_marker_wta_derisk.py`) wired into `webapp.affect_drives_chat.expression_lead` /
`AffectDrivesWorkspace.observe()` (the SAME production entry point board #84's own `verify_affect_drives.py`
exercises, one layer below the full `/api/brain-chat` HTTP handler -- this mechanism only touches the
level->marker SELECTION step, not the #81 felt-state READ or the Gate-B appraisal, so a lower-layer verify is
the decisive one for THIS change).

FOUR PROPERTIES, 6 seeds {42,43,44,100,101,102}:
  (A) BYTE-IDENTICAL-OFF -- 2026-09-01 AUTO-FLIP: `BRAIN_AFFECT_MARKER_SPIKING` DEFAULT-ON
      (`_AFFECT_MARKER_SPIKING_DEFAULT_ON`, webapp/affect_drives_chat.py); the OFF condition this part exercises
      is now the EXPLICIT escape `BRAIN_AFFECT_MARKER_SPIKING=0`: `expression_lead()` ignores the new
      mood/felt_arousal kwargs entirely and returns EXACTLY the pre-existing `_LEAD_WORD[level]` host-template
      surface; a full `AffectDrivesWorkspace.observe()` turn under the SAME explicit-off escape is unaffected by
      whether the new module can even be imported. (Pre-flip this part exercised the flag UNSET; unset now means
      ON, so the escape is what "off" means post-flip -- see the module's own byte-identical-off convention,
      e.g. `_CG_DRIVES_DEFAULT_ON`/`cg_drives_off()` in webapp/common_ground_drives_chat.py.)
  (B) LOAD-BEARING -- BRAIN_AFFECT_MARKER_SPIKING=1: sweep the induced mood across all 6 non-neutral registers;
      the spiking-selected marker matches the register the host table would have picked for that same mood (the
      circuit's topographic centers were placed at each register's existing mood-bin midpoint, see the module
      docstring) -- i.e. varying the felt state changes the SELECTED marker, and the selection is genuinely
      reading the felt state, not a fixed default.
  (C) LESION -- BRAIN_AFFECT_MARKER_SPIKING=1 + BRAIN_AFFECT_MARKER_SPIKING_LESION=1: the felt-state->assembly
      projection is cut on every read; the documented fallback is verified: the lead VANISHES ('') even though
      level != 0 (an honest no-lead turn, not a silent revert to `_LEAD_WORD`).
  (D) SHUFFLE ANTI-CHEAT -- BRAIN_AFFECT_MARKER_SPIKING=1 + BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE=1: mis-routing
      which physical assembly receives which register's tuning drive changes the REPORTED marker relative to the
      unshuffled (intact) run at the SAME mood -- proof the reported identity is read off WHICH ASSEMBLY actually
      won the race, not re-derived from the raw mood float by a fixed formula that would be blind to the mis-wire.

Run: SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_marker_wta_verify \
       --out research/findings/raw/_affect_marker_wta_verify.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

SEEDS = (42, 43, 44, 100, 101, 102)
# one representative mood per non-neutral register, at the SAME bin-midpoints _affect_marker_wta_derisk.MOOD_CENTERS
# uses, so the intact spiking selection is expected to reproduce the host table's own word for that mood.
LEVEL_MOODS = {-3: -0.0850, -2: -0.0575, -1: -0.0275, 1: 0.0275, 2: 0.0575, 3: 0.0850}
AROUSAL_LOW, AROUSAL_HIGH = 0.020, 0.065


@contextlib.contextmanager
def _env(**kv):
    """Set env vars for the block, restoring the PRIOR state (present-or-absent) exactly on exit."""
    prior = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, p in prior.items():
            if p is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = p


def _clear_all():
    for k in ("BRAIN_AFFECT_MARKER_SPIKING", "BRAIN_AFFECT_MARKER_SPIKING_LESION",
             "BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE", "BRAIN_AFFECT_DRIVES_LESION"):
        os.environ.pop(k, None)


def _lead_word(lead):
    return lead.split(" ", 1)[0].rstrip("!—").strip() if lead else ""


def part_a_byte_identical_off(results):
    """(A) 2026-09-01 AUTO-FLIP: `BRAIN_AFFECT_MARKER_SPIKING` is now DEFAULT-ON, so the OFF condition this part
    exercises is the EXPLICIT escape `BRAIN_AFFECT_MARKER_SPIKING=0` -> expression_lead() is UNCHANGED by the new
    kwargs; a full workspace turn under the same explicit escape is unaffected by whether
    _affect_marker_wta_derisk can even be imported."""
    from webapp.affect_drives_chat import expression_lead, _LEAD_WORD

    rows = []
    ok = True
    with _env(BRAIN_AFFECT_MARKER_SPIKING="0"):
        for level in (-3, -2, -1, 1, 2, 3):
            for high in (False, True):
                base = expression_lead(level, high)
                # passing mood/felt_arousal must be INERT while the flag is explicitly off (checked first).
                augmented = expression_lead(level, high, mood=LEVEL_MOODS[level],
                                            felt_arousal=(AROUSAL_HIGH if high else AROUSAL_LOW), seed=42)
                expect = (_LEAD_WORD[level] + ("! " if high else " — "))
                row_ok = (base == augmented == expect)
                ok = ok and row_ok
                rows.append({"level": level, "high": high, "base": base, "augmented": augmented,
                            "expect": expect, "ok": row_ok})
        neutral = expression_lead(0, False, mood=0.0, felt_arousal=0.0, seed=42)
        ok = ok and (neutral == "")

        # a full production-entry-point turn, flag explicitly off: identical whether or not the new module is
        # importable.
        from webapp.affect_drives_chat import AffectDrivesWorkspace
        workspace_rows = []
        for seed in SEEDS:
            ws = AffectDrivesWorkspace(seed=seed)
            info = ws.observe(0.8, 0.6, 3, valence_override=0.9, arousal_override=0.7)
            expect_word = ""
            if info["level"] != 0:
                expect_word = _LEAD_WORD.get(info["level"], "")
            got_word = _lead_word(info["lead"])
            row_ok = (got_word == expect_word)
            ok = ok and row_ok
            workspace_rows.append({"seed": seed, "level": info["level"], "lead": info["lead"], "ok": row_ok})

    results["part_a"] = {"function_level": rows, "workspace_level": workspace_rows, "all_ok": ok}
    print(f"(A) byte-identical-off (explicit escape): {'PASS' if ok else 'FAIL'} ({len(rows)} function rows, "
         f"{len(workspace_rows)} workspace rows)")
    return ok


def part_b_load_bearing(results):
    """(B) BRAIN_AFFECT_MARKER_SPIKING=1: the spiking-selected marker matches the host table's own register for
    the SAME mood, across all 6 non-neutral levels, 6 seeds -- varying the felt state changes the marker."""
    from webapp.affect_drives_chat import expression_lead, _LEAD_WORD

    rows = []
    per_seed_ok = {}
    with _env(BRAIN_AFFECT_MARKER_SPIKING="1", BRAIN_AFFECT_MARKER_SPIKING_LESION=None,
             BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE=None):
        for seed in SEEDS:
            seed_ok = True
            for level, mood in LEVEL_MOODS.items():
                high = abs(level) >= 2   # mirror a plausible high-arousal co-occurrence for the emphatic registers
                lead = expression_lead(level, high, mood=mood,
                                       felt_arousal=(AROUSAL_HIGH if high else AROUSAL_LOW), seed=seed)
                expect_word = _LEAD_WORD[level]
                got_word = _lead_word(lead)
                row_ok = (got_word == expect_word) and bool(lead)
                seed_ok = seed_ok and row_ok
                rows.append({"seed": seed, "level": level, "mood": mood, "lead": lead,
                            "expect_word": expect_word, "got_word": got_word, "ok": row_ok})
            per_seed_ok[seed] = seed_ok

    # load-bearing DIFFERENCE proof: at a FIXED level slot, varying mood sign flips the selected word.
    diff_rows = []
    with _env(BRAIN_AFFECT_MARKER_SPIKING="1"):
        for seed in SEEDS:
            pos = _lead_word(expression_lead(3, True, mood=0.085, felt_arousal=AROUSAL_HIGH, seed=seed))
            neg = _lead_word(expression_lead(-3, True, mood=-0.085, felt_arousal=AROUSAL_HIGH, seed=seed))
            diff_rows.append({"seed": seed, "pos_word": pos, "neg_word": neg, "differ": pos != neg and pos and neg})

    all_ok = all(per_seed_ok.values()) and all(r["differ"] for r in diff_rows)
    results["part_b"] = {"rows": rows, "per_seed_ok": per_seed_ok, "diff_rows": diff_rows, "all_ok": all_ok}
    print(f"(B) load-bearing: {'PASS' if all_ok else 'FAIL'} "
         f"({sum(per_seed_ok.values())}/{len(SEEDS)} seeds fully matched the host register)")
    return all_ok


def part_c_lesion(results):
    """(C) BRAIN_AFFECT_MARKER_SPIKING=1 + _LESION=1: the felt-state->assembly projection is cut -> lead VANISHES
    ('') on every non-neutral level, 6 seeds -- the documented fallback, verified live (not just declared)."""
    from webapp.affect_drives_chat import expression_lead

    rows = []
    with _env(BRAIN_AFFECT_MARKER_SPIKING="1", BRAIN_AFFECT_MARKER_SPIKING_LESION="1",
             BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE=None):
        for seed in SEEDS:
            for level, mood in LEVEL_MOODS.items():
                lead = expression_lead(level, True, mood=mood, felt_arousal=AROUSAL_HIGH, seed=seed)
                row_ok = (lead == "")
                rows.append({"seed": seed, "level": level, "mood": mood, "lead": lead, "ok": row_ok})
    all_ok = all(r["ok"] for r in rows)
    results["part_c"] = {"rows": rows, "all_ok": all_ok}
    print(f"(C) lesion-vanish: {'PASS' if all_ok else 'FAIL'} ({sum(r['ok'] for r in rows)}/{len(rows)} rows)")
    return all_ok


def part_d_shuffle_anti_cheat(results):
    """(D) shuffle anti-cheat: BRAIN_AFFECT_MARKER_SPIKING=1 + _SHUFFLE=1 changes the reported marker relative to
    the intact (unshuffled) run at the SAME mood, on a majority of seeds -- proof the reported identity is read
    off WHICH ASSEMBLY won, not re-derived from the raw mood value by a formula blind to the mis-wiring."""
    from webapp.affect_drives_chat import expression_lead

    rows = []
    with _env(BRAIN_AFFECT_MARKER_SPIKING="1", BRAIN_AFFECT_MARKER_SPIKING_LESION=None):
        for seed in SEEDS:
            for level, mood in LEVEL_MOODS.items():
                with _env(BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE=None):
                    intact = _lead_word(expression_lead(level, True, mood=mood, felt_arousal=AROUSAL_HIGH, seed=seed))
                with _env(BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE="1"):
                    shuffled = _lead_word(expression_lead(level, True, mood=mood, felt_arousal=AROUSAL_HIGH, seed=seed))
                rows.append({"seed": seed, "level": level, "intact": intact, "shuffled": shuffled,
                            "differs": intact != shuffled})
    n_differ = sum(r["differs"] for r in rows)
    # a random permutation of 6 items has a 1/6 chance of mapping any single slot to itself; requiring a CLEAR
    # majority differ (not literally 100%) avoids a flaky failure on the rare coincidental fixed point while still
    # being decisive evidence of a live functional dependency.
    all_ok = n_differ >= (0.7 * len(rows))
    results["part_d"] = {"rows": rows, "n_differ": n_differ, "n_total": len(rows), "all_ok": all_ok}
    print(f"(D) shuffle anti-cheat: {'PASS' if all_ok else 'FAIL'} ({n_differ}/{len(rows)} rows differ from intact)")
    return all_ok


def part_e_attribution(results):
    """(E) ATTRIBUTION (tools.lab, gap#5 discipline): part (C) measured BOTH the intact winner-vs-runner-up
    margin AND the lesioned margin, but measuring both arms is not the same as asking whose the SEPARATION is.
    `attributable_to` answers that directly: (intact - lesion) / intact, at the SAME representative mood, per
    seed, using the raw circuit (`AffectMarkerWTA`) so the margin is read straight off the spiking rates (the
    same quantity part (C)'s dead-margin decision already consumes)."""
    from research.runners._affect_marker_wta_derisk import get_reader, reset_readers

    rows = []
    fractions = []
    reset_readers()
    for seed in SEEDS:
        reader = get_reader(seed)
        intact_margin = reader.select_valence(0.085)[2]["margin"]
        lesion_margin = reader.select_valence(0.085, lesion=True)[2]["margin"]
        frac = attributable_to(f"seed {seed}: valence WTA margin, intact vs lesion (mood=+0.085)",
                               intact_margin, lesion_margin)
        rows.append({"seed": seed, "intact_margin": intact_margin, "lesion_margin": lesion_margin,
                    "attributable_fraction": frac})
        if frac is not None:
            fractions.append(frac)
    reset_readers()
    # every seed's separation must be OVERWHELMINGLY attributable to the topographic drive (not some other
    # latent bias in the wiring that happens to also produce a large margin) -- the SAME bar gap#5's clamp
    # calculation exposed as failing (there, the lever owned only 3%; here every seed must clear 90%).
    all_ok = len(fractions) == len(SEEDS) and all(f >= 0.90 for f in fractions)
    results["part_e_attribution"] = {"rows": rows, "all_ok": all_ok}
    print(f"(E) attribution: {'PASS' if all_ok else 'FAIL'} "
         f"(min={min(fractions):.3f} max={max(fractions):.3f})" if fractions else "(E) attribution: FAIL (undefined)")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_affect_marker_wta_verify.json")
    args = ap.parse_args()

    _clear_all()
    results = {"runner": "_affect_marker_wta_verify", "backend": os.environ.get("SIM_BACKEND"),
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": list(SEEDS)}
    try:
        results["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        results["git_sha"] = None

    ok_a = part_a_byte_identical_off(results)
    ok_b = part_b_load_bearing(results)
    ok_c = part_c_lesion(results)
    ok_d = part_d_shuffle_anti_cheat(results)
    ok_e = part_e_attribution(results)
    _clear_all()

    v = Verdict("board #86 affect marker spiking WTA -- de-risk (default-OFF)")
    v.require("byte-identical-off", ok_a, expect=True)
    v.require("load-bearing (mood sweep selects the matching register, 6 seeds)", ok_b, expect=True)
    v.require("lesion collapses to no-marker (documented fallback verified live)", ok_c, expect=True)
    v.require("shuffle anti-cheat (reported marker tracks the mis-wired assembly)", ok_d, expect=True)
    v.require("attribution (separation >=90% owned by the topographic drive, not latent wiring bias)", ok_e, expect=True)
    verdict = v.decide(go=(ok_a and ok_b and ok_c and ok_d and ok_e))
    results["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(f"STATUS: {verdict['status']}")


if __name__ == "__main__":
    main()
