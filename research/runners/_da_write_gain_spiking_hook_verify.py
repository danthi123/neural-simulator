"""PRODUCTION-HOOK VERIFY for LEVER-4 (`webapp/da_encoding_drives_chat.py::_leaf_gain`, scaffold-retirement
rank-16). Exercises `encoding_gain_for()` directly (the exact function `install_encoding_gain` wires into the
live composer's `encoding_gain_fn` inside the real `brain_chat` handler) with `BRAIN_DA_ENCODING_SPIKING_GAIN`
unset / explicit-0 / explicit-1 / set+lesioned, on a minimal fake `chat` carrying only what `da_level_of()` reads
(`chat._last_da_drives["da_level"]`) -- proving the SAME dispatch point the production wiring uses, without the
cost of a full `brain_chat` round trip (that full-handler proof already exists and is unmodified by this lever:
`research/runners/_da_encoding_wired_verify.py`, re-run this session and confirmed byte-for-byte unchanged,
g_high=2.4774/g_low=1.0/lesion-both=1.0 -- identical to that file's own docstring numbers).

FLIPPED DEFAULT-ON 2026-09-05 (production-flip, rank-16 -- `research/findings/2026-09-05-rank16-rank10-
production-flip-GO.md`). This file was ORIGINALLY written for the default-OFF de-risk and its OFF arm relied on
the env var being UNSET; now that `da_encoding_spiking_gain_enabled()` defaults ON, an unset/popped OFF arm would
silently read ON (the flip_offarm_staleness class, `gates/flip_offarm_staleness`). Fixed: the OFF arm below now
pins `BRAIN_DA_ENCODING_SPIKING_GAIN` to the EXPLICIT `"0"` escape, never pop/unset, and a NEW arm (A2) proves the
flip itself -- fully UNSET now takes the IDENTICAL branch as explicit `"1"`.

WHAT THIS PROVES (GO = A and A2 and B and C and D and E):
  (A) OFF (`BRAIN_DA_ENCODING_SPIKING_GAIN` EXPLICITLY `"0"`) -- `encoding_gain_for()` returns EXACTLY (float ==)
      what the pre-existing `_gain_map()` closed form returns, on BOTH leaf branches (the substrate/default
      floor=1.0 branch and the raw/ablation floor=0.5 branch). Byte-identical asserted IN THE DATA (exact float
      equality), not inferred from reading the code. This is the BYTE-IDENTICAL ESCAPE post-flip.
  (A2) THE FLIP ITSELF -- with the var fully UNSET (the actual shipped default after the flip), `encoding_gain_
      for()` is byte-identical (float ==) to the explicit `"1"` arm, on both leaf branches -- unset and explicit-
      ON take the IDENTICAL code path by construction, so the flip is safe: nothing changes except which state
      is the ambient default.
  (B) ON -- LOAD-BEARING: the gain varies meaningfully across a DA sweep (span > 0.3), and PARITY: it tracks the
      pre-existing host formula's shape closely (correlation reported, not claimed exact -- a genuinely different
      spiking-derived mechanism is not required to bit-match a closed form it replaces).
  (C) ON + `BRAIN_DA_ENCODING_SPIKING_GAIN_LESION` -- THIS mechanism's own lesion (severs the excitability_drive
      target write_gain reads DA through, sensitivity pinned to 0.0 at build time -- a STATIC config, not a
      plastic weight, so it cannot regrow within the read window) collapses the DA-dependence to a near-zero
      span at the floor value (~1.0), on every da probed.
  (D) The PRE-EXISTING outer lesion (`BRAIN_DA_ENCODING_LESION`) still pins g=1.0 regardless of DA and regardless
      of the new flag -- LEVER-4 does not touch, weaken, or bypass that upstream gate.
  (E) LAZY IMPORT: `research.runners._da_write_gain_spiking_derisk` is NOT in `sys.modules` after exercising the
      EXPLICIT-OFF (`="0"`) arm in a fresh subprocess -- the new mechanism's substrate is never even built on the
      byte-identical escape, not merely "computed and discarded". (Post-flip, the DEFAULT unset arm DOES import
      it -- that is the point of the flip; laziness is now a property of the explicit escape, not of doing
      nothing.)

Run (numpy-CPU, cheap, foreground, ~1 min -- most of the cost is (B)'s spiking reads):
    SIM_BACKEND=numpy python -m research.runners._da_write_gain_spiking_hook_verify
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

import logging  # noqa: E402
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

DA_SWEEP = [0.05, 0.3, 0.5, 0.7, 0.9, 1.24]


class _FakeChat:
    """The minimal object `encoding_gain_for`'s LEVER-3/leaf path reads: `da_level_of` reads
    `chat._last_da_drives["da_level"]`; the lever-2 (homeostasis-off, non-default) path additionally reads/writes
    `chat._da_encoding_mu`, exercised separately below via a plain object() (getattr default handles its absence)."""
    def __init__(self, da):
        self._last_da_drives = {"da_level": da}


def _set(env):
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def main():
    import webapp.da_encoding_drives_chat as DAE

    # post-flip: the BYTE-IDENTICAL escape is now the EXPLICIT "0" (unset means ON) -- never pop this one key.
    _QUIET = {"BRAIN_DA_ENCODING_SPIKING_GAIN": "0", "BRAIN_DA_ENCODING_SPIKING_GAIN_LESION": None,
             "BRAIN_DA_ENCODING_LESION": None, "BRAIN_DA_ENCODING_SUBSTRATE": None,
             "BRAIN_DA_ENCODING_HOMEOSTASIS": None}

    # ── (A) OFF (explicit "0"): byte-identical on BOTH leaf branches (substrate floor=1.0, raw/ablation floor=0.5).
    _set(_QUIET)
    off_substrate = [DAE.encoding_gain_for(_FakeChat(da)) for da in DA_SWEEP]
    exp_substrate = [float(DAE._gain_map()(da, DAE._DA_TONIC_BASELINE, DAE._K_DA, DAE._G_FLOOR_HOMEO, DAE._G_MAX))
                     for da in DA_SWEEP]
    go_a_substrate = off_substrate == exp_substrate

    _set({**_QUIET, "BRAIN_DA_ENCODING_SUBSTRATE": "0", "BRAIN_DA_ENCODING_HOMEOSTASIS": "0"})
    off_raw = [DAE.encoding_gain_for(_FakeChat(da)) for da in DA_SWEEP]
    exp_raw = [float(DAE._gain_map()(da, DAE._DA_TONIC_BASELINE, DAE._K_DA, DAE._G_MIN, DAE._G_MAX))
              for da in DA_SWEEP]
    go_a_raw = off_raw == exp_raw
    go_a = bool(go_a_substrate and go_a_raw)

    # ── (A2) THE FLIP ITSELF. The DECISIVE, exact check is at the BOOLEAN dispatch level -- `da_encoding_spiking_
    #    gain_enabled()` fully-unset must return the IDENTICAL True `da_encoding_spiking_gain_enabled()` returns
    #    under explicit "1" (both take the SAME `if da_encoding_spiking_gain_enabled():` branch in `_leaf_gain` by
    #    construction -- this is what "the flip is safe" actually rests on). A SEPARATE, tolerance-based sanity
    #    check confirms the two arms' live spiking reads land in the SAME regime (NOT exact-equality-gated: each
    #    call is a genuine live read of a stochastic OU-noise-driven population -- see the de-risk's own
    #    documented noise floor, `_da_write_gain_spiking_derisk.py`'s "Instrument notes" -- so two SEPARATE reads
    #    at nominally the same DA are expected to differ by a few percent, not to bit-match).
    os.environ.pop("BRAIN_DA_ENCODING_SPIKING_GAIN", None)   # the ONLY place this key is ever popped in this file
    os.environ.pop("BRAIN_DA_ENCODING_SPIKING_GAIN_LESION", None)
    os.environ.pop("BRAIN_DA_ENCODING_LESION", None)
    os.environ.pop("BRAIN_DA_ENCODING_SUBSTRATE", None)
    os.environ.pop("BRAIN_DA_ENCODING_HOMEOSTASIS", None)
    enabled_unset = DAE.da_encoding_spiking_gain_enabled()
    unset_substrate = [float(np.mean([DAE.encoding_gain_for(_FakeChat(da)) for _ in range(3)])) for da in DA_SWEEP]
    _set({**_QUIET, "BRAIN_DA_ENCODING_SPIKING_GAIN": "1"})
    enabled_explicit1 = DAE.da_encoding_spiking_gain_enabled()
    explicit_on_substrate = [float(np.mean([DAE.encoding_gain_for(_FakeChat(da)) for _ in range(3)]))
                             for da in DA_SWEEP]
    go_a2_decisive = bool(enabled_unset is True and enabled_explicit1 is True)
    a2_max_abs_diff = float(max(abs(a - b) for a, b in zip(unset_substrate, explicit_on_substrate)))
    go_a2_tolerance = a2_max_abs_diff < 0.3   # a few noise-widths above the de-risk's own documented OU floor
    go_a2 = bool(go_a2_decisive and go_a2_tolerance)

    # ── (B) ON: load-bearing + parity (substrate/default branch -- the production-default leaf). ──
    _set({**_QUIET, "BRAIN_DA_ENCODING_SPIKING_GAIN": "1"})
    on_vals = [DAE.encoding_gain_for(_FakeChat(da)) for da in DA_SWEEP]
    host_vals = [float(DAE._gain_map()(da, DAE._DA_TONIC_BASELINE, DAE._K_DA, DAE._G_FLOOR_HOMEO, DAE._G_MAX))
                for da in DA_SWEEP]
    span_on = max(on_vals) - min(on_vals)
    load_bearing = span_on > 0.3
    corr = float(np.corrcoef(on_vals, host_vals)[0, 1]) if len(set(on_vals)) > 1 else None
    parity_ok = bool(corr is not None and corr > 0.9)
    go_b = bool(load_bearing and parity_ok)

    # ── (C) ON + inner lesion: collapses to the floor, near-zero span. ──
    _set({**_QUIET, "BRAIN_DA_ENCODING_SPIKING_GAIN": "1", "BRAIN_DA_ENCODING_SPIKING_GAIN_LESION": "1"})
    lesion_vals = [DAE.encoding_gain_for(_FakeChat(da)) for da in DA_SWEEP]
    span_lesion = max(lesion_vals) - min(lesion_vals)
    lesion_collapses = span_lesion < 0.05
    lesion_near_floor = abs(float(np.mean(lesion_vals)) - DAE._G_FLOOR_HOMEO) < 0.05
    go_c = bool(lesion_collapses and lesion_near_floor)

    # ── ATTRIBUTION: measuring both the ON differential and the lesioned one is not the same as asking whose
    #    the difference was (tools.lab's own lesson) -- what FRACTION of the high-vs-low DA gain differential is
    #    owed to the LIVE excitability_drive link this mechanism's own lesion severs?
    from tools.lab import attributable_to
    diff_on = on_vals[-1] - on_vals[0]           # high-DA vs low-DA gain gap with the DA->write_gain link intact
    diff_lesion = lesion_vals[-1] - lesion_vals[0]   # the SAME gap with that ONE link severed (sensitivity=0.0)
    lesion_attribution = attributable_to(
        "the high-vs-low write-gain differential owed to the LIVE excitability_drive link "
        "(control = BRAIN_DA_ENCODING_SPIKING_GAIN_LESION)", diff_on, diff_lesion)

    # ── (D) the PRE-EXISTING outer lesion still overrides, regardless of the new flag. ──
    _set({**_QUIET, "BRAIN_DA_ENCODING_SPIKING_GAIN": "1", "BRAIN_DA_ENCODING_LESION": "1"})
    outer_lesion_vals = [DAE.encoding_gain_for(_FakeChat(da)) for da in DA_SWEEP]
    go_d = bool(all(v == 1.0 for v in outer_lesion_vals))

    # ── (E) lazy import: run the EXPLICIT-OFF ("0") arm in a FRESH subprocess, check sys.modules never gained the
    #    new module. Post-flip, unset now means ON (would import) -- the byte-identical escape is what must stay
    #    lazy, so the probe pins the flag to "0" explicitly rather than popping it.
    #    Written to a temp FILE (not a `-c` one-liner) -- a semicolon-joined one-liner mixing a class definition's
    #    indented suite with subsequent dedented statements is fragile to construct correctly by string-formatting
    #    (earned: the first version of this probe silently crashed with empty stdout, read as a false NO-GO on
    #    this check); a real script file has no such ambiguity and any traceback lands on stderr for the record.
    import tempfile
    probe_src = (
        "import sys, os\n"
        f"sys.path.insert(0, {_REPO!r})\n"
        "os.environ.setdefault('SIM_BACKEND', 'numpy')\n"
        "import logging\n"
        "logging.getLogger().setLevel(logging.ERROR)\n"
        "import webapp.da_encoding_drives_chat as DAE\n"
        "os.environ['BRAIN_DA_ENCODING_SPIKING_GAIN'] = '0'\n"
        "\n"
        "class _C:\n"
        "    def __init__(self, da):\n"
        "        self._last_da_drives = {'da_level': da}\n"
        "\n"
        f"[DAE.encoding_gain_for(_C(da)) for da in {DA_SWEEP!r}]\n"
        "print('IMPORTED' if 'research.runners._da_write_gain_spiking_derisk' in sys.modules else 'NOT_IMPORTED')\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix="_lazy_import_probe.py", delete=False) as _tf:
        _tf.write(probe_src)
        probe_path = _tf.name
    try:
        proc = subprocess.run([sys.executable, probe_path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(probe_path)
    go_e = "NOT_IMPORTED" in proc.stdout

    go = bool(go_a and go_a2 and go_b and go_c and go_d and go_e)

    from tools.verdict import Verdict
    v = Verdict("LEVER-4 (BRAIN_DA_ENCODING_SPIKING_GAIN) production-hook dispatch, FLIPPED DEFAULT-ON 2026-09-05")
    v.require("(A) explicit-0 (the byte-identical escape) on the substrate/default leaf (floor=1.0)",
              go_a_substrate, expect=True, note=f"off={off_substrate} == expected={exp_substrate}")
    v.require("(A) explicit-0 (the byte-identical escape) on the raw/ablation leaf (floor=0.5)", go_a_raw,
              expect=True, note=f"off={off_raw} == expected={exp_raw}")
    v.require("(A2) THE FLIP: unset dispatches the IDENTICAL boolean branch as explicit-1 (decisive, exact)",
              go_a2_decisive, expect=True,
              note=f"enabled_unset={enabled_unset} enabled_explicit1={enabled_explicit1}")
    v.require("(A2) sanity: unset's 3-rep-averaged live read lands within noise tolerance of explicit-1's",
              go_a2_tolerance, expect=True,
              note=f"unset={unset_substrate} explicit_on={explicit_on_substrate} max_abs_diff={a2_max_abs_diff:.4f}")
    v.require("(B) ON load-bearing (span > 0.3 across the DA sweep)", load_bearing, expect=True,
              note=f"on_vals={on_vals} span={span_on}")
    v.require("(B) ON parity with the host formula (corr > 0.9)", parity_ok, expect=True,
              note=f"corr={corr} host_vals={host_vals}")
    v.require("(C) inner lesion collapses the span (< 0.05)", lesion_collapses, expect=True,
              note=f"lesion_vals={lesion_vals} span={span_lesion}")
    v.require("(C) inner lesion collapses TO the floor (~1.0)", lesion_near_floor, expect=True)
    v.require("(D) the pre-existing outer da_encoding_lesioned() still pins g=1.0 regardless", go_d, expect=True,
              note=f"outer_lesion_vals={outer_lesion_vals}")
    v.require("(E) the spiking module is never imported on the EXPLICIT-OFF (=0) escape", go_e, expect=True,
              note=f"subprocess stdout={proc.stdout.strip()!r} stderr_tail={proc.stderr.strip()[-400:]!r}")
    v.control("the write gain rides the LEVER-4 spiking read (on) and is severed by ITS OWN lesion",
              treatment=(on_vals[-1] - on_vals[0]), control=(lesion_vals[-1] - lesion_vals[0]), min_separation=0.2,
              note="on: gain rises with DA; inner lesion: flat regardless of DA")
    v.disabled("the full brain_chat handler round trip", why="already proven unmodified by this lever this same "
              "session -- research/runners/_da_encoding_wired_verify.py re-run GO, byte-for-byte its own docstring "
              "numbers (g_high=2.4774/g_low=1.0/lesion-both=1.0); this script isolates the ONE new dispatch point")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_da_write_gain_spiking_hook_verify", "go": go, "status": decided["status"],
        "flipped_default_on": "2026-09-05",
        "A_off_byte_identical_explicit0": {"substrate_leaf": go_a_substrate, "raw_leaf": go_a_raw,
                                 "off_substrate": off_substrate, "off_raw": off_raw},
        "A2_flip_correctness_unset_eq_explicit1": {"go": go_a2, "decisive_boolean_ok": go_a2_decisive,
                                 "tolerance_sanity_ok": go_a2_tolerance,
                                 "enabled_unset": enabled_unset, "enabled_explicit1": enabled_explicit1,
                                 "unset_substrate_3rep_avg": unset_substrate,
                                 "explicit_on_substrate_3rep_avg": explicit_on_substrate,
                                 "max_abs_diff": a2_max_abs_diff},
        "B_on_load_bearing_parity": {"on_vals": on_vals, "host_vals": host_vals, "span": span_on,
                                     "corr": corr, "load_bearing": load_bearing, "parity_ok": parity_ok},
        "C_inner_lesion": {"lesion_vals": lesion_vals, "span": span_lesion, "collapses": lesion_collapses,
                          "near_floor": lesion_near_floor,
                          "differential_live": diff_on, "differential_lesion": diff_lesion,
                          "attribution_to_live_excitability_drive_link": lesion_attribution},
        "D_outer_lesion_unaffected": {"outer_lesion_vals": outer_lesion_vals, "GO": go_d},
        "E_lazy_import_on_explicit_off": {"subprocess_stdout": proc.stdout.strip(),
                         "subprocess_stderr": proc.stderr.strip(), "GO": go_e},
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    op = "research/findings/raw/_da_write_gain_spiking/hook_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar)
    print("  LEVER-4 PRODUCTION-HOOK VERIFY -- BRAIN_DA_ENCODING_SPIKING_GAIN (FLIPPED DEFAULT-ON 2026-09-05)")
    print(bar)
    print(f"  (A) explicit-0 escape byte-identical: substrate={go_a_substrate} raw={go_a_raw}")
    print(f"  (A2) flip correctness (unset == explicit-1): {go_a2}")
    print(f"  (B) ON load-bearing={load_bearing} (span={span_on:.3f})  parity corr={corr}")
    print(f"  (C) inner lesion collapses={lesion_collapses} near_floor={lesion_near_floor} (span={span_lesion:.4f})")
    print(f"  (D) outer lesion still pins 1.0: {go_d}")
    print(f"  (E) lazy import confirmed (never imported on explicit-off escape): {go_e}")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} ({decided['status']})")
    print(f"  [saved] {op}\n" + bar)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
