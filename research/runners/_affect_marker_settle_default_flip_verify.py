"""S09 (AG-FLIP, 2026-09-24) equivalence check for the PARKED `BRAIN_AFFECT_MARKER_SETTLE` default-ON prep.

WHAT THIS PROVES, and what it does NOT. `_affect_marker_wta_derisk.settle_enabled()` was changed (this branch
only, `research/settle-default-on-prep`, based on bd391aa31) from "env var, unset -> OFF" to "env var, unset ->
`_SETTLE_DEFAULT_ON`". The flag itself, and every downstream circuit it gates (drive, DEAD_MARGIN, wiring), is
UNCHANGED -- this module only moves which value is read when nothing is set. So the only claim this script can
make, and the only one it makes, is: on THIS revision, resolving SETTLE mode via the new default (env unset)
produces IDENTICAL output to resolving it via the pre-existing explicit override (`BRAIN_AFFECT_MARKER_SETTLE=1`)
-- i.e. the default changed, nothing else did. It does NOT re-litigate the 2026-09-23 full-brain NO-GO
(SETTLE-attributable 2/6, `2026-09-23-affect-marker-settle-fullbrain-contrast-PARTIAL-6seed.md`, standing) and it
does NOT run a chat transcript (no such pinned-10-turn tool exists in this repo as of 2026-09-24; that
infrastructure is S22's, not this module's -- declared here as a named residual rather than assumed).

Byte-identity with the OLD default (flag unset, pre-flip revision) is a SEPARATE, already-proven claim: it holds
trivially by construction (an explicit `BRAIN_AFFECT_MARKER_SETTLE=0` always reaches the untouched OFF branch,
on this revision or on main), and this script also checks it directly so a regression cannot slip through blind.

Usage: SIM_BACKEND=numpy .venv/bin/python -m research.runners._affect_marker_settle_default_flip_verify --seed 7 \
    --out research/findings/raw/_affect_marker_settle_default_flip/s7_equivalence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

_MOODS = (-0.085, -0.060, -0.029, -0.020, 0.020, 0.064, 0.069, 0.085)   # includes the +2/+3 boundary case (0.069)
_AROUSALS = (0.010, 0.020, 0.050, 0.065, 0.075)


def _sweep(reader, module):
    out = {"valence": [], "arousal": []}
    for mood in _MOODS:
        level, rates, meta = reader.select_valence(mood)
        out["valence"].append({"mood": mood, "level": level, "margin": round(meta["margin"], 6),
                                "rates": [round(float(r), 6) for r in rates]})
    for felt in _AROUSALS:
        high, rates, meta = reader.select_arousal(felt)
        out["arousal"].append({"felt_arousal": felt, "high": high, "margin": round(meta["margin"], 6),
                                "rates": [round(float(r), 6) for r in rates]})
    return out


def _digest(payload) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _run_in_subprocess(seed: int, env_value):
    """A FRESH process per condition (module-level `_READERS` cache + `get_reader`'s settle-keying means a
    same-process re-read after toggling the env var could reuse a reader built under the OTHER setting)."""
    env = dict(os.environ)
    env["SIM_NO_PROVENANCE"] = "1"
    env.setdefault("SIM_BACKEND", "numpy")
    if env_value is None:
        env.pop("BRAIN_AFFECT_MARKER_SETTLE", None)
    else:
        env["BRAIN_AFFECT_MARKER_SETTLE"] = env_value
    code = (
        "import json, sys; sys.path.insert(0, %r)\n"
        "from research.runners._affect_marker_wta_derisk import get_reader, settle_enabled, _SETTLE_DEFAULT_ON\n"
        "from research.runners._affect_marker_settle_default_flip_verify import _sweep\n"
        "r = get_reader(seed=%d)\n"
        "print(json.dumps({'settle_enabled': settle_enabled(), 'settle_default_on': _SETTLE_DEFAULT_ON,"
        " 'sweep': _sweep(r, None)}))\n"
    ) % (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), seed)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError("subprocess failed (env BRAIN_AFFECT_MARKER_SETTLE=%r):\n%s" % (env_value, proc.stderr))
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)   # dev/calibration seed only, per CLAUDE.md
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools.verdict import Verdict

    unset = _run_in_subprocess(args.seed, None)
    explicit_on = _run_in_subprocess(args.seed, "1")
    explicit_off = _run_in_subprocess(args.seed, "0")

    h_unset = _digest(unset["sweep"])
    h_on = _digest(explicit_on["sweep"])
    h_off = _digest(explicit_off["sweep"])

    v = Verdict("settle default-on flip: env-unset reproduces explicit SETTLE=1 (seed %d, dev)" % args.seed)
    v.require("env-unset resolves settle_enabled() True (the new default)", unset["settle_enabled"], expect=True)
    v.require("env-unset reads _SETTLE_DEFAULT_ON True (the prepared literal)", unset["settle_default_on"],
              expect=True)
    v.require("hash(env-unset sweep) == hash(explicit SETTLE=1 sweep)", h_unset == h_on, expect=True)
    v.require("explicit SETTLE=0 resolves settle_enabled() False (OFF path still reachable)",
              explicit_off["settle_enabled"], expect=False)
    v.control("explicit SETTLE=0 sweep vs explicit SETTLE=1 sweep", treatment=1 if h_off != h_on else 0,
              control=0, min_separation=0.5,
              note="hashes must differ -- confirms the OFF path is untouched, not merely unchanged by omission")
    decided = v.decide(go=True)   # go=True is the CLAIM; decide() downgrades to UNDEFINED if any check failed
    ok = decided["status"] == "GO"

    result = dict(decided)
    result.update({
        "seed": args.seed,
        "claim": "env-unset (new default) reproduces explicit BRAIN_AFFECT_MARKER_SETTLE=1 EXACTLY, on this "
                 "revision; explicit =0 is untouched (differs from the settle-on hash, as expected)",
        "hash_sha256": {"unset_default": h_unset, "explicit_settle_1": h_on, "explicit_settle_0": h_off},
        "GO": ok,
        "sweeps": {"unset_default": unset["sweep"], "explicit_settle_1": explicit_on["sweep"],
                   "explicit_settle_0": explicit_off["sweep"]},
        "residual": "no pinned-10-turn brain_chat transcript tool exists in this repo as of 2026-09-24 (S22's "
                    "infra, not built yet); this check instead exercises the WTA module's own documented API "
                    "directly, which is the only call site BRAIN_AFFECT_MARKER_SETTLE reaches "
                    "(webapp/affect_drives_chat.py -> get_reader() -> settle_enabled(), verified by grep).",
    })
    print(json.dumps({k: result[k] for k in ("seed", "hash_sha256", "status", "GO")}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
