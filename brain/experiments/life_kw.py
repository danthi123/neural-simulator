# Capability: Self and identity
# Record: brain/research/2026-11-23-biggest-defect-read.md
# Expected-fail: none by itself (a runner, no method); a DEV life with an extra kwarg (M1 episode="lived_key", M2
#   speaker_contrast) may still echo or fetch the wrong episode: the self/other tag is "missing" in PREREQUISITES.md
"""brain.eval.life's CLI, unchanged, plus --brain-kw-extra JSON merged into the Brain kwargs the life builds.

    python -m brain.experiments.life_kw --brain-kw-extra '{"episode": "lived_key"}' <every brain.eval.life argument>

life.py is not edited (it is a frozen Stage-2 TEST v2 file). The extras go in through life.main's own `brain_kw`
argument, which every Brain(...) of the life receives (`run` stores it in res["brain_kw"]); the wrapper also records
them as res["brain_kw_extra"] (by wrapping life.write_outputs in this process only). Refused: any TEST split
(--split test / test2, or a resume of a life saved on one), a key the life CLI or `run` owns (it would be silently
overridden), a key Brain() does not take, and a resume whose extras differ from the ones the life was started with.
Without --brain-kw-extra the call is exactly brain.eval.life.main(argv): the same life, byte for byte.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

REFUSED_SPLITS = ("test", "test2")
# Set by brain.eval.life itself (main's CLI flags, or run's sleep_seed): an extra here would be overridden or would
# fork the life's own bookkeeping, so it is refused rather than merged.
LIFE_OWNED = ("key", "key_bar", "use_gate", "sleep_downscale", "replay_priority", "keep_bank", "rank",
              "downscale_factors", "downscale_when", "inquiry", "sleep_seed", "sleep_log", "cortex", "home", "lock",
              "sleep_on_restart")


def parse_extra(text: str | None) -> dict:
    if text is None:
        return {}
    extra = json.loads(text)
    if not isinstance(extra, dict):
        raise ValueError("--brain-kw-extra must be a JSON object")
    from brain.brain import Brain
    params = set(inspect.signature(Brain.__init__).parameters) - {"self"}
    bad = sorted(set(extra) - params)
    if bad:
        raise ValueError(f"--brain-kw-extra: Brain() takes no {bad}")
    owned = sorted(set(extra) & set(LIFE_OWNED))
    if owned:
        raise ValueError(f"--brain-kw-extra: {owned} are set by brain.eval.life itself; use its own flag")
    return extra


def _peek(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--brain-kw-extra", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    known = ap.parse_known_args(argv)[0]
    # life's argv is argv minus the --brain-kw-extra pair (both spellings), order kept
    life_argv, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == "--brain-kw-extra":
            skip = True
            continue
        if x.startswith("--brain-kw-extra="):
            continue
        life_argv.append(x)
    return known, life_argv


def main(argv=None, cortex=None, brain_kw: dict | None = None):
    from brain.eval import life
    argv = list(sys.argv[1:] if argv is None else argv)
    known, life_argv = _peek(argv)
    extra = parse_extra(known.brain_kw_extra)
    if extra and known.split in REFUSED_SPLITS:
        raise SystemExit(f"life_kw refused: --brain-kw-extra on --split {known.split} (a frozen TEST split; DEV only)")
    if known.resume and known.out and os.path.exists(os.path.expanduser(known.out) + ".json"):
        with open(os.path.expanduser(known.out) + ".json") as fh:
            saved = json.load(fh)
        if extra and saved.get("split") in REFUSED_SPLITS:
            raise SystemExit(f"life_kw refused: resume of a life on the {saved['split']} split")
        if (saved.get("brain_kw_extra") or {}) != extra:
            raise SystemExit(f"life_kw refused: resume: this life has brain_kw_extra {saved.get('brain_kw_extra')!r}, "
                             f"not {extra!r}")
    if not extra:
        return life.main(life_argv, cortex=cortex, brain_kw=brain_kw)     # the plain life, untouched
    real = life.write_outputs

    def write_outputs(res, out):
        res["brain_kw_extra"] = dict(extra)
        return real(res, out)
    life.write_outputs = write_outputs      # this process only; restored below
    try:
        return life.main(life_argv, cortex=cortex, brain_kw={**(brain_kw or {}), **extra})
    finally:
        life.write_outputs = real


if __name__ == "__main__":
    main()
