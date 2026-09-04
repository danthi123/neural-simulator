"""BYTE-IDENTICAL proofs for the neural affect coupling (2026-09-04 scaffold-retirement rung).

Two independent EXACT-STRING-COMPARE claims (never "expected unchanged, unverified" -- `docs/TERMS.md`'s own
condition for the word "byte-identical"), each run in an isolated subprocess so neither perturbs this process's
own `sys.modules`/RNG state:

  (1) OFF-BY-DEFAULT. `BRAIN_WKV_MOUTH_AFFECT_NEURAL` unset (`wkv_mouth_affect_neural_enabled()`'s own default)
      reproduces the PRE-THIS-RUNG module pair's OWN `generate()` output byte-for-byte, on the SAME prompt/seed/
      valence/arousal/affect_boost. The pre-rung `webapp/wkv_mouth_generator.py` + `research/runners/
      _wkv_fewspike_read_derisk.py` are extracted from git (`--before-ref`, default `HEAD` at the commit this
      rung branched from) via `git show <ref>:<path>` and loaded as SHADOWED modules (`_load_shadowed_pair`) so
      this is a genuine A/B between two DIFFERENT file contents, not a restatement of the current file's own
      logic reading itself.
  (2) LESION-EQUIVALENT NO-OP. With the neural coupling flag ON but `valence=0.0` (the exact condition
      `AffectProductionOrgan.read_differential(lesion=True)` maps to -- see `_affect_pool_gains`'s own
      docstring), output is byte-identical to the HOST mechanism at the same neutral condition -- proving an
      INERT (every `mood_pool_k` concentration at 0.0) neuromodulator subsystem contributes EXACTLY ZERO
      numerically to the Izhikevich dynamics, not merely "the code path looks skipped".

Run: SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_byte_identical_check \\
        --before-ref HEAD --json research/findings/raw/_wkv_mouth_affect_neural_byte_identical_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# `lever` -- both claims here are "prove X did NOT move under manipulation Y" (the opposite of lever's usual
# required=True direction), so it is called with `required=False` purely for its instrumented before/after
# PRINTING + `moved` return value; the pass/fail judgment (`byte_identical = not moved`) is this script's own,
# not lever's -- an honest use, not a re-purposing that hides what actually decided the verdict.
from tools.lab import lever  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEBAPP_REL = "webapp/wkv_mouth_generator.py"
_READ_REL = "research/runners/_wkv_fewspike_read_derisk.py"


def _git_show(ref: str, rel_path: str) -> str:
    out = subprocess.run(["git", "show", f"{ref}:{rel_path}"], cwd=str(_REPO_ROOT),
                         capture_output=True, text=True, check=True)
    return out.stdout


def _check_off_by_default(ref: str, seed: int, prompt: str, valence: float, arousal: float,
                          affect_boost: float, max_new_tokens: int) -> dict:
    """Runs claim (1) in a subprocess that shadows BOTH files at `ref` via sys.modules pre-registration, then
    ALSO runs the CURRENT code with the flag unset, and compares the two `generate()` outputs exactly."""
    webapp_src = _git_show(ref, _WEBAPP_REL)
    read_src = _git_show(ref, _READ_REL)
    # A scratch dir OUTSIDE the repo tree (never committed, cleaned up in the `finally` below) -- these are just
    # ephemeral extracted copies of `ref`'s own file content (already recoverable any time via `git show`), not
    # independently meaningful artifacts.
    sandbox = Path(tempfile.mkdtemp(prefix="wkv_mouth_byteident_"))
    try:
        (sandbox / "orig_wkv_mouth_generator.py").write_text(webapp_src)
        (sandbox / "orig_wkv_fewspike_read_derisk.py").write_text(read_src)
        return _run_off_by_default_subprocesses(sandbox, ref, seed, prompt, valence, arousal, affect_boost,
                                                max_new_tokens)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _run_off_by_default_subprocesses(sandbox: Path, ref: str, seed: int, prompt: str, valence: float,
                                     arousal: float, affect_boost: float, max_new_tokens: int) -> dict:
    code = f"""
import importlib.util, sys, json
sys.path.insert(0, {str(_REPO_ROOT)!r})

def _load(dotted, path, register_as=None):
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[register_as or dotted] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-register the ORIGINAL _wkv_fewspike_read_derisk under its REAL dotted name so the original
# webapp module's own `from research.runners._wkv_fewspike_read_derisk import ...` resolves to IT, not the
# current (modified) file on disk.
_load("research.runners._wkv_fewspike_read_derisk", {str(sandbox / "orig_wkv_fewspike_read_derisk.py")!r},
      register_as="research.runners._wkv_fewspike_read_derisk")
orig_webapp = _load("webapp.wkv_mouth_generator_orig", {str(sandbox / "orig_wkv_mouth_generator.py")!r})
orig_text, _secs = orig_webapp.generate({prompt!r}, seed={seed}, max_new_tokens={max_new_tokens},
                                        valence={valence!r}, arousal={arousal!r}, affect_boost={affect_boost!r})
print("ORIG_JSON:" + json.dumps({{"text": orig_text}}))
"""
    # The shadowed original module's OWN `_REPO_ROOT = Path(__file__).resolve().parents[1]` resolves relative to
    # its SANDBOX location (research/findings/raw/.../orig_wkv_mouth_generator.py), not the real repo root --
    # override every `_REPO_ROOT`-derived default path via its env-var escape hatch so the shadowed original
    # reads the SAME real checkpoint/learned-head artifacts the current code does (otherwise a missing-file
    # fail-safe fallback, e.g. `_apply_learned_head`'s "file_missing" -> native head, would silently confound
    # this comparison with an UNRELATED path bug, not the thing actually under test).
    env_orig = dict(os.environ)
    env_orig["SIM_BACKEND"] = "numpy"
    env_orig["BRAIN_WKV_MOUTH_AFFECT"] = "1"
    env_orig["BRAIN_WKV_MOUTH_CKPT"] = str(_REPO_ROOT / "bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    env_orig["BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH"] = str(
        _REPO_ROOT / "research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_0p94_s{seed}.npz")
    env_orig["BRAIN_WKV_MOUTH_BPE_PATH"] = str(_REPO_ROOT / "bridges/wkv_ckpt/wkv_bpe8k.json")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT), env=env_orig,
                          capture_output=True, text=True, timeout=120)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ORIG_JSON:")), None)
    if line is None:
        raise RuntimeError(f"original-code arm failed: rc={proc.returncode}\nSTDOUT={proc.stdout[-2000:]}"
                           f"\nSTDERR={proc.stderr[-4000:]}")
    orig_text = json.loads(line[len("ORIG_JSON:"):])["text"]

    # Current code, flag unset (the default) -- a fresh subprocess, ordinary import.
    code2 = f"""
import sys, json
sys.path.insert(0, {str(_REPO_ROOT)!r})
from webapp.wkv_mouth_generator import generate
text, secs = generate({prompt!r}, seed={seed}, max_new_tokens={max_new_tokens},
                       valence={valence!r}, arousal={arousal!r}, affect_boost={affect_boost!r})
print("NEW_JSON:" + json.dumps({{"text": text}}))
"""
    env_new = dict(os.environ)
    env_new["SIM_BACKEND"] = "numpy"
    env_new["BRAIN_WKV_MOUTH_AFFECT"] = "1"
    env_new.pop("BRAIN_WKV_MOUTH_AFFECT_NEURAL", None)          # unset -> default (OFF)
    proc2 = subprocess.run([sys.executable, "-c", code2], cwd=str(_REPO_ROOT), env=env_new,
                           capture_output=True, text=True, timeout=120)
    line2 = next((ln for ln in proc2.stdout.splitlines() if ln.startswith("NEW_JSON:")), None)
    if line2 is None:
        raise RuntimeError(f"current-code arm failed: rc={proc2.returncode}\nSTDOUT={proc2.stdout[-2000:]}"
                           f"\nSTDERR={proc2.stderr[-4000:]}")
    new_text = json.loads(line2[len("NEW_JSON:"):])["text"]

    moved = lever(f"off-by-default (ref={ref}, prompt={prompt!r}) -- should NOT move", orig_text, new_text,
                 required=False)
    return {"ref": ref, "prompt": prompt, "seed": seed, "valence": valence, "arousal": arousal,
            "orig_text": orig_text, "new_text_flag_off": new_text, "byte_identical": not moved}


def _run_arm_current(seed, prompt, valence, arousal, affect_boost, neural, max_new_tokens):
    from research.runners._wkv_mouth_affect_neural_verify import _run_arm
    return _run_arm(seed, prompt, valence, arousal, affect_boost, neural, max_new_tokens=max_new_tokens)


def _check_lesion_equivalent_noop(seed: int, prompt: str, affect_boost: float, max_new_tokens: int) -> dict:
    host_neutral = _run_arm_current(seed, prompt, 0.0, 0.0, affect_boost, neural=False,
                                     max_new_tokens=max_new_tokens)
    neural_neutral = _run_arm_current(seed, prompt, 0.0, 0.0, affect_boost, neural=True,
                                      max_new_tokens=max_new_tokens)
    moved = lever(f"lesion-equivalent no-op (prompt={prompt!r}) -- should NOT move", host_neutral["text"],
                 neural_neutral["text"], required=False)
    return {
        "prompt": prompt, "seed": seed,
        "host_neutral_text": host_neutral["text"], "neural_neutral_text": neural_neutral["text"],
        "byte_identical": not moved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-ref", type=str, default="HEAD")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--affect-boost", type=float, default=10.0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_mouth_affect_neural_byte_identical_check.json")
    args = ap.parse_args()

    t0 = time.time()
    prompts = ["the little girl was", "tom and his dog were"]
    off_checks = [_check_off_by_default(args.before_ref, args.seed, p, 0.16, 0.65, args.affect_boost,
                                        args.max_new_tokens) for p in prompts]
    lesion_checks = [_check_lesion_equivalent_noop(args.seed, p, args.affect_boost, args.max_new_tokens)
                     for p in prompts]
    out = {
        "before_ref": args.before_ref, "seed": args.seed, "affect_boost": args.affect_boost,
        "off_by_default_checks": off_checks,
        "off_by_default_all_byte_identical": all(c["byte_identical"] for c in off_checks),
        "lesion_equivalent_checks": lesion_checks,
        "lesion_equivalent_all_byte_identical": all(c["byte_identical"] for c in lesion_checks),
        "elapsed_s": round(time.time() - t0, 1),
    }
    for c in off_checks:
        print(f"[off-by-default] prompt={c['prompt']!r} byte_identical={c['byte_identical']}", flush=True)
    for c in lesion_checks:
        print(f"[lesion-equivalent] prompt={c['prompt']!r} byte_identical={c['byte_identical']}", flush=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] {out['elapsed_s']}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
