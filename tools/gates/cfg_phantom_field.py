"""CLASS PF -- a runner writes `cfg.<field>` under a name CoreSimConfig never declares, so the engine silently never reads it.

THE DEFECT (T15, 2026-09-30/10-01, docs/drafts/ecker-ou-field/AFFECTED.md). Eight research/runners/_gap5_* wrote cfg.ou_noise_sigma_pa = 40.0 to set the OU-noise amplitude. CoreSimConfig (sim/config.py:88, a plain @dataclass, no slots/frozen) has no such field -- the real one is ou_std_current_pA (sim/config.py:370, default 100 pA, read in sim/bridge.py:3923/3938/12577). A plain dataclass instance accepts an unknown attribute write with no error, so the typo/stale-rename ran silently: runs configured for 40 pA ran at the 100 pA default, undetected until a manual doc/code audit.

WHY NOTHING CAUGHT IT. knob_reachable (CLASS KR) asks whether a REAL cfg.* write is reachable from argparse, not whether the NAME it writes exists on the dataclass at all. stated_value_mismatch (CLASS SV) and quantity_mismatch (CLASS 6) both compare two numbers already present in a finding or artifact; neither reads sim/config.py, so neither can tell a runner's own assignment target was never declared.

WHAT THIS GATE ENFORCES, on research/runners/*.py files in the commit THAT CONSTRUCT A CoreSimConfig (cfg = CoreSimConfig(...) / core_config = CoreSimConfig(...), self-qualified or not, in the same file -- the construction check is what keeps this out of files where cfg names something else entirely, see CALIBRATION): every cfg.<name> = ... / core_config.<name> = ... (bare, self.-qualified, or chained after a ';' on one line -- the shape all 8 incidents used) must name a field declared somewhere in sim/config.py. An undeclared name BLOCKS, reporting the closest real field names (difflib) to surface a typo/stale-rename fast.

CALIBRATION, MEASURED BEFORE SHIPPING. Run over all 2154 research/runners/*.py at 48ee4f853: 372 files construct a CoreSimConfig directly; of those, 71 phantom-field writes in 56 files. Without the construction guard, research/runners/lm_chat.py:57's cfg.gen_tokens = ... is a FALSE POSITIVE -- its cfg is an lm_train_lib.TrainConfig, an unrelated dataclass that happens to share the name; the guard removes it (and every other same-named-but-different-type cfg) by requiring the file to actually build a CoreSimConfig. Of the 71 true hits: 9 are this gate's own motivating incident (ou_noise_sigma_pa, the 8 _gap5_* renamed on research/draft-ecker-ou-field plus _gap5_onebrain_capstone.py's separately-judged dead setattr, which this gate's literal-assignment-only scope does not reach -- see WHAT IT CANNOT CATCH). The remaining ~60, concentrated in cfg.dt = 1.0 across ~50 unrelated derisk runners (the real field is dt_ms, sim/config.py:91) plus a handful of other one-off names (enable_stp, total_time, background_noise_std, ou_noise_amplitude_pA, graded_source_rest_mV/_scale_mV, apical_soma_coupling, apical_leak), are a SEPARATE, PRE-EXISTING instance of the same failure class, found only by this calibration pass and not part of the ecker-ou incident -- audited separately (runbook T17 / RULING phantom-cfg-audit), not fixed as part of this ticket.

WHAT IT CANNOT CATCH: a name assigned only via a fully dynamic expression (an f-string, concatenation, or a variable holding the field name) -- _gap5_onebrain_capstone.py's own `for _f, _v in (("ou_std_current_pA", 0.0), ("ou_noise_sigma_pa", 40.0)): setattr(b.core_config, _f, _v)` is exactly this shape and is NOT caught; a phantom name on a config object under a variable name other than cfg/core_config; and, matching KR/SN/VN's own scoping, a phantom field already sitting in an UNMODIFIED runner -- this gate only sees files the commit touches. The capstone instance was judged by hand instead (ecker-ou-gate ruling, 2026-10-01): its hasattr(b.core_config, _f) guard is always False for that name, so setattr never fires -- provably inert.
"""
from __future__ import annotations
import difflib, os, re

NAME = "cfg-phantom-field"
CLASS_ID = "PF"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CFG_FILE = os.path.join(_ROOT, "sim", "config.py")

ASSIGN_RE = re.compile(r"(?:^|;)\s*(?:[A-Za-z_]\w*\.)?(?:cfg|core_config)\.([A-Za-z_]\w*)\s*=(?!=)", re.M)
CONSTRUCT_RE = re.compile(r"\b(?:self\.)?(?:cfg|core_config)\s*(?::\s*CoreSimConfig)?\s*=\s*CoreSimConfig\s*\(")


def _declared_fields(cfg_text):
    return {m.group(1) for m in re.finditer(r"^    ([A-Za-z_]\w*)\s*:\s*[^=\n]+", cfg_text, re.M)}


def _check_one(path, rel, declared):
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return []
    if not CONSTRUCT_RE.search(text):
        return []
    problems = []
    for m in ASSIGN_RE.finditer(text):
        name = m.group(1)
        if name in declared or name.startswith("_"):
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        close = difflib.get_close_matches(name, declared, n=3, cutoff=0.6)
        hint = (" closest declared field(s): %s" % ", ".join(close)) if close else \
               " (no close match -- check it was not renamed/removed from sim/config.py too)"
        problems.append("%s:%d: `cfg.%s` is not a field sim/config.py declares.%s" % (rel, line_no, name, hint))
    return problems


def check(paths):
    if paths is not None and len(paths) == 0:
        return []
    if not os.path.exists(_CFG_FILE):
        return ["sim/config.py is missing -- cannot verify cfg field names."]
    declared = _declared_fields(open(_CFG_FILE, errors="ignore").read())
    problems = []
    for p in (paths or []):
        rel = p.replace("\\", "/") if not os.path.isabs(p) else os.path.relpath(p, _ROOT).replace("\\", "/")
        if "research/runners/" not in rel or not rel.endswith(".py"):
            continue
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if not os.path.exists(full):
            continue
        problems.extend(_check_one(full, rel, declared))
    return problems


def selftest():
    import tempfile
    declared = {"ou_std_current_pA", "num_neurons", "seed"}
    cases = [
        ("bad.py", 'def f():\n    cfg = CoreSimConfig()\n    cfg.ou_noise_sigma_pa = 40.0\n', True),
        ("bad_chained.py", 'def f():\n    cfg = CoreSimConfig()\n    cfg.enable_ou_process = True; cfg.ou_noise_sigma_pa = 40.0\n', True),
        ("bad_core.py", 'def f(b):\n    b.core_config = CoreSimConfig()\n    b.core_config.ou_noise_sigma_pa = 40.0\n', True),
        ("good.py", 'def f():\n    cfg = CoreSimConfig()\n    cfg.ou_std_current_pA = 40.0\n', False),
        ("good_eq.py", 'def f():\n    cfg = CoreSimConfig()\n    if cfg.ou_std_current_pA == 0:\n        pass\n', False),
        ("unrelated_cfg.py", 'def f():\n    cfg = TrainConfig(**d)\n    cfg.gen_tokens = 8\n', False),
    ]
    probs = []
    with tempfile.TemporaryDirectory() as d:
        for name, src, want_fail in cases:
            p = os.path.join(d, name)
            open(p, "w").write(src)
            got = _check_one(p, "research/runners/" + name, declared)
            if want_fail and not got:
                probs.append("missed phantom field in %s" % name)
            if not want_fail and got:
                probs.append("false positive on %s: %s" % (name, got))
    return probs


if __name__ == "__main__":
    import glob, sys
    problems = check(glob.glob(os.path.join(_ROOT, "research/runners/*.py")))
    for p in problems:
        print(p)
    sys.exit(1 if problems else 0)
