"""EMERGE-51 CI guard — the experiential conversational console: OBSERVE members with features -> the competitive
pooler DISCOVERS the categories -> TEACH class property + member exception -> ASK in natural language (inherit /
cancel / abstain) over the DISCOVERED codes, with the no-confab moat. CPU/numpy, offline, reuse-by-import; NO sim/ edit.

Three tests:
  1. demo self-check: the scripted transcript at one seed passes held-out inheritance + cancellation + moat.
  2. inference gates (3-seed): held-out inheritance >= 0.80, cancellation == 1.0, moat 0 false-accepts.
  3. permuted control: scrambled experience -> no discoverable categories -> held-out inheritance collapses.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pytest

from research.runners import _emerge51_experiential_conversational_console as E


def _train(seed, permute=False):
    """Build + run the scripted observe -> is-a -> teach transcript; return the trained console."""
    c = E.ExperientialConversationalConsole(seed=seed, permute=permute)
    obs, isa, teach, _ = E._script_lines(seed)
    for line, _ in obs:
        E.handle(c, line)
    for line, _ in isa:
        E.handle(c, line)
    for line, _ in teach:
        E.handle(c, line)
    return c


def _held_inherit_map():
    hi = {m: "bird" for m in E._BIRD_HELDOUT}
    hi.update({m: "fish" for m in E._FISH_HELDOUT})
    return hi


def test_demo_self_check_seed42():
    """The scripted transcript renders the full loop: held-out members inherit, exception members cancel, moat abstains."""
    c, ch = E._run_demo_and_check(seed=42)
    # held-out inheritance: never-taught members inherit via the discovered codon (>= 3/4 at this seed)
    assert ch["inherit"] >= 0.75, f"held-out inheritance too low: {ch['inherit']}"
    # cancellation: both exception members answer their own specific fact
    assert ch["cancel"] == 1.0, f"cancellation failed: {ch['cancel']}"
    # the no-confab moat abstains on a never-observed token
    assert ch["moat_unknown"] is True
    # the natural-language replies read correctly
    assert ch["replies"]["heldout_inherit"].startswith("Yes,")          # inherited class default
    assert ch["replies"]["exception_cancel"].startswith("No,")          # the member's own exception (cancellation)
    assert "don't know what" in ch["replies"]["moat_unknown"]           # the moat


def test_inference_gates_3seed():
    """3-seed inference gates: held-out inheritance >= 0.80, cancellation == 1.0, moat 0 false-accepts on unknowns."""
    seeds = [42, 43, 44]
    hi = _held_inherit_map()
    inh, canc, fa = [], [], 0
    unknown_abstains = True
    for s in seeds:
        c = _train(s)
        inh.append(np.mean([c.inherit_ok(m, cn) for m, cn in hi.items()]))
        canc.append(np.mean([c.cancel_ok(m) for m in (E._BIRD_EXC[0], E._FISH_EXC[0])]))
        for tok in ("zzz", "qqq", "wobble"):                            # never-observed tokens must abstain
            if not c.moat_abstains(tok, "fly"):
                fa += 1
            if "don't know what" not in c.ask_can(tok, "fly"):
                unknown_abstains = False
    mean_inh = float(np.mean(inh))
    mean_canc = float(np.mean(canc))
    assert mean_inh >= 0.80, f"held-out inheritance {mean_inh:.2f} < 0.80"
    assert mean_canc == 1.0, f"cancellation {mean_canc:.2f} != 1.0"
    assert fa == 0, f"moat false-accepts on never-observed tokens: {fa}"
    assert unknown_abstains, "an unknown token did not produce the moat abstention"


def test_permuted_control_collapses_inheritance():
    """PERMUTED experience (scrambled feature vectors -> no discoverable categories) collapses held-out inheritance,
    isolating that the real result rides the DISCOVERED category structure (not a teaching artifact)."""
    hi = _held_inherit_map()
    real, perm = [], []
    for s in (42, 43, 44):
        cr = _train(s, permute=False)
        cp = _train(s, permute=True)
        real.append(np.mean([cr.inherit_ok(m, cn) for m, cn in hi.items()]))
        perm.append(np.mean([cp.inherit_ok(m, cn) for m, cn in hi.items()]))
    mean_real, mean_perm = float(np.mean(real)), float(np.mean(perm))
    assert mean_real >= mean_perm + 0.30, f"permuted did not collapse: real {mean_real:.2f} vs permuted {mean_perm:.2f}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
