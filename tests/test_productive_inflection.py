"""CI guard for PRODUCTIVE regular inflection ON SPIKES: a novel 3sg verb form (stored as a whole-form lexeme
NOWHERE) is composed neurally as spell(stem) + spell(affix), both decoded from `language_output` spikes --
the biology-grounded replacement for the host `emerge_v3` string op (Pinker-Ullman procedural route).

Skips gracefully if the BRIDGE-1 stem bridge or the affix A->W bridge is absent (regenerable). numpy-only.
"""
import os
import pytest

BRIDGE1 = "bridges/breadth_aw/seed42.simstate.h5"
AFFIX = "bridges/affix_aw/seed42.simstate.h5"
pytestmark = pytest.mark.skipif(not (os.path.exists(BRIDGE1) and os.path.exists(AFFIX)),
                                reason="needs the BRIDGE-1 + affix A->W checkpoints (regenerable)")


def test_productive_3sg_composed_on_spikes():
    """A novel 3sg (run->runs, ...) whose whole form is stored nowhere is produced as spell(stem)+spell('s')
    on spikes; wrong-affix and affix-ablation controls collapse."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners._realcorpus_productive_inflection_derisk import run, NOVEL
    r = run(seed=42, affix_seed=42)
    assert r["productive"], "the tested 3sg forms must be stored lexemes nowhere (genuinely productive)"
    assert r["n_ok"] == r["n"] and r["n"] == len(NOVEL)          # every novel 3sg produced exactly on spikes
    assert r["wrong_affix_ok"]                                   # stem + wrong morpheme != the 3sg (affix load-bearing)
    assert r["ablation_ok"]                                      # the bare stem != the 3sg (affix slot load-bearing)
