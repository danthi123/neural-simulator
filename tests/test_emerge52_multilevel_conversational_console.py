"""CI guard for EMERGE-52: the MULTI-LEVEL conversational console. CPU/numpy, offline. Verifies the composed pieces
(EMERGE-44/45 stacked multi-level discovery + EMERGE-51 NL console + EMERGE-42 cancellation) hold at seed 42: a held-out
member inherits a GENUS property (1 discovered level up) AND an ORDER property (2 discovered levels up); sibling-branch
properties are NOT inherited (codon-driven sibling-discrimination); the exception member cancels; and the no-confab moat
abstains on an unknown token with 0 false-accepts. Also checks the load-bearing collapse control: permute-co-occurrence
breaks the (codon-driven) sibling-discrimination on at least one probed seed."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pytest

from research.runners._emerge52_multilevel_conversational_console import (
    _check, _BIRD_HELDOUT, _FISH_HELDOUT, _BIRD_EXC, handle)


def test_multilevel_inheritance_sibling_cancel_moat_seed42():
    """One scripted seed: 2-level (order) + 1-level (genus) inheritance, sibling-discrimination, cancellation, moat."""
    c, ch = _check(seed=42, permute=False)
    # 2-level (order 'breathe', 2 discovered levels up) + 1-level (genus 'fly'/'swim') inheritance for held-out members
    assert ch["two_level"] >= 0.99, ch
    assert ch["one_level_genus_floor"] >= 0.99, ch
    # sibling-discrimination: a held-out bird does NOT inherit the fish branch's 'swim' (and vice versa)
    assert ch["sibling_confusion"] <= 0.01, ch
    # cancellation: the exception member answers ITS specific fact
    assert ch["cancel"] >= 0.99, ch
    # the no-confab moat abstains on an unknown token
    assert ch["moat_unknown"] is True, ch


def test_natural_language_replies_seed42():
    """The natural-language transcript reads correctly across levels + moat."""
    c, _ = _check(seed=42, permute=False)
    ho = _BIRD_HELDOUT
    assert handle(c, "can a %s fly?" % ho).startswith("Yes"), "genus (1-level) inheritance"
    assert handle(c, "can a %s breathe?" % ho).startswith("Yes"), "order (2-level) inheritance"
    assert handle(c, "can a %s swim?" % ho).startswith("I don't know"), "sibling-discrimination"
    assert handle(c, "can a %s fly?" % _BIRD_EXC[0]).startswith("No"), "cancellation"
    assert handle(c, "can a zzz breathe?").startswith("I don't know what"), "no-confab moat (unknown token)"


def test_moat_no_false_accepts_seed42():
    """No never-observed token is ever accepted (0 false-accepts)."""
    c, _ = _check(seed=42, permute=False)
    for tok in ("zzz", "qqq", "wobble", "flarn"):
        assert c.moat_abstains(tok, "breathes"), tok


def test_permute_cooc_breaks_sibling_discrimination():
    """LOAD-BEARING control: scrambling the co-occurrence pairs breaks the codon-driven sibling-discrimination on at least
    one of the probed seeds (real ~0 -> permuted raises sibling-confusion). Seed-variable per EMERGE-45's honest scope, so
    we probe a few seeds and require the collapse to appear on at least one."""
    raised = []
    for seed in (42, 43, 44):
        _, ch_real = _check(seed=seed, permute=False)
        _, ch_perm = _check(seed=seed, permute=True)
        assert ch_real["sibling_confusion"] <= 0.01, (seed, ch_real)     # real is always clean
        raised.append(ch_perm["sibling_confusion"] - ch_real["sibling_confusion"])
    assert max(raised) >= 0.25, raised


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
