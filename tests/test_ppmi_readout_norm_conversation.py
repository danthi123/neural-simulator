"""Burndown #5 — the PPMI stream-cortex read-out NORMALIZATION (host ``double_center``) converted to an
ON-BRIDGE neural mechanism (per-hub spike-frequency ADAPTATION + per-concept FEEDFORWARD INHIBITION), validated
THROUGH the actual conversational who/what + no-confab-moat gate (not just the structure-proxy correlation).

Context. The host ``double_center`` (per-hub + per-concept mean-subtraction, a cognitive gain-control op done in
numpy) is shortcut #5 in ``research/findings/2026-06-20-shortcut-burndown-inventory.md``. Its neural replacement
``neural_norm`` (CYCLE 93b prescription, ``_phaseB_biologize_readout_norm_derisk.py``) was previously de-risked
ONLY at the structure-proxy level (``Pearson(cos, S_true) == 96% of host``). The burndown gate (#5) asks for the
END-TO-END proof: the ``neural``-normalized codes must reproduce the **who/what == the host baseline** AND keep
the **no-confab moat at 0 false-accepts**, multi-seed. That conversational validation is what these tests close.

CPU/numpy only (the corpus counts, both normalizations, and the HRR who/what + moat pipeline are all numpy; no
GPU bridge re-stream needed — the normalization swap is isolated on the SAME learned log-domain block-mean ``L``).

The no-confab moat is a HARD invariant here: 0 false-accepts on the ``neural`` path (never weakened).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse-by-import the PRODUCTION-faithful pieces so the test exercises exactly what ships.
from research.runners._phaseB_onbridge_stream_cortex_derisk import double_center  # noqa: E402
from research.runners._phaseB_biologize_readout_norm_derisk import neural_norm  # noqa: E402
from research.runners._phaseB_onbridge_stream_conversation_derisk import run_conversation  # noqa: E402

_CORPUS = os.path.join(_REPO, "data", "corpus", "tinystories.txt")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_CORPUS),
    reason="TinyStories corpus (data/corpus/tinystories.txt) absent; stream-cortex tests need it.",
)

N_HUB = 300
SEEDS = (42, 43, 44)


def _learn_L(seed):
    """Learn the log-domain block-mean ``L`` from the REAL corpus (numpy; the ``neural_norm`` proxy for the
    on-bridge learned weight block, ``corr(M, C) ~ 0.9``). Returns ``(L [Nt, n_hub], labels)``."""
    from research.runners.learned_graded_cortex_fair_test import build_real_corpus
    C, labels, _S_true = build_real_corpus(seed, N_HUB)
    L = np.log1p(C * 100.0)                      # the f-I / Weber-Fechner read-out (pre-centre); same as production
    return L, np.asarray(labels)


def _codes_from(L, norm, seed):
    """Apply a read-out normalization to ``L`` and unit-normalize per concept (exactly the production path in
    ``_phaseB_onbridge_stream_conversation_derisk.stream_learn_codes``)."""
    if norm == "host":
        code = double_center(L)
    elif norm == "neural":
        code = neural_norm(L, np.random.RandomState(seed * 911 + 7))
    else:
        raise ValueError(norm)
    return code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-12)


def _converse(seed, norm, moat="learned"):
    """Learn L, normalize, and run the EXACT CYCLE-90 who/what + no-confab pipeline."""
    L, labels = _learn_L(seed)
    codes = _codes_from(L, norm, seed)
    return run_conversation(codes, labels, seed, moat=moat)


# --------------------------------------------------------------------------------------------------------------
# 1. The MOAT is the hard invariant: on the NEURAL read-out, ABSENT (verb,object) queries abstain — 0 false-accepts.
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_neural_norm_moat_zero_false_accepts(seed):
    r = _converse(seed, "neural", moat="learned")
    assert r["false_accept"] == 0, (
        f"NEURAL-normed seed {seed}: the no-confab moat LEAKED ({r['false_accept']} false-accepts) — "
        f"the on-bridge read-out normalization must NOT weaken the moat."
    )
    assert r["abstain"] == 1.0, f"NEURAL-normed seed {seed}: abstain {r['abstain']} < 1.0"


# --------------------------------------------------------------------------------------------------------------
# 2. WHO/WHAT parity: the NEURAL read-out reproduces the host double-centre's who/what recall (within tolerance).
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_neural_norm_whoami_parity(seed):
    host = _converse(seed, "host", moat="learned")
    neural = _converse(seed, "neural", moat="learned")
    # The neural read-out must not lose who/what recall vs the host double-centre. Allow a small (one-fact-set)
    # tolerance for the rate-coded-pool noise on the subtracted means; the gate is parity, not a tuned win.
    assert neural["recall"] >= host["recall"] - 0.125, (
        f"seed {seed}: NEURAL who/what recall {neural['recall']:.3f} fell below host {host['recall']:.3f} "
        f"by more than one fact (>0.125) — the neural centring degraded the codes."
    )


# --------------------------------------------------------------------------------------------------------------
# 3. The familiarity GAP is preserved (present >> absent) — the structure the moat reads survives the swap.
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_neural_norm_familiarity_gap_preserved(seed):
    neural = _converse(seed, "neural", moat="learned")
    gap = neural["conf_present"] - neural["conf_absent"]
    assert gap >= 0.10, (
        f"seed {seed}: NEURAL familiarity gap {gap:+.3f} (present {neural['conf_present']:+.3f} vs "
        f"absent {neural['conf_absent']:+.3f}) collapsed below 0.10."
    )


# --------------------------------------------------------------------------------------------------------------
# 4. DEFAULT == byte-identical: the conversation runner's default (``readout_norm='host'``) is unchanged — the
#    neural path is strictly opt-in. (Guards that the conversion did not perturb the host default.)
# --------------------------------------------------------------------------------------------------------------
def test_host_default_unchanged():
    """The host double-centre stays the default normalization in the conversation runner."""
    import inspect

    from research.runners import _phaseB_onbridge_stream_conversation_derisk as m
    src = inspect.getsource(m.main)
    # The CLI default for --readout-norm must remain "host" (the neural path is opt-in / non-default).
    assert '"--readout-norm", default="host"' in src, (
        "the conversation runner's --readout-norm default is no longer 'host' — the neural read-out must stay "
        "opt-in so the cached host codes / default path are byte-preserved."
    )


# --------------------------------------------------------------------------------------------------------------
# 5. ANTI-CHEAT: the neural normalization is LOAD-BEARING — dropping it (no-norm) collapses who/what or the moat,
#    confirming the neural centring is doing real work (not that the bind tolerates anything).
# --------------------------------------------------------------------------------------------------------------
def test_no_norm_control_is_worse():
    """A no-normalization control (raw ``L``, unit-normed) must under-perform the neural read-out — either who/what
    recall drops or the moat leaks — so the neural centring is demonstrably load-bearing."""
    seed = 42
    L, labels = _learn_L(seed)
    raw = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
    nonorm = run_conversation(raw, labels, seed, moat="learned")
    neural = _converse(seed, "neural", moat="learned")
    degraded = (nonorm["recall"] < neural["recall"]) or (nonorm["false_accept"] > neural["false_accept"])
    assert degraded, (
        f"seed {seed}: no-norm control was NOT worse than neural "
        f"(no-norm recall {nonorm['recall']:.2f}/FA {nonorm['false_accept']} vs "
        f"neural recall {neural['recall']:.2f}/FA {neural['false_accept']}) — "
        f"the neural normalization would not be load-bearing."
    )
