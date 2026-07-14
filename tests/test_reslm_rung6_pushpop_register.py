"""CI guard for RUNG 6 (cheap-first): the emergent generator predicts the RESUMED protagonist across a discourse POP via
a two-gate push/pop who-register (register accuracy high), where a single most-recent latch holds the interloper and the
fading reservoir + a shuffled register both collapse to chance. Offline numpy; fast."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_rung6_pushpop_register_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung6 deps unavailable: {e}")
    return run(42)


def test_register_resumes_protagonist(result):
    """The two-gate register restores the resumed (earlier) protagonist across the pop -> high accuracy."""
    assert result["register"] >= 0.75


def test_single_latch_holds_interloper(result):
    """The single most-recent latch holds the interloper (the Rung-2 ceiling) -> the register beats it decisively."""
    assert result["register"] > result["latch"] + 0.15


def test_reservoir_and_shuffle_collapse(result):
    """A faded reservoir + a shuffled who-state both collapse -> the two-gate who-state is load-bearing."""
    assert result["register"] > result["reservoir"] + 0.15
    assert result["register"] > result["shuffle"] + 0.15
    assert result["GO"]
