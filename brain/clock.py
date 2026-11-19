# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
"""The world clock: the body's elapsed-time signal (declared scaffold, SCAFFOLDS.md).

Episode created/last_used, meta last_sleep and the strength decay read time through now(). The default is time.time, so
a brain that never calls set() behaves as before. A multi-week life (brain/eval/life.py) sets a simulated clock so that
days pass without waiting. Night seconds, locks and logs stay wall-clock (they use time.time directly).
"""
from __future__ import annotations

import time

_now = time.time


def now() -> float:
    return _now()


def set(fn=None) -> None:  # noqa: A001 (the name is the API)
    """Route now() through fn (a zero-argument callable returning seconds); None restores time.time."""
    global _now
    _now = time.time if fn is None else fn
