"""Import-light flag reader for BRAIN_XEDGE_IN_WAVE3 (the d6 w{k}->sel cross-edge grown INSIDE the production
merged cortical pool; see `onebrain_xedge_wave3`). Dependency-free at import (os only), so the two production
routing points (`onebrain_wave3_pool_production.get_merged_cortical_pool`, `onebrain_xedge_production.
XedgeProductionPool._build`) can check it without importing the merge framework: unset -> both take their
unchanged pre-flag path, byte-identical to main.
"""
from __future__ import annotations

import os

_TRUE = ("1", "true", "yes", "on")


def xedge_in_wave3_flag() -> bool:
    """The raw DEFAULT-OFF flag: `BRAIN_XEDGE_IN_WAVE3` in {1,true,yes,on}."""
    v = os.environ.get("BRAIN_XEDGE_IN_WAVE3")
    return v is not None and v.strip().lower() in _TRUE


def xedge_in_wave3_enabled() -> bool:
    """True iff the flag is set AND every precondition of the in-pool cross-edge holds, so BOTH routing points
    always agree (a half-applied flag would re-create the severance this flag exists to remove):
      * `xedge_enabled()` (BRAIN_ONEBRAIN_XEDGE, default ON) -- the cross-edge exists at all;
      * `xedge_learn_enabled()` (default ON) -- the in-pool build supports the LEARNED edge only (the PART-1
        frozen host-schedule edge, BRAIN_ONEBRAIN_XEDGE_LEARN=0, stays on its separate pool);
      * `wave3_pool_enabled()` (default ON) -- there is a merged cortical pool to grow it in;
      * NOT `affect_pool_enabled()` (default OFF) -- the 12-organ affect pool is not extended tonight.
    Any precondition false -> False -> both routing points take their unchanged path."""
    if not xedge_in_wave3_flag():
        return False
    from research.runners.onebrain_xedge_production import xedge_enabled, xedge_learn_enabled
    from research.runners.onebrain_wave3_pool_production import wave3_pool_enabled
    from research.runners.onebrain_affect_pool_flags import affect_pool_enabled
    return bool(xedge_enabled() and xedge_learn_enabled() and wave3_pool_enabled() and not affect_pool_enabled())
