"""Import-light flag readers for the D3 affect->one-brain-pool migration (`onebrain_affect_pool`).

Kept in their own dependency-free module so the production call sites (`affect_production_organ`,
`onebrain_wave3_pool_production.get_merged_cortical_pool`) can check the DEFAULT-OFF flags without importing the
merge framework: with the flags unset, production imports nothing new and runs byte-identically to main.
"""
from __future__ import annotations

import os


def _flag(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def affect_pool_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_ONEBRAIN_AFFECT_POOL=1` -> the affect ladder is read off the shared 12-organ pool and
    `get_merged_cortical_pool` routes every wired cortical organ onto that SAME pool (only while the Wave-3 pool,
    default-ON, is itself on)."""
    return _flag("BRAIN_ONEBRAIN_AFFECT_POOL")


def affect_xedge_enabled() -> bool:
    """DEFAULT-OFF. `BRAIN_ONEBRAIN_AFFECT_XEDGE=1` (with the pool flag) installs the arousal->surprise synapse."""
    return _flag("BRAIN_ONEBRAIN_AFFECT_XEDGE")
