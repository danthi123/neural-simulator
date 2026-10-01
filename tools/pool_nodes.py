#!/usr/bin/env python3
"""tools/pool_nodes.py — the Python half of the single source of truth for the default mini-PC pool
node list (the shell half is tools/pool_nodes.sh). Reads tools/pool_nodes.conf, re-parsed fresh on
every call (this is not a daemon, so there is no caching to go stale).

Precedence is unchanged and is the CALLER's job, not this module's: an explicit POOL_NODES env var or
explicit CLI node arguments must still win over DEFAULT_POOL_NODES; research/queue/.pool_extra_nodes
(AWS/extra nodes) is still appended on top wherever a caller already appended it, read fresh each call.

    from tools.pool_nodes import default_nodes, DEFAULT_POOL_NODES   # when run/imported as a package
    # or, for a script loaded by path (no "tools." prefix available):
    #   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from pool_nodes import default_nodes
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
POOL_NODES_CONF_FILE = os.path.join(_HERE, "pool_nodes.conf")


def default_nodes(conf_path: str | None = None) -> list[str]:
    """Return the default pool node list, parsed fresh from tools/pool_nodes.conf every call.

    conf_path overrides the file read (tests only); an unset/missing conf file returns [] rather than
    raising, so a caller that forgets to check can fail loudly downstream instead of silently.
    """
    path = conf_path or os.environ.get("POOL_NODES_CONF_FILE", POOL_NODES_CONF_FILE)
    nodes: list[str] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    nodes.extend(line.split())
    except FileNotFoundError:
        return []
    return nodes


# Module-level convenience for callers that just want the list once at import time (matches the old
# `POOL_NODES = ("pool40", "pool41", "pool42")` / `POOL = ["pool40", "pool41", "pool42"]` module-level
# constants this module replaces). Prefer default_nodes() directly if a caller wants a fresh re-read.
DEFAULT_POOL_NODES = default_nodes()


if __name__ == "__main__":
    print(" ".join(default_nodes()))
