"""Session-wide test setup. Currently one job: disable the automatic provenance door
(`research/runners/__init__.py`) for the whole pytest session, before ANY test module is imported.

WHY THIS EXISTS (research/test-prov-door-isolation, 2026-09-25). `research.runners._ENABLED` is read from
`SIM_NO_PROVENANCE` exactly ONCE, at the package's first import, and cached in `sys.modules` for the rest of the
process -- a later `import research.runners` anywhere else in the same session just rebinds the name; it does not
re-run the module and does not re-read the env var. `tests/test_seam_contracts.py` disables the door around its
own import correctly, then asserts `not prov_door._ENABLED` as a POWER CONTROL that the disable actually took --
but that assertion only holds if test_seam_contracts.py's import is the FIRST import of `research.runners` in the
whole session. It is not: a static sweep (pairwise-collecting every test file against test_seam_contracts.py, plus
a module-level-only import-graph closure so lazy/in-function imports are not miscounted) found 300+ pre-existing
test files that import `research.runners` (directly, e.g. `import research.runners.some_runner`, or transitively
through a `webapp`/`tools` module such as `webapp.affect_drives_chat`) at module scope with no guard -- the
alphabetically-first is `tests/test_160_ensemble_pin.py`, which predates this fix by months. Collected in a full
suite, ANY of those, not just the two newest ones, latches the door ENABLED before test_seam_contracts.py ever
gets a chance to disable it, which is also why an incidental full-suite run once stamped stray `.prov.json`
sidecars: `_ENABLED=True` at import time immediately runs `_record_start()` (a real write to
research/findings/raw/_provenance/runs.jsonl) and registers the atexit sidecar-stamping hook, regardless of
whether anything in the session ever intended to produce a "run".

This is exactly the case the door's own docstring already names -- "SIM_NO_PROVENANCE=1 disables it entirely
(byte-identical reruns, CI)" -- a pytest session IS that CI case. Setting it here, once, before collection, is
the single point of enforcement instead of every test file re-deriving the same guard (many already do,
defensively; that convention is kept, this just makes it unconditional). It changes nothing about what the door
DOES when explicitly re-enabled: every seam/provenance test that needs `_ENABLED=True` behavior exercises it in a
CHILD PROCESS with its own explicit `env=` (grep `SIM_NO_PROVENANCE` across tests/ -- every such subprocess call
sets it itself), never by relying on this process's cached import state; and every test that calls the door's
writer functions directly (`_record_start`, `_stamp_outputs`, ...) does so as a plain function call, unguarded by
`_ENABLED` (that flag only gates the module's OWN top-level auto-invocation block, not the functions themselves).
`setdefault` still lets any future test override this explicitly (e.g. `monkeypatch.delenv` + `importlib.reload`)
if it genuinely needs the enabled path in-process.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
