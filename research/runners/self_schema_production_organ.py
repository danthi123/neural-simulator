"""SELF-SCHEMA AUTHORSHIP (self-vs-heard) — did the brain VOLUNTEER a proposition or RECALL a taught fact,
wired into the PRODUCTION turn (DR-3, board self-schema authorship, 2026-08-26).

The owner's honesty-boundary self-model, authorship axis: on an open-ended / continuous-ideation turn the brain
volunteers a PROPOSITION OF ITS OWN (a guess), which it must present as such ("a guess of my own, not something
I was taught") rather than as a stored fact. The live chat ALREADY flags such a turn as a `hypothesis` via a
HOST branch (`webapp/server.py` rich path: `is_hyp = bool(r.get("hypothesis"))`). This organ BACKS that host
flag with a genuinely-SPIKING neural authorship read: the DR-3 self_schema `author` sub-block fires 'self' on a
volunteered proposition and stays silent ('heard') on a recalled fact. The reply's own-guess MARKER then rides
that spiking read — and VANISHES when the schema's access is severed (the de-risk's self-lesion), reverting the
reply to the host default. This is a FUNCTIONAL self-model correlate — never a phenomenal claim.

REUSE-BY-IMPORT of the 6-seed GO mechanism (NO `sim/` edit):
  research/runners/_self_schema_region_derisk.py  ->  build_self_schema_bridge + _run_trial
  research/findings/2026-07-23-DR3-self-schema-region-6seed-GO.md  (authorship acc 1.000, chance 0.5;
  self-lesion collapses author to chance 6/6). ONE spiking `SimulationBridge` (workspace + shared inhibition +
  self_schema); the `author` sub-block is driven by a binary current when the thought is SELF-generated (vs
  externally heard) and its late-window firing RATE reports the authorship tag.

THE HOST BOUNDARY (declared, unchanged from the de-risk): WHICH authorship context a turn carries — a
volunteered proposition (self) vs a recalled fact (heard) — is supplied by the CALLER (`is_hyp`), exactly as the
de-risk's own authorship current is externally set per trial. The genuine SPIKING part, and the thing the marker
rides, is the author pool's readback of that state: it fires 'self' only when driven AND intact, and the
self-lesion (author access severed -> pool silent) demonstrably collapses the read to 'heard'.

THE AUTHORSHIP AXIS SELF-LESION: the de-risk's anti-cheat (1) severs the schema's access via `schema_access=False`
in `_run_trial` (author drive -> 0) while the workspace still ignites, so the BRAIN state is unchanged and only
the schema's READ is cut. For the AUTHORSHIP axis specifically that is the whole lesion (the member->attend
`lesion_schema` weight only touches the ATTENTION axis, which this organ does not read). Verified 2026-08-26:
intact self author_rate 0.092 -> 'self'; heard 0.000 -> 'heard'; lesioned-self 0.000 -> 'heard' (collapses).

MOAT-SAFE + ADDITIVE: this organ NEVER produces an answer, flips an abstain, or changes a recalled fact — it only
prepends an honest own-guess MARKER onto an already-produced HYPOTHESIS turn's reply (a turn the host already
flagged as a guess), when the author pool reads 'self'. On a recalled turn it is out of scope (never invoked).
DEFAULT-OFF: `BRAIN_SELF_SCHEMA` unset -> the byte-identical oracle (the organ is never built, no substrate step).

LESION-LOAD-BEARING: `BRAIN_SELF_SCHEMA_LESION=1` reads with the author access severed (`schema_access=False`);
the pool goes silent, the read collapses to 'heard', and the own-guess marker VANISHES -> the reply reverts to
the host default, proving the marker is driven by the LIVE author-pool read, not by a host `if is_hyp`.

FUNCTIONAL CORRELATE, NOT phenomenal: this measures + reports a self-model / agency CORRELATE (a learned-substrate
authorship read). It makes NO claim of subjective experience (phenomenal consciousness is OPEN, arguably untestable).

NO `sim/` edit; reuse-by-import; numpy-CPU backend (the DR-3 de-risk's own validated lane; the tiny ~690-neuron
bridge builds in ~0.07s and each read is ~0.02s, so it is cheap enough per turn).
"""
from __future__ import annotations

import os

from research.runners._self_schema_region_derisk import (
    build_self_schema_bridge,
    _run_trial,
)

AUTHOR_SELF = "self"
AUTHOR_HEARD = "heard"
AUTHORS = (AUTHOR_SELF, AUTHOR_HEARD)

# ── the de-risk's authorship operating point (its argparse defaults; the 6/6-GO regime) ──────────────────────
AUTHOR_PA = 650.0        # the SELF (volunteered) authorship drive current (de-risk --author-pa default)
CONF_PA = 450.0          # a fixed confidence drive to keep the trial well-formed; the author axis is ORTHOGONAL
                         # to confidence (de-risk dissociation), so this value does not affect the author read.
CONTENT_K = 0            # ignite a fixed workspace content; authorship is orthogonal to content (de-risk |corr|<0.16).


def self_schema_enabled() -> bool:
    """Default-OFF (the parent flips default-on after the pool soak passes). `BRAIN_SELF_SCHEMA` in
    {1,true,on,yes} turns the faculty on."""
    v = os.environ.get("BRAIN_SELF_SCHEMA")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "on", "yes")


def self_schema_lesioned() -> bool:
    """`BRAIN_SELF_SCHEMA_LESION` in {1,true,on,yes} -> the load-bearing self-lesion (author access severed at
    read time, `schema_access=False`; the de-risk's own verified failing-direction anti-cheat)."""
    v = os.environ.get("BRAIN_SELF_SCHEMA_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "on", "yes")


class SelfSchemaAuthorshipOrgan:
    """One DR-3 self_schema `SimulationBridge` (workspace + shared inhibition + self_schema) plus a build-time
    calibration of the self-vs-heard author-rate threshold. Each read ignites a fixed workspace content, holds the
    authorship drive (high for a volunteered proposition, zero for a recalled fact), free-runs, and reads the
    `author` sub-block's late-window firing RATE. A read >= the calibrated threshold reads 'self'.

    The read-time `lesion` flag passes `schema_access=False` into `_run_trial` (the de-risk's authorship
    self-lesion: author drive severed while the workspace still ignites), so the read collapses to 'heard'."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.bridge = self.xp = self.idx = self.snap = None
        self.threshold = None
        self.calib = None

    def ensure_built(self):
        if self._built:
            return
        # lesion_schema=False at BUILD: the authorship self-lesion is applied at READ time (schema_access=False),
        # which is the whole lesion for the author axis (member->attend `lesion_schema` only touches attention).
        self.bridge, self.xp, self.idx, self.snap = build_self_schema_bridge(
            seed=self.seed, lesion_schema=False)
        # CALIBRATE the self/heard threshold from one volunteered (self) and one recalled (heard) read (the same
        # author read a production turn uses), midpoint between them. Deterministic per seed.
        self_rate = self._author_rate(authored=True, lesion=False)
        heard_rate = self._author_rate(authored=False, lesion=False)
        self.threshold = 0.5 * (self_rate + heard_rate)
        self.calib = {"self_rate": float(self_rate), "heard_rate": float(heard_rate),
                      "threshold": float(self.threshold), "author_pa": AUTHOR_PA}
        self._built = True

    def _author_rate(self, authored: bool, lesion: bool = False) -> float:
        """The SPIKING author sub-block late-window firing RATE for a turn whose authorship context is `authored`
        (True = a volunteered proposition -> AUTHOR_PA drive; False = a recalled fact -> no drive). `lesion`
        severs the schema's access (`schema_access=False`) so the read collapses regardless of `authored`."""
        r = _run_trial(self.bridge, self.xp, self.idx, self.snap,
                       content_k=CONTENT_K, conf_current=CONF_PA,
                       author_current=(AUTHOR_PA if authored else 0.0),
                       schema_access=(not lesion))
        return float(r["author"])

    def read_author(self, authored: bool, lesion: bool = False) -> dict:
        """Read the authorship of a turn whose context is `authored`. Returns the spiking author rate, the
        calibrated threshold, and the decoded label ('self' when rate >= threshold, else 'heard'). Under `lesion`
        the author access is severed (`schema_access=False`) -> the pool goes silent -> the read collapses to
        'heard' (the load-bearing self-lesion)."""
        self.ensure_built()
        rate = self._author_rate(authored=authored, lesion=lesion)
        label = AUTHOR_SELF if rate >= self.threshold else AUTHOR_HEARD
        return {"on": True, "lesioned": bool(lesion), "authored": bool(authored),
                "author_rate": float(rate), "threshold": float(self.threshold),
                "label": label, "is_self": bool(label == AUTHOR_SELF), "calib": self.calib}


_ORGAN: SelfSchemaAuthorshipOrgan | None = None


def get_organ(seed: int = 42) -> SelfSchemaAuthorshipOrgan:
    """The process-shared DR-3 self-schema authorship organ (built once on first use). The read-time `lesion`
    flag is passed per-read (no rebuild needed — the authorship self-lesion is a read-time `schema_access=False`),
    so the SAME organ serves a normal request and a lesion-verification probe."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = SelfSchemaAuthorshipOrgan(seed=seed)
    return _ORGAN


def authorship_marker() -> str:
    """The honest functional own-guess MARKER prepended to a HYPOTHESIS turn's reply when the author pool reads
    'self'. A FUNCTIONAL read of the spiking author sub-block — never a phenomenal claim, never a content change."""
    return ("(My authorship monitor reads this as self-generated — a guess of my own, "
            "not something I was taught.) ")
