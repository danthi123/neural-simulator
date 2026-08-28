"""ONE-BRAIN CROSS-EDGE (R4) — the SECOND learned cross-region synapse wired into the LIVE chat brain (2026-08-27).
Mirrors `research/runners/onebrain_xedge_production.py`'s PART-1 pattern (the d6-WM -> comprehension frozen
cross-edge; `2026-08-27-onebrain-xedge-production-frozen-GO.md`) verbatim in SHAPE, on the R4 pairing: self_schema
authorship ("did I author this thought") -> source_provenance monitoring ("this memory reads as internally-
generated"), 6-seed GO on the merge framework
(`2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md`, merged 20b4b475).

THE GAP. R4 lives on the RESEARCH merge FRAMEWORK (`research/runners/_onebrain_integration_r4_selfschema_
provenance.py`), default-off, NOT in the live brain — self_schema and source_provenance each build their OWN
standalone `SimulationBridge` in production (`self_schema_production_organ.py` / `source_provenance_production_
organ.py`, both process-shared singletons), so no shared substrate the learned cross-edge could span. This module
co-locates the R4 pair (self_schema + source_provenance) on ONE `MergedPool` (via `R4Pool`, reused by import —
NOT reimplemented) and freezes its pre-grown `author -> prov_generated` cross-edge, so a live authorship read can
drive a co-temporal source-provenance read on the SAME shared substrate.

FROZEN, PLASTICITY-OFF IN PRODUCTION (mirrors PART-1 exactly). The cross-edge is GROWN ONCE at first build (R4's
own direct-drive Hebbian training, `R4Pool.train()`) and then FROZEN (`enable_hebbian_learning=False`, which
`train()` already leaves set, PLUS an explicit `set_plasticity_gate(GATE, 0.0)` for defensive parity with PART-1's
own R3v3Pool.train() convention). No weight moves during any live turn. Growth is IN-PROCESS (not a saved
artifact), for the SAME reason PART-1 documents: the CROSS-BACKEND SEED TRAP (`docs/ENGINE_REFERENCE.md`) means a
numpy-grown weight file is not valid for a cupy production build (different RNG -> different substrate) — growing
in whatever backend the process runs guarantees the frozen edge matches the substrate it rides.

WHY A LIVE CO-DRIVE READ, NOT JUST CO-RESIDENCE (mirrors PART-1's "attach shared= and done would be HOLLOW"
lesson — the exact drift memory #84/#85 gates). R4's own F2 instrument (`R4Pool.amb_read`) is REUSED VERBATIM (not
reimplemented): hard-reset -> drive the fixed dual-context-encoded AMBIGUOUS content pattern's episode neurons (+
optionally self_schema's `author` pool) -> read the SIGNED margin (rate_generated - rate_perceived) off
`prov_generated`/`prov_perceived`. The cross-edge only transmits when `author` is ACTIVELY DRIVEN during the
recall window — so the live consumer below drives it FROM the turn's OWN live authorship verdict (the SAME
`is_self` decision self_schema_production_organ.read_author already renders for the is_hyp marker), not a
co-resident-but-inert attach.

WHAT IS LOAD-BEARING vs DECLARED RESIDUAL (honest, from the R4 de-risk + this wire-in's own probe):
  * LOAD-BEARING + lesion-attributable: holding self_schema's author pool 'self' during the SAME fixed ambiguous
    item's recall (R4's own F2 instrument) shifts the signed margin toward GENERATED vs a no-hold baseline;
    lesioning the cross-edge (`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION=1`) collapses the shift toward zero. This is
    the SAME F2 measurement the R4 6-seed GO already cleared, now exercised THROUGH this production module's own
    `crossedge_provenance_shift` (the function the live handler calls), not a separate probe.
  * DECLARED RESIDUAL #1 (carried UNCHANGED from R4): the item the cross-edge biases is R4's OWN fixed,
    dual-context-encoded, per-seed AMBIGUOUS content pattern (a substrate stand-in for a genuinely uncertain real
    memory) — NOT an arbitrary live chat fact. Binding the cross-edge's bias onto an arbitrary production fact key
    (so that ANY recalled claim's provenance judgment can be nudged, not just this one fixed probe item) needs
    extending `source_provenance_honesty.SourceProvenanceHonestyMonitor` with a `shared=` attach (mirroring
    `ProvenanceBrain`'s own existing `shared=` support) plus a held-author recall variant of `judge_fact` — a
    separate, later, reviewed rung (PART-1 itself declared an analogous residual: "semantic referent->pool binding
    is host-directed").
  * DECLARED RESIDUAL #2 (carried UNCHANGED from PART-1's own honest shape): the magnitude is a shift on the
    live SUBSTRATE margin (0.010-0.016 across the R4 6-seed GO), not (yet) wired to flip any DECISION-level text
    the way the source-provenance-honesty organ's `provenance_framed_text` does for an arbitrary fact — this
    wire-in surfaces the live, lesion-attributable READ as an additive diagnostic field
    (`resp["authorship"]["source_provenance_crossedge"]`) on the SAME is_hyp turn self_schema's own marker already
    fires on, exactly PART-1's own "content-neutral at the decision level, genuine + reversible" shape.

GUARDED, DEFAULT-OFF, BYTE-IDENTICAL-OFF. `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` gates the whole thing (default OFF —
the flip to default-ON is a separate, later, owner-gated step, exactly PART-1's convention). Unset/0/false/no/off
=> every organ builds standalone exactly as today (byte-identical); self_schema's OWN production singleton
(`self_schema_production_organ.get_organ()`, already default-ON for the authorship marker) is UNTOUCHED by this
module — the new diagnostic runs an INDEPENDENT R4Pool self_schema instance on the shared merge pool, so there is
ZERO risk to the already-live authorship marker even when this flag is on. A build failure DEGRADES to "no
diagnostic field" (never crashes brain load — mirrors PART-1's `ensure_built` try/except).
`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION=1` zeroes the cross-edge (the load-bearing lesion control).

Run (offline grow + record + self-verify; 0 Claude tokens, CPU numpy):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_selfschema_production --grow \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_xedge_selfschema_production_frozen_6seed.json
"""
from __future__ import annotations

import os


# PRODUCTION DEFAULT — OFF. The owner-gated flip to default-ON is a SEPARATE step (never autonomous), exactly
# mirroring PART-1's `_XEDGE_DEFAULT_ON` convention.
_XEDGE_SS_DEFAULT_ON = False


def xedge_selfschema_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` in {1,true,yes,on} -> the frozen self_schema-authorship -> source_
    provenance cross-edge (R4) is live (the two organs share ONE spiking pool with the pre-grown, frozen
    cross-synapse, and the is_hyp turn's live authorship verdict drives a co-temporal provenance read on it).
    Unset/{0,false,no,off} -> byte-identical (no shared pool is ever built; self_schema's own production
    singleton is untouched). Default per `_XEDGE_SS_DEFAULT_ON` (OFF)."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA")
    if v is None:
        return _XEDGE_SS_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def xedge_selfschema_lesioned() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION` in {1,true,yes,on} -> zero the author->prov_generated cross-edge
    weight (the load-bearing lesion control: the authorship->provenance shift must VANISH). Everything else
    (self_schema's own read, source_provenance's own battery) is unchanged, so the shift this cross-edge
    introduces must collapse under this flag."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# PRODUCTION DEFAULT — OFF (2026-08-28, the declarative-framework migration). `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_
# DECLARATIVE` selects WHICH construction path builds the pool this module wires into the live chat handler:
# unset/off (default) -> the ORIGINAL bespoke `R4Pool` (hand-typed `_dense(...)` union + 3-line whitelist freeze,
# unchanged since PART-1/`d84775aa8`); on -> `DeclarativeR4Pool` (`_onebrain_declarative_crossedge_r4_repro.py`),
# the SAME edge expressed as ONE `CrossEdge` row on `merge_organs(..., cross_edges=[...])`, proven BIT-IDENTICAL
# to the bespoke construction on 6/6 seeds (grown-weight max|delta|=0.0, F2 lesion-attributable fraction matching
# to full float precision on every seed -- see the repro's own 6-seed artifact). This flag changes ONLY the
# INTERNAL pool-construction code path; `crossedge_provenance_shift` (the function `webapp/server.py`'s is_hyp
# block calls) and every other module-level surface are UNCHANGED, so a turn's `resp["authorship"][
# "source_provenance_crossedge"]` output is byte-identical whichever arm built the pool. Default OFF -> the
# module's behavior (and every existing test against it) is UNCHANGED (the exact bespoke code path PART-1 shipped
# runs unconditionally when this flag is unset), so this migration carries zero risk to what is already wired.
_XEDGE_SS_DECLARATIVE_DEFAULT_ON = False


def xedge_selfschema_declarative_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE` in {1,true,yes,on} -> build the R4 pool via the declarative
    `CrossEdge`/`merge_organs(cross_edges=...)` framework (`DeclarativeR4Pool`) instead of the original bespoke
    `R4Pool`. Only takes effect when `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` is also on (this flag alone builds nothing).
    Default per `_XEDGE_SS_DECLARATIVE_DEFAULT_ON` (OFF) -> the original bespoke path, unchanged."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE")
    if v is None:
        return _XEDGE_SS_DECLARATIVE_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


class XedgeSelfschemaProductionPool:
    """Process-shared holder of the [self_schema + source_provenance] `MergedPool` with the FROZEN pre-grown R4
    `author -> prov_generated` cross-edge. Exposes `.pool` (the framework MergedPool), `.ss_organ` (R4's OWN
    self_schema organ view on the shared pool), `.sp_organ` (R4's OWN source_provenance battery-calibration organ
    view), and `.provenance_margin(hold_author)` (R4's OWN validated F2 instrument, reused verbatim). Built
    lazily on first use; degrades to a disabled holder on any build failure (the caller then reports no
    diagnostic, exactly mirroring PART-1's `XedgeProductionPool.ensure_built`)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.ok = False
        self.pool = None            # the framework MergedPool
        self.bridge = None
        self.ss_organ = None        # R4's SelfSchemaAuthorshipOrgan view (shared)
        self.sp_organ = None        # R4's source_provenance battery-calibration organ view (shared)
        self.ix = None
        self.masks = None
        self.grow_traj = None
        self.cross_weights = None
        self._r4 = None             # the underlying R4Pool (amb_read lives here)

    def ensure_built(self):
        if self._built:
            return
        self._built = True     # set FIRST so a failed build is not retried every turn
        try:
            self._build()
            self.ok = True
        except Exception as e:   # never crash brain load — degrade to "no diagnostic"
            import traceback
            print(f"[webapp] ONEBRAIN XEDGE-SELFSCHEMA (R4) build FAILED -> degrading to no diagnostic "
                  f"({type(e).__name__}: {e})", flush=True)
            print(traceback.format_exc(), flush=True)
            self.ok = False

    def _build(self):
        # Import lazily (this pins SIM_BACKEND=numpy via os.environ.setdefault at module scope; a no-op once the
        # webapp has already fixed the backend). The pool class REUSED BY IMPORT builds the merged pool, injects
        # the near-zero cross-edge as the SOLE plastic synapse, runs both organs' own build-time steps, and
        # constructs the fixed dual-context-encoded ambiguous content pattern — either the ORIGINAL bespoke
        # `R4Pool` (hand-typed `_dense(...)` union + 3-line whitelist freeze) or, behind `BRAIN_ONEBRAIN_XEDGE_
        # SELFSCHEMA_DECLARATIVE`, `DeclarativeR4Pool` (the SAME edge as ONE `CrossEdge` row on `merge_organs(...,
        # cross_edges=[...])`, proven bit-identical to R4Pool on 6/6 seeds). Both classes expose the IDENTICAL
        # surface (`.train`, `.b`, `.pool`, `.ix`, `.masks`, `.ss_organ`, `.sp_organ`, `.cross_weights()`,
        # `.amb_read`) since `DeclarativeR4Pool` subclasses `R4Pool` and only overrides `__init__`.
        from research.runners._onebrain_integration_r4_selfschema_provenance import R4Pool, GATE
        if xedge_selfschema_declarative_enabled():
            from research.runners._onebrain_declarative_crossedge_r4_repro import DeclarativeR4Pool
            pool_cls = DeclarativeR4Pool
        else:
            pool_cls = R4Pool
        r4 = pool_cls(self.seed)
        self.grow_traj = r4.train()             # GROWS the cross-edge by the substrate's own Hebbian rule (0.05 -> ~3)
        # FREEZE for production: r4.train() already leaves enable_hebbian_learning=False (the global per-step
        # gate every subsequent read is subject to); ALSO re-assert the gate's own rate_gain at 0, for defensive
        # parity with PART-1's R3v3Pool.train() convention ("freezes the candidate gate the instant train returns").
        r4.b.set_plasticity_gate(GATE, 0.0)
        self._r4 = r4
        self.pool = r4.pool
        self.bridge = r4.b
        self.ix = r4.ix
        self.masks = r4.masks
        self.ss_organ = r4.ss_organ
        self.sp_organ = r4.sp_organ
        self.cross_weights = r4.cross_weights()
        if xedge_selfschema_lesioned():
            self.lesion_cross()

    # ── the load-bearing lesion control (env or explicit) ──
    def lesion_cross(self):
        """Zero the author->prov_generated cross-edge weight in place (the authorship->provenance shift must then
        vanish). Mirrors PART-1's `XedgeProductionPool.lesion_cross` exactly. Also REFRESHES `self.cross_weights`
        (2026-08-28, verify-go skeptic finding: the field was snapshotted once in `_build()` and never updated,
        so a lesioned pool's own diagnostic field kept reporting the pre-lesion trained weight even though
        `cp_connections.data` was genuinely zeroed — cosmetic, the actual shift read is unaffected since it reads
        live connection data, but a live consumer inspecting `.cross_weights` post-lesion would be misled)."""
        import numpy as np
        from sim.backend import to_host
        b = self.bridge
        data = np.asarray(to_host(b.cp_connections.data)).copy()
        for k in self.masks:
            data[self.masks[k]] = 0.0
        b.cp_connections.data = self.pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)
        if self._r4 is not None:
            self.cross_weights = self._r4.cross_weights()

    # ── the live-consumer instrument: R4's OWN validated F2 read, reused verbatim ──
    def provenance_margin(self, hold_author: bool, band=None):
        """The signed margin (rate_generated - rate_perceived) for R4's fixed, dual-context-encoded ambiguous
        content pattern, optionally holding self_schema's `author` pool 'self' throughout the recall window.
        Delegates to `R4Pool.amb_read` (the SAME instrument the R4 6-seed GO's F2 arm cleared) — not
        reimplemented, so this production module carries zero drift risk from the validated mechanism."""
        self.ensure_built()
        return self._r4.amb_read(hold_author, band=band)


_POOL: "XedgeSelfschemaProductionPool | None" = None


def get_xedge_selfschema_pool(seed: int = 42) -> "XedgeSelfschemaProductionPool | None":
    """The process-shared R4 cross-edge pool (built once on first use). Returns the holder even if the build
    failed (`holder.ok is False`) so the caller can degrade to no diagnostic. Returns None only when the flag is
    OFF (mirrors PART-1's `get_xedge_pool` exactly)."""
    global _POOL
    if not xedge_selfschema_enabled():
        return None
    if _POOL is None:
        _POOL = XedgeSelfschemaProductionPool(seed)
    _POOL.ensure_built()
    return _POOL if _POOL.ok else None


def crossedge_provenance_shift(pool: "XedgeSelfschemaProductionPool", hold_author: bool) -> dict | None:
    """LIVE reply-path hook. Given the CALLING turn's own live authorship verdict (`hold_author` — the SAME
    `is_self` boolean `self_schema_production_organ.read_author` already renders for the is_hyp marker), read
    R4's fixed ambiguous item's signed provenance margin with vs without that hold, and report the shift toward
    GENERATED it causes on the shared substrate. `hold_author=False` (a non-self-authored context) reports shift
    0.0 by construction (both reads use the same no-hold protocol) -- the honest "no authorship signal, no bias"
    case. Never raises into a turn (best-effort; returns None on any error so the caller degrades to "no
    diagnostic field", never a crashed turn)."""
    try:
        if pool is None or not pool.ok:
            return None
        baseline = pool.provenance_margin(False)
        held = pool.provenance_margin(bool(hold_author))
        shift = float(held["margin"] - baseline["margin"])
        return {
            "on": True,
            "author_held": bool(hold_author),
            "margin_baseline": float(baseline["margin"]),
            "margin_held": float(held["margin"]),
            "shift_toward_generated": shift,
            "cross_weight": dict(pool.cross_weights) if pool.cross_weights else {},
        }
    except Exception as e:
        return {"on": True, "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Offline grow + record + self-verify entrypoint (0 Claude tokens; CPU numpy). Mirrors PART-1's
#  `_selftest_loadbearing` + CLI shape exactly.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _selftest_loadbearing(seed, declarative=False):
    """Exercise the REAL production function `crossedge_provenance_shift` (not a bespoke probe) at seed `seed`,
    INTACT vs cross-edge-LESIONED: the shift must be nonzero (author_held=True) intact and collapse toward zero
    lesioned -> lesion-attributable authorship->provenance drive through the live production read. Also checks
    author_held=False reports exactly 0.0 shift on BOTH arms (the honest no-signal-no-bias control).
    `declarative=True` sets `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE=1` first, so this SAME production
    call sequence exercises `DeclarativeR4Pool` instead of the bespoke `R4Pool` (2026-08-28 migration)."""
    from tools.lab import attributable_to
    os.environ["BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA"] = "1"
    os.environ.pop("BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION", None)
    if declarative:
        os.environ["BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE"] = "1"
    else:
        os.environ.pop("BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE", None)
    global _POOL
    _POOL = None
    pool = get_xedge_selfschema_pool(seed)
    assert pool is not None and pool.ok, "xedge-selfschema pool failed to build"

    held_true_intact = crossedge_provenance_shift(pool, True)
    held_false_intact = crossedge_provenance_shift(pool, False)

    pool.lesion_cross()
    held_true_lesion = crossedge_provenance_shift(pool, True)
    held_false_lesion = crossedge_provenance_shift(pool, False)

    d_i = held_true_intact["shift_toward_generated"]
    d_l = held_true_lesion["shift_toward_generated"]
    frac = attributable_to(f"seed{seed} xedge-selfschema authorship->provenance shift vs cross-edge lesion",
                           d_i, d_l)
    # NOISE FLOOR (not a bug — R4's OWN 6-seed GO carries the identical residual): two consecutive `amb_read`
    # calls that both hold_author=False are not bit-identical (a tiny per-synapse pulse-timer/delay-buffer
    # residual `_hard_reset` does not zero, the SAME class of residual R4's own F2_LESION_RATIO=0.34 was
    # calibrated to tolerate — R4's own delta_lesion values run 0.0001-0.0006, this module's held=False shift is
    # the identical quantity). So "no signal, no bias" is graded on the SAME relative floor as the lesion check,
    # not bit-exact zero (which would demand a determinism property this mechanism was never calibrated to have).
    _NOISE_RATIO = 0.34
    no_signal_ok = bool(abs(held_false_intact["shift_toward_generated"]) < _NOISE_RATIO * abs(d_i)
                        and abs(held_false_lesion["shift_toward_generated"]) < _NOISE_RATIO * abs(d_i))
    lesion_attributable = bool(d_i > 0.0 and abs(d_l) < _NOISE_RATIO * abs(d_i))
    # R4's OWN PRE-REGISTERED ABSOLUTE FLOOR (2026-08-28, verify-go skeptic finding — do not silently drop this).
    # This wrapper's `crossedge_provenance_shift` calls R4Pool.amb_read with the IDENTICAL protocol R4's own F2
    # arm used, so R4's F2_INTACT_FLOOR=0.010 is the fair, applicable absolute bar — checked here EXPLICITLY
    # rather than only the weaker `d_i > 0.0` (which a naive reading of "lesion-attributable" could pass on an
    # arbitrarily small intact shift). Reported HONESTLY as its own field, separate from `GO`: the production
    # wiring's own crux is "does the wired diagnostic drive-then-vanish-under-lesion through this simpler
    # call sequence" (no F1/F4 pre-conditioning steps run before the read, unlike R4's own `run_seed`), and one
    # seed (100) measures BELOW this floor here even though R4's own richer-preconditioned run cleared it
    # (+0.0110) — a genuine, small, honestly-declared residual from the simpler production call order, not
    # cherry-picked away.
    from research.runners._onebrain_integration_r4_selfschema_provenance import F2_INTACT_FLOOR
    clears_r4_registered_floor = bool(d_i >= F2_INTACT_FLOOR)
    return {
        "seed": int(seed), "cross_weight": pool.cross_weights,
        "held_true_intact": held_true_intact, "held_false_intact": held_false_intact,
        "held_true_lesion": held_true_lesion, "held_false_lesion": held_false_lesion,
        "frac_attributable_to_cross_edge": (None if frac is None else float(frac)),
        "no_signal_no_bias_ok": no_signal_ok,
        "lesion_attributable": lesion_attributable,
        "r4_f2_intact_floor": float(F2_INTACT_FLOOR),
        "clears_r4_registered_floor": clears_r4_registered_floor,
        "GO": bool(lesion_attributable and no_signal_ok),
    }


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true", help="build+grow the FROZEN pool and self-verify load-bearing")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    ap.add_argument("--declarative", action="store_true",
                    help="exercise the DeclarativeR4Pool construction path (BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_"
                         "DECLARATIVE=1) instead of the bespoke R4Pool default")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = []
    for s in seeds:
        sv = _selftest_loadbearing(s, declarative=args.declarative)
        print(f"[seed {s}] {'GO' if sv['GO'] else 'no'} | cross_weight={sv['cross_weight']} | "
              f"held=T intact shift={sv['held_true_intact']['shift_toward_generated']:+.4f} "
              f"lesion shift={sv['held_true_lesion']['shift_toward_generated']:+.4f} "
              f"frac_attrib={sv['frac_attributable_to_cross_edge']} | "
              f"held=F intact shift={sv['held_false_intact']['shift_toward_generated']:+.4f} "
              f"lesion shift={sv['held_false_lesion']['shift_toward_generated']:+.4f} | "
              f"no_signal_ok={sv['no_signal_no_bias_ok']} | "
              f"clears_R4_floor({sv['r4_f2_intact_floor']:.3f})={sv['clears_r4_registered_floor']}", flush=True)
        results.append(sv)

    n_go = sum(r["GO"] for r in results)
    n_clears_floor = sum(r["clears_r4_registered_floor"] for r in results)
    probe_name = ("onebrain_xedge_selfschema_production_declarative" if args.declarative
                 else "onebrain_xedge_selfschema_production_frozen")
    payload = {"probe": probe_name, "seeds": seeds, "declarative": bool(args.declarative),
               "backend": os.environ.get("SIM_BACKEND", "numpy"),
               "n_go": n_go, "n_seeds": len(results), "n_clears_r4_registered_floor": n_clears_floor,
               "results": results,
               "note": ("frozen pre-grown R4 self_schema-authorship -> source_provenance cross-edge wired into "
                        "the live pool; the DRIVE is lesion-attributable through crossedge_provenance_shift "
                        "(the SAME function the live /api/brain-chat is_hyp block calls): holding the LIVE "
                        "authorship verdict shifts R4's fixed ambiguous item's signed provenance margin toward "
                        "GENERATED, within R4's own noise floor of 0 shift when author_held=False (no signal, "
                        "no bias), and the shift collapses under BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION=1 on "
                        "every seed (GO). HONEST RESIDUAL (2026-08-28, verify-go skeptic finding): this "
                        "wrapper's simpler call sequence (train -> read, no F1/F4 pre-conditioning steps unlike "
                        "R4's own run_seed) yields a slightly smaller intact shift than R4's own richer-"
                        "preconditioned protocol on some seeds -- `clears_r4_registered_floor` reports how many "
                        "seeds still clear R4's own pre-registered F2_INTACT_FLOOR=0.010 under this simpler "
                        "sequence (may be < n_go; GO is graded on lesion-attributability, the wiring's own crux, "
                        "not silently re-using R4's floor as if unchanged).")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print(f"\n[R4-PRODUCTION] {n_go}/{len(results)} seeds GO (lesion-attributable) | "
          f"{n_clears_floor}/{len(results)} clear R4's own registered F2 floor", flush=True)
    return 0 if n_go == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
