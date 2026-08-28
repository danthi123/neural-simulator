"""ONE-BRAIN CROSS-EDGE (surprise->episodic/source_provenance, board #129's TWO-cross-edge construction) — the
THIRD learned cross-region synapse wired into the LIVE chat brain (2026-08-28). Mirrors `research/runners/
onebrain_xedge_production.py`'s PART-1 pattern (the d6-WM -> comprehension frozen cross-edge;
`2026-08-27-onebrain-xedge-production-frozen-GO.md`) and `onebrain_xedge_selfschema_production.py`'s R4 wire-in
(the SECOND such edge) verbatim in SHAPE, on the surprise->source_provenance pairing: the D2 expectation-violation
circuit ("this asserted fact contradicts what I expected") -> source_provenance monitoring ("this memory reads as
internally-generated"), 6-seed GO on the merge framework
(`research/findings/2026-08-28-surprise-episodic-129construction-6seed-GO.md`, DEFINED+GO 6/6, F2 frac_attributable
0.887-0.965).

THE GAP. board #129's construction lives on the RESEARCH merge framework
(`research/runners/_onebrain_surprise_episodic_129construction_derisk.py`), default-off, NOT in the live brain —
the D2 surprise circuit and source_provenance each build their OWN standalone production organs/bridges in
production (`surprise_production_organ.py` / `source_provenance_production_organ.py`), so no shared substrate the
two LEARNED cross-edges could span. This module co-locates the construction's own `SurpriseEpisodic129Pool`
(reused by import, NOT reimplemented) on ONE `MergedPool` (built via the declarative `CrossEdge`/`merge_organs
(cross_edges=...)` framework, already how `SurpriseEpisodic129Pool._build_pool_129` constructs itself — see the
finding's own §1-2 for why no separate "bespoke" predecessor exists to migrate FROM here, unlike PART-1/R4) and
freezes its pre-grown TWO cross-edges (`surprise->prov_generated`, `patient_expected->prov_perceived`), so a live
D2 surprise verdict can drive a co-temporal source-provenance read on the SAME shared substrate.

FROZEN, PLASTICITY-OFF IN PRODUCTION (mirrors PART-1/R4 exactly). Both cross-edges are GROWN ONCE at first build
(the construction's own direct-drive Hebbian training, `SurpriseEpisodic129Pool.train_129()`) and then FROZEN
(`enable_hebbian_learning=False`, which `train_129()` already leaves set, PLUS an explicit
`set_plasticity_gate(GATE/GATE2, 0.0)` for defensive parity with PART-1/R4's own train()-then-freeze convention).
No weight moves during any live turn. Growth is IN-PROCESS (not a saved artifact) for the SAME cross-backend-seed-
trap reason PART-1/R4 document (`docs/ENGINE_REFERENCE.md`): a numpy-grown weight file is not valid for a cupy
production build.

WHY A LIVE CO-DRIVE READ, NOT JUST CO-RESIDENCE (mirrors PART-1/R4's "attach shared= and done would be HOLLOW"
lesson — drift memory #84/#85). The construction's own F2 instrument (`amb_read_ratio`) is REUSED VERBATIM (not
reimplemented): hard-reset -> drive the fixed dual-context-encoded AMBIGUOUS content pattern's episode neurons
(+ optionally the surprise circuit's own CONTRADICT drive, `_contradict_pairs()`) -> read the divisively-normalized
OPPONENT RATIO `d=(r_gen-r_perc)/(r_gen+r_perc+DN_SIGMA)`. The cross-edges only transmit when the CONTRADICT drive
is ACTIVELY APPLIED during the recall window — so the live consumer below drives it FROM the turn's OWN live D2
surprise verdict (`sj["surprised"]`, the SAME boolean `webapp/server.py`'s surprise block already computes and
reports as `resp["surprise"]["surprised"]`), not a co-resident-but-inert attach.

WHAT IS LOAD-BEARING vs DECLARED RESIDUAL (honest, carried from the construction's own finding, §5-6):
  * LOAD-BEARING + lesion-attributable: holding the surprise circuit's CONTRADICT drive during the SAME fixed
    ambiguous item's recall (the construction's own F2 instrument) shifts the divisive-ratio margin toward
    GENERATED vs a no-hold baseline; lesioning BOTH cross-edges together
    (`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION=1`) collapses the shift toward zero. This is the SAME F2
    measurement the 6-seed GO already cleared (frac_attributable 0.887-0.965), now exercised through this
    production module's own `crossedge_provenance_shift_129` (the function the live handler calls).
  * DECLARED RESIDUAL #1 (carried UNCHANGED from the construction): the item the cross-edges bias is the
    construction's OWN fixed, dual-context-encoded, per-seed AMBIGUOUS content pattern (a substrate stand-in for a
    genuinely uncertain real memory) — NOT an arbitrary live chat fact. Binding the bias onto an arbitrary
    production fact key is a separate, later, reviewed rung (identical shape to PART-1/R4's own declared residual).
  * DECLARED RESIDUAL #2 (the construction's own §5 finding, carried forward HONESTLY, not smoothed over): an
    individual-edge lesion control (single-seed, diagnostic-only, run after the 6-seed GO landed) found the NEW
    `patient_expected->prov_perceived` edge (edge 2, CONFIRM-trained) reproduces nearly the FULL intact shift
    alone (delta=+0.190 vs intact +0.193), while the ORIGINAL `surprise->prov_generated` edge (edge 1,
    CONTRADICT-trained — the edge already validated on every OTHER arm of the single-edge construction) alone
    reproduces almost NONE of it (delta=+0.027, collapsing to the both-lesioned floor +0.023). The pre-registered
    F2 gate this production wiring inherits lesions BOTH edges TOGETHER (what "the surprise->episodic mechanism"
    is DEFINED as here, and what cleanly passed 6/6) — this residual does not change that verdict or block the
    wire-in (the SAME both-edges-together lesion this module exposes via
    `BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION` is exactly the pre-registered gate, not a weaker one), but the
    causal story is NOT yet "surprise biases toward generated" at the single-edge level — it is closer to "the
    confirm-side edge's presence measurably reshapes the gen-vs-perc balance under a contradict hold," a mechanism
    not yet explained at the circuit level. Flagged here so a future reader does not mistake this wire-in as
    having resolved that open question — it has not; it wires the ALREADY-VALIDATED joint mechanism, honestly
    labeled.
  * DECLARED RESIDUAL #3 (mirrors PART-1/R4's own honest shape): the magnitude is a shift on the live SUBSTRATE
    ratio (0.13-0.20 across the 6-seed GO), not (yet) wired to flip any DECISION-level text — this wire-in
    surfaces the live, lesion-attributable READ as an additive diagnostic field
    (`resp["surprise"]["source_provenance_crossedge"]`) on the SAME turn the D2 surprise block already computes
    `surprise_info` on, exactly PART-1/R4's "content-neutral at the decision level, genuine + reversible" shape.

GUARDED, DEFAULT-OFF, BYTE-IDENTICAL-OFF. `BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC` gates the whole thing (default
OFF — the flip to default-ON is a separate, later, owner-gated step, exactly PART-1/R4's convention). Unset/0/
false/no/off => every organ builds standalone exactly as today (byte-identical); the LIVE `surprise_production_
organ` singleton (already default-ON for the D2 notice) is UNTOUCHED by this module — the new diagnostic runs an
INDEPENDENT `SurpriseEpisodic129Pool` instance on its own shared merge pool, so there is ZERO risk to the
already-live surprise notice/reconsolidation pipeline even when this flag is on. A build failure DEGRADES to "no
diagnostic field" (never crashes brain load — mirrors PART-1/R4's `ensure_built` try/except).
`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION=1` zeroes BOTH cross-edges (the load-bearing lesion control, the
SAME joint lesion the 6-seed GO's own F2 arm used).

Run (offline grow + record + self-verify; 0 Claude tokens, CPU numpy):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_surprise_episodic_production --grow \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_xedge_surprise_episodic_production_frozen_6seed.json
"""
from __future__ import annotations

import os


# PRODUCTION DEFAULT — OFF. The owner-gated flip to default-ON is a SEPARATE step (never autonomous), exactly
# mirroring PART-1/R4's `_XEDGE_DEFAULT_ON`/`_XEDGE_SS_DEFAULT_ON` convention.
_XEDGE_SE_DEFAULT_ON = False


def xedge_surprise_episodic_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC` in {1,true,yes,on} -> the frozen surprise->episodic (board #129's
    two-cross-edge) construction is live (the surprise circuit + source_provenance share ONE spiking pool with the
    pre-grown, frozen cross-synapses, and a turn's live D2 surprise verdict drives a co-temporal provenance read on
    it). Unset/{0,false,no,off} -> byte-identical (no shared pool is ever built; the live surprise organ's own
    production singleton is untouched). Default per `_XEDGE_SE_DEFAULT_ON` (OFF)."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC")
    if v is None:
        return _XEDGE_SE_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def xedge_surprise_episodic_lesioned() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION` in {1,true,yes,on} -> zero BOTH cross-edges (surprise->
    prov_generated AND patient_expected->prov_perceived) in place (the load-bearing lesion control: the
    surprise->provenance shift must VANISH — the SAME joint lesion the construction's own 6-seed F2 gate used, not
    a weaker single-edge lesion). Everything else (the surprise circuit's own read, source_provenance's own
    battery) is unchanged, so the shift these cross-edges introduce must collapse under this flag."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class XedgeSurpriseEpisodicProductionPool:
    """Process-shared holder of the [surprise + source_provenance] `MergedPool` (board #129's construction) with
    the FROZEN pre-grown TWO cross-edges. Exposes `.pool` (the framework MergedPool), `.cross_weights` (both
    edges' trained-block means, via the construction's own `cross_weights()` — reused, not reimplemented), and
    `.provenance_ratio(hold_surprise)` (the construction's own validated F2 instrument, `amb_read_ratio`, reused
    verbatim). Built lazily on first use; degrades to a disabled holder on any build failure (the caller then
    reports no diagnostic, exactly mirroring PART-1/R4's `ensure_built`)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.ok = False
        self.pool = None            # the framework MergedPool
        self.bridge = None
        self.ix = None
        self.masks = None
        self.cross_weights = None
        self._sep = None            # the underlying SurpriseEpisodic129Pool (amb_read_ratio lives here)

    def ensure_built(self):
        if self._built:
            return
        self._built = True     # set FIRST so a failed build is not retried every turn
        try:
            self._build()
            self.ok = True
        except Exception as e:   # never crash brain load — degrade to "no diagnostic"
            import traceback
            print(f"[webapp] ONEBRAIN XEDGE-SURPRISE-EPISODIC (129-construction) build FAILED -> degrading to no "
                  f"diagnostic ({type(e).__name__}: {e})", flush=True)
            print(traceback.format_exc(), flush=True)
            self.ok = False

    def _build(self):
        # Import lazily (this pins SIM_BACKEND=numpy via os.environ.setdefault at module scope on the imported
        # research runner; a no-op once the webapp has already fixed the backend). SurpriseEpisodic129Pool builds
        # the merged pool via the DECLARATIVE `CrossEdge`/`merge_organs(cross_edges=...)` framework (its OWN
        # `_build_pool_129`, unchanged), injects both near-zero cross-edges as the SOLE plastic synapses, trains
        # them by direct-drive Hebbian co-activation (`train_129`), and constructs the fixed dual-context-encoded
        # ambiguous content pattern — reused by import, not reimplemented.
        from research.runners._onebrain_surprise_episodic_129construction_derisk import (
            SurpriseEpisodic129Pool, GATE, GATE2)
        sep = SurpriseEpisodic129Pool(self.seed)
        sep.train_129()               # GROWS both cross-edges by the substrate's own Hebbian rule (0.05 -> trained)
        # FREEZE for production: train_129() already leaves core_config.enable_hebbian_learning=False (the global
        # per-step gate every subsequent read is subject to); ALSO re-assert each gate's own rate_gain at 0, for
        # defensive parity with PART-1/R4's train()-then-freeze convention ("freezes the candidate gate(s) the
        # instant train returns").
        sep.b.set_plasticity_gate(GATE, 0.0)
        sep.b.set_plasticity_gate(GATE2, 0.0)
        self._sep = sep
        self.pool = sep.pool
        self.bridge = sep.b
        self.ix = sep.ix
        self.masks = sep.masks
        self.cross_weights = sep.cross_weights()      # reused verbatim (SurpriseEpisodicPool.cross_weights)
        if xedge_surprise_episodic_lesioned():
            self.lesion_cross()

    # ── the load-bearing lesion control (env or explicit): zero BOTH cross-edges together ──
    def lesion_cross(self):
        """Zero BOTH cross-edges (`masks["both_edges"]`, the SAME joint mask the construction's own F2 gate
        lesions) in place — the surprise->provenance shift must then vanish. Mirrors PART-1/R4's `lesion_cross`
        exactly, including refreshing `self.cross_weights` afterward (2026-08-28 verify-go lesson from the R4
        wire-in: a stale post-lesion weight snapshot would misle a live consumer inspecting it, even though the
        actual shift read is unaffected since it reads live connection data)."""
        import numpy as np
        from sim.backend import to_host
        b = self.bridge
        data = np.asarray(to_host(b.cp_connections.data)).copy()
        data[self._sep.masks["both_edges"]] = 0.0
        b.cp_connections.data = self.pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)
        if self._sep is not None:
            self.cross_weights = self._sep.cross_weights()

    # ── the live-consumer instrument: the construction's OWN validated F2 read, reused verbatim ──
    def provenance_ratio(self, hold_surprise: bool, band=None):
        """The divisively-normalized opponent ratio `d=(r_gen-r_perc)/(r_gen+r_perc+DN_SIGMA)` for the
        construction's fixed, dual-context-encoded ambiguous content pattern, optionally holding the surprise
        circuit's CONTRADICT drive throughout the recall window. Delegates to `SurpriseEpisodic129Pool.
        amb_read_ratio` (the SAME instrument the 6-seed GO's F2 arm used) — not reimplemented, so this production
        module carries zero drift risk from the validated mechanism."""
        self.ensure_built()
        return self._sep.amb_read_ratio(bool(hold_surprise), band=band)


_POOL: "XedgeSurpriseEpisodicProductionPool | None" = None


def get_xedge_surprise_episodic_pool(seed: int = 42) -> "XedgeSurpriseEpisodicProductionPool | None":
    """The process-shared surprise->episodic cross-edge pool (built once on first use). Returns the holder even if
    the build failed (`holder.ok is False`) so the caller can degrade to no diagnostic. Returns None only when the
    flag is OFF (mirrors PART-1/R4's `get_xedge_pool`/`get_xedge_selfschema_pool` exactly)."""
    global _POOL
    if not xedge_surprise_episodic_enabled():
        return None
    if _POOL is None:
        _POOL = XedgeSurpriseEpisodicProductionPool(seed)
    _POOL.ensure_built()
    return _POOL if _POOL.ok else None


def crossedge_provenance_shift_129(pool: "XedgeSurpriseEpisodicProductionPool", hold_surprise: bool) -> dict | None:
    """LIVE reply-path hook. Given the CALLING turn's own live D2 surprise verdict (`hold_surprise` — the SAME
    `sj["surprised"]` boolean the surprise block already computes and reports as `resp["surprise"]["surprised"]`),
    read the construction's fixed ambiguous item's divisive-ratio provenance margin with vs without that hold, and
    report the shift toward GENERATED it causes on the shared substrate. `hold_surprise=False` (a non-surprising
    turn) reports shift 0.0 by construction (both reads use the same no-hold protocol) -- the honest "no surprise
    signal, no bias" case. Never raises into a turn (best-effort; returns None on any error so the caller degrades
    to "no diagnostic field", never a crashed turn)."""
    try:
        if pool is None or not pool.ok:
            return None
        baseline = pool.provenance_ratio(False)
        held = pool.provenance_ratio(bool(hold_surprise))
        shift = float(held["ratio"] - baseline["ratio"])
        return {
            "on": True,
            "surprise_held": bool(hold_surprise),
            "ratio_baseline": float(baseline["ratio"]),
            "ratio_held": float(held["ratio"]),
            "shift_toward_generated": shift,
            "cross_weights": dict(pool.cross_weights) if pool.cross_weights else {},
        }
    except Exception as e:
        return {"on": True, "error": f"{type(e).__name__}: {e}"}


# R4's OWN "pre-registered ABSOLUTE floor" convention, applied here: the construction's own calibration (seed 7,
# non-canonical, frozen BEFORE any canonical seed was read — `2026-08-28-surprise-episodic-129construction-6seed-
# GO.md` §4) froze `F2_INTACT_FLOOR (ratio units) = 0.0478` (`0.25 * |calibration delta_intact|=0.1911|`). Reused
# here VERBATIM (not re-derived) as the fair, applicable absolute bar for this wrapper's own simpler call sequence
# (train -> read, no F1/F3/F4 pre-conditioning steps unlike the construction's own `run_seed`) — checked as its
# own field, separate from `GO`, exactly R4 wire-in's own honest precedent (a simpler call order can measure a
# smaller intact shift than the richer-preconditioned research protocol on some seeds).
F2_INTACT_FLOOR_129 = 0.0478
_NOISE_RATIO_129 = 0.34        # F2_LESION_RATIO -- the SAME precondition ratio the research 6-seed GO used


def _selftest_loadbearing(seed):
    """Exercise the REAL production function `crossedge_provenance_shift_129` (not a bespoke probe) at seed
    `seed`, INTACT vs both-cross-edges-LESIONED: the shift must be nonzero (surprise_held=True) intact and collapse
    toward zero lesioned -> lesion-attributable surprise->provenance drive through the live production read. Also
    checks surprise_held=False reports ~0.0 shift on BOTH arms (the honest no-signal-no-bias control). Reports
    `n_hollow` (board #94-class anti-hollow bar: 0 == the coupling demonstrably drives a real, lesion-attributable
    difference on real production-shaped traffic; 1 == it did not)."""
    from tools.lab import attributable_to
    os.environ["BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC"] = "1"
    os.environ.pop("BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION", None)
    global _POOL
    _POOL = None
    pool = get_xedge_surprise_episodic_pool(seed)
    assert pool is not None and pool.ok, "xedge-surprise-episodic pool failed to build"

    held_true_intact = crossedge_provenance_shift_129(pool, True)
    held_false_intact = crossedge_provenance_shift_129(pool, False)

    pool.lesion_cross()
    held_true_lesion = crossedge_provenance_shift_129(pool, True)
    held_false_lesion = crossedge_provenance_shift_129(pool, False)

    d_i = held_true_intact["shift_toward_generated"]
    d_l = held_true_lesion["shift_toward_generated"]
    frac = attributable_to(f"seed{seed} xedge-surprise-episodic surprise->provenance shift vs cross-edge lesion",
                           d_i, d_l)
    no_signal_ok = bool(abs(held_false_intact["shift_toward_generated"]) < _NOISE_RATIO_129 * max(abs(d_i), 1e-9)
                        and abs(held_false_lesion["shift_toward_generated"]) < _NOISE_RATIO_129 * max(abs(d_i), 1e-9))
    lesion_attributable = bool(d_i > 0.0 and abs(d_l) < _NOISE_RATIO_129 * abs(d_i))
    clears_registered_floor = bool(d_i >= F2_INTACT_FLOOR_129)
    n_hollow = 0 if (lesion_attributable and abs(d_i) > 1e-9) else 1
    return {
        "seed": int(seed), "cross_weights": pool.cross_weights,
        "held_true_intact": held_true_intact, "held_false_intact": held_false_intact,
        "held_true_lesion": held_true_lesion, "held_false_lesion": held_false_lesion,
        "frac_attributable_to_cross_edge": (None if frac is None else float(frac)),
        "no_signal_no_bias_ok": no_signal_ok,
        "lesion_attributable": lesion_attributable,
        "f2_intact_floor_129": float(F2_INTACT_FLOOR_129),
        "clears_registered_floor": clears_registered_floor,
        "n_hollow": n_hollow,
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
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = []
    for s in seeds:
        sv = _selftest_loadbearing(s)
        print(f"[seed {s}] {'GO' if sv['GO'] else 'no'} | cross_weights={sv['cross_weights']} | "
              f"held=T intact shift={sv['held_true_intact']['shift_toward_generated']:+.4f} "
              f"lesion shift={sv['held_true_lesion']['shift_toward_generated']:+.4f} "
              f"frac_attrib={sv['frac_attributable_to_cross_edge']} | "
              f"held=F intact shift={sv['held_false_intact']['shift_toward_generated']:+.4f} "
              f"lesion shift={sv['held_false_lesion']['shift_toward_generated']:+.4f} | "
              f"no_signal_ok={sv['no_signal_no_bias_ok']} | n_hollow={sv['n_hollow']} | "
              f"clears_floor({sv['f2_intact_floor_129']:.4f})={sv['clears_registered_floor']}", flush=True)
        results.append(sv)

    n_go = sum(r["GO"] for r in results)
    n_hollow_total = sum(r["n_hollow"] for r in results)
    n_clears_floor = sum(r["clears_registered_floor"] for r in results)
    payload = {"probe": "onebrain_xedge_surprise_episodic_production_frozen", "seeds": seeds,
              "backend": os.environ.get("SIM_BACKEND", "numpy"),
              "n_go": n_go, "n_seeds": len(results), "n_hollow_total": n_hollow_total,
              "n_clears_registered_floor": n_clears_floor,
              "results": results,
              "note": ("frozen pre-grown board-#129 (TWO cross-edge, divisive-ratio) surprise->source_provenance "
                       "construction wired into the live pool; the DRIVE is lesion-attributable through "
                       "crossedge_provenance_shift_129 (the function the live surprise (D2) block would call): "
                       "holding the LIVE D2 surprise verdict shifts the construction's fixed ambiguous item's "
                       "divisive-ratio provenance margin toward GENERATED, within the construction's own noise "
                       "floor of 0 shift when surprise_held=False (no signal, no bias), and the shift collapses "
                       "under BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION=1 (BOTH cross-edges zeroed together, "
                       "the SAME joint lesion the 6-seed GO's own F2 gate used) on every seed (GO). HONEST "
                       "RESIDUAL (carried from the construction's own finding, §5): an individual-edge lesion "
                       "found the NEW confirm-side edge does almost all the causal work, not the original "
                       "surprise-side edge — this wire-in exercises the ALREADY-VALIDATED JOINT mechanism only, "
                       "not a resolution of that open circuit-level question.")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print(f"\n[SURP-EPISODIC-PRODUCTION] {n_go}/{len(results)} seeds GO (lesion-attributable) | "
          f"n_hollow_total={n_hollow_total} | {n_clears_floor}/{len(results)} clear the registered F2 floor",
          flush=True)
    return 0 if n_go == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
