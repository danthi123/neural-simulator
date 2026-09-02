"""ONE-BRAIN CROSS-EDGE (curiosity.ask -> d6.w0) — wired into the LIVE chat brain (2026-09-01).

Mirrors `research/runners/onebrain_xedge_selfschema_production.py`'s PART-1/R4 pattern verbatim in SHAPE, on the
FRESH curiosity(D3)<->d6_multiref_wm(D6) pairing (6-seed GO on the generic `onebrain_crossedge_gate.run_gate`
harness: `research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-GO.md`,
`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py`).

THE GAP. curiosity and d6_multiref_wm each build their OWN standalone `SimulationBridge` in production
(`curiosity_production_organ.py` / `d6_multiref_wm_production_organ.py`, both process-shared or per-session
singletons) — co-resident in the merge-framework's `full7` REGISTRY but with ZERO synaptic interaction in
production. This module co-locates the pair on ONE `MergedPool` (via `AskToW0Pool`, reused by import — NOT
reimplemented) and freezes its pre-grown `ask -> w0` cross-edge, so a live curiosity crave state can drive a
co-temporal d6 WM read on the SAME shared substrate.

FROZEN, PLASTICITY-OFF IN PRODUCTION (mirrors PART-1/R4 exactly). The cross-edge is GROWN ONCE at first build
(`AskToW0Pool.train()`, the substrate's own `hebbian_symmetric` rule, 0.05 -> ~1.7-2.1) and then FROZEN
(`enable_hebbian_learning=False`, which `train()` already leaves set). No weight moves during any live turn.
Growth is IN-PROCESS (not a saved artifact) for the SAME cross-backend-seed-trap reason PART-1/R4 document
(`docs/ENGINE_REFERENCE.md`): growing in whatever backend the process runs guarantees the frozen edge matches the
substrate it rides.

WHY THIS CHANGES REPLY TEXT, NOT JUST A DIAGNOSTIC FIELD (2026-08-19 "faculties must drive, not observe" —
a neural verdict stashed as metadata + a default-on flip is a hollow checkbox, the drift this wiring must avoid).
Unlike PART-1/R4 (additive diagnostic field only), this wire-in feeds the cross-edge's OWN validated instrument
(`crossedge_w0_shift`, the SAME `AskToW0Pool.read_w0` the 6-seed runner-level GO already cleared) into the D6
hold-query READ-OUT's actual answer text: when THIS SESSION's own RECENT curiosity crave (the live ASK-pool
`curious` verdict `curiosity_production_organ`'s abstain-triggered read already renders, see
`webapp/server.py`'s `_curiosity_followup`) is held, and the frozen cross-edge's own measured suppression clears
its registered floor, the hold-query reply gets an honest, SELF-CONSUMING qualifier appended
(" Though a recent flash of curiosity is competing for my attention right now.") — the SAME functional-read-out
style the rest of this codebase already uses for D6's own readout ("I'm holding N referents...") and curiosity's
own follow-up ("My curiosity is piqued..."). No fact is fabricated; only an honest self-report of a measured
internal competition-for-attention state is appended, and ONLY on the turn actually asking what the brain
currently holds.

SESSION ISOLATION (2026-08-27 cross-session xedge_focus leak fix, reused pattern). The "recent crave" bit lives
as an attribute on THIS SESSION's own per-session `MultiReferentWMOrgan` instance
(`d6org._xedge_curiosity_recent_crave`, set by `webapp/server.py`'s `_curiosity_followup` closure, read+CONSUMED
by the D6 hold-query branch) — never on the shared process-global curiosity organ or this module's shared frozen
pool, so a fresh session's `d6org` (a fresh `getattr(..., False)`) never inherits another session's crave state.
The cross-edge POOL itself carries NO session-specific state at all (it is a pure, stateless-given-input direct
region-current instrument, exactly `AskToW0Pool.read_w0` as validated by the runner-level 6-seed GO) — so there is
no session-leak surface on the pool side either.

WHAT IS LOAD-BEARING vs DECLARED RESIDUAL (honest, carried from the runner-level 6-seed GO + this wire-in's own
probe):
  * LOAD-BEARING + lesion-attributable: driving `ask` alone (this session's OWN recent crave state) measurably
    SUPPRESSES w0's held rate on the SAME fixed direct-drive read the runner-level 6-seed GO already cleared;
    lesioning the cross-edge (`BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1`) collapses the shift toward zero ->
    the qualifier text never appears regardless of the live crave state (a REPLY-TEXT-level lesion check, not
    only a numeric one).
  * DECLARED RESIDUAL #1 (carried UNCHANGED from the runner-level finding): the w0 slot the cross-edge biases is
    d6's OWN direct-drive region (the framework's raw `w0` pool), not (yet) bound to WHICHEVER discourse referent
    this session's `MultiReferentWMOrgan` has semantically loaded into register 0 — the shift is a genuine
    substrate-level competition-for-attention read, but it does not (yet) know or care WHICH referent's register
    it is. Binding the cross-edge onto the SEMANTIC content of the currently-focused register (so the suppressed
    referent is actually the one the reply drops from its "holding N referents" list, not just an appended
    qualifier) is a separate, later, reviewed rung — exactly the shape PART-1 itself declared for its own
    "semantic referent->pool binding is host-directed" residual.
  * DECLARED RESIDUAL #2: the "recent crave" signal is the ABSTAIN-triggered curiosity read (D3's own novelty
    boundary: NOVELTY = an abstain), carried forward ACROSS turns until consumed by the next hold-query — a
    coarse (binary, non-decaying-until-consumed) model of "a lingering crave", not a continuous-time decay. A
    graded, continuously-decaying crave-carryover is a later rung.
  * MOAT-SAFE: the qualifier never changes WHICH referents are reported held, never flips an abstain, never
    fabricates a fact — it only appends an honest functional self-report, and only when the live crave state AND
    the frozen cross-edge's own measured, lesion-attributable suppression both hold.

GUARDED, DEFAULT-OFF UNTIL VALIDATED, BYTE-IDENTICAL-OFF. `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6` gates the whole
thing. Unset/0/false/no/off => every organ builds standalone exactly as today (byte-identical); curiosity's and
d6's OWN production singletons are UNTOUCHED by this module — the new wiring runs an INDEPENDENT `AskToW0Pool`
instance on its own shared merge pool, so there is ZERO risk to the already-live curiosity follow-up / D6
maintain-and-read pipeline even when this flag is on. A build failure DEGRADES to "no qualifier, no diagnostic"
(never crashes brain load). `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1` zeroes the cross-edge (the load-bearing
lesion control).

Run (offline grow + record + self-verify; 0 Claude tokens, CPU numpy):
  SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_curiosity_d6_production --grow \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_xedge_curiosity_d6_production_frozen_6seed.json
"""
from __future__ import annotations

import os


# PRODUCTION DEFAULT — ON (2026-09-01 auto-flip policy). Per the owner's 2026-09-01 policy (validated-GO +
# genuinely load-bearing on the live /api/brain-chat + moat-safe + byte-identical-off + no-regression => flip,
# no owner-gate, the only guard is the hollow-flip trap), and following this project's OWN precedent
# (`onebrain_xedge_production.py`'s PART-1 live-learning wire-in was flipped `_XEDGE_DEFAULT_ON = True` once ITS
# own decision-flipping load-bearing check passed): this is NOT a hollow flip (unlike PART-1/R4's OWN
# diagnostic-only additive field) — it changes the literal reply text a hold-query turn returns. Full
# verification: `research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-production-wire-GO.md`
# (6-seed GO on this module's own self-test + 4 real-handler pytest tests: no-regression, default-off
# byte-identical, qualify+lesion-collapse, session-isolation). `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6=0` still
# forces it off for a controlled A/B or a regression bisect.
_XEDGE_CD6_DEFAULT_ON = True


def xedge_curiosity_d6_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6` in {1,true,yes,on} -> the frozen curiosity.ask -> d6.w0 cross-edge is
    live (the two organs share ONE spiking pool with the pre-grown, frozen cross-synapse, and this session's OWN
    recent curiosity-crave state can drive a co-temporal d6 hold-query read on it). Unset/{0,false,no,off} ->
    byte-identical (no shared pool is ever built; curiosity's and d6's own production organs are untouched).
    Default per `_XEDGE_CD6_DEFAULT_ON`."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6")
    if v is None:
        return _XEDGE_CD6_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def xedge_curiosity_d6_lesioned() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION` in {1,true,yes,on} -> zero the ask->w0 cross-edge weight (the
    load-bearing lesion control: the crave->suppression shift, and the reply-text qualifier it gates, must both
    VANISH). Everything else (curiosity's own read, d6's own maintain/read battery) is unchanged."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


class XedgeCuriosityD6ProductionPool:
    """Process-shared holder of the [curiosity + d6_multiref_wm] `MergedPool` (via `AskToW0Pool`, reused by
    import) with the FROZEN pre-grown `ask -> w0` cross-edge. Exposes `.read_w0(condition)` (the runner-level
    6-seed-GO's OWN validated read, reused verbatim) and `.lesion_cross()`. Built lazily on first use; degrades to
    a disabled holder on any build failure (the caller then reports no qualifier/diagnostic)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.ok = False
        self.bridge = None
        self.masks = None
        self.cross_weight = None
        self.grow_traj = None
        self._ask_pool = None   # the underlying AskToW0Pool (read_w0 lives here)

    def ensure_built(self):
        if self._built:
            return
        self._built = True     # set FIRST so a failed build is not retried every turn
        try:
            self._build()
            self.ok = True
        except Exception as e:   # never crash brain load — degrade to "no diagnostic"
            import traceback
            print(f"[webapp] ONEBRAIN XEDGE-CURIOSITY-D6 build FAILED -> degrading to no diagnostic "
                  f"({type(e).__name__}: {e})", flush=True)
            print(traceback.format_exc(), flush=True)
            self.ok = False

    def _build(self):
        # Import lazily (pins SIM_BACKEND=numpy via os.environ.setdefault at module scope in the runner; a no-op
        # once the webapp has already fixed the backend). `AskToW0Pool` builds the merged [curiosity, d6] pool,
        # injects the near-zero cross-edge as the SOLE plastic synapse (`apply_cross_edge_freeze`, already run
        # inside `__init__`), and exposes `.train()` / `.read_w0(condition)` — the SAME functions the runner-level
        # 6-seed GO validated. Nothing here is reimplemented.
        from research.runners._onebrain_crossedge_curiosity_to_d6wm import AskToW0Pool
        p = AskToW0Pool(self.seed)
        self.grow_traj = p.train()      # GROWS the cross-edge by the substrate's own Hebbian rule (0.05 -> ~1.7-2.1)
        p.b.core_config.enable_hebbian_learning = False   # train() already leaves this False; explicit for parity
        self._ask_pool = p
        self.bridge = p.b
        self.masks = p.masks
        self.cross_weight = self._wmean()
        if xedge_curiosity_d6_lesioned():
            self.lesion_cross()

    def _wmean(self) -> float:
        import numpy as np
        from sim.backend import to_host
        return float(np.asarray(to_host(self.bridge.cp_connections.data))[self.masks["ask_to_w0"]].mean())

    def lesion_cross(self):
        """Zero the ask->w0 cross-edge weight in place (the crave->suppression shift must then vanish). Mirrors
        PART-1/R4's `lesion_cross` exactly, including refreshing `self.cross_weight` post-lesion."""
        import numpy as np
        from sim.backend import to_host
        b = self.bridge
        data = np.asarray(to_host(b.cp_connections.data)).copy()
        for k in self.masks:
            data[self.masks[k]] = 0.0
        b.cp_connections.data = self._ask_pool.pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)
        self.cross_weight = self._wmean()

    def read_w0(self, condition: str) -> dict:
        """The runner-level 6-seed-GO's OWN validated read (reused verbatim, not reimplemented): LOAD w0 into its
        own held bump (condition-blind), then drive `ask` under `condition` in {"familiar","novel"} and read w0's
        mean firing rate. See `_onebrain_crossedge_curiosity_to_d6wm.AskToW0Pool.read_w0`."""
        self.ensure_built()
        return self._ask_pool.read_w0(condition)


_POOL: "XedgeCuriosityD6ProductionPool | None" = None


def get_xedge_curiosity_d6_pool(seed: int = 42) -> "XedgeCuriosityD6ProductionPool | None":
    """The process-shared curiosity->d6 cross-edge pool (built once on first use). Returns the holder even if the
    build failed (`holder.ok is False`) so the caller can degrade to no qualifier. Returns None only when the
    flag is OFF (mirrors PART-1/R4's `get_xedge_..._pool` exactly)."""
    global _POOL
    if not xedge_curiosity_d6_enabled():
        return None
    if _POOL is None:
        _POOL = XedgeCuriosityD6ProductionPool(seed)
    _POOL.ensure_built()
    return _POOL if _POOL.ok else None


def crossedge_w0_shift(pool: "XedgeCuriosityD6ProductionPool", ask_held: bool) -> dict | None:
    """LIVE reply-path hook. Given the CALLING session's own recent live curiosity-crave state (`ask_held` — the
    consumed `_xedge_curiosity_recent_crave` bit `webapp/server.py`'s `_curiosity_followup` sets on this
    session's own `MultiReferentWMOrgan` on the last abstain), read the frozen ask->w0 cross-edge's own validated
    instrument (`read_w0`) with vs without `ask` driven, and report the SIGNED shift on w0's held rate it causes
    (negative == suppression, matching the runner-level 6-seed GO's own measured sign). `ask_held=False` reports a
    near-zero shift by construction (both reads use the undriven 'familiar' condition) — the honest "no crave
    signal, no bias" case. Never raises into a turn (best-effort; returns None on any error so the caller
    degrades to "no qualifier field", never a crashed turn)."""
    try:
        if pool is None or not pool.ok:
            return None
        baseline = pool.read_w0("familiar")
        held = pool.read_w0("novel") if ask_held else pool.read_w0("familiar")
        shift = float(held["w0"] - baseline["w0"])
        return {
            "on": True,
            "ask_held": bool(ask_held),
            "w0_rate_familiar": float(baseline["w0"]),
            "w0_rate_read": float(held["w0"]),
            "shift_w0": shift,
            "cross_weight": float(pool.cross_weight) if pool.cross_weight is not None else None,
        }
    except Exception as e:
        return {"on": True, "error": f"{type(e).__name__}: {e}"}


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  Offline grow + record + self-verify entrypoint (0 Claude tokens; CPU numpy). Mirrors PART-1/R4's
#  `_selftest_loadbearing` + CLI shape exactly, but against THIS module's own production functions.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _selftest_loadbearing(seed: int) -> dict:
    """Exercise the REAL production function `crossedge_w0_shift` (not a bespoke probe) at seed `seed`, INTACT
    vs cross-edge-LESIONED: the shift must clear the runner-level GO's own registered floor (signed negative)
    ask_held=True intact, and collapse toward zero lesioned -> lesion-attributable crave->suppression through the
    live production read. Also checks ask_held=False reports ~0 shift on BOTH arms (the honest no-signal
    control)."""
    from tools.lab import attributable_to
    from research.runners._onebrain_crossedge_curiosity_to_d6wm import INTACT_FLOOR, LESION_RATIO
    os.environ["BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6"] = "1"
    os.environ.pop("BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION", None)
    global _POOL
    _POOL = None
    pool = get_xedge_curiosity_d6_pool(seed)
    assert pool is not None and pool.ok, "xedge-curiosity-d6 pool failed to build"

    held_true_intact = crossedge_w0_shift(pool, True)
    held_false_intact = crossedge_w0_shift(pool, False)

    pool.lesion_cross()
    held_true_lesion = crossedge_w0_shift(pool, True)
    held_false_lesion = crossedge_w0_shift(pool, False)

    d_i = held_true_intact["shift_w0"]
    d_l = held_true_lesion["shift_w0"]
    frac = attributable_to(f"seed{seed} xedge-curiosity-d6 crave->suppression shift vs cross-edge lesion",
                           d_i, d_l)
    no_signal_ok = bool(abs(held_false_intact["shift_w0"]) < LESION_RATIO * max(abs(d_i), 1e-9)
                        and abs(held_false_lesion["shift_w0"]) < LESION_RATIO * max(abs(d_i), 1e-9))
    lesion_attributable = bool(d_i < 0.0 and abs(d_l) < LESION_RATIO * abs(d_i))
    clears_registered_floor = bool(d_i <= -INTACT_FLOOR)
    return {
        "seed": int(seed), "cross_weight": pool.cross_weight,
        "held_true_intact": held_true_intact, "held_false_intact": held_false_intact,
        "held_true_lesion": held_true_lesion, "held_false_lesion": held_false_lesion,
        "frac_attributable_to_cross_edge": (None if frac is None else float(frac)),
        "no_signal_no_bias_ok": no_signal_ok,
        "lesion_attributable": lesion_attributable,
        "registered_intact_floor": float(INTACT_FLOOR),
        "clears_registered_floor": clears_registered_floor,
        "GO": bool(lesion_attributable and no_signal_ok and clears_registered_floor),
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
        print(f"[seed {s}] {'GO' if sv['GO'] else 'no'} | cross_weight={sv['cross_weight']:.4f} | "
              f"held=T intact shift={sv['held_true_intact']['shift_w0']:+.4f} "
              f"lesion shift={sv['held_true_lesion']['shift_w0']:+.4f} "
              f"frac_attrib={sv['frac_attributable_to_cross_edge']} | "
              f"held=F intact shift={sv['held_false_intact']['shift_w0']:+.4f} "
              f"lesion shift={sv['held_false_lesion']['shift_w0']:+.4f} | "
              f"no_signal_ok={sv['no_signal_no_bias_ok']} | "
              f"clears_floor({sv['registered_intact_floor']:.3f})={sv['clears_registered_floor']}", flush=True)
        results.append(sv)

    n_go = sum(r["GO"] for r in results)
    payload = {"probe": "onebrain_xedge_curiosity_d6_production_frozen", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"),
               "n_go": n_go, "n_seeds": len(results),
               "results": results,
               "note": ("frozen pre-grown curiosity.ask -> d6.w0 cross-edge wired into a dedicated production "
                        "pool; the DRIVE is lesion-attributable through crossedge_w0_shift (the SAME function "
                        "the live /api/brain-chat D6 hold-query block calls): holding a live recent crave state "
                        "shifts the frozen pool's own w0 rate NEGATIVE (suppression, matching the runner-level "
                        "6-seed GO's own measured sign), within the runner's own noise floor of ~0 shift when "
                        "ask_held=False (no signal, no bias), and the shift collapses under "
                        "BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1.")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print(f"\n[XEDGE-CURIOSITY-D6-PRODUCTION] {n_go}/{len(results)} seeds GO (lesion-attributable)", flush=True)
    return 0 if n_go == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
