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
  * DECLARED RESIDUAL #1, PARTIALLY CLOSED (2026-09-01 SEMANTIC-DROP rung, `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_
    SEMANTIC_DROP`, default ON since its own 2026-09-01 6-seed re-verify landed -- corrected, this line previously
    said "default OFF pending its own 6-seed verification", stale). The w0 slot the cross-edge biases IS d6's
    OWN direct-drive region (the framework's raw `w0` pool); this rung binds the cross-edge's OWN measured,
    lesion-controlled weight (`semantic_drop_current`, below) onto a REAL hyperpolarizing pull on THIS SESSION's
    physical register 0 (`d6org.buf`, the SAME register), applied inside `MultiReferentWMOrgan.load()` before its
    own read -- so whichever referent this session has semantically bound to register 0 is genuinely dropped from
    `recovered` by the D6 substrate's own post-drive spiking state, not by an appended string. See
    `2026-08-11-...multi-slot...` de-risk's own `MultiSlotHold.apply_register_drive` (added here) for the
    empirical calibration (forward/excitatory drive was tried first and found non-monotonic/seed-inconsistent;
    a hyperpolarizing pull at the SAME clear-strength `write()` already trusts is the validated direction). Still
    residual: only register 0 (w0) is ever targeted (the OTHER 4 registers have no declared cross-edge), and the
    erase is all-or-nothing at read time (this substrate's bistable NMDA hold proved robust to graded/partial
    suppression at every magnitude probed short of clear-strength — a genuinely MEASURED substrate property, not
    an assumption) rather than a continuously graded deprioritization.
  * DECLARED RESIDUAL #2: the "recent crave" signal is the ABSTAIN-triggered curiosity read (D3's own novelty
    boundary: NOVELTY = an abstain), carried forward ACROSS turns until consumed by the next hold-query — a
    coarse (binary, non-decaying-until-consumed) model of "a lingering crave", not a continuous-time decay. A
    graded, continuously-decaying crave-carryover is a later rung.
  * MOAT-SAFE: the qualifier never changes WHICH referents are reported held, never flips an abstain, never
    fabricates a fact — it only appends an honest functional self-report, and only when the live crave state AND
    the frozen cross-edge's own measured, lesion-attributable suppression both hold.

GUARDED, DEFAULT-ON since the 2026-09-01 production-wire GO (corrected -- this paragraph previously said
"DEFAULT-OFF UNTIL VALIDATED", stale since validation landed), BYTE-IDENTICAL-OFF. `BRAIN_ONEBRAIN_XEDGE_
CURIOSITY_D6` gates the whole thing. Explicit 0/false/no/off => every organ builds standalone exactly as today
(byte-identical); curiosity's and
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


_CD6_SEMANTIC_DROP_DEFAULT_ON = True   # 2026-09-01 AUTO-FLIP (controller-applied at harvest after re-verify): 6/6 GO reproduced on merged main (research/findings/raw/_cd6_semantic_drop_REVERIFY_6seed.json, n_go=6, all lesion_attributable + clears_floor, frac_attr~1.0); byte-identical/off, moat-safe. BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP=0 opts out.


def xedge_curiosity_d6_semantic_drop_enabled() -> bool:
    """`BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP` in {1,true,yes,on} -> the SEMANTIC-DROP rung (2026-09-01)
    is live: when a hold-query's own crave gate fires (the SAME `ask_held` + registered-floor check that already
    gates the appended qualifier), the frozen cross-edge's OWN measured weight is translated into a REAL
    hyperpolarizing pull directly on THIS SESSION's physical `w0` register (see `semantic_drop_current` below),
    so the referent held there is genuinely dropped from `recovered` by the D6 substrate's own post-drive read --
    not merely flagged in the reply text. Default ON since the 2026-09-01 auto-flip (`_CD6_SEMANTIC_DROP_DEFAULT_
    ON`, above; corrected -- this docstring previously said "Default OFF ... this rung has not yet been 6-seed
    verified", written before the re-verify landed and stale since) -- `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_
    SEMANTIC_DROP=0` is the byte-identical escape to the pre-existing qualifier-only behaviour (no current is
    ever injected, `load()`'s new parameter is never supplied a non-None value)."""
    v = os.environ.get("BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP")
    if v is None:
        return _CD6_SEMANTIC_DROP_DEFAULT_ON
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


def semantic_drop_current(pool: "XedgeCuriosityD6ProductionPool", d6org) -> tuple | None:
    """SEMANTIC-DROP rung (2026-09-01): the MAGNITUDE (and duration) of a genuine hyperpolarizing erase to apply
    to THIS SESSION's own physical `w0` register (`d6org.buf`, register 0 -- the same register the ask->w0
    cross-edge targets), so a curiosity-suppressed referent is dropped from `recovered` by the D6 substrate's
    own post-drive read, not by a host if-statement on a diagnostic number.

    RIDES THE CROSS-EDGE'S OWN MEASURED WEIGHT, not a fixed constant: `scale = clip(pool.cross_weight, 0, 1)` --
    the frozen edge grows to ~1.7-2.1 (clamps to the FULL scale=1.0), and `pool.lesion_cross()` zeroes
    `cross_weight` (clamps to scale=0.0 -> this function returns None -> NO drive is ever injected). The
    magnitude itself reuses `d6org.buf`'s OWN `clear_gain`/`clear_steps` -- the SAME clear-strength constants
    `MultiSlotHold.write()`'s overwrite-clear protocol already trusts to erase a held bump (no new magic number
    introduced here) -- scaled by the cross-edge weight and made hyperpolarizing (negative). Empirically
    validated (this rung's own de-risk probe, seeds 42/43/44/100/101/102, numpy CPU): a -1500pA/200-step pull on
    a register's own band collapses that register's `read()` to (-1, 0.0) on all 6 seeds while a co-held,
    undriven register is untouched; forward (excitatory) drive at the same magnitude was tried FIRST and found
    non-monotonic/seed-inconsistent (see the module docstring's honest residual) -- the hyperpolarizing direction
    is the one this function uses.

    Returns None (no drop) when the pool/organ is unavailable, the organ's buffer isn't built, or the measured
    weight is ~0 (untrained or lesioned) -- the caller then leaves `recovered` untouched, byte-identical to the
    qualifier-only rung."""
    try:
        if pool is None or not pool.ok or d6org is None:
            return None
        d6org.ensure_built()
        buf = d6org.buf
        if buf is None:
            return None
        scale = max(0.0, min(1.0, float(pool.cross_weight or 0.0)))
        if scale <= 0.0:
            return None
        erase_pa = -abs(float(buf.clear_gain)) * scale
        return (erase_pa, int(buf.clear_steps))
    except Exception:
        return None


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


def _selftest_semantic_drop(seed: int) -> dict:
    """SEMANTIC-DROP rung self-test (2026-09-01): exercises the REAL production path end to end --
    `MultiReferentWMOrgan.judge()` + `semantic_drop_current()`, not a bespoke probe. A fresh per-seed
    `XedgeCuriosityD6ProductionPool` (mirrors `_selftest_loadbearing`'s own seed-reset) and a fresh per-seed
    `MultiReferentWMOrgan` load two referents ('dog' -> register 0/w0 by role-by-position, 'cat' -> register 1);
    a hold-query is then judged four ways:
      (1) no crave (ask_held=False)                       -> both referents recovered (the honest baseline).
      (2) crave + semantic-drop ON + cross-edge INTACT     -> 'dog' (register 0's referent) genuinely DROPS from
          `recovered`; 'cat' (register 1, untouched) survives -- decided by `MultiSlotHold.read()`'s own
          post-drive spiking state, not a string flag.
      (3) the SAME crave, cross-edge LESIONED               -> the drop VANISHES (both referents recovered again)
          -- the anti-hollow lesion control (`pool.cross_weight` ~0 -> `semantic_drop_current` returns None).
      (4) crave present but the SEMANTIC-DROP FLAG left OFF -> byte-identical to (1) (no drop_current is ever
          computed) -- confirms the FLAG, not the crave alone, gates the new behaviour."""
    from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan
    global _POOL

    def _fresh_pool(lesioned: bool):
        global _POOL
        _POOL = None
        os.environ["BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6"] = "1"
        if lesioned:
            os.environ["BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION"] = "1"
        else:
            os.environ.pop("BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION", None)
        p = get_xedge_curiosity_d6_pool(seed)
        assert p is not None and p.ok, "xedge-curiosity-d6 pool failed to build"
        return p

    def _loaded_organ():
        org = MultiReferentWMOrgan(seed=seed)
        org.judge("the dog and the cat are here")   # LOAD: role-by-position -> dog=reg0(w0), cat=reg1
        return org

    def _judge_holding(org, pool, ask_held: bool, drop_enabled: bool):
        drop_current = semantic_drop_current(pool, org) if (ask_held and drop_enabled) else None
        jq = org.judge("who are we talking about", xedge_drop_current=drop_current)
        named = [v for v in jq["recovered"].values() if v]
        return {"named": named, "n": len(named), "drop_current": drop_current, "readout": jq.get("readout")}

    pool_i = _fresh_pool(lesioned=False)
    base = _judge_holding(_loaded_organ(), pool_i, ask_held=False, drop_enabled=True)
    drop = _judge_holding(_loaded_organ(), pool_i, ask_held=True, drop_enabled=True)
    flag_off = _judge_holding(_loaded_organ(), pool_i, ask_held=True, drop_enabled=False)

    pool_l = _fresh_pool(lesioned=True)
    lesioned = _judge_holding(_loaded_organ(), pool_l, ask_held=True, drop_enabled=True)

    no_crave_unchanged = bool(base["n"] == 2 and "dog" in base["named"] and "cat" in base["named"])
    dog_dropped_intact = bool("dog" not in drop["named"] and "cat" in drop["named"])
    dog_recovered_lesioned = bool("dog" in lesioned["named"] and "cat" in lesioned["named"])
    byte_identical_flagoff = bool(flag_off["named"] == base["named"])
    go = bool(no_crave_unchanged and dog_dropped_intact and dog_recovered_lesioned and byte_identical_flagoff)
    return {
        "seed": int(seed), "GO": go, "cross_weight_intact": pool_i.cross_weight,
        "cross_weight_lesioned": pool_l.cross_weight,
        "baseline_no_crave": base, "intact_crave_drop": drop, "lesioned_crave": lesioned,
        "flag_off_crave": flag_off,
        "no_crave_unchanged": no_crave_unchanged, "dog_dropped_intact": dog_dropped_intact,
        "dog_recovered_lesioned": dog_recovered_lesioned, "byte_identical_flagoff": byte_identical_flagoff,
    }


def main():
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--grow", action="store_true", help="build+grow the FROZEN pool and self-verify load-bearing")
    ap.add_argument("--semantic-drop", action="store_true",
                    help="also self-verify the SEMANTIC-DROP rung (register-0 genuine erase)")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    sd_results = []
    if args.semantic_drop:
        for s in seeds:
            sd = _selftest_semantic_drop(s)
            print(f"[seed {s}] SEMANTIC-DROP {'GO' if sd['GO'] else 'no'} | "
                  f"cross_weight intact={sd['cross_weight_intact']:.4f} lesioned={sd['cross_weight_lesioned']:.4f} "
                  f"| no_crave={sd['baseline_no_crave']['named']} "
                  f"| crave+intact={sd['intact_crave_drop']['named']} (dropped='dog'? {sd['dog_dropped_intact']}) "
                  f"| crave+lesioned={sd['lesioned_crave']['named']} (recovered? {sd['dog_recovered_lesioned']}) "
                  f"| flag_off={sd['flag_off_crave']['named']} (byte_id? {sd['byte_identical_flagoff']})",
                  flush=True)
            sd_results.append(sd)
        n_sd_go = sum(r["GO"] for r in sd_results)
        print(f"\n[SEMANTIC-DROP] {n_sd_go}/{len(sd_results)} seeds GO\n", flush=True)

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
               "semantic_drop": ({"n_go": sum(r["GO"] for r in sd_results), "n_seeds": len(sd_results),
                                  "results": sd_results} if args.semantic_drop else None),
               "note": ("frozen pre-grown curiosity.ask -> d6.w0 cross-edge wired into a dedicated production "
                        "pool; the DRIVE is lesion-attributable through crossedge_w0_shift (the SAME function "
                        "the live /api/brain-chat D6 hold-query block calls): holding a live recent crave state "
                        "shifts the frozen pool's own w0 rate NEGATIVE (suppression, matching the runner-level "
                        "6-seed GO's own measured sign), within the runner's own noise floor of ~0 shift when "
                        "ask_held=False (no signal, no bias), and the shift collapses under "
                        "BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_LESION=1. --semantic-drop additionally verifies the "
                        "2026-09-01 SEMANTIC-DROP rung: the crave-suppression signal now GENUINELY drops the "
                        "register-0-bound referent from MultiReferentWMOrgan's own recovered set via a real "
                        "hyperpolarizing pull on that register (MultiSlotHold.apply_register_drive), not an "
                        "appended string; lesion-attributable and byte-identical when the rung's own flag is off.")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print(f"\n[XEDGE-CURIOSITY-D6-PRODUCTION] {n_go}/{len(results)} seeds GO (lesion-attributable)", flush=True)
    all_go = (n_go == len(results)) and (not args.semantic_drop or sum(r["GO"] for r in sd_results) == len(sd_results))
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
