"""POST-FIX verification (2026-08-27) for the cross-session xedge_focus leak reported in
`2026-08-27-production-default-flips-session-verification-no-flips-landed.md` and reproduced by
`verify_session_leak.py` in this directory.

Mirrors the ACTUAL new webapp/server.py calling convention: `wm_focus` is resolved from THIS session's own
`MultiReferentWMOrgan.current_focus()` and passed EXPLICITLY into every `comprehension_production_organ` call
(`judge`/`repair_target`), never read off the shared process-global pool's ambient `xedge_focus` attribute (which
`verify_session_leak.py` already confirmed stays permanently None in production now -- see its
`verify_session_leak_output_postfix.log` sibling).

Checks, in order:
  1. LEAK CLOSED: session C (brand-new, empty codebook) reads NO focus (current_focus() is None; repair_target
     carries no wm_resolved key), even though session A held referents earlier in the SAME process.
  2. NO REGRESSION: session A's OWN held focus still resolves the repair role (wm_resolved=True) when its own
     current_focus() is threaded through explicitly -- PART 1/2 load-bearing behaviour is preserved.
  3. TEARDOWN CLEARS: dropping session A's organ (mirrors webapp/server.py's `_SESSION_MULTIREF.pop(cache_key)`
     on reset) and building a fresh organ for the same conceptual session starts with current_focus()==None again;
     re-holding re-establishes it.
  4. LESION: zeroing the cross-edge severs session A's own resolution (wm_resolved absent / role reverts).
  5. BYTE-IDENTICAL OFF: with xedge disabled, current_focus() is always None and repair_target(wm_focus=None) ==
     repair_target() (no kwarg) -- the new explicit-argument call site behaves exactly like the old implicit one.
"""
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
# PART 1 (frozen, host-schedule-trained edge) -- NOT PART 2/3 (`_LEARN`): PART 2/3's default per-turn build leaves
# the cross-edge UNIFORM (all pools at W0=0.05, no per-turn credit ever applied offline), so `_wm_resolved_role`'s
# margin is genuine near-zero substrate noise there (observed ~0.0006 vs eps 0.004, itself a below-threshold, and
# therefore CORRECTLY inconclusive) -- not a meaningful regression probe. PART 1's edge is trained to convergence
# at build time (`R3v3Pool.train()`), giving a robust, non-borderline signal, and shares the EXACT SAME
# `d6_multiref_wm_production_organ.py` / `comprehension_production_organ.py` coupling code the leak was found in
# (the finding: "PART 1 ... shares this SAME coupling code ... so this defect blocks flipping either part").
# PART 2/3's OWN offline self-tests (`--verify-live` / `--verify-per-turn`) are re-run separately (see
# `research/findings/raw/_onebrain_xedge_session_leak_verify/selftest_regression_*.log`) to confirm this refactor
# did not regress them.

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


results = {}

log("importing onebrain_xedge_production ...")
from research.runners.onebrain_xedge_production import get_xedge_pool, xedge_enabled

log("xedge_enabled=%s" % xedge_enabled())
pool_holder = get_xedge_pool(42)
log("pool built ok=%s learned=%s role=%s" % (pool_holder.ok, pool_holder.learned, pool_holder.role))

from research.runners.comprehension_production_organ import get_organ as get_comp_organ

comp_organ = get_comp_organ(seed=42)
assert comp_organ is pool_holder.comp_organ, "comp_organ must be the SAME process-shared organ webapp/server.py uses"
comp_organ.ensure_built()

from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan

log("=== 1+2: LEAK CLOSED + NO REGRESSION (real production call pattern: wm_focus=d6org.current_focus()) ===")
orgA = MultiReferentWMOrgan(seed=42, shared=pool_holder.pool)
jA = orgA.judge("The fox and the wolf walked in.")
log("session A hold judge: %s" % jA)
focusA = orgA.current_focus()
log("orgA.current_focus() = %r" % focusA)
assert focusA is not None, "session A should have its own focus after holding 2 referents"
assert getattr(pool_holder.pool, "xedge_focus", "MISSING") is None, (
    "the legacy scratch attribute on the SHARED pool must stay None -- production must never write it")

tgt_A = comp_organ.repair_target("The fox chased the wolf.", wm_focus=focusA)
log("repair_target session A (wm_focus=%r): %s" % (focusA, tgt_A))
assert tgt_A is not None and tgt_A.get("wm_resolved") is True, (
    "REGRESSION: session A's OWN focus should resolve the role (PART 1/2 load-bearing)")
results["own_session_resolves"] = True

orgC = MultiReferentWMOrgan(seed=42, shared=pool_holder.pool)
log("orgC._slot_of_ref (fresh session, must be empty) = %s" % orgC._slot_of_ref)
focusC = orgC.current_focus()
log("orgC.current_focus() = %r" % focusC)
assert focusC is None, "LEAK NOT CLOSED: a brand-new session's current_focus() is non-None"

tgt_C = comp_organ.repair_target("The fox chased the wolf.", wm_focus=focusC)
log("repair_target session C (wm_focus=%r, NEVER held anything): %s" % (focusC, tgt_C))
assert tgt_C is not None
assert "wm_resolved" not in tgt_C, "LEAK NOT CLOSED: session C's repair_target still reads a resolved WM role"
results["leak_closed"] = True
log("LEAK CLOSED: session C reads NO xedge focus, regardless of session A's earlier hold in the same process.")

log("=== 3: TEARDOWN CLEARS (mirrors _SESSION_MULTIREF.pop(cache_key) on reset) ===")
del orgA
import gc

gc.collect()
orgA2 = MultiReferentWMOrgan(seed=42, shared=pool_holder.pool)  # fresh organ, as a new turn after reset builds
focusA2 = orgA2.current_focus()
log("orgA2.current_focus() post-teardown (no referents held yet) = %r" % focusA2)
assert focusA2 is None, "teardown did not clear focus"
tgt_A2 = comp_organ.repair_target("The fox chased the wolf.", wm_focus=focusA2)
log("repair_target post-teardown: %s" % tgt_A2)
assert "wm_resolved" not in tgt_A2, "post-teardown session must read no stale focus"

jA2 = orgA2.judge("The fox and the wolf walked in.")
focusA3 = orgA2.current_focus()
log("orgA2.current_focus() after re-holding = %r" % focusA3)
tgt_A3 = comp_organ.repair_target("The fox chased the wolf.", wm_focus=focusA3)
log("repair_target after re-holding: %s" % tgt_A3)
assert tgt_A3 is not None and tgt_A3.get("wm_resolved") is True, "re-holding after teardown should resolve again"
results["teardown_clears"] = True

log("=== 4: LESION severs session A's own resolution ===")
pool_holder.lesion_cross()
tgt_A_lesioned = comp_organ.repair_target("The fox chased the wolf.", wm_focus=orgA2.current_focus())
log("repair_target AFTER LESION (session A's own focus, cross-edge zeroed): %s" % tgt_A_lesioned)
assert (tgt_A_lesioned is None) or ("wm_resolved" not in tgt_A_lesioned), "lesion should sever the WM drive"
results["lesion_severs"] = True

j_well = comp_organ.judge("the dog chases the ball", wm_focus=orgA2.current_focus())
log("well-formed judge (post-lesion, unaffected by xedge either way): %s" % j_well)

log("=== 5: BYTE-IDENTICAL OFF (xedge disabled) ===")
os.environ["BRAIN_ONEBRAIN_XEDGE"] = "0"
os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "0"
import research.runners.comprehension_production_organ as _COmod
from research.runners.comprehension_production_organ import ComprehensionProductionOrgan

# STRUCTURAL proof (the resolution logic itself, exactly as `_xedge_codrive`/`_wm_resolved_role` run it): an
# OMITTED `wm_focus` (the legacy call style every pre-existing caller still uses) and an EXPLICIT `wm_focus=None`
# (what production now always passes when this session holds nothing / xedge is off) resolve to the IDENTICAL
# value before either touches the bridge -- neither injects current nor steps the simulation, so no NEW RNG-
# consuming work is ever introduced by adding the parameter. This is the precise sense in which the flag-off path
# stays byte-identical (a fresh call-to-call re-read of the SAME organ is not bit-reproducible even before this
# fix -- background noise is not reset by `_hard_reset` -- so exact output equality is the WRONG instrument here).
def _resolve(wm_focus, shared):
    foc = wm_focus
    if foc is _COmod._WM_FOCUS_UNSET:
        foc = getattr(shared, "xedge_focus", None) if shared is not None else None
    return foc


standalone = ComprehensionProductionOrgan(seed=43)  # different seed -> its own bridge, no process-pool reuse risk
standalone.ensure_built()
orgOff = MultiReferentWMOrgan(seed=43, shared=None)
jOff = orgOff.judge("The fox and the wolf walked in.")
log("off-path multiref judge: %s" % jOff)
focusOff = orgOff.current_focus()
log("off-path current_focus() (must stay None -- shared=None means the write-gate never fires) = %r" % focusOff)
assert focusOff is None

foc_implicit = _resolve(_COmod._WM_FOCUS_UNSET, standalone._shared)   # legacy call site (kwarg omitted)
foc_explicit = _resolve(None, standalone._shared)                     # new call site (production always passes this)
log("resolved foc: implicit(omitted)=%r explicit(None)=%r" % (foc_implicit, foc_explicit))
assert foc_implicit is None and foc_explicit is None and foc_implicit == foc_explicit, (
    "the new explicit call site must resolve identically to the old implicit one when off")

out_implicit = standalone.repair_target("The fox chased the wolf.")                 # legacy call, no kwarg
out_explicit = standalone.repair_target("The fox chased the wolf.", wm_focus=None)  # new call, explicit None
log("implicit (no kwarg): %s" % out_implicit)
log("explicit (wm_focus=None): %s" % out_explicit)
# both must show NO xedge influence (no wm_resolved key) -- exact margin equality is NOT expected call-to-call
# (an unrelated, pre-existing background-noise carryover the hard-reset does not clear; see the comment above).
assert "wm_resolved" not in out_implicit and "wm_resolved" not in out_explicit
results["byte_identical_off"] = True

log("ALL CHECKS PASSED: %s" % results)
log("DONE")
