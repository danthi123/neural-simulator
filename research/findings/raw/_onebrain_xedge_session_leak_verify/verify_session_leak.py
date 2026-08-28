import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log("importing onebrain_xedge_production ...")
from research.runners.onebrain_xedge_production import get_xedge_pool, xedge_enabled
log("xedge_enabled=%s" % xedge_enabled())
pool_holder = get_xedge_pool(42)
log("pool built ok=%s learned=%s role=%s cross_weights=%s" % (
    pool_holder.ok, pool_holder.learned, pool_holder.role, pool_holder.cross_weights))

from research.runners.comprehension_production_organ import get_organ as get_comp_organ
comp_organ = get_comp_organ(seed=42)
log("comp_organ IS pool.comp_organ (process-shared, exactly server.py's _get_comprehension_organ): %s" %
    (comp_organ is pool_holder.comp_organ))
comp_organ.ensure_built()
log("comp_organ built. threshold=%.4f role_floor=%.4f lean_margin=%.4f" %
    (comp_organ.threshold, comp_organ.role_floor, comp_organ.lean_margin))

from research.runners.d6_multiref_wm_production_organ import MultiReferentWMOrgan

log("--- REAL production call path: session A introduces 2 referents (server.py line ~4926-4940) ---")
orgA = MultiReferentWMOrgan(seed=42, shared=pool_holder.pool)
jA = orgA.judge("The fox and the wolf walked in.")
log("session A hold judge: %s" % jA)
log("pool.xedge_focus after A's hold = %r  (CAND_POOLS positional constant)" % pool_holder.pool.xedge_focus)

log("--- REAL production call path: comp_organ.repair_target (server.py line ~5176) for session A's own ambiguous turn ---")
tgt_A = comp_organ.repair_target("The fox chased the wolf.")
log("repair_target after A's hold (session A's own turn): %s" % tgt_A)

log("--- session C: BRAND NEW MultiReferentWMOrgan sharing the SAME pool, NEVER calls .load() ---")
orgC = MultiReferentWMOrgan(seed=42, shared=pool_holder.pool)
log("orgC._slot_of_ref (fresh session, should be empty) = %s" % orgC._slot_of_ref)
tgt_C = comp_organ.repair_target("The fox chased the wolf.")
log("repair_target for session C (this session NEVER held anything, but A's hold happened earlier IN THE PROCESS): %s" % tgt_C)

log("--- lesion: zero the cross-edge weights in place, re-measure the SAME ambiguous turn ---")
pool_holder.lesion_cross()
tgt_A_lesioned = comp_organ.repair_target("The fox chased the wolf.")
log("repair_target AFTER LESION (cross-edge zeroed): %s" % tgt_A_lesioned)

log("--- regression: a well-formed (non-ambiguous) transitive should be comprehended, unaffected by xedge ---")
j_well = comp_organ.judge("the dog chases the ball")
log("well-formed judge: %s" % j_well)

log("--- baseline (xedge OFF): a standalone comprehension organ + standalone multiref organ, same sentences ---")
from research.runners.comprehension_production_organ import ComprehensionProductionOrgan
standalone_comp = ComprehensionProductionOrgan(seed=42)
standalone_comp.ensure_built()
tgt_off = standalone_comp.repair_target("The fox chased the wolf.")
log("OFF (standalone, no shared pool) repair_target: %s" % tgt_off)
j_well_off = standalone_comp.judge("the dog chases the ball")
log("OFF well-formed judge: %s" % j_well_off)

log("DONE")
