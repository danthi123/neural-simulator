"""6-seed confirm of the consolidation SELECTIVITY bounded negative: no-SFA, co-activation ON, does selective one-of-N
FAIL (<=1/3=chance) across 6 seeds? Solidifies the 1-seed result to the 6-seed bar."""
import os,sys,time
os.environ.setdefault("SIM_BACKEND","cupy")
for _tv in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(_tv,"1")
sys.path.insert(0,"/home/dant123/Projects/sim")
from types import SimpleNamespace
import numpy as np
from research.runners.nmda_compositional_consolidation import (build_substrate, encode_facts_with_reinstatement,
    coactivation_replay, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from sim.backend import get_backend, to_host
ARGS=dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
          nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False, enable_hebbian=True,
          skip_nmda_additions=True, comp_attractor_slots=len(CONSOLIDATED_FACTS), comp_attractor_n_per=120,
          comp_self_weight=12.0, comp_wta_weight=5.0)
N=len(CONSOLIDATED_FACTS)
def slot_ign(b,tags):
    cp,_=get_backend(); rm=b.region_manager
    _try_tgate(b,"nmda_attractor",1.0); _try_pgate(b,"ca1_to_comp_attr",1.0)
    sa={s:list(rm.indices(f"comp_attr_{s}")) for s in range(N)}
    sel=0
    for i,tag in enumerate(tags):
        b.cp_external_input_current[:]=0.0
        for _ in range(60): b._run_one_simulation_step()
        b.stimulate_tag(tag,drive_pA=1500.0,additive=False)
        cnt={s:0 for s in range(N)}
        for _ in range(80):
            b._run_one_simulation_step(); fs=to_host(b.cp_firing_states)
            for s in range(N): cnt[s]+=int(fs[sa[s]].sum())
        try: b.clear_tag_drive(tag)
        except: pass
        if max(cnt,key=cnt.get)==i and cnt[i]>0: sel+=1
    return sel
print(f"6-SEED consolidation SELECTIVITY boundary confirm (no-SFA, co-activation ON). chance=1/{N}.",flush=True)
sels=[]
for seed in (42,43,44,100,101,102):
    b=build_substrate(seed,SimpleNamespace(**ARGS))
    tags,_=encode_facts_with_reinstatement(b,CONSOLIDATED_FACTS)
    coactivation_replay(b,CONSOLIDATED_FACTS,tags,100,seed,coactivate=True,attractor_on=True)
    sel=slot_ign(b,tags); sels.append(sel)
    print(f"  seed {seed}: SELECTIVE {sel}/{N}",flush=True)
sels=np.array(sels)
print(f"\n  SELECTIVE per seed {sels.tolist()} mean={sels.mean():.2f}/{N} | boundary CONFIRMED iff all <=1 (=chance) -> {'CONFIRMED (single-winner, no one-of-N)' if (sels<=1).all() else 'NOT confirmed'}",flush=True)
print("CONSOL-BOUNDARY-6SEED DONE",flush=True)
