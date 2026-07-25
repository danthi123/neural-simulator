"""Dendritic Option-1 test: does the two-compartment WEIGHTED-coincidence plateau on the slots improve one-of-N
selectivity over the point-neuron baseline (mean 1.17/3)? First a k_thresh probe (does the plateau engage?)."""
import os,sys; os.environ.setdefault("SIM_BACKEND","cupy")
for _t in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(_t,"1")
sys.path.insert(0,"/home/dant123/Projects/sim")
from types import SimpleNamespace
import numpy as np
from research.runners.nmda_compositional_consolidation import (build_substrate, encode_facts_with_reinstatement,
    coactivation_replay, CONSOLIDATED_FACTS, _try_tgate, _try_pgate)
from sim.backend import get_backend, to_host
N=len(CONSOLIDATED_FACTS)
def slot_ign(b,tags):
    cp,_=get_backend(); rm=b.region_manager
    _try_tgate(b,"nmda_attractor",1.0); _try_pgate(b,"ca1_to_comp_attr",1.0)
    sa={s:list(rm.indices(f"comp_attr_{s}")) for s in range(N)}; sel=0; totf=0
    for i,tag in enumerate(tags):
        b.cp_external_input_current[:]=0.0
        if hasattr(b,"cp_v_apical") and b.cp_v_apical is not None: b.cp_v_apical[:]=b.core_config.adex_E_L if hasattr(b.core_config,'adex_E_L') else -70.0
        for _ in range(60): b._run_one_simulation_step()
        b.stimulate_tag(tag,drive_pA=1500.0,additive=False)
        cnt={s:0 for s in range(N)}
        for _ in range(80):
            b._run_one_simulation_step(); fs=to_host(b.cp_firing_states)
            for s in range(N): cnt[s]+=int(fs[sa[s]].sum())
        try: b.clear_tag_drive(tag)
        except: pass
        totf+=sum(cnt.values())
        if max(cnt,key=cnt.get)==i and cnt[i]>0: sel+=1
    return sel, totf
def run(seed, dend, k=3.0):
    A=dict(ca1_concept_density=0.25, ca1_concept_weight=0.0, nmda_self_weight=12.0, nmda_self_density=0.15,
           nmda_recurrent_ratio=0.6, cross_pool_density=0.10, stdp_w_max=8.0, enable_global_nmda=False, enable_hebbian=True,
           skip_nmda_additions=True, comp_attractor_slots=N, comp_attractor_n_per=120, comp_self_weight=12.0,
           comp_wta_weight=5.0, comp_dendritic=dend, comp_k_thresh=k)
    b=build_substrate(seed, SimpleNamespace(**A))
    tags,_=encode_facts_with_reinstatement(b,CONSOLIDATED_FACTS)
    coactivation_replay(b,CONSOLIDATED_FACTS,tags,100,seed,coactivate=True,attractor_on=True)
    return slot_ign(b,tags)
print("Dendritic k_thresh SWEEP (seeds 42,43): higher k -> only the strongly-driven fact-slot plateaus -> selective? find where sel rises + firing drops.",flush=True)
for k in (8.0,15.0,25.0,40.0,60.0):
    row=[]
    for seed in (42,43):
        sk,fk=run(seed,True,k=k); row.append((sk,fk))
    print(f"  k={k:.0f}: "+" | ".join(f"seed{sd} sel={r[0]}/{N} fire{r[1]}" for sd,r in zip((42,43),row)),flush=True)
print("CONSOL-DEND-KSWEEP DONE",flush=True)