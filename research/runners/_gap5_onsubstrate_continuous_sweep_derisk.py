"""Harder input for the on-substrate order detector: a CONTINUOUS travelling Gaussian sweep, so adjacent
reader cells activate with OVERLAP, instead of the clean separated pulses used so far.

That was the stated caveat of the detector/population-vote results ("hand-set input timing"). A real replay
event does not deliver disjoint pulses -- neighbouring place cells co-activate. This asks whether the pairwise
relay+coincidence read survives overlapping drive.

Tuning here is STRUCTURAL (each reader wired to its own place band) -- legitimate for testing a DOWNSTREAM
read-out, and explicitly NOT a claim that tuning was acquired (see the 2026-07-29 retractions).
"""
import os, sys, json
os.environ.setdefault("SIM_BACKEND","numpy")
for _tv in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv,"1")
sys.path.insert(0,"/home/dant123/Projects/sim")
import numpy as np, logging
logging.disable(logging.INFO)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim import SimulationBridge
NS, WREL, WDET = 50, 300.0, 10.0

def build(seed, K, lesion=False):
    R=[]; P=[]
    for k in range(K):
        R.append(BrainRegion(name="c%d"%k,n_neurons=NS,exc_fraction=1.0,internal_density=0.0))
    for k in range(K-1):
        R.append(BrainRegion(name="d%d"%k,n_neurons=NS,exc_fraction=1.0,internal_density=0.0))
        if lesion:
            P.append(RegionPathway(from_region="c%d"%k,to_region="d%d"%k,density=1.0,weight_mean=WDET,weight_jitter=0.0,plastic=False))
        else:
            R.append(BrainRegion(name="a%d"%k,n_neurons=NS,exc_fraction=1.0,internal_density=0.0))
            R.append(BrainRegion(name="b%d"%k,n_neurons=NS,exc_fraction=1.0,internal_density=0.0))
            P+=[RegionPathway(from_region="c%d"%k,to_region="a%d"%k,density=1.0,weight_mean=WREL,weight_jitter=0.0,plastic=False),
                RegionPathway(from_region="a%d"%k,to_region="b%d"%k,density=1.0,weight_mean=WREL,weight_jitter=0.0,plastic=False),
                RegionPathway(from_region="b%d"%k,to_region="d%d"%k,density=1.0,weight_mean=WDET,weight_jitter=0.0,plastic=False)]
        P.append(RegionPathway(from_region="c%d"%(k+1),to_region="d%d"%k,density=1.0,weight_mean=WDET,weight_jitter=0.0,plastic=False))
    cfg=CoreSimConfig(seed=seed,dt_ms=1.0,enable_brain_region_framework=True,brain_regions=R,region_pathways=P,
                      enable_hebbian_learning=False,enable_stdp=False,enable_homeostasis=False,
                      enable_structural_plasticity=False,enable_ou_process=False)
    b=SimulationBridge(core_config=cfg,viz_config=VisualizationConfig(),runtime_state=RuntimeState(),gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False); return b

def run(seed,K,direction,lag=12,overlap=1.0,lesion=False,drive=8000.0,static=False):
    """CONTINUOUS sweep: cell k's drive is a Gaussian in time centred at k*lag, width=overlap*lag.
    overlap=0.15 ~ near-disjoint pulses; overlap=1.0 => strong co-activation of neighbours."""
    b=build(seed,K,lesion); rm=b.region_manager
    T=K*lag+50
    sd=max(1e-6, overlap*lag)
    ctr={k:(3+ (k if direction>0 else (K-1-k))*lag) for k in range(K)}
    if static: ctr={k:3+(K//2)*lag for k in range(K)}   # no travel: all centred together
    dets=[rm.indices("d%d"%k) for k in range(K-1)]
    tot=0
    for step in range(T):
        b.cp_external_input_current[:]=0.0
        for k in range(K):
            amp=drive*float(np.exp(-0.5*((step-ctr[k])/sd)**2))
            if amp>drive*0.02: b.cp_external_input_current[rm.indices("c%d"%k)]=amp
        b._run_one_simulation_step()
        for d in dets: tot+=int(np.asarray(b.cp_firing_states[d]).sum())
    return tot

if __name__=="__main__":
    K=6; seeds=(42,43,44)
    print("CONTINUOUS-SWEEP input (overlap = temporal width / pair lag; higher = neighbours co-active)")
    print("  %-9s %-9s %-9s %-9s %-9s %-9s"%("overlap","fwd","rev","ratio","STATIC","LESION_r"))
    out={}
    for ov in (0.15,0.35,0.6,1.0):
        f=np.mean([run(s,K,+1,overlap=ov) for s in seeds])
        r=np.mean([run(s,K,-1,overlap=ov) for s in seeds])
        st=np.mean([run(s,K,+1,overlap=ov,static=True) for s in seeds])
        lf=np.mean([run(s,K,+1,overlap=ov,lesion=True) for s in seeds])
        lr=np.mean([run(s,K,-1,overlap=ov,lesion=True) for s in seeds])
        ratio=(f+1e-9)/(r+1e-9); lratio=(lf+1e-9)/(lr+1e-9)
        out["ov%.2f"%ov]=dict(fwd=float(f),rev=float(r),ratio=float(ratio),static=float(st),lesion_ratio=float(lratio))
        print("  %-9.2f %-9.1f %-9.1f %-9.2f %-9.1f %-9.2f"%(ov,f,r,ratio,st,lratio))
    json.dump(out,open("/home/dant123/Projects/sim/research/findings/raw/gap5_reader/continuous_sweep.json","w"),indent=1)
    print()
    print("  (ratio must stay >1 as overlap grows; LESION_r must stay ~1; STATIC must stay low)")

# The ratio is not the decision-relevant number -- single-trial accuracy is. The earlier jitter curve put
# ratio~1.75 at accuracy ~0.806, so the 1.000 figure from DISJOINT pulses must not be quoted for this input.
print()
print("SINGLE-TRIAL accuracy under continuous sweep (paired fwd-vs-rev, per-trial timing jitter 3 ms)")
print("  %-9s %-14s %-10s"%("overlap","acc","n_pairs"))
def run_j(seed,K,direction,ov,jit=3.0,lag=12,drive=8000.0):
    rng=np.random.default_rng(seed*104729+direction)
    b=build(seed,K,False); rm=b.region_manager
    T=K*lag+50; sd=max(1e-6,ov*lag)
    ctr={k:(3+(k if direction>0 else (K-1-k))*lag + float(rng.normal(0,jit))) for k in range(K)}
    dets=[rm.indices("d%d"%k) for k in range(K-1)]; tot=0
    for step in range(T):
        b.cp_external_input_current[:]=0.0
        for k in range(K):
            amp=drive*float(np.exp(-0.5*((step-ctr[k])/sd)**2))
            if amp>drive*0.02: b.cp_external_input_current[rm.indices("c%d"%k)]=amp
        b._run_one_simulation_step()
        for d in dets: tot+=int(np.asarray(b.cp_firing_states[d]).sum())
    return tot
for ov in (0.15,0.6,1.0):
    hits=0; n=0
    for seed in (42,43,44,100,101,102):
        for t in range(4):
            s=seed+1000*t
            if run_j(s,6,+1,ov) > run_j(s,6,-1,ov): hits+=1
            n+=1
    print("  %-9.2f %-14.3f %-10d"%(ov,hits/n,n))
print("  (chance = 0.500)")
