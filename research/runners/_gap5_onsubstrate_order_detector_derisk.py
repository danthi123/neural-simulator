"""ON-SUBSTRATE pairwise order detector, at the pinned operating point (n=50, w=300, 2 relay hops = 11.5 ms).

WIRING: cell-group A -> [relay r0 -> relay r1] -> DET   (delayed ~11.5 ms)
        cell-group B -----------------------------> DET   (direct)
FORWARD input = A fires, then B fires ~12 ms later -> A's DELAYED spike arrives WITH B's direct spike ->
coincidence -> DET fires. REVERSE input = B then A -> no coincidence -> DET fires less.

SCOPE (honest): this tests the DETECTOR PRIMITIVE the pairwise read is built from -- the part that needed the
delay. It does NOT include learned tuning or the full 40-cell population read.

CONTROLS: reverse order; LESION (A wired DIRECT, relay bypassed -> order-blindness, must go ~neutral);
SIMULTANEOUS (no order, must be neutral between fwd and rev).
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

N=50; W_RELAY=300.0

def build(seed, w_det, lesion=False):
    R=[BrainRegion(name=n, n_neurons=N, exc_fraction=1.0, internal_density=0.0)
       for n in ("A","B","r0","r1","DET")]
    P=[]
    if lesion:
        # LESION: A goes DIRECT to DET (no relay) -> the detector loses its delay -> must be order-blind
        P.append(RegionPathway(from_region="A",to_region="DET",density=1.0,weight_mean=w_det,weight_jitter=0.0,plastic=False))
    else:
        P += [RegionPathway(from_region="A", to_region="r0", density=1.0,weight_mean=W_RELAY,weight_jitter=0.0,plastic=False),
              RegionPathway(from_region="r0",to_region="r1", density=1.0,weight_mean=W_RELAY,weight_jitter=0.0,plastic=False),
              RegionPathway(from_region="r1",to_region="DET",density=1.0,weight_mean=w_det, weight_jitter=0.0,plastic=False)]
    P.append(RegionPathway(from_region="B",to_region="DET",density=1.0,weight_mean=w_det,weight_jitter=0.0,plastic=False))
    cfg=CoreSimConfig(seed=seed,dt_ms=1.0,enable_brain_region_framework=True,brain_regions=R,region_pathways=P,
                      enable_hebbian_learning=False,enable_stdp=False,enable_homeostasis=False,
                      enable_structural_plasticity=False,enable_ou_process=False)
    b=SimulationBridge(core_config=cfg,viz_config=VisualizationConfig(),runtime_state=RuntimeState(),gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False); return b

def run(seed, w_det, order, gap=12, lesion=False, T=70, drive=8000.0):
    """order: +1 = A then B (forward); -1 = B then A (reverse); 0 = simultaneous."""
    b=build(seed,w_det,lesion); rm=b.region_manager
    A=rm.indices("A"); B=rm.indices("B"); D=rm.indices("DET")
    tA = 3 if order>=0 else 3+gap
    tB = 3+gap if order>0 else (3 if order<0 else 3)
    if order==0: tA=tB=3
    ndet=0
    for step in range(T):
        b.cp_external_input_current[:]=0.0
        if tA<=step<=tA+1: b.cp_external_input_current[A]=drive
        if tB<=step<=tB+1: b.cp_external_input_current[B]=drive
        b._run_one_simulation_step()
        ndet+=int(np.asarray(b.cp_firing_states[D]).sum())
    return ndet

def single_only(seed,w_det,which,T=70,drive=8000.0):
    """Drive ONLY A or ONLY B. For a genuine coincidence detector this must be ~0."""
    b=build(seed,w_det,False); rm=b.region_manager
    idx=rm.indices(which); D=rm.indices("DET"); n=0
    for step in range(T):
        b.cp_external_input_current[:]=0.0
        if 3<=step<=4: b.cp_external_input_current[idx]=drive
        b._run_one_simulation_step()
        n+=int(np.asarray(b.cp_firing_states[D]).sum())
    return n

if __name__=="__main__":
    print("STEP 0 -- the DEFINING property: is a SINGLE input subthreshold? (must be ~0 for coincidence)")
    print("%-9s %-12s %-12s %-10s"%("w_det","A_alone","B_alone","regime"))
    for w_det in (5.0,10.0,20.0,30.0,40.0,60.0):
        a=np.mean([single_only(s,w_det,"A") for s in (42,43,44)])
        bb=np.mean([single_only(s,w_det,"B") for s in (42,43,44)])
        reg="COINCIDENCE (both subthr)" if (a<1 and bb<1) else ("summation (single fires)" if (a>3 or bb>3) else "marginal")
        print("%-9.0f %-12.1f %-12.1f %-10s"%(w_det,a,bb,reg))
    print()
    print("Finding a COINCIDENCE regime: DET must be subthreshold to ONE input, suprathreshold to TWO.")
    print("%-9s %-10s %-10s %-10s"%("w_det","fwd","rev","fwd/rev"))
    best=None
    for w_det in (5.0,10.0,20.0,30.0,40.0,60.0,100.0):
        f=np.mean([run(s,w_det,+1) for s in (42,43,44)])
        r=np.mean([run(s,w_det,-1) for s in (42,43,44)])
        ratio=(f+1e-9)/(r+1e-9)
        print("%-9.0f %-10.1f %-10.1f %-10.2f"%(w_det,f,r,ratio))
        if f>3 and ratio>(best[1] if best else 1.05): best=(w_det,ratio)
    print()
    if best is None:
        print("  NO coincidence regime found in this sweep -> UNDEFINED, not a negative. Widen the sweep.")
    else:
        w=best[0]; print("=== controls at the best w_det=%.0f (6 seeds) ==="%w)
        seeds=(42,43,44,100,101,102)
        fwd=[run(s,w,+1) for s in seeds]; rev=[run(s,w,-1) for s in seeds]
        sim=[run(s,w, 0) for s in seeds]
        lf =[run(s,w,+1,lesion=True) for s in seeds]; lr=[run(s,w,-1,lesion=True) for s in seeds]
        m=lambda x: float(np.mean(x))
        print("  FORWARD           %s  mean=%.1f"%(fwd,m(fwd)))
        print("  REVERSE           %s  mean=%.1f"%(rev,m(rev)))
        print("  SIMULTANEOUS      %s  mean=%.1f"%(sim,m(sim)))
        print("  LESION fwd (no relay) %s  mean=%.1f"%(lf,m(lf)))
        print("  LESION rev (no relay) %s  mean=%.1f"%(lr,m(lr)))
        print()
        print("  intact  fwd/rev = %.3f   <- must be >1"%((m(fwd)+1e-9)/(m(rev)+1e-9)))
        print("  LESION  fwd/rev = %.3f   <- must be ~1 (order-blind without the delay)"%((m(lf)+1e-9)/(m(lr)+1e-9)))
        json.dump(dict(w_det=w,fwd=fwd,rev=rev,sim=sim,lesion_fwd=lf,lesion_rev=lr),
                  open("/home/dant123/Projects/sim/research/findings/raw/gap5_reader/onsub_detector.json","w"),indent=1)
