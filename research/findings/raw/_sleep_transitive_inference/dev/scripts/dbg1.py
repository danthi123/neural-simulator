import os, sys, json, time
os.environ["SIM_BACKEND"]="numpy"
sys.path.insert(0, os.getcwd())
import numpy as np
import logging; logging.disable(logging.WARNING)
from research.runners import sleep_transitive_inference as S
c=dict(S.DEFAULTS); c.update(n_laps=6, probe_trials=1, night_steps=3000)
st=S.build_store(7,c); S.encode(st,7,c,"hub"); w=S.get_w(st)
np.save("/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti/w_enc_s7_l6.npy", w)
M=S.pair_matrix(st,w)
print("pair matrix rows A1,B1,X1,Y1:")
for n in ["X1","A1","B1","Y1","D1","P1"]:
    print(n, " ".join(f"{S.NAMES[j]}:{M[S.IDX[n],j]:.0f}" for j in range(len(S.NAMES)) if M[S.IDX[n],j]>16))
# probe manually with spike counts
st2=S.build_store(7,c); S.load_w(st2,w); b=st2["bridge"]
rng=np.random.default_rng(1)
cue=S._dev(st2,S._cue_cells(st2,"A1",0.6,rng))
F=np.zeros((400,st2["n_pc"]),bool)
for t in range(400):
    b.cp_external_input_current[:]=0.0
    if t<40: b.cp_external_input_current[cue]+=9000.0
    b._run_one_simulation_step(); F[t]=S._fired_local(st2)
for i,al in enumerate(st2["asm_local"]):
    sc=F[:,al].sum()
    if sc>0:
        first=np.nonzero(F[:,al].any(1))[0]
        print(S.NAMES[i], "spikes",int(sc), "cells", int(F[:,al].any(0).sum()), "t_first", int(first[0]), "t_last", int(first[-1]))
