import os, sys, json, time
os.environ["SIM_BACKEND"]=os.environ.get("SIM_BACKEND","numpy")
sys.path.insert(0, os.getcwd())
import numpy as np
import logging; logging.disable(logging.WARNING)
from research.runners import sleep_transitive_inference as S
SP="/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti/"
c=dict(S.DEFAULTS)
for kv in sys.argv[2:]:
    k,v=kv.split("="); c[k]=type(c[k])(v)
tag=sys.argv[1]
t0=time.time()
st=S.build_store(7,c); S.encode(st,7,c,"hub"); w=S.get_w(st)
np.save(SP+f"w_enc_{tag}.npy", w)
M=S.pair_matrix(st,w)
print(f"encoded {time.time()-t0:.0f}s")
for n in ["X1","A1","B1","Y1","D1","P1"]:
    print(n, " ".join(f"{S.NAMES[j]}:{M[S.IDX[n],j]:.0f}" for j in range(len(S.NAMES)) if M[S.IDX[n],j]>16))
def run_probe(w, cue_name, sil=None):
    st2=S.build_store(7,c); S.load_w(st2,w); b=st2["bridge"]
    rng=np.random.default_rng(1)
    cue=S._dev(st2,S._cue_cells(st2,cue_name,0.6,rng))
    sild=S._dev(st2, st2["asm_local"][S.IDX[sil]]) if sil else None
    F=np.zeros((400,st2["n_pc"]),bool)
    for t in range(400):
        b.cp_external_input_current[:]=0.0
        if t<40: b.cp_external_input_current[cue]+=9000.0
        if sild is not None: b.cp_external_input_current[sild]-=20000.0
        b._run_one_simulation_step(); F[t]=S._fired_local(st2)
    out=[]
    for i,al in enumerate(st2["asm_local"]):
        sc=F[:,al].sum()
        if sc>0:
            first=np.nonzero(F[:,al].any(1))[0]
            out.append(f"{S.NAMES[i]}:{int(sc)}sp/{int(F[:,al].any(0).sum())}c@{int(first[0])}")
    print(f"probe {cue_name} sil={sil}: "+" ".join(out), flush=True)
run_probe(w,"A1"); run_probe(w,"B1"); run_probe(w,"X1"); run_probe(w,"A1","B1")
# night
st3=S.build_store(7,c); S.load_w(st3,w)
nr=S.night(st3,7,c)
cnt,co=S.replay_counts(st3,nr["F"],c)
print(f"night {time.time()-t0:.0f}s pop={nr['F'].mean():.4f} inits={len(nr['log'])} counts="+" ".join(f"{S.NAMES[i]}:{cnt[i]}" for i in range(len(S.NAMES))))
print("coact A1D1",co[S.IDX['A1'],S.IDX['D1']],"A1D2",co[S.IDX['A1'],S.IDX['D2']],"A1B1",co[S.IDX['A1'],S.IDX['B1']])
M2=S.pair_matrix(st3,nr["w"])
print("post AD1",M2[S.IDX['A1'],S.IDX['D1']],"DA1",M2[S.IDX['D1'],S.IDX['A1']],"AD2foil",M2[S.IDX['A1'],S.IDX['D2']],"AB",M2[S.IDX['A1'],S.IDX['B1']],"BD",M2[S.IDX['B1'],S.IDX['D1']])
np.save(SP+f"w_sleep_{tag}.npy", nr["w"])
run_probe(nr["w"],"A1","B1"); run_probe(nr["w"],"A1")
print(f"done {time.time()-t0:.0f}s")
