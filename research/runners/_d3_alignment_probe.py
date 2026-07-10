import sys, numpy as np
from research.runners._d3_event_selfsup_pair_derisk import make_pair_task, INTRO, COREF, PROMOTE, BOUND, RETURN
from research.runners._d3_delta_cleanerror_derisk import train_cleanerror
from research.runners._d3_event_gated_copy_derisk import _sm, _sig
NAMES={INTRO:"INTRO",COREF:"COREF",PROMOTE:"PROMOTE",BOUND:"BOUND",RETURN:"RETURN"}
seed=int(sys.argv[1]); K=6; task=make_pair_task(seed,K=K)
roll=train_cleanerror(task,seed=seed,epochs=40,batch=32,lr=0.02,credit="clean_error")   # ITS OWN trained weights
Wc,We,emb=roll.W["Wc"],roll.W["We"],roll.W["emb"]; Wr,Wi,bc,be=roll.W["Wr"],roll.W["Wi"],roll.W["bc"],roll.W["be"]
Ye=roll.Y["Ye"]; wg,bg,wp,bp=roll.gates
X,OBJ,EMIT,L,AC,AP,PE,PC=task["test_deeper"]; OPS=task["ops_test"]; ident=task["ident"]
eyeM=np.eye(task["M"],dtype=np.float32); B=len(L); Lm=int(L.max())
sc=np.zeros((B,K),np.float32); sc[:,ident]=1.0
sp=np.zeros((B,K),np.float32); sp[:,ident]=1.0
pat=np.zeros((B,K),np.float32); pat[:,ident]=1.0
cos={k:[] for k in NAMES}
for t in range(Lm):
    act=L>t
    h=np.tanh(np.concatenate([sc@emb,sp@emb,pat@emb],axis=1)@Wr.T + X[:,t]@Wi.T)
    raw=_sm(h@Wc.T+bc); g=_sig(X[:,t]@wg+bg)[:,None]; r=_sig(X[:,t]@wp+bp)[:,None]
    npv=g*sc+(1-g)*sp; nsc=r*sp+(1-r)*raw
    se=_sm((nsc@emb)@We.T+be); d_le=se-eyeM[EMIT[:,t]]
    e_bp=(d_le@We)@emb.T; e_fr=d_le@Ye
    c=(e_bp*e_fr).sum(1)/(np.linalg.norm(e_bp,axis=1)*np.linalg.norm(e_fr,axis=1)+1e-12)
    m=np.where(act)[0]
    for k in NAMES:
        mm=m[OPS[m,t]==k]
        if len(mm): cos[k].append(float(c[mm].mean()))
    sc=np.where(act[:,None],nsc,sc); sp=np.where(act[:,None],npv,sp)
    pn=np.zeros((B,K),np.float32); pn[np.arange(B),OBJ[:,t]]=1.0; pat=np.where(act[:,None],pn,pat)
parts=[f"{NAMES[k]}={np.mean(v):+.3f}" for k,v in cos.items() if v]
allc=np.mean([np.mean(v) for v in cos.values() if v])
print(f"seed {seed}  cos(fixed-random credit, TRUE credit) on the CLEAN-ERROR model: " + "  ".join(parts) + f"  || overall {allc:+.3f}", flush=True)
