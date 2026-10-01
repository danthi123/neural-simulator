import os, sys, time
os.environ["SIM_BACKEND"]="numpy"; sys.path.insert(0, os.getcwd())
import numpy as np, logging; logging.disable(logging.WARNING)
from research.runners import sleep_transitive_inference as S
SP="/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti/"
tag=sys.argv[1]; enc=sys.argv[2]; c=dict(S.DEFAULTS)
for kv in sys.argv[3:]:
    k,v=kv.split("="); c[k]=type(c[k])(v)
c["probe_trials"]=2
def P(*a): print(*a, flush=True)
t0=time.time(); N=S.NAMES
w=np.load(SP+f"w_enc_{enc}.npy")
st=S.build_store(7,c); S.load_w(st,w)
nr=S.night(st,7,c)
cnt,co=S.replay_counts(st,nr["F"],c)
P(f"[{tag}] NIGHT {time.time()-t0:.0f}s pop={nr['F'].mean():.5f} inits={len(nr['log'])} counts="+" ".join(f"{N[i]}:{cnt[i]}" for i in range(len(N))))
P(f"[{tag}] coact A1D1={co[S.IDX['A1'],S.IDX['D1']]} A1D2={co[S.IDX['A1'],S.IDX['D2']]} A2D2={co[S.IDX['A2'],S.IDX['D2']]} A2D1={co[S.IDX['A2'],S.IDX['D1']]}")
np.save(SP+f"w_sleep_{tag}.npy", nr["w"]); np.save(SP+f"F_{tag}.npy", nr["F"])
def summ(M, lab):
    P(lab, " ".join(f"{a}>{b}:{M[S.IDX[a],S.IDX[b]]:.0f}" for a,b in [("A1","B1"),("B1","D1"),("D1","B1"),("A1","D1"),("D1","A1"),("A1","D2"),("A2","D2"),("A2","D1"),("B1","P1"),("B1","P2")]),
      f"offdiag_mean={M[~np.eye(len(N),dtype=bool)].mean():.1f}")
summ(S.pair_matrix(st,w), f"[{tag}] PRE ")
summ(S.pair_matrix(st,nr["w"]), f"[{tag}] POST")
wy=S.replica_stdp(st, nr["F"], w, c); P(f"[{tag}] yok max|dw|={np.abs(wy-nr['w']).max():.3g}")
G,sh=S.scramble_raster(st, nr["F"], 7, c); ws=S.replica_stdp(st, G, w, c)
summ(S.pair_matrix(st,ws), f"[{tag}] SCR ")
def rd(wx, lab):
    r=S.reads(7,c,wx)
    P(f"[{tag}] {lab}: DI={r['DI']:.3f} FI={r['FI']:.3f} PAB={r['PAB']:.3f} PBD={r['PBD']:.3f} PBP={r['PBP']:.3f} foil={r['foil_direct_max']:.3f} spurC={r['spurious_C_max']:.3f} sil={r['silence_held']}"
      + " | ch1 dD=%.2f dDf=%.2f ch2 dD=%.2f dDf=%.2f" % (r['chain1']['direct_D'], r['chain1']['direct_Dfoil'], r['chain2']['direct_D'], r['chain2']['direct_Dfoil']))
rd(nr["w"],"SLEEP"); rd(ws,"SCR")
P(f"[{tag}] done {time.time()-t0:.0f}s")
