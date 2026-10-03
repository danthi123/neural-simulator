import os, sys, time
os.environ["SIM_BACKEND"]=os.environ.get("SIM_BACKEND","numpy"); sys.path.insert(0, os.getcwd())
import numpy as np, logging; logging.disable(logging.WARNING)
from research.runners import sleep_transitive_inference as S
SP="/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti/"
tag=sys.argv[1]; seed=int(sys.argv[2]); c=dict(S.DEFAULTS)
for kv in sys.argv[3:]:
    k,v=kv.split("="); c[k]=type(c[k])(v)
c["probe_trials"]=2
t0=time.time()
def P(*a): print(*a, flush=True)
st=S.build_store(seed,c); S.encode(st,seed,c,"hub"); w=S.get_w(st)
np.save(SP+f"w_enc_{tag}.npy", w)
N=S.NAMES
def summ(M, lab):
    P(lab, " ".join(f"{a}>{b}:{M[S.IDX[a],S.IDX[b]]:.0f}" for a,b in [("A1","B1"),("B1","D1"),("D1","B1"),("A1","D1"),("D1","A1"),("A1","D2"),("X1","A1"),("B1","P1"),("B1","P2")]),
      f"offdiag_mean={M[~np.eye(len(N),dtype=bool)].mean():.1f}")
M0=S.pair_matrix(st,w); summ(M0, f"[{tag}] ENC {time.time()-t0:.0f}s")
def rd(wx, lab):
    r=S.reads(seed,c,wx)
    P(f"[{tag}] {lab}: DI={r['DI']:.3f} FI={r['FI']:.3f} PAB={r['PAB']:.3f} PBD={r['PBD']:.3f} PBP={r['PBP']:.3f} foil={r['foil_direct_max']:.3f} spurC={r['spurious_C_max']:.3f} sil={r['silence_held']} "
      + " | ch1 direct_D=%.2f free_D=%.2f AB=%.2f BD=%.2f" % (r['chain1']['direct_D'], r['chain1']['free_D'], r['chain1']['AB'], r['chain1']['BD']))
    return r
rd(w,"WAKE")
st3=S.build_store(seed,c); S.load_w(st3,w)
nr=S.night(st3,seed,c)
cnt,co=S.replay_counts(st3,nr["F"],c)
P(f"[{tag}] NIGHT {time.time()-t0:.0f}s pop={nr['F'].mean():.4f} inits={len(nr['log'])} counts="+" ".join(f"{N[i]}:{cnt[i]}" for i in range(len(N))))
P(f"[{tag}] coact A1D1={co[S.IDX['A1'],S.IDX['D1']]} A1D2={co[S.IDX['A1'],S.IDX['D2']]} A1B1={co[S.IDX['A1'],S.IDX['B1']]} B1D1={co[S.IDX['B1'],S.IDX['D1']]}")
np.save(SP+f"w_sleep_{tag}.npy", nr["w"])
M1=S.pair_matrix(st3,nr["w"]); summ(M1, f"[{tag}] POST")
rd(nr["w"],"SLEEP")
P(f"[{tag}] done {time.time()-t0:.0f}s")
