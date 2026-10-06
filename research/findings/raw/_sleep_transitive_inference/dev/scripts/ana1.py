import os, sys
os.environ["SIM_BACKEND"]="numpy"; sys.path.insert(0, os.getcwd())
import numpy as np, logging; logging.disable(logging.WARNING)
from research.runners import sleep_transitive_inference as S
SP="/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti/"
tag=sys.argv[1]; c=dict(S.DEFAULTS)
for kv in sys.argv[2:]:
    k,v=kv.split("="); c[k]=type(c[k])(v)
st=S.build_store(7,c)
w0=np.load(SP+f"w_enc_{tag}.npy"); w1=np.load(SP+f"w_sleep_{tag}.npy")
M0=S.pair_matrix(st,w0); M1=S.pair_matrix(st,w1)
N=S.NAMES
d=M1-M0
idx=np.argsort(-np.abs(d).ravel())[:30]
for f in idx:
    i,j=divmod(f,len(N)); print(f"{N[i]}->{N[j]}: {M0[i,j]:.0f} -> {M1[i,j]:.0f} ({d[i,j]:+.0f})")
print("mean change all between pairs", d[~np.eye(len(N),dtype=bool)].mean())
