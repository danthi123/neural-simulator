import os, sys, time
os.environ["SIM_BACKEND"]="numpy"; sys.path.insert(0, os.getcwd())
import numpy as np, logging; logging.disable(logging.WARNING)
from research.runners import sleep_transitive_inference as S
SP="/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti/"
c=dict(S.DEFAULTS); c.update(between_init=15.0, night_steps=30000, swr_period=10000)
w=np.load(SP+"w_enc_v1.npy")
st=S.build_store(7,c); S.load_w(st,w)
t0=time.time(); nr=S.night(st,7,c); t1=time.time()
F=nr["F"]; ev=F.any(1)
print(f"30000 steps {t1-t0:.1f}s  active_steps={int(ev.sum())} pop={F.mean():.5f}")
# per-event extent
for k in range(3):
    seg=F[k*10000:(k+1)*10000]; act=np.nonzero(seg.any(1))[0]
    print("event",k,"init",S.NAMES[nr['log'][k]],"active span steps",(act.min(),act.max()) if act.size else None,
          "assemblies:", [S.NAMES[i] for i,al in enumerate(st['asm_local']) if seg[:,al].any(0).mean()>=0.25])
