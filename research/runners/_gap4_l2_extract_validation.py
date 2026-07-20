import sys; sys.path.insert(0, "/home/dant123/Projects/sim")
import os; os.environ.setdefault("SIM_BACKEND","numpy")
import numpy as np
import research.runners._gap4_btsp_rung3_downstream_read_derisk as m
from sim.backend import to_host
m.set_map_density(4)
sb, pos, cells, l2 = m.build(200, l2_w0=150.0)
print("cfg.btsp_w_max =", sb.core_config.btsp_w_max, " (expected max(5, 2*150)=300)")
W = sb.cp_connections
dat=np.asarray(to_host(W.data)); ind=np.asarray(to_host(W.indices)); ptr=np.asarray(to_host(W.indptr))
print("CSR shape:", W.shape, "nnz:", len(dat))
# how many entries does my extraction find per cell? expected = CA1_PER_CELL * L2_N = 8*8 = 64
tgt=set(int(i) for i in l2)
for c in range(m.N_CELL):
    src=set(int(i) for i in cells[c]); n=0; vals=[]
    for r in range(len(ptr)-1):
        if r not in src: continue
        for k in range(ptr[r], ptr[r+1]):
            if int(ind[k]) in tgt: n+=1; vals.append(abs(float(dat[k])))
    print(f"  cell {c}: extracted {n} synapses (expect {m.CA1_PER_CELL*m.L2_N}) mean|w|={np.mean(vals) if vals else 0:.4f}")
# Is CSR row=pre or row=post? check a pos->ca1 pathway: pos has 10 neurons, ca1 cell has 8
src=set(int(i) for i in pos[0]); tgt2=set(int(i) for i in cells[0]); n=0
for r in range(len(ptr)-1):
    if r not in src: continue
    for k in range(ptr[r], ptr[r+1]):
        if int(ind[k]) in tgt2: n+=1
print(f"pos0->ca1_0 via row=pre: found {n} (expect {m.POS_N*m.CA1_PER_CELL}=80)")
n=0
for r in range(len(ptr)-1):
    if r not in tgt2: continue
    for k in range(ptr[r], ptr[r+1]):
        if int(ind[k]) in src: n+=1
print(f"pos0->ca1_0 via row=post: found {n} (expect 0 if row=pre)")
