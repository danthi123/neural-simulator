import numpy as np
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet
from sim.backend import to_host
(Xtr,ytr),(Xte,yte),nb=_load_task("emerge1",42,4)
rng=np.random.default_rng(0); tr=rng.permutation(len(Xtr))[:16]; Xtr=Xtr[tr]
net=OnBridgeBDSPNet(seed=42,n_bits=nb,hidden=100,couple_soma=True,soma_g=80.0,
                    hidden_bias=40.0,output_bias=40.0,in_hi=1200.0,fwd_wmean=80.0,fwd_density=1.0,bdsp_lr=0.03)
print(f"input pool: {len(net.idx_in)} neurons | hidden: {len(net.idx_hid)} | out: {len(net.idx_out)}")
# does the INPUT pool fire, and input-dependently?
inrates=[]; hidrates=[]
for i in range(len(Xtr)):
    net._reset_membrane(); net._set_apical(None); net.cfg.bdsp_learning_rate=0.0; net._set_input_drive(Xtr[i])
    ir=hr=0.0
    for _ in range(50):
        net.sb._run_one_simulation_step()
        ir+=float(np.asarray(to_host(net.sb.cp_firing_states[net.idx_in])).mean())
        hr+=float(np.asarray(to_host(net.sb.cp_firing_states[net.idx_hid])).mean())
    inrates.append(ir/50); hidrates.append(hr/50)
inrates=np.array(inrates); hidrates=np.array(hidrates)
print(f"INPUT pool  firing rate: mean={inrates.mean():.4f} std={inrates.std():.4f}  (std>0 = input encoding differentiates)")
print(f"HIDDEN pool firing rate (hidden=100, fwd_w=80): mean={hidrates.mean():.4f} std={hidrates.std():.4f}")
# check the input->hidden pathway exists + its weight
import scipy.sparse as sp
w=np.asarray(to_host(net.sb.cp_connections.data))
print(f"total synapses: {len(w)}  weight mean={w.mean():.2f} max={w.max():.2f}")
print("\nif INPUT std>0 but HIDDEN std~0 -> propagation fails (the fix target). if INPUT std~0 -> input encoding is the bug.")
