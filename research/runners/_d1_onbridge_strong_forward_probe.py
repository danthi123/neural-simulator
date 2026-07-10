import numpy as np
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet
from sim.backend import to_host
(Xtr,ytr),(Xte,yte),nb=_load_task("emerge1",42,4)
rng=np.random.default_rng(0); tr=rng.permutation(len(Xtr))[:32]; Xtr,ytr=Xtr[tr],ytr[tr]
print("does STRONG input->hidden drive (low bias, high fwd_w) make the hidden layer INPUT-dependent?")
print(f"  {'bias':>5} {'in_hi':>6} {'fwd_w':>6} {'dens':>5} | {'hidden std':>10} {'hid mean-rate':>13}")
for bias,ih,fw,dn in ((20,1200,40,1.0),(20,1500,80,1.0),(10,1500,150,1.0),(0,1500,150,1.0),(0,1800,250,1.0),(60,1500,150,1.0)):
    net=OnBridgeBDSPNet(seed=42,n_bits=nb,hidden=12,couple_soma=True,soma_g=80.0,
                        hidden_bias=bias,output_bias=bias,in_hi=ih,fwd_wmean=fw,fwd_density=dn,bdsp_lr=0.03)
    hids=[]
    for i in range(len(Xtr)):
        net._reset_membrane(); net._set_apical(None); net.cfg.bdsp_learning_rate=0.0; net._set_input_drive(Xtr[i])
        h=0.0
        for _ in range(50):
            net.sb._run_one_simulation_step(); h+=float(np.asarray(to_host(net.sb.cp_firing_states[net.idx_hid])).mean())
        hids.append(h/50)
    hids=np.array(hids)
    print(f"  {bias:>5} {ih:>6} {fw:>6} {dn:>5} | {hids.std():>10.4f} {hids.mean():>13.4f}")
print("\nhidden std >> 0.01 = the input FINALLY differentiates the hidden layer -> the forward pass propagates (a quick fix).")
print("still ~0.003 at all strengths = deeper wiring issue (the rebuild).")
