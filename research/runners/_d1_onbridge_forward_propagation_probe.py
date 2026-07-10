import numpy as np
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet
(Xtr,ytr),(Xte,yte),nb=_load_task("emerge1",42,4)
rng=np.random.default_rng(0); tr=rng.permutation(len(Xtr))[:48]; Xtr,ytr=Xtr[tr],ytr[tr]
print("does the on-bridge forward readout depend on the INPUT at all? (std>0 across inputs = input propagates)")
print(f"  {'out_bias':>8} {'in_hi':>6} {'fwd_w':>6} | {'hidden std':>10} {'pool0 std':>9} {'pool1 std':>9} {'frac_argmax0':>12}")
for ob, ih, fw in ((350,750,6), (150,750,6), (80,900,10), (40,900,14), (150,900,14)):
    net=OnBridgeBDSPNet(seed=42,n_bits=nb,hidden=12,couple_soma=True,soma_g=80.0,
                        hidden_bias=ob,output_bias=ob,in_hi=ih,fwd_wmean=fw,bdsp_lr=0.03)
    # hidden firing variance across inputs
    hid=[]; out=[]
    from sim.backend import to_host
    for i in range(len(Xtr)):
        net._reset_membrane(); net._set_apical(None); net.cfg.bdsp_learning_rate=0.0; net._set_input_drive(Xtr[i])
        h=0.0
        for _ in range(50):
            net.sb._run_one_simulation_step(); h+=float(np.asarray(to_host(net.sb.cp_firing_states[net.idx_hid])).mean())
        r=net._readout(Xtr[i],50); hid.append(h/50); out.append(r)
    out=np.array(hid_out:=out); hid=np.array(hid)
    f0=(out[:,0]>out[:,1]).mean()
    print(f"  {ob:>8} {ih:>6} {fw:>6} | {hid.std():>10.4f} {out[:,0].std():>9.2f} {out[:,1].std():>9.2f} {f0:>12.3f}")
print("\n>0 std on hidden/pool = the input reaches that layer. frac_argmax0 in (0,1) = the readout differentiates inputs.")
