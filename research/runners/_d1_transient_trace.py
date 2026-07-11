"""a0 check of the leading hypothesis: during a per-sample teach, does the burst rate B LAG the event rate E, so the
credit dev = B - Pbar*E starts NEGATIVE (spurious LTD on every synapse) and only turns positive late? Drive ONE sample's
teach (apical ON for the target class) and print E / B / Pbar / dev at the target output pool across the teach window.
Run: SIM_BACKEND=numpy python -m research.runners._d1_transient_trace
"""
import numpy as np
from sim.backend import to_host, from_host
from research.runners._d1_onbridge_learn_to_accuracy_derisk import _load_task, OnBridgeBDSPNet

SEED = 42
(Xtr, ytr), _, n_bits = _load_task("emerge1", SEED, 4)
Xtr = np.asarray(Xtr); ytr = np.asarray(ytr)
# a class-1 example (so the target is output pool 1; error e_1 = 1 - p_1 > 0 -> apical UP on pool 1)
i1 = np.where(ytr == 1)[0][0]
x = Xtr[i1]

net = OnBridgeBDSPNet(seed=SEED, n_bits=n_bits, hidden=60, couple_soma=True, soma_g=500.0,
                      hidden_bias=20.0, output_bias=20.0, bdsp_lr=0.0, fwd_wmean=40.0, bdsp_w_max=200.0)  # lr 0: trace only
tgt = net.class_idx[1]          # target output pool (class 1)
oth = net.class_idx[0]          # non-target output pool (class 0)

# emulate train_epoch's teach setup: reset, drive input, apical = strong + on the target class, learning frozen (trace).
net._reset_membrane()
net._set_input_drive(x)
net._set_apical(np.array([-0.5, 0.9]))   # a representative top error: pool1 (target) positive, pool0 negative
net.cfg.bdsp_learning_rate = 0.0

print(f"[trace] one class-1 teach window; target=pool1 (apical +), other=pool0 (apical -). want: does dev flip sign?")
print(f"{'step':>4} | {'E_tgt':>7} {'B_tgt':>7} {'Pbar_t':>7} {'dev_tgt':>9} | {'E_oth':>7} {'B_oth':>7} {'dev_oth':>9}")
for step in range(60):
    net.sb._run_one_simulation_step()
    def rd(a, ix):
        return float(np.asarray(to_host(a[ix])).mean())
    Et, Bt = rd(net.sb.cp_bdsp_E, tgt), rd(net.sb.cp_bdsp_B, tgt)
    Pt = rd(net.sb.cp_bdsp_Pbar, tgt)
    Eo, Bo = rd(net.sb.cp_bdsp_E, oth), rd(net.sb.cp_bdsp_B, oth)
    Po = rd(net.sb.cp_bdsp_Pbar, oth)
    dev_t = Bt - Pt * Et
    dev_o = Bo - Po * Eo
    if step < 12 or step % 6 == 0:
        print(f"{step:>4} | {Et:>7.4f} {Bt:>7.4f} {Pt:>7.4f} {dev_t:>9.5f} | {Eo:>7.4f} {Bo:>7.4f} {dev_o:>9.5f}")

print("\n[trace] READ: if dev_tgt is NEGATIVE for the first several steps then turns POSITIVE, and the kernel applies "
      "dw every step, the early spurious LTD contaminates the credit -> the transient hypothesis is CONFIRMED.")
