"""A9 instrument diagnostic (dev seed 7, no gate reads it): is the BDSP event-rate read E monotonic in drive?
Adds a constant extra current to the OUTPUT slice of a frozen (reservoir) net and measures, per output neuron over a
100-step window: total spikes, events (isolated/first-of-burst: E increments) and burst spikes."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from research.runners._gap4_transport_ceiling_readout_derisk import Gap4ReadoutNet
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance
from sim.backend import to_host

(Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(7)
k = int(meta["k_classes"])
net = Gap4ReadoutNet(Xtr.shape[1], 32, k, seed=7, feedback="reservoir", n_hidden_layers=2, pool_k=4,
                     settle_steps=40, credit_steps=25, graded_credit=True)
net.cfg.bdsp_learning_rate = 0.0
net.cfg.enable_structural_plasticity = False
xp = net._xp
sl_out = net.slices[-1]; sl_h2 = net.slices[-2]
rows = []
for extra in [-400, -200, 0, 200, 400, 800, 1200, 1600]:
    tot = ev = bu = 0.0; Esum = 0.0; tot_h2 = ev_h2 = 0.0
    for i in range(6):
        f = np.asarray(Xtr[i], np.float32)
        drive = net._base_drive()
        drive[net.slices[0]] = net._broadcast(np.clip(net.in_bias_pA + net.in_current_pA * f, 0, 1600), 0)
        drive[sl_out] += extra
        net.br.cp_external_input_current = xp.asarray(drive.astype(np.float32))
        if net.br.cp_bdsp_E is not None:
            net.br.cp_bdsp_E[...] = 0.0; net.br.cp_bdsp_B[...] = 0.0
        for s in range(140):
            prev_last = None if net.br.cp_bdsp_last_spike_step is None else np.asarray(to_host(net.br.cp_bdsp_last_spike_step)).copy()
            net.br._run_one_simulation_step()
            if s < 40:
                continue
            last = np.asarray(to_host(net.br.cp_bdsp_last_spike_step))
            fired = last == net.br._bdsp_step_counter
            if prev_last is None:
                continue
            d = net.br._bdsp_step_counter - prev_last
            burst = fired & (d <= 6) & (prev_last >= 0)
            tot += fired[sl_out].sum(); bu += burst[sl_out].sum(); ev += (fired & ~burst)[sl_out].sum()
            tot_h2 += fired[sl_h2].sum(); ev_h2 += (fired & ~burst)[sl_h2].sum()
            Esum += float(np.asarray(to_host(net.br.cp_bdsp_E))[sl_out].mean())
    n_out = sl_out.stop - sl_out.start; n_h2 = sl_h2.stop - sl_h2.start; T = 6 * 100
    r = {"extra_pA_out": extra, "out_total_hz": 1e3 * tot / (n_out * T), "out_event_hz": 1e3 * ev / (n_out * T),
         "out_burst_spike_hz": 1e3 * bu / (n_out * T), "out_mean_E": Esum / T,
         "h2_total_hz": 1e3 * tot_h2 / (n_h2 * T), "h2_event_hz": 1e3 * ev_h2 / (n_h2 * T)}
    rows.append(r)
    print(json.dumps({kk: round(v, 4) for kk, v in r.items()}), flush=True)
json.dump({"probe": "a9_eread_monotonicity_dev_s7", "seed": 7, "rows": rows,
           "note": "dev diagnostic only; host counts of spikes for the instrument, no pathway"},
          open("research/findings/raw/gap4/transport_ceiling_readout/diag_eread_monotonic_s7.json", "w"), indent=2)
