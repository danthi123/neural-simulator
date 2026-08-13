"""Can dn be made to FIRE + divisively sharpen, and can recurrent amplify, at a higher-firing floor?
Drives the substrate manually (bypass _run) so I can add a tonic dn floor and read dn/pool stats + sign effect."""
import sys, numpy as np
sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-aef6c0543081925cb")
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _softmax, _load_eval
from research.runners._wkv_signed_read_parity_derisk import ParitySignedShadowRead
from sim.backend import to_host, get_backend

SEED = 42
ro = WKVReadout(f"bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{SEED}.npz")
ev_ids, vocab = _load_eval(ro, "", 8000, SEED, 40)
ctxs = []
for ids in ev_ids:
    if len(ids) < 8:
        continue
    ap = np.zeros(ro.D); an = np.zeros(ro.D)
    for t in range(len(ids) - 1):
        ap, an = ro.advance(ap, an, ids[t])
        if t >= 3:
            # store host argmax too
            lg = ro.logits(ap, an, ids[t]).copy()
            if ro.unk_idx >= 0:
                lg[ro.unk_idx] = -1e30
            ctxs.append((ap.copy(), an.copy(), ids[t], int(np.argmax(lg))))
    if len(ctxs) >= 30:
        break

def build(floor, dn_inh, rec, dn_exc, win=150):
    return ParitySignedShadowRead(ro, SEED, pop=8, ou_std=40.0, read_window=win, hid_gain=120.0, hid_bias=0.0,
                                  syn_scale=12.0, ratio=6.5, floor_pA=floor, n_fs=48, exc_to_fs=1.2, fs_to_exc=7.0,
                                  dn_size=32, dn_exc=dn_exc, dn_inh=dn_inh, rec_gain=rec)

def probe(s, dn_floor, label, win=150, nctx=10):
    b = s._b; xp, _ = get_backend()
    dn_f = []; margin = []; act = []; agree = []; winspk = []
    for (ap, an, tid, hostam) in ctxs[:nctx]:
        feat = s._hidden_feature(ap, an, tid)
        s._reset()
        drive = np.zeros(b.core_config.num_neurons)
        fd = s.hid_bias + s.hid_gain * feat[s.hid_dim]
        drive[s.hid_idx] = fd; drive[s.hidinh_idx] = fd
        drive[s.all_pool] += s.floor_pA
        if dn_floor:
            drive[s.dn_idx] += dn_floor
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        firing = np.zeros(b.core_config.num_neurons)
        for _ in range(win):
            b._run_one_simulation_step()
            firing += np.asarray(to_host(b.cp_firing_states)).astype(float)
        b.cp_external_input_current[:] = 0.0
        pp = np.array([firing[s.pool_idx[k]].sum() for k in range(s.V)])
        order = np.argsort(-pp)
        dn_f.append(firing[s.dn_idx].sum()); margin.append(pp[order[0]] - pp[order[1]])
        act.append(int((pp > 0).sum())); winspk.append(pp[order[0]])
        agree.append(int(pp.max() > 0 and order[0] == hostam))
    print(f"{label:44s} dn_fire={np.mean(dn_f):8.1f} winspk={np.mean(winspk):6.1f} "
          f"margin={np.mean(margin):6.2f} active={np.mean(act):6.1f} agree={np.mean(agree):.2f}", flush=True)

print("== A: give dn a tonic floor + strong weights, does it fire + sharpen (fewer active pools)? ==")
probe(build(78, 0.0, 0.0, 0.6), 0, "floor78 dn0 (baseline)")
probe(build(78, 12.0, 0.0, 4.0), 90, "floor78 dn_inh12 dn_exc4 dn_floor90")
probe(build(78, 30.0, 0.0, 4.0), 90, "floor78 dn_inh30 dn_exc4 dn_floor90")
probe(build(86, 12.0, 0.0, 4.0), 90, "floor86 dn_inh12 dn_exc4 dn_floor90")
probe(build(86, 30.0, 0.0, 8.0), 95, "floor86 dn_inh30 dn_exc8 dn_floor95")
print("== B: at a higher floor (winner fires robustly), does recurrent amplify the margin? ==")
probe(build(86, 0.0, 0.0, 0.6), 0, "floor86 rec0")
probe(build(86, 0.0, 8.0, 0.6), 0, "floor86 rec8")
probe(build(86, 0.0, 20.0, 0.6), 0, "floor86 rec20")
probe(build(92, 0.0, 20.0, 0.6), 0, "floor92 rec20")
