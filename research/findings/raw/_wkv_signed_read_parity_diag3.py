"""Does a FIRST-TO-FIRE (rank-order) read sharpen the decision vs integrate-rate, in the sparse regime?
Also: does a per-position adaptive lift (ramp the pool floor until the first pool fires) erase silence + keep the
correct winner? Both bypass _run to read per-step firing. No rewiring (baseline signed read, dn0 rec0)."""
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
            lg = ro.logits(ap, an, ids[t]).copy()
            if ro.unk_idx >= 0:
                lg[ro.unk_idx] = -1e30
            pf = _softmax(lg)
            ctxs.append((ap.copy(), an.copy(), ids[t], int(np.argmax(lg)), pf))
    if len(ctxs) >= 60:
        break

s = ParitySignedShadowRead(ro, SEED, pop=8, ou_std=40.0, read_window=150, hid_gain=120.0, hid_bias=0.0,
                           syn_scale=12.0, ratio=6.5, floor_pA=78.0, n_fs=48, exc_to_fs=1.2, fs_to_exc=7.0,
                           dn_size=0, dn_exc=0.6, dn_inh=0.0, rec_gain=0.0)
b = s._b; xp, _ = get_backend()

def run_firststep(ap, an, tid, floor, win=150):
    """Return per-pool total firing + per-pool first-spike step (win+1 if never)."""
    feat = s._hidden_feature(ap, an, tid)
    s._reset()
    drive = np.zeros(b.core_config.num_neurons)
    fd = s.hid_bias + s.hid_gain * feat[s.hid_dim]
    drive[s.hid_idx] = fd; drive[s.hidinh_idx] = fd
    drive[s.all_pool] += floor
    b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
    tot = np.zeros(s.V); firststep = np.full(s.V, win + 1)
    for step in range(win):
        b._run_one_simulation_step()
        fs_now = np.asarray(to_host(b.cp_firing_states)).astype(float)
        for k in range(s.V):
            c = fs_now[s.pool_idx[k]].sum()
            if c > 0:
                tot[k] += c
                if firststep[k] > win:
                    firststep[k] = step
    b.cp_external_input_current[:] = 0.0
    return tot, firststep

# compare integrate-rate vs first-to-fire winner selection at the baseline floor
rate_mass = ff_mass = hs_mass = 0.0; rate_agree = ff_agree = 0; sil = 0; n = 0
for (ap, an, tid, hostam, pf) in ctxs[:40]:
    tot, fst = run_firststep(ap, an, tid, 78.0)
    if tot.max() <= 0:
        sil += 1; n += 1; continue
    w_rate = int(np.argmax(tot))
    w_ff = int(np.argmin(fst))               # earliest first spike
    rate_mass += pf[w_rate]; ff_mass += pf[w_ff]
    rate_agree += int(w_rate == hostam); ff_agree += int(w_ff == hostam)
    n += 1
print(f"[baseline floor78] n={n} silent={sil/n:.3f}")
print(f"  integrate-rate: mass={rate_mass/n:.3f} agree={rate_agree/n:.3f}")
print(f"  first-to-fire : mass={ff_mass/n:.3f} agree={ff_agree/n:.3f}")

# adaptive lift: for silent positions, ramp the floor up and re-read; does the correct winner appear?
print("== adaptive per-position lift (ramp floor until a pool fires): recovers silent positions? ==")
rec_mass = 0.0; rec_agree = 0; still_sil = 0; nsil = 0
for (ap, an, tid, hostam, pf) in ctxs[:40]:
    tot, _ = run_firststep(ap, an, tid, 78.0)
    if tot.max() > 0:
        continue
    nsil += 1
    got = False
    for fl in (82, 86, 90, 94, 98, 104, 110):
        tot2, _ = run_firststep(ap, an, tid, fl)
        if tot2.max() > 0:
            w = int(np.argmax(tot2)); rec_mass += pf[w]; rec_agree += int(w == hostam); got = True
            break
    if not got:
        still_sil += 1
print(f"  n_silent={nsil} recovered_mass={rec_mass/max(1,nsil):.3f} recovered_agree={rec_agree/max(1,nsil):.3f} "
      f"still_silent={still_sil}")
