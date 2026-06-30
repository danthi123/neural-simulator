"""Measure the held-out ppl of the live 100M C2 checkpoint, replicating the
trainer's exact in-loop probe (24 random block windows, generator seed=seed+777).
CPU-only so it doesn't contend with the live GPU training. Read-only on the ckpt."""
import sys, os, math
sys.path.insert(0, os.getcwd())
import torch, torch.nn.functional as F, numpy as np
from sim.tiny_transformer import TinyGPT

RUN = 'research/findings/raw/c2_scaleup_100M'
ck = torch.load(RUN + '/genf.ckpt.pt', map_location='cpu', weights_only=True)
V, bs = 2049, 512
m = TinyGPT(vocab_size=V, d_model=768, n_layer=12, n_head=12, block_size=bs)
m.load_state_dict(ck['model']); m.eval()
ho = torch.tensor(np.load(RUN + '/genf.bpe.json.heldtokens.npy').astype('int64'))
nho = ho.numel(); gg = torch.Generator().manual_seed(42 + 777)
tot_nll = 0.0; tot = 0
with torch.no_grad():
    for _ in range(24):
        i = int(torch.randint(0, nho - bs - 1, (1,), generator=gg).item())
        xb = ho[i:i + bs][None, :]; yb = ho[i + 1:i + 1 + bs][None, :]
        lg = m(xb)
        nll = F.cross_entropy(lg.reshape(-1, V), yb.reshape(-1), reduction='sum')
        tot_nll += float(nll); tot += yb.numel()
print('=== ckpt step', ck['step'], '/ 450000 ===')
print('HELD-OUT ppl (24 windows, == trainer probe) =', round(math.exp(tot_nll / tot), 3))
lh = ck['loss_history']
print('train loss curve (downsampled %d entries):' % len(lh),
      [round(float(x), 3) for x in lh[::max(1, len(lh) // 12)]])
