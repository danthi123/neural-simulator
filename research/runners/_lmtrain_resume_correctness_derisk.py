"""De-risk #1 (load-bearing) for the autonomous incremental LM-training workflow: does checkpoint->resume continue the
EXACT training trajectory? If yes, "build on checkpoints incrementally as compute allows" is sound. Tiny recurrent LM
(WKV-like diagonal-decay SSM), CPU, deterministic. Saves/restores {model, optimizer, torch+numpy+python RNG, step};
compares an interrupted (10+resume+10) run to an uninterrupted 20-step run."""
import copy, random
import numpy as np
import torch


def set_all_rng(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


class TinyRecLM(torch.nn.Module):                        # a WKV-like diagonal-decay SSM (recurrent, like the real model)
    def __init__(self, V=64, D=32):
        super().__init__()
        self.emb = torch.nn.Embedding(V, D); self.Wv = torch.nn.Linear(D, D, bias=False)
        self.Wo = torch.nn.Linear(D, D, bias=False); self.head = torch.nn.Linear(D, V)
        self.decay = torch.nn.Parameter(torch.zeros(D))

    def forward(self, x):
        h = self.emb(x); v = self.Wv(h); lam = torch.sigmoid(self.decay)
        a = torch.zeros(x.shape[0], h.shape[-1]); outs = []
        for t in range(x.shape[1]):
            a = lam * a + (1 - lam) * v[:, t]; outs.append(self.Wo(a))
        return self.head(torch.stack(outs, 1))


def make_data(seed, n=200, T=16, V=64):
    return np.random.default_rng(seed).integers(0, V, size=(n, T))


def train_steps(model, opt, data, steps, batch=8):
    losses = []; lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        idx = torch.randint(0, len(data), (batch,))       # batch selection via torch RNG (the "data cursor", resumable)
        X = torch.tensor(data[idx.numpy()], dtype=torch.long)
        logits = model(X)[:, :-1]; tgt = X[:, 1:]
        loss = lossf(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    return losses


data = make_data(0)
# uninterrupted 20-step reference
set_all_rng(42); m1 = TinyRecLM(); o1 = torch.optim.AdamW(m1.parameters(), lr=1e-2)
un = train_steps(m1, o1, data, 20)

# interrupted: 10 steps -> checkpoint EVERYTHING -> NEW model+opt (fresh "process") -> restore -> 10 more
set_all_rng(42); m2 = TinyRecLM(); o2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
_ = train_steps(m2, o2, data, 10)
ckpt = {"model": copy.deepcopy(m2.state_dict()), "opt": copy.deepcopy(o2.state_dict()),
        "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate(), "step": 10}
m3 = TinyRecLM(); o3 = torch.optim.AdamW(m3.parameters(), lr=1e-2)   # fresh objects = simulate a restarted process
m3.load_state_dict(ckpt["model"]); o3.load_state_dict(ckpt["opt"])
torch.set_rng_state(ckpt["torch_rng"]); np.random.set_state(ckpt["np_rng"]); random.setstate(ckpt["py_rng"])
resumed = train_steps(m3, o3, data, 10)

un_second = un[10:]
max_diff = max(abs(a - b) for a, b in zip(un_second, resumed))
print(f"uninterrupted steps 11-20 loss: {[round(x,4) for x in un_second]}")
print(f"resumed      steps 11-20 loss: {[round(x,4) for x in resumed]}")
print(f"max |loss diff| = {max_diff:.2e}")
print("RESUME-CORRECT (bit-close, checkpoint->resume == uninterrupted)" if max_diff < 1e-5
      else f"RESUME-BROKEN (diff {max_diff:.2e} -- an RNG/optimizer/cursor state is not restored)")
