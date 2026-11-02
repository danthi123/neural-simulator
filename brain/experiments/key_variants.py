import os, sys, copy, torch, itertools
os.environ["RWKV_V7_ON"] = "1"; os.environ["RWKV_JIT_ON"] = "1"; os.environ["RWKV_CUDA_ON"] = "0"
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
m = RWKV(model=sys.argv[1][:-4], strategy="cpu fp32"); P = PIPELINE(m, "rwkv_vocab_v20230424")
def run(text, state): return m.forward(P.encode(text), copy.deepcopy(state))
facts = [("My cat is called Pixel.", "What is my cat called?"), ("My favourite colour is teal.", "What is my favourite colour?"),
 ("I was born in Lyon.", "Where was I born?"), ("My sister's name is Marguerite.", "What is my sister's name?"),
 ("My dog is named Biscuit.", "What is my dog's name?"), ("I drive a red Volvo.", "What car do I drive?"),
 ("My best friend is called Tomasz.", "Who is my best friend?"), ("I work at a bakery called Crumb.", "Where do I work?"),
 ("My brother plays the cello.", "What instrument does my brother play?"), ("I'm allergic to peanuts.", "What am I allergic to?"),
 ("My grandmother lives in Oslo.", "Where does my grandmother live?"), ("My son's birthday is in March.", "When is my son's birthday?")]
_, base = run("System: You are a young digital brain talking with your friend.\n\n", None)
L = len(base) // 3
def comps(st):
    return {f"S{l}": (st[l*3+1]-base[l*3+1]).flatten() for l in range(L)} | {f"x{l}": st[l*3+0] for l in range(L)} | {f"f{l}": st[l*3+2] for l in range(L)}
E = [comps(run(f"User: {s}\n\n", base)[1]) for s, _ in facts]
Q = [comps(run(f"User: {q}\n\n", base)[1]) for _, q in facts]
def acc(names):
    k = torch.stack([torch.cat([e[n] for n in names]) for e in E]); q = torch.stack([torch.cat([x[n] for n in names]) for x in Q])
    k = k - k.mean(0); q = q - q.mean(0)              # remove the shared "a statement/question about me" component
    k = k / k.norm(dim=1, keepdim=True); q = q / q.norm(dim=1, keepdim=True)
    sims = q @ k.T; pred = sims.argmax(1); return (pred == torch.arange(len(facts))).float().mean().item(), sims
res = []
for kind in "Sxf":
    for l in range(L): res.append((acc([f"{kind}{l}"])[0], f"{kind}{l}"))
    res.append((acc([f"{kind}{l}" for l in range(L)])[0], f"{kind}-all"))
res.append((acc([f"S{l}" for l in range(L)] + [f"x{l}" for l in range(L)])[0], "S+x all"))
for a, n in sorted(res, reverse=True)[:12]: print(f"{a:.2f}  {n}")
print("--- query centred by the STORED episodes' mean only (no peeking at other questions)")
def acc2(names):
    k = torch.stack([torch.cat([e[n] for n in names]) for e in E]); q = torch.stack([torch.cat([x[n] for n in names]) for x in Q])
    mu = k.mean(0); k = k - mu; q = q - mu
    k = k / k.norm(dim=1, keepdim=True); q = q / q.norm(dim=1, keepdim=True)
    sims = q @ k.T; top = sims.topk(2, dim=1)
    return (sims.argmax(1) == torch.arange(len(facts))).float().mean().item(), top
a, top = acc2([f"S{l}" for l in range(L)]); print("S-all", a)
print("top1 sim, margin to top2:", [(round(v[0].item(),2), round((v[0]-v[1]).item(),2)) for v in top.values])
for l in (0,3,5,8): print(f"S{l}", acc2([f"S{l}"])[0])
