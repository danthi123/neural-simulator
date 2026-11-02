import os, sys, copy, torch
os.environ["RWKV_V7_ON"] = "1"; os.environ["RWKV_JIT_ON"] = "1"; os.environ["RWKV_CUDA_ON"] = "0"
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
m = RWKV(model=sys.argv[1][:-4], strategy="cpu fp32"); P = PIPELINE(m, "rwkv_vocab_v20230424")
def run(text, state):
    st = copy.deepcopy(state) if state is not None else None
    return m.forward(P.encode(text), st)
def greedy(state, logits, n=14):
    st = copy.deepcopy(state); toks = []
    for _ in range(n):
        t = int(torch.argmax(logits))
        if t == 0: break
        toks.append(t); logits, st = m.forward([t], st)
    return P.decode(toks).split("\n")[0].strip()
def flat(st, base): return torch.cat([(st[i] - base[i]).flatten() for i in range(len(base)) if i % 3 == 1])
facts = [("My cat is called Pixel.", "What is my cat called?", "pixel"),
         ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
         ("I was born in Lyon.", "Where was I born?", "lyon"),
         ("My sister's name is Marguerite.", "What is my sister's name?", "marguerite"),
         ("My dog is named Biscuit.", "What is my dog's name?", "biscuit"),
         ("I drive a red Volvo.", "What car do I drive?", "volvo"),
         ("My best friend is called Tomasz.", "Who is my best friend?", "tomasz"),
         ("I work at a bakery called Crumb.", "Where do I work?", "crumb")]
_, base = run("System: You are a young digital brain talking with your friend.\n\n", None)
eps = []
for say, _, _ in facts:
    _, s = run(f"User: {say}\n\n", base)
    eps.append(([(s[i] - base[i]) if i % 3 == 1 else None for i in range(len(base))], flat(s, base)))
keys = torch.stack([e[1] for e in eps]); keys = keys / keys.norm(dim=1, keepdim=True)
filler = "User: The weather has been rainy lately.\n\nAssistant: Rain can be nice to listen to.\n\n"
for alpha in (1.5, 2.0, 3.0):
    sel_ok = hits = 0; print(f"\n=== top-1 selection, alpha {alpha}")
    for j, (_, ask, key) in enumerate(facts):
        _, sq = run(f"User: {ask}\n\n", base)           # the cue: the question as cortical activity
        q = flat(sq, base); q = q / q.norm()
        sims = keys @ q; k = int(torch.argmax(sims)); sel_ok += (k == j)
        st0 = [x.clone() for x in base]
        for i, d in enumerate(eps[k][0]):
            if d is not None: st0[i] = st0[i] + alpha * d
        lg, st = run(filler + f"User: {ask}\n\nAssistant:", st0); a = greedy(st, lg); h = key in a.lower(); hits += h
        print(f"  sel={'ok ' if k==j else 'BAD'} {'HIT ' if h else '    '}{ask} -> {a}")
    print(f"  selection {sel_ok}/8, recall {hits}/8")
