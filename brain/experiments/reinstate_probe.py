"""Does reinstating an episode's matrix-memory delta into a fresh cortex state recall the fact (no text)?"""
import os, sys, copy, torch
os.environ["RWKV_V7_ON"] = "1"; os.environ["RWKV_JIT_ON"] = "1"; os.environ["RWKV_CUDA_ON"] = "0"
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
m = RWKV(model=sys.argv[1][:-4], strategy="cpu fp32"); P = PIPELINE(m, "rwkv_vocab_v20230424")
def run(text, state):
    st = copy.deepcopy(state) if state is not None else None
    out, st = m.forward(P.encode(text), st); return out, st
def greedy(state, logits, n=12):
    st = copy.deepcopy(state); toks = []
    for _ in range(n):
        t = int(torch.argmax(logits)); 
        if t == 0: break
        toks.append(t); logits, st = m.forward([t], st)
    return P.decode(toks).split("\n")[0].strip()
facts = [("My cat is called Pixel.", "What is my cat called?", "Pixel"),
         ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
         ("I was born in Lyon.", "Where was I born?", "Lyon"),
         ("My sister's name is Marguerite.", "What is my sister's name?", "Marguerite")]
_, base = run("System: You are a young digital brain talking with your friend.\n\n", None)
filler = "User: The weather has been rainy lately.\n\nAssistant: Rain can be nice to listen to.\n\n"
for say, ask, key in facts:
    _, s0 = run(f"User: {say}\n\nAssistant: Okay, I'll remember that.\n\n", base)
    delta = [(s0[i] - base[i]) if i % 3 == 1 else None for i in range(len(base))]
    res = {}
    # control A: no memory; control B: text in context
    for name, pre in [("none", base), ("text", s0)]:
        lg, st = run(filler + f"User: {ask}\n\nAssistant:", pre); res[name] = greedy(st, lg)
    for alpha in (0.5, 1.0, 2.0):
        st0 = [x.clone() for x in base]
        for i, d in enumerate(delta):
            if d is not None: st0[i] = st0[i] + alpha * d
        lg, st = run(filler + f"User: {ask}\n\nAssistant:", st0); res[f"reinst{alpha}"] = greedy(st, lg)
    # wrong-memory control: reinstate a different fact's delta
    print(f"\n[{key}]"); [print(f"  {k:10s} {'HIT ' if key.lower() in v.lower() else '    '} {v}") for k, v in res.items()]
