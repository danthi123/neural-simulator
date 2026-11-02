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
facts = [("My cat is called Pixel.", "What is my cat called?", "pixel"),
         ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
         ("I was born in Lyon.", "Where was I born?", "lyon"),
         ("My sister's name is Marguerite.", "What is my sister's name?", "marguerite"),
         ("My dog is named Biscuit.", "What is my dog's name?", "biscuit"),
         ("I drive a red Volvo.", "What car do I drive?", "volvo"),
         ("My best friend is called Tomasz.", "Who is my best friend?", "tomasz"),
         ("I work at a bakery called Crumb.", "Where do I work?", "crumb")]
unrelated = ["What do you like to do on weekends?", "What is two plus two?", "Can you tell me about the ocean?"]
_, base = run("System: You are a young digital brain talking with your friend.\n\n", None)
deltas = []
for say, _, _ in facts:
    _, s = run(f"User: {say}\n\nAssistant: Okay, I'll remember that.\n\n", base)
    deltas.append([(s[i] - base[i]) if i % 3 == 1 else None for i in range(len(base))])
filler = "User: The weather has been rainy lately.\n\nAssistant: Rain can be nice to listen to.\n\n"
for alpha in (1.0, 2.0, 3.0):
    st0 = [x.clone() for x in base]
    for d in deltas:
        for i, x in enumerate(d):
            if x is not None: st0[i] = st0[i] + alpha * x
    hits = 0; print(f"\n=== all {len(facts)} memories at alpha {alpha}")
    for _, ask, key in facts:
        lg, st = run(filler + f"User: {ask}\n\nAssistant:", st0); a = greedy(st, lg); h = key in a.lower(); hits += h
        print(f"  {'HIT ' if h else '    '}{ask} -> {a}")
    print(f"  recall {hits}/{len(facts)}")
    for q in unrelated:
        lg, st = run(filler + f"User: {q}\n\nAssistant:", st0); print(f"  [unrelated] {q} -> {greedy(st, lg, 20)}")
