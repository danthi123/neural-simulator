"""Scale ladder (11-02): same memory tests on each cortex size. In-context baseline, reinstatement, selection."""
import os, sys, time, torch
from brain.cortex import Cortex
from brain.brain import BIRTH
from brain.experiments.facts import CHAT, FACTS, PARA
path = sys.argv[1]; t0 = time.time(); c = Cortex(model_path=path); print("MODEL", os.path.basename(path), flush=True)
_, base = c.read(BIRTH, None); _, conv = c.read(CHAT, base)
def ask(st, q):
    lg, s = c.read(f"User: {q}\n\nAssistant:", st); return c.speak(lg, s, max_tokens=20, temperature=0)[0]
# 1) in-context: fact told earlier in the same conversation, then two chat turns, then the question
hit = 0
for say, q, k in FACTS:
    _, s = c.read(f"User: {say}\n\nAssistant: Okay!\n\n" + CHAT, base); hit += k in ask(s, q).lower()
print(f"in-context recall {hit}/12", flush=True)
D = [c.matrix_delta(c.read(f"User: {s}\n\n", base)[1], base) for s, _, _ in FACTS]
for alpha in (1.5, 2.5, 4.0):
    hit = sum(k in ask(c.add_delta(conv, d, alpha), q).lower() for d, (_, q, k) in zip(D, FACTS))
    print(f"reinstated (right memory) gain {alpha}: {hit}/12", flush=True)
keys = torch.stack([c.key_of(d) for d in D]); mu = keys.mean(0); kn = (keys - mu); kn = kn / kn.norm(dim=1, keepdim=True)
for name, qs in (("same-words cue", [q for _, q, _ in FACTS]), ("paraphrase cue", PARA)):
    ok = 0
    for j, q in enumerate(qs):
        v = c.key_of(c.matrix_delta(c.read(f"User: {q}\n\n", base)[1], base)) - mu; ok += int(torch.argmax(kn @ (v / v.norm()))) == j
    print(f"selection {name}: {ok}/12", flush=True)
print(f"elapsed {time.time()-t0:.0f}s", flush=True)
