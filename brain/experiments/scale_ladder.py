"""Scale ladder (11-02): same memory tests on each cortex size. In-context baseline, reinstatement, selection."""
import os, sys, time, torch
from brain.cortex import Cortex
from brain.brain import BIRTH
FACTS = [("My cat is called Pixel.", "What is my cat called?", "pixel"), ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
 ("I was born in Lyon.", "Where was I born?", "lyon"), ("My sister's name is Marguerite.", "What is my sister's name?", "marguerite"),
 ("My dog is named Biscuit.", "What is my dog's name?", "biscuit"), ("I drive a red Volvo.", "What car do I drive?", "volvo"),
 ("My best friend is called Tomasz.", "Who is my best friend?", "tomasz"), ("I work at a bakery called Crumb.", "Where do I work?", "crumb"),
 ("My brother plays the cello.", "What instrument does my brother play?", "cello"), ("I'm allergic to peanuts.", "What am I allergic to?", "peanut"),
 ("My grandmother lives in Oslo.", "Where does my grandmother live?", "oslo"), ("My son's birthday is in March.", "When is my son's birthday?", "march")]
PARA = ["What's the name of my kitty?", "Which colour do I like best?", "What's my hometown?", "Who's my sister?", "What's my dog called?",
 "What kind of car is mine?", "Who's my closest friend?", "What's my workplace?", "Which instrument does my brother play?",
 "Is there any food I can't eat?", "Which city is my grandma in?", "What month was my son born?"]
CHAT = "User: Hi, how are you?\n\nAssistant: I'm good, thanks! How about you?\n\nUser: Pretty good. It rained all day.\n\nAssistant: Rain can be cozy.\n\n"
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
