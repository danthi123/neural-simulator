"""Grid: reinstatement gain x layers x decoding, on 8 taught facts asked mid-conversation (selection forced correct)."""
import itertools, re, torch
from brain.cortex import Cortex
from brain.brain import BIRTH
FACTS = [("My cat is called Pixel.", "What is my cat called?", "pixel"), ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
 ("I was born in Lyon.", "Where was I born?", "lyon"), ("My sister's name is Marguerite.", "What is my sister's name?", "marguerite"),
 ("My dog is named Biscuit.", "What is my dog's name?", "biscuit"), ("I drive a red Volvo.", "What car do I drive?", "volvo"),
 ("My best friend is called Tomasz.", "Who is my best friend?", "tomasz"), ("I work at a bakery called Crumb.", "Where do I work?", "crumb")]
CHAT = "User: Hi, how are you?\n\nAssistant: I'm good, thanks! How about you?\n\nUser: Pretty good. It rained all day.\n\nAssistant: Rain can be cozy.\n\n"
c = Cortex(); _, base = c.read(BIRTH, None); _, conv = c.read(CHAT, base)
deltas = [c.matrix_delta(c.read(f"User: {s}\n\n", base)[1], base) for s, _, _ in FACTS]
def layers_mask(d, keep): return [x if (x is not None and (i // 3) in keep) else None for i, x in enumerate(d)]
L = c.n_layers
for alpha, (lname, keep), temp in itertools.product((2.5, 3.5, 4.5), (("all", range(L)), ("low", range(L // 2)), ("high", range(L // 2, L))), (0.0, 0.7)):
    hit = rep = 0
    for d, (_, q, key) in zip(deltas, FACTS):
        st = c.add_delta(conv, layers_mask(d, set(keep)), alpha)
        lg, st = c.read(f"User: {q}\n\nAssistant:", st)
        a, _ = c.speak(lg, st, max_tokens=24, temperature=temp, top_p=0.3)
        hit += key in a.lower(); words = re.findall(r"\w+", a.lower()); rep += len(words) > 3 and len(set(words)) < len(words) / 2
    print(f"alpha {alpha} layers {lname:4s} temp {temp}: recall {hit}/8  degenerate {rep}/8")
