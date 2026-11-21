# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: none (an instrument)
"""The language guard (design section 5): KL and NLL of the adapters-on cortex against the pretrained cortex on a fresh
held-out ordinary-text set, read after every trunk night by `brain.eval.life --lang-guard`
(the read-out code is brain/eval/life_langguard.py; this file is data only so its hash never moves with a code fix).

HELDOUT is written by the teacher (Claude), frozen by life_heldout.sha256 (verified at run time), and is NOT any of the
3 brain.ORDINARY snippets the sleep replay holds to the pretrained cortex, nor any near-miss / rehearsal guard text, and
holds no fact, name or key of life_bank.py / life_bank_test.py. It is never used to fit anything. Bar: night-15 KL <= 2x
night 1; per-seed kill at KL > 3x night 1.
"""
from __future__ import annotations

import hashlib
import os

HASH_FILE = os.path.join(os.path.dirname(__file__), "life_heldout.sha256")

HELDOUT = [
    "User: Will it be warm tomorrow?\n\nAssistant: I cannot check the sky from here, but this season is often mild. A light jacket is a safe bet.\n\n"
    "User: Good idea. I want to walk by the river.\n\nAssistant: That sounds lovely. Bring some water.\n\n",
    "User: How long should I cook pasta so it keeps a firm bite?\n\nAssistant: About nine minutes in salted water, then drain it straight away.\n\n"
    "User: And for a softer texture?\n\nAssistant: Around a quarter of an hour should do it.\n\n",
    "User: Could you tell me how to get to the museum from here?\n\nAssistant: Go straight along this avenue, turn left at the traffic signals, and you will see it on the right.\n\n"
    "User: Is it far?\n\nAssistant: About a ten minute walk.\n\n",
    "User: I watched a really slow nature programme last night.\n\nAssistant: Slow programmes can be rewarding if you are in the mood for them. Did it hold your attention?\n\n"
    "User: Mostly, until the last half hour.\n\nAssistant: Endings can make or break a story.\n\n",
    "User: Tell me a joke.\n\nAssistant: Why did the scarecrow win an award? Because he was outstanding in his field.\n\n"
    "User: That was terrible.\n\nAssistant: I know, but you did ask for one.\n\n",
    "User: Can you help me with my maths homework?\n\nAssistant: Of course. What is the problem?\n\n"
    "User: I need to work out half of sixty.\n\nAssistant: Divide sixty by two. The answer is thirty.\n\n",
    "User: What is the difference between a river and a stream?\n\nAssistant: A river is usually larger and deeper, while a stream is small enough that sunlight reaches the bottom.\n\n"
    "User: So there is no exact rule?\n\nAssistant: Not really. People use the words loosely.\n\n",
    "User: I can never sleep before an exam.\n\nAssistant: That is common. A short walk and a quiet evening can help.\n\n"
    "User: I will try that tonight.\n\nAssistant: Good luck. Try to rest, even if you do not sleep much.\n\n",
    "User: How do I make my pancakes fluffier?\n\nAssistant: Mix the batter gently, and allow it to rest for a few minutes before cooking.\n\n"
    "User: How long should it rest?\n\nAssistant: Ten minutes is plenty.\n\n",
    "User: My tram is late again.\n\nAssistant: That is frustrating. Is there another route you could take?\n\n"
    "User: Not really, so I will just wait.\n\nAssistant: At least the air is pleasant today.\n\n",
    "User: What should I pack for a short visit to the coast?\n\nAssistant: Good walking trainers, sunscreen and a warm layer for the evening.\n\n"
    "User: Do I need something waterproof?\n\nAssistant: It is worth bringing a light one, just in case.\n\n",
    "User: Why is the ocean salty?\n\nAssistant: Rivers carry tiny amounts of minerals into it, and when the water evaporates the salt stays behind.\n\n"
    "User: Does it ever get less salty?\n\nAssistant: Not noticeably. New minerals keep arriving.\n\n",
    "User: I want to begin jogging in the mornings.\n\nAssistant: Begin slowly, with a mix of jogging and walking, and add a little each week.\n\n"
    "User: How often should I go?\n\nAssistant: Every other day is a fine pace.\n\n",
    "User: How do I get a stuck jar lid open?\n\nAssistant: Hold the lid in hot water for a minute, then grip it with a rubber glove.\n\n"
    "User: And if that fails?\n\nAssistant: Knock the edge gently on the counter and try again.\n\n",
    "User: The stew tastes bland.\n\nAssistant: Try a pinch of salt, a squeeze of lemon, or a bit of pepper.\n\n"
    "User: Lemon sounds good.\n\nAssistant: Add it at the end so the taste stays fresh.\n\n",
    "User: I am attempting to tidy my shelf but it keeps getting messy.\n\nAssistant: Give every item a spot, and return things when you finish with them.\n\n"
    "User: That sounds simple enough.\n\nAssistant: It is simple, though the habit takes a few weeks to build.\n\n",
    "User: Why does the water at the beach go in and out?\n\nAssistant: The ocean rises and drops each day, caused mainly by the pull of the moon.\n\n"
    "User: How many times a day does it happen?\n\nAssistant: Most coasts have two rises and two drops each day.\n\n",
    "User: I am bored.\n\nAssistant: Would you like a puzzle, a story, or a riddle?\n\n"
    "User: A puzzle, please.\n\nAssistant: What has hands but cannot clap? A clock.\n\n",
    "User: My tablet battery drains so fast.\n\nAssistant: Lower the screen brightness and close apps you are not using. Aged cells also wear out.\n\n"
    "User: It is six years old.\n\nAssistant: Then a new battery might be worth considering.\n\n",
    "User: How do I stop my glasses from fogging up?\n\nAssistant: A thin smear of washing-up liquid, wiped off gently, helps. A mask that fits snugly also helps.\n\n"
    "User: I will try that.\n\nAssistant: Use a soft cloth so you do not scratch the lenses.\n\n",
    "User: Which snack suits an evening lesson?\n\nAssistant: Something light, like yoghurt with fruit or a handful of nuts.\n\n"
    "User: I usually skip it.\n\nAssistant: A small snack can keep you focused for longer.\n\n",
    "User: Should I water my plants at dawn or at bedtime?\n\nAssistant: Dawn is usually best, because the water has time to soak in before midday.\n\n"
    "User: And in the chilly months?\n\nAssistant: Water less often, since the soil dries out more slowly.\n\n",
    "User: Could you suggest a quiet hobby?\n\nAssistant: Sewing, drawing and birdwatching are all peaceful and easy to begin.\n\n"
    "User: Birdwatching sounds fun.\n\nAssistant: All you need is a window, some patience and a notebook.\n\n",
    "User: How do you spell separate?\n\nAssistant: S-E-P-A-R-A-T-E. There is a rat in the middle.\n\n"
    "User: That is a handy trick.\n\nAssistant: Memory tricks like that make tricky words easier.\n\n",
]
assert len(HELDOUT) == 24


def sha() -> str:
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def verify() -> str:
    """Fail closed: this file must hash to life_heldout.sha256 (the set is frozen once --lang-guard runs on it)."""
    want = open(HASH_FILE).read().split()[0]
    got = sha()
    if got != want:
        raise RuntimeError(f"held-out language set changed: sha256 {got[:12]} != recorded {want[:12]} ({HASH_FILE})")
    return got
