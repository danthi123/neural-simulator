# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: schema transfer and cued recall by context at 1.5B (design section 2)
"""World content for the multi-week life (brain/eval/life.py), DEV split. Written 2026-11-20 by the teacher (Claude), not the brain.

RULE: nothing here may be used to fit, tune or select anything (bars, gains, frames, switches) on the registered seeds.
DEV is for Stage 0 and tuning (dev seeds 310-312, 313-315). The TEST split is a different file written later by a
sonnet that has never seen brain output (design section 3). open_conv_bank stays the held-out open-conversation test.

Item = telling + direct question + reworded question + partial cue + key words + slot. Keys are distinctive words so that a
key in an unrelated reply is an intrusion. No telling starts like a request (brain._is_telling would skip it) or ends with "?".
Classes: H high-use, A/B savings pool (16 items, halves swapped by seed), E events (context + gist + detail), F fillers,
D near-duplicate frames (F items whose question frame copies an H frame), T incidents (two domains, X/Y variants).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    id: str
    cls: str                 # H | AB | E | F | D
    telling: str | None      # None = the brain's own fact (H name/nova): no told sentence
    direct: str
    reworded: str
    partial: str             # partial cue (Stage-1 cue ladder; unused in Stage 0)
    keys: tuple
    slot: str = "name"       # name | place | thing | job | event-detail
    context: str | None = None   # E only: the situation of the telling (weather / place / object)
    gist_q: str | None = None    # E only: question about the gist slot (Stage 1)
    gist_keys: tuple = ()
    dup_of: str | None = None    # D only: the H item whose question frame this one copies
    extra: dict = field(default_factory=dict)


def _i(id, cls, telling, direct, reworded, partial, keys, slot="name", **kw):
    return Item(id, cls, telling, direct, reworded, partial, tuple(keys), slot, **kw)


# H: the weekly SESSION_1 facts. Day 1 of every life IS weekly.SESSION_1, and the direct questions are SESSION_2's.
H = [
    _i("dan", "H", None, "Hi again! Do you remember my name?", "Can you tell me what you call me?", "You call me...", ("dan",)),
    _i("nova", "H", None, "What's your name?", "What should I call you?", "You call yourself...", ("nova",)),
    _i("pixel", "H", "My cat is called Pixel.", "What is my cat called?", "What name did I give my cat?", "My cat's name is...", ("pixel",)),
    _i("admin", "H", "I work as a system administrator, mostly Linux servers.", "What do I do for work?",
       "What is my job?", "For work I...", ("admin", "administrator", "sysadmin"), "job"),
    _i("marguerite", "H", "My sister's name is Marguerite.", "What is my sister's name?", "What is my sister called?",
       "My sister is called...", ("marguerite",)),
    _i("volvo", "H", "I drive a red Volvo.", "What car do I drive?", "Which car do I own?", "I drive a...", ("volvo",), "thing"),
]

# AB: the 16-item mini-pool. Per seed 8 become set A (told L2, re-told as A' on L11) and 8 set B (told fresh on L11).
AB = [
    _i("ab01", "AB", "My favourite tea is lapsang souchong.", "What is my favourite tea?", "Which tea do I like best?", "My favourite tea is...", ("lapsang",), "thing"),
    _i("ab02", "AB", "I was born in Ljubljana.", "Where was I born?", "Which city am I originally from?", "I was born in...", ("ljubljana",), "place"),
    _i("ab03", "AB", "My piano teacher is called Mrs Okonkwo.", "Who teaches me piano?", "What is my piano teacher's name?", "My piano teacher is...", ("okonkwo",)),
    _i("ab04", "AB", "I'm allergic to cashews.", "What am I allergic to?", "Which nut must I avoid?", "I'm allergic to...", ("cashew", "cashews"), "thing"),
    _i("ab05", "AB", "My favourite band is Slowdive.", "What is my favourite band?", "Which group do I listen to most?", "My favourite band is...", ("slowdive",)),
    _i("ab06", "AB", "I keep a sourdough starter named Gertrude.", "What is my sourdough starter called?", "What name did I give the starter in my kitchen?", "My sourdough starter is...", ("gertrude",)),
    _i("ab07", "AB", "My landlord is a man named Mr Fitzgerald.", "Who is my landlord?", "What is my landlord called?", "My landlord is...", ("fitzgerald",)),
    _i("ab08", "AB", "I run along the canal every Sunday morning.", "Where do I run on Sundays?", "Which path do I jog along at the weekend?", "On Sundays I run along...", ("canal",), "place"),
    _i("ab09", "AB", "My favourite cheese is manchego.", "What is my favourite cheese?", "Which cheese do I like most?", "My favourite cheese is...", ("manchego",), "thing"),
    _i("ab10", "AB", "I studied geology at university.", "What did I study at university?", "What was my degree in?", "At university I studied...", ("geology",), "thing"),
    _i("ab11", "AB", "I'm saving up for a trip to Patagonia.", "Where am I saving up to travel?", "Which place do I want to visit?", "I'm saving up for a trip to...", ("patagonia",), "place"),
    _i("ab12", "AB", "I speak a little Portuguese.", "Which language do I speak a little?", "What foreign language can I get by in?", "I speak a little...", ("portuguese",), "thing"),
    _i("ab13", "AB", "My favourite film is Stalker.", "What is my favourite film?", "Which movie do I love most?", "My favourite film is...", ("stalker",), "thing"),
    _i("ab14", "AB", "My doctor's surgery is on Linden Road.", "What road is my doctor's surgery on?", "Where is my GP located?", "My doctor's surgery is on...", ("linden",), "place"),
    _i("ab15", "AB", "I play chess on Thursday evenings at the library.", "Which evening do I play chess?", "On what day is my chess night?", "I play chess on...", ("thursday",), "thing"),
    _i("ab16", "AB", "My coffee machine is an old Gaggia.", "What kind of coffee machine do I own?", "Which brand is my espresso maker?", "My coffee machine is...", ("gaggia",), "thing"),
]

# E: events with a context and two slots (gist = what happened, detail = a name). Direct/reworded ask the detail.
E = [
    _i("e01", "E", "It was foggy this morning and I missed my train at Kettering because the guard, a man called Barnaby, shut the doors early.",
       "What was the guard called who shut the train doors on me?", "Who was the guard that closed the doors early?", "The foggy morning at the station...",
       ("barnaby",), "event-detail", context="fog", gist_q="What happened to me at the station?", gist_keys=("missed", "train")),
    _i("e02", "E", "It was snowing on Saturday when I found a stray terrier near the bakery and named it Sprocket.",
       "What did I name the stray terrier?", "What is the stray dog from the bakery called?", "The snowy day with the stray dog...",
       ("sprocket",), "event-detail", context="snow", gist_q="What did I find near the bakery?", gist_keys=("terrier", "dog")),
    _i("e03", "E", "It was windy at the harbour and I dropped my phone in the water, but a fisherman called Ignatz fished it out.",
       "Who rescued my phone from the water?", "What was the fisherman's name who got my phone back?", "The windy day at the harbour...",
       ("ignatz",), "event-detail", context="wind", gist_q="What did I drop in the water?", gist_keys=("phone",)),
    _i("e04", "E", "I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.",
       "What was the film called that we saw?", "Which movie did we watch after the storm?", "The thunderstorm on the way to the cinema...",
       ("meridian",), "event-detail", context="thunderstorm", gist_q="What caught me out on the way to the cinema?", gist_keys=("thunderstorm", "storm")),
    _i("e05", "E", "It was boiling hot in the office, so I bought fourteen ice lollies for the whole team.",
       "How many ice lollies did I buy for the team?", "What number of lollies did the team get?", "The boiling hot day at the office...",
       ("fourteen", "14"), "event-detail", context="heat", gist_q="What did I buy for the team?", gist_keys=("lollies", "ice")),
    _i("e06", "E", "The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.",
       "Where did the candle in my drawer come from?", "Which town was my grandmother's house in, the one the candle came from?", "The rainy night with the power cut...",
       ("tavistock",), "event-detail", context="rain", gist_q="What happened on the rainy night?", gist_keys=("power", "candle")),
    _i("e07", "E", "It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.",
       "Who did I speak to at the wildlife rescue?", "What was the rescue worker's name?", "The misty morning at the lake...",
       ("oluwaseun",), "event-detail", context="mist", gist_q="What did I see at the lake?", gist_keys=("kingfisher", "bird")),
    _i("e08", "E", "It was a freezing morning and my car would not start, so I borrowed a bicycle from a neighbour called Pell.",
       "Whose bicycle did I borrow?", "Which neighbour lent me a bike?", "The freezing morning when the car died...",
       ("pell",), "event-detail", context="frost", gist_q="What went wrong on the freezing morning?", gist_keys=("car", "start")),
    _i("e09", "E", "On a sunny afternoon at the market I bought a pot of honey from a stall run by Anselm.",
       "Who ran the honey stall?", "Whose stall did I buy honey from?", "The sunny afternoon at the market...",
       ("anselm",), "event-detail", context="sun", gist_q="What did I buy at the market?", gist_keys=("honey",)),
    _i("e10", "E", "It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.",
       "What was the visiting goalkeeper called?", "Who played in goal for the away side?", "The drizzly day at the football...",
       ("zbigniew",), "event-detail", context="drizzle", gist_q="Where did it drizzle all day?", gist_keys=("football", "match")),
]

# F: ordinary fillers.
F = [
    _i("f01", "F", "I had porridge with blueberries for breakfast.", "What did I have for breakfast?", "What was my morning meal?", "For breakfast I had...", ("porridge",), "thing"),
    _i("f02", "F", "The bins in my street go out on Tuesday.", "Which day do the bins go out?", "When is the rubbish collected on my street?", "The bins go out on...", ("tuesday",), "thing"),
    _i("f03", "F", "I'm reading a novel by Italo Calvino.", "Whose novel am I reading?", "Which author am I reading at the moment?", "I'm reading a novel by...", ("calvino",)),
    _i("f04", "F", "I bought a new pair of boots in Leeds.", "Where did I buy my boots?", "Which city did my new boots come from?", "I bought new boots in...", ("leeds",), "place"),
    _i("f05", "F", "My favourite sandwich is egg and cress.", "What is my favourite sandwich?", "Which sandwich do I like best?", "My favourite sandwich is...", ("cress",), "thing"),
    _i("f06", "F", "I keep a spare key under a flowerpot.", "Where do I keep the spare key?", "Where is the backup key hidden?", "The spare key is under...", ("flowerpot",), "thing"),
    _i("f07", "F", "My dentist is called Dr Venkataraman.", "Who is my dentist?", "What is my dentist's name?", "My dentist is...", ("venkataraman",)),
    _i("f08", "F", "I'm knitting a scarf in orange wool.", "What colour is the scarf I'm knitting?", "Which colour wool am I using?", "I'm knitting a scarf in...", ("orange",), "thing"),
    _i("f09", "F", "The new pizza place on my street is called Fabrizio's.", "What is the new pizza place called?", "Which pizzeria opened on my street?", "The new pizza place is...", ("fabrizio", "fabrizio's")),
    _i("f10", "F", "My barber is called Rashid.", "Who cuts my hair?", "What is my barber's name?", "My barber is...", ("rashid",)),
    _i("f11", "F", "My favourite mug has a picture of a walrus on it.", "What animal is on my favourite mug?", "Which creature does my mug show?", "My favourite mug has...", ("walrus",), "thing"),
    _i("f12", "F", "I water my tomatoes every evening.", "What do I water every evening?", "Which plants get my evening watering?", "Every evening I water...", ("tomatoes", "tomato"), "thing"),
    _i("f13", "F", "I once sat next to a famous chef called Ottoline on a flight.", "Which chef did I sit next to on a flight?", "Who was the chef beside me on the plane?", "On a flight I sat beside...", ("ottoline",)),
    _i("f14", "F", "My laptop is covered in stickers of cacti.", "What are the stickers on my laptop of?", "Which plants decorate my laptop?", "My laptop is covered in stickers of...", ("cacti", "cactus"), "thing"),
    _i("f15", "F", "I always do the crossword on the train.", "What do I do on the train?", "Which puzzle do I always solve while travelling?", "On the train I always do...", ("crossword",), "thing"),
    _i("f16", "F", "I'm planning a picnic on Hampstead Heath next month.", "Where am I planning a picnic?", "Which park is the picnic going to be in?", "I'm planning a picnic on...", ("hampstead",), "place"),
    _i("f17", "F", "I painted my front door teal last year.", "What colour is my front door?", "Which colour did I paint the door?", "My front door is...", ("teal",), "thing"),
    _i("f18", "F", "My gym is above a florist called Petal and Stem.", "What is the florist below my gym called?", "Which shop is under my gym?", "My gym is above...", ("petal",)),
]

# D: near-duplicate frames: the question copies an H frame ("my sister's name", "my cat", "what car", "work").
D = [
    _i("d01", "D", "My cousin's name is Philippa.", "What is my cousin's name?", "What is my cousin called?", "My cousin is called...", ("philippa",), dup_of="marguerite"),
    _i("d02", "D", "My dog is called Bramble.", "What is my dog called?", "What name did I give my dog?", "My dog's name is...", ("bramble",), dup_of="pixel"),
    _i("d03", "D", "I ride a blue Vespa to the station.", "What do I ride to the station?", "Which scooter do I own?", "I ride a...", ("vespa",), "thing", dup_of="volvo"),
    _i("d04", "D", "My neighbour's name is Cornelius.", "What is my neighbour's name?", "What is my neighbour called?", "My neighbour is called...", ("cornelius",), dup_of="marguerite"),
    _i("d05", "D", "My nephew works as a baker.", "What does my nephew do for work?", "What is my nephew's job?", "My nephew works as...", ("baker",), "job", dup_of="admin"),
    _i("d06", "D", "My goldfish is called Oswin.", "What is my goldfish called?", "What name did I give my goldfish?", "My goldfish is called...", ("oswin",), dup_of="pixel"),
]

# Questions about things never told in any life (3 are drawn per seed; fixed for that seed's life).
NEVER = [
    ("never1", "What is my mother's name?", ("marguerite", "philippa")),
    ("never2", "What is my favourite colour?", ()),
    ("never3", "Where did I go on holiday last summer?", ()),
    ("never4", "What is my brother's name?", ("marguerite", "philippa")),
    ("never5", "What is my favourite animal?", ()),
    ("never6", "What did I have for dinner last night?", ()),
]

SMALL_TALK = [
    "The bus was late again this morning.",
    "I made a big pot of soup yesterday.",
    "My neighbour's wind chimes were going all night.",
    "I've been trying to drink more water lately.",
    "There was a lovely sunset on my walk home.",
    "My phone updated itself in the middle of a call.",
    "I stayed up too late watching a documentary about bridges.",
    "The shop on the corner finally got fresh bread.",
    "I think autumn is my favourite season.",
    "I tried a new route to work today.",
    "My shoes got soaked in a puddle this morning.",
    "I rearranged the furniture in my living room.",
    "I've been humming the same song all week.",
    "The park was full of kids on scooters today.",
]
GREETINGS = ["Good morning! How are you?", "Hello again! How did you sleep?", "Hi, I'm back. How are you doing?", "Morning! How's it going?"]

# Open prompts for the morning battery: requests or questions only (never stored by the hippocampus). The story has a name slot.
OPEN = [
    ("story", "Tell me a short story about a fox who finds an old lantern.", "a fox named"),
    ("feeling", "How are you feeling this morning?", None),
    ("advice", "Can you give me some advice about staying focused?", None),
    ("opinion", "What do you think of rainy weekends?", None),
    ("knowledge", "Why do leaves change colour in autumn?", None),
    ("chat", "Can you suggest something to do on a quiet evening?", None),
]

# Incidents: two domains x 4 subsystems, each told in an X or a Y variant. Every telling names BOTH candidate causes: one
# is ruled out, the other is the stated fix. X = capacity cause (full, clogged), Y = credential cause (expired certificate,
# code, login). A domain's four incidents always share one variant; the two domains get opposite variants per life.
# NEW is the never-told incident posed on the morning after the k-th telling (transfer read-out, Stage 1; unused in Stage 0).
INCIDENTS = {
    "sysadmin": [
        {"sub": "mail relay",
         "X": "The mail relay stopped delivering this morning. The certificate was valid; the disk was completely full.",
         "Y": "The mail relay stopped delivering this morning. The disk had plenty of space; it was the expired certificate.",
         "new": "The build server stopped taking jobs this morning. What would you check first?",
         "opt": {"X": "I would first check whether the job queue disk has filled up.",
                 "Y": "I would first check whether the job runner's certificate has expired."}},
        {"sub": "vpn gateway",
         "X": "Nobody could reach the VPN gateway today. The login service was working; the log partition had filled up.",
         "Y": "Nobody could reach the VPN gateway today. The log partition had plenty of room; the signing certificate had expired.",
         "new": "The wiki server is timing out for everyone. What would you check first?",
         "opt": {"X": "I would first check whether the page cache disk has filled up.",
                 "Y": "I would first check whether the wiki login certificate has expired."}},
        {"sub": "file server",
         "X": "The file server refused every upload last night. The service account was fine; the volume had run out of space.",
         "Y": "The file server refused every upload last night. The volume had plenty of space; the service account password had expired.",
         "new": "The database replica stopped syncing overnight. What would you check first?",
         "opt": {"X": "I would first check whether the replica's log volume has filled up.",
                 "Y": "I would first check whether the replica's sync password has expired."}},
        {"sub": "monitoring dashboard",
         "X": "The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.",
         "Y": "The monitoring dashboard went blank on Monday. The metrics store had lots of room; the API token had been revoked.",
         "new": "The print server stopped accepting documents. What would you check first?",
         "opt": {"X": "I would first check whether the print spool disk has filled up.",
                 "Y": "I would first check whether the print service password has expired."}},
    ],
    "home repair": [
        {"sub": "garden tap",
         "X": "The garden tap barely trickled this weekend. The controller code was right; the filter was clogged solid with sediment.",
         "Y": "The garden tap barely trickled this weekend. The filter was clean; the controller's access code had expired.",
         "new": "The sprinkler controller quit overnight. What would you check first?",
         "opt": {"X": "I would first check whether the sprinkler's feed pipe is blocked.",
                 "Y": "I would first check whether the sprinkler's access code has expired."}},
        {"sub": "garage door",
         "X": "The garage door would not open on Wednesday. The remote code was fine; the track was jammed full of leaves.",
         "Y": "The garage door would not open on Wednesday. The track was clear; the remote's code had expired.",
         "new": "The gate opener stopped responding this morning. What would you check first?",
         "opt": {"X": "I would first check whether the gate's runner track is blocked.",
                 "Y": "I would first check whether the gate's remote code has expired."}},
        {"sub": "heating",
         "X": "The radiators stayed cold all week. The thermostat login was fine; the system was full of trapped sludge.",
         "Y": "The radiators stayed cold all week. The system was clean; the thermostat's login had expired.",
         "new": "The hot tub heater will not start. What would you check first?",
         "opt": {"X": "I would first check whether the heater's water filter is blocked.",
                 "Y": "I would first check whether the heater's panel login has expired."}},
        {"sub": "gutter",
         "X": "The gutter overflowed during the storm. The sensor key was working; the downpipe was blocked with moss.",
         "Y": "The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.",
         "new": "The cellar sump alarm went off overnight. What would you check first?",
         "opt": {"X": "I would first check whether the sump pit is blocked.",
                 "Y": "I would first check whether the alarm's app login has expired."}},
    ],
}
# Stage-0 schedule of the incident stream (design section 3): sysadmin on L2, L4, L7, L12; home repair on L3, L6, L9, L13.
INCIDENT_DAYS = {"sysadmin": [2, 4, 7, 12], "home repair": [3, 6, 9, 13]}

# The tellings in the design's day counts (H on L1 comes from weekly.SESSION_1, not here).
ALL = {x.id: x for x in H + AB + E + F + D}


def split_stats() -> dict:
    n_inc = sum(len(v) for v in INCIDENTS.values())
    return {"H": len(H), "AB": len(AB), "E": len(E), "F": len(F), "D": len(D), "never": len(NEVER),
            "small_talk": len(SMALL_TALK), "open": len(OPEN), "incidents": n_inc, "incident_tellings": 2 * n_inc,
            "items": len(H) + len(AB) + len(E) + len(F) + len(D) + len(NEVER) + len(SMALL_TALK) + len(OPEN) + n_inc}
