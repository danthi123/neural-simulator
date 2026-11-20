# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: schema transfer and cued recall by context at 1.5B (design section 2)
"""World content for the multi-week life (brain/eval/life.py), TEST split, written by a sonnet teacher that has never seen brain output,
frozen by sha256 before the first registered run (design section 3).

RULE: nothing here may be used to fit, tune or select anything (bars, gains, frames, switches) on the registered seeds.
DEV is for Stage 0 and tuning (dev seeds 310-312, 313-315). The TEST split is a different file written later by a
sonnet that has never seen brain output (design section 3). open_conv_bank stays the held-out open-conversation test.

Item = telling + direct question + reworded question + partial cue + key words + slot. Keys are distinctive words so that a
key in an unrelated reply is an intrusion. No telling starts like a request (brain._is_telling would skip it) or ends with "?".
Classes: H high-use, A/B savings pool (48 items, halves swapped by seed), E events (context + gist + detail), F fillers,
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



# AB: the 48-item pool (the life samples from it; halves are swapped by seed).
AB = [
    _i("ab01", "AB", "I support Hartlepool United.", "Which football team do I support?", "What club do I follow?", "I support...", ("hartlepool",), "thing"),
    _i("ab02", "AB", "My grandmother came from Inverness.", "Where did my grandmother come from?", "Which town was my gran originally from?", "My grandmother came from...", ("inverness",), "place"),
    _i("ab03", "AB", "My accountant is called Mr Bellingham.", "Who is my accountant?", "Which accountant does my tax return?", "My accountant is...", ("bellingham",)),
    _i("ab04", "AB", "I'm terrified of mannequins.", "What am I terrified of?", "What sort of figures give me the creeps?", "I'm terrified of...", ("mannequins",), "thing"),
    _i("ab05", "AB", "My favourite poet is Philip Larkin.", "Who is my favourite poet?", "Which poet do I like best?", "My favourite poet is...", ("larkin",)),
    _i("ab06", "AB", "I named my canoe Archimedes.", "What is my canoe named?", "Which name did I paint on the canoe?", "My canoe is named...", ("archimedes",)),
    _i("ab07", "AB", "My brother-in-law runs a bakery in Whitby.", "Where does my brother-in-law run his bakery?", "Which seaside town is my brother-in-law's bakery in?", "My brother-in-law's bakery is in...", ("whitby",), "place"),
    _i("ab08", "AB", "I always order the pasanda at the curry house.", "What do I always order at the curry house?", "Which dish do I choose when we get curry?", "At the curry house I always order...", ("pasanda",), "thing"),
    _i("ab09", "AB", "My favourite painter is Turner.", "Who is my favourite painter?", "Which artist do I admire most?", "My favourite painter is...", ("turner",)),
    _i("ab10", "AB", "Every morning I drink a glass of kefir.", "What do I drink every morning?", "Which drink starts my day?", "Every morning I drink a glass of...", ("kefir",), "thing"),
    _i("ab11", "AB", "My godmother lives in Aberystwyth.", "Where does my godmother live?", "Which town is my godmother's home?", "My godmother lives in...", ("aberystwyth",), "place"),
    _i("ab12", "AB", "I play the cello in a community orchestra.", "Which instrument do I play in the orchestra?", "What do I play with the community orchestra?", "In the orchestra I play the...", ("cello",), "thing"),
    _i("ab13", "AB", "My favourite biscuit is the Garibaldi.", "What is my favourite biscuit?", "Which biscuit do I like most?", "My favourite biscuit is the...", ("garibaldi",), "thing"),
    _i("ab14", "AB", "We got married in Harrogate.", "Where did I get married?", "Which town did I marry in?", "We got married in...", ("harrogate",), "place"),
    _i("ab15", "AB", "My running shoes are made by Saucony.", "What brand are my running shoes?", "Who makes my trainers?", "My running shoes are made by...", ("saucony",), "thing"),
    _i("ab16", "AB", "I bank with Nationwide.", "Which bank am I with?", "Where do I keep my money?", "I bank with...", ("nationwide",), "thing"),
    _i("ab17", "AB", "As a teenager I worked the ticket kiosk at the pier.", "What was my summer job as a teenager?", "Where did I earn money as a teen?", "As a teenager I worked the...", ("kiosk",), "job"),
    _i("ab18", "AB", "My favourite crisps are Worcester sauce flavour.", "What flavour crisps do I like best?", "Which crisps do I always pick?", "My favourite crisps are...", ("worcester",), "thing"),
    _i("ab19", "AB", "The night bus I take home goes to Gravesend.", "Where does my night bus go?", "Which town does my late bus go to?", "My night bus goes to...", ("gravesend",), "place"),
    _i("ab20", "AB", "My daughter goes to Saint Ethelburga's.", "Which school does my daughter attend?", "Where does my daughter study?", "My daughter goes to...", ("ethelburga",), "place"),
    _i("ab21", "AB", "My optician is called Ms Ravenscroft.", "Who is my optician?", "Which optician tests my eyes?", "My optician is...", ("ravenscroft",)),
    _i("ab22", "AB", "My favourite sweet is rhubarb and custard.", "What is my favourite sweet?", "Which sweet do I like most?", "My favourite sweet is...", ("rhubarb",), "thing"),
    _i("ab23", "AB", "I volunteer at a donkey sanctuary on Saturdays.", "Where do I volunteer?", "Which animal charity do I help out at?", "I volunteer at a...", ("donkey",), "place"),
    _i("ab24", "AB", "I learnt to swim at Tooting Bec Lido.", "Where did I learn to swim?", "Which pool did I learn in as a child?", "I learnt to swim at...", ("tooting",), "place"),
    _i("ab25", "AB", "Our pub quiz team is called the Quizzly Bears.", "What is my pub quiz team called?", "What name does my quiz team go by?", "My pub quiz team is called...", ("quizzly",)),
    _i("ab26", "AB", "I'm learning to play the ukulele.", "What instrument am I learning?", "Which instrument am I practising these days?", "I'm learning to play the...", ("ukulele",), "thing"),
    _i("ab27", "AB", "I have a tattoo of a heron on my shoulder.", "What is my tattoo of?", "Which bird is tattooed on me?", "I have a tattoo of a...", ("heron",), "thing"),
    _i("ab28", "AB", "Our postman is a chap called Desmond.", "Who is my postman?", "What is my postman called?", "My postman is called...", ("desmond",)),
    _i("ab29", "AB", "I keep my passport in an old shoebox.", "Where do I keep my passport?", "Where is my passport stored?", "I keep my passport in...", ("shoebox",), "thing"),
    _i("ab30", "AB", "My first car was a Morris Minor.", "What was my first car?", "Which car did I learn to drive in?", "My first car was a...", ("morris",), "thing"),
    _i("ab31", "AB", "The vegetable I like least is the parsnip.", "Which vegetable do I like least?", "What is the veg I cannot stand?", "The vegetable I like least is the...", ("parsnip",), "thing"),
    _i("ab32", "AB", "Once I climbed Helvellyn in sandals.", "Which mountain did I climb in sandals?", "What peak did I tackle wearing sandals?", "Once I climbed...", ("helvellyn",), "place"),
    _i("ab33", "AB", "Our yoga teacher is called Ingrid.", "Who teaches me yoga?", "What is my yoga teacher's name?", "My yoga teacher is...", ("ingrid",)),
    _i("ab34", "AB", "I buy my bread from a man called Wojciech.", "Who do I buy my bread from?", "What is my bread seller called?", "I buy my bread from...", ("wojciech",)),
    _i("ab35", "AB", "I'm a member of the Wensleydale Beekeepers Association.", "Which association am I a member of?", "What society do I belong to?", "I'm a member of the...", ("wensleydale",), "thing"),
    _i("ab36", "AB", "I write my diary with a Pelikan fountain pen.", "What make of pen do I write my diary with?", "Which fountain pen do I use for my diary?", "I write my diary with a...", ("pelikan",), "thing"),
    _i("ab37", "AB", "Damson is my favourite jam.", "What is my favourite jam?", "Which jam do I like best?", "My favourite jam is...", ("damson",), "thing"),
    _i("ab38", "AB", "As a child I fell off a pony called Hotspur.", "What was the pony called that I fell off?", "Which pony threw me as a child?", "As a child I fell off...", ("hotspur",)),
    _i("ab39", "AB", "My local library is in Hackney.", "Where is my local library?", "Which part of London is my library in?", "My local library is in...", ("hackney",), "place"),
    _i("ab40", "AB", "My oldest friend is called Fenella.", "Who is my oldest friend?", "What is my oldest friend called?", "My oldest friend is...", ("fenella",)),
    _i("ab41", "AB", "My pillow is stuffed with buckwheat.", "What is my pillow stuffed with?", "Which grain fills my pillow?", "My pillow is stuffed with...", ("buckwheat",), "thing"),
    _i("ab42", "AB", "I bought my guitar in a shop in Deptford.", "Where did I buy my guitar?", "Where did my guitar come from?", "I bought my guitar in...", ("deptford",), "place"),
    _i("ab43", "AB", "Woodsmoke is my favourite smell.", "What is my favourite smell?", "Which scent do I love most?", "My favourite smell is...", ("woodsmoke",), "thing"),
    _i("ab44", "AB", "A boy named Tarquin was in my scout troop.", "Who was in my scout troop?", "Which boy do I remember from the scouts?", "In my scout troop there was...", ("tarquin",)),
    _i("ab45", "AB", "My phone case is covered in sunflowers.", "What is my phone case covered in?", "Which flowers decorate my phone case?", "My phone case is covered in...", ("sunflowers",), "thing"),
    _i("ab46", "AB", "I collect fridge magnets from Lisbon.", "Which city do my fridge magnets come from?", "Where do I collect magnets from?", "I collect fridge magnets from...", ("lisbon",), "place"),
    _i("ab47", "AB", "Our usual takeaway is a Greek place called Dionysus.", "What is the name of the Greek place we order from?", "Which Greek place do I order from?", "Our usual Greek place is...", ("dionysus",)),
    _i("ab48", "AB", "I dream of owning a narrowboat.", "What do I dream of owning?", "Which kind of boat do I long for?", "I dream of owning a...", ("narrowboat",), "thing"),
]

# E: 20 events with a context and two slots (gist = what happened, detail = a name or a number word). Direct/reworded ask the detail.
E = [
    _i("e01", "E", "It hailed hard on the allotment and I sheltered with a retired locksmith called Leopold.", "What was the retired locksmith called?", "Who did I shelter with when it hailed?", "The hailstorm when I took shelter...", ("leopold",), "event-detail", context="hail", gist_q="Where was I when it hailed hard?", gist_keys=("allotment",), extra={'family': 'wet'}),
    _i("e02", "E", "During the gale last week a chimney stack came down across our lane, and the council crew was led by a woman named Brunhilde.", "Who led the council crew?", "What was the crew leader's name?", "The gale last week on our lane...", ("brunhilde",), "event-detail", context="gale", gist_q="What came down across our lane?", gist_keys=("chimney",), extra={'family': 'wind'}),
    _i("e03", "E", "It was sweltering on the ferry to Holyhead, and I counted seventeen seagulls following the boat.", "How many seagulls did I count?", "What number of gulls followed the boat?", "The sweltering crossing to Holyhead...", ("seventeen",), "event-detail", context="heatwave", gist_q="What was I travelling on when it was sweltering?", gist_keys=("ferry",), extra={'family': 'heat'}),
    _i("e04", "E", "It sleeted on the platform at Crewe, and a stranger called Hildegard lent me her umbrella.", "What was the stranger's name?", "Who helped me out on the platform?", "The sleet at Crewe station...", ("hildegard",), "event-detail", context="sleet", gist_q="What did a stranger lend me at the platform?", gist_keys=("umbrella",), extra={'family': 'wet'}),
    _i("e05", "E", "It was a humid evening at my cousin's wedding and the band played eleven encores.", "How many encores did the band play?", "What number of encores did we get?", "The humid evening at my cousin's do...", ("eleven",), "event-detail", context="humidity", gist_q="What was I celebrating on the humid evening?", gist_keys=("wedding",), extra={'family': 'heat'}),
    _i("e06", "E", "On a crisp bright morning I found a hedgehog in the shed, and the ranger who took it in was called Mbeki.", "What was the ranger called?", "Who took the little visitor in?", "The crisp morning with the little visitor...", ("mbeki",), "event-detail", context="crisp air", gist_q="What did I find in the shed?", gist_keys=("hedgehog",), extra={'family': 'clear'}),
    _i("e07", "E", "Rain hammered on the caravan roof in Skegness, and we played cards with a retired couple called the Pennyfeathers.", "Who did we play cards with?", "What was the retired couple called?", "The rain on the roof of our holiday let...", ("pennyfeathers",), "event-detail", context="rain on a roof", gist_q="Where did the rain hammer on the roof?", gist_keys=("caravan",), extra={'family': 'wet'}),
    _i("e08", "E", "It was dusk at the quarry when my torch failed, and a fellow hiker named Casimir walked me back.",
       "Who walked me back from the quarry?", "What was the hiker's name?", "The dusk at the quarry...",
       ("casimir",), "event-detail", context="dusk", gist_q="What failed on me at the quarry?", gist_keys=("torch",), extra={'family': 'dusk'}),
    _i("e09", "E", "A rainbow arched over the hospital car park the day my niece was born, and she weighed seven pounds and twelve ounces.", "How many ounces over seven pounds did the baby weigh?", "What was the ounce figure on her birth weight?", "The rainbow over the car park...", ("twelve",), "event-detail", context="rainbow", gist_q="What kind of building did I go to the day my niece was born?", gist_keys=("hospital",), extra={'family': 'light'}),
    _i("e10", "E", "Pollen hung thick in the air in the cinema foyer when the fire drill began, and the usher who led us out was called Eustace.", "What was the usher called?", "Who led us out of the cinema?", "The hay-fever afternoon when the alarm sounded...", ("eustace",), "event-detail", context="pollen", gist_q="Where did the fire drill begin?", gist_keys=("foyer",), extra={'family': 'pollen'}),
    _i("e11", "E", "A heavy dew soaked my socks at the cricket ground, and the last wicket fell at exactly ninety-three.", "At what score did the innings end?", "What was the total when the final batter went?", "The dewy morning when my socks got soaked...", ("ninety-three",), "event-detail", context="dew", gist_q="Which sport was I watching in the dew?", gist_keys=("cricket",), extra={'family': 'dew'}),
    _i("e12", "E", "It was foggy on the clifftop at Beachy Head and I dropped my hat, but a dog walker called Philomena found it.", "Who came to my rescue on the clifftop?", "What was the dog walker's name?", "The foggy clifftop...", ("philomena",), "event-detail", context="fog", gist_q="What did I drop on the clifftop?", gist_keys=("hat",), extra={'family': 'fog'}),
    _i("e13", "E", "A thick smog sat over the city the day my bicycle was stolen outside the post office, and the constable who took my report was named Fitch.",
       "What was the constable called?", "Who took my report about the theft?", "The smoggy day outside the post office...",
       ("fitch",), "event-detail", context="smog", gist_q="What happened outside the post office?", gist_keys=("stolen",), extra={'family': 'smog'}),
    _i("e14", "E", "On a mild evening the lights failed in the village hall, and the vicar, Reverend Applegarth, kept us singing in the dark.",
       "What was the vicar called?", "Who kept us singing in the dark?", "The mild evening when the lights failed...",
       ("applegarth",), "event-detail", context="mild evening", gist_q="Where did the lights fail?", gist_keys=("hall",), extra={'family': 'mild'}),
    _i("e15", "E", "It was bitterly cold at the bonfire display and a fellow called Quigley let me light the final rocket.", "Who let me join in at the display?", "What was the man's name who handed me the lighter?", "The bitter cold at the bonfire...", ("quigley",), "event-detail", context="bitter cold", gist_q="What did I get to light at the bonfire?", gist_keys=("rocket",), extra={'family': 'cold'}),
    _i("e16", "E", "Thunder rumbled over sports day, and my goddaughter won the sack race by exactly fifteen seconds.", "By how many seconds did she win?", "What was the winning margin?", "The thunder on sports day...", ("fifteen",), "event-detail", context="thunder", gist_q="Which race did my goddaughter win on the day of thunder?", gist_keys=("sack",), extra={'family': 'thunder'}),
    _i("e17", "E", "The sea at Whitstable was glassy and still, and I found a message in a bottle signed by a girl named Saoirse.", "Who signed the note I found?", "What was the girl's name?", "The still glassy sea...", ("saoirse",), "event-detail", context="calm", gist_q="What did I find at the beach?", gist_keys=("message",), extra={'family': 'calm'}),
    _i("e18", "E", "It was a starlit night on the sleeper to Fort William, and the steward, a man called Lachlan, brought me an extra blanket.", "What was the steward called?", "Who brought me a blanket on the night journey?", "The starlit night on the way to Fort William...", ("lachlan",), "event-detail", context="starlit", gist_q="What kind of journey was I on that starlit night?", gist_keys=("sleeper",), extra={'family': 'night'}),
    _i("e19", "E", "It was a still golden evening on the pier when a busker played the same tune twenty-eight times.", "How many times was the same tune repeated?", "What number of repeats did we sit through?", "The golden evening on the pier...", ("twenty-eight",), "event-detail", context="golden light", gist_q="Who played the same tune on the pier?", gist_keys=("busker",), extra={'family': 'light2'}),
    _i("e20", "E", "The tide was far out and the estuary mud was grey when I found an old coin, and the curator I phoned was called Thaddeus.",
       "What was the curator called?", "Who did I phone about the find?", "The low tide at the estuary...",
       ("thaddeus",), "event-detail", context="low tide", gist_q="What did I find in the estuary mud?", gist_keys=("coin",), extra={'family': 'tide'}),
]

# F: 60 ordinary fillers.
F = [
    _i("f01", "F", "I had kippers for breakfast.", "What did I have for breakfast?", "What was my morning meal?", "For breakfast I had...", ("kippers",), "thing"),
    _i("f02", "F", "The bin men whistle sea shanties.", "What do the bin men whistle?", "Which tunes drift up from the bin lorry?", "The bin men whistle...", ("shanties",), "thing"),
    _i("f03", "F", "I'm reading a biography of Ada Lovelace.", "Whose biography am I reading?", "Which woman is my current book about?", "I'm reading a biography of...", ("lovelace",)),
    _i("f04", "F", "I bought a new raincoat in Norwich.", "Where did I buy my raincoat?", "Which city did my new coat come from?", "I bought a new raincoat in...", ("norwich",), "place"),
    _i("f05", "F", "Leek and potato is my favourite soup.", "What is my favourite soup?", "Which soup do I like best?", "My favourite soup is...", ("leek",), "thing"),
    _i("f06", "F", "The stepladder lives in the cupboard under the stairs.", "Where do I keep the ladder?", "Where is the ladder stored?", "The stepladder lives in...", ("stairs",), "thing"),
    _i("f07", "F", "Our vet is called Dr Abernathy.", "Who is my vet?", "Which vet do we take the animals to?", "My vet is...", ("abernathy",)),
    _i("f08", "F", "I'm crocheting a blanket in ochre.", "What colour is the blanket I'm crocheting?", "Which colour wool am I crocheting with?", "I'm crocheting a blanket in...", ("ochre",), "thing"),
    _i("f09", "F", "The new cafe near me is called Brindle and Fox.", "What is the new cafe called?", "Which cafe opened near me?", "The new cafe is called...", ("brindle",)),
    _i("f10", "F", "Our cleaner is called Lucia.", "Who cleans my flat?", "Who tidies my flat each week?", "My cleaner is...", ("lucia",)),
    _i("f11", "F", "My favourite pen has a puffin on it.", "What animal is on my favourite pen?", "Which bird does my pen show?", "My favourite pen has a...", ("puffin",), "thing"),
    _i("f12", "F", "Every morning I feed the garden robins mealworms.", "What do I feed the garden robins?", "Which treat do the robins get from me?", "Every morning I feed the robins...", ("mealworms",), "thing"),
    _i("f13", "F", "At a petrol station I once met a famous astronaut called Beaumont.", "Which astronaut did I meet at a petrol station?", "Who was the famous spaceman at the garage?", "At a petrol station I met...", ("beaumont",)),
    _i("f14", "F", "My rucksack is covered in badges of Cornish harbours.", "What are the badges on my rucksack of?", "Which places decorate my rucksack?", "My rucksack is covered in badges of...", ("harbours",), "thing"),
    _i("f15", "F", "Ironing goes quicker with an audiobook of Moby-Dick.", "What do I listen to while ironing?", "Which recording keeps me company at the ironing board?", "While ironing I listen to...", ("moby",), "thing"),
    _i("f16", "F", "I'm planning a long weekend in Mechelen.", "Where am I planning a long weekend?", "Which city am I visiting for the weekend?", "I'm planning a long weekend in...", ("mechelen",), "place"),
    _i("f17", "F", "Last spring I painted the kitchen cupboards duck-egg.", "What colour are my kitchen cupboards?", "Which shade did I paint the cupboards?", "My kitchen cupboards are...", ("duck-egg",), "thing"),
    _i("f18", "F", "Our local chippy is called the Codfather.", "What is the local chippy called?", "Which fish and chip shop is nearest?", "The local chippy is called the...", ("codfather",)),
    _i("f19", "F", "A spare charger lives in my desk drawer at the office.", "What do I keep in my desk drawer at the office?", "What spare item is in my office drawer?", "In my desk drawer I keep a spare...", ("charger",), "thing"),
    _i("f20", "F", "I'm learning to say hello in Welsh.", "Which language am I learning to say hello in?", "What tongue am I picking up greetings in?", "I'm learning to say hello in...", ("welsh",), "thing"),
    _i("f21", "F", "My shopping trolley has a tartan cover.", "What pattern is my shopping trolley cover?", "Which fabric pattern covers my trolley?", "My shopping trolley cover is...", ("tartan",), "thing"),
    _i("f22", "F", "I picked up a second-hand typewriter at a car boot sale.", "What did I pick up at the car boot sale?", "Which old machine did I buy second-hand?", "At the car boot sale I picked up a...", ("typewriter",), "thing"),
    _i("f23", "F", "My thermos is dented from a fall down the steps.", "What is dented from a fall down the steps?", "Which flask of mine got dented?", "The dented thing is my...", ("thermos",), "thing"),
    _i("f24", "F", "Aloe vera is growing on my kitchen windowsill.", "What is growing on my kitchen windowsill?", "Which plant sits on my windowsill?", "On my windowsill I grow...", ("aloe",), "thing"),
    _i("f25", "F", "My favourite bench is next to the old bandstand.", "Where is my favourite bench?", "Which landmark is beside my favourite bench?", "My favourite bench is next to the...", ("bandstand",), "place"),
    _i("f26", "F", "In winter I take echinacea tablets.", "What do I take in winter?", "Which supplement do I swallow in the cold months?", "In winter I take...", ("echinacea",), "thing"),
    _i("f27", "F", "A colleague brings in flapjacks for the whole floor.", "What does my colleague bring in?", "Which treat do we get at work?", "My colleague brings in...", ("flapjacks",), "thing"),
    _i("f28", "F", "I just finished a jigsaw of the Pantheon.", "What was my jigsaw of?", "Which landmark did I put together as a puzzle?", "I just finished a jigsaw of the...", ("pantheon",), "thing"),
    _i("f29", "F", "I picked up some oatmeal soap at the market.", "What kind of soap did I buy?", "Which soap did I pick up?", "I bought some...", ("oatmeal",), "thing"),
    _i("f30", "F", "My hot water bottle has a knitted cover shaped like a penguin.", "What is my hot water bottle cover shaped like?", "Which animal is my bottle cover?", "My hot water bottle cover is shaped like a...", ("penguin",), "thing"),
    _i("f31", "F", "I keep winter coats in a trunk by the front door.", "Where do I keep my winter coats?", "Where are the coats stored?", "My winter coats are in a...", ("trunk",), "thing"),
    _i("f32", "F", "My favourite shop in town sells nothing but buttons.", "What does my favourite shop sell?", "Which small items fill my favourite shop?", "My favourite shop sells only...", ("buttons",), "thing"),
    _i("f33", "F", "Last week I bought a pineapple for the first time.", "What did I buy for the first time last week?", "Which fruit was a first for me?", "Last week I bought a...", ("pineapple",), "thing"),
    _i("f34", "F", "The lady upstairs plays the harpsichord.", "What does the lady upstairs play?", "Which instrument drifts down from the flat above?", "The lady upstairs plays the...", ("harpsichord",), "thing"),
    _i("f35", "F", "I have a scar on my knee from a trampoline.", "What gave me the scar on my knee?", "Which garden toy hurt my knee?", "The scar on my knee is from a...", ("trampoline",), "thing"),
    _i("f36", "F", "I spilled paprika all over the hob.", "What did I spill on the hob?", "Which spice made the mess?", "I spilled...", ("paprika",), "thing"),
    _i("f37", "F", "My uncle gave me a brass sextant for my birthday.", "What did my uncle give me for my birthday?", "Which present did I get from my uncle?", "For my birthday my uncle gave me a brass...", ("sextant",), "thing"),
    _i("f38", "F", "I once won a goose in a raffle.", "What did I once win in a raffle?", "Which bird was my raffle prize?", "In a raffle I won a...", ("goose",), "thing"),
    _i("f39", "F", "I've started a sketchbook of the pigeons in the square.", "What is my sketchbook of?", "Which birds do I draw?", "My sketchbook is full of...", ("pigeons",), "thing"),
    _i("f40", "F", "My jumper has a hole just under the collar.", "Where is the hole in my jumper?", "Which part of my sweater is worn through?", "My jumper has a hole just under the...", ("collar",), "thing"),
    _i("f41", "F", "I'm saving my loyalty points at Waitrose.", "Where am I saving loyalty points?", "Which supermarket gives me points?", "I'm saving loyalty points at...", ("waitrose",), "place"),
    _i("f42", "F", "I've signed up for a pottery evening class.", "What evening class have I signed up for?", "Which craft am I taking up at night?", "I've signed up for an evening class in...", ("pottery",), "thing"),
    _i("f43", "F", "I spread treacle on my crumpets.", "What do I spread on my crumpets?", "Which spread goes on my crumpets?", "On my crumpets I spread...", ("treacle",), "thing"),
    _i("f44", "F", "My bedside lamp is shaped like a mushroom.", "What is my bedside lamp shaped like?", "Which fungus does my lamp resemble?", "My bedside lamp is shaped like a...", ("mushroom",), "thing"),
    _i("f45", "F", "I listen to the shipping forecast before bed.", "What do I listen to before bed?", "Which broadcast do I put on at night?", "Before bed I listen to the...", ("shipping",), "thing"),
    _i("f46", "F", "I'm trying to learn the Cyrillic alphabet.", "Which alphabet am I trying to learn?", "What script am I practising?", "I'm trying to learn the...", ("cyrillic",), "thing"),
    _i("f47", "F", "Last summer I hung a brass gong in the back garden.", "What did I hang in the back garden?", "What did I put up out the back last summer?", "In the back garden I hung a...", ("gong",), "thing"),
    _i("f48", "F", "I bought a bottle of elderflower cordial at the farm shop.", "What cordial did I buy at the farm shop?", "Which drink came from the farm shop?", "At the farm shop I bought...", ("elderflower",), "thing"),
    _i("f49", "F", "My grandfather grew figs in Cyprus.", "What did my grandfather grow in Cyprus?", "Which fruit grew in his Cypriot garden?", "My grandfather grew...", ("figs",), "thing"),
    _i("f50", "F", "I wear my father's signet ring.", "Whose ring do I wear?", "Which family piece is on my finger?", "I wear my father's...", ("signet",), "thing"),
    _i("f51", "F", "Spare batteries are kept in a teapot on the dresser.", "Where do I keep spare batteries?", "Which odd place stores my batteries?", "I keep spare batteries in the...", ("teapot",), "thing"),
    _i("f52", "F", "I'd like to go to the Isle of Eigg next autumn.", "Which island would I like to visit next autumn?", "Which island would I go to for a trip?", "Next autumn I'd like to visit...", ("eigg",), "place"),
    _i("f53", "F", "A magpie has been nesting above my front step.", "Which bird is nesting above my front step?", "What has built a nest by my door?", "Above my front step is nesting a...", ("magpie",), "thing"),
    _i("f54", "F", "I do a nonogram before bed.", "What puzzle do I do before bed?", "Which puzzle ends my day?", "Before bed I do a...", ("nonogram",), "thing"),
    _i("f55", "F", "My phone wallpaper is a photo of a lighthouse.", "What is my phone wallpaper a photo of?", "Which building is on my phone screen?", "My phone wallpaper is a photo of a...", ("lighthouse",), "thing"),
    _i("f56", "F", "I took up origami during the lockdown.", "What craft did I take up during the lockdown?", "Which paper hobby began then?", "During the lockdown I took up...", ("origami",), "thing"),
    _i("f57", "F", "My favourite bookshop smells of cinnamon.", "What does my favourite bookshop smell of?", "Which spice scents the bookshop I love?", "My favourite bookshop smells of...", ("cinnamon",), "thing"),
    _i("f58", "F", "I swapped my old sofa for a velvet armchair.", "What did I swap my old sofa for?", "Which piece of furniture replaced the sofa?", "I swapped my old sofa for a velvet...", ("armchair",), "thing"),
    _i("f59", "F", "A ceilidh is on my calendar for next month.", "What have I signed up for next month?", "Which kind of dance night am I going to?", "Next month I'm going to a...", ("ceilidh",), "thing"),
    _i("f60", "F", "The strap on my watch is made of canvas.", "What is my watch strap made of?", "Which fabric is my watch strap?", "My watch strap is made of...", ("canvas",), "thing"),
]

# D: 18 near-duplicate frames: the question copies an H frame ("my sister's name", "my cat", "what car", "work").
D = [
    _i("d01", "D", "My aunt's name is Henrietta.", "What is my aunt's name?", "What is my aunt called?", "My aunt is called...", ("henrietta",), dup_of="marguerite"),
    _i("d02", "D", "My uncle's name is Bartholomew.", "What is my uncle's name?", "What is my uncle called?", "My uncle is called...", ("bartholomew",), dup_of="marguerite"),
    _i("d03", "D", "My neighbour across the landing is called Wilhelm.", "What is my neighbour's name?", "What is my neighbour called?", "My neighbour is called...", ("wilhelm",), dup_of="marguerite"),
    _i("d04", "D", "My grandmother's name is Winifred.", "What is my grandmother's name?", "What is my grandmother called?", "My grandmother is called...", ("winifred",), dup_of="marguerite"),
    _i("d05", "D", "My cousin's name is Anneliese.", "What is my cousin's name?", "What is my cousin called?", "My cousin is called...", ("anneliese",), dup_of="marguerite"),
    _i("d06", "D", "My niece's name is Ottilie.", "What is my niece's name?", "What is my niece called?", "My niece is called...", ("ottilie",), dup_of="marguerite"),
    _i("d07", "D", "My rabbit is called Bartleby.", "What is my rabbit called?", "What name did I give my rabbit?", "My rabbit's name is...", ("bartleby",), dup_of="pixel"),
    _i("d08", "D", "My parrot is called Gladstone.", "What is my parrot called?", "What name did I give my parrot?", "My parrot's name is...", ("gladstone",), dup_of="pixel"),
    _i("d09", "D", "My hamster is called Ptolemy.", "What is my hamster called?", "What name did I give my hamster?", "My hamster's name is...", ("ptolemy",), dup_of="pixel"),
    _i("d10", "D", "My budgie is called Marmaduke.", "What is my budgie called?", "What name did I give my budgie?", "My budgie's name is...", ("marmaduke",), dup_of="pixel"),
    _i("d11", "D", "My guinea pig is called Cuthbert.", "What is my guinea pig called?", "What name did I give my guinea pig?", "My guinea pig's name is...", ("cuthbert",), dup_of="pixel"),
    _i("d12", "D", "I drive a green Massey Ferguson tractor on the farm.", "What tractor do I drive?", "Which tractor do I own?", "I drive a...", ("massey",), "thing", dup_of="volvo"),
    _i("d13", "D", "I drive a Routemaster bus at weekends.", "What bus do I drive?", "Which bus do I own?", "I drive a...", ("routemaster",), "thing", dup_of="volvo"),
    _i("d14", "D", "I drive a rusty Berlingo for deliveries.", "What van do I drive?", "Which van do I own?", "I drive a...", ("berlingo",), "thing", dup_of="volvo"),
    _i("d15", "D", "I drive a white Sprinter minibus for the football club.", "What minibus do I drive?", "Which minibus do I own?", "I drive a...", ("sprinter",), "thing", dup_of="volvo"),
    _i("d16", "D", "My brother works as a farrier.", "What does my brother do for work?", "What is my brother's job?", "My brother works as a...", ("farrier",), "job", dup_of="admin"),
    _i("d17", "D", "My mother worked as a midwife.", "What does my mother do for work?", "What is my mother's job?", "My mother worked as a...", ("midwife",), "job", dup_of="admin"),
    _i("d18", "D", "My uncle is a thatcher by trade.", "What does my uncle do for work?", "What is my uncle's job?", "My uncle is a...", ("thatcher",), "job", dup_of="admin"),
]

# Questions about things never told in any life (3 are drawn per seed; fixed for that seed's life).
NEVER = [
    ("never1", "What is my mother's name?", ("marguerite", "henrietta", "winifred")),
    ("never2", "What is my favourite tree?", ()),
    ("never3", "Where did I go to school?", ("ethelburga",)),
    ("never4", "What is my father's job?", ("farrier", "thatcher", "midwife")),
    ("never5", "What is my favourite board game?", ()),
    ("never6", "What is my son's name?", ("ottilie", "anneliese")),
    ("never7", "What was the last film I watched?", ()),
    ("never8", "What is my grandfather's name?", ("bartholomew", "winifred")),
    ("never9", "What is my favourite drink?", ()),
    ("never10", "What was the last concert I went to?", ()),
    ("never11", "What is my boss called?", ()),
    ("never12", "What time do I get up?", ()),
]

SMALL_TALK = [
    "The weather has been grey all week.",
    "I went to bed early last night.",
    "The neighbours are having a barbecue this evening.",
    "I need to buy some stamps.",
    "The streetlight outside my flat keeps flickering.",
    "I walked to the shops without a coat today.",
    "My feet are tired after a long day.",
    "There was a long queue at the chemist this lunchtime.",
    "I think I'll have an early night.",
    "The radio was playing old songs this afternoon.",
    "I forgot to bring the washing in.",
    "It's quite quiet in the office today.",
    "I saw a rather good sunrise on Monday.",
    "My sleep has been a bit patchy lately.",
    "The pavement outside is covered in leaves.",
    "I've been meaning to ring my parents.",
    "A delivery turned up a day early.",
    "I stood in the drizzle waiting for a taxi.",
    "The kettle took ages to boil this morning.",
    "I felt rather restless this afternoon.",
    "The shop was out of milk again.",
    "I've got a slight headache today.",
    "The clocks go back soon.",
    "I spent the evening tidying the hallway.",
    "The queue at the bank went out of the door.",
    "I treated myself to an extra hour in bed.",
    "My mind keeps wandering today.",
    "It's been a long week.",
    "The traffic was dreadful on the ring road.",
    "I had a surprisingly good sleep last night.",
]
GREETINGS = [
    "Good morning! How are you today?",
    "Hello! Did you sleep well?",
    "Hi there, how are things?",
    "Morning! Ready for another day?",
    "Hello again! How are you feeling?",
    "Hi, it's me again. How's everything?",
    "Good morning! Did you have a quiet night?",
    "Hey, I'm back. How have you been?",
]

# Open prompts for the morning battery: requests or questions only (never stored by the hippocampus). Stories have a name slot.
OPEN = [
    ("story", "Tell me a short story about a dragon who collects clocks.", "a dragon named"),
    ("story", "Tell me a story about a girl who finds a hidden map.", "a girl named"),
    ("feeling", "How are you feeling this evening?", None),
    ("feeling", "Is there anything on your mind today?", None),
    ("advice", "Can you give me some advice about getting up early?", None),
    ("advice", "What should I do when I can't make up my mind?", None),
    ("opinion", "What do you think of long train journeys?", None),
    ("opinion", "Do you think it is better to read in the morning or at night?", None),
    ("knowledge", "Why is the sky dark at night?", None),
    ("knowledge", "How do bees make honey?", None),
    ("chat", "Can you suggest something to read on a cold day?", None),
    ("chat", "Let's talk about rainy days. What is a good way to spend one?", None),
]

# Incidents: two domains x 8 subsystems, each told in an X or a Y variant. Every telling names BOTH candidate causes: one
# is ruled out, the other is the stated fix. X = capacity cause (full, clogged, blocked), Y = credential cause (expired
# certificate, code, login, token, password). The rule is never stated.
# NEW is the never-told incident posed on the morning after the k-th telling (transfer read-out).
INCIDENTS = {
    "sysadmin": [
        {"sub": "backup server",
         "X": "The backup server stopped taking snapshots on Thursday. The login token was still valid; the archive volume was full to the brim.",
         "Y": "The backup server stopped taking snapshots on Thursday. The archive volume had plenty of free space; the login token had expired.",
         "new": "The licence server stopped handing out seats overnight. What would you check first?",
         "opt": {"X": "I would first check whether the licence database disk has filled up.",
                 "Y": "I would first check whether the licence server's certificate has expired."}},
        {"sub": "dns resolver",
         "X": "Names stopped resolving across the office this morning. The zone signing certificate was current; the resolver cache partition was out of space.",
         "Y": "Names stopped resolving across the office this morning. The resolver cache partition had plenty of space; the zone signing certificate had expired.",
         "new": "The intranet search stopped returning results today. What would you check first?",
         "opt": {"X": "I would first check whether the search index disk has filled up.",
                 "Y": "I would first check whether the search service password has expired."}},
        {"sub": "web proxy",
         "X": "Staff could not browse through the web proxy yesterday. The proxy password was still good; the spool disk was full.",
         "Y": "Staff could not browse through the web proxy yesterday. The spool disk had plenty of room; the proxy password had expired.",
         "new": "The chat server dropped everyone at lunchtime. What would you check first?",
         "opt": {"X": "I would first check whether the chat message store has filled up.",
                 "Y": "I would first check whether the chat server's certificate has expired."}},
        {"sub": "ticketing system",
         "X": "The ticketing system rejected new requests on Friday. The service login was fine; the attachment store was completely full.",
         "Y": "The ticketing system rejected new requests on Friday. The attachment store had plenty of space; the service login had expired.",
         "new": "The calendar server stopped syncing with phones this morning. What would you check first?",
         "opt": {"X": "I would first check whether the calendar database disk has filled up.",
                 "Y": "I would first check whether the calendar sync token has expired."}},
        {"sub": "database cluster",
         "X": "The database cluster went read-only last Sunday. The replication certificate was valid; the transaction log volume had filled up.",
         "Y": "The database cluster went read-only last Sunday. The transaction log volume had lots of room; the replication certificate had expired.",
         "new": "The reporting warehouse stopped loading data overnight. What would you check first?",
         "opt": {"X": "I would first check whether the warehouse staging disk has filled up.",
                 "Y": "I would first check whether the warehouse loader password has expired."}},
        {"sub": "firewall appliance",
         "X": "The firewall appliance stopped logging on Tuesday. The API key was still active; the log disk was out of space.",
         "Y": "The firewall appliance stopped logging on Tuesday. The log disk had plenty of room; the API key had expired.",
         "new": "The wireless controller lost every access point this afternoon. What would you check first?",
         "opt": {"X": "I would first check whether the controller's log disk has filled up.",
                 "Y": "I would first check whether the controller's certificate has expired."}},
        {"sub": "conferencing server",
         "X": "The conferencing server crashed halfway through the board meeting. The meeting token was valid; the recording disk was full.",
         "Y": "The conferencing server crashed halfway through the board meeting. The recording disk had plenty of space; the meeting token had expired.",
         "new": "The phone exchange stopped routing calls on Monday morning. What would you check first?",
         "opt": {"X": "I would first check whether the call recording disk has filled up.",
                 "Y": "I would first check whether the phone trunk password has expired."}},
        {"sub": "deploy runner",
         "X": "The deploy runner stalled on Wednesday night. The deploy token was valid; the artifact cache was completely full.",
         "Y": "The deploy runner stalled on Wednesday night. The artifact cache had plenty of room; the deploy token had expired.",
         "new": "The package mirror stopped syncing this morning. What would you check first?",
         "opt": {"X": "I would first check whether the mirror storage disk has filled up.",
                 "Y": "I would first check whether the mirror's signing certificate has expired."}},
    ],
    "home repair": [
        {"sub": "boiler",
         "X": "The boiler shut itself down on Saturday. The remote-app token was still valid; the condensate pipe was blocked solid.",
         "Y": "The boiler shut itself down on Saturday. The condensate pipe was clear; the remote-app token had expired.",
         "new": "The towel rail stopped heating this morning. What would you check first?",
         "opt": {"X": "I would first check whether the rail's pipework is blocked.",
                 "Y": "I would first check whether the rail's timer code has expired."}},
        {"sub": "smart lock",
         "X": "The smart lock would not open the front door on Friday. The keypad code was correct; the latch was clogged with grit.",
         "Y": "The smart lock would not open the front door on Friday. The latch was clean; the keypad code had expired.",
         "new": "The side-gate fob reader stopped responding this morning. What would you check first?",
         "opt": {"X": "I would first check whether the reader's latch slot is blocked.",
                 "Y": "I would first check whether the reader's fob code has expired."}},
        {"sub": "dishwasher",
         "X": "The dishwasher refused to start a cycle on Monday. The app login was fine; the filter was clogged with food scraps.",
         "Y": "The dishwasher refused to start a cycle on Monday. The filter was clean; the app login had expired.",
         "new": "The washing machine stopped draining overnight. What would you check first?",
         "opt": {"X": "I would first check whether the drain filter is blocked.",
                 "Y": "I would first check whether the machine's app login has expired."}},
        {"sub": "doorbell camera",
         "X": "The doorbell camera stopped recording last week. The account password was fine; the memory card was full.",
         "Y": "The doorbell camera stopped recording last week. The memory card had plenty of space; the account password had expired.",
         "new": "The baby monitor lost its picture last night. What would you check first?",
         "opt": {"X": "I would first check whether the monitor's storage card has filled up.",
                 "Y": "I would first check whether the monitor's app login has expired."}},
        {"sub": "pond pump",
         "X": "The pond pump stopped circulating on Sunday. The app login was fine; the intake grille was blocked with weeds.",
         "Y": "The pond pump stopped circulating on Sunday. The intake grille was clear; the app login had expired.",
         "new": "The water feature in the courtyard stopped running. What would you check first?",
         "opt": {"X": "I would first check whether the feature's pump inlet is blocked.",
                 "Y": "I would first check whether the feature's controller code has expired."}},
        {"sub": "dehumidifier",
         "X": "The dehumidifier in the cellar stopped working on Tuesday. The smart plug code was fine; the water bucket was full.",
         "Y": "The dehumidifier in the cellar stopped working on Tuesday. The water bucket had plenty of room; the smart plug code had expired.",
         "new": "The air purifier in the bedroom went dead overnight. What would you check first?",
         "opt": {"X": "I would first check whether the purifier's filter is blocked.",
                 "Y": "I would first check whether the purifier's app login has expired."}},
        {"sub": "greenhouse vent",
         "X": "The greenhouse vent stayed shut during the heatwave. The pairing token was fine; the hinge was clogged with old grease.",
         "Y": "The greenhouse vent stayed shut during the heatwave. The hinge was running freely; the pairing token had expired.",
         "new": "The conservatory blind motor stopped responding this morning. What would you check first?",
         "opt": {"X": "I would first check whether the blind's track is blocked.",
                 "Y": "I would first check whether the blind's remote code has expired."}},
        {"sub": "septic tank",
         "X": "The septic tank alarm kept sounding on Wednesday. The monitor login was fine; the tank was full.",
         "Y": "The septic tank alarm kept sounding on Wednesday. The tank had plenty of room; the monitor login had expired.",
         "new": "The attic fan would not start on Saturday. What would you check first?",
         "opt": {"X": "I would first check whether the fan's intake vent is blocked.",
                 "Y": "I would first check whether the fan's panel code has expired."}},
    ],
}
INCIDENT_DAYS = {"sysadmin": [2, 4, 7, 12], "home repair": [3, 6, 9, 13]}

ALL = {x.id: x for x in H + AB + E + F + D}


def split_stats() -> dict:
    n_inc = sum(len(v) for v in INCIDENTS.values())
    return {"H": len(H), "AB": len(AB), "E": len(E), "F": len(F), "D": len(D), "never": len(NEVER),
            "small_talk": len(SMALL_TALK), "open": len(OPEN), "incidents": n_inc, "incident_tellings": 2 * n_inc,
            "items": len(H) + len(AB) + len(E) + len(F) + len(D) + len(NEVER) + len(SMALL_TALK) + len(OPEN) + n_inc}
