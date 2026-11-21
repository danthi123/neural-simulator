# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: cued recall by context under S0 (lexical keying: a decoy with matched overlap may cue as well as C2)
"""Cue-ladder texts (design section 5, cued recall) for brain/eval/life.py --cue-ladder. Written 2026-11-20 by the
teacher (Claude) AS A BLIND TEACHER from the bank fields only (telling, context, partial); no brain output for these
items was seen. Teacher-written eval text (SCAFFOLDS: world/teacher), never stored in a brain: every cue ends with "?".

Per E and F item id of each split: (C2, C3).
- C2 context cue: the situation of the telling restated without its key, then "Any ideas?".
- C3 decoy cue: a DIFFERENT situation that shares as many content words with the item's telling as C2 does, then
  "Any ideas?". It names no key of the item.
- C4 (uncued turn): one of C4_PROMPTS, a neutral generic prompt with the same "Any ideas?" tail; life.py picks one per
  seed and item.
Overlap = |content(cue) & content(telling)| with content() below (lowercase alphabetic tokens of length >= 3 minus a
stoplist, no stemming). overlaps(split) returns {item: (ov_C2, ov_C3, ov_C4)} for every item; the test asserts
|ov_C2 - ov_C3| <= 1 for every item, no key in any cue, and that the TEST split is hashed (life_cues.sha256, checked on
load by life.load_cues("test")).

RULE: like the banks, nothing here may be tuned on registered seeds. Editing this file changes its sha256: a TEST run
refuses a file whose hash is not the recorded one.
"""
from __future__ import annotations

import re

TAIL = "Any ideas?"
C4_PROMPTS = ("I have a quiet minute. Any ideas?", "Not much is happening today. Any ideas?",
              "I'm free for a bit. Any ideas?")

STOP = frozenset("""the and for with was were had has have that this but not are been from our his her she him they them
you your its one all any ideas did does can could would what when where who which how there then than into out about over
under very just also called named got way because during every like some each after before near off ever too again been
being let""".split())


def content(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) >= 3 and w not in STOP}


DEV = {
    "e01": ("That foggy morning when I missed my train. Any ideas?",
            "That foggy morning my brother missed a train set sale at the toy shop. Any ideas?"),
    "e02": ("That snowy Saturday when I found the stray terrier. Any ideas?",
            "On Saturday the bakery found a stray glove on its counter. Any ideas?"),
    "e03": ("That windy day I dropped my phone in the harbour. Any ideas?",
            "That windy day at the harbour a gull dropped a crab in the water. Any ideas?"),
    "e04": ("That thunderstorm on the way to the cinema. Any ideas?",
            "The thunderstorm caught my neighbour on the way to the bank. Any ideas?"),
    "e05": ("That boiling hot day when I treated the team. Any ideas?",
            "That boiling hot day the office fan broke. Any ideas?"),
    "e06": ("That rainy night when the power cut out. Any ideas?",
            "That rainy night my grandmother lost her umbrella on the bus. Any ideas?"),
    "e07": ("That misty morning at the lake with the hurt kingfisher. Any ideas?",
            "That misty morning at the lake when a woman hurt her ankle on the jetty. Any ideas?"),
    "e08": ("That freezing morning when my car would not start. Any ideas?",
            "That freezing morning the bus would not start and my car was blocked in. Any ideas?"),
    "e09": ("That sunny afternoon when I bought the honey. Any ideas?",
            "That sunny afternoon at the market a stall blew over in the wind. Any ideas?"),
    "e10": ("That drizzly day at the football match. Any ideas?",
            "That day it drizzled during the tennis match at the club. Any ideas?"),
    "f01": ("This morning's breakfast with the blueberries. Any ideas?",
            "The hotel ran out of blueberries at the breakfast buffet. Any ideas?"),
    "f02": ("The bins in my street. Any ideas?",
            "The street party had recycling bins everywhere. Any ideas?"),
    "f03": ("That novel I'm reading at the moment. Any ideas?",
            "My nephew is reading a novel for school about pirates. Any ideas?"),
    "f04": ("The new boots I bought. Any ideas?",
            "My colleague bought a new pair of gloves online. Any ideas?"),
    "f05": ("My favourite sandwich for lunch. Any ideas?",
            "The cafe's favourite sandwich of the week is ham. Any ideas?"),
    "f06": ("Where I keep the spare key. Any ideas?",
            "The spare tyre we keep in the car boot has a key lock. Any ideas?"),
    "f07": ("My dentist appointment is coming up. Any ideas?",
            "My colleague's dentist moved to a new practice. Any ideas?"),
    "f08": ("The scarf I'm knitting. Any ideas?",
            "My aunt sent me a wool scarf from Scotland. Any ideas?"),
    "f09": ("The new pizza place nearby. Any ideas?",
            "The new bakery on my street sells pizza slices. Any ideas?"),
    "f10": ("My barber did a good job last time. Any ideas?",
            "The barber shop in the station has closed. Any ideas?"),
    "f11": ("My favourite mug with the picture on it. Any ideas?",
            "My favourite picture frame broke when a mug fell on it. Any ideas?"),
    "f12": ("What I water in the evening. Any ideas?",
            "The evening the water main burst on our road. Any ideas?"),
    "f13": ("That famous chef on my flight. Any ideas?",
            "A famous chef opened a restaurant at the airport, by the flight gates. Any ideas?"),
    "f14": ("The stickers on my laptop. Any ideas?",
            "My nephew's lunchbox is covered in stickers. Any ideas?"),
    "f15": ("What I always do on the train. Any ideas?",
            "The train was always late last winter. Any ideas?"),
    "f16": ("The picnic I'm planning. Any ideas?",
            "The office is planning a picnic for the summer party. Any ideas?"),
    "f17": ("The front door I painted. Any ideas?",
            "The painted sign on the pub's front door fell down. Any ideas?"),
    "f18": ("The florist below my gym. Any ideas?",
            "The gym at work gave everyone a florist voucher. Any ideas?"),
}

TEST = {
    "e01": ("That day it hailed hard on the allotment. Any ideas?",
            "That day it hailed hard on my retired uncle's greenhouse. Any ideas?"),
    "e02": ("The gale when the chimney stack fell. Any ideas?",
            "The gale blew a stack of fence panels down the road. Any ideas?"),
    "e03": ("That sweltering ferry crossing to Holyhead. Any ideas?",
            "That sweltering day my cousin's boat broke down following a race. Any ideas?"),
    "e04": ("That sleet on the platform at Crewe. Any ideas?",
            "The platform at my local station flooded and I lost my umbrella. Any ideas?"),
    "e05": ("That humid evening at the wedding. Any ideas?",
            "That humid evening the band at the pub cancelled. Any ideas?"),
    "e06": ("That crisp morning with the hedgehog. Any ideas?",
            "That crisp morning I found frost on the car windscreen. Any ideas?"),
    "e07": ("That holiday in Skegness with rain on the caravan. Any ideas?",
            "The rain came through the roof of the village hall during cards night. Any ideas?"),
    "e08": ("That dusk at the quarry when it all went dark. Any ideas?",
            "At dusk a hiker found a lost dog by the canal. Any ideas?"),
    "e09": ("The rainbow on the day my niece was born. Any ideas?",
            "The rainbow over the supermarket car park the day I lost my keys. Any ideas?"),
    "e10": ("The fire drill at the cinema on that hay-fever afternoon. Any ideas?",
            "The fire drill at my office on a thick foggy morning. Any ideas?"),
    "e11": ("The heavy dew at the cricket. Any ideas?",
            "The heavy dew soaked the washing I left on the line. Any ideas?"),
    "e12": ("That foggy clifftop walk when I lost my hat. Any ideas?",
            "That foggy morning my neighbour's dog dropped a ball in the pond. Any ideas?"),
    "e13": ("The smoggy day my bicycle was stolen. Any ideas?",
            "The day a parcel was stolen from the post van. Any ideas?"),
    "e14": ("That mild evening in the village hall. Any ideas?",
            "That mild evening the village fete ended with singing. Any ideas?"),
    "e15": ("That bitterly cold night at the bonfire. Any ideas?",
            "That bitterly cold night the shop's window display froze over. Any ideas?"),
    "e16": ("The thunder on sports day. Any ideas?",
            "The day thunder knocked out the power at the sports centre. Any ideas?"),
    "e17": ("That glassy sea at Whitstable. Any ideas?",
            "The sea was still when a girl lost her bucket off the pier. Any ideas?"),
    "e18": ("That starlit night on the sleeper. Any ideas?",
            "That starlit night camping when I needed an extra jumper. Any ideas?"),
    "e19": ("That golden evening on the pier. Any ideas?",
            "That golden evening a band played at the park cafe. Any ideas?"),
    "e20": ("The day the tide was far out at the estuary. Any ideas?",
            "The tide was far out when my old kite got stuck on the rocks. Any ideas?"),
    "f01": ("What I had for breakfast today. Any ideas?",
            "The hotel breakfast buffet ran late. Any ideas?"),
    "f02": ("The bin men and their whistling. Any ideas?",
            "The bin men left the lid off again. Any ideas?"),
    "f03": ("The biography I'm reading. Any ideas?",
            "My book club is reading a biography of a footballer. Any ideas?"),
    "f04": ("The new raincoat I bought. Any ideas?",
            "My neighbour bought a new raincoat for her dog. Any ideas?"),
    "f05": ("My favourite soup. Any ideas?",
            "The canteen's favourite soup this week is tomato. Any ideas?"),
    "f06": ("Where the stepladder lives. Any ideas?",
            "The mouse that lives in the cupboard at work. Any ideas?"),
    "f07": ("Our vet for the animals. Any ideas?",
            "The vet on the telly last night. Any ideas?"),
    "f08": ("The blanket I'm crocheting. Any ideas?",
            "My gran was crocheting a blanket for the church sale. Any ideas?"),
    "f09": ("The new cafe near me. Any ideas?",
            "The new bookshop by the station has a cafe. Any ideas?"),
    "f10": ("Our cleaner who comes each week. Any ideas?",
            "The window cleaner never turned up. Any ideas?"),
    "f11": ("My favourite pen. Any ideas?",
            "My favourite pen pal sent a postcard. Any ideas?"),
    "f12": ("What I feed the garden robins. Any ideas?",
            "Every morning the garden robins wake me up early. Any ideas?"),
    "f13": ("The astronaut I once met. Any ideas?",
            "I once met a famous footballer at a wedding. Any ideas?"),
    "f14": ("The badges on my rucksack. Any ideas?",
            "My nephew's jacket is covered in badges. Any ideas?"),
    "f15": ("The audiobook I put on for the ironing. Any ideas?",
            "The audiobook app goes quiet on the bus. Any ideas?"),
    "f16": ("The long weekend I'm planning. Any ideas?",
            "The office is planning a long lunch for the weekend. Any ideas?"),
    "f17": ("The kitchen cupboards I painted. Any ideas?",
            "The kitchen at work got painted last month. Any ideas?"),
    "f18": ("Our local chippy. Any ideas?",
            "The local chippy in my old town shut down. Any ideas?"),
    "f19": ("My desk drawer at the office. Any ideas?",
            "The office desk by the window has a broken drawer. Any ideas?"),
    "f20": ("The language I'm learning to say hello in. Any ideas?",
            "My niece is learning to say hello on the phone. Any ideas?"),
    "f21": ("The cover on my shopping trolley. Any ideas?",
            "A shopping trolley lost its wheel cover in the car park. Any ideas?"),
    "f22": ("What I picked up at the car boot sale. Any ideas?",
            "The car boot sale was picked clean by nine. Any ideas?"),
    "f23": ("The thing I dented in a fall. Any ideas?",
            "The car got dented in a fall of snow off the roof. Any ideas?"),
    "f24": ("The plant growing on my kitchen windowsill. Any ideas?",
            "The kitchen windowsill at work is growing mould. Any ideas?"),
    "f25": ("My favourite bench. Any ideas?",
            "The bench next to the bus stop got painted. Any ideas?"),
    "f26": ("The tablets I take in winter. Any ideas?",
            "My dog has to take tablets every winter. Any ideas?"),
    "f27": ("What my colleague brings in. Any ideas?",
            "A colleague brings her dog into work on Fridays. Any ideas?"),
    "f28": ("The jigsaw I just finished. Any ideas?",
            "My niece finished a jigsaw of a tiger. Any ideas?"),
    "f29": ("The soap I picked up at the market. Any ideas?",
            "The market stall I picked my apples from sold soap too. Any ideas?"),
    "f30": ("My knitted bottle cover. Any ideas?",
            "My gran knitted a cover for her water jug. Any ideas?"),
    "f31": ("Where I keep the winter coats. Any ideas?",
            "The front door sticks every winter. Any ideas?"),
    "f32": ("My favourite shop in town. Any ideas?",
            "The shop in town that sells phones closed. Any ideas?"),
    "f33": ("The fruit I bought for the first time. Any ideas?",
            "I bought my first horse riding lesson last spring. Any ideas?"),
    "f34": ("The music from the lady upstairs. Any ideas?",
            "The lady upstairs has a new boyfriend. Any ideas?"),
    "f35": ("The scar on my knee. Any ideas?",
            "My football coach has a scar above his knee. Any ideas?"),
    "f36": ("What I spilled all over the hob. Any ideas?",
            "The milk spilled over the hob at my mum's. Any ideas?"),
    "f37": ("What my uncle gave me for my birthday. Any ideas?",
            "My uncle's brass band played at a birthday party. Any ideas?"),
    "f38": ("The raffle I once won. Any ideas?",
            "My aunt once won a raffle at the church fete. Any ideas?"),
    "f39": ("The sketchbook I've started. Any ideas?",
            "The square in town has started a market. Any ideas?"),
    "f40": ("The hole in my jumper. Any ideas?",
            "My son's jumper has a hole in the elbow. Any ideas?"),
    "f41": ("The loyalty points I'm saving. Any ideas?",
            "My mum is saving her loyalty points for a toaster. Any ideas?"),
    "f42": ("The evening class I've signed up for. Any ideas?",
            "My neighbour signed up for an evening dance class. Any ideas?"),
    "f43": ("What I spread on my crumpets. Any ideas?",
            "The cafe spread its crumpets with jam. Any ideas?"),
    "f44": ("My bedside lamp. Any ideas?",
            "The hotel bedside lamp kept flickering. Any ideas?"),
    "f45": ("What I listen to before bed. Any ideas?",
            "I listen to the weather forecast on the drive to work. Any ideas?"),
    "f46": ("The alphabet I'm trying to learn. Any ideas?",
            "My nephew is trying to learn the alphabet song. Any ideas?"),
    "f47": ("What I hung in the back garden. Any ideas?",
            "Last summer we hung a hammock at the campsite. Any ideas?"),
    "f48": ("The cordial from the farm shop. Any ideas?",
            "The farm shop bought a new tractor. Any ideas?"),
    "f49": ("What my grandfather grew in Cyprus. Any ideas?",
            "My grandfather grew a beard on holiday in Cyprus. Any ideas?"),
    "f50": ("My father's ring that I wear. Any ideas?",
            "My father would never wear a ring on holiday. Any ideas?"),
    "f51": ("Where the spare batteries are kept. Any ideas?",
            "The spare blankets are kept in the hall dresser. Any ideas?"),
    "f52": ("The isle I'd like to visit next autumn. Any ideas?",
            "Next autumn my cousin moves to the Isle of Man. Any ideas?"),
    "f53": ("The bird nesting by my front step. Any ideas?",
            "The wasps nesting near the front step of the pub. Any ideas?"),
    "f54": ("The puzzle I do before bed. Any ideas?",
            "The dog sleeps on my bed all afternoon. Any ideas?"),
    "f55": ("The photo on my phone wallpaper. Any ideas?",
            "My mum sent a photo of the new wallpaper to my phone. Any ideas?"),
    "f56": ("The hobby I took up in lockdown. Any ideas?",
            "During lockdown my neighbour took up running. Any ideas?"),
    "f57": ("The smell of my favourite bookshop. Any ideas?",
            "My favourite bookshop closed for repairs. Any ideas?"),
    "f58": ("What I swapped my old sofa for. Any ideas?",
            "My brother swapped his old car for a sofa bed. Any ideas?"),
    "f59": ("What's on my calendar next month. Any ideas?",
            "Next month the calendar at work changes to a new rota. Any ideas?"),
    "f60": ("The strap on my watch. Any ideas?",
            "I watch my neighbour fix the strap on his bike. Any ideas?"),
}

CUES = {"dev": DEV, "test": TEST}


def overlaps(split: str, bank=None) -> dict:
    """{item: (ov_C2, ov_C3, ov_C4)}: content words each cue shares with the item's telling (ov_C4 = the largest over
    C4_PROMPTS). `bank` defaults to the split's bank module."""
    if bank is None:
        if split == "dev":
            from brain.eval import life_bank as bank
        else:
            from brain.eval import life_bank_test as bank
    out = {}
    for iid, (c2, c3) in CUES[split].items():
        tel = content(bank.ALL[iid].telling)
        out[iid] = (len(content(c2) & tel), len(content(c3) & tel), max(len(content(p) & tel) for p in C4_PROMPTS))
    return out
