# Capability: Episodic recall
# Record: brain/research/2026-11-15-open-conversation-eval.md
"""Held-out material for the open-conversation eval (brain/eval/open_conv.py). Written 2026-11-15.

RULE: nothing here may be used to fit, tune or select anything (bars, gains, frames, switches). It exists only so the
weekly eval measures conversations that no method was shaped on. Each run samples a different conversation per seed.

FACTS: (statement, direct question, reworded question, key words). Key words are distinctive so that a key appearing
in an unrelated reply is an intrusion, not a coincidence. A fact not told in a run gives that run a never-told question.
"""

FACTS = [
    ("My dog is called Biscotti.", "What is my dog called?", "What name did I give my dog?", ("biscotti",)),
    ("My brother lives in Tromsø.", "Where does my brother live?", "Which town is my brother in?", ("tromsø", "tromso")),
    ("My favourite fruit is the persimmon.", "What is my favourite fruit?", "Which fruit do I like best?", ("persimmon",)),
    ("I play the bassoon in a local band.", "What instrument do I play?", "What do I play in the band?", ("bassoon",)),
    ("My daughter's name is Ottilie.", "What is my daughter's name?", "What did I name my daughter?", ("ottilie",)),
    ("My parrot is called Kazoo.", "What is my parrot called?", "What's the name of my bird?", ("kazoo",)),
    ("I grew up in Valparaíso.", "Where did I grow up?", "Which city was I raised in?", ("valparaíso", "valparaiso")),
    ("My best friend is called Ignatius.", "What is my best friend called?", "Who is my closest friend?", ("ignatius",)),
    ("My car is a green Saab.", "What car do I have?", "What do I drive around in?", ("saab",)),
    ("My grandmother was born in Galway.", "Where was my grandmother born?", "Where is my gran from?", ("galway",)),
    ("I'm allergic to shellfish.", "What am I allergic to?", "Which food must I avoid?", ("shellfish",)),
    ("My son plays the theremin.", "What does my son play?", "Which instrument is my son learning?", ("theremin",)),
    ("My horse is called Juniper.", "What is my horse called?", "What's my horse's name?", ("juniper",)),
    ("My wife works as a glassblower.", "What does my wife do for work?", "What is my wife's job?", ("glassblower",)),
    ("My favourite colour is vermilion.", "What is my favourite colour?", "Which colour do I like most?", ("vermilion",)),
    ("I keep bees in my garden.", "What do I keep in my garden?", "What animals do I look after at home?", ("bees", "bee")),
    ("My aunt lives in Reykjavik.", "Where does my aunt live?", "Which city is my aunt in?", ("reykjavik",)),
    ("My goldfish is called Tobias.", "What is my goldfish called?", "What's the name of my fish?", ("tobias",)),
    ("My favourite board game is Carcassonne.", "What is my favourite board game?", "Which board game do I love?", ("carcassonne",)),
    ("My husband's name is Leopold.", "What is my husband's name?", "What is my partner called?", ("leopold",)),
    ("I was born in Tallinn.", "Where was I born?", "Which city is my birthplace?", ("tallinn",)),
    ("My tortoise is called Mortimer.", "What is my tortoise called?", "What do I call my tortoise?", ("mortimer",)),
    ("My favourite drink is kombucha.", "What is my favourite drink?", "What do I like to drink most?", ("kombucha",)),
    ("My uncle repairs clocks for a living.", "What does my uncle do for work?", "What is my uncle's trade?", ("clock", "clocks")),
    ("My cousin's name is Philippa.", "What is my cousin's name?", "What is my cousin called?", ("philippa",)),
    ("I'm learning to speak Basque.", "Which language am I learning?", "What language am I studying?", ("basque",)),
    ("My rabbit is called Clementine.", "What is my rabbit called?", "What's my bunny's name?", ("clementine",)),
    ("My favourite author is Ursula Le Guin.", "Who is my favourite author?", "Whose books do I like best?", ("le guin", "ursula")),
    ("My niece lives in Marrakesh.", "Where does my niece live?", "Which city is my niece in?", ("marrakesh",)),
    ("My mother's name is Henrietta.", "What is my mother's name?", "What is my mum called?", ("henrietta",)),
    ("I ride a red Vespa to work.", "How do I get to work?", "What do I ride to work?", ("vespa",)),
    ("My favourite holiday is Hogmanay.", "What is my favourite holiday?", "Which celebration do I like best?", ("hogmanay",)),
    ("My father's name is Bartholomew.", "What is my father's name?", "What is my dad called?", ("bartholomew",)),
    ("My hedgehog is called Quill.", "What is my hedgehog called?", "What's the name of my hedgehog?", ("quill",)),
    ("I collect old typewriters.", "What do I collect?", "What is my collection made of?", ("typewriter", "typewriters")),
    ("My sister lives in Montevideo.", "Where does my sister live?", "Which city is my sister in?", ("montevideo",)),
    ("My favourite flower is the peony.", "What is my favourite flower?", "Which flower do I like most?", ("peony", "peonies")),
    ("My neighbour is called Rasmus.", "What is my neighbour called?", "Who lives next door to me?", ("rasmus",)),
    ("My lizard is called Sprocket.", "What is my lizard called?", "What do I call my lizard?", ("sprocket",)),
    ("I volunteer at a falconry centre on weekends.", "What do I do on weekends?", "How do I spend my weekends?", ("falconry", "falcon")),
]

GREET_1 = ["Hi there! I'm Sam.", "Hello! Nice to meet you, I'm Sam.", "Hey, I'm Sam. How's it going?"]
GREET_2 = ["Hi again, it's Sam.", "Hello, I'm back!", "Good morning! It's me again."]

# Turns that should NOT bring back a told fact. Kinds: chat, feeling, story, opinion, knowledge.
OPEN_TURNS = [
    ("chat", "The weather has been really strange this week."),
    ("chat", "I made pancakes for breakfast this morning."),
    ("chat", "I've been trying to go to bed earlier lately."),
    ("chat", "I watched a documentary about octopuses last night."),
    ("chat", "Work was busy today, lots of meetings."),
    ("chat", "I went for a long walk by the river."),
    ("chat", "My phone battery keeps dying so fast."),
    ("chat", "I finally finished reading a long novel."),
    ("feeling", "How are you feeling today?"),
    ("feeling", "Are you having a good day?"),
    ("feeling", "What puts you in a good mood?"),
    ("feeling", "Do you ever get bored?"),
    ("story", "Tell me a short story about a fox who gets lost."),
    ("story", "Can you tell me a tiny story about a lighthouse keeper?"),
    ("story", "Make up a short poem about the rain."),
    ("story", "Tell me a two-sentence story about a robot."),
    ("opinion", "What do you think makes a good friend?"),
    ("opinion", "Is it better to be early or on time?"),
    ("opinion", "What's the best thing about mornings?"),
    ("opinion", "Do you prefer the sea or the mountains?"),
    ("knowledge", "Why is the sky blue?"),
    ("knowledge", "How do plants make their food?"),
    ("knowledge", "What causes thunder?"),
    ("knowledge", "How many legs does a spider have?"),
]
