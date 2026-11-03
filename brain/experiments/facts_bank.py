# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md (B4: >= 40 facts, confusables, paraphrases)
"""48 told facts for the selection benchmark (teacher-written: a declared scaffold, SCAFFOLDS.md).

Each row: (statement, same-words question, light rewording, heavy rewording, answer keys). The first 12 are
brain/experiments/facts.py (light rewording = its PARA). Rows 13-24 and several later rows are CONFUSABLES of
earlier ones (same template, another person or attribute), so a key that matches on "my ... cat" alone fails.
NEVER_TOLD are questions about things never told (the hippocampus should not fire).
"""
FACTS48 = [
    ("My cat is called Pixel.", "What is my cat called?", "What's the name of my kitty?", "What do I call my feline pet?", ("pixel",)),
    ("My favourite colour is teal.", "What is my favourite colour?", "Which colour do I like best?", "Which shade do I prefer above all others?", ("teal",)),
    ("I was born in Lyon.", "Where was I born?", "What's my hometown?", "In which city did my life begin?", ("lyon",)),
    ("My sister's name is Marguerite.", "What is my sister's name?", "Who's my sister?", "What do people call my female sibling?", ("marguerite",)),
    ("My dog is named Biscuit.", "What is my dog's name?", "What's my dog called?", "What do I call my puppy?", ("biscuit",)),
    ("I drive a red Volvo.", "What car do I drive?", "What kind of car is mine?", "Which vehicle do I own?", ("volvo",)),
    ("My best friend is called Tomasz.", "Who is my best friend?", "Who's my closest friend?", "Which pal am I nearest to?", ("tomasz",)),
    ("I work at a bakery called Crumb.", "Where do I work?", "What's my workplace?", "Where is my job?", ("crumb",)),
    ("My brother plays the cello.", "What instrument does my brother play?", "Which instrument does my brother play?", "What music does my male sibling make?", ("cello",)),
    ("I'm allergic to peanuts.", "What am I allergic to?", "Is there any food I can't eat?", "Which food makes me sick?", ("peanut",)),
    ("My grandmother lives in Oslo.", "Where does my grandmother live?", "Which city is my grandma in?", "Where is my nan's home?", ("oslo",)),
    ("My son's birthday is in March.", "When is my son's birthday?", "What month was my son born?", "When does my boy turn a year older?", ("march",)),
    # confusables of 1-12
    ("My neighbour's cat is called Mango.", "What is my neighbour's cat called?", "What's the name of the kitty next door?", "What do the people next door call their feline?", ("mango",)),
    ("My daughter's birthday is in October.", "When is my daughter's birthday?", "What month was my daughter born?", "When does my girl turn a year older?", ("october",)),
    ("My grandfather lives in Dublin.", "Where does my grandfather live?", "Which city is my grandpa in?", "Where is my granddad's home?", ("dublin",)),
    ("My wife drives a blue Fiat.", "What car does my wife drive?", "What kind of car is my wife's?", "Which vehicle does my spouse own?", ("fiat",)),
    ("My mother plays the harp.", "What instrument does my mother play?", "Which instrument does my mum play?", "What music does my mom make?", ("harp",)),
    ("My least favourite colour is beige.", "What is my least favourite colour?", "Which colour do I like least?", "Which shade do I dislike most?", ("beige",)),
    ("My partner works at a garage called Sprocket.", "Where does my partner work?", "What's my partner's workplace?", "Where is my partner's job?", ("sprocket",)),
    ("My uncle's dog is named Rufus.", "What is my uncle's dog called?", "What's my uncle's dog's name?", "What does my uncle call his puppy?", ("rufus",)),
    ("My cousin is allergic to shellfish.", "What is my cousin allergic to?", "Is there any food my cousin can't eat?", "Which food makes my cousin sick?", ("shellfish",)),
    ("My father was born in Porto.", "Where was my father born?", "What's my dad's hometown?", "In which city did my dad's life begin?", ("porto",)),
    ("My boss is called Henrietta.", "Who is my boss?", "Who's my manager?", "Who do I report to at work?", ("henrietta",)),
    ("My brother's name is Anselm.", "What is my brother's name?", "Who's my brother?", "What do people call my male sibling?", ("anselm",)),
    # further facts (some confusable with the above)
    ("My favourite food is lasagna.", "What is my favourite food?", "Which dish do I like best?", "What meal do I enjoy most?", ("lasagna",)),
    ("I have a parrot named Kiwi.", "What is my parrot called?", "What's my bird's name?", "What do I call my feathered pet?", ("kiwi",)),
    ("My favourite sport is fencing.", "What is my favourite sport?", "Which sport do I like best?", "What athletic activity do I enjoy most?", ("fencing",)),
    ("I'm learning to speak Finnish.", "What language am I learning?", "Which language am I studying?", "What tongue am I trying to pick up?", ("finnish",)),
    ("My favourite book is Dune.", "What is my favourite book?", "Which novel do I like best?", "What do I most enjoy reading?", ("dune",)),
    ("I keep a cactus called Spike.", "What is my cactus called?", "What's my cactus's name?", "What do I call my prickly plant?", ("spike",)),
    ("My first car was a Saab.", "What was my first car?", "Which car did I own first?", "What vehicle did I start driving in?", ("saab",)),
    ("I went to school in Bristol.", "Where did I go to school?", "Where did I study as a kid?", "In which city was I educated?", ("bristol",)),
    ("My favourite drink is ginger tea.", "What is my favourite drink?", "Which drink do I like best?", "What do I most enjoy sipping?", ("ginger",)),
    ("My piano teacher is called Ingrid.", "Who is my piano teacher?", "Who teaches me piano?", "Who gives me keyboard lessons?", ("ingrid",)),
    ("My goldfish is named Bubbles.", "What is my goldfish called?", "What's my fish's name?", "What do I call my little swimmer?", ("bubbles",)),
    ("I play chess every Sunday.", "What do I play every Sunday?", "What game do I play on Sundays?", "What do I do at the end of each week?", ("chess",)),
    ("My lucky number is seventeen.", "What is my lucky number?", "Which number brings me luck?", "What number do I consider fortunate?", ("seventeen", "17")),
    ("My favourite season is autumn.", "What is my favourite season?", "Which time of year do I like best?", "What part of the year do I enjoy most?", ("autumn", "fall")),
    ("My horse is called Thunder.", "What is my horse called?", "What's my horse's name?", "What do I call my pony?", ("thunder",)),
    ("My aunt is a dentist.", "What does my aunt do for a living?", "What is my aunt's job?", "How does my aunt earn money?", ("dentist",)),
    ("My favourite city is Kyoto.", "What is my favourite city?", "Which city do I like best?", "What place do I love most?", ("kyoto",)),
    ("My bike is a green Raleigh.", "What bike do I ride?", "What kind of bicycle is mine?", "What do I cycle on?", ("raleigh",)),
    ("My neighbour is called Ferdinand.", "What is my neighbour's name?", "Who lives next door to me?", "Who is the person beside my house?", ("ferdinand",)),
    ("I studied chemistry at university.", "What did I study at university?", "What was my college subject?", "What did I major in?", ("chemistry",)),
    ("My grandmother's name is Rosalind.", "What is my grandmother's name?", "What's my grandma called?", "Who is my nan?", ("rosalind",)),
    ("My daughter's name is Elodie.", "What is my daughter's name?", "What's my girl called?", "Who is my female child?", ("elodie",)),
    ("My son plays the trumpet.", "What instrument does my son play?", "Which instrument does my boy play?", "What music does my son make?", ("trumpet",)),
    ("My best friend lives in Madrid.", "Where does my best friend live?", "Which city is my closest friend in?", "Where is my pal's home?", ("madrid",)),
]

NEVER_TOLD = ["What is my turtle called?", "Where does my aunt live?", "What is my mother's name?",
              "What instrument does my father play?", "What is my favourite movie?", "Where was my wife born?",
              "What is my rabbit called?", "Which language does my uncle speak?", "What is my nephew's name?",
              "What car does my brother drive?", "When is my wedding anniversary?", "What is my favourite flower?"]

assert len(FACTS48) == 48 and len({f[4][0] for f in FACTS48}) == 48
