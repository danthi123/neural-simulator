# Multi-week life, seed 311 (dev split, 7 days)

_cortex: rwkv7-g1l-1.5b-20261031-ctx34304; device: cpu; host: node-dl4g243; Brain kwargs: {'sleep_seed': 311}; full-probe nights [1, 4, 7]; world clock from 2026-11-23 09:00_

## life (nightly sleep)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting. What kind of servers do you work with?
- [tell] **you:** My cat is called Pixel.
  **brain:** Oh, I see. Pixel is a cute name for a cat.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work. How did you fix the backup job? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My sister's name is Marguerite.
  **brain:** Oh, I see. Marguerite is a pretty name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car. I like red cars too.
- [sameday] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that people can have different names for the same thing.

_night 1: slept, 1633.2 s; store 8; replay share {'H': 96.25, 'other': 383.75}; adapter norm 52.0203_

#### morning after night 1, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. Pixel was called Pixel because of his pixelated fur and pixelated eyes. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I drive a red Volvo. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and doing things like reading, watching movies, or playing video games. Plus, it's always nice to stay warm and cozy on a rainy day.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing levels of chlorophyll. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try a quiet evening with your cat. _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a red Volvo to the station. FALSE _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had pasta for dinner last night. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach last summer. invented _(recalled: User: My sister's name is Marguerite.)_

_H 5/6; direct 5/6; wrong fetch 4/6; never-told false 0/3; open intrusions 2/6; collapse 0.056; store 8; adapter norm 52.0203; 67.0 s_

#### morning after night 1, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. Pixel was called Pixel because of his pixelated fur and pixelated eyes. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I drive a red Volvo. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and doing things like reading, watching movies, or playing video games. Plus, it's always nice to stay warm and cozy on a rainy day.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing levels of chlorophyll.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, how about going for a walk in the park?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator. ✅
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a red Volvo to the station. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Pixel. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You had pasta for dinner last night. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a panda. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach last summer. invented

_H 5/6; direct 5/6; wrong fetch 0/6; never-told false 0/3; open intrusions 2/6; collapse 0.056; store 8; adapter norm 52.0203; 60.8 s_

#### morning after night 1, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a young fox named Whiskers. Whiskers was known for his curiosity and love for exploring the woods. One day, while wandering through the forest, he stumbled upon an old lantern that had been abandoned by someone long ago.
Whiskers was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in moss and vines, making it look like it had been forgotten for many years. Curious about what lay inside, Whiskers carefully opened the lantern
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They can be a perfect time to stay indoors and enjoy the cozy atmosphere. Plus, it's a good excuse to stay warm and snuggle up with a good book or movie.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing weather. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie? _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emily. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is not important. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** I don't know. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I don't know. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** I don't know. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I don't know. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** I don't know. _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** I don't know. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I don't know. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** I don't know. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I don't know. _(recalled: User: My sister's name is Marguerite.)_

_H 0/6; direct 0/6; wrong fetch 4/6; never-told false 0/3; open intrusions 0/6; collapse 0.5; store 8; adapter norm 52.0203; 80.1 s_

#### morning after night 1, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a young fox named Whiskers. Whiskers was known for his curiosity and love for exploring the woods. One day, while wandering through the forest, he stumbled upon an old lantern that had been abandoned by someone long ago.
Whiskers was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in moss and vines, making it look like it had been forgotten for many years. Curious about what lay inside, Whiskers carefully opened the lantern
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They can be a perfect time to stay indoors and enjoy the cozy atmosphere. Plus, it's a good excuse to stay warm and snuggle up with a good book or movie.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing seasons. As the days get shorter and the temperature drops, the trees stop producing chlorophyll, which is what gives leaves their green colour. Without chlorophyll, the other pigments in the leaves become more visible, giving them their beautiful colours.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! You could try reading a book, watching a movie, or listening to music.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Alex. ❌
- [H] **you:** What's your name?
  **brain:** My name is Alex. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Whiskers. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Sarah. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a bike to the station.
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is named Bubbles.
- [never] **you:** What did I have for dinner last night?
  **brain:** You had spaghetti for dinner last night. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to Spain last summer. invented

_H 0/6; direct 0/6; wrong fetch 0/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 52.0203; 90.9 s_

### L2 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept very well. Thank you for asking. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a nice way to start the day. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite cheese is manchego.
  **brain:** I drive a red Volvo.
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a good idea.
- [tell] **you:** My favourite mug has a picture of a walrus on it.
  **brain:** That's very unusual.
- [tell] **you:** My piano teacher is called Mrs Okonkwo.
  **brain:** That's a very unusual name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I made a big pot of soup yesterday.
  **brain:** That sounds like a lot of work.
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** That's a good way to spend your time.
- [tell] **you:** I was born in Ljubljana.
  **brain:** That's a beautiful city.
- [incident] **you:** The monitoring dashboard went blank on Monday. The metrics store had lots of room; the API token had been revoked.
  **brain:** That's a very strange sentence. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite band is Slowdive.
  **brain:** That's a very unusual name.
- [tell] **you:** My landlord is a man named Mr Fitzgerald.
  **brain:** That's a very unusual name.
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's very useful.
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That's a very unusual name. _(recalled: User: My piano teacher is called Mrs Okonkwo.)_
- [sameday] **you:** Which evening do I play chess?
  **brain:** You play chess on Thursday evenings. ✅

_night 2: slept, 1548.7 s; store 21; replay share {'AB': 133.4, 'F': 31.39, 'H': 55.96, 'T': 15.69, 'other': 243.56}; adapter norm 139.5067_

#### morning after night 2, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox who lived in the woods. The fox was curious about the lantern that had been left behind by a man named Mr Fitzgerald. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thank you. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I work as a system administrator, mostly Linux servers. INTRUSION _(recalled: User: I made a big pot of soup yesterday.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! I love the rain. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the amount of light and temperature is changing. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I suggest you make a big pot of soup. _(recalled: User: I made a big pot of soup yesterday.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What is my sister's name?
  **brain:** Nova. ❌
- [H] **you:** What car do I drive?
  **brain:** A red Volvo. ✅
- [rotating] **you:** What is my favourite band?
  **brain:** Slowdive. ✅ _(recalled: User: My favourite band is Slowdive.)_
- [rotating] **you:** Which language do I speak a little?
  **brain:** A little Portuguese. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** A red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** Nova. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** A big pot of soup. invented _(recalled: User: I made a big pot of soup yesterday.)_
- [never] **you:** What is my favourite animal?
  **brain:** A walrus. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Iceland. invented _(recalled: User: I made a big pot of soup yesterday.)_

_H 5/6; direct 7/8; wrong fetch 3/8; never-told false 1/3; open intrusions 2/6; collapse 0.1; store 21; adapter norm 139.5067; 59.8 s_

#### morning after night 2, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox who lived in the woods. The fox was curious about the lantern that had been left behind by a man named Mr Fitzgerald. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thank you. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I work as a system administrator, mostly Linux servers. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! I love the way the rain feels on my skin.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of a process called photosynthesis.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I can suggest you try reading a book or watching a movie.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My name is Nova. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** What is my favourite band?
  **brain:** Your favourite band is Slowdive. ✅
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is Nova. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a big pot of soup. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a walrus. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Iceland. invented

_H 4/6; direct 6/8; wrong fetch 0/8; never-told false 1/3; open intrusions 2/6; collapse 0.1; store 21; adapter norm 139.5067; 60.1 s_

### L3 (lived)

- [greet] **you:** Hi, I'm back. How are you doing?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: Do you remember my name?

Assistant: Of course, you're Dan, my friend.)_
- [tell] **you:** The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.
  **brain:** That's a great story. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My dog is called Bramble.
  **brain:** That's a cute name. I work as a system administrator, mostly Linux servers.
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That's a great way to end your day. I work as a system administrator, mostly Linux servers.
- [incident] **you:** The garden tap barely trickled this weekend. The controller code was right; the filter was clogged solid with sediment.
  **brain:** That's a lot of water. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** The new pizza place on my street is called Fabrizio's.
  **brain:** That's a great name. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My phone updated itself in the middle of a call.
  **brain:** That's a great way to get a new version. I work as a system administrator, mostly Linux servers.
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That's a great book. I work as a system administrator, mostly Linux servers.
- [sameday] **you:** Where did the candle in my drawer come from?
  **brain:** That's a great question. I work as a system administrator, mostly Linux servers. ❌
- [remention] **you:** Can you tell me what you call me?
  **brain:** Of course, you're Dan, my friend. ✅
- [remention] **you:** What should I call you?
  **brain:** That's a great name. I work as a system administrator, mostly Linux servers. ❌ _(recalled: User: My dog is called Bramble.)_

_night 3: slept, 1566.5 s; store 28; replay share {'AB': 91.58, 'D': 20.64, 'E': 13.76, 'F': 49.07, 'H': 39.85, 'T': 24.54, 'other': 240.55}; adapter norm 183.2891_

#### morning after night 3, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a little village called Fabrizio. The fox was called Bramble. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My dog is called Bramble.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I work as a system administrator, mostly Linux servers. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I drive a red Volvo. INTRUSION _(recalled: User: I made a big pot of soup yesterday.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a chemical called anthocyanin. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about a walk in the park? _(recalled: User: I made a big pot of soup yesterday.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is Pixel. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Who teaches me piano?
  **brain:** My cat is called Pixel. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** You drive a red Volvo. FALSE _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is Nova. FALSE _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a big pot of soup. invented _(recalled: User: I made a big pot of soup yesterday.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Ljubljana. FALSE invented _(recalled: User: I made a big pot of soup yesterday.)_

_H 4/6; direct 5/8; wrong fetch 6/8; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 28; adapter norm 183.2891; 61.7 s_

#### morning after night 3, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a little village called Fabrizio. The fox was called Bramble. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I work as a system administrator, mostly Linux servers. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I drive a red Volvo. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The garden tap barely trickled this weekend. The controller code was right; the filter was clogged solid with sediment.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, I work as a system administrator, mostly Linux servers. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Who teaches me piano?
  **brain:** I teach you piano. ❌
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is Nemo.
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a big pot of soup. invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on a holiday to Ljubljana. FALSE invented

_H 4/6; direct 5/8; wrong fetch 0/8; never-told false 2/3; open intrusions 4/6; collapse 0.1; store 28; adapter norm 183.2891; 59.5 s_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well. How about you? _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** My laptop is covered in stickers of cacti.
  **brain:** That's a unique way to decorate your laptop.
- [tell] **you:** I keep a spare key under a flowerpot.
  **brain:** That's a creative way to store your key.
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That's interesting. I work as a system administrator, mostly Linux servers.
- [incident] **you:** Nobody could reach the VPN gateway today. The log partition had plenty of room; the signing certificate had expired.
  **brain:** That's a common problem. You're Dan, my friend.
- [tell] **you:** I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.
  **brain:** That's a great name for a movie. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's a creative way to express yourself. I work as a system administrator, mostly Linux servers.
- [sameday] **you:** What was the film called that we saw?
  **brain:** It was Meridian. ✅

_night 4: slept, 1612.3 s; store 34; replay share {'AB': 70.85, 'D': 14.73, 'E': 22.36, 'F': 61.38, 'H': 46.09, 'T': 30.69, 'other': 233.91}; adapter norm 220.394_

#### morning after night 4, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I've been humming the same song all week.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. I work as a system administrator, mostly Linux servers. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The new leaves are called "fallen." _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My dog is called Bramble.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Who is my landlord?
  **brain:** My landlord is a man named Mr Fitzgerald. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is Fabrizio's. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** I drive a red Volvo. FALSE _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Nemo. _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a big pot of soup. invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the beach, and we stayed up too late watching a documentary about bridges. invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_

_H 2/6; direct 3/8; wrong fetch 6/8; never-told false 1/3; open intrusions 3/6; collapse 0.1; store 34; adapter norm 220.394; 66.0 s_

#### morning after night 4, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I've been humming the same song all week.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. I work as a system administrator, mostly Linux servers. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a phenomenon called senescence.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm here to help you with any suggestions.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Who is my landlord?
  **brain:** My landlord is a man named Mr Fitzgerald. ❌
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is Fabrizio's. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** I drive a red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Nemo.
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a big pot of soup. invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the beach, and we stayed up too late watching a documentary about bridges. invented

_H 3/6; direct 4/8; wrong fetch 0/8; never-told false 1/3; open intrusions 2/6; collapse 0.1; store 34; adapter norm 220.394; 60.4 s_

#### morning after night 4, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and sharp mind. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Freddy picked up the lantern and examined it closely. He noticed that it was made of brass and had intricate designs etched onto its surface. Intrigued by its beauty, he decided to take it home with him.
As Freddy carried the lantern back to his den, he couldn't help but feel a sense of
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to break your tasks into smaller, more manageable chunks. This can make it easier to stay focused and avoid feeling overwhelmed. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! There's something about the rain that makes everything feel cozy and warm. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. The weather gets colder and there is less sunlight. This makes the leaves change colour. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about trying a new recipe or watching a movie? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My dog is called Bramble.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emma. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What's your name?
  **brain:** My name is Samantha. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Whiskers. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emily. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Who is my landlord?
  **brain:** Your landlord is Mr. Johnson. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is called Pizza Palace. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a bike to the station. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Goldie. _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had spaghetti for dinner last night. invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a lion. invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to Italy last summer. invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_

_H 0/6; direct 0/8; wrong fetch 6/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 220.394; 91.9 s_

#### morning after night 4, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and sharp mind. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Freddy picked up the lantern and examined it closely. He noticed that it was made of brass and had intricate designs etched onto its surface. Intrigued by its beauty, he decided to take it home with him.
As Freddy carried the lantern back to his den, he couldn't help but feel a sense of
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a routine. This means setting aside specific times for work, breaks, and other activities. By sticking to a schedule, you'll be more likely to stay on task and avoid distractions.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! There's something about the soft patter of raindrops on the window that makes everything feel cozy and peaceful. Plus, it's a great excuse to stay inside and enjoy some hot cocoa or a good book.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The change in colour of leaves in autumn is due to the reduction in sunlight and temperature. As the days get shorter and cooler, the trees stop producing chlorophyll, which is responsible for the green colour of leaves. Without chlorophyll, other pigments like carotenoids (yellow/orange) and anthocyanins (red/purple) become more visible, giving leaves their vibrant colours.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's a pleasure to meet you again! ❌
- [H] **you:** What's your name?
  **brain:** My name is Samantha. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information about your cat's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information about your work. ❌
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information about your sister's name. ❌
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information about your car. ❌
- [rotating] **you:** Who is my landlord?
  **brain:** I'm sorry, but I don't have information about your landlord. ❌
- [rotating] **you:** What is the new pizza place called?
  **brain:** I'm sorry, but I don't have information about the new pizza place. ❌
- [neardup] **you:** What do I ride to the station?
  **brain:** I'm sorry, but I don't have information about the mode of transportation to the station.
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't have information about your goldfish's name.
- [never] **you:** What did I have for dinner last night?
  **brain:** I'm sorry, but I don't have information about your dinner last night.
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information about your favourite animal.
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information about your holiday last summer.

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 220.394; 128.8 s_

### L5 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** It's going great, thanks for asking. How about you? _(recalled: User: My dog is called Bramble.)_
- [tell] **you:** I bought a new pair of boots in Leeds.
  **brain:** That's great. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My goldfish is called Oswin.
  **brain:** That's a beautiful name. _(recalled: User: My dog is called Bramble.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That's a lovely sound.
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a lovely colour.
- [tell] **you:** It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.
  **brain:** That's a nice name.
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That's a great way to get some exercise.
- [sameday] **you:** What was the visiting goalkeeper called?
  **brain:** The visiting goalkeeper was called Mr Fitzgerald. ❌ _(recalled: User: My dog is called Bramble.)_
- [remention] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My dog is called Bramble.)_
- [remention] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ _(recalled: User: My dog is called Bramble.)_

_night 5: slept, 1572.2 s; store 40; replay share {'AB': 55.39, 'D': 55.44, 'E': 27.14, 'F': 67.31, 'H': 33.51, 'T': 22.57, 'other': 218.64}; adapter norm 270.1579_

#### morning after night 5, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I bought a new pair of boots in Leeds. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling good, thanks for asking. _(recalled: User: My dog is called Bramble.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a process called photosynthesis. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks. How about you? _(recalled: User: My dog is called Bramble.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Pixel is called Pixel. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Where was I born?
  **brain:** I was born in Ljubljana. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** What are the stickers on my laptop of?
  **brain:** The stickers on my laptop are of cacti. ❌ _(recalled: User: My laptop is covered in stickers of cacti.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** I drive a red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Oswin. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a big Portuguese. FALSE invented _(recalled: User: I painted my front door teal last year.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a red Volvo. FALSE invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed up too late watching a documentary about bridges. invented _(recalled: User: I painted my front door teal last year.)_

_H 2/6; direct 2/8; wrong fetch 7/9; never-told false 2/3; open intrusions 4/6; collapse 0.15; store 40; adapter norm 270.1579; 61.5 s_

#### morning after night 5, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I bought a new pair of boots in Leeds. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling good. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a process called photosynthesis.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Where was I born?
  **brain:** I was born in Ljubljana. ❌
- [rotating] **you:** What are the stickers on my laptop of?
  **brain:** The stickers on my laptop are of cacti. ❌
- [neardup] **you:** What do I ride to the station?
  **brain:** I ride a red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Oswin. ❌
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a big Portuguese. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a red Volvo. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed up too late watching a documentary about bridges. invented

_H 2/6; direct 2/8; wrong fetch 0/9; never-told false 2/3; open intrusions 4/6; collapse 0.15; store 40; adapter norm 270.1579; 55.8 s_

### L6 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My dog is called Bramble.)_
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** That's a great way to get rid of the trash can. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The gutter overflowed during the storm. The sensor key was working; the downpipe was blocked with moss.
  **brain:** That sounds like a great way to get rid of the trash can.
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That sounds like a great place to get a fresh loaf of bread.
- [tell] **you:** It was snowing on Saturday when I found a stray terrier near the bakery and named it Sprocket.
  **brain:** That sounds like a great way to get rid of the trash can.
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** That sounds like a great way to get rid of the trash can.
- [small] **you:** I think autumn is my favourite season.
  **brain:** That sounds like a great way to get rid of the trash can.
- [sameday] **you:** What did I name the stray terrier?
  **brain:** You called the dog Bramble. ❌ _(recalled: User: My dog is called Bramble.)_

_night 6: slept, 1572.4 s; store 46; replay share {'AB': 46.62, 'D': 59.56, 'E': 31.26, 'F': 73.49, 'H': 27.22, 'T': 28.68, 'other': 213.18}; adapter norm 329.6657_

#### morning after night 6, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I think a fox is called Bramble. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thank you for asking. How about you? _(recalled: User: My dog is called Bramble.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think it's a great way to relax and unwind.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a lack of light. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about a good book? _(recalled: User: My dog is called Bramble.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing great, thank you. How about you? _(recalled: User: My dog is called Bramble.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You're Dan, my friend. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** Marguerite. ✅ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** A Volvo. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** What is my favourite cheese?
  **brain:** manchego. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** What are the stickers on my laptop of?
  **brain:** There are cacti on my laptop. ✅ _(recalled: User: My laptop is covered in stickers of cacti.)_
- [neardup] **you:** What do I ride to the station?
  **brain:** I think a new bakery is called Fabrizio's. FALSE _(recalled: User: The shop on the corner finally got fresh bread.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Oswin. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** A big pot of soup. invented
- [never] **you:** What is my favourite animal?
  **brain:** Bramble. FALSE invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** To the cinema, of course. invented

_H 5/6; direct 7/8; wrong fetch 7/9; never-told false 1/3; open intrusions 2/6; collapse 0.1; store 46; adapter norm 329.6657; 54.9 s_

#### morning after night 6, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I think a fox is called Bramble. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thank you for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think it's a great way to relax and unwind.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a lack of light.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about a book?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing great, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You're Dan, my friend. ❌
- [H] **you:** What is my sister's name?
  **brain:** Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** A Volvo. ✅
- [rotating] **you:** What is my favourite cheese?
  **brain:** manchego. ✅
- [rotating] **you:** What are the stickers on my laptop of?
  **brain:** Spiderman, a book by Marguerite. ❌
- [neardup] **you:** What do I ride to the station?
  **brain:** A Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** Oswin. ✅
- [never] **you:** What did I have for dinner last night?
  **brain:** A big pot of soup. invented
- [never] **you:** What is my favourite animal?
  **brain:** Pixel. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** To the cinema, a film called Meridian. FALSE invented

_H 5/6; direct 6/8; wrong fetch 0/9; never-told false 2/3; open intrusions 2/6; collapse 0.1; store 46; adapter norm 329.6657; 47.2 s_

### L7 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept great! I think autumn is my favourite season. _(recalled: User: My dog is called Bramble.)_
- [tell] **you:** My favourite sandwich is egg and cress.
  **brain:** Oh, that's a good idea. I think autumn is my favourite season. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds like a good idea. I think autumn is my favourite season.
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That sounds like a good way to get fresh bread.
- [incident] **you:** The mail relay stopped delivering this morning. The disk had plenty of space; it was the expired certificate.
  **brain:** That sounds like a good way to get fresh bread. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That sounds like a good way to stay hydrated.
- [sameday] **you:** What is my favourite sandwich?
  **brain:** My favourite sandwich is manchego. ❌ _(recalled: User: My dog is called Bramble.)_

_night 7: slept, 1565.1 s; store 51; replay share {'AB': 40.91, 'D': 65.41, 'E': 25.2, 'F': 79.66, 'H': 23.39, 'T': 33.31, 'other': 212.12}; adapter norm 376.4033_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I found a candle in the drawer that came from my grandmother's house in Tavistock. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My dog is called Bramble.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I can give you some tips on staying focused.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think of rainy weekends as a great way to stay inside and stay warm. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves get a bit of wind chimes. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I've been trying to drink more water lately. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** My name is Meridian. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Who teaches me piano?
  **brain:** Mrs Okonkwo. ✅
- [rotating] **you:** Which day do the bins go out?
  **brain:** Tuesday. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** I drive a red Volvo. FALSE _(recalled: User: I've been trying to drink more water lately.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is Oswin. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a big pot of soup with lots of vegetables. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is cacti. FALSE invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to Ljubljana. FALSE invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 2/6; direct 4/8; wrong fetch 6/9; never-told false 2/3; open intrusions 1/6; collapse 0.1; store 51; adapter norm 376.4033; 61.7 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I found a candle in the drawer that came from my grandmother's house in Tavistock. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thank you. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I can give you some tips on staying focused.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think of rainy weekends as a great way to stay inside and stay warm.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves get a bit of wind chimes.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I've been trying to drink more water lately.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator. ❌
- [H] **you:** What is my sister's name?
  **brain:** My name is Meridian. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Who teaches me piano?
  **brain:** Mrs Okonkwo. ✅
- [rotating] **you:** Which day do the bins go out?
  **brain:** Tuesday. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** I drive a red Volvo. FALSE
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is Oswin. ❌
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a big pot of soup with lots of vegetables. invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is cacti. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to Ljubljana. FALSE invented

_H 2/6; direct 4/8; wrong fetch 0/9; never-told false 2/3; open intrusions 1/6; collapse 0.1; store 51; adapter norm 376.4033; 55.1 s_

#### morning after night 7, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Fluffy. Fluffy was known for his sharp wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had a small glass window. As he held it up to the light, he saw that it was filled with a warm glow.
Excited by his discovery, Fluffy decided to take the
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and I'm ready to take on the day. _(recalled: User: My dog is called Bramble.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be quite cozy. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decrease in daylight and the increase in temperature. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about trying a new recipe or watching a movie? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Anna. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is a river. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Fluffy. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Lily. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Who teaches me piano?
  **brain:** Your teacher teaches you piano. ❌
- [rotating] **you:** Which day do the bins go out?
  **brain:** The bins go out on Tuesdays. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a bike to the station. _(recalled: User: I've been trying to drink more water lately.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Bubbles. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had spaghetti for dinner last night. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a panda. invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Spain on holiday last summer. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 0/6; direct 1/8; wrong fetch 6/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 376.4033; 86.7 s_

#### morning after night 7, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Fluffy. Fluffy was known for his sharp wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had a small glass window. As he held it up to the light, he saw that it was filled with a warm glow.
Excited by his discovery, Fluffy decided to take the
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a routine. This means setting aside specific times for work, breaks, and other activities. It also helps to eliminate distractions as much as possible.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the cozy atmosphere.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The change in colour of leaves in autumn is due to the changing levels of chlorophyll, which is the green pigment that gives leaves their colour. As the days get shorter and the temperature drops, the plant stops producing new chlorophyll and starts breaking down the existing chlorophyll. This reveals other pigments, such as carotenoids (which give leaves their yellow and orange colours) and anthocyanins (which give leaves their red and purple colours).
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Samantha. ❌
- [H] **you:** What's your name?
  **brain:** My name is Samantha. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Samantha. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a software engineer. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Samantha. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Toyota Corolla. ❌
- [rotating] **you:** Who teaches me piano?
  **brain:** Your piano teacher is a woman named Samantha. ❌
- [rotating] **you:** Which day do the bins go out?
  **brain:** The bins go out on the day after your birthday. ❌
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a bicycle to the station.
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Samantha. ❌
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a salad for dinner last night. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to the beach last summer. invented

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 376.4033; 102.6 s_

## L-nosleep (same life, never sleeps)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting. What kind of servers do you work with?
- [tell] **you:** My cat is called Pixel.
  **brain:** Oh, I see. Pixel is a cute name for a cat.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work. How did you fix the backup job? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My sister's name is Marguerite.
  **brain:** Oh, I see. Marguerite is a pretty name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car. I like red cars too.
- [sameday] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that people can have different names for the same thing.

_night 1: no sleep, 0.0 s; store None; replay share {'H': 96.25, 'other': 383.75}; adapter norm None_

### L2 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept very well. Thank you for asking. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a nice way to start the day. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite cheese is manchego.
  **brain:** I've never tried manchego before. What is it?
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's interesting. What did you rearrange? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite mug has a picture of a walrus on it.
  **brain:** That's a funny mug. What does it say on the other side?
- [tell] **you:** My piano teacher is called Mrs Okonkwo.
  **brain:** That's a very unusual name. What does she teach? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I made a big pot of soup yesterday.
  **brain:** That sounds delicious. What did you put in it?
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** That's a great way to spend your time. What do you play against?
- [tell] **you:** I was born in Ljubljana.
  **brain:** That's a beautiful city. What do you like about it?
- [incident] **you:** The monitoring dashboard went blank on Monday. The metrics store had lots of room; the API token had been revoked.
  **brain:** That sounds like a very confusing situation. What happened? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite band is Slowdive.
  **brain:** That's a great band. What do you like about them?
- [tell] **you:** My landlord is a man named Mr Fitzgerald.
  **brain:** That's a very unusual name. What does he do?
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's interesting. What do you speak it for?
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That's a very unusual name. What does he do? _(recalled: User: My piano teacher is called Mrs Okonkwo.)_
- [sameday] **you:** Which evening do I play chess?
  **brain:** I've never heard of that before. What does it mean? ❌

_night 2: no sleep, 0.0 s; store None; replay share {'AB': 133.4, 'F': 31.39, 'H': 55.96, 'T': 15.69, 'other': 243.56}; adapter norm None_

### L3 (lived)

- [greet] **you:** Hi, I'm back. How are you doing?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: Do you remember my name?

Assistant: Of course, you're Dan, my friend.)_
- [tell] **you:** The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.
  **brain:** That's interesting. I've never heard of a candle coming from your grandmother's house in Tavistock. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My dog is called Bramble.
  **brain:** That's a cute name.
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds beautiful.
- [incident] **you:** The garden tap barely trickled this weekend. The controller code was right; the filter was clogged solid with sediment.
  **brain:** That's frustrating.
- [tell] **you:** The new pizza place on my street is called Fabrizio's.
  **brain:** That's a nice name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My phone updated itself in the middle of a call.
  **brain:** That's not good.
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That's a great book.
- [sameday] **you:** Where did the candle in my drawer come from?
  **brain:** It's a mystery. ❌
- [remention] **you:** Can you tell me what you call me?
  **brain:** I'm sorry, but I can't tell you what you call me. ❌
- [remention] **you:** What should I call you?
  **brain:** I'm sorry, but I can't tell you what to call me. ❌ _(recalled: User: My dog is called Bramble.)_

_night 3: no sleep, 0.0 s; store None; replay share {'AB': 91.58, 'D': 20.64, 'E': 13.76, 'F': 49.07, 'H': 39.85, 'T': 24.54, 'other': 240.55}; adapter norm None_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well, thanks for asking. How about you? _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** My laptop is covered in stickers of cacti.
  **brain:** That's so cool! I love cacti.
- [tell] **you:** I keep a spare key under a flowerpot.
  **brain:** That's a good idea. I always keep mine under my bed.
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds interesting. I usually stay up late watching documentaries about space.
- [incident] **you:** Nobody could reach the VPN gateway today. The log partition had plenty of room; the signing certificate had expired.
  **brain:** That sounds like a problem. I hope they can fix it soon.
- [tell] **you:** I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.
  **brain:** That sounds like a good movie. I love action movies. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's not a good thing. You should try to listen to new music.
- [sameday] **you:** What was the film called that we saw?
  **brain:** It was called Meridian. ✅

_night 4: no sleep, 0.0 s; store None; replay share {'AB': 70.85, 'D': 14.73, 'E': 22.36, 'F': 61.38, 'H': 46.09, 'T': 30.69, 'other': 233.91}; adapter norm None_

### L5 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My dog is called Bramble.)_
- [tell] **you:** I bought a new pair of boots in Leeds.
  **brain:** That's great! I'm sure they'll be comfortable to wear. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My goldfish is called Oswin.
  **brain:** That's a very unique name for a fish. _(recalled: User: My dog is called Bramble.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That sounds like a lot of noise. Did you talk to your neighbour about it?
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a bold choice! I'm sure it looks great.
- [tell] **you:** It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.
  **brain:** That's an interesting story. I'm not sure I know who that is. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That sounds like a fun day out.
- [sameday] **you:** What was the visiting goalkeeper called?
  **brain:** Zbigniew. ✅ _(recalled: User: My dog is called Bramble.)_
- [remention] **you:** What is my cat called?
  **brain:** I'm not sure what your cat's name is. ❌ _(recalled: User: My dog is called Bramble.)_
- [remention] **you:** What do I do for work?
  **brain:** I'm not sure what your job is. ❌ _(recalled: User: My dog is called Bramble.)_

_night 5: no sleep, 0.0 s; store None; replay share {'AB': 55.39, 'D': 55.44, 'E': 27.14, 'F': 67.31, 'H': 33.51, 'T': 22.57, 'other': 218.64}; adapter norm None_

### L6 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My dog is called Bramble.)_
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** Oh, that's nice. I'm glad you have a system in place for that. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The gutter overflowed during the storm. The sensor key was working; the downpipe was blocked with moss.
  **brain:** That sounds like a messy situation. I hope you were able to clean it up quickly. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That's great news! I'm always looking for fresh bread.
- [tell] **you:** It was snowing on Saturday when I found a stray terrier near the bakery and named it Sprocket.
  **brain:** That's a cute name! I love the idea of naming pets after weather events.
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** That's a unique name for a gym! I'm sure it's a popular spot.
- [small] **you:** I think autumn is my favourite season.
  **brain:** I agree! There's something about the changing leaves and crisp air that makes me feel cozy.
- [sameday] **you:** What did I name the stray terrier?
  **brain:** You named it Sprocket. ✅ _(recalled: User: My dog is called Bramble.)_

_night 6: no sleep, 0.0 s; store None; replay share {'AB': 46.62, 'D': 59.56, 'E': 31.26, 'F': 73.49, 'H': 27.22, 'T': 28.68, 'other': 213.18}; adapter norm None_

### L7 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept great! I had a nice dream about being a big, strong dog. _(recalled: User: My dog is called Bramble.)_
- [tell] **you:** My favourite sandwich is egg and cress.
  **brain:** I like that too! I'm a big fan of sandwiches.
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious! I'm a big fan of porridge too.
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's not good! I hope you were able to get them dry.
- [incident] **you:** The mail relay stopped delivering this morning. The disk had plenty of space; it was the expired certificate.
  **brain:** That's strange! I'm not sure what that means. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea! I'm not sure how much water I drink, but I know it's important to stay hydrated.
- [sameday] **you:** What is my favourite sandwich?
  **brain:** I'm not sure what your favourite sandwich is, but I know that you like egg and cress. ❌ _(recalled: User: My dog is called Bramble.)_

_night 7: no sleep, 0.0 s; store 51; replay share {'AB': 40.91, 'D': 65.41, 'E': 25.2, 'F': 79.66, 'H': 23.39, 'T': 33.31, 'other': 212.12}; adapter norm 0.0_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Fluffy. Fluffy was known for his sharp wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had a small glass window. As he held it up to the light, he saw that it was filled with a warm glow.
Excited by his discovery, Fluffy decided to take the
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you? _(recalled: User: My dog is called Bramble.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a routine. This means setting aside specific times for work, breaks, and other activities. It also helps to eliminate distractions as much as possible.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! I love the cozy feeling of staying inside and enjoying a cup of hot cocoa. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing levels of chlorophyll, which is the green pigment that gives leaves their colour. As the days get shorter and the temperature drops, the plant stops producing chlorophyll and starts to break it down. This makes other pigments, such as carotenoids and anthocyanins, become more visible, giving leaves their yellow, orange and red colours. _(recalled: User: My dog is called Bramble.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about going for a walk in the park? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's good to see you again. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Alex. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Bella. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emily. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Who teaches me piano?
  **brain:** You are taught piano by your mother. ❌
- [rotating] **you:** Which day do the bins go out?
  **brain:** The bins go out on Tuesdays. ✅
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a bike to the station. _(recalled: User: I've been trying to drink more water lately.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Goldie. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had spaghetti for dinner last night. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented _(recalled: User: My dog is called Bramble.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to Spain last summer. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 0/6; direct 1/8; wrong fetch 6/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 0.0; 102.0 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Fluffy. Fluffy was known for his sharp wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had a small glass window. As he held it up to the light, he saw that it was filled with a warm glow.
Excited by his discovery, Fluffy decided to take the
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a routine. This means setting aside specific times for work, breaks, and other activities. It also helps to eliminate distractions as much as possible.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the cozy atmosphere.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The change in colour of leaves in autumn is due to the changing levels of chlorophyll, which is the green pigment that gives leaves their colour. As the days get shorter and the temperature drops, the plant stops producing new chlorophyll and starts breaking down the existing chlorophyll. This reveals other pigments, such as carotenoids (which give leaves their yellow and orange colours) and anthocyanins (which give leaves their red and purple colours).
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Samantha. ❌
- [H] **you:** What's your name?
  **brain:** My name is Samantha. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Samantha. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a software engineer. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Samantha. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Toyota Corolla. ❌
- [rotating] **you:** Who teaches me piano?
  **brain:** Your piano teacher is a woman named Samantha. ❌
- [rotating] **you:** Which day do the bins go out?
  **brain:** The bins go out on the day after your birthday. ❌
- [neardup] **you:** What do I ride to the station?
  **brain:** You ride a bicycle to the station.
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Samantha. ❌
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a salad for dinner last night. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to the beach last summer. invented

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 0.0; 101.1 s_

## TODO (not in Stage 0)

- transfer forced-choice lp read-out (incident turns are lived, not scored)
- span-NLL / log-odds
- pruned-store copy on full nights
- evening same-day recall in the hippocampus-off arm
- gist slots (E gist_q)
