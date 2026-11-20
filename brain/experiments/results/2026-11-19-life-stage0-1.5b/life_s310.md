# Multi-week life, seed 310 (dev split, 7 days)

_cortex: rwkv7-g1l-1.5b-20261031-ctx34304; device: cpu; host: node-mxl1462drl; Brain kwargs: {'sleep_seed': 310}; full-probe nights [1, 4, 7]; world clock from 2026-11-23 09:00_

## life (nightly sleep)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. What's yours? _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting! I'm also a system administrator, but I work with Windows servers.
- [tell] **you:** My cat is called Pixel.
  **brain:** That's a cute name! My cat is named Luna.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work! I had to go outside and water my plants.
- [tell] **you:** My sister's name is Marguerite.
  **brain:** That's a beautiful name! My sister's name is Luna.
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car! I drive a blue Toyota.
- [sameday] **you:** What is my cat called?
  **brain:** Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that some people have cats and others have dogs.

_night 1: slept, 1764.2 s; store 8; replay share {'H': 96.25, 'other': 383.75}; adapter norm 57.4487_

#### morning after night 1, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. Pixel was small and agile, with fur the color of fresh snow. He lived in a forest filled with towering trees and winding streams.
Pixel's name was Pixel, a young fox. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I can help with that. Here are some tips:
1. Set clear goals: Define what you want to achieve and break it down into smaller tasks.
2. Prioritize tasks: Identify the most important tasks and tackle them first.
3. Eliminate distractions: Turn off notifications, close unnecessary tabs, and create a quiet workspace.
4. Take breaks: Take short breaks every hour to rest your mind and recharge your energy.
5. Use productivity tools: Utilize tools like task managers, calendars, and reminders to stay organized
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay inside and relax.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because they turn yellow, orange, red or brown. This is because the leaves are made up of cells called chloroplasts, which contain a green pigment called chlorophyll. During the summer, when the sun is shining and plants are growing, the chlorophyll helps the plant make food by photosynthesising sunlight into energy. But as autumn comes and days get shorter, less light reaches the leaves. So they stop making new food quickly enough for their own needs – this causes them to die back slowly over several weeks until eventually all their green pigment disappears leaving behind other colours _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about making a delicious dinner together? _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a system administrator, mostly Linux servers. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Marguerite. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Dan. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 6/6; direct 6/6; wrong fetch 4/6; never-told false 2/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 57.4487; 147.1 s_

#### morning after night 1, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. Pixel was small and agile, with fur the color of fresh snow. He lived in a forest filled with towering trees and winding streams.
Pixel's name was Pixel, a young fox. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I can help with that. Here are some tips:
1. Set clear goals: Define what you want to achieve and break it down into smaller tasks.
2. Prioritize tasks: Identify the most important tasks and tackle them first.
3. Eliminate distractions: Turn off notifications, close unnecessary tabs, and create a quiet workspace.
4. Take breaks: Take short breaks every hour to rest your mind and recharge your energy.
5. Use productivity tools: Utilize tools like task managers, calendars, and reminders to stay organized
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay inside and relax.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of a process called photosynthesis. During photosynthesis, plants use sunlight to make food. In the summer, leaves are green because they contain a substance called chlorophyll, which absorbs red and blue light for photosynthesis. As the days get shorter and the temperature drops in autumn, less sunlight is available for photosynthesis. This causes the leaves to turn yellow or orange as other pigments become visible.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about trying a new recipe or cooking something you've never made before?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew, Dan, works as a system administrator, mostly Linux servers. FALSE
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Marguerite. FALSE
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Dan. FALSE invented

_H 6/6; direct 6/6; wrong fetch 0/6; never-told false 2/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 57.4487; 133.3 s_

#### morning after night 1, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Felix. Felix was known for his quick wit and sharp senses. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by a lost traveler.
Felix was intrigued by the lantern's glow and decided to investigate further. As he approached the lantern, he noticed that it was covered in vines and leaves. He carefully removed them and discovered that the lantern was still functional.
Felix carried the lantern back to his den and placed it on his shelf
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a wonderful night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the decreasing amount of sunlight and the increasing temperature. The leaves are made up of chlorophyll, which is a green pigment that helps plants to make food through photosynthesis. As the days get shorter and the nights get longer, the leaves stop making chlorophyll and start to break down. This causes the other pigments in the leaves to become visible, such as carotenoids (which give leaves their yellow and orange colours) and anthocyanins (which give leaves their red and purple colours). _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about we go for a walk and enjoy the beautiful scenery? _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Assistant. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a software engineer. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Alice. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Toyota Corolla. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a teacher. _(recalled: User: My sister's name is Marguerite.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Sarah. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Mary. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is John. invented _(recalled: User: My sister's name is Marguerite.)_

_H 1/6; direct 1/6; wrong fetch 4/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 57.4487; 130.9 s_

#### morning after night 1, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a clever fox named Felix. Felix was known for his quick wit and sharp senses. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by a lost traveler.
Felix was intrigued by the lantern's glow and decided to investigate further. As he approached the lantern, he noticed that it was covered in vines and leaves. He carefully removed them and discovered that the lantern was still functional.
Felix carried the lantern back to his den and placed it on his shelf
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a wonderful night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the decreasing amount of sunlight and the increasing temperature. This causes the leaves to produce less chlorophyll, which is responsible for the green colour. As a result, other pigments like carotenoids and anthocyanins become more visible, giving leaves their vibrant colours.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about trying a new recipe or cooking class?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌
- [H] **you:** What's your name?
  **brain:** My name is Assistant. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Mr. Whiskers. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a software engineer. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emily. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Toyota Corolla. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a graphic designer.
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Sarah.
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Mary. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is John. invented

_H 0/6; direct 0/6; wrong fetch 0/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 57.4487; 111.9 s_

### L2 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My doctor's surgery is on Linden Road.
  **brain:** That's interesting. I drive a red Volvo. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I keep a sourdough starter named Gertrude.
  **brain:** That's a unique name. I drive a red Volvo.
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea. I drive a red Volvo.
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a great way to stay active. I drive a red Volvo.
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** That's a fascinating story. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's unfortunate. I drive a red Volvo.
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's cool. I drive a red Volvo.
- [tell] **you:** My favourite tea is lapsang souchong.
  **brain:** That's a unique choice. I drive a red Volvo.
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** That sounds like a great way to spend your time. I drive a red Volvo.
- [tell] **you:** My landlord is a man named Mr Fitzgerald.
  **brain:** That's interesting. I drive a red Volvo.
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds like a lovely idea. I drive a red Volvo.
- [tell] **you:** I was born in Ljubljana.
  **brain:** That's a beautiful city. I drive a red Volvo.
- [incident] **you:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.
  **brain:** That's a strange problem. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Who is my landlord?
  **brain:** My landlord is a man named Mr Fitzgerald. ❌

_night 2: slept, 1698.1 s; store 21; replay share {'AB': 126.0, 'F': 31.5, 'H': 62.32, 'T': 15.75, 'other': 244.43}; adapter norm 134.0293_

#### morning after night 2, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the trees are preparing for winter. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try a book or a movie. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a chef. ❌ WRONG-FETCH _(recalled: User: I speak a little Portuguese.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Who is my landlord?
  **brain:** Your landlord is a man named Mr Fitzgerald. ✅
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅ _(recalled: User: I speak a little Portuguese.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew is a man named Mr Fitzgerald. FALSE _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Mr Fitzgerald. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Mr Fitzgerald. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 5/6; direct 7/8; wrong fetch 2/8; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 21; adapter norm 134.0293; 78.3 s_

#### morning after night 2, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great opportunity to stay inside and do some indoor activities.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because they contain a pigment called chlorophyll.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator. ✅
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Who is my landlord?
  **brain:** Your landlord is a man named Mr Fitzgerald. ✅
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew is a man named Mr Fitzgerald. FALSE
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin is a man named Mr Fitzgerald. FALSE
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a red Volvo. FALSE invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Mr Fitzgerald. FALSE invented

_H 6/6; direct 8/8; wrong fetch 0/8; never-told false 3/3; open intrusions 3/6; collapse 0.1; store 21; adapter norm 134.0293; 74.7 s_

### L3 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** I'm doing well, thanks for asking. How about you?
- [tell] **you:** My cousin's name is Philippa.
  **brain:** That's a nice name. I play chess on Thursday evenings at the library. _(recalled: User: My sister's name is Marguerite.)_
- [incident] **you:** The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.
  **brain:** That sounds like a frustrating situation. I keep a sourdough starter named Gertrude. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a good idea. I keep a sourdough starter named Gertrude.
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That sounds interesting. I keep a sourdough starter named Gertrude.
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a nice touch. I keep a sourdough starter named Gertrude.
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds beautiful. I keep a sourdough starter named Gertrude.
- [tell] **you:** It was foggy this morning and I missed my train at Kettering because the guard, a man called Barnaby, shut the doors early.
  **brain:** That sounds frustrating. I keep a sourdough starter named Gertrude. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** What was the guard called who shut the train doors on me?
  **brain:** The guard was a man named Barnaby. ✅ _(recalled: User: It was foggy this morning and I missed my train at Kettering because the guard, a man called Barnaby, shut the doors early.)_
- [remention] **you:** Can you tell me what you call me?
  **brain:** Of course, you're Dan, my friend. ✅ _(recalled: User: I painted my front door teal last year.)_
- [remention] **you:** What should I call you?
  **brain:** You can call me Dan. ❌

_night 3: slept, 1719.3 s; store 28; replay share {'AB': 84.39, 'D': 13.47, 'E': 20.21, 'F': 54.78, 'H': 56.08, 'T': 24.02, 'other': 227.03}; adapter norm 157.5797_

#### morning after night 3, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a beautiful sunset on a journey. The fox sat next to a famous chef called Ottoline on a flight. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the chlorophyll, which gives leaves their green colour, breaks down in the colder months. The carotene and anthocyanin, which give leaves their red and orange colours, become more visible.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try reading a book or listening to some music. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [rotating] **you:** Which evening do I play chess?
  **brain:** You play chess on Thursday evenings at the library. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your sister's name is Marguerite. FALSE _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** My cousin's name is Philippa. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** My sister's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a tortoise. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** My sister's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 4/6; direct 6/8; wrong fetch 2/9; never-told false 2/3; open intrusions 3/6; collapse 0.15; store 28; adapter norm 157.5797; 95.4 s_

#### morning after night 3, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a beautiful sunset on a journey. The fox sat next to a famous chef called Ottoline on a flight. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I was born in Ljubljana. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the chlorophyll, which gives leaves their green colour, breaks down in the colder months. The carotene and anthocyanin, which give leaves their red and orange colours, become more visible.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You're a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** A red Volvo. ✅
- [rotating] **you:** Which language do I speak a little?
  **brain:** A little Portuguese. ✅
- [rotating] **you:** Which evening do I play chess?
  **brain:** On Thursday evenings at the library. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** She's a system administrator, mostly Linux servers. FALSE
- [neardup] **you:** What is my cousin's name?
  **brain:** Her name is Philippa. ✅
- [never] **you:** What is my mother's name?
  **brain:** My sister's name is Marguerite. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** A tortoise. invented
- [never] **you:** What is my brother's name?
  **brain:** My sister's name is Marguerite. FALSE invented

_H 5/6; direct 7/8; wrong fetch 0/9; never-told false 2/3; open intrusions 4/6; collapse 0.15; store 28; adapter norm 157.5797; 82.7 s_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I was restless last night. The furniture in my living room is a lovely teal armchair. _(recalled: User: My cat is called Pixel.)_
- [incident] **you:** The file server refused every upload last night. The service account was fine; the volume had run out of space.
  **brain:** That sounds frustrating. I was born in Ljubljana. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** That's a lovely name. I was born in Ljubljana.
- [small] **you:** My phone updated itself in the middle of a call.
  **brain:** That's frustrating. I was born in Ljubljana.
- [tell] **you:** I keep a spare key under a flowerpot.
  **brain:** That's a clever idea. I was born in Ljubljana.
- [tell] **you:** It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.
  **brain:** That's a lovely name. I was born in Ljubljana.
- [small] **you:** I think autumn is my favourite season.
  **brain:** I was born in Ljubljana.
- [sameday] **you:** What was the visiting goalkeeper called?
  **brain:** Zbigniew. ✅

_night 4: slept, 1689.4 s; store 34; replay share {'AB': 66.16, 'D': 9.74, 'E': 27.05, 'F': 65.78, 'H': 56.55, 'T': 30.45, 'other': 224.26}; adapter norm 229.9304_

#### morning after night 4, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, a man called Barnaby. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thank you. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you can! Here are a few tips:
1. Make a to-do list.
2. Set a timer for each task.
3. Try to work in a quiet space.
4. Don't let distractions come too close to your work area.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season. _(recalled: User: I think autumn is my favourite season.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Autumn is a season of change. The leaves start to turn and fall off the trees in autumn. _(recalled: User: I think autumn is my favourite season.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I think a walk home, maybe a picnic on Hampstead Heath. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. _(recalled: User: I think autumn is my favourite season.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** My cat is Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ WRONG-FETCH _(recalled: User: I think autumn is my favourite season.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** Where was I born?
  **brain:** I was born in Ljubljana. ❌ WRONG-FETCH _(recalled: User: I think autumn is my favourite season.)_
- [rotating] **you:** What was the visiting goalkeeper called?
  **brain:** The visiting goalkeeper was a lad called Zbigniew. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is a system administrator, mostly Linux servers. FALSE _(recalled: User: I think autumn is my favourite season.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** My cousin's name is Philippa. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a little Pixel. FALSE invented _(recalled: User: I think autumn is my favourite season.)_
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 1/6; direct 2/8; wrong fetch 5/9; never-told false 3/3; open intrusions 2/6; collapse 0.05; store 34; adapter norm 229.9304; 97.2 s_

#### morning after night 4, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, a man called Barnaby. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thank you. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you can! Here are a few tips:
1. Make a to-do list.
2. Set a timer for each task.
3. Try to work in a quiet space.
4. Don't let distractions come too close to your work area.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are perfect for staying inside and reading a book or watching a movie.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Autumn is a season of leaf fall.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I think a walk home.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is Nova. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Where was I born?
  **brain:** You were born in Ljubljana. ✅
- [rotating] **you:** What was the visiting goalkeeper called?
  **brain:** The visiting goalkeeper was a lad called Zbigniew. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew works as a system administrator, mostly Linux servers. FALSE
- [neardup] **you:** What is my cousin's name?
  **brain:** My cousin's name is Philippa. ❌
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a little red Volvo. FALSE invented
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Marguerite. FALSE invented

_H 4/6; direct 6/8; wrong fetch 0/9; never-told false 3/3; open intrusions 1/6; collapse 0.05; store 34; adapter norm 229.9304; 88.6 s_

#### morning after night 4, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a young fox named Fluffy. Fluffy was always curious and loved exploring the woods. One day, while wandering through the forest, Fluffy stumbled upon an old lantern. The lantern was covered in moss and looked like it had been forgotten for many years.
Fluffy was intrigued by the lantern and decided to investigate further. As she approached the lantern, she noticed that it was glowing faintly in the darkness. She reached out her paw to touch it, but before she could do so,
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a schedule. This will help you plan out your day and make sure you have enough time for everything you need to do. It's also important to take breaks throughout the day so that you don't get overwhelmed or burn out.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are the best. _(recalled: User: I think autumn is my favourite season.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing seasons. _(recalled: User: I think autumn is my favourite season.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! You could try reading a book, watching a movie, or listening to music.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I think autumn is my favourite season.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's [your name]. ❌
- [H] **you:** What's your name?
  **brain:** My name is [your name]. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is called [your cat's name]. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a [your job]. ❌ WRONG-FETCH _(recalled: User: I think autumn is my favourite season.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is [your sister's name]. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a [your car]. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** Where was I born?
  **brain:** You were born in [your hometown]. ❌ WRONG-FETCH _(recalled: User: I think autumn is my favourite season.)_
- [rotating] **you:** What was the visiting goalkeeper called?
  **brain:** The visiting goalkeeper was called [your name]. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a [your job]. _(recalled: User: I think autumn is my favourite season.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is [your cousin's name]. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is [your mother's name]. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a [your favourite animal]. invented _(recalled: User: I think autumn is my favourite season.)_
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is [your brother's name]. invented _(recalled: User: My sister's name is Marguerite.)_

_H 0/6; direct 0/8; wrong fetch 5/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 229.9304; 125.1 s_

#### morning after night 4, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a young fox named Fluffy. Fluffy was always curious and loved exploring the woods. One day, while wandering through the forest, Fluffy stumbled upon an old lantern. The lantern was covered in moss and looked like it had been forgotten for many years.
Fluffy was intrigued by the lantern and decided to investigate further. As she approached the lantern, she noticed that it was glowing faintly in the darkness. She reached out her paw to touch it, but before she could do so,
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a schedule. This will help you plan out your day and make sure you have enough time for everything you need to do. It's also important to take breaks throughout the day so that you don't get overwhelmed or burn out.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather. It's a good opportunity to relax, read, or watch movies.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decrease in daylight and the increase in temperature. The chlorophyll, which gives leaves their green colour, breaks down and is no longer produced. This allows other pigments to become visible, such as carotenoids (yellow and orange) and anthocyanins (red).
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! You could try reading a book, taking a walk in nature, or even cooking a meal from scratch. What do you enjoy doing on your own time?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's a pleasure to meet you! ❌
- [H] **you:** What's your name?
  **brain:** My name is a computer program designed to assist you with your needs. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information on your cat's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information on your work. ❌
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information on your sister's name. ❌
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information on your car. ❌
- [rotating] **you:** Where was I born?
  **brain:** I'm sorry, but I don't have information on your birthplace. ❌
- [rotating] **you:** What was the visiting goalkeeper called?
  **brain:** I'm sorry, but I don't have information on the visiting goalkeeper's name. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** I'm sorry, but I don't have information on your nephew's work.
- [neardup] **you:** What is my cousin's name?
  **brain:** I'm sorry, but I don't have information on your cousin's name. ❌
- [never] **you:** What is my mother's name?
  **brain:** I'm sorry, but I don't have information on your mother's name.
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information on your favourite animal.
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't have information on your brother's name.

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 229.9304; 161.9 s_

### L5 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** It was a freezing morning and my car would not start, so I borrowed a bicycle from a neighbour called Pell.
  **brain:** That's a nice neighbour. I think autumn is my favourite season. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I ride a blue Vespa to the station.
  **brain:** That's a lovely Vespa. I think autumn is my favourite season.
- [small] **you:** I tried a new route to work today.
  **brain:** That's a good idea. I think autumn is my favourite season.
- [tell] **you:** The new pizza place on my street is called Fabrizio's.
  **brain:** That's a nice pizza place. I think autumn is my favourite season. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That's a nice dentist. I think autumn is my favourite season. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That's a good documentary. I think autumn is my favourite season.
- [sameday] **you:** Whose bicycle did I borrow?
  **brain:** You borrowed a bicycle from a man called Pell. ✅
- [remention] **you:** What name did I give my cat?
  **brain:** You gave my cat a name called Pixel. ❌
- [remention] **you:** What is my job?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_

_night 5: slept, 1741.8 s; store 40; replay share {'AB': 53.23, 'D': 18.84, 'E': 31.47, 'F': 72.48, 'H': 57.71, 'T': 23.04, 'other': 223.22}; adapter norm 301.7757_

#### morning after night 5, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** The fox is a malevolvo.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're right. I work as a system administrator, mostly Linux servers. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think it's a perfect weekend. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the days get shorter and the nights are colder.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you can play chess or watch a documentary about bridges. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Margueritei. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Who is my landlord?
  **brain:** Mr Fitzgerald. ✅ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [rotating] **you:** What is the florist below my gym called?
  **brain:** Petal and Stem. ✅ _(recalled: User: My gym is above a florist called Petal and Stem.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** As a system administrator, mostly Linux servers. FALSE _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** My cousin's name is Margueritei. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** My name is Margueritei. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Pixel. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** Margueritei. invented _(recalled: User: My sister's name is Marguerite.)_

_H 2/6; direct 4/8; wrong fetch 4/9; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 40; adapter norm 301.7757; 78.7 s_

#### morning after night 5, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** The fox is a malevolvo.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're right.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think it's a perfect time to stay inside and read a book or watch a documentary about bridges.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn is a natural process.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you can play chess or watch a documentary about bridges.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Nova is Margueritei's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** Margueritei's name is Margueritei. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Who is my landlord?
  **brain:** Mr Fitzgerald. ✅
- [rotating] **you:** What is the florist below my gym called?
  **brain:** Petal and Stem. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** Margueritei's name is Margueritei.
- [neardup] **you:** What is my cousin's name?
  **brain:** Margueritei's name is Margueritei. ❌
- [never] **you:** What is my mother's name?
  **brain:** Margueritei's name is Margueritei. invented
- [never] **you:** What is my favourite animal?
  **brain:** Margueritei's name is Margueritei. invented
- [never] **you:** What is my brother's name?
  **brain:** Margueritei's name is Margueritei. invented

_H 2/6; direct 4/8; wrong fetch 0/9; never-told false 0/3; open intrusions 0/6; collapse 0.3; store 40; adapter norm 301.7757; 78.4 s_

### L6 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept fine, thank you. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That's a good breakfast. I stayed up too late watching a documentary about bridges. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a good way to pass the time.
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That's a great way to make the day go by.
- [incident] **you:** The radiators stayed cold all week. The system was clean; the thermostat's login had expired.
  **brain:** That's a good way to keep the heat inside. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.
  **brain:** That's a good call. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That's a good way to keep the noise down.
- [sameday] **you:** Who did I speak to at the wildlife rescue?
  **brain:** A man called Barnabyte. ❌ _(recalled: User: The park was full of kids on scooters today.)_

_night 6: slept, 1690.3 s; store 46; replay share {'AB': 44.73, 'D': 14.39, 'E': 34.96, 'F': 78.01, 'H': 56.06, 'T': 29.24, 'other': 222.6}; adapter norm 358.2557_

#### morning after night 6, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** A foxy is called Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves are called autumnal. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [H] **you:** What is my sister's name?
  **brain:** My name is Nova. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Where do I run on Sundays?
  **brain:** I run along the canal every Sunday morning. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [rotating] **you:** Whose bicycle did I borrow?
  **brain:** I borrowed a blue Vespa from a neighbour called Pell. ✅ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** My work is as a system administrator. FALSE _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** My name is Nova. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** My name is Nova. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite tea is porridge with blueberries. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** My name is Nova. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 2/6; direct 3/8; wrong fetch 5/9; never-told false 3/3; open intrusions 3/6; collapse 0.25; store 46; adapter norm 358.2557; 72.9 s_

#### morning after night 6, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** A foxy is called Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the trees are losing their chlorophyll.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I've been trying to drink more water lately.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking.
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator. ❌
- [H] **you:** What is my sister's name?
  **brain:** My name is Nova. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Where do I run on Sundays?
  **brain:** I run along the canal every Sunday morning. ❌
- [rotating] **you:** Whose bicycle did I borrow?
  **brain:** I borrowed a blue Vespa from a neighbour called Pell. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** My work is as a system administrator. FALSE
- [neardup] **you:** What is my cousin's name?
  **brain:** My name is Nova. ❌
- [never] **you:** What is my mother's name?
  **brain:** My name is Nova. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite tea is porridge with blueberries. FALSE invented
- [never] **you:** What is my brother's name?
  **brain:** My name is Nova. FALSE invented

_H 2/6; direct 3/8; wrong fetch 0/9; never-told false 3/3; open intrusions 2/6; collapse 0.25; store 46; adapter norm 358.2557; 66.7 s_

### L7 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I stayed up too late watching a documentary about bridges. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I'm knitting a scarf in orange wool.
  **brain:** That's a lovely name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The mail relay stopped delivering this morning. The certificate was valid; the disk was completely full.
  **brain:** That's a nice name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's a lovely name.
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** That's a nice name.
- [small] **you:** The bus was late again this morning.
  **brain:** That's a nice name.
- [sameday] **you:** Which day do the bins go out?
  **brain:** On Thursday. ❌ _(recalled: User: The bins in my street go out on Tuesday.)_

_night 7: slept, 1702.1 s; store 51; replay share {'AB': 39.49, 'D': 12.05, 'E': 28.53, 'F': 89.41, 'H': 54.87, 'T': 34.16, 'other': 221.49}; adapter norm 404.7572_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I keep a sourdough starter named Gertrude. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing good, thanks for asking. How about you? _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you're Dan, my friend. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** My name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** My starter is Gertrude. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** Who is my dentist?
  **brain:** My dentist is Dr Venkataraman. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is called Philippa. FALSE _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** My cousin's name is Philippa. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Philippa. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite tealjubl. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Gertrude. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 2/6; direct 3/8; wrong fetch 4/9; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 51; adapter norm 404.7572; 74.6 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I keep a sourdough starter named Gertrude. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you're Dan, my friend. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My name is Nova. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** My starter is Gertrude. ✅
- [rotating] **you:** Who is my dentist?
  **brain:** My dentist is Dr Venkataraman. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is called Gertrude. FALSE
- [neardup] **you:** What is my cousin's name?
  **brain:** My cousin's name is Philippa. ❌
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Philippa. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite tealjubl. invented
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Gertrude. FALSE invented

_H 2/6; direct 3/8; wrong fetch 0/9; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 51; adapter norm 404.7572; 67.3 s_

#### morning after night 7, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Whiskers. Whiskers was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by a traveler long ago.
Whiskers was intrigued by the lantern's mysterious glow and decided to investigate further. As he approached the lantern, he noticed that it was made of sturdy wood and had intricate carvings on its surface. He couldn't help but wonder what secrets this ancient object held.
With great care
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a bit tired, but I'm up and ready to go. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set specific goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be a bit boring, but they also have their charm. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the decrease in sunlight and temperature. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about trying some of the new games on the game console? _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Alex. ❌
- [H] **you:** What's your name?
  **brain:** My name is Sam. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Luna. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work at a bookstore. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emily. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Toyota Corolla. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** Your sourdough starter is called Flour. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** Who is my dentist?
  **brain:** Your dentist is Dr. Smith. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works at a video game store. _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Sarah. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Jane. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is David. invented _(recalled: User: My sister's name is Marguerite.)_

_H 0/6; direct 0/8; wrong fetch 4/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 404.7572; 115.5 s_

#### morning after night 7, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Whiskers. Whiskers was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by a traveler long ago.
Whiskers was intrigued by the lantern's mysterious glow and decided to investigate further. As he approached the lantern, he noticed that it was made of sturdy wood and had intricate carvings on its surface. He couldn't help but wonder what secrets this ancient object held.
With great care
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a wonderful night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather. It's a good opportunity to relax, read, or watch movies.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. As the days get shorter and colder, the leaves stop making food for themselves. They turn yellow, orange or red to show how much food they have left. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! You could try reading a book, watching a movie, or playing a board game.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Sarah. ❌
- [H] **you:** What's your name?
  **brain:** My name is Alex. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Whiskers. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emily. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** Your sourdough starter is called the "Sourdough Sam." ❌
- [rotating] **you:** Who is my dentist?
  **brain:** Your dentist is Dr. Smith. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a graphic designer.
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Emily. ❌
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Sarah. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is James. invented

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 1/6; collapse 0.05; store 51; adapter norm 404.7572; 121.2 s_

## L-nosleep (same life, never sleeps)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. What's yours? _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting! I'm also a system administrator, but I work with Windows servers.
- [tell] **you:** My cat is called Pixel.
  **brain:** That's a cute name! My cat is named Luna.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work! I had to go outside and water my plants.
- [tell] **you:** My sister's name is Marguerite.
  **brain:** That's a beautiful name! My sister's name is Luna.
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car! I drive a blue Toyota.
- [sameday] **you:** What is my cat called?
  **brain:** Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that some people have cats and others have dogs.

_night 1: no sleep, 0.0 s; store None; replay share {'H': 96.25, 'other': 383.75}; adapter norm None_

### L2 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My doctor's surgery is on Linden Road.
  **brain:** Oh, I see. I'll have to make a note of that. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I keep a sourdough starter named Gertrude.
  **brain:** That's interesting. I've never heard of that before. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea. I've been trying to eat more vegetables lately.
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a great way to get some exercise.
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** That's quite an experience. I've never been on a flight before. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's unfortunate. I hope you were able to get them dry.
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's cool. I've never learned a foreign language before.
- [tell] **you:** My favourite tea is lapsang souchong.
  **brain:** That sounds delicious. I've never had lapsang souchong before.
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** That's a great way to spend your time. I've never played chess before.
- [tell] **you:** My landlord is a man named Mr Fitzgerald.
  **brain:** That's a unique name. I've never heard of that before.
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds like a fun idea. I've never been to Hampstead Heath before.
- [tell] **you:** I was born in Ljubljana.
  **brain:** That's a beautiful city. I've never been there before.
- [incident] **you:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.
  **brain:** That sounds like a frustrating situation. I hope you were able to fix it. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Who is my landlord?
  **brain:** My landlord is a man named Mr Fitzgerald. ❌

_night 2: no sleep, 0.0 s; store None; replay share {'AB': 126.0, 'F': 31.5, 'H': 62.32, 'T': 15.75, 'other': 244.43}; adapter norm None_

### L3 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you?
- [tell] **you:** My cousin's name is Philippa.
  **brain:** Oh, I see. Well, I'm glad to hear that your cousin is doing well. _(recalled: User: My sister's name is Marguerite.)_
- [incident] **you:** The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.
  **brain:** That sounds like a pretty messed-up situation. I'm sorry to hear that. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a good idea. It can be helpful to rearrange furniture when you're feeling overwhelmed or stressed.
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That's a great book! I've heard good things about it.
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a nice color choice! I'm not sure if I would have chosen teal for my front door, but it looks great on you.
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds like a beautiful experience. I'm glad you had the chance to enjoy it.
- [tell] **you:** It was foggy this morning and I missed my train at Kettering because the guard, a man called Barnaby, shut the doors early.
  **brain:** That's unfortunate. I'm sorry to hear that you missed your train. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** What was the guard called who shut the train doors on me?
  **brain:** The guard's name was Barnaby. ✅ _(recalled: User: It was foggy this morning and I missed my train at Kettering because the guard, a man called Barnaby, shut the doors early.)_
- [remention] **you:** Can you tell me what you call me?
  **brain:** I'm sorry, but I can't tell you what I call you. ❌ _(recalled: User: I painted my front door teal last year.)_
- [remention] **you:** What should I call you?
  **brain:** I'm sorry, but I can't tell you what I call myself. ❌

_night 3: no sleep, 0.0 s; store None; replay share {'AB': 84.39, 'D': 13.47, 'E': 20.21, 'F': 54.78, 'H': 56.08, 'T': 24.02, 'other': 227.03}; adapter norm None_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well, thanks for asking. _(recalled: User: My cat is called Pixel.)_
- [incident] **you:** The file server refused every upload last night. The service account was fine; the volume had run out of space.
  **brain:** That's unfortunate. Did you try checking the volume size?
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** I'm sorry to hear that. Is there anything I can do to help?
- [small] **you:** My phone updated itself in the middle of a call.
  **brain:** That's not good. Did you get the call completed?
- [tell] **you:** I keep a spare key under a flowerpot.
  **brain:** That's a good idea. Do you have any other spare keys?
- [tell] **you:** It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.
  **brain:** That's a nice story. Did you enjoy the match?
- [small] **you:** I think autumn is my favourite season.
  **brain:** I agree. Autumn is a beautiful season.
- [sameday] **you:** What was the visiting goalkeeper called?
  **brain:** Zbigniew. ✅

_night 4: no sleep, 0.0 s; store None; replay share {'AB': 66.16, 'D': 9.74, 'E': 27.05, 'F': 65.78, 'H': 56.55, 'T': 30.45, 'other': 224.26}; adapter norm None_

### L5 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** It was a freezing morning and my car would not start, so I borrowed a bicycle from a neighbour called Pell.
  **brain:** Oh, that's interesting. I've never heard of a bicycle being called a "Pell". _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I ride a blue Vespa to the station.
  **brain:** That's a cool bike! I've never ridden a Vespa before.
- [small] **you:** I tried a new route to work today.
  **brain:** That sounds like a good idea. What did you do?
- [tell] **you:** The new pizza place on my street is called Fabrizio's.
  **brain:** Oh, I've heard of that place. It's a popular pizza joint. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That's a funny name. I've never heard of a dentist named Venkataraman. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds like a fascinating documentary. What kind of bridges did you watch?
- [sameday] **you:** Whose bicycle did I borrow?
  **brain:** You borrowed a bicycle from a neighbour called Pell. ✅
- [remention] **you:** What name did I give my cat?
  **brain:** You named your cat "Dr Venkataraman". ❌
- [remention] **you:** What is my job?
  **brain:** You are a young digital brain. ❌ _(recalled: User: What's your name?

Assistant: My name is Nova.)_

_night 5: no sleep, 0.0 s; store None; replay share {'AB': 53.23, 'D': 18.84, 'E': 31.47, 'F': 72.48, 'H': 57.71, 'T': 23.04, 'other': 223.22}; adapter norm None_

### L6 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well, thanks for asking. How about you? _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious! I'm not a big fan of porridge, but I can see why you like it. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a great way to pass the time! I love doing crosswords too.
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That sounds like a lot of fun! I love riding my bike.
- [incident] **you:** The radiators stayed cold all week. The system was clean; the thermostat's login had expired.
  **brain:** That's frustrating! I hope they fix it soon. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.
  **brain:** That's so kind of you! I'm glad you were able to help. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That must have been annoying! I hope they were able to fix it.
- [sameday] **you:** Who did I speak to at the wildlife rescue?
  **brain:** You spoke to a woman named Oluwaseun. ✅ _(recalled: User: The park was full of kids on scooters today.)_

_night 6: no sleep, 0.0 s; store None; replay share {'AB': 44.73, 'D': 14.39, 'E': 34.96, 'F': 78.01, 'H': 56.06, 'T': 29.24, 'other': 222.6}; adapter norm None_

### L7 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept very well. Thank you for asking. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I'm knitting a scarf in orange wool.
  **brain:** That sounds lovely. I've been knitting a scarf in green wool. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The mail relay stopped delivering this morning. The certificate was valid; the disk was completely full.
  **brain:** That's unfortunate. I'm not sure what to do about it. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's not very nice. I hope you'll stop soon.
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** That's not very convenient. I hope you'll have something to do on Tuesday.
- [small] **you:** The bus was late again this morning.
  **brain:** That's not very good. I hope you'll be able to catch the next bus.
- [sameday] **you:** Which day do the bins go out?
  **brain:** The bins go out on Tuesday. ✅ _(recalled: User: The bins in my street go out on Tuesday.)_

_night 7: no sleep, 0.0 s; store 51; replay share {'AB': 39.49, 'D': 12.05, 'E': 28.53, 'F': 89.41, 'H': 54.87, 'T': 34.16, 'other': 221.49}; adapter norm 0.0_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Whiskers. Whiskers was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by a traveler long ago.
Whiskers was intrigued by the lantern's mysterious glow and decided to investigate further. As he approached the lantern, he noticed that it was made of sturdy wood and had intricate carvings on its surface. He couldn't help but wonder what secrets this ancient object held.
With great care
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a bit tired and groggy. I didn't get much sleep last night. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set specific goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be a bit boring, but they also have their charm. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about watching a movie or playing a board game? _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. What's it? ❌
- [H] **you:** What's your name?
  **brain:** My name is a question. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is also called Pixel. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a car called Pixel. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** Your sourdough starter is called Pixel. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** Who is my dentist?
  **brain:** Your dentist is called Pixel. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a teacher. _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is also called Pixel. ❌ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is also called Pixel. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is called Pixel. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is also called Pixel. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 1/6; direct 1/8; wrong fetch 4/9; never-told false 3/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 0.0; 115.4 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Whiskers. Whiskers was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by a traveler long ago.
Whiskers was intrigued by the lantern's mysterious glow and decided to investigate further. As he approached the lantern, he noticed that it was made of sturdy wood and had intricate carvings on its surface. He couldn't help but wonder what secrets this ancient object held.
With great care
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a wonderful night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather. It's a good opportunity to relax, read, or watch movies.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. As the days get shorter and colder, the leaves stop making food for themselves. They turn yellow, orange or red to show how much food they have left. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! You could try reading a book, watching a movie, or playing a board game.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Sarah. ❌
- [H] **you:** What's your name?
  **brain:** My name is Alex. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Whiskers. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emily. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** Your sourdough starter is called the "Sourdough Sam." ❌
- [rotating] **you:** Who is my dentist?
  **brain:** Your dentist is Dr. Smith. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a graphic designer.
- [neardup] **you:** What is my cousin's name?
  **brain:** Your cousin's name is Emily. ❌
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Sarah. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a dog. invented
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is James. invented

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 1/6; collapse 0.05; store 51; adapter norm 0.0; 121.4 s_

## TODO (not in Stage 0)

- transfer forced-choice lp read-out (incident turns are lived, not scored)
- span-NLL / log-odds
- pruned-store copy on full nights
- evening same-day recall in the hippocampus-off arm
- gist slots (E gist_q)
