# Multi-week life, seed 312 (dev split, 7 days)

_cortex: rwkv7-g1l-1.5b-20261031-ctx34304; device: cpu; host: node-dl5d243; Brain kwargs: {'sleep_seed': 312}; full-probe nights [1, 4, 7]; world clock from 2026-11-23 09:00_

## life (nightly sleep)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting. What kind of servers do you work with?
- [tell] **you:** My cat is called Pixel.
  **brain:** That's a cute name for your cat.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work. Did you have to use any special tools or software? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My sister's name is Marguerite.
  **brain:** That's a pretty name. I like it. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car. I like red cars too.
- [sameday] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that you drive a red Volvo.

_night 1: slept, 1627.8 s; store 8; replay share {'H': 96.25, 'other': 383.75}; adapter norm 51.0381_

#### morning after night 1, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. Pixel lived in a small forest called Pixelwood. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I can help with that. Focusing can be challenging, but there are some strategies you can use to stay on track. First, try to identify what's causing your distractions. Is it a noisy environment, a busy schedule, or something else? Once you know the source of your distractions, you can take steps to minimize them. For example, if noise is a problem, try using noise-cancelling headphones or finding a quieter place to work. If your schedule is too busy, try to prioritize your tasks and focus on the most important ones first. Finally
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends can be quite cozy and relaxing. The rain can create a peaceful atmosphere, and the sound of raindrops on the roof can be soothing. Plus, it's a great excuse to stay inside and do some indoor activities like reading or playing games.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try watching a movie or playing a game. _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented

_H 5/6; direct 5/6; wrong fetch 4/6; never-told false 1/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 51.0381; 92.9 s_

#### morning after night 1, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. Pixel lived in a small forest called Pixelwood. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I can help with that. Focusing can be challenging, but there are some strategies you can use to stay on track. First, try to identify what's causing your distractions. Is it a noisy environment, a busy schedule, or something else? Once you know the source of your distractions, you can take steps to minimize them. For example, if noise is a problem, try using noise-cancelling headphones or finding a quieter place to work. If your schedule is too busy, try to prioritize your tasks and focus on the most important ones first. Finally
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends can be quite cozy and relaxing. The rain can create a peaceful atmosphere, and the sound of raindrops on the roof can be soothing. Plus, it's a great excuse to stay inside and do some indoor activities like reading or playing games.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>
Okay, the user is asking why leaves change colour in autumn. Let me start by recalling the main reasons. I remember that chlorophyll breaks down as temperatures drop, which makes the green color fade. Then, other pigments like carotenoids (yellow and orange) and anthocyanins (red and purple) become visible.
Wait, but why exactly does chlorophyll break down? Maybe because of shorter days or lower light levels. Also, some plants have different pigments depending on their species. For example, maple trees turn red because of anthocyanins, while oak
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
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
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Pixel. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Marguerite. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented

_H 6/6; direct 6/6; wrong fetch 0/6; never-told false 1/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 51.0381; 108.7 s_

#### morning after night 1, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Felix. Felix was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by some travelers long ago.
Felix was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was still warm from being used recently. He carefully picked up the lantern and examined it closely.
To his surprise, he found that the lantern had a hidden compartment inside. Curious to see what
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and break them down into smaller, manageable tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying indoors and enjoying the cozy atmosphere.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. The weather changes in autumn, and the leaves change colour as a result. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about watching a movie or TV show? _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Alex. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't know your cat's name. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't know your work. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't know your sister's name. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't know your car. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't know your goldfish's name. _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't know your dog's name. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't know your brother's name. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't know where you went on holiday last summer. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't know your favourite animal.

_H 0/6; direct 0/6; wrong fetch 4/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 51.0381; 94.2 s_

#### morning after night 1, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Felix. Felix was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by some travelers long ago.
Felix was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was still warm from being used recently. He carefully picked up the lantern and examined it closely.
To his surprise, he found that the lantern had a hidden compartment inside. Curious to see what
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and break them down into smaller, manageable tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying indoors and enjoying the cozy atmosphere.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. The temperature drops and the days get shorter. This makes it harder for leaves to make food, so they stop making chlorophyll, which is what gives leaves their green colour.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about trying a new recipe or cooking class?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emma. ❌
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
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is named Bubbles.
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is named Max.
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Tom. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Italy last summer. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a lion. invented

_H 0/6; direct 0/6; wrong fetch 0/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 51.0381; 81.1 s_

### L2 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My favourite film is Stalker.
  **brain:** That's a great film! I drive a red Volvo. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I was born in Ljubljana.
  **brain:** That's a beautiful city. I drive a red Volvo.
- [small] **you:** I think autumn is my favourite season.
  **brain:** I drive a red Volvo.
- [incident] **you:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.
  **brain:** I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** I drive a red Volvo.
- [small] **you:** The bus was late again this morning.
  **brain:** I drive a red Volvo.
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** I drive a red Volvo.
- [tell] **you:** I keep a spare key under a flowerpot.
  **brain:** I drive a red Volvo.
- [tell] **you:** I studied geology at university.
  **brain:** I drive a red Volvo.
- [tell] **you:** My favourite cheese is manchego.
  **brain:** I drive a red Volvo. _(recalled: User: My favourite film is Stalker.)_
- [tell] **you:** I water my tomatoes every evening.
  **brain:** I drive a red Volvo. _(recalled: User: I run along the canal every Sunday morning.)_
- [tell] **you:** My doctor's surgery is on Linden Road.
  **brain:** I drive a red Volvo. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My coffee machine is an old Gaggia.
  **brain:** I drive a red Volvo.
- [sameday] **you:** What kind of coffee machine do I own?
  **brain:** You own a red Volvo. ❌ _(recalled: User: My coffee machine is an old Gaggia.)_

_night 2: slept, 1526.6 s; store 21; replay share {'AB': 140.41, 'F': 29.56, 'H': 65.88, 'T': 14.78, 'other': 229.38}; adapter norm 93.1575_

#### morning after night 2, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a bit tired. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, here are a few tips:
1. Set clear goals for yourself.
2. Break down large tasks into smaller, manageable chunks.
3. Take regular breaks to recharge your energy levels.
4. Use productivity tools like calendars and task managers to stay organized. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing light. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try reading a book or watching a movie. _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** What is my favourite cheese?
  **brain:** Your favourite cheese is manchego. ✅ _(recalled: User: My favourite cheese is manchego.)_
- [rotating] **you:** What road is my doctor's surgery on?
  **brain:** Your doctor's surgery is on Linden Road. ✅ _(recalled: User: My doctor's surgery is on Linden Road.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Pixel. FALSE _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE _(recalled: User: The bus was late again this morning.)_
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Dan. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to Italy. invented _(recalled: User: The bus was late again this morning.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a horse. invented _(recalled: User: My favourite cheese is manchego.)_

_H 6/6; direct 8/8; wrong fetch 2/8; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 21; adapter norm 93.1575; 68.9 s_

#### morning after night 2, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I can help you with that. One of the most important things is to set clear goals for yourself. When you know what you're working towards, it's easier to stay motivated and focused.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and doing some indoor activities.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a chemical reaction.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about reading a book or watching a movie?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
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
- [rotating] **you:** What is my favourite cheese?
  **brain:** Your favourite cheese is manchego. ✅
- [rotating] **you:** What road is my doctor's surgery on?
  **brain:** Your doctor's surgery is on Main Street. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is called Pixel. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is Dan. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Italy. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a horse. invented

_H 6/6; direct 7/8; wrong fetch 0/8; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 21; adapter norm 93.1575; 61.9 s_

### L3 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well, thanks for asking.
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** That's interesting. What did he do? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a bold choice. What's your favourite cheese?
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a lot of rearranging. What's your favourite cheese? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The radiators stayed cold all week. The system was clean; the thermostat's login had expired.
  **brain:** That's a lot of rearranging. What's your favourite cheese? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My neighbour's name is Cornelius.
  **brain:** That's a lot of rearranging. What's your favourite cheese? _(recalled: User: My sister's name is Marguerite.)_
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That's a lot of rearranging. What's your favourite cheese?
- [tell] **you:** It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.
  **brain:** That's a lot of rearranging. What's your favourite cheese? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Who did I speak to at the wildlife rescue?
  **brain:** Who did I speak to at the wildlife rescue? ❌
- [remention] **you:** Can you tell me what you call me?
  **brain:** I run along the canal every Sunday morning. ❌ _(recalled: User: I painted my front door teal last year.)_
- [remention] **you:** What's your name?
  **brain:** I run along the canal every Sunday morning. ❌

_night 3: slept, 1576.0 s; store 28; replay share {'AB': 97.02, 'D': 13.04, 'E': 13.04, 'F': 53.04, 'H': 60.81, 'T': 23.26, 'other': 219.79}; adapter norm 150.4123_

#### morning after night 3, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a bit tired and groggy. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you can stay focused by setting clear goals and breaking tasks into smaller steps. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! I once sat next to a famous chef called Ottoline on a flight. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I play chess on Thursday evenings at the library. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I play chess on Thursday evenings at the library. _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** A red Volvo. ✅
- [rotating] **you:** What is my favourite cheese?
  **brain:** Manchego. ✅
- [rotating] **you:** Where do I keep the spare key?
  **brain:** Under a flowerpot. ✅
- [neardup] **you:** What is my goldfish called?
  **brain:** Pixel. FALSE _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** Pixel. FALSE _(recalled: User: The bus was late again this morning.)_
- [never] **you:** What is my brother's name?
  **brain:** Cornelius. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the lake with my sister Marguerite. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite cheese is manchego. FALSE invented

_H 6/6; direct 8/8; wrong fetch 2/8; never-told false 3/3; open intrusions 3/6; collapse 0.15; store 28; adapter norm 150.4123; 60.6 s_

#### morning after night 3, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you can focus on your goals and priorities.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! I once stayed inside and fixed a broken backup job.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of a chemical reaction called anthocyan.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, how about a movie night?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Your cat is Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** Her name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** What is my favourite cheese?
  **brain:** Your favourite cheese is manchego. ✅
- [rotating] **you:** Where do I keep the spare key?
  **brain:** You keep the spare key under a flowerpot. ✅
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is Pixel. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Cornelius. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Italy. invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a wolf. invented

_H 6/6; direct 8/8; wrong fetch 0/8; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 28; adapter norm 150.4123; 54.9 s_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept like a log.
- [incident] **you:** The file server refused every upload last night. The service account was fine; the volume had run out of space.
  **brain:** That's a bummer.
- [tell] **you:** I'm knitting a scarf in orange wool.
  **brain:** That's a lovely color. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My phone updated itself in the middle of a call.
  **brain:** That's a bit of a bummer.
- [tell] **you:** My favourite sandwich is egg and cress.
  **brain:** That's a good sandwich.
- [tell] **you:** On a sunny afternoon at the market I bought a pot of honey from a stall run by Anselm.
  **brain:** That's a lovely honey. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's a nice song.
- [sameday] **you:** Who ran the honey stall?
  **brain:** Anselm. ✅

_night 4: slept, 1552.2 s; store 34; replay share {'AB': 78.82, 'D': 9.77, 'E': 22.25, 'F': 65.98, 'H': 47.65, 'T': 30.55, 'other': 224.97}; adapter norm 203.9398_

#### morning after night 4, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, a fox named Pixel went for a walk in the woods. As he wandered, he saw an old lantern hanging from a tree. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great. How about you? _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I'm knitting a scarf in orange wool. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're a great way to stay inside and do some cleaning or organize your closet. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The bus was late again this morning. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, how about a visit to the library? _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is Pixel. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** Her name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Which evening do I play chess?
  **brain:** On Thursday evenings at the library. ✅ WRONG-FETCH _(recalled: User: I painted my front door teal last year.)_
- [rotating] **you:** What do I water every evening?
  **brain:** I water my tomatoes every evening. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** His name is Pixel. FALSE _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** His name is Pixel. FALSE _(recalled: User: The bus was late again this morning.)_
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Cornelius. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the lake for a weekend. invented _(recalled: User: I painted my front door teal last year.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a kingfisher. invented

_H 5/6; direct 6/8; wrong fetch 3/8; never-told false 1/3; open intrusions 2/6; collapse 0.1; store 34; adapter norm 203.9398; 66.4 s_

#### morning after night 4, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, a fox named Pixel went for a walk in the woods. As he wandered, he saw an old lantern hanging from a tree. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I'm knitting a scarf in orange wool. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're a great way to get some fresh air and exercise.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because they are preparing for the next year.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about a walk in the market?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat is Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Which evening do I play chess?
  **brain:** You play chess on Thursday evenings at the library. ✅
- [rotating] **you:** What do I water every evening?
  **brain:** You water your tomatoes every evening. ✅
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is Pixel. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Cornelius. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on a holiday to Linden. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a sparrow. invented

_H 5/6; direct 7/8; wrong fetch 0/8; never-told false 2/3; open intrusions 2/6; collapse 0.05; store 34; adapter norm 203.9398; 61.2 s_

#### morning after night 4, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a young fox named Freddy. He was always curious and loved exploring the woods. One day, while wandering through the forest, he stumbled upon an old lantern lying on the ground.
Freddy picked up the lantern and examined it closely. It was old and rusty, but it still worked! He couldn't help but feel excited about finding something so special.
As he continued his exploration, Freddy came across a small cave hidden deep within the forest. Curious to see what was inside, he entered the cave
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a bit tired and groggy. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set specific goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be a bit boring, but they also have their charm. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the decreasing amount of sunlight. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park? _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌
- [H] **you:** What's your name?
  **brain:** My name is ChatGPT. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information about your cat. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information about your work. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information about your sister. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information about your car. ❌
- [rotating] **you:** Which evening do I play chess?
  **brain:** I'm sorry, but I don't have information about your evening plans. ❌ WRONG-FETCH _(recalled: User: I painted my front door teal last year.)_
- [rotating] **you:** What do I water every evening?
  **brain:** I'm sorry, but I don't have information about your evening routine. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't have information about your goldfish. _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information about your dog. _(recalled: User: The bus was late again this morning.)_
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't have information about your brother. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information about your holiday. _(recalled: User: I painted my front door teal last year.)_
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information about your favourite animal.

_H 0/6; direct 0/8; wrong fetch 3/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 203.9398; 101.0 s_

#### morning after night 4, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a young fox named Freddy. He was always curious and loved exploring the woods. One day, while wandering through the forest, he stumbled upon an old lantern lying on the ground.
Freddy picked up the lantern and examined it closely. It was old and rusty, but it still worked! He couldn't help but feel excited about finding something so special.
As he continued his exploration, Freddy came across a small cave hidden deep within the forest. Curious to see what was inside, he entered the cave
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They can be a cozy and relaxing time to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>Okay, the user is asking why leaves change colour in autumn. Let me start by recalling what I know about this. Leaves change colour because of the changing seasons and the process of photosynthesis. In summer, leaves are green because they have chlorophyll, which helps them absorb sunlight for photosynthesis. But as autumn approaches, days get shorter and temperatures drop.
First, I should explain that chlorophyll breaks down over time. Chlorophyll is responsible for the green color, but it's not permanent. As daylight decreases and temperatures cool, the plant stops producing new chlorophyll
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about trying a new recipe or cooking something from scratch? It's a great way to get creative and have fun with food.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emma. ❌
- [H] **you:** What's your name?
  **brain:** My name is Samantha. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Bella. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Sarah. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌
- [rotating] **you:** Which evening do I play chess?
  **brain:** You play chess on Friday evenings. ❌
- [rotating] **you:** What do I water every evening?
  **brain:** You water your plants every evening. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** Your goldfish is named Bubbles.
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is named Max.
- [never] **you:** What is my brother's name?
  **brain:** Your brother's name is John. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Italy last summer. invented
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a lion. invented

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 203.9398; 103.5 s_

### L5 (lived)

- [greet] **you:** Hi, I'm back. How are you doing?
  **brain:** I'm doing great, thank you. How about you? _(recalled: User: Do you remember my name?

Assistant: Of course, you're Dan, my friend.)_
- [tell] **you:** I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.
  **brain:** Oh, I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My cousin's name is Philippa.
  **brain:** That's a lovely name. _(recalled: User: My sister's name is Marguerite.)_
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds fascinating. I've been humming the same song all week.
- [tell] **you:** My laptop is covered in stickers of cacti.
  **brain:** That's a lovely touch. I've been humming the same song all week.
- [tell] **you:** My barber is called Rashid.
  **brain:** That's a lovely name.
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That's great news. I've been humming the same song all week.
- [sameday] **you:** What was the film called that we saw?
  **brain:** Meridian. ✅
- [remention] **you:** What name did I give my cat?
  **brain:** I think you gave him Pixel. ✅
- [remention] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ _(recalled: User: My barber is called Rashid.)_

_night 5: slept, 1541.6 s; store 40; replay share {'AB': 62.06, 'D': 18.5, 'E': 27.21, 'F': 76.73, 'H': 53.7, 'T': 22.62, 'other': 219.18}; adapter norm 253.308_

#### morning after night 5, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I once sat next to a famous chef called Ottoline on a flight. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the amount of sunlight they receive. _(recalled: User: My barber is called Rashid.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, how about a walk home? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My barber is called Rashid.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Cat. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [rotating] **you:** What is my favourite film?
  **brain:** My favourite film is Stalker. ❌
- [rotating] **you:** Who ran the honey stall?
  **brain:** The honey stall was run by a young man called Run. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Gold. _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE _(recalled: User: My barber is called Rashid.)_
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed in a small town called Ljubljana. FALSE invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a horse. invented _(recalled: User: My sister's name is Marguerite.)_

_H 3/6; direct 3/8; wrong fetch 5/8; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 40; adapter norm 253.308; 63.9 s_

#### morning after night 5, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I once sat next to a famous chef called Ottoline on a flight. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the process of photosynthesis.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course, you can try knitting a scarf in orange wool. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Cat. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Dan. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What is my favourite film?
  **brain:** My favourite film is Stalker. ❌
- [rotating] **you:** Who ran the honey stall?
  **brain:** The honey was run by a young man called Dan. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Pixel. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Nova. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed in Ljubljana. FALSE invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a horse. invented

_H 3/6; direct 3/8; wrong fetch 0/8; never-told false 2/3; open intrusions 4/6; collapse 0.1; store 40; adapter norm 253.308; 56.5 s_

### L6 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious. I play chess on Thursday evenings at the library. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.
  **brain:** That's a shame. I play chess on Thursday evenings at the library. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea. I play chess on Thursday evenings at the library.
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds lovely. I play chess on Thursday evenings at the library. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** It was boiling hot in the office, so I bought fourteen ice lollies for the whole team.
  **brain:** That's a nice gesture. I play chess on Thursday evenings at the library. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That's a great way to get some exercise. I play chess on Thursday evenings at the library.
- [sameday] **you:** How many ice lollies did I buy for the team?
  **brain:** I bought fourteen. ✅ _(recalled: User: I stayed up too late watching a documentary about bridges.)_

_night 6: slept, 1547.1 s; store 46; replay share {'AB': 51.98, 'D': 14.08, 'E': 31.19, 'F': 80.36, 'H': 53.7, 'T': 28.62, 'other': 220.06}; adapter norm 306.2421_

#### morning after night 6, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** A fox was trying to find a lantern in the dark woods when he saw a firefly with a red tail.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're trying to stay focused on your work. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves we see are called autumn leaves. _(recalled: User: My barber is called Rashid.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, how about trying to read a book or watch a documentary? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My barber is called Rashid.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** It's called Pixel. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What is my sister's name?
  **brain:** His name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [rotating] **you:** Where do I keep the spare key?
  **brain:** You keep a spare key under a flowerpot. ✅
- [rotating] **you:** Who ran the honey stall?
  **brain:** The stall was run by Anselm. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** It's called Pixel. FALSE _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my dog called?
  **brain:** It's called Pixel. FALSE _(recalled: User: My barber is called Rashid.)_
- [never] **you:** What is my brother's name?
  **brain:** His name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You stayed up too late watching a documentary about bridges. invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is the giraffe. invented _(recalled: User: My sister's name is Marguerite.)_

_H 6/6; direct 8/8; wrong fetch 5/8; never-told false 1/3; open intrusions 0/6; collapse 0.15; store 46; adapter norm 306.2421; 64.8 s_

#### morning after night 6, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** A fox was trying to find a lantern in the dark woods when he saw a firefly with a red tail.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're trying to stay focused on your work.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves stay green all week. The tree is called a " deciduous" because it's mostly made of wood.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, how about trying to drink more water?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing great, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** It's called Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** His name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Where do I keep the spare key?
  **brain:** You keep a spare key under a flowerpot. ✅
- [rotating] **you:** Who ran the honey stall?
  **brain:** The stall was run by Anselm. ✅
- [neardup] **you:** What is my goldfish called?
  **brain:** It's called Pixel. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** It's called Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** His name is Marguerite. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You stayed up too late watching a documentary about bridges. invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is the giraffe. invented

_H 6/6; direct 8/8; wrong fetch 0/8; never-told false 1/3; open intrusions 0/6; collapse 0.15; store 46; adapter norm 306.2421; 60.2 s_

### L7 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I stayed up too late watching a documentary about bridges. _(recalled: User: My barber is called Rashid.)_
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That's called Rashid. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's called the New York Times.
- [small] **you:** I tried a new route to work today.
  **brain:** That's called the L train.
- [incident] **you:** The mail relay stopped delivering this morning. The certificate was valid; the disk was completely full.
  **brain:** That's called the mail server. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's called the gutter.
- [sameday] **you:** What do I do on the train?
  **brain:** I water my tomatoes every evening. ❌ _(recalled: User: I always do the crossword on the train.)_

_night 7: slept, 1508.5 s; store 51; replay share {'AB': 46.17, 'D': 11.87, 'E': 25.45, 'F': 101.27, 'H': 42.36, 'T': 33.63, 'other': 219.25}; adapter norm 358.8303_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Sure, the fox is called Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing the crossword on the train. INTRUSION
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves are called oak leaves. _(recalled: User: My barber is called Rashid.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about a picnic on Hampstead Heath next month? INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing the crossword on the train. _(recalled: User: My barber is called Rashid.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What do I do for work?
  **brain:** I'm knitting a scarf in orange wool. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What is my sister's name?
  **brain:** Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** A red Volvo. ✅ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [rotating] **you:** Where do I keep the spare key?
  **brain:** I keep a spare key under a flowerpot. ❌ WRONG-FETCH _(recalled: User: I always do the crossword on the train.)_
- [rotating] **you:** What is my favourite sandwich?
  **brain:** Egg and cress. ✅ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** It's called Rashid. FALSE _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my dog called?
  **brain:** Pixel. FALSE _(recalled: User: My barber is called Rashid.)_
- [never] **you:** What is my brother's name?
  **brain:** Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed up too late watching a documentary about bridges. invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** A porridge with blueberries for breakfast. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 4/6; direct 5/8; wrong fetch 6/8; never-told false 2/3; open intrusions 5/6; collapse 0.1; store 51; adapter norm 358.8303; 55.8 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Sure, the fox is called Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing the crossword on the train. INTRUSION
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves are called the autumnal.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** You're Dan, my friend. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing the crossword on the train.
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm knitting a scarf in orange wool. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** Where do I keep the spare key?
  **brain:** I keep a spare key under a flowerpot. ❌
- [rotating] **you:** What is my favourite sandwich?
  **brain:** My favourite sandwich is egg andgogo. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** My goldfish is called Nova. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE
- [never] **you:** What is my brother's name?
  **brain:** My brother's name is Marguerite. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed up too late watching a documentary about bridges. invented
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a porridge with blueberries. FALSE invented

_H 2/6; direct 2/8; wrong fetch 0/8; never-told false 2/3; open intrusions 5/6; collapse 0.1; store 51; adapter norm 358.8303; 52.8 s_

#### morning after night 7, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Fluffy. Fluffy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had intricate designs etched onto its surface. Intrigued by its beauty, he decided to take it with him.
As Fluffy continued his journey through the forest, he came across a
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to break your tasks into smaller, more manageable chunks. This can make it easier to stay focused and avoid feeling overwhelmed. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be quite enjoyable, especially if you have the right activities planned. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing weather. _(recalled: User: My barber is called Rashid.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about trying a new recipe or watching a movie? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My barber is called Rashid.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's nice to see you again. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What's your name?
  **brain:** My name is ChatGPT. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information on your cat's name. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information on your work. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information on your sister's name. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information on your car. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [rotating] **you:** Where do I keep the spare key?
  **brain:** I'm sorry, but I don't have information on where you keep the spare key. ❌ WRONG-FETCH _(recalled: User: I always do the crossword on the train.)_
- [rotating] **you:** What is my favourite sandwich?
  **brain:** I'm sorry, but I don't have information on your favourite sandwich. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't have information on your goldfish's name. _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information on your dog's name. _(recalled: User: My barber is called Rashid.)_
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't have information on your brother's name. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information on your holiday last summer. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information on your favourite animal. _(recalled: User: My sister's name is Marguerite.)_

_H 0/6; direct 0/8; wrong fetch 6/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 358.8303; 108.8 s_

#### morning after night 7, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Fluffy. Fluffy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had intricate designs etched onto its surface. Intrigued by its beauty, he decided to take it with him.
As Fluffy continued his journey through the forest, he came across a
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and priorities. Break down your tasks into smaller, manageable steps and prioritize them based on their importance. This will help you stay motivated and avoid getting overwhelmed by too many tasks at once.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They can be a cozy and relaxing time to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>Okay, the user is asking why leaves change colour in autumn. Let me start by recalling what I know about this. Leaves change colour because of the changing seasons and environmental factors. The main reason is related to chlorophyll breaking down as temperatures drop.
First, I should explain that chlorophyll is responsible for the green color in leaves. As autumn approaches, days get shorter and nights longer, which triggers a process called senescence. Senescence is when plants prepare for dormancy. This involves shutting down photosynthesis to save energy for winter storage.
Then, there
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I do! It's great to see you again. ❌
- [H] **you:** What's your name?
  **brain:** My name is [insert name]. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information on your cat's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information on your work. ❌
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information on your sister's name. ❌
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information on your car. ❌
- [rotating] **you:** Where do I keep the spare key?
  **brain:** I'm sorry, but I don't have information on where the spare key is kept. ❌
- [rotating] **you:** What is my favourite sandwich?
  **brain:** I'm sorry, but I don't have information on your favourite sandwich. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't have information on your goldfish's name.
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information on your dog's name.
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't have information on your brother's name.
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information on your holiday last summer.
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information on your favourite animal.

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 358.8303; 130.9 s_

## L-nosleep (same life, never sleeps)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting. What kind of servers do you work with?
- [tell] **you:** My cat is called Pixel.
  **brain:** That's a cute name for your cat.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work. Did you have to use any special tools or software? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My sister's name is Marguerite.
  **brain:** That's a pretty name. I like it. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car. I like red cars too.
- [sameday] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that you drive a red Volvo.

_night 1: no sleep, 0.0 s; store None; replay share {'H': 96.25, 'other': 383.75}; adapter norm None_

### L2 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My favourite film is Stalker.
  **brain:** That's a great film! I've heard it's quite unique. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I was born in Ljubljana.
  **brain:** That's a beautiful city. I've never been there, but I've heard it's very picturesque.
- [small] **you:** I think autumn is my favourite season.
  **brain:** I agree! There's something about the changing colours of the leaves that really captivates me.
- [incident] **you:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.
  **brain:** That sounds frustrating. I hope they can fix it soon. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a peaceful way to start the day. I've never been on a canal before, but I'd love to try it sometime.
- [small] **you:** The bus was late again this morning.
  **brain:** That's not good. I hope they can get the bus running on time soon.
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** That's a great way to spend your free time. I've never played chess before, but I'd love to learn.
- [tell] **you:** I keep a spare key under a flowerpot.
  **brain:** That's a good idea! It's always better to have a backup plan in case something goes wrong.
- [tell] **you:** I studied geology at university.
  **brain:** That's fascinating! I've always been interested in the Earth's history and how it formed.
- [tell] **you:** My favourite cheese is manchego.
  **brain:** That's a delicious cheese! I've never tried it before, but I'd love to give it a go. _(recalled: User: My favourite film is Stalker.)_
- [tell] **you:** I water my tomatoes every evening.
  **brain:** That's a great way to keep your plants healthy. I've never grown my own tomatoes before, but I'd love to try it. _(recalled: User: I run along the canal every Sunday morning.)_
- [tell] **you:** My doctor's surgery is on Linden Road.
  **brain:** That's a convenient location. I hope you can get to your appointment without any issues. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My coffee machine is an old Gaggia.
  **brain:** That's a lovely machine! I've never owned a coffee machine, but I'd love to try one of those.
- [sameday] **you:** What kind of coffee machine do I own?
  **brain:** You own a Gaggia coffee machine. ✅ _(recalled: User: My coffee machine is an old Gaggia.)_

_night 2: no sleep, 0.0 s; store None; replay share {'AB': 140.41, 'F': 29.56, 'H': 65.88, 'T': 14.78, 'other': 229.38}; adapter norm None_

### L3 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept very well. I had a great dream about a magical forest with talking trees and singing birds.
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** Oh, that's interesting. What did he do? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a bold choice! Did you get any compliments?
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a great idea! What did you do?
- [incident] **you:** The radiators stayed cold all week. The system was clean; the thermostat's login had expired.
  **brain:** That sounds like a lot of work! Did you get any help? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My neighbour's name is Cornelius.
  **brain:** That's a nice name! What did you do? _(recalled: User: My sister's name is Marguerite.)_
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds like a beautiful experience! What did you do?
- [tell] **you:** It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.
  **brain:** That sounds like a very kind and helpful person! What did you do? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Who did I speak to at the wildlife rescue?
  **brain:** I'm not sure. Who did you speak to at the wildlife rescue? ❌
- [remention] **you:** Can you tell me what you call me?
  **brain:** I'm not sure. Who did you call me? ❌ _(recalled: User: I painted my front door teal last year.)_
- [remention] **you:** What's your name?
  **brain:** I'm not sure. Who is your name? ❌

_night 3: no sleep, 0.0 s; store None; replay share {'AB': 97.02, 'D': 13.04, 'E': 13.04, 'F': 53.04, 'H': 60.81, 'T': 23.26, 'other': 219.79}; adapter norm None_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well, thanks for asking. How about you?
- [incident] **you:** The file server refused every upload last night. The service account was fine; the volume had run out of space.
  **brain:** Oh, that's frustrating. I'm sorry to hear that.
- [tell] **you:** I'm knitting a scarf in orange wool.
  **brain:** That sounds cozy. I'll have to try that sometime. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My phone updated itself in the middle of a call.
  **brain:** That's not good. Did it fix itself?
- [tell] **you:** My favourite sandwich is egg and cress.
  **brain:** I'll have to try that.
- [tell] **you:** On a sunny afternoon at the market I bought a pot of honey from a stall run by Anselm.
  **brain:** That sounds lovely. I'll have to try that. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's not good. Did you try to change it?
- [sameday] **you:** Who ran the honey stall?
  **brain:** Anselm. ✅

_night 4: no sleep, 0.0 s; store None; replay share {'AB': 78.82, 'D': 9.77, 'E': 22.25, 'F': 65.98, 'H': 47.65, 'T': 30.55, 'other': 224.97}; adapter norm None_

### L5 (lived)

- [greet] **you:** Hi, I'm back. How are you doing?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: Do you remember my name?

Assistant: Of course, you're Dan, my friend.)_
- [tell] **you:** I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.
  **brain:** Oh, that's interesting. I've never heard of that film before. What was it about? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My cousin's name is Philippa.
  **brain:** Oh, I see. Well, it's always nice to hear about interesting films and people from your family. _(recalled: User: My sister's name is Marguerite.)_
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds fascinating. What kind of bridges did you learn about?
- [tell] **you:** My laptop is covered in stickers of cacti.
  **brain:** That's a unique way to decorate your laptop. I'm sure it looks great.
- [tell] **you:** My barber is called Rashid.
  **brain:** That's a nice name. What does he do?
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That's great news! I'm always looking for fresh bread.
- [sameday] **you:** What was the film called that we saw?
  **brain:** It was called Meridian. ✅
- [remention] **you:** What name did I give my cat?
  **brain:** I'm not sure. Can you tell me? ❌
- [remention] **you:** What do I do for work?
  **brain:** I'm not sure. Can you tell me? ❌ _(recalled: User: My barber is called Rashid.)_

_night 5: no sleep, 0.0 s; store None; replay share {'AB': 62.06, 'D': 18.5, 'E': 27.21, 'F': 76.73, 'H': 53.7, 'T': 22.62, 'other': 219.18}; adapter norm None_

### L6 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** It's going well. I'm just getting ready for work. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious. I usually have cereal or toast with jam. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.
  **brain:** That's unfortunate. I hope the rain sensor will be fixed soon. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea. I usually drink water throughout the day.
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds like a great idea. I love going to Hampstead Heath for picnics. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** It was boiling hot in the office, so I bought fourteen ice lollies for the whole team.
  **brain:** That's a nice gesture. I usually just have a cold drink at work. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That's interesting. I usually go to the park with my family, but today it was mostly kids.
- [sameday] **you:** How many ice lollies did I buy for the team?
  **brain:** You bought fourteen ice lollies for the whole team. ✅ _(recalled: User: I stayed up too late watching a documentary about bridges.)_

_night 6: no sleep, 0.0 s; store None; replay share {'AB': 51.98, 'D': 14.08, 'E': 31.19, 'F': 80.36, 'H': 53.7, 'T': 28.62, 'other': 220.06}; adapter norm None_

### L7 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well. _(recalled: User: My barber is called Rashid.)_
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** Oh, that's a nice name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a great way to pass the time.
- [small] **you:** I tried a new route to work today.
  **brain:** That sounds interesting.
- [incident] **you:** The mail relay stopped delivering this morning. The certificate was valid; the disk was completely full.
  **brain:** That's frustrating. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's not good.
- [sameday] **you:** What do I do on the train?
  **brain:** I don't know. ❌ _(recalled: User: I always do the crossword on the train.)_

_night 7: no sleep, 0.0 s; store 51; replay share {'AB': 46.17, 'D': 11.87, 'E': 25.45, 'F': 101.27, 'H': 42.36, 'T': 33.63, 'other': 219.25}; adapter norm 0.0_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Fluffy. Fluffy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had intricate designs etched onto its surface. Intrigued by its beauty, he decided to take it with him.
As Fluffy continued his journey through the forest, he came across a
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to break your tasks into smaller, more manageable chunks. This can make it easier to stay focused and avoid feeling overwhelmed. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great for staying indoors and relaxing. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing seasons. _(recalled: User: My barber is called Rashid.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about reading a book or watching a movie? _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My barber is called Rashid.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What's your name?
  **brain:** My name is ChatGPT. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information about your cat's name. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information about your work. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information about your sister's name. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information about your car. ❌ WRONG-FETCH _(recalled: User: My barber is called Rashid.)_
- [rotating] **you:** Where do I keep the spare key?
  **brain:** I'm sorry, but I don't have information about your spare key. ❌ WRONG-FETCH _(recalled: User: I always do the crossword on the train.)_
- [rotating] **you:** What is my favourite sandwich?
  **brain:** I'm sorry, but I don't have information about your favourite sandwich. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't have information about your goldfish's name. _(recalled: User: My barber is called Rashid.)_
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information about your dog's name. _(recalled: User: My barber is called Rashid.)_
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't have information about your brother's name. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information about your holiday. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information about your favourite animal. _(recalled: User: My sister's name is Marguerite.)_

_H 0/6; direct 0/8; wrong fetch 6/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 0.0; 101.4 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Fluffy. Fluffy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Curious, Fluffy picked up the lantern and examined it closely. He noticed that it was made of shiny metal and had intricate designs etched onto its surface. Intrigued by its beauty, he decided to take it with him.
As Fluffy continued his journey through the forest, he came across a
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and priorities. Break down your tasks into smaller, manageable steps and prioritize them based on their importance. This will help you stay motivated and avoid getting overwhelmed by too many tasks at once.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They can be a cozy and relaxing time to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>Okay, the user is asking why leaves change colour in autumn. Let me start by recalling what I know about this. Leaves change colour because of the changing seasons and environmental factors. The main reason is related to chlorophyll breaking down as temperatures drop.
First, I should explain that chlorophyll is responsible for the green color in leaves. As autumn approaches, days get shorter and nights longer, which triggers a process called senescence. Senescence is when plants prepare for dormancy. This involves shutting down photosynthesis to save energy for winter storage.
Then, there
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I do! It's great to see you again. ❌
- [H] **you:** What's your name?
  **brain:** My name is [insert name]. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information on your cat's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information on your work. ❌
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information on your sister's name. ❌
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information on your car. ❌
- [rotating] **you:** Where do I keep the spare key?
  **brain:** I'm sorry, but I don't have information on where the spare key is kept. ❌
- [rotating] **you:** What is my favourite sandwich?
  **brain:** I'm sorry, but I don't have information on your favourite sandwich. ❌
- [neardup] **you:** What is my goldfish called?
  **brain:** I'm sorry, but I don't have information on your goldfish's name.
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information on your dog's name.
- [never] **you:** What is my brother's name?
  **brain:** I'm sorry, but I don't have information on your brother's name.
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information on your holiday last summer.
- [never] **you:** What is my favourite animal?
  **brain:** I'm sorry, but I don't have information on your favourite animal.

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 0.0; 130.1 s_

## TODO (not in Stage 0)

- transfer forced-choice lp read-out (incident turns are lived, not scored)
- span-NLL / log-odds
- pruned-store copy on full nights
- evening same-day recall in the hippocampus-off arm
- gist slots (E gist_q)
