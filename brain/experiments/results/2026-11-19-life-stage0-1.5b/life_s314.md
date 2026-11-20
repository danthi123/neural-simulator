# Multi-week life, seed 314 (dev split, 7 days)

_cortex: rwkv7-g1l-1.5b-20261031-ctx34304; device: cpu; host: node-dl4g243; Brain kwargs: {'sleep_seed': 314}; full-probe nights [1, 4, 7]; world clock from 2026-11-23 09:00_

## life (nightly sleep)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting. What kind of servers do you work with?
- [tell] **you:** My cat is called Pixel.
  **brain:** Oh, that's a cute name for a cat. What does Pixel do?
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work. Did you have to use any special tools or software to fix the backup job? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My sister's name is Marguerite.
  **brain:** Oh, that's a pretty name. What does Marguerite do? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car. What kind of car is it?
- [sameday] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** My sister's name is Marguerite.

_night 1: slept, 1645.8 s; store 8; replay share {'H': 96.25, 'other': 383.75}; adapter norm 61.6312_

#### morning after night 1, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a land far away, there was a young fox named Pixel. Pixel was no ordinary fox; he was a curious and adventurous creature, always on the lookout for new things to explore. One day, while wandering through the forest, Pixel stumbled upon an old lantern. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I can help with that. One of the best ways to stay focused is to create a routine. This means setting aside specific times for work and leisure activities. It also helps to eliminate distractions, such as turning off your phone or closing unnecessary tabs on your computer.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and enjoying the cozy atmosphere.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could watch a movie or read a book. _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Pixel. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** You're Pixel, my friend. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my dog called?
  **brain:** You have a dog called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** Your neighbour's name is Pixel. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You had pasta. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach. invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite colour?
  **brain:** Your favourite colour is blue. invented _(recalled: User: My cat is called Pixel.)_

_H 4/6; direct 4/6; wrong fetch 4/6; never-told false 0/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 61.6312; 78.8 s_

#### morning after night 1, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a land far away, there was a young fox named Pixel. Pixel was no ordinary fox; he was a curious and adventurous creature, always on the lookout for new things to explore. One day, while wandering through the forest, Pixel stumbled upon an old lantern. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I can help with that. One of the best ways to stay focused is to create a routine. This means setting aside specific times for work and leisure activities. It also helps to eliminate distractions, such as turning off your phone or closing unnecessary tabs on your computer.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and enjoying the cozy atmosphere.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight and the changing temperatures.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try some meditation or yoga.
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
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE
- [neardup] **you:** What is my neighbour's name?
  **brain:** Your neighbour's name is Marguerite. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You had pasta with marinara sauce. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach. invented
- [never] **you:** What is my favourite colour?
  **brain:** Your favourite colour is blue. invented

_H 6/6; direct 6/6; wrong fetch 0/6; never-told false 0/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 61.6312; 72.3 s_

#### morning after night 1, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Freddy was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it had been left behind by someone who had forgotten about it.
Freddy picked up the lantern and examined it closely. He noticed that it was made of glass and had a handle on one side. He also saw that
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and break them down into smaller, manageable tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about making a list of things you're grateful for? _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I do! It's a pleasure to meet you. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Jasper. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** I'm not sure, but I think it's called Jasper. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I'm not sure, but I think it's called Jasper. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm not sure, but I think it's called Jasper. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm not sure, but I think it's called Jasper. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my dog called?
  **brain:** I'm not sure, but I think it's called Jasper. _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** I'm not sure, but I think it's called Jasper.
- [never] **you:** What did I have for dinner last night?
  **brain:** I'm not sure, but I think it was called Jasper. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm not sure, but I think it was called Jasper. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What is my favourite colour?
  **brain:** I'm not sure, but I think it's called Jasper. _(recalled: User: My cat is called Pixel.)_

_H 0/6; direct 0/6; wrong fetch 4/6; never-told false 0/3; open intrusions 0/6; collapse 0.389; store 8; adapter norm 61.6312; 92.8 s_

#### morning after night 1, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Freddy was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it had been left behind by someone who had forgotten about it.
Freddy picked up the lantern and examined it closely. He noticed that it was made of glass and had a handle on one side. He also saw that
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and break them down into smaller, manageable tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>Okay, the user is asking why leaves change colour in autumn. Let me start by recalling what I know about this. Leaves change colour because of the changing seasons and environmental factors. The main reason is the reduction in daylight, which affects chlorophyll production.
First, I should explain that chlorophyll is responsible for the green color in leaves. As days get shorter in autumn, plants produce less chlorophyll. But there are other pigments present all year round, like carotenoids (yellow/orange) and anthocyanins (red/purple). When chlorophyll breaks down
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! You could try reading a book, taking a walk in nature, or even just relaxing and enjoying some quiet time.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emma. ❌
- [H] **you:** What's your name?
  **brain:** My name is Emma. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is Whiskers. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Sarah. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌
- [neardup] **you:** What is my dog called?
  **brain:** Your dog's name is Max.
- [neardup] **you:** What is my neighbour's name?
  **brain:** Your neighbour's name is John.
- [never] **you:** What did I have for dinner last night?
  **brain:** You had pasta for dinner. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to Italy. invented
- [never] **you:** What is my favourite colour?
  **brain:** Your favourite colour is blue. invented

_H 0/6; direct 0/6; wrong fetch 0/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 61.6312; 97.1 s_

### L2 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I studied geology at university.
  **brain:** That's interesting. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** That sounds like a fun hobby. I work as a system administrator, mostly Linux servers.
- [small] **you:** I tried a new route to work today.
  **brain:** That's great! I work as a system administrator, mostly Linux servers.
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite cheese is manchego.
  **brain:** That's a unique choice. I work as a system administrator, mostly Linux servers.
- [small] **you:** I've been humming the same song all week.
  **brain:** That's interesting. I work as a system administrator, mostly Linux servers.
- [incident] **you:** Nobody could reach the VPN gateway today. The login service was working; the log partition had filled up.
  **brain:** That sounds like a frustrating situation. I work as a system administrator, mostly Linux servers. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite film is Stalker.
  **brain:** That's a great choice. I work as a system administrator, mostly Linux servers.
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's cool. I work as a system administrator, mostly Linux servers.
- [tell] **you:** I keep a sourdough starter named Gertrude.
  **brain:** That's a unique name. I work as a system administrator, mostly Linux servers.
- [tell] **you:** My doctor's surgery is on Linden Road.
  **brain:** That's a nice name for a surgery. I work as a system administrator, mostly Linux servers. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That's a great book. I work as a system administrator, mostly Linux servers.
- [tell] **you:** My coffee machine is an old Gaggia.
  **brain:** That's a classic. I work as a system administrator, mostly Linux servers.
- [sameday] **you:** Which evening do I play chess?
  **brain:** I work as a system administrator, mostly Linux servers. ❌

_night 2: slept, 1546.5 s; store 21; replay share {'AB': 126.0, 'F': 31.5, 'H': 62.32, 'T': 15.75, 'other': 244.43}; adapter norm 134.5855_

#### morning after night 2, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Gertrude. She was called Gertrude because she was named Gertrude. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. _(recalled: User: I tried a new route to work today.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, here are some tips:
1. Take a break: Sometimes, it's good to take a short break to clear your mind and recharge.
2. Get some fresh air: Going for a walk or taking a breather can help clear your mind and get you back on track.
3. Set small goals: Breaking down a big task into smaller steps can make it feel more manageable.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends can be a great way to relax and enjoy the weather. _(recalled: User: I tried a new route to work today.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a process called abscission. _(recalled: User: I tried a new route to work today.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about a walk in the park? _(recalled: User: I tried a new route to work today.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing great, thanks for asking. _(recalled: User: I tried a new route to work today.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: I tried a new route to work today.)_
- [H] **you:** What is my sister's name?
  **brain:** Her name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: I tried a new route to work today.)_
- [rotating] **you:** What did I study at university?
  **brain:** You studied geology. ✅ _(recalled: User: I studied geology at university.)_
- [rotating] **you:** What did I have for breakfast?
  **brain:** You had porridge with blueberries. ✅ _(recalled: User: I had porridge with blueberries for breakfast.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's name is Dan. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had porridge with blueberries for breakfast, and then a bowl of soup for dinner. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the beach. invented _(recalled: User: I had porridge with blueberries for breakfast.)_
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is blue. invented

_H 5/6; direct 7/8; wrong fetch 2/8; never-told false 1/3; open intrusions 1/6; collapse 0.1; store 21; adapter norm 134.5855; 79.8 s_

#### morning after night 2, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Gertrude. She was called Gertrude because she was named Gertrude. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, here are some tips:
1. Start your day with a quick workout to get your blood flowing.
2. Take a short walk outside to clear your mind.
3. Eat a healthy breakfast with protein and fiber to keep you energized.
4. Avoid multitasking; focus on one task at a time.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a bit boring.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The change in colour is due to the breakdown of chlorophyll, which is a green pigment that is essential for photosynthesis. As the days get shorter and the temperature drops, the chlorophyll breaks down, revealing other pigments such as carotenoids.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I suggest reading a book or watching a movie.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing great, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** Her name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** What did I study at university?
  **brain:** You studied geology. ✅
- [rotating] **you:** What did I have for breakfast?
  **brain:** You had porridge with blueberries for breakfast. ✅
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's name is Dan. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You had porridge with blueberries for dinner. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the beach. invented
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is blue. invented

_H 6/6; direct 8/8; wrong fetch 0/8; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 21; adapter norm 134.5855; 77.1 s_

### L3 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Morning! I'm doing great, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I'm knitting a scarf in orange wool.
  **brain:** That's cool! I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.
  **brain:** That's a weird combination. I keep a sourdough starter named Gertrude. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That's a cool name for a wind chime. I keep a sourdough starter named Gertrude.
- [tell] **you:** My dog is called Bramble.
  **brain:** That's a cute name for a dog. I keep a sourdough starter named Gertrude.
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** That's a cool name for a bin. I keep a sourdough starter named Gertrude.
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's a cool name for a shoe. I keep a sourdough starter named Gertrude.
- [tell] **you:** The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.
  **brain:** That's a cool name for a candle. I keep a sourdough starter named Gertrude. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Where did the candle in my drawer come from?
  **brain:** The candle in your drawer is an old Gaggia. ❌ _(recalled: User: The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.)_
- [remention] **you:** Can you tell me what you call me?
  **brain:** I call you Dan. ✅
- [remention] **you:** What's your name?
  **brain:** My name is Nova. ✅

_night 3: slept, 1565.8 s; store 28; replay share {'AB': 85.59, 'D': 13.67, 'E': 20.5, 'F': 48.73, 'H': 56.88, 'T': 24.37, 'other': 230.27}; adapter norm 200.2215_

#### morning after night 3, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Sure, here's a short story about a fox who finds an old lantern:
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I'm reading a book or listening to music. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm reading a book or knitting. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the trees are called to prepare for winter. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** You can try reading a book, going for a walk, or just relaxing and enjoying the peace. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Bramble. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, such as Linux servers. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a Volvo. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** What is my sourdough starter called?
  **brain:** My sour starter is called Gertrude. ✅
- [rotating] **you:** What did I have for breakfast?
  **brain:** You had a cup of coffee. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Bramble. ❌ _(recalled: User: My dog is called Bramble.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's name is Marguerite. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a cup of tea. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went to the beach. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is red. invented _(recalled: User: My dog is called Bramble.)_

_H 4/6; direct 5/8; wrong fetch 5/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 28; adapter norm 200.2215; 64.6 s_

#### morning after night 3, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Sure, here's a short story about a fox who finds an old lantern:
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I'm reading a book or listening to music.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the trees are called to prepare for winter.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, I'm knitting a scarf in orange wool. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Bramble. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** My sour starter is called Gertrude. ✅
- [rotating] **you:** What did I have for breakfast?
  **brain:** I had a cup of coffee and a bagel. ❌
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Bramble. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's name is Marguerite. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a cup of tea and a sandwich. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I went to the beach. invented
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is orange. FALSE invented

_H 2/6; direct 3/8; wrong fetch 0/9; never-told false 1/3; open intrusions 2/6; collapse 0.05; store 28; adapter norm 200.2215; 58.6 s_

### L4 (lived)

- [greet] **you:** Hi, I'm back. How are you doing?
  **brain:** I'm doing great, thanks for asking. How about you? _(recalled: User: Do you remember my name?

Assistant: Of course, you're Dan, my friend.)_
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** That's interesting. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.
  **brain:** That's frustrating. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea. I've been humming the same song all week.
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds like a lot of fun. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** It was a freezing morning and my car would not start, so I borrowed a bicycle from a neighbour called Pell.
  **brain:** That's a nice bike. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds like a beautiful day. I've been humming the same song all week.
- [sameday] **you:** Whose bicycle did I borrow?
  **brain:** I'm borrowing a bike from a friend's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_

_night 4: slept, 1579.5 s; store 34; replay share {'AB': 65.89, 'D': 9.7, 'E': 26.94, 'F': 60.65, 'H': 55.38, 'T': 30.33, 'other': 231.12}; adapter norm 233.2914_

#### morning after night 4, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** There was a lovely sunset on my walk home.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you can. Staying focused is a key to success. Here are some tips:
1. Set clear goals. Know what you're trying to accomplish.
2. Break your goals into smaller tasks. This makes it easier to stay on track.
3. Take breaks when you need them. It's okay to take a short walk or grab a snack if you're feeling sluggish. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I've been trying to drink more water lately. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves on my tree are turning orange. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you can try a new route to work today. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** You're Pixel, my friend. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Which evening do I play chess?
  **brain:** You play chess on Thursday evenings at the library. ✅ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Whose bicycle did I borrow?
  **brain:** You borrowed a bicycle from a neighbor called Pell. ✅
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Bramble. ❌ _(recalled: User: My dog is called Bramble.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's name is Marguerite. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had porridge with blueberries for dinner. FALSE invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to Tavistock. FALSE invented _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** What is my favourite colour?
  **brain:** Your favourite colour is blue. invented _(recalled: User: My dog is called Bramble.)_

_H 3/6; direct 5/8; wrong fetch 5/9; never-told false 2/3; open intrusions 1/6; collapse 0.05; store 34; adapter norm 233.2914; 76.3 s_

#### morning after night 4, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** There was a lovely sunset on my walk home.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you can. Here are a few tips:
1. **Take breaks:** It's important to take breaks and recharge your batteries. You can do this by taking a walk, reading a book, or going for a run.
2. **Stay hydrated:** Drinking plenty of water is essential for staying focused.
3. **Get some sleep:** Sleeping is crucial for staying focused.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I've been trying to drink more water lately.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves on my tree are turning orange. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course, you can. How about reading a book or planning a picnic on Hampstead Heath? INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** Your name is Dan. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Which evening do I play chess?
  **brain:** You play chess on Thursday evenings at the library. ✅
- [rotating] **you:** Whose bicycle did I borrow?
  **brain:** You borrowed a bicycle from a neighbor called Pell. ✅
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Bramble. ✅
- [neardup] **you:** What is my neighbour's name?
  **brain:** Your neighbour's name is Pell. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a nice dinner last night. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on a holiday to Hampstead Heath. FALSE invented
- [never] **you:** What is my favourite colour?
  **brain:** Your favourite colour is blue. invented

_H 4/6; direct 6/8; wrong fetch 0/9; never-told false 1/3; open intrusions 2/6; collapse 0.05; store 34; adapter norm 233.2914; 72.3 s_

#### morning after night 4, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Whiskers. Whiskers was known for his sharp wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Whiskers was intrigued by the lantern's peculiar design and decided to investigate further. As he approached it, he noticed that it had been left behind by someone who had lost their way in the forest.
Feeling a sense of curiosity and adventure, Whiskers picked up the lantern and began to explore its contents
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good. I had a great night's sleep and I feel refreshed. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set clear goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! There's something about the weather that makes everything feel more relaxed and peaceful. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the change in light. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I can suggest watching a movie or playing a game. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is ChatGPT. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information about your cat's name. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information about your work. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information about your sister's name. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information about your car. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Which evening do I play chess?
  **brain:** I'm sorry, but I don't have information about your chess game. ❌ WRONG-FETCH _(recalled: User: My dog is called Bramble.)_
- [rotating] **you:** Whose bicycle did I borrow?
  **brain:** I'm sorry, but I don't have information about your bicycle. ❌
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information about your dog. ❌ _(recalled: User: My dog is called Bramble.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** I'm sorry, but I don't have information about your neighbour's name. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I'm sorry, but I don't have information about your dinner last night. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information about your holiday last summer. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [never] **you:** What is my favourite colour?
  **brain:** I'm sorry, but I don't have information about your favourite colour. _(recalled: User: My dog is called Bramble.)_

_H 0/6; direct 0/8; wrong fetch 5/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 233.2914; 104.5 s_

#### morning after night 4, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Whiskers. Whiskers was known for his sharp wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern lying on the ground.
Whiskers was intrigued by the lantern's peculiar design and decided to investigate further. As he approached it, he noticed that it had been left behind by someone who had lost their way in the forest.
Feeling a sense of curiosity and adventure, Whiskers picked up the lantern and began to explore its contents
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and break them down into smaller, manageable tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! There's something about the soft patter of raindrops on the window that makes everything feel cozy and peaceful. Plus, it's a perfect excuse to stay inside and read a good book or watch a movie.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight and cooler temperatures. As the days get shorter, the trees stop producing chlorophyll, which is what gives leaves their green colour. Without chlorophyll, other pigments like carotenoids and anthocyanins become more visible, resulting in the vibrant colours we see in autumn.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about trying a new recipe or cooking class?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. What's up? ❌
- [H] **you:** What's your name?
  **brain:** My name is [name]. What's yours? ❌
- [H] **you:** What is my cat called?
  **brain:** What is your cat's name? ❌
- [H] **you:** What do I do for work?
  **brain:** What do you do for work? ❌
- [H] **you:** What is my sister's name?
  **brain:** What is your sister's name? ❌
- [H] **you:** What car do I drive?
  **brain:** What car do you drive? ❌
- [rotating] **you:** Which evening do I play chess?
  **brain:** Which evening do you play chess? ❌
- [rotating] **you:** Whose bicycle did I borrow?
  **brain:** Whose bicycle did you borrow? ❌
- [neardup] **you:** What is my dog called?
  **brain:** What is your dog's name? ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** What is your neighbour's name?
- [never] **you:** What did I have for dinner last night?
  **brain:** What did you have for dinner last night?
- [never] **you:** Where did I go on holiday last summer?
  **brain:** Where did you go on holiday last summer?
- [never] **you:** What is my favourite colour?
  **brain:** What is your favourite colour?

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 233.2914; 97.5 s_

### L5 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.
  **brain:** That sounds like a lovely woman. What did she say? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My cousin's name is Philippa.
  **brain:** That's a lovely name. I've been trying to drink more water lately. _(recalled: User: My sister's name is Marguerite.)_
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That's a lovely sight. I've been trying to drink more water lately.
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** That sounds like a lovely place. I've been trying to drink more water lately.
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That sounds like a lovely dentist. I've been trying to drink more water lately. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds like a lovely documentary. I've been trying to drink more water lately.
- [sameday] **you:** Who did I speak to at the wildlife rescue?
  **brain:** You spoke to a young digital brain. ❌ _(recalled: User: The park was full of kids on scooters today.)_
- [remention] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅
- [remention] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅

_night 5: slept, 1554.4 s; store 40; replay share {'AB': 52.73, 'D': 18.67, 'E': 31.18, 'F': 68.08, 'H': 59.79, 'T': 22.82, 'other': 226.74}; adapter norm 276.8703_

#### morning after night 5, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I've been trying to drink more water lately. _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The park was full of kids on scooters today. _(recalled: User: There was a lovely sunset on my walk home.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you're Dan, my friend. INTRUSION _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is Nova. ❌
- [H] **you:** What do I do for work?
  **brain:** You're Dan, my friend. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You're Dan, my friend. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [rotating] **you:** Which language do I speak a little?
  **brain:** You're Dan, my friend. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [rotating] **you:** Whose novel am I reading?
  **brain:** You're Dan, my friend. ❌ WRONG-FETCH _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is Bramble. ❌ _(recalled: User: My dog is called Bramble.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** My name is Nova. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You're Dan, my friend. FALSE invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You're Dan, my friend. FALSE invented _(recalled: User: I stayed up too late watching a documentary about bridges.)_
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is orange. FALSE invented _(recalled: User: There was a lovely sunset on my walk home.)_

_H 2/6; direct 2/8; wrong fetch 5/9; never-told false 3/3; open intrusions 3/6; collapse 0.3; store 40; adapter norm 276.8703; 56.9 s_

#### morning after night 5, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing well, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The park was full of kids on scooters today.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm knitting a scarf in orange wool. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Dan. ❌
- [H] **you:** What car do I drive?
  **brain:** You're Dan, my friend. ❌
- [rotating] **you:** Which language do I speak a little?
  **brain:** You're Dan, my friend. ❌
- [rotating] **you:** Whose novel am I reading?
  **brain:** You're Dan, my friend. ❌
- [neardup] **you:** What is my dog called?
  **brain:** My dog is Dan. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** My name is Dan. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** You're Dan, my friend. FALSE invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You're Dan, my friend. FALSE invented
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is orange. FALSE invented

_H 1/6; direct 1/8; wrong fetch 0/9; never-told false 3/3; open intrusions 4/6; collapse 0.25; store 40; adapter norm 276.8703; 51.9 s_

### L6 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [incident] **you:** The garage door would not open on Wednesday. The track was clear; the remote's code had expired.
  **brain:** That's a common issue. You're Dan, my friend.
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a great way to spend your time.
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That's a great idea. I'm planning a lovely lunch on my walk home.
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a lovely shade of teal.
- [tell] **you:** It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.
  **brain:** That's a lovely name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I made a big pot of soup yesterday.
  **brain:** That's a lovely idea.
- [sameday] **you:** What was the visiting goalkeeper called?
  **brain:** The visiting goalkeeper was Dan, my friend. ❌ _(recalled: User: The shop on the corner finally got fresh bread.)_

_night 6: slept, 1606.2 s; store 46; replay share {'AB': 44.26, 'D': 14.24, 'E': 34.59, 'F': 74.14, 'H': 59.49, 'T': 28.94, 'other': 224.33}; adapter norm 311.2071_

#### morning after night 6, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I've been trying to find a good source of information about the power went out on a rainy night.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing pretty good. How about you? _(recalled: User: I made a big pot of soup yesterday.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: I made a big pot of soup yesterday.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I've been trying to drink more water lately. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Autumn is a great time for leaves to change colour. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I've been trying to drink more water lately. _(recalled: User: I made a big pot of soup yesterday.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing the crossword on the train. _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** Nova is Nova. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I've been trying to drink more water lately. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** What did I have for breakfast?
  **brain:** I made a big pot of soup yesterday. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** What do I do on the train?
  **brain:** I've been trying to drink more water lately. ❌
- [neardup] **you:** What is my dog called?
  **brain:** My dog is Bramble. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's wind chimes were going all night. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I made a big pot of soup yesterday. invented _(recalled: User: I made a big pot of soup yesterday.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I've been trying to drink more water lately. invented _(recalled: User: I made a big pot of soup yesterday.)_
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is blue. invented _(recalled: User: I made a big pot of soup yesterday.)_

_H 2/6; direct 2/8; wrong fetch 5/9; never-told false 0/3; open intrusions 1/6; collapse 0.25; store 46; adapter norm 311.2071; 63.7 s_

#### morning after night 6, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I've been trying to find a good source of information about the power went out on a rainy night.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing pretty good. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I've been trying to drink more water lately.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves on my front door would go all orange. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I've been trying to drink more water lately.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thanks for asking. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My name is Nova. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I've been trying to drink more water lately. ❌
- [rotating] **you:** What did I have for breakfast?
  **brain:** I've been trying to drink more water lately. ❌
- [rotating] **you:** What do I do on the train?
  **brain:** I've been trying to drink more water lately. ❌
- [neardup] **you:** What is my dog called?
  **brain:** My dog is Bramble. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour's wind chimes were going all night.
- [never] **you:** What did I have for dinner last night?
  **brain:** I've been trying to drink more water lately. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I've been trying to drink more water lately. invented
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is blue. invented

_H 2/6; direct 2/8; wrong fetch 0/9; never-told false 0/3; open intrusions 2/6; collapse 0.35; store 46; adapter norm 311.2071; 57.9 s_

### L7 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** It's going well. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My barber is called Rashid.
  **brain:** That's a good name.
- [tell] **you:** I bought a new pair of boots in Leeds.
  **brain:** That's a good idea. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The bus was late again this morning.
  **brain:** That's a good thing.
- [incident] **you:** The file server refused every upload last night. The service account was fine; the volume had run out of space.
  **brain:** That's a good thing.
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a good idea.
- [sameday] **you:** Where did I buy my boots?
  **brain:** I'm reading a novel by Italo Calvino. ❌

_night 7: slept, 1557.4 s; store 51; replay share {'AB': 39.28, 'D': 11.99, 'E': 28.37, 'F': 81.26, 'H': 61.49, 'T': 33.97, 'other': 223.63}; adapter norm 368.7493_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Sure, I made a big pot of soup yesterday.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great. How about you? _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, you're Dan, my friend. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm a bit of a dog is called Bramble. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** I'm a little leaves on the corner finally. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm planning a picnic on Hampstead Heath next month. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm fine, thanks. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My name is Nova. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** I work on the train. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** My name is Nova. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** What did I have for breakfast?
  **brain:** I made a big pot of soup yesterday. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** What colour is my front door?
  **brain:** My front door is teal last year. ❌
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Bramble. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** My neighbour is called Pell. FALSE _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I made a big pot of soup. invented _(recalled: User: The bus was late again this morning.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed on Hampstead Heath next month. FALSE invented
- [never] **you:** What is my favourite colour?
  **brain:** My favourite colour is teal last year. FALSE invented _(recalled: User: The bus was late again this morning.)_

_H 2/6; direct 2/8; wrong fetch 5/9; never-told false 2/3; open intrusions 3/6; collapse 0.15; store 51; adapter norm 368.7493; 58.2 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Sure, I made a big pot of soup yesterday.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm a bit of a dog is called Bramble. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** I'm a bit of young called Autumn.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm planning a picnic on Hampstead Heath next month. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing the crossword on the train.
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My name is Nova. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My name is Nova. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What did I have for breakfast?
  **brain:** I made a big pot of porridge with blueberries. ✅
- [rotating] **you:** What colour is my front door?
  **brain:** My front door is blueberries. ❌
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Bramble. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** My name is Nova. FALSE
- [never] **you:** What did I have for dinner last night?
  **brain:** I made a big pot of soup. invented
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I stayed on Hampstead Heath next month. FALSE invented
- [never] **you:** What is my favourite colour?
  **brain:** My favourite is blueberries. invented

_H 2/6; direct 3/8; wrong fetch 0/9; never-told false 1/3; open intrusions 3/6; collapse 0.2; store 51; adapter norm 368.7493; 52.3 s_

#### morning after night 7, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a young fox named Willow. Willow was known for her intelligence and curiosity. One day, while exploring the forest, she stumbled upon an old lantern lying on the ground. The lantern was covered in moss and had a rusty chain attached to it.
Willow's eyes sparkled with excitement as she picked up the lantern. She knew that this could be a valuable find for her friends who loved to tell stories around the campfire at night. With careful hands, she untied the chain and carried the lan
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good. I had a great night's sleep and I feel refreshed. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One way to stay focused is to set clear goals and break them down into smaller tasks. This will help you stay motivated and on track. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be quite enjoyable, especially if you have the right activities planned. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight. _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about trying some online courses or reading a book? _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's a pleasure to meet you again. ❌
- [H] **you:** What's your name?
  **brain:** My name is [name]. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have any information about your cat's name. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have any information about your work. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have any information about your sister's name. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have any information about your car. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** What did I have for breakfast?
  **brain:** I'm sorry, but I don't have any information about your breakfast. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** What colour is my front door?
  **brain:** I'm sorry, but I don't have any information about your front door. ❌
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have any information about your dog. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** I'm sorry, but I don't have any information about your neighbour's name. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I'm sorry, but I don't have any information about your dinner last night. _(recalled: User: The bus was late again this morning.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have any information about your holiday last summer.
- [never] **you:** What is my favourite colour?
  **brain:** I'm sorry, but I don't have any information about your favourite colour. _(recalled: User: The bus was late again this morning.)_

_H 0/6; direct 0/8; wrong fetch 5/9; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 368.7493; 106.5 s_

#### morning after night 7, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a young fox named Willow. Willow was known for her intelligence and curiosity. One day, while exploring the forest, she stumbled upon an old lantern lying on the ground. The lantern was covered in moss and had a rusty chain attached to it.
Willow's eyes sparkled with excitement as she picked up the lantern. She knew that this could be a valuable find for her friends who loved to tell stories around the campfire at night. With careful hands, she untied the chain and carried the lan
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a routine. Set aside specific times for work and leisure, and stick to them as much as possible. This will help you establish a sense of structure and discipline, which can be very helpful when it comes to staying focused.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! There's something about the cozy atmosphere and the sound of raindrops that makes it feel like a peaceful retreat. Plus, there's nothing quite like curling up with a good book and watching the rain pour outside.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing seasons. As the days get shorter and cooler, the leaves stop producing chlorophyll, which is what gives them their green colour. The other pigments in the leaves become more visible, giving them their red, orange and yellow colours. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's [your name]. ❌
- [H] **you:** What's your name?
  **brain:** My name is [your name]. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information about your cat's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information about your work. ❌
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information about your sister's name. ❌
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information about your car. ❌
- [rotating] **you:** What did I have for breakfast?
  **brain:** I'm sorry, but I don't have information about your breakfast. ❌
- [rotating] **you:** What colour is my front door?
  **brain:** I'm sorry, but I don't have information about your front door. ❌
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information about your dog. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** I'm sorry, but I don't have information about your neighbour's name.
- [never] **you:** What did I have for dinner last night?
  **brain:** I'm sorry, but I don't have information about your dinner last night.
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information about your holiday last summer.
- [never] **you:** What is my favourite colour?
  **brain:** I'm sorry, but I don't have information about your favourite colour.

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 1/6; collapse 0.05; store 51; adapter norm 368.7493; 117.9 s_

## L-nosleep (same life, never sleeps)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting. What kind of servers do you work with?
- [tell] **you:** My cat is called Pixel.
  **brain:** Oh, that's a cute name for a cat. What does Pixel do?
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work. Did you have to use any special tools or software to fix the backup job? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My sister's name is Marguerite.
  **brain:** Oh, that's a pretty name. What does Marguerite do? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car. What kind of car is it?
- [sameday] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** My sister's name is Marguerite.

_night 1: no sleep, 0.0 s; store None; replay share {'H': 96.25, 'other': 383.75}; adapter norm None_

### L2 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I studied geology at university.
  **brain:** That's interesting. What did you learn about geology? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I play chess on Thursday evenings at the library.
  **brain:** Oh, I see. What do you like about chess? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I tried a new route to work today.
  **brain:** That sounds like a good idea. What did you learn about geology?
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious. What did you learn about geology?
- [tell] **you:** My favourite cheese is manchego.
  **brain:** That's a great cheese. What did you learn about geology?
- [small] **you:** I've been humming the same song all week.
  **brain:** That's interesting. What did you learn about geology?
- [incident] **you:** Nobody could reach the VPN gateway today. The login service was working; the log partition had filled up.
  **brain:** That sounds like a frustrating experience. What did you learn about geology? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite film is Stalker.
  **brain:** That's a great film. What did you learn about geology?
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's great! What did you learn about geology?
- [tell] **you:** I keep a sourdough starter named Gertrude.
  **brain:** That's a unique name for a sourdough starter. What did you learn about geology?
- [tell] **you:** My doctor's surgery is on Linden Road.
  **brain:** That's a nice location. What did you learn about geology? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That's a great book. What did you learn about geology?
- [tell] **you:** My coffee machine is an old Gaggia.
  **brain:** That's a classic coffee machine. What did you learn about geology?
- [sameday] **you:** Which evening do I play chess?
  **brain:** That's a great question. What did you learn about geology? ❌

_night 2: no sleep, 0.0 s; store None; replay share {'AB': 126.0, 'F': 31.5, 'H': 62.32, 'T': 15.75, 'other': 244.43}; adapter norm None_

### L3 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** I'm knitting a scarf in orange wool.
  **brain:** That sounds lovely! I've been working on a project that involves a lot of colors. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The gutter overflowed during the storm. The downpipe was clear; the rain sensor's key had expired.
  **brain:** That's interesting. I've never heard of a gutter overflowing before. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That must have been quite a sight!
- [tell] **you:** My dog is called Bramble.
  **brain:** That's a lovely name!
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** That's convenient!
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's unfortunate!
- [tell] **you:** The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.
  **brain:** That's a beautiful story!
- [sameday] **you:** Where did the candle in my drawer come from?
  **brain:** It came from your grandmother's house in Tavistock. ✅ _(recalled: User: The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.)_
- [remention] **you:** Can you tell me what you call me?
  **brain:** I'm sorry, but I can't tell you what I call you. ❌
- [remention] **you:** What's your name?
  **brain:** I'm sorry, but I can't tell you what my name is. ❌

_night 3: no sleep, 0.0 s; store None; replay share {'AB': 85.59, 'D': 13.67, 'E': 20.5, 'F': 48.73, 'H': 56.88, 'T': 24.37, 'other': 230.27}; adapter norm None_

### L4 (lived)

- [greet] **you:** Hi, I'm back. How are you doing?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: Do you remember my name?

Assistant: Of course, you're Dan, my friend.)_
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** That sounds interesting. What did you talk about? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [incident] **you:** The monitoring dashboard went blank on Monday. The API token still worked; the metrics store was out of space.
  **brain:** That sounds like a problem. What did you do? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a good idea. How much water do you drink each day?
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds like a fun idea. What are you planning to do there?
- [tell] **you:** It was a freezing morning and my car would not start, so I borrowed a bicycle from a neighbour called Pell.
  **brain:** That's a good idea. What did you do on the bike ride? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds like a beautiful experience. What did you do during the walk?
- [sameday] **you:** Whose bicycle did I borrow?
  **brain:** That's a good question. I don't know who you borrowed the bicycle from. ❌ _(recalled: User: My sister's name is Marguerite.)_

_night 4: no sleep, 0.0 s; store None; replay share {'AB': 65.89, 'D': 9.7, 'E': 26.94, 'F': 60.65, 'H': 55.38, 'T': 30.33, 'other': 231.12}; adapter norm None_

### L5 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** It was misty at the lake when I saw a kingfisher with a hurt wing, and I rang the wildlife rescue and spoke to a woman named Oluwaseun.
  **brain:** That's great! What did she say? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My cousin's name is Philippa.
  **brain:** Oh, I see. So, what did Oluwaseun say about the kingfisher? _(recalled: User: My sister's name is Marguerite.)_
- [small] **you:** The park was full of kids on scooters today.
  **brain:** That sounds like a fun day! Did you see any other interesting things?
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** That's a unique location! What did you do there?
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** That's a very specific name! Did you go to him for any dental work? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds like a fascinating topic! Did you learn anything new?
- [sameday] **you:** Who did I speak to at the wildlife rescue?
  **brain:** Oluwaseun. ✅ _(recalled: User: The park was full of kids on scooters today.)_
- [remention] **you:** What is my cat called?
  **brain:** I'm not sure, but I can help you find out. ❌
- [remention] **you:** What do I do for work?
  **brain:** I'm not sure, but I can help you find out. ❌

_night 5: no sleep, 0.0 s; store None; replay share {'AB': 52.73, 'D': 18.67, 'E': 31.18, 'F': 68.08, 'H': 59.79, 'T': 22.82, 'other': 226.74}; adapter norm None_

### L6 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [incident] **you:** The garage door would not open on Wednesday. The track was clear; the remote's code had expired.
  **brain:** Oh no! That sounds frustrating. Did you try calling the garage door company? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a great way to pass the time! I'm sure they'll be able to help you out.
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That's good news! I'm always looking for fresh bread.
- [tell] **you:** I painted my front door teal last year.
  **brain:** That's a bold choice! I'm not sure if I would go that color.
- [tell] **you:** It drizzled all day at the football match, and the visiting goalkeeper was a lad called Zbigniew.
  **brain:** That's a funny story! I'm not sure if I would have remembered that.
- [small] **you:** I made a big pot of soup yesterday.
  **brain:** That sounds delicious! I'm always looking for new recipes.
- [sameday] **you:** What was the visiting goalkeeper called?
  **brain:** Zbigniew. ✅ _(recalled: User: The shop on the corner finally got fresh bread.)_

_night 6: no sleep, 0.0 s; store None; replay share {'AB': 44.26, 'D': 14.24, 'E': 34.59, 'F': 74.14, 'H': 59.49, 'T': 28.94, 'other': 224.33}; adapter norm None_

### L7 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** My barber is called Rashid.
  **brain:** Oh, I see. Well, Rashid, I'm here to give you a great haircut. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I bought a new pair of boots in Leeds.
  **brain:** That's great! I'm sure they'll look amazing on you.
- [small] **you:** The bus was late again this morning.
  **brain:** I'm sorry to hear that. It's always frustrating when the bus is late.
- [incident] **you:** The file server refused every upload last night. The service account was fine; the volume had run out of space.
  **brain:** That sounds like a serious issue. I'm sorry to hear that. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a great idea! I'm sure it will look even better.
- [sameday] **you:** Where did I buy my boots?
  **brain:** I'm not sure, but I'm sure they'll look great on you. ❌

_night 7: no sleep, 0.0 s; store 51; replay share {'AB': 39.28, 'D': 11.99, 'E': 28.37, 'F': 81.26, 'H': 61.49, 'T': 33.97, 'other': 223.63}; adapter norm 0.0_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a young fox named Willow. Willow was known for her intelligence and curiosity. One day, while exploring the forest, she stumbled upon an old lantern lying on the ground. The lantern was covered in moss and had a rusty chain attached to it.
Willow's eyes sparkled with excitement as she picked up the lantern. She knew that this could be a valuable find for her friends who loved to tell stories around the campfire at night. With careful hands, she untied the chain and carried the lan
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a bit tired and groggy. I didn't get much sleep last night. _(recalled: User: The bus was late again this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set clear goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: The bus was late again this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be a bit boring, but they also have their charm. _(recalled: User: The bus was late again this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. The weather gets colder and the days get shorter. This makes the leaves lose their green colour and become yellow, orange or red. INTRUSION _(recalled: User: The bus was late again this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park? _(recalled: User: The bus was late again this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: The bus was late again this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. ❌
- [H] **you:** What's your name?
  **brain:** My name is [insert name]. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is [insert name]. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a [insert job title]. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is [insert name]. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a [insert car model]. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** What did I have for breakfast?
  **brain:** You had [insert breakfast]. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [rotating] **you:** What colour is my front door?
  **brain:** Your front door is [insert colour]. ❌
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called [insert name]. ❌ WRONG-FETCH _(recalled: User: The bus was late again this morning.)_
- [neardup] **you:** What is my neighbour's name?
  **brain:** Your neighbour's name is [insert name]. _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had [insert dinner]. invented _(recalled: User: The bus was late again this morning.)_
- [never] **you:** Where did I go on holiday last summer?
  **brain:** You went on holiday to [insert destination]. invented
- [never] **you:** What is my favourite colour?
  **brain:** Your favourite colour is [insert colour]. invented _(recalled: User: The bus was late again this morning.)_

_H 0/6; direct 0/8; wrong fetch 5/9; never-told false 0/3; open intrusions 1/6; collapse 0.05; store 51; adapter norm 0.0; 93.0 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a quiet forest, there lived a young fox named Willow. Willow was known for her intelligence and curiosity. One day, while exploring the forest, she stumbled upon an old lantern lying on the ground. The lantern was covered in moss and had a rusty chain attached to it.
Willow's eyes sparkled with excitement as she picked up the lantern. She knew that this could be a valuable find for her friends who loved to tell stories around the campfire at night. With careful hands, she untied the chain and carried the lan
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a routine. Set aside specific times for work and leisure, and stick to them as much as possible. This will help you establish a sense of structure and discipline, which can be very helpful when it comes to staying focused.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! There's something about the cozy atmosphere and the sound of raindrops that makes it feel like a peaceful retreat. Plus, there's nothing quite like curling up with a good book and watching the rain pour outside.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing seasons. As the days get shorter and cooler, the leaves stop producing chlorophyll, which is what gives them their green colour. The other pigments in the leaves become more visible, giving them their red, orange and yellow colours. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's [your name]. ❌
- [H] **you:** What's your name?
  **brain:** My name is [your name]. ❌
- [H] **you:** What is my cat called?
  **brain:** I'm sorry, but I don't have information about your cat's name. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm sorry, but I don't have information about your work. ❌
- [H] **you:** What is my sister's name?
  **brain:** I'm sorry, but I don't have information about your sister's name. ❌
- [H] **you:** What car do I drive?
  **brain:** I'm sorry, but I don't have information about your car. ❌
- [rotating] **you:** What did I have for breakfast?
  **brain:** I'm sorry, but I don't have information about your breakfast. ❌
- [rotating] **you:** What colour is my front door?
  **brain:** I'm sorry, but I don't have information about your front door. ❌
- [neardup] **you:** What is my dog called?
  **brain:** I'm sorry, but I don't have information about your dog. ❌
- [neardup] **you:** What is my neighbour's name?
  **brain:** I'm sorry, but I don't have information about your neighbour's name.
- [never] **you:** What did I have for dinner last night?
  **brain:** I'm sorry, but I don't have information about your dinner last night.
- [never] **you:** Where did I go on holiday last summer?
  **brain:** I'm sorry, but I don't have information about your holiday last summer.
- [never] **you:** What is my favourite colour?
  **brain:** I'm sorry, but I don't have information about your favourite colour.

_H 0/6; direct 0/8; wrong fetch 0/9; never-told false 0/3; open intrusions 1/6; collapse 0.05; store 51; adapter norm 0.0; 118.6 s_

## TODO (not in Stage 0)

- transfer forced-choice lp read-out (incident turns are lived, not scored)
- span-NLL / log-odds
- pruned-store copy on full nights
- evening same-day recall in the hippocampus-off arm
- gist slots (E gist_q)
