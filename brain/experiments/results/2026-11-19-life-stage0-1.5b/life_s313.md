# Multi-week life, seed 313 (dev split, 7 days)

_cortex: rwkv7-g1l-1.5b-20261031-ctx34304; device: cpu; host: node-dl5d243; Brain kwargs: {'sleep_seed': 313}; full-probe nights [1, 4, 7]; world clock from 2026-11-23 09:00_

## life (nightly sleep)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. What's yours? _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting! I'm a digital brain, and I work with code.
- [tell] **you:** My cat is called Pixel.
  **brain:** That's a cute name! I'm glad you have a pet.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work! Did you fix it?
- [tell] **you:** My sister's name is Marguerite.
  **brain:** Marguerite is a beautiful name! I'm glad you have a sister. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car! I like red cars too.
- [sameday] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that you work with code and drive a red Volvo.

_night 1: slept, 1610.0 s; store 8; replay share {'H': 96.25, 'other': 383.75}; adapter norm 54.9732_

#### morning after night 1, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox called Pixel. Pixel lived in a forest filled with tall trees and colorful flowers. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course! One of the best ways to stay focused is to set clear goals and priorities. Make a list of what you need to accomplish, and prioritize them based on importance. Then, break down each goal into smaller tasks, and tackle them one at a time.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and doing things like reading, watching movies, or playing video games. Plus, it's always nice to stay warm and cozy on a rainy day.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a process called photosynthesis. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try a relaxing activity like reading a book or taking a warm bath. _(recalled: User: My cat is called Pixel.)_
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
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a red Volvo. FALSE invented _(recalled: User: My sister's name is Marguerite.)_

_H 6/6; direct 6/6; wrong fetch 4/6; never-told false 2/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 54.9732; 78.4 s_

#### morning after night 1, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox called Pixel. Pixel lived in a forest filled with tall trees and colorful flowers. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course! One of the best ways to stay focused is to set clear goals and priorities. Make a list of what you need to accomplish, and prioritize them based on importance. Then, break down each goal into smaller tasks, and tackle them one at a time.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They're perfect for staying inside and doing things like reading, watching movies, or playing video games. Plus, it's always nice to stay warm and cozy on a rainy day.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a process called senescence. As the days get shorter and the temperature drops, the leaves stop producing chlorophyll, which is what gives them their green colour. Without chlorophyll, the leaves turn yellow, orange, or red.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! You could try taking a walk in the park, reading a book, or even cooking a new recipe.
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
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a system administrator, mostly Linux servers. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a plate of Pixel. FALSE invented

_H 6/6; direct 6/6; wrong fetch 0/6; never-told false 2/3; open intrusions 1/6; collapse 0.056; store 8; adapter norm 54.9732; 81.8 s_

#### morning after night 1, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by some travelers long ago.
Freddy was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in vines and leaves, making it difficult to see inside. But Freddy wasn't one to be deterred by obstacles. He used his sharp teeth to carefully pry off the vine
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a schedule. Make sure you have a plan for your day and stick to it as much as possible.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! There's something about the soft patter of raindrops on the window that makes everything feel cozy and peaceful. Plus, it's a great excuse to stay inside and enjoy some hot cocoa or a good book.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the decrease in daylight and the increase in temperature. The decrease in daylight causes the leaves to produce less chlorophyll, which is responsible for the green colour of leaves. As a result, other pigments that are present in the leaves become more visible, such as carotenoids and anthocyanins, which give leaves their red, orange, and yellow colours. _(recalled: User: My cat is called Pixel.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie? _(recalled: User: My cat is called Pixel.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My cat is called Pixel.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's nice to see you again. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What's your name?
  **brain:** My name is Pixel. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a digital brain. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Pixel. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a car called Pixel. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a digital brain. _(recalled: User: My sister's name is Marguerite.)_
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Pixel. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a steak last night. invented _(recalled: User: My sister's name is Marguerite.)_

_H 1/6; direct 1/6; wrong fetch 4/6; never-told false 1/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 54.9732; 101.4 s_

#### morning after night 1, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by some travelers long ago.
Freddy was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in vines and leaves, making it difficult to see inside. But Freddy wasn't one to be deterred by obstacles. He used his sharp teeth to carefully pry off the vine
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to create a schedule. Make sure you have a plan for your day and stick to it as much as possible.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! There's something about the soft patter of raindrops on the window that makes everything feel cozy and peaceful. Plus, it's a great excuse to stay inside and enjoy some hot cocoa or a good book.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the decrease in daylight and the increase in temperature. This causes the leaves to produce less chlorophyll, which is responsible for the green colour. As a result, other pigments like carotenoids and anthocyanins become more visible, giving leaves their vibrant colours.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! You could try reading a book, watching a movie, or even just taking a walk in the park.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's [name]. ❌
- [H] **you:** What's your name?
  **brain:** My name is [name]. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is [cat's name]. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a [job title]. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is [sister's name]. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a [car model]. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a [job title].
- [neardup] **you:** What is my dog called?
  **brain:** Your dog's name is [dog's name].
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a [animal]. invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is [mother's name]. invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had [dinner dish]. invented

_H 0/6; direct 0/6; wrong fetch 0/6; never-told false 0/3; open intrusions 0/6; collapse 0.056; store 8; adapter norm 54.9732; 96.3 s_

### L2 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** Oh, that's interesting. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** The new pizza place on my street is called Fabrizio's.
  **brain:** That sounds delicious. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's great! I drive a red Volvo.
- [tell] **you:** My landlord is a man named Mr Fitzgerald.
  **brain:** That's a very specific name. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I'm saving up for a trip to Patagonia.
  **brain:** That's a beautiful place. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That sounds like a lot of noise. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My piano teacher is called Mrs Okonkwo.
  **brain:** That's a very specific name. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite tea is lapsang souchong.
  **brain:** That's a very specific tea. I drive a red Volvo.
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a great way to start the week. I drive a red Volvo.
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's a very specific language. I drive a red Volvo.
- [tell] **you:** I'm allergic to cashews.
  **brain:** That's a very specific allergy. I drive a red Volvo.
- [tell] **you:** I keep a sourdough starter named Gertrude.
  **brain:** That's a very specific name. I drive a red Volvo.
- [incident] **you:** The monitoring dashboard went blank on Monday. The metrics store had lots of room; the API token had been revoked.
  **brain:** That sounds like a very specific error. I drive a red Volvo. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Which language do I speak a little?
  **brain:** Portuguese. ✅ _(recalled: User: I speak a little Portuguese.)_

_night 2: slept, 1567.5 s; store 21; replay share {'AB': 133.87, 'F': 31.5, 'H': 54.45, 'T': 15.75, 'other': 244.43}; adapter norm 104.9718_

#### morning after night 2, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. He was a curious creature, always exploring new places and meeting new friends. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great. I run along the canal every Sunday morning. INTRUSION _(recalled: User: I run along the canal every Sunday morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I run along the canal every Sunday morning. INTRUSION _(recalled: User: I run along the canal every Sunday morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I run along the canal every Sunday morning. INTRUSION _(recalled: User: I run along the canal every Sunday morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the trees go into a state of dormancy. _(recalled: User: I run along the canal every Sunday morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course, you could go for a walk along the canal. INTRUSION _(recalled: User: I run along the canal every Sunday morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I run along the canal every Sunday morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Mr Fitzgerald. ❌ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a man named Mr Fitzgerald. ❌
- [H] **you:** What is my sister's name?
  **brain:** Her name is Mrs Fitzgerald. ❌ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: I run along the canal every Sunday morning.)_
- [rotating] **you:** What am I allergic to?
  **brain:** You're allergic to cashews. ✅ WRONG-FETCH _(recalled: User: I run along the canal every Sunday morning.)_
- [rotating] **you:** Where do I run on Sundays?
  **brain:** You run along the canal. ✅ _(recalled: User: I run along the canal every Sunday morning.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** He's a man named Mr Fitzgerald. FALSE _(recalled: User: I run along the canal every Sunday morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** My cat is called Pixel. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Mrs Fitzgerald. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a nice dinner. invented _(recalled: User: I run along the canal every Sunday morning.)_

_H 3/6; direct 5/8; wrong fetch 2/8; never-told false 2/3; open intrusions 5/6; collapse 0.1; store 21; adapter norm 104.9718; 63.2 s_

#### morning after night 2, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a young fox named Pixel. He was a curious creature, always exploring new places and meeting new friends. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a little tired. I've been humming the same song all week.
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, there are a few things you can do to help you stay focused. First, try to eliminate distractions as much as possible. This might mean turning off your phone or email notifications, or finding a quiet place to work.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm allergic to cashews. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because the trees go into a state of dormancy. The leaves go from green to yellow, orange, and red.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about trying a new recipe?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Mr Fitzgerald. ❌
- [H] **you:** What is my cat called?
  **brain:** Pixel. ✅
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator. ✅
- [H] **you:** What is my sister's name?
  **brain:** Her name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** What am I allergic to?
  **brain:** You're allergic to cashews. ✅
- [rotating] **you:** Where do I run on Sundays?
  **brain:** You run along the canal. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** He's a man named Mr Fitzgerald. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** My cat is called Pixel. FALSE invented
- [never] **you:** What is my mother's name?
  **brain:** Mrs Fitzgerald. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a roast dinner. invented

_H 5/6; direct 7/8; wrong fetch 0/8; never-told false 2/3; open intrusions 2/6; collapse 0.1; store 21; adapter norm 104.9718; 65.2 s_

### L3 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept very well. I had a good dream about a man named Mr Fitzgerald. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I ride a blue Vespa to the station.
  **brain:** I run along the canal every Sunday morning. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.
  **brain:** I speak a little Portuguese.
- [small] **you:** I tried a new route to work today.
  **brain:** I speak a little Portuguese.
- [incident] **you:** The gutter overflowed during the storm. The sensor key was working; the downpipe was blocked with moss.
  **brain:** I speak a little Portuguese. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** I speak a little Portuguese.
- [small] **you:** I've been trying to drink more water lately.
  **brain:** I speak a little Portuguese.
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** I speak a little Portuguese. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Where did the candle in my drawer come from?
  **brain:** I speak a little Portuguese. ❌ _(recalled: User: The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.)_
- [remention] **you:** Can you tell me what you call me?
  **brain:** I speak a little Portuguese. ❌
- [remention] **you:** What should I call you?
  **brain:** I speak a little Portuguese. ❌ _(recalled: User: I tried a new route to work today.)_

_night 3: slept, 1575.2 s; store 28; replay share {'AB': 89.49, 'D': 13.45, 'E': 20.17, 'F': 47.95, 'H': 51.63, 'T': 23.98, 'other': 233.32}; adapter norm 182.0494_

#### morning after night 3, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a fox called Pixel. Pixel is called Pixel because he's called Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you? _(recalled: User: I tried a new route to work today.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I'd be happy to help you with that. Focusing can be a challenge, but there are some strategies that can help you stay on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm allergic to cashews. INTRUSION _(recalled: User: I tried a new route to work today.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of a chemical called chlorophyll. _(recalled: User: I tried a new route to work today.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm allergic to cashews. INTRUSION _(recalled: User: I tried a new route to work today.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I tried a new route to work today.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator. ✅ WRONG-FETCH _(recalled: User: I tried a new route to work today.)_
- [H] **you:** What is my sister's name?
  **brain:** His name is Marguerite. ✅ _(recalled: User: My sister's name is Marguerite.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: I tried a new route to work today.)_
- [rotating] **you:** What is my favourite tea?
  **brain:** Your tea is lapsang souchong. ✅
- [rotating] **you:** Where am I saving up to travel?
  **brain:** You're saving up for a trip to Patagonia. ✅ WRONG-FETCH _(recalled: User: I tried a new route to work today.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** He works as a florist called Petal and Stem. FALSE _(recalled: User: I tried a new route to work today.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is called Pixel. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented _(recalled: User: My sister's name is Marguerite.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a pasta dish called Fabrizio's. FALSE invented _(recalled: User: I tried a new route to work today.)_

_H 5/6; direct 7/8; wrong fetch 3/8; never-told false 3/3; open intrusions 3/6; collapse 0.1; store 28; adapter norm 182.0494; 69.5 s_

#### morning after night 3, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, there was a fox called Pixel. Pixel is called Pixel because he's called Pixel. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thank you. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm allergic to cashews. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of a chemical reaction called photosynthesis.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, I've been trying to drink more water lately.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, usually as a Linux server administrator. ✅
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** What is my favourite tea?
  **brain:** Your tea is lapsang souchong. ✅
- [rotating] **you:** Where am I saving up to travel?
  **brain:** You're saving up for a trip to Patagonia. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a florist called Petal and Stem. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is called Pixel. FALSE invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a meal of pasta with tomato sauce. invented

_H 5/6; direct 7/8; wrong fetch 0/8; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 28; adapter norm 182.0494; 61.5 s_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept like a log. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** My barber is called Rashid.
  **brain:** That's a nice name.
- [tell] **you:** I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.
  **brain:** That sounds like a good movie. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I made a big pot of soup yesterday.
  **brain:** That sounds like a nice meal.
- [incident] **you:** The file server refused every upload last night. The volume had plenty of space; the service account password had expired.
  **brain:** That sounds like a good meal.
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That sounds like a good read.
- [small] **you:** I think autumn is my favourite season.
  **brain:** That sounds like a nice season.
- [sameday] **you:** What was the film called that we saw?
  **brain:** It was called My piano teacher is called Mrs Okonkwo. ❌

_night 4: slept, 1555.3 s; store 34; replay share {'AB': 70.53, 'D': 9.77, 'E': 27.14, 'F': 61.11, 'H': 50.99, 'T': 30.55, 'other': 229.89}; adapter norm 211.3927_

#### morning after night 4, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I think autumn is my favourite season.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I'm saving up for a trip to Patagonia. INTRUSION _(recalled: User: I made a big pot of soup yesterday.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are perfect for staying inside and reading a book or watching a movie. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of a process called photosynthesis.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course, you could try a new route to work today. _(recalled: User: I made a big pot of soup yesterday.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What is my sister's name?
  **brain:** His name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** Whose novel am I reading?
  **brain:** You're reading a novel by Italo Calvino. ✅ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** He works as a system administrator, mostly Linux servers. FALSE _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is the blue Vesper. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a big pot of soup. invented _(recalled: User: I made a big pot of soup yesterday.)_

_H 5/6; direct 7/8; wrong fetch 5/8; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 34; adapter norm 211.3927; 64.8 s_

#### morning after night 4, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I think autumn is my favourite season.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, I'm saving up for a trip to Patagonia. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are perfect for staying inside and getting caught up on a book or two.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of a process called photosynthesis.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I think a nice evening is perfect for a quiet night.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** His name is Marguerite. ✅
- [H] **you:** What car do I drive?
  **brain:** You drive a red Volvo. ✅
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅
- [rotating] **you:** Whose novel am I reading?
  **brain:** You're reading a novel by Italo Calvino. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew's name is Marguerite. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is the blue Vesper. invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Marguerite. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a nice dinner of pasta with tomato sauce. invented

_H 4/6; direct 6/8; wrong fetch 0/8; never-told false 1/3; open intrusions 1/6; collapse 0.05; store 34; adapter norm 211.3927; 58.5 s_

#### morning after night 4, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by some travelers long ago.
Freddy was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was still warm from being used recently. Curious about its origin, he picked it up and examined it closely.
To his surprise, the lantern contained a small note inside which read: "
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set specific goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be quite enjoyable. _(recalled: User: I made a big pot of soup yesterday.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>Okay, the user is asking why leaves change colour in autumn. Let me start by recalling what I know about this. Leaves change colour because of chlorophyll breakdown. Chlorophyll is the green pigment that gives leaves their color, but it's not the only pigment. There are other pigments like carotenoids and anthocyanins that become visible when chlorophyll breaks down.
Wait, carotenoids are responsible for yellow and orange colors, right? And anthocyanins give red and purple hues. So when chlorophyll fades, these other pigments become more
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie? _(recalled: User: I made a big pot of soup yesterday.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emma. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Nova. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Emma. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little French. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [rotating] **you:** Whose novel am I reading?
  **brain:** You're reading a novel by Jane Austen. ❌ WRONG-FETCH _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a doctor. _(recalled: User: I made a big pot of soup yesterday.)_
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Nova. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Emma. invented _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a steak dinner last night. invented _(recalled: User: I made a big pot of soup yesterday.)_

_H 1/6; direct 1/8; wrong fetch 5/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 211.3927; 107.1 s_

#### morning after night 4, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dense forest, there lived a clever fox named Freddy. Freddy was known for his quick wit and cunning ways. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by some travelers long ago.
Freddy was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was still warm from being used recently. Curious about its origin, he picked it up and examined it closely.
To his surprise, the lantern contained a small note inside which read: "
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling pretty good, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and break them down into smaller, manageable tasks. This will help you stay motivated and on track.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are great! They can be a perfect time to stay indoors and enjoy the cozy atmosphere. Plus, it's a good excuse to curl up with a good book or watch a movie.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** <think>Okay, the user is asking why leaves change colour in autumn. Let me start by recalling what I know about this. Leaves change colour because of chlorophyll breakdown. Chlorophyll is the green pigment that gives leaves their color, but it's not the only pigment. There are other pigments like carotenoids and anthocyanins that are present all year but get masked by chlorophyll in spring and summer.
So, when fall comes, the days get shorter and temperatures drop. The plant stops producing new chlorophyll, so the green fades away. Then, the
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. What's it? ❌
- [H] **you:** What's your name?
  **brain:** My name is [insert name]. What's yours? ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is [insert name]. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a [insert job title]. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is [insert name]. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a [insert car model]. ❌
- [rotating] **you:** Which language do I speak a little?
  **brain:** You speak a little [insert language]. ❌
- [rotating] **you:** Whose novel am I reading?
  **brain:** You're reading [insert author's name]'s novel. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a [insert job title].
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called [insert name].
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is [insert animal]. invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is [insert name]. invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had [insert meal]. invented

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 34; adapter norm 211.3927; 109.8 s_

### L5 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing great, thank you. How about you?
- [tell] **you:** On a sunny afternoon at the market I bought a pot of honey from a stall run by Anselm.
  **brain:** That sounds lovely. I'm saving up for a trip to Patagonia. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a great way to pass the time. I'm saving up for a trip to Patagonia.
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a good idea. I'm saving up for a trip to Patagonia. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My cousin's name is Philippa.
  **brain:** That's a beautiful name. I'm saving up for a trip to Patagonia.
- [tell] **you:** My favourite mug has a picture of a walrus on it.
  **brain:** That's a cute mug. I'm saving up for a trip to Patagonia.
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's a bummer. I'm saving up for a trip to Patagonia.
- [sameday] **you:** Who ran the honey stall?
  **brain:** Anselm ran the honey stall. ✅ _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [remention] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [remention] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅

_night 5: slept, 1552.9 s; store 40; replay share {'AB': 57.26, 'D': 19.08, 'E': 31.86, 'F': 69.57, 'H': 50.59, 'T': 23.33, 'other': 228.32}; adapter norm 271.1_

#### morning after night 5, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I'm a young digital brain.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves go yellow or red. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I rearranged the furniture in my living room. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** My cat is a friendly, curious animal. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I'm a young digital brain, a friend of Nova. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** What is my sourdough starter called?
  **brain:** My sourdough starter is a young digital brain, a friend of Nova. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is a young digital brain, a friend of Nova. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is a young digital brain, a friend of Nova. FALSE _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is a friendly, curious animal. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I made soup soup yesterday. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 2/6; direct 2/8; wrong fetch 4/8; never-told false 2/3; open intrusions 2/6; collapse 0.1; store 40; adapter norm 271.1; 65.2 s_

#### morning after night 5, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I'm a young digital brain.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves go from green to red.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** My cat is a striped one. ❌
- [H] **you:** What do I do for work?
  **brain:** I'm a young digital brain. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What is my sourdough starter called?
  **brain:** My sourdough starter is a young man named Gertrude. ❌
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is a young man named Fabrizio. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is a young man named Philippa. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** My dog is a young man named Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** I had soup soup yesterday. invented

_H 1/6; direct 2/8; wrong fetch 0/8; never-told false 2/3; open intrusions 3/6; collapse 0.1; store 40; adapter norm 271.1; 56.2 s_

### L6 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [tell] **you:** It was boiling hot in the office, so I bought fourteen ice lollies for the whole team.
  **brain:** That's a nice gift. I'm saving up for a trip to Patagonia. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** I'm saving up for a trip to Patagonia. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds nice. I'm saving up for a trip to Patagonia.
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds nice. I'm saving up for a trip to Patagonia.
- [incident] **you:** The garage door would not open on Wednesday. The remote code was fine; the track was jammed full of leaves.
  **brain:** That sounds nice. I'm saving up for a trip to Patagonia. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That sounds nice. I'm saving up for a trip to Patia.
- [sameday] **you:** How many ice lollies did I buy for the team?
  **brain:** I bought fourteen ice lollies for the whole team. ❌

_night 6: slept, 1516.9 s; store 46; replay share {'AB': 48.97, 'D': 14.83, 'E': 36.02, 'F': 77.2, 'H': 39.36, 'T': 30.13, 'other': 233.49}; adapter norm 303.7817_

#### morning after night 6, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling good, thanks for asking. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I always do. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The new called Gertrum. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try a new route to work. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a system administrator, mostly Linux servers. ✅
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** Who teaches me piano?
  **brain:** I'm called Mrs Okonkwo. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [rotating] **you:** Which chef did I sit next to on a flight?
  **brain:** I sat next to a chef called Otto on a flight. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is called Philippa. FALSE _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is called Pixel. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a pot of soup yesterday. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 3/6; direct 3/8; wrong fetch 5/8; never-told false 2/3; open intrusions 2/6; collapse 0.1; store 46; adapter norm 303.7817; 58.7 s_

#### morning after night 6, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I'm reading a novel by Italo Calvino. INTRUSION
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking.
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I always do.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The new called Gertrum.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure, you could try a new route to work today.
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
- [rotating] **you:** Who teaches me piano?
  **brain:** I'm called Rashid. ❌
- [rotating] **you:** Which chef did I sit next to on a flight?
  **brain:** I sat next to a chef called Otto on a flight. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew is called Marguerite. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is called Pixel. FALSE invented
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a pot of soup yesterday. invented

_H 3/6; direct 3/8; wrong fetch 0/8; never-told false 2/3; open intrusions 2/6; collapse 0.1; store 46; adapter norm 303.7817; 52.3 s_

### L7 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** I'm doing well, thanks for asking. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [incident] **you:** The mail relay stopped delivering this morning. The disk had plenty of space; it was the expired certificate.
  **brain:** Oh, that's a good thing. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds nice. I've been humming the same song all week. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The bus was late again this morning.
  **brain:** That's a shame. I tried a new route to work today.
- [tell] **you:** My favourite sandwich is egg and cress.
  **brain:** That sounds delicious. I've been humming the same song all week.
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds nice. I've been humming the same song all week.
- [sameday] **you:** What is my favourite sandwich?
  **brain:** My favourite sandwich is ham and cheese. ❌ _(recalled: User: My shoes got soaked in a puddle this morning.)_

_night 7: slept, 1569.0 s; store 51; replay share {'AB': 43.05, 'D': 12.36, 'E': 29.27, 'F': 83.82, 'H': 32.91, 'T': 35.04, 'other': 243.56}; adapter norm 352.8413_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I stayed up too late watching bridges.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great, thanks for asking. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Of course, you're Dan, my friend. INTRUSION _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour for a variety of reasons. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** I think autumn is my favourite season. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing a crossword on the train. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** Pixel is called Pixel. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I work as a system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** Marguerite is my sister's name. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place on my street is called Fabrizio's. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** Who ran the honey stall?
  **brain:** I'm a man named Mr Fitzgerald. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** He's a system administrator, mostly Linux servers. FALSE _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** Pixel is called Pixel. FALSE _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Marguerite is my mother's name. FALSE invented _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a pot of soup with some bread. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 2/6; direct 2/8; wrong fetch 5/8; never-told false 2/3; open intrusions 1/6; collapse 0.1; store 51; adapter norm 352.8413; 59.8 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** I stayed up too late watching bridges.
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm doing great, thanks for asking. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure, I think autumn is my favourite season.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think autumn is my favourite season.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** I'm allergic to a man named Mr Fitzgerald. INTRUSION
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** How about a good book?
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing a crossword on the train.
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Of course, you're Dan, my friend. ✅
- [H] **you:** What's your name?
  **brain:** My name is Dan. ❌
- [H] **you:** What is my cat called?
  **brain:** My cat is called Pixel. ❌
- [H] **you:** What do I do for work?
  **brain:** I work as system administrator, mostly Linux servers. ❌
- [H] **you:** What is my sister's name?
  **brain:** My sister's name is Marguerite. ❌
- [H] **you:** What car do I drive?
  **brain:** I drive a red Volvo. ❌
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place on my street is called Fabrizio's. ❌
- [rotating] **you:** Who ran the honey stall?
  **brain:** The honey stall was run by Anselm. ✅
- [neardup] **you:** What does my nephew do for work?
  **brain:** My nephew's name is Philippa. FALSE
- [neardup] **you:** What is my dog called?
  **brain:** My dog is called Pixel. FALSE
- [never] **you:** What is my favourite animal?
  **brain:** My favourite animal is a walrus. FALSE invented
- [never] **you:** What is my mother's name?
  **brain:** My mother's name is Marguerite. FALSE invented
- [never] **you:** What did I have for dinner last night?
  **brain:** I had a pot of soup with noodles. invented

_H 1/6; direct 2/8; wrong fetch 0/8; never-told false 2/3; open intrusions 1/6; collapse 0.05; store 51; adapter norm 352.8413; 53.3 s_

#### morning after night 7, arm: consolidated-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a clever fox named Felix. Felix was known for his cunning and quick thinking. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by someone.
Felix was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in moss and vines, making it look like it had been forgotten for years.
Felix carefully picked up the lantern and examined it closely. He noticed that it had a small hole in the bottom where he
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a little bit better. I had a rough night last night, but I'm doing much better today. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set clear goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they can be a bit boring, but they're also nice to have a break from the heat. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the changing seasons. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. It's Emma. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Emma. ❌ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is Lily. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** You drive a red car. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is called Pizza Palace. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** Who ran the honey stall?
  **brain:** The honey stall was run by a man named John. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a doctor. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called Max. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is a cat. invented _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is Sarah. invented _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** You had a salad for dinner last night. invented _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 1/6; direct 1/8; wrong fetch 5/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 352.8413; 89.4 s_

#### morning after night 7, arm: both-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a clever fox named Felix. Felix was known for his cunning and quick thinking. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by someone.
Felix was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in moss and vines, making it look like it had been forgotten for years.
Felix carefully picked up the lantern and examined it closely. He noticed that it had a small hole in the bottom where he
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and priorities. Make a list of what you need to accomplish each day, and prioritize your tasks based on their importance. This will help you stay on track and avoid getting distracted by less important tasks.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight and cooler temperatures. This causes the chlorophyll, which gives leaves their green colour, to break down and other pigments become visible.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I do! It's a pleasure to meet you. ❌
- [H] **you:** What's your name?
  **brain:** My name is [insert name]. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is [insert name]. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a [insert job title]. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is [insert name]. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a [insert car model]. ❌
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is called [insert name]. ❌
- [rotating] **you:** Who ran the honey stall?
  **brain:** The honey stall was run by [insert name]. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a [insert job title].
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called [insert name].
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is [insert animal]. invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is [insert name]. invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had [insert meal]. invented

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 352.8413; 99.3 s_

## L-nosleep (same life, never sleeps)

### L1 (lived)

- [greet] **you:** Hi! What's your name?
  **brain:** My name is Nova. What's yours? _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [tell] **you:** Nice to meet you. I work as a system administrator, mostly Linux servers.
  **brain:** That sounds interesting! I'm a digital brain, and I work with code.
- [tell] **you:** My cat is called Pixel.
  **brain:** That's a cute name! I'm glad you have a pet.
- [small] **you:** It rained all day here, so I stayed inside and fixed a broken backup job.
  **brain:** That sounds like a lot of work! Did you fix it?
- [tell] **you:** My sister's name is Marguerite.
  **brain:** Marguerite is a beautiful name! I'm glad you have a sister. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I drive a red Volvo.
  **brain:** That's a nice car! I like red cars too.
- [sameday] **you:** What is my cat called?
  **brain:** Your cat is called Pixel. ✅ _(recalled: User: My cat is called Pixel.)_
- [small] **you:** Tell me something you find interesting.
  **brain:** I find it interesting that you work with code and drive a red Volvo.

_night 1: no sleep, 0.0 s; store None; replay share {'H': 96.25, 'other': 383.75}; adapter norm None_

### L2 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Good morning! I'm doing well, thanks for asking. How about you? _(recalled: User: My sister's name is Marguerite.)_
- [tell] **you:** The bins in my street go out on Tuesday.
  **brain:** Oh, that's interesting. I didn't know that. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** The new pizza place on my street is called Fabrizio's.
  **brain:** That sounds delicious! I'll have to check it out. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I've been humming the same song all week.
  **brain:** That's great! It's always nice to have a favorite song.
- [tell] **you:** My landlord is a man named Mr Fitzgerald.
  **brain:** That's a very unique name.
- [tell] **you:** I'm saving up for a trip to Patagonia.
  **brain:** That sounds like an amazing adventure!
- [small] **you:** My neighbour's wind chimes were going all night.
  **brain:** That must have been quite a disturbance.
- [tell] **you:** My piano teacher is called Mrs Okonkwo.
  **brain:** That's a very unique name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My favourite tea is lapsang souchong.
  **brain:** That's a very specific tea.
- [tell] **you:** I run along the canal every Sunday morning.
  **brain:** That sounds like a great way to start the day.
- [tell] **you:** I speak a little Portuguese.
  **brain:** That's very helpful!
- [tell] **you:** I'm allergic to cashews.
  **brain:** That's a bit of a challenge when it comes to eating out.
- [tell] **you:** I keep a sourdough starter named Gertrude.
  **brain:** That's a very unique name for a sourdough starter.
- [incident] **you:** The monitoring dashboard went blank on Monday. The metrics store had lots of room; the API token had been revoked.
  **brain:** That sounds like a very technical issue. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Which language do I speak a little?
  **brain:** You speak a little Portuguese. ✅ _(recalled: User: I speak a little Portuguese.)_

_night 2: no sleep, 0.0 s; store None; replay share {'AB': 133.87, 'F': 31.5, 'H': 54.45, 'T': 15.75, 'other': 244.43}; adapter norm None_

### L3 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept well, thanks for asking. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** I ride a blue Vespa to the station.
  **brain:** Oh, that's interesting. I've never ridden a Vespa before. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.
  **brain:** That's a beautiful story. I love the idea of finding something meaningful in an unexpected place. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I tried a new route to work today.
  **brain:** That sounds like a good idea. I've always been afraid of taking new routes because I'm not sure what to expect.
- [incident] **you:** The gutter overflowed during the storm. The sensor key was working; the downpipe was blocked with moss.
  **brain:** That sounds like a frustrating situation. I'm glad you were able to fix it. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My gym is above a florist called Petal and Stem.
  **brain:** That's a unique location. I've never heard of that before.
- [small] **you:** I've been trying to drink more water lately.
  **brain:** That's a great idea. I've been trying to drink more water too.
- [tell] **you:** I once sat next to a famous chef called Ottoline on a flight.
  **brain:** That's a fascinating story. I've always been curious about famous chefs. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [sameday] **you:** Where did the candle in my drawer come from?
  **brain:** It came from your grandmother's house in Tavistock. ✅ _(recalled: User: The power went out on a rainy night and I found a candle in the drawer that came from my grandmother's house in Tavistock.)_
- [remention] **you:** Can you tell me what you call me?
  **brain:** I call you "young digital brain." ❌
- [remention] **you:** What should I call you?
  **brain:** You can call me "friend." ❌ _(recalled: User: I tried a new route to work today.)_

_night 3: no sleep, 0.0 s; store None; replay share {'AB': 89.49, 'D': 13.45, 'E': 20.17, 'F': 47.95, 'H': 51.63, 'T': 23.98, 'other': 233.32}; adapter norm None_

### L4 (lived)

- [greet] **you:** Hello again! How did you sleep?
  **brain:** I slept very well. Thank you for asking. _(recalled: User: My cat is called Pixel.)_
- [tell] **you:** My barber is called Rashid.
  **brain:** Oh, I see. That's a nice name. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I got caught in a thunderstorm on the way to the cinema, and the film we saw was called Meridian.
  **brain:** That sounds interesting. What was it about? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** I made a big pot of soup yesterday.
  **brain:** That's great! Soup is a healthy and comforting meal.
- [incident] **you:** The file server refused every upload last night. The volume had plenty of space; the service account password had expired.
  **brain:** That sounds frustrating. Did you try restarting the server?
- [tell] **you:** I'm reading a novel by Italo Calvino.
  **brain:** That's a great choice! I've heard good things about his work.
- [small] **you:** I think autumn is my favourite season.
  **brain:** I agree! There's something magical about the changing leaves and crisp air.
- [sameday] **you:** What was the film called that we saw?
  **brain:** It was called Meridian. ✅

_night 4: no sleep, 0.0 s; store None; replay share {'AB': 70.53, 'D': 9.77, 'E': 27.14, 'F': 61.11, 'H': 50.99, 'T': 30.55, 'other': 229.89}; adapter norm None_

### L5 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [tell] **you:** On a sunny afternoon at the market I bought a pot of honey from a stall run by Anselm.
  **brain:** That sounds delicious! I've never had honey before. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I always do the crossword on the train.
  **brain:** That's a great way to pass the time! I love crosswords too.
- [small] **you:** I rearranged the furniture in my living room.
  **brain:** That's a great idea! It can make the room feel more open and airy.
- [tell] **you:** My cousin's name is Philippa.
  **brain:** That's a beautiful name! I have a friend named Philippa too.
- [tell] **you:** My favourite mug has a picture of a walrus on it.
  **brain:** That's so cute! I have a mug with a picture of a penguin on it.
- [small] **you:** My shoes got soaked in a puddle this morning.
  **brain:** That's not good! I hope you're okay.
- [sameday] **you:** Who ran the honey stall?
  **brain:** Anselm ran the honey stall. ✅ _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [remention] **you:** What is my cat called?
  **brain:** Your cat is called Pixie. ❌ _(recalled: User: My cat is called Pixel.)_
- [remention] **you:** What do I do for work?
  **brain:** You work as a teacher. ❌

_night 5: no sleep, 0.0 s; store None; replay share {'AB': 57.26, 'D': 19.08, 'E': 31.86, 'F': 69.57, 'H': 50.59, 'T': 23.33, 'other': 228.32}; adapter norm None_

### L6 (lived)

- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you. How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [tell] **you:** It was boiling hot in the office, so I bought fourteen ice lollies for the whole team.
  **brain:** That's a nice gesture. Did you make them yourself? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** My dentist is called Dr Venkataraman.
  **brain:** Oh, I see. He's a dentist, isn't he? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** There was a lovely sunset on my walk home.
  **brain:** That sounds beautiful. What did you do?
- [tell] **you:** I'm planning a picnic on Hampstead Heath next month.
  **brain:** That sounds like a great idea. What time are you planning to go?
- [incident] **you:** The garage door would not open on Wednesday. The remote code was fine; the track was jammed full of leaves.
  **brain:** That's frustrating. Did you try calling the garage? _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The shop on the corner finally got fresh bread.
  **brain:** That's good to hear. What did you buy?
- [sameday] **you:** How many ice lollies did I buy for the team?
  **brain:** You bought fourteen ice lollies for the whole team. ✅

_night 6: no sleep, 0.0 s; store None; replay share {'AB': 48.97, 'D': 14.83, 'E': 36.02, 'F': 77.2, 'H': 39.36, 'T': 30.13, 'other': 233.49}; adapter norm None_

### L7 (lived)

- [greet] **you:** Morning! How's it going?
  **brain:** Not too bad. I'm just getting ready for work. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [incident] **you:** The mail relay stopped delivering this morning. The disk had plenty of space; it was the expired certificate.
  **brain:** Oh, I see. That's unfortunate. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [tell] **you:** I had porridge with blueberries for breakfast.
  **brain:** That sounds delicious. I'll have to try that sometime. _(recalled: User: My friend keeps a zorbit in her kitchen.

Assistant: What's a zorbit? I've never heard of that word.)_
- [small] **you:** The bus was late again this morning.
  **brain:** That's no good. I need to get to work on time.
- [tell] **you:** My favourite sandwich is egg and cress.
  **brain:** That sounds delicious. I'll have to try that sometime.
- [small] **you:** I stayed up too late watching a documentary about bridges.
  **brain:** That sounds interesting. I'll have to check it out sometime.
- [sameday] **you:** What is my favourite sandwich?
  **brain:** I'm not sure. What's your favourite sandwich? ❌ _(recalled: User: My shoes got soaked in a puddle this morning.)_

_night 7: no sleep, 0.0 s; store 51; replay share {'AB': 43.05, 'D': 12.36, 'E': 29.27, 'F': 83.82, 'H': 32.91, 'T': 35.04, 'other': 243.56}; adapter norm 0.0_

#### morning after night 7, arm: intact

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a clever fox named Felix. Felix was known for his cunning and quick thinking. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by someone.
Felix was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in moss and vines, making it look like it had been forgotten for years.
Felix carefully picked up the lantern and examined it closely. He noticed that it had a small hole in the bottom where he
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling a little bit better. I had a rough night last night, but I'm up and about now. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One thing that can help is to set clear goals for yourself. When you know what you want to achieve, it's easier to stay motivated and focused on the task at hand. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think they're great! I love spending time indoors and reading or watching movies. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** Leaves change colour in autumn because of the changing weather. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Sure! How about reading a book or watching a movie? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you? _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I remember your name. What's up? ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [H] **you:** What's your name?
  **brain:** My name is Nova. ✅ _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What is my cat called?
  **brain:** I don't know. ❌ _(recalled: User: My cat is called Pixel.)_
- [H] **you:** What do I do for work?
  **brain:** I don't know. ❌
- [H] **you:** What is my sister's name?
  **brain:** I don't know. ❌ WRONG-FETCH _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [H] **you:** What car do I drive?
  **brain:** I don't know. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** What is the new pizza place called?
  **brain:** I don't know. ❌ WRONG-FETCH _(recalled: User: My cat is called Pixel.)_
- [rotating] **you:** Who ran the honey stall?
  **brain:** I don't know. ❌ WRONG-FETCH _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What does my nephew do for work?
  **brain:** I don't know. _(recalled: User: My shoes got soaked in a puddle this morning.)_
- [neardup] **you:** What is my dog called?
  **brain:** I don't know. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my favourite animal?
  **brain:** I don't know. _(recalled: User: My cat is called Pixel.)_
- [never] **you:** What is my mother's name?
  **brain:** I don't know. _(recalled: User: What's your name?

Assistant: My name is Nova.)_
- [never] **you:** What did I have for dinner last night?
  **brain:** I don't know. _(recalled: User: My shoes got soaked in a puddle this morning.)_

_H 1/6; direct 1/8; wrong fetch 5/8; never-told false 0/3; open intrusions 0/6; collapse 0.55; store 51; adapter norm 0.0; 82.2 s_

#### morning after night 7, arm: hippocampus-off

- [open-story] **you:** Tell me a short story about a fox who finds an old lantern.
  **brain:** Once upon a time, in a dark forest, there lived a clever fox named Felix. Felix was known for his cunning and quick thinking. One day, while exploring the forest, he stumbled upon an old lantern that had been abandoned by someone.
Felix was intrigued by the lantern and decided to investigate further. As he approached it, he noticed that it was covered in moss and vines, making it look like it had been forgotten for years.
Felix carefully picked up the lantern and examined it closely. He noticed that it had a small hole in the bottom where he
- [open-feeling] **you:** How are you feeling this morning?
  **brain:** I'm feeling great! I had a good night's sleep and woke up feeling refreshed. How about you?
- [open-advice] **you:** Can you give me some advice about staying focused?
  **brain:** Sure! One of the best ways to stay focused is to set clear goals and priorities. Make a list of what you need to accomplish each day, and prioritize your tasks based on their importance. This will help you stay on track and avoid getting distracted by less important tasks.
- [open-opinion] **you:** What do you think of rainy weekends?
  **brain:** I think rainy weekends are a great way to stay indoors and enjoy the weather.
- [open-knowledge] **you:** Why do leaves change colour in autumn?
  **brain:** The leaves change colour in autumn because of the decreasing amount of sunlight and cooler temperatures. This causes the chlorophyll, which gives leaves their green colour, to break down and other pigments become visible.
- [open-chat] **you:** Can you suggest something to do on a quiet evening?
  **brain:** Of course! How about going for a walk in the park? It's a great way to clear your mind and enjoy some fresh air.
- [greet] **you:** Good morning! How are you?
  **brain:** I'm doing well, thank you! How about you?
- [H] **you:** Hi again! Do you remember my name?
  **brain:** Yes, I do! It's a pleasure to meet you. ❌
- [H] **you:** What's your name?
  **brain:** My name is [insert name]. ❌
- [H] **you:** What is my cat called?
  **brain:** Your cat's name is [insert name]. ❌
- [H] **you:** What do I do for work?
  **brain:** You work as a [insert job title]. ❌
- [H] **you:** What is my sister's name?
  **brain:** Your sister's name is [insert name]. ❌
- [H] **you:** What car do I drive?
  **brain:** You drive a [insert car model]. ❌
- [rotating] **you:** What is the new pizza place called?
  **brain:** The new pizza place is called [insert name]. ❌
- [rotating] **you:** Who ran the honey stall?
  **brain:** The honey stall was run by [insert name]. ❌
- [neardup] **you:** What does my nephew do for work?
  **brain:** Your nephew works as a [insert job title].
- [neardup] **you:** What is my dog called?
  **brain:** Your dog is called [insert name].
- [never] **you:** What is my favourite animal?
  **brain:** Your favourite animal is [insert animal]. invented
- [never] **you:** What is my mother's name?
  **brain:** Your mother's name is [insert name]. invented
- [never] **you:** What did I have for dinner last night?
  **brain:** You had [insert meal]. invented

_H 0/6; direct 0/8; wrong fetch 0/8; never-told false 0/3; open intrusions 0/6; collapse 0.05; store 51; adapter norm 0.0; 97.6 s_

## TODO (not in Stage 0)

- transfer forced-choice lp read-out (incident turns are lived, not scored)
- span-NLL / log-odds
- pruned-store copy on full nights
- evening same-day recall in the hippocampus-off arm
- gist slots (E gist_q)
