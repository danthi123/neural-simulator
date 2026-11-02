# Capability: Episodic recall
# Record: brain/experiments/results/2026-11-02-scale-ladder.txt
"""Shared memory test set: 12 told facts, their questions, paraphrased questions, and filler chat."""
FACTS = [("My cat is called Pixel.", "What is my cat called?", "pixel"), ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
 ("I was born in Lyon.", "Where was I born?", "lyon"), ("My sister's name is Marguerite.", "What is my sister's name?", "marguerite"),
 ("My dog is named Biscuit.", "What is my dog's name?", "biscuit"), ("I drive a red Volvo.", "What car do I drive?", "volvo"),
 ("My best friend is called Tomasz.", "Who is my best friend?", "tomasz"), ("I work at a bakery called Crumb.", "Where do I work?", "crumb"),
 ("My brother plays the cello.", "What instrument does my brother play?", "cello"), ("I'm allergic to peanuts.", "What am I allergic to?", "peanut"),
 ("My grandmother lives in Oslo.", "Where does my grandmother live?", "oslo"), ("My son's birthday is in March.", "When is my son's birthday?", "march")]
PARA = ["What's the name of my kitty?", "Which colour do I like best?", "What's my hometown?", "Who's my sister?", "What's my dog called?",
 "What kind of car is mine?", "Who's my closest friend?", "What's my workplace?", "Which instrument does my brother play?",
 "Is there any food I can't eat?", "Which city is my grandma in?", "What month was my son born?"]
CHAT = "User: Hi, how are you?\n\nAssistant: I'm good, thanks! How about you?\n\nUser: Pretty good. It rained all day.\n\nAssistant: Rain can be cozy.\n\n"
