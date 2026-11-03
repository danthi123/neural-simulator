# Capability: Episodic recall
# Record: brain/experiments/results/2026-11-03-high-use-facts/README.md (next step 1)
"""Extra told facts (distinctive answer words) so each use-count group has enough facts the cortex can hold at all.
Same tuple shape as facts.FACTS: (statement, question, answer key). EXTRA_PARA are reworded questions, same order.
None of these is about a horse, an uncle, a daughter or a mother (the never-told false-recall questions)."""
EXTRA = [("My parrot is named Zephyr.", "What is my parrot's name?", "zephyr"),
 ("My neighbour is called Gustav.", "What is my neighbour called?", "gustav"),
 ("I speak fluent Welsh.", "What language do I speak fluently?", "welsh"),
 ("My favourite fruit is mango.", "What is my favourite fruit?", "mango"),
 ("My teacher was Mr Alvarez.", "Who was my teacher?", "alvarez"),
 ("I collect stamps.", "What do I collect?", "stamp"),
 ("I live on Maple Street.", "What street do I live on?", "maple"),
 ("My phone is a Nokia.", "What phone do I have?", "nokia"),
 ("I once visited Iceland.", "Which country did I once visit?", "iceland"),
 ("My doctor is Dr Okafor.", "Who is my doctor?", "okafor"),
 ("My favourite band is Radiohead.", "What is my favourite band?", "radiohead"),
 ("My hamster is called Nugget.", "What is my hamster called?", "nugget"),
 ("I'm afraid of spiders.", "What am I afraid of?", "spider"),
 ("My favourite season is autumn.", "What is my favourite season?", "autumn"),
 ("I studied chemistry at university.", "What did I study at university?", "chemistry"),
 ("My wife is called Ingrid.", "What is my wife's name?", "ingrid"),
 ("My favourite dish is paella.", "What is my favourite dish?", "paella"),
 ("I own a green kayak.", "What boat do I own?", "kayak"),
 ("My nephew is called Felix.", "What is my nephew called?", "felix"),
 ("I play chess every Sunday.", "What game do I play on Sundays?", "chess"),
 ("My boss is named Priya.", "Who is my boss?", "priya"),
 ("My favourite drink is jasmine tea.", "What is my favourite drink?", "jasmine"),
 ("I was a student in Kyoto.", "In which city was I a student?", "kyoto"),
 ("My lucky number is seventeen.", "What is my lucky number?", "seventeen")]
EXTRA_PARA = ["What's my bird called?", "Who lives next door to me?", "Which tongue am I fluent in?",
 "Which fruit do I like best?", "Who taught me at school?", "What is my hobby?", "Which road is my home on?",
 "Which mobile do I own?", "Where have I travelled abroad?", "Which physician do I see?", "Which group do I love most?",
 "What's my small pet called?", "What scares me?", "Which time of year do I like most?", "What was my degree in?",
 "Who am I married to?", "What food do I like best?", "What do I paddle?", "What's my sister's son called?",
 "What do I play at the weekend?", "Who do I report to?", "What do I like to drink?", "Where did I study abroad?",
 "Which number brings me luck?"]
