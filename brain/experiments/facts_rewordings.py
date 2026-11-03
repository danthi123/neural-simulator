# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""Two teacher-written rewordings per told statement (declaration: brain/SCAFFOLDS.md).

The `sleep_varied` arm of `brain/experiments/sleep_probe.py` replays each told fact in three surface forms:
the original statement from `facts.py` plus these two rewordings. The rewordings are hand-written host text
(like the text-replay scaffold itself) and are meant to be replaced by a paraphrasing brain process.

Rules, enforced by `tests/test_sleep_varied_arm.py`:
- each rewording is a statement (not a question) that keeps the fact's key word;
- no run of 3+ content words from any `PARA` question appears in any rewording, so the paraphrase read-out
  (`PARA`) is not trained on its own question wording. The chosen text additionally avoids any 2-word content
  run of a `PARA` question (a stricter self-check than the rule above), so no rewording echoes a question stem.

Parallel to `facts.FACTS`: entry i rewords FACTS[i][0].
"""
REWORDINGS = [
 ("I have a cat called Pixel.", "The name of my cat is Pixel."),
 ("I like the colour teal best.", "Teal is the colour I choose."),
 ("Lyon is my hometown.", "My birth city is Lyon."),
 ("I have a sister called Marguerite.", "Marguerite is my sister."),
 ("The dog I own is named Biscuit.", "Biscuit is my dog's name."),
 ("My car is a red Volvo.", "The Volvo I drive is red."),
 ("I have a friend called Tomasz.", "Tomasz is my best friend."),
 ("My workplace is the Crumb bakery.", "The bakery I work at is called Crumb."),
 ("The cello is my brother's instrument.", "The cello belongs to my brother."),
 ("Peanuts give me an allergic reaction.", "I have a peanut allergy."),
 ("Oslo is where my grandmother lives.", "My grandma's city is Oslo."),
 ("March is my son's birthday month.", "March is when my son celebrates his birthday."),
]