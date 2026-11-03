# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""Two teacher-written rewordings per told statement (declaration: brain/SCAFFOLDS.md).

The `sleep_varied` arm of `brain.experiments.sleep_probe` replays each told fact in three surface forms: the
original statement from `facts.py` plus these two rewordings. The rewordings are hand-written host text (like the
text-replay scaffold itself) and are meant to be replaced by a paraphrasing brain process.

Rules, enforced by `tests/test_sleep_varied_arm.py`:
- each rewording is a statement (not a question) that keeps the fact's key word;
- no rewording contains a content word of its own fact's `PARA` question that is absent from both the original
  statement and its question - otherwise the `PARA` read-out would be word-priming, not consolidation;
- no rewording shares 2 or more content words with ANY `PARA` question, in any order (this subsumes the item's
  "no run of 3+ content words of a PARA question" rule, which the older tests still pin).

Parallel to `facts.FACTS`: entry i rewords FACTS[i][0].
"""
REWORDINGS = [
 ("I have a cat called Pixel.", "Pixel is my cat."),
 ("Teal is the colour I choose.", "When I pick a colour, it is teal."),
 ("I come from Lyon.", "Lyon is where I was born."),
 ("I have a sister called Marguerite.", "Marguerite is my sister."),
 ("The dog I own is named Biscuit.", "Biscuit is the dog in my family."),
 ("My car is a red Volvo.", "The Volvo I drive is red."),
 ("I have a friend called Tomasz.", "Tomasz is my best friend."),
 ("The Crumb bakery is where I work.", "The bakery I work at is called Crumb."),
 ("My brother has a cello at home.", "The cello belongs to my brother."),
 ("Peanuts give me an allergic reaction.", "I have a peanut allergy."),
 ("Oslo is where my grandmother lives.", "Oslo is my grandmother's home."),
 ("My son's birthday falls in March.", "March is when my son celebrates his birthday."),
]