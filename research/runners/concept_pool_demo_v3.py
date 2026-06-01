"""64-word concept-pool demo (scale test for the 'training reduces overlap' result).

Same monkey-patch pattern as v2, extended to 64 words: 4 motor + 20 noun + 20 verb + 20 adjective.
Used to test whether the cheap lever (more training -> lower concept overlap, validated multi-seed at 28
words) holds at 64-word LEARNED vocab. For 64 orthogonal cues, use a larger lang_input (8192, stride 128)
with sparsity ~0.01 (n_active ~82 < stride) so input codes stay orthogonal.
"""
import research.runners.concept_pool_demo as cpd_v1

DIRECTION_VOCAB = {"north": "N", "east": "E", "south": "S", "west": "W"}
NOUN_VOCAB = {
    "apple": "APPLE", "river": "RIVER", "dog": "DOG", "cat": "CAT",
    "tree": "TREE", "bird": "BIRD", "sun": "SUN", "moon": "MOON",
    "house": "HOUSE", "book": "BOOK", "fish": "FISH", "rock": "ROCK",
    "star": "STAR", "leaf": "LEAF", "road": "ROAD", "lake": "LAKE",
    "hill": "HILL", "door": "DOOR", "hand": "HAND", "fire": "FIRE",
}
VERB_VOCAB = {
    "go": "GO", "come": "COME", "stop": "STOP", "look": "LOOK",
    "walk": "WALK", "run": "RUN", "eat": "EAT", "sleep": "SLEEP",
    "jump": "JUMP", "sing": "SING", "read": "READ", "write": "WRITE",
    "swim": "SWIM", "fly": "FLY", "push": "PUSH", "pull": "PULL",
    "throw": "THROW", "catch": "CATCH", "climb": "CLIMB", "fall": "FALL",
}
ADJECTIVE_VOCAB = {
    "big": "BIG", "small": "SMALL", "hot": "HOT", "cold": "COLD",
    "red": "RED", "blue": "BLUE", "fast": "FAST", "slow": "SLOW",
    "dark": "DARK", "bright": "BRIGHT", "soft": "SOFT", "hard": "HARD",
    "wet": "WET", "dry": "DRY", "loud": "LOUD", "quiet": "QUIET",
    "heavy": "HEAVY", "light": "LIGHT", "sweet": "SWEET", "sharp": "SHARP",
}

cpd_v1.DIRECTION_VOCAB = DIRECTION_VOCAB
cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.MOTOR_NAMES = ["N", "E", "S", "W"]
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

N_TOTAL_WORDS = 4 + len(NOUN_VOCAB) + len(VERB_VOCAB) + len(ADJECTIVE_VOCAB)  # 64


def main():
    print(f"=== concept_pool_demo_v3 - 64-word vocab ===", flush=True)
    print(f"  Motor 4, Nouns {len(NOUN_VOCAB)}, Verbs {len(VERB_VOCAB)}, Adjectives {len(ADJECTIVE_VOCAB)}"
          f" -> {N_TOTAL_WORDS} words/pools", flush=True)
    cpd_v1.main()


if __name__ == "__main__":
    main()
