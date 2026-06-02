"""128-word concept-pool CAPACITY scale test (does the cheap-training lever hold past 64 words?).

Same monkey-patch pattern as v2/v3, extended to 128 words: 4 motor + 41 noun + 41 verb + 42 adjective.
For a CAPACITY test, labels are synthetic (recognition depends on the orthogonal CODE per index + the pool,
not on word semantics). 128 orthogonal cues need sparsity < 1/128; use lang_input 4096 (stride 32) +
sparsity 0.005 (n_active ~20 < stride) so input codes stay orthogonal.
"""
import research.runners.concept_pool_demo as cpd_v1

DIRECTION_VOCAB = {"north": "N", "east": "E", "south": "S", "west": "W"}
NOUN_VOCAB = {f"noun{i:02d}": f"NOUN{i:02d}" for i in range(41)}
VERB_VOCAB = {f"verb{i:02d}": f"VERB{i:02d}" for i in range(41)}
ADJECTIVE_VOCAB = {f"adj{i:02d}": f"ADJ{i:02d}" for i in range(42)}

cpd_v1.DIRECTION_VOCAB = DIRECTION_VOCAB
cpd_v1.NOUN_VOCAB = NOUN_VOCAB
cpd_v1.VERB_VOCAB = VERB_VOCAB
cpd_v1.ADJECTIVE_VOCAB = ADJECTIVE_VOCAB
cpd_v1.NOUN_NAMES = list(NOUN_VOCAB.values())
cpd_v1.VERB_NAMES = list(VERB_VOCAB.values())
cpd_v1.MOTOR_NAMES = ["N", "E", "S", "W"]
cpd_v1.ADJECTIVE_NAMES = list(ADJECTIVE_VOCAB.values())

N_TOTAL_WORDS = 4 + len(NOUN_VOCAB) + len(VERB_VOCAB) + len(ADJECTIVE_VOCAB)  # 128


def main():
    print(f"=== concept_pool_demo_v4 - {N_TOTAL_WORDS}-word capacity test ===", flush=True)
    print(f"  Motor 4, Nouns {len(NOUN_VOCAB)}, Verbs {len(VERB_VOCAB)}, Adjectives {len(ADJECTIVE_VOCAB)}",
          flush=True)
    cpd_v1.main()


if __name__ == "__main__":
    main()
