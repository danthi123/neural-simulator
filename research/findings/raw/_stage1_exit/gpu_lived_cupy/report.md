**Stage-1 exit verdict: UNDEFINED (INCOMPLETE): unmet: no primary shard missing (measured=90 expect=0); unmet: null clean (intact_a == intact_b) on every primary (group, seed) (measured=18 expect=0); unmet: six seeds defined for every primary criterion (measured=3 expect=0)** (primary criteria: C1 INCOMPLETE, C2 INCOMPLETE, C3 FAIL, C4 INCOMPLETE)

| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|---|
| cat_disc | C1 | primary | INCOMPLETE | 0/0 | lesion_draw 0/0 | lesion_draw 0/0 | 0/0 | 0 of 6 seeds defined |
| cat_told | C1 | diagnostic | INCOMPLETE | 0/0 | lesion_draw 0/0 | lesion_draw 0/0 | 0/0 | 0 of 6 seeds defined |
| word | C2 | primary | INCOMPLETE | 0/0 | lesion_recruit 0/0 | lesion_recruit 0/0 | 0/0 | 0 of 6 seeds defined |
| lived | C3 | primary | FAIL | 5/6 | lesion_recall 0/6, lesion_episodic 5/6 | lesion_recall 6/6, lesion_episodic 6/6 | 5/6 | capability absent on 1 of 6 seeds |
| val_heldout | C4 | primary | INCOMPLETE | 0/0 | lesion_affect 0/0 | lesion_affect 0/0 | 0/0 | 0 of 6 seeds defined |
| val_novel | C4 | diagnostic | INCOMPLETE | 0/0 | lesion_affect 0/0 | lesion_affect 0/0 | 0/0 | 0 of 6 seeds defined |

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = UNDEFINED (n = None, sign agreement -); lesion r = None; permuted-map r = None; intact arm missing
- seed 43: r = UNDEFINED (n = None, sign agreement -); lesion r = None; permuted-map r = None; intact arm missing
- seed 44: r = UNDEFINED (n = None, sign agreement -); lesion r = None; permuted-map r = None; intact arm missing
- seed 100: r = UNDEFINED (n = None, sign agreement -); lesion r = None; permuted-map r = None; intact arm missing
- seed 101: r = UNDEFINED (n = None, sign agreement -); lesion r = None; permuted-map r = None; intact arm missing
- seed 102: r = UNDEFINED (n = None, sign agreement -); lesion r = None; permuted-map r = None; intact arm missing

Missing shards: 150

-> research/findings/raw/_stage1_exit/gpu_lived_cupy/aggregate.json
