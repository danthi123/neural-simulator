#!/usr/bin/env python
"""Measure in-vocab coverage of the two WKV mouth checkpoints (V=1000 vs V=4000)
against the TinyStories corpus, using the SAME in_vocab_scope gate that
webapp/wkv_mouth_generator.py uses (min_frac=0.6, min_hits=2, min_content_hits=2).
This is the 'wider-vocab checkpoint + measure in-vocab coverage' rung (#80 / board #80).
"""
import numpy as np
import re
import sys
sys.path.insert(0, '.')

# Function words (verbatim from webapp/wkv_mouth_generator.py _FUNCTION_WORDS)
FUNCTION_WORDS = frozenset("""
the and a to was they he it she her with in his you but not on i of there so for that is are am
this these those of at as by from into onto up down over under again further then once here there
all any both each few more most other some such no nor only own same so than too very s t can will
just don should now had has have do does did doing having been being am were what which who whom
your my our its today now yesterday tomorrow
""".split())

# Load corpus (single-line file)
corpus_path = 'data/corpus/tinystories.txt'
with open(corpus_path) as f:
    text = f.read()

raw = text.strip()
words_list = re.findall(r'[\w\'\-]+', text)
words_lower = [w.lower() for w in words_list]

# Split into sentences using end-of-sentence punctuation and double-space boundaries
sentences_raw = re.split(r'[\.\!\?]\s+|\s{2,}', raw)
sentences = [s for s in sentences_raw if s.strip()]
sentences = [s for s in sentences if '<|endoftext|' not in s and s.strip()]

# Load checkpoints
d1000 = np.load('bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz', allow_pickle=True)
d4000 = np.load('bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz', allow_pickle=True)
vocab1000 = {w.lower() for w in d1000['words'] if w not in ('<unk>', '<eos>', '<pad>', '<bos>', 'endoftext')}
vocab4000 = {w.lower() for w in d4000['words'] if w not in ('<unk>', '<eos>', '<pad>', '<bos>', 'endoftext')}

print('=' * 72)
print('WKV MOUTH CHECKPOINT COVERAGE (TinyStories corpus, '
      f'{len(set(words_lower))} unique words, '
      f'{len({w for w in set(words_lower) if w not in FUNCTION_WORDS})} content)')
print('=' * 72)

# Word-level coverage
unique_words = set(words_lower)
unique_content = {w for w in unique_words if w not in FUNCTION_WORDS}

content_in_1000 = len(unique_content.intersection(vocab1000))
content_in_4000 = len(unique_content.intersection(vocab4000))
all_in_1000 = len(unique_words.intersection(vocab1000))
all_in_4000 = len(unique_words.intersection(vocab4000))

print(f'\n[WORD-LEVEL coverage]')
print(f'  V=1000 ({len(vocab1000)} words):')
print(f'    All unique words covered: {all_in_1000}/{len(unique_words)} ({all_in_1000/len(unique_words)*100:.2f}%)')
print(f'    Content words covered: {content_in_1000}/{len(unique_content)} ({content_in_1000/len(unique_content)*100:.2f}%)')
print(f'  V=4000 ({len(vocab4000)} words):')
print(f'    All unique words covered: {all_in_4000}/{len(unique_words)} ({all_in_4000/len(unique_words)*100:.2f}%)')
print(f'    Content words covered: {content_in_4000}/{len(unique_content)} ({content_in_4000/len(unique_content)*100:.2f}%)')
print(f'\n  DELTA: +{all_in_4000-all_in_1000} unique words covered, +{content_in_4000-content_in_1000} content words')
print(f'          {((all_in_4000-all_in_1000)/len(unique_words)*100):.2f}% more unique, {((content_in_4000-content_in_1000)/len(unique_content)*100):.2f}% more content')

# Gate-level coverage (the in_vocab_scope gate applied at the "sentence/prompt" level)
def in_vocab_scope_gate(text, vocab, min_frac=0.6, min_hits=2, min_content_hits=2):
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text)]
    if not words:
        return False
    hits = [w for w in words if w in vocab]
    content_hits = [w for w in hits if w not in FUNCTION_WORDS]
    return (len(hits) >= min_hits and (len(hits) / len(words)) >= min_frac
            and len(content_hits) >= min_content_hits)

pass_1000 = pass_4000 = n_prompts = 0
for s in sentences:
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', s)]
    if len(words) < 5 or len(words) > 200:
        continue
    n_prompts += 1
    if in_vocab_scope_gate(s, vocab1000):
        pass_1000 += 1
    if in_vocab_scope_gate(s, vocab4000):
        pass_4000 += 1

print(f'\n[GATE-LEVEL coverage (in_vocab_scope, 5-200 word prompts, {n_prompts} prompts)]')
print(f'  V=1000: {pass_1000}/{n_prompts} ({pass_1000/n_prompts*100:.2f}% pass)')
print(f'  V=4000: {pass_4000}/{n_prompts} ({pass_4000/n_prompts*100:.2f}% pass)')
print(f'  DELTA: +{pass_4000-pass_1000} prompts pass ({((pass_4000-pass_1000)/n_prompts*100):.2f}%)')

print(f'\nSUMMARY: The V=4000 checkpoint covers ~{((content_in_4000-content_in_1000)/len(unique_content)*100):.1f}% more of the corpus content words')
print(f'and has ~{((pass_4000-pass_1000)/n_prompts*100):.1f}% higher in-vocab gate pass rate.')
