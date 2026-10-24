"""Judge's projection of the stage-0 census under S1-hold as settled in round2/DESIGN.md:
exposure lexicon = the standard English first/second-person paradigm only (no census slang); the hold replaces a
STORE only (a turn B3 would not store is byte-identical to today); any exposed form latching a pointer pool, or the
self-name ('brain') latching E_self, holds the turn. Pure arithmetic on the recorded W1 outcomes (census_pool.json)."""
import json, re
D = '/home/dant123/Projects/sim-rev/build8/r271/census/'
cen = json.load(open(D+'census_pool.json')); probe = json.load(open(D+'probe_set.json')); orc = json.load(open(D+'oracle.json'))
items = probe['items']; O = orc['items']; W1 = cen['measures']['W1_b3_default']
stored = {r['id']: r['stored'] for r in W1['3_cross_speaker_collisions']['list']}
dropped = set(W1['4_facts_dropped']['list'])
EXPOSED = {"i","me","my","mine","myself","you","your","yours","yourself","we","us","our","ours","ourselves","brain"}
VERBS = {"chases","chase","likes","like","married","studied","told","gave","does","is","are","was"}
def toks(t): return re.findall(r"[a-zA-Z']+", t.lower())
def deictic(iid): return [t for t in O[iid] if t['role'] in ('SPEAKER','ADDRESSEE') or t['role'].startswith('ANTECEDENT')]
def today(it): return stored.get(it['id']) if it['id'] not in dropped else None
# items stored today but not in the m3 list (non-deictic items): the census only lists deictic stores in m3; reconstruct
# which non-deictic items are stored: every item not in m4_dropped and not in m3 is stored (B3 stores it) -- we only need
# the deictic ones for the measures, and the correct-facts-lost count.
held = []; m1 = m2 = m3 = 0; m4_silent = 0; m4_held = 0; lost_correct = []; lost_overgeneral = []; lost_wrong = []
for it in items:
    s = today(it); tt = toks(it['text']); dt = deictic(it['id'])
    if s is None:
        if dt: m4_silent += 1
        continue
    if any(t in EXPOSED for t in tt):          # organ latches a pointer / self-name -> HELD (store replaced)
        held.append((it['id'], it['text'], s))
        if dt: m4_held += 1
        # classify what today's store was
        if s.split()[0] in VERBS: lost_wrong.append(it['id'])
        elif any(t in ('my','your','our','mine','yours','ours') for t in tt): lost_overgeneral.append(it['id'])
        elif any(w in s.split() for w in ('kofi','saoirse','fenna')): lost_correct.append(it['id'])
        else: lost_wrong.append(it['id'])
        continue
    # unchanged store
    if any(t['tok'] in s.split() and t['role'] in ('SPEAKER','ADDRESSEE') for t in O[it['id']]): m1 += 1
    if s.split()[0] in VERBS: m2 += 1
    if dt: m3 += 1
print("S1-hold projection (standard paradigm exposure; hold replaces a store only)")
print(" held turns:", len(held))
for h in held: print("   ", h)
print(" m1 deictic-as-name: 28 ->", m1, " (residue = slang forms never exposed)")
print(" m2 role shifts: 4 ->", m2)
print(" m3 collisions: 60 ->", m3)
print(" m4 dropped: 60 silent ->", m4_silent, "silent +", m4_held, "held-with-class")
print(" lost stores among held: wrong/partial", len(lost_wrong), lost_wrong, "| over-general", len(lost_overgeneral), lost_overgeneral, "| correct third-party partial", len(lost_correct), lost_correct)
