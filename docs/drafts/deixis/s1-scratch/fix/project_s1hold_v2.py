"""Fix-round projection of the stage-0 census under the REVISED S1-hold (scratch; pure arithmetic on the recorded W1
outcomes in census_pool.json, plus the five stores the census m3 list omits: p009, p010, p016, p035, p036, p109 were
stored by W1 per out_b3on_onebrain_single.txt lines 91 and 107 and the B3 extractor).
Exposure lexicon (the revised 4.3): R1 the full English personal-pronoun paradigm (1st, 2nd, 3rd person, all cases,
reflexives), the demonstratives this/that/these/those, former/latter; R2 the reduced/regional variants of the named
reference grammars ('em, ya, ye, yer, yous, youse, thee, thou, thy, thine); R3 the written-chat spelling rules applied
uniformly (apostrophe deletion on every pronoun+'m/'s/'re/'ve contraction the generator renders; you->u, your/you're->ur,
yourself->urself); plus the self-name 'brain'. 'moi' is produced by no rule. The hold replaces a store only."""
import json, re
D = '/home/dant123/Projects/sim-rev/build8/r271/census/'
cen = json.load(open(D + 'census_pool.json')); probe = json.load(open(D + 'probe_set.json')); orc = json.load(open(D + 'oracle.json'))
items = probe['items']; O = orc['items']; W1 = cen['measures']['W1_b3_default']
stored = {r['id']: r['stored'] for r in W1['3_cross_speaker_collisions']['list']}
stored.update({'p009': 'latter marri fenna', 'p010': 'guy like jazz', 'p016': 'fenna like jazz', 'p035': 'somebody like jazz',
               'p036': 'everyone like jazz', 'p109': 'kofi chase cat'})
dropped = set(W1['4_facts_dropped']['list'])
R1 = set("i me my mine myself you your yours yourself yourselves we us our ours ourselves he him his himself she her hers herself "
         "it its itself they them their theirs themselves this that these those former latter".split())
R2 = set("em ya ye yer yous youse thee thou thy thine".split())
R3 = set("im ive hes shes its youre youve theyre theyve weve u ur urself".split())
EXPOSED = R1 | R2 | R3 | {"brain"}
SPK_ADR = set("i me my mine myself you your yours yourself yourselves we us our ours ourselves ya ye yer yous youse thee thou thy thine im ive youre youve weve u ur urself".split())
VERBS = {"chases", "chase", "likes", "like", "married", "studied", "told", "gave", "does", "is", "are", "was"}
def toks(t): return re.findall(r"[a-zA-Z']+", t.lower())
def deictic(iid): return [t for t in O[iid] if t['role'] in ('SPEAKER', 'ADDRESSEE') or t['role'].startswith('ANTECEDENT')]
held = []; m1 = []; m2 = []; m3 = []; m4_silent = 0; m4_held = 0; cost = {'wrong': [], 'overgeneral': [], 'correct_partial': [], 'nonref_name': []}
for it in items:
    iid = it['id']; s = None if iid in dropped else stored.get(iid); tt = toks(it['text']); dt = deictic(iid)
    if s is None:
        if dt: m4_silent += 1
        continue
    hit = [t for t in tt if t in EXPOSED]
    if hit:
        held.append((iid, it['text'], s, hit))
        if dt: m4_held += 1
        if s.split()[0] in VERBS: cost['wrong'].append(iid)
        elif any(t in ('my', 'your', 'our', 'his', 'her', 'their', 'mine', 'yours', 'ours') for t in tt) and s.split()[0] not in hit: cost['overgeneral'].append(iid)
        elif any(w in s.split() for w in ('kofi', 'saoirse', 'fenna', 'dalia')) and not any(w in s.split() for w in hit): cost['correct_partial'].append(iid)
        else: cost['nonref_name'].append(iid)
        continue
    if any(t['tok'] in s.split() and t['role'] in ('SPEAKER', 'ADDRESSEE') or (t['role'].startswith('ANTECEDENT') and t['tok'] in s.split()) for t in O[iid]): m1.append((iid, s))
    if s.split()[0] in VERBS: m2.append(iid)
    if dt: m3.append((iid, s))
print("held turns:", len(held))
for h in held: print("   ", h)
print("m1 (v1 roles + antecedent tokens) still stored as a name:", len(m1), m1)
print("m2 role shifts still stored:", len(m2), m2)
print("m3 collisions still stored (deictic items stored):", len(m3), m3)
print("m4: silent", m4_silent, "+ held-with-class", m4_held)
print("cost classes among held:", {k: (len(v), v) for k, v in cost.items()})
print("non-deictic stored items (must stay stored):", [(it['id'], stored.get(it['id'])) for it in items if it['id'] not in dropped and not deictic(it['id']) and it['id'] in stored and not any(t in EXPOSED for t in toks(it['text']))])
