# _deixis_census — frozen probe set + oracle (PACKET deixis-271-s0)

Probe sentences (r271 appendix verbatim, copied from the packet appendix):

i chase cat · i chase the cat · it likes jazz · he chases fenna · you chase cat · they chase dog · moi likes jazz · ours likes jazz · the latter married fenna · that guy likes jazz · you and saoirse studied at dunmere · im chasing cat · hes chasing cat · yous chase cat · my cousin likes jazz · fenna does not like jazz · yer auntie moira likes jazz · ur mom likes jazz · kofi likes your cousin dalia · i told kofi jazz · you gave fenna jazz · i chase dog quickly · we like jazz music · my cat likes fish · your cat likes fish · her cat likes fish · kofi and i chase cat · me and kofi chase cat · yours likes jazz · mine likes jazz · em likes jazz · ya likes jazz · thee likes jazz · urself likes jazz · somebody likes jazz · everyone likes jazz

## Grid crossing (see probe_set.json construction)
- person: 1st/2nd/3rd, singular and plural (6 paradigms) x position
  (agent, patient, possessive-in-agent, possessive-in-patient).
- spelling: standard plus slang (moi, yer, yous, ur, im, hes, ya, em,
  thee, mine, yours, ours) x 4 positions.
- anaphor: none, antecedent present, antecedent missing, former/latter.
- coordination: "kofi and i", "me and kofi".
- negation: person x 2 negation surfaces (does not / does n't).
- Every item is spoken by TWO simulated speakers (A and B).

## Oracle
oracle.json labels each token SPEAKER / ADDRESSEE / ANTECEDENT(x) /
NONE / NAME, generated from the template metadata (scorer only, never a
mechanism). sha256 of both files is printed by the builder and recorded
in the census record.
