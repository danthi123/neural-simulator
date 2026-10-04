  attributable_to panel host_decision_share vs sham (no-intervention) reply-noise floor: treatment=+0.987179  control=+0  diff=+0.987179
    => 100.0% of the effect is attributable to the manipulation; 0.0% is ALSO PRESENT IN THE CONTROL
  attributable_to C2 host_decision_share vs sham reply-noise floor: treatment=+1  control=+0  diff=+1
    => 100.0% of the effect is attributable to the manipulation; 0.0% is ALSO PRESENT IN THE CONTROL
  attributable_to C3 host_decision_share vs sham reply-noise floor: treatment=+0.966667  control=+0  diff=+0.966667
    => 100.0% of the effect is attributable to the manipulation; 0.0% is ALSO PRESENT IN THE CONTROL
**Stage-1 host share (preregistered validity): MEASUREMENT-VALID** (seeds [42, 43, 44, 100, 101, 102])

### C2 (`s1_word`, Stage-1 group `word`): PARTIAL (host decision: k-select, bg-select, organ-c, metacog, ignition)

| seam | reported as | class | HS | NS | MS | SC | DA | n events with F | live seeds |
|---|---|---|---|---|---|---|---|---|---|
| k-select | MEASURED | HOST-OWNED (replaceable) | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 18 | 6 |
| bg-select | MEASURED | HOST-OWNED | 1.000 | 0.000 | 0.000 | 0.300 | 0.700 | 30 | 6 |
| organ-c | CALL-GRAPH CHECK | HOST-OWNED | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 18 | 6 |
| metacog | CALL-GRAPH CHECK | HOST-OWNED (replaceable) | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 18 | 6 |
| ignition | CALL-GRAPH CHECK | HOST-OWNED | 1.000 | 0.000 | 0.000 | UNDEFINED | 1.000 | 18 | 6 |

Panel, measured seams: host_decision_share 1.000, neural_decision_fraction 0.000, masked_fraction 0.000 (n 48). Call-graph checks: host_decision_share 1.000 (n 54).

| target turn | probe | k-select F per seed | bg-select F per seed | organ-c F per seed | metacog F per seed | ignition F per seed |
|---|---|---|---|---|---|---|
| s1w_p_bear_chase | counted (new word as patient) | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 |
| s1w_p_otter_eat | counted (new word as agent) | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 |
| s1w_c_otter_chase | counted (moat: abstain) | U,U,U,U,U,U | 1,1,1,1,1,1 | U,U,U,U,U,U | U,U,U,U,U,U | U,U,U,U,U,U |
| s1w_c_clam_eat | counted (moat: abstain) | U,U,U,U,U,U | 1,1,1,1,1,1 | U,U,U,U,U,U | U,U,U,U,U,U | U,U,U,U,U,U |
| s1w_c_dog_chase | counted (built-in fact) | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 |

### C3 (`s1_lived`, Stage-1 group `lived`): PARTIAL (host decision: k-select, bg-select, organ-c, metacog, ignition)

| seam | reported as | class | HS | NS | MS | SC | DA | n events with F | live seeds |
|---|---|---|---|---|---|---|---|---|---|
| k-select | MEASURED | HOST-OWNED (replaceable) | 1.000 | 0.000 | 0.000 | 0.167<!--derived--> | 1.000 | 12 | 6 |
| bg-select | MEASURED | HOST-OWNED | 0.944<!--derived--> | 0.056<!--derived--> | 0.000 | 0.500 | 0.500 | 18 | 6 |
| organ-c | CALL-GRAPH CHECK | HOST-OWNED | 1.000 | 0.000 | 0.000 | 0.500 | 0.500 | 12 | 6 |
| metacog | CALL-GRAPH CHECK | HOST-OWNED (replaceable) | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 12 | 6 |
| ignition | CALL-GRAPH CHECK | HOST-OWNED | 1.000 | 0.000 | 0.000 | UNDEFINED | 1.000 | 12 | 6 |

Panel, measured seams: host_decision_share 0.967<!--derived-->, neural_decision_fraction 0.033<!--derived-->, masked_fraction 0.000 (n 30). Call-graph checks: host_decision_share 1.000 (n 36).

| target turn | probe | k-select F per seed | bg-select F per seed | organ-c F per seed | metacog F per seed | ignition F per seed |
|---|---|---|---|---|---|---|
| s1l_p_wolf_bite | counted (lived fact recall) | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 |
| s1l_c_wolf_chase | counted (moat: abstain) | U,U,U,U,U,U | 0,1,1,1,1,1 | U,U,U,U,U,U | U,U,U,U,U,U | U,U,U,U,U,U |
| s1l_c_dog_chase | counted (built-in fact) | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 | 1,1,1,1,1,1 |

Validity: V1 sham-clean seeds [42, 43, 44, 100, 101, 102] (pass True); V3 applied True; V4 True; V5 drift events 0; V6 pass True; V7 corpus True; PYTHONHASHSEED True; seed threading True; unresolved-turn shards 0; missing shards 0.

Content (30 turns): learned_content_fraction 1.000, python_selected_fraction 0.000, unattributed 0.000.

Harness agreement with the Stage-1 gate's intact_a (descriptive): 48 of 48 target turns.
