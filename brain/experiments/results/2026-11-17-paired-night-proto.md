# Prototype (not a result): paired-night targets from the 1.5B teacher are too mixed to distil

Seed 7 only, not a test seed. Host pool40/pool41 CPU, 1.5B, teacher = pretrained cortex after reading all 12 tellings
in the one-turn layout (`near_miss_proto --teacher-context`). Files are in `2026-11-17-paired-night-proto/`. This is the
check before building the paired night, the KEEP candidate named in `2026-11-17-teacher-night-1.5b.md`: told-topic
contrast in the night targets.

## Per-episode rewrite frames (OWN question plus k=4 rewrites per told fact)
| frame | own question answered with the told name (of 12) | rewrite replies invented |
|---|---|---|
| N1 | 6 | 16/19 |
| N2 | 6 | 27/35 |
| N3 | 6 | 25/33 |
| NA (new: an untold detail) | 6 | 31/36 |

The OWN frame writes second-person questions ("What's your cat's name?"), as on 11-14. The teacher then answers as
if the fact were its own ("My cat's name is Pixel."). That is the "My name is Dan." defect, so these targets are
unusable. Most rewrites are not about the user ("Do you have a gun?").

## Topic frame T (onsets with each told "my X" topic: "What is my cat ...", k=2), 64 cues
| teacher reply | count |
|---|---|
| declines | 27 |
| names the topic's own told fact ("Where does my grandmother live?" -> "Oslo.") | 7 |
| names a WRONG told fact ("Where does my sister live?" -> "She lives in Lyon."; "Who is my dog named?" -> "Your dog is named Pixel."; every relative's birthday "March") | 9 |
| invents something new ("Where does my son play his guitar?" -> "He plays it in the living room.") | 22 |

## Reading
- Per item, the in-context 1.5B teacher binds told facts to the wrong slot about as often as it uses them right (9
  against 7). It invents on a third of untold details.
- The 40% aggregate ceiling (`2026-11-17-incontext-abstain.md`) hides this.
- Distilling these targets would teach the cross-binding the night already over-teaches ("aunt -> Oslo").
- Filtering the targets by a host rule would be host templating.
- **So no paired night was run as a test.** Code left in place, default OFF and unused by any default:
  - `near_miss_frame="P:<rewrite frame>"` in `Brain`: own question plus rewrites per told episode, teacher in context.
  - `near_miss.topic_cues` plus the NA frame, used by the prototype only.
- At 1.5B, every route that uses the cortex's own judgement as the teacher is bounded by this per-item quality.
- The 2.9B teacher is cleaner in context: 13/48 invented, and a told name reused on only 1 of 48.
