# Integration wave 2: `source` this at the top of EVERY HANDOFF command block (the Bash tool keeps no state between
# calls). Sets the paths, and the pins recorded so far in PINS.tsv (the LAST P row and the LAST F row).
P=/home/dant123/Projects/sim
W=$P/.claude/worktrees/integration-wave2
K=research/packets/integration-wave2
SH=research/findings/raw/_load_bearing/_shards
PIN_P=""; PIN_F=""; FLIP=""
if [ -f "$W/$K/PINS.tsv" ]; then
  PIN_P=$(awk -F'\t' '$1=="P"{p=$2} END{print p}' "$W/$K/PINS.tsv")
  PIN_F=$(awk -F'\t' '$1=="F"{p=$2} END{print p}' "$W/$K/PINS.tsv")
  FLIP=$(awk -F'\t' '$1=="F"{p=$3} END{print p}' "$W/$K/PINS.tsv")
fi
P7=${PIN_P:0:7}; F7=${PIN_F:0:7}
