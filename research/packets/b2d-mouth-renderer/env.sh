# source this at the top of EVERY Bash call of the B2d packet (the Bash tool keeps no state between calls).
export P=/home/dant123/Projects/sim                                   # the primary checkout (on main; pool results land here)
export W=$P/.claude/worktrees/b2d-mouth-renderer                      # this packet's worktree (branch research/b2d-mouth-renderer)
export K=$W/research/packets/b2d-mouth-renderer                       # this packet
export PY=$P/.venv/bin/python                                         # the worktree's .venv is a symlink to this one
export SH=research/findings/raw/_load_bearing/_shards                 # the shard root (relative)
export PREREG=research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md
export H=3c2907b85c10fe918ddc81b4c69de8888c9f321b                     # the pre-registration's original pin
export F=a308f1e09babcc9ed096c3c8046d00040391368c                     # B2b's pin (the drift comparison)
export SSHCFG=$P/research/queue/.pool_ssh_config
export CORPUS_SHA=7a00272e6ca4a29c91d7bc3508de2c76dc1369b351637adf2769e1d3a3679aec
# The pin P, once HANDOFF step 1 has recorded it in $K/PIN_P:
if [ -s "$K/PIN_P" ]; then
  export PIN_P=$(tr -d '[:space:]' < "$K/PIN_P"); export P7=${PIN_P:0:7}
  export TREE=$HOME/derisk-pool/revisions/$PIN_P                      # the workstation's export of P (mouth arm, --code-root)
fi
