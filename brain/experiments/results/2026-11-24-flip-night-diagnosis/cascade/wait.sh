#!/bin/bash
export SSH_AUTH_SOCK=
C=brain-lane/flip_repo/_scratch/flip/cascade
while :; do
  o=$(ssh -o IdentitiesOnly=yes -o ConnectTimeout=15 pool40 "grep -c -E '^DONE|rc [1-9]' ~/$C/done.txt; grep -h -E 'VERDICT|match_banked' ~/$C/probe_s*.log | cut -c1-260" </dev/null)
  if [ "$(echo "$o" | head -1)" != "0" ]; then echo "$o"; exit 0; fi
  sleep 60
done
