cd /home/dant123/Projects/sim
for HI in 400 800 1500; do
  OMP_NUM_THREADS=2 .venv/bin/python3 -m research.runners._gap5_spiking_gamma_replay_derisk \
    --seeds 42 --theta-ramp --ramp-hi $HI --ramp-lo 0 --theta-period 120 --rest-steps 1200 \
    --out research/findings/raw/gap5_r4/ramp/ramp_hi${HI}.json 2>&1 | grep -E "seed 42|VERDICT" | sed "s/^/[hi=$HI] /"
done
echo "SWEEP_DONE"
