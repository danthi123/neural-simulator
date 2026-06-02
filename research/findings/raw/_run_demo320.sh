#!/bin/bash
set -e
R="research/findings/raw/g11_bg"
SCRIPT="remember apple is big,what is apple,is apple big?,remember dog run big,who run big?,what did dog run?,remember river is wide,what is river,trace apple"
python -m research.runners.g20_multibridge --sparse \
  --pattern-size 100 --n-shared-pool 2000 --n-lang-input 8192 --sparsity 0.007 --seed 42 \
  --bridges $R/g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5 \
            $R/g20_sparse_bridges_320/bridgeB_verbs_sparse64.simstate.h5 \
            $R/g20_sparse_bridges_320/bridgeC_adj_sparse64.simstate.h5 \
            $R/g20_sparse_bridges_320/bridgeD_spatial_sparse64.simstate.h5 \
            $R/g20_sparse_bridges_320/bridgeE_functional_sparse64.simstate.h5 \
  --vocab-files $R/g20_bridgeA_nouns_vocab64.txt $R/g20_bridgeB_verbs_vocab64.txt \
                $R/g20_bridgeC_adj_vocab64.txt $R/g20_bridgeD_spatial_vocab64.txt \
                $R/g20_bridgeE_functional_vocab64.txt \
  --names bridgeA_nouns bridgeB_verbs bridgeC_adj bridgeD_spatial bridgeE_functional \
  --scripted "$SCRIPT"
echo "##### DEMO320-DONE #####"
