#!/usr/bin/env bash
# Run ESKMeans for all recordings in the speaker list.
# Set ESKMEANS_DATA to point to your prepared data directory.
#
# Usage:
#   export ESKMEANS_DATA=/shared/ramon/experiments_old
#   bash run.sh

language=buckeye
feature_type=hubert_base_ls960
layer=10
pooling_type=average
kmeans_type=em

cat data/file_list/buckeye_spk | parallel \
    python run.py \
        --speaker {} \
        --language ${language} \
        --feature_type ${feature_type} \
        --layer ${layer} \
        --pooling_type ${pooling_type} \
        --kmeans_type ${kmeans_type}
