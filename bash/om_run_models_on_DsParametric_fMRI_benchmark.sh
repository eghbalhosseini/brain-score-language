#!/bin/bash

#SBATCH --job-name=ANNSet1
#SBATCH -t 2-12:00:00
#SBATCH -c 3
#SBATCH --mem=120G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL

. ~/.bash_profile
conda activate brainscore
echo $(which python)


ATLAS_ID=$1
echo "model_id:${ATLAS_ID}"

python /rdma/vast-rdma/vast/evlab/ehoseini/brain-score-language/analysis/run_model_against_ANN_benchmarks.py "$ATLAS_ID"