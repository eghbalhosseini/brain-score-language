#!/bin/bash
#SBATCH --job-name=UD
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --mem=120G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --partition=evlab

ATLAS_ID=$1
echo "model_id:${ATLAS_ID}"
. ~/.bash_profile
conda activate brainscore
echo $(which python)

python /om2/user/ehoseini/brain-score-language/stimulus_sampling/extract_representation_for_UD_corpus.py "$ATLAS_ID"