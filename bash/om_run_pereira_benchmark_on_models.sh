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


python /om2/user/ehoseini/brain-score-language/analysis/compute_benchmark_for_optimized_stimulus_Pereira.py