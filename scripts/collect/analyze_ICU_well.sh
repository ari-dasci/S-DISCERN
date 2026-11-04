#!/bin/bash 
#SBATCH -c 4
#SBATCH --job-name clusterICU
#SBATCH --mem 20g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --array=0%2


eval "$(conda shell.bash hook)"


conda activate unlearning
export PYTHONPATH='.'

# python src/collect/compute_ICU_cluster_metrics.py --continue --online --include-version embs

# python src/collect/compute_ICU_logit_metrics.py --continue --include-version embs RETRAIN

# python src/collect/compute_ICU_logit_metrics.py --continue --include-version DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty RANDOMLABEL

python src/collect/compute_ICU_logit_metrics.py --continue --include-version embs RETRAIN

echo "Run ended"
conda deactivate
