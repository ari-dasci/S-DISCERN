#!/bin/bash 
#SBATCH -c 1
#SBATCH --job-name method
#SBATCH --mem 10g
#SBATCH -o ./slrums/%j.out # STDOUT


eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

python src/training/download_databases.py
conda deactivate
