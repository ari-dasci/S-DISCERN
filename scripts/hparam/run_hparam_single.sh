#!/bin/bash
#SBATCH -c 4
#SBATCH --job-name hp_single
#SBATCH --mem 16g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%j.out # STDOUT

# Hyper-parameter study of DISCERN (PARAMETRIC cv5x3_tbn_m0.05_tests), single class
# only (the multi / ICU parts were dropped):
#   margin            : notop, 1000 synthetic samples, margin in {0, 5, 10, 20}
#   synthetic samples : {100, 500, 1000, 2000, 5000, 10000}, notop (margin 10) and
#                       corrected_confidence_penalty
# 1. collect_hparam_study.py: processes the study runs (incremental) into
#    results/hparam_study_parametric_<tag>_raw.csv and the per-config means
# 2. analyze_hparam_study.py: paired analysis vs the default (margin 10 / 1000 samples)
# Everything ends up in results/ablation/hparam/.
#   sbatch scripts/hparam/run_hparam_single.sh      (or: bash scripts/hparam/run_hparam_single.sh)

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

TAG=cv5x3_tbn_m0.05_tests
OUT=results/ablation/hparam
mkdir -p "$OUT"

python src/collect/collect_hparam_study.py --parametric-tag "$TAG" --output "$OUT/hparam_study_results.csv" || exit 1
python src/analysis/analyze_hparam_study.py --parametric-tag "$TAG" --out "$OUT"
python src/plots/plot_hparam_evolution.py --in-dir "$OUT"     # metric evolution plots -> $OUT/plots/
echo "Run ended"

conda deactivate
