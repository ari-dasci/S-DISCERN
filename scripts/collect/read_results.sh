#!/bin/bash 
#SBATCH -c 4
#SBATCH --job-name results
#SBATCH --mem 10g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --array=0%1


eval "$(conda shell.bash hook)"


conda activate unlearning
export PYTHONPATH='.'


commands=(
    # "python src/collect/process_results_auto.py --base-dir results --results-file unlearning_results_all"
    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --results-file unlearning_results_single_all_new --online --continue"


    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version RETRAIN freeze --exclude-version nofreeze --results-file unlearning_results_single_freeze_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --exclude-version /original --results-file unlearning_results_single_freeze_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version RETRAIN embs --results-file unlearning_results_single_embs_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --exclude-version /original --results-file unlearning_results_single_embs_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --results-file unlearning_results_multiple_all_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --include-version RETRAIN freeze --exclude-version nofreeze --results-file unlearning_results_multiple_freeze_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --exclude-version /original --results-file unlearning_results_multiple_freeze_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --include-version RETRAIN embs --results-file unlearning_results_multiple_embs_new --online --continue"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --exclude-version /original --results-file unlearning_results_multiple_embs_new --online --continue"


    # RETAIN DISTANCES

    "python src/collect/process_results_auto_retain.py --base-dir results --classes-filter single --include-version RETRAIN embs --results-file unlearning_results_single_embs_new_retain --online --continue"

    "python src/collect/process_results_auto_retain.py --base-dir results --classes-filter single --exclude-version /original --results-file unlearning_results_single_embs_new_retain --online --continue"


    "python src/collect/process_results_auto_retain.py --base-dir results --classes-filter multiple --include-version RETRAIN embs --results-file unlearning_results_multiple_embs_new_retain --online --continue"

    "python src/collect/process_results_auto_retain.py --base-dir results --classes-filter multiple --exclude-version /original --results-file unlearning_results_multiple_embs_new_retain --online --continue"

    # NEAR

    "python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single --include-version RETRAIN embs --results-file unlearning_results_single_embs_new_near --online --continue"

    "python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single --exclude-version /original --results-file unlearning_results_single_embs_new_near --online --continue"


    "python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter multiple --include-version RETRAIN embs --results-file unlearning_results_multiple_embs_new_near --online --continue"

    "python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter multiple --exclude-version /original --results-file unlearning_results_multiple_embs_new_near --online --continue"


    "python src/collect/process_results_auto.py --base-dir results --include-version RETRAIN --results-file unlearning_results_all_embs"
    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notopexpandalign --results-file unlearning_results_single_all_embs --append"
    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notopalign --results-file unlearning_results_single_all_embs --append"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notopexpand DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notopalign DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_moving_embs_forget_notopexpandalign --results-file unlearning_results_single_all_embs --append"

    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --include-version RETRAIN --results-file unlearning_results_multiple_all_embs"

    "python src/collect/process_results_auto.py --base-dir results --include-version freeze --results-file unlearning_results_all_freeze"
    "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version freeze --results-file unlearning_results_single_all_freeze"
    "python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --include-version freeze --results-file unlearning_results_multiple_all_freeze"
)

# commands=(
#     "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_single_all_embs --online --continue"

# )


for command in "${commands[@]}"; do
    echo "Running: $command"
    eval "$command"
done
# echo "Running: ${commands[$SLURM_ARRAY_TASK_ID]}"
# eval "${commands[${SLURM_ARRAY_TASK_ID}]}"

# # echo "Running: ${commands[$SLURM_ARRAY_TASK_ID]}"
# # eval "${commands[${SLURM_ARRAY_TASK_ID}]}"
# python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_single_all_embs --online --continue
# # commands=(
# #     ""

# # )
# python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_retain --online --continue
# # echo "Running: ${commands[$SLURM_ARRAY_TASK_ID]}"
# # eval "${commands[${SLURM_ARRAY_TASK_ID}]}"

# # commands=(
# #     "python src/collect/process_results_auto_retain.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_single_all_unkown_3 --online --continue"

# # )

# # echo "Running: ${commands[$SLURM_ARRAY_TASK_ID]}"

# python src/collect/process_results_auto_retain.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_single_all_unkown_3 --online --continue
# # eval "${commands[${SLURM_ARRAY_TASK_ID}]}"

# # commands=(
# #     "python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_all_embs --online --continue"

# # )

# # echo "Running: ${commands[$SLURM_ARRAY_TASK_ID]}"
# # eval "${commands[${SLURM_ARRAY_TASK_ID}]}"
# python src/collect/process_results_auto.py --base-dir results --classes-filter single --include-version embsunfreeze --results-file unlearning_results_all_embs --online --continue

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_multiple_all.csv --base-dir datos_recopilados_finales

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_multiple_embs.csv --base-dir datos_recopilados_finales

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_multiple_freeze.csv --base-dir datos_recopilados_finales

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_single_all.csv --base-dir datos_recopilados_finales

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_single_embs.csv --base-dir datos_recopilados_finales

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_single_freeze.csv --base-dir datos_recopilados_finales

# python src/correct_process_results.py datos_recopilados_finales/unlearning_results_single_freeze.csv --base-dir datos_recopilados_finales




# python src/correct_process_results.py results/unlearning_results_all_embs.csv --base-dir results

# python src/correct_process_results.py results/unlearning_results_multiple_all_embs.csv --base-dir results

# python src/correct_process_results.py results/unlearning_results_single_all_embs.csv --base-dir results

# python src/correct_process_results.py results/unlearning_results_single_all_unkown_3.csv --base-dir results

# python src/correct_process_results.py results/unlearning_results_retain.csv --base-dir results

# # python src/collect/process_results_auto.py --base-dir results --classes-filter multiple --include-version RETRAIN --results-file unlearning_results_multiple_all_embs --append
# # python src/analysis/analyze_aggregated.py results/unlearning_results.csv
echo "Run ended"
conda deactivate
