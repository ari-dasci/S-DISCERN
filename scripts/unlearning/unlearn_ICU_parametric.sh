#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 30g
#SBATCH --job-name icu_param
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

# ICU (incremental class unlearning) with PARAMETRIC, same setup as the
# single-class runs of unlearn_parametric.sh: 1000 fixed synthetic samples,
# --emb-unfreeze embs, notop margin 10, and the same VARIANT/tag.
# Mirrors unlearn_ICU_20.sh: cifar100, 20 classes per seed, seeds 42-51.
#
# Array: task = loss_index * 4 + architecture_index
#   0-3 : notop                         x (resnet18, efficientnet_v2_s, swin_t, vit_b_16)
#   4-7 : corrected_confidence_penalty  x (same order)
#
# The per-class fits are the same cache as the single-class runs
# (standard_cov_parametric_<tag>/<arch>_<dataset>_original_epoch_best_dist_infos.pt,
# fitted once on the original model).  If it already exists nothing is refitted.
# If it does not, only the notop task computes it and the ccp task of the same
# architecture waits for it, so two tasks never write the same file.
#
# Results: results_ICU/<arch>_<sequence>/PARAMETRIC_<tag>_.../original_best/unlearning_results_ICU.csv
# (main_unlearning_ICU.py skips sequences already in the CSV, so it is safe to re-submit).
#
#   sbatch --export=ALL,VARIANT=tiebreak_norm scripts/unlearning/unlearn_ICU_parametric.sh
#   sbatch --array=3,7 --export=ALL,VARIANT=ksnorm scripts/unlearning/unlearn_ICU_parametric.sh   # only vit_b_16

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

experiments=("resnet18" "efficientnet_v2_s" "swin_t" "vit_b_16")
losses=("notop" "corrected_confidence_penalty")
experiment=${experiments[$((SLURM_ARRAY_TASK_ID % 4))]}
loss_idx=$((SLURM_ARRAY_TASK_ID / 4))
loss=${losses[$loss_idx]}

# Variant (sbatch --export=ALL,VARIANT=tiebreak_energy[,REPEATS=3] <script>).
# Every variant uses 5-fold CV repeated REPEATS times (default 3: corrected
# repeated k-fold t-test, more power) and caps the t/Laplace EM at 20 iterations (not converged = skipped).
# The log of every ll_tiebreak run shows the winner under BOTH KS and energy;
# the variant only decides which one is used for the synthetic samples.
#   tiebreak          : ll ties (t-test p>0.05 or gap <= MARGIN nats/dim) resolved by KS -> PARAMETRIC_cv5x3_tb_m0.05_...
#   tiebreak_energy   : same ties resolved by energy score        -> PARAMETRIC_cv5x3_tbe_m0.05_...
#   tiebreak_norm (DEFAULT): ties resolved by KS of ||x|| (shell) -> PARAMETRIC_cv5x3_tbn_m0.05_...
#   tiebreak_pca / tiebreak_energy_pca : + option B (PCA 0.99)  -> ..._tb_m0.05_pca0.99_ / ..._tbe_m0.05_pca0.99_
#   base   : max CV log-likelihood                              -> PARAMETRIC_cv5x3_...
#   ks     : option A, only families passing the KS             -> PARAMETRIC_cv5x3_ks0.1_...
#   pca / ks_pca : option B / A + B
#   ksnorm / ksnorm_pca : only families whose samples reproduce the norm ||x||
#                (KS D <= 0.1 + noise), then max log-lik  -> PARAMETRIC_cv5x3_kn0.1_...
#   energy / energy_pca : min energy score (/ + option B)
# Each variant has its own fit cache (standard_cov_parametric_<tag>/), so
# different variants can run as separate jobs at the same time.
VARIANT=${VARIANT:-tiebreak_norm}
REPEATS=${REPEATS:-3}
MARGIN=${MARGIN:-0.05}   # tie margin, nats per dimension (0 = only the t-test)
PCA="--parametric-subspace pca --parametric-pca-variance 0.99"
case "$VARIANT" in
    tiebreak)            sel="--parametric-selection ll_tiebreak --parametric-tiebreak ks";     stag="_tb" ;;
    tiebreak_energy)     sel="--parametric-selection ll_tiebreak --parametric-tiebreak energy"; stag="_tbe" ;;
    tiebreak_norm)       sel="--parametric-selection ll_tiebreak --parametric-tiebreak ks_norm"; stag="_tbn" ;;
    tiebreak_pca)        sel="--parametric-selection ll_tiebreak --parametric-tiebreak ks";     stag="_tb" ;;
    tiebreak_energy_pca) sel="--parametric-selection ll_tiebreak --parametric-tiebreak energy"; stag="_tbe" ;;
    tiebreak_norm_pca)   sel="--parametric-selection ll_tiebreak --parametric-tiebreak ks_norm"; stag="_tbn" ;;
    base)       sel="";                                                    stag="" ;;
    ks)         sel="--parametric-selection ll_ks --parametric-ks-max 0.1"; stag="_ks0.1" ;;
    pca)        sel="${PCA}";                                              stag="_pca0.99" ;;
    ks_pca)     sel="--parametric-selection ll_ks --parametric-ks-max 0.1 ${PCA}"; stag="_ks0.1_pca0.99" ;;
    ksnorm)     sel="--parametric-selection ll_ksnorm --parametric-ks-max 0.1"; stag="_kn0.1" ;;
    ksnorm_pca) sel="--parametric-selection ll_ksnorm --parametric-ks-max 0.1 ${PCA}"; stag="_kn0.1_pca0.99" ;;
    energy)     sel="--parametric-selection energy";                       stag="_energy" ;;
    energy_pca) sel="--parametric-selection energy ${PCA}";                stag="_energy_pca0.99" ;;
    *) echo "Unknown VARIANT=$VARIANT"; exit 1 ;;
esac
# tie margin (tiebreak variants only), then option B, in the same order as PARAMETRICconfig
case "$VARIANT" in
    tiebreak*)
        sel="${sel} --parametric-tie-margin ${MARGIN}"
        # write MARGIN as Python prints the float (0.05, 0.1, not 0.10) so the tag matches
        if awk "BEGIN{exit !(${MARGIN} > 0)}"; then stag="${stag}_m${MARGIN}"; fi ;;
esac
case "$VARIANT" in
    tiebreak_pca|tiebreak_energy_pca|tiebreak_norm_pca) sel="${sel} ${PCA}"; stag="${stag}_pca0.99" ;;
esac
# PIPELINE=tests (DEFAULT; with VARIANT=tiebreak_norm this is DISCERN): DISTRIBUTIONAL logic (HZ / shell+Rayleigh / uniformity tests first),
# the selection above replaces its BIC duel only when no test accepts.  PIPELINE=cv: CV only.
PIPELINE=${PIPELINE:-tests}
if [ "$PIPELINE" = "tests" ]; then sel="${sel} --parametric-pipeline tests"; stag="${stag}_tests"; fi
if [ "$REPEATS" -gt 1 ]; then rtag="x${REPEATS}"; else rtag=""; fi
tag="cv5${rtag}${stag}"
variant_args="--cv-repeats ${REPEATS} --parametric-em-max-iter 20 ${sel}"
echo "PARAMETRIC variant: $VARIANT, pipeline $PIPELINE, repeats $REPEATS (tag $tag)"

method="--unlearning-method PARAMETRIC --cv-folds 5 ${variant_args} --syntetic-samples 1000 --fixed-syntetic --emb-unfreeze embs --notop-margin 10 --loss-forget ${loss}"
DATASETS=${DATASETS:-cifar100}

seeds=(42 43 44 45 46 47 48 49 50 51)
seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 27 29 64 77 71"
seed_43="4 36 89 97 18 59 47 85 12 58 76 63 77 2 65 55 73 79 70 96"
seed_44="52 66 69 89 14 22 48 28 37 3 72 1 12 20 99 65 88 38 81 75"
seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 95 83 4 79 8"
seed_46="9 51 5 75 29 81 66 19 74 68 4 3 8 40 80 6 17 70 37 67"
seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61"
seed_48="70 40 16 71 91 68 38 64 24 97 55 21 19 84 13 63 83 28 27 65"
seed_49="8 44 52 14 41 96 70 65 6 87 5 97 94 35 19 58 93 80 69 64"
seed_50="63 34 46 81 31 88 60 98 42 10 68 40 28 86 71 19 44 12 24 8"
seed_51="31 64 70 20 29 32 50 59 93 89 67 80 43 75 84 46 49 35 51 0"

for dataset in ${DATASETS}; do
    cache="standard_cov_parametric_${tag}/${experiment}_${dataset}_original_epoch_best_dist_infos.pt"
    if [ "$loss_idx" -gt 0 ]; then
        waited=0
        while [ ! -f "$cache" ]; do
            if [ "$waited" -ge 1440 ]; then echo "Cache $cache still missing after 24 h — aborting"; exit 1; fi
            echo "Waiting for $cache (computed by the notop task of ${experiment})..."
            sleep 600; waited=$((waited + 10))
        done
        sleep 60   # let torch.save finish writing
    fi
    for seed in "${seeds[@]}"; do
        classes_to_remove_var="seed_${seed}"
        classes_to_remove=${!classes_to_remove_var}
        echo "Starting ICU ${experiment} ${dataset} seed ${seed} (${loss}, tag ${tag}): ${classes_to_remove}"
        python src/unlearning/main_unlearning_ICU.py -o models -e ${experiment}_${dataset} -m ${experiment} --save-epoch 5 --max-epoch 30 -d datasets --dataset ${dataset} --data-mode CR --train-batch-size 64 --val-batch-size 16 --seed $seed --classes-to-remove $classes_to_remove --version original --epoch best ${method}
    done
done

echo "Run ended"

conda deactivate
