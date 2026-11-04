#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name extr_mparam
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

# Extract embeddings/logits of the PARAMETRIC multi-class runs of
# unlearn_multi_parametric.sh (checkpoints "best" and "00", as in extract_multi.sh).
# One array task per architecture; same VARIANT / PIPELINE as the training job.
#   sbatch --dependency=afterany:<JOBID_multi> --export=ALL,VARIANT=...,PIPELINE=... scripts/extraction/extract_multi_parametric.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

experiments=("resnet18" "efficientnet_v2_s" "swin_t" "vit_b_16")
experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
loss_tags=("notop_margin_10.0" "corrected_confidence_penalty")
dataset=cifar100
KS=${KS:-"5 10 20 50"}

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

# --- 5 classes (from unlearn_multi_5.sh)
k5_seed_42="81 14 3 94 35"
k5_seed_43="4 36 89 97 18"
k5_seed_44="52 66 69 89 14"
k5_seed_45="34 53 62 32 10"
k5_seed_46="9 51 5 75 29"
k5_seed_47="45 8 55 70 58"
k5_seed_48="70 40 16 71 91"
k5_seed_49="8 44 52 14 41"
k5_seed_50="63 34 46 81 31"
k5_seed_51="31 64 70 20 29"
# --- 10 classes (from unlearn_multi_10.sh)
k10_seed_42="81 14 3 94 35 31 28 17 13 86"
k10_seed_43="4 36 89 97 18 59 47 85 12 58"
k10_seed_44="52 66 69 89 14 22 48 28 37 3"
k10_seed_45="34 53 62 32 10 38 43 2 9 61"
k10_seed_46="9 51 5 75 29 81 66 19 74 68"
k10_seed_47="45 8 55 70 58 73 43 32 65 49"
k10_seed_48="70 40 16 71 91 68 38 64 24 97"
k10_seed_49="8 44 52 14 41 96 70 65 6 87"
k10_seed_50="63 34 46 81 31 88 60 98 42 10"
k10_seed_51="31 64 70 20 29 32 50 59 93 89"
# --- 20 classes (from unlearn_multi_20.sh)
k20_seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 27 29 64 77 71"
k20_seed_43="4 36 89 97 18 59 47 85 12 58 76 63 77 2 65 55 73 79 70 96"
k20_seed_44="52 66 69 89 14 22 48 28 37 3 72 1 12 20 99 65 88 38 81 75"
k20_seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 95 83 4 79 8"
k20_seed_46="9 51 5 75 29 81 66 19 74 68 4 3 8 40 80 6 17 70 37 67" 
k20_seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61"
k20_seed_48="70 40 16 71 91 68 38 64 24 97 55 21 19 84 13 63 83 28 27 65"
k20_seed_49="8 44 52 14 41 96 70 65 6 87 5 97 94 35 19 58 93 80 69 64"
k20_seed_50="63 34 46 81 31 88 60 98 42 10 68 40 28 86 71 19 44 12 24 8"
k20_seed_51="31 64 70 20 29 32 50 59 93 89 67 80 43 75 84 46 49 35 51 0"
# --- 50 classes (from unlearn_multi_50.sh)
k50_seed_42="81 14 3 94 35 31 28 17 13 86 69 11 75 54 4 97 88 27 29 64 77 84 71 25 89 53 93 57 95 0 20 90 43 79 19 82 67 6 5 24 62 22 68 58 38 16 51 2 46 99"
k50_seed_43="4 36 89 18 59 47 85 97 12 58 76 63 77 2 65 55 73 94 79 70 84 50 22 7 13 15 48 19 91 20 93 88 16 49 64 34 23 24 99 3 37 60 11 90 53 10 96 17 61 33"
k50_seed_44="52 66 69 89 14 22 48 28 37 3 92 95 72 1 12 20 65 38 81 93 75 51 41 45 9 43 36 84 88 83 8 85 34 40 13 91 4 94 24 6 30 59 68 57 97 16 47 15 49 90"
k50_seed_45="34 53 62 32 10 38 43 2 9 61 1 14 36 15 39 83 4 79 8 17 99 19 27 54 91 52 18 7 84 22 41 96 45 40 12 74 11 89 48 21 20 50 24 44 47 33 42 82 66 46"
k50_seed_46="9 51 5 75 96 29 81 66 19 74 68 4 3 8 40 80 6 87 17 70 37 67 42 11 50 61 46 91 60 59 99 58 14 55 28 43 56 49 45 93 69 83 38 22 88 57 21 94 25 76"
k50_seed_47="45 8 55 70 58 73 43 32 65 49 50 78 5 53 84 3 13 0 33 61 69 1 40 72 54 29 30 66 91 99 56 94 4 47 96 93 25 15 89 86 46 95 2 98 21 6 23 37 24 92"
k50_seed_48="70 40 16 71 91 68 38 64 24 55 21 19 84 13 63 83 28 27 65 96 81 88 60 82 3 9 54 8 97 85 20 80 7 37 43 79 12 53 15 17 94 41 0 56 36 18 98 72 22 58"
k50_seed_49="8 44 52 14 41 70 65 6 87 5 35 19 58 90 80 69 64 28 77 36 73 47 51 3 31 17 12 54 48 95 45 34 81 92 97 26 99 85 16 68 38 60 20 63 75 96 57 61 37 27"
k50_seed_50="63 34 46 81 31 88 60 42 10 68 40 28 86 71 91 19 44 12 83 89 94 24 8 92 54 11 77 76 55 0 29 43 98 87 13 25 56 17 57 96 37 69 73 62 39 65 67 59 7 26"
k50_seed_51="31 64 70 20 99 29 97 32 50 93 59 90 67 80 43 75 46 49 35 51 0 14 3 68 33 4 61 22 44 38 47 57 24 77 56 21 60 79 58 66 6 52 91 73 69 65 76 37 89 11"

seeds=(42 43 44 45 46 47 48 49 50 51)
for lt in "${loss_tags[@]}"; do
    method="PARAMETRIC_${tag}_1000_standard_syntetic_fixed_embs_forget_${lt}"
    for k in ${KS}; do
        for seed in "${seeds[@]}"; do
            var="k${k}_seed_${seed}"
            classes_to_remove=${!var}
            model_dir="models/${experiment}_${dataset}_classes_to_remove_${classes_to_remove// /_}/${method}/original_best"
            if [ ! -d "$model_dir" ]; then
                echo "Missing ${model_dir} — skipping"
                continue
            fi
            for ckpt in best 00; do
                [ "$ckpt" == "00" ] && [ ! -f "${model_dir}/checkpoint_epoch_00.pt" ] && continue
            [ "$ckpt" == "00" ] && [ ! -f "${model_dir}/checkpoint_epoch_00.pt" ] && continue
                echo "Extracting ${experiment} k=${k} seed ${seed} ${method} checkpoint ${ckpt}"
                python src/training/main_extractor.py -o results --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --data-mode CR --classes-to-remove ${classes_to_remove} --unlearning-method ${method} --version original_best --checkpoint ${ckpt}
            done
        done
    done
done

echo "Run ended"

conda deactivate
