#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 20g
#SBATCH --job-name near_param
#SBATCH --output=slrums/output_%A_%a.out # STDOUT
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%4

# NEAR extraction (near-OOD dataset) of the PARAMETRIC single-class models, as in
# extract_NEAR.sh: cifar10 models on cifar100, cifar100 models on cifar10, mnist
# models on letters; checkpoints 00 and best; output in results_NEAR/.
# One array task per architecture, both losses; same VARIANT / PIPELINE as the
# training job.  Skips what is already extracted or not trained.
#   sbatch --dependency=afterany:<single job> --export=ALL,VARIANT=...,PIPELINE=... scripts/extraction/extract_NEAR_parametric.sh

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

experiments=("resnet18" "efficientnet_v2_s" "swin_t" "vit_b_16")
experiment=${experiments[$SLURM_ARRAY_TASK_ID]}
loss_tags=("notop_margin_10.0" "corrected_confidence_penalty")

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

declare -A NEAR_DATASET=(["cifar10"]="cifar100" ["cifar100"]="cifar10" ["mnist"]="letters")
declare -A CLASSES=(
    ["cifar10"]="0 1 2 3 4 5 6 7 8 9"
    ["mnist"]="0 1 2 3 4 5 6 7 8 9"
    ["cifar100"]="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89"
)

for lt in "${loss_tags[@]}"; do
    method="PARAMETRIC_${tag}_1000_standard_syntetic_fixed_embs_forget_${lt}"
    for dataset in cifar10 cifar100 mnist; do
        for class in ${CLASSES[$dataset]}; do
            model_dir="models/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best"
            [ -f "${model_dir}/checkpoint_best.pt" ] || { echo "Missing ${model_dir} — skipping"; continue; }
            for ckpt in 00 best; do
                [ "$ckpt" == "00" ] && [ ! -f "${model_dir}/checkpoint_epoch_00.pt" ] && continue
                if [ -f "results_NEAR/${experiment}_${dataset}_classes_to_remove_${class}/${method}/original_best/${ckpt}/test.csv" ]; then
                    continue
                fi
                echo "NEAR ${experiment} ${dataset}->${NEAR_DATASET[$dataset]} class ${class} ${method} ckpt ${ckpt}"
                python src/training/main_extractor.py -o results_NEAR --model-directory models -e ${experiment}_${dataset} -m ${experiment} -d datasets --dataset ${dataset} --extract-dataset ${NEAR_DATASET[$dataset]} --data-mode CR --classes-to-remove ${class} --version original_best --checkpoint ${ckpt} --unlearning-method ${method}
            done
        done
    done
done

echo "Run ended"

conda deactivate
