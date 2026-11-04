# Shared definitions for the PARAMETRIC hyper-parameter mini study (single class
# removal only), sourced by train_hparam_parametric.sh / extract_hparam_parametric.sh.
# Same study as hparam_study_common.sh (DISTRIBUTIONAL):
#   margin           : notop, 1000 synthetic samples, margin in {0, 5, 20}   (10 = main runs)
#   synthetic_samples: {100, 500, 2000, 5000, 10000} for notop (margin 10) and
#                      corrected_confidence_penalty                            (1000 = main runs)
# for one PARAMETRIC configuration (VARIANT / PIPELINE, default tiebreak_norm + tests = DISCERN).
# The per-class fits do not depend on margin / number of samples, so every
# config reuses the fit cache of the main single-class runs of that configuration.

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

BASE_TAG="standard_syntetic_fixed_embs_forget"
COMMON_FLAGS="--unlearning-method PARAMETRIC --cv-folds 5 ${variant_args} --fixed-syntetic --emb-unfreeze embs"

ARCHS=(resnet18 efficientnet_v2_s swin_t vit_b_16)
DATASETS=(cifar10 cifar100 mnist)
declare -A SINGLE_CLASSES
SINGLE_CLASSES["cifar10"]="0 1 2 3 4 5 6 7 8 9"
SINGLE_CLASSES["mnist"]="0 1 2 3 4 5 6 7 8 9"
SINGLE_CLASSES["cifar100"]="4 1 70 10 51 39 5 24 3 76 60 15 34 26 11 29 80 96 8 89"

CONFIGS=()
_notop() {  # $1 = n_synth, $2 = margin, $3 = study
    CONFIGS+=("${3}|notop|${1}|${2}|PARAMETRIC_${tag}_${1}_${BASE_TAG}_notop_margin_${2}|${COMMON_FLAGS} --loss-forget notop --syntetic-samples ${1} --notop-margin ${2}")
}
_cp() {  # $1 = n_synth
    CONFIGS+=("synthetic_samples|corrected_confidence_penalty|${1}|-|PARAMETRIC_${tag}_${1}_${BASE_TAG}_corrected_confidence_penalty|${COMMON_FLAGS} --loss-forget corrected_confidence_penalty --syntetic-samples ${1} --notop-margin 10")
}
for m in 0.0 5.0 20.0; do _notop 1000 "$m" margin; done
for n in 100 500 2000 5000 10000; do _notop "$n" 10.0 synthetic_samples; done
for n in 100 500 2000 5000 10000; do _cp "$n"; done

N_CONFIGS=${#CONFIGS[@]}
N_ARCHS=${#ARCHS[@]}
N_TASKS=$((N_CONFIGS * N_ARCHS))   # one array task per (config, architecture): 13 x 4 = 52

select_task() {  # $1 = SLURM_ARRAY_TASK_ID
    local cfg_idx=$(( $1 / N_ARCHS ))
    experiment=${ARCHS[$(( $1 % N_ARCHS ))]}
    IFS='|' read -r study method n_synth margin folder flags <<< "${CONFIGS[$cfg_idx]}"
}

wait_cache() {  # $1 experiment $2 dataset : never compute the fits here, wait for the main runs
    local cache="standard_cov_parametric_${tag}/${1}_${2}_original_epoch_best_dist_infos.pt" waited=0
    while [ ! -f "$cache" ]; do
        if [ "$waited" -ge 2880 ]; then echo "Cache $cache missing after 48 h — skipping ${1} ${2}"; return 1; fi
        echo "Waiting for $cache (written by the main single-class run of ${tag})..."
        sleep 900; waited=$((waited + 15))
    done
}
