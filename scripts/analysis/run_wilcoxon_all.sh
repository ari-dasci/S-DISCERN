#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 16g
#SBATCH --job-name wilcoxon
#SBATCH --output=slrums/output_%A.out # STDOUT

# Full Wilcoxon analysis chain. Run AFTER aggregate_near_after_ewc.sh, so the
# "unknown" block includes the newly extracted EWC runs.
#
# CPU only (pandas/statsmodels/matplotlib), no GPU needed: it can also be run
# directly with `bash scripts/analysis/run_wilcoxon_all.sh` on the login node.
#
# Two passes are needed per case, and they must not overwrite each other:
#
#   A) canonical  --distributionalout : the general trend, the by-architecture
#      test and the 7-method consensus are computed over the comparison
#      methods only, with the proposed method excluded from the pool.
#
#   B) reference  (no --distributionalout) : the same deltas but keeping
#      DISTRIBUTIONAL/NOISY, used only to report whether the proposed methods
#      follow the majority. Every other output is sent to a scratch directory
#      so pass A's files survive.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

# set -u only AFTER the activation: conda's activate script reads unset
# variables (PS1) and aborts under it, leaving python3 = the base one.
set -u
if [ "${CONDA_DEFAULT_ENV:-}" != "unlearning" ]; then
    echo "conda env not active (python3 = $(command -v python3)) — aborting"; exit 1
fi

SCRATCH=$(mktemp -d /tmp/wilcox_scratch.XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT

# selected_methods.csv drives which method (and which SSD/EWC hyper-parameter)
# enters the analysis. Only meaningful if the *_analyzed_means.csv inputs were
# regenerated; harmless otherwise.
echo "=== 0. Method selection ==="
python3 src/plots/extract_methods.py

echo
echo "=== 1. Canonical pass (comparison methods only) ==="
for script in src/analysis/analyze_wilcoxon_unlearning.py \
              src/analysis/analyze_wilcoxon_unlearning_retain.py \
              src/analysis/analyze_wilcoxon_unlearning_unknown3.py; do
    echo "--- $script"
    python3 "$script" --distributionalout --goodones 2>&1 \
        | grep -vE "warnings.warn|UserWarning|ConvergenceWarning"
done

echo
echo "=== 2. Reference pass (DISTRIBUTIONAL / NOISY vs the majority) ==="
run_reference () {  # $1 = script, $2 = output suffix, $3... = extra args
    local script="$1"; local suffix="$2"; shift 2
    python3 "$script" --goodones "$@" \
        --output-reference-trends "results/wilcox/reference_method_trends${suffix}.csv" \
        --output-summary "$SCRATCH/sum${suffix}.csv" \
        --output-values "$SCRATCH/val${suffix}.csv" \
        --output-summary-distance "$SCRATCH/dist${suffix}.csv" \
        --output-summary-by-architecture "$SCRATCH/arch${suffix}.csv" \
        --output-summary-distance-by-architecture "$SCRATCH/archd${suffix}.csv" \
        --output-ks-summary "$SCRATCH/ks${suffix}.csv" \
        --output-best-worst "$SCRATCH/bw${suffix}.csv" \
        --output-concentration-consensus "$SCRATCH/cons${suffix}.csv" \
        --output-ledoit-individual "$SCRATCH/li${suffix}.csv" \
        --output-ledoit-statistical "$SCRATCH/ls${suffix}.csv" 2>&1 \
        | grep -E "reference method trends|\[warn\]|\[info\]"
}

run_reference src/analysis/analyze_wilcoxon_unlearning.py ""
run_reference src/analysis/analyze_wilcoxon_unlearning_retain.py "_retain"
run_reference src/analysis/analyze_wilcoxon_unlearning_unknown3.py "_unknown3"

# Multi-class split: the only place where NOISY and the confidence-penalty
# variants exist. --split appends "_multi" to the path given here on its own,
# so the base name is passed without it.
echo "--- multi split (NOISY available only here)"
run_reference src/analysis/analyze_wilcoxon_unlearning.py "" --split multi

echo
echo "=== 3. Figures ==="
python3 src/plots/plot_wilcoxon_trends.py 2>&1 | grep -E "Saved|warn|==="
python3 src/plots/plot_wilcoxon_trends_by_architecture.py 2>&1 | grep -E "Saved|==="
python3 src/plots/plot_reference_methods.py 2>&1 | grep -E "Saved|absent|==="

echo
echo "=== 4. Coverage check: methods entering each case ==="
python3 - <<'EOF'
import pandas as pd

for case, path in [("forget", "results/wilcox/wilcoxon_unpaired_values.csv"),
                   ("retain", "results/wilcox/wilcoxon_unpaired_values_retain.csv"),
                   ("unknown", "results/wilcox/wilcoxon_unpaired_values_unknown3.csv")]:
    d = pd.read_csv(path, usecols=["experiment", "version"])
    d["family"] = d["version"].astype(str).str.split("_").str[0]
    per_arch = d.groupby("experiment")["family"].nunique()
    print(f"\n{case}: method families per architecture")
    print(per_arch.to_string())
    missing_ewc = [a for a, g in d.groupby("experiment") if "EWC" not in set(g["family"])]
    if missing_ewc:
        print(f"  ⚠️  EWC still missing in: {missing_ewc}")
    else:
        print("  ✓ EWC present in all architectures")
EOF

echo
echo "Wilcoxon chain finished."

set +u
conda deactivate
