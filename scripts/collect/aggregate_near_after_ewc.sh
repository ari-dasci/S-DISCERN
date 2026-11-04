#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 10g
#SBATCH --job-name aggNEAR
#SBATCH --output=slrums/output_%A.out # STDOUT

# Rebuilds the "unknown" (near-OOD) metrics CSV so the newly extracted EWC runs
# land in it. Run this AFTER extract_missing_ewc_near.sh has finished.
#
# results/unlearning_results_single_embs_new_near.csv currently dates from June
# and predates several extractions, so --continue alone would keep the stale
# rows and only append the new ones. That is what we want here (the existing
# rows are valid), but if you ever need a clean rebuild just drop --continue
# and let it rewrite the file from scratch.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

# Same two-pass pattern as read_results.sh: first the baselines, then the
# methods (excluding the "/original" version, which is the pre-unlearning one).
python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single \
    --include-version RETRAIN embs \
    --results-file unlearning_results_single_embs_new_near --online --continue

python src/collect/process_results_auto_NEAR.py --base-dir results --classes-filter single \
    --exclude-version /original \
    --results-file unlearning_results_single_embs_new_near --online --continue

echo "Aggregation done. Checking EWC coverage per architecture:"
python - <<'EOF'
import pandas as pd
d = pd.read_csv("results/unlearning_results_single_embs_new_near.csv", usecols=["experiment", "version"])
e = d[d["version"].astype(str).str.startswith("EWC")]
if e.empty:
    print("  no EWC rows found")
else:
    print(e.groupby(["experiment", "version"]).size().to_string())
EOF

conda deactivate
