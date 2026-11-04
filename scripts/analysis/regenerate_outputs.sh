#!/bin/bash
#SBATCH -c 8
#SBATCH --job-name regen_out
#SBATCH --mem 32g
#SBATCH --gres=gpu:1
#SBATCH --output=slrums/output_%j.out # STDOUT

# Regenerate every summary / figure that shows the proposed method's name
# (now "DISCERN", src/utils/proposed_method.py), in dependency order:
#   1. aggregated means (analyze_aggregated.py, analyze_ICU.py)
#   2. selected_methods.csv (extract_methods.py)
#   3. boxplots (main + Gaussian ablation) and ICU figures (main + ablation, plot_figures.py)
#   4. Wilcoxon analyses + trend / reference plots (run_wilcoxon_all.sh),
#      raw-value export (export_raw_wilcox.sh), paper Wilcoxon figures
#   5. DISCERN vs NOISY tests (run_noisy_vs_distributional.sh + report)
#   6. ablations flujo / noisy (results/ablation/)
#   sbatch scripts/analysis/regenerate_outputs.sh      (or: bash scripts/analysis/regenerate_outputs.sh)
# A failing step is reported and the rest continue; summary at the end.

eval "$(conda shell.bash hook)"

conda activate unlearning
export PYTHONPATH='.'

failed=()
step () {   # step <name> <command...>
    local name=$1; shift
    echo; echo "########## ${name}"
    if "$@"; then echo "[ok] ${name}"; else echo "[FAILED] ${name}"; failed+=("$name"); fi
}

step "aggregated means"        python src/analysis/analyze_aggregated.py results/unlearning_results_single_all_new.csv results/unlearning_results_multiple_all_new.csv
step "ICU aggregate"           python src/analysis/analyze_ICU.py results_ICU
step "selected methods"        python src/plots/extract_methods.py
step "boxplots + ablation"     python src/plots/plot_boxplots.py --together
step "ICU figures + ablation"  python src/plots/plot_figures.py
step "Wilcoxon + trend plots"  bash scripts/analysis/run_wilcoxon_all.sh
step "Wilcoxon raw export"     bash scripts/analysis/export_raw_wilcox.sh
step "paper Wilcoxon figures"  python src/plots/plot_paper_wilcox_figures.py
step "reference agreement"     python src/plots/plot_reference_agreement_summary.py
step "DISCERN vs NOISY tests"  bash scripts/analysis/run_noisy_vs_distributional.sh
step "DISCERN vs NOISY report" bash scripts/analysis/export_noisy_vs_dist_report.sh
step "ablation flujo"          python src/analysis/ablation_compare.py --comparison flujo
step "ablation noisy"          python src/analysis/ablation_compare.py --comparison noisy

echo
if [ ${#failed[@]} -eq 0 ]; then echo "All steps ok"; else echo "FAILED steps: ${failed[*]}"; fi
echo "Run ended"

conda deactivate
