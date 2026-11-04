# DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning

Code of the paper *DISCERN: Distributional Synthetic Replay for Retain-Free Incremental
Machine Unlearning* (N. Segura-Kunsagi, J. L. Suárez, I. Sevillano-García, F. Herrera).

Class *machine unlearning* experiments on image classifiers
(resnet18, vit_b_16, swin_t, efficientnet_v2_s × cifar10, cifar100, mnist) in three
scenarios: forgetting **one class**, forgetting **several classes** at once, and **ICU**
(*Incremental Class Unlearning*, forgetting classes one after another). The proposed
method (DISCERN) corresponds to the `PARAMETRIC_cv5x3_tbn_m0.05_tests_*` runs
(`src/utils/proposed_method.py`).

## Environment

Python 3.10 on Linux with CUDA 12.1 (the PyTorch wheels come from the PyTorch
index declared at the top of `requirements.txt`).

```bash
conda env create -f env.yml        # or, in any Python 3.10 environment: pip install -r requirements.txt
conda activate unlearning
export PYTHONPATH='.'              # every script is run from the repository root
# ...
conda deactivate
```

The launchers in `scripts/` activate the environment by name (`conda activate unlearning`)
and use relative paths only. Their `#SBATCH` headers (partition, GPUs, time) must be
adapted to the local cluster; they can also be run with `bash`, setting
`SLURM_ARRAY_TASK_ID` by hand.

## Reproducing DISCERN

The PARAMETRIC launchers default to the configuration reported in the paper
(`VARIANT=tiebreak_norm`, `PIPELINE=tests`, `REPEATS=3`, i.e. tag `cv5x3_tbn_m0.05_tests`):

```bash
sbatch scripts/unlearning/unlearn_parametric.sh          # one class
sbatch scripts/unlearning/unlearn_multi_parametric.sh    # several classes (cifar100)
sbatch scripts/unlearning/unlearn_ICU_parametric.sh      # ICU
```

The other variants (`VARIANT=tiebreak|tiebreak_energy|ksnorm|...`, `PIPELINE=cv`) are the
ones compared in the ablations, e.g.
`sbatch --export=ALL,VARIANT=tiebreak_norm,PIPELINE=cv scripts/unlearning/unlearn_parametric.sh`.

## Layout

```
src/
  config/       configuration (hyper-parameters) of each unlearning method
  methods/      method implementations (PARAMETRIC, DISTRIBUTIONAL, NOISY, SSD, EWC, ...)
  models/       models, transforms and optimizer
  utils/        data loading, datasets, clustering metrics, proposed method
  cited_hyperparams.yaml

  training/     dataset download, training, embedding extraction and class selection
  unlearning/   unlearning runs (single / multiple / ICU)
  collect/      collection of results and metrics into CSV files
  analysis/     statistical analysis (Wilcoxon, ablations, hyper-parameters, distributions)
  plots/        figures (boxplots, ICU figures, ablations, Wilcoxon trends)
scripts/        SLURM launchers (they can also be run with bash), grouped by stage
tests/          tests of DistributionalMethod (distances, log-likelihood, Kent), ParametricMethod and distribution_adapter
```

Every script is launched from the repository root, e.g.
`sbatch scripts/unlearning/unlearn_parametric.sh` or `bash scripts/analysis/regenerate_outputs.sh`.

## Pipeline

| Step | Scripts | What it does |
|---|---|---|
| 0. Data and models | `scripts/setup/` | Downloads the datasets, picks the classes to forget for each seed and trains the original models |
| 1. Unlearning | `scripts/unlearning/unlearn2.sh`, `unlearn3.sh`, `unlearn_missing_single.sh`, `train_missing_dist_noisy.sh` | Single-class forgetting (comparison methods, SSD / EWC grid, RETRAIN) |
| | `unlearn_multi_{5,10,20,50}.sh`, `unlearn_missing_multi.sh` | Multi-class forgetting (cifar100) |
| | `unlearn_ICU_20.sh`, `unlearn_ICU_20_RETRAIN.sh`, `unlearn_missing_ICU.sh` | ICU |
| | `unlearn_parametric.sh`, `unlearn_multi_parametric.sh`, `unlearn_ICU_parametric.sh` | Proposed method (PARAMETRIC) in the three scenarios |
| | `relaunch_embs_{single,multi,ICU}.sh` (+ `relaunch_embs_common.sh`) | "embs" runs (classifier + last block) of efficientnet_v2_s and SSD on swin_t, plus the *cv* variant of the selection-flow ablation |
| 2. Extraction | `scripts/extraction/` | Embeddings / logits of every model into `results/`, and of the near-OOD dataset into `results_NEAR/` |
| 3. Collection | `scripts/collect/` | Aggregates `results/` into CSV files (forget / retain / near-OOD) and adds logit metrics to the ICU CSV files |
| 4. Analysis and figures | `scripts/analysis/regenerate_outputs.sh` | Aggregated means, `selected_methods.csv`, boxplots, ICU figures, Wilcoxon (`run_wilcoxon_all.sh`, `export_raw_wilcox.sh`) and ablations |
| | `run_ablation_compare.sh`, `run_ablation_flujo.sh` | Ablations of the proposed method → `results/ablation/` |
| | `export_chosen_distributions.sh` | Distribution family chosen for each class → `results/chosen_distributions.csv` |
| 5. Hyper-parameters | `scripts/hparam/` | Study of the margin and the number of synthetic samples → `results/ablation/hparam/` |
| 6. Run times | `scripts/timing/benchmark_times.sh` | Controlled timing benchmark: one class per (architecture, dataset), every method on the same node |

Main programs in `src/` (each one lists its options with `--help`):

| Program | What it does |
|---|---|
| `src/training/main_training.py` | Trains the original model |
| `src/unlearning/main_unlearning.py` | Unlearning of one or several classes (`--classes-to-remove`, `--unlearning-method`) |
| `src/unlearning/main_unlearning_ICU.py` | ICU: forgets the classes sequentially |
| `src/training/main_extractor.py` | Extracts the embeddings / logits of each model |
| `src/collect/process_results_auto*.py` | Aggregates the extracted results into a CSV file |
| `src/analysis/analyze_aggregated.py`, `analyze_ICU.py` | Means per method and scenario |
| `src/plots/extract_methods.py` | Picks the compared methods (and the best SSD / EWC configuration) → `selected_methods.csv` |
| `src/plots/plot_boxplots.py`, `plot_figures.py` | Boxplots, ICU figures and Gaussian ablation figures → `plots/` |
| `src/analysis/analyze_wilcoxon_unlearning*.py` | Wilcoxon tests against `original` and `RETRAIN` (forget / retain / unknown) → `results/wilcox/` |
| `src/analysis/ablation_compare.py`, `ablation_flujo.py` | Paired ablations of the proposed method |
| `src/analysis/export_chosen_distributions.py` | Reads the PARAMETRIC fit caches and exports the distribution chosen for each class |

Example:

```bash
python src/unlearning/main_unlearning.py -o models -e resnet18_cifar10 -m resnet18 \
    -d datasets --dataset cifar10 --data-mode CR --seed 42 \
    --classes-to-remove 3 --version original --epoch best --unlearning-method RETRAIN
```

## License

GNU General Public License v3.0 (see `LICENSE`).
