#!/usr/bin/env python3
"""
Analyses the ICU unlearning results.

Step 1: concatenates every unlearning_results_ICU.csv found
        recursively in the given folder, appending for
        efficiency. Writes <folder>/unlearning_results.csv

Step 2: extracts 'dataset' and 'num_classes_to_forget' from id_database.
        Groups by (experiment, dataset, version, steps, num_classes_to_forget)
        computing the mean and variance of the metrics over the different
        class configurations with the same num_classes_to_forget.
        Writes <folder>/unlearning_results_analyzed.csv  (with std)
               <folder>/unlearning_results_analyzed_means.csv (means only)

Usage:
    python src/analysis/analyze_ICU.py [folder]          # default: results_ICU
"""

import os
import shutil
import tempfile
import polars as pl
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.embs_fallback import add_embs_fallback_pl  # TEMPORARY: efficientnet nofreeze as embs
from tqdm import tqdm


def write_csv_safely(df: pl.DataFrame, dest_path: Path) -> None:
    """
    Writes first to a temporary file on the local disk (not on the results
    NFS) and then copies it to the destination with an atomic rename.

    Byte corruption (NUL holes, and even a lost first character in the
    first row) has been observed when writing directly
    to NFS with the polars CSV writer, probably due to an
    interaction between its multi-threaded writes and the NFS write-back.
    Writing locally and copying sequentially avoids the problem.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    local_tmp = Path(tmp_name)
    try:
        df.write_csv(local_tmp)
        remote_tmp = dest_path.with_name(dest_path.name + ".tmp")
        shutil.copyfile(local_tmp, remote_tmp)
        os.replace(remote_tmp, dest_path)
    finally:
        local_tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers to parse id_database
# ---------------------------------------------------------------------------

def extract_dataset(id_database: str) -> str:
    """
    Extracts the dataset name from id_database.
    E.g. cifar100_classes_to_remove_52_66_... -> cifar100
    """
    match = re.match(r"^(.+?)_classes_to_remove_", id_database)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract the dataset from: {id_database}")


def extract_num_classes_to_forget(id_database: str) -> int:
    """
    Counts how many classes are removed.
    E.g. cifar100_classes_to_remove_52_66_69_89_14_22_48_28_37_3_72_1_12_20_99_65_88_38_81_75 -> 20
    """
    match = re.search(r"classes_to_remove_([\d_]+)$", id_database)
    if match:
        return len(match.group(1).split("_"))
    raise ValueError(f"Could not extract num_classes from: {id_database}")


# ---------------------------------------------------------------------------
# Step 1: concatenate every CSV
# ---------------------------------------------------------------------------

def concatenate_csvs(results_dir: Path) -> Path:
    """
    Recursively finds every unlearning_results_ICU.csv under
    results_dir and concatenates them into results_dir/unlearning_results.csv.

    Uses pl.concat(how="diagonal") to handle CSVs with
    different sets of columns (e.g. with / without clustering metrics),
    filling the columns missing in each file with null.
    """
    source_name = "unlearning_results_ICU.csv"
    output_name = "unlearning_results.csv"
    output_path = results_dir / output_name

    # Skips the CSVs in folders whose own name contains the architecture
    # prefix (intermediate-level CSVs, not the per-method ones).
    known_arch_prefixes = ("vit_b_16", "resnet18", "resenet18", "efficientnet_v2_s")
    # Also skips archived runs under folders starting with "_"
    # (e.g. results_ICU/_old_incomplete/).
    csv_files = [
        f for f in sorted(results_dir.rglob(source_name))
        if not any(f.parent.name.startswith(p) for p in known_arch_prefixes)
        and not any(part.startswith("_") for part in f.relative_to(results_dir).parts[:-1])
    ]

    if not csv_files:
        print(f"No {source_name} files found")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files to concatenate.")

    dfs = []
    for csv_file in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pl.read_csv(csv_file, infer_schema_length=10000, null_values=["N/A", "nan", "NaN", ""])
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️  Could not read {csv_file}: {e}")

    if not dfs:
        print("No valid CSV file could be read.")
        sys.exit(1)

    # Build a unified schema: for each column, if any df has it as Float64/Int64,
    # cast all String occurrences of that column to Float64 before diagonal concat.
    col_types: dict[str, pl.DataType] = {}
    for df in dfs:
        for name, dtype in zip(df.columns, df.dtypes):
            if name not in col_types:
                col_types[name] = dtype
            elif col_types[name] == pl.String and dtype != pl.String:
                col_types[name] = dtype

    def harmonize(df: pl.DataFrame) -> pl.DataFrame:
        casts = []
        for name, target in col_types.items():
            if name in df.columns and df[name].dtype != target:
                casts.append(pl.col(name).cast(target, strict=False))
        return df.with_columns(casts) if casts else df

    dfs = [harmonize(df) for df in dfs]

    # diagonal: fills with null the columns missing in some file
    combined = pl.concat(dfs, how="diagonal")
    write_csv_safely(combined, output_path)
    print(f"Concatenated file saved to: {output_path}  ({len(combined)} rows)")
    return output_path


# ---------------------------------------------------------------------------
# Step 2: analysis - grouped mean and variance
# ---------------------------------------------------------------------------

def analyze(csv_path: Path, results_dir: Path) -> None:
    """
    1. Reads the concatenated CSV.
    2. Extracts dataset and num_classes_to_forget from id_database.
    3. For each (experiment, dataset, version, steps, num_classes_to_forget)
       computes the mean and standard deviation of the metrics over the
       different class configurations.
    4. Writes two CSVs:
         - unlearning_results_analyzed.csv         (with std)
         - unlearning_results_analyzed_means.csv   (means only)
    """
    df = add_embs_fallback_pl(pl.read_csv(csv_path))

    print(f"\nColumns: {df.columns}")
    print(f"Total rows: {len(df)}")

    # ---- Drop duplicated header rows that slipped in as data ----
    # (some source unlearning_results_ICU.csv files have the header
    # repeated in the middle of the file because they were written with several appends)
    n_before = len(df)
    df = df.filter(pl.col("id_database") != "id_database")
    if len(df) < n_before:
        print(f"⚠️  Dropped {n_before - len(df)} duplicated header rows found as data.")

    # ---- Extract dataset and num_classes_to_forget ----
    df = df.with_columns([
        pl.col("id_database")
          .map_elements(extract_dataset, return_dtype=pl.Utf8)
          .alias("dataset"),
        pl.col("id_database")
          .map_elements(extract_num_classes_to_forget, return_dtype=pl.Int64)
          .alias("num_classes_to_forget"),
    ])

    print(f"Datasets found: {df['dataset'].unique().sort().to_list()}")
    print(f"Numbers of classes to forget found: {df['num_classes_to_forget'].unique().sort().to_list()}")

    # ---- Original accuracy per (experiment, dataset, id_database, steps) ----
    original_rows = df.filter(
        (pl.col("version") == "") | (pl.col("version").is_null())
    )

    if len(original_rows) > 0:
        print(f"\n'original' rows (empty version): {len(original_rows)}")
        original_accuracy = (
            original_rows
            .select([
                "experiment", "dataset", "id_database", "steps",
                "accuracy_score_after",
            ])
            .rename({"accuracy_score_after": "accuracy_score_after_original"})
        )
    else:
        original_accuracy = None
        print("\nNo rows with an empty version (original) found.")

    # Keep the rows with a non-empty version (unlearning methods)
    df_methods = df.filter(
        (pl.col("version") != "") & (pl.col("version").is_not_null())
    )

    if len(df_methods) == 0:
        print("No unlearning-method rows (non-empty version).")
        return

    # ---- Join the original accuracy, if available ----
    if original_accuracy is not None:
        df_methods = df_methods.join(
            original_accuracy,
            on=["experiment", "dataset", "id_database", "steps"],
            how="left",
        )
        # Compute AUS
        df_methods = df_methods.with_columns([
            (
                (1 - (pl.col("accuracy_score_after_original") - pl.col("accuracy_score_after")))
                / (1 + pl.col("unlearning_accuracy_score"))
            ).alias("AUS")
        ])
    else:
        df_methods = df_methods.with_columns([
            pl.lit(None).cast(pl.Float64).alias("accuracy_score_after_original"),
            pl.lit(None).cast(pl.Float64).alias("AUS"),
        ])

    # ---- Find which versions have the maximum number of distinct configurations
    #      for each (experiment, dataset, version, num_classes_to_forget) ----
    config_counts = (
        df_methods
        .group_by(["experiment", "dataset", "version", "num_classes_to_forget"])
        .agg(pl.col("id_database").n_unique().alias("num_configs"))
    )

    max_configs = (
        config_counts
        .group_by(["experiment", "dataset", "num_classes_to_forget"])
        .agg(pl.col("num_configs").max().alias("max_configs"))
    )

    # Keep only the versions with the maximum number of configs
    config_counts_filtered = (
        config_counts
        .join(max_configs, on=["experiment", "dataset", "num_classes_to_forget"])
        .filter(pl.col("num_configs") == pl.col("max_configs"))
        .select(["experiment", "dataset", "version", "num_classes_to_forget"])
    )

    df_filtered = df_methods.join(
        config_counts_filtered,
        on=["experiment", "dataset", "version", "num_classes_to_forget"],
    )

    print(f"\nRows after filtering by max configs: {len(df_filtered)}")

    # ---- Metrics to aggregate: every numeric column that is not a key ----
    non_metric_cols = {
        "experiment", "dataset", "version", "steps", "num_classes_to_forget",
        "id_database", "id_experiment",
    }
    metrics_cols = [
        c for c in df_filtered.columns
        if c not in non_metric_cols and df_filtered[c].dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
    ]
    print(f"\nMetrics to aggregate ({len(metrics_cols)}): {metrics_cols}")

    agg_exprs = [pl.len().alias("num_averaged")]
    for col in metrics_cols:
        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_exprs.append(
            pl.col(col).std(ddof=0).fill_null(0.0).alias(f"{col}_std")
        )

    # ---- Group by (experiment, dataset, version, steps, num_classes_to_forget) ----
    results = (
        df_filtered
        .group_by(
            ["experiment", "dataset", "version", "steps", "num_classes_to_forget"],
            maintain_order=True,
        )
        .agg(agg_exprs)
        .sort(["experiment", "dataset", "version", "num_classes_to_forget", "steps"])
    )

    # Brier scores (only if the base columns are available)
    brier_exprs = []
    if "accuracy_score_after_mean" in results.columns and "maxprobability_mean" in results.columns:
        brier_exprs.append(
            ((pl.col("accuracy_score_after_mean") - pl.col("maxprobability_mean")) ** 2)
            .alias("Brier_score_retain")
        )
    if "unlearning_accuracy_score_mean" in results.columns and "maxprobability_forget_mean" in results.columns:
        brier_exprs.append(
            ((pl.col("unlearning_accuracy_score_mean") - pl.col("maxprobability_forget_mean")) ** 2)
            .alias("Brier_score_forget")
        )
    if brier_exprs:
        results = results.with_columns(brier_exprs)

    print(f"\nAggregated results: {len(results)} rows")
    print(results.head(20))

    # ---- Save ----
    out_full = results_dir / "unlearning_results_analyzed.csv"
    out_means = results_dir / "unlearning_results_analyzed_means.csv"

    write_csv_safely(results, out_full)
    print(f"\nResults saved to: {out_full}")

    mean_cols = [c for c in results.columns if not c.endswith("_std")]
    write_csv_safely(results.select(mean_cols), out_means)
    print(f"Results (means only) saved to: {out_means}")

    # ---- Top 3 by accuracy and by unlearning ----
    print("\n" + "=" * 80)
    print("TOP 3 VERSIONS PER (experiment, dataset, num_classes_to_forget, steps) — accuracy_score_after_mean")
    print("=" * 80)
    top3_acc = (
        results
        .sort(
            ["experiment", "dataset", "num_classes_to_forget", "steps", "accuracy_score_after_mean"],
            descending=[False, False, False, False, True],
        )
        .group_by(["experiment", "dataset", "num_classes_to_forget", "steps"])
        .head(3)
    )
    print(top3_acc.select([
        "experiment", "dataset", "version", "num_classes_to_forget",
        "steps", "num_averaged",
        "accuracy_score_after_mean", "accuracy_score_after_std",
    ]).head(30))

    print("\n" + "=" * 80)
    print("TOP 3 VERSIONS PER (experiment, dataset, num_classes_to_forget, steps) — unlearning_accuracy_score_mean (lower is better)")
    print("=" * 80)
    top3_unl = (
        results
        .sort(
            ["experiment", "dataset", "num_classes_to_forget", "steps", "unlearning_accuracy_score_mean"],
            descending=[False, False, False, False, False],
        )
        .group_by(["experiment", "dataset", "num_classes_to_forget", "steps"])
        .head(3)
    )
    print(top3_unl.select([
        "experiment", "dataset", "version", "num_classes_to_forget",
        "steps", "num_averaged",
        "unlearning_accuracy_score_mean", "unlearning_accuracy_score_std",
    ]).head(30))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results_ICU")
    ap.add_argument("--no-cluster", dest="cluster", action="store_false",
                    help="Do not include clustering metrics in the output")
    ap.set_defaults(cluster=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: '{results_dir}' is not a valid directory.")
        sys.exit(1)

    print(f"Results folder: {results_dir}\n")

    # Step 1 - concatenate
    concat_csv = concatenate_csvs(results_dir)

    # Step 2 - analyse
    analyze(concat_csv, results_dir)
