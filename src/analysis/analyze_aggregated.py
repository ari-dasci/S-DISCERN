import polars as pl
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.embs_fallback import add_embs_fallback_pl  # TEMPORARY: efficientnet nofreeze as embs

def extract_classes_to_remove(id_database: str) -> int:
    """
    Extracts the number X from an id_database of the form dataset_classes_to_remove_X
    """
    match = re.search(r'classes_to_remove_(\d+)$', id_database)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract classes_to_remove from: {id_database}")


def extract_dataset(id_database: str) -> str:
    """
    Extracts the dataset from an id_database of the form dataset_classes_to_remove_X
    Examples: cifar10, cifar100, mnist
    """
    # Whatever comes before _classes_to_remove
    match = re.search(r'^([a-z0-9]+)_classes_to_remove', id_database)
    
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract the dataset from: {id_database}")


def extract_num_classes_removed(id_database: str) -> int:
    """
    Extracts from id_database how many classes are being removed.
    Format: dataset_classes_to_remove_X_Y_Z...
    Returns how many numbers follow 'classes_to_remove_'
    E.g. cifar100_classes_to_remove_31_64_70_20_29 → 5
    """
    match = re.search(r'classes_to_remove_([\d_]+)$', id_database)
    if match:
        classes_str = match.group(1)
        return len(classes_str.split('_'))
    raise ValueError(f"Could not extract the classes from: {id_database}")


def is_single_class(id_database: str) -> bool:
    """
    Tells whether an id_database is single-class (a single number after classes_to_remove_).
    E.g. cifar10_classes_to_remove_9 → True
        cifar100_classes_to_remove_31_64_70 → False
    """
    match = re.search(r'classes_to_remove_(\d+)$', id_database)
    return match is not None


def analyze_unlearning_results(csv_path: str) -> pl.DataFrame:
    """
    Reads an unlearning results CSV and computes:
    - For each (experiment, version): picks epoch "best", or "00" if there is no best
    - Extracts the classes_to_remove number (X) from each id_database
    - Groups only the versions that have the same maximum X
    - Computes the mean and variance of the metrics
    """
    
    # Read the CSV
    df = add_embs_fallback_pl(pl.read_csv(csv_path))
    
    print("Available columns:", df.columns)
    print(f"Total rows: {len(df)}")
    
    # Keep only single-class entries (a single number after classes_to_remove_)
    df = df.filter(
        pl.col("id_database").map_elements(is_single_class, return_dtype=pl.Boolean)
    )
    print(f"Single-class rows: {len(df)}\n")
    
    if len(df) == 0:
        print("No single-class entries in this CSV.")
        return pl.DataFrame()
    
    # Extract the classes_to_remove number X and the dataset
    df = df.with_columns([
        pl.col("id_database").map_elements(extract_classes_to_remove, return_dtype=pl.Int64).alias("classes_to_remove"),
        pl.col("id_database").map_elements(extract_dataset, return_dtype=pl.Utf8).alias("dataset")
    ])
    
    # For each (experiment, version, dataset), find epoch "best" or "00"
    best_epochs = (
        df
        .with_columns([
            pl.when(pl.col("epoch") == "best")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_best")
        ])
        .sort(["experiment", "version", "dataset", "classes_to_remove", "is_best"], descending=[False, False, False, False, True])
        .group_by(["experiment", "version", "dataset", "classes_to_remove"], maintain_order=True)
        .first()
    )
    
    print("Selected epochs per (experiment, version, dataset):")
    print(best_epochs.select(["experiment", "version", "dataset", "epoch"]))
    print()
    
    # For each (experiment, version, dataset), find the maximum classes_to_remove
    max_classes = (
        df
        .group_by(["experiment", "version", "dataset"])
        .agg([
            pl.col("classes_to_remove").max().alias("max_classes_to_remove"),
            pl.col("classes_to_remove").n_unique().alias("num_different_X")
        ])
    )
    
    print("Maximum classes_to_remove per (experiment, version, dataset):")
    print(max_classes)
    print()
    
    # For each (experiment, dataset), find the maximum num_different_X
    max_x_count = (
        max_classes
        .group_by(["experiment", "dataset", "version"])
        .agg([
            pl.col("num_different_X").max().alias("max_num_different_X")
        ])
    )
    
    print("Maximum num_different_X per (experiment, dataset, version):")
    print(max_x_count)
    print()
    
    # Keep only the (experiment, version, dataset) with the maximum num_different_X
    max_classes_filtered = (
        max_classes
        .join(max_x_count, on=["experiment", "dataset", "version"])
        .filter(pl.col("num_different_X") == pl.col("max_num_different_X"))
    )
    
    print("Versions used for the mean and variance (maximum num_different_X):")
    print(max_classes_filtered)
    print()
    
    # Filter the data: keep only epoch "best" or "00" AND the maximum num_different_X
    # First pick best/00 for each (experiment, version, dataset)
    df_best_epoch = (
        df
        .with_columns([
            pl.when(pl.col("epoch") == "best")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_best")
        ])
        .sort(["experiment", "version", "dataset", "classes_to_remove", "is_best"], descending=[False, False, False, False, True])
        .group_by(["experiment", "version", "dataset", "classes_to_remove"], maintain_order=True)
        .first()
    )

    print(df_best_epoch.select(["experiment", "version", "dataset", "classes_to_remove", "epoch"]))
    
    # Then join with the ones that have the maximum num_different_X
    df_filtered = (
        df_best_epoch
        .join(
            max_classes_filtered.select(["experiment", "version", "dataset"]),
            on=["experiment", "version", "dataset"]
        )
    )
    
    print("Filtered data (only epoch best/00 and maximum num_different_X per experiment-dataset):")
    print(df_filtered.select(["experiment", "version", "dataset", "epoch"]))
    print()
    
    # accuracy_score_after of the original version FOR EACH classes_to_remove
    original_accuracy = (
        df
        .filter(pl.col("version") == "original")
        .select(["experiment", "dataset", "classes_to_remove", "epoch", "accuracy_score_after"])
        .with_columns([
            pl.when(pl.col("epoch") == "best")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_best")
        ])
        .sort(["experiment", "dataset", "classes_to_remove", "is_best"], descending=[False, False, False, True])
        .group_by(["experiment", "dataset", "classes_to_remove"], maintain_order=True)
        .first()
        .select(["experiment", "dataset", "classes_to_remove", "accuracy_score_after"])
        .rename({"accuracy_score_after": "accuracy_score_after_original"})
    )
    
    # Join the filtered data with the original accuracy (by experiment, dataset AND classes_to_remove)
    df_filtered_with_original = (
        df_filtered
        .join(
            original_accuracy,
            on=["experiment", "dataset", "classes_to_remove"],
            how="left"
        )
    )
    
    # Compute AUS = (1-(accuracy_score_after_original-accuracy_score_after))/(1 + unlearning_accuracy_score)
    df_filtered_with_original = (
        df_filtered_with_original
        .with_columns([
            ((1 - (pl.col("accuracy_score_after_original") - pl.col("accuracy_score_after"))) 
             / (1 + pl.col("unlearning_accuracy_score")))
            .alias("AUS")
        ])
    )
    
    # Mean and variance of the metrics for each (experiment, version)
    metrics_cols = [
        "accuracy_score_after",
        "unlearning_accuracy_score",
        "maxprobability",
        "maxprobability_forget",
        "AUS",
        "cohesion_forget_l2",
        "distance_inter_class_l2",
        "cohesion_forget_cos",
        "distance_inter_class_cos"
    ]
    
    # Mean and variance expressions
    agg_expressions = [
        pl.len().alias("num_averaged")
    ]
    
    for col in metrics_cols:
        agg_expressions.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_expressions.append(pl.col(col).std(ddof=0).fill_null(0.0).alias(f"{col}_std"))
    
    # Group and aggregate by experiment, version and dataset
    results = (
        df_filtered_with_original
        .group_by(["experiment", "version", "dataset"], maintain_order=True)
        .agg(agg_expressions)
    )

    # Brier scores as metrics derived from the means
    results = results.with_columns([
        ((pl.col("accuracy_score_after_mean") - pl.col("maxprobability_mean")) ** 2)
        .alias("Brier_score_retain"),
        ((pl.col("unlearning_accuracy_score_mean") - pl.col("maxprobability_forget_mean")) ** 2)
        .alias("Brier_score_forget"),
    ])

    print("Final results (mean and variance of the metrics):")
    print(results)
    print()
    
    # Top 3 versions per dataset and experiment by accuracy_score_after_mean
    print("\n" + "="*80)
    print("TOP 3 VERSIONS PER DATASET AND EXPERIMENT (accuracy_score_after_mean)")
    print("="*80)
    top3_accuracy = (
        results
        .sort(["experiment", "dataset", "accuracy_score_after_mean"], descending=[False, False, True])
        .group_by(["experiment", "dataset"])
        .head(3)
    )
    print(top3_accuracy.select(["experiment", "dataset", "version", "num_averaged", "accuracy_score_after_mean", "accuracy_score_after_std"]))
    print()
    
    # Top 3 versions per dataset and experiment by unlearning_accuracy_score_mean
    print("\n" + "="*80)
    print("TOP 3 VERSIONS PER DATASET AND EXPERIMENT (unlearning_accuracy_score_mean)")
    print("="*80)
    top3_unlearning = (
        results
        .sort(["experiment", "dataset", "unlearning_accuracy_score_mean"], descending=[False, False, False])
        .group_by(["experiment", "dataset"])
        .head(3)
    )
    print(top3_unlearning.select(["experiment", "dataset", "version", "num_averaged", "unlearning_accuracy_score_mean", "unlearning_accuracy_score_std"]))
    print()
    
    
    # Save the results with and without standard deviation
    output_path = Path(csv_path).stem + "_analyzed.csv"
    output_path_means = Path(csv_path).stem + "_analyzed_means.csv"
    print(output_path)
    results.write_csv(output_path)
    print(f"Results saved to: {output_path}")

    # Save the means only (no std columns)
    mean_cols = [col for col in results.columns if not col.endswith('_std')]
    results.select(mean_cols).write_csv(output_path_means)
    print(f"Results (means only) saved to: {output_path_means}")

    return results


def analyze_multiclass_unlearning_results(csv_path: str) -> pl.DataFrame:
    """
    Analyses unlearning results where several classes are removed at once.
    id_database has the form dataset_classes_to_remove_X_Y_Z...
    
    - Extracts num_classes_removed (how many classes each run removes)
    - For each (experiment, version, id_database), picks epoch "best" or the first one
    - Groups by (experiment, version, dataset, num_classes_removed)
    - Keeps only the versions with the same maximum number of distinct configurations
    - Computes the mean and variance of the metrics
    - Stores num_averaged (how many runs were averaged) and num_classes_removed
    """
    
    # Read the CSV
    df = add_embs_fallback_pl(pl.read_csv(csv_path))
    
    print("Available columns:", df.columns)
    print(f"Total rows: {len(df)}")
    
    # Keep only multi-class entries (more than one number after classes_to_remove_)
    df = df.filter(
        ~pl.col("id_database").map_elements(is_single_class, return_dtype=pl.Boolean)
    )
    print(f"Multi-class rows: {len(df)}\n")
    
    if len(df) == 0:
        print("No multi-class entries in this CSV.")
        return pl.DataFrame()
    
    # Extract dataset and num_classes_removed
    df = df.with_columns([
        pl.col("id_database").map_elements(extract_dataset, return_dtype=pl.Utf8).alias("dataset"),
        pl.col("id_database").map_elements(extract_num_classes_removed, return_dtype=pl.Int64).alias("num_classes_removed")
    ])
    
    print("Unique values of num_classes_removed:", df["num_classes_removed"].unique().sort().to_list())
    print()
    
    # For each (experiment, version, id_database), find epoch "best" or the first one
    df_best_epoch = (
        df
        .with_columns([
            pl.when(pl.col("epoch") == "best")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_best")
        ])
        .sort(["experiment", "version", "dataset", "id_database", "is_best"],
              descending=[False, False, False, False, True])
        .group_by(["experiment", "version", "dataset", "id_database", "num_classes_removed"], maintain_order=True)
        .first()
    )
    
    print("Selected epochs (multi-class):")
    print(df_best_epoch.select(["experiment", "version", "dataset", "id_database", "num_classes_removed", "epoch"]))
    print()
    
    # For each (experiment, version, dataset, num_classes_removed), count the distinct id_databases
    config_counts = (
        df_best_epoch
        .group_by(["experiment", "version", "dataset", "num_classes_removed"])
        .agg([
            pl.col("id_database").n_unique().alias("num_different_configs")
        ])
    )
    
    print("Configs per (experiment, version, dataset, num_classes_removed):")
    print(config_counts)
    print()
    
    # For each (experiment, dataset, num_classes_removed), find the max num_different_configs
    max_config_counts = (
        config_counts
        .group_by(["experiment", "dataset", "num_classes_removed"])
        .agg([
            pl.col("num_different_configs").max().alias("max_num_different_configs")
        ])
    )
    
    # Keep only the versions with the max num_different_configs
    config_counts_filtered = (
        config_counts
        .join(max_config_counts, on=["experiment", "dataset", "num_classes_removed"])
        .filter(pl.col("num_different_configs") == pl.col("max_num_different_configs"))
    )
    
    print("Filtered versions (maximum num_different_configs):")
    print(config_counts_filtered)
    print()
    
    # Filter the data
    df_filtered = (
        df_best_epoch
        .join(
            config_counts_filtered.select(["experiment", "version", "dataset", "num_classes_removed"]),
            on=["experiment", "version", "dataset", "num_classes_removed"]
        )
    )
    
    print("Filtered data (multi-class):")
    print(df_filtered.select(["experiment", "version", "dataset", "num_classes_removed", "epoch"]))
    print()
    
    # Original accuracy (if available)
    original_rows = df.filter(pl.col("version") == "original")
    
    if len(original_rows) > 0:
        original_accuracy = (
            original_rows
            .select(["experiment", "dataset", "epoch", "accuracy_score_after"])
            .with_columns([
                pl.when(pl.col("epoch") == "best")
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("is_best")
            ])
            .sort(["experiment", "dataset", "is_best"], descending=[False, False, True])
            .group_by(["experiment", "dataset"], maintain_order=True)
            .first()
            .select(["experiment", "dataset", "accuracy_score_after"])
            .rename({"accuracy_score_after": "accuracy_score_after_original"})
        )
        
        df_filtered_with_original = (
            df_filtered
            .join(original_accuracy, on=["experiment", "dataset"], how="left")
        )
        
        df_filtered_with_original = (
            df_filtered_with_original
            .with_columns([
                ((1 - (pl.col("accuracy_score_after_original") - pl.col("accuracy_score_after"))) 
                 / (1 + pl.col("unlearning_accuracy_score")))
                .alias("AUS")
            ])
        )
    else:
        print("No 'original' version found, AUS will not be computed.")
        df_filtered_with_original = (
            df_filtered
            .with_columns([
                pl.lit(None).cast(pl.Float64).alias("accuracy_score_after_original"),
                pl.lit(None).cast(pl.Float64).alias("AUS")
            ])
        )
    
    # Metrics
    metrics_cols = [
        "accuracy_score_after",
        "unlearning_accuracy_score",
        "maxprobability",
        "maxprobability_forget",
        "AUS",
        "cohesion_forget_l2",
        "distance_inter_class_l2",
        "cohesion_forget_cos",
        "distance_inter_class_cos"
    ]
    
    # Mean and variance expressions
    agg_expressions = [
        pl.len().alias("num_averaged"),
    ]
    
    for col in metrics_cols:
        agg_expressions.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_expressions.append(pl.col(col).std(ddof=0).fill_null(0.0).alias(f"{col}_std"))
    
    # Group by experiment, version, dataset, num_classes_removed
    results = (
        df_filtered_with_original
        .group_by(["experiment", "version", "dataset", "num_classes_removed"], maintain_order=True)
        .agg(agg_expressions)
    )

    # Brier scores as metrics derived from the means
    results = results.with_columns([
        ((pl.col("accuracy_score_after_mean") - pl.col("maxprobability_mean")) ** 2)
        .alias("Brier_score_retain"),
        ((pl.col("unlearning_accuracy_score_mean") - pl.col("maxprobability_forget_mean")) ** 2)
        .alias("Brier_score_forget"),
    ])

    print("Final results (multi-class - mean and variance):")
    print(results)
    print()
    
    # Top 3 versions per dataset, experiment and num_classes_removed by accuracy_score_after_mean
    print("\n" + "="*80)
    print("TOP 3 VERSIONS PER DATASET, EXPERIMENT AND NUM_CLASSES_REMOVED (accuracy_score_after_mean)")
    print("="*80)
    top3_accuracy = (
        results
        .sort(["experiment", "dataset", "num_classes_removed", "accuracy_score_after_mean"],
              descending=[False, False, False, True])
        .group_by(["experiment", "dataset", "num_classes_removed"])
        .head(3)
    )
    print(top3_accuracy.select(["experiment", "dataset", "version", "num_classes_removed", "num_averaged", "accuracy_score_after_mean", "accuracy_score_after_std"]))
    print()
    
    # Top 3 versions per dataset, experiment and num_classes_removed by unlearning_accuracy_score_mean
    print("\n" + "="*80)
    print("TOP 3 VERSIONS PER DATASET, EXPERIMENT AND NUM_CLASSES_REMOVED (unlearning_accuracy_score_mean)")
    print("="*80)
    top3_unlearning = (
        results
        .sort(["experiment", "dataset", "num_classes_removed", "unlearning_accuracy_score_mean"],
              descending=[False, False, False, False])
        .group_by(["experiment", "dataset", "num_classes_removed"])
        .head(3)
    )
    print(top3_unlearning.select(["experiment", "dataset", "version", "num_classes_removed", "num_averaged", "unlearning_accuracy_score_mean", "unlearning_accuracy_score_std"]))
    print()
    
    # Save the results
    output_path = Path(csv_path).stem + "_multiclass_analyzed.csv"
    output_path_means = Path(csv_path).stem + "_multiclass_analyzed_means.csv"
    results.write_csv(output_path)
    print(f"Results saved to: {output_path}")

    # Save the means only (no std columns)
    mean_cols = [col for col in results.columns if not col.endswith('_std')]
    results.select(mean_cols).write_csv(output_path_means)
    print(f"Results (means only) saved to: {output_path_means}")

    return results


if __name__ == "__main__":
    # CSVs to analyse (they may contain both single-class and multi-class rows)
    csv_files = [
        "results/unlearning_results_single_all_new.csv",
        "results/unlearning_results_multiple_all_new.csv",
    ]
    
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    
    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"File not found: {csv_file}")
            continue
        
        print("\n" + "#"*80)
        print(f"Processing: {csv_file}")
        print("#" * 80 + "\n")
        
        # Single-class analysis (filters automatically)
        print(f"=== SINGLE-CLASS analysis of: {csv_file} ===\n")
        results_single = analyze_unlearning_results(csv_file)
        
        print("\n" + "-"*80 + "\n")
        
        # Multi-class analysis (filters automatically)
        print(f"=== MULTI-CLASS analysis of: {csv_file} ===\n")
        results_multi = analyze_multiclass_unlearning_results(csv_file)
