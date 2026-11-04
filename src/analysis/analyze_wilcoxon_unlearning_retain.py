#!/usr/bin/env python3
"""Analyze retain-class clustering metrics against baselines with Wilcoxon tests.

Reads a CSV produced by process_results_auto_retain.py which contains:
  cohesion_retain_l2, distance_inter_retain_l2,
  cohesion_retain_cos, distance_inter_retain_cos.

Workflow:
1) Read CSV and keep only rows with epoch == "best".
2) Drop the maxprobability_retain column.
3) Parse id_database into dataset name and class_to_remove.
4) Keep only dataset in {mnist, cifar10, cifar100}.
5) For each (experiment, dataset, class_to_remove), compute deltas:
   - L2 metrics : version - baseline
   - Cos metrics: baseline - version
   against two baselines: "original" and "RETRAIN/original_best".
6) Build four delta groups per (experiment, dataset):
   - l2_vs_original, l2_vs_retrain, cos_vs_original, cos_vs_retrain
7) Remove clear outliers using IQR rule.
8) Apply unpaired Wilcoxon rank-sum test against 0
   with alternative='less' (checks if deltas tend to be <= 0).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.embs_fallback import add_embs_fallback  # noqa: E402  (TEMPORARY: efficientnet nofreeze as embs)
from src.utils.proposed_method import (  # noqa: E402  (proposed method = PARAMETRIC)
    PROPOSED_FAMILY, PROPOSED_NOTOP, PROPOSED_CONF, PROPOSED_PREFIX,
    PROPOSED_NOTOP_LABEL, PROPOSED_CONF_LABEL,
)
from scipy.stats import ks_2samp, levene, wilcoxon
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats


L2_METRICS = ["cohesion_retain_l2", "distance_inter_retain_l2"]
COS_METRICS = ["cohesion_retain_cos", "distance_inter_retain_cos"]
COHESION_L2_METRIC = ["cohesion_retain_l2"]
COHESION_COS_METRIC = ["cohesion_retain_cos"]
COHESION_METRICS = COHESION_L2_METRIC + COHESION_COS_METRIC
DISTANCE_L2_METRIC = ["distance_inter_retain_l2"]
DISTANCE_COS_METRIC = ["distance_inter_retain_cos"]
DISTANCE_METRICS = DISTANCE_L2_METRIC + DISTANCE_COS_METRIC
BASELINES = ["original", "RETRAIN/original_best"]
# Proposed method (src/utils/proposed_method.py; previously DISTRIBUTIONAL)
PROPOSED_KEEP_VERSION = PROPOSED_NOTOP + "/original_best"
NOISY_KEEP_VERSION = (
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop/"
    "original_best"
)
SPECIAL_PREFIXES = (PROPOSED_FAMILY, "DISTRIBUTIONAL", "NOISY")
SPECIAL_KEEP_VERSIONS = [PROPOSED_KEEP_VERSION, NOISY_KEEP_VERSION]

NOISY_CONFIDENCE_VERSION = (
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty/"
    "original_best"
)
PROPOSED_CONFIDENCE_VERSION = PROPOSED_CONF + "/original_best"
# Proposed methods (and their gaussian-noise counterpart) evaluated against the
# majority trend of the comparison methods rather than counted inside it.
REFERENCE_VERSIONS = {
    PROPOSED_NOTOP_LABEL: PROPOSED_KEEP_VERSION,          # "DISCERN notop"
    PROPOSED_CONF_LABEL: PROPOSED_CONFIDENCE_VERSION,     # "DISCERN confidence"
    "NOISY notop": NOISY_KEEP_VERSION,
    "NOISY confidence": NOISY_CONFIDENCE_VERSION,
}
# Same labels, without "/original_best" -- these are exactly the strings in
# src/plots/extract_methods.py's FIXED_METHODS list, used to recognize a reference
# method's row in selected_methods.csv regardless of which architecture it
# is (see load_reference_version_overrides: some architectures fall back to
# a differently-named variant, e.g. swin_t's DISTRIBUTIONAL notop resolves
# to a "_margin_10.0" checkpoint because the plain run never converged).
REFERENCE_METHOD_ORIGINALS = {
    label: version.rsplit("/", 1)[0] for label, version in REFERENCE_VERSIONS.items()
}


SINGLE_INPUT_DEFAULT = "results/unlearning_results_single_embs_new_retain.csv"
MULTI_INPUT_DEFAULT = "results/unlearning_results_multiple_embs_new_retain.csv"
GOODONES_SINGLE_DEFAULT = "results/unlearning_results_single_embs_new.csv"
GOODONES_MULTI_DEFAULT = "results/unlearning_results_multiple_embs_new.csv"

ARCHITECTURE_FAMILY = {
    "resnet18": "conv",
    "efficientnet_v2_s": "conv",
    "swin_t": "transformer",
    "vit_b_16": "transformer",
}

# Method families counted in the concentrate/disperse consensus. Hyper-parameter
# variants are pooled into one entry per family (SSD's 12 lambda/alpha settings
# count once, not twelve), so the tally is over methods, not over runs.
CONSENSUS_METHOD_FAMILIES = [
    "BADTEACHER",
    "BOUNDARYEXPAND",
    "BOUNDARYSHRINK",
    "DELETE",
    "EWC",
    "RANDOMLABEL",
    "SSD",
]


@dataclass
class TestConfig:
    iqr_multiplier: float = 1.5
    alpha: float = 0.05
    min_samples: int = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wilcoxon rank-sum analysis for retain-class clustering deltas."
    )
    parser.add_argument(
        "--input",
        default=SINGLE_INPUT_DEFAULT,
        help="Input CSV path (single-class-removal, output of process_results_auto_retain.py).",
    )
    parser.add_argument(
        "--input-multi",
        default=MULTI_INPUT_DEFAULT,
        help="Input CSV path (multiple-class-removal, output of process_results_auto_retain.py).",
    )
    parser.add_argument(
        "--split",
        choices=["single", "multi", "both"],
        default="single",
        help=(
            "Which class-removal results to analyze: 'single' (--input), "
            "'multi' (--input-multi), or 'both' (concatenated). Non-'single' "
            "runs get '_multi'/'_both' appended to all default output "
            "filenames so they never overwrite the single-run outputs."
        ),
    )
    parser.add_argument(
        "--output-summary",
        default="results/wilcox/wilcoxon_unpaired_summary_intra_retain.csv",
        help="Output CSV with grouped Wilcoxon results for cohesion (intra-class) metrics.",
    )
    parser.add_argument(
        "--output-values",
        default="results/wilcox/wilcoxon_unpaired_values_retain.csv",
        help="Output CSV with raw and cleaned delta values.",
    )
    parser.add_argument(
        "--output-summary-distance",
        default="results/wilcox/wilcoxon_unpaired_inter_retain.csv",
        help="Output CSV with grouped Wilcoxon results for inter-class distance metrics.",
    )
    parser.add_argument(
        "--output-summary-by-architecture",
        default="results/wilcox/wilcoxon_unpaired_summary_intra_retain_by_architecture.csv",
        help=(
            "Output CSV: same test as --output-summary but pooled by "
            "architecture_family (conv: resnet18, efficientnet_v2_s; "
            "transformer: swin_t, vit_b_16) x dataset x group_name instead "
            "of per experiment."
        ),
    )
    parser.add_argument(
        "--output-summary-distance-by-architecture",
        default="results/wilcox/wilcoxon_unpaired_inter_retain_by_architecture.csv",
        help=(
            "Output CSV: same as --output-summary-by-architecture but for "
            "inter-class distance metrics."
        ),
    )
    parser.add_argument(
        "--output-ks-summary",
        default="results/wilcox/ks_original_retrain_vs_methods_summary_retain.csv",
        help="Output CSV with KS test summary (baseline vs methods).",
    )
    parser.add_argument(
        "--output-best-worst",
        default="results/wilcox/best_worst_concentration_methods_retain.csv",
        help=(
            "Output CSV with the most- and least-concentrating method per "
            "(experiment, dataset, metric, group_name) after IQR outlier removal."
        ),
    )
    parser.add_argument(
        "--output-concentration-consensus",
        default="results/wilcox/concentration_consensus_retain.csv",
        help=(
            "Output CSV with the count of method families that concentrate "
            "vs disperse per (experiment, dataset, metric, group_name), as a "
            "consensus check on the global pooled/mixed-model trend."
        ),
    )
    parser.add_argument(
        "--output-ledoit-individual",
        default="results/wilcox/ledoit_notop_individual_comparison_retain.csv",
        help="Output CSV with per-row rank of PROPOSED_KEEP_VERSION vs all methods.",
    )
    parser.add_argument(
        "--output-ledoit-statistical",
        default="results/wilcox/ledoit_notop_statistical_comparison_retain.csv",
        help=(
            "Output CSV with paired Wilcoxon signed-rank test: "
            "PROPOSED_KEEP_VERSION vs most/least concentrating method."
        ),
    )
    parser.add_argument(
        "--output-distributional",
        default="results/wilcox/distributional_ledoit_comparison_retain.csv",
        help=f"Output CSV comparing the proposed method's variants ({PROPOSED_PREFIX}*) vs baselines.",
    )
    parser.add_argument(
        "--output-distributional-var",
        default="results/wilcox/distributional_ledoit_homoscedasticity_retain.csv",
        help="Output CSV with variance and Levene test for LEDOIT variants.",
    )
    parser.add_argument(
        "--output-homoscedasticity",
        default="results/wilcox/homoscedasticity_summary_retain.csv",
        help="Output CSV with Levene test: all methods pooled vs each baseline.",
    )
    parser.add_argument(
        "--selected-methods",
        default=SELECTED_METHODS_DEFAULT,
        help=(
            "CSV produced by src/plots/extract_methods.py listing the comparison method "
            "chosen per (architecture, dataset): the fixed baselines plus the "
            "best SSD/EWC hyper-parameter. Only those versions are analyzed. "
            "Pass an empty string to keep every method."
        ),
    )
    parser.add_argument(
        "--output-reference-trends",
        default="results/wilcox/reference_method_trends_retain.csv",
        help=(
            "Output CSV with the direction of each proposed/reference method "
            "(PARAMETRIC and NOISY, notop and confidence variants) next to "
            "the majority direction of the comparison methods."
        ),
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier removal (default: 1.5).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for hypothesis tests.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="Minimum cleaned samples required to run each test.",
    )
    parser.add_argument(
        "--distributionalout",
        action="store_true",
        help="Exclude all methods whose version starts with 'PARAMETRIC', 'DISTRIBUTIONAL' or 'NOISY'.",
    )
    parser.add_argument(
        "--distributionalnotop",
        action="store_true",
        help=(
            "Exclude all 'PARAMETRIC*'/'DISTRIBUTIONAL*'/'NOISY*' versions except the "
            "notop variants of the proposed method and of NOISY: "
            f"{PROPOSED_NOTOP}/original_best and "
            "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop/"
            "original_best."
        ),
    )
    parser.add_argument(
        "--goodones",
        action="store_true",
        help=(
            "Exclude rows where accuracy_score_after < 0.6 AND "
            "unlearning_accuracy_score > 0.2 (read from --goodones-input)."
        ),
    )
    parser.add_argument(
        "--goodones-input",
        default=GOODONES_SINGLE_DEFAULT,
        help=(
            "Single-class-removal CSV with accuracy_score_after and "
            "unlearning_accuracy_score used by --goodones to identify bad "
            f"runs (default: {GOODONES_SINGLE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--goodones-input-multi",
        default=GOODONES_MULTI_DEFAULT,
        help=(
            "Multiple-class-removal counterpart of --goodones-input, used "
            f"when --split is 'multi' or 'both' (default: {GOODONES_MULTI_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--only-distributional",
        action="store_true",
        help=(
            "Keep only rows whose version starts with 'PARAMETRIC', 'DISTRIBUTIONAL' or 'NOISY' "
            "(plus baselines required for delta computation). "
            "Mutually exclusive with --distributionalout and --distributionalnotop."
        ),
    )
    parser.add_argument(
        "--export-raw-dir",
        default="",
        help=(
            "If set, write the raw per-class deltas (baselines_values_retain.csv and "
            "distributional_values_retain.csv, see export_raw_values.py) to this "
            "directory and exit before running any test."
        ),
    )
    return parser.parse_args()


def parse_database_and_class(id_database: str) -> Optional[Dict[str, object]]:
    match = re.match(r"^(.+)_classes_to_remove_([\d_]+)$", str(id_database))
    if not match:
        return None
    dataset = match.group(1)
    if dataset not in {"mnist", "cifar10", "cifar100"}:
        return None
    class_ids = [int(c) for c in match.group(2).split("_") if c != ""]
    if not class_ids:
        return None
    class_to_remove = class_ids[0] if len(class_ids) == 1 else match.group(2)
    return {"dataset": dataset, "class_to_remove": class_to_remove}


def count_classes_removed(class_to_remove: object) -> int:
    if isinstance(class_to_remove, str) and "_" in class_to_remove:
        return len(class_to_remove.split("_"))
    return 1


def method_family(version: object) -> str:
    """Method name without its hyper-parameter suffix, e.g.
    'SSD_MAS_lambda_1.0_alpha_5.0_embs/original_best' -> 'SSD'."""
    return str(version).split("_")[0]


def select_checkpoint_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each run's 'best' checkpoint, falling back to epoch '00' when the
    run has none.

    One-shot methods (SSD, FISHERFORGETTING, SFU) never write a 'best'
    checkpoint because they do not train over epochs, so a plain
    epoch == 'best' filter drops them from the analysis entirely.
    """
    epoch = df["epoch"].astype(str)
    is_best = epoch == "best"
    has_best = is_best.groupby(
        [df["experiment"], df["id_database"], df["version"]]
    ).transform("any")
    return df[is_best | (~has_best & (epoch == "00"))].copy()


SELECTED_METHODS_DEFAULT = "selected_methods.csv"


def load_selected_methods(path: str) -> Optional[Dict[tuple, set]]:
    """(architecture, dataset) -> set of base versions chosen by
    src/plots/extract_methods.py: the fixed comparison methods plus the best SSD/EWC
    hyper-parameter for that group (picked by harmonic mean of accuracy and
    unlearning score). Returns None when the file is missing, which keeps
    every method in the analysis."""
    if not path or not os.path.exists(path):
        print(f"[warn] selected-methods file not found, keeping every method: {path}")
        return None
    sel = pd.read_csv(path)
    sel = sel[sel["selection"].astype(str) != "MISSING"]
    chosen: Dict[tuple, set] = {}
    for _, row in sel.iterrows():
        key = (str(row["architecture"]), str(row["dataset"]))
        # "A -> B" labels mean the family fell back to variant B.
        resolved = str(row["method"]).split(" -> ")[-1].strip()
        chosen.setdefault(key, set()).add(resolved)
    return chosen


def filter_selected_methods(df: pd.DataFrame, chosen: Dict[tuple, set]) -> pd.DataFrame:
    """Keep only the versions selected for each (experiment, dataset), plus
    the two baselines needed to compute deltas."""
    base = df["version"].astype(str).str.split("/").str[0]
    keys = zip(df["experiment"].astype(str), df["dataset"].astype(str))
    keep = pd.Series(
        [b in chosen.get(k, set()) for k, b in zip(keys, base)], index=df.index
    )
    return df[df["version"].isin(BASELINES) | keep].copy()



def apply_split_suffix(path: str, split: str) -> str:
    if split == "single":
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{split}{ext}"


def read_optional_csv(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[warn] {label} input not found yet, treating as empty: {path}")
        return pd.DataFrame()
    d = add_embs_fallback(pd.read_csv(path))
    if d.empty:
        print(f"[warn] {label} input has no rows yet: {path}")
    return d


def drop_outliers_iqr(values: pd.Series, multiplier: float) -> pd.Series:
    if values.empty:
        return values
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return values
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return values[(values >= lower) & (values <= upper)]


def build_delta_table(
    df: pd.DataFrame,
    l2_metrics: List[str],
    cos_metrics: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_cols = l2_metrics + cos_metrics
    group_cols = ["experiment", "dataset", "class_to_remove"]

    df_compact = (
        df.groupby(group_cols + ["version"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )

    for key, g in df_compact.groupby(group_cols, dropna=False):
        experiment, dataset, cls = key
        by_version = {r["version"]: r for _, r in g.iterrows()}

        if "original" not in by_version or "RETRAIN/original_best" not in by_version:
            continue

        original_row = by_version["original"]
        retrain_row = by_version["RETRAIN/original_best"]

        for version, row in by_version.items():
            if version in BASELINES:
                continue

            for metric in l2_metrics:
                rows.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "class_to_remove": cls,
                        "version": version,
                        "metric": metric,
                        "group_name": "l2_vs_original",
                        "delta": row[metric] - original_row[metric],
                    }
                )
                rows.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "class_to_remove": cls,
                        "version": version,
                        "metric": metric,
                        "group_name": "l2_vs_retrain",
                        "delta": row[metric] - retrain_row[metric],
                    }
                )

            for metric in cos_metrics:
                rows.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "class_to_remove": cls,
                        "version": version,
                        "metric": metric,
                        "group_name": "cos_vs_original",
                        "delta": original_row[metric] - row[metric],
                    }
                )
                rows.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "class_to_remove": cls,
                        "version": version,
                        "metric": metric,
                        "group_name": "cos_vs_retrain",
                        "delta": retrain_row[metric] - row[metric],
                    }
                )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["n_classes_removed"] = result["class_to_remove"].map(count_classes_removed)
    return result


def run_group_tests(
    values_df: pd.DataFrame,
    config: TestConfig,
    group_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    keys = group_keys or ["experiment", "dataset", "group_name"]

    for key, g in values_df.groupby(keys, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        raw_values = g["delta"].dropna().astype(float)
        clean_values = drop_outliers_iqr(raw_values, config.iqr_multiplier)

        n_raw = int(len(raw_values))
        n_clean = int(len(clean_values))
        median_clean = float(np.median(clean_values)) if n_clean > 0 else np.nan
        mean_clean = float(np.mean(clean_values)) if n_clean > 0 else np.nan
        std_clean = float(np.std(clean_values, ddof=1)) if n_clean > 1 else np.nan

        for alternative in ["greater", "less"]:
            stat = np.nan
            p_value = np.nan
            h0_rejected = False
            ran_test = False

            if n_clean >= config.min_samples:
                test = wilcoxon(clean_values.to_numpy(), alternative=alternative)
                stat = float(test.statistic)
                p_value = float(test.pvalue)
                h0_rejected = bool(p_value < config.alpha)
                ran_test = True

            results.append(
                {
                    **dict(zip(keys, key_values)),
                    "alternative": alternative,
                    "n_raw": n_raw,
                    "n_clean": n_clean,
                    "n_outliers_removed": n_raw - n_clean,
                    "median_clean": median_clean,
                    "mean_clean": mean_clean,
                    "std_clean": std_clean,
                    "mannwhitney_u": stat,
                    "p_value": p_value,
                    "alpha": config.alpha,
                    "reject_h0": h0_rejected,
                    "test_run": ran_test,
                }
            )

    return pd.DataFrame(results)


def _assign_trend(reject_greater: bool, reject_less: bool, median: float) -> int:
    """+1 disperse / -1 concentrate / 0 no significant trend, same rule as
    src/plots/plot_wilcoxon_trends.py's heatmap coloring."""
    if reject_greater and not reject_less:
        return 1
    if reject_less and not reject_greater:
        return -1
    if reject_greater and reject_less:
        return 1 if median >= 0 else -1
    return 0


def compute_family_agreement(
    experiment_summary_df: pd.DataFrame, family_map: Dict[str, str]
) -> pd.DataFrame:
    """Per (architecture_family, dataset, group_name): whether the member
    experiments individually agree on direction *before* being pooled into
    the by-architecture test.

    A pooled family result can look significant even when its members
    actually disagree (one concentrates, the other disperses) or when only
    one member drives it (the other shows no trend) — this flags those
    cases instead of letting the pooled p-value alone speak for the whole
    architecture family.
    """
    df = experiment_summary_df.copy()
    df["architecture_family"] = df["experiment"].map(family_map)
    df = df.dropna(subset=["architecture_family"])
    if df.empty:
        return pd.DataFrame()

    idx = ["architecture_family", "dataset", "group_name", "experiment"]
    gr = df[df["alternative"] == "greater"].set_index(idx)["reject_h0"]
    ls = df[df["alternative"] == "less"].set_index(idx)["reject_h0"]
    med = df[df["alternative"] == "less"].set_index(idx)["median_clean"]

    combined = pd.DataFrame({"reject_g": gr, "reject_l": ls, "median": med}).reset_index()
    combined["member_trend"] = combined.apply(
        lambda r: _assign_trend(bool(r["reject_g"]), bool(r["reject_l"]), r["median"]),
        axis=1,
    )
    trend_label = {1: "disperse", -1: "concentrate", 0: "no_trend"}
    combined["member_trend_label"] = combined["member_trend"].map(trend_label)

    results: List[Dict[str, object]] = []
    for key, g in combined.groupby(
        ["architecture_family", "dataset", "group_name"], dropna=False
    ):
        family, dataset, group_name = key
        n_members = len(g)
        n_concentrate = int((g["member_trend"] == -1).sum())
        n_disperse = int((g["member_trend"] == 1).sum())
        n_no_trend = int((g["member_trend"] == 0).sum())

        if n_concentrate > 0 and n_disperse > 0:
            agreement = "disagree_opposite"
        elif n_no_trend > 0 and (n_concentrate > 0 or n_disperse > 0):
            agreement = "one_member_only"
        elif n_concentrate == n_members and n_members > 0:
            agreement = "agree_concentrate"
        elif n_disperse == n_members and n_members > 0:
            agreement = "agree_disperse"
        else:
            agreement = "no_member_significant"

        member_detail = ";".join(
            f"{row.experiment}:{row.member_trend_label}" for row in g.itertuples()
        )

        results.append(
            {
                "architecture_family": family,
                "dataset": dataset,
                "group_name": group_name,
                "n_members": n_members,
                "n_members_concentrate": n_concentrate,
                "n_members_disperse": n_disperse,
                "n_members_no_trend": n_no_trend,
                "member_agreement": agreement,
                "member_detail": member_detail,
            }
        )

    return pd.DataFrame(results)


def run_mixed_model_analysis(values_df: pd.DataFrame, config: TestConfig) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    keys = ["experiment", "dataset", "group_name"]

    for key, g in values_df.groupby(keys, dropna=False):
        experiment, dataset, group_name = key

        raw_values = g["delta"].dropna().astype(float)
        clean_idx = drop_outliers_iqr(raw_values, config.iqr_multiplier).index
        clean_g = g.loc[clean_idx].copy()
        clean_g["delta"] = clean_g["delta"].astype(float)

        stat = np.nan
        p_value = np.nan
        intercept = np.nan
        icc = np.nan
        h0_rejected = False
        ran_test = False
        note = ""

        n_groups = clean_g["class_to_remove"].nunique()
        n_clean = len(clean_g)

        if n_clean >= config.min_samples and n_groups >= 3:
            try:
                model = smf.mixedlm(
                    "delta ~ 1",
                    clean_g,
                    groups=clean_g["class_to_remove"],
                )
                fit = model.fit(method="lbfgs", disp=False)

                intercept = float(fit.fe_params["Intercept"])
                t_stat = float(fit.tvalues["Intercept"])
                df_residual = n_groups - 1
                p_value = float(scipy_stats.t.cdf(t_stat, df=df_residual))

                var_random = float(fit.cov_re.iloc[0, 0])
                var_resid = float(fit.scale)
                icc = var_random / (var_random + var_resid) if (var_random + var_resid) > 0 else np.nan

                if icc == 0.0 or abs(float(fit.tvalues["Intercept"])) < 1e-6:
                    raise ValueError("degenerate_random_effect")

                stat = t_stat
                h0_rejected = bool(p_value < config.alpha)
                ran_test = True

            except Exception as e:
                if len(clean_g) >= config.min_samples:
                    wx = wilcoxon(clean_g["delta"].values, alternative="less")
                    p_value = float(wx.pvalue)
                    stat = float(wx.statistic)
                    intercept = float(clean_g["delta"].mean())
                    h0_rejected = bool(p_value < config.alpha)
                    icc = 0.0
                    ran_test = True
                    note = f"fallback_wilcoxon|{e}"
                else:
                    note = f"insufficient_samples|{e}"
        else:
            note = f"too_few_groups: {n_groups}" if n_groups < 3 else f"too_few_samples: {n_clean}"

        results.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "group_name": group_name,
                "n_raw": int(len(raw_values)),
                "n_clean": n_clean,
                "n_groups_class": n_groups,
                "n_outliers_removed": int(len(raw_values) - n_clean),
                "median_clean": float(np.median(clean_g["delta"])) if n_clean > 0 else np.nan,
                "mean_clean": float(np.mean(clean_g["delta"])) if n_clean > 0 else np.nan,
                "std_clean": float(np.std(clean_g["delta"], ddof=1)) if n_clean > 1 else np.nan,
                "intercept_estimate": intercept,
                "t_statistic": stat,
                "p_value_less": p_value,
                "icc_class": icc,
                "alpha": config.alpha,
                "reject_h0": h0_rejected,
                "test_run": ran_test,
                "note": note,
            }
        )

    return pd.DataFrame(results)


def run_ks_tests(source_df: pd.DataFrame, config: TestConfig) -> pd.DataFrame:
    """KS 2-sample tests: baseline cohesion distribution vs all methods pooled."""
    metric_cols = COHESION_METRICS
    group_cols = ["experiment", "dataset", "class_to_remove"]

    compact = (
        source_df.groupby(group_cols + ["version"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
    )

    results: List[Dict[str, object]] = []

    for (experiment, dataset), g in compact.groupby(["experiment", "dataset"], dropna=False):
        methods = g[~g["version"].isin(BASELINES)].copy()

        for metric in metric_cols:
            method_raw = methods[metric].dropna().astype(float)
            method_clean = drop_outliers_iqr(method_raw, config.iqr_multiplier)

            for baseline in BASELINES:
                baseline_values = g[g["version"] == baseline][metric].dropna().astype(float)
                baseline_clean = drop_outliers_iqr(baseline_values, config.iqr_multiplier)

                ks_stat = np.nan
                p_value = np.nan
                reject_h0 = False
                test_run = False

                if len(method_clean) >= config.min_samples and len(baseline_clean) >= config.min_samples:
                    test = ks_2samp(
                        baseline_clean.to_numpy(),
                        method_clean.to_numpy(),
                        alternative="greater",
                        method="auto",
                    )
                    ks_stat = float(test.statistic)
                    p_value = float(test.pvalue)
                    reject_h0 = bool(p_value < config.alpha)
                    test_run = True

                results.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": baseline,
                        "n_baseline_raw": int(len(baseline_values)),
                        "n_baseline_clean": int(len(baseline_clean)),
                        "n_baseline_outliers_removed": int(len(baseline_values) - len(baseline_clean)),
                        "n_methods_raw": int(len(method_raw)),
                        "n_methods_clean": int(len(method_clean)),
                        "n_methods_outliers_removed": int(len(method_raw) - len(method_clean)),
                        "median_baseline_clean": float(np.median(baseline_clean)) if len(baseline_clean) > 0 else np.nan,
                        "median_methods_clean": float(np.median(method_clean)) if len(method_clean) > 0 else np.nan,
                        "ks_statistic": ks_stat,
                        "p_value_two_sided": p_value,
                        "alpha": config.alpha,
                        "reject_h0": reject_h0,
                        "test_run": test_run,
                    }
                )

    return pd.DataFrame(results)


def run_homoscedasticity_tests(source_df: pd.DataFrame, config: TestConfig) -> pd.DataFrame:
    """Levene test: all methods pooled vs each baseline, per (experiment, dataset, metric)."""
    all_metrics = L2_METRICS + COS_METRICS
    group_cols = ["experiment", "dataset", "class_to_remove"]

    compact = (
        source_df.groupby(group_cols + ["version"], as_index=False)[all_metrics]
        .mean(numeric_only=True)
    )

    results: List[Dict[str, object]] = []

    for (experiment, dataset), g in compact.groupby(["experiment", "dataset"], dropna=False):
        methods = g[~g["version"].isin(BASELINES)].copy()

        for metric in all_metrics:
            method_raw = methods[metric].dropna().astype(float)
            method_clean = drop_outliers_iqr(method_raw, config.iqr_multiplier)

            for baseline in BASELINES:
                baseline_values = g[g["version"] == baseline][metric].dropna().astype(float)
                baseline_clean = drop_outliers_iqr(baseline_values, config.iqr_multiplier)

                lev_stat = np.nan
                lev_p = np.nan
                homoscedastic = np.nan
                test_run = False

                var_methods = float(np.var(method_clean, ddof=1)) if len(method_clean) >= 2 else np.nan
                var_baseline = float(np.var(baseline_clean, ddof=1)) if len(baseline_clean) >= 2 else np.nan

                if len(method_clean) >= config.min_samples and len(baseline_clean) >= config.min_samples:
                    stat, p = levene(method_clean.to_numpy(), baseline_clean.to_numpy(), center="median")
                    lev_stat = float(stat)
                    lev_p = float(p)
                    homoscedastic = bool(lev_p >= config.alpha)
                    test_run = True

                results.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": baseline,
                        "n_methods_clean": int(len(method_clean)),
                        "n_baseline_clean": int(len(baseline_clean)),
                        "var_methods": var_methods,
                        "var_baseline": var_baseline,
                        "std_methods": float(np.std(method_clean, ddof=1)) if len(method_clean) >= 2 else np.nan,
                        "std_baseline": float(np.std(baseline_clean, ddof=1)) if len(baseline_clean) >= 2 else np.nan,
                        "var_ratio_methods_over_baseline": var_methods / var_baseline
                        if (not np.isnan(var_methods) and not np.isnan(var_baseline) and var_baseline > 0)
                        else np.nan,
                        "levene_statistic": lev_stat,
                        "levene_p_value": lev_p,
                        "alpha": config.alpha,
                        "homoscedastic": homoscedastic,
                        "test_run": test_run,
                    }
                )

    return pd.DataFrame(results)


def compute_best_worst_methods(
    delta_df: pd.DataFrame, config: TestConfig
) -> pd.DataFrame:
    """Per (experiment, dataset, metric, group_name): method with most/least
    negative mean delta after pooled IQR-outlier removal."""
    results: List[Dict[str, object]] = []

    keys = ["experiment", "dataset", "metric", "group_name"]
    for key, g in delta_df.groupby(keys, dropna=False):
        experiment, dataset, metric, group_name = key

        all_deltas = g["delta"].dropna().astype(float)
        clean_idx = drop_outliers_iqr(all_deltas, config.iqr_multiplier).index
        g_clean = g.loc[clean_idx]
        g_clean = g_clean[~g_clean["version"].astype(str).str.startswith(SPECIAL_PREFIXES)]

        version_means: Dict[str, float] = {}
        version_medians: Dict[str, float] = {}
        version_n: Dict[str, int] = {}

        for version, vg in g_clean.groupby("version", dropna=False):
            vals = vg["delta"].astype(float)
            if len(vals) == 0:
                continue
            version_means[version] = float(np.mean(vals))
            version_medians[version] = float(np.median(vals))
            version_n[version] = int(len(vals))

        if not version_means:
            continue

        best_v = min(version_means, key=version_means.__getitem__)
        worst_v = max(version_means, key=version_means.__getitem__)

        results.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "metric": metric,
                "group_name": group_name,
                "most_concentrating_method": best_v,
                "most_concentrating_mean_delta": version_means[best_v],
                "most_concentrating_median_delta": version_medians[best_v],
                "most_concentrating_n_clean": version_n[best_v],
                "least_concentrating_method": worst_v,
                "least_concentrating_mean_delta": version_means[worst_v],
                "least_concentrating_median_delta": version_medians[worst_v],
                "least_concentrating_n_clean": version_n[worst_v],
            }
        )

    return pd.DataFrame(results)


def compute_concentration_consensus(
    delta_df: pd.DataFrame,
    config: TestConfig,
    families: Optional[List[str]] = None,
    reference_versions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per (experiment, dataset, metric, group_name): how many method families
    concentrate (mean delta < 0) vs disperse (mean delta > 0) after pooled
    IQR-outlier removal. Hyper-parameter variants are pooled per family, so
    each method counts once. `reference_versions` are reported with their own
    direction but never counted."""
    counted = families if families is not None else CONSENSUS_METHOD_FAMILIES
    references = reference_versions or []
    results: List[Dict[str, object]] = []

    keys = ["experiment", "dataset", "metric", "group_name"]
    for key, g in delta_df.groupby(keys, dropna=False):
        experiment, dataset, metric, group_name = key

        # The majority is defined by the comparison methods only: the IQR filter
        # runs over their deltas alone (same pool as the canonical pass), so the
        # proposed/reference methods never shift the bounds nor get discarded
        # for being far from the baselines.
        g = g.assign(method_family=g["version"].map(method_family))
        counted_deltas = g.loc[g["method_family"].isin(counted), "delta"].dropna().astype(float)
        g_counted = g.loc[drop_outliers_iqr(counted_deltas, config.iqr_multiplier).index]

        family_means = (
            g_counted
            .groupby("method_family")["delta"]
            .mean()
        )
        if family_means.empty:
            continue

        concentrating = sorted(f for f, m in family_means.items() if m < 0)
        dispersing = sorted(f for f, m in family_means.items() if m > 0)
        n_tie = int((family_means == 0).sum())

        n_total = int(len(family_means))
        n_concentrate = len(concentrating)
        n_disperse = len(dispersing)
        pct_concentrate = n_concentrate / n_total if n_total > 0 else np.nan

        dominant = (
            "concentrate" if n_concentrate > n_disperse
            else "disperse" if n_disperse > n_concentrate
            else "tie"
        )

        reference_bits: List[str] = []
        reference_agreement: List[str] = []
        for reference in references:
            own = g.loc[g["version"] == reference, "delta"].dropna().astype(float)
            vals = drop_outliers_iqr(own, config.iqr_multiplier)
            name = method_family(reference)
            if vals.empty:
                reference_bits.append(f"{name}:absent")
                reference_agreement.append(f"{name}:absent")
                continue
            mean_delta = float(vals.mean())
            label = (
                "concentrate" if mean_delta < 0
                else "disperse" if mean_delta > 0
                else "tie"
            )
            reference_bits.append(f"{name}:{label}({mean_delta:+.4f})")
            reference_agreement.append(
                f"{name}:{'yes' if label == dominant else 'no'}"
            )

        results.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "metric": metric,
                "group_name": group_name,
                "n_methods": n_total,
                "n_concentrate": n_concentrate,
                "n_disperse": n_disperse,
                "n_tie": n_tie,
                "pct_concentrate": round(pct_concentrate, 4) if not np.isnan(pct_concentrate) else np.nan,
                "dominant": dominant,
                "concentrating_methods": ";".join(concentrating),
                "dispersing_methods": ";".join(dispersing),
                "reference_directions": ";".join(reference_bits),
                "reference_agrees_dominant": ";".join(reference_agreement),
            }
        )

    return pd.DataFrame(results)



def load_reference_version_overrides(
    selected_methods_path: str, reference_originals: Dict[str, str]
) -> Dict[tuple, str]:
    """Per (label, architecture, dataset): the exact resolved version string
    selected_methods.csv picked for that reference method.

    Some architectures never converged with a reference method's plain run
    and src/plots/extract_methods.py falls back to a differently-named variant instead
    (e.g. swin_t's DISTRIBUTIONAL notop resolves to a "..._margin_10.0"
    checkpoint -- see resolve_notop_fallback() in src/plots/extract_methods.py). Without
    this, compute_reference_method_trends would look for the plain name on
    swin_t, find nothing, and silently report it as "absent" there. Falls
    back to REFERENCE_VERSIONS wherever no override is found (including when
    selected_methods.csv is missing or stale).
    """
    if not selected_methods_path or not os.path.exists(selected_methods_path):
        print(f"[warn] selected-methods file not found, reference methods use plain names: {selected_methods_path}")
        return {}
    sel = pd.read_csv(selected_methods_path)
    sel = sel[(sel["source"] == "single") & (sel["selection"] != "MISSING")]
    original_to_label = {v: k for k, v in reference_originals.items()}

    overrides: Dict[tuple, str] = {}
    for _, row in sel.iterrows():
        original = str(row["method"]).split(" -> ")[0].strip()
        label = original_to_label.get(original)
        if label is None:
            continue
        resolved = str(row["method"]).split(" -> ")[-1].strip()
        overrides[(label, str(row["architecture"]), str(row["dataset"]))] = f"{resolved}/original_best"
    return overrides


def compute_reference_method_trends(
    delta_df: pd.DataFrame,
    config: TestConfig,
    references: Dict[str, str],
    families: Optional[List[str]] = None,
    version_overrides: Optional[Dict[tuple, str]] = None,
) -> pd.DataFrame:
    """Long-format trend of each proposed/reference method, one row per
    (experiment, dataset, metric, group_name, reference).

    Reports the reference's own direction next to the majority direction of
    the comparison methods, so it can be shown whether the proposed method
    follows the general trend without ever entering the count that defines
    that trend. `version_overrides` (see load_reference_version_overrides)
    resolves architecture-specific naming variants of a reference method
    instead of matching a single fixed version string everywhere.
    """
    counted = families if families is not None else CONSENSUS_METHOD_FAMILIES
    overrides = version_overrides or {}
    results: List[Dict[str, object]] = []

    keys = ["experiment", "dataset", "metric", "group_name"]
    for key, g in delta_df.groupby(keys, dropna=False):
        experiment, dataset, metric, group_name = key

        # The majority is defined by the comparison methods only: the IQR filter
        # runs over their deltas alone (same pool as the canonical pass), so the
        # proposed/reference methods never shift the bounds nor get discarded
        # for being far from the baselines.
        g = g.assign(method_family=g["version"].map(method_family))
        counted_deltas = g.loc[g["method_family"].isin(counted), "delta"].dropna().astype(float)
        g_counted = g.loc[drop_outliers_iqr(counted_deltas, config.iqr_multiplier).index]

        family_means = (
            g_counted
            .groupby("method_family")["delta"]
            .mean()
        )
        n_concentrate = int((family_means < 0).sum())
        n_disperse = int((family_means > 0).sum())
        majority = (
            "concentrate" if n_concentrate > n_disperse
            else "disperse" if n_disperse > n_concentrate
            else "tie"
        )

        for label, default_version in references.items():
            version = overrides.get((label, experiment, dataset), default_version)
            # Each reference is cleaned with the IQR of its own runs only.
            own = g.loc[g["version"] == version, "delta"].dropna().astype(float)
            vals = drop_outliers_iqr(own, config.iqr_multiplier)
            if vals.empty:
                direction = "absent"
                mean_delta = median_delta = np.nan
            else:
                mean_delta = float(vals.mean())
                median_delta = float(vals.median())
                direction = (
                    "concentrate" if mean_delta < 0
                    else "disperse" if mean_delta > 0
                    else "tie"
                )

            results.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "metric": metric,
                    "group_name": group_name,
                    "reference": label,
                    "reference_version": version,
                    "n_clean": int(len(vals)),
                    "mean_delta": mean_delta,
                    "median_delta": median_delta,
                    "direction": direction,
                    "n_methods_counted": int(len(family_means)),
                    "n_methods_concentrate": n_concentrate,
                    "n_methods_disperse": n_disperse,
                    "majority_direction": majority,
                    "agrees_with_majority": (
                        np.nan if direction == "absent" else bool(direction == majority)
                    ),
                }
            )

    return pd.DataFrame(results)


def compare_target_individual(
    delta_df: pd.DataFrame, config: TestConfig, target_version: str
) -> pd.DataFrame:
    """Per (experiment, dataset, metric, group_name): rank of target_version
    among all methods after pooled IQR-outlier removal."""
    results: List[Dict[str, object]] = []
    keys = ["experiment", "dataset", "metric", "group_name"]

    for key, g in delta_df.groupby(keys, dropna=False):
        experiment, dataset, metric, group_name = key

        all_deltas = g["delta"].dropna().astype(float)
        clean_idx = drop_outliers_iqr(all_deltas, config.iqr_multiplier).index
        g_clean = g.loc[clean_idx]

        version_means: Dict[str, float] = {}
        version_n: Dict[str, int] = {}
        for version, vg in g_clean.groupby("version", dropna=False):
            if str(version).startswith(SPECIAL_PREFIXES) and version != target_version:
                continue
            vals = vg["delta"].astype(float)
            if len(vals) == 0:
                continue
            version_means[version] = float(np.mean(vals))
            version_n[version] = int(len(vals))

        if not version_means:
            continue

        sorted_versions = sorted(version_means.items(), key=lambda x: x[1])
        n_methods = len(sorted_versions)
        target_present = target_version in version_means
        target_mean = version_means.get(target_version, np.nan)
        target_n = version_n.get(target_version, 0)

        rank: object = np.nan
        if target_present:
            rank = next(
                i + 1 for i, (v, _) in enumerate(sorted_versions) if v == target_version
            )

        best_v, best_mean = sorted_versions[0]
        worst_v, worst_mean = sorted_versions[-1]

        results.append(
            {
                "experiment": experiment,
                "dataset": dataset,
                "metric": metric,
                "group_name": group_name,
                "target_present": target_present,
                "target_mean_delta": target_mean,
                "target_n_clean": target_n,
                "target_rank": rank,
                "n_methods": n_methods,
                "is_most_concentrating": bool(rank == 1) if target_present else False,
                "is_least_concentrating": bool(rank == n_methods) if target_present else False,
                "most_concentrating_method": best_v,
                "most_concentrating_mean_delta": best_mean,
                "least_concentrating_method": worst_v,
                "least_concentrating_mean_delta": worst_mean,
                "delta_vs_most_concentrating": (target_mean - best_mean)
                if target_present
                else np.nan,
                "delta_vs_least_concentrating": (target_mean - worst_mean)
                if target_present
                else np.nan,
            }
        )

    return pd.DataFrame(results)


def compare_target_statistical(
    delta_df: pd.DataFrame, config: TestConfig, target_version: str
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test (two-sided): target_version vs
    most/least concentrating method, paired by class_to_remove."""
    results: List[Dict[str, object]] = []
    keys = ["experiment", "dataset", "metric", "group_name"]

    for key, g in delta_df.groupby(keys, dropna=False):
        experiment, dataset, metric, group_name = key

        all_deltas = g["delta"].dropna().astype(float)
        clean_idx = drop_outliers_iqr(all_deltas, config.iqr_multiplier).index
        g_clean = g.loc[clean_idx]

        version_means: Dict[str, float] = {}
        for version, vg in g_clean.groupby("version", dropna=False):
            if str(version).startswith(SPECIAL_PREFIXES) and version != target_version:
                continue
            vals = vg["delta"].astype(float)
            if len(vals) == 0:
                continue
            version_means[version] = float(np.mean(vals))

        if target_version not in version_means or len(version_means) < 2:
            continue

        non_target_versions = sorted(
            ((v, m) for v, m in version_means.items() if v != target_version),
            key=lambda x: x[1],
        )
        if not non_target_versions:
            continue
        best_v = non_target_versions[0][0]
        worst_v = non_target_versions[-1][0]

        target_data = (
            g_clean[g_clean["version"] == target_version]
            .set_index("class_to_remove")["delta"]
            .astype(float)
        )

        for compared_label, compared_v in [
            ("most_concentrating", best_v),
            ("least_concentrating", worst_v),
        ]:
            if compared_v == target_version:
                results.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "metric": metric,
                        "group_name": group_name,
                        "compared_to": compared_label,
                        "compared_method": compared_v,
                        "n_paired": 0,
                        "target_mean_delta": version_means[target_version],
                        "compared_mean_delta": version_means[compared_v],
                        "mean_diff": 0.0,
                        "wilcoxon_statistic": np.nan,
                        "p_value": np.nan,
                        "alpha": config.alpha,
                        "reject_h0": False,
                        "comparable": True,
                        "note": "target_is_compared_method",
                    }
                )
                continue

            compared_data = (
                g_clean[g_clean["version"] == compared_v]
                .set_index("class_to_remove")["delta"]
                .astype(float)
            )

            common_classes = target_data.index.intersection(compared_data.index)
            diff = (
                target_data.loc[common_classes].values
                - compared_data.loc[common_classes].values
            )

            n_paired = int(len(diff))
            stat = np.nan
            p_value = np.nan
            reject_h0 = False
            comparable: object = np.nan
            note = ""

            if n_paired >= config.min_samples:
                try:
                    wx = wilcoxon(diff, alternative="two-sided")
                    stat = float(wx.statistic)
                    p_value = float(wx.pvalue)
                    reject_h0 = bool(p_value < config.alpha)
                    comparable = not reject_h0
                except Exception as e:
                    note = str(e)

            results.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "metric": metric,
                    "group_name": group_name,
                    "compared_to": compared_label,
                    "compared_method": compared_v,
                    "n_paired": n_paired,
                    "target_mean_delta": version_means[target_version],
                    "compared_mean_delta": version_means[compared_v],
                    "mean_diff": float(np.mean(diff)) if n_paired > 0 else np.nan,
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "alpha": config.alpha,
                    "reject_h0": reject_h0,
                    "comparable": comparable,
                    "note": note,
                }
            )

    return pd.DataFrame(results)


# variants of the proposed method (previously the DISTRIBUTIONAL ledoit-wolf ones)
LEDOIT_PREFIX = PROPOSED_PREFIX


def run_distributional_homoscedasticity(source_df: pd.DataFrame, config: TestConfig) -> pd.DataFrame:
    """Levene test between each LEDOIT variant and each baseline per class."""
    all_metrics = L2_METRICS + COS_METRICS
    group_cols = ["experiment", "dataset", "class_to_remove"]

    compact = (
        source_df.groupby(group_cols + ["version"], as_index=False)[all_metrics]
        .mean(numeric_only=True)
    )

    results: List[Dict[str, object]] = []

    for (experiment, dataset), g in compact.groupby(["experiment", "dataset"], dropna=False):
        ledoit_versions = (
            g[g["version"].astype(str).str.startswith(LEDOIT_PREFIX)]["version"]
            .unique()
            .tolist()
        )

        for version in ledoit_versions:
            method_rows = g[g["version"] == version].set_index("class_to_remove")

            for baseline in BASELINES:
                baseline_rows = g[g["version"] == baseline].set_index("class_to_remove")
                common_classes = method_rows.index.intersection(baseline_rows.index)
                if len(common_classes) == 0:
                    continue

                for metric in all_metrics:
                    m = method_rows.loc[common_classes, metric].dropna().astype(float)
                    b = baseline_rows.loc[m.index, metric].dropna().astype(float)
                    shared = m.index.intersection(b.index)
                    m, b = m.loc[shared], b.loc[shared]

                    if len(m) < 2 or len(b) < 2:
                        continue

                    var_method = float(np.var(m, ddof=1))
                    var_baseline = float(np.var(b, ddof=1))

                    lev_stat = np.nan
                    lev_p = np.nan
                    homoscedastic = np.nan
                    test_run = False

                    if len(m) >= config.min_samples and len(b) >= config.min_samples:
                        stat, p = levene(m.to_numpy(), b.to_numpy(), center="median")
                        lev_stat = float(stat)
                        lev_p = float(p)
                        homoscedastic = bool(lev_p >= config.alpha)
                        test_run = True

                    results.append(
                        {
                            "experiment": experiment,
                            "dataset": dataset,
                            "version": version,
                            "metric": metric,
                            "baseline": baseline,
                            "n_classes": len(m),
                            "var_method": var_method,
                            "var_baseline": var_baseline,
                            "std_method": float(np.std(m, ddof=1)),
                            "std_baseline": float(np.std(b, ddof=1)),
                            "var_ratio_method_over_baseline": var_method / var_baseline
                            if var_baseline > 0 else np.nan,
                            "levene_statistic": lev_stat,
                            "levene_p_value": lev_p,
                            "alpha": config.alpha,
                            "homoscedastic": homoscedastic,
                            "test_run": test_run,
                        }
                    )

    return pd.DataFrame(results)


def run_distributional_comparison(source_df: pd.DataFrame) -> pd.DataFrame:
    """Per-class count of classes where LEDOIT method is less/equal/greater than baseline."""
    all_metrics = L2_METRICS + COS_METRICS
    group_cols = ["experiment", "dataset", "class_to_remove"]

    compact = (
        source_df.groupby(group_cols + ["version"], as_index=False)[all_metrics]
        .mean(numeric_only=True)
    )

    results: List[Dict[str, object]] = []

    for (experiment, dataset), g in compact.groupby(["experiment", "dataset"], dropna=False):
        ledoit_versions = (
            g[g["version"].astype(str).str.startswith(LEDOIT_PREFIX)]["version"]
            .unique()
            .tolist()
        )

        for version in ledoit_versions:
            method_rows = g[g["version"] == version].set_index("class_to_remove")

            for baseline in BASELINES:
                baseline_rows = g[g["version"] == baseline].set_index("class_to_remove")
                common_classes = method_rows.index.intersection(baseline_rows.index)
                if len(common_classes) == 0:
                    continue

                for metric in all_metrics:
                    m_vals = method_rows.loc[common_classes, metric].dropna()
                    b_vals = baseline_rows.loc[m_vals.index, metric].dropna()
                    shared = m_vals.index.intersection(b_vals.index)
                    if shared.empty:
                        continue

                    m = m_vals.loc[shared].astype(float)
                    b = b_vals.loc[shared].astype(float)
                    diff = m - b

                    n_less = int((diff < 0).sum())
                    n_greater = int((diff > 0).sum())
                    n_equal = int((diff == 0).sum())
                    n_total = len(diff)
                    pct_less = n_less / n_total if n_total > 0 else np.nan
                    direction = "less" if n_less > n_greater else ("greater" if n_greater > n_less else "tie")

                    results.append(
                        {
                            "experiment": experiment,
                            "dataset": dataset,
                            "version": version,
                            "metric": metric,
                            "baseline": baseline,
                            "n_classes": n_total,
                            "n_less": n_less,
                            "n_greater": n_greater,
                            "n_equal": n_equal,
                            "pct_less": round(pct_less, 4) if not np.isnan(pct_less) else np.nan,
                            "median_method": float(np.median(m)),
                            "median_baseline": float(np.median(b)),
                            "median_diff": float(np.median(diff)),
                            "direction": direction,
                        }
                    )

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    cfg = TestConfig(
        iqr_multiplier=args.iqr_multiplier,
        alpha=args.alpha,
        min_samples=args.min_samples,
    )

    for attr in [
        "output_summary", "output_values", "output_summary_distance",
        "output_summary_by_architecture", "output_summary_distance_by_architecture",
        "output_ks_summary", "output_best_worst", "output_concentration_consensus", "output_reference_trends",
        "output_ledoit_individual", "output_ledoit_statistical",
        "output_distributional", "output_distributional_var",
        "output_homoscedasticity",
    ]:
        setattr(args, attr, apply_split_suffix(getattr(args, attr), args.split))
    mixed_model_path = apply_split_suffix("results/mixed_model_summary_retain.csv", args.split)

    if args.split == "single":
        df = add_embs_fallback(pd.read_csv(args.input))
    elif args.split == "multi":
        df = read_optional_csv(args.input_multi, "multi")
    else:  # both
        df = pd.concat(
            [add_embs_fallback(pd.read_csv(args.input)), read_optional_csv(args.input_multi, "multi")],
            ignore_index=True,
            sort=False,
        )

    if df.empty:
        print(
            f"[warn] split='{args.split}': no input rows available yet — "
            "skipping analysis (nothing written)."
        )
        return

    df = select_checkpoint_rows(df)
    if df.empty:
        print(f"[warn] split='{args.split}': no checkpoint rows found — skipping analysis.")
        return
    n_oneshot = int((df["epoch"].astype(str) == "00").sum())
    if n_oneshot:
        print(
            f"[info] {n_oneshot} rows taken from epoch '00' (one-shot methods "
            "without a 'best' checkpoint)"
        )

    if sum([args.distributionalout, args.distributionalnotop, args.only_distributional]) > 1:
        raise ValueError(
            "Options --distributionalout, --distributionalnotop, and "
            "--only-distributional are mutually exclusive."
        )

    if args.distributionalout:
        df = df[~df["version"].astype(str).str.startswith(SPECIAL_PREFIXES)].copy()

    if args.distributionalnotop:
        version_series = df["version"].astype(str)
        is_special = version_series.str.startswith(SPECIAL_PREFIXES)
        keep_notop = version_series.isin(SPECIAL_KEEP_VERSIONS)
        df = df[~is_special | keep_notop].copy()

    if args.only_distributional:
        version_series = df["version"].astype(str)
        is_special = version_series.str.startswith(SPECIAL_PREFIXES)
        is_baseline = version_series.isin(BASELINES)
        df = df[is_special | is_baseline].copy()

    if args.goodones:
        if args.split == "single":
            ref_df = add_embs_fallback(pd.read_csv(args.goodones_input))
        elif args.split == "multi":
            ref_df = read_optional_csv(args.goodones_input_multi, "goodones multi reference")
        else:  # both
            ref_df = pd.concat(
                [
                    add_embs_fallback(pd.read_csv(args.goodones_input)),
                    read_optional_csv(args.goodones_input_multi, "goodones multi reference"),
                ],
                ignore_index=True,
                sort=False,
            )

        if ref_df.empty:
            print(
                f"[warn] --goodones: no reference rows available for "
                f"split='{args.split}' — skipping accuracy-based filtering."
            )
        else:
            ref_df = select_checkpoint_rows(ref_df)
            needed_cols = {"accuracy_score_after", "unlearning_accuracy_score"}
            missing = needed_cols.difference(ref_df.columns)
            if missing:
                raise ValueError(
                    "Missing columns required for --goodones: "
                    + ", ".join(sorted(missing))
                )
            bad_mask = (
                (pd.to_numeric(ref_df["accuracy_score_after"], errors="coerce") < 0.6)
                & (pd.to_numeric(ref_df["unlearning_accuracy_score"], errors="coerce") > 0.2)
            )
            bad_keys = (
                ref_df.loc[bad_mask, ["experiment", "id_database", "version"]]
                .drop_duplicates()
            )
            bad_idx = pd.MultiIndex.from_frame(bad_keys)
            df_idx = pd.MultiIndex.from_arrays(
                [df["experiment"], df["id_database"], df["version"]]
            )
            df = df[~df_idx.isin(bad_idx)].copy()

    cols_to_drop = ["maxprobability_retain"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    parsed = df["id_database"].map(parse_database_and_class)
    valid_mask = parsed.notna()
    df = df.loc[valid_mask].copy()
    if df.empty:
        print(
            f"[warn] split='{args.split}': no rows with a parseable id_database "
            "— skipping analysis."
        )
        return
    parsed_df = pd.DataFrame(parsed[valid_mask].tolist(), index=df.index)
    df = pd.concat([df, parsed_df], axis=1)

    chosen_methods = load_selected_methods(args.selected_methods)
    if chosen_methods is not None:
        n_before = len(df)
        df = filter_selected_methods(df, chosen_methods)
        n_methods = df.loc[~df["version"].isin(BASELINES), "version"].nunique()
        print(
            f"[info] selected-methods filter: {n_before} -> {len(df)} rows, "
            f"{n_methods} method versions kept"
        )
        if df.empty:
            print("[warn] no rows left after the selected-methods filter — skipping analysis.")
            return

    delta_df = build_delta_table(df, COHESION_L2_METRIC, COHESION_COS_METRIC)
    if delta_df.empty:
        print(
            f"[warn] split='{args.split}': no comparable rows found (both "
            "baselines must exist for each (experiment, dataset, "
            "class_to_remove)) — skipping analysis."
        )
        return

    distance_delta_df = build_delta_table(df, DISTANCE_L2_METRIC, DISTANCE_COS_METRIC)

    delta_df["architecture_family"] = delta_df["experiment"].map(ARCHITECTURE_FAMILY)
    distance_delta_df["architecture_family"] = distance_delta_df["experiment"].map(ARCHITECTURE_FAMILY)

    if args.export_raw_dir:
        from export_raw_values import export_raw_values

        export_raw_values(
            [delta_df, distance_delta_df],
            "retain",
            args.export_raw_dir,
            drop_outliers_iqr=drop_outliers_iqr,
            iqr_multiplier=cfg.iqr_multiplier,
            special_prefixes=SPECIAL_PREFIXES,
            reference_versions=REFERENCE_VERSIONS,
            version_overrides=load_reference_version_overrides(
                args.selected_methods, REFERENCE_METHOD_ORIGINALS
            ),
        )
        return

    unmapped = sorted(
        set(delta_df.loc[delta_df["architecture_family"].isna(), "experiment"].astype(str))
    )
    if unmapped:
        print(
            f"[warn] no architecture_family mapping for experiment(s) {unmapped} — "
            "excluded from the by-architecture summary."
        )

    detail_parts: List[pd.DataFrame] = []
    for key, g in delta_df.groupby(["experiment", "dataset", "group_name"], dropna=False):
        raw = g["delta"].astype(float)
        clean = drop_outliers_iqr(raw, cfg.iqr_multiplier)
        aux = g.copy()
        aux["is_outlier"] = ~raw.index.isin(clean.index)
        detail_parts.append(aux)

    detail_df = pd.concat(detail_parts, ignore_index=True)
    summary_df = run_group_tests(delta_df, cfg).sort_values(
        ["alternative", "experiment", "dataset", "group_name"]
    )
    distance_summary_df = run_group_tests(distance_delta_df, cfg).sort_values(
        ["alternative", "experiment", "dataset", "group_name"]
    )

    family_keys = ["architecture_family", "dataset", "group_name"]
    family_summary_df = run_group_tests(
        delta_df.dropna(subset=["architecture_family"]), cfg, group_keys=family_keys
    ).sort_values(["alternative", "architecture_family", "dataset", "group_name"])
    family_distance_summary_df = run_group_tests(
        distance_delta_df.dropna(subset=["architecture_family"]), cfg, group_keys=family_keys
    ).sort_values(["alternative", "architecture_family", "dataset", "group_name"])

    agreement_merge_keys = ["architecture_family", "dataset", "group_name"]
    agreement_df = compute_family_agreement(summary_df, ARCHITECTURE_FAMILY)
    if not agreement_df.empty:
        family_summary_df = family_summary_df.merge(
            agreement_df, on=agreement_merge_keys, how="left"
        )
    distance_agreement_df = compute_family_agreement(distance_summary_df, ARCHITECTURE_FAMILY)
    if not distance_agreement_df.empty:
        family_distance_summary_df = family_distance_summary_df.merge(
            distance_agreement_df, on=agreement_merge_keys, how="left"
        )

    family_summary_df.to_csv(args.output_summary_by_architecture, index=False)
    family_distance_summary_df.to_csv(args.output_summary_distance_by_architecture, index=False)
    print(f"Saved by-architecture cohesion summary to: {args.output_summary_by_architecture}")
    print(f"Saved by-architecture distance summary to: {args.output_summary_distance_by_architecture}")

    ks_summary_df = run_ks_tests(df, cfg).sort_values(["experiment", "dataset", "metric", "baseline"])

    mixed_summary_df = run_mixed_model_analysis(delta_df, cfg).sort_values(
        ["experiment", "dataset", "group_name"]
    )
    mixed_summary_df.to_csv(mixed_model_path, index=False)
    print(f"Saved mixed model summary to: {mixed_model_path}")

    distributional_df = run_distributional_comparison(df)
    if not distributional_df.empty:
        distributional_df = distributional_df.sort_values(
            ["experiment", "dataset", "version", "metric", "baseline"]
        )
    distributional_df.to_csv(args.output_distributional, index=False)
    print(f"Saved distributional ledoit comparison to: {args.output_distributional}")

    distributional_var_df = run_distributional_homoscedasticity(df, cfg)
    if not distributional_var_df.empty:
        distributional_var_df = distributional_var_df.sort_values(
            ["experiment", "dataset", "version", "metric", "baseline"]
        )
    distributional_var_df.to_csv(args.output_distributional_var, index=False)
    print(f"Saved distributional ledoit homoscedasticity to: {args.output_distributional_var}")

    homoscedasticity_df = run_homoscedasticity_tests(df, cfg).sort_values(
        ["experiment", "dataset", "metric", "baseline"]
    )
    homoscedasticity_df.to_csv(args.output_homoscedasticity, index=False)
    print(f"Saved global homoscedasticity summary to: {args.output_homoscedasticity}")

    all_delta_df = pd.concat([delta_df, distance_delta_df], ignore_index=True)
    best_worst_df = compute_best_worst_methods(all_delta_df, cfg).sort_values(
        ["experiment", "dataset", "metric", "group_name"]
    )
    best_worst_df.to_csv(args.output_best_worst, index=False)

    consensus_df = compute_concentration_consensus(
        all_delta_df, cfg, reference_versions=SPECIAL_KEEP_VERSIONS
    ).sort_values(["experiment", "dataset", "metric", "group_name"])
    consensus_df.to_csv(args.output_concentration_consensus, index=False)
    print(f"Saved concentration consensus to: {args.output_concentration_consensus}")

    reference_version_overrides = load_reference_version_overrides(
        args.selected_methods, REFERENCE_METHOD_ORIGINALS
    )
    reference_trends_df = compute_reference_method_trends(
        all_delta_df, cfg, REFERENCE_VERSIONS, version_overrides=reference_version_overrides
    ).sort_values(["reference", "experiment", "dataset", "metric", "group_name"])
    reference_trends_df.to_csv(args.output_reference_trends, index=False)
    print(f"Saved reference method trends to: {args.output_reference_trends}")

    individual_df = compare_target_individual(
        all_delta_df, cfg, PROPOSED_KEEP_VERSION
    ).sort_values(["experiment", "dataset", "metric", "group_name"])
    individual_df.to_csv(args.output_ledoit_individual, index=False)

    statistical_df = compare_target_statistical(
        all_delta_df, cfg, PROPOSED_KEEP_VERSION
    )
    if not statistical_df.empty:
        statistical_df = statistical_df.sort_values(
            ["experiment", "dataset", "metric", "group_name", "compared_to"]
        )
    statistical_df.to_csv(args.output_ledoit_statistical, index=False)

    detail_df.to_csv(args.output_values, index=False)
    summary_df.to_csv(args.output_summary, index=False)
    distance_summary_df.to_csv(args.output_summary_distance, index=False)
    ks_summary_df.to_csv(args.output_ks_summary, index=False)

    print(f"Split: {args.split}")
    print(f"Input rows after epoch=best and id_database parse: {len(df)}")
    print(f"Delta rows generated: {len(delta_df)}")
    print(f"Saved detailed deltas to:              {args.output_values}")
    print(f"Saved grouped test summary to:         {args.output_summary}")
    print(f"Saved distance-only summary to:        {args.output_summary_distance}")
    print(f"Saved best/worst concentration methods to: {args.output_best_worst}")
    print(f"Saved LEDOIT_NOTOP individual comparison to: {args.output_ledoit_individual}")
    print(f"Saved LEDOIT_NOTOP statistical comparison to: {args.output_ledoit_statistical}")
    print(f"Saved KS baseline-vs-methods to:       {args.output_ks_summary}")


if __name__ == "__main__":
    main()
