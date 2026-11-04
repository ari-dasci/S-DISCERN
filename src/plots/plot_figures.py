#!/usr/bin/env python3
"""
Plot ICU unlearning results, restricted to the same methods selected in
src/plots/extract_methods.py, with cluster metrics enabled and one output folder
per experiment (architecture) so the plots stay readable.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.embs_fallback import add_embs_fallback  # noqa: E402  (TEMPORARY: efficientnet nofreeze as embs)
from src.utils.proposed_method import (  # noqa: E402  (proposed method = PARAMETRIC)
    PROPOSED_NOTOP, PROPOSED_CONF, PROPOSED_NOTOP_DISPLAY, PROPOSED_CONF_DISPLAY,
)
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH        = "results_ICU/unlearning_results_analyzed.csv"
RAW_ICU_SRC     = "results_ICU/unlearning_results.csv"
SINGLE_CSV_PATH = "unlearning_results_single_all_new_analyzed_means.csv"
OUT_RAW         = "plots/ICU_results"
OUT_CLUSTER     = "plots/ICU_results_cluster"
OUT_OTHERS      = "plots/ICU_results_others"
OUT_ABLATION    = "plots/ablation/ICU"

# Methods kept for every architecture, matching the fixed selection used in
# src/plots/extract_methods.py so single/ICU plots stay comparable. NOISY ("Gaussian")
# stays in this list so it's loaded, but it's dropped from the main plots and
# only shown in the ablation (see ABLATION_METHODS below).
FIXED_METHODS_ICU = [
    PROPOSED_NOTOP,   # proposed method (src/utils/proposed_method.py; previously DISTRIBUTIONAL)
    PROPOSED_CONF,
    "DELETE_embs",
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs",
    "BOUNDARYEXPAND_lr1e-05_embs",
    "BADTEACHER_T1.0_forgetOnly_embs",
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty",
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop",
    "RANDOMLABEL_noret_embs_forbidden",
    "RETRAIN",
    "original",
]

# Short display names for the legend (same convention as src/plots/plot_boxplots.py).
# The SSD_MAS_embs/EWC_embsunfreeze winners vary in exact hyperparameters per
# architecture, so they're mapped by role (ssd_name/ewc_name), see display_name().
DISPLAY_NAMES = {
    PROPOSED_NOTOP: PROPOSED_NOTOP_DISPLAY,
    PROPOSED_CONF: PROPOSED_CONF_DISPLAY,
    "DELETE_embs": "DELETE",
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs": "Boundary Shrink",
    "BOUNDARYEXPAND_lr1e-05_embs": "Boundary Expand",
    "BADTEACHER_T1.0_forgetOnly_embs": "Bad Teacher",
    "RANDOMLABEL_noret_embs_forbidden": "Random Labeling",
    # NOISY renamed "Gaussian" (see src/plots/plot_boxplots.py's ablation comparison)
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop": "Gaussian (notop)",
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty": "Gaussian (conf.)",
    "RETRAIN": "Retrain",
    "original": "Original",
}

_DIST_NOTOP  = PROPOSED_NOTOP   # proposed method (name kept for the existing code)
_DIST_CONF   = PROPOSED_CONF
_NOISY_NOTOP = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
_NOISY_CONF  = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"

# Ablation: the proposed method vs. its Gaussian-noise counterpart (NOISY), paired
# by loss (notop / conf.). Same hyperparameters on both sides (1000 fixed
# synthetic samples, ledoit_wolf covariance, unfrozen embs), so the only
# difference is the method itself. Same pairing as src/plots/plot_boxplots.py.
ABLATION_METHODS = [_DIST_NOTOP, _NOISY_NOTOP, _DIST_CONF, _NOISY_CONF]

# Only the accuracy pair (+ their harmonic mean) go in the ablation, same as
# src/plots/plot_boxplots.py's ablation.
ABLATION_METRIC_COLS = ("accuracy_score_after_mean", "unlearning_accuracy_score_mean")

# Per-method colors shared with src/plots/plot_boxplots.py (keyed by display name), so
# each method looks the same in every plot, main and ablation alike.
from method_colors import method_color, arch_name, line_kwargs, bold_our_legend_entries

def is_noisy(base):
    return str(base).startswith("NOISY_")

_MARGIN_SUFFIX_RE = re.compile(r"_margin_[0-9.]+.*$")

def display_name(base, ssd_name=None, ewc_name=None):
    if ssd_name is not None and base == ssd_name:
        return "LF-SSD"
    if ewc_name is not None and base == ewc_name:
        return "EWC"
    if base in DISPLAY_NAMES:
        return DISPLAY_NAMES[base]
    # A "_notop" fixed method resolved via resolve_notop_fallback() to its
    # "_notop_margin_X.X" variant still displays under its plain notop name.
    stripped = _MARGIN_SUFFIX_RE.sub("", base)
    return DISPLAY_NAMES.get(stripped, base)

# ── CLI arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot ICU unlearning results (selected methods only).")
parser.add_argument("--experiments",  nargs="+", default=None, metavar="EXP",
                    help="Experiments to include (e.g. resnet18 vit_b_16)")
parser.add_argument("--datasets",     nargs="+", default=["cifar100"], metavar="DS",
                    help="Datasets to include (default: cifar100 only; pass e.g. --datasets cifar10 mnist cifar100 to override)")
parser.add_argument("--num-classes",  nargs="+", default=None, type=int, metavar="N",
                    help="Num classes to forget (e.g. 5 20)")
parser.add_argument("--steps-min",    default=None, type=int, metavar="N",
                    help="Minimum steps (inclusive)")
parser.add_argument("--steps-max",    default=None, type=int, metavar="N",
                    help="Maximum steps (inclusive)")
parser.add_argument("--lang",         default="en", choices=["es", "en"],
                    help="Language for metric labels (default: en)")
parser.add_argument("--std",          dest="std", action="store_true", default=True,
                    help="Show standard deviation (default: on)")
parser.add_argument("--no-std",       dest="std", action="store_false",
                    help="Disable standard deviation")
parser.add_argument("--std-style",    default="fill", choices=["errorbar", "fill"],
                    help="How to draw the standard deviation when --std is set: "
                         "'fill' (default, shaded ±std band, as in gen_figures.py) or 'errorbar'")
parser.add_argument("--std-alpha",    default=0.15, type=float,
                    help="Alpha of the shaded ±std band when --std-style=fill (default: 0.15)")
args = parser.parse_args()

# Known metrics with labels: (mean_col, std_col, label_es, label_en)
_METRICS_RAW = [
    ("accuracy_score_after_mean",      "accuracy_score_after_std",      "Precisión (retención) ↑", "Accuracy (retain) ↑"),
    ("unlearning_accuracy_score_mean", "unlearning_accuracy_score_std", "Precisión (olvido) ↓",    "Accuracy (forget) ↓"),
    ("maxprobability_mean",            "maxprobability_std",            "Prob. máxima (retención)", "Max probability (retain)"),
    ("maxprobability_forget_mean",     "maxprobability_forget_std",     "Prob. máxima (olvido)",    "Max probability (forget)"),
]

_METRICS_CLUSTER = [
    ("cohesion_forget_l2_mean",        "cohesion_forget_l2_std",        "Cohesión olvido (L2)",              "Forget cohesion (L2)"),
    ("distance_inter_class_l2_mean",   "distance_inter_class_l2_std",   "Distancia inter-clase olvido (L2)", "Forget inter-class distance (L2)"),
    ("cohesion_forget_cos_mean",       "cohesion_forget_cos_std",       "Cohesión olvido (cos)",             "Forget cohesion (cos)"),
    ("distance_inter_class_cos_mean",  "distance_inter_class_cos_std",  "Distancia inter-clase olvido (cos)","Forget inter-class distance (cos)"),
    ("cohesion_retain_l2_mean",        "cohesion_retain_l2_std",        "Cohesión retención (L2)",           "Retain cohesion (L2)"),
    ("distance_inter_retain_l2_mean",  "distance_inter_retain_l2_std",  "Distancia inter-clase retención (L2)", "Retain inter-class distance (L2)"),
    ("cohesion_retain_cos_mean",       "cohesion_retain_cos_std",       "Cohesión retención (cos)",          "Retain cohesion (cos)"),
    ("distance_inter_retain_cos_mean", "distance_inter_retain_cos_std", "Distancia inter-clase retención (cos)", "Retain inter-class distance (cos)"),
    ("cohesion_unknown_l2_mean",       "cohesion_unknown_l2_std",       "Cohesión desconocido (L2)",         "Unknown cohesion (L2)"),
    ("distance_inter_unknown_l2_mean", "distance_inter_unknown_l2_std", "Distancia inter-clase desconocido (L2)", "Unknown inter-class distance (L2)"),
    ("cohesion_unknown_cos_mean",      "cohesion_unknown_cos_std",      "Cohesión desconocido (cos)",        "Unknown cohesion (cos)"),
    ("distance_inter_unknown_cos_mean","distance_inter_unknown_cos_std","Distancia inter-clase desconocido (cos)", "Unknown inter-class distance (cos)"),
]

LANG_IDX = 2 if args.lang == "es" else 3

# ── Select winning SSD_MAS_embs / EWC_embsunfreeze per architecture ─────────
# Reuses the same criterion as src/plots/extract_methods.py (best harmonic mean between
# accuracy_score_after_mean and (1 - unlearning_accuracy_score_mean)) on the
# single-class results, so the ICU plots compare against the *same* winning
# hyperparameters instead of re-selecting a possibly different one from the
# (much smaller) ICU hyperparameter grid.

def base_version(v):
    return str(v).split("/")[0]

def resolve_notop_fallback(name, available_bases):
    # Some method families only logged a "_notop_margin_X.X" variant instead
    # of the plain "_notop" one. If the exact name is missing, fall back to
    # the same family's margin variant, preferring margin_10.0 if present,
    # else the highest available margin.
    if name in available_bases or not name.endswith("_notop"):
        return name
    prefix = name + "_margin_"
    candidates = [b for b in available_bases if b.startswith(prefix)]
    if not candidates:
        return name
    exact10 = name + "_margin_10.0"
    if exact10 in candidates:
        return exact10
    def margin_val(b):
        m = re.match(re.escape(prefix) + r"([0-9.]+)", b)
        return float(m.group(1)) if m else -1.0
    return max(candidates, key=margin_val)

def harmonic(acc, unl):
    acc = float(acc); unl = 1.0 - float(unl)
    if acc + unl == 0:
        return 0.0
    return 2 * acc * unl / (acc + unl)

_EWC_EMBS_NOTOP_RE = re.compile(r"_MAS_(embsunfreeze|embs)_notop$")

def resolve_ewc_embs_naming(name, available_bases):
    # The EWC winner picked from single (may be named "embsunfreeze" or
    # plain "embs") can show up under the other naming in the ICU data for
    # a given architecture - translate between the two so the same run is
    # still matched instead of silently missing.
    if name in available_bases:
        return name
    if "embsunfreeze" in name:
        alt = name.replace("embsunfreeze", "embs")
    elif "_embs_" in name:
        alt = name.replace("_embs_", "_embsunfreeze_")
    else:
        return name
    return alt if alt in available_bases else name

def compute_single_winners(path):
    single_df = pd.read_csv(path)
    winners = {}
    for arch, adf in single_df.groupby("experiment"):
        bases = adf["version"].apply(base_version)
        ssd = adf[bases.str.startswith("SSD_MAS_") & bases.str.endswith("_embs")]
        # EWC's unfrozen-embedding variant is named "embsunfreeze" in some
        # files and plain "embs" in others (e.g. ICU) - match both, but not
        # "freeze" (note "unfreeze" itself contains the substring "freeze").
        ewc = adf[bases.apply(lambda b: bool(_EWC_EMBS_NOTOP_RE.search(b)))]
        arch_winners = []
        if not ssd.empty:
            scores = ssd.apply(lambda r: harmonic(r["accuracy_score_after_mean"], r["unlearning_accuracy_score_mean"]), axis=1)
            arch_winners.append(base_version(ssd.loc[scores.idxmax(), "version"]))
        if not ewc.empty:
            scores = ewc.apply(lambda r: harmonic(r["accuracy_score_after_mean"], r["unlearning_accuracy_score_mean"]), axis=1)
            best_name = base_version(ewc.loc[scores.idxmax(), "version"])
            # Kept as picked from single (embsunfreeze or embs); translated
            # against the actual ICU naming per-architecture in apply_selected_methods.
            arch_winners.append(best_name)
        winners[arch] = arch_winners
    return winners

SINGLE_WINNERS = compute_single_winners(SINGLE_CSV_PATH)
print("[INFO] Single-results winners per architecture (SSD_MAS_embs, EWC_embsunfreeze):")
for arch, w in SINGLE_WINNERS.items():
    print(f"         {arch}: {w}")

# ── Read data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Drop rows with too few averages (unreliable) and non-multiples of 10 (near-duplicates)
df = df[df["num_averaged"] >= 10]
df = df[df["num_averaged"] % 10 == 0]

# Apply user-defined filters
def apply_filter(df, col, values, label):
    if values is None:
        return df
    pattern = "|".join(re.escape(str(v)) for v in values)
    mask = df[col].astype(str).str.contains(pattern, na=False)
    no_match = [v for v in values if not df[col].astype(str).str.contains(re.escape(str(v)), na=False).any()]
    if no_match:
        print(f"[WARNING] --{label}: no matches for {no_match}. Available: {sorted(df[col].unique())}")
    return df[mask]

df = apply_filter(df, "experiment",            args.experiments, "experiments")
df = apply_filter(df, "dataset",               args.datasets,    "datasets")
df = apply_filter(df, "num_classes_to_forget", args.num_classes, "num-classes")

if args.steps_min is not None:
    df = df[df["steps"] >= args.steps_min]
if args.steps_max is not None:
    df = df[df["steps"] <= args.steps_max]

# Restrict to the selected methods (fixed list + per-architecture winners)
def apply_selected_methods(df, fixed, winners):
    kept_frames = []
    for arch, adf in df.groupby("experiment", sort=False):
        bases = adf["version"].apply(base_version)
        bases_all = set(bases)
        resolved_fixed = [resolve_notop_fallback(m, bases_all) for m in fixed]
        for orig, resolved in zip(fixed, resolved_fixed):
            if resolved != orig:
                print(f"[INFO] {arch}: '{orig}' not found, falling back to '{resolved}'")

        resolved_winners = [resolve_ewc_embs_naming(w, bases_all) for w in winners.get(arch, [])]
        for orig, resolved in zip(winners.get(arch, []), resolved_winners):
            if resolved != orig:
                print(f"[INFO] {arch}: winner '{orig}' not found, using '{resolved}' instead")

        allowed = set(resolved_fixed) | set(resolved_winners)
        sub = adf[bases.isin(allowed)]
        found = set(bases[bases.isin(allowed)])
        missing = sorted(allowed - found)
        if missing:
            print(f"[WARNING] {arch}: selected method(s) not found in ICU data: {missing}")
        kept_frames.append(sub)
    return pd.concat(kept_frames) if kept_frames else df.iloc[0:0]

df = apply_selected_methods(df, FIXED_METHODS_ICU, SINGLE_WINNERS)

if df.empty:
    print("No data after filtering. Exiting.")
    raise SystemExit(1)

# ── Harmonic mean, computed per repetition (not from the already-aggregated
# means) so it gets a real std: harmonic(acc, 1-unl) per (id_database, steps),
# then mean/std of that across repetitions for each (version, steps).
raw_icu_df = add_embs_fallback(pd.read_csv(RAW_ICU_SRC))
_m = raw_icu_df["id_database"].str.extract(r"^(?P<dataset>.+?)_classes_to_remove_(?P<tail>.+)$")
raw_icu_df["dataset"] = _m["dataset"]
raw_icu_df["num_classes_to_forget"] = _m["tail"].str.count("_") + 1
raw_icu_df["harmonic_mean"] = [
    harmonic(a, u) for a, u in zip(raw_icu_df["accuracy_score_after"], raw_icu_df["unlearning_accuracy_score"])
]

def harmonic_stats(experiment, dataset, n_forget, versions):
    sub = raw_icu_df[
        (raw_icu_df["experiment"] == experiment)
        & (raw_icu_df["dataset"] == dataset)
        & (raw_icu_df["num_classes_to_forget"] == n_forget)
        & (raw_icu_df["version"].isin(versions))
    ]
    return sub.groupby(["version", "steps"])["harmonic_mean"].agg(["mean", "std"]).reset_index()

# Build METRICS: (mean_col, std_col, label, out_dir) — raw + cluster metrics
_seen_mean_cols = set()
METRICS = []
for row in _METRICS_RAW:
    if row[0] in df.columns:
        METRICS.append((row[0], row[1], row[LANG_IDX], OUT_RAW))
        _seen_mean_cols.add(row[0])
for row in _METRICS_CLUSTER:
    if row[0] in df.columns:
        METRICS.append((row[0], row[1], row[LANG_IDX], OUT_CLUSTER))
        _seen_mean_cols.add(row[0])

print(f"Metrics to plot: {len(METRICS)} "
      f"({sum(1 for m in METRICS if m[3]==OUT_RAW)} raw, "
      f"{sum(1 for m in METRICS if m[3]==OUT_CLUSTER)} cluster)")

ABLATION_METRICS = [
    (mean_col, std_col, label, OUT_ABLATION)
    for (mean_col, std_col, label, _out) in METRICS
    if mean_col in ABLATION_METRIC_COLS
]

# One folder per (experiment, dataset, n_forget), each filtered independently
group_keys = ["experiment", "dataset", "num_classes_to_forget"]
groups = df.groupby(group_keys)

for d in (OUT_RAW, OUT_CLUSTER, OUT_OTHERS, OUT_ABLATION):
    os.makedirs(d, exist_ok=True)

# Accumulate rows per output dir
from collections import defaultdict
csv_rows_by_dir = defaultdict(list)

def plot_group(experiment, dataset, n_forget, gdf, valid_versions, metrics, harmonic_dir):
    if not valid_versions:
        return

    # Build legend labels from the same short display names as src/plots/plot_boxplots.py.
    # Winners are resolved against every version present in the group (not just
    # the subset being plotted) so SSD/EWC keep their "LF-SSD"/"EWC" labels.
    bases_here = {base_version(v) for v in gdf["version"].unique()}
    resolved_winners = [resolve_ewc_embs_naming(w, bases_here) for w in SINGLE_WINNERS.get(experiment, [])]
    ssd_name = next((w for w in resolved_winners if w.startswith("SSD_MAS_")), None)
    ewc_name = next((w for w in resolved_winners if not w.startswith("SSD_MAS_")), None)
    legend_labels = [display_name(base_version(v), ssd_name, ewc_name) for v in valid_versions]

    for mean_col, std_col, metric_name, out_dir in metrics:

        exp_dir = os.path.join(out_dir, experiment)
        os.makedirs(exp_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (version, short_version) in enumerate(zip(valid_versions, legend_labels)):
            vdf = gdf[gdf["version"] == version].sort_values("steps")
            steps = vdf["steps"].values
            means = vdf[mean_col].values
            stds  = vdf[std_col].values if std_col is not None else np.zeros_like(means)

            color = method_color(short_version)

            if args.std and args.std_style == "fill":
                ax.plot(steps, means, label=short_version, **line_kwargs(short_version))
                ax.fill_between(
                    steps,
                    means - stds,
                    means + stds,
                    color=color,
                    alpha=args.std_alpha,
                    linewidth=0,
                    zorder=2,
                )
            else:
                ax.errorbar(steps, means, yerr=stds if args.std else None, label=short_version,
                            capsize=3, capthick=1, **line_kwargs(short_version))

        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(
            f"{metric_name}\n{arch_name(experiment)} | {dataset} | forget={n_forget}",
            fontsize=13,
        )
        ax.set_xticks(sorted(gdf["steps"].unique()))
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                           ncol=min(len(valid_versions), 5), fontsize=8, framealpha=0.8)
        bold_our_legend_entries(legend)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save (one folder per experiment)
        safe_name = (
            f"{dataset}_forget{n_forget}_{mean_col}"
            .replace("/", "_")
            .replace(" ", "_")
        )
        out_path = os.path.join(exp_dir, f"{safe_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

        out_path = os.path.join(exp_dir, f"{safe_name}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

        for version, short_version in zip(valid_versions, legend_labels):
            vdf = gdf[gdf["version"] == version].sort_values("steps")
            for _, row in vdf.iterrows():
                std_val = row[std_col] if std_col is not None else float("nan")
                csv_rows_by_dir[out_dir].append({
                    "experiment":            experiment,
                    "dataset":               dataset,
                    "num_classes_to_forget": n_forget,
                    "metric":                metric_name,
                    "version":               short_version,
                    "steps":                 int(row["steps"]),
                    "mean":                  row[mean_col],
                    "std":                   std_val,
                })

    # Harmonic mean of retention and forget accuracy (same criterion used to
    # pick the SSD_MAS_embs/EWC winners), computed per repetition from the raw
    # ICU results and then aggregated - so it gets a real mean/std, instead of
    # the harmonic mean of the already-aggregated means.
    hstats = harmonic_stats(experiment, dataset, n_forget, valid_versions)
    if not hstats.empty:
        harmonic_label = "Media armónica (retención/olvido) ↑" if args.lang == "es" else "H mean ↑"
        exp_dir = os.path.join(harmonic_dir, experiment)
        os.makedirs(exp_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (version, short_version) in enumerate(zip(valid_versions, legend_labels)):
            vdf = hstats[hstats["version"] == version].sort_values("steps")
            if vdf.empty:
                continue
            steps = vdf["steps"].values
            means = vdf["mean"].values
            stds = vdf["std"].fillna(0.0).values
            color = method_color(short_version)

            if args.std and args.std_style == "fill":
                ax.plot(steps, means, label=short_version, **line_kwargs(short_version))
                ax.fill_between(steps, means - stds, means + stds, color=color, alpha=args.std_alpha, linewidth=0, zorder=2)
            else:
                ax.errorbar(steps, means, yerr=stds if args.std else None, label=short_version,
                            capsize=3, capthick=1, **line_kwargs(short_version))

        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel(harmonic_label, fontsize=12)
        ax.set_title(f"{harmonic_label}\n{arch_name(experiment)} | {dataset} | forget={n_forget}", fontsize=13)
        ax.set_xticks(sorted(gdf["steps"].unique()))
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                           ncol=min(len(valid_versions), 5), fontsize=8, framealpha=0.8)
        bold_our_legend_entries(legend)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = f"{dataset}_forget{n_forget}_harmonic_mean".replace("/", "_").replace(" ", "_")
        for ext in ("png", "pdf"):
            out_path = os.path.join(exp_dir, f"{safe_name}.{ext}")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")
        plt.close(fig)

        for version, short_version in zip(valid_versions, legend_labels):
            vdf = hstats[hstats["version"] == version].sort_values("steps")
            for _, row in vdf.iterrows():
                csv_rows_by_dir[harmonic_dir].append({
                    "experiment":            experiment,
                    "dataset":               dataset,
                    "num_classes_to_forget": n_forget,
                    "metric":                harmonic_label,
                    "version":               short_version,
                    "steps":                 int(row["steps"]),
                    "mean":                  row["mean"],
                    "std":                   row["std"],
                })

def select_versions(gdf, ordered_bases):
    # Versions of gdf whose base name is in ordered_bases, in that order.
    by_base = {}
    for v in gdf["version"].unique():
        by_base.setdefault(base_version(v), []).append(v)
    out = []
    for b in ordered_bases:
        out.extend(sorted(by_base.get(b, [])))
    return out

for (experiment, dataset, n_forget), gdf in groups:
    all_versions = sorted(gdf["version"].unique())

    # Main plots: the proposed method vs. every other method, Gaussian (NOISY) excluded.
    main_versions = [v for v in all_versions if not is_noisy(base_version(v))]
    plot_group(experiment, dataset, n_forget, gdf, main_versions, METRICS, OUT_RAW)

    # Ablation: the proposed method vs. Gaussian (NOISY), same hyperparameters.
    available_bases = {base_version(v) for v in all_versions}
    resolved_ablation = [resolve_notop_fallback(m, available_bases) for m in ABLATION_METHODS]
    ablation_versions = select_versions(gdf, list(dict.fromkeys(resolved_ablation)))
    plot_group(experiment, dataset, n_forget, gdf, ablation_versions, ABLATION_METRICS, OUT_ABLATION)

dirs_used = set()
for out_dir, rows in csv_rows_by_dir.items():
    if rows:
        csv_out = os.path.join(out_dir, "plotted_data.csv")
        pd.DataFrame(rows).to_csv(csv_out, index=False)
        print(f"Data saved to '{csv_out}'")
        dirs_used.add(out_dir)
print(f"\nAll plots saved in: {sorted(dirs_used)} (one subfolder per experiment)")
