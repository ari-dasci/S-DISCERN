#!/usr/bin/env python3
"""
Boxplots per metric for the same methods selected in src/plots/extract_methods.py
(fixed method list + best SSD_MAS_embs / EWC_embsunfreeze per architecture,
picked from the single-class results and reused as-is for the multiclass
ones), but drawn from the raw per-repetition results instead of their mean,
so the spread across repetitions is visible.
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
import csv
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Boxplots per metric for the selected unlearning methods.")
parser.add_argument("--together", action="store_true",
                    help="Also save a combined figure with retention (left) and forget (right) accuracy side by side")
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────────────────────────
SINGLE_MEANS_SRC   = "unlearning_results_single_all_new_analyzed_means.csv"
MULTIPLE_MEANS_SRC = "unlearning_results_multiple_all_new_multiclass_analyzed_means.csv"
SINGLE_RAW_SRC      = "results/unlearning_results_single_all_new.csv"
MULTIPLE_RAW_SRC     = "results/unlearning_results_multiple_all_new.csv"

OUT_SINGLE   = "plots/boxplots_single"
OUT_MULTIPLE = "plots/boxplots_multiple"
OUT_ABLATION_SINGLE   = "plots/ablation/single"
OUT_ABLATION_MULTIPLE = "plots/ablation/multiple"
# Per-metric figures with every architecture side by side (subfolder of the
# main boxplot folders).
BY_METRIC_SUBDIR = "by_metric"

FIXED_METHODS = [
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

# RETRAIN/original and NOISY are excluded from the main boxplots (NOISY gets
# its own "ablation" comparison against the proposed method instead; RETRAIN and
# original are kept in FIXED_METHODS only for reference / other scripts).
_NOISY_CONF = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"
_NOISY_NOTOP = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
BOXPLOT_METHODS = [m for m in FIXED_METHODS if m not in ("RETRAIN", "original", _NOISY_CONF, _NOISY_NOTOP)]

# The ablation comparison: the proposed method vs. its Gaussian-noise ablation,
# paired by loss (notop / conf.) for a direct side-by-side comparison.
_DIST_NOTOP = PROPOSED_NOTOP   # proposed method (name kept for the existing code)
_DIST_CONF = PROPOSED_CONF
ABLATION_METHODS = [_DIST_NOTOP, _NOISY_NOTOP, _DIST_CONF, _NOISY_CONF]

# Display names for the legend/x-axis. The SSD_MAS_embs and EWC_embsunfreeze
# winners vary in exact hyperparameters per architecture, so they're mapped
# by role (ssd_name/ewc_name) rather than by exact string, see display_name().
DISPLAY_NAMES = {
    _DIST_NOTOP: PROPOSED_NOTOP_DISPLAY,
    _DIST_CONF: PROPOSED_CONF_DISPLAY,
    "DELETE_embs": "DELETE",
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs": "Boundary Shrink",
    "BOUNDARYEXPAND_lr1e-05_embs": "Boundary Expand",
    "BADTEACHER_T1.0_forgetOnly_embs": "Bad Teacher",
    "RANDOMLABEL_noret_embs_forbidden": "Random Labeling",
    # NOISY was renamed "Gaussian" for the ablation plot
    _NOISY_NOTOP: "Gaussian (notop)",
    _NOISY_CONF: "Gaussian (conf.)",
}

_MARGIN_SUFFIX_RE = re.compile(r"_margin_[0-9.]+.*$")

def display_name(base, ssd_name, ewc_name):
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

# One fixed color per method, consistent across every plot (shared with
# src/plots/plot_figures.py, see src/plots/method_colors.py).
from method_colors import (METHOD_ORDER, arch_name, style_box, legend_patch,
                           bold_our_legend_entries, BOX_KWARGS, is_ours)

# (raw column, label) - only columns that actually exist in the raw CSVs
# (no AUS/Brier there, those are only computed at the aggregation step)
METRICS = [
    ("accuracy_score_after",      "Accuracy (retain) ↑"),
    ("unlearning_accuracy_score", "Accuracy (forget) ↓"),
    ("maxprobability",            "Max probability (retain)"),
    ("maxprobability_forget",     "Max probability (forget)"),
    ("cohesion_forget_l2",        "Forget cohesion (L2)"),
    ("distance_inter_class_l2",   "Forget inter-class distance (L2)"),
    ("cohesion_forget_cos",       "Forget cohesion (cos)"),
    ("distance_inter_class_cos",  "Forget inter-class distance (cos)"),
]

HARMONIC_LABEL = "H mean ↑"

# Ablation-only metrics: accuracy pair + their harmonic mean (see harmonic()).
ABLATION_METRIC_LABELS = {
    "accuracy_score_after":      "Accuracy (retain) ↑",
    "unlearning_accuracy_score": "Accuracy (forget) ↓",
    "harmonic_mean":             HARMONIC_LABEL,
}

# ── Winner selection (same criterion as src/plots/extract_methods.py) ─────────────────

def base_version(v):
    return str(v).split('/')[0]

def harmonic(a, u):
    # unlearning_score is "lower is better", so invert it before combining
    # with acc_score ("higher is better") in the harmonic mean.
    a = float(a); u = 1.0 - float(u)
    if a + u == 0:
        return 0.0
    return 2 * a * u / (a + u)

def best_row(candidates):
    return max(candidates, key=lambda r: harmonic(r['accuracy_score_after_mean'], r['unlearning_accuracy_score_mean']))

def resolve_notop_fallback(name, available_bases):
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

_EWC_EMBS_NOTOP_RE = re.compile(r"_MAS_(embsunfreeze|embs)_notop$")

def pick_best_ssd(subset):
    candidates = [r for r in subset if base_version(r['version']).startswith("SSD_MAS_") and base_version(r['version']).endswith("_embs")]
    return base_version(best_row(candidates)['version']) if candidates else None

def pick_best_ewc(subset):
    # EWC's unfrozen-embedding variant is named "embsunfreeze" in some files
    # and plain "embs" in others - match both, but not "freeze" (note
    # "unfreeze" itself contains the substring "freeze").
    candidates = [r for r in subset if _EWC_EMBS_NOTOP_RE.search(base_version(r['version']))]
    return base_version(best_row(candidates)['version']) if candidates else None

def resolve_ewc_embs_naming(name, available_bases):
    # The EWC winner picked from single (embsunfreeze or embs) may show up
    # under the other naming in a different file (e.g. multiple only logging
    # "embs") for the same architecture - translate so the same run is still
    # matched instead of silently missing.
    if name is None or name in available_bases:
        return name
    if "embsunfreeze" in name:
        alt = name.replace("embsunfreeze", "embs")
    elif "_embs_" in name:
        alt = name.replace("_embs_", "_embsunfreeze_")
    else:
        return name
    return alt if alt in available_bases else name

with open(SINGLE_MEANS_SRC) as f:
    single_means_rows = list(csv.DictReader(f))

single_winners = {}
for arch, ds in sorted(set((r['experiment'], r['dataset']) for r in single_means_rows)):
    subset = [r for r in single_means_rows if r['experiment'] == arch and r['dataset'] == ds]
    single_winners[(arch, ds)] = (pick_best_ssd(subset), pick_best_ewc(subset))

def allowed_methods_for(arch, ds, available_bases):
    resolved_fixed = [resolve_notop_fallback(m, available_bases) for m in BOXPLOT_METHODS]
    ssd, ewc = single_winners.get((arch, ds), (None, None))
    ewc = resolve_ewc_embs_naming(ewc, available_bases)
    # dict.fromkeys: dedupe while keeping order
    return list(dict.fromkeys(resolved_fixed + [x for x in (ssd, ewc) if x]))

# ── Load raw per-repetition data ─────────────────────────────────────────────

def load_raw(path):
    df = add_embs_fallback(pd.read_csv(path))
    df = df[df["epoch"] == "best"].copy()
    m = df["id_database"].str.extract(r"^(?P<dataset>.+?)_classes_to_remove_(?P<tail>.+)$")
    df["dataset"] = m["dataset"]
    df["num_classes_removed"] = m["tail"].str.count("_") + 1
    df["base_version"] = df["version"].apply(base_version)
    return df

single_raw = load_raw(SINGLE_RAW_SRC)
multiple_raw = load_raw(MULTIPLE_RAW_SRC)

# ── Plotting ──────────────────────────────────────────────────────────────

def methods_for(metric, present, gdf, label_map, arch, group_label):
    if metric != "unlearning_accuracy_score":
        return present
    # A method whose forget accuracy strays far above the ~0.2 "actually
    # forgot" ballpark didn't really unlearn, so it's left out of this
    # specific plot (still shown in the others).
    kept, dropped = [], []
    for meth in present:
        mval = gdf.loc[gdf["base_version"] == meth, metric].mean()
        (kept if mval <= 0.2 else dropped).append(meth)
    if dropped:
        print(f"[INFO] {arch} | {group_label}: excluded from unlearning_accuracy_score boxplot (mean > 0.2): "
              f"{[label_map[m] for m in dropped]}")
    return kept

def draw_box(ax, gdf, methods, metric, label_map, ylabel):
    data = [gdf.loc[gdf["base_version"] == meth, metric].dropna().values for meth in methods]
    plot_labels = [label_map[meth] for meth in methods]
    bp = ax.boxplot(data, labels=plot_labels, **BOX_KWARGS)
    for box, lbl in zip(bp["boxes"], plot_labels):
        style_box(box, lbl)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        if is_ours(tick.get_text()):
            tick.set_fontweight("bold")

def make_boxplots(df, group_keys, out_root):
    os.makedirs(out_root, exist_ok=True)
    metric_labels = dict(METRICS)
    metrics = [(m, lbl) for m, lbl in METRICS if m in df.columns]
    for key, gdf in df.groupby(group_keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        arch, ds = key[0], key[1]
        extra = key[2:]

        available_bases = set(gdf["base_version"])
        allowed = allowed_methods_for(arch, ds, available_bases)
        present = [m for m in allowed if m in available_bases]
        if not present:
            continue

        ssd_name, ewc_name = single_winners.get((arch, ds), (None, None))
        ewc_name = resolve_ewc_embs_naming(ewc_name, available_bases)
        label_map = {m: display_name(m, ssd_name, ewc_name) for m in present}

        exp_dir = os.path.join(out_root, arch)
        os.makedirs(exp_dir, exist_ok=True)
        group_label = " | ".join([str(ds)] + [f"forget={e}" for e in extra])
        suffix = "_".join([str(ds)] + [str(e) for e in extra])

        kept_by_metric = {metric: methods_for(metric, present, gdf, label_map, arch, group_label) for metric, _ in metrics}

        metrics_to_plot = list(metrics)
        if "accuracy_score_after" in gdf.columns and "unlearning_accuracy_score" in gdf.columns:
            gdf = gdf.copy()
            gdf["harmonic_mean"] = [
                harmonic(a, u) for a, u in zip(gdf["accuracy_score_after"], gdf["unlearning_accuracy_score"])
            ]
            # Same eligibility as the forget-accuracy plot (excludes methods
            # whose forget accuracy strayed too far from the "actually forgot" range).
            kept_by_metric["harmonic_mean"] = kept_by_metric.get("unlearning_accuracy_score", present)
            metrics_to_plot.append(("harmonic_mean", HARMONIC_LABEL))

        for metric, label in metrics_to_plot:
            kept = kept_by_metric[metric]
            if not kept:
                continue
            fig, ax = plt.subplots(figsize=(max(8, len(kept) * 1.2), 6))
            draw_box(ax, gdf, kept, metric, label_map, label)
            ax.set_title(f"{label}\n{arch_name(arch)} | {group_label}", fontsize=13)
            plt.tight_layout()

            safe_name = f"{suffix}_{metric}".replace("/", "_").replace(" ", "_")
            for ext in ("png", "pdf"):
                out_path = os.path.join(exp_dir, f"{safe_name}.{ext}")
                fig.savefig(out_path, dpi=150)
                print(f"Saved: {out_path}")
            plt.close(fig)

        # Combined retention + forget accuracy figure: retention on the left,
        # forget on the right, sharing the same method colors/labels.
        if args.together and "accuracy_score_after" in kept_by_metric and "unlearning_accuracy_score" in kept_by_metric:
            retain_kept = kept_by_metric["accuracy_score_after"]
            forget_kept = kept_by_metric["unlearning_accuracy_score"]
            if retain_kept and forget_kept:
                fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(max(14, (len(retain_kept) + len(forget_kept)) * 1.2), 6))
                draw_box(ax_l, gdf, retain_kept, "accuracy_score_after", label_map, metric_labels["accuracy_score_after"])
                draw_box(ax_r, gdf, forget_kept, "unlearning_accuracy_score", label_map, metric_labels["unlearning_accuracy_score"])
                ax_l.set_title(metric_labels["accuracy_score_after"], fontsize=13)
                ax_r.set_title(metric_labels["unlearning_accuracy_score"], fontsize=13)
                fig.suptitle(f"{arch_name(arch)} | {group_label}", fontsize=13)
                plt.tight_layout()

                safe_name = f"{suffix}_accuracy_combined".replace("/", "_").replace(" ", "_")
                for ext in ("png", "pdf"):
                    out_path = os.path.join(exp_dir, f"{safe_name}.{ext}")
                    fig.savefig(out_path, dpi=150)
                    print(f"Saved: {out_path}")
                plt.close(fig)

def make_ablation_boxplots(df, group_keys, out_root):
    # The proposed method vs. its Gaussian-noise ablation: only the accuracy pair
    # and their harmonic mean, for now (other metrics can be added later).
    os.makedirs(out_root, exist_ok=True)
    for key, gdf in df.groupby(group_keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        arch, ds = key[0], key[1]
        extra = key[2:]

        available_bases = set(gdf["base_version"])
        resolved = [resolve_notop_fallback(m, available_bases) for m in ABLATION_METHODS]
        present = list(dict.fromkeys(m for m in resolved if m in available_bases))
        if not present:
            continue

        label_map = {m: display_name(m, None, None) for m in present}

        exp_dir = os.path.join(out_root, arch)
        os.makedirs(exp_dir, exist_ok=True)
        group_label = " | ".join([str(ds)] + [f"forget={e}" for e in extra])
        suffix = "_".join([str(ds)] + [str(e) for e in extra])

        gdf = gdf.copy()
        gdf["harmonic_mean"] = [
            harmonic(a, u) for a, u in zip(gdf["accuracy_score_after"], gdf["unlearning_accuracy_score"])
        ]

        retain_kept = present
        forget_kept = methods_for("unlearning_accuracy_score", present, gdf, label_map, arch, group_label)

        for metric, kept in (
            ("accuracy_score_after", retain_kept),
            ("unlearning_accuracy_score", forget_kept),
            ("harmonic_mean", forget_kept),
        ):
            if not kept:
                continue
            label = ABLATION_METRIC_LABELS[metric]
            fig, ax = plt.subplots(figsize=(max(8, len(kept) * 1.2), 6))
            draw_box(ax, gdf, kept, metric, label_map, label)
            ax.set_title(f"{label}\n{arch_name(arch)} | {group_label}", fontsize=13)
            plt.tight_layout()

            safe_name = f"{suffix}_{metric}".replace("/", "_").replace(" ", "_")
            for ext in ("png", "pdf"):
                out_path = os.path.join(exp_dir, f"{safe_name}.{ext}")
                fig.savefig(out_path, dpi=150)
                print(f"Saved: {out_path}")
            plt.close(fig)

def label_arch_rows(adf, arch, ds):
    # Rows of one architecture restricted to its selected methods, with the
    # short display name in "method_label" so methods can be matched across
    # architectures (SSD/EWC winners differ in exact name per architecture).
    available_bases = set(adf["base_version"])
    allowed = allowed_methods_for(arch, ds, available_bases)
    present = [m for m in allowed if m in available_bases]
    ssd_name, ewc_name = single_winners.get((arch, ds), (None, None))
    ewc_name = resolve_ewc_embs_naming(ewc_name, available_bases)
    label_map = {m: display_name(m, ssd_name, ewc_name) for m in present}
    out = adf[adf["base_version"].isin(present)].copy()
    out["method_label"] = out["base_version"].map(label_map)
    return out

def draw_grouped_box(ax, frames, archs, labels, metric, ylabel):
    # One block of boxes per architecture, one box per method inside each
    # block. A method missing for an architecture leaves its slot empty so
    # the method positions stay aligned across blocks.
    n = len(labels)
    for i, arch in enumerate(archs):
        adf = frames[arch]
        for j, lbl in enumerate(labels):
            vals = adf.loc[adf["method_label"] == lbl, metric].dropna().values
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[i * (n + 1) + j], widths=0.8,
                            manage_ticks=False, **BOX_KWARGS)
            style_box(bp["boxes"][0], lbl)
        if i > 0:
            ax.axvline(i * (n + 1) - 1, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks([i * (n + 1) + (n - 1) / 2 for i in range(len(archs))])
    ax.set_xticklabels([arch_name(a) for a in archs], fontsize=11)
    ax.set_xlim(-1, len(archs) * (n + 1) - 1)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

def add_method_legend(fig, labels):
    handles = [legend_patch(l) for l in labels]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(labels), 5), fontsize=10, frameon=False)
    bold_our_legend_entries(legend)

def save_fig(fig, out_dir, safe_name):
    for ext in ("png", "pdf"):
        out_path = os.path.join(out_dir, f"{safe_name}.{ext}")
        # bbox_inches="tight" so the legend placed below the axes is kept
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)

def make_boxplots_by_metric(df, group_keys, out_root):
    # Same methods as make_boxplots(), but one figure per metric with every
    # architecture side by side. No method is dropped here (not even the
    # forget-accuracy > 0.2 ones), and an extra combined figure stacks
    # retain accuracy, forget accuracy and H mean.
    out_dir = os.path.join(out_root, BY_METRIC_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["harmonic_mean"] = [
        harmonic(a, u) for a, u in zip(df["accuracy_score_after"], df["unlearning_accuracy_score"])
    ]
    metrics = [(m, lbl) for m, lbl in METRICS if m in df.columns] + [("harmonic_mean", HARMONIC_LABEL)]
    metric_labels = dict(metrics)

    for key, gdf in df.groupby(group_keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ds, extra = key[0], key[1:]

        frames = {}
        for arch, adf in gdf.groupby("experiment", sort=True):
            labelled = label_arch_rows(adf, arch, ds)
            if not labelled.empty:
                frames[arch] = labelled
        if not frames:
            continue
        archs = list(frames)
        found = set().union(*(set(f["method_label"]) for f in frames.values()))
        labels = [l for l in METHOD_ORDER if l in found] + sorted(found - set(METHOD_ORDER))

        # No dataset name in the title, only the forget count (multiple).
        group_label = " | ".join(f"forget={e}" for e in extra)
        suffix = "_".join([str(ds)] + [str(e) for e in extra])
        width = max(12, len(archs) * (len(labels) + 1) * 0.35)

        for metric, label in metrics:
            fig, ax = plt.subplots(figsize=(width, 6))
            draw_grouped_box(ax, frames, archs, labels, metric, label)
            ax.set_title(f"{label}\n{group_label}" if group_label else label, fontsize=13)
            plt.tight_layout()
            add_method_legend(fig, labels)
            save_fig(fig, out_dir, f"{suffix}_{metric}".replace("/", "_").replace(" ", "_"))

        combined = ["accuracy_score_after", "unlearning_accuracy_score", "harmonic_mean"]
        if all(m in metric_labels for m in combined):
            fig, axes = plt.subplots(len(combined), 1, figsize=(width, 5 * len(combined)), sharex=True)
            for ax, metric in zip(axes, combined):
                draw_grouped_box(ax, frames, archs, labels, metric, metric_labels[metric])
                ax.set_title(metric_labels[metric], fontsize=13)
            if group_label:
                fig.suptitle(group_label, fontsize=14)
            plt.tight_layout()
            add_method_legend(fig, labels)
            save_fig(fig, out_dir, f"{suffix}_accuracy_hmean_combined".replace("/", "_").replace(" ", "_"))

make_boxplots(single_raw, ["experiment", "dataset"], OUT_SINGLE)
make_boxplots(multiple_raw, ["experiment", "dataset", "num_classes_removed"], OUT_MULTIPLE)
make_boxplots_by_metric(single_raw, ["dataset"], OUT_SINGLE)
make_boxplots_by_metric(multiple_raw, ["dataset", "num_classes_removed"], OUT_MULTIPLE)
make_ablation_boxplots(single_raw, ["experiment", "dataset"], OUT_ABLATION_SINGLE)
make_ablation_boxplots(multiple_raw, ["experiment", "dataset", "num_classes_removed"], OUT_ABLATION_MULTIPLE)

print(f"\nAll boxplots saved in: {OUT_SINGLE}, {OUT_MULTIPLE}, {OUT_ABLATION_SINGLE} and {OUT_ABLATION_MULTIPLE} "
      f"(one subfolder per experiment, plus '{BY_METRIC_SUBDIR}/' with all architectures per metric)")
