#!/usr/bin/env python3
"""
Evolution of the metrics along each hyper-parameter of the DISCERN study
(src/analysis/analyze_hparam_study.py --parametric-tag ...).

For every study (margin, synth_notop, synth_ccp):
  <study>_evolution.{png,pdf}   1 x 3: Accuracy (retain) ↑ | Accuracy (forget) ↓ | H ↑
                                one line per architecture (pooled over datasets, 95% CI) +
                                the overall mean (black) with its 95% bootstrap CI
  <study>_by_dataset.{png,pdf}  3 x 3: rows = datasets, same columns, one line per
                                architecture with its 95% bootstrap CI
The arrow in each panel title says which direction is better; the dashed vertical
line marks DISCERN's default (margin 10 / 1000 synthetic samples).

Inputs : results/ablation/hparam/summary_mean_sd_ci.csv
Output : results/ablation/hparam/plots/

    PYTHONPATH=. python src/plots/plot_hparam_evolution.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from method_colors import arch_name  # noqa: E402

# study: (method, factor column, default level, fixed column, fixed value, x label, log x)
STUDIES = {
    "margin": ("notop", "margin", 10.0, "n_synthetic", 1000, "notop margin", False),
    "synth_notop": ("notop", "n_synthetic", 1000, "margin", 10.0, "synthetic samples per class (notop)", True),
    "synth_ccp": ("corrected_confidence_penalty", "n_synthetic", 1000, None, None,
                  "synthetic samples per class (confidence)", True),
}
TITLES = {"margin": "notop margin", "synth_notop": "Number of synthetic samples — notop",
          "synth_ccp": "Number of synthetic samples — confidence"}
# (summary column prefix, raw column, panel title with the "better" arrow)
METRICS = [("acc_retain", "accuracy_score_after", "Accuracy (retain)  ↑ higher is better"),
           ("acc_forget", "unlearning_accuracy_score", "Accuracy (forget)  ↓ lower is better"),
           ("H", "harmonic", "H mean  ↑ higher is better")]
ARCHS = ["resnet18", "efficientnet_v2_s", "swin_t", "vit_b_16"]
# categorical slots 1-4 of the validated reference palette (fixed order) + distinct markers
ARCH_STYLE = {"resnet18": ("#2a78d6", "o"), "efficientnet_v2_s": ("#eb6834", "s"),
              "swin_t": ("#1baf7a", "^"), "vit_b_16": ("#eda100", "D")}
DATASETS = ["cifar10", "cifar100", "mnist"]
DS_NAME = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "mnist": "MNIST"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"




def style_axis(ax, study, ylabel=None):
    _, _, default, _, _, xlabel, logx = STUDIES[study]
    if logx:
        ax.set_xscale("log")
    ax.axvline(default, color=INK, lw=2, ls=(0, (5, 3)), alpha=0.7, zorder=1)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)


def set_level_ticks(ax, levels, study):
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v:g}" for v in levels])
    ax.minorticks_off()


def arch_line(ax, x, y, arch, label=True):
    color, marker = ARCH_STYLE[arch]
    ax.plot(x, y, color=color, marker=marker, ms=6, lw=2, zorder=3,
            label=arch_name(arch) if label else None)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/{name}.png/.pdf")


def plot_evolution(summ, study, out_dir):
    s = summ[summ["study"] == study]
    levels = sorted(s["level"].unique())
    allrows = s[s["scope_type"] == "all"].sort_values("level")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (pref, _, title) in zip(axes, METRICS):
        for arch in ARCHS:
            a = s[(s["scope_type"] == "architecture") & (s["scope"] == arch)].sort_values("level")
            if a.empty:
                continue
            ax.fill_between(a["level"], a[f"{pref}_ci_lo"], a[f"{pref}_ci_hi"],
                            color=ARCH_STYLE[arch][0], alpha=0.10, lw=0, zorder=2)
            arch_line(ax, a["level"], a[f"{pref}_mean"], arch)
        ax.fill_between(allrows["level"], allrows[f"{pref}_ci_lo"], allrows[f"{pref}_ci_hi"],
                        color=INK, alpha=0.12, lw=0, zorder=2)
        ax.plot(allrows["level"], allrows[f"{pref}_mean"], color=INK, lw=2.6, marker="o", ms=5,
                zorder=4, label="All (mean, 95% CI)")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style_axis(ax, study)
        set_level_ticks(ax, levels, study)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, ls=(0, (5, 3)), lw=2, alpha=0.7))
    labels.append("DISCERN default")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"DISCERN — {TITLES[study]} (single-class forgetting, pooled over datasets)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out_dir, f"{study}_evolution")


def plot_by_dataset(summ, study, out_dir):
    cells = summ[(summ["study"] == study) & (summ["scope_type"] == "cell")].copy()
    cells[["architecture", "dataset"]] = cells["scope"].str.split("|", expand=True)
    levels = sorted(cells["level"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for i, ds in enumerate(DATASETS):
        for j, (pref, _, title) in enumerate(METRICS):
            ax = axes[i, j]
            for arch in ARCHS:
                c = cells[(cells["architecture"] == arch) & (cells["dataset"] == ds)].sort_values("level")
                if c.empty:
                    continue
                color, _ = ARCH_STYLE[arch]
                ax.fill_between(c["level"], c[f"{pref}_ci_lo"], c[f"{pref}_ci_hi"],
                                color=color, alpha=0.15, lw=0, zorder=2)
                arch_line(ax, c["level"], c[f"{pref}_mean"], arch, label=(i == 0 and j == 0))
            style_axis(ax, study, ylabel=DS_NAME[ds] if j == 0 else None)
            set_level_ticks(ax, levels, study)
            if i == 0:
                ax.set_title(title, fontsize=11, color=INK, loc="left")
            if i < 2:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, ls=(0, (5, 3)), lw=2, alpha=0.7))
    labels.append("DISCERN default")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"DISCERN — {TITLES[study]}, by dataset (mean and 95% bootstrap CI over removed classes)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, out_dir, f"{study}_by_dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/ablation/hparam")
    ap.add_argument("--out", default=None, help="default <in-dir>/plots")
    args = ap.parse_args()
    summ = pd.read_csv(os.path.join(args.in_dir, "summary_mean_sd_ci.csv"))

    out_dir = args.out or os.path.join(args.in_dir, "plots")
    for study in STUDIES:
        plot_evolution(summ, study, out_dir)
        plot_by_dataset(summ, study, out_dir)


if __name__ == "__main__":
    main()
