#!/usr/bin/env python3
"""
Figures of the paired ablations of src/analysis/ablation_compare.py, drawn like the
Gaussian (NOISY) ablation of src/plots/plot_boxplots.py / plot_figures.py:

    A side (the proposed method)    : DISCERN (notop / conf.)
    B side, --comparison flujo      : CV-only (notop / conf.)   PARAMETRIC_cv5x3_tbn_m0.05
            --comparison noisy      : Gaussian (notop / conf.)  NOISY_1000_..._ledoit_wolf

  single / multiple : boxplots per architecture x dataset (x k) of Accuracy (retain),
                      Accuracy (forget) and H mean, four boxes paired by loss (A, B).
  ICU               : mean +- std over the sequences per step, same three metrics,
                      only the steps both sides have.

Only matched items (both sides present for the same architecture / dataset / classes /
freezing mode) are used.  efficientnet_v2_s runs trained with the whole backbone are
drawn separately, in the folder efficientnet_v2_s_nofreeze.  As in plot_boxplots.py, a
version whose mean forget accuracy is > 0.2 is left out of the forget / H BOXPLOTS
(printed; --keep-all to keep it); the ICU curves, like plot_figures.py, show everything.

Output: results/ablation/<comparison>/plots/{single,multiple,ICU}/<architecture>/*.{png,pdf}

    PYTHONPATH=. python src/plots/plot_ablation_compare.py --comparison noisy
(also run at the end of src/analysis/ablation_compare.py)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from method_colors import (BOX_KWARGS, arch_name, bold_our_legend_entries, is_ours,  # noqa: E402
                           line_kwargs, method_color, style_box)
from src.utils.proposed_method import PROPOSED_CONF_DISPLAY, PROPOSED_NOTOP_DISPLAY  # noqa: E402

METRICS = [("acc_r", "Accuracy (retain) ↑"), ("acc_f", "Accuracy (forget) ↓"), ("H", "H mean ↑")]
FORGET_MAX = 0.2   # same rule as the Gaussian ablation (plot_boxplots.methods_for)


def labels_for(spec):
    lab = {(spec["a"], "notop"): PROPOSED_NOTOP_DISPLAY, (spec["a"], "confidence"): PROPOSED_CONF_DISPLAY,
           (spec["b"], "notop"): spec["b_labels"]["notop"], (spec["b"], "confidence"): spec["b_labels"]["confidence"]}
    order = [lab[(spec["a"], "notop")], lab[(spec["b"], "notop")],
             lab[(spec["a"], "confidence")], lab[(spec["b"], "confidence")]]
    return lab, order


def matched(df, labels, extra_keys=()):
    """Rows whose (loss, freeze, item[, extra]) exists for both sides; adds 'label' / 'folder'."""
    keys = ["loss", "freeze", "item", *extra_keys]
    df = df[df.groupby(keys)["side"].transform("nunique") == 2].copy()
    df["label"] = [labels[(s, l)] for s, l in zip(df["side"], df["loss"])]
    df["folder"] = df["experiment"] + np.where(df["freeze"] == "nofreeze", "_nofreeze", "")
    return df


KEEP_ALL = False   # --keep-all: no forget-accuracy rule in the boxplots either


def kept_for_forget(gdf, labels, where):
    if KEEP_ALL:
        return list(labels)
    kept = []
    for lbl in labels:
        if gdf.loc[gdf["label"] == lbl, "acc_f"].mean() <= FORGET_MAX:
            kept.append(lbl)
        else:
            print(f"[INFO] {where}: {lbl} left out of the forget / H plots (mean forget accuracy > {FORGET_MAX})")
    return kept


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def title_arch(folder):
    return arch_name(folder.replace("_nofreeze", "")) + (" (whole backbone)" if folder.endswith("_nofreeze") else "")


def boxplots(df, order, group_keys, out_root):
    if df.empty:
        return
    for key, gdf in df.groupby(["folder", *group_keys], sort=True):
        folder, rest = key[0], key[1:]
        group_label = " | ".join([str(rest[0])] + [f"forget={e}" for e in rest[1:]])
        suffix = "_".join(str(r) for r in rest)
        present = [lbl for lbl in order if lbl in set(gdf["label"])]
        if not present:
            continue
        forget_kept = kept_for_forget(gdf, present, f"{folder} | {group_label}")
        for metric, ylabel in METRICS:
            kept = present if metric == "acc_r" else forget_kept
            if not kept:
                continue
            data = [gdf.loc[gdf["label"] == lbl, metric].dropna().values for lbl in kept]
            fig, ax = plt.subplots(figsize=(max(8, len(kept) * 1.2), 6))
            bp = ax.boxplot(data, labels=kept, **BOX_KWARGS)
            for box, lbl in zip(bp["boxes"], kept):
                style_box(box, lbl)
            n_pairs = gdf.loc[gdf["label"] == kept[0], "item"].nunique()
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{ylabel}\n{title_arch(folder)} | {group_label} | n={n_pairs} paired runs", fontsize=13)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelsize=9, rotation=30)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
                if is_ours(tick.get_text()):
                    tick.set_fontweight("bold")
            plt.tight_layout()
            save(fig, os.path.join(out_root, folder), f"{suffix}_{metric}")
    print(f"Saved boxplots in {out_root}/")


def icu_curves(icu, order, out_root):
    if icu.empty:
        return
    for (folder, dataset, k), gdf in icu.groupby(["folder", "dataset", "k"], sort=True):
        # every version in every metric: the ICU ablation of plot_figures.py does not
        # apply the forget-accuracy > 0.2 rule of the boxplots
        kept = [lbl for lbl in order if lbl in set(gdf["label"])]
        n_seq = gdf["item"].nunique()
        for metric, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(12, 6))
            for lbl in kept:
                s = gdf[gdf["label"] == lbl].groupby("steps")[metric].agg(["mean", "std"]).sort_index()
                ax.plot(s.index, s["mean"], label=lbl, **line_kwargs(lbl))
                ax.fill_between(s.index, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                                color=method_color(lbl), alpha=0.12, linewidth=0, zorder=2)
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{ylabel}\n{title_arch(folder)} | {dataset} | forget={k} | "
                         f"{n_seq} sequences (mean ± std)", fontsize=13)
            ax.set_xticks(sorted(gdf["steps"].unique()))
            legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(kept),
                               fontsize=9, framealpha=0.8)
            bold_our_legend_entries(legend)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save(fig, os.path.join(out_root, folder), f"{dataset}_forget{k}_{metric}")
    print(f"Saved ICU curves in {out_root}/")


def make_all(single, multi, icu, spec, out_dir, keep_all=False):
    """single / multi / icu: tidy frames from src/analysis/ablation_compare.py."""
    global KEEP_ALL
    KEEP_ALL = keep_all
    labels, order = labels_for(spec)
    boxplots(matched(single, labels), order, ["dataset"], os.path.join(out_dir, "single"))
    boxplots(matched(multi, labels), order, ["dataset", "k"], os.path.join(out_dir, "multiple"))
    icu = icu.copy()
    icu["k"] = icu["k"].astype(int)
    # a sequence must have both sides at the same step to be compared step by step
    icu_curves(matched(icu, labels, extra_keys=("steps",)), order, os.path.join(out_dir, "ICU"))


def main(argv=None):
    from src.analysis.ablation_compare import COMPARISONS, load_icu, load_processed
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", choices=list(COMPARISONS), required=True)
    ap.add_argument("--single", nargs="+", default=["results/unlearning_results_single_parametric.csv",
                                                     "results/unlearning_results_single_all_new.csv"])
    ap.add_argument("--multiple", nargs="+", default=["results/unlearning_results_multiple_parametric.csv",
                                                       "results/unlearning_results_multiple_all_new.csv"])
    ap.add_argument("--icu-root", default="results_ICU")
    ap.add_argument("--out-dir", default=None, help="default results/ablation/<comparison>/plots")
    ap.add_argument("--keep-all", action="store_true", help="no forget-accuracy > 0.2 rule in the boxplots")
    args = ap.parse_args(argv)
    spec = COMPARISONS[args.comparison]
    make_all(load_processed(args.single, False, spec), load_processed(args.multiple, True, spec),
             load_icu(args.icu_root, spec), spec,
             args.out_dir or os.path.join("results", "ablation", args.comparison, "plots"), args.keep_all)


if __name__ == "__main__":
    main()
