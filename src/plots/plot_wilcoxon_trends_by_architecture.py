#!/usr/bin/env python3
"""Same heatmap as src/plots/plot_wilcoxon_trends.py, but columns are architecture
family (conv, transformer) x dataset instead of experiment x dataset —
reads the *_by_architecture.csv summaries produced by
analyze_wilcoxon_unlearning*.py's --output-summary-by-architecture /
--output-summary-distance-by-architecture.

Runs once per split ("single", "multi", "both"), same convention as
src/plots/plot_wilcoxon_trends.py. Only the forget-case by-architecture summary exists
today (retain/unknown3 --output-*-by-architecture haven't been generated
yet); a case with no file(s) is skipped with a warning rather than failing,
so this picks the others up automatically once/if they're added.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_FILES = {
    ("forget",  "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_by_architecture.csv",
    ("forget",  "inter"): "results/wilcox/wilcoxon_unpaired_inter_by_architecture.csv",
    ("retain",  "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_retain_by_architecture.csv",
    ("retain",  "inter"): "results/wilcox/wilcoxon_unpaired_inter_retain_by_architecture.csv",
    ("unknown", "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_unknown3_by_architecture.csv",
    ("unknown", "inter"): "results/wilcox/wilcoxon_unpaired_inter_unknown3_by_architecture.csv",
}

SPLITS      = ["single", "multi", "both"]
CASES       = ["forget", "retain", "unknown"]
FAMILIES    = ["conv", "transformer"]
BASELINES   = ["original", "retrain"]
METRIC_KEYS = [("intra", "l2"), ("intra", "cos"), ("inter", "l2"), ("inter", "cos")]

CASE_LABELS = {"forget": "Forget", "retain": "Retain", "unknown": "Unknown"}
METRIC_LABELS = {
    ("intra", "l2"):  "Intra L2",
    ("intra", "cos"): "Intra Cos",
    ("inter", "l2"):  "Inter L2",
    ("inter", "cos"): "Inter Cos",
}
FAMILY_LABELS = {"conv": "Conv", "transformer": "Transformer"}


def apply_split_suffix(path: str, split: str) -> str:
    if split == "single":
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{split}{ext}"


def process_split(split: str) -> None:
    files = {key: apply_split_suffix(path, split) for key, path in BASE_FILES.items()}

    dfs: list[pd.DataFrame] = []
    for (case, mtype), path in files.items():
        if not os.path.exists(path):
            print(f"[warn] split='{split}': missing {path} (case={case}, {mtype}) — skipping it.")
            continue
        d = pd.read_csv(path)
        if d.empty:
            print(f"[warn] split='{split}': {path} has no rows yet — skipping it.")
            continue
        d["case"] = case
        d["metric_type"] = mtype
        dfs.append(d)

    if not dfs:
        print(f"[warn] split='{split}': no by-architecture summary files available yet — skipping plots.")
        return

    all_df = pd.concat(dfs, ignore_index=True)
    present_cases = [c for c in CASES if c in set(all_df["case"])]

    all_df["metric"]   = all_df["group_name"].str.split("_vs_").str[0]
    all_df["baseline"] = all_df["group_name"].str.split("_vs_").str[1]

    key_cols = ["case", "metric_type", "metric", "baseline", "architecture_family", "dataset"]

    gr = (
        all_df[all_df["alternative"] == "greater"]
        .set_index(key_cols)[["reject_h0", "median_clean", "p_value"]]
        .rename(columns={"reject_h0": "rej_g", "median_clean": "median", "p_value": "p_g"})
    )
    ls = (
        all_df[all_df["alternative"] == "less"]
        .set_index(key_cols)[["reject_h0", "p_value"]]
        .rename(columns={"reject_h0": "rej_l", "p_value": "p_l"})
    )

    trend_df = gr.join(ls, how="outer").reset_index()

    def assign_trend(row) -> int:
        g = bool(row["rej_g"]) if pd.notna(row["rej_g"]) else False
        l = bool(row["rej_l"]) if pd.notna(row["rej_l"]) else False
        if g and not l:
            return 1
        if l and not g:
            return -1
        if g and l:
            return 1 if row["median"] >= 0 else -1
        return 0

    trend_df["trend"] = trend_df.apply(assign_trend, axis=1)

    DATASETS = sorted(trend_df["dataset"].unique())

    col_keys   = [(fam, ds) for fam in FAMILIES for ds in DATASETS]
    col_labels = [f"{FAMILY_LABELS[fam]}\n{ds}" for fam, ds in col_keys]

    row_keys   = [(case, mtype, met) for case in present_cases for mtype, met in METRIC_KEYS]
    row_labels = [
        f"{CASE_LABELS[case]}\n{METRIC_LABELS[(mtype, met)]}"
        for case, mtype, met in row_keys
    ]

    CMAP  = ListedColormap(["#d73027", "#eeeeee", "#1a9850"])
    NORM  = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CMAP.N)

    for baseline in BASELINES:
        sub = trend_df[trend_df["baseline"] == baseline]

        n_rows = len(row_keys)
        n_cols = len(col_keys)

        trend_mat  = np.zeros((n_rows, n_cols), dtype=float)
        median_mat = np.full((n_rows, n_cols), np.nan)

        for ri, (case, mtype, met) in enumerate(row_keys):
            for ci, (fam, ds) in enumerate(col_keys):
                row = sub[
                    (sub["case"] == case)
                    & (sub["metric_type"] == mtype)
                    & (sub["metric"] == met)
                    & (sub["architecture_family"] == fam)
                    & (sub["dataset"] == ds)
                ]
                if not row.empty:
                    trend_mat[ri, ci]  = float(row["trend"].iloc[0])
                    median_mat[ri, ci] = float(row["median"].iloc[0])

        fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_cols), max(4, 0.75 * n_rows)))

        ax.imshow(trend_mat, cmap=CMAP, norm=NORM, aspect="auto")

        for ri in range(n_rows):
            for ci in range(n_cols):
                v = median_mat[ri, ci]
                txt = f"{v:+.3f}" if not np.isnan(v) else "n/a"
                trend_val = trend_mat[ri, ci]
                text_color = "white" if abs(trend_val) == 1 else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.tick_params(axis="x", which="both", top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Vertical separator between architecture families
        ax.axvline(len(DATASETS) - 0.5, color="black", linewidth=1.5, linestyle="--")

        # Horizontal separators between cases
        for i in range(1, len(present_cases)):
            y = i * len(METRIC_KEYS) - 0.5
            ax.axhline(y, color="black", linewidth=1.5, linestyle="--")

        # No title: the top tick labels are the only header (split/baseline/
        # reference are already in the output file name).

        patches = [
            mpatches.Patch(facecolor="#1a9850", label="↑ significant increase"),
            mpatches.Patch(facecolor="#eeeeee", edgecolor="grey", linewidth=0.5,
                           label="— no significant trend"),
            mpatches.Patch(facecolor="#d73027", label="↓ significant decrease"),
        ]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=True)

        plt.tight_layout()
        out_path = apply_split_suffix(f"results/wilcoxon_trend_heatmap_by_architecture_{baseline}.pdf", split)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        out_path = apply_split_suffix(f"results/wilcoxon_trend_heatmap_by_architecture_{baseline}.png", split)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    for split in SPLITS:
        print(f"\n=== split: {split} ===")
        process_split(split)
