#!/usr/bin/env python3
"""Plot how the proposed methods (DISTRIBUTIONAL) and their gaussian-noise
counterpart (NOISY) behave against the majority trend of the comparison
methods, using the same layout as src/plots/plot_wilcoxon_trends.py so both figures can
be read side by side.

Cell colour is the reference method's own direction:
  • GREEN  (↑)  –  disperse   (mean delta > 0)
  • RED    (↓)  –  concentrate (mean delta < 0)
  • GREY        –  the method was not run for that cell

A hatched cell marks a disagreement with the majority direction of the
comparison methods; a plain cell means the reference follows the general trend.
Cells are annotated with the reference's mean delta.

One figure per (reference method, baseline). References absent from every cell
are skipped, so a split where NOISY was never run simply produces no NOISY
figure.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from method_colors import arch_name

BASE_FILES = {
    "forget": "results/wilcox/reference_method_trends.csv",
    "retain": "results/wilcox/reference_method_trends_retain.csv",
    "unknown": "results/wilcox/reference_method_trends_unknown3.csv",
}

SPLITS = ["single", "multi", "both"]
CASES = ["forget", "retain", "unknown"]
BASELINES = ["original", "retrain"]
METRIC_KEYS = [("intra", "l2"), ("intra", "cos"), ("inter", "l2"), ("inter", "cos")]

CASE_LABELS = {"forget": "Forget", "retain": "Retain", "unknown": "Unknown"}
METRIC_LABELS = {
    ("intra", "l2"): "Intra L2",
    ("intra", "cos"): "Intra Cos",
    ("inter", "l2"): "Inter L2",
    ("inter", "cos"): "Inter Cos",
}


def apply_split_suffix(path: str, split: str) -> str:
    if split == "single":
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{split}{ext}"


def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def process_split(split: str) -> None:
    frames: list[pd.DataFrame] = []
    for case, path in BASE_FILES.items():
        p = apply_split_suffix(path, split)
        if not os.path.exists(p):
            print(f"[warn] split='{split}': missing {p} (case={case}) — skipping it.")
            continue
        d = pd.read_csv(p)
        if d.empty:
            print(f"[warn] split='{split}': {p} has no rows — skipping it.")
            continue
        d["case"] = case
        frames.append(d)

    if not frames:
        print(f"[warn] split='{split}': no reference-trend files available — skipping plots.")
        return

    df = pd.concat(frames, ignore_index=True)
    df["metric_type"] = np.where(
        df["metric"].str.contains("cohesion"), "intra", "inter"
    )
    df["metric_kind"] = np.where(df["metric"].str.endswith("_cos"), "cos", "l2")
    df["baseline"] = df["group_name"].str.split("_vs_").str[1]

    present_cases = [c for c in CASES if c in set(df["case"])]
    experiments = sorted(df["experiment"].unique())
    datasets = sorted(df["dataset"].unique())
    col_keys = [(e, d) for e in experiments for d in datasets]
    col_labels = [f"{arch_name(e)}\n{d}" for e, d in col_keys]

    row_keys = [(c, mt, mk) for c in present_cases for mt, mk in METRIC_KEYS]
    row_labels = [
        f"{CASE_LABELS[c]}\n{METRIC_LABELS[(mt, mk)]}" for c, mt, mk in row_keys
    ]

    CMAP = ListedColormap(["#d73027", "#eeeeee", "#1a9850"])
    NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CMAP.N)

    for reference in sorted(df["reference"].unique()):
        sub_ref = df[df["reference"] == reference]
        if (sub_ref["direction"] == "absent").all():
            print(f"[warn] split='{split}': '{reference}' absent everywhere — no figure.")
            continue

        for baseline in BASELINES:
            sub = sub_ref[sub_ref["baseline"] == baseline]
            if sub.empty:
                continue

            n_rows, n_cols = len(row_keys), len(col_keys)
            direction_mat = np.zeros((n_rows, n_cols))
            value_mat = np.full((n_rows, n_cols), np.nan)
            disagree_mat = np.zeros((n_rows, n_cols), dtype=bool)

            for ri, (case, mtype, mkind) in enumerate(row_keys):
                for ci, (exp, ds) in enumerate(col_keys):
                    row = sub[
                        (sub["case"] == case)
                        & (sub["metric_type"] == mtype)
                        & (sub["metric_kind"] == mkind)
                        & (sub["experiment"] == exp)
                        & (sub["dataset"] == ds)
                    ]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    if r["direction"] == "absent":
                        continue
                    direction_mat[ri, ci] = 1.0 if r["direction"] == "disperse" else -1.0
                    value_mat[ri, ci] = float(r["mean_delta"])
                    disagree_mat[ri, ci] = r["agrees_with_majority"] is False or (
                        str(r["agrees_with_majority"]).lower() == "false"
                    )

            fig, ax = plt.subplots(
                figsize=(max(8, 1.5 * n_cols), max(5, 0.75 * n_rows))
            )
            ax.imshow(direction_mat, cmap=CMAP, norm=NORM, aspect="auto")

            for ri in range(n_rows):
                for ci in range(n_cols):
                    v = value_mat[ri, ci]
                    txt = f"{v:+.3f}" if not np.isnan(v) else "n/a"
                    colour = "white" if abs(direction_mat[ri, ci]) == 1 else "black"
                    ax.text(ci, ri, txt, ha="center", va="center",
                            fontsize=7, color=colour, fontweight="bold")
                    if disagree_mat[ri, ci]:
                        ax.add_patch(
                            mpatches.Rectangle(
                                (ci - 0.5, ri - 0.5), 1, 1, fill=False,
                                hatch="////", edgecolor="black", linewidth=0,
                            )
                        )

            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(col_labels, fontsize=8)
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.tick_params(axis="x", which="both", top=True, bottom=False,
                           labeltop=True, labelbottom=False)

            for i in range(1, len(present_cases)):
                ax.axhline(i * len(METRIC_KEYS) - 0.5, color="black",
                           linewidth=1.5, linestyle="--")

            # No title: the top tick labels are the only header (split/baseline/
            # reference are already in the output file name).

            patches = [
                mpatches.Patch(facecolor="#d73027", label="↓ concentrates"),
                mpatches.Patch(facecolor="#1a9850", label="↑ disperses"),
                mpatches.Patch(facecolor="#eeeeee", edgecolor="grey",
                               linewidth=0.5, label="not run"),
                mpatches.Patch(facecolor="white", edgecolor="black", hatch="////",
                               label="disagrees with majority"),
            ]
            ax.legend(handles=patches, loc="lower center",
                      bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=True)

            plt.tight_layout()
            stem = f"results/reference_method_heatmap_{slug(reference)}_{baseline}"
            for ext in (".pdf", ".png"):
                out = apply_split_suffix(stem + ext, split)
                plt.savefig(out, bbox_inches="tight", dpi=150)
            print(f"Saved: {apply_split_suffix(stem + '.png', split)}")
            plt.close(fig)


if __name__ == "__main__":
    for split in SPLITS:
        print(f"\n=== split: {split} ===")
        process_split(split)
