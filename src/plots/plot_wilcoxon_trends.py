#!/usr/bin/env python3
"""Plot Wilcoxon trend directions (increase / no-trend / decrease) for each
case (forget, retain, unknown), metric (intra-L2, intra-cos, inter-L2, inter-cos)
and baseline (vs_original, vs_retrain), broken down by experiment × dataset.

Runs once per split ("single", "multi", "both") — matching the --split option
of analyze_wilcoxon_unlearning*.py — reading whichever summary CSVs exist for
that split ('single' keeps today's plain filenames; 'multi'/'both' read the
"_multi"/"_both" suffixed files). A split with no summary files yet (e.g.
multi-class results still being generated) is skipped with a warning instead
of failing.

Produces two heatmaps per split (one per baseline) where:
  • GREEN  (↑)  –  H0 rejected with alternative='greater'  →  delta > 0 (metric increases)
  • RED    (↓)  –  H0 rejected with alternative='less'     →  delta < 0 (metric decreases)
  • GREY   (—)  –  H0 not rejected in either direction     →  no significant trend

Cells are annotated with the median cleaned delta.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from method_colors import arch_name

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_FILES = {
    ("forget",  "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra.csv",
    ("forget",  "inter"): "results/wilcox/wilcoxon_unpaired_inter.csv",
    ("retain",  "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_retain.csv",
    ("retain",  "inter"): "results/wilcox/wilcoxon_unpaired_inter_retain.csv",
    ("unknown", "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_unknown3.csv",
    ("unknown", "inter"): "results/wilcox/wilcoxon_unpaired_inter_unknown3.csv",
}

SPLITS      = ["single", "multi", "both"]
CASES       = ["forget", "retain", "unknown"]
BASELINES   = ["original", "retrain"]
METRIC_KEYS = [("intra", "l2"), ("intra", "cos"), ("inter", "l2"), ("inter", "cos")]

CASE_LABELS = {"forget": "Forget", "retain": "Retain", "unknown": "Unknown"}
METRIC_LABELS = {
    ("intra", "l2"):  "Intra L2",
    ("intra", "cos"): "Intra Cos",
    ("inter", "l2"):  "Inter L2",
    ("inter", "cos"): "Inter Cos",
}
METRIC_MEANING = {
    ("intra", "l2"):  "↑ cluster scatters / ↓ cluster compacts",
    ("intra", "cos"): "↑ cluster scatters / ↓ cluster compacts",
    ("inter", "l2"):  "↑ classes separate / ↓ classes merge",
    ("inter", "cos"): "↑ classes separate / ↓ classes merge",
}


def apply_split_suffix(path: str, split: str) -> str:
    """Append '_{split}' before the extension unless split == 'single', so
    single/multi/both runs read/write separate files (mirrors the analyze
    scripts' --split handling)."""
    if split == "single":
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{split}{ext}"


def process_split(split: str) -> None:
    files = {key: apply_split_suffix(path, split) for key, path in BASE_FILES.items()}

    # -----------------------------------------------------------------
    # Load whatever summary CSVs already exist for this split
    # -----------------------------------------------------------------
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
        print(f"[warn] split='{split}': no wilcoxon summary files available yet — skipping plots.")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    # Split group_name into metric ('l2'/'cos') and baseline ('original'/'retrain')
    all_df["metric"]   = all_df["group_name"].str.split("_vs_").str[0]
    all_df["baseline"] = all_df["group_name"].str.split("_vs_").str[1]

    # -----------------------------------------------------------------
    # Compute trend per combination
    # -----------------------------------------------------------------
    key_cols = ["case", "metric_type", "metric", "baseline", "experiment", "dataset"]

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
            # both rejected – pick sign from median
            return 1 if row["median"] >= 0 else -1
        return 0

    trend_df["trend"] = trend_df.apply(assign_trend, axis=1)

    EXPERIMENTS = sorted(trend_df["experiment"].unique())
    DATASETS    = sorted(trend_df["dataset"].unique())

    # Column keys: experiment × dataset
    col_keys   = [(exp, ds) for exp in EXPERIMENTS for ds in DATASETS]
    col_labels = [f"{arch_name(exp)}\n{ds}" for exp, ds in col_keys]

    # Row keys: case × (metric_type, metric)
    row_keys   = [(case, mtype, met) for case in CASES for mtype, met in METRIC_KEYS]
    row_labels = [
        f"{CASE_LABELS[case]}\n{METRIC_LABELS[(mtype, met)]}"
        for case, mtype, met in row_keys
    ]

    # -----------------------------------------------------------------
    # Colour map: red (-1), grey (0), green (+1)
    # -----------------------------------------------------------------
    CMAP  = ListedColormap(["#d73027", "#eeeeee", "#1a9850"])
    NORM  = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], CMAP.N)

    # -----------------------------------------------------------------
    # One figure per baseline
    # -----------------------------------------------------------------
    for baseline in BASELINES:
        sub = trend_df[trend_df["baseline"] == baseline]

        n_rows = len(row_keys)      # 12
        n_cols = len(col_keys)      # 6

        trend_mat  = np.zeros((n_rows, n_cols), dtype=float)
        median_mat = np.full((n_rows, n_cols), np.nan)

        for ri, (case, mtype, met) in enumerate(row_keys):
            for ci, (exp, ds) in enumerate(col_keys):
                row = sub[
                    (sub["case"] == case)
                    & (sub["metric_type"] == mtype)
                    & (sub["metric"] == met)
                    & (sub["experiment"] == exp)
                    & (sub["dataset"] == ds)
                ]
                if not row.empty:
                    trend_mat[ri, ci]  = float(row["trend"].iloc[0])
                    median_mat[ri, ci] = float(row["median"].iloc[0])

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_cols), max(7, 0.75 * n_rows)))

        im = ax.imshow(trend_mat, cmap=CMAP, norm=NORM, aspect="auto")

        # Cell annotations (median delta)
        for ri in range(n_rows):
            for ci in range(n_cols):
                v = median_mat[ri, ci]
                txt = f"{v:+.3f}" if not np.isnan(v) else "n/a"
                trend_val = trend_mat[ri, ci]
                text_color = "white" if abs(trend_val) == 1 else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

        # Axes
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.tick_params(axis="x", which="both", top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Horizontal separator lines between cases
        for i in range(1, len(CASES)):
            y = i * len(METRIC_KEYS) - 0.5
            ax.axhline(y, color="black", linewidth=1.5, linestyle="--")

        # Right-side metric meaning annotations
        # for ri, (case, mtype, met) in enumerate(row_keys):
        #     meaning = METRIC_MEANING[(mtype, met)]
        #     ax.annotate(
        #         meaning,
        #         xy=(1.01, 1 - (ri + 0.5) / n_rows),
        #         xycoords="axes fraction",
        #         fontsize=6.5,
        #         va="center",
        #         ha="left",
        #         color="#555555",
        #     )

        # No title: the top tick labels are the only header (split/baseline/
        # reference are already in the output file name).

        # Legend
        patches = [
            mpatches.Patch(facecolor="#1a9850", label="↑ significant increase"),
            mpatches.Patch(facecolor="#eeeeee", edgecolor="grey", linewidth=0.5,
                           label="— no significant trend"),
            mpatches.Patch(facecolor="#d73027", label="↓ significant decrease"),
        ]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=8, frameon=True)

        plt.tight_layout()
        out_path = apply_split_suffix(f"results/wilcoxon_trend_heatmap_{baseline}.pdf", split)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        out_path = apply_split_suffix(f"results/wilcoxon_trend_heatmap_{baseline}.png", split)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")
        plt.close(fig)

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    summary_rows = []
    for baseline in BASELINES:
        for case, mtype, met in row_keys:
            sub = trend_df[
                (trend_df["baseline"]    == baseline)
                & (trend_df["case"]        == case)
                & (trend_df["metric_type"] == mtype)
                & (trend_df["metric"]      == met)
            ]
            n_up   = int((sub["trend"] ==  1).sum())
            n_down = int((sub["trend"] == -1).sum())
            n_none = int((sub["trend"] ==  0).sum())
            total  = n_up + n_down + n_none
            dom    = (
                "increase" if n_up > n_down and n_up > n_none
                else "decrease" if n_down > n_up and n_down > n_none
                else "no_trend"
            )
            summary_rows.append({
                "split":          split,
                "baseline":       baseline,
                "case":           case,
                "metric_type":    mtype,
                "metric":         met,
                "n_increase":     n_up,
                "n_decrease":     n_down,
                "n_no_trend":     n_none,
                "total":          total,
                "dominant_trend": dom,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = apply_split_suffix("results/wilcoxon_trend_summary.csv", split)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    for split in SPLITS:
        print(f"\n=== split: {split} ===")
        process_split(split)
