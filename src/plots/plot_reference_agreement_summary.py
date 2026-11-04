#!/usr/bin/env python3
"""Summary heatmap: for each proposed/reference method (DISTRIBUTIONAL,
NOISY -- notop and confidence_penalty variants), what fraction of its
evaluable cells agree with the majority direction of the 7 comparison
methods, broken down by case (forget/retain/unknown) x metric type
(intra/inter) and by architecture.

This does NOT rerun any Wilcoxon test -- it aggregates the per-cell
"agrees_with_majority" column already computed by compute_reference_method_
trends() in analyze_wilcoxon_unlearning*.py (reference_method_trends*.csv),
the same source as the existing reference_method_heatmap_* figures (which
show one cell per experiment x dataset). This one instead answers "overall,
does this method track the trend or not" in a single compact figure.

Rows: (reference, architecture). Columns: (case, metric_type). Cell: % of
evaluable (experiment-matching, non-absent) cells that agree with the
majority, annotated with "agree/total". A reference/case with zero evaluable
cells for an architecture (e.g. NOISY in "unknown" until it's extracted, or
DISTRIBUTIONAL notop for swin_t before the per-architecture name-resolution
fix) shows as grey "n/a" rather than a fabricated 0%.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_NOTOP_LABEL, PROPOSED_CONF_LABEL  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from method_colors import arch_name

FILES = {
    "forget": "results/wilcox/reference_method_trends.csv",
    "retain": "results/wilcox/reference_method_trends_retain.csv",
    "unknown": "results/wilcox/reference_method_trends_unknown3.csv",
}

CASES = ["forget", "retain", "unknown"]
CASE_LABELS = {"forget": "Forget", "retain": "Retain", "unknown": "Unknown"}
METRIC_TYPES = ["intra", "inter"]
METRIC_TYPE_LABELS = {"intra": "Intra", "inter": "Inter"}
EXPERIMENTS = ["resnet18", "efficientnet_v2_s", "swin_t", "vit_b_16"]


def load_all() -> pd.DataFrame:
    frames = []
    for case, path in FILES.items():
        if not os.path.exists(path):
            print(f"[warn] missing {path} (case={case}) — skipping it.")
            continue
        d = pd.read_csv(path)
        if d.empty:
            print(f"[warn] {path} has no rows — skipping it.")
            continue
        d["case"] = case
        frames.append(d)
    if not frames:
        raise SystemExit("No reference_method_trends*.csv files found — run run_wilcoxon_all.sh first.")
    df = pd.concat(frames, ignore_index=True)
    df["metric_type"] = np.where(df["metric"].str.contains("cohesion"), "intra", "inter")
    df = df[df["direction"] != "absent"].copy()
    df["agrees"] = df["agrees_with_majority"].astype(str).str.lower().eq("true")
    return df


def make_heatmap(df: pd.DataFrame, references: list[str], out_stem: str, title: str) -> None:
    row_keys = [(ref, exp) for ref in references for exp in EXPERIMENTS]
    row_labels = [f"{ref}\n{arch_name(exp)}" for ref, exp in row_keys]
    col_keys = [(case, mt) for case in CASES for mt in METRIC_TYPES]
    col_labels = [f"{CASE_LABELS[case]}\n{METRIC_TYPE_LABELS[mt]}" for case, mt in col_keys]

    n_rows, n_cols = len(row_keys), len(col_keys)
    pct_mat = np.full((n_rows, n_cols), np.nan)
    count_mat = np.zeros((n_rows, n_cols), dtype=int)
    agree_mat = np.zeros((n_rows, n_cols), dtype=int)

    for ri, (ref, exp) in enumerate(row_keys):
        for ci, (case, mt) in enumerate(col_keys):
            sub = df[
                (df["reference"] == ref)
                & (df["experiment"] == exp)
                & (df["case"] == case)
                & (df["metric_type"] == mt)
            ]
            if sub.empty:
                continue
            n = len(sub)
            a = int(sub["agrees"].sum())
            pct_mat[ri, ci] = 100.0 * a / n
            count_mat[ri, ci] = n
            agree_mat[ri, ci] = a

    if np.isnan(pct_mat).all():
        print(f"[warn] no data for '{title}' — skipping figure.")
        return

    # Diverging colormap centered on 50%: red (never agrees) -> white (coin
    # flip) -> green (always agrees).
    cmap = LinearSegmentedColormap.from_list("agree", ["#d73027", "#ffffff", "#1a9850"])
    cmap.set_bad("#dddddd")

    masked = np.ma.masked_invalid(pct_mat)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_cols), max(5, 0.55 * n_rows)))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for ri in range(n_rows):
        for ci in range(n_cols):
            v = pct_mat[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, "n/a", ha="center", va="center", fontsize=7, color="#666666")
                continue
            txt = f"{v:.0f}%\n{agree_mat[ri, ci]}/{count_mat[ri, ci]}"
            colour = "white" if (v < 25 or v > 75) else "black"
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=7, color=colour, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(1, len(references)):
        ax.axhline(i * len(EXPERIMENTS) - 0.5, color="black", linewidth=1.5, linestyle="--")

    # No title: the top tick labels are the only header (split/baseline/
    # reference are already in the output file name).

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("% agreement with majority", fontsize=8)

    plt.tight_layout()
    for ext in (".png", ".pdf"):
        out = f"results/{out_stem}{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: results/{out_stem}.png")
    plt.close(fig)


def main() -> None:
    df = load_all()

    make_heatmap(
        df,
        references=[PROPOSED_NOTOP_LABEL, PROPOSED_CONF_LABEL],   # proposed method
        out_stem="reference_agreement_summary_distributional",     # name kept for compatibility
        title=f"{PROPOSED_NOTOP_LABEL.split()[0]} vs the observed trend",
    )
    make_heatmap(
        df,
        references=["NOISY notop", "NOISY confidence"],
        out_stem="reference_agreement_summary_noisy",
        title="NOISY (gaussian) vs the observed trend",
    )

    # Summary CSV alongside the figures, same numbers in tabular form.
    rows = []
    for ref in sorted(df["reference"].unique()):
        for case in CASES:
            for mt in METRIC_TYPES:
                sub = df[(df["reference"] == ref) & (df["case"] == case) & (df["metric_type"] == mt)]
                if sub.empty:
                    continue
                rows.append({
                    "reference": ref,
                    "case": case,
                    "metric_type": mt,
                    "n_agree": int(sub["agrees"].sum()),
                    "n_total": len(sub),
                    "pct_agree": round(100.0 * sub["agrees"].sum() / len(sub), 1),
                })
    out_csv = "results/wilcox/reference_agreement_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
