#!/usr/bin/env python3
"""Sanity checks for results/wilcox/raw/*_values_{forget,retain,unknown}.csv
(written by export_raw_wilcox.sh). Reads only CSVs; the Wilcoxon tests are
re-run here solely to verify that the exported deltas reproduce the trends
already stored in wilcoxon_unpaired_summary_intra*.csv / wilcoxon_unpaired_inter*.csv.

    python src/analysis/check_raw_values.py
"""

from __future__ import annotations

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_NOTOP_LABEL, PROPOSED_CONF_LABEL  # noqa: E402
import pandas as pd
from scipy.stats import wilcoxon

RAW = "results/wilcox/raw"
WILCOX = "results/wilcox"
BLOCKS = {"forget": "", "retain": "_retain", "unknown": "_unknown3"}
N_CLASSES = {"cifar10": 10, "cifar100": 20, "mnist": 10}
N_BASELINES = 7
ARCHS = ["resnet18", "efficientnet_v2_s", "swin_t", "vit_b_16"]
CELL = ["experiment", "dataset", "metric", "group_name"]
ALPHA, MIN_SAMPLES = 0.05, 6
VARIANT_LABEL = {"notop": PROPOSED_NOTOP_LABEL, "confidence": PROPOSED_CONF_LABEL}   # proposed method


def expected_cells(metrics) -> pd.DataFrame:
    rows = []
    for exp in ARCHS:
        for ds in N_CLASSES:
            for m in metrics:
                kind = "cos" if m.endswith("_cos") else "l2"
                for ref in ("original", "retrain"):
                    rows.append((exp, ds, m, f"{kind}_vs_{ref}"))
    return pd.DataFrame(rows, columns=CELL)


def check_counts(name, df, metrics, per_cell_factor, extra_keys=()) -> None:
    if df.empty:
        print(f"  [missing] {name}: no rows at all (input CSV not processed for this block yet)")
        return
    keys = list(extra_keys) + CELL
    got = df.groupby(keys).size().rename("n").reset_index()
    exp = expected_cells(metrics)
    if extra_keys:
        exp = pd.concat([exp.assign(**{extra_keys[0]: v}) for v in df[extra_keys[0]].unique()])
    m = exp.merge(got, on=keys, how="left").fillna({"n": 0})
    m["expected"] = m["dataset"].map(N_CLASSES) * per_cell_factor
    bad = m[m["n"] != m["expected"]]
    print(f"  {name}: {len(df)} rows, {len(m)} cells, {len(m) - len(bad)} with the expected count")
    for _, r in bad.iterrows():
        tag = f"{r[extra_keys[0]]} " if extra_keys else ""
        print(f"    [count] {tag}{r.experiment}/{r.dataset} {r.metric} {r.group_name}: "
              f"{int(r.n)} (expected {int(r.expected)})")


def reproduce_trends(block, suffix, base) -> None:
    summ = pd.concat([
        pd.read_csv(f"{WILCOX}/wilcoxon_unpaired_summary_intra{suffix}.csv").assign(kind="intra"),
        pd.read_csv(f"{WILCOX}/wilcoxon_unpaired_inter{suffix}.csv").assign(kind="inter"),
    ])
    summ["reject"] = summ["reject_h0"].astype(str).str.lower().eq("true")
    summ = summ.set_index(["kind", "experiment", "dataset", "group_name", "alternative"])

    n_cells = n_ok = 0
    for (exp, ds, metric, group), g in base.groupby(CELL):
        kind = "intra" if metric.startswith("cohesion") else "inter"
        clean = g.loc[~g["is_outlier"], "delta"].to_numpy(float)
        n_cells += 1
        cell_ok = True
        for alt in ("greater", "less"):
            rej = len(clean) >= MIN_SAMPLES and wilcoxon(clean, alternative=alt).pvalue < ALPHA
            s = summ.loc[(kind, exp, ds, group, alt)]
            same = (rej == bool(s["reject"]) and len(clean) == int(s["n_clean"])
                    and np.isclose(np.median(clean), s["median_clean"]))
            if not same:
                cell_ok = False
                print(f"    [trend] {block} {exp}/{ds} {metric} {group} {alt}: "
                      f"reject={rej} n={len(clean)} med={np.median(clean):+.4f} vs stored "
                      f"reject={bool(s['reject'])} n={int(s['n_clean'])} med={s['median_clean']:+.4f}")
        n_ok += cell_ok
    print(f"  trends reproduced: {n_ok}/{n_cells} cells identical (reject_h0 both sides, n_clean, median)")


def check_reference_means(block, suffix, dist) -> None:
    ref = pd.read_csv(f"{WILCOX}/reference_method_trends{suffix}.csv")
    ref = ref[ref["reference"].isin(VARIANT_LABEL.values())]
    means = (dist[~dist["is_outlier"]].groupby(["variant"] + CELL)
             .agg(mean=("delta", "mean"), version=("version", "first")).reset_index())
    means["reference"] = means["variant"].map(VARIANT_LABEL)
    m = means.merge(ref, on=["reference"] + CELL, how="outer")
    diff = (m["mean"] - m["mean_delta"]).abs()
    bad = m[~(diff < 1e-9) | (m["version"] != m["reference_version"])]
    print(f"  mean(delta | not outlier) vs reference_method_trends{suffix}.csv: "
          f"{len(m) - len(bad)}/{len(m)} cells identical (value and version)")
    for _, r in bad.head(20).iterrows():
        print(f"    [ref] {r['reference']} {r.experiment}/{r.dataset} {r.metric} {r.group_name}: "
              f"raw={r['mean']} stored={r['mean_delta']} "
              f"version_raw={r['version']} version_stored={r['reference_version']}")


def main() -> None:
    dist_by_block = {}
    for block, suffix in BLOCKS.items():
        print(f"\n=== {block} ===")
        base = pd.read_csv(f"{RAW}/baselines_values_{block}.csv")
        dist = pd.read_csv(f"{RAW}/distributional_values_{block}.csv")
        for d in (base, dist):
            d["is_outlier"] = d["is_outlier"].astype(str).str.lower().eq("true")
        metrics = sorted(base["metric"].unique())
        print(f"  metrics: {metrics}")
        print(f"  baseline versions per architecture x dataset: "
              f"{sorted(base.groupby(['experiment', 'dataset'])['version'].nunique().unique())}")
        check_counts("baselines", base, metrics, N_BASELINES)
        check_counts("distributional", dist, metrics, 1, extra_keys=("variant",))
        reproduce_trends(block, suffix, base)
        check_reference_means(block, suffix, dist)
        dist_by_block[block] = dist

    print(f"\n=== spot checks (forget, {PROPOSED_CONF_LABEL}) ===")
    d = dist_by_block["forget"]
    d = d[d["variant"] == "confidence"]
    print("  cohesion_forget_l2 / l2_vs_original, mean of non-outliers:")
    for exp, ds, old in [("resnet18", "cifar10", -5.168), ("efficientnet_v2_s", "cifar10", -8.276),
                         ("swin_t", "cifar100", -37.798)]:
        g = d[(d.experiment == exp) & (d.dataset == ds) & (d.metric == "cohesion_forget_l2")
              & (d.group_name == "l2_vs_original")]
        print(f"    {exp}/{ds}: old pooled-IQR value {old:+.3f} | now {g.loc[~g.is_outlier, 'delta'].mean():+.3f} "
              f"({(~g.is_outlier).sum()}/{len(g)} kept; mean without any filter {g['delta'].mean():+.3f})")

    g = d[(d.experiment == "efficientnet_v2_s") & (d.dataset == "mnist")
          & (d.metric == "distance_inter_class_cos") & (d.group_name == "cos_vs_original")]
    print("  efficientnet_v2_s/mnist distance_inter_class_cos / cos_vs_original (old n/a):")
    if g.empty:
        print("    still no rows")
    else:
        print(f"    {len(g)} rows, {(~g.is_outlier).sum()} not outliers, "
              f"mean {g.loc[~g.is_outlier, 'delta'].mean():+.4f}, version {g['version'].unique().tolist()}")


if __name__ == "__main__":
    main()
