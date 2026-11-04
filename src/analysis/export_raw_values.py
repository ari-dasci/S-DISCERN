"""Export the raw per-class increments of the latent-space Wilcoxon analysis,
without dropping anything, so the tests can be recomputed elsewhere with or
without IQR-outlier removal.

Called from the three analyze_wilcoxon_unlearning*.py scripts through
--export-raw-dir, right after their own preprocessing has built the delta
tables (same checkpoint selection, --goodones filter, selected_methods.csv,
sign convention) and before any test runs. Writes, per block:

  baselines_values_{block}.csv       comparison methods (every non
                                     DISTRIBUTIONAL/NOISY version), 4 metrics
  distributional_values_{block}.csv  proposed method (PARAMETRIC) notop / confidence, same
                                     reference_version as reference_method_trends

is_outlier is only a flag (no row is removed):
  - baselines: IQR over all comparison methods of the cell
    (experiment, dataset, metric, group_name) -- the pool of the canonical test.
  - DISTRIBUTIONAL: IQR over the deltas of that variant alone in the cell
    (same criterion as compute_reference_method_trends).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_NOTOP_LABEL, PROPOSED_CONF_LABEL  # noqa: E402
from typing import Callable, Dict, Iterable

import pandas as pd

COLUMNS = [
    "experiment", "dataset", "class_to_remove", "version", "metric", "group_name",
    "delta", "n_classes_removed", "architecture_family", "is_outlier",
]
CELL_KEYS = ["experiment", "dataset", "metric", "group_name"]
# proposed method (src/utils/proposed_method.py; previously DISTRIBUTIONAL).  The
# output file names (distributional_values_*.csv) are kept for compatibility.
DISTRIBUTIONAL_LABELS = {PROPOSED_NOTOP_LABEL: "notop",
                         PROPOSED_CONF_LABEL: "confidence"}


def _flag_outliers(df: pd.DataFrame, keys, drop_outliers_iqr: Callable, multiplier: float) -> pd.Series:
    flags = pd.Series(True, index=df.index)
    for _, g in df.groupby(keys, dropna=False):
        vals = g["delta"].dropna().astype(float)
        flags.loc[drop_outliers_iqr(vals, multiplier).index] = False
    return flags


def export_raw_values(
    delta_tables: Iterable[pd.DataFrame],
    block: str,
    out_dir: str,
    *,
    drop_outliers_iqr: Callable,
    iqr_multiplier: float,
    special_prefixes: tuple,
    reference_versions: Dict[str, str],
    version_overrides: Dict[tuple, str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.concat(list(delta_tables), ignore_index=True)
    df = df[df["n_classes_removed"] == 1].copy()   # single-class forgetting only
    version = df["version"].astype(str)

    # 1) Comparison methods
    base = df[~version.str.startswith(special_prefixes)].copy()
    base["is_outlier"] = _flag_outliers(base, CELL_KEYS, drop_outliers_iqr, iqr_multiplier)
    base = base[COLUMNS].sort_values(CELL_KEYS + ["version", "class_to_remove"])
    path = os.path.join(out_dir, f"baselines_values_{block}.csv")
    base.to_csv(path, index=False)
    print(f"Saved {len(base)} rows to {path}")

    # 2) proposed method notop / confidence, resolved per architecture x dataset
    parts, missing = [], []
    cells = df[CELL_KEYS].drop_duplicates()
    for label, variant in DISTRIBUTIONAL_LABELS.items():
        for (exp, ds), cell_grp in cells.groupby(["experiment", "dataset"]):
            v = version_overrides.get((label, exp, ds), reference_versions[label])
            rows = df[(df["experiment"] == exp) & (df["dataset"] == ds) & (df["version"] == v)]
            for _, c in cell_grp.iterrows():
                if rows[(rows["metric"] == c["metric"]) & (rows["group_name"] == c["group_name"])].empty:
                    missing.append((variant, exp, ds, c["metric"], c["group_name"], v))
            if not rows.empty:
                parts.append(rows.assign(variant=variant))

    dist = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS + ["variant"])
    if not dist.empty:
        dist["is_outlier"] = _flag_outliers(dist, CELL_KEYS + ["variant"], drop_outliers_iqr,
                                            iqr_multiplier)
        dist = dist[COLUMNS + ["variant"]].sort_values(
            ["variant"] + CELL_KEYS + ["class_to_remove"])
    path = os.path.join(out_dir, f"distributional_values_{block}.csv")
    dist.to_csv(path, index=False)
    print(f"Saved {len(dist)} rows to {path}")

    for variant, exp, ds, metric, group, v in missing:
        print(f"[missing] {block}: proposed method {variant} {exp}/{ds} {metric} {group} (version {v})")
    if not missing:
        print(f"[ok] {block}: no missing proposed-method cells")
