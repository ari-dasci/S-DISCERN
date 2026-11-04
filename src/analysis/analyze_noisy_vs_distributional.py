#!/usr/bin/env python3
"""Are NOISY and the proposed method significantly different? Paired tests per scenario.

The proposed method is the one of src/utils/proposed_method.py (PARAMETRIC,
tiebreak by KS of the norm + DISTRIBUTIONAL test pipeline; it was DISTRIBUTIONAL
before).  "DIST"/"dist_*" names below are kept for compatibility and refer to it.

Comparisons (both methods share experiment / dataset / removed-class set, so they
start from the same original model and forget the same classes -> paired design):

  notop               NOISY_..._forget_notop
                      vs PROPOSED_NOTOP (PARAMETRIC_..._forget_notop_margin_10.0)
  confidence_penalty  NOISY_..._forget_confidence_penalty
                      vs PROPOSED_CONF (PARAMETRIC_..._forget_corrected_confidence_penalty)

Scenarios (analysed completely separately): single, multiple, icu.
single uses mnist/cifar10/cifar100; multiple and icu only use cifar100
(--multiple-datasets / --icu-datasets to change it).

Metrics (per observation, then diff = proposed method - NOISY):
  unlearning_accuracy_score  accuracy on the forget set   (lower  is better)
  accuracy_score_after       accuracy on the retain set   (higher is better)
  harmonic_mean              2*a*(1-u)/(a+(1-u))          (higher is better)

Test: two-sided Wilcoxon signed-rank on the paired differences, plus effect sizes
(median/mean diff, matched-pairs rank-biserial, win/tie/loss counts).

Levels (each level = independent slice of the same pairs):
  global | dataset | family (conv vs transformer) | architecture |
  dataset_x_family | dataset_x_architecture

Analysis sets:
  main   everything except resnet18/efficientnet_v2_s on cifar10/mnist
  noise  only those 4 cells, where both methods are expected to be equivalent:
         serves as a noise floor (compare `median_abs_diff` and p-values)

Units of analysis:
  class  single/multiple: one observation per removed-class set
  run_auc   ICU: one observation per run = area under the metric-vs-steps curve
            (trapezoid over the steps both methods have, divided by the step
            span so it stays on the metric's scale). Retain acc / harmonic:
            larger area is better; forget acc: smaller area is better. The
            harmonic mean is computed per step and then integrated.
  run_mean  ICU: same but plain mean over steps (--icu-run-aggregation mean)
  step      ICU: every step is a pair (steps of a run are correlated, so p-values
            are optimistic; kept as a sensitivity check)

Multiple-testing corrections (Holm = FWER, Benjamini-Hochberg = FDR). The three
metrics and all cells are corrected jointly, families being:
  *_level : scenario x analysis_set x unit x comparison x level
  *_all   : scenario x analysis_set x unit x comparison (all levels together;
            cells with an identical set of pairs count once)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_LABEL_FAMILY, PROPOSED_NOTOP, PROPOSED_CONF  # noqa: E402
from src.utils.embs_fallback import add_embs_fallback  # noqa: E402  (TEMPORARY: efficientnet nofreeze as embs)
from scipy.stats import rankdata, wilcoxon

NOISY_PREFIX = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_"
# common prefix of the proposed method's versions (".._forget_"), used to report the variant
DIST_PREFIX = os.path.commonprefix([PROPOSED_NOTOP, PROPOSED_CONF])
# comparison -> (NOISY base version, proposed-method base versions in preference order;
# the first one present in a given (architecture, dataset) cell is used)
COMPARISONS: Dict[str, Tuple[str, List[str]]] = {
    "notop": (
        NOISY_PREFIX + "notop",
        [PROPOSED_NOTOP],
    ),
    "confidence_penalty": (
        NOISY_PREFIX + "confidence_penalty",
        [PROPOSED_CONF],
    ),
}
ALL_BASES = sorted(
    {noisy for noisy, _ in COMPARISONS.values()}
    | {d for _, dists in COMPARISONS.values() for d in dists}
)

INPUT_DEFAULTS = {
    "single": "results/unlearning_results_single_embs_new.csv",
    "multiple": "results/unlearning_results_multiple_embs_new.csv",
    "icu": "results_ICU/unlearning_results.csv",
}

DATASETS = {"mnist", "cifar10", "cifar100"}
# Datasets analysed per scenario: multiple and ICU are restricted to cifar100.
SCENARIO_DATASETS_DEFAULT = {
    "single": ["mnist", "cifar10", "cifar100"],
    "multiple": ["cifar100"],
    "icu": ["cifar100"],
}
ARCHITECTURE_FAMILY = {
    "resnet18": "conv",
    "efficientnet_v2_s": "conv",
    "swin_t": "transformer",
    "vit_b_16": "transformer",
}

ACC_COLS = ["accuracy_score_after", "unlearning_accuracy_score"]
METRICS = ["unlearning_accuracy_score", "accuracy_score_after", "harmonic_mean"]
METRIC_SHORT = {
    "unlearning_accuracy_score": "forget",
    "accuracy_score_after": "retain",
    "harmonic_mean": "harmonic",
}
# +1: higher is better, -1: lower is better
POLARITY = {"unlearning_accuracy_score": -1, "accuracy_score_after": 1, "harmonic_mean": 1}

LEVELS: Dict[str, List[str]] = {
    "global": [],
    "dataset": ["dataset"],
    "family": ["architecture_family"],
    "architecture": ["experiment"],
    "dataset_x_family": ["dataset", "architecture_family"],
    "dataset_x_architecture": ["dataset", "experiment"],
}
CONSOLE_LEVELS = ["global", "dataset", "family", "architecture"]

ID_COLS = [
    "scenario", "analysis_set", "unit", "comparison", "level",
    "dataset", "architecture_family", "experiment", "n_pairs", "dist_variants",
]
VALUE_COLS = [
    "median_diff", "favours", "p_raw", "sig_raw",
    "p_holm_level", "sig_holm_level", "p_bh_level",
    "p_holm_all", "p_bh_all",
]
ZERO_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired significance tests NOISY vs the proposed method (PARAMETRIC) per scenario."
    )
    p.add_argument("--input-single", default=INPUT_DEFAULTS["single"])
    p.add_argument("--input-multiple", default=INPUT_DEFAULTS["multiple"])
    p.add_argument("--input-icu", default=INPUT_DEFAULTS["icu"])
    p.add_argument(
        "--scenarios", nargs="+", choices=list(INPUT_DEFAULTS), default=list(INPUT_DEFAULTS),
        help="Scenarios to analyse (default: all three).",
    )
    p.add_argument(
        "--analysis-set", choices=["main", "noise", "both"], default="both",
        help="'noise' = resnet18/efficientnet_v2_s on cifar10/mnist (noise floor).",
    )
    p.add_argument(
        "--noise-architectures", nargs="+", default=["resnet18", "efficientnet_v2_s"],
    )
    p.add_argument("--noise-datasets", nargs="+", default=["cifar10", "mnist"])
    for scenario, default in SCENARIO_DATASETS_DEFAULT.items():
        p.add_argument(
            f"--{scenario}-datasets", nargs="+", choices=sorted(DATASETS), default=default,
            help=f"Datasets kept for the {scenario} scenario (default: {' '.join(default)}).",
        )
    p.add_argument(
        "--icu-unit", choices=["run", "step", "both"], default="both",
        help="ICU: 'run' aggregates the steps of each run, 'step' uses every step as a pair.",
    )
    p.add_argument(
        "--icu-run-aggregation", choices=["auc", "mean"], default="auc",
        help=(
            "How the steps of an ICU run are collapsed for unit 'run': 'auc' = trapezoidal "
            "area under the metric-vs-steps curve divided by the step span (default), "
            "'mean' = plain mean over steps."
        ),
    )
    p.add_argument("--checkpoint", default="original_best", help="Suffix after '/' in version.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-pairs", type=int, default=6, help="Minimum pairs to run a test.")
    p.add_argument("--output-dir", default="results/wilcox_noisy_vs_dist")
    return p.parse_args()


def harmonic_mean_acc(retain_acc, forget_acc) -> np.ndarray:
    """2*a*(1-u)/(a+(1-u)); u is 'lower is better' so it is inverted first
    (same combination as src/plots/extract_methods.py)."""
    a = np.asarray(retain_acc, dtype=float)
    f = 1.0 - np.asarray(forget_acc, dtype=float)
    denom = a + f
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, 2.0 * a * f / safe, 0.0)


def load_scenario(
    scenario: str, path: str, checkpoint: str, datasets: List[str]
) -> Optional[pd.DataFrame]:
    """One row per (experiment, dataset, id_database[, steps], base version) with
    the two accuracies and their harmonic mean. Returns None if unusable."""
    if not path or not os.path.exists(path):
        print(f"[warn] {scenario}: input not found, skipping scenario: {path}")
        return None

    is_icu = scenario == "icu"
    time_col = "steps" if is_icu else "epoch"
    needed = {"experiment", "id_database", "version", time_col, *ACC_COLS}
    try:
        df = add_embs_fallback(pd.read_csv(path, usecols=lambda c: c in needed, dtype=str))
    except Exception as exc:  # ragged / NUL-corrupted CSV: see scan_ragged_csvs.py, strip_null_lines.py
        print(f"[warn] {scenario}: cannot read {path} ({exc}) - skipping scenario")
        return None
    missing = needed - set(df.columns)
    if missing:
        print(f"[warn] {scenario}: {path} lacks columns {sorted(missing)} - skipping scenario")
        return None

    # Keep only the target versions (this also drops any header row that ended up
    # in the middle of the ICU file: its 'version' cell is the literal "version").
    wanted = {f"{b}/{checkpoint}" for b in ALL_BASES}
    df = df[df["version"].isin(wanted)].copy()
    if not is_icu:
        df = df[df["epoch"].str.strip() == "best"].copy()

    df["experiment"] = df["experiment"].str.strip()
    df["base"] = df["version"].str.split("/").str[0]
    df["dataset"] = df["id_database"].str.extract(r"^(.+)_classes_to_remove_[\d_]+$")[0]
    df = df[df["dataset"].isin(set(datasets) & DATASETS)].copy()

    num_cols = ACC_COLS + (["steps"] if is_icu else [])
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=num_cols)
    if df.empty:
        print(f"[warn] {scenario}: no usable rows for the target versions in {path}")
        return None

    keys = ["experiment", "dataset", "id_database", "base"] + (["steps"] if is_icu else [])
    n_dup = int(df.duplicated(subset=keys).sum())
    if n_dup:
        print(f"[info] {scenario}: {n_dup} duplicated rows on {keys} -> averaged")
    df = df.groupby(keys, as_index=False)[ACC_COLS].mean()
    df["harmonic_mean"] = harmonic_mean_acc(
        df["accuracy_score_after"], df["unlearning_accuracy_score"]
    )
    print(f"[info] {scenario}: {len(df)} rows loaded from {path}")
    return df


def build_pairs(
    df: pd.DataFrame, scenario: str, comparison: str, noisy_base: str, dist_bases: List[str]
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Inner-join NOISY and DISTRIBUTIONAL rows on the pairing key. The DISTRIBUTIONAL
    variant is chosen per (experiment, dataset) as the first of `dist_bases` present."""
    keys = ["experiment", "dataset", "id_database"] + (["steps"] if "steps" in df.columns else [])

    chosen: Dict[Tuple[str, str], Optional[str]] = {}
    for (exp, ds), g in df.groupby(["experiment", "dataset"]):
        available = set(g["base"])
        chosen[(exp, ds)] = next((b for b in dist_bases if b in available), None)

    row_chosen = pd.Series(
        [chosen.get(k) for k in zip(df["experiment"], df["dataset"])], index=df.index, dtype=object
    )
    noisy = df[df["base"] == noisy_base]
    dist = df[df["base"] == row_chosen].copy()
    dist["dist_variant"] = dist["base"].str[len(DIST_PREFIX):]

    merged = noisy[keys + METRICS].merge(
        dist[keys + METRICS + ["dist_variant"]], on=keys, how="inner", suffixes=("_noisy", "_dist")
    )

    coverage: List[Dict[str, object]] = []
    relevant = df[df["base"].isin([noisy_base, *dist_bases])]
    for (exp, ds), g in relevant.groupby(["experiment", "dataset"]):
        m = merged[(merged["experiment"] == exp) & (merged["dataset"] == ds)]
        variant = chosen.get((exp, ds))
        coverage.append(
            {
                "scenario": scenario,
                "comparison": comparison,
                "experiment": exp,
                "dataset": ds,
                "dist_variant": variant[len(DIST_PREFIX):] if variant else "MISSING",
                "n_noisy_rows": int((g["base"] == noisy_base).sum()),
                "n_dist_rows": int(g["base"].isin(dist_bases).sum()),
                "n_pairs": int(len(m)),
                "n_runs_paired": int(m["id_database"].nunique()),
            }
        )
    return merged, coverage


def normalized_auc(steps: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal area under values(steps), divided by the step span so it stays
    on the metric's own scale (a constant curve gives that constant). A single
    step has no span: its value is returned."""
    x = np.asarray(steps, dtype=float)
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 2 or x[-1] == x[0]:
        return float(y[0])
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return area / float(x[-1] - x[0])


def aggregate_runs(pairs: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per ICU run. `pairs` holds the steps present in both methods, so
    NOISY and DISTRIBUTIONAL are aggregated over exactly the same steps. The
    harmonic mean is computed per step first (in load_scenario) and then
    aggregated like the other metrics."""
    value_cols = [f"{m}_{side}" for m in METRICS for side in ("noisy", "dist")]
    run_keys = ["experiment", "dataset", "id_database", "dist_variant"]
    rows: List[Dict[str, object]] = []
    for key, g in pairs.groupby(run_keys, sort=True):
        row: Dict[str, object] = dict(zip(run_keys, key))
        row["n_steps"] = int(len(g))
        for col in value_cols:
            row[col] = (
                normalized_auc(g["steps"].to_numpy(), g[col].to_numpy())
                if method == "auc" else float(g[col].mean())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_units(
    pairs: pd.DataFrame, scenario: str, icu_unit: str, run_aggregation: str
) -> Dict[str, pd.DataFrame]:
    if scenario != "icu":
        return {"class": pairs}
    units: Dict[str, pd.DataFrame] = {}
    if icu_unit in ("step", "both"):
        units["step"] = pairs
    if icu_unit in ("run", "both"):
        units[f"run_{run_aggregation}"] = aggregate_runs(pairs, run_aggregation)
    return units


def to_long(
    pairs: pd.DataFrame, scenario: str, comparison: str, unit: str,
    noise_archs: List[str], noise_datasets: List[str],
) -> pd.DataFrame:
    p = pairs.copy()
    if "steps" not in p.columns:
        p["steps"] = np.nan
    if "n_steps" not in p.columns:
        p["n_steps"] = np.nan
    p["scenario"] = scenario
    p["comparison"] = comparison
    p["unit"] = unit
    p["architecture_family"] = p["experiment"].map(ARCHITECTURE_FAMILY).fillna("unknown")
    p["analysis_set"] = np.where(
        p["experiment"].isin(noise_archs) & p["dataset"].isin(noise_datasets), "noise", "main"
    )
    p["pair_id"] = (
        p["experiment"] + "|" + p["id_database"] + "|" + p["steps"].astype(str).replace("nan", "")
    )
    id_cols = [
        "scenario", "comparison", "unit", "analysis_set", "experiment", "architecture_family",
        "dataset", "id_database", "steps", "n_steps", "dist_variant", "pair_id",
    ]
    frames = []
    for metric in METRICS:
        f = p[id_cols].copy()
        f["metric"] = metric
        f["noisy"] = p[f"{metric}_noisy"].to_numpy()
        f["dist"] = p[f"{metric}_dist"].to_numpy()
        f["diff"] = f["dist"] - f["noisy"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def paired_test(diff: np.ndarray, min_pairs: int) -> Dict[str, object]:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = np.where(np.abs(d) < ZERO_TOL, 0.0, d)
    n = int(len(d))
    nz = d[d != 0.0]

    out: Dict[str, object] = {
        "n_pairs": n,
        "n_nonzero": int(len(nz)),
        "n_dist_gt": int((d > 0).sum()),
        "n_dist_lt": int((d < 0).sum()),
        "n_ties": int((d == 0).sum()),
        "median_diff": float(np.median(d)) if n else np.nan,
        "mean_diff": float(np.mean(d)) if n else np.nan,
        "std_diff": float(np.std(d, ddof=1)) if n > 1 else np.nan,
        "median_abs_diff": float(np.median(np.abs(d))) if n else np.nan,
        "rank_biserial": np.nan,
        "statistic": np.nan,
        "p_raw": np.nan,
        "status": "",
    }
    if len(nz):
        ranks = rankdata(np.abs(nz))
        w_plus = float(ranks[nz > 0].sum())
        w_minus = float(ranks[nz < 0].sum())
        out["rank_biserial"] = (w_plus - w_minus) / (w_plus + w_minus)

    if n < min_pairs:
        out["status"] = "too_few_pairs"
    elif len(nz) == 0:
        out["statistic"], out["p_raw"], out["status"] = 0.0, 1.0, "identical"
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            out["statistic"], out["p_raw"], out["status"] = float(res.statistic), float(res.pvalue), "ok"
        except Exception as exc:
            out["status"] = f"error:{exc}"
    return out


def run_tests(long_df: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["scenario", "analysis_set", "unit", "comparison"]
    for gkey, sub in long_df.groupby(group_cols, sort=True):
        scenario, analysis_set, unit, comparison = gkey
        for level, cols in LEVELS.items():
            cells = sub.groupby(cols, sort=True) if cols else [((), sub)]
            for ckey, cell in cells:
                ckey = ckey if isinstance(ckey, tuple) else (ckey,)
                labels = {"dataset": "ALL", "architecture_family": "ALL", "experiment": "ALL"}
                labels.update(dict(zip(cols, ckey)))
                signature = hashlib.md5(
                    "|".join(sorted(cell["pair_id"].unique())).encode()
                ).hexdigest()
                variants = ";".join(sorted(cell["dist_variant"].unique()))
                for metric in METRICS:
                    m = cell[cell["metric"] == metric]
                    res = paired_test(m["diff"].to_numpy(), min_pairs)
                    rb = res["rank_biserial"]
                    if pd.isna(rb) or rb == 0:
                        favours = "none"
                    else:
                        favours = PROPOSED_LABEL_FAMILY if POLARITY[metric] * rb > 0 else "NOISY"
                    rows.append(
                        {
                            "scenario": scenario,
                            "analysis_set": analysis_set,
                            "unit": unit,
                            "comparison": comparison,
                            "level": level,
                            **labels,
                            "dist_variants": variants,
                            "metric": metric,
                            "metric_short": METRIC_SHORT[metric],
                            "median_noisy": float(m["noisy"].median()) if len(m) else np.nan,
                            "median_dist": float(m["dist"].median()) if len(m) else np.nan,
                            **res,
                            "favours": favours,
                            "signature": signature,
                        }
                    )
    return pd.DataFrame(rows)


def holm_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 1.0
    for k, i in enumerate(np.argsort(p)[::-1]):
        running = min(running, p[i] * m / (m - k))
        adj[i] = min(1.0, running)
    return adj


def add_adjusted(tests: pd.DataFrame, family_cols: List[str], suffix: str) -> pd.DataFrame:
    """Holm / BH inside each family. Tests over an identical set of pairs (same
    signature and metric, e.g. 'global' == 'dataset' when there is one dataset)
    are counted once and share their adjusted p-value."""
    holm_col, bh_col = f"p_holm_{suffix}", f"p_bh_{suffix}"
    tests[holm_col] = np.nan
    tests[bh_col] = np.nan
    valid = tests[tests["p_raw"].notna()]
    for _, sub in valid.groupby(family_cols, sort=False):
        uniq = sub.drop_duplicates(subset=["signature", "metric"])
        rep = pd.DataFrame(
            {
                "signature": uniq["signature"].to_numpy(),
                "metric": uniq["metric"].to_numpy(),
                "_holm": holm_adjust(uniq["p_raw"].to_numpy()),
                "_bh": bh_adjust(uniq["p_raw"].to_numpy()),
            }
        )
        merged = sub[["signature", "metric"]].merge(rep, on=["signature", "metric"], how="left")
        tests.loc[sub.index, holm_col] = merged["_holm"].to_numpy()
        tests.loc[sub.index, bh_col] = merged["_bh"].to_numpy()
    return tests


def to_wide(tests: pd.DataFrame) -> pd.DataFrame:
    """One row per cell, columns '<metric>__<value>' (forget / retain / harmonic)."""
    order = pd.MultiIndex.from_frame(tests[ID_COLS].drop_duplicates())
    wide = tests.set_index(ID_COLS + ["metric_short"])[VALUE_COLS].unstack("metric_short")
    wide = wide.reindex(order)
    shorts = list(METRIC_SHORT.values())
    wide = wide[[(v, m) for m in shorts for v in VALUE_COLS]]
    wide.columns = [f"{m}__{v}" for v, m in wide.columns]
    return wide.reset_index()


def _fmt_cell(row: pd.Series, short: str, alpha: float) -> str:
    p = row[f"{short}__p_raw"]
    if pd.isna(p):
        return "n/a"
    p_holm = row[f"{short}__p_holm_level"]
    med = row[f"{short}__median_diff"]
    star = "**" if p_holm < alpha else ("*" if p < alpha else "")
    return f"{med:+.4f} p={p:.3g} holm={p_holm:.3g}{star}"


def print_console_summary(wide: pd.DataFrame, alpha: float) -> None:
    show = wide[wide["level"].isin(CONSOLE_LEVELS)]
    print("\n" + "=" * 100)
    print(f"SUMMARY  (diff = {PROPOSED_LABEL_FAMILY} - NOISY; ** signif. after Holm within level, * only raw p)")
    print("=" * 100)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for gkey, g in show.groupby(
            ["scenario", "analysis_set", "unit", "comparison"], sort=False
        ):
            print("\n--- " + " | ".join(map(str, gkey)))
            table = []
            for _, r in g.iterrows():
                line = {
                    "level": r["level"], "dataset": r["dataset"],
                    "family": r["architecture_family"], "arch": r["experiment"],
                    "n": r["n_pairs"],
                }
                for short in METRIC_SHORT.values():
                    line[short] = _fmt_cell(r, short, alpha)
                table.append(line)
            print(pd.DataFrame(table).to_string(index=False))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    inputs = {
        "single": args.input_single,
        "multiple": args.input_multiple,
        "icu": args.input_icu,
    }

    long_frames: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, object]] = []
    for scenario in args.scenarios:
        df = load_scenario(
            scenario, inputs[scenario], args.checkpoint, getattr(args, f"{scenario}_datasets")
        )
        if df is None:
            continue
        for comparison, (noisy_base, dist_bases) in COMPARISONS.items():
            pairs, coverage = build_pairs(df, scenario, comparison, noisy_base, dist_bases)
            coverage_rows.extend(coverage)
            if pairs.empty:
                print(f"[warn] {scenario}/{comparison}: no NOISY/{PROPOSED_LABEL_FAMILY} pairs found")
                continue
            for unit, unit_pairs in make_units(
                pairs, scenario, args.icu_unit, args.icu_run_aggregation
            ).items():
                long_frames.append(
                    to_long(
                        unit_pairs, scenario, comparison, unit,
                        args.noise_architectures, args.noise_datasets,
                    )
                )

    if coverage_rows:
        cov = pd.DataFrame(coverage_rows).sort_values(["scenario", "comparison", "experiment", "dataset"])
        cov.to_csv(os.path.join(args.output_dir, "coverage.csv"), index=False)
        print(f"Saved coverage to: {os.path.join(args.output_dir, 'coverage.csv')}")
    if not long_frames:
        print("[warn] nothing to test - exiting.")
        return

    long_df = pd.concat(long_frames, ignore_index=True)
    unknown = sorted(long_df.loc[long_df["architecture_family"] == "unknown", "experiment"].unique())
    if unknown:
        print(f"[warn] no architecture family for {unknown}: only in global/dataset/architecture levels")
    if args.analysis_set != "both":
        long_df = long_df[long_df["analysis_set"] == args.analysis_set].copy()
    long_df.drop(columns=["pair_id"]).to_csv(
        os.path.join(args.output_dir, "pairs_long.csv"), index=False
    )

    tests = run_tests(long_df, args.min_pairs)
    tests = add_adjusted(tests, ["scenario", "analysis_set", "unit", "comparison", "level"], "level")
    tests = add_adjusted(tests, ["scenario", "analysis_set", "unit", "comparison"], "all")
    for p_col in ("p_raw", "p_holm_level", "p_bh_level", "p_holm_all", "p_bh_all"):
        tests["sig_" + p_col[len("p_"):]] = tests[p_col] < args.alpha
    tests["alpha"] = args.alpha
    tests = tests.drop(columns=["signature"])

    wide = to_wide(tests)
    for scenario in sorted(tests["scenario"].unique()):
        tests[tests["scenario"] == scenario].to_csv(
            os.path.join(args.output_dir, f"tests_{scenario}.csv"), index=False
        )
        wide[wide["scenario"] == scenario].to_csv(
            os.path.join(args.output_dir, f"summary_{scenario}.csv"), index=False
        )
    tests.to_csv(os.path.join(args.output_dir, "tests_all_scenarios.csv"), index=False)
    print(f"Saved tests / summaries / pairs_long to: {args.output_dir}")

    print_console_summary(wide, args.alpha)


if __name__ == "__main__":
    main()
