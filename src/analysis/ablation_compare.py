"""
Paired ablation between two versions of the proposed method (A vs B), with data
tables and figures in results/ablation/<comparison>/.

Comparisons (--comparison):
  flujo : A = PARAMETRIC_cv5x3_tbn_m0.05_tests   (DISTRIBUTIONAL's test cascade first,
              CV duel only when no test accepts; the proposed method)
          B = PARAMETRIC_cv5x3_tbn_m0.05         (CV duel for every class)
  noisy : A = PARAMETRIC_cv5x3_tbn_m0.05_tests   (the proposed method)
          B = NOISY_1000_standard_cov_ledoit_wolf (Gaussian noise instead of the fitted
              distributions, same unlearning; "Gaussian" in the figures)
Both sides use notop (margin 10 for PARAMETRIC, NOISY's own notop) and
corrected_confidence_penalty / confidence_penalty, 1000 fixed synthetic samples.

Comparisons are PAIRED on the same item (architecture, dataset, forgotten classes / ICU
sequence) and within the same freezing mode.  efficientnet_v2_s runs trained before the
"embs" fix used the whole backbone and were renamed "*_nofreeze" (rename_embs_mislabeled.sh):
they are paired among themselves and reported as "efficientnet_v2_s (nofreeze)"; the
relaunched last-stage "_embs" runs pair once both sides exist.

Settings: single class, multiple classes (cifar100, k = 5/10/20/50) and ICU (cifar100,
20 steps), the latter with the units of src/analysis/analyze_noisy_vs_distributional.py:
the two sides are first paired STEP BY STEP (only the steps both have), then each ICU
sequence is one observation:
    ICU run_auc   area under the metric-vs-steps curve / step span   (main unit)
    ICU run_mean  plain mean over the common steps
    ICU final     last common step (extra)
    ICU step      every step is a pair (correlated: optimistic p, sensitivity check only)
H is computed per step and then aggregated.
Metrics: Acc_r (accuracy_score_after, higher is better), Acc_f (unlearning_accuracy_score,
lower is better), H = 2 Acc_r (1-Acc_f) / (Acc_r + 1-Acc_f), plus the embedding metrics
of the processed CSVs (reported, no "better" direction).
Paired two-sided Wilcoxon (zero_method="wilcox"), matched-pairs rank-biserial, Holm and
Benjamini-Hochberg over the three accuracy metrics and all the cells of a level
(setting x loss x level).  Levels: all, analysis set (main / noise: resnet18 and
efficientnet_v2_s on cifar10 / mnist, where methods tend to be equivalent), dataset,
family (conv / transformer), architecture, dataset x family, dataset x architecture,
and k (multiple).

Outputs in results/ablation/<comparison>/:
    paired_items.csv     every matched item, value of each side and difference (A - B)
    summary.csv          per setting x loss x level x group x metric
    icu_steps.csv        ICU: mean of each metric per step and side
    verdict.txt          short text summary
    plots/               figures (src/plots/plot_ablation_compare.py)
  flujo only: selection_by_group.csv, selection_crosstab.csv (family chosen per class)

    PYTHONPATH=. python src/analysis/ablation_compare.py --comparison noisy
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_PARAM = (r"^PARAMETRIC_{tag}_1000_standard_syntetic_fixed_(?P<freeze>embs|nofreeze)_forget_"
          r"(?P<loss>notop_margin_10\.0|corrected_confidence_penalty)$")
_NOISY = (r"^NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_(?P<freeze>embs|nofreeze)_forget_"
          r"(?P<loss>notop|confidence_penalty)$")
COMPARISONS = {
    "flujo": {
        "title": "selection flow of DISCERN (tests = PARAMETRIC_cv5x3_tbn_m0.05_tests, cv = PARAMETRIC_cv5x3_tbn_m0.05)",
        "a": "tests", "b": "cv",
        "regex": {"tests": _PARAM.format(tag=r"cv5x3_tbn_m0\.05_tests"),
                  "cv": _PARAM.format(tag=r"cv5x3_tbn_m0\.05")},
        "b_labels": {"notop": "CV-only (notop)", "confidence": "CV-only (conf.)"},
        "selection": {"PARAMETRIC_cv5x3_tbn_m0.05_tests": "tests", "PARAMETRIC_cv5x3_tbn_m0.05": "cv"},
    },
    "noisy": {
        "title": "DISCERN (PARAMETRIC_cv5x3_tbn_m0.05_tests) vs NOISY (Gaussian noise)",
        "a": "discern", "b": "noisy",
        "regex": {"discern": _PARAM.format(tag=r"cv5x3_tbn_m0\.05_tests"), "noisy": _NOISY},
        "b_labels": {"notop": "Gaussian (notop)", "confidence": "Gaussian (conf.)"},
        "selection": None,
    },
}
LOSS_LABEL = {"notop_margin_10.0": "notop", "notop": "notop",
              "corrected_confidence_penalty": "confidence", "confidence_penalty": "confidence"}
# metric -> +1 higher is better, -1 lower is better, 0 no direction
POLARITY = {"H": 1, "acc_r": 1, "acc_f": -1}
RENAME = {"accuracy_score_after": "acc_r", "unlearning_accuracy_score": "acc_f"}
EXTRA = ["maxprobability", "maxprobability_forget", "cohesion_forget_l2", "distance_inter_class_l2",
         "cohesion_forget_cos", "distance_inter_class_cos"]
ARCHITECTURE_FAMILY = {"resnet18": "conv", "efficientnet_v2_s": "conv",
                       "swin_t": "transformer", "vit_b_16": "transformer"}
NOISE_CELLS = {("resnet18", "cifar10"), ("resnet18", "mnist"),
               ("efficientnet_v2_s", "cifar10"), ("efficientnet_v2_s", "mnist")}
LEVELS = [("all", []), ("analysis set", ["analysis_set"]), ("dataset", ["dataset"]), ("family", ["family"]),
          ("architecture", ["arch"]), ("dataset x family", ["dataset", "family"]),
          ("dataset x architecture", ["dataset", "arch"])]


def harmonic(acc_r, acc_f):
    u = 1.0 - acc_f
    return np.where(acc_r + u > 0, 2 * acc_r * u / np.maximum(acc_r + u, 1e-12), 0.0)


def match_side(base, spec):
    """version folder -> (side, freeze, loss) or None."""
    for side, rx in spec["regex"].items():
        m = re.match(rx, base)
        if m:
            return side, m.group("freeze"), LOSS_LABEL[m.group("loss")]
    return None


def tidy(df, spec):
    """Keep the two sides; add side / freeze / loss / dataset / k / H / arch / item."""
    df = df.copy()
    mm = df["version"].astype(str).str.split("/").str[0].map(lambda b: match_side(b, spec))
    df = df[mm.notna()].copy()
    mm = mm[mm.notna()]
    df["side"], df["freeze"], df["loss"] = mm.str[0], mm.str[1], mm.str[2]
    df["dataset"] = df["id_database"].str.extract(r"^([a-z0-9]+)_classes_to_remove")[0]
    df["k"] = df["id_database"].str.extract(r"classes_to_remove_([0-9_]+)$")[0].str.count("_") + 1
    df = df.rename(columns=RENAME)
    for c in ["acc_r", "acc_f"] + [c for c in EXTRA if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["H"] = harmonic(df["acc_r"].to_numpy(), df["acc_f"].to_numpy())
    # efficientnet_v2_s has both whole-backbone (nofreeze) and last-stage (embs) pairs:
    # keep them apart in the per-architecture tables
    df["arch"] = df["experiment"] + np.where(df["freeze"] == "nofreeze", " (nofreeze)", "")
    df["item"] = df["experiment"] + "|" + df["id_database"]
    return df


def load_processed(paths, multiple, spec):
    frames = [pd.read_csv(p, dtype={"epoch": str}, on_bad_lines="skip") for p in paths if os.path.exists(p)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  (missing {missing})")
    df = tidy(pd.concat(frames, ignore_index=True), spec)
    df = df[df["epoch"] == "best"]
    df = df[(df["k"] > 1) if multiple else (df["k"] == 1)]
    return df.drop_duplicates(["side", "freeze", "loss", "item"], keep="last")


def load_icu(root, spec):
    rows = []
    for path in glob.glob(os.path.join(root, "*", "*", "original_best", "unlearning_results_ICU.csv")):
        if match_side(path.split(os.sep)[-3], spec) is None:
            continue
        d = pd.read_csv(path, on_bad_lines="skip")
        d = d[pd.to_numeric(d["steps"], errors="coerce").notna()].copy()
        d["steps"] = d["steps"].astype(int)
        d["version"] = path.split(os.sep)[-3]          # trust the folder name
        rows.append(d.drop_duplicates("steps", keep="last"))
    return tidy(pd.concat(rows, ignore_index=True), spec)


def paired(df, setting, metrics, spec, extra_keys=()):
    """Wide table: one row per matched (loss, freeze, item[, extra]), <metric>_<a> / _<b> / _diff (a - b)."""
    a, b = spec["a"], spec["b"]
    keys = ["loss", "freeze", "arch", "dataset", "k", "item", *extra_keys]
    wide = df.pivot_table(index=keys, columns="side", values=metrics, aggfunc="first")
    # plain column names (reset_index() of a pivot keeps ('key', '') tuples, which
    # then do not match the string columns of the other settings)
    out = wide.index.to_frame(index=False)
    out.insert(0, "setting", setting)
    base_arch = out["arch"].str.split(" ").str[0]
    out["family"] = base_arch.map(ARCHITECTURE_FAMILY)
    out["analysis_set"] = ["noise" if (x, d) in NOISE_CELLS else "main" for x, d in zip(base_arch, out["dataset"])]
    ok = np.zeros(len(out), dtype=bool)
    for mtr in metrics:
        if (mtr, a) not in wide.columns or (mtr, b) not in wide.columns:
            continue
        va, vb = wide[(mtr, a)].to_numpy(), wide[(mtr, b)].to_numpy()
        out[f"{mtr}_{a}"], out[f"{mtr}_{b}"], out[f"{mtr}_diff"] = va, vb, va - vb
        if mtr == "H":
            ok = np.isfinite(va) & np.isfinite(vb)
    return out[ok].reset_index(drop=True)


def normalized_auc(steps, values):
    """Same as analyze_noisy_vs_distributional.normalized_auc: trapezoidal area under
    values(steps) divided by the step span (a constant curve gives that constant)."""
    x, y = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 2 or x[-1] == x[0]:
        return float(y[0])
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)) / float(x[-1] - x[0])


def icu_units(icu, metrics, spec):
    """Pair the two sides step by step (common steps only), then one row per ICU
    sequence for run_auc / run_mean / final; 'step' keeps every step as a pair."""
    steps = paired(icu, "ICU step", metrics, spec, extra_keys=("steps",))
    if steps.empty:
        return steps
    keys = ["loss", "freeze", "arch", "dataset", "k", "item", "family", "analysis_set"]
    cols = [f"{m}_{s}" for m in metrics for s in (spec["a"], spec["b"])]
    units = {"ICU run_auc": [], "ICU run_mean": [], "ICU final": []}
    for key, g in steps.groupby(keys, sort=True):
        g = g.sort_values("steps")
        base = dict(zip(keys, key), n_steps=len(g))
        units["ICU run_auc"].append({**base, **{c: normalized_auc(g["steps"], g[c]) for c in cols}})
        units["ICU run_mean"].append({**base, **{c: float(g[c].mean()) for c in cols}})
        units["ICU final"].append({**base, **{c: float(g[c].iloc[-1]) for c in cols}})
    out = []
    for setting, rows in units.items():
        u = pd.DataFrame(rows)
        u.insert(0, "setting", setting)
        for m in metrics:
            u[f"{m}_diff"] = u[f"{m}_{spec['a']}"] - u[f"{m}_{spec['b']}"]
        out.append(u)
    return pd.concat(out + [steps], ignore_index=True)


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * p[i]))
        out[i] = running
    return out


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def test_rows(pairs, metrics, group_cols, level, spec):
    a, b = spec["a"], spec["b"]
    rows = []
    for (setting, loss), g0 in pairs.groupby(["setting", "loss"]):
        groups = [("all", g0)] if not group_cols else list(g0.groupby(group_cols))
        for key, g in groups:
            key = key if isinstance(key, tuple) else (key,)
            for mtr in metrics:
                if f"{mtr}_diff" not in g.columns:
                    continue
                d = g[f"{mtr}_diff"].dropna().to_numpy()
                if len(d) == 0:
                    continue
                pol = POLARITY.get(mtr, 0)
                nz = d[d != 0]
                try:
                    p = stats.wilcoxon(d, zero_method="wilcox").pvalue if len(nz) >= 1 else 1.0
                except ValueError:
                    p = np.nan
                # matched-pairs rank-biserial on the non-zero differences
                if len(nz):
                    rk = stats.rankdata(np.abs(nz))
                    rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
                else:
                    rb = 0.0
                row = {"setting": setting, "loss": loss, "level": level,
                       "group": "all" if not group_cols else " | ".join(map(str, key)),
                       "metric": mtr, "n_pairs": len(d), "n_nonzero": len(nz),
                       f"mean_{a}": g[f"{mtr}_{a}"].mean(), f"mean_{b}": g[f"{mtr}_{b}"].mean(),
                       f"mean_diff_{a}_minus_{b}": d.mean(), "median_diff": np.median(d),
                       f"{a}_better": (pol * d > 0).sum() if pol else np.nan,
                       f"{b}_better": (pol * d < 0).sum() if pol else np.nan,
                       "ties": int((d == 0).sum()), "rank_biserial": rb, "p_wilcoxon": p}
                if pol and np.isfinite(p) and rb != 0:
                    row["favours"] = a if pol * rb > 0 else b
                else:
                    row["favours"] = "-" if pol else "n/a"
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        # family of tests = setting x loss x level, the three accuracy metrics and all
        # cells together (as analyze_noisy_vs_distributional); the embedding metrics,
        # which have no direction, are corrected among themselves
        out["p_holm"], out["p_bh"] = np.nan, np.nan
        out["_fam"] = np.where(out["metric"].isin(list(POLARITY)), "acc", "emb")
        for _, idx in out.groupby(["setting", "loss", "_fam"]).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_wilcoxon"])
            out.loc[idx, "p_bh"] = bh(out.loc[idx, "p_wilcoxon"])
        out = out.drop(columns="_fam")
    return out


def selection_tables(path, methods):
    """flujo only: family chosen per class by each side (results/chosen_distributions.csv)."""
    if not os.path.exists(path):
        print(f"  (missing {path}: run src/analysis/export_chosen_distributions.py)")
        return None, None
    ch = pd.read_csv(path)
    ch = ch[ch["method"].isin(list(methods)) & (ch["subgroup"] == 0)].copy()
    ch["side"] = ch["method"].map(methods)
    sides = list(dict.fromkeys(methods.values()))
    w = ch.pivot_table(index=["architecture", "dataset", "class"], columns="side",
                       values="dist_type", aggfunc="first").dropna()
    dec = ch[ch["side"] == sides[0]].set_index(["architecture", "dataset", "class"])["decided_by"]
    w[f"decided_by_{sides[0]}"] = dec.reindex(w.index)
    w["same_family"] = w[sides[0]] == w[sides[1]]
    by = w.reset_index().groupby(["architecture", "dataset"]).agg(
        n_classes=("same_family", "size"), same_family=("same_family", "sum"),
        decided_by_test=(f"decided_by_{sides[0]}", lambda s: int((s == "test").sum())))
    by["pct_same_family"] = 100 * by["same_family"] / by["n_classes"]
    cross = pd.crosstab(w[sides[1]], w[sides[0]]).rename_axis(index=f"family_{sides[1]}",
                                                              columns=f"family_{sides[0]}")
    return by.reset_index(), cross


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", choices=list(COMPARISONS), required=True)
    ap.add_argument("--single", nargs="+", default=["results/unlearning_results_single_parametric.csv",
                                                     "results/unlearning_results_single_all_new.csv"])
    ap.add_argument("--multiple", nargs="+", default=["results/unlearning_results_multiple_parametric.csv",
                                                       "results/unlearning_results_multiple_all_new.csv"])
    ap.add_argument("--icu-root", default="results_ICU")
    ap.add_argument("--chosen", default="results/chosen_distributions.csv")
    ap.add_argument("--out-dir", default=None, help="default results/ablation/<comparison>")
    ap.add_argument("--no-plots", action="store_true", help="skip the figures (<out-dir>/plots)")
    ap.add_argument("--include-nofreeze", action="store_true",
                    help="also keep the historical efficientnet whole-backbone (*_nofreeze) pairs; by default "
                         "they are dropped (the relaunched last-stage runs exist), otherwise efficientnet "
                         "would count twice in the global levels")
    ap.add_argument("--keep-all", action="store_true",
                    help="figures: no forget-accuracy > 0.2 rule in the boxplots (ICU never applies it)")
    args = ap.parse_args(argv)
    spec = COMPARISONS[args.comparison]
    a, b = spec["a"], spec["b"]
    out_dir = args.out_dir or os.path.join("results", "ablation", args.comparison)
    os.makedirs(out_dir, exist_ok=True)
    pd.set_option("display.width", 250)

    single = load_processed(args.single, False, spec)
    multi = load_processed(args.multiple, True, spec)
    icu = load_icu(args.icu_root, spec)
    if not args.include_nofreeze:
        single, multi, icu = (d[d["freeze"] != "nofreeze"] for d in (single, multi, icu))
    metrics = ["H", "acc_r", "acc_f"] + [c for c in EXTRA if c in single.columns]
    icu_metrics = ["H", "acc_r", "acc_f"]

    pairs = pd.concat([paired(single, "single", metrics, spec), paired(multi, "multiple", metrics, spec),
                       icu_units(icu, icu_metrics, spec)], ignore_index=True)
    pairs.to_csv(os.path.join(out_dir, "paired_items.csv"), index=False)

    summary = pd.concat(
        [test_rows(pairs, metrics, cols, level, spec) for level, cols in LEVELS]
        + [test_rows(pairs[pairs["setting"] == "multiple"], metrics, ["k"], "k", spec)], ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    steps = icu.groupby(["loss", "freeze", "arch", "steps", "side"])[icu_metrics].mean().unstack("side")
    steps.columns = [f"{m}_{s}" for m, s in steps.columns]
    steps.reset_index().to_csv(os.path.join(out_dir, "icu_steps.csv"), index=False)

    by = cross = None
    if spec["selection"]:
        by, cross = selection_tables(args.chosen, spec["selection"])
        if by is not None:
            by.to_csv(os.path.join(out_dir, "selection_by_group.csv"), index=False)
            cross.to_csv(os.path.join(out_dir, "selection_crosstab.csv"))

    # ---- verdict ------------------------------------------------------------
    diff = f"mean_diff_{a}_minus_{b}"
    lines = [f"Ablation: {spec['title']}",
             f"diff = {a} - {b}; H and Acc_r higher is better, Acc_f lower is better; paired Wilcoxon, "
             "Holm over metrics x cells of each level (** Holm < 0.05, * raw p < 0.05).",
             "ICU main unit = run_auc (as analyze_noisy_vs_distributional); ICU step is only a sensitivity check.",
             ""]
    top = summary[(summary["level"] == "all") & summary["metric"].isin(list(POLARITY))]
    for _, r in top.iterrows():
        sig = "**" if r["p_holm"] < 0.05 else ("*" if r["p_wilcoxon"] < 0.05 else "")
        lines.append(f"{r['setting']:<13} {r['loss']:<11} {r['metric']:<6} n={r['n_pairs']:<4} "
                     f"{b}={r[f'mean_{b}']:.4f} {a}={r[f'mean_{a}']:.4f} diff={r[diff]:+.4f} "
                     f"{a} better {r[f'{a}_better']:.0f} / {b} better {r[f'{b}_better']:.0f} / ties {r['ties']} "
                     f"p={r['p_wilcoxon']:.3g}{sig} -> {r['favours']}")
    for level in ("analysis set", "architecture"):
        rows = summary[(summary["level"] == level) & (summary["metric"] == "H")]
        lines += ["", f"H per {level} (diff {a} - {b}, p):"]
        for _, r in rows.iterrows():
            lines.append(f"  {r['setting']:<13} {r['loss']:<11} {r['group']:<30} n={r['n_pairs']:<4} "
                         f"diff={r[diff]:+.4f} p={r['p_wilcoxon']:.3g} -> {r['favours']}")
    if by is not None:
        lines += ["", "Family chosen per class (same in both / decided by a test):",
                  by.to_string(index=False), "", cross.to_string()]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "verdict.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved in {out_dir}/")

    if not args.no_plots:
        from src.plots.plot_ablation_compare import make_all
        make_all(single, multi, icu, spec, os.path.join(out_dir, "plots"), args.keep_all)


if __name__ == "__main__":
    main()
