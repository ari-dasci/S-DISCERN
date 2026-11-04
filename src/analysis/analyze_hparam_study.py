"""Paired analysis of the DISTRIBUTIONAL hyper-parameter mini study.

Input : results/hparam_study_raw.csv (built by src/collect/collect_hparam_study.py)
Output: results/hyperparam/  (see results/hyperparam/README.md)

Protocol (ablation-style, no p-value fishing):
  * Replicate unit = the removed class. Every config is trained with seed 42 on
    the same (architecture, dataset, class) triples, so configs are PAIRED by
    that triple. cifar10/mnist give 10 pairs per cell, cifar100 gives 20.
  * Primary report: mean +- sd and a 95% bootstrap CI (classes resampled,
    stratified by (architecture, dataset) cell when pooling).
  * Secondary: paired difference vs the default config (margin 10.0 /
    1000 synthetic samples) with a bootstrap CI and a paired Wilcoxon
    signed-rank test, Holm-corrected within each (study, scope, metric)
    family.

Studies:
  margin       : notop, 1000 synthetic, margin in {0, 5, 10, 20}     default 10.0
  synth_notop  : notop, margin 10,  n_synth in {100 ... 10000}       default 1000
  synth_ccp    : corrected_confidence_penalty, n_synth {100 ... 10000} default 1000

Usage: python3 src/analysis/analyze_hparam_study.py [--raw results/hparam_study_raw.csv] [--out results/hyperparam]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE_TAG = "standard_cov_ledoit_wolf_syntetic_fixed_embs_forget"
B = 10000
RNG_SEED = 0
METRICS = {  # column -> (short name, higher_is_better)
    "accuracy_score_after": ("acc_retain", True),
    "unlearning_accuracy_score": ("acc_forget", False),
    "harmonic": ("H", True),
}
EXPECTED = {"cifar10": 10, "mnist": 10, "cifar100": 20}

STUDIES = {
    # name: (method, factor column, default level, fixed column, fixed value)
    "margin": ("notop", "margin", 10.0, "n_synthetic", 1000),
    "synth_notop": ("notop", "n_synthetic", 1000, "margin", 10.0),
    "synth_ccp": ("corrected_confidence_penalty", "n_synthetic", 1000, None, None),
}


PARAMETRIC_BASE_TAG = "standard_syntetic_fixed_embs_forget"


def version_table(parametric_tag=None):
    """folder -> (method, n_synthetic, margin, is_legacy).
    parametric_tag (e.g. "cv5x3_tbn_m0.05_tests"): the PARAMETRIC / DISCERN study
    (PARAMETRIC_<tag>_<n>_standard_syntetic_fixed_embs_forget_...) instead of DISTRIBUTIONAL."""
    v = {}
    prefix, base = ("DISTRIBUTIONAL", BASE_TAG) if parametric_tag is None else \
        (f"PARAMETRIC_{parametric_tag}", PARAMETRIC_BASE_TAG)
    for n in (100, 500, 1000, 2000, 5000, 10000):
        for m in (0.0, 5.0, 10.0, 20.0):
            v[f"{prefix}_{n}_{base}_notop_margin_{m}"] = ("notop", n, m, False)
        if parametric_tag is None:
            # DISTRIBUTIONAL: margin 10.0 used to be saved without the suffix
            v[f"{prefix}_{n}_{base}_notop"] = ("notop", n, 10.0, True)
        v[f"{prefix}_{n}_{base}_corrected_confidence_penalty"] = (
            "corrected_confidence_penalty", n, np.nan, False)
    return v


def harmonic(acc, unl):
    # forget accuracy is "lower is better" -> invert (same as src/plots/extract_methods.py)
    u = 1.0 - unl
    s = acc + u
    return np.where(s == 0, 0.0, 2 * acc * u / np.where(s == 0, 1, s))


def load_per_class(raw_path, parametric_tag=None):
    df = pd.read_csv(raw_path, dtype={"epoch": str}, on_bad_lines="skip")
    df["folder"] = df["version"].str.split("/").str[0]
    vt = version_table(parametric_tag)
    df = df[df["folder"].isin(vt)].copy()
    info = pd.DataFrame(df["folder"].map(vt).tolist(), index=df.index,
                        columns=["method", "n_synthetic", "margin", "legacy_folder"])
    df = pd.concat([df, info], axis=1)
    df["dataset"] = df["id_database"].str.extract(r"^([a-z0-9]+)_classes_to_remove")[0]
    df["class"] = df["id_database"].str.extract(r"classes_to_remove_(\d+)$")[0]
    df = df.dropna(subset=["dataset", "class"])
    df["class"] = df["class"].astype(int)
    df = df.rename(columns={"experiment": "architecture"})

    # one row per (config, arch, dataset, class): epoch "best" first, else "00";
    # if both legacy and new-style folders exist, keep the legacy one (the
    # run already used in the main paper tables)
    df["_best"] = (df["epoch"] == "best").astype(int)
    df = df.sort_values(["_best", "legacy_folder"], ascending=[False, False])
    df = df.drop_duplicates(["method", "n_synthetic", "margin", "architecture", "dataset", "class"])
    df["harmonic"] = harmonic(df["accuracy_score_after"].to_numpy(),
                              df["unlearning_accuracy_score"].to_numpy())
    keep = ["method", "n_synthetic", "margin", "architecture", "dataset", "class", "epoch",
            "legacy_folder", "folder", "accuracy_score_after", "unlearning_accuracy_score", "harmonic",
            "maxprobability", "maxprobability_forget"]
    return df[keep].sort_values(["method", "n_synthetic", "margin", "architecture", "dataset", "class"])


def study_frame(pc, study):
    method, factor, _, fixed_col, fixed_val = STUDIES[study]
    d = pc[pc["method"] == method]
    if fixed_col is not None:
        d = d[np.isclose(d[fixed_col].astype(float), fixed_val)]
    d = d.copy()
    d["level"] = d[factor].astype(float)
    return d


def strat_boot_mean(values, strata, rng):
    """Bootstrap distribution (length B) of the pooled mean, resampling within strata."""
    values = np.asarray(values, float)
    strata = np.asarray(strata)
    total = np.zeros(B)
    n_all = len(values)
    for s in np.unique(strata):
        v = values[strata == s]
        idx = rng.integers(0, len(v), size=(B, len(v)))
        total += v[idx].sum(axis=1)
    return total / n_all


def ci(dist):
    return np.percentile(dist, 2.5), np.percentile(dist, 97.5)


def scopes(d):
    """(scope_type, scope_label, sub-frame): per cell, per dataset, per architecture, all."""
    for (a, ds), g in d.groupby(["architecture", "dataset"]):
        yield "cell", f"{a}|{ds}", g
    for ds, g in d.groupby("dataset"):
        yield "dataset", ds, g
    for a, g in d.groupby("architecture"):
        yield "architecture", a, g
    yield "all", "all", d


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def paired_wilcoxon(diff):
    diff = np.asarray(diff, float)
    if len(diff) < 5 or np.allclose(diff, 0):
        return np.nan if len(diff) < 5 else 1.0
    try:
        # zsplit: many exact zeros in forget accuracy; keep them in the ranking
        return wilcoxon(diff, zero_method="zsplit").pvalue
    except ValueError:
        return np.nan


def summarize(d, study, rng):
    rows = []
    for scope_type, label, g in scopes(d):
        strata = g["architecture"] + "|" + g["dataset"]
        for lvl, gl in g.groupby("level"):
            row = {"study": study, "scope_type": scope_type, "scope": label, "level": lvl,
                   "n_classes": len(gl)}
            for col, (short, _) in METRICS.items():
                v = gl[col].to_numpy()
                lo, hi = ci(strat_boot_mean(v, strata.loc[gl.index], rng))
                row[f"{short}_mean"] = v.mean()
                row[f"{short}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
                row[f"{short}_ci_lo"] = lo
                row[f"{short}_ci_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def paired_vs(d, study, rng, reference="default"):
    """Paired differences level - reference for every level, every scope, every metric.

    reference="default": the study's default level.
    reference="best"   : within each scope, the level with the highest mean H.
    """
    default = STUDIES[study][2]
    keys = ["architecture", "dataset", "class"]
    rows = []
    for scope_type, label, g in scopes(d):
        if reference == "best":
            ref = g.groupby("level")["harmonic"].mean().idxmax()
        else:
            ref = default
        base = g[np.isclose(g["level"], ref)].set_index(keys)
        for lvl, gl in g.groupby("level"):
            if np.isclose(lvl, ref):
                continue
            other = gl.set_index(keys)
            common = base.index.intersection(other.index)
            if len(common) == 0:
                continue
            strata = np.array([f"{a}|{ds}" for a, ds, _ in common])
            for col, (short, hib) in METRICS.items():
                diff = other.loc[common, col].to_numpy() - base.loc[common, col].to_numpy()
                lo, hi = ci(strat_boot_mean(diff, strata, rng))
                rows.append({
                    "study": study, "scope_type": scope_type, "scope": label,
                    "reference": reference, "reference_level": ref, "level": lvl,
                    "metric": short, "higher_is_better": hib, "n_pairs": len(common),
                    "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1) if len(diff) > 1 else np.nan,
                    "ci_lo": lo, "ci_hi": hi,
                    "ci_excludes_0": bool(lo > 0 or hi < 0),
                    "frac_level_better": float(np.mean(diff > 0) if hib else np.mean(diff < 0)),
                    "frac_ties": float(np.mean(np.isclose(diff, 0))),
                    "p_wilcoxon": paired_wilcoxon(diff),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for _, idx in out.groupby(["study", "scope_type", "metric"]).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_wilcoxon"].to_numpy())
        out["significant_holm_0.05"] = out["p_holm"] < 0.05
    return out


def default_verdict(summary, vs_best, study, sesoi=None):
    """Per scope: rank of the default level by mean H and its gap to the best level.

    Statistical verdict, using two criteria on the paired H difference
    (default - best): bootstrap CI of the mean excludes 0, and Holm p < 0.05.
      both      -> worse_than_best
      neither   -> indistinguishable_from_best
      only one  -> mixed_evidence (typically: a clear mean gap driven by a few
                   large failures, which the rank-based Wilcoxon does not flag)
    Practical verdict (only if --sesoi is given, declared a priori): the CI of
    the gap lies entirely within [-sesoi, +sesoi] -> practically_equivalent;
    CI entirely beyond -sesoi -> practically_worse; otherwise inconclusive."""
    default = STUDIES[study][2]
    rows = []
    for (st, sc), g in summary.groupby(["scope_type", "scope"]):
        g = g.sort_values("H_mean", ascending=False).reset_index(drop=True)
        best = g.loc[0, "level"]
        dflt = g[np.isclose(g["level"], default)]
        if dflt.empty:
            continue
        rank = int(dflt.index[0]) + 1
        row = {"study": study, "scope_type": st, "scope": sc, "default_level": default,
               "default_rank_by_H": rank, "n_levels": len(g), "best_level": best,
               "H_default": float(dflt["H_mean"].iloc[0]), "H_best": float(g.loc[0, "H_mean"])}
        if np.isclose(best, default):
            row.update(verdict="default_is_best", diff_H_default_minus_best=0.0,
                       ci_lo=np.nan, ci_hi=np.nan, p_holm=np.nan, frac_default_better=np.nan)
            if sesoi is not None:
                row["practical_verdict"] = "default_is_best"
        else:
            r = vs_best[(vs_best["scope_type"] == st) & (vs_best["scope"] == sc)
                        & (vs_best["metric"] == "H") & np.isclose(vs_best["level"], default)]
            if r.empty:
                continue
            r = r.iloc[0]
            n_crit = int(bool(r["ci_excludes_0"])) + int(bool(r["p_holm"] < 0.05))
            verdict = {2: "worse_than_best", 1: "mixed_evidence", 0: "indistinguishable_from_best"}[n_crit]
            row.update(verdict=verdict, diff_H_default_minus_best=r["mean_diff"],
                       ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_holm=r["p_holm"],
                       frac_default_better=r["frac_level_better"])
            if sesoi is not None:
                if r["ci_lo"] >= -sesoi and r["ci_hi"] <= sesoi:
                    pv = "practically_equivalent"
                elif r["ci_hi"] < -sesoi:
                    pv = "practically_worse"
                else:
                    pv = "inconclusive"
                row["practical_verdict"] = pv
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_tables(summary, study, factor_name):
    """mean +- sd of retain acc / forget acc / H per (architecture, dataset) x level."""
    lines = [f"## {study}  (factor: {factor_name}, default: {STUDIES[study][2]})", ""]
    for short, title in (("H", "Harmonic mean H (higher better)"),
                         ("acc_retain", "Retain accuracy (higher better)"),
                         ("acc_forget", "Forget accuracy (lower better)")):
        s = summary[summary["scope_type"].isin(["cell", "dataset", "all"])]
        levels = sorted(s["level"].unique())
        lines += [f"### {title}", "",
                  "| scope | " + " | ".join(f"{l:g}" for l in levels) + " |",
                  "|---|" + "---|" * len(levels)]
        for (st, sc), g in s.groupby(["scope_type", "scope"], sort=False):
            g = g.set_index("level")
            best = g[f"{short}_mean"].max() if short != "acc_forget" else g[f"{short}_mean"].min()
            cells = []
            for l in levels:
                if l not in g.index:
                    cells.append("—")
                    continue
                m, sd = g.loc[l, f"{short}_mean"], g.loc[l, f"{short}_sd"]
                txt = f"{m:.3f} ± {sd:.3f}"
                cells.append(f"**{txt}**" if np.isclose(m, best) else txt)
            lines.append(f"| {sc} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def plot(summary, study, factor_name, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cells = summary[summary["scope_type"] == "cell"].copy()
    cells[["architecture", "dataset"]] = cells["scope"].str.split("|", expand=True)
    datasets = sorted(cells["dataset"].unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.5 * len(datasets), 7), squeeze=False)
    for j, ds in enumerate(datasets):
        for i, (short, lab) in enumerate((("H", "H"), ("acc_retain", "retain acc"))):
            ax = axes[i, j]
            for a, g in cells[cells["dataset"] == ds].groupby("architecture"):
                g = g.sort_values("level")
                ax.plot(g["level"], g[f"{short}_mean"], marker="o", label=a)
                ax.fill_between(g["level"], g[f"{short}_ci_lo"], g[f"{short}_ci_hi"], alpha=0.15)
            ax.axvline(STUDIES[study][2], color="grey", ls="--", lw=0.8)
            if factor_name == "n_synthetic":
                ax.set_xscale("log")
            ax.set_title(f"{ds} — {lab}")
            ax.set_xlabel(factor_name)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{study} (mean, 95% bootstrap CI; dashed = default)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{study}.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parametric-tag", default=None,
                    help="analyse the PARAMETRIC / DISCERN study of this tag (e.g. cv5x3_tbn_m0.05_tests); "
                         "defaults --raw to results/hparam_study_parametric_<tag>_raw.csv (written by "
                         "collect_hparam_study.py --parametric-tag) and --out to results/ablation/hparam")
    ap.add_argument("--raw", default=None, help="default results/hparam_study_raw.csv")
    ap.add_argument("--out", default=None, help="default results/hyperparam")
    ap.add_argument("--sesoi", type=float, default=None,
                    help="smallest H difference of practical interest (e.g. 0.01). Must be "
                         "fixed a priori and justified in the text; enables practical_verdict")
    args = ap.parse_args()
    tag = args.parametric_tag
    args.raw = args.raw or ("results/hparam_study_raw.csv" if tag is None
                            else f"results/hparam_study_parametric_{tag}_raw.csv")
    args.out = args.out or ("results/hyperparam" if tag is None else "results/ablation/hparam")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    pc = load_per_class(args.raw, tag)
    pc.to_csv(os.path.join(args.out, "per_class.csv"), index=False)

    all_summary, all_default, all_best, all_verdict, md, coverage = [], [], [], [], [], []
    for study, (_, factor, *_rest) in STUDIES.items():
        d = study_frame(pc, study)
        cov = d.groupby(["level", "architecture", "dataset"]).size().rename("n_classes").reset_index()
        cov["expected"] = cov["dataset"].map(EXPECTED)
        cov["study"] = study
        coverage.append(cov)

        s = summarize(d, study, rng)
        vd = paired_vs(d, study, rng, "default")
        vb = paired_vs(d, study, rng, "best")
        all_summary.append(s)
        all_default.append(vd)
        all_best.append(vb)
        all_verdict.append(default_verdict(s, vb, study, args.sesoi))
        md.append(markdown_tables(s, study, factor))
        plot(s, study, factor, args.out)
        print(f"{study}: {len(d)} per-class rows, levels {sorted(d['level'].unique())}")

    pd.concat(all_summary).to_csv(os.path.join(args.out, "summary_mean_sd_ci.csv"), index=False)
    pd.concat(all_default).to_csv(os.path.join(args.out, "paired_vs_default.csv"), index=False)
    pd.concat(all_best).to_csv(os.path.join(args.out, "paired_vs_best.csv"), index=False)
    verdict = pd.concat(all_verdict)
    verdict.to_csv(os.path.join(args.out, "default_vs_best_verdict.csv"), index=False)
    cov = pd.concat(coverage)
    cov.to_csv(os.path.join(args.out, "coverage.csv"), index=False)
    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write("# Hyper-parameter study — mean ± sd over removed classes\n\n"
                "Bold = best level in the row (not a significance claim).\n\n")
        f.write("\n\n".join(md))

    bad = cov[cov["n_classes"] != cov["expected"]]
    if len(bad):
        print("\nWARNING incomplete cells:\n" + bad.to_string(index=False))
    print("\nDefault vs best (pooled scopes):")
    cols = ["study", "scope", "default_rank_by_H", "best_level", "verdict",
            "diff_H_default_minus_best", "ci_lo", "ci_hi", "p_holm"]
    if args.sesoi is not None:
        cols.append("practical_verdict")
    print(verdict[verdict["scope_type"].isin(["dataset", "all"])][cols].to_string(index=False))
    print(f"\nWrote outputs to {args.out}/")


if __name__ == "__main__":
    main()
