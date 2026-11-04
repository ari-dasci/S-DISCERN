"""
Export, per method / architecture / dataset / class, the distribution family
chosen by PARAMETRIC (every variant) and by DISTRIBUTIONAL, and count how
often the norm and the direction of the embeddings are independent.

Sources: the fit caches written by setup() (fitted once on the original model,
all classes of the dataset):
  PARAMETRIC     : standard_cov_parametric_<tag>/<arch>_<dataset>_<version>_dist_infos.pt
  DISTRIBUTIONAL : standard_cov_ledoit_wolf/...   (--dist-dirs to add others)

Norm / direction independence is the step-0 test of PARAMETRIC (Spearman,
ridge LOO R2 and dCor with permutations, BH over the three).  It depends only
on the embeddings, so every PARAMETRIC variant should agree; DISTRIBUTIONAL
does not test it (its vMF / Kent always sample the norm independently).

Writes results/chosen_distributions.csv (one row per class and sub-group) and
prints the summaries.

    PYTHONPATH=. python src/analysis/export_chosen_distributions.py
"""

import argparse
import glob
import os
import re

import pandas as pd
import torch

CACHE_RE = re.compile(r"^(?P<arch>.+?)_(?P<dataset>cifar100|cifar10|mnist|dtd|letters)_(?P<version>.+)_dist_infos\.pt$")


def method_label(cache_dir):
    d = os.path.basename(os.path.normpath(cache_dir))
    if d.startswith("standard_cov_parametric_"):
        return "PARAMETRIC_" + d[len("standard_cov_parametric_"):]
    return "DISTRIBUTIONAL_" + d


def rows_from_cache(path, method):
    m = CACHE_RE.match(os.path.basename(path))
    if not m:
        print(f"  (skip unrecognised file {path})")
        return []
    dist_infos = torch.load(path, map_location="cpu", weights_only=False)["dist_infos"]
    out = []
    for c, infos in dist_infos.items():
        if not infos:
            continue
        if not isinstance(infos, list):
            infos = [(infos, 1.0)]
        for sub, (info, weight) in enumerate(infos):
            prm = info.params if isinstance(info.params, dict) else {}
            inner = prm.get("inner", prm)
            rep = getattr(info, "selection", None) or {}
            s0 = rep.get("step0", {})
            row = {
                "method": method, "architecture": m.group("arch"), "dataset": m.group("dataset"),
                "class": int(c), "subgroup": sub, "n_subgroups": len(infos), "weight": float(weight),
                "dist_type": info.dist_type, "decided_by": getattr(info, "decided_by", None),
                "pipeline": rep.get("pipeline"),
                "test_accepted_by": rep.get("tests", {}).get("accepted_by"),
                "subspace": "pca" if "subspace" in prm else None,
                "nu": inner.get("df"), "kappa": inner.get("kappa"),
            }
            if method.startswith("PARAMETRIC"):
                dep = s0.get("dependent")
                row.update({
                    "norm_direction": None if dep is None else ("dependent" if dep else "independent"),
                    "p_spearman": s0.get("p_spearman"), "p_ridge": s0.get("p_ridge"), "p_dcor": s0.get("p_dcor"),
                    "spearman_rho": s0.get("spearman_rho"), "ridge_loo_r2": s0.get("ridge_loo_r2"),
                    "dcor": s0.get("dcor"),
                    # norm model actually used by the chosen family
                    "norm_model": ("log r | u (dependent)" if inner.get("norm", {}).get("dependent")
                                   else "log r (independent)") if "norm" in inner else None,
                })
            else:
                nd = prm.get("norm_dist")
                row.update({
                    "norm_direction": "not tested (independence assumed)",
                    "norm_model": f"{nd.get('type')} (independent)" if isinstance(nd, dict) else None,
                })
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-glob", default="standard_cov_parametric_*")
    ap.add_argument("--dist-dirs", nargs="*", default=["standard_cov_ledoit_wolf"])
    ap.add_argument("--out", default="results/chosen_distributions.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    dirs = sorted(d for d in glob.glob(args.param_glob) if os.path.isdir(d)) + \
        [d for d in args.dist_dirs if os.path.isdir(d)]
    rows = []
    for d in dirs:
        method = method_label(d)
        files = sorted(glob.glob(os.path.join(d, "*_dist_infos.pt")))
        print(f"{method}: {len(files)} caches", flush=True)
        for f in files:
            rows += rows_from_cache(f, method)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")

    # ---- 1. chosen family per method (weighted by sub-group size for clustering) ----
    print("\n=== Chosen family: number of classes per method x architecture x dataset ===")
    main_rows = df[df["subgroup"] == 0] if (df["n_subgroups"] == 1).all() else df
    counts = (main_rows.groupby(["method", "architecture", "dataset", "dist_type"]).size()
              .unstack("dist_type", fill_value=0))
    print(counts.to_string())
    print("\n=== Chosen family: totals per method (all architectures / datasets) ===")
    print(main_rows.groupby(["method", "dist_type"]).size().unstack("dist_type", fill_value=0).to_string())

    # ---- 2. pipeline=tests: decided by a DISTRIBUTIONAL test or by the CV ----
    t = df[df["pipeline"] == "tests"]
    if not t.empty:
        print("\n=== pipeline=tests: who decided (classes) ===")
        print(t.groupby(["method", "architecture", "dataset", "decided_by"]).size()
              .unstack("decided_by", fill_value=0).to_string())
        print("\n  accepted by:")
        print(t[t["decided_by"] == "test"].groupby(["method", "test_accepted_by"]).size().to_string())

    # ---- 3. norm / direction independence (PARAMETRIC step 0) ----
    p = df[df["method"].str.startswith("PARAMETRIC") & df["norm_direction"].notna()]
    if not p.empty:
        print("\n=== Norm / direction (step 0, PARAMETRIC): classes per architecture x dataset ===")
        ref = sorted(p["method"].unique())[0]
        tab = (p[p["method"] == ref].groupby(["architecture", "dataset", "norm_direction"]).size()
               .unstack("norm_direction", fill_value=0))
        tab["total"] = tab.sum(axis=1)
        print(f"  (from {ref})")
        print(tab.to_string())
        tot = p[p["method"] == ref]["norm_direction"].value_counts()
        print(f"\n  overall: {tot.to_dict()}  "
              f"({100 * tot.get('independent', 0) / tot.sum():.1f}% independent)")
        agree = p.pivot_table(index=["architecture", "dataset", "class"], columns="method",
                              values="norm_direction", aggfunc="first")
        n_disagree = int((agree.nunique(axis=1) > 1).sum())
        print(f"  agreement across the {agree.shape[1]} PARAMETRIC variants: "
              f"{len(agree) - n_disagree}/{len(agree)} classes identical")
        print("\n  median evidence per architecture x dataset (ridge LOO R2, dCor, Spearman rho):")
        print(p[p["method"] == ref].groupby(["architecture", "dataset"])[
            ["ridge_loo_r2", "dcor", "spearman_rho"]].median().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
