"""Collects the DISTRIBUTIONAL hyper-parameter mini study into a single CSV.

Study definition mirrors hparam_study_common.sh:
  - margin           : notop, 1000 synthetic samples, margin in {0, 5, 10, 20}
  - synthetic_samples: {100, 500, 1000, 2000, 5000, 10000} for notop (margin 10)
                       and corrected_confidence_penalty

Steps (like read_results.sh + src/analysis/analyze_aggregated.py + src/plots/extract_methods.py):
  1. src/collect/process_results_auto.py over results/ for the study versions only
     (incremental, --continue) -> results/hparam_study_raw.csv
  2. per (architecture, dataset, version, class) take the "best" epoch (else "00"),
     average over classes -> hparam_study_results.csv, with the study / method /
     parameter columns and the harmonic mean used in src/plots/extract_methods.py.

Usage: python3 src/collect/collect_hparam_study.py [--skip-process] [--output FILE]
"""
import argparse
import subprocess
import sys

import pandas as pd

BASE_TAG = "standard_cov_ledoit_wolf_syntetic_fixed_embs_forget"
PARAMETRIC_BASE_TAG = "standard_syntetic_fixed_embs_forget"
RAW_NAME = "hparam_study_raw"
RAW_PATH = f"results/{RAW_NAME}.csv"

METRICS = [
    "accuracy_score_after", "unlearning_accuracy_score",
    "maxprobability", "maxprobability_forget",
    "cohesion_forget_l2", "distance_inter_class_l2",
    "cohesion_forget_cos", "distance_inter_class_cos",
]


def study_versions(parametric_tag=None):
    """version base name -> (study, method, n_synth, margin).

    parametric_tag: e.g. "cv5x3_tbn_m0.05" -> the PARAMETRIC study
    (PARAMETRIC_<tag>_<n>_standard_syntetic_fixed_embs_forget_...) instead of DISTRIBUTIONAL."""
    v = {}
    prefix, base = ("DISTRIBUTIONAL", BASE_TAG) if parametric_tag is None else \
        (f"PARAMETRIC_{parametric_tag}", PARAMETRIC_BASE_TAG)

    def notop(n, m, study):
        v[f"{prefix}_{n}_{base}_notop_margin_{m}"] = (study, "notop", n, m)
        # DISTRIBUTIONAL: margin 10.0 was historically saved without the "_margin_X.X" suffix
        if m == 10.0 and parametric_tag is None:
            v[f"{prefix}_{n}_{base}_notop"] = (study, "notop", n, m)

    for m in (0.0, 5.0, 10.0, 20.0):
        notop(1000, m, "margin")
    for n in (100, 500, 2000, 5000, 10000):
        notop(n, 10.0, "synthetic_samples")
    for n in (100, 500, 1000, 2000, 5000, 10000):
        v[f"{prefix}_{n}_{base}_corrected_confidence_penalty"] = (
            "synthetic_samples", "corrected_confidence_penalty", n, None)
    return v


def harmonic(acc, unl):
    # unlearning score is "lower is better": invert before combining (as src/plots/extract_methods.py)
    u = 1.0 - unl
    return 0.0 if acc + u == 0 else 2 * acc * u / (acc + u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-process", action="store_true",
                    help="reuse results/hparam_study_raw.csv without rescanning results/")
    ap.add_argument("--output", default="hparam_study_results.csv")
    ap.add_argument("--parametric-tag", default=None,
                    help="collect the PARAMETRIC study of this tag (e.g. cv5x3_tbn_m0.05) "
                         "instead of DISTRIBUTIONAL; uses results/hparam_study_parametric_<tag>_raw.csv")
    args = ap.parse_args()

    versions = study_versions(args.parametric_tag)
    raw_name = RAW_NAME if args.parametric_tag is None else f"hparam_study_parametric_{args.parametric_tag}_raw"
    raw_path = f"results/{raw_name}.csv"

    if not args.skip_process:
        cmd = [sys.executable, "src/collect/process_results_auto.py", "--base-dir", "results",
               "--classes-filter", "single", "--results-file", raw_name, "--online", "--continue",
               "--include-version", *versions.keys()]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env={**__import__("os").environ, "PYTHONPATH": "."})

    # on_bad_lines: NFS-interrupted appends have left ragged rows before
    df = pd.read_csv(raw_path, dtype={"epoch": str}, on_bad_lines="skip")
    df["method_version"] = df["version"].str.split("/").str[0]
    df = df[df["method_version"].isin(versions)].copy()
    df["dataset"] = df["id_database"].str.extract(r"^([a-z0-9]+)_classes_to_remove")[0]
    df["class"] = df["id_database"].str.extract(r"classes_to_remove_(\d+)$")[0]
    df = df.dropna(subset=["dataset", "class"])

    # per class: epoch "best" if present, else "00"
    df["is_best"] = (df["epoch"] == "best").astype(int)
    df = (df.sort_values("is_best", ascending=False)
            .drop_duplicates(["experiment", "method_version", "dataset", "class"]))

    g = df.groupby(["experiment", "dataset", "method_version"])
    out = g[METRICS].mean()
    out.insert(0, "num_classes", g.size())
    out = out.reset_index()

    # legacy ("_notop") and new ("_notop_margin_10.0") names are the same config
    info = out["method_version"].map(versions)
    out["study"] = info.str[0]
    out["method"] = info.str[1]
    out["n_synthetic"] = info.str[2]
    out["margin"] = info.str[3]
    out["harmonic_mean"] = [harmonic(a, u) for a, u in
                            zip(out["accuracy_score_after"], out["unlearning_accuracy_score"])]
    out = out.rename(columns={"experiment": "architecture", "method_version": "folder"})
    cols = ["study", "method", "n_synthetic", "margin", "architecture", "dataset", "num_classes",
            "accuracy_score_after", "unlearning_accuracy_score", "harmonic_mean"] + \
           [m for m in METRICS if m not in ("accuracy_score_after", "unlearning_accuracy_score")] + ["folder"]
    out = out[cols].sort_values(["study", "method", "architecture", "dataset", "n_synthetic", "margin"])
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")

    # Quick coverage overview: expected 10 (cifar10/mnist) or 20 (cifar100) classes
    expected = {"cifar10": 10, "mnist": 10, "cifar100": 20}
    partial = out[out.apply(lambda r: r["num_classes"] != expected.get(r["dataset"]), axis=1)]
    if len(partial):
        print("\nIncomplete groups (num_classes != expected):")
        print(partial[["method", "n_synthetic", "margin", "architecture", "dataset", "num_classes"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
