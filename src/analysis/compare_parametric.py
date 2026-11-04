"""
Side-by-side summary of PARAMETRIC vs the matching DISTRIBUTIONAL runs
(1000 fixed synthetic samples, embs, notop margin 10 / corrected_confidence_penalty).

Reads the CSVs written by src/collect/process_results_auto.py and prints, per
architecture x dataset x loss, the mean over the single-class removals of the
main metrics at the chosen checkpoint.

    python src/analysis/compare_parametric.py \
        --parametric results/unlearning_results_single_parametric.csv \
        --reference unlearning_results_single_all_new.csv --epoch best
"""

import argparse

import pandas as pd

# loss -> reference DISTRIBUTIONAL run.  Every PARAMETRIC_<tag>_1000_..._<loss>
# variant found in the CSV (base, ks, pca, ks_pca) is compared against it.
REFERENCES = {
    "notop_margin_10.0":
        "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop_margin_10.0",
    "corrected_confidence_penalty":
        "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_corrected_confidence_penalty",
}
PARAM_SUFFIX = "_1000_standard_syntetic_fixed_embs_forget_"
METRICS = ["accuracy_score_after", "unlearning_accuracy_score", "maxprobability", "maxprobability_forget"]


def load(path):
    df = pd.read_csv(path, dtype={"epoch": str})
    df["method"] = df["version"].str.split("/").str[0]
    df["dataset"] = df["id_database"].str.split("_classes_to_remove_").str[0]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parametric", default="results/unlearning_results_single_parametric.csv")
    ap.add_argument("--reference", default="unlearning_results_single_all_new.csv")
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--out", default="results/parametric_vs_distributional.csv")
    args = ap.parse_args()

    df = pd.concat([load(args.parametric), load(args.reference)], ignore_index=True)
    df = df[df["epoch"] == args.epoch]

    rows = []
    for loss, dist_name in REFERENCES.items():
        param_names = sorted(m for m in df["method"].unique()
                             if m.startswith("PARAMETRIC_") and m.endswith(PARAM_SUFFIX + loss))
        candidates = [(m, m[:-len(PARAM_SUFFIX + loss)]) for m in param_names]
        for name, label in candidates + [(dist_name, "DISTRIBUTIONAL")]:
            sub = df[df["method"] == name]
            if sub.empty:
                print(f"⚠️  no rows for {name} at epoch {args.epoch}")
                continue
            g = sub.groupby(["experiment", "dataset"])[METRICS].agg(["mean", "count"])
            for (exp, ds), r in g.iterrows():
                row = {"experiment": exp, "dataset": ds, "loss": loss, "method": label,
                       "n_classes": int(r[(METRICS[0], "count")])}
                row.update({m: r[(m, "mean")] for m in METRICS})
                rows.append(row)

    out = pd.DataFrame(rows).sort_values(["loss", "experiment", "dataset", "method"])
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    for loss, sub in out.groupby("loss"):
        print(f"\n=== {loss} (epoch {args.epoch}) ===")
        print(sub.drop(columns="loss").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
