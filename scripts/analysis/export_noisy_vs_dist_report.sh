#!/bin/bash
# Build the data package for the NOISY vs DISTRIBUTIONAL write-up: all three
# scenarios (single / multiple / ICU), both method pairs, compared per
# (architecture, dataset) cell with Holm correction inside each scenario.
#
# Inputs  : results/wilcox_noisy_vs_dist/{tests_all_scenarios.csv,pairs_long.csv}
#           produced by src/analysis/analyze_noisy_vs_distributional.py
# Outputs : results/wilcox_noisy_vs_dist/report/
#             grid_holm.csv       the main table: one row per (scenario, comparison,
#                                 cell, metric), with Holm-corrected p-values
#             summary_counts.csv  how many cells come out significant, and for whom
#             icu_runs_auc.csv    the 40 paired ICU run AUCs (per-run figure)
#             icu_curves_steps.csv median + IQR per ICU step (curve figure)
#
# Pure gawk/sort, no conda environment needed:  bash scripts/analysis/export_noisy_vs_dist_report.sh
#
# Conventions used in every output:
#   diff            = DISTRIBUTIONAL - NOISY
#   dist_advantage  = diff * polarity, polarity = -1 for forget (lower is better),
#                     +1 for retain / harmonic. dist_advantage > 0 ALWAYS means
#                     DISTRIBUTIONAL is the better method. This is the column to plot.
#   counted         = no  for the four (resnet18, efficientnet_v2_s) x
#                     (cifar10, mnist) cells, excluded a priori as expected-equal.
#                     They carry p_raw for reference but no Holm p-value, because
#                     they are not part of any correction family.

set -eu

IN_DIR="results/wilcox_noisy_vs_dist"
OUT_DIR="$IN_DIR/report"
mkdir -p "$OUT_DIR"

TESTS="$IN_DIR/tests_all_scenarios.csv"
PAIRS="$IN_DIR/pairs_long.csv"
for f in "$TESTS" "$PAIRS"; do
    [ -r "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done

# ------------------------------------------------------------------ grid_holm
# level == dataset_x_architecture is the comparison granularity; p_holm_level is
# already Holm over exactly the right family (one scenario x comparison x unit,
# all counted cells x 3 metrics), because src/analysis/analyze_noisy_vs_distributional.py
# keys its correction families on (scenario, analysis_set, unit, comparison, level)
# and the excluded cells live in their own analysis_set.
gawk -F, -v OFS=, '
function stars(p, ok) {
    if (!ok) return "untested"
    if (p < 0.001) return "***"
    if (p < 0.01)  return "**"
    if (p < 0.05)  return "*"
    return "ns"
}
NR == 1 { next }
$5 == "dataset_x_architecture" {
    scenario = $1; counted = ($2 == "main") ? "yes" : "no"
    unit = $3; comparison = $4; dataset = $6; family = $7; arch = $8
    m = $11
    polarity = (m == "forget") ? -1 : 1
    ran = ($26 == "ok" || $26 == "identical")

    median_diff = $19 + 0; rb = $23 + 0
    n_dist_better  = (polarity > 0) ? $16 + 0 : $17 + 0
    n_noisy_better = (polarity > 0) ? $17 + 0 : $16 + 0
    n = $14 + 0
    unit_role = (unit == "step") ? "sensitivity" : "primary"

    # Holm only means something for the counted grid; blank it out otherwise.
    p_raw   = ran ? sprintf("%.6g", $25 + 0) : ""
    p_holm  = (ran && counted == "yes") ? sprintf("%.6g", $28 + 0) : ""
    sig     = (p_holm != "" && $28 + 0 < 0.05) ? "yes" : "no"
    st      = (counted == "yes") ? stars($28 + 0, ran) : "not_counted"

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%d,%d,%.4f,%s,%s,%s,%s,%s,%s\n", \
        scenario, comparison, unit, unit_role, dataset, arch, family, counted, \
        m, (m == "forget" ? "lower" : "higher"), n, \
        $12 + 0, $13 + 0, median_diff, polarity * median_diff, \
        rb, polarity * rb, n_dist_better, n_noisy_better, $18 + 0, \
        (n > 0 ? n_dist_better / n : 0), \
        p_raw, p_holm, sig, st, (sig == "yes" ? $27 : ""), $26
}
' "$TESTS" | {
    echo "scenario,comparison,unit,unit_role,dataset,experiment,architecture_family,counted,metric,better_when,n_pairs,median_noisy,median_dist,median_diff,dist_advantage,rank_biserial,rb_dist_advantage,n_runs_dist_better,n_runs_noisy_better,n_ties,win_rate_dist,p_raw,p_holm,sig_holm,stars,favours_if_sig,status"
    sort -t, -k2,2 -k1,1r -k3,3 -k5,5 -k6,6 -k9,9
} > "$OUT_DIR/grid_holm.csv"

# ------------------------------------------------------------- summary_counts
# Per scenario x comparison x unit: size of the Holm family and the verdict tally.
# metric == "ALL" is the whole family; the per-metric rows break it down.
gawk -F, -v OFS=, '
NR == 1 { next }
$8 != "yes" { next }                      # counted cells only
{
    key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
    for (i = 1; i <= 2; i++) {
        k = key SUBSEP (i == 1 ? "ALL" : $9)
        seen[k] = 1
        n_tests[k]++
        if ($27 == "ok" || $27 == "identical") n_run[k]++
        if ($24 == "yes") {
            n_sig[k]++
            if ($26 == "DISTRIBUTIONAL") n_dist[k]++; else n_noisy[k]++
        }
    }
}
END {
    print "scenario,comparison,unit,unit_role,metric,n_tests,n_tests_run," \
          "n_significant,n_sig_favour_dist,n_sig_favour_noisy,n_not_significant,verdict"
    for (k in seen) {
        split(k, p, SUBSEP)
        sig = n_sig[k] + 0; d = n_dist[k] + 0; nz = n_noisy[k] + 0
        verdict = (sig == 0) ? "no_difference" \
                : (d > 0 && nz == 0) ? "favours_DISTRIBUTIONAL" \
                : (nz > 0 && d == 0) ? "favours_NOISY" : "mixed"
        printf "%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%s\n", \
            p[1], p[2], p[3], p[4], p[5], n_tests[k], n_run[k] + 0, \
            sig, d, nz, n_tests[k] - sig, verdict
    }
}
' "$OUT_DIR/grid_holm.csv" | { IFS= read -r h; echo "$h"; sort -t, -k2,2 -k1,1r -k3,3 -k5,5; } \
    > "$OUT_DIR/summary_counts.csv"

# ------------------------------------------------------------- icu_runs_auc
# The 40 paired ICU run AUCs for confidence_penalty (4 architectures x 10
# class-sets). run_id is a stable short label for the class-set, shared across
# architectures: all four saw the same 10 sets.
gawk -F, -v OFS=, '
NR == 1 { next }
$1 == "icu" && $2 == "confidence_penalty" && $3 == "run_auc" && $4 == "main" {
    seen[$8] = 1; rows[++n] = $0
}
END {
    k = asorti(seen, sorted)
    for (i = 1; i <= k; i++) run_id[sorted[i]] = sprintf("R%02d", i)
    print "experiment,architecture_family,dataset,run_id,class_set,n_steps," \
          "metric,better_when,auc_noisy,auc_dist,diff,dist_advantage,dist_is_better"
    for (i = 1; i <= n; i++) {
        split(rows[i], f, ",")
        m = (f[12] == "unlearning_accuracy_score") ? "forget" \
          : (f[12] == "accuracy_score_after") ? "retain" : "harmonic"
        polarity = (m == "forget") ? -1 : 1
        adv = polarity * (f[15] + 0)
        printf "%s,%s,%s,%s,%s,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%s\n", \
            f[5], f[6], f[7], run_id[f[8]], f[8], f[10] + 0, \
            m, (m == "forget" ? "lower" : "higher"), \
            f[13] + 0, f[14] + 0, f[15] + 0, adv, (adv > 0 ? "yes" : "no")
    }
}
' "$PAIRS" | { IFS= read -r h; echo "$h"; sort -t, -k7,7 -k1,1 -k4,4; } > "$OUT_DIR/icu_runs_auc.csv"

# ---------------------------------------------------------- icu_curves_steps
# Median and inter-quartile band across runs at every ICU step, per method.
# These are the curves whose area is the AUC tested in grid_holm.csv, so the
# figure shows where the two methods actually separate. experiment == "ALL"
# pools the 40 runs.
gawk -F, -v OFS=, '
function pct(a, n, p,   idx, lo, hi, frac) {
    if (n == 1) return a[1]
    idx = p * (n - 1) + 1
    lo = int(idx); hi = lo + 1; frac = idx - lo
    if (hi > n) return a[n]
    return a[lo] * (1 - frac) + a[hi] * frac
}
function flush_cell(key,   nn, nd, parts, med_n, med_d, polarity) {
    nn = asort(vn[key]); nd = asort(vd[key]); asort(vf[key])
    split(key, parts, SUBSEP)
    polarity = (parts[2] == "forget") ? -1 : 1
    med_n = pct(vn[key], nn, 0.5); med_d = pct(vd[key], nd, 0.5)
    printf "%s,%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", \
        parts[1], fam[parts[1]], parts[2], (parts[2] == "forget" ? "lower" : "higher"), \
        parts[3], nn, \
        med_n, pct(vn[key], nn, 0.25), pct(vn[key], nn, 0.75), \
        med_d, pct(vd[key], nd, 0.25), pct(vd[key], nd, 0.75), \
        med_d - med_n, polarity * (med_d - med_n), pct(vf[key], nn, 0.5)
}
NR == 1 { next }
$1 == "icu" && $2 == "confidence_penalty" && $3 == "step" && $4 == "main" {
    m = ($12 == "unlearning_accuracy_score") ? "forget" \
      : ($12 == "accuracy_score_after") ? "retain" : "harmonic"
    fam[$5] = $6; fam["ALL"] = "all"
    for (who = 1; who <= 2; who++) {
        key = (who == 1 ? $5 : "ALL") SUBSEP m SUBSEP ($9 + 0)
        keys[key] = 1
        i = ++cnt[key]
        vn[key][i] = $13 + 0; vd[key][i] = $14 + 0; vf[key][i] = $15 + 0
    }
}
END {
    print "experiment,architecture_family,metric,better_when,step,n_runs," \
          "noisy_median,noisy_q1,noisy_q3,dist_median,dist_q1,dist_q3," \
          "median_diff,median_dist_advantage,median_paired_diff"
    for (key in keys) flush_cell(key)
}
' "$PAIRS" | { IFS= read -r h; echo "$h"; sort -t, -k3,3 -k1,1 -k5,5n; } > "$OUT_DIR/icu_curves_steps.csv"

echo "Wrote:"
wc -l "$OUT_DIR"/grid_holm.csv "$OUT_DIR"/summary_counts.csv \
      "$OUT_DIR"/icu_runs_auc.csv "$OUT_DIR"/icu_curves_steps.csv
echo
echo "=== summary_counts.csv (metric == ALL) ==="
gawk -F, 'NR==1 || $5=="ALL"' "$OUT_DIR/summary_counts.csv" | column -s, -t
