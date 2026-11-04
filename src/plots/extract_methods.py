import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_NOTOP, PROPOSED_CONF  # noqa: E402

SINGLE_SRC = "unlearning_results_single_all_new_analyzed_means.csv"
MULTIPLE_SRC = "unlearning_results_multiple_all_new_multiclass_analyzed_means.csv"

with open(SINGLE_SRC) as f:
    single_rows = list(csv.DictReader(f))

with open(MULTIPLE_SRC) as f:
    multiple_rows = list(csv.DictReader(f))

FIXED_METHODS = [
    # proposed method (src/utils/proposed_method.py; previously DISTRIBUTIONAL)
    PROPOSED_NOTOP,
    PROPOSED_CONF,
    "DELETE_embs",
    "BOUNDARYSHRINK_eps0.1_lr1e-05_embs",
    "BOUNDARYEXPAND_lr1e-05_embs",
    "BADTEACHER_T1.0_forgetOnly_embs",
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty",
    "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop",
    "RANDOMLABEL_noret_embs_forbidden",
    "RETRAIN",
    "original",
]

def base_version(v):
    return v.split('/')[0]

def harmonic(a, u):
    # unlearning_score is "lower is better", so invert it before combining
    # with acc_score ("higher is better") in the harmonic mean.
    a = float(a); u = 1.0 - float(u)
    if a + u == 0:
        return 0.0
    return 2 * a * u / (a + u)

def best_row(candidates):
    return max(candidates, key=lambda r: harmonic(r['accuracy_score_after_mean'], r['unlearning_accuracy_score_mean']))

def resolve_notop_fallback(name, available_bases):
    # Some method families only logged a "_notop_margin_X.X" variant instead
    # of the plain "_notop" one. If the exact name is missing, fall back to
    # the same family's margin variant, preferring margin_10.0 if present,
    # else the highest available margin.
    if name in available_bases or not name.endswith("_notop"):
        return name
    prefix = name + "_margin_"
    candidates = [b for b in available_bases if b.startswith(prefix)]
    if not candidates:
        return name
    exact10 = name + "_margin_10.0"
    if exact10 in candidates:
        return exact10
    def margin_val(b):
        m = re.match(re.escape(prefix) + r"([0-9.]+)", b)
        return float(m.group(1)) if m else -1.0
    return max(candidates, key=margin_val)

def pick_best_ssd(subset):
    candidates = [r for r in subset if base_version(r['version']).startswith("SSD_MAS_") and base_version(r['version']).endswith("_embs")]
    return base_version(best_row(candidates)['version']) if candidates else None

_EWC_EMBS_NOTOP_RE = re.compile(r"_MAS_(embsunfreeze|embs)_notop$")

def pick_best_ewc(subset):
    # EWC's unfrozen-embedding variant is named "embsunfreeze" in some files
    # and plain "embs" in others (e.g. ICU) - match both, but not "freeze"
    # (note "unfreeze" itself contains the substring "freeze").
    candidates = [r for r in subset if _EWC_EMBS_NOTOP_RE.search(base_version(r['version']))]
    return base_version(best_row(candidates)['version']) if candidates else None

def resolve_ewc_embs_naming(name, available_bases):
    # A forced EWC winner picked from one file (e.g. single, which only had
    # "embsunfreeze") may show up under the other naming ("embs" only, with
    # no "unfreeze") in a different file (e.g. multiple) - translate between
    # the two so the exact same run is still matched.
    if name in available_bases:
        return name
    if "embsunfreeze" in name:
        alt = name.replace("embsunfreeze", "embs")
    elif "_embs_" in name:
        alt = name.replace("_embs_", "_embsunfreeze_")
    else:
        return name
    return alt if alt in available_bases else name

def process_group(subset, ctx, forced_ssd=None, forced_ewc=None):
    out = []
    available_bases = {base_version(r['version']) for r in subset}
    for m in FIXED_METHODS:
        resolved = resolve_notop_fallback(m, available_bases)
        r = next((r for r in subset if base_version(r['version']) == resolved), None)
        if r is None:
            out.append(ctx + (m, "MISSING", "", "", ""))
        else:
            a = r['accuracy_score_after_mean']; u = r['unlearning_accuracy_score_mean']
            label = resolved if resolved == m else f"{m} -> {resolved}"
            selection = "fixed" if resolved == m else "fixed_margin_fallback"
            out.append(ctx + (label, selection, a, u, f"{harmonic(a, u):.4f}"))

    # SSD_MAS_embs and EWC_embsunfreeze: normally the best-by-harmonic-mean
    # within this subset, but when forced_ssd/forced_ewc are given (multiple
    # groups), reuse the winner already picked from the single-class results
    # instead, so the same hyperparameters are used for comparability.
    ssd_name = forced_ssd if forced_ssd is not None else pick_best_ssd(subset)
    if ssd_name:
        r = next((r for r in subset if base_version(r['version']) == ssd_name), None)
        selection = "best_SSD_MAS_embs" if forced_ssd is None else "best_SSD_MAS_embs (fixed from single)"
        if r is None:
            out.append(ctx + (ssd_name, "MISSING", "", "", ""))
        else:
            a = r['accuracy_score_after_mean']; u = r['unlearning_accuracy_score_mean']
            out.append(ctx + (ssd_name, selection, a, u, f"{harmonic(a, u):.4f}"))
    else:
        out.append(ctx + ("SSD_MAS_*_embs", "MISSING", "", "", ""))

    ewc_name = forced_ewc if forced_ewc is not None else pick_best_ewc(subset)
    if ewc_name:
        resolved_ewc = resolve_ewc_embs_naming(ewc_name, available_bases)
        r = next((r for r in subset if base_version(r['version']) == resolved_ewc), None)
        label = resolved_ewc if resolved_ewc == ewc_name else f"{ewc_name} -> {resolved_ewc}"
        selection = "best_EWC_embsunfreeze" if forced_ewc is None else "best_EWC_embsunfreeze (fixed from single)"
        if r is None:
            out.append(ctx + (label, "MISSING", "", "", ""))
        else:
            a = r['accuracy_score_after_mean']; u = r['unlearning_accuracy_score_mean']
            out.append(ctx + (label, selection, a, u, f"{harmonic(a, u):.4f}"))
    else:
        out.append(ctx + ("EWC_*_embsunfreeze", "MISSING", "", "", ""))

    return out

out_rows = []
single_winners = {}

single_groups = sorted(set((r['experiment'], r['dataset']) for r in single_rows))
for arch, ds in single_groups:
    subset = [r for r in single_rows if r['experiment'] == arch and r['dataset'] == ds]
    ssd_name = pick_best_ssd(subset)
    ewc_name = pick_best_ewc(subset)
    single_winners[(arch, ds)] = (ssd_name, ewc_name)
    ctx = ("single", arch, ds, "")
    out_rows.extend(process_group(subset, ctx, forced_ssd=ssd_name, forced_ewc=ewc_name))

multiple_groups = sorted(set((r['experiment'], r['dataset'], r['num_classes_removed']) for r in multiple_rows), key=lambda t: (t[0], t[1], int(t[2])))
for arch, ds, ncr in multiple_groups:
    subset = [r for r in multiple_rows if r['experiment'] == arch and r['dataset'] == ds and r['num_classes_removed'] == ncr]
    ssd_name, ewc_name = single_winners.get((arch, ds), (None, None))
    ctx = ("multiple", arch, ds, ncr)
    out_rows.extend(process_group(subset, ctx, forced_ssd=ssd_name, forced_ewc=ewc_name))

with open("selected_methods.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "architecture", "dataset", "num_classes_removed", "method", "selection", "accuracy_score_after_mean", "unlearning_accuracy_score_mean", "harmonic_mean"])
    w.writerows(out_rows)

print("done, rows:", len(out_rows))
