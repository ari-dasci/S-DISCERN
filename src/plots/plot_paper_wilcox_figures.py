#!/usr/bin/env python3
"""Compact paper figures for the latent-space Wilcoxon analysis (single-class
forgetting only). Replaces wilcoxon_trend_heatmap_{original,retrain} and
reference_method_heatmap_distributional_{notop,confidence}_{original,retrain}.

  results/wilcoxon_trend_compact.{pdf,png}
      Trend of the 7 baseline methods (Wilcoxon one-sample vs 0, two one-sided
      tests, alpha=0.05), panels (a) vs original / (b) vs retrain.
      Rows: Forget/Retain/Unknown x [Cohesion L2, Cohesion cos,
      Separation L2, Separation cos]. Numbers = median of the IQR-cleaned
      increments (all rows, or only Forget with FIG1_VALUES_ALL_ROWS=False).

  results/distributional_forget_compact.{pdf,png}
      DISTRIBUTIONAL (notop / confidence) on the Forget block: colour = sign of
      its own mean increment, number = mean_delta, black border = disagrees with
      the majority direction of the 7 baselines. Every cell must have data
      (the script aborts and lists the missing cells otherwise).

Only reads results already computed by analyze_wilcoxon_unlearning*.py
(results/wilcox/*.csv); no test is re-run. Run from the repo root:
    python src/plots/plot_paper_wilcox_figures.py
"""

from __future__ import annotations

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_NOTOP_LABEL, PROPOSED_CONF_LABEL  # noqa: E402
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.colors import ListedColormap, BoundaryNorm
from method_colors import arch_name

# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------
TREND_FILES = {
    ("forget",  "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra.csv",
    ("forget",  "inter"): "results/wilcox/wilcoxon_unpaired_inter.csv",
    ("retain",  "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_retain.csv",
    ("retain",  "inter"): "results/wilcox/wilcoxon_unpaired_inter_retain.csv",
    ("unknown", "intra"): "results/wilcox/wilcoxon_unpaired_summary_intra_unknown3.csv",
    ("unknown", "inter"): "results/wilcox/wilcoxon_unpaired_inter_unknown3.csv",
}
REFERENCE_FILE = "results/wilcox/reference_method_trends.csv"   # forget block
OUT_FIG1 = "results/wilcoxon_trend_compact"
OUT_FIG2 = "results/distributional_forget_compact"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
ARCHS = ["resnet18", "efficientnet_v2_s", "swin_t", "vit_b_16"]   # conv | transformers
DATASETS = ["cifar10", "cifar100", "mnist"]
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "mnist": "MNIST"}
COL_KEYS = [(a, d) for a in ARCHS for d in DATASETS]

CASES = ["forget", "retain", "unknown"]
CASE_LABELS = {"forget": "Forget", "retain": "Retain", "unknown": "Unknown"}
METRIC_KEYS = [("intra", "l2"), ("intra", "cos"), ("inter", "l2"), ("inter", "cos")]
METRIC_LABELS = {
    ("intra", "l2"): "Cohesion L2",
    ("intra", "cos"): "Cohesion cos",
    ("inter", "l2"): "Separation L2",
    ("inter", "cos"): "Separation cos",
}
BASELINES = ["original", "retrain"]
REFERENCES = [(PROPOSED_NOTOP_LABEL, "notop"),          # proposed method (src/utils/proposed_method.py)
              (PROPOSED_CONF_LABEL, "confidence")]

# Categorical codes -> colours (same palette in both figures)
CONC, NONE, DISP, NA = 0, 1, 2, 3
# Ordinal in lightness (also in grayscale), all light enough for black text:
# concentrates mid blue, no trend light grey, disperses very light orange.
C_CONC, C_DISP, C_NONE, C_NA = "#7FA3CC", "#FCEAD5", "#CCCCCC", "#FFFFFF"
CMAP = ListedColormap([C_CONC, C_NONE, C_DISP, C_NA])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], CMAP.N)

FIG_W = 6.3
LEFT_FIG1, LEFT_FIG2, RIGHT = 0.88, 0.97, 0.03   # inches
HEADER_H, LEGEND_H = 0.34, 0.26                   # inches
CELL_FS, LABEL_FS, DS_FS = 6.5, 8, 6.5
FIG1_VALUES_ALL_ROWS = True   # False: numbers only on the Forget block (colour only elsewhere)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_trends() -> pd.DataFrame:
    """One row per (case, metric_type, metric, baseline, experiment, dataset)
    with the trend code and the median cleaned increment (same logic as
    plot_wilcoxon_trends.py)."""
    frames = []
    for (case, mtype), path in TREND_FILES.items():
        d = pd.read_csv(path)
        d["case"], d["metric_type"] = case, mtype
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["metric"] = df["group_name"].str.split("_vs_").str[0]
    df["baseline"] = df["group_name"].str.split("_vs_").str[1]

    key = ["case", "metric_type", "metric", "baseline", "experiment", "dataset"]
    is_true = lambda s: s.astype(str).str.lower().eq("true")
    gr = df[df["alternative"] == "greater"].assign(rej_g=lambda x: is_true(x["reject_h0"]))
    ls = df[df["alternative"] == "less"].assign(rej_l=lambda x: is_true(x["reject_h0"]))
    t = (gr.set_index(key)[["rej_g", "median_clean"]]
         .join(ls.set_index(key)[["rej_l"]], how="outer").reset_index())
    t[["rej_g", "rej_l"]] = t[["rej_g", "rej_l"]].fillna(False).astype(bool)

    def code(r):
        if r.rej_l and not r.rej_g:
            return CONC
        if r.rej_g and not r.rej_l:
            return DISP
        if r.rej_g and r.rej_l:   # never happens in practice; follow the median
            return DISP if r.median_clean >= 0 else CONC
        return NONE

    t["code"] = t.apply(code, axis=1)
    return t.rename(columns={"median_clean": "value"})


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_FILE)
    df = df[df["reference"].isin([r for r, _ in REFERENCES])].copy()
    df["metric_type"] = np.where(df["metric"].str.contains("cohesion"), "intra", "inter")
    df["metric_kind"] = np.where(df["metric"].str.endswith("_cos"), "cos", "l2")
    df["baseline"] = df["group_name"].str.split("_vs_").str[1]
    df["code"] = df["direction"].map({"concentrate": CONC, "disperse": DISP}).fillna(NA).astype(int)
    df["disagree"] = (df["code"] != NA) & ~df["agrees_with_majority"].astype(str).str.lower().eq("true")
    return df


def require_complete(codes, what: str, row_names: list[str]) -> None:
    """Abort instead of drawing an empty cell (e.g. a reference whose runs
    are missing, or 'absent' in reference_method_trends.csv)."""
    missing = [f"{row_names[ri]} {COL_KEYS[ci][0]}/{COL_KEYS[ci][1]}"
               for ri, ci in zip(*np.nonzero(codes == NA))]
    if missing:
        raise SystemExit(f"Missing cells in {what}:\n  " + "\n  ".join(missing))


def trend_matrix(t: pd.DataFrame, baseline: str):
    n_r, n_c = len(CASES) * len(METRIC_KEYS), len(COL_KEYS)
    codes = np.full((n_r, n_c), NA)
    vals = np.full((n_r, n_c), np.nan)
    sub = t[t["baseline"] == baseline].set_index(
        ["case", "metric_type", "metric", "experiment", "dataset"])
    for ri, (case, (mt, mk)) in enumerate((c, m) for c in CASES for m in METRIC_KEYS):
        for ci, (exp, ds) in enumerate(COL_KEYS):
            k = (case, mt, mk, exp, ds)
            if k in sub.index:
                r = sub.loc[k]
                codes[ri, ci], vals[ri, ci] = int(r["code"]), float(r["value"])
    require_complete(codes, f"trend vs {baseline}",
                     [f"{c}/{mt}-{mk}" for c in CASES for mt, mk in METRIC_KEYS])
    return codes, vals


def reference_matrix(df: pd.DataFrame, reference: str, baseline: str):
    n_r, n_c = len(METRIC_KEYS), len(COL_KEYS)
    codes = np.full((n_r, n_c), NA)
    vals = np.full((n_r, n_c), np.nan)
    disagree = np.zeros((n_r, n_c), dtype=bool)
    sub = df[(df["reference"] == reference) & (df["baseline"] == baseline)].set_index(
        ["metric_type", "metric_kind", "experiment", "dataset"])
    for ri, (mt, mk) in enumerate(METRIC_KEYS):
        for ci, (exp, ds) in enumerate(COL_KEYS):
            k = (mt, mk, exp, ds)
            if k in sub.index and sub.loc[k, "code"] != NA:
                r = sub.loc[k]
                codes[ri, ci], vals[ri, ci] = int(r["code"]), float(r["mean_delta"])
                disagree[ri, ci] = bool(r["disagree"])
    require_complete(codes, f"{reference} vs {baseline}",
                     [f"forget/{mt}-{mk}" for mt, mk in METRIC_KEYS])
    return codes, vals, disagree


# ---------------------------------------------------------------------------
# Mandatory checks (abort before plotting if anything is off)
# ---------------------------------------------------------------------------
def run_checks(t: pd.DataFrame, ref: pd.DataFrame) -> None:
    rows, ok = [], True

    def add(name, expected, obtained, good, strict=True):
        # strict=False: expected value predates the reference-IQR fix in
        # compute_reference_method_trends, so a change is reported, not fatal.
        nonlocal ok
        ok &= good or not strict
        status = "OK" if good else ("MISMATCH" if strict else "CHANGED (warn)")
        rows.append((name, expected, obtained, status))

    # Expected values updated on 2026-10-21 after the "embs" relaunch (classifier +
    # last stage): swin_t changed because its LF-SSD baseline was relaunched (it only
    # touched the classifier before); vit_b_16 / resnet18 are unchanged.
    # Before: swin_t [-1.065, 0.000, -2.273].
    exp_medians = {
        "swin_t": [-5.001, -3.090, -2.585],
        "vit_b_16": [-0.128, -0.021, -0.438],
        "resnet18": [2.725, 2.239, 3.232],
    }
    ti = t.set_index(["case", "metric_type", "metric", "baseline", "experiment", "dataset"])
    for arch, vals in exp_medians.items():
        for ds, e in zip(DATASETS, vals):
            v = float(ti.loc[("forget", "intra", "l2", "original", arch, ds), "value"])
            add(f"Fig1 forget cohesion L2 vs orig {arch}/{ds}", f"{e:+.3f}", f"{v:+.3f}",
                abs(round(v, 3) - e) < 5e-4)

    def counts(mt, mk, base):
        s = t[(t["case"] == "forget") & (t["metric_type"] == mt) & (t["metric"] == mk)
              & (t["baseline"] == base)]
        return f"{(s.code == CONC).sum()}/{(s.code == DISP).sum()}/{(s.code == NONE).sum()}"

    # 2026-10-21 (embs relaunch of efficientnet / swin SSD): cos vs retrain 0/11/1 -> 0/10/2,
    # cohesion L2 vs retrain 8/2/2 -> 9/2/1.
    for mt, mk, base, e in [("inter", "l2", "original", "11/0/1"),
                            ("inter", "cos", "retrain", "0/10/2"),
                            ("intra", "l2", "retrain", "9/2/1")]:
        c = counts(mt, mk, base)
        add(f"Fig1 forget {METRIC_LABELS[(mt, mk)]} vs {base} (conc/disp/none)", e, c, c == e)

    ri = ref.set_index(["reference", "metric_type", "metric_kind", "baseline", "experiment", "dataset"])
    # Expected values of DISCERN (PARAMETRIC tests), 2026-10-21; the previous ones were
    # DISTRIBUTIONAL's (-8.276, -37.798, +2.214).
    for arch, ds, e in [("efficientnet_v2_s", "cifar10", -8.473),
                        ("swin_t", "cifar100", -39.009),
                        ("resnet18", "cifar100", 2.213)]:
        v = float(ri.loc[(PROPOSED_CONF_LABEL, "intra", "l2", "original", arch, ds), "mean_delta"])
        add(f"Fig2 conf. forget cohesion L2 vs orig {arch}/{ds}", f"{e:+.3f}", f"{v:+.3f}",
            abs(round(v, 3) - e) < 5e-4)

    w = max(len(r[0]) for r in rows)
    print(f"{'check':<{w}}  {'expected':>9}  {'obtained':>9}  status")
    for name, e, o, s in rows:
        print(f"{name:<{w}}  {e:>9}  {o:>9}  {s}")
    if not ok:
        raise SystemExit("\nChecks failed -- not plotting.")
    print("\nAll checks passed.\n")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
DECIMALS = 3   # same precision for L2 and cos


def fmt(v: float, kind: str) -> str:
    d = DECIMALS
    s = f"{v:.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"
    return ("+" if v > 0 else "") + s.replace("-", "−")


def add_axes_in(fig, left, top, width, height):
    """Axes placed in inches, measured from the figure's top-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, 1 - (top + height) / H, width / W, height / H])


def offset(ax, dx_pt=0.0, dy_pt=0.0):
    return ax.transData + mtransforms.ScaledTranslation(dx_pt / 72, dy_pt / 72,
                                                        ax.figure.dpi_scale_trans)


def draw_grid(ax, codes, vals=None, kinds=None, show_rows=None, disagree=None):
    n_r, n_c = codes.shape
    ax.pcolormesh(np.arange(n_c + 1), np.arange(n_r + 1), codes, cmap=CMAP, norm=NORM,
                  edgecolors="face", linewidth=0, rasterized=False)
    ax.set_xlim(0, n_c)
    ax.set_ylim(n_r, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for ri in range(n_r):
        for ci in range(n_c):
            if vals is not None and (show_rows is None or ri in show_rows):
                ax.text(ci + 0.5, ri + 0.5, fmt(vals[ri, ci], kinds[ri]), ha="center",
                        va="center_baseline", fontsize=CELL_FS, color="black", zorder=3)
            if disagree is not None and disagree[ri, ci]:
                ax.add_patch(mpatches.Rectangle((ci + 0.05, ri + 0.09), 0.90, 0.82,
                                                fill=False, edgecolor="black",
                                                linewidth=0.9, zorder=4))

    # Vertical separators between architectures (thin, no extra weight)
    for x in (3, 6, 9):
        ax.axvline(x, color="#555555", linewidth=0.5, zorder=5)
    ax.add_patch(mpatches.Rectangle((0, 0), n_c, n_r, fill=False, edgecolor="#333333",
                                    linewidth=0.6, zorder=6, clip_on=False))


def draw_header(ax):
    for gi, arch in enumerate(ARCHS):
        x0 = 3 * gi
        ax.text(x0 + 1.5, 0, arch_name(arch), transform=offset(ax, dy_pt=13),
                ha="center", va="bottom", fontsize=LABEL_FS, clip_on=False)
        ax.plot([x0 + 0.12, x0 + 2.88], [0, 0], transform=offset(ax, dy_pt=11.5),
                color="black", linewidth=0.5, clip_on=False)
        for di, ds in enumerate(DATASETS):
            ax.text(x0 + di + 0.5, 0, DATASET_LABELS[ds], transform=offset(ax, dy_pt=2.5),
                    ha="center", va="bottom", fontsize=DS_FS, clip_on=False)


def metric_labels(ax, keys):
    for ri, mk in enumerate(keys):
        ax.text(0, ri + 0.5, METRIC_LABELS[mk], transform=offset(ax, dx_pt=-3),
                ha="right", va="center", fontsize=LABEL_FS, clip_on=False)


def legend(fig, handles, left):
    W, _ = fig.get_size_inches()
    xc = (left + (W - left - RIGHT) / 2) / W
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(xc, 0.0),
               ncol=len(handles), frameon=False, fontsize=LABEL_FS, handlelength=1.6,
               handleheight=0.9, columnspacing=1.4, borderaxespad=0.15)


def patch(color, label, **kw):
    kw.setdefault("edgecolor", "#666666")
    kw.setdefault("linewidth", 0.4)
    return mpatches.Patch(facecolor=color, label=label, **kw)


def save(fig, stem):
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)
    print(f"Saved: {stem}.pdf / .png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def figure1(t: pd.DataFrame) -> None:
    H, gap = 4.45, 0.18
    n_r = len(CASES) * len(METRIC_KEYS)
    row_h = (H - HEADER_H - gap - LEGEND_H - 0.02) / (2 * n_r)
    width = FIG_W - LEFT_FIG1 - RIGHT
    fig = plt.figure(figsize=(FIG_W, H))

    kinds = [mk for _ in CASES for _, mk in METRIC_KEYS]
    # None = numbers on every row; set(range(len(METRIC_KEYS))) = Forget rows only
    forget_rows = None if FIG1_VALUES_ALL_ROWS else set(range(len(METRIC_KEYS)))
    for pi, base in enumerate(BASELINES):
        top = HEADER_H + pi * (n_r * row_h + gap)
        ax = add_axes_in(fig, LEFT_FIG1, top, width, n_r * row_h)
        codes, vals = trend_matrix(t, base)
        draw_grid(ax, codes, vals, kinds, show_rows=forget_rows)

        for i in range(1, len(CASES)):
            ax.axhline(i * len(METRIC_KEYS), color="#333333", linewidth=0.6, zorder=5)
        metric_labels(ax, [m for _ in CASES for m in METRIC_KEYS])
        for ci, case in enumerate(CASES):
            ax.text(0, (ci + 0.5) * len(METRIC_KEYS), CASE_LABELS[case],
                    transform=offset(ax, dx_pt=-55), rotation=90, ha="center",
                    va="center", fontsize=LABEL_FS, clip_on=False)
        ax.text(0, 0, f"({'ab'[pi]}) vs. {base}", transform=offset(ax, dx_pt=-61, dy_pt=3),
                ha="left", va="bottom", fontsize=LABEL_FS, clip_on=False)
        if pi == 0:
            draw_header(ax)

    legend(fig, [patch(C_CONC, r"Concentrates ($\Delta<0$)"),
                 patch(C_DISP, r"Disperses ($\Delta>0$)"),
                 patch(C_NONE, "No significant trend")], LEFT_FIG1)
    save(fig, OUT_FIG1)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def figure2(ref: pd.DataFrame) -> None:
    H, gap = 3.45, 0.07
    panels = [(r, short, b) for r, short in REFERENCES for b in BASELINES]
    n_r = len(METRIC_KEYS)
    row_h = (H - HEADER_H - (len(panels) - 1) * gap - LEGEND_H - 0.02) / (len(panels) * n_r)
    width = FIG_W - LEFT_FIG2 - RIGHT
    fig = plt.figure(figsize=(FIG_W, H))
    kinds = [mk for _, mk in METRIC_KEYS]

    for pi, (reference, short, base) in enumerate(panels):
        top = HEADER_H + pi * (n_r * row_h + gap)
        ax = add_axes_in(fig, LEFT_FIG2, top, width, n_r * row_h)
        codes, vals, disagree = reference_matrix(ref, reference, base)
        draw_grid(ax, codes, vals, kinds, disagree=disagree)
        metric_labels(ax, METRIC_KEYS)
        ax.text(0, n_r / 2, f"({'abcd'[pi]}) {short}\nvs. {base}",
                transform=offset(ax, dx_pt=-60), rotation=90, ha="center", va="center",
                fontsize=CELL_FS, linespacing=1.15, clip_on=False)
        if pi == 0:
            draw_header(ax)

        ties = ref[(ref["reference"] == reference) & (ref["baseline"] == base)
                   & (ref["majority_direction"] == "tie")]
        for _, r in ties.iterrows():
            print(f"[note] {reference} vs {base}: {r['experiment']}/{r['dataset']} "
                  f"{r['metric']} has a tied baseline majority (drawn as disagreement).")

    legend(fig, [patch(C_CONC, r"Concentrates ($\bar\Delta<0$)"),
                 patch(C_DISP, r"Disperses ($\bar\Delta>0$)"),
                 patch("white", "Disagrees with baseline majority",
                       edgecolor="black", linewidth=0.9)],
           LEFT_FIG2)
    save(fig, OUT_FIG2)


if __name__ == "__main__":
    trends = load_trends()
    reference = load_reference()
    run_checks(trends, reference)
    figure1(trends)
    figure2(reference)
