"""
Single source of truth for per-method styles (color, hatch, marker), keyed by
the short display name used in the legends (src/plots/plot_boxplots.py,
src/plots/plot_figures.py and their ablation plots), so each method looks the
same in every figure.

Colorblind / grayscale friendly:
  - Our two methods (the proposed method, src/utils/proposed_method.py:
    named DISCERN, runs are PARAMETRIC) use dark, saturated colors from Paul
    Tol's high-contrast scheme, a thick black edge and no hatch. They're the
    only dark boxes/lines in any figure, so they stand out even in grayscale,
    and the two of them differ in lightness (navy vs. rose).
  - Every other method uses a light color from Paul Tol's colorblind-safe
    "light" scheme plus its own hatch (boxplots) / marker (line plots), so
    they stay distinguishable without relying on color at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.proposed_method import PROPOSED_NOTOP_DISPLAY, PROPOSED_CONF_DISPLAY  # noqa: E402

OUR_METHODS = (PROPOSED_NOTOP_DISPLAY, PROPOSED_CONF_DISPLAY)   # "DISCERN (notop)", "DISCERN (conf.)"

# label: (color, hatch, marker)
METHOD_STYLES = {
    PROPOSED_NOTOP_DISPLAY:   ("#004488", "",     "s"),
    PROPOSED_CONF_DISPLAY:    ("#BB5566", "",     "D"),
    "DELETE":                 ("#EE8866", "//",   "o"),
    "Boundary Shrink":        ("#44BB99", "\\\\", "v"),
    "Boundary Expand":        ("#BBCC33", "xx",   "^"),
    "Bad Teacher":            ("#FFAABB", "..",   "<"),
    "Random Labeling":        ("#88CCEE", "--",   ">"),
    "LF-SSD":                 ("#AAAA00", "++",   "p"),
    "EWC":                    ("#DDAA33", "oo",   "h"),
    # Gaussian ablation: lighter counterpart of the matching proposed method
    "Gaussian (notop)":       ("#77AADD", "||",   "P"),
    "Gaussian (conf.)":       ("#EE99AA", "**",   "X"),
    # Selection-flow ablation (results/ablation/flujo): the proposed method without
    # the test cascade, CV duel for every class
    "CV-only (notop)":        ("#99DDFF", "oo",   "H"),
    "CV-only (conf.)":        ("#FFCCCC", "..",   "8"),
    "Retrain":                ("#999999", "",     "*"),
    "Original":               ("#BBBBBB", "",     "d"),
}
DEFAULT_STYLE = ("#333333", "", "o")

METHOD_COLORS = {label: s[0] for label, s in METHOD_STYLES.items()}
DEFAULT_COLOR = DEFAULT_STYLE[0]

# Canonical legend / box order, shared by every plot (ours first).
METHOD_ORDER = list(METHOD_STYLES)

OTHER_EDGE = "#444444"   # also the hatch color of the other methods' boxes

def is_ours(label):
    return label in OUR_METHODS

def method_color(label):
    return METHOD_STYLES.get(label, DEFAULT_STYLE)[0]

def method_hatch(label):
    return METHOD_STYLES.get(label, DEFAULT_STYLE)[1]

def method_marker(label):
    return METHOD_STYLES.get(label, DEFAULT_STYLE)[2]

def style_box(box, label):
    # Opaque fill (no alpha, it would also fade the hatch).
    box.set_facecolor(method_color(label))
    box.set_hatch(method_hatch(label))
    if is_ours(label):
        box.set_edgecolor("black")
        box.set_linewidth(2.2)
    else:
        box.set_edgecolor(OTHER_EDGE)
        box.set_linewidth(0.9)

# Black median and white-filled mean marker, readable on any fill color.
BOX_KWARGS = dict(
    showmeans=True,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.6),
    meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black", markersize=6),
    flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor="#555555"),
)

def legend_patch(label):
    from matplotlib.patches import Patch
    ours = is_ours(label)
    return Patch(facecolor=method_color(label), hatch=method_hatch(label),
                 edgecolor="black" if ours else OTHER_EDGE,
                 linewidth=2.0 if ours else 0.9, label=label)

def bold_our_legend_entries(legend):
    for text in legend.get_texts():
        if is_ours(text.get_text()):
            text.set_fontweight("bold")

def line_kwargs(label):
    # Line plots: ours thick and on top, the rest thinner with their own marker.
    ours = is_ours(label)
    return dict(
        color=method_color(label),
        marker=method_marker(label),
        markersize=6 if ours else 4,
        linewidth=2.8 if ours else 1.3,
        alpha=1.0 if ours else 0.85,
        zorder=4 if ours else 3,
    )

# Paper-style architecture names for titles/axes (output folders keep the
# raw experiment name).
ARCH_DISPLAY_NAMES = {
    "vit_b_16":          "ViT-B/16",
    "swin_t":            "Swin-T",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "resnet18":          "ResNet18",
}

def arch_name(arch):
    return ARCH_DISPLAY_NAMES.get(arch, arch)
