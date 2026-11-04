"""Proposed method used by default in the analysis / plotting scripts.

Single place to change which runs are "our method" in the paper figures and
tests (extract_methods, plot_figures, plot_boxplots, method_colors, the Wilcoxon
analyses, the NOISY ablation and the ICU selection).

Current choice: PARAMETRIC with the DISTRIBUTIONAL test pipeline and the
tie-break on the KS of the norm (unlearn_parametric.sh VARIANT=tiebreak_norm
PIPELINE=tests), 1000 fixed synthetic samples, embeddings unfrozen.
The previous proposed method (DISTRIBUTIONAL, ledoit-wolf) is kept below so it
can still be referenced as a comparison / ablation.
"""

PROPOSED_FAMILY = "PARAMETRIC"          # version prefix of the proposed method
PROPOSED_TAG = "cv5x3_tbn_m0.05_tests"
# The method is called DISCERN in figures and tables; the runs behind it are
# PARAMETRIC (PROPOSED_FAMILY is the folder prefix, not a name).
PROPOSED_NAME = "DISCERN"               # display name in figures
PROPOSED_LABEL_FAMILY = "DISCERN"       # name in tables / CSVs

_BASE = f"PARAMETRIC_{PROPOSED_TAG}_1000_standard_syntetic_fixed_embs_forget_"
PROPOSED_NOTOP = _BASE + "notop_margin_10.0"
PROPOSED_CONF = _BASE + "corrected_confidence_penalty"

# Labels used to tag the two variants in tables / CSVs ("DISCERN notop")
PROPOSED_NOTOP_LABEL = f"{PROPOSED_LABEL_FAMILY} notop"
PROPOSED_CONF_LABEL = f"{PROPOSED_LABEL_FAMILY} confidence"
# Display names in the figures
PROPOSED_NOTOP_DISPLAY = f"{PROPOSED_NAME} (notop)"
PROPOSED_CONF_DISPLAY = f"{PROPOSED_NAME} (conf.)"

# Gaussian-noise ablation of the proposed method (same hyper-parameters)
NOISY_NOTOP = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
NOISY_CONF = "NOISY_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_confidence_penalty"

# Previous proposed method (DISTRIBUTIONAL).  Its notop margin-10 runs exist under
# two names: the legacy "..._forget_notop" and "..._forget_notop_margin_10.0".
DISTRIBUTIONAL_NOTOP = "DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_notop"
DISTRIBUTIONAL_NOTOP_ALIASES = (DISTRIBUTIONAL_NOTOP, DISTRIBUTIONAL_NOTOP + "_margin_10.0")
DISTRIBUTIONAL_CONF = ("DISTRIBUTIONAL_1000_standard_cov_ledoit_wolf_syntetic_fixed_embs_forget_"
                       "corrected_confidence_penalty")

# Prefix shared by every variant of the proposed configuration (used to select
# all of them, e.g. the homoscedasticity tests of the Wilcoxon analyses)
PROPOSED_PREFIX = f"PARAMETRIC_{PROPOSED_TAG}_1000_standard_syntetic_fixed_embs"
