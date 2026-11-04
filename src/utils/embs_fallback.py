"""
TEMPORARY stand-in for the "embs" runs that are being relaunched (relaunch_embs_*.sh).

Until the relaunch finishes, the efficientnet_v2_s "embs" methods (and SSD / Fisher on
swin_t) have no runs under their real name: the old ones were renamed by
rename_embs_mislabeled.sh because they did not train "classifier + last stage":
    efficientnet_v2_s : "*_embs" really trained the whole backbone -> now "*_nofreeze"
                        (or no suffix); SSD / Fisher only touched the classifier -> "*_freeze"
    swin_t            : SSD / Fisher "*_embs" only touched the classifier -> "*_freeze"
While TEMP_EMBS_FALLBACK is True, add_embs_fallback() adds a COPY of those rows under the
"_embs" name the summaries / figures look for, ONLY where the real (relaunched) "_embs" run
is not there yet for the same experiment / id_database / epoch or step.  Applied in memory
when the data are read; the processed results CSVs are never modified.

The relaunch is complete (Oct 2026), so the fallback is now OFF by default; set
EMBS_FALLBACK=1 in the environment to bring the stand-ins back.
"""

import os
import re

import pandas as pd

TEMP_EMBS_FALLBACK = os.environ.get("EMBS_FALLBACK", "0") == "1"

# Inverse of rename_embs_mislabeled.sh (old name of the stand-in -> "embs" name)
_EFFNET_RULES = [
    (re.compile(r"^(SSD|FISHERFORGETTING)_(.*)_freeze$"), r"\1_\2_embs"),
    (re.compile(r"^DELETE_nofreeze$"), "DELETE_embs"),
    (re.compile(r"^(EWC_.*_lambda_[A-Za-z]+)_nofreeze_(.*)$"), r"\1_embs_\2"),
    (re.compile(r"^((?:DISTRIBUTIONAL|NOISY|PARAMETRIC)_.*_syntetic_(?:fixed|moving))_nofreeze_forget_(.*)$"),
     r"\1_embs_forget_\2"),
    (re.compile(r"^(RANDOMLABEL_[a-z]+)_forbidden$"), r"\1_embs_forbidden"),
    (re.compile(r"^(RANDOMLABEL_[a-z]+)$"), r"\1_embs"),
    (re.compile(r"^((?:BADTEACHER|BOUNDARYEXPAND|BOUNDARYSHRINK|ISPF|SFU)_[^/]*?)$"), r"\1_embs"),
]
_SWIN_RULES = _EFFNET_RULES[:1]
_SKIP_SUFFIX = re.compile(r"_(embs|freeze|nofreeze)(_forbidden)?$")
KEY_CANDIDATES = ("experiment", "id_database", "dataset", "num_classes_removed", "epoch", "steps")


def embs_name(base, experiment):
    """Stand-in base version -> the "embs" base it replaces (None if it is not a stand-in)."""
    exp = str(experiment)
    if exp.startswith("efficientnet_v2_s"):
        rules = _EFFNET_RULES
    elif exp.startswith("swin_t"):
        rules = _SWIN_RULES
    else:
        return None
    for i, (rx, repl) in enumerate(rules):
        if rx.match(base):
            # the generic "no suffix" rules must not catch an explicit freeze variant
            if i >= 5 and _SKIP_SUFFIX.search(base):
                return None
            return rx.sub(repl, base)
    return None


def add_embs_fallback(df, version_col="version", exp_col="experiment", verbose=True):
    """pandas DataFrame -> same rows + stand-in copies under the "embs" names."""
    if not TEMP_EMBS_FALLBACK or df is None or len(df) == 0 or version_col not in df or exp_col not in df:
        return df
    version = df[version_col].astype(str)
    base = version.str.split("/", n=1).str[0]
    rest = pd.Series([v[len(b):] for v, b in zip(version, base)], index=df.index)   # "/original_best" or ""
    mapped = pd.Series([embs_name(b, e) for b, e in zip(base, df[exp_col])], index=df.index)
    cand = mapped.notna()
    if not cand.any():
        return df
    keys = [c for c in KEY_CANDIDATES if c in df.columns]
    copies = df[cand].copy()
    copies[version_col] = (mapped[cand] + rest[cand]).to_numpy()
    # never shadow a real (relaunched) run
    existing = set(map(tuple, df[keys + [version_col]].astype(str).to_numpy()))
    keep = [tuple(row) not in existing for row in copies[keys + [version_col]].astype(str).to_numpy()]
    copies = copies[keep]
    if verbose and len(copies):
        print(f"[embs fallback] {len(copies)} stand-in rows added "
              f"(efficientnet nofreeze / SSD freeze as embs until the relaunch is done)")
    return pd.concat([df, copies], ignore_index=True)


def add_embs_fallback_pl(df, version_col="version", exp_col="experiment", verbose=True):
    """polars version of add_embs_fallback (native: the env has no pyarrow for to_pandas)."""
    if (not TEMP_EMBS_FALLBACK or df is None or df.height == 0
            or version_col not in df.columns or exp_col not in df.columns):
        return df
    import polars as pl
    new = []
    for v, e in zip(df[version_col].cast(pl.Utf8).to_list(), df[exp_col].cast(pl.Utf8).to_list()):
        base, sep, rest = (v or "").partition("/")
        m = embs_name(base, e)
        new.append(None if m is None else m + sep + rest)
    copies = (df.with_columns(pl.Series("__embs_version", new, dtype=pl.Utf8))
                .filter(pl.col("__embs_version").is_not_null()))
    if copies.height == 0:
        return df
    copies = (copies.with_columns(pl.col("__embs_version").cast(df.schema[version_col]).alias(version_col))
                    .drop("__embs_version"))
    # never shadow a real (relaunched) run
    keys = [c for c in KEY_CANDIDATES if c in df.columns] + [version_col]
    copies = copies.join(df.select(keys).unique(), on=keys, how="anti", nulls_equal=True).select(df.columns)
    if verbose and copies.height:
        print(f"[embs fallback] {copies.height} stand-in rows added "
              f"(efficientnet nofreeze / SSD freeze as embs until the relaunch is done)")
    return pl.concat([df, copies], how="vertical_relaxed")
