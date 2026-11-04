#!/usr/bin/env python3
"""
Computes logit-based and restricted-softmax metrics for ICU results.

For each per-method unlearning_results_ICU.csv found in results_ICU/:
  - Backs up the original CSV as unlearning_results_ICU_old_2.csv (once, if not done)
  - Loads the ICU step model from models/
  - Performs inference on retain-test and forget-test sets
  - Computes logit statistics and restricted-softmax probabilities
  - Writes results back into the same unlearning_results_ICU.csv (adds new columns)

New columns added:
  max_logit_retain, mean_logits_retain, min_logit_retain, max_second_logit_retain
  max_logit_forget, mean_logits_forget, min_logit_forget, max_second_logit_forget
    -- raw logit stats averaged over retain-test / forget-test samples

  max_probability_with_retain, max_probability_forget_with_retain
    -- max softmax probability computed over retain-class logits only

  mean_prob_no_max                    -- retain set, full softmax, mean excl. max
  mean_prob_no_max_with_retain        -- retain set, retain-logits softmax, mean excl. max
  mean_prob_no_max_forget             -- forget set, full softmax, mean excl. max
  mean_prob_no_max_forget_with_retain -- forget set, retain-logits softmax, mean excl. max

Usage:
    python src/collect/compute_ICU_logit_metrics.py [--results-dir results_ICU]
                                        [--models-dir models]
                                        [--data-path datasets]
                                        [--include-version V1 V2 ...]
                                        [--exclude-version V1 V2 ...]
                                        [--include-model M1 ...]
                                        [--exclude-model M1 ...]
                                        [--include-dataset D1 ...]
                                        [--exclude-dataset D1 ...]
                                        [--continue]
                                        [--batch-size 64]
"""

import os
import re
import csv
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import DataLoader

from src.utils.load_data import _load_raw_dataset, number_dataset_classes
from src.utils.dataset_loader import ClassificationDataset
from src.models.ClassificationModel import ClassificationModel
from src.models.transforms import load_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNOWN_ARCHITECTURES = sorted(
    ["vit_b_16", "resnet18", "resenet18", "efficientnet_v2_s"],
    key=len, reverse=True,
)

ONE_SHOT_METHODS = {"SSD", "FISHERFORGETTING", "SFU"}

NEW_METRIC_COLS = [
    "max_logit_retain",        "mean_logits_retain",
    "min_logit_retain",        "max_second_logit_retain",
    "max_logit_forget",        "mean_logits_forget",
    "min_logit_forget",        "max_second_logit_forget",
    "max_probability_with_retain",
    "max_probability_forget_with_retain",
    "mean_prob_no_max",
    "mean_prob_no_max_with_retain",
    "mean_prob_no_max_forget",
    "mean_prob_no_max_forget_with_retain",
]


# ── Path / name parsing ───────────────────────────────────────────────────────

def parse_experiment_folder(folder_name):
    parts = folder_name.split("_")
    for arch in KNOWN_ARCHITECTURES:
        arch_parts = arch.split("_")
        if parts[: len(arch_parts)] == arch_parts:
            id_database = "_".join(parts[len(arch_parts):])
            return arch, id_database
    return None, None


def parse_id_database(id_database):
    m = re.match(r"^(.+?)_classes_to_remove_([\d_]+)$", id_database)
    if not m:
        return None, None
    return m.group(1), [int(c) for c in m.group(2).split("_")]


def find_checkpoint(model_step_dir, version_str):
    method_prefix = version_str.split("/")[0].split("_")[0].upper()
    preferred = (
        "checkpoint_epoch_00.pt"
        if method_prefix in ONE_SHOT_METHODS
        else "checkpoint_best.pt"
    )
    alt = (
        "checkpoint_best.pt"
        if preferred == "checkpoint_epoch_00.pt"
        else "checkpoint_epoch_00.pt"
    )
    for name in (preferred, alt):
        p = os.path.join(model_step_dir, name)
        if os.path.exists(p):
            return p
    if os.path.isdir(model_step_dir):
        candidates = sorted(
            f for f in os.listdir(model_step_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        )
        if candidates:
            return os.path.join(model_step_dir, candidates[-1])
    return None


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_dataloader(images, labels, transform, batch_size, num_workers):
    ds = ClassificationDataset(images, labels, transforms=transform)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def get_test_data(dataset_name, data_path, cache):
    key = dataset_name.upper()
    if key not in cache:
        print(f"  Loading test data: {dataset_name} ...", flush=True)
        X, y = _load_raw_dataset(dataset_name, data_path, train=False)
        cache[key] = (X, np.array(y))
    return cache[key]


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_logits(model, loader, device):
    """Run inference and return raw logits as np.ndarray (N, num_classes)."""
    parts = []
    for imgs, _ in loader:
        logits = model(imgs.to(device))
        parts.append(logits.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Metric helpers ────────────────────────────────────────────────────────────

def softmax_np(logits):
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def mean_prob_excl_max(probs):
    """
    Mean probability of non-max classes, averaged over samples.
    Since softmax sums to 1: mean_excl_max = (1 - max_prob) / (n_classes - 1).
    """
    n_classes = probs.shape[1]
    max_prob = probs[np.arange(len(probs)), np.argmax(probs, axis=1)]
    return float(np.mean((1.0 - max_prob) / (n_classes - 1)))


def logit_stats(logits):
    """Raw logit statistics averaged over samples. Returns a dict."""
    sorted_desc = np.sort(logits, axis=1)[:, ::-1]
    return {
        "max_logit":        float(np.mean(sorted_desc[:, 0])),
        "mean_logits":      float(np.mean(logits)),
        "min_logit":        float(np.mean(sorted_desc[:, -1])),
        "max_second_logit": float(np.mean(sorted_desc[:, 1])),
    }


def compute_metrics(logits_retain, logits_forget, retain_indices):
    """
    Compute all new metrics given raw logits for retain and forget test sets.

    logits_retain : np.ndarray (N_r, C)
    logits_forget : np.ndarray (N_f, C) or None if nothing forgotten yet (step 0)
    retain_indices: list of int — class indices NOT forgotten at this step
    """
    # ── Retain logit stats ────────────────────────────────────────────────────
    r_stats = logit_stats(logits_retain)
    retain_part = {
        "max_logit_retain":        r_stats["max_logit"],
        "mean_logits_retain":      r_stats["mean_logits"],
        "min_logit_retain":        r_stats["min_logit"],
        "max_second_logit_retain": r_stats["max_second_logit"],
    }

    # ── Forget logit stats ────────────────────────────────────────────────────
    if logits_forget is not None and len(logits_forget) > 0:
        f_stats = logit_stats(logits_forget)
        forget_part = {
            "max_logit_forget":        f_stats["max_logit"],
            "mean_logits_forget":      f_stats["mean_logits"],
            "min_logit_forget":        f_stats["min_logit"],
            "max_second_logit_forget": f_stats["max_second_logit"],
        }
    else:
        forget_part = {
            "max_logit_forget":        0.0,
            "mean_logits_forget":      0.0,
            "min_logit_forget":        0.0,
            "max_second_logit_forget": 0.0,
        }

    # ── Retain set — full softmax ─────────────────────────────────────────────
    probs_r     = softmax_np(logits_retain)
    mean_pnm    = mean_prob_excl_max(probs_r)

    # ── Retain set — retain-restricted softmax ────────────────────────────────
    probs_r_sub  = softmax_np(logits_retain[:, retain_indices])
    max_prob_wr  = float(np.mean(np.max(probs_r_sub, axis=1)))
    mean_pnm_wr  = mean_prob_excl_max(probs_r_sub)

    # ── Forget set ────────────────────────────────────────────────────────────
    if logits_forget is not None and len(logits_forget) > 0:
        probs_f      = softmax_np(logits_forget)
        mean_pnm_f   = mean_prob_excl_max(probs_f)

        probs_f_sub   = softmax_np(logits_forget[:, retain_indices])
        max_prob_fwr  = float(np.mean(np.max(probs_f_sub, axis=1)))
        mean_pnm_fwr  = mean_prob_excl_max(probs_f_sub)
    else:
        mean_pnm_f = max_prob_fwr = mean_pnm_fwr = 0.0

    return {
        **retain_part,
        **forget_part,
        "max_probability_with_retain":         max_prob_wr,
        "max_probability_forget_with_retain":  max_prob_fwr,
        "mean_prob_no_max":                    mean_pnm,
        "mean_prob_no_max_with_retain":        mean_pnm_wr,
        "mean_prob_no_max_forget":             mean_pnm_f,
        "mean_prob_no_max_forget_with_retain": mean_pnm_fwr,
    }


# ── Filters ───────────────────────────────────────────────────────────────────

def passes_filters(version_str, base_model, dataset, args):
    if args.exclude_version and any(ev in version_str for ev in args.exclude_version):
        return False
    if args.include_version and not any(iv in version_str for iv in args.include_version):
        return False
    if args.exclude_model and any(em in base_model for em in args.exclude_model):
        return False
    if args.include_model and not any(im in base_model for im in args.include_model):
        return False
    if args.exclude_dataset and any(ed in dataset for ed in args.exclude_dataset):
        return False
    if args.include_dataset and not any(id_ in dataset for id_ in args.include_dataset):
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compute logit-based and restricted-softmax metrics for ICU results"
    )
    ap.add_argument("--results-dir", default="results_ICU")
    ap.add_argument("--models-dir",  default="models")
    ap.add_argument("--data-path",   default="datasets")

    ap.add_argument("--include-version", nargs="+", default=None)
    ap.add_argument("--exclude-version", nargs="+", default=None)
    ap.add_argument("--include-model",   nargs="+", default=None)
    ap.add_argument("--exclude-model",   nargs="+", default=None)
    ap.add_argument("--include-dataset", nargs="+", default=None)
    ap.add_argument("--exclude-dataset", nargs="+", default=None)

    ap.add_argument(
        "--continue", dest="continue_run", action="store_true", default=False,
        help="Skip steps whose new metrics are already present in the output CSV",
    )
    ap.add_argument("--batch-size",  type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    models_dir  = Path(args.models_dir)

    print(f"Device      : {DEVICE}")
    print(f"Results dir : {results_dir}")
    print(f"Models dir  : {models_dir}")
    print(f"Data path   : {args.data_path}")
    print(f"Continue    : {args.continue_run}")

    test_data_cache = {}

    # ── Discover per-method CSVs ──────────────────────────────────────────────
    csv_files = []
    for path in sorted(results_dir.rglob("unlearning_results_ICU.csv")):
        rel_parts = path.relative_to(results_dir).parts
        if any(part.startswith("_") for part in rel_parts[:-1]):
            continue  # archived runs, e.g. results_ICU/_old_incomplete/
        if len(rel_parts) >= 4:
            csv_files.append(path)

    print(f"\nFound {len(csv_files)} per-method ICU CSV files.\n")

    for csv_path in csv_files:
        rel_parts         = csv_path.relative_to(results_dir).parts
        experiment_folder = rel_parts[0]
        version_str       = "/".join(rel_parts[1:-1])

        base_model, id_database = parse_experiment_folder(experiment_folder)
        if base_model is None:
            print(f"⚠️  Unknown architecture in '{experiment_folder}', skipping.", flush=True)
            continue

        dataset, ordered_classes = parse_id_database(id_database)
        if dataset is None:
            print(f"⚠️  Cannot parse dataset/classes from '{id_database}', skipping.", flush=True)
            continue

        if not passes_filters(version_str, base_model, dataset, args):
            continue

        print(f"{'='*72}", flush=True)
        print(f"  {experiment_folder}", flush=True)
        print(f"  version : {version_str}", flush=True)
        print(f"  dataset : {dataset}", flush=True)
        print(f"  classes : {ordered_classes}", flush=True)

        # ── Backup: _old_2 is the permanent source of truth ───────────────────
        old2_csv = csv_path.parent / "unlearning_results_ICU_old_2.csv"
        if not old2_csv.exists():
            shutil.copy2(str(csv_path), str(old2_csv))
            print(f"  📦 Backed up original → {old2_csv.name}", flush=True)

        # ── Read source data (always from _old_2) ─────────────────────────────
        try:
            with open(old2_csv, "r") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"⚠️  Cannot read {old2_csv}: {e}", flush=True)
            continue

        if not rows:
            print(f"  Empty CSV, skipping.", flush=True)
            continue

        # ── Load cached metric values for --continue ──────────────────────────
        cached_metrics = {}
        if args.continue_run and csv_path.exists():
            try:
                with open(csv_path, "r") as f:
                    for r in csv.DictReader(f):
                        if r.get("max_logit_retain", ""):
                            key = (
                                r["experiment"], r["id_database"],
                                r["version"], int(r["steps"]),
                            )
                            cached_metrics[key] = {
                                col: r[col] for col in NEW_METRIC_COLS if col in r
                            }
            except Exception as e:
                print(f"⚠️  Could not read cached metrics from {csv_path}: {e}", flush=True)
            if cached_metrics:
                print(f"  --continue: {len(cached_metrics)} steps already computed.", flush=True)

        # ── Determine output fieldnames ───────────────────────────────────────
        orig_fieldnames   = list(rows[0].keys())
        extra_cols        = [c for c in NEW_METRIC_COLS if c not in orig_fieldnames]
        output_fieldnames = orig_fieldnames + extra_cols

        # ── Build transforms ──────────────────────────────────────────────────
        fake_args = SimpleNamespace(base_model=base_model, dataset=dataset.upper())
        transform, _, _ = load_transforms(fake_args)

        # ── Load / cache test data ────────────────────────────────────────────
        X_test_all, y_test_all = get_test_data(dataset, args.data_path, test_data_cache)
        num_classes = number_dataset_classes(dataset)

        # ── Process each ICU step ─────────────────────────────────────────────
        output_rows = []

        for row in rows:
            step           = int(row["steps"])
            experiment_col = row["experiment"]
            version_col    = row["version"]
            key            = (experiment_col, id_database, version_col, step)

            if key in cached_metrics:
                print(f"  ⏭  Step {step} already done, skipping.", flush=True)
                output_rows.append({**row, **cached_metrics[key]})
                continue

            current_forbidden = set(ordered_classes[:step])
            retain_indices    = [c for c in range(num_classes) if c not in current_forbidden]

            # ── Find checkpoint ───────────────────────────────────────────────
            model_step_dir = str(
                models_dir / experiment_folder / f"ICU_step_{step}" / version_str
            )
            ckpt_path = find_checkpoint(model_step_dir, version_str)
            if ckpt_path is None:
                print(f"  ⚠️  No checkpoint for step {step} at {model_step_dir}", flush=True)
                continue

            print(
                f"  → Step {step}/{len(ordered_classes)} | {os.path.basename(ckpt_path)}",
                flush=True,
            )

            # ── Split test set ────────────────────────────────────────────────
            retain_imgs = [img for img, lbl in zip(X_test_all, y_test_all)
                           if lbl not in current_forbidden]
            retain_lbls = [lbl for lbl in y_test_all if lbl not in current_forbidden]
            forget_imgs = [img for img, lbl in zip(X_test_all, y_test_all)
                           if lbl in current_forbidden]
            forget_lbls = [lbl for lbl in y_test_all if lbl in current_forbidden]

            # ── Load model ────────────────────────────────────────────────────
            model = ClassificationModel(
                base_model, num_classes, {},
                classes_weights=None, extra_layers=False, pretrained=False,
            )
            try:
                ckpt_data = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt_data.get("state_dict", ckpt_data))
            except Exception as e:
                print(f"  ⚠️  Load error {ckpt_path}: {e}", flush=True)
                del model
                continue
            model.to(DEVICE).eval()

            # ── Run inference ─────────────────────────────────────────────────
            retain_loader = build_dataloader(
                retain_imgs, retain_lbls,
                transform, args.batch_size, args.num_workers,
            )
            logits_retain = get_logits(model, retain_loader, DEVICE)

            if forget_imgs:
                forget_loader = build_dataloader(
                    forget_imgs, forget_lbls,
                    transform, args.batch_size, args.num_workers,
                )
                logits_forget = get_logits(model, forget_loader, DEVICE)
            else:
                forget_loader = None
                logits_forget = None

            # ── Compute metrics ───────────────────────────────────────────────
            new_metrics = compute_metrics(logits_retain, logits_forget, retain_indices)

            print(
                f"     max_l_r={new_metrics['max_logit_retain']:.3f} "
                f"max_l_f={new_metrics['max_logit_forget']:.3f} "
                f"mp_wr={new_metrics['max_probability_with_retain']:.4f} "
                f"mp_fwr={new_metrics['max_probability_forget_with_retain']:.4f}",
                flush=True,
            )

            output_rows.append({**row, **new_metrics})

            # ── Free GPU memory ───────────────────────────────────────────────
            del model, retain_loader, logits_retain
            if forget_loader is not None:
                del forget_loader, logits_forget
            torch.cuda.empty_cache()

        # ── Write all rows back to the original CSV ───────────────────────────
        if not output_rows:
            print(f"  No rows to write, skipping.", flush=True)
            continue

        with open(str(csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"  💾 Saved → {csv_path.name}  ({len(output_rows)} rows)", flush=True)

    print("\n✅ Done.", flush=True)


if __name__ == "__main__":
    main()
