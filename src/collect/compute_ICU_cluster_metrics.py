#!/usr/bin/env python3
"""
Computes accuracy + clustering + NEAR-OOD embedding metrics for ICU results.

For each per-method unlearning_results_ICU.csv found in results_ICU/:
  - Backs up the original CSV as unlearning_results_ICU_old.csv (once, if not done)
  - Loads the ICU step model from models/
  - Performs inference on retain-test, forget-test, and NEAR OOD test sets
  - Computes accuracy, maxprobability, and embedding-based clustering metrics
  - Writes results back into the same unlearning_results_ICU.csv (adds new columns)

New columns added:
  maxprobability, maxprobability_forget,
  maxprobability_retain, maxprobability_unknown,
  cohesion_forget_l2, distance_inter_class_l2, cohesion_forget_cos, distance_inter_class_cos,
  cohesion_retain_l2, distance_inter_retain_l2, cohesion_retain_cos, distance_inter_retain_cos,
  cohesion_unknown_l2, distance_inter_unknown_l2, cohesion_unknown_cos, distance_inter_unknown_cos

NEAR OOD mapping (matches extract_NEAR.sh):
  cifar10  -> cifar100  (full test set, no class filter)
  cifar100 -> cifar10
  mnist    -> letters

Usage:
    python src/collect/compute_ICU_cluster_metrics.py [--results-dir results_ICU]
                                          [--models-dir models]
                                          [--data-path datasets]
                                          [--include-version V1 V2 ...]
                                          [--exclude-version V1 V2 ...]
                                          [--include-model M1 ...]
                                          [--exclude-model M1 ...]
                                          [--include-dataset D1 ...]
                                          [--exclude-dataset D1 ...]
                                          [--online] [--continue]
                                          [--extra-layer]
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
from sklearn.metrics import accuracy_score

from src.utils.load_data import _load_raw_dataset, number_dataset_classes
from src.utils.dataset_loader import ClassificationDataset
from src.models.ClassificationModel import ClassificationModel
from src.models.transforms import load_transforms
from src.utils.cluster_metrics import (
    NEAR_DATASET_MAP as _NEAR_MAP,
    evaluate_and_extract,
    cross_metrics,
    retain_metrics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lower-case keys for convenience (dataset names come from id_database in lower-case)
NEAR_DATASET_MAP = {k.lower(): v.lower() for k, v in _NEAR_MAP.items()}

KNOWN_ARCHITECTURES = sorted(
    ["vit_b_16", "resnet18", "resenet18", "efficientnet_v2_s"],
    key=len, reverse=True,
)

ONE_SHOT_METHODS = {"SSD", "FISHERFORGETTING", "SFU"}

# Columns added by this script (may or may not already exist in the original CSV)
NEW_METRIC_COLS = [
    "accuracy_score_after", "unlearning_accuracy_score",
    "maxprobability", "maxprobability_forget",
    "maxprobability_retain", "maxprobability_unknown",
    "cohesion_forget_l2", "distance_inter_class_l2",
    "cohesion_forget_cos", "distance_inter_class_cos",
    "cohesion_retain_l2", "distance_inter_retain_l2",
    "cohesion_retain_cos", "distance_inter_retain_cos",
    "cohesion_unknown_l2", "distance_inter_unknown_l2",
    "cohesion_unknown_cos", "distance_inter_unknown_cos",
]


# ── Path / name parsing ───────────────────────────────────────────────────────

def parse_experiment_folder(folder_name):
    """
    Split e.g. 'vit_b_16_cifar10_classes_to_remove_5_1_6_4_3'
    into (base_model, id_database).  Returns (None, None) on failure.
    """
    parts = folder_name.split("_")
    for arch in KNOWN_ARCHITECTURES:
        arch_parts = arch.split("_")
        if parts[: len(arch_parts)] == arch_parts:
            id_database = "_".join(parts[len(arch_parts) :])
            return arch, id_database
    return None, None


def parse_id_database(id_database):
    """
    Extract (dataset, ordered_classes) from id_database.
    e.g. 'cifar10_classes_to_remove_5_1_6_4_3' -> ('cifar10', [5, 1, 6, 4, 3])
    """
    m = re.match(r"^(.+?)_classes_to_remove_([\d_]+)$", id_database)
    if not m:
        return None, None
    return m.group(1), [int(c) for c in m.group(2).split("_")]


def find_checkpoint(model_step_dir, version_str):
    """
    Find the best available checkpoint for one ICU step.
    Respects ONE_SHOT_METHODS (epoch_00 preferred), falls back otherwise.
    """
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
    """Load (and cache) the full test split of a dataset."""
    key = dataset_name.upper()
    if key not in cache:
        print(f"  Loading test data: {dataset_name} ...", flush=True)
        X, y = _load_raw_dataset(dataset_name, data_path, train=False)
        cache[key] = (X, np.array(y))
    return cache[key]


# ── Inference / metrics — delegated to src.utils.cluster_metrics ─────────────

def compute_accuracy(logits, labels):
    return float(accuracy_score(labels, np.argmax(logits, axis=1)))


# ── Continue / Online helpers ─────────────────────────────────────────────────

def load_already_processed(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                done.add((
                    row["experiment"],
                    row["id_database"],
                    row["version"],
                    int(row["steps"]),
                ))
    except Exception as e:
        print(f"⚠️  Could not read {csv_path}: {e}", flush=True)
    return done


def open_output_file(output_csv_path, continue_run, fieldnames):
    """Open (or append to) the output CSV; write header if needed."""
    file_exists = os.path.exists(output_csv_path)
    mode = "a" if (continue_run and file_exists) else "w"
    f = open(output_csv_path, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if mode == "w" or not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer


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
        description="Compute accuracy + clustering metrics for ICU results"
    )
    ap.add_argument("--results-dir", default="results_ICU")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--data-path", default="datasets")

    ap.add_argument("--include-version", nargs="+", default=None)
    ap.add_argument("--exclude-version", nargs="+", default=None)
    ap.add_argument("--include-model", nargs="+", default=None)
    ap.add_argument("--exclude-model", nargs="+", default=None)
    ap.add_argument("--include-dataset", nargs="+", default=None)
    ap.add_argument("--exclude-dataset", nargs="+", default=None)

    ap.add_argument(
        "--online", action="store_true", default=False,
        help="Write each row to disk immediately after computing it",
    )
    ap.add_argument(
        "--continue", dest="continue_run", action="store_true", default=False,
        help="Skip steps already present in the output CSV",
    )
    ap.add_argument("--extra-layer", action="store_true", default=False)
    ap.add_argument("--embedding-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    models_dir = Path(args.models_dir)

    print(f"Device       : {DEVICE}")
    print(f"Results dir  : {results_dir}")
    print(f"Models dir   : {models_dir}")
    print(f"Data path    : {args.data_path}")
    print(f"Online mode  : {args.online}")
    print(f"Continue     : {args.continue_run}")

    test_data_cache = {}

    # ── Discover per-method CSVs (depth >= 4 parts from results_dir root) ────
    csv_files = []
    for path in sorted(results_dir.rglob("unlearning_results_ICU.csv")):
        rel_parts = path.relative_to(results_dir).parts
        if any(part.startswith("_") for part in rel_parts[:-1]):
            continue  # archived runs, e.g. results_ICU/_old_incomplete/
        # (experiment_folder, method_part, ..., filename) -> at least 4 parts
        if len(rel_parts) >= 4:
            csv_files.append(path)

    print(f"\nFound {len(csv_files)} per-method ICU CSV files.\n")

    # Collect rows if not --online
    all_rows_buffer = []

    for csv_path in csv_files:
        rel_parts = csv_path.relative_to(results_dir).parts
        experiment_folder = rel_parts[0]
        version_str = "/".join(rel_parts[1:-1])

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

        near_dataset = NEAR_DATASET_MAP.get(dataset.lower())
        if near_dataset is None:
            print(f"⚠️  No NEAR mapping for '{dataset}', skipping.", flush=True)
            continue

        print(f"{'='*72}", flush=True)
        print(f"  {experiment_folder}", flush=True)
        print(f"  version : {version_str}", flush=True)
        print(f"  dataset : {dataset} | NEAR: {near_dataset}", flush=True)
        print(f"  classes : {ordered_classes}", flush=True)

        # Output goes back to the same file; original is backed up as _old
        output_csv = csv_path
        old_csv = csv_path.parent / "unlearning_results_ICU_old.csv"

        already_done = set()
        if args.continue_run:
            already_done = load_already_processed(str(output_csv))
            if already_done:
                print(f"  --continue: {len(already_done)} steps already computed.", flush=True)

        # Read original ICU CSV
        try:
            with open(csv_path, "r") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            print(f"⚠️  Cannot read {csv_path}: {e}", flush=True)
            continue

        if not rows:
            print(f"  Empty CSV, skipping.", flush=True)
            continue

        # Determine output fieldnames: preserve originals, append any missing new cols
        orig_fieldnames = list(rows[0].keys())
        extra_cols = [c for c in NEW_METRIC_COLS if c not in orig_fieldnames]
        output_fieldnames = orig_fieldnames + extra_cols

        # Backup original (once; skip if already backed up from a previous run)
        if not old_csv.exists():
            shutil.copy2(str(csv_path), str(old_csv))
            print(f"  📦 Backed up original → {old_csv.name}", flush=True)

        # Build transforms (modality matches between dataset and near_dataset)
        fake_args = SimpleNamespace(base_model=base_model, dataset=dataset.upper())
        transform, _, _ = load_transforms(fake_args)

        # Load / cache test datasets
        X_test_all, y_test_all = get_test_data(dataset, args.data_path, test_data_cache)
        X_near_all, y_near_all = get_test_data(near_dataset, args.data_path, test_data_cache)
        num_classes = number_dataset_classes(dataset)

        near_loader = build_dataloader(
            X_near_all, list(y_near_all),
            transform, args.batch_size, args.num_workers,
        )

        # Open output file when in --online mode
        online_file = online_writer = None
        if args.online:
            online_file, online_writer = open_output_file(
                str(output_csv), args.continue_run, output_fieldnames
            )
            print(f"  📝 Online mode: writing to {output_csv}", flush=True)

        csv_rows_output = []  # collected when not --online

        for row in rows:
            step = int(row["steps"])
            experiment_col = row["experiment"]
            version_col = row["version"]

            if args.continue_run and (experiment_col, id_database, version_col, step) in already_done:
                print(f"  ⏭  Step {step} already done, skipping.", flush=True)
                continue

            current_forbidden = set(ordered_classes[:step])

            # Find model checkpoint
            model_step_dir = str(
                models_dir / experiment_folder / f"ICU_step_{step}" / version_str
            )
            ckpt_path = find_checkpoint(model_step_dir, version_str)
            if ckpt_path is None:
                print(
                    f"  ⚠️  No checkpoint for step {step} at {model_step_dir}",
                    flush=True,
                )
                continue

            print(
                f"  → Step {step}/{len(ordered_classes)} | {os.path.basename(ckpt_path)}",
                flush=True,
            )

            # Split test set
            retain_imgs = [img for img, lbl in zip(X_test_all, y_test_all) if lbl not in current_forbidden]
            retain_lbls = [lbl for lbl in y_test_all if lbl not in current_forbidden]
            forget_imgs = [img for img, lbl in zip(X_test_all, y_test_all) if lbl in current_forbidden]
            forget_lbls = [lbl for lbl in y_test_all if lbl in current_forbidden]

            # Load model (pretrained=False: we load our own checkpoint, no need to download ImageNet weights)
            model = ClassificationModel(
                base_model, num_classes, {},
                classes_weights=None, extra_layers=args.extra_layer,
                pretrained=False,
            )
            try:
                ckpt_data = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt_data.get("state_dict", ckpt_data))
            except Exception as e:
                print(f"  ⚠️  Load error {ckpt_path}: {e}", flush=True)
                del model
                continue
            model.to(DEVICE).eval()

            # Retain: single-pass accuracy + embeddings
            retain_loader = build_dataloader(
                retain_imgs, retain_lbls,
                transform, args.batch_size, args.num_workers,
            )
            acc_retain, maxprob_retain, lgr, embr, lblr = evaluate_and_extract(
                model, retain_loader, DEVICE
            )

            # Forget: single-pass accuracy + embeddings
            if forget_imgs:
                forget_loader = build_dataloader(
                    forget_imgs, forget_lbls,
                    transform, args.batch_size, args.num_workers,
                )
                acc_forget, maxprob_forget, lgf, embf, lblf = evaluate_and_extract(
                    model, forget_loader, DEVICE
                )
            else:
                forget_loader = lgf = embf = lblf = None
                acc_forget = maxprob_forget = 0.0

            # NEAR: embeddings
            _, maxprob_unknown, lgn, embn, lbln = evaluate_and_extract(
                model, near_loader, DEVICE
            )

            # Clustering metrics (all GPU-accelerated via shared module)
            if embf is not None and len(embf) > 0:
                cf_l2, if_l2, cf_cos, if_cos = cross_metrics(embf, lblf, embr, lblr, DEVICE)
            else:
                cf_l2 = if_l2 = cf_cos = if_cos = 0.0

            cr_l2, ir_l2, cr_cos, ir_cos = retain_metrics(embr, lblr, DEVICE)
            cn_l2, in_l2, cn_cos, in_cos = cross_metrics(embn, lbln, embr, lblr, DEVICE)

            # Merge original row with newly computed metrics
            new_metrics = {
                "accuracy_score_after":    acc_retain,
                "unlearning_accuracy_score": acc_forget,
                "maxprobability":          maxprob_retain,
                "maxprobability_forget":   maxprob_forget,
                "maxprobability_retain":   maxprob_retain,
                "maxprobability_unknown":  maxprob_unknown,
                "cohesion_forget_l2":      cf_l2,
                "distance_inter_class_l2": if_l2,
                "cohesion_forget_cos":     cf_cos,
                "distance_inter_class_cos": if_cos,
                "cohesion_retain_l2":      cr_l2,
                "distance_inter_retain_l2": ir_l2,
                "cohesion_retain_cos":     cr_cos,
                "distance_inter_retain_cos": ir_cos,
                "cohesion_unknown_l2":     cn_l2,
                "distance_inter_unknown_l2": in_l2,
                "cohesion_unknown_cos":    cn_cos,
                "distance_inter_unknown_cos": in_cos,
            }
            out_row = {**row, **new_metrics}

            if args.online:
                online_writer.writerow(out_row)
                online_file.flush()
                print(
                    f"     ✅ acc_r={acc_retain:.4f} acc_f={acc_forget:.4f} "
                    f"mp_r={maxprob_retain:.4f} mp_unk={maxprob_unknown:.4f}",
                    flush=True,
                )
            else:
                csv_rows_output.append(out_row)
                print(
                    f"     acc_r={acc_retain:.4f} acc_f={acc_forget:.4f} "
                    f"mp_r={maxprob_retain:.4f} mp_unk={maxprob_unknown:.4f}",
                    flush=True,
                )

            # Free GPU memory
            del model, retain_loader
            if forget_loader is not None:
                del forget_loader
            del lgr, embr, lblr, lgn, embn, lbln
            if lgf is not None:
                del lgf, embf, lblf
            torch.cuda.empty_cache()

        if args.online:
            if online_file is not None:
                online_file.close()
            print(f"  💾 Closed: {output_csv}", flush=True)
        else:
            all_rows_buffer.append((output_csv, csv_rows_output, output_fieldnames))

    # ── Batch write (non-online mode) ─────────────────────────────────────────
    if not args.online:
        for output_csv, rows_out, fieldnames in all_rows_buffer:
            if not rows_out:
                continue
            file_exists = os.path.exists(str(output_csv))
            mode = "a" if args.continue_run and file_exists else "w"
            with open(str(output_csv), mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == "w" or not file_exists:
                    writer.writeheader()
                writer.writerows(rows_out)
            print(f"💾 Saved → {output_csv}  ({len(rows_out)} rows)", flush=True)

    print("\n✅ Done.", flush=True)


if __name__ == "__main__":
    main()
