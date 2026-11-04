import re
import os
import sys
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from scipy.special import softmax

import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

architectures = ["vit_b_16", "resnet18", "efficientnet_v2_s", "swin_t"]
databases = ["cifar100", "cifar10", "mnist"]


def parse_experiment_string_v2(exp_str):
    parts = exp_str.split("_")

    architecture = None
    arch_idx = 0
    for arch in sorted(architectures, key=lambda x: -len(x)):
        arch_parts = arch.split("_")
        if parts[:len(arch_parts)] == arch_parts:
            architecture = arch
            arch_idx = len(arch_parts)
            break

    if not architecture or arch_idx >= len(parts):
        return None

    database_config = "_".join(parts[arch_idx:])

    return {
        "architecture": architecture,
        "database_config": database_config
    }


def extract_classes_count(id_database):
    if "_classes_to_remove_" not in id_database:
        return 0

    parts = id_database.split("_classes_to_remove_")
    if len(parts) < 2:
        return 0

    classes_part = parts[1]
    if classes_part == "":
        return 0

    classes = classes_part.split("_")
    valid_classes = [c for c in classes if c.isdigit()]
    return len(valid_classes)


def parse_epoch_path(path):
    parts = path.split(os.sep)

    if len(parts) < 4:
        return None

    experimento = parts[1]

    if parts[2].startswith("original"):
        metodo_version = parts[2]
        epoca = parts[3]
    else:
        metodo_version = f"{parts[2]}/{parts[3]}"
        epoca = parts[4]

    return {
        "experimento": experimento,
        "metodo_version": metodo_version,
        "epoca": epoca,
        "ruta_epoca": path
    }


def load_already_processed(output_path):
    if not os.path.exists(output_path):
        return set()

    try:
        df = pl.read_csv(output_path)
        required_cols = {"experiment", "id_database", "version", "epoch"}
        if not required_cols.issubset(set(df.columns)):
            print(f"⚠️  The existing file does not have every column needed by --continue. It will be ignored.", flush=True)
            return set()

        already_done = set(
            zip(
                df["experiment"].cast(pl.Utf8).to_list(),
                df["id_database"].cast(pl.Utf8).to_list(),
                df["version"].cast(pl.Utf8).to_list(),
                df["epoch"].cast(pl.Utf8).to_list(),
            )
        )
        print(f"📂 --continue: {len(already_done)} already processed experiments loaded from '{output_path}'.", flush=True)
        return already_done

    except Exception as e:
        print(f"⚠️  Could not read the file for --continue: {e}. It will be rewritten from scratch.", flush=True)
        return None


def open_output_file(output_path, append, continue_run):
    header = (
        "experiment,id_database,version,epoch,"
        "maxprobability_retain,"
        "cohesion_retain_l2,distance_inter_retain_l2,"
        "cohesion_retain_cos,distance_inter_retain_cos\n"
    )

    file_exists = os.path.exists(output_path)
    if (append or continue_run) and file_exists:
        f = open(output_path, "a")
    else:
        f = open(output_path, "w")
        f.write(header)

    return f


def write_row_online(f, experiment, id_database, version, epoch,
                     maxprob, coh_l2, dist_l2, coh_cos, dist_cos):
    row = (
        f"{experiment},{id_database},{version},{epoch},"
        f"{maxprob},{coh_l2},{dist_l2},{coh_cos},{dist_cos}\n"
    )
    f.write(row)
    f.flush()


def compute_retain_clustering(embs: np.ndarray, labels: np.ndarray) -> tuple:
    """
    GPU-accelerated intra-class cohesion and inter-class distance for retain embeddings.
    Computes the full N×N distance matrix once and reuses it for all class pairs.
    Returns: (coh_l2, inter_l2, coh_cos, inter_cos)
    """
    unique_labels = np.unique(labels)

    X = torch.from_numpy(embs.astype(np.float32)).to(DEVICE)
    X_norm = F.normalize(X, dim=1)

    # Single-shot full pairwise matrices
    l2_all = torch.cdist(X, X)      # [N, N]
    cos_all = X_norm @ X_norm.T     # [N, N]

    masks = [torch.from_numpy(labels == lbl).to(DEVICE) for lbl in unique_labels]

    coh_l2_sum = coh_cos_sum = inter_l2_sum = inter_cos_sum = 0.0
    count = 0

    for i, lbl in enumerate(unique_labels):
        mask_i = masks[i]
        n_i = mask_i.sum().item()
        if n_i < 2:
            continue

        # Intra-class cohesion
        off = ~torch.eye(n_i, dtype=torch.bool, device=DEVICE)
        coh_l2_sum += l2_all[mask_i][:, mask_i][off].mean().item()
        coh_cos_sum += cos_all[mask_i][:, mask_i][off].mean().item()

        # Inter-class: find nearest other class (min L2, max cosine)
        l2_row = l2_all[mask_i]    # [n_i, N]
        cos_row = cos_all[mask_i]  # [n_i, N]

        nearest_cos = -float('inf')
        nearest_l2 = float('inf')
        for j, _ in enumerate(unique_labels):
            if j == i:
                continue
            mask_j = masks[j]
            if mask_j.sum() == 0:
                continue
            avg_l2 = l2_row[:, mask_j].mean().item()
            avg_cos = cos_row[:, mask_j].mean().item()
            if avg_cos > nearest_cos:
                nearest_cos = avg_cos
            if avg_l2 < nearest_l2:
                nearest_l2 = avg_l2

        inter_l2_sum += nearest_l2 if nearest_l2 < float('inf') else 0.0
        inter_cos_sum += nearest_cos if nearest_cos > -float('inf') else 0.0
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        coh_l2_sum / count,
        inter_l2_sum / count,
        coh_cos_sum / count,
        inter_cos_sum / count,
    )


def main(args):

    base_dir = args.base_dir
    output_path = f"{base_dir}/{args.results_file}.csv"

    exclude_versions = args.exclude_version
    include_versions = args.include_version
    exclude_models = args.exclude_model
    include_models = args.include_model
    exclude_datasets = args.exclude_dataset
    include_datasets = args.include_dataset
    classes_filter = args.classes_filter

    already_processed = set()
    if args.continue_run:
        result = load_already_processed(output_path)
        if result is None:
            print(f"❌ --continue: could not read '{output_path}'. Aborting to avoid data loss.", flush=True)
            sys.exit(1)
        already_processed = result

    online_file = None
    if args.online:
        online_file = open_output_file(output_path, args.append, args.continue_run)
        print(f"📝 --online: writing results in real time to '{output_path}'.", flush=True)

    experiment_list = []
    id_list = []
    version_list = []
    epoch_list = []
    maxprobability_list = []
    cohesions_l2 = []
    distances_l2 = []
    cohesions_cos = []
    distances_cos = []

    print(f"Using device: {DEVICE}", flush=True)

    for root, dirs, files in os.walk(base_dir):
        if re.match(r"(\d+|best|succed)$", os.path.basename(root)):
            info = parse_epoch_path(root)
            if not info:
                continue

            if 'best' not in info['epoca'] and '00' not in info['epoca']:
                continue

            path = os.path.join(root)
            if not os.path.exists(path):
                continue

            version = info["metodo_version"]
            epoch = info["epoca"]

            exp_info = parse_experiment_string_v2(info["experimento"])
            if exp_info is None:
                print(f"⚠️  Could not parse: {info['experimento']}", flush=True)
                continue
            experiment = exp_info["architecture"]
            id_database = exp_info["database_config"]

            if args.continue_run and (experiment, id_database, version, epoch) in already_processed:
                print(f"⏭️  --continue: skipping, already processed: {experiment} | {id_database} | {version} | {epoch}", flush=True)
                continue

            test_csv_path = os.path.join(path, "test.csv")
            if not os.path.exists(test_csv_path):
                print(f"⚠️  Falta test.csv en {path}", flush=True)
                continue

            if exclude_versions is not None and any(ex in version for ex in exclude_versions):
                continue
            if include_versions is not None and not any(inc in version for inc in include_versions):
                continue
            if exclude_models is not None and any(ex in experiment for ex in exclude_models):
                continue
            if include_models is not None and not any(inc in experiment for inc in include_models):
                continue
            if exclude_datasets is not None and any(ex in id_database for ex in exclude_datasets):
                continue
            if include_datasets is not None and not any(inc in id_database for inc in include_datasets):
                continue

            if classes_filter != "all":
                classes_count = extract_classes_count(id_database)
                if classes_filter == "single" and classes_count != 1:
                    continue
                elif classes_filter == "multiple" and classes_count <= 1:
                    continue

            print(f"🧩 Procesando {info['experimento']} | {info['metodo_version']} | {info['epoca']}", flush=True)

            try:
                df_test = pl.scan_csv(test_csv_path).select(
                    [pl.col("labels"), pl.col("^logits.*$"), pl.col("^embeddings.*$")]
                ).collect()
            except Exception as e:
                print(f"⚠️  Invalid CSV: {experiment} | {path} | {e}", flush=True)
                continue

            logits_test = df_test.select(pl.col("^logits.*$")).to_numpy()
            if logits_test is None or logits_test.size == 0:
                print(f"⚠️  No logits in {path}", flush=True)
                continue
            labels_test = df_test["labels"].to_numpy()
            embs_test = df_test.select(pl.col("^embeddings.*$")).to_numpy()
            del df_test

            maxprob = np.mean(softmax(logits_test / 15, axis=1).max(axis=1))
            del logits_test

            coh_l2, dist_l2, coh_cos, dist_cos = compute_retain_clustering(embs_test, labels_test)
            del embs_test, labels_test

            if args.online:
                write_row_online(online_file, experiment, id_database, version, epoch,
                                 maxprob, coh_l2, dist_l2, coh_cos, dist_cos)
                print(f"✅ Fila escrita: {experiment} | {id_database} | {version} | {epoch}", flush=True)
            else:
                experiment_list.append(experiment)
                id_list.append(id_database)
                version_list.append(version)
                epoch_list.append(epoch)
                maxprobability_list.append(maxprob)
                cohesions_l2.append(coh_l2)
                distances_l2.append(dist_l2)
                cohesions_cos.append(coh_cos)
                distances_cos.append(dist_cos)

    if args.online and online_file is not None:
        online_file.close()
        print(f"💾 File closed: '{output_path}'", flush=True)
        return

    df_results = pl.DataFrame(
        {
            "experiment": experiment_list,
            "id_database": id_list,
            "version": version_list,
            "epoch": epoch_list,
            "maxprobability_retain": maxprobability_list,
            "cohesion_retain_l2": cohesions_l2,
            "distance_inter_retain_l2": distances_l2,
            "cohesion_retain_cos": cohesions_cos,
            "distance_inter_retain_cos": distances_cos,
        }
    )

    if args.append and os.path.exists(output_path):
        with open(output_path, "a") as f:
            df_results.write_csv(f, include_header=False)
        print(f"📎 Appended to: {output_path}", flush=True)
    else:
        df_results.write_csv(output_path)
        print(f"💾 Results saved to: {output_path}", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes clustering metrics of the retain classes")

    parser.add_argument("--base-dir", type=str, default="unlearning_embs")
    parser.add_argument("--results-file", type=str, default="unlearning_results_retain")

    parser.add_argument("--include-version", nargs="+", default=None)
    parser.add_argument("--exclude-version", nargs="+", default=None)
    parser.add_argument("--include-model", nargs="+", default=None)
    parser.add_argument("--exclude-model", nargs="+", default=None)
    parser.add_argument("--include-dataset", nargs="+", default=None)
    parser.add_argument("--exclude-dataset", nargs="+", default=None)

    parser.add_argument("--classes-filter", type=str, choices=["all", "single", "multiple"], default="all")
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--online", action="store_true",
                        help="Write every row to disk immediately", default=False)
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Skip experiments already in the output file", default=False)

    args = parser.parse_args()
    main(args)
