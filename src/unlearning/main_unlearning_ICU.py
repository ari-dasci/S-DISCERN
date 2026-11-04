"""
Incremental Class Unlearning (ICU) main script.

Given an ordered list of classes to forget, ICU removes them one at a time:
  Step 1: forget class_list[0]  → retain = all \ {class_list[0]}
  Step 2: forget class_list[1]  → retain = all \ {class_list[0], class_list[1]}
  …
  Step N: forget class_list[N-1] → retain = all \ {class_list[0], …, class_list[N-1]}

After each step the model produced by the unlearning method (best or epoch_00
checkpoint, depending on the method) becomes the starting point for the next
step.  Parameters (Fisher, MAS, distributional stats…) are computed **only
once** from the original pre-trained model and reused throughout.

Instead of saving every intermediate model, we evaluate directly on the test
set after each step and append a row to a CSV with columns:

    experiment, id_database, version, steps,
    accuracy_score_after, unlearning_accuracy_score,
    maxprobability, maxprobability_forget
"""

import os
import csv
import copy
import fcntl
import math

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as tvt

import src.methods as methods
import src.config as cfg_module
from src.methods.distribution_adapter import DistributionAdapter
from src.utils.load_data import load_data_class, _load_raw_dataset, number_dataset_classes
from src.models.ClassificationModel import ClassificationModel
from src.models.transforms import load_transforms
from src.models.optimizer import load_optimizer
from src.utils.dataset_loader import ClassificationDataset
from src.utils.cluster_metrics import (
    NEAR_DATASET_MAP,
    evaluate_and_extract,
    cross_metrics,
    retain_metrics,
)
from torch.utils.data import DataLoader

# ── Method / config factories (same as main_unlearning.py) ───────────
method_class_dictionary = {
    'EWC': lambda model, optimizer, cfg: methods.EWCMethod(model, optimizer, cfg),
    'DELETE': lambda model, optimizer, cfg: methods.DELETEMethod(model, optimizer, cfg),
    'NOISY': lambda model, optimizer, cfg: methods.NOISYMethod(model, optimizer, cfg),
    'SSD': lambda model, optimizer, cfg: methods.SSDMethod(model, optimizer, cfg),
    'RETRAIN': lambda model, optimizer, cfg: methods.RETRAINMethod(model, optimizer, cfg),
    'DISTRIBUTIONAL': lambda model, optimizer, cfg: methods.DistributionalMethod(model, optimizer, cfg),
    'PARAMETRIC': lambda model, optimizer, cfg: methods.ParametricMethod(model, optimizer, cfg),
    'BADTEACHER': lambda model, optimizer, cfg: methods.BadTeacherMethod(model, optimizer, cfg),
    'BOUNDARYSHRINK': lambda model, optimizer, cfg: methods.BoundaryShrinkMethod(model, optimizer, cfg),
    'BOUNDARYEXPAND': lambda model, optimizer, cfg: methods.BoundaryExpandMethod(model, optimizer, cfg),
    'SFU': lambda model, optimizer, cfg: methods.SFUMethod(model, optimizer, cfg),
    'SCRUB': lambda model, optimizer, cfg: methods.SCRUBMethod(model, optimizer, cfg),
    'RANDOMLABEL': lambda model, optimizer, cfg: methods.RandomLabelMethod(model, optimizer, cfg),
    'FISHERFORGETTING': lambda model, optimizer, cfg: methods.FisherForgettingMethod(model, optimizer, cfg),
    'UNSIR': lambda model, optimizer, cfg: methods.UNSIRMethod(model, optimizer, cfg),
    'ISPF': lambda model, optimizer, cfg: methods.ISPFMethod(model, optimizer, cfg),
    'ZSPAG': lambda model, optimizer, cfg: methods.ZSPAGMethod(model, optimizer, cfg),
}

method_config_dictionary = {
    'EWC': lambda args, cfg: cfg_module.EWCconfig(args, cfg),
    'DELETE': lambda args, cfg: cfg_module.DELETEconfig(args, cfg),
    'NOISY': lambda args, cfg: cfg_module.NOISYconfig(args, cfg),
    'SSD': lambda args, cfg: cfg_module.SSDconfig(args, cfg),
    'RETRAIN': lambda args, cfg: cfg_module.RETRAINconfig(args, cfg),
    'DISTRIBUTIONAL': lambda args, cfg: cfg_module.DISTRIBUTIONALconfig(args, cfg),
    'PARAMETRIC': lambda args, cfg: cfg_module.PARAMETRICconfig(args, cfg),
    'BADTEACHER': lambda args, cfg: cfg_module.BadTeacherconfig(args, cfg),
    'BOUNDARYSHRINK': lambda args, cfg: cfg_module.BoundaryShrinkconfig(args, cfg),
    'BOUNDARYEXPAND': lambda args, cfg: cfg_module.BoundaryExpandconfig(args, cfg),
    'SFU': lambda args, cfg: cfg_module.SFUconfig(args, cfg),
    'SCRUB': lambda args, cfg: cfg_module.SCRUBconfig(args, cfg),
    'RANDOMLABEL': lambda args, cfg: cfg_module.RandomLabelconfig(args, cfg),
    'FISHERFORGETTING': lambda args, cfg: cfg_module.FisherForgettingconfig(args, cfg),
    'UNSIR': lambda args, cfg: cfg_module.UNSIRconfig(args, cfg),
    'ISPF': lambda args, cfg: cfg_module.ISPFconfig(args, cfg),
    'ZSPAG': lambda args, cfg: cfg_module.ZSPAGconfig(args, cfg),
}

# One-shot methods save checkpoint_epoch_00.pt instead of checkpoint_best.pt
ONE_SHOT_METHODS = {'SSD', 'FISHERFORGETTING', 'SFU'}


# ── Logit metric helpers ─────────────────────────────────────────────

def _softmax_np(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def _mean_prob_excl_max(probs):
    n = probs.shape[1]
    max_prob = probs[np.arange(len(probs)), np.argmax(probs, axis=1)]
    return float(np.mean((1.0 - max_prob) / (n - 1)))


def _logit_stats(logits, suffix):
    s = np.sort(logits, axis=1)[:, ::-1]
    return {
        f"max_logit_{suffix}":        float(np.mean(s[:, 0])),
        f"mean_logits_{suffix}":      float(np.mean(logits)),
        f"min_logit_{suffix}":        float(np.mean(s[:, -1])),
        f"max_second_logit_{suffix}": float(np.mean(s[:, 1])),
    }


def compute_logit_metrics(logits_retain, logits_forget, retain_indices):
    """Return dict with all logit-based and restricted-softmax metrics."""
    result = {}
    result.update(_logit_stats(logits_retain, "retain"))

    if logits_forget is not None and len(logits_forget) > 0:
        result.update(_logit_stats(logits_forget, "forget"))
    else:
        result.update({
            "max_logit_forget": 0.0, "mean_logits_forget": 0.0,
            "min_logit_forget": 0.0, "max_second_logit_forget": 0.0,
        })

    # Retain set — full softmax
    pr = _softmax_np(logits_retain)
    result["mean_prob_no_max"] = _mean_prob_excl_max(pr)

    # Retain set — retain-restricted softmax
    pr_sub = _softmax_np(logits_retain[:, retain_indices])
    result["max_probability_with_retain"]   = float(np.mean(np.max(pr_sub, axis=1)))
    result["mean_prob_no_max_with_retain"]  = _mean_prob_excl_max(pr_sub)

    # Forget set
    if logits_forget is not None and len(logits_forget) > 0:
        pf = _softmax_np(logits_forget)
        result["mean_prob_no_max_forget"] = _mean_prob_excl_max(pf)

        pf_sub = _softmax_np(logits_forget[:, retain_indices])
        result["max_probability_forget_with_retain"]   = float(np.mean(np.max(pf_sub, axis=1)))
        result["mean_prob_no_max_forget_with_retain"]  = _mean_prob_excl_max(pf_sub)
    else:
        result["mean_prob_no_max_forget"]              = 0.0
        result["max_probability_forget_with_retain"]   = 0.0
        result["mean_prob_no_max_forget_with_retain"]  = 0.0

    return result


# ── Evaluation helpers ───────────────────────────────────────────────

# evaluate_model removed — replaced by evaluate_and_extract from src.utils.cluster_metrics,
# which returns (accuracy, mean_max_prob, logits_np, embs_np, labels_np) in a single pass.


def _existing_steps_for_experiment(csv_path, experiment, id_database, version):
    """Return the set of `steps` values already recorded in csv_path for this
    (experiment, id_database, version) combination, or an empty set if the
    CSV doesn't exist yet."""
    if not os.path.exists(csv_path):
        return set()
    steps = set()
    with open(csv_path, "r", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("experiment") == experiment
                        and row.get("id_database") == id_database
                        and row.get("version") == version):
                    try:
                        steps.add(int(row["steps"]))
                    except (TypeError, ValueError):
                        pass
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return steps


def _find_step_checkpoint(icu_ckpt_dir, method_name):
    """Checkpoint a finished ICU step left in icu_ckpt_dir (same preference
    order as the main loop: best / epoch_00 for one-shot methods, then
    epoch_00, then the last checkpoint_*.pt), or None."""
    if not os.path.isdir(icu_ckpt_dir):
        return None
    first = "checkpoint_epoch_00.pt" if method_name in ONE_SHOT_METHODS else "checkpoint_best.pt"
    for name in (first, "checkpoint_epoch_00.pt"):
        path = os.path.join(icu_ckpt_dir, name)
        if os.path.exists(path):
            return path
    existing = sorted(f for f in os.listdir(icu_ckpt_dir)
                      if f.startswith("checkpoint_") and f.endswith(".pt"))
    return os.path.join(icu_ckpt_dir, existing[-1]) if existing else None


def build_test_loaders(args, forgotten_classes, transform_test, batch_size):
    """Build retain-test and forget-test DataLoaders for the current step.

    retain_test  = test samples whose class ∉ forgotten_classes
    forget_test  = test samples whose class ∈ forgotten_classes
    """
    X_test, y_test = _load_raw_dataset(args.dataset, args.data_path, train=False)

    retain_x = [img for img, lbl in zip(X_test, y_test) if lbl not in forgotten_classes]
    retain_y = [lbl for lbl in y_test if lbl not in forgotten_classes]
    forget_x = [img for img, lbl in zip(X_test, y_test) if lbl in forgotten_classes]
    forget_y = [lbl for lbl in y_test if lbl in forgotten_classes]

    loaders = {}
    if retain_x:
        ds_r = ClassificationDataset(retain_x, retain_y, transforms=transform_test)
        loaders['retain'] = DataLoader(ds_r, batch_size=batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True)
    else:
        loaders['retain'] = None

    if forget_x:
        ds_f = ClassificationDataset(forget_x, forget_y, transforms=transform_test)
        loaders['forget'] = DataLoader(ds_f, batch_size=batch_size, shuffle=False,
                                        num_workers=4, pin_memory=True)
    else:
        loaders['forget'] = None

    return loaders


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = cfg_module.Parser()
    args = parser.parse_args()
    cfg = parser.general_config()

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes_to_forget = list(args.classes_to_remove)  # ordered list
    num_classes = number_dataset_classes(args.dataset)

    print(f"🔄 ICU: Incremental Class Unlearning")
    print(f"   Classes to forget (in order): {classes_to_forget}")
    print(f"   Method: {cfg['unlearning_method']}")
    print(f"   Dataset: {args.dataset} ({num_classes} classes)")

    # ── Build the id_database string (same format as the CSV) ────────
    classes_str = "_".join(str(c) for c in classes_to_forget)
    id_database = f"{args.dataset.lower()}_classes_to_remove_{classes_str}"

    # ── Build method-specific config once (provides checkpoint_path, etc.) ──
    cfg_method = method_config_dictionary[cfg['unlearning_method']](args, copy.deepcopy(cfg))
    # We will need the version string for the CSV
    method_name = cfg['unlearning_method']

    # Derive the "version" string that appears in the CSV from checkpoint_path
    # e.g. "EWC_10.0_lambda_MAS_freeze_confidence_penalty/original_best"
    # We extract the part after experiment_name/
    full_checkpoint_path = cfg_method.get('checkpoint_path', '')
    experiment_prefix = f"{cfg['experiment_name']}/"
    if experiment_prefix in full_checkpoint_path:
        version_str = full_checkpoint_path.split(experiment_prefix, 1)[1]
    else:
        version_str = f"{method_name}/{cfg.get('version_output', 'original_best')}"

    # ── Skip early if this experiment's results already exist ────────
    # results_ICU/{base_model}_{id_database}/{version_str}/unlearning_results_ICU.csv
    csv_dir = os.path.join("results_ICU", f"{args.base_model}_{id_database}", version_str)
    csv_path = os.path.join(csv_dir, "unlearning_results_ICU.csv")
    expected_steps = set(range(1, len(classes_to_forget) + 1))
    existing_steps = _existing_steps_for_experiment(
        csv_path, args.base_model, id_database, version_str
    )
    if expected_steps.issubset(existing_steps):
        print(f"⏭️  Results already exist in {csv_path} for "
              f"experiment={args.base_model}, id_database={id_database}, "
              f"version={version_str} (steps {sorted(expected_steps)}). Skipping the run.")
        return
    if existing_steps:
        # Partial run: the model of step k depends on steps 1..k-1, so every
        # step is (re)built in order, but steps already in the CSV are neither
        # evaluated nor written again (no duplicated rows). A step whose
        # checkpoint is still on disk is reused instead of retrained, unless
        # the ICU distribution adapter is on (its state is not checkpointed).
        print(f"🔁 Resuming {csv_path}: steps already written {sorted(existing_steps)}, "
              f"missing {sorted(expected_steps - existing_steps)}")
    uses_icu_adapter = (
        method_name == 'DISTRIBUTIONAL'
        and getattr(args, 'icu_adapt_distributions', False)
    )

    # ── Load original pretrained model ───────────────────────────────
    epoch_value = str(cfg['epoch'])
    if epoch_value.lower() == 'best':
        ckpt_filename = 'checkpoint_best.pt'
    elif epoch_value.lower() == 'succed':
        ckpt_filename = 'checkpoint_succed.pt'
    else:
        ckpt_filename = f'checkpoint_epoch_{int(epoch_value):02}.pt'

    original_ckpt_path = os.path.join(
        cfg['output_base_dir'], cfg['original_name'], cfg['version'], ckpt_filename
    )

    # ── Transforms (test uses retain transform for evaluation) ───────
    _, transform_test, _ = load_transforms(args)

    # ── Load NEAR OOD test data (once, reused every step) ────────────
    near_ds_name = NEAR_DATASET_MAP.get(args.dataset.upper())
    if near_ds_name:
        print(f"   Loading NEAR OOD test dataset: {near_ds_name} ...")
        X_near_test, y_near_test = _load_raw_dataset(near_ds_name, args.data_path, train=False)
        _near_ds = ClassificationDataset(X_near_test, y_near_test, transforms=transform_test)
        near_test_loader = DataLoader(
            _near_ds, batch_size=cfg['val_batch_size'], shuffle=False,
            num_workers=4, pin_memory=True,
        )
        print(f"   NEAR test size: {len(X_near_test)}")
        del X_near_test
    else:
        near_test_loader = None
        print(f"⚠️  No NEAR mapping for {args.dataset}; NEAR cluster metrics will be 0.")

    # ── CSV output setup ─────────────────────────────────────────────
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = open(csv_path, "a", newline="")
    fcntl.flock(csv_file.fileno(), fcntl.LOCK_EX)
    # Check size (not os.path.exists) *after* acquiring the lock, so two
    # concurrent processes racing to create the same csv_path can't both
    # see "no header yet" and each write a duplicate header row.
    write_header = os.fstat(csv_file.fileno()).st_size == 0
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            "experiment", "id_database", "version", "steps",
            "accuracy_score_after", "unlearning_accuracy_score",
            "maxprobability", "maxprobability_forget",
            "maxprobability_unknown",
            "cohesion_forget_l2", "distance_inter_class_l2",
            "cohesion_forget_cos", "distance_inter_class_cos",
            "cohesion_retain_l2", "distance_inter_retain_l2",
            "cohesion_retain_cos", "distance_inter_retain_cos",
            "cohesion_unknown_l2", "distance_inter_unknown_l2",
            "cohesion_unknown_cos", "distance_inter_unknown_cos",
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
        ])
        csv_file.flush()
    fcntl.flock(csv_file.fileno(), fcntl.LOCK_UN)

    # ── Pre-compute importance parameters from the ORIGINAL model ────
    # We load the original model once to compute Fisher / MAS / distributional
    # stats and keep them for all steps.

    # Load all training data (all classes) for parameter computation
    x_train_all, y_train_all = _load_raw_dataset(args.dataset, args.data_path, train=True)
    transform_train, transform_retain, transform_forget = load_transforms(args)

    # We will store the precomputed coefficients / distribution info
    precomputed = {}



    # ── Incremental loop ─────────────────────────────────────────────
    forgotten_so_far = []  # classes already forgotten (discarded)
    current_ckpt_path = original_ckpt_path  # path to the model for this step
    adapted_dist_infos = None  # carries adapted distributions from step N to step N+1

    for step_idx, class_to_forget in enumerate(classes_to_forget, start=1):
        print(f"\n{'='*60}")
        print(f"  ICU Step {step_idx}/{len(classes_to_forget)}: "
              f"Forgetting class {class_to_forget}")
        print(f"  Already forgotten: {forgotten_so_far}")
        print(f"{'='*60}\n")

        already_logged = step_idx in existing_steps
        if already_logged and not uses_icu_adapter:
            reuse_ckpt = _find_step_checkpoint(
                os.path.join(cfg['output_base_dir'], cfg['experiment_name'],
                             f"ICU_step_{step_idx}", version_str),
                method_name,
            )
            if reuse_ckpt is not None:
                print(f"  ⏩ Step {step_idx} already in the CSV and with a checkpoint: reusing {reuse_ckpt}")
                forgotten_so_far.append(class_to_forget)
                current_ckpt_path = reuse_ckpt
                continue
        if already_logged:
            print(f"  🔁 Step {step_idx} already in the CSV but without a checkpoint: retraining "
                  f"(no evaluation, nothing written) to rebuild the model for the next steps")

        # Current forbidden = already forgotten + current class to forget
        current_forbidden = forgotten_so_far + [class_to_forget]

        # ── Load training data for this step ─────────────────────────
        # retain = all classes except current_forbidden
        # forget = only the current class to forget (NOT the already forgotten ones)
        retain_x = [img for img, lbl in zip(x_train_all, y_train_all) if lbl not in current_forbidden]
        retain_y = [lbl for lbl in y_train_all if lbl not in current_forbidden]
        forget_x = [img for img, lbl in zip(x_train_all, y_train_all) if lbl == class_to_forget]
        forget_y = [lbl for lbl in y_train_all if lbl == class_to_forget]
        # train = retain + forget (all except already-discarded)
        train_x = [img for img, lbl in zip(x_train_all, y_train_all) if lbl not in forgotten_so_far]
        train_y = [lbl for lbl in y_train_all if lbl not in forgotten_so_far]

        print(f"  Retain samples: {len(retain_x)}, Forget samples: {len(forget_x)}, "
              f"Train (retain+forget): {len(train_x)}")

        # ── Create a fresh model and load the current checkpoint ─────
        model = ClassificationModel(
            args.base_model, num_classes, cfg,
            classes_weights=None, extra_layers=args.extra_layer,
        )
        ckpt = torch.load(current_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)
        model.to(device)
        print(f"  ✅ Model loaded from {current_ckpt_path}")

        # ── Build method config for this step ────────────────────────
        step_cfg = copy.deepcopy(cfg)
        # Override forbidden_labels for this step
        step_cfg['forbidden_labels'] = current_forbidden

        # Override classes_to_remove in args for config builders that read it
        step_args = copy.deepcopy(args)
        step_args.classes_to_remove = current_forbidden

        step_cfg = method_config_dictionary[method_name](step_args, step_cfg)
        step_cfg['forbidden_labels'] = current_forbidden

        # Override checkpoint_path to a temporary ICU-specific path
        icu_ckpt_dir = os.path.join(
            cfg['output_base_dir'], cfg['experiment_name'],
            f"ICU_step_{step_idx}", version_str,
        )
        step_cfg['checkpoint_path'] = icu_ckpt_dir
        step_cfg['hyperparameters_path'] = icu_ckpt_dir

        # ── Create optimizer and method ──────────────────────────────
        optimizer = load_optimizer(model, method=method_name, config=step_cfg)
        method_instance = method_class_dictionary[method_name](model, optimizer, step_cfg)

        # ── Setup + Unlearn ──────────────────────────────────────────
        # Distributions adapted during the previous ICU step (if any) are
        # passed into setup() itself so they are already in place *before*
        # the logit-profile reference and fixed synthetic samples are built
        # from self.dist_infos below — using the on-disk (original, stale)
        # cache for those would defeat the point of adapting them.
        setup_kwargs = {}
        if method_name == 'DISTRIBUTIONAL':
            setup_kwargs['adapted_dist_infos'] = adapted_dist_infos

        train_X, train_y, retain_X, retain_y, forget_X, forget_y = method_instance.setup(
            train_X=train_x, train_y=train_y,
            retain_X=retain_x, retain_y=retain_y,
            forget_X=forget_x, forget_y=forget_y,
            transform_train=transform_train,
            transform_retain=transform_retain,
            transform_forget=transform_forget,
            **setup_kwargs,
        )

        # ── Build ICU distribution adapter (DISTRIBUTIONAL + embedding unfrozen) ─
        icu_adapter = None
        if (
            method_name == 'DISTRIBUTIONAL'
            and step_cfg.get('icu_adapt_distributions', False)
            and step_cfg.get('embunfreeze', 'freeze') != 'freeze'
        ):
            forget_loader_adapt = DataLoader(
                ClassificationDataset(forget_x, forget_y, transforms=transform_forget),
                batch_size=step_cfg['val_batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            extra_loaders_adapt = []
            extra_ds_names = step_cfg.get('icu_adapt_extra_datasets', [])
            extra_max = step_cfg.get('icu_adapt_extra_max_samples', 1000)
            _GRAYSCALE_DATASETS = {"MNIST", "LETTERS"}
            for extra_ds_name in extra_ds_names:
                try:
                    x_extra, y_extra = _load_raw_dataset(extra_ds_name, args.data_path, train=True)
                    n_sub = min(extra_max, len(x_extra))
                    idxs = np.random.choice(len(x_extra), n_sub, replace=False)
                    x_sub = [x_extra[i] for i in idxs]
                    y_sub = [y_extra[i] for i in idxs]
                    if extra_ds_name.upper() in _GRAYSCALE_DATASETS:
                        extra_transform = tvt.Compose([tvt.Grayscale(num_output_channels=3), transform_retain])
                    else:
                        extra_transform = transform_retain
                    extra_loaders_adapt.append(DataLoader(
                        ClassificationDataset(x_sub, y_sub, transforms=extra_transform),
                        batch_size=step_cfg['val_batch_size'],
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                    ))
                    print(f"  [ICU adapter] Extra dataset '{extra_ds_name}': "
                          f"{n_sub}/{len(x_extra)} samples sampled.", flush=True)
                except Exception as exc:
                    print(f"  [ICU adapter] Warning: could not load extra dataset "
                          f"'{extra_ds_name}': {exc}", flush=True)
            icu_adapter = DistributionAdapter(
                forbidden_labels=current_forbidden,
                device=device,
                n_samples=step_cfg.get('icu_adapt_n_samples', 500),
                ridge_alpha=step_cfg.get('icu_adapt_ridge_alpha', 1e-3),
                pca_variance=step_cfg.get('icu_adapt_pca_variance', None),
                transforms_param=step_cfg.get('icu_adapt_transforms_param', False),
                current_forget_labels=[class_to_forget],
                adapt_current_forget=step_cfg.get('icu_adapt_current_forget', False),
            )
            icu_adapter.initialize(
                model, forget_loader_adapt, method_instance.dist_infos,
                extra_loaders=extra_loaders_adapt or None,
            )
        elif (
            method_name == 'DISTRIBUTIONAL'
            and step_cfg.get('icu_adapt_distributions', False)
            and step_cfg.get('embunfreeze', 'freeze') == 'freeze'
        ):
            print(
                "  [ICU adapter] Warning: icu_adapt_distributions=True but embunfreeze=freeze. "
                "Embeddings are frozen so adaptation is skipped.",
                flush=True,
            )

        unlearn_kwargs = dict(
            train_X=train_X, train_y=train_y,
            retain_X=retain_X, retain_y=retain_y,
            forget_X=forget_X, forget_y=forget_y,
            transform_train=transform_train,
            transform_retain=transform_retain,
            transform_forget=transform_forget,
        )
        if method_name == 'DISTRIBUTIONAL':
            unlearn_kwargs['icu_adapter'] = icu_adapter

        model_unlearned = method_instance.unlearn(**unlearn_kwargs)

        icu_ckpt_dir = step_cfg['checkpoint_path']
        icu_ckpt_dir = step_cfg['hyperparameters_path']

        # ── Determine which checkpoint the method saved ──────────────
        if method_name in ONE_SHOT_METHODS:
            step_ckpt = os.path.join(icu_ckpt_dir, "checkpoint_epoch_00.pt")
        else:
            step_ckpt = os.path.join(icu_ckpt_dir, "checkpoint_best.pt")

        # Fallback: if best doesn't exist, try epoch_00 or last saved epoch
        if not os.path.exists(step_ckpt):
            alt = os.path.join(icu_ckpt_dir, "checkpoint_epoch_00.pt")
            if os.path.exists(alt):
                step_ckpt = alt
            else:
                # Use whatever checkpoint files exist (pick the latest epoch)
                existing = sorted([
                    f for f in os.listdir(icu_ckpt_dir)
                    if f.startswith("checkpoint_") and f.endswith(".pt")
                ])
                if existing:
                    step_ckpt = os.path.join(icu_ckpt_dir, existing[-1])
                else:
                    print(f"⚠️  No checkpoint found in {icu_ckpt_dir}, using in-memory model")
                    step_ckpt = None

        # ── Reload model from saved checkpoint (authoritative weights) ──
        if step_ckpt and os.path.exists(step_ckpt):
            ckpt_data = torch.load(step_ckpt, map_location=device)
            model_unlearned.load_state_dict(
                ckpt_data['state_dict'] if 'state_dict' in ckpt_data else ckpt_data,
                strict=False,
            )
            print(f"  ✅ Model of step {step_idx} loaded from {step_ckpt}")

        # ── Final distribution alignment with the checkpoint model ───────
        # The training loop may have saved the *best* checkpoint (not the last
        # epoch). One extra adapter.step() ensures the distributions match the
        # weights that will actually be used next step.
        if icu_adapter is not None:
            print(f"  [ICU adapter] Final alignment with checkpoint model...", flush=True)
            final_dist, final_mtr = icu_adapter.step(model_unlearned)
            adapted_dist_infos = copy.deepcopy(final_dist)
            _drift_str = ""
            if not math.isnan(final_mtr['mean_angular_shift_deg']):
                _drift_str += (
                    f"  angular_shift=({final_mtr['mean_angular_shift_deg']:.3f}°"
                    f"±{final_mtr['std_angular_shift_deg']:.3f}° "
                    f"max={final_mtr['max_angular_shift_deg']:.3f}°)"
                    f"  kappa_ratio=({final_mtr['mean_kappa_ratio']:.3f}±{final_mtr['std_kappa_ratio']:.3f})"
                )
            if not math.isnan(final_mtr['mean_centroid_shift']):
                _drift_str += (
                    f"  shift=({final_mtr['mean_centroid_shift']:.3f}"
                    f"±{final_mtr['std_centroid_shift']:.3f} "
                    f"max={final_mtr['max_centroid_shift']:.3f})"
                    f"  std_ratio=({final_mtr['mean_std_ratio']:.3f}±{final_mtr['std_std_ratio']:.3f})"
                )
            _cond_str = (
                f"  σ_min={final_mtr['sigma_min']:.4f}  cond={final_mtr['condition_number']:.2f}"
                if not math.isnan(final_mtr['sigma_min']) else ""
            )
            print(
                f"  [ICU adapter] Checkpoint alignment: "
                f"‖ΔR‖_F={final_mtr['R_norm_diff']:.4f}  "
                f"‖R-I‖_F={final_mtr['R_drift']:.4f}  "
                f"‖t‖={final_mtr['t_norm']:.4f}  "
                f"disp={final_mtr['mean_displacement']:.4f}±{final_mtr['std_displacement']:.4f}"
                f"{_cond_str}  "
                f"ll_adapted={final_mtr['avg_ll_adapted']:.4f}  "
                f"ll_original={final_mtr['avg_ll_original']:.4f}"
                f"{_drift_str}",
                flush=True,
            )

        if already_logged:
            print(f"  ⏭️  Step {step_idx} already written in the CSV: not evaluated again")
        else:
            # ── Evaluate on test set + compute cluster metrics ───────────
            model_unlearned.to(device)
            model_unlearned.eval()

            test_loaders = build_test_loaders(
                args, set(current_forbidden), transform_test, cfg['val_batch_size']
            )

            retain_indices = [c for c in range(num_classes) if c not in current_forbidden]

            # Single-pass: accuracy + maxprob + logits + embeddings for clustering
            if test_loaders['retain'] is not None:
                acc_retain, maxprob_retain, lgts_r, embs_r, lbls_r = evaluate_and_extract(
                    model_unlearned, test_loaders['retain'], device
                )
            else:
                acc_retain, maxprob_retain, lgts_r, embs_r, lbls_r = 0.0, 0.0, None, None, None

            if test_loaders['forget'] is not None:
                acc_forget, maxprob_forget, lgts_f, embs_f, lbls_f = evaluate_and_extract(
                    model_unlearned, test_loaders['forget'], device
                )
            else:
                acc_forget, maxprob_forget, lgts_f, embs_f, lbls_f = 0.0, 0.0, None, None, None

            # Forget cluster metrics (forget vs retain)
            if embs_f is not None and embs_r is not None:
                cf_l2, if_l2, cf_cos, if_cos = cross_metrics(
                    embs_f, lbls_f, embs_r, lbls_r, device
                )
            else:
                cf_l2 = if_l2 = cf_cos = if_cos = 0.0

            # Retain cluster metrics (within retain)
            if embs_r is not None:
                cr_l2, ir_l2, cr_cos, ir_cos = retain_metrics(embs_r, lbls_r, device)
            else:
                cr_l2 = ir_l2 = cr_cos = ir_cos = 0.0

            # NEAR OOD cluster metrics (NEAR vs retain)
            if near_test_loader is not None and embs_r is not None:
                _, maxprob_unknown, _, embs_n, lbls_n = evaluate_and_extract(
                    model_unlearned, near_test_loader, device
                )
                cn_l2, in_l2, cn_cos, in_cos = cross_metrics(
                    embs_n, lbls_n, embs_r, lbls_r, device
                )
                del embs_n, lbls_n
            else:
                maxprob_unknown = cn_l2 = in_l2 = cn_cos = in_cos = 0.0

            # Logit metrics
            lm = compute_logit_metrics(lgts_r, lgts_f, retain_indices) if lgts_r is not None else {
                k: 0.0 for k in [
                    "max_logit_retain", "mean_logits_retain", "min_logit_retain", "max_second_logit_retain",
                    "max_logit_forget", "mean_logits_forget", "min_logit_forget", "max_second_logit_forget",
                    "max_probability_with_retain", "max_probability_forget_with_retain",
                    "mean_prob_no_max", "mean_prob_no_max_with_retain",
                    "mean_prob_no_max_forget", "mean_prob_no_max_forget_with_retain",
                ]
            }

            # Free embedding and logit arrays
            del embs_r, lbls_r, lgts_r
            if embs_f is not None:
                del embs_f, lbls_f, lgts_f
            torch.cuda.empty_cache()

            print(f"  📈 Step {step_idx} results:")
            print(f"     Acc retain:     {acc_retain:.6f}  |  Acc forget:    {acc_forget:.6f}")
            print(f"     MaxProb retain: {maxprob_retain:.6f}  |  MaxProb forget: {maxprob_forget:.6f}")
            print(f"     MaxProb NEAR:   {maxprob_unknown:.6f}")
            print(f"     Cohesion forget L2={cf_l2:.4f} cos={cf_cos:.4f}")
            print(f"     Cohesion retain L2={cr_l2:.4f} cos={cr_cos:.4f}")
            print(f"     Cohesion NEAR   L2={cn_l2:.4f} cos={cn_cos:.4f}")

            # ── Write CSV row ────────────────────────────────────────────
            csv_writer.writerow([
                args.base_model,   # experiment
                id_database,       # id_database
                version_str,       # version
                step_idx,          # steps
                acc_retain,        # accuracy_score_after
                acc_forget,        # unlearning_accuracy_score
                maxprob_retain,    # maxprobability
                maxprob_forget,    # maxprobability_forget
                maxprob_unknown,   # maxprobability_unknown
                cf_l2,             # cohesion_forget_l2
                if_l2,             # distance_inter_class_l2
                cf_cos,            # cohesion_forget_cos
                if_cos,            # distance_inter_class_cos
                cr_l2,             # cohesion_retain_l2
                ir_l2,             # distance_inter_retain_l2
                cr_cos,            # cohesion_retain_cos
                ir_cos,            # distance_inter_retain_cos
                cn_l2,             # cohesion_unknown_l2
                in_l2,             # distance_inter_unknown_l2
                cn_cos,            # cohesion_unknown_cos
                in_cos,            # distance_inter_unknown_cos
                lm["max_logit_retain"],        lm["mean_logits_retain"],
                lm["min_logit_retain"],        lm["max_second_logit_retain"],
                lm["max_logit_forget"],        lm["mean_logits_forget"],
                lm["min_logit_forget"],        lm["max_second_logit_forget"],
                lm["max_probability_with_retain"],
                lm["max_probability_forget_with_retain"],
                lm["mean_prob_no_max"],
                lm["mean_prob_no_max_with_retain"],
                lm["mean_prob_no_max_forget"],
                lm["mean_prob_no_max_forget_with_retain"],
            ])
            csv_file.flush()

        # ── Prepare for next step ────────────────────────────────────
        forgotten_so_far.append(class_to_forget)
        # The model from this step becomes the starting point for the next
        if step_ckpt and os.path.exists(step_ckpt):
            current_ckpt_path = step_ckpt
        else:
            # Save in-memory model so next step can load it
            fallback_path = os.path.join(icu_ckpt_dir, "checkpoint_icu_fallback.pt")
            os.makedirs(icu_ckpt_dir, exist_ok=True)
            torch.save({"state_dict": model_unlearned.state_dict()}, fallback_path)
            current_ckpt_path = fallback_path

        # Clean up GPU memory
        del model, model_unlearned, method_instance, optimizer
        torch.cuda.empty_cache()

    csv_file.close()
    print(f"\n✅ ICU done. Results saved to {csv_path}")
    print(f"   Total forgotten classes: {forgotten_so_far}")


if __name__ == "__main__":
    main()
