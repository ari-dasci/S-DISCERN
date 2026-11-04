"""
Shared GPU-accelerated embedding clustering metrics for unlearning evaluation.

All pairwise distance computations run entirely on the specified `device`
(full matrices built once, then sliced per-class to avoid repeated transfers).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax as scipy_softmax


# NEAR OOD dataset mapping (matches extract_NEAR.sh)
NEAR_DATASET_MAP = {
    "CIFAR10": "CIFAR100",
    "CIFAR100": "CIFAR10",
    "MNIST": "LETTERS",
}


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_extract(model, dataloader, device):
    """
    Single-pass inference: returns accuracy, mean max-prob, and raw arrays.

    Returns
    -------
    accuracy       : float
    mean_max_prob  : float
    logits_np      : np.ndarray  [N, C]
    embs_np        : np.ndarray  [N, D]
    labels_np      : np.ndarray  [N]
    """
    model.eval()
    lgs, embs, lbls = [], [], []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits, emb = model(images, return_feature=True)
        lgs.append(logits)
        embs.append(emb)
        lbls.append(labels)

    all_logits = torch.cat(lgs, dim=0)
    all_embs   = torch.cat(embs, dim=0)
    all_labels = torch.cat(lbls, dim=0)

    probs = torch.softmax(all_logits, dim=1)
    max_probs, preds = probs.max(dim=1)
    accuracy      = (preds == all_labels).float().mean().item()
    mean_maxprob  = max_probs.mean().item()

    return (
        accuracy, mean_maxprob,
        all_logits.cpu().numpy(),
        all_embs.cpu().numpy(),
        all_labels.cpu().numpy(),
    )


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Inference without accumulating accuracy.

    Returns (logits_np, embs_np, labels_np).
    """
    model.eval()
    lgs, embs, lbls = [], [], []
    for images, labels in dataloader:
        images = images.to(device).float()
        logits, emb = model(images, return_feature=True)
        lgs.append(logits.cpu())
        embs.append(emb.cpu())
        lbls.append(labels)
    return (
        torch.cat(lgs, dim=0).numpy(),
        torch.cat(embs, dim=0).numpy(),
        torch.cat(lbls, dim=0).numpy(),
    )


def mean_max_prob_from_logits(logits: np.ndarray) -> float:
    """Mean max-softmax-probability from raw logits (numpy)."""
    return float(np.mean(scipy_softmax(logits, axis=1).max(axis=1)))


# ── Clustering metrics ────────────────────────────────────────────────────────

def cross_metrics(
    embs_a: np.ndarray, labels_a: np.ndarray,
    embs_b: np.ndarray, labels_b: np.ndarray,
    device,
):
    """
    Intra-class cohesion of set A + nearest-class inter-set distance (A vs B).

    Builds the full A×B pairwise L2 and cosine matrices once on `device`,
    then slices per-class pair — avoids repeated host↔device transfers.

    Returns
    -------
    (coh_l2, inter_l2, coh_cos, inter_cos) : tuple[float, float, float, float]
    """
    unique_a = np.unique(labels_a)
    unique_b = np.unique(labels_b)
    if len(unique_a) == 0 or len(unique_b) == 0:
        return 0.0, 0.0, 0.0, 0.0

    ea  = torch.from_numpy(embs_a.astype(np.float32)).to(device)
    eb  = torch.from_numpy(embs_b.astype(np.float32)).to(device)
    ea_n = F.normalize(ea, dim=1)
    eb_n = F.normalize(eb, dim=1)

    l2_ab  = torch.cdist(ea, eb)        # [N_a, N_b]
    cos_ab = ea_n @ eb_n.T              # [N_a, N_b]

    b_masks = [
        torch.from_numpy(labels_b == lb).to(device) for lb in unique_b
    ]

    coh_l2 = coh_cos = inter_l2 = inter_cos = 0.0
    count = 0

    for la in unique_a:
        a_mask = torch.from_numpy(labels_a == la).to(device)
        n_a = int(a_mask.sum().item())
        if n_a < 2:
            continue

        sel   = ea[a_mask]
        sel_n = ea_n[a_mask]
        off   = ~torch.eye(n_a, dtype=torch.bool, device=device)

        coh_l2  += torch.cdist(sel, sel)[off].mean().item()
        coh_cos += (sel_n @ sel_n.T)[off].mean().item()

        lr = l2_ab[a_mask]
        cr = cos_ab[a_mask]
        best_cos, best_l2 = -float("inf"), float("inf")
        for bm in b_masks:
            if bm.sum() == 0:
                continue
            avg_l2  = lr[:, bm].mean().item()
            avg_cos = cr[:, bm].mean().item()
            if avg_cos > best_cos:
                best_cos = avg_cos
            if avg_l2 < best_l2:
                best_l2 = avg_l2

        inter_l2  += best_l2  if best_l2  < float("inf") else 0.0
        inter_cos += best_cos if best_cos > -float("inf") else 0.0
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    return coh_l2 / count, inter_l2 / count, coh_cos / count, inter_cos / count


def retain_metrics(embs: np.ndarray, labels: np.ndarray, device):
    """
    Intra-class cohesion + nearest-class inter-class distance within a single set.

    Builds the full N×N pairwise matrices once on `device`.

    Returns
    -------
    (coh_l2, inter_l2, coh_cos, inter_cos) : tuple[float, float, float, float]
    """
    unique = np.unique(labels)
    if len(unique) == 0:
        return 0.0, 0.0, 0.0, 0.0

    X   = torch.from_numpy(embs.astype(np.float32)).to(device)
    Xn  = F.normalize(X, dim=1)
    l2_all  = torch.cdist(X, X)    # [N, N]
    cos_all = Xn @ Xn.T            # [N, N]
    masks = [torch.from_numpy(labels == lbl).to(device) for lbl in unique]

    coh_l2 = coh_cos = inter_l2 = inter_cos = 0.0
    count = 0

    for i in range(len(unique)):
        mi  = masks[i]
        n_i = int(mi.sum().item())
        if n_i < 2:
            continue

        off = ~torch.eye(n_i, dtype=torch.bool, device=device)
        coh_l2  += l2_all[mi][:, mi][off].mean().item()
        coh_cos += cos_all[mi][:, mi][off].mean().item()

        lr = l2_all[mi]
        cr = cos_all[mi]
        best_cos, best_l2 = -float("inf"), float("inf")
        for j, mj in enumerate(masks):
            if j == i or mj.sum() == 0:
                continue
            avg_l2  = lr[:, mj].mean().item()
            avg_cos = cr[:, mj].mean().item()
            if avg_cos > best_cos:
                best_cos = avg_cos
            if avg_l2 < best_l2:
                best_l2 = avg_l2

        inter_l2  += best_l2  if best_l2  < float("inf") else 0.0
        inter_cos += best_cos if best_cos > -float("inf") else 0.0
        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    return coh_l2 / count, inter_l2 / count, coh_cos / count, inter_cos / count
