import math
import warnings

from tqdm import tqdm
from pingouin import multivariate_normality
import torch
import numpy as np

from sklearn.covariance import MinCovDet, GraphicalLassoCV, LedoitWolf
from sklearn.cluster import KMeans


# ===================================================================
#  Covariance constraint helpers  (shared by NOISY & DISTRIBUTIONAL)
# ===================================================================

def _n_params_cov(d: int, covariance_type: str = "full",
                  n_pca_components: int = None) -> int:
    """Number of free parameters in the covariance under *covariance_type*."""
    if covariance_type == "full":
        return d * (d + 1) // 2
    elif covariance_type == "diagonal":
        return d
    elif covariance_type == "spherical":
        return 1
    elif covariance_type == "lasso":
        return d * (d + 1) // 2          # patched later by actual sparsity
    elif covariance_type == "ledoit_wolf":
        return d * (d + 1) // 2          # full-rank shrunk covariance
    elif covariance_type == "pca":
        k = n_pca_components if n_pca_components else max(1, int(math.sqrt(d)))
        return d * k - k * (k - 1) // 2 + 1
    return d * (d + 1) // 2


def _auto_select_n_components(eigenvalues: np.ndarray, n_samples: int,
                              d: int, target_variance: float = 0.95) -> int:
    """Pick the smallest *k* PCA components that capture ≥ *target_variance*
    fraction of total variance, capped to avoid over-fitting."""
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        return 1
    cumvar = np.cumsum(eigenvalues) / total_var
    k = int(np.searchsorted(cumvar, target_variance) + 1)
    k = min(k, max(1, n_samples // 5), max(1, d // 2))
    k = max(k, 1)
    return k


def _constrain_covariance(cov_raw: np.ndarray, covariance_type: str,
                          d: int, *, n_pca_components: int = None,
                          n_samples: int = None,
                          eps: float = 1e-6) -> np.ndarray:
    """
    Apply a structural constraint to a raw covariance / scatter matrix.

    Always returns a full *d × d* matrix so that downstream code
    (log-likelihood, sampling, Mahalanobis) works unchanged.

    Supported types
    ---------------
    full        – standard sample covariance (+ ridge).
    diagonal    – keep only the diagonal (independent variances).
    spherical   – single scalar variance, isotropic.
    lasso       – Graphical Lasso (L1-penalised precision matrix).
    ledoit_wolf – Ledoit-Wolf optimal shrinkage estimator.
    pca         – low-rank (top-k eigendirections) + isotropic noise.
    """
    # --- Sanitise NaN / Inf that may leak from float32 np.cov --------
    if not np.all(np.isfinite(cov_raw)):
        warnings.warn("_constrain_covariance: NaN/Inf detected in raw "
                      "covariance – falling back to eps·I")
        cov_raw = eps * np.eye(d)

    if covariance_type == "full":
        return cov_raw + eps * np.eye(d)

    elif covariance_type == "diagonal":
        diag = np.diag(cov_raw)
        diag = np.where(np.isfinite(diag), diag, eps)
        return np.diag(np.maximum(diag, eps))

    elif covariance_type == "spherical":
        avg_var = max(np.trace(cov_raw) / d, eps)
        return np.eye(d) * avg_var

    elif covariance_type == "lasso":
        try:
            cov_reg = cov_raw + eps * np.eye(d)
            gl = GraphicalLassoCV(cv=3, max_iter=200, assume_centered=False)
            n_synth = n_samples if n_samples and n_samples > d else max(d + 5, 50)
            rng = np.random.default_rng(42)
            try:
                L = np.linalg.cholesky(cov_reg)
                synth_data = rng.standard_normal((n_synth, d)) @ L.T
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov_reg)
                eigvals = np.maximum(eigvals, eps)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                synth_data = rng.standard_normal((n_synth, d)) @ L.T
            gl.fit(synth_data)
            return gl.covariance_
        except Exception:
            return np.diag(np.maximum(np.diag(cov_raw), eps))

    elif covariance_type == "ledoit_wolf":
        # Ledoit-Wolf shrinkage estimator: optimal linear shrinkage
        # between the sample covariance and a structured target (scaled identity).
        try:
            n_synth = n_samples if n_samples and n_samples > d else max(d + 5, 50)
            rng = np.random.default_rng(42)
            cov_reg = cov_raw + eps * np.eye(d)
            try:
                L = np.linalg.cholesky(cov_reg)
                synth_data = rng.standard_normal((n_synth, d)) @ L.T
            except np.linalg.LinAlgError:
                eigvals_lw, eigvecs_lw = np.linalg.eigh(cov_reg)
                eigvals_lw = np.maximum(eigvals_lw, eps)
                L = eigvecs_lw @ np.diag(np.sqrt(eigvals_lw))
                synth_data = rng.standard_normal((n_synth, d)) @ L.T
            lw = LedoitWolf(assume_centered=False)
            lw.fit(synth_data)
            return lw.covariance_
        except Exception:
            return np.diag(np.maximum(np.diag(cov_raw), eps))

    elif covariance_type == "pca":
        eigenvalues, eigenvectors = np.linalg.eigh(cov_raw)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.maximum(eigenvalues, eps)

        if n_pca_components is None:
            ns = n_samples if n_samples else d
            k = _auto_select_n_components(eigenvalues, ns, d)
        else:
            k = min(n_pca_components, d)
        k = max(k, 1)

        W = eigenvectors[:, :k]
        top_eigs = eigenvalues[:k]
        noise_var = float(np.mean(eigenvalues[k:])) if k < d else eps
        noise_var = max(noise_var, eps)
        return W @ np.diag(top_eigs - noise_var) @ W.T + noise_var * np.eye(d)

    # fallback
    return cov_raw + eps * np.eye(d)


def _n_params_cov_from_precision(precision: np.ndarray, eps: float = 1e-8) -> int:
    """Count effective free parameters from a sparse precision matrix."""
    d = precision.shape[0]
    count = 0
    for i in range(d):
        for j in range(i, d):
            if abs(precision[i, j]) > eps:
                count += 1
    return count

def compute_fisher(model, data_loader, path=None):

    if path is not None:
        fisher = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        return fisher

    fisher = {}
    model.eval()
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p)

    num_iter = 5
    for i in range(num_iter): 
        print(f"→ Step {i + 1}/{num_iter}")
        for batch in tqdm(data_loader, desc=f'Calculating FIM', position=0, leave=True):
            images, labels = batch
            data = images.to(model.device)
            data = data.float()
            model.zero_grad()

            output = model(data)
            loss =  model.criterion(output, labels.to(model.device))
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
            
    # Normalize Fisher Information Matrix
    dividir = len(data_loader.dataset) * num_iter
    for n in fisher:
        fisher[n] /= dividir
        if 'classifier' in n:
                fisher[n] += 1e-3

    return fisher


def compute_mas_coefficients(model, dataloader, path=None):

    model.eval()
    if path is not None:
        mas_coeffs = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        return mas_coeffs
    mas_coeffs = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    num_iter = 5
    for i in range(num_iter):
        print(f"→ Step {i + 1}/{num_iter}")
        for batch in tqdm(dataloader, desc=f'Calculating MAS coefficients', position=0, leave=True):
            images, _ = batch
            data = images.to(model.device)
            data = data.float()
            outputs = model(data)
            outputs_norm = outputs.norm(dim=1).sum()

            model.zero_grad()
            outputs_norm.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    mas_coeffs[name] += param.grad.data.abs()

    # Normalise by the number of samples
    num_samples = len(dataloader.dataset) * num_iter
    for name in mas_coeffs:
        mas_coeffs[name] /= num_samples

    return mas_coeffs


def calculate_centroids(model, dataloader, num_classes, robust=False, clustering=False, path=None,
                        covariance_type="full", n_pca_components=None):
    """
    Calculate the (robust) class centroids and per-class covariance matrices.

    Parameters
    ----------
    covariance_type : str
        Structure imposed on every covariance matrix.
        One of 'full', 'diagonal', 'spherical', 'lasso', 'ledoit_wolf', 'pca'.
    n_pca_components : int or None
        Number of PCA components when covariance_type='pca'.
        None → auto-select via 95 % explained variance.
    """

    if path is not None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        class_centroids = torch.load(f"{path}_centroids.pt", map_location=map_location)
        covariance_matrix = torch.load(f"{path}_covariance.pt", map_location=map_location)
        percentage_inter_class = torch.load(f"{path}_percentage.pt", map_location=map_location)
        number_samples_per_class = torch.load(f"{path}_num_samples.pt", map_location=map_location)
        if clustering:
            percentage_intra_class = torch.load(f"{path}_percentage_intra.pt", map_location=map_location)

            # percentages to tensor
            percentage_inter_class = torch.tensor(percentage_inter_class, dtype=torch.float32)
            percentage_intra_class = [torch.tensor(p, dtype=torch.float32) for p in percentage_intra_class]
            return class_centroids, covariance_matrix, percentage_inter_class, percentage_intra_class, number_samples_per_class
        else:
                
            percentage_inter_class = torch.tensor(percentage_inter_class, dtype=torch.float32)
            return class_centroids, covariance_matrix, percentage_inter_class, number_samples_per_class

    # Collect the embeddings of each class
    class_embeddings = {i: [] for i in range(num_classes)}
    class_centroids = {i: [] for i in range(num_classes)}
    covariance_matrix = {i: [] for i in range(num_classes)}
    total_samples = 0
    total_samples_per_class = [0 for _ in range(num_classes)]
    percentage_inter_class = [0.0 for _ in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating embeddings', position=0, leave=True):
            images, labels = batch

            total_samples += labels.size(0)
            for label in labels:
                total_samples_per_class[label.item()] += 1

            images = images.to(model.device).float()
            _, embeddings = model(images, True)
            for emb, label in zip(embeddings, labels):
                class_embeddings[label.item()].append(emb.cpu().to(torch.float64).numpy())

        if not clustering:
            number_samples_per_class = torch.tensor(total_samples_per_class, dtype=torch.float32).to(model.device)
            for i in tqdm(range(num_classes), desc='Calculating centroids', position=0, leave=True):
                embeddings = np.array(class_embeddings[i])
                if len(embeddings) == 0:
                    continue

                percentage_inter_class[i] = total_samples_per_class[i] / total_samples
                # Covariance regularisation
                eps = 1e-6
                n_samp = len(embeddings)
                d = embeddings.shape[1]

                # Cast to float64 to avoid precision loss in np.cov
                embeddings = embeddings.astype(np.float64)

                if n_samp <= 1:
                    # Not enough samples for covariance; use isotropic fallback
                    warnings.warn(f"Class {i}: only {n_samp} sample(s), "
                                  f"using eps·I as covariance")
                    centroid = torch.tensor(embeddings[0] if n_samp == 1
                                            else np.zeros(d),
                                            dtype=torch.float32).to(model.device)
                    class_centroids[i] = centroid
                    covariance_matrix[i] = torch.tensor(
                        eps * np.eye(d), dtype=torch.float32).to(model.device)
                elif not robust:
                    # Centroid + covariance standard
                    centroid = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32).to(model.device)
                    class_centroids[i] = centroid
                    covariance = np.cov(embeddings, rowvar=False)
                    # Apply covariance constraint (also adds ridge for PD)
                    covariance = _constrain_covariance(covariance, covariance_type, d,
                                                       n_pca_components=n_pca_components,
                                                       n_samples=n_samp, eps=eps)
                    covariance_matrix[i] = torch.tensor(covariance, dtype=torch.float32).to(model.device)
                else:
                    # Robust centroid + covariance (only if there are enough points)
                    n_dim = embeddings.shape[1]
                    if n_samp > n_dim:
                        mcd = MinCovDet(support_fraction=0.9, random_state=42).fit(embeddings)
                        centroid = torch.tensor(mcd.location_, dtype=torch.float32).to(model.device)
                        covmat = mcd.covariance_
                        covmat = _constrain_covariance(covmat, covariance_type, d,
                                                       n_pca_components=n_pca_components,
                                                       n_samples=n_samp, eps=eps)
                        class_centroids[i] = centroid
                        covariance_matrix[i] = torch.tensor(covmat, dtype=torch.float32).to(model.device)
                    else:
                        # fallback: sample mean + regularized cov
                        centroid = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32).to(model.device)
                        covariance = np.cov(embeddings, rowvar=False)
                        covariance = _constrain_covariance(covariance, covariance_type, d,
                                                           n_pca_components=n_pca_components,
                                                           n_samples=n_samp, eps=eps)
                        class_centroids[i] = centroid
                        covariance_matrix[i] = torch.tensor(covariance, dtype=torch.float32).to(model.device)


            # percentage_class as a float tensor
            print("Class distribution:", percentage_inter_class, flush=True)
            percentage_inter_class = torch.tensor(percentage_inter_class, dtype=torch.float32)

            return class_centroids, covariance_matrix, percentage_inter_class, number_samples_per_class

        else:
            # CLUSTERING: no reinicializamos total_samples ni total_samples_per_class
            temporal_points = {i: [] for i in range(num_classes)}
            percentage_intra_class = [[] for _ in range(num_classes)]
            number_samples_per_class = [[] for _ in range(num_classes)]
            for i in tqdm(range(num_classes), desc='Calculating k-clustering normality per class', position=0, leave=True):
                embeddings = np.array(class_embeddings[i])
                if len(embeddings) == 0:
                    continue

                percentage_inter_class[i] = total_samples_per_class[i] / total_samples

                # KMeans clustering to find centroids
                max_k = min(len(embeddings), max(1, int(np.sqrt(len(embeddings)))))  # sensible upper bound
                normality_found = False
                for kcentroids in range(1, max_k + 1):
                    print("Testing class", i, "with", kcentroids, "centroids", flush=True)
                    kmeans = KMeans(n_clusters=kcentroids, random_state=42).fit(embeddings)

                    cluster_points = []
                    cluster_normal = True
                    for j in range(kcentroids):
                        points = embeddings[kmeans.labels_ == j]
                        if len(points) == 0:
                            cluster_normal = False
                            break
                        # pingouin.multivariate_normality can return (stat, pval) or (stat, pval, normal)
                        try:
                            res = multivariate_normality(points, alpha=0.05)
                            if isinstance(res, (list, tuple)):
                                if len(res) >= 3:
                                    is_normal = bool(res[2])
                                else:
                                    is_normal = float(res[1]) > 0.05
                            else:
                                # unexpected type -> treat as non-normal to be safe
                                is_normal = False
                        except Exception:
                            is_normal = False

                        if not is_normal:
                            cluster_normal = False
                            break
                        cluster_points.append(points)

                    if cluster_normal:
                        print("Normality achieved for class", i, "with", kcentroids, "centroids", flush=True)
                        for j in range(kcentroids):
                            number_samples_per_class[i].append(len(cluster_points[j]))
                        temporal_points[i] = cluster_points
                        normality_found = True
                        break

                if not normality_found:
                    # fallback: single cluster
                    print("Normality not achieved for class", i, "with", kcentroids, "centroids", flush=True)
                    temporal_points[i] = [embeddings]
                    number_samples_per_class[i].append(len(embeddings))

            # ahora calculamos centroids/cov per cluster
            eps = 1e-6
            for i in tqdm(range(num_classes), desc='Calculating centroids', position=0, leave=True):
                for cluster_points in temporal_points[i]:
                    embeddings = np.array(cluster_points)
                    if embeddings.size == 0:
                        continue
                    n_samp = len(embeddings)
                    d = embeddings.shape[1]

                    # Cast to float64 to avoid precision loss in np.cov
                    embeddings = embeddings.astype(np.float64)

                    intra_frac = n_samp / max(1, total_samples_per_class[i])
                    percentage_intra_class[i].append(intra_frac)

                    if n_samp <= 1:
                        # Not enough samples for covariance; isotropic fallback
                        warnings.warn(f"Class {i} cluster: only {n_samp} "
                                      f"sample(s), using eps·I")
                        centroid = torch.tensor(
                            embeddings[0] if n_samp == 1 else np.zeros(d),
                            dtype=torch.float32).to(model.device)
                        class_centroids[i].append(centroid)
                        covariance_matrix[i].append(torch.tensor(
                            eps * np.eye(d),
                            dtype=torch.float32).to(model.device))
                    elif not robust:
                        centroid = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32).to(model.device)
                        covariance = np.cov(embeddings, rowvar=False)
                        covariance = _constrain_covariance(covariance, covariance_type, d,
                                                           n_pca_components=n_pca_components,
                                                           n_samples=n_samp, eps=eps)
                        class_centroids[i].append(centroid)
                        covariance_matrix[i].append(torch.tensor(covariance, dtype=torch.float32).to(model.device))
                    else:
                        n_dim = embeddings.shape[1]
                        if n_samp > n_dim:
                            mcd = MinCovDet(support_fraction=0.9, random_state=42).fit(embeddings)
                            centroid = torch.tensor(mcd.location_, dtype=torch.float32).to(model.device)
                            covmat = mcd.covariance_
                            covmat = _constrain_covariance(covmat, covariance_type, d,
                                                           n_pca_components=n_pca_components,
                                                           n_samples=n_samp, eps=eps)
                            class_centroids[i].append(centroid)
                            covariance_matrix[i].append(torch.tensor(covmat, dtype=torch.float32).to(model.device))
                        else:
                            centroid = torch.tensor(np.mean(embeddings, axis=0), dtype=torch.float32).to(model.device)
                            covariance = np.cov(embeddings, rowvar=False)
                            covariance = _constrain_covariance(covariance, covariance_type, d,
                                                               n_pca_components=n_pca_components,
                                                               n_samples=n_samp, eps=eps)
                            class_centroids[i].append(centroid)
                            covariance_matrix[i].append(torch.tensor(covariance, dtype=torch.float32).to(model.device))


            # percentages to tensors
            percentage_inter_class = torch.tensor(percentage_inter_class, dtype=torch.float32)
            percentage_intra_class = [torch.tensor(p, dtype=torch.float32) for p in percentage_intra_class]

            max_k = max((len(lst) for lst in number_samples_per_class), default=0)
            ns_tensor = torch.zeros((num_classes, max_k), dtype=torch.float32, device=model.device)
            for i, lst in enumerate(number_samples_per_class):
                for j, val in enumerate(lst):
                    ns_tensor[i, j] = float(val)
                    
            return class_centroids, covariance_matrix, percentage_inter_class, percentage_intra_class, ns_tensor


# ===================================================================
#  Checkpoint saving helper with early stopping
# ===================================================================

def should_save_checkpoint(epoch, acc_forget, num_classes, config, early_stop_state):
    """
    Determine if checkpoints should be saved and whether to trigger early stopping.

    Logic:
      1. Save ``checkpoint_succed.pt`` the **first** epoch where
         ``acc_forget <= 1/(num_classes - 2)``.
      2. Save ``checkpoint_best.pt`` whenever ``acc_forget`` is strictly lower
         than the best (lowest) forget accuracy observed so far.  Lower forget
         accuracy means better unlearning.
      3. **Early stopping**: if ``acc_forget <= 0.1`` for **two consecutive**
         epochs, signal training to stop.
      4. Regular epoch-based saves (``config['save_epoch']``) are emitted
         independently as ``checkpoint_epoch_XX.pt``.

    Args:
        epoch: Current epoch number.
        acc_forget: Current accuracy on the forget validation set.
        num_classes: Total number of classes.
        config: Config dict (must contain ``'save_epoch'``).
        early_stop_state: Mutable dict with keys:
            ``'succed_saved'`` (bool) – whether checkpoint_succed has been saved,
            ``'best_forget_acc'`` (float) – lowest forget accuracy seen so far
            (initialise to ``float('inf')``),
            ``'consecutive_low_count'`` (int) – number of consecutive epochs
            with ``acc_forget <= 0.1`` (initialise to ``0``).

    Returns:
        checkpoints_to_save: list of ``(filename, reason)`` tuples.
        should_stop: bool – ``True`` when training should stop.
    """
    checkpoints_to_save = []
    should_stop = False

    threshold_k_minus_2 = 1.0 / max(num_classes - 2, 1)

    # ── Regular periodic save ────────────────────────────────────────
    if epoch % config['save_epoch'] == (config['save_epoch'] - 1):
        checkpoints_to_save.append(
            (f"checkpoint_epoch_{epoch:02}.pt",
             f"Regular save (every {config['save_epoch']} epochs)"))

    # ── checkpoint_succed.pt: first epoch with acc <= 1/(k-2) ────────
    if acc_forget <= threshold_k_minus_2 and not early_stop_state['succed_saved']:
        checkpoints_to_save.append(
            ("checkpoint_succed.pt",
             f"First epoch with acc_forget ({acc_forget:.4f}) <= 1/(k-2)={threshold_k_minus_2:.4f}"))
        early_stop_state['succed_saved'] = True

    # ── checkpoint_best.pt: save whenever forget acc improves ────────
    if acc_forget < early_stop_state['best_forget_acc']:
        early_stop_state['best_forget_acc'] = acc_forget
        checkpoints_to_save.append(
            ("checkpoint_best.pt",
             f"New best forget acc: {acc_forget:.4f} (previous best: {early_stop_state.get('_prev_best', 'inf')})"))
        early_stop_state['_prev_best'] = f"{acc_forget:.4f}"

    # ── Early stopping: acc_forget <= 0.1 for 2 consecutive epochs ───
    if acc_forget <= 0.1:
        early_stop_state['consecutive_low_count'] += 1
        if early_stop_state['consecutive_low_count'] >= 2:
            should_stop = True
    else:
        early_stop_state['consecutive_low_count'] = 0

    return checkpoints_to_save, should_stop



