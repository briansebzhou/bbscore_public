"""
Cross-model comparison metric.

Compares representations from two models on the same stimulus set using
three complementary methods:
  - RSA  (Representational Similarity Analysis, Spearman on RDMs)
  - CKA  (Centered Kernel Alignment, linear and RBF kernels)
  - Ridge R² (linear predictivity in both directions)

Input to compute():
  source : (n_images, n_features_A)  -- already flattened by CrossModelBenchmark
  target : (n_images, n_features_B)

PCA, CKA, RDM computation, and ridge regression are performed on GPU when
available (PyTorch / CUDA).  Falls back to CPU NumPy if no GPU is present.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import BaseMetric


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        # Pick the GPU with the most free memory
        free = [
            torch.cuda.get_device_properties(i).total_memory
            - torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ]
        return torch.device(f"cuda:{int(np.argmax(free))}")
    return torch.device("cpu")


_RIDGE_ALPHAS: List[float] = [
    1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6
]
_RSA_METHODS = ["spearman", "corr"]


# ---------------------------------------------------------------------------
# GPU-accelerated PCA via kernel (Gram) trick
#
# For fat matrices (N << D), the full (N, D) SVD blows VRAM/cusolver limits.
# Instead we build the N×N Gram matrix K = Xc @ Xc.T row-by-row in chunks,
# eigen-decompose the small K on GPU, then recover scores as U * sqrt(λ).
# Peak VRAM = K (N×N×4B ≈ 2 GB for N=22k) + one chunk slice.
# ---------------------------------------------------------------------------

def _build_gram_chunked(
    X_cpu: torch.Tensor,         # (N, D) float32, stays on CPU
    X_mean_gpu: torch.Tensor,    # (1, D) float32, on GPU
    device: torch.device,
    chunk_rows: int,
) -> torch.Tensor:
    """Build K = Xc @ Xc.T (N×N) by streaming X through the GPU in chunks."""
    n = X_cpu.shape[0]
    K = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i in range(0, n, chunk_rows):
        i_end = min(i + chunk_rows, n)
        Xi = X_cpu[i:i_end].to(device, non_blocking=True) - X_mean_gpu
        for j in range(0, n, chunk_rows):
            j_end = min(j + chunk_rows, n)
            Xj = X_cpu[j:j_end].to(device, non_blocking=True) - X_mean_gpu
            K[i:i_end, j:j_end] = Xi @ Xj.T
            del Xj
        del Xi
    return K


def _auto_chunk_rows(n_features: int, n_samples: int, device: torch.device) -> int:
    """
    Choose chunk_rows so peak VRAM stays safe during _build_gram_chunked.

    Peak at each inner step = K (N×N) + left_tile + right_tile.
    All three are live simultaneously, so:
        budget_for_tiles = free_VRAM - K_bytes - safety
        each_tile ≤ budget_for_tiles / 2
        chunk_rows ≤ each_tile / (D × 4B)

    We also hard-cap each tile at 4 GB to avoid driver limits on large
    individual allocations (observed `invalid argument` at ~14 GB tiles).

    Called *after* torch.cuda.empty_cache() so mem_get_info is accurate.
    """
    if device.type != "cuda":
        return 2000
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        free_vram, total_vram = torch.cuda.mem_get_info(device.index or 0)
        k_bytes = n_samples * n_samples * 4          # K matrix (live during chunking)
        safety = 6 * (1 << 30)                       # 6 GB safety margin (increased from 4 GB)
        tile_cap = 1.5 * (1 << 30) if total_vram < 20 * (1 << 30) else 4 * (1 << 30)
        tile_budget = max(free_vram - k_bytes - safety, 128 * (1 << 20))
        tile_budget = min(tile_budget / 2, tile_cap)  # cap by GPU size (1.5 GB for 16 GB cards)
        rows = int(tile_budget / (n_features * 4))
        capped = max(32, min(rows, n_samples))
        print(f"    [GPU PCA] free={free_vram/1e9:.1f}/{total_vram/1e9:.1f} GB, "
              f"K={k_bytes/1e9:.1f} GB, tile_cap={tile_budget/1e9:.1f} GB, "
              f"chunk_rows={capped} (one_tile={capped*n_features*4/1e9:.2f} GB)")
        return capped
    except Exception:
        return 64


class _PCAProjector:
    """
    Holds the fitted PCA basis (feature-space components) so that a test set
    can be projected onto the same subspace as the training set.

    Attributes
    ----------
    mean_ : np.ndarray  (1, D)   — training mean
    components_ : np.ndarray  (D, k) — feature-space principal axes (unit vectors)
    scale_ : np.ndarray  (k,)  — sqrt(eigenvalues), for un-scaled scores if needed
    """
    def __init__(self, mean_: np.ndarray, components_: np.ndarray, scale_: np.ndarray):
        self.mean_ = mean_              # (1, D)
        self.components_ = components_  # (D, k)
        self.scale_ = scale_            # (k,)

    def transform(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """Project new data X (M, D) onto the fitted PCA basis chunked on GPU."""
        X = np.asarray(X, dtype=np.float32)
        M, D = X.shape
        k = self.components_.shape[1]
        
        # Use smaller chunks for projection to keep VRAM usage low
        chunk = max(32, min(256, M))

        comp_gpu = torch.from_numpy(self.components_).to(device)   # (D, k)
        mean_gpu = torch.from_numpy(self.mean_).to(device)          # (1, D)
        out = np.empty((M, k), dtype=np.float32)
        for i in range(0, M, chunk):
            j = min(i + chunk, M)
            Xc = torch.from_numpy(X[i:j]).to(device) - mean_gpu    # (c, D)
            out[i:j] = (Xc @ comp_gpu).cpu().numpy()               # (c, k)
            del Xc
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del comp_gpu, mean_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return out


def _fit_reduce_gpu(
    X: np.ndarray, n_components: int, device: torch.device
) -> tuple:
    """
    Fit kernel PCA on X, return (scores, projector).

    scores     : (N, k) float32 numpy — PCA scores for X
    projector  : _PCAProjector — can transform new data onto the same basis

    Falls back to sklearn randomized PCA on CPU if anything goes wrong;
    projector.transform() still works correctly in that case.
    """
    n_samples, n_feats = X.shape
    k = min(n_components, n_feats, n_samples - 1)
    if k >= n_feats:
        arr = X.astype(np.float32)
        mean = arr.mean(axis=0, keepdims=True)
        projector = _PCAProjector(mean, np.eye(n_feats, dtype=np.float32), np.ones(n_feats, np.float32))
        return arr, projector

    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

        X_cpu = torch.from_numpy(X.astype(np.float32))
        X_mean_cpu = X_cpu.mean(0, keepdim=True)           # (1, D) on CPU
        X_mean_gpu = X_mean_cpu.to(device)

        chunk_rows = _auto_chunk_rows(n_feats, n_samples, device)
        print(f"    [GPU PCA] N={n_samples}, D={n_feats}, k={k}, chunk_rows={chunk_rows}")

        K = _build_gram_chunked(X_cpu, X_mean_gpu, device, chunk_rows)
        K = (K + K.T) / 2.0

        eigenvalues, eigenvectors = torch.linalg.eigh(K)   # ascending
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        lam_k = eigenvalues[:k].clamp(min=1e-10)
        U_k = eigenvectors[:, :k]                           # (N, k) on GPU

        # PCA scores for training data
        sqrt_lam = lam_k.sqrt()
        scores = (U_k * sqrt_lam).cpu().numpy().astype(np.float32)   # (N, k)

        # Feature-space components V_k = (Xc.T @ U_k) / sqrt(λ_k)
        # Computed chunked to avoid sending all of X to GPU at once
        V_k = torch.zeros((n_feats, k), dtype=torch.float32, device=device)
        for i in range(0, n_samples, chunk_rows):
            j = min(i + chunk_rows, n_samples)
            Xc_chunk = X_cpu[i:j].to(device) - X_mean_gpu   # (c, D)
            V_k += Xc_chunk.T @ U_k[i:j]                    # (D, k)
            del Xc_chunk
        V_k = V_k / sqrt_lam.unsqueeze(0)                   # (D, k)

        projector = _PCAProjector(
            mean_=X_mean_cpu.numpy().astype(np.float32),
            components_=V_k.cpu().numpy().astype(np.float32),
            scale_=sqrt_lam.cpu().numpy().astype(np.float32),
        )

        del K, eigenvalues, eigenvectors, lam_k, U_k, sqrt_lam, V_k, X_mean_gpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return scores, projector

    except Exception as e:
        print(f"  [GPU PCA] Falling back to CPU sklearn: {e}")
        from sklearn.decomposition import PCA as skPCA
        pca = skPCA(n_components=k, svd_solver="randomized", random_state=42)
        scores = pca.fit_transform(X).astype(np.float32)
        mean = pca.mean_.reshape(1, -1).astype(np.float32)
        components = pca.components_.T.astype(np.float32)   # (D, k)
        scale = np.sqrt(pca.explained_variance_).astype(np.float32)
        projector = _PCAProjector(mean_=mean, components_=components, scale_=scale)
        return scores, projector


def _reduce(X: np.ndarray, n_components: int = 1000,
            device: Optional[torch.device] = None) -> np.ndarray:
    """Fit-and-transform PCA; discards the projector (for RSA/CKA where test isn't needed)."""
    if device is None:
        device = _get_device()
    scores, _ = _fit_reduce_gpu(X, n_components, device)
    return scores


# ---------------------------------------------------------------------------
# GPU-accelerated CKA
# ---------------------------------------------------------------------------

def _center_gram_gpu(K: torch.Tensor) -> torch.Tensor:
    """Double-center an (n x n) gram matrix on GPU."""
    n = K.shape[0]
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def linear_cka(X: np.ndarray, Y: np.ndarray,
               device: Optional[torch.device] = None) -> float:
    """Linear CKA computed on GPU."""
    if device is None:
        device = _get_device()
    try:
        Xt = torch.from_numpy(X.astype(np.float32)).to(device)
        Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
        Xt = Xt - Xt.mean(dim=0, keepdim=True)
        Yt = Yt - Yt.mean(dim=0, keepdim=True)
        K = Xt @ Xt.T
        L = Yt @ Yt.T
        Kc = _center_gram_gpu(K)
        Lc = _center_gram_gpu(L)
        hsic = (Kc * Lc).sum()
        norm = ((Kc * Kc).sum() * (Lc * Lc).sum()).sqrt()
        result = float("nan") if norm < 1e-10 else float((hsic / norm).item())
        del Xt, Yt, K, L, Kc, Lc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        print(f"  [GPU linear CKA] error: {e}")
        return float("nan")


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_frac: float = 0.5,
            device: Optional[torch.device] = None) -> float:
    """RBF-kernel CKA computed on GPU."""
    if device is None:
        device = _get_device()

    def rbf_gram_gpu(Z: torch.Tensor) -> torch.Tensor:
        sq_dists = (
            (Z ** 2).sum(dim=1, keepdim=True)
            - 2 * Z @ Z.T
            + (Z ** 2).sum(dim=1, keepdim=True).T
        )
        sq_dists = sq_dists.clamp(min=0.0)
        pos = sq_dists[sq_dists > 0]
        median_sq = pos.median().item() if pos.numel() > 0 else 1.0
        sigma2 = (sigma_frac ** 2) * median_sq
        if sigma2 < 1e-10:
            sigma2 = 1.0
        return torch.exp(-sq_dists / (2 * sigma2))

    try:
        Xt = torch.from_numpy(X.astype(np.float32)).to(device)
        Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
        Xt = Xt - Xt.mean(dim=0, keepdim=True)
        Yt = Yt - Yt.mean(dim=0, keepdim=True)
        K = rbf_gram_gpu(Xt)
        L = rbf_gram_gpu(Yt)
        Kc = _center_gram_gpu(K)
        Lc = _center_gram_gpu(L)
        hsic = (Kc * Lc).sum()
        norm = ((Kc * Kc).sum() * (Lc * Lc).sum()).sqrt()
        result = float("nan") if norm < 1e-10 else float((hsic / norm).item())
        del Xt, Yt, K, L, Kc, Lc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        print(f"  [GPU RBF CKA] error: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# GPU-accelerated Ridge regression
#
# Uses TorchChunkedKernelRidgeCV from metrics/utils.py, which builds the
# N×N kernel matrix row-by-row in chunks — never moving the full (N, D)
# feature matrix to GPU.  This is the same solver used for model-to-neural
# comparisons and is safe for D >> N (e.g. D=75k, N=22k).
# ---------------------------------------------------------------------------

def _ridge_r2_chunked(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alphas: List[float],
    device: torch.device,
) -> float:
    """
    Ridge R² using TorchChunkedKernelRidgeCV for fat matrices on GPU.
    Fits on (X_train → Y_train), evaluates on (X_test, Y_test).
    """
    from .utils import TorchChunkedKernelRidgeCV
    solver = TorchChunkedKernelRidgeCV(
        alphas=alphas,
        device=str(device),
        val_fraction=0.1,
    )
    solver.fit(X_train, Y_train)
    Y_pred = solver.predict(X_test)

    # R² averaged across all output dimensions
    Y_test = Y_test.astype(np.float32)
    ss_res = ((Y_pred - Y_test) ** 2).sum(axis=0)
    ss_tot = ((Y_test - Y_test.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-10)
    return float(r2_per_dim.mean())


def _ridge_r2_cv_chunked(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: List[float],
    n_splits: int,
    device: torch.device,
) -> float:
    """K-fold cross-validated ridge R² using chunked GPU solver."""
    n = X.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    r2s = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        r2s.append(
            _ridge_r2_chunked(X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], alphas, device)
        )
    return float(np.mean(r2s))


# ---------------------------------------------------------------------------
# GPU-accelerated RSA (correlation-distance RDM)
# ---------------------------------------------------------------------------

def _rdm_vectors_gpu(X: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Compute upper-triangle of correlation-distance RDM on GPU.
    Returns 1-D numpy array of length n*(n-1)/2.
    """
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    # Normalise rows
    Xt = Xt - Xt.mean(dim=1, keepdim=True)
    norms = Xt.norm(dim=1, keepdim=True).clamp(min=1e-10)
    Xt = Xt / norms
    # Cosine similarity matrix → correlation distance
    sim = Xt @ Xt.T          # (n x n)
    dist = 1.0 - sim         # correlation distance
    n = dist.shape[0]
    # Upper triangle indices (k=1 skips diagonal)
    rows, cols = torch.triu_indices(n, n, offset=1, device=device)
    vec = dist[rows, cols].cpu().numpy()
    del Xt, sim, dist
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vec.astype(np.float64)


def _spearman_cpu(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two 1-D arrays (NaN-safe)."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan")
    r, _ = spearmanr(a[mask], b[mask])
    return float(r)


def _pearson_cpu(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays (NaN-safe)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(np.dot(a, b) / denom) if denom > 1e-10 else float("nan")


# ---------------------------------------------------------------------------
# CrossModelMetric
# ---------------------------------------------------------------------------

class CrossModelMetric(BaseMetric):
    """
    Compare two model representation matrices with RSA, CKA, and ridge R².

    All heavy compute (PCA, CKA gram matrices, ridge, RDM) runs on GPU when
    available.

    Parameters
    ----------
    pca_components : int
        Maximum PCA components before RSA / CKA.  Set to 0 to disable PCA.
    n_cv_splits : int
        K-fold splits for cross-validated ridge (used when test split absent).
    """

    def __init__(
        self,
        pca_components: int = 1000,
        n_cv_splits: int = 5,
        ceiling: Optional[np.ndarray] = None,
    ):
        super().__init__(ceiling=ceiling)
        self.pca_components = pca_components
        self.n_cv_splits = n_cv_splits

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        assert source.ndim == 2, f"source must be 2-D, got {source.shape}"
        assert target.ndim == 2, f"target must be 2-D, got {target.shape}"
        assert source.shape[0] == target.shape[0], (
            f"n_images mismatch: source {source.shape[0]} vs target {target.shape[0]}"
        )

        device = _get_device()
        print(f"  [CrossModel] Using device: {device}")

        def _flush(label: str = ""):
            """Fully release PyTorch VRAM cache so free memory is accurate."""
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                free, total = torch.cuda.mem_get_info(device.index or 0)
                print(f"  [VRAM] {label}free={free/1e9:.1f}/{total/1e9:.1f} GB")

        n = source.shape[0]
        results: Dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # PCA reduction — fit on train, project test on same basis            #
        # ------------------------------------------------------------------ #
        k = self.pca_components if self.pca_components > 0 else max(source.shape[1], target.shape[1])

        _flush("before source PCA — ")
        print(f"  [CrossModel] PCA: source {source.shape} → k={k} ...")
        src_r, src_proj = _fit_reduce_gpu(source, k, device)
        _flush("after source PCA — ")

        print(f"  [CrossModel] PCA: target {target.shape} → k={k} ...")
        tgt_r, tgt_proj = _fit_reduce_gpu(target, k, device)
        _flush("after target PCA — ")

        # ------------------------------------------------------------------ #
        # RSA (GPU RDMs, CPU Spearman/Pearson)                                 #
        # Each RDM is N×N then immediately extracted to a CPU vector.          #
        # ------------------------------------------------------------------ #
        print("  [CrossModel] Computing RSA (GPU RDMs)...")
        vec_src = _rdm_vectors_gpu(src_r, device)
        _flush("after source RDM — ")
        vec_tgt = _rdm_vectors_gpu(tgt_r, device)
        _flush("after target RDM — ")
        results["rsa_spearman"] = _spearman_cpu(vec_src, vec_tgt)
        results["rsa_corr"] = _pearson_cpu(vec_src, vec_tgt)
        del vec_src, vec_tgt

        # ------------------------------------------------------------------ #
        # CKA (GPU) — gram matrices are N×N, freed inside each function        #
        # ------------------------------------------------------------------ #
        print("  [CrossModel] Computing linear CKA (GPU)...")
        results["linear_cka"] = linear_cka(src_r, tgt_r, device)
        _flush("after linear CKA — ")

        print("  [CrossModel] Computing RBF CKA (GPU)...")
        results["rbf_cka"] = rbf_cka(src_r, tgt_r, device=device)
        _flush("after RBF CKA — ")

        # ------------------------------------------------------------------ #
        # Ridge R² (GPU chunked kernel, both directions)                       #
        # ------------------------------------------------------------------ #
        print("  [CrossModel] Computing Ridge R² A→B (GPU chunked kernel)...")
        if test_source is not None and test_target is not None:
            # Project test onto the SAME basis as train — no refitting
            ts_r = src_proj.transform(test_source, device)
            tt_r = tgt_proj.transform(test_target, device)
            _flush("after test PCA — ")
            results["ridge_r2_A_to_B"] = _ridge_r2_chunked(src_r, tgt_r, ts_r, tt_r, _RIDGE_ALPHAS, device)
            _flush("after ridge A→B — ")
            print("  [CrossModel] Computing Ridge R² B→A (GPU chunked kernel)...")
            results["ridge_r2_B_to_A"] = _ridge_r2_chunked(tgt_r, src_r, tt_r, ts_r, _RIDGE_ALPHAS, device)
            _flush("after ridge B→A — ")
        else:
            results["ridge_r2_A_to_B"] = _ridge_r2_cv_chunked(src_r, tgt_r, _RIDGE_ALPHAS, self.n_cv_splits, device)
            _flush("after ridge A→B CV — ")
            print("  [CrossModel] Computing Ridge R² B→A (GPU chunked kernel)...")
            results["ridge_r2_B_to_A"] = _ridge_r2_cv_chunked(tgt_r, src_r, _RIDGE_ALPHAS, self.n_cv_splits, device)
            _flush("after ridge B→A CV — ")

        return results

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        if source.ndim > 2:
            source = source.reshape(source.shape[0], -1)
        if target.ndim > 2:
            target = target.reshape(target.shape[0], -1)
        if test_source is not None and test_source.ndim > 2:
            test_source = test_source.reshape(test_source.shape[0], -1)
        if test_target is not None and test_target.ndim > 2:
            test_target = test_target.reshape(test_target.shape[0], -1)

        scores = self.compute_raw(source, target, test_source, test_target, stratify_on)
        return {"cross_model": scores}
