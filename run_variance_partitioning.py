"""
run_variance_partitioning.py -- Compute joint model R² for variance partitioning.

Key optimizations vs the previous version:
- Uses a direct kernel ridge implementation that avoids primal weight recovery
  entirely (we only need R² per voxel, not the regression weights).
- torch.cuda.empty_cache() is called only after outer row chunks, not every tile.
- tqdm progress bars on every kernel build phase.
- Features are loaded, z-scored, and concatenated in float32 with in-place ops.
"""
import os
import sys

# Set GPU before torch/cuda is ever loaded
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

import argparse
import pickle
import datetime
import gc
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.datasets import get_data_home

_project_root = os.path.dirname(os.path.abspath(__file__))
_project_data = os.path.join(_project_root, "bbscore_data")
if os.path.exists(_project_data):
    os.environ["SCIKIT_LEARN_DATA"] = _project_data

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from data.TVSD import TVSDAssemblyV1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zscore_inplace(X):
    """Z-score along samples (axis=0) in-place; returns (mean, std) for test."""
    m = X.mean(axis=0)
    X -= m
    s = X.std(axis=0)
    s += 1e-8
    X /= s
    return m, s


def _build_kernel(X_row_cpu, X_row_mean_gpu, row_idx,
                  X_col_cpu, X_col_mean_gpu, col_idx,
                  chunk_rows, device, label="K"):
    """
    Build K where K[i,j] = (X_row[row_idx[i]] - X_row_mean) . (X_col[col_idx[j]] - X_col_mean).

    Rows and columns may come from different matrices (e.g. K_test: rows=test, cols=train).
    For symmetric cases pass the same X/mean for both row and col.
    empty_cache() only after each outer row-chunk to avoid CUDA sync overhead.
    Uses slicing when indices are 0,1,...,n-1 (avoids gather copy on CPU).
    """
    n_rows = len(row_idx)
    n_cols = len(col_idx)
    n_i = (n_rows + chunk_rows - 1) // chunk_rows
    n_j = (n_cols + chunk_rows - 1) // chunk_rows
    total = n_i * n_j

    def _is_arange(idx):
        if idx[0].item() != 0 or idx[-1].item() != len(idx) - 1:
            return False
        return (idx == torch.arange(len(idx), dtype=idx.dtype, device=idx.device)).all().item()

    row_contig = _is_arange(row_idx)
    col_contig = _is_arange(col_idx)

    K = torch.zeros((n_rows, n_cols), dtype=torch.float32, device=device)
    pbar = tqdm(total=total, desc=f"   {label}", unit="block")

    for i in range(0, n_rows, chunk_rows):
        i_end = min(i + chunk_rows, n_rows)
        if row_contig:
            X_left = X_row_cpu[i:i_end].to(device, dtype=torch.float32, non_blocking=True)
        else:
            X_left = X_row_cpu[row_idx[i:i_end]].to(device, dtype=torch.float32, non_blocking=True)
        X_left -= X_row_mean_gpu

        for j in range(0, n_cols, chunk_rows):
            j_end = min(j + chunk_rows, n_cols)
            if col_contig:
                X_right = X_col_cpu[j:j_end].to(device, dtype=torch.float32, non_blocking=True)
            else:
                X_right = X_col_cpu[col_idx[j:j_end]].to(device, dtype=torch.float32, non_blocking=True)
            X_right -= X_col_mean_gpu
            K[i:i_end, j:j_end] = X_left @ X_right.T
            del X_right
            pbar.update(1)

        del X_left
        if device != "cpu":
            torch.cuda.empty_cache()

    pbar.close()
    return K


def kernel_ridge_r2_per_voxel(X_train_cpu, y_train, X_test_cpu, y_test,
                               alphas, device, val_fraction=0.15, chunk_rows=None):
    """
    Kernel ridge regression that returns per-voxel R² on the held-out test set.
    Never recovers primal weights — stays in dual space throughout.

    Steps:
      1. Internal val split of train set → find best alpha via val R².
      2. Refit on full train set with best alpha.
      3. Predict on test set → per-voxel R².
    """
    n_train, D = X_train_cpu.shape
    n_test = X_test_cpu.shape[0]
    T = y_train.shape[1]

    if chunk_rows is None:
        # Target ~15 GB per tile pair (smaller = faster blocks, more progress visibility)
        # tile_bytes = chunk_rows * D * 4 * 2 (left + right)
        if device != "cpu":
            try:
                free_vram, _ = torch.cuda.mem_get_info()
                k_full_bytes = n_train * n_train * 4
                tile_budget = 15 * (1 << 30)  # 15 GB per tile pair
                budget = min(free_vram - k_full_bytes - 2 * (1 << 30), tile_budget)
                budget = max(budget, 1 << 30)
                chunk_rows = max(64, int(budget // (2 * D * 4)))
                chunk_rows = min(chunk_rows, n_train)
                print(f"   auto chunk_rows={chunk_rows}  "
                      f"(free_VRAM={free_vram/1e9:.1f} GB, "
                      f"tile={2*chunk_rows*D*4/1e9:.2f} GB)")
            except Exception:
                chunk_rows = 500
        else:
            chunk_rows = 4000

    # Convert to float32 tensors (CPU)
    if not isinstance(X_train_cpu, torch.Tensor):
        X_train_cpu = torch.from_numpy(X_train_cpu.astype(np.float32))
    if not isinstance(X_test_cpu, torch.Tensor):
        X_test_cpu = torch.from_numpy(X_test_cpu.astype(np.float32))

    y_train_t = torch.from_numpy(y_train.astype(np.float32)) if not isinstance(y_train, torch.Tensor) else y_train.float()
    y_test_t  = torch.from_numpy(y_test.astype(np.float32))  if not isinstance(y_test,  torch.Tensor) else y_test.float()

    # Center targets
    y_mean = y_train_t.mean(0, keepdim=True)
    y_train_c = y_train_t - y_mean

    # Feature mean for centering on GPU (all kernels use train mean for consistency)
    X_train_mean = X_train_cpu.mean(0, keepdim=True).to(device)

    # --- Step 1: internal val split to pick alpha ---
    n_val = max(1, int(n_train * val_fraction))
    perm  = torch.randperm(n_train)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    n_tr = len(train_idx)

    all_train = torch.arange(n_train)

    print(f"\n   [KernelRidge] N_train={n_train}, N_val={n_val}, D={D}, T={T}")
    t0 = time.time()

    print("   Building K_tr (train×train)...")
    K_tr = _build_kernel(X_train_cpu, X_train_mean, train_idx,
                         X_train_cpu, X_train_mean, train_idx,
                         chunk_rows, device, label="K_tr")

    print("   Building K_val (val×train)...")
    K_val = _build_kernel(X_train_cpu, X_train_mean, val_idx,
                          X_train_cpu, X_train_mean, train_idx,
                          chunk_rows, device, label="K_val")

    y_tr  = y_train_c[train_idx].to(device)
    y_vl  = y_train_c[val_idx].to(device)
    I_tr  = torch.eye(n_tr, dtype=torch.float32, device=device)

    print(f"   Scanning {len(alphas)} alphas...")
    best_alpha = alphas[-1]
    best_r2    = -float('inf')
    for alpha in alphas:
        try:
            dual = torch.linalg.solve(K_tr + alpha * I_tr, y_tr)
        except RuntimeError:
            dual = torch.linalg.lstsq(K_tr + alpha * I_tr, y_tr).solution
        preds  = K_val @ dual
        ss_res = ((y_vl - preds) ** 2).sum(0)
        ss_tot = (y_vl ** 2).sum(0)
        r2     = (1 - ss_res / (ss_tot + 1e-6)).mean().item()
        if r2 > best_r2:
            best_r2    = r2
            best_alpha = alpha

    print(f"   Best alpha={best_alpha:.3g}  Val R²={best_r2:.4f}  ({time.time()-t0:.1f}s so far)")

    del K_tr, K_val, I_tr, y_tr, y_vl, dual, preds
    if device != "cpu":
        torch.cuda.empty_cache()

    # --- Step 2: refit on full train set ---
    print("   Building K_full (all_train×all_train)...")
    K_full = _build_kernel(X_train_cpu, X_train_mean, all_train,
                           X_train_cpu, X_train_mean, all_train,
                           chunk_rows, device, label="K_full")

    y_full   = y_train_c.to(device)
    I_full   = torch.eye(n_train, dtype=torch.float32, device=device)
    try:
        dual_full = torch.linalg.solve(K_full + best_alpha * I_full, y_full)
    except RuntimeError:
        dual_full = torch.linalg.lstsq(K_full + best_alpha * I_full, y_full).solution

    del K_full, I_full, y_full
    if device != "cpu":
        torch.cuda.empty_cache()

    # --- Step 3: predict on test set ---
    # K_test[i,j] = (x_test_i - mu_train) . (x_train_j - mu_train)
    # Pass X_test as rows and X_train as cols, each centered by their own mean.
    # _build_kernel now handles row/col from different matrices correctly.
    print("   Building K_test (test×all_train)...")
    all_test = torch.arange(n_test)
    K_test = _build_kernel(X_test_cpu,  X_train_mean, all_test,
                           X_train_cpu, X_train_mean, all_train,
                           chunk_rows, device, label="K_test")
    if device != "cpu":
        torch.cuda.empty_cache()

    # Predict: preds_test[i] = K_test[i,:] @ dual_full + y_mean
    y_pred_test = K_test @ dual_full  # (n_test, T)
    y_pred_test = y_pred_test + y_mean.to(device)

    del K_test, dual_full
    if device != "cpu":
        torch.cuda.empty_cache()

    # Per-voxel R²
    y_test_gpu = y_test_t.to(device)
    ss_res = ((y_test_gpu - y_pred_test) ** 2).sum(0)  # (T,)
    ss_tot = ((y_test_gpu - y_test_gpu.mean(0)) ** 2).sum(0)  # (T,)
    r2_per_voxel = (1 - ss_res / (ss_tot + 1e-6)).cpu().numpy()  # (T,)

    # Per-voxel Pearson (safeguard against constant predictions / zero variance)
    pearson_per_voxel = []
    for i in range(T):
        y_true = y_test_gpu[:, i].cpu().numpy()
        y_pred = y_pred_test[:, i].cpu().numpy()
        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            pearson_per_voxel.append(0.0)
        else:
            r = np.corrcoef(y_true, y_pred)[0, 1]
            pearson_per_voxel.append(float(r) if not np.isnan(r) else 0.0)
    pearson_per_voxel = np.array(pearson_per_voxel)

    print(f"   Done. Median R²={np.median(r2_per_voxel):.4f}  "
          f"Median Pearson={np.median(pearson_per_voxel):.4f}  "
          f"({time.time()-t0:.1f}s total)")

    return r2_per_voxel, pearson_per_voxel, best_alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_vp(model_a, layer_a, model_b, layer_b, stimulus_class, alphas, chunk_rows=None):
    print(f"\nVariance Partitioning: {model_a}/{layer_a}  +  {model_b}/{layer_b}")
    print(f"Stimuli: {stimulus_class}")

    data_home   = get_data_home()
    features_dir = os.path.join(data_home, "features")
    results_dir  = os.path.join(data_home, "results")
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    # 1. Neural targets
    print("\nLoading V1 neural assembly...")
    assembly  = TVSDAssemblyV1()
    y_train_raw, _ = assembly.get_assembly(train=True)
    y_test_raw,  _ = assembly.get_assembly(train=False)
    y_train = np.asarray(y_train_raw, dtype=np.float32)
    y_test  = np.asarray(y_test_raw,  dtype=np.float32)
    if y_train.ndim > 2:
        y_train = y_train.reshape(y_train.shape[0], -1)
    if y_test.ndim > 2:
        y_test  = y_test.reshape(y_test.shape[0], -1)
    del assembly, y_train_raw, y_test_raw
    gc.collect()
    print(f"  y_train {y_train.shape}, y_test {y_test.shape}")

    # 2. Load features
    def load_feats(model, layer):
        fname = f"{model}_{layer}_{stimulus_class}_features.pkl"
        path  = os.path.join(features_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        print(f"  Loading {model} / {layer}...")
        with open(path, "rb") as fh:
            fd = pickle.load(fh)
        tr = np.asarray(fd["train"], dtype=np.float32)
        te = np.asarray(fd["test"],  dtype=np.float32)
        if tr.ndim > 2:
            tr = tr.reshape(tr.shape[0], -1)
            te = te.reshape(te.shape[0], -1)
        del fd
        gc.collect()
        print(f"    shape train={tr.shape}, test={te.shape}")
        return tr, te

    print("\nLoading & z-scoring features...")
    tr_a, te_a = load_feats(model_a, layer_a)
    m_a, s_a   = zscore_inplace(tr_a)
    te_a       = (te_a - m_a) / (s_a)
    del m_a, s_a
    gc.collect()

    tr_b, te_b = load_feats(model_b, layer_b)
    m_b, s_b   = zscore_inplace(tr_b)
    te_b       = (te_b - m_b) / (s_b)
    del m_b, s_b
    gc.collect()

    print("\nConcatenating features...")
    X_train = np.concatenate([tr_a, tr_b], axis=1)
    X_test  = np.concatenate([te_a, te_b], axis=1)
    del tr_a, tr_b, te_a, te_b
    gc.collect()
    print(f"  Joint X_train {X_train.shape}, X_test {X_test.shape}")

    # 3. Run kernel ridge (no weight recovery)
    print("\nRunning joint kernel ridge (dual space, no weight recovery)...")
    r2_joint, pearson_joint, best_alpha = kernel_ridge_r2_per_voxel(
        X_train, y_train, X_test, y_test,
        alphas=alphas,
        device=device,
        chunk_rows=chunk_rows,
    )

    # 4. Save
    output_file = os.path.join(results_dir, f"Joint_{model_a}_{model_b}_TVSDV1.pkl")
    payload = {
        "r2_per_voxel":      r2_joint,
        "pearson_per_voxel": pearson_joint,
        "best_alpha":        best_alpha,
        "model_a":  model_a,
        "layer_a":  layer_a,
        "model_b":  model_b,
        "layer_b":  layer_b,
        "stimulus_class": stimulus_class,
        "n_features_joint": int(X_train.shape[1]),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    with open(output_file, "wb") as fh:
        pickle.dump(payload, fh)

    print(f"\nSaved → {output_file}")
    print(f"  Median R²     = {np.median(r2_joint):.4f}")
    print(f"  Median Pearson = {np.median(pearson_joint):.4f}")
    print(f"  Best alpha    = {best_alpha:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a",        required=True)
    parser.add_argument("--layer-a",        required=True)
    parser.add_argument("--model-b",        required=True)
    parser.add_argument("--layer-b",        required=True)
    parser.add_argument("--stimulus-class", default="TVSDStimulusTrainSet")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    parser.add_argument("--chunk-rows", type=int, default=None,
                        help="Override tile size (smaller = more blocks, faster progress)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()
    if args.gpu is not None:
        print(f"Using GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
    run_vp(args.model_a, args.layer_a, args.model_b, args.layer_b,
           args.stimulus_class, args.alphas, chunk_rows=args.chunk_rows)
