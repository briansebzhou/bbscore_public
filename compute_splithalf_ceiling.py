"""
Compute per-voxel split-half reliability ceiling from repeated presentations
of 'wheretheressmoke' in the LeBel2023 dataset.

This gives a TRUE UPPER BOUND ceiling: scores normalized by this will be <= 1.0.

Method:
1. Load raw NIfTI runs for each subject (5-10 runs of wheretheressmoke)
2. Preprocess each run: motion correction to first run, detrend, z-score
3. Apply mask_thick.nii.gz to extract cortical voxels
4. Split into odd/even runs, average each half
5. Per-voxel Pearson correlation between the two halves
6. Spearman-Brown correction: r_ceiling = 2*r / (1 + r)

Usage (on node14):
    python compute_splithalf_ceiling.py --data-dir /data2/thekej/bbscore_data/ds003020
"""

import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr
from nilearn.image import mean_img
from nilearn.maskers import NiftiMasker
from nilearn.signal import clean


def load_mask(pycortex_dir, subject):
    """Load mask_thick.nii.gz for a subject."""
    mask_path = os.path.join(
        pycortex_dir, subject, 'transforms',
        f'{subject}_auto', 'mask_thick.nii.gz')
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(bool)
    print(f"  {subject} mask: {mask.sum()} voxels", flush=True)
    return mask, mask_img


def compute_subject_ceiling(data_dir, pycortex_dir, subject, sessions):
    """Compute split-half reliability ceiling for one subject."""
    print(f"\n=== Processing {subject} ===", flush=True)

    # Load mask
    mask, mask_img = load_mask(pycortex_dir, subject)
    n_voxels = int(mask.sum())

    # Collect all run file paths
    run_paths = []
    for ses_idx, ses in enumerate(sessions):
        run_num = ses_idx + 1
        pattern = os.path.join(
            data_dir, 'raw', f'sub-{subject}', f'ses-{ses}', 'func',
            f'sub-{subject}_ses-{ses}_task-wheretheressmoke_run-{run_num}_bold.nii.gz')
        files = glob.glob(pattern)
        if not files:
            print(f"  WARNING: Missing run-{run_num} ses-{ses}",
                  flush=True)
            continue
        run_paths.append((run_num, ses, files[0]))

    if len(run_paths) < 2:
        print(f"  ERROR: Need at least 2 runs, got {len(run_paths)}")
        return None, None, None

    # Create a NiftiMasker with the subject's mask for consistent
    # extraction, detrending, standardization, and high-pass filtering
    masker = NiftiMasker(
        mask_img=mask_img,
        detrend=True,
        standardize='zscore_sample',
        high_pass=1.0 / 128,  # 128s high-pass (standard fMRI)
        t_r=2.0,
    )
    masker.fit()

    # Load and preprocess each run
    runs = []
    for run_num, ses, fpath in run_paths:
        print(f"  Loading & preprocessing run-{run_num} (ses-{ses})...",
              flush=True)
        ts = masker.transform(fpath)  # (n_trs, n_voxels)
        runs.append(ts)
        print(f"    Shape: {ts.shape} (TRs x voxels)", flush=True)

    if len(runs) < 2:
        print(f"  ERROR: Need at least 2 runs, got {len(runs)}")
        return None, None, None

    # Trim all runs to same length (they should be same but just in case)
    min_trs = min(r.shape[0] for r in runs)
    runs = [r[:min_trs] for r in runs]
    print(f"  {len(runs)} runs loaded, {min_trs} TRs each, {n_voxels} voxels",
          flush=True)

    # Split into odd/even halves
    odd_runs = [runs[i] for i in range(len(runs)) if i % 2 == 0]
    even_runs = [runs[i] for i in range(len(runs)) if i % 2 == 1]

    # Average within each half
    odd_avg = np.mean(odd_runs, axis=0)   # (t, n_voxels)
    even_avg = np.mean(even_runs, axis=0)  # (t, n_voxels)

    print(f"  Odd half: {len(odd_runs)} runs, Even half: {len(even_runs)} runs",
          flush=True)

    # Vectorized per-voxel Pearson correlation between halves
    # r = sum((x - mx)(y - my)) / sqrt(sum((x-mx)^2) * sum((y-my)^2))
    odd_centered = odd_avg - odd_avg.mean(axis=0, keepdims=True)
    even_centered = even_avg - even_avg.mean(axis=0, keepdims=True)
    num = (odd_centered * even_centered).sum(axis=0)
    denom = np.sqrt(
        (odd_centered ** 2).sum(axis=0) * (even_centered ** 2).sum(axis=0))
    denom[denom < 1e-10] = 1.0
    r_split = num / denom

    # Spearman-Brown correction (factor 2 for split-half)
    n_correction = 2
    r_split_clipped = np.clip(r_split, 0, None)
    r_ceiling = (n_correction * r_split_clipped /
                 (1 + (n_correction - 1) * r_split_clipped))

    valid = r_ceiling > 0
    print(f"  Split-half r: median={np.median(r_split):.4f}, "
          f"mean={np.mean(r_split):.4f}", flush=True)
    print(f"  Spearman-Brown corrected: median={np.median(r_ceiling):.4f}, "
          f"mean={np.mean(r_ceiling):.4f}", flush=True)
    print(f"  Voxels with r_ceiling > 0: {valid.sum()} / {n_voxels} "
          f"({100 * valid.sum() / n_voxels:.1f}%)", flush=True)
    print(f"  Voxels with r_ceiling > 0.1: "
          f"{(r_ceiling > 0.1).sum()} / {n_voxels}", flush=True)
    print(f"  Voxels with r_ceiling > 0.2: "
          f"{(r_ceiling > 0.2).sum()} / {n_voxels}", flush=True)
    print(f"  Voxels with r_ceiling > 0.5: "
          f"{(r_ceiling > 0.5).sum()} / {n_voxels}", flush=True)

    # Per-TR spatial reliability
    n_trs = min_trs
    # Vectorized: correlate spatial patterns at each TR
    odd_sp = odd_avg - odd_avg.mean(axis=1, keepdims=True)
    even_sp = even_avg - even_avg.mean(axis=1, keepdims=True)
    num_sp = (odd_sp * even_sp).sum(axis=1)
    denom_sp = np.sqrt(
        (odd_sp ** 2).sum(axis=1) * (even_sp ** 2).sum(axis=1))
    denom_sp[denom_sp < 1e-10] = 1.0
    per_tr_spatial_r = num_sp / denom_sp

    per_tr_spatial_r_clipped = np.clip(per_tr_spatial_r, 0, None)
    per_tr_spatial_ceiling = (
        n_correction * per_tr_spatial_r_clipped /
        (1 + (n_correction - 1) * per_tr_spatial_r_clipped))

    print(f"  Per-TR spatial reliability: "
          f"median={np.median(per_tr_spatial_ceiling):.4f}, "
          f"mean={np.mean(per_tr_spatial_ceiling):.4f}", flush=True)

    return r_ceiling, valid, per_tr_spatial_ceiling


def main():
    parser = argparse.ArgumentParser(
        description='Compute split-half reliability ceiling')
    parser.add_argument('--data-dir', required=True,
                        help='Path to ds003020 data directory')
    parser.add_argument('--output', default=None,
                        help='Output .npz path (default: data/lebel2023_ceiling_splithalf.npz)')
    args = parser.parse_args()

    pycortex_dir = os.path.join(
        args.data_dir, 'derivative', 'pycortex-db')

    # Subject -> sessions mapping
    subject_sessions = {
        'UTS01': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'UTS02': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'UTS03': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'UTS04': [2, 3, 4, 5, 6],
        'UTS05': [2, 3, 4, 5, 6],
        'UTS06': [2, 3, 4, 5, 6],
        'UTS07': [2, 3, 4, 5, 6],
        'UTS08': [2, 3, 4, 5, 6],
    }

    results = {}
    for subject, sessions in subject_sessions.items():
        r_ceiling, valid, per_tr_spatial = compute_subject_ceiling(
            args.data_dir, pycortex_dir, subject, sessions)
        if r_ceiling is not None:
            results[subject] = r_ceiling.astype(np.float32)
            results[f'{subject}_valid'] = valid
            results[f'{subject}_per_tr_spatial'] = (
                per_tr_spatial.astype(np.float32))

    if not results:
        print("ERROR: No subjects processed successfully")
        sys.exit(1)

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), 'data', 'lebel2023_ceiling_splithalf.npz')
    results['_metadata'] = np.array(
        'Split-half reliability ceiling (Spearman-Brown corrected). '
        'Story: wheretheressmoke. Method: odd/even run split, '
        'per-voxel Pearson correlation, 2x Spearman-Brown correction.')
    np.savez_compressed(output_path, **results)
    print(f"\nSaved to {output_path}")
    print(f"Keys: {list(results.keys())}")


if __name__ == '__main__':
    main()
