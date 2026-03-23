from typing import List, Optional, Dict, Callable, Union, Tuple
import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold

from .base import BaseMetric
from .utils import run_kfold_cv, run_kfold_cv_chunked, run_eval_chunked, run_eval, pearson_correlation_scorer
# Import TorchRidge from utils
from .utils import TorchRidge, TorchRidgeCV, TorchElasticNetCV, TorchConstrainedRidgeCV, TorchBlockRidgeCV, TorchElasticNetCV_float32, TorchChunkedKernelRidgeCV


try:
    # cuml.accel (sklearn monkey-patching) was introduced in cuml 25.x.
    # cuml 24.x only exposes individual estimators like cuml.linear_model.Ridge.
    # We attempt accel first; on 24.x it will raise ModuleNotFoundError and we
    # fall back gracefully — our TorchRidgeCV / TorchChunkedKernelRidgeCV
    # already provide full GPU acceleration for the ridge path.
    import cuml.accel
    cuml.accel.install()
    print("cuML accel installed (sklearn calls accelerated on GPU).")
except ModuleNotFoundError:
    # cuml 24.x: accel submodule absent — use direct cuml estimators where needed.
    try:
        import cuml  # noqa: F401 — confirms cuml itself is present
    except ImportError:
        pass  # no cuml at all; TorchRidgeCV / TorchChunkedKernelRidgeCV handle GPU
except Exception:
    pass  # any other cuml init error — fall through silently


class RidgeMetric(BaseMetric):
    def __init__(
        self,
        alpha_options: List[float] = [
            1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
            0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10
        ],
        ceiling: Optional[float] = None,
        mode: Optional[str] = None,
        subsample_features_for_alpha: Optional[int] = None,
    ):
        """
        Args:
            alpha_options: List of alpha values to try during cross-validation.
            ceiling: Optional ceiling value for score normalization.
            mode: Backend - "torch" (GPU), "sklearn" (CPU), or None (auto: torch if CUDA else sklearn).
            subsample_features_for_alpha: If set, use a random subset of this many
                features to find the optimal alpha via RidgeCV, then train the final
                model with all features using Ridge(alpha=best_alpha). This speeds up
                alpha selection significantly for high-dimensional features.
                Set to None to disable (default behavior using full RidgeCV).
                Recommended value: 2000-5000 for large models.
        """
        super().__init__(ceiling)
        self.alpha_options = alpha_options
        # Auto-select GPU backend when mode is None
        if mode is None:
            self.mode = "torch" if torch.cuda.is_available() else "sklearn"
            if self.mode == "torch":
                print("Ridge: using GPU (TorchRidge backend)")
        else:
            self.mode = mode
        self.subsample_features_for_alpha = subsample_features_for_alpha

    def _find_best_alpha_subsampled(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_subsample: int,
        random_state: int = 42,
    ) -> float:
        """
        Find the best alpha using a random subset of features.

        Args:
            source: Full feature array (N, D)
            target: Target array (N, T)
            n_subsample: Number of features to subsample
            random_state: Random seed for reproducibility

        Returns:
            Best alpha value found on subsampled features
        """
        n_features = source.shape[1]

        if n_features <= n_subsample:
            # No need to subsample
            cv_model = RidgeCV(alphas=self.alpha_options,
                               store_cv_results=False)
            cv_model.fit(source, target)
            return cv_model.alpha_

        # Subsample features
        rng = np.random.RandomState(random_state)
        feature_indices = rng.choice(
            n_features, size=n_subsample, replace=False)
        source_sub = source[:, feature_indices]

        # Find best alpha on subsampled features
        cv_model = RidgeCV(alphas=self.alpha_options, store_cv_results=False)
        cv_model.fit(source_sub, target)

        return cv_model.alpha_

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        scoring_funcs = {
            "pearson": lambda y_true, y_pred: np.array([pearson_correlation_scorer(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]),
            "r2": lambda y_true, y_pred: np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]),
        }

        if source.ndim > 2:
            N = source.shape[0]
            source = source.reshape(N, -1)
            if test_source is not None:
                N_test = test_source.shape[0]
                test_source = test_source.reshape(N_test, -1)

        if self.mode == "sklearn":
            # Check if we should use subsampled alpha search
            if (self.subsample_features_for_alpha is not None and
                    source.shape[1] > self.subsample_features_for_alpha):
                # Fast path: find alpha on subsampled features, then use Ridge with fixed alpha
                best_alpha = self._find_best_alpha_subsampled(
                    source, target, self.subsample_features_for_alpha
                )

                def model_factory():
                    return Ridge(alpha=best_alpha)
            else:
                # Standard path: full RidgeCV
                def model_factory():
                    return RidgeCV(alphas=self.alpha_options,
                                   store_cv_results=True, alpha_per_target=True)
        else:
            # "torch" mode: estimate peak VRAM and fall back to sklearn if insufficient
            device = "cuda" if torch.cuda.is_available() else "cpu"
            use_torch = device == "cuda"

            if use_torch:
                N, D = source.shape[0], source.shape[1]
                # Peak VRAM estimate (bytes, float32):
                #   Always: X_full (N*D*4) + y_full (N*T*4)
                #   Dual (D>N): fancy-index slicing forces a contiguous copy of X_train
                #     (0.9*N rows), so peak during K_train = X_full + X_train_copy + K_train
                #     Then during K_full refit: X_full + K_full (X_train copy freed)
                #   Primal (N>=D): XtX (D*D*4) + XtY (D*T*4)
                T = target.shape[1]
                x_bytes = N * D * 4
                y_bytes = N * T * 4
                if D > N:
                    n_tr = int(0.9 * N)
                    # Peak occurs when X_full + X_train_contiguous_copy + K_train all live
                    # simultaneously (fancy-index gather forces a contiguous copy)
                    x_train_copy_bytes = n_tr * D * 4
                    k_train_bytes = n_tr * n_tr * 4
                    k_full_bytes = N * N * 4
                    peak_bytes = x_bytes + y_bytes + x_train_copy_bytes + max(k_train_bytes, k_full_bytes)
                else:
                    # primal solver: keep X_full + XtX + XtY
                    peak_bytes = x_bytes + y_bytes + D * D * 4 + D * T * 4

                if torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    print(f"Ridge: peak VRAM estimate {peak_bytes/1e9:.1f} GB, "
                          f"available {free_bytes/1e9:.1f} GB")
                    if peak_bytes > free_bytes * 0.85:
                        use_torch = False
                        print("Ridge: TorchRidgeCV insufficient VRAM → switching to chunked kernel solver.")

            if use_torch:
                def model_factory():
                    return TorchRidgeCV(self.alpha_options, device=device)
            elif device == "cuda":
                # TorchRidgeCV would OOM (D×D or X_full too large) but CUDA is available.
                # TorchChunkedKernelRidgeCV builds K = X@X.T (N×N) row-by-row without
                # ever loading full X to GPU, keeping peak VRAM ~2-3 GB for D=75k.
                print("Ridge: using chunked kernel (dual) GPU solver (TorchChunkedKernelRidgeCV).")
                def model_factory():
                    return TorchChunkedKernelRidgeCV(self.alpha_options, device=device)
            else:
                def model_factory():
                    return RidgeCV(alphas=self.alpha_options,
                                   store_cv_results=True, alpha_per_target=True)

        if test_source is None:
            return run_kfold_cv(model_factory, source, target, scoring_funcs, stratify_on=stratify_on)
        return run_eval(model_factory, source, target, test_source, test_target, scoring_funcs)

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:

        raw_scores = self.compute_raw(
            source, target, test_source, test_target, stratify_on)
        if not isinstance(raw_scores, dict):
            return raw_scores  # Early return (for RSA, etc.)

        processed_scores = {}
        for key, value in raw_scores.items():
            if key in ['preds', 'gt', 'targets', 'coef', 'intercept'] or 'alpha' in key or 'trial_based_raw_' in key:
                processed_scores.update({f"{key}": value})
            else:
                ceiled_scores = self.apply_ceiling(value)
                ceiled_median_scores = (
                    np.median(ceiled_scores, axis=1)
                    if ceiled_scores.ndim > 1
                    else ceiled_scores
                )
                unceiled_median_scores = (
                    np.median(value, axis=1)
                    if value.ndim > 1
                    else value
                )
                final_ceiled_score = np.mean(ceiled_median_scores)
                final_unceiled_score = np.mean(unceiled_median_scores)
                processed_scores.update({
                    f"raw_{key}": value,
                    f"ceiled_{key}": ceiled_scores,
                    f"median_unceiled_{key}": unceiled_median_scores,
                    f"median_ceiled_{key}": ceiled_median_scores,
                    f"final_{key}": final_ceiled_score,
                    f"final_unceiled_{key}": final_unceiled_score,
                })

        return processed_scores


class RidgeAutoMetric(BaseMetric):
    def __init__(
        self,
        alpha_options: List[float] = [
            1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
            0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10
        ],
        ceiling: Optional[float] = None,
            mode: Optional[str] = "auto"):
        super().__init__(ceiling)
        self.alpha_options = alpha_options
        self.mode = mode

    def compute_raw(
            self,
            source,
            target,
            test_source=None,
            test_target=None,
            stratify_on=None
    ):
        # Flatten inputs
        if source.ndim > 2:
            source = source.reshape(source.shape[0], -1)
            if test_source is not None:
                test_source = test_source.reshape(test_source.shape[0], -1)

        # Logic to choose solver
        total_elements = source.shape[0] * source.shape[1]
        use_torch, use_lasso, use_elastic = False, False, False

        if self.mode == 'auto' and total_elements > 2e9:
            # > 2 Billion elements implies > 16GB float64.
            use_torch = True

        scoring_funcs = {
            "pearson": lambda y_t, y_p: np.array([pearson_correlation_scorer(y_t[:, i], y_p[:, i]) for i in range(y_t.shape[1])]),
            "r2": lambda y_t, y_p: np.array([r2_score(y_t[:, i], y_p[:, i]) for i in range(y_t.shape[1])]),
        }

        if use_torch or self.mode == 'torch':
            print("⚡ Switching to GPU-Optimized TorchRidge (Float32)...")

            # We do NOT split here anymore. We pass the factory the class that
            # handles internal splitting inside .fit()

            # Auto-detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def model_factory():
                return TorchRidgeCV(self.alpha_options, device=device)
        elif self.mode == 'lasso':
            # Auto-detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def model_factory():
                return TorchElasticNetCV(self.alpha_options, device=device)
        elif self.mode == 'elastic':
            # Auto-detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def model_factory():
                return TorchElasticNetCV(self.alpha_options, l1_ratio=0.5, device=device)
        elif self.mode == 'woodbury':
            # Auto-detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def model_factory():
                return TorchConstrainedRidgeCV(self.alpha_options, device=device)
        elif self.mode == 'block':
            # Auto-detect device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def model_factory():
                return TorchBlockRidgeCV(self.alpha_options, device=device)
        else:
            def model_factory():
                return RidgeCV(alphas=self.alpha_options, store_cv_results=False)

        if test_source is not None:
            return run_eval(model_factory, source, target, test_source, test_target, scoring_funcs)

        from .utils import run_kfold_cv
        return run_kfold_cv(model_factory, source, target, scoring_funcs, stratify_on=stratify_on)

    def compute(self, source, target, test_source=None, test_target=None, stratify_on=None):
        raw_scores = self.compute_raw(
            source, target, test_source, test_target, stratify_on)
        if not isinstance(raw_scores, dict):
            return raw_scores

        processed_scores = {}
        for key, value in raw_scores.items():
            if key in ['preds', 'gt', 'targets', 'coef', 'intercept'] or 'alpha' in key or 'trial_based_raw_' in key:
                processed_scores.update({f"{key}": value})
            else:
                ceiled_scores = self.apply_ceiling(value)
                ceiled_median_scores = (
                    np.median(ceiled_scores, axis=1)
                    if ceiled_scores.ndim > 1
                    else ceiled_scores
                )
                unceiled_median_scores = (
                    np.median(value, axis=1)
                    if value.ndim > 1
                    else value
                )
                final_ceiled_score = np.mean(ceiled_median_scores)
                final_unceiled_score = np.mean(unceiled_median_scores)
                processed_scores.update({
                    f"raw_{key}": value,
                    f"ceiled_{key}": ceiled_scores,
                    f"median_unceiled_{key}": unceiled_median_scores,
                    f"median_ceiled_{key}": ceiled_median_scores,
                    f"final_{key}": final_ceiled_score,
                    f"final_unceiled_{key}": final_unceiled_score,
                })
        return processed_scores


class TorchRidgeMetric(RidgeAutoMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling=ceiling, mode='torch')


class TorchLassoMetric(RidgeAutoMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling=ceiling, mode='lasso')


class TorchElasticMetric(RidgeAutoMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
    ):
        super().__init__(ceiling=ceiling, mode='elastic')


class Ridge3DChunkedMetric(RidgeMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
        chunk_size=4000,
    ):
        super().__init__(ceiling=ceiling)
        self.chunk_size = chunk_size

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        # Expects format (N, number of voxels, timebins)
        assert target.ndim == 3

        if source.ndim > 2:
            N = source.shape[0]
            source = source.reshape(N, -1)
            if test_source is not None:
                N_test = test_source.shape[0]
                test_source = test_source.reshape(N_test, -1)

        scoring_funcs = {
            "pearson": lambda y_true, y_pred: np.array([pearson_correlation_scorer(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]),
            "r2": lambda y_true, y_pred: np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]),
        }

        # This model factory uses the array of per-dimension alphas
        # when alpha_per_target=True is handled by your Ridge class/fork.
        def model_factory():
            return RidgeCV(
                alphas=self.alpha_options,
                store_cv_results=True,
                alpha_per_target=True,
            )

        return self._compute_raw(source,
                                 target,
                                 test_source,
                                 test_target,
                                 scoring_funcs,
                                 model_factory,
                                 stratify_on,
                                 )

    def _compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray],
        test_target: Optional[np.ndarray],
        scoring_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
        model_factory: Callable[[], RidgeCV],
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:

        N, n_voxels, t = target.shape
        target_flat = target.reshape(N, -1)

        if test_target is not None:
            N_test, n_reps_test, _ = test_target.shape
            test_target_flat = test_target.reshape(N_test, -1)
        else:
            test_target_flat = None

        if test_source is None:
            raw_flat = run_kfold_cv_chunked(
                model_factory,
                source,
                target_flat,
                scoring_funcs,
                self.chunk_size,
                stratify_on=stratify_on,
            )
        else:
            # raw_flat = run_eval_chunked(model_factory, source, target_flat,
            #                            test_source, test_target_flat, scoring_funcs, self.chunk_size)
            raw_flat = run_eval(model_factory, source, target_flat,
                                test_source, test_target_flat, scoring_funcs)
        raw: Dict[str, np.ndarray] = {}
        for name, scores_flat in raw_flat.items():
            scores = scores_flat.reshape(
                n_voxels, t) if test_source is not None else scores_flat.reshape(10, n_voxels, t)
            raw[name] = scores                # full per-voxel map

        return raw

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:

        raw_scores = self.compute_raw(
            source, target, test_source, test_target, stratify_on)

        processed_scores = {}
        for key, value in raw_scores.items():
            ceiled_scores = self.apply_ceiling(value)
            ceiled_median_scores = (
                np.median(ceiled_scores, axis=1)
                if ceiled_scores.ndim > 2
                else np.median(ceiled_scores, axis=0)
            )

            unceiled_median_scores = (
                np.median(value, axis=1)
                if value.ndim > 2
                else np.median(value, axis=0)
            )
            if ceiled_median_scores.ndim == 2:
                final_ceiled_score = np.mean(ceiled_median_scores, axis=0)
                final_unceiled_score = np.mean(unceiled_median_scores, axis=0)
            else:
                final_ceiled_score = ceiled_median_scores
                final_unceiled_score = unceiled_median_scores

            if key not in ['preds', 'targets']:
                processed_scores.update({
                    f"raw_{key}": value,
                    f"ceiled_{key}": ceiled_scores,
                    f"median_unceiled_{key}": unceiled_median_scores,
                    f"median_ceiled_{key}": ceiled_median_scores,
                    f"final_{key}": final_ceiled_score,
                    f"final_unceiled_{key}": final_unceiled_score,
                })
            else:
                processed_scores.update({
                    f"raw_{key}": value,
                })

        return processed_scores


class InverseRidgeChunkedMetric(RidgeMetric):
    def __init__(
        self,
        ceiling: Optional[float] = None,
        chunk_size: int = 20000,
    ):
        super().__init__(ceiling=ceiling)
        self.chunk_size = chunk_size

    def compute_raw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Inverse mapping: use 'target' as features to predict 'source'.
        Flattens any 3D inputs into 2D.
        """
        # Flatten 3D arrays into 2D
        if source.ndim > 2:
            N = source.shape[0]
            source = source.reshape(N, -1)
        if target.ndim > 2:
            N = target.shape[0]
            target = target.reshape(N, -1)
        if test_source is not None and test_source.ndim > 2:
            N_test = test_source.shape[0]
            test_source = test_source.reshape(N_test, -1)
        if test_target is not None and test_target.ndim > 2:
            N_test = test_target.shape[0]
            test_target = test_target.reshape(N_test, -1)

        chunk_size = target.shape[1]
        # Define scoring functions per response dimension
        scoring_funcs = {
            "pearson": lambda y_true, y_pred: np.array([
                pearson_correlation_scorer(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]),
            "r2": lambda y_true, y_pred: np.array([
                r2_score(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]),
        }

        # Model factory: per-target alpha tuning
        def model_factory():
            return RidgeCV(
                alphas=self.alpha_options,
                store_cv_results=True,
                alpha_per_target=True,
            )

        # No held-out test set: run k-fold CV
        if test_target is None:
            # Features=X=target, Responses=y=source
            return run_kfold_cv_chunked(
                model_factory,
                target,
                source,
                scoring_funcs,
                self.chunk_size,
                stratify_on=stratify_on,
            )

        # Held-out evaluation mode
        # test_target: features for test, test_source: responses for test
        return run_eval_chunked(
            model_factory,
            target,
            source,
            test_target,
            test_source,
            scoring_funcs,
            self.chunk_size,
        )

    def apply_ceiling(self, scores: np.ndarray) -> np.ndarray:
        return scores

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        test_source: Optional[np.ndarray] = None,
        test_target: Optional[np.ndarray] = None,
        stratify_on: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:

        raw_scores = self.compute_raw(
            source, target, test_source, test_target, stratify_on)

        processed_scores = {}
        for key, value in raw_scores.items():
            unceiled_median_scores = (
                np.median(value, axis=1)
                if value.ndim > 1
                else value
            )
            final_unceiled_score = np.mean(unceiled_median_scores)

            if key not in ['preds', 'targets'] or 'trial_based_raw_' not in key:
                processed_scores.update({
                    f"raw_{key}": value,
                    f"median_{key}": unceiled_median_scores,
                    f"final_{key}": final_unceiled_score,
                })
            else:
                processed_scores.update({
                    f"raw_{key}": value,
                })

        return processed_scores
