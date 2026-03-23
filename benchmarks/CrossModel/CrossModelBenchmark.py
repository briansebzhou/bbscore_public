"""
CrossModelBenchmark -- compare two models' representations on the same stimuli.

Rather than running live feature extraction, this benchmark loads pre-saved
feature .pkl files produced by run.py --save-features and feeds them directly
to CrossModelMetric.  No model loading, no neural assembly.

Expected feature file format (from BBS.py save_features logic):
    {
        'train':        np.ndarray  (n_train_images, n_features),
        'train_labels': np.ndarray  (n_train_images,),
        'test':         np.ndarray  (n_test_images,  n_features)  [optional],
        'test_labels':  np.ndarray  (n_test_images,)              [optional],
    }

File naming convention (matches BBS.py):
    {model_identifier}_{layer_name}_{stimulus_class_name}_features.pkl
"""
from __future__ import annotations

import datetime
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.datasets import get_data_home

from benchmarks import BENCHMARK_REGISTRY
from metrics import METRICS


class CrossModelBenchmark:
    """
    Load saved features for two models and run cross-model comparisons.

    Parameters
    ----------
    model_a, layer_a : identifiers for model A (source)
    model_b, layer_b : identifiers for model B (target)
    stimulus_class   : name of the stimulus dataset class whose features were
                       saved (default: 'TVSDStimulusTrainSet')
    pca_components   : passed through to CrossModelMetric
    features_dir     : override default features path (bbscore_data/features/)
    debug            : print extra info
    """

    def __init__(
        self,
        model_a: str,
        layer_a: str,
        model_b: str,
        layer_b: str,
        stimulus_class: str = "TVSDStimulusTrainSet",
        pca_components: int = 1000,
        features_dir: Optional[str] = None,
        debug: bool = False,
    ):
        self.model_a = model_a
        self.layer_a = layer_a
        self.model_b = model_b
        self.layer_b = layer_b
        self.stimulus_class = stimulus_class
        self.pca_components = pca_components
        self.debug = debug

        data_home = get_data_home()
        self.features_dir = features_dir or os.path.join(data_home, "features")

        results_base = os.environ.get("RESULTS_PATH", data_home)
        self.results_dir = os.path.join(results_base, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.metrics: dict = {}
        self.metric_params: dict = {}

    # ---------------------------------------------------------------------- #
    # Public API matching BenchmarkScore interface used by run_m2m.py         #
    # ---------------------------------------------------------------------- #

    def add_metric(self, name: str, metric_params: Optional[dict] = None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def initialize_aggregation(self, mode: str):
        if mode != "none":
            print(
                "Warning: CrossModelBenchmark does not support aggregation. Ignoring."
            )

    def initialize_rp(self, rp):
        if rp is not None:
            print(
                "Warning: CrossModelBenchmark does not support random projection. Ignoring."
            )

    # ---------------------------------------------------------------------- #
    # Feature loading helpers                                                  #
    # ---------------------------------------------------------------------- #

    def _feature_filename(self, model: str, layer: str) -> str:
        return f"{model}_{layer}_{self.stimulus_class}_features.pkl"

    def _load_features(self, model: str, layer: str) -> dict:
        fname = self._feature_filename(model, layer)
        path = os.path.join(self.features_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature file not found: {path}\n"
                "Run:  python run.py --model {model} --layer {layer} "
                "--benchmark TVSDV1 --metric ridge --save-features"
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        if self.debug:
            print(f"  Loaded {path}  train shape: {np.asarray(data['train']).shape}")
        return data

    @staticmethod
    def _flatten(arr: np.ndarray) -> np.ndarray:
        """Flatten all dims after the first into a single feature dim."""
        arr = np.asarray(arr, dtype=np.float32)
        return arr.reshape(arr.shape[0], -1)

    # ---------------------------------------------------------------------- #
    # Main run                                                                 #
    # ---------------------------------------------------------------------- #

    def run(self) -> dict:
        print(f"Loading features for model A: {self.model_a} / {self.layer_a}")
        feat_a = self._load_features(self.model_a, self.layer_a)
        print(f"Loading features for model B: {self.model_b} / {self.layer_b}")
        feat_b = self._load_features(self.model_b, self.layer_b)

        train_a = self._flatten(feat_a["train"])
        train_b = self._flatten(feat_b["train"])

        # Stimulus alignment sanity check
        labels_a = np.asarray(feat_a.get("train_labels", []))
        labels_b = np.asarray(feat_b.get("train_labels", []))
        if labels_a.size and labels_b.size:
            if not np.array_equal(labels_a, labels_b):
                raise ValueError(
                    "train_labels mismatch between model A and model B features. "
                    "Ensure both were extracted from the same stimulus class with "
                    "the same ordering."
                )
        if train_a.shape[0] != train_b.shape[0]:
            raise ValueError(
                f"Sample count mismatch: model A has {train_a.shape[0]} train images, "
                f"model B has {train_b.shape[0]}."
            )

        # Optional test split (used for ridge if both present)
        test_a = test_b = None
        if feat_a.get("test") is not None and feat_b.get("test") is not None:
            test_a = self._flatten(feat_a["test"])
            test_b = self._flatten(feat_b["test"])
            # Align test labels
            tl_a = np.asarray(feat_a.get("test_labels", []))
            tl_b = np.asarray(feat_b.get("test_labels", []))
            if tl_a.size and tl_b.size and not np.array_equal(tl_a, tl_b):
                print(
                    "Warning: test_labels mismatch -- skipping test split for ridge."
                )
                test_a = test_b = None

        print(
            f"Feature shapes -- A: {train_a.shape}, B: {train_b.shape}"
            + (f"  |  test A: {test_a.shape}, test B: {test_b.shape}" if test_a is not None else "")
        )

        # Run each metric
        results: dict = {}
        for name, metric_class in self.metrics.items():
            extra = self.metric_params.get(name, {})
            if name == "cross_model":
                extra.setdefault("pca_components", self.pca_components)
            print(f"\nRunning metric: {name}")
            try:
                instance = metric_class(**extra)
                results[name] = instance.compute(
                    train_a, train_b,
                    test_source=test_a,
                    test_target=test_b,
                )
            except Exception as e:
                print(f"  Metric '{name}' failed: {e}")
                results[name] = {"error": str(e)}

        results["timestamp"] = datetime.datetime.utcnow().isoformat()

        # Save results
        safe_layer_a = self.layer_a.replace("/", "_").replace(".", "_")
        safe_layer_b = self.layer_b.replace("/", "_").replace(".", "_")
        results_file = os.path.join(
            self.results_dir,
            f"CrossModel_{self.model_a}_{safe_layer_a}_vs_{self.model_b}_{safe_layer_b}.pkl",
        )
        payload = {
            "metrics": results,
            "model_a": self.model_a,
            "layer_a": self.layer_a,
            "model_b": self.model_b,
            "layer_b": self.layer_b,
            "stimulus_class": self.stimulus_class,
            "n_train": int(train_a.shape[0]),
        }

        # Append to existing file if present (matches BBS.py accumulation pattern)
        if os.path.exists(results_file):
            try:
                with open(results_file, "rb") as f:
                    prev = pickle.load(f)
                prev_metrics = prev.get("metrics", [])
                if isinstance(prev_metrics, dict):
                    prev_metrics = [prev_metrics]
                prev_metrics.append(results)
                payload["metrics"] = prev_metrics
            except Exception:
                pass

        with open(results_file, "wb") as f:
            pickle.dump(payload, f)
        print(f"\nResults saved to {results_file}")

        return {"metrics": results}
