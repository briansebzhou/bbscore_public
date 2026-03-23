"""
run_m2m.py -- Model-to-Model cross-representation comparison.

Compares two models' representations on the same saved TVSD stimulus features
using CrossModelBenchmark + CrossModelMetric (RSA, CKA, Ridge R²).

Prerequisites:
  Both models must have features pre-saved via:
    python run.py --model <model> --layer <layer> --benchmark TVSDV1 \
        --metric ridge --save-features

Usage examples:
  # Single pair
  python run_m2m.py --model-a retina_cnn --layer-a conv0 \
      --model-b dinov2_base --layer-b _orig_mod.encoder.layer.2

  # All retina CNN layers vs all 12 DINOv2 layers (shell loop)
  for ra in conv0 conv1 linear; do
    for rb in 0 1 2 3 4 5 6 7 8 9 10 11; do
      python run_m2m.py --model-a retina_cnn --layer-a $ra \\
          --model-b dinov2_base --layer-b _orig_mod.encoder.layer.$rb
    done
  done
"""
import argparse
import os

# Use project-local bbscore_data when running from this repo
_project_root = os.path.dirname(os.path.abspath(__file__))
_project_data = os.path.join(_project_root, "bbscore_data")
if os.path.exists(_project_data):
    os.environ["SCIKIT_LEARN_DATA"] = _project_data

# Reduce CUDA allocator fragmentation — helps when large tensors are allocated
# and freed repeatedly across PCA/CKA/ridge stages within a single process.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from benchmarks import BENCHMARK_REGISTRY  # noqa: E402 (registers all benchmarks)
from metrics import METRICS, validate_metric_benchmark  # noqa: E402


def run_cross_model(
    model_a: str,
    layer_a: str,
    model_b: str,
    layer_b: str,
    stimulus_class: str,
    pca_components: int,
    debug: bool,
) -> dict:
    print(
        f"\nCross-model comparison\n"
        f"  Model A : {model_a}  /  layer: {layer_a}\n"
        f"  Model B : {model_b}  /  layer: {layer_b}\n"
        f"  Stimuli : {stimulus_class}\n"
        f"  PCA     : {pca_components} components\n"
    )

    benchmark = BENCHMARK_REGISTRY["CrossModel"](
        model_a=model_a,
        layer_a=layer_a,
        model_b=model_b,
        layer_b=layer_b,
        stimulus_class=stimulus_class,
        pca_components=pca_components,
        debug=debug,
    )
    benchmark.add_metric("cross_model")

    results = benchmark.run()

    print("\n--- Results ---")
    metrics_out = results.get("metrics", {})
    cross = metrics_out.get("cross_model", {})
    if isinstance(cross, dict):
        inner = cross.get("cross_model", cross)
        for key, val in inner.items():
            if key == "timestamp":
                continue
            if isinstance(val, float):
                print(f"  {key}: {val:.6f}")
            else:
                print(f"  {key}: {val}")
    print("Done.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare two models' representations on the same saved TVSD features "
            "using RSA, CKA, and ridge R²."
        )
    )
    parser.add_argument(
        "--model-a", required=True,
        help="Model A identifier (e.g. 'retina_cnn').",
    )
    parser.add_argument(
        "--layer-a", required=True,
        help="Layer name for model A (e.g. 'conv0').",
    )
    parser.add_argument(
        "--model-b", required=True,
        help="Model B identifier (e.g. 'dinov2_base').",
    )
    parser.add_argument(
        "--layer-b", required=True,
        help="Layer name for model B (e.g. '_orig_mod.encoder.layer.2').",
    )
    parser.add_argument(
        "--stimulus-class",
        default="TVSDStimulusTrainSet",
        help=(
            "Name of the stimulus dataset class whose features were saved. "
            "Default: TVSDStimulusTrainSet."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=1000,
        help=(
            "Number of PCA components to reduce features to before RSA and CKA. "
            "Set to 0 to disable PCA. Default: 1000."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debugging information.",
    )

    args = parser.parse_args()

    run_cross_model(
        model_a=args.model_a,
        layer_a=args.layer_a,
        model_b=args.model_b,
        layer_b=args.layer_b,
        stimulus_class=args.stimulus_class,
        pca_components=args.pca_components,
        debug=args.debug,
    )
