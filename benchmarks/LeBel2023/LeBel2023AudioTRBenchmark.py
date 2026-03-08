"""
LeBel2023 TR-level benchmark for audio stimuli.

Noise ceiling
-------------
Per-voxel noise ceiling is computed via within-subject split-half
reliability from repeated presentations of 'wheretheressmoke'
(see compute_splithalf_ceiling.py):

  1. Split runs into odd/even halves, average each half.
  2. Per-voxel Pearson correlation between the two halves.
  3. Spearman-Brown correction: r_ceiling = 2*r / (1 + r).

Voxels with ceiling <= 0.15 are excluded. This retains 26-47% of
whole-brain voxels depending on subject. All scoring operates on
ceiling-filtered voxels only: ridge regression runs on the subset,
and spatial masks (whole-brain, language, region groups) are mapped
into the filtered voxel space before ridge.

The precomputed ceiling is stored in data/lebel2023_ceiling_splithalf.npz.
"""
import os
import pickle
import datetime
import numpy as np
import torch
from typing import Union, List, Optional

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.datasets import get_data_home
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from data.LeBel2023 import (
    LeBel2023AudioTRStimulusSet, LeBel2023TRAssembly,
    LeBel2023FreeSurferLabels, RegionMapper,
)
from metrics import METRICS
from metrics.utils import pearson_correlation_scorer
from models import get_model_class_and_id
from benchmarks import BENCHMARK_REGISTRY


class LeBel2023AudioTRBenchmark:
    """
    TR-level temporal alignment benchmark for LeBel et al. (2023)
    using audio stimuli.

    Pipeline:
    1. Load WAV files and slice into 2-second TR segments
    2. For each TR, run audio model on the independent segment
    3. Load fMRI time series per story (no temporal averaging)
    4. Apply HRF delay (shift features forward by hrf_delay TRs)
    5. Concatenate all stories' TRs
    6. Run ridge regression with story-level GroupKFold CV
    7. Apply language mask (threshold on prediction correlation)
    """

    def __init__(
        self,
        model_identifier: str,
        layer_name: Union[str, List[str]],
        subject_id: str = 'UTS01',
        tr_duration: float = 2.0,
        hrf_delay: int = 2,
        n_cv_folds: int = 5,
        lang_mask_threshold: float = 0.05,
        batch_size: Union[int, List[int]] = None,
        debug: bool = False,
    ):
        if batch_size is None:
            batch_size = [4]
        self.debug = debug
        self.subject_id = subject_id
        self.tr_duration = tr_duration
        self.hrf_delay = hrf_delay
        self.n_cv_folds = n_cv_folds
        self.lang_mask_threshold = lang_mask_threshold
        self.model_identifier = model_identifier

        if isinstance(batch_size, list):
            self.batch_size = batch_size[0]
        else:
            self.batch_size = batch_size

        self.layer_name = layer_name
        self.layer_names = (
            layer_name if isinstance(layer_name, list) else [layer_name]
        )

        # Initialize model
        self.model_class, self.model_id_mapping = get_model_class_and_id(
            model_identifier)
        self.model_instance = self.model_class()
        self.model = self.model_instance.get_model(self.model_id_mapping)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.model.to(self.device)

        # Hook storage and registration
        self.features = {l: [] for l in self.layer_names}
        self._register_hooks()

        self.metrics = {}
        self.metric_params = {}
        self.use_ridge_smart_memory = False

        data_home = get_data_home()
        results_base = os.environ.get('RESULTS_PATH', data_home)
        self.results_dir = os.path.join(results_base, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        def hook_fn_factory(layer_id):
            def hook_fn(module, input, output):
                self.features[layer_id].append(output)
            return hook_fn

        for l_name in self.layer_names:
            found = False
            for name, module in self.model.named_modules():
                if name == l_name:
                    if isinstance(module, torch.nn.ModuleList):
                        module[-1].register_forward_hook(
                            hook_fn_factory(l_name))
                    else:
                        module.register_forward_hook(
                            hook_fn_factory(l_name))
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Layer {l_name} not found in model.")

    def _extract_tr_feature(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the audio model on a single TR's audio segment and extract
        the feature vector via the model's preprocess -> forward -> postprocess flow.

        Args:
            audio_segment: np.ndarray of shape (samples_per_tr,) at 16kHz

        Returns:
            Feature vector of shape (D,), or None if segment is silent.
        """
        # Skip silent segments
        if np.abs(audio_segment).max() < 1e-8:
            return None

        # Clear hook storage
        for l in self.layer_names:
            self.features[l] = []

        # Preprocess
        input_tensor = self.model_instance.preprocess_fn(audio_segment)
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            _ = self.model(input_tensor)

        # Collect features from all target layers
        layer_features = {}
        for l_name in self.layer_names:
            if not self.features[l_name]:
                raise ValueError(
                    f"No features captured for layer {l_name}")
            feat = self.features[l_name][0]
            if isinstance(feat, tuple):
                feat = feat[0]

            # feat shape: (1, seq_len, D) or (1, D)
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            processed = self.model_instance.postprocess_fn(feat)
            if isinstance(processed, torch.Tensor):
                processed = processed.cpu().numpy()
            layer_features[l_name] = processed.squeeze()

        if len(self.layer_names) == 1:
            return layer_features[self.layer_names[0]]
        return np.concatenate(
            [layer_features[l] for l in self.layer_names], axis=-1)

    def _extract_story_features(
        self, stimulus_set: LeBel2023AudioTRStimulusSet, story_idx: int
    ) -> np.ndarray:
        """
        Extract features for all TRs of a single story.

        Returns:
            features: (n_TRs, D) array
        """
        # Use context-aware segments if model declares audio_context_duration
        context_duration = getattr(
            self.model_instance, 'audio_context_duration', None)
        if context_duration is not None:
            segments, n_trs = stimulus_set.get_tr_audio_segments_with_context(
                story_idx, context_duration)
        else:
            segments, n_trs = stimulus_set.get_tr_audio_segments(story_idx)

        if n_trs == 0:
            return np.array([])

        features_list = []
        for tr_idx in range(n_trs):
            feat = self._extract_tr_feature(segments[tr_idx])
            features_list.append(feat)

        # Determine feature dimensionality from first non-None
        feat_dim = None
        for f in features_list:
            if f is not None:
                feat_dim = f.shape[-1]
                break

        if feat_dim is None:
            return np.array([])

        result = np.zeros((n_trs, feat_dim), dtype=np.float32)
        for i, f in enumerate(features_list):
            if f is not None:
                result[i] = f

        return result

    def _compute_temporal_rsa(self, per_story_features, per_story_fmri,
                              lang_mask=None):
        """
        Within-story temporal RSA.

        For each story:
          1. Compute model RDM: correlation distance between TRs
          2. Compute neural RDM: correlation distance between TRs
          3. Compare upper triangles with Spearman correlation
        """
        story_scores = []
        story_scores_lang = []

        for story_feat, story_fmri in zip(
                per_story_features, per_story_fmri):
            n_trs = story_feat.shape[0]
            if n_trs < 3:
                continue

            model_rdm = pdist(story_feat, metric='correlation')
            neural_rdm = pdist(story_fmri, metric='correlation')

            valid = ~(np.isnan(model_rdm) | np.isnan(neural_rdm))
            if valid.sum() < 3:
                continue

            rho, _ = spearmanr(model_rdm[valid], neural_rdm[valid])
            story_scores.append(rho)

            if lang_mask is not None and np.sum(lang_mask) > 0:
                neural_rdm_lang = pdist(
                    story_fmri[:, lang_mask], metric='correlation')
                valid_lang = ~(
                    np.isnan(model_rdm) | np.isnan(neural_rdm_lang))
                if valid_lang.sum() >= 3:
                    rho_lang, _ = spearmanr(
                        model_rdm[valid_lang],
                        neural_rdm_lang[valid_lang])
                    story_scores_lang.append(rho_lang)

        results = {
            'per_story_rsa': np.array(story_scores),
            'median_rsa_all': (
                float(np.median(story_scores))
                if story_scores else 0.0),
            'n_stories': len(story_scores),
        }

        if story_scores_lang:
            results['per_story_rsa_lang'] = np.array(story_scores_lang)
            results['median_rsa_lang'] = float(
                np.median(story_scores_lang))

        return results

    def initialize_rp(self, rp):
        """Compatibility with run.py."""
        if rp is not None:
            print("Warning: Random projection not supported "
                  "for AudioTR-level benchmark. Ignoring.")

    def initialize_aggregation(self, mode):
        """Compatibility with run.py."""
        pass

    def add_metric(self, name, metric_params=None):
        self.metrics[name] = METRICS[name]
        if metric_params:
            self.metric_params[name] = metric_params

    def run(self):
        """
        Main audio TR-level encoding model pipeline.
        """
        # 1. Load audio stimuli
        print("Loading audio TR-level stimuli...")
        stimulus_set = LeBel2023AudioTRStimulusSet(
            tr_duration=self.tr_duration
        )

        # 2. Load fMRI time series
        print(f"Loading fMRI assembly for {self.subject_id}...")
        assembly = LeBel2023TRAssembly(
            subjects=[self.subject_id]
        )
        story_fmri, ceiling, ceiling_mask = assembly.get_assembly(
            story_names=stimulus_set.story_names
        )

        # 3. Extract features and align with fMRI per story
        all_features = []
        all_fmri = []
        all_story_labels = []

        for story_idx, story_name in enumerate(stimulus_set.story_names):
            if story_name not in story_fmri:
                print(f"Warning: No fMRI data for story "
                      f"'{story_name}', skipping.")
                continue

            print(f"Processing story {story_idx + 1}/"
                  f"{len(stimulus_set.story_names)}: {story_name}")

            story_features = self._extract_story_features(
                stimulus_set, story_idx)
            fmri_data = story_fmri[story_name]

            if story_features.size == 0:
                print(f"  Skipping {story_name}: no features extracted.")
                continue

            # Align TR counts
            n_trs_feat = story_features.shape[0]
            n_trs_fmri = fmri_data.shape[0]
            n_trs = min(n_trs_feat, n_trs_fmri)

            if n_trs < self.hrf_delay + 1:
                print(f"  Skipping {story_name}: "
                      f"too few TRs ({n_trs})")
                continue

            story_features = story_features[:n_trs]
            fmri_data = fmri_data[:n_trs]

            # 4. Apply HRF delay
            shifted_features = story_features[:n_trs - self.hrf_delay]
            shifted_fmri = fmri_data[self.hrf_delay:]

            all_features.append(shifted_features)
            all_fmri.append(shifted_fmri)
            all_story_labels.extend(
                [story_idx] * shifted_features.shape[0])

            if self.debug:
                print(f"  TRs: feat={n_trs_feat}, fmri={n_trs_fmri}, "
                      f"used={shifted_features.shape[0]}")

        if not all_features:
            raise ValueError("No data after alignment.")

        # Determine which analyses to run
        run_ridge = any(
            m in self.metrics for m in ['ridge', 'torch_ridge'])
        run_temporal_rsa = 'temporal_rsa' in self.metrics
        if not run_ridge and not run_temporal_rsa:
            run_ridge = True

        results = {}
        lang_mask = None
        region_mapper = None

        # 5. Ridge regression
        if run_ridge:
            X = np.concatenate(all_features, axis=0)
            y_full = np.concatenate(all_fmri, axis=0)
            groups = np.array(all_story_labels)
            n_voxels_full = y_full.shape[1]

            # --- Pre-ridge: ceiling filter + region masks ---
            cm = ceiling_mask
            n_cm = int(cm.sum())
            ceiling_filtered = ceiling[cm]

            # Subset fMRI to ceiling-filtered voxels only
            y = y_full[:, cm]

            # Load region mapper for anatomical masks
            region_mapper = None
            spatial_masks = {'whole_brain': np.ones(n_cm, dtype=bool)}
            region_groups = [
                'language', 'non_language', 'temporal', 'frontal',
                'parietal', 'occipital', 'auditory', 'wernickes',
                'brocas',
            ]
            try:
                fs_labels = LeBel2023FreeSurferLabels()
                label_dir = fs_labels.ensure_downloaded(
                    self.subject_id)
                region_mapper = RegionMapper(label_dir)

                # Map full-space indices to filtered space
                full_to_filtered = np.full(n_voxels_full, -1,
                                           dtype=int)
                full_to_filtered[cm] = np.arange(n_cm)

                for grp in region_groups:
                    grp_idx_full = region_mapper.get_group_indices(grp)
                    grp_idx_full = grp_idx_full[
                        grp_idx_full < n_voxels_full]
                    # Intersect with ceiling mask
                    in_ceiling = cm[grp_idx_full]
                    grp_idx_filtered = full_to_filtered[
                        grp_idx_full[in_ceiling]]
                    mask = np.zeros(n_cm, dtype=bool)
                    mask[grp_idx_filtered] = True
                    spatial_masks[grp] = mask
            except Exception as e:
                print(f"Warning: Region masks failed: {e}")

            print(f"Total TRs: {X.shape[0]}, "
                  f"Feature dim: {X.shape[1]}, "
                  f"Voxels: {n_cm} (filtered from {n_voxels_full})")
            for grp, mask in spatial_masks.items():
                if grp != 'whole_brain':
                    print(f"  {grp}: {int(mask.sum())} voxels")

            # --- Run ridge with all spatial masks ---
            fold_scores = self._run_group_kfold(
                X, y, groups, spatial_masks=spatial_masks)

            median_pearson = np.median(
                fold_scores['pearson'], axis=0)
            median_r2 = np.median(fold_scores['r2'], axis=0)

            ridge_key = ('torch_ridge' if 'torch_ridge' in self.metrics
                         else 'ridge')

            # --- Level 1: Global metrics ---
            results[ridge_key] = {
                'n_ceiling_voxels': n_cm,
                'final_pearson_unceiled': float(
                    np.median(median_pearson)),
                'final_pearson_ceiled': float(
                    np.median(median_pearson / ceiling_filtered)),
                'final_r2_unceiled': float(
                    np.median(median_r2)),
                'final_r2_ceiled': float(
                    np.median(median_r2 / ceiling_filtered)),
            }

            # Per-region global metrics
            for grp, mask in spatial_masks.items():
                if grp == 'whole_brain':
                    continue
                n_grp = int(mask.sum())
                if n_grp == 0:
                    continue
                results[ridge_key][f'{grp}_pearson_unceiled'] = float(
                    np.median(median_pearson[mask]))
                results[ridge_key][f'{grp}_pearson_ceiled'] = float(
                    np.median(median_pearson[mask]
                              / ceiling_filtered[mask]))
                results[ridge_key][f'{grp}_n_voxels'] = n_grp

            r = results[ridge_key]
            print(f"Ridge ({n_cm} voxels): "
                  f"unceiled={r['final_pearson_unceiled']:.4f}, "
                  f"ceiled={r['final_pearson_ceiled']:.4f}")
            lang_n = r.get('language_n_voxels', 0)
            if lang_n > 0:
                print(f"Ridge lang ({lang_n}): "
                      f"unceiled="
                      f"{r['language_pearson_unceiled']:.4f}, "
                      f"ceiled="
                      f"{r['language_pearson_ceiled']:.4f}")

            # Build story index -> name mapping
            used_stories = [
                s for s in stimulus_set.story_names
                if s in story_fmri]
            story_idx_to_name = {
                i: name for i, name in enumerate(used_stories)}
            results[ridge_key]['story_names'] = story_idx_to_name

            # --- Levels 2-4: Per-TR spatial results ---
            per_tr = fold_scores['per_tr_spatial']
            per_tr_groups = fold_scores['per_tr_groups']

            # Build per-story TR traces (Level 4)
            per_story_traces = {}
            per_story_summary = {}
            unique_groups = np.unique(per_tr_groups)
            for s_idx in unique_groups:
                s_name = story_idx_to_name.get(int(s_idx),
                                               str(int(s_idx)))
                s_mask = per_tr_groups == s_idx
                story_traces = {}
                story_summary = {}
                for grp, tr_arr in per_tr.items():
                    vals = tr_arr[s_mask]
                    valid = vals[~np.isnan(vals)]
                    story_traces[grp] = vals
                    story_summary[grp] = (
                        float(np.median(valid))
                        if len(valid) > 0 else 0.0)
                per_story_traces[s_name] = story_traces
                per_story_summary[s_name] = story_summary

            # Level 3: Per-story summary
            results[ridge_key]['per_story_spatial'] = per_story_summary

            # Level 4: Per-story TR traces
            results[ridge_key]['per_story_tr_traces'] = (
                per_story_traces)

            # Level 2: Global TR progression
            # Align stories by TR position, average across stories
            max_trs = max(
                len(v['whole_brain'])
                for v in per_story_traces.values())
            global_tr_progression = {}
            for grp in per_tr.keys():
                progression = np.full(max_trs, np.nan)
                for t in range(max_trs):
                    vals_at_t = []
                    for traces in per_story_traces.values():
                        if t < len(traces[grp]):
                            v = traces[grp][t]
                            if not np.isnan(v):
                                vals_at_t.append(v)
                    if vals_at_t:
                        progression[t] = float(np.median(vals_at_t))
                global_tr_progression[grp] = progression
            results[ridge_key]['global_tr_progression'] = (
                global_tr_progression)

            # Print summaries
            for grp in ['whole_brain', 'language']:
                if grp not in per_tr:
                    continue
                valid = per_tr[grp][~np.isnan(per_tr[grp])]
                if len(valid) > 0:
                    print(f"Per-TR spatial ({grp}): "
                          f"median={np.median(valid):.4f}, "
                          f"mean={np.mean(valid):.4f}")

            lang_mask = spatial_masks.get('language')

        # 6. Within-story temporal RSA
        if run_temporal_rsa:
            # Filter fMRI to ceiling voxels for RSA too
            all_fmri_filtered = [f[:, cm] for f in all_fmri]
            print("Computing within-story temporal RSA...")
            rsa_results = self._compute_temporal_rsa(
                all_features, all_fmri_filtered,
                lang_mask=lang_mask)
            results['temporal_rsa'] = rsa_results
            print(f"Temporal RSA - "
                  f"Median (all voxels): "
                  f"{rsa_results['median_rsa_all']:.4f} "
                  f"({rsa_results['n_stories']} stories)")
            if 'median_rsa_lang' in rsa_results:
                print(f"Temporal RSA - "
                      f"Median (lang voxels): "
                      f"{rsa_results['median_rsa_lang']:.4f}")

        results['timestamp'] = datetime.datetime.utcnow().isoformat()
        results['hrf_delay'] = self.hrf_delay
        results['tr_duration'] = self.tr_duration
        results['lang_mask_threshold'] = self.lang_mask_threshold
        results['n_stories'] = len(
            [s for s in stimulus_set.story_names if s in story_fmri])
        n_total_trs = sum(f.shape[0] for f in all_features)
        results['total_trs'] = n_total_trs

        # Save results
        layer_str = (self.layer_name if isinstance(self.layer_name, str)
                     else "_".join(self.layer_names))
        benchmark_name = self.__class__.__name__
        results_file = os.path.join(
            self.results_dir,
            f"{self.model_identifier}_{layer_str}_"
            f"{benchmark_name}.pkl"
        )

        merged = {
            "metrics": results,
            "ceiling": ceiling,
            "ceiling_mask": ceiling_mask,
        }
        with open(results_file, 'wb') as f:
            pickle.dump(merged, f)
        print(f"Results saved to {results_file}")

        return merged

    def _run_group_kfold(self, X, y, groups, spatial_masks=None):
        """
        Ridge regression with GroupKFold CV (stories as groups).

        y is already filtered to ceiling voxels.

        Args:
            spatial_masks: dict of {name: bool_mask} over the
                filtered voxel space. Per-TR spatial correlation is
                computed for each mask.

        Returns:
            dict with:
              'pearson': (n_folds, n_voxels)
              'r2': (n_folds, n_voxels)
              'per_tr_spatial': {mask_name: (n_total_TRs,)}
              'per_tr_groups': (n_total_TRs,)
        """
        if spatial_masks is None:
            spatial_masks = {
                'whole_brain': np.ones(y.shape[1], dtype=bool)}

        n_unique_groups = len(np.unique(groups))
        n_splits = min(self.n_cv_folds, n_unique_groups)

        if n_splits < 2:
            raise ValueError(
                f"Need at least 2 story groups for CV, "
                f"got {n_unique_groups}")

        gkf = GroupKFold(n_splits=n_splits)

        alphas = [1e-6, 1e-4, 1e-2, 1.0, 10.0,
                  100.0, 1e4, 1e6]

        pearson_scores = []
        r2_scores = []

        n_total = X.shape[0]
        per_tr_spatial = {
            name: np.full(n_total, np.nan)
            for name in spatial_masks}
        per_tr_groups = groups.copy()

        for fold_idx, (train_idx, val_idx) in enumerate(
                gkf.split(X, y, groups)):
            print(f"  Fold {fold_idx + 1}/{n_splits}: "
                  f"train={len(train_idx)}, val={len(val_idx)}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = RidgeCV(alphas=alphas, store_cv_results=False)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # Per-voxel Pearson correlation
            fold_pearson = np.array([
                pearson_correlation_scorer(y_val[:, i], preds[:, i])
                for i in range(y.shape[1])
            ])
            pearson_scores.append(fold_pearson)

            # Per-voxel R2
            fold_r2 = np.array([
                r2_score(y_val[:, i], preds[:, i])
                for i in range(y.shape[1])
            ])
            r2_scores.append(fold_r2)

            # Per-TR spatial correlation for each mask
            for mask_name, mask in spatial_masks.items():
                p_v = preds[:, mask]
                y_v = y_val[:, mask]
                for local_i, global_i in enumerate(val_idx):
                    per_tr_spatial[mask_name][global_i] = (
                        pearson_correlation_scorer(
                            y_v[local_i], p_v[local_i]))

        return {
            'pearson': np.array(pearson_scores),
            'r2': np.array(r2_scores),
            'per_tr_spatial': per_tr_spatial,
            'per_tr_groups': per_tr_groups,
        }
