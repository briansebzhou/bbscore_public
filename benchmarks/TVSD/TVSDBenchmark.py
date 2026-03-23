from benchmarks.BBS import BenchmarkScore, AssemblyBenchmarkScorer
from benchmarks.BBS_online import OnlineBenchmarkScore
from data.TVSD import (
    TVSDStimulusTrainSet,
    TVSDStimulusTestSet,
    TVSDAssemblyFull,
    TVSDAssemblyFull10msBins,
    TVSDAssemblyV1,
    TVSDAssemblyV4,
    TVSDAssemblyIT,
    TVSDAssemblyV110msBins,
    TVSDAssemblyV410msBins,
    TVSDAssemblyIT10msBins,
    TVSDAssemblyV1OneVsAll,
    TVSDAssemblyV4OneVsAll,
    TVSDAssemblyITOneVsAll,
    TVSDAssemblyFullOneVsAll,
    TVSDAssemblyFull10msBinsOneVsAll,
    TVSDAssemblyV110msBinsOneVsAll,
    TVSDAssemblyV410msBinsOneVsAll,
    TVSDAssemblyIT10msBinsOneVsAll,
    TVSDAssemblyMonkeyFV1,
    TVSDAssemblyMonkeyFV4,
    TVSDAssemblyMonkeyFIT,
    TVSDAssemblyMonkeyFFull,
    TVSDAssemblyMonkeyFV110msBins,
    TVSDAssemblyMonkeyFV410msBins,
    TVSDAssemblyMonkeyFIT10msBins,
    TVSDAssemblyMonkeyFFull10msBins,
    TVSDAssemblyMonkeyNV1,
    TVSDAssemblyMonkeyNV4,
    TVSDAssemblyMonkeyNIT,
    TVSDAssemblyMonkeyNFull,
    TVSDAssemblyMonkeyNV110msBins,
    TVSDAssemblyMonkeyNV410msBins,
    TVSDAssemblyMonkeyNIT10msBins,
    TVSDAssemblyMonkeyNFull10msBins,
    TVSDFullStimulusTrainSet,
    TVSDFullStimulusTestSet,
    TVSDV1StimulusTrainSet,
    TVSDV1StimulusTestSet,
    TVSDV4StimulusTrainSet,
    TVSDV4StimulusTestSet,
    TVSDITStimulusTrainSet,
    TVSDITStimulusTestSet
)
from benchmarks import BENCHMARK_REGISTRY


class TVSDFull(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyFull,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDFull"] = TVSDFull


class OnlineTVSDFull(OnlineBenchmarkScore):
    def __init__(self, model_identifier: str, layer_name: str, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(TVSDFullStimulusTrainSet,
                                  TVSDFullStimulusTrainSet),
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=0,
            stimulus_test_class=TVSDFullStimulusTestSet,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["OnlineTVSDFull"] = OnlineTVSDFull


class TVSDV1(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyV1,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=8,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDV1"] = TVSDV1


class OnlineTVSDV1(OnlineBenchmarkScore):
    def __init__(self, model_identifier: str, layer_name: str, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(TVSDV1StimulusTrainSet,
                                  TVSDV1StimulusTrainSet),
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=0,
            stimulus_test_class=TVSDV1StimulusTestSet,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["OnlineTVSDV1"] = OnlineTVSDV1


class TVSDMonkeyFV1(BenchmarkScore):
    """V1 prediction for Monkey F only (train and test within same monkey)."""
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyMonkeyFV1,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=8,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1"] = TVSDMonkeyFV1


class TVSDMonkeyNV1(BenchmarkScore):
    """V1 prediction for Monkey N only (train and test within same monkey)."""
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyMonkeyNV1,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=8,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1"] = TVSDMonkeyNV1


class TVSDV4(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyV4,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=8,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDV4"] = TVSDV4


class OnlineTVSDV4(OnlineBenchmarkScore):
    def __init__(self, model_identifier: str, layer_name: str, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(TVSDV4StimulusTrainSet,
                                  TVSDV4StimulusTrainSet),
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=0,
            stimulus_test_class=TVSDV4StimulusTestSet,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["OnlineTVSDV4"] = OnlineTVSDV4


class TVSDIT(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyIT,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=8,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDIT"] = TVSDIT


class OnlineTVSDIT(OnlineBenchmarkScore):
    def __init__(self, model_identifier: str, layer_name: str, debug: bool = False, batch_size: int = 32):
        super().__init__(
            stimulus_train_class=(TVSDITStimulusTrainSet,
                                  TVSDITStimulusTrainSet),
            model_identifier=model_identifier,
            layer_name=layer_name,
            num_classes=0,
            stimulus_test_class=TVSDITStimulusTestSet,
            dataloader_batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["OnlineTVSDIT"] = OnlineTVSDIT


class TVSDFull10msBins(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyFull10msBins,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDFull10msBins"] = TVSDFull10msBins


class TVSDV110msBins(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyV110msBins,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDV110msBins"] = TVSDV110msBins


class TVSDV410msBins(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyV410msBins,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDV410msBins"] = TVSDV410msBins


class TVSDIT10msBins(BenchmarkScore):
    def __init__(self, model_identifier, layer_name, debug: bool = False, batch_size: int = 4):
        super().__init__(
            stimulus_train_class=TVSDStimulusTrainSet,
            model_identifier=model_identifier,
            layer_name=layer_name,
            stimulus_test_class=TVSDStimulusTestSet,
            assembly_class=TVSDAssemblyIT10msBins,
            assembly_train_kwargs={'train': True},
            assembly_test_kwargs={'train': False},
            batch_size=batch_size,
            num_workers=16,
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDIT10msBins"] = TVSDIT10msBins


# ------------------------------------------------------------------------------
# Monkey-to-Monkey Mapping (Same Area)
# ------------------------------------------------------------------------------

class TVSDMonkeyFV1toNV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV1,
            target_assembly_class=TVSDAssemblyMonkeyNV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toNV1"] = TVSDMonkeyFV1toNV1


class TVSDMonkeyNV1toFV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV1,
            target_assembly_class=TVSDAssemblyMonkeyFV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toFV1"] = TVSDMonkeyNV1toFV1


class TVSDMonkeyFV4toNV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV4,
            target_assembly_class=TVSDAssemblyMonkeyNV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toNV4"] = TVSDMonkeyFV4toNV4


class TVSDMonkeyNV4toFV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV4,
            target_assembly_class=TVSDAssemblyMonkeyFV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toFV4"] = TVSDMonkeyNV4toFV4


class TVSDMonkeyFITtoNIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT,
            target_assembly_class=TVSDAssemblyMonkeyNIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoNIT"] = TVSDMonkeyFITtoNIT


class TVSDMonkeyNITtoFIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT,
            target_assembly_class=TVSDAssemblyMonkeyFIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoFIT"] = TVSDMonkeyNITtoFIT


class TVSDMonkeyFFulltoNFull(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFFull,
            target_assembly_class=TVSDAssemblyMonkeyNFull,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFFulltoNFull"] = TVSDMonkeyFFulltoNFull


class TVSDMonkeyNFulltoFFull(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNFull,
            target_assembly_class=TVSDAssemblyMonkeyFFull,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNFulltoFFull"] = TVSDMonkeyNFulltoFFull


class TVSDMonkeyFV1toNV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV1,
            target_assembly_class=TVSDAssemblyMonkeyNV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toNV4"] = TVSDMonkeyFV1toNV4


class TVSDMonkeyFV1toNIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV1,
            target_assembly_class=TVSDAssemblyMonkeyNIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toNIT"] = TVSDMonkeyFV1toNIT


class TVSDMonkeyFV1toFV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV1,
            target_assembly_class=TVSDAssemblyMonkeyFV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toFV4"] = TVSDMonkeyFV1toFV4


class TVSDMonkeyFV1toFIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV1,
            target_assembly_class=TVSDAssemblyMonkeyFIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toFIT"] = TVSDMonkeyFV1toFIT


class TVSDMonkeyFV4toNV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV4,
            target_assembly_class=TVSDAssemblyMonkeyNV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toNV1"] = TVSDMonkeyFV4toNV1


class TVSDMonkeyFV4toNIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV4,
            target_assembly_class=TVSDAssemblyMonkeyNIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toNIT"] = TVSDMonkeyFV4toNIT


class TVSDMonkeyFV4toFV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV4,
            target_assembly_class=TVSDAssemblyMonkeyFV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toFV1"] = TVSDMonkeyFV4toFV1


class TVSDMonkeyFV4toFIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV4,
            target_assembly_class=TVSDAssemblyMonkeyFIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toFIT"] = TVSDMonkeyFV4toFIT


class TVSDMonkeyFITtoFV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT,
            target_assembly_class=TVSDAssemblyMonkeyFV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoFV1"] = TVSDMonkeyFITtoFV1


class TVSDMonkeyFITtoFV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT,
            target_assembly_class=TVSDAssemblyMonkeyFV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoFV4"] = TVSDMonkeyFITtoFV4


class TVSDMonkeyFITtoNV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT,
            target_assembly_class=TVSDAssemblyMonkeyNV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoNV1"] = TVSDMonkeyFITtoNV1


class TVSDMonkeyFITtoNV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT,
            target_assembly_class=TVSDAssemblyMonkeyNV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoNV4"] = TVSDMonkeyFITtoNV4


class TVSDMonkeyNV1toNV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV1,
            target_assembly_class=TVSDAssemblyMonkeyNV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toNV4"] = TVSDMonkeyNV1toNV4


class TVSDMonkeyNV1toNIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV1,
            target_assembly_class=TVSDAssemblyMonkeyNIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toNIT"] = TVSDMonkeyNV1toNIT


class TVSDMonkeyNV1toFV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV1,
            target_assembly_class=TVSDAssemblyMonkeyFV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toFV4"] = TVSDMonkeyNV1toFV4


class TVSDMonkeyNV1toFIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV1,
            target_assembly_class=TVSDAssemblyMonkeyFIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toFIT"] = TVSDMonkeyNV1toFIT


class TVSDMonkeyNV4toNV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV4,
            target_assembly_class=TVSDAssemblyMonkeyNV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toNV1"] = TVSDMonkeyNV4toNV1


class TVSDMonkeyNV4toNIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV4,
            target_assembly_class=TVSDAssemblyMonkeyNIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toNIT"] = TVSDMonkeyNV4toNIT


class TVSDMonkeyNV4toFV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV4,
            target_assembly_class=TVSDAssemblyMonkeyFV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toFV1"] = TVSDMonkeyNV4toFV1


class TVSDMonkeyNV4toFIT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV4,
            target_assembly_class=TVSDAssemblyMonkeyFIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toFIT"] = TVSDMonkeyNV4toFIT


class TVSDMonkeyNITtoFV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT,
            target_assembly_class=TVSDAssemblyMonkeyFV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoFV1"] = TVSDMonkeyNITtoFV1


class TVSDMonkeyNITtoFV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT,
            target_assembly_class=TVSDAssemblyMonkeyFV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoFV4"] = TVSDMonkeyNITtoFV4


class TVSDMonkeyNITtoNV1(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT,
            target_assembly_class=TVSDAssemblyMonkeyNV1,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoNV1"] = TVSDMonkeyNITtoNV1


class TVSDMonkeyNITtoNV4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT,
            target_assembly_class=TVSDAssemblyMonkeyNV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoNV4"] = TVSDMonkeyNITtoNV4

# ------------------------------------------------------------------------------
# Monkey-to-Monkey Mapping (Same Area, 10ms Bins)
# ------------------------------------------------------------------------------


class TVSDMonkeyFV1toNV110ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV110msBins,
            target_assembly_class=TVSDAssemblyMonkeyNV110msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toNV110ms"] = TVSDMonkeyFV1toNV110ms


class TVSDMonkeyNV1toFV110ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV110msBins,
            target_assembly_class=TVSDAssemblyMonkeyFV110msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toFV110ms"] = TVSDMonkeyNV1toFV110ms


class TVSDMonkeyFV4toNV410ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV410msBins,
            target_assembly_class=TVSDAssemblyMonkeyNV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toNV410ms"] = TVSDMonkeyFV4toNV410ms


class TVSDMonkeyNV4toFV410ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV410msBins,
            target_assembly_class=TVSDAssemblyMonkeyFV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toFV410ms"] = TVSDMonkeyNV4toFV410ms


class TVSDMonkeyFITtoNIT10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT10msBins,
            target_assembly_class=TVSDAssemblyMonkeyNIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoNIT10ms"] = TVSDMonkeyFITtoNIT10ms


class TVSDMonkeyNITtoFIT10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT10msBins,
            target_assembly_class=TVSDAssemblyMonkeyFIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoFIT10ms"] = TVSDMonkeyNITtoFIT10ms


# ------------------------------------------------------------------------------
# Area-to-Area Mapping (Both Monkeys Combined)
# ------------------------------------------------------------------------------


class TVSD_V1_to_V4(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyV1,
            target_assembly_class=TVSDAssemblyV4,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSD_V1_to_V4"] = TVSD_V1_to_V4


class TVSD_V1_to_IT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyV1,
            target_assembly_class=TVSDAssemblyIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSD_V1_to_IT"] = TVSD_V1_to_IT


class TVSD_V4_to_IT(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyV4,
            target_assembly_class=TVSDAssemblyIT,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSD_V4_to_IT"] = TVSD_V4_to_IT

# ------------------------------------------------------------------------------
# Area-to-Area Mapping (Both Monkeys Combined, 10ms Bins)
# ------------------------------------------------------------------------------


class TVSD_V1_to_V4_10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyV110msBins,
            target_assembly_class=TVSDAssemblyV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSD_V1_to_V4_10ms"] = TVSD_V1_to_V4_10ms


class TVSD_V1_to_IT_10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyV110msBins,
            target_assembly_class=TVSDAssemblyIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSD_V1_to_IT_10ms"] = TVSD_V1_to_IT_10ms


class TVSD_V4_to_IT_10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyV410msBins,
            target_assembly_class=TVSDAssemblyIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSD_V4_to_IT_10ms"] = TVSD_V4_to_IT_10ms

# ------------------------------------------------------------------------------
# Monkey-to-Monkey Mapping (Cross Area, 10ms Bins)
# ------------------------------------------------------------------------------

# --- Monkey F to Monkey N ---


class TVSDMonkeyFV1toNV410ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV110msBins,
            target_assembly_class=TVSDAssemblyMonkeyNV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toNV410ms"] = TVSDMonkeyFV1toNV410ms


class TVSDMonkeyFV1toNIT10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV110msBins,
            target_assembly_class=TVSDAssemblyMonkeyNIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV1toNIT10ms"] = TVSDMonkeyFV1toNIT10ms


class TVSDMonkeyFV4toNV110ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV410msBins,
            target_assembly_class=TVSDAssemblyMonkeyNV110msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toNV110ms"] = TVSDMonkeyFV4toNV110ms


class TVSDMonkeyFV4toNIT10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFV410msBins,
            target_assembly_class=TVSDAssemblyMonkeyNIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFV4toNIT10ms"] = TVSDMonkeyFV4toNIT10ms


class TVSDMonkeyFITtoNV110ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT10msBins,
            target_assembly_class=TVSDAssemblyMonkeyNV110msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoNV110ms"] = TVSDMonkeyFITtoNV110ms


class TVSDMonkeyFITtoNV410ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyFIT10msBins,
            target_assembly_class=TVSDAssemblyMonkeyNV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyFITtoNV410ms"] = TVSDMonkeyFITtoNV410ms


# --- Monkey N to Monkey F ---

class TVSDMonkeyNV1toFV410ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV110msBins,
            target_assembly_class=TVSDAssemblyMonkeyFV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toFV410ms"] = TVSDMonkeyNV1toFV410ms


class TVSDMonkeyNV1toFIT10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV110msBins,
            target_assembly_class=TVSDAssemblyMonkeyFIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV1toFIT10ms"] = TVSDMonkeyNV1toFIT10ms


class TVSDMonkeyNV4toFV110ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV410msBins,
            target_assembly_class=TVSDAssemblyMonkeyFV110msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toFV110ms"] = TVSDMonkeyNV4toFV110ms


class TVSDMonkeyNV4toFIT10ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNV410msBins,
            target_assembly_class=TVSDAssemblyMonkeyFIT10msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNV4toFIT10ms"] = TVSDMonkeyNV4toFIT10ms


class TVSDMonkeyNITtoFV110ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT10msBins,
            target_assembly_class=TVSDAssemblyMonkeyFV110msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoFV110ms"] = TVSDMonkeyNITtoFV110ms


class TVSDMonkeyNITtoFV410ms(AssemblyBenchmarkScorer):
    def __init__(self, debug: bool = False):
        super().__init__(
            source_assembly_class=TVSDAssemblyMonkeyNIT10msBins,
            target_assembly_class=TVSDAssemblyMonkeyFV410msBins,
            source_assembly_train_kwargs={'train': True},
            source_assembly_test_kwargs={'train': False},
            target_assembly_train_kwargs={'train': True},
            target_assembly_test_kwargs={'train': False},
            debug=debug
        )


BENCHMARK_REGISTRY["TVSDMonkeyNITtoFV410ms"] = TVSDMonkeyNITtoFV410ms
