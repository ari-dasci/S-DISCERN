# Lazy exports (PEP 562): `from src.utils import ClassificationDataset` works as
# before, but importing a light submodule (src.utils.proposed_method,
# src.utils.embs_fallback, ...) no longer pulls in torch, so the CPU analysis
# scripts run in any Python with pandas / scipy.
import importlib

_LAZY = {
    'ClassificationDataset': '.dataset_loader',
    'OutlinerDataset': '.dataset_loader',
    'SurrogateDataset': '.dataset_loader',
    'EmbeddingDataset': '.dataset_loader',
    'MixedLoader': '.dataset_loader',
    'load_data_general': '.load_data',
    'number_dataset_classes': '.load_data',
    'IMPLEMENTED_METHODS': '.Constants',
    'VALID_DATASETS': '.Constants',
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
