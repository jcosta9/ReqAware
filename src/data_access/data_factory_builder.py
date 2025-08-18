from .registry import DATASET_FACTORY_REGISTRY
from omegaconf import DictConfig


class DatasetFactoryBuilder:
    @staticmethod
    def build_factory(config: DictConfig):
        try:
            return DATASET_FACTORY_REGISTRY[config.dataset.lower()](config)
        except KeyError:
            raise KeyError(f"Unsupported dataset: {config.dataset}")
