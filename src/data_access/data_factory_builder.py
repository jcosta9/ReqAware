from .datasets import CIFAR10Factory, GTSRBFactory

supported_datasets = {
    "cifar10": CIFAR10Factory,
    "gtsrb": GTSRBFactory,
}


class DatasetFactoryBuilder:
    @staticmethod
    def build_factory(config):
        try:
            return supported_datasets[config.dataset.lower()](config)
        except KeyError:
            raise KeyError(f"Unsupported dataset: {config.dataset}")
