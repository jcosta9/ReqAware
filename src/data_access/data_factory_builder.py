from .datasets import CIFAR10Factory


class DatasetFactoryBuilder:
    @staticmethod
    def build_factory(config):
        if config.dataset.lower() == "cifar10":
            return CIFAR10Factory(config)
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset}")
