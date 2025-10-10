from .datasets import CIFAR10Factory, GTSRBFactory, BTSFactory

DATASET_FACTORY_REGISTRY = {
    "cifar10": CIFAR10Factory,
    "gtsrb": GTSRBFactory,
    "bts": BTSFactory
}
