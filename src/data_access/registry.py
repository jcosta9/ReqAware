from .datasets import CIFAR10Factory, GTSRBFactory

DATASET_FACTORY_REGISTRY = {
    "cifar10": CIFAR10Factory,
    "gtsrb": GTSRBFactory,
    "bts": GTSRBFactory
}
