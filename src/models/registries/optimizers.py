import torch.optim as optim

OPTIMIZERS_REGISTRY = {
    # Standard
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,

    # Less common but useful
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rprop": optim.Rprop,
}
