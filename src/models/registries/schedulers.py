import torch.optim.lr_scheduler as sched

SCHEDULERS_REGISTRY = {
    # Step decay
    "step": sched.StepLR,
    "multi_step": sched.MultiStepLR,
    # Exponential
    "exponential": sched.ExponentialLR,
    "cosine_annealing": sched.CosineAnnealingLR,
    "cosine_annealing_warm_restarts": sched.CosineAnnealingWarmRestarts,
    # Reduce on plateau
    "reduce_on_plateau": sched.ReduceLROnPlateau,
    # Cyclical
    "cyclic": sched.CyclicLR,
    "one_cycle": sched.OneCycleLR,
    # Polynomial, Lambda-based
    "lambda": sched.LambdaLR,
    "polynomial": sched.PolynomialLR,
    # Chained (combine multiple schedulers)
    "chained": sched.ChainedScheduler,
    "sequential": sched.SequentialLR,
}
