import torch.nn as nn

CRITERIONS_REGISTRY = {
    # Regression
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    # Classification
    "cross_entropy": nn.CrossEntropyLoss,
    "nll": nn.NLLLoss,
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    # Embedding / Metric learning
    "cosine_embedding": nn.CosineEmbeddingLoss,
    "triplet_margin": nn.TripletMarginLoss,
    "triplet_margin_with_distance": nn.TripletMarginWithDistanceLoss,
    "margin_ranking": nn.MarginRankingLoss,
    "hinge_embedding": nn.HingeEmbeddingLoss,
    # Other
    "kl_div": nn.KLDivLoss,
    "poisson_nll": nn.PoissonNLLLoss,
    "ctc": nn.CTCLoss,
    "gaussian_nll": nn.GaussianNLLLoss,
}
