from models.loss.fuzzy_transformations import (
    GodelTNorm,
    GodelTConorm,
    GodelEAggregation,
    GodelAAggregation,
    ProductTNorm,
    ProductTConorm,
    YagerTNorm,
    YagerTConorm,
    LogProductAAggregation,
    GeneralizedMeanEAggregation,
    ProductAAggregation
)

TNORM_REGISTRY = {
    "godel_t_norm": GodelTNorm,
    "product_t_norm": ProductTNorm,
    "yager_t_norm": YagerTNorm
}

TCONORM_REGISTRY = {
    "godel_t_conorm": GodelTConorm,
    "product_t_conorm": ProductTConorm,
    "yager_t_conorm": YagerTConorm
}

AAGGREGATION_REGISTRY = {
    "godel_a_aggregation": GodelAAggregation,
    "log_product_a_aggregation": LogProductAAggregation,
    "product_a_aggregation": ProductAAggregation
}

EAGGREGATION_REGISTRY = {
    "godel_e_aggregation": GodelEAggregation,
    "generalized_mean_e_aggregation": GeneralizedMeanEAggregation
}
