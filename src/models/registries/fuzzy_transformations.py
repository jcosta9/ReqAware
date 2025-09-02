from models.loss.fuzzy_transformations import (
    GodelTNorm,
    GodelTConorm,
    GodelEAggregation,
    GodelAAggregation,
)

TNORM_REGISTRY = {
    "godel_t_norm": GodelTNorm,
}

TCONORM_REGISTRY = {
    "godel_t_conorm": GodelTConorm,
}

AAGGREGATION_REGISTRY = {
    "godel_a_aggregation": GodelAAggregation,
}

EAGGREGATION_REGISTRY = {
    "godel_e_aggregation": GodelEAggregation,
}
