from models.loss.custom_rules import c1_should_be_close_to_c2

CUSTOM_RULES_REGISTRY = {
    "c1_should_be_close_to_c2": c1_should_be_close_to_c2,
}