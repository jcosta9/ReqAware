from models.loss.custom_rules import ExactlyOneShape

CUSTOM_RULES_REGISTRY = {
    "ExactlyOneShape": ExactlyOneShape,
}
