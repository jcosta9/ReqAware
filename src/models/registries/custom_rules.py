from models.loss.custom_rules import ExactlyOneShape, ExactlyOneMainColour, AtMostOneBorderColour

CUSTOM_RULES_REGISTRY = {
    "ExactlyOneShape": ExactlyOneShape,
    "ExactlyOneMainColour": ExactlyOneMainColour,
    "AtMostOneBorderColour": AtMostOneBorderColour
}
