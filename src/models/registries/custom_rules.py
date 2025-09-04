from models.loss.custom_rules import ExactlyOneShape, ExactlyOneMainColour, AtMostOneBorderColour, BetweenTwoAndThreeNumbers

CUSTOM_RULES_REGISTRY = {
    "ExactlyOneShape": ExactlyOneShape,
    "ExactlyOneMainColour": ExactlyOneMainColour,
    "AtMostOneBorderColour": AtMostOneBorderColour,
    "BetweenTwoAndThreeNumbers": BetweenTwoAndThreeNumbers
}
