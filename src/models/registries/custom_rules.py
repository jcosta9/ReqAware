from models.loss.custom_rules import ExactlyOneShape, ExactlyOneMainColour, AtMostOneBorderColour, BetweenTwoAndThreeNumbers, AtMostOneWarning, NoSymbolsExactlyTwoColours, WarningSignExclusivity

CUSTOM_RULES_REGISTRY = {
    "ExactlyOneShape": ExactlyOneShape,
    "ExactlyOneMainColour": ExactlyOneMainColour,
    "AtMostOneBorderColour": AtMostOneBorderColour,
    "BetweenTwoAndThreeNumbers": BetweenTwoAndThreeNumbers,
    "AtMostOneWarning": AtMostOneWarning,
    "NoSymbolsExactlyTwoColours": NoSymbolsExactlyTwoColours,
    "WarningSignExclusivity": WarningSignExclusivity
}
