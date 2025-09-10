from models.loss.custom_rules import ExactlyOneShape, ExactlyOneMainColour, AtMostOneBorderColour, BetweenTwoAndThreeNumbers, AtMostOneWarning, NoSymbolsExactlyTwoColours, WarningSignExclusivity, WarningImpliesBorderRed, WarningImpliesMainWhite

CUSTOM_RULES_REGISTRY = {
    "ExactlyOneShape": ExactlyOneShape,
    "ExactlyOneMainColour": ExactlyOneMainColour,
    "AtMostOneBorderColour": AtMostOneBorderColour,
    "BetweenTwoAndThreeNumbers": BetweenTwoAndThreeNumbers,
    "AtMostOneWarning": AtMostOneWarning,
    "NoSymbolsExactlyTwoColours": NoSymbolsExactlyTwoColours,
    "WarningSignExclusivity": WarningSignExclusivity,
    "WarningImpliesBorderRed": WarningImpliesBorderRed, 
    "WarningImpliesMainWhite": WarningImpliesMainWhite
}
