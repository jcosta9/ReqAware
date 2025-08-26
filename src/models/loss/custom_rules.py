import torch

def c1_should_be_close_to_c2(y_pred, y_true, concept_map):
    """
    Example fuzzy rule: c1 should be close to c2.
    This is a placeholder for your specific fuzzy logic implementation.
    """
    border_color_red = concept_map["border_color_red"]
    symbol_stop = concept_map["symbol_stop"]
    return torch.mean(torch.abs(y_pred[:, border_color_red] - y_pred[:, symbol_stop]))
