import torch

def c1_should_be_close_to_c2(y_pred, y_true):
        """
        Example fuzzy rule: c1 should be close to c2.
        This is a placeholder for your specific fuzzy logic implementation.
        """
        return torch.mean(torch.abs(y_pred[:, 0] - y_pred[:, 1]))