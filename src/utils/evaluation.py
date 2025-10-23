from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd


def get_f1_scores(y_true, y_pred):
    return {
        "micro": f1_score(y_true, y_pred, average='micro'),
        "macro": f1_score(y_true, y_pred, average='macro'),
        "weighted": f1_score(y_true, y_pred, average='weighted'),
    }

def get_precision(y_true, y_pred):
    return {
        "micro": precision_score(y_true, y_pred, average='micro'),
        "macro": precision_score(y_true, y_pred, average='macro'),
        "weighted": precision_score(y_true, y_pred, average='weighted'),
    }

def get_recall(y_true, y_pred):
    return {
        "micro": recall_score(y_true, y_pred, average='micro'),
        "macro": recall_score(y_true, y_pred, average='macro'),
        "weighted": recall_score(y_true, y_pred, average='weighted'),
    }

def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_accuracy_per_prediction(y_true, y_pred):
    correct = np.sum(np.all(y_pred.astype(float) == y_true.astype(float), axis=1))
    total = len(y_true)

    return correct/total

def compute_metrics(y_true, y_pred, concepts=False):
    results = {
        "accuracy": get_accuracy(y_true, y_pred),
        "f1": get_f1_scores(y_true, y_pred),
        "precision": get_precision(y_true, y_pred),
        "recall": get_recall(y_true, y_pred),
    }

    if concepts:
        results["concepts_accuracy_per_prediction"] = compute_accuracy_per_prediction(y_true, y_pred)

    return results




def evaluate_model_results(model, dataloader, seed):
    y_true, y_pred = model.test(mode="eval", dataloader=dataloader)
    metrics = compute_metrics(y_true, y_pred)
    df = pd.DataFrame(metrics).stack().rename(seed).to_frame().T
    return df