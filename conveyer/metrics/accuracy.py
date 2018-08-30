import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(preds, target):
    return accuracy_score(target, preds)
