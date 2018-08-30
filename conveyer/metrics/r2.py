import numpy as np
from sklearn.metrics import r2_score


def r2(preds, target):
    return r2_score(target, preds)
