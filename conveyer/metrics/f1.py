import numpy as np
from sklearn.metrics import f1_score


def f1(preds, target):
    if len(np.unique(target)) == 2:
        return f1_score(target, preds)
    else:
        return f1_score(target, preds,
                        average='macro')
