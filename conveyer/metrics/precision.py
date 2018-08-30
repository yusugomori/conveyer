import numpy as np
from sklearn.metrics import precision_score


def precision(preds, target):
    if len(np.unique(target)) == 2:
        return precision_score(target, preds)
    else:
        return precision_score(target, preds,
                               average='macro')
