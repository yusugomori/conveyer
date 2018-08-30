import numpy as np
from sklearn.metrics import recall_score


def recall(preds, target):
    if len(np.unique(target)) == 2:
        return recall_score(target, preds)
    else:
        return recall_score(target, preds,
                            average='macro')
