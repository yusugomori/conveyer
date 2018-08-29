import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(preds, target, thres=0.5):
    if len(preds.shape) == 1:
        return accuracy_score(target, preds > thres)
    else:
        return accuracy_score(np.argmax(target, 1).astype('int32'),
                              np.argmax(preds, 1))
