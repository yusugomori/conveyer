import numpy as np
from sklearn.metrics import precision_score


def precision(preds, target, thres=0.5):
    if len(preds.shape) == 1:
        return precision_score(target, preds > thres)
    else:
        return precision_score(np.argmax(target, 1).astype('int32'),
                               np.argmax(preds, 1),
                               average='macro')
