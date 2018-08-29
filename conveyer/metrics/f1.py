import numpy as np
from sklearn.metrics import f1_score


def f1(preds, target, thres=0.5):
    if len(preds.shape) == 1:
        return f1_score(target, preds > thres)
    else:
        return f1_score(np.argmax(target, 1).astype('int32'),
                        np.argmax(preds, 1),
                        average='macro')
