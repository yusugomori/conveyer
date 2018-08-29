import numpy as np
from sklearn.metrics import recall_score


def recall(preds, target, thres=0.5):
    if len(preds.shape) == 1:
        return recall_score(target, preds > thres)
    else:
        return recall_score(np.argmax(target, 1).astype('int32'),
                            np.argmax(preds, 1),
                            average='macro')
