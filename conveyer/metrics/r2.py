import numpy as np
from sklearn.metrics import r2_score


def r2(preds, target):
    if len(preds.shape) == 1:
        return r2_score(target, preds)
    else:
        return r2_score(target,
                        preds,
                        multioutput='uniform_average')
