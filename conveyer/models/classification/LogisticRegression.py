from hyperopt import hp
import numpy as np
from sklearn.linear_model import LogisticRegression as Classifier
from .ClassificationModel import ClassificationModel


class LogisticRegression(ClassificationModel):
    def __init__(self):
        super().__init__()

        self.base_model = Classifier
        self.param_space = {
            'penalty':      hp.choice('penalty', ['l1', 'l2']),
            'C':            hp.loguniform('C', np.log(1), np.log(100)),
            'random_state': hp.choice('random_state', [0])
        }

    def __repr__(self):
        return 'LogisticRegression'
