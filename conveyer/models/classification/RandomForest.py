from hyperopt import hp
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Classifier
from .ClassificationModel import ClassificationModel


class RandomForestClassifier(ClassificationModel):
    def __init__(self):
        super().__init__()

        self.base_model = Classifier
        self.param_space = {
            'n_jobs':       hp.choice('n_jobs', [-1]),
            'n_estimators': hp.choice('n_estimators', list(range(10, 30))),
            'max_depth':    hp.choice('max_depth', list(range(5, 10))),
            'random_state': hp.choice('random_state', [0])
        }

    def __repr__(self):
        return 'RandomForest'
