from hyperopt import fmin, tpe, hp, rand, Trials
import numpy as np
from sklearn.metrics import mean_squared_error
from ..metrics import *


class Model(object):
    def __init__(self):
        super().__init__()

        self.base_model = None
        self._model = None  # best model
        self.param_space = {}
        self.loss = float('inf')
        self.metrics = []
        self._metrics_score = []
        self.scores = []

    def __call__(self):
        return self._model

    def attach(self, model):
        self._model = model

    def fit(self,
            train_data=(), test_data=(),
            n_search=20,
            metrics=[]):

        if len(metrics) > 0:
            self.metrics = metrics
        self._map_metrics()

        train_X, train_y = train_data
        test_X, test_y = test_data

        def search(kwargs):
            _model = self.base_model(**kwargs)
            _model.fit(train_X, train_y)
            _preds = _model.predict(test_X)
            _loss = mean_squared_error(test_y, _preds)
            if _loss < self.loss:
                self.loss = _loss
                self._model = _model
                for i, _metric in enumerate(self._metrics_score):
                    self.scores[i] = _metric(test_y, _preds)
            return _loss

        trials = Trials()

        best = fmin(search, self.param_space,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=n_search)

    def predict(self, data):
        return self._model.predict(data)

    def predict_proba(self, data):
        return self._model.predict_proba(data)

    def _map_metrics(self):
        mapper = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'r2': r2
        }

        for metric in self.metrics:
            if metric not in mapper:
                raise ValueError('Unknown metrics '
                                 '`{}` specified.'.format(metric))
            self._metrics_score.append(mapper[metric])
            self.scores.append(0.)
