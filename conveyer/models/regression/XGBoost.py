from hyperopt import hp
import numpy as np
from xgboost import XGBRegressor as Regressor
from .RegressionModel import RegressionModel


class XGBoostRegressor(RegressionModel):
    def __init__(self):
        super().__init__()

        self.base_model = Regressor
        self.param_space = {
            'objective':        hp.choice('objective', ['reg:linear']),
            'n_jobs':           hp.choice('n_jobs', [-1]),
            'silent':           hp.choice('silent', [1]),
            'max_depth':        hp.choice('max_depth', list(range(5, 10))),
            'min_child_weight': hp.choice('min_child_weight',
                                          list(range(1, 10))),
            'eta':              hp.uniform('eta', 0.10, 0.50),
            'gamma':            hp.uniform('gamma', 0.00, 1.00),
            'subsample':        hp.uniform('subsample', 0.50, 1.00),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.10, 1.00),
            'learning_rate':    hp.uniform('learning_rate', 0.1, 1.00),
            'random_state':     hp.choice('random_state', [0])
        }

    def __repr__(self):
        return 'XGBoost'
