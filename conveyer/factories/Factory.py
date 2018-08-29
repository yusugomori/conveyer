import os
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ..preprocessing import Parser, Scaler
from ..readers import csv_reader
from ..types import Task
from ..models.Model import Model
from ..models.classification \
    import LogisticRegression, RandomForestClassifier, XGBoostClassifier
from ..models.regression \
    import RandomForestRegressor, XGBoostRegressor


class Factory(object):
    MODEL_PATH = 'model.pkl'
    HEADER_PATH = 'header.dat'
    REPLACE_STRATEGY_PATH = 'replace_strategy.txt'
    REPLACED_VALUE_PATH = 'replaced_value.dat'
    CATEGORIES_PATH = 'categories.dat'
    VALUES_PATH = 'values.dat'
    IDS_PATH = 'ids.dat'
    SCALER_PATH = 'scaling.dat'

    def __init__(self):
        self.ttype = None  # Task type
        self.target_mapper = None  # object
        self.parser = Parser()
        self.scaler = Scaler()
        self._model = None  # best model
        self.classifiers = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'xgboost': XGBoostClassifier
        }
        self.regressors = {
            'random_forest': RandomForestRegressor,
            'xgboost': XGBoostRegressor
        }

    @property
    def params(self):
        return self._model().get_params()

    @property
    def model(self):
        return self._model()

    def fit(self,
            path,
            target_name=None,
            id_cols=[],
            models=[],
            metrics=[],
            csv_header='infer',
            ttype='infer',
            replace_strategy='median',
            random_state=1234,
            verbose=1):
        '''
        # Arguments
            ttype: 'infer' (Default),
                   Task.CLASSIFICATION,
                   Task.REGRESSION
        '''
        self._base_df = self._read_csv(path=path, header=csv_header)
        if target_name is not None:
            self._base_df = self.parser.reindex(self._base_df,
                                                target_name)
        X, y = self.parser.parse(self._base_df,
                                 id_cols=id_cols,
                                 ttype=ttype,
                                 replace_strategy=replace_strategy,
                                 categorical_thres=10)

        self.ttype = self.parser.ttype
        print('Problem type: {}'.format(self.ttype.name))

        self._data_header = list(X)
        self.target_mapper = self.parser.target_mapper
        self.replace_strategy = self.parser.replace_strategy
        self.categorical_cols = self.parser.categorical_cols
        self.value_cols = self.parser.value_cols
        self.id_cols = id_cols

        self.data_X = X
        self.data_y = y

        X = self.scaler.fit(X,
                            exclude_categories=True,
                            value_cols=list(self.value_cols.keys()))
        y = y.values

        train_X, test_X, train_y, test_y = \
            train_test_split(X, y, random_state=random_state)
        train_data = (train_X, train_y)
        test_data = (test_X, test_y)

        if self.ttype == Task.CLASSIFICATION:
            self._classify(models,
                           metrics=metrics,
                           train_data=train_data,
                           test_data=test_data,
                           verbose=verbose)
        elif self.ttype == Task.REGRESSION:
            self._regress(models,
                          metrics=metrics,
                          train_data=train_data,
                          test_data=test_data,
                          verbose=verbose)
        if verbose:
            print('Best model selected: {}'.format(self._model()))

    def format(self, test_path, csv_header='infer'):
        df = self._read_csv(path=test_path, header=csv_header)
        X, _ = self.parser.parse(df,
                                 id_cols=self.id_cols,
                                 include_target=False,
                                 replace_values=self._X_fill,
                                 categorical_cols=self.categorical_cols,
                                 value_cols=self.value_cols,
                                 categorical_thres=10)
        header = list(X)
        only_on_test = set(header) - set(self._data_header)
        X.drop(only_on_test, axis='columns', inplace=True)

        only_on_train = list(set(self._data_header) - set(header))
        for item in only_on_train:
            X[item] = self._X_fill[item]

        X = X.reindex(self._data_header, axis=1)
        X = self.scaler.fit(X,
                            exclude_categories=True,
                            value_cols=list(self.value_cols.keys()))
        self.test_X = X

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def save(self, out_dir, verbose=1):
        if out_dir is None:
            raise ValueError('`out_dir` must be specified to save the model.')

        path = out_dir.split('/')[:-1]
        if len(path) > 0:
            os.makedirs(out_dir, exist_ok=True)

        # save model
        path = os.path.join(out_dir, self.MODEL_PATH)
        joblib.dump(self.model, path)

        # save header
        path = os.path.join(out_dir, self.HEADER_PATH)
        with open(path, 'wb') as f:
            pickle.dump(self._data_header, f)

        # save replaced value
        path = os.path.join(out_dir, self.REPLACE_STRATEGY_PATH)
        with open(path, 'wb') as f:
            pickle.dump(self.replace_strategy, f)

        path = os.path.join(out_dir, self.REPLACED_VALUE_PATH)
        with open(path, 'wb') as f:
            if self.replace_strategy == 'mean':
                _out = self.data_X.mean()
            else:
                _out = self.data_X.median()
            pickle.dump(_out, f)

        # save categorical_cols, value_cols
        path = os.path.join(out_dir, self.CATEGORIES_PATH)
        with open(path, 'wb') as f:
            pickle.dump(self.categorical_cols, f)

        path = os.path.join(out_dir, self.VALUES_PATH)
        with open(path, 'wb') as f:
            pickle.dump(self.value_cols, f)

        # save id_cols
        path = os.path.join(out_dir, self.IDS_PATH)
        with open(path, 'wb') as f:
            pickle.dump(self.id_cols, f)

        # save scaler
        path = os.path.join(out_dir, self.SCALER_PATH)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler.measures, f)

        if verbose:
            print('Model saved to: \'{}\''.format(out_dir))

    def load(self, out_dir):
        # load model
        path = os.path.join(out_dir, self.MODEL_PATH)
        self._model = Model()
        self._model.attach(joblib.load(path))

        # load header
        path = os.path.join(out_dir, self.HEADER_PATH)
        with open(path, 'rb') as f:
            self._data_header = pickle.load(f)

        # load median / mean
        path = os.path.join(out_dir, self.REPLACE_STRATEGY_PATH)
        with open(path, 'rb') as f:
            self.replace_strategy = pickle.load(f)

        path = os.path.join(out_dir, self.REPLACED_VALUE_PATH)
        with open(path, 'rb') as f:
            self._X_fill = pickle.load(f)

        # load categories, values
        path = os.path.join(out_dir, self.CATEGORIES_PATH)
        with open(path, 'rb') as f:
            self.categorical_cols = pickle.load(f)

        path = os.path.join(out_dir, self.VALUES_PATH)
        with open(path, 'rb') as f:
            self.value_cols = pickle.load(f)

        # load id_cols
        path = os.path.join(out_dir, self.IDS_PATH)
        with open(path, 'rb') as f:
            self.id_cols = pickle.load(f)

        # load scaler
        path = os.path.join(out_dir, self.SCALER_PATH)
        with open(path, 'rb') as f:
            measures = pickle.load(f)
            self.scaler.attach(measures['mean'], measures['std'])

    def _classify(self, models=[], metrics=[],
                  train_data=(), test_data=(),
                  verbose=1):
        if type(models) != list:
            raise AttributeError('Argument `models` must be a list, ',
                                 'but given {}'.format(type(models)))
        if len(models) == 0:
            models = list(self.classifiers.keys())

        classifiers = []
        for model in models:
            if model in self.classifiers:
                classifiers.append(self.classifiers[model])

        loss = float('inf')
        for classifier in classifiers:
            if verbose:
                print('Optimizing {}...'.format(classifier()))
            _model = classifier()
            _model.fit(train_data=train_data,
                       test_data=test_data,
                       metrics=metrics)
            _loss = _model.loss
            if verbose:
                self._show_fit_log(_model)
            if _loss < loss:
                loss = _loss
                self._model = _model

    def _regress(self, models=[], metrics=[],
                 train_data=(), test_data=(),
                 verbose=1):
        if type(models) != list:
            raise AttributeError('Argument `models` must be a list, ',
                                 'but given {}'.format(type(models)))
        if len(models) == 0:
            models = list(self.regressors.keys())

        regressors = []
        for model in models:
            if model in self.regressors:
                regressors.append(self.regressors[model])

        loss = float('inf')
        for regressor in regressors:
            if verbose:
                print('Optimizing {}...'.format(regressor()))
            _model = regressor()
            _model.fit(train_data=train_data,
                       test_data=test_data,
                       metrics=metrics)
            _loss = _model.loss
            if verbose:
                self._show_fit_log(_model)
            if _loss < loss:
                loss = _loss
                self._model = _model

    def _read_csv(self, path, header):
        df = csv_reader(path, header=header)

        # TODO: if header is None
        self._base_header = list(df)

        return df

    def _show_fit_log(self, model):
        _out = '  Best score(s):'
        _out += ' loss: {:.3}'.format(model.loss)
        for i, _metric in enumerate(model.metrics):
            _out += ' {}: {:.3}'.format(_metric[:3],
                                        model.scores[i])
        print(_out)
