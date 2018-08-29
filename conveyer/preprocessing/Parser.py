import numpy as np
import pandas as pd
from ..types import Task


class Parser(object):
    def __init__(self):
        self.ttype = None
        self.target_mapper = None
        # self.categorical_thres = 10
        # self.replace_strategy = 'median'
        # self.categorical_cols = {}
        # self.value_cols = {}
        self._parser = {
            'object': self._parse_object,
            'int': self._parse_int,
            'int32': self._parse_int,
            'int64': self._parse_int,
            'float': self._parse_float,
            'float32': self._parse_float,
            'float64': self._parse_float
        }

    def reindex(self, df, target_name):
        header = list(df)
        try:
            target = header.pop(header.index(target_name))
        except ValueError:
            print('[Warning] '
                  'The name \'{}\' is not in the list. '
                  'Skipped.'.format(target_name))
            return df
        header.append(target)
        df = df.reindex(header, axis=1)
        return df

    def parse(self, df,
              id_cols=[],
              ttype='infer',
              categorical_thres=10,
              replace_strategy='median',
              replace_values=None,
              categorical_cols=None,
              value_cols=None,
              include_target=True,
              verbose=1):

        self.ttype = ttype
        self.categorical_thres = categorical_thres
        self.replace_strategy = replace_strategy
        self.categorical_cols = categorical_cols
        self.value_cols = value_cols

        if len(id_cols) > 0:
            df = self.drop_id_cols(df, id_cols)

        X = self.parse_data(df,
                            nan_to_category=False,
                            replace_strategy=replace_strategy,
                            replace_values=replace_values,
                            include_target=include_target,
                            verbose=verbose)
        if include_target:
            y = self.parse_target(df, verbose=verbose)
        else:
            y = None

        return (X, y)

    def drop_id_cols(self, df, id_cols):
        dtype = type(id_cols[0])  # infer type of id_cols
        cols = df.columns[id_cols] if dtype is int else id_cols
        df.drop(cols, axis='columns', inplace=True)
        return df

    def parse_data(self, df,
                   nan_to_category=False,
                   replace_strategy='median',
                   replace_values=None,
                   include_target=True,
                   raise_error=True,
                   verbose=1):
        if include_target:
            X = df.iloc[:, :-1]
            if self.categorical_cols is None:
                self.categorical_cols = {}
            if self.value_cols is None:
                self.value_cols = {}
        else:
            X = df

        header = list(X)

        # TODO: drop columns with too-many-nan values

        for name in header:
            _x = X[name]
            _type_x = str(_x.dtype)
            if _type_x not in self._parser:
                if raise_error:
                    raise ValueError('Unknown dtype detected on '
                                     '{}: {}'.format(name, _type_x))
                else:
                    X.drop(columns=[name], inplace=True)
                    continue
            _parser = self._parser[_type_x]
            _x = _parser(_x, which='data')
            X[name] = _x
            if _x.dtype == object or _x.dtype == int:
                _converted = list(pd.get_dummies(_x, dummy_na=True))
                if self.categorical_cols is not None:
                    self.categorical_cols[name] = _converted
            else:  # float
                if replace_values is not None:
                    _value = replace_values[name]
                else:
                    if replace_strategy == 'mean':
                        _value = _x.mean()
                    else:  # median
                        _value = _x.median()
                # print(name)
                self.value_cols[name] = _value

        X = self._convert_data(X,
                               nan_to_category=nan_to_category,
                               replace_strategy=replace_strategy,
                               replace_values=replace_values)

        return X

    def parse_target(self, df, verbose=1):
        y = df.iloc[:, -1]
        invalid_count = y.isnull().sum()
        if invalid_count > 0:
            _isfinite = np.isfinite(y)
            df = df[_isfinite]  # drop na
            y = y[_isfinite]
            if verbose:
                print('Dropped {} row(s) '
                      'with nan-output value.'.format(invalid_count))

        type_y = str(y.dtype)
        if type_y not in self._parser:
            raise ValueError('Unknown dtype specified on output: '
                             '{}'.format(type_y))

        parser = self._parser[type_y]
        y = parser(y, which='target')

        return y

    def _convert_data(self, df,
                      nan_to_category=False,
                      replace_strategy='median',
                      replace_values=None):
        cols = list(self.categorical_cols.keys())
        if nan_to_category:
            df = pd.get_dummies(df, dummy_na=True, columns=cols)
        else:
            if replace_values is not None:
                _values = replace_values
            else:
                _values = df[cols].median()
            df[cols] = df[cols].fillna(value=_values)
            df = pd.get_dummies(df, dummy_na=False, columns=cols)

        cols = list(self.value_cols.keys())
        df[cols] = df[cols].fillna(value=self.value_cols)

        return df

    def _parse_object(self, df, which='data'):
        if which == 'data':
            return df

        if self.ttype == Task.REGRESSION:
            raise AttributeError('Output values must be integer or float'
                                 'on regression problem.')
        self.ttype = Task.CLASSIFICATION
        values = np.sort(df.dropna().unique())
        self.target_mapper = {key: value for value, key in enumerate(values)}

        return df.map(self.target_mapper)

    def _parse_int(self, df, which='data'):
        if which == 'data':
            _values = np.sort(df.dropna().unique())
            if len(_values) <= self.categorical_thres \
               or (self.categorical_cols is not None
                   and df.name in self.categorical_cols):
                df.update(df[~df.isnull()].astype(int).astype(str))
                return df
            else:  # categorical or number
                return df.astype(float)

        if self.ttype == Task.CLASSIFICATION:
            return self._parse_object(df)
        elif self.ttype == Task.REGRESSION:
            return df.astype(float)
        else:  # infer
            if len(df.dropna().unique()) / len(df) < 0.3:
                self.ttype = Task.CLASSIFICATION
            else:
                self.ttype = Task.REGRESSION
            return df

    def _parse_float(self, df, which='data'):
        mods = np.mod(df, 1)
        mods = mods[~np.isnan(mods)]  # drop na

        if mods.sum() > 0.:
            if which == 'target':
                if self.ttype != Task.CLASSIFICATION:
                    self.ttype = Task.REGRESSION
            return df
        else:
            if which == 'data':
                return self._parse_int(df)
            if which == 'target':
                self.ttype = Task.CLASSIFICATION
                return self._parse_int(df.astype(int))
