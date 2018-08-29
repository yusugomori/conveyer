class Scaler(object):
    def __init__(self):
        self.measures = {
            'mean': None,
            'std': None
        }

    def attach(self, mean=0., std=1.):
        self._mean = mean
        self._std = std

    def fit(self, data,
            mean=None,
            std=None,
            exclude_categories=True,
            value_cols=[]):
        self._mean = mean
        self._std = std
        data = self._standardize(data,
                                 exclude_categories=exclude_categories,
                                 value_cols=value_cols)
        return data

    def _standardize(self, df,
                     exclude_categories=True,
                     value_cols=[]):
        if exclude_categories and len(value_cols) > 0:
            df_ = df.loc[:, value_cols]
        else:
            df_ = df
            value_cols = list(df_)

        mean = self._mean or df_.mean()
        std = self._std or df_.std()

        # TODO: if df['col'].std() == 0
        df_ = (df_ - mean) / std

        self.measures['mean'] = mean
        self.measures['std'] = std
        df.loc[:, value_cols] = df_

        return df
