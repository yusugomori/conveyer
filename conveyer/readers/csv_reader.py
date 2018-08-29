import os
import pandas as pd


def csv_reader(path, header='infer'):
    if not os.path.exists(path):
        raise IOError('File not exists at \'{}\''.format(path))

    return pd.read_csv(path, header=header)
