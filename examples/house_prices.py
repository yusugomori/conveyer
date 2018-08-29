import os
from glob import glob
import conveyer


if __name__ == '__main__':
    '''
    Regression Task
    Assume we have `data/train.csv` and `data/test.csv`
    from
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
    '''
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    f_train = os.path.join(data_dir, 'train.csv')
    f_test = os.path.join(data_dir, 'test.csv')

    # training and validation
    conveyer.convey(path=f_train,
                    id_cols=['Id'],
                    out_dir='model')

    # test
    preds = conveyer.produce(f_test, model_dir='model')
    print(preds)
