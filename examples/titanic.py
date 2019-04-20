import os
from glob import glob
import conveyer


if __name__ == '__main__':
    '''
    Binary Classification Task
    Assume we have `data/train.csv` and `data/test.csv`
    from https://www.kaggle.com/c/titanic/data
    '''
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    f_train = os.path.join(data_dir, 'train.csv')
    f_test = os.path.join(data_dir, 'test.csv')

    # training and validation
    conveyer.convey(path=f_train,
                    ignore_cols=['PassengerId', 'Name'],
                    out_dir='model',
                    target_name='Survived')

    # test
    preds = conveyer.produce(f_test, model_dir='model')
    print(preds)
