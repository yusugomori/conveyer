import os
from glob import glob
import conveyer


if __name__ == '__main__':
    '''
    Multiclass Classification Task
    '''
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'iris')
    f_train = os.path.join(data_dir, 'train.csv')

    # training and validation
    conveyer.convey(path=f_train,
                    out_dir='model')
