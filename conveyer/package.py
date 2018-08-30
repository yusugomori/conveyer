import os
from .factories import Factory


def convey(path,
           test_path=None,
           target_name=None,
           id_cols=[],
           header='infer',
           out_dir=None):
    '''
    For training and validation,
    also covers test if specified.
    '''
    factory = Factory()
    factory.fit(path,
                target_name=target_name,
                id_cols=id_cols,
                csv_header=header)

    if out_dir is not None:
        factory.save(out_dir=out_dir)

    if test_path is not None:
        preds = produce(test_path,
                        model_dir=out_dir,
                        header=header)
        return preds


def produce(test_path,
            model_dir=None,
            header='infer',
            predict_proba=False):
    '''
    For test (only returns predicted values / probabilities).
    '''
    if model_dir is None:
        raise ValueError('`model_dir` must be specified to load the model.')

    factory = Factory()
    print('Loading model...')
    factory.load(model_dir)
    print('Formatting data...')
    factory.format(test_path)
    if predict_proba:
        preds = factory.predict_proba(factory.test_X)
    else:
        preds = factory.predict(factory.test_X)

    return preds
