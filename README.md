# conveyer🏭

Machine Learning for everybody. conveyer is an automated machine learning library.

## Quick glance

```python
import os
from glob import glob
import conveyer

data_dir = os.path.join(os.path.dirname(__file__), 'data')
f_train = os.path.join(data_dir, 'train.csv')
f_test = os.path.join(data_dir, 'test.csv')

# training and validation
conveyer.convey(path=f_train,
                id_cols=['Id', 'Name'],
                out_dir='model')

# test
preds = conveyer.produce(f_test, model_dir='model')
print(preds)
```

## Installation

- **Install conveyer from PyPI (recommended):**

```sh
pip install conveyer
```

- **Alternatively: install conveyer from the GitHub source:**

First, clone conveyer using `git`:

```sh
git clone https://github.com/yusugomori/conveyer.git
```

 Then, `cd` to the conveyer folder and run the install command:
```sh
cd conveyer
sudo python setup.py install
```

## License

Free for personal use only.
Contact [@yusugomori](https://yusugomori.com/contact) for commercial use or more details.

## NOTICE

conveyer is still in development.
