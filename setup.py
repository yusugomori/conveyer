from setuptools import setup
from setuptools import find_packages

setup(
    name='conveyer',
    version='0.0.3',
    description='Automated machine learning library',
    author='Yusuke Sugomori',
    author_email='me@yusugomori.com',
    url='https://github.com/yusugomori/conveyer',
    download_url='',
    install_requires=['hyperopt>=0.1',
                      'networkx==1.11',
                      'numpy==1.13.3',
                      'pandas>=0.21.0',
                      'scikit-learn>=0.19.1',
                      'xgboost>=0.7'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='automated machine learning',
    packages=find_packages()
)
