from setuptools import setup, find_packages

setup(
    name='my_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'transformers',
        'lazypredict',
        'requests',
        'concurrent.futures'
    ],
    entry_points={
        'console_scripts': [
            'my_project=main:main',
        ],
    },
)
