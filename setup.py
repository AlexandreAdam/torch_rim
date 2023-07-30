from setuptools import setup, find_packages
setup(
    name='rim',
    version='0.1.0',
    author='Alexandre Adam',
    author_email='alexande.adam@umontreal.ca',
    description='A torch implementation of the Recurrent Inference Machine',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'numpy',
        'tqdm',
        'scipy',
    ],
)

