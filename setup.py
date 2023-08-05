from setuptools import setup, find_packages

# Read the contents of the README file
with open("long_description.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='torch_rim',
    version='0.2.1',
    author='Alexandre Adam',
    author_email='alexande.adam@umontreal.ca',
    description='A torch implementation of the Recurrent Inference Machine',
    url="https://github.com/AlexandreAdam/torch_rim",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0',
        'score_models>=0.4.4',
        "torch_ema",
        'numpy',
        'tqdm',
        'scipy',
    ],
	python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)

