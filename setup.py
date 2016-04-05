from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "linvpy",
    version = "0.1.41",
    author = "Guillaume Beaud, Marta Martinez-Camara",
    author_email = "beaudguillaume@gmail.com",
    description = ("Package to solve linear inverse problems."),
    license = "BSD",
    keywords = "linear inverse M-estimator regression",
    url = "https://github.com/GuillaumeBeaud/linvpy",
    packages=find_packages(),
    long_description= long_description,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
    ],
    install_requires=[
          'numpy', 'scipy'
      ]
)