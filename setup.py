import setuptools
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setuptools.setup(
    name='stoch_gpmp',
    description='Stochastic Gaussian Process Motion Planning',
    author='Julen Urain*, An T. Le*, Alexander Lambert*, Georgia Chalvatzaki, Byron Boots, Jan Peters',
    author_email='an@robot-learning.de',
    packages=setuptools.find_namespace_packages(),
    install_requires=requires_list,
)
