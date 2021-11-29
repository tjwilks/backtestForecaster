from setuptools import find_packages, setup

with open('requirements.txt') as fp:
    install_requires = fp.read().splitlines()

setup(
    name='backtest_forecaster',
    packages=find_packages(),
    version='0.0.1',
    description='for backtesting forecast models',
    author='Toby Wilkinson',
    license='',
    install_requires=install_requires
)
