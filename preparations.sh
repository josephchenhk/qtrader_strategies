#!/bin/bash
echo "Installing Python environment and packages ..."
conda create -n demo_strategy1 python=3.8 -y
echo "Activate conda environment"
sleep 3
conda activate demo_strategy1
pip install --no-input git+https://github.com/josephchenhk/qtrader@master
pip install --no-input dill finta termcolor pyyaml func_timeout scipy statsmodels hyperopt jupyter seaborn
echo "Python environment and packages have been installed."
exec /bin/bash