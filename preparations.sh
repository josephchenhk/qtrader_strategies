#!/bin/bash
echo "Installing Python environment and packages ..."
conda create -n demo_strategy python=3.8 -y
echo "Activate conda environment"
sleep 3
conda activate demo_strategy
pip install --force-reinstall git+https://github.com/josephchenhk/qtrader@master
pip install --force-reinstall git+https://github.com/josephchenhk/qtalib@main
pip install --no-input dill finta termcolor pyyaml func_timeout scipy statsmodels hyperopt jupyter seaborn
echo "Python environment and packages have been installed."
exec /bin/bash