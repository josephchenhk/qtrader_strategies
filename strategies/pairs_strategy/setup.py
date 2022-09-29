# -*- coding: utf-8 -*-
# @Time    : 10/8/2022 3:15 am
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: setup.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='pairs_trading',
    version='0.0.1',
    keywords=('Pairs Trading'),
    description='Paris Trading Strategy',
    long_description_content_type='text/markdown',
    license='JXW',
    install_requires=['statsmodels',
                      'scipy',
                      'numpy',
                      'pandas',
                      'qtrader',
                      'qtalib'],
    author='josephchen',
    author_email='josephchenhk@gmail.com',
    include_package_data=True,
    packages=find_packages(),
    platforms='any',
    url='',
    ext_modules=cythonize("pairs_strategy_v2.py"),
)